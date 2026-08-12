"""One-arm isolated pilot and outer-fold worker for Loop175 Phase B."""

from __future__ import annotations

import hashlib
import io
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import torch

from src.loop167_phase_b.contracts import canonical_json_bytes

from .model import RegionNetConfig
from .phase_b_contract import (
    load_json_object,
    load_phase_b_protocol,
    sha256_file,
    validate_bound_evidence,
    write_exclusive_json,
)
from .phase_b_data import (
    FULL_TRAIN_ROWS,
    IdentityFreePhaseBFitPayload,
    load_aligned_phase_b_data,
    load_ragged_region_cache,
)
from .phase_b_engine import EngineConfig, run_inner_pilot, train_neural_arm
from .phase_b_receipt import write_arm_fold_artifact
from .phase_b_source_closure import validate_phase_b_source_closure
from .phase_b_training import (
    fit_frozen_b0_hgb,
    generate_e_weights_from_b0_inner_oof,
    predict_b0_scores,
)
from .resource_guard import gpu_allocated_bytes, process_rss_bytes

NEURAL_ARMS = frozenset({"B", "C", "D", "E"})


class PhaseBWorkerError(RuntimeError):
    """Raised when one worker cannot prove its frozen scope."""


def _hash_payload(payload: Mapping[str, object], *, domain: bytes) -> str:
    return hashlib.sha256(domain + canonical_json_bytes(dict(payload))).hexdigest()


def _peak_rss_bytes() -> int:
    current = process_rss_bytes()
    try:
        import psutil

        memory = psutil.Process(os.getpid()).memory_info()
        return max(current, int(getattr(memory, "peak_wset", current)))
    except ImportError:
        return current


def _runtime_environment() -> dict[str, object]:
    import sklearn

    return {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _validate_closure_and_cache(
    *,
    project_root: Path,
    source_closure_path: Path,
    cache_receipt_path: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    protocol = load_phase_b_protocol(project_root)
    validate_bound_evidence(project_root, protocol)
    closure = load_json_object(source_closure_path)
    validate_phase_b_source_closure(project_root, closure)
    cache_receipt = load_json_object(cache_receipt_path)
    if (
        cache_receipt.get("schema") != "axon_loop175_full_train_region_cache_receipt_v1"
        or cache_receipt.get("decision")
        != "full_train_region_cache_gate_pass_seed41_pilot_may_begin"
        or cache_receipt.get("val_test_or_full_rows_opened") != 0
        or cache_receipt.get("training_runs") != 0
    ):
        raise PhaseBWorkerError("full-Train region cache receipt did not pass safely")
    inputs = cache_receipt.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("protocol_sha256") != protocol.sha256:
        raise PhaseBWorkerError("region cache protocol commitment drifted")
    return protocol, closure, cache_receipt


def _load_payload(
    project_root: Path,
    cache_receipt: Mapping[str, Any],
) -> IdentityFreePhaseBFitPayload:
    aligned = load_aligned_phase_b_data(project_root)
    cache_binding = cache_receipt.get("cache")
    if not isinstance(cache_binding, Mapping):
        raise PhaseBWorkerError("region cache receipt has no cache binding")
    regions = load_ragged_region_cache(
        Path(str(cache_binding["path"])),
        expected_sha256=str(cache_binding["sha256"]),
        expected_rows=FULL_TRAIN_ROWS,
    )
    return aligned.make_fit_payload(regions)


def _full_e_weight_vector(payload: IdentityFreePhaseBFitPayload, *, outer_fold: int) -> np.ndarray:
    result = generate_e_weights_from_b0_inner_oof(
        payload.b0_values,
        payload.labels,
        payload.folds,
        outer_holdout_fold=outer_fold,
    )
    weights = np.ones(FULL_TRAIN_ROWS, dtype=np.float32)
    weights[result.row_indices] = result.weights
    if np.intersect1d(result.row_indices, np.flatnonzero(payload.folds == outer_fold)).size:
        raise PhaseBWorkerError("E inner-OOF weight rows intersect outer holdout")
    return weights


def _resource_payload(*, started: float, output_paths: tuple[Path, ...]) -> dict[str, object]:
    disk_bytes = sum(path.stat().st_size for path in output_paths if path.is_file())
    return {
        "wall_seconds": time.monotonic() - started,
        "rss_bytes": _peak_rss_bytes(),
        "gpu_allocated_bytes": gpu_allocated_bytes(),
        "new_disk_bytes": disk_bytes,
        "oom": False,
        "timeout": False,
        "nonfinite": False,
    }


def run_pilot(
    *,
    project_root: Path,
    source_closure_path: Path,
    cache_receipt_path: Path,
    arm: str,
    output: Path,
    microbatch: int = 2,
    gradient_accumulation: int = 16,
) -> dict[str, object]:
    if arm not in NEURAL_ARMS:
        raise PhaseBWorkerError("pilot arm must be B, C, D, or E")
    protocol, _closure, cache_receipt = _validate_closure_and_cache(
        project_root=project_root,
        source_closure_path=source_closure_path,
        cache_receipt_path=cache_receipt_path,
    )
    payload = _load_payload(project_root, cache_receipt)
    pilot_fit = np.flatnonzero(np.isin(payload.folds, (2, 3, 4))).astype(np.int64)
    selection = np.flatnonzero(payload.folds == 1).astype(np.int64)
    if np.intersect1d(np.concatenate((pilot_fit, selection)), np.flatnonzero(payload.folds == 0)).size:
        raise PhaseBWorkerError("epoch pilot touched outer fold 0")
    sample_weights = _full_e_weight_vector(payload, outer_fold=0) if arm == "E" else None
    engine = EngineConfig(
        seed=41,
        microbatch=microbatch,
        gradient_accumulation=gradient_accumulation,
        device="cuda",
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = run_inner_pilot(
        payload,
        pilot_fit_indices=pilot_fit,
        selection_indices=selection,
        arm=arm,
        max_epochs=12,
        model_config=RegionNetConfig(),
        engine=engine,
        sample_weights=sample_weights,
        protocol_sha256=protocol.sha256 if arm == "D" else None,
        outer_fold=0,
    )
    resources = _resource_payload(started=started, output_paths=())
    receipt: dict[str, object] = {
        "schema": "axon_loop175_epoch_pilot_receipt_v1",
        "loop_id": "Loop175",
        "claim_scope": "train_only_inner_epoch_selection_not_outer_oof_or_model_quality",
        "arm": arm,
        "seed": 41,
        "outer_fold_never_read": 0,
        "pilot_fit_folds": [2, 3, 4],
        "inner_selection_fold": 1,
        "pilot_fit_rows": int(pilot_fit.size),
        "selection_rows": int(selection.size),
        "selected_epoch": result.selected_epoch,
        "selection_losses": list(result.selection_losses),
        "selection_is_unweighted": result.selection_is_unweighted,
        "protocol_sha256": protocol.sha256,
        "cache_sha256": cache_receipt["cache"]["sha256"],
        "engine": asdict(engine),
        "model_config": asdict(RegionNetConfig()),
        "runtime_environment": _runtime_environment(),
        "resources": resources,
        "raw_rows_opened": 0,
        "val_test_or_full_rows_opened": 0,
        "decision": "pilot_pass_freeze_epoch_for_all_outer_folds",
    }
    write_exclusive_json(output, receipt)
    return receipt


def _save_hgb_checkpoint(path: Path, estimator: Any, *, outer_fold: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PhaseBWorkerError("refusing to overwrite B0 HGB checkpoint")
    buffer = io.BytesIO()
    joblib.dump(
        {
            "schema": "axon_loop175_b0_hgb_checkpoint_v1",
            "outer_fold": outer_fold,
            "estimator": estimator,
        },
        buffer,
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(buffer.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def run_outer_fold(
    *,
    project_root: Path,
    source_closure_path: Path,
    cache_receipt_path: Path,
    pilot_receipt_path: Path | None,
    artifact_directory: Path,
    worker_receipt_path: Path,
    checkpoint_path: Path,
    arm: str,
    outer_fold: int,
    microbatch: int = 2,
    gradient_accumulation: int = 16,
) -> dict[str, object]:
    if arm not in {"A", *NEURAL_ARMS} or outer_fold not in range(5):
        raise PhaseBWorkerError("outer worker arm or fold is invalid")
    protocol, _closure, cache_receipt = _validate_closure_and_cache(
        project_root=project_root,
        source_closure_path=source_closure_path,
        cache_receipt_path=cache_receipt_path,
    )
    aligned = load_aligned_phase_b_data(project_root)
    fit_indices = np.flatnonzero(aligned.folds != outer_fold).astype(np.int64)
    holdout_indices = np.flatnonzero(aligned.folds == outer_fold).astype(np.int64)
    if fit_indices.size != 16_000 or holdout_indices.size != 4_000:
        raise PhaseBWorkerError("outer worker partition counts drifted")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    selected_epoch: int | None = None
    optimizer_steps = 0
    final_training_loss: float | None = None
    sample_weight_sum: float | None = None
    if arm == "A":
        estimator = fit_frozen_b0_hgb(aligned.b0_values[fit_indices], aligned.labels[fit_indices])
        scores = predict_b0_scores(estimator, aligned.b0_values[holdout_indices])
        model_commitment = _save_hgb_checkpoint(checkpoint_path, estimator, outer_fold=outer_fold)
    else:
        if pilot_receipt_path is None:
            raise PhaseBWorkerError("neural outer worker requires a pilot receipt")
        pilot = load_json_object(pilot_receipt_path)
        if (
            pilot.get("schema") != "axon_loop175_epoch_pilot_receipt_v1"
            or pilot.get("arm") != arm
            or pilot.get("protocol_sha256") != protocol.sha256
            or pilot.get("decision") != "pilot_pass_freeze_epoch_for_all_outer_folds"
        ):
            raise PhaseBWorkerError("neural pilot receipt drifted")
        selected_epoch = int(pilot["selected_epoch"])
        payload = _load_payload(project_root, cache_receipt)
        weights = _full_e_weight_vector(payload, outer_fold=outer_fold) if arm == "E" else None
        engine = EngineConfig(
            seed=41,
            microbatch=microbatch,
            gradient_accumulation=gradient_accumulation,
            device="cuda",
        )
        neural = train_neural_arm(
            payload,
            fit_indices=fit_indices,
            holdout_indices=holdout_indices,
            arm=arm,
            frozen_epoch=selected_epoch,
            model_config=RegionNetConfig(),
            engine=engine,
            sample_weights=weights,
            protocol_sha256=protocol.sha256 if arm == "D" else None,
            outer_fold=outer_fold,
            checkpoint_path=checkpoint_path,
        )
        scores = neural.holdout_scores
        optimizer_steps = neural.optimizer_steps
        final_training_loss = neural.final_training_loss
        sample_weight_sum = neural.sample_weight_sum
        if neural.checkpoint_sha256 is None:
            raise PhaseBWorkerError("neural worker did not seal its checkpoint")
        model_commitment = neural.checkpoint_sha256

    resources = _resource_payload(started=started, output_paths=(checkpoint_path,))
    runtime_receipt: dict[str, object] = {
        "schema": "axon_loop175_arm_fold_worker_receipt_v1",
        "loop_id": "Loop175",
        "claim_scope": "train_only_outer_fit_and_holdout_score_not_val_test_or_full",
        "arm": arm,
        "fold": outer_fold,
        "seed": 41,
        "fit_rows": int(fit_indices.size),
        "holdout_rows": int(holdout_indices.size),
        "selected_epoch": selected_epoch,
        "optimizer_steps": optimizer_steps,
        "final_training_loss": final_training_loss,
        "sample_weight_sum": sample_weight_sum,
        "protocol_sha256": protocol.sha256,
        "cache_sha256": cache_receipt["cache"]["sha256"],
        "model_commitment": model_commitment,
        "checkpoint_path": str(checkpoint_path),
        "runtime_environment": _runtime_environment(),
        "resources": resources,
        "raw_rows_opened": 0,
        "val_test_or_full_rows_opened": 0,
        "decision": "arm_fold_outer_oof_complete",
    }
    runtime_commitment = _hash_payload(
        runtime_receipt,
        domain=b"axon_loop175_arm_fold_worker_receipt_v1\0",
    )
    runtime_receipt["runtime_commitment"] = runtime_commitment
    artifact = write_arm_fold_artifact(
        artifact_directory,
        arm=arm,
        fold=outer_fold,
        holdout_indices=holdout_indices,
        scores=scores,
        fit_count=int(fit_indices.size),
        protocol_commitment=protocol.sha256,
        cache_commitment=str(cache_receipt["cache"]["sha256"]),
        config_commitment=protocol.sha256,
        model_commitment=model_commitment,
        runtime_commitment=runtime_commitment,
    )
    runtime_receipt["artifact_numeric_commitment"] = artifact.metadata["numeric_commitment"]
    write_exclusive_json(worker_receipt_path, runtime_receipt)
    return runtime_receipt


__all__ = ["PhaseBWorkerError", "run_outer_fold", "run_pilot"]
