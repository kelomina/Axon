"""Resumable seed-41 orchestration and resource gates for Loop175 Phase B."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Mapping

from src.loop167_phase_b.contracts import canonical_json_bytes

from .phase_b_contract import load_json_object, load_phase_b_protocol, sha256_file
from .phase_b_data import load_aligned_phase_b_data
from .phase_b_receipt import (
    ARM_NAMES,
    aggregate_seed41_receipt,
    artifact_paths,
    load_arm_fold_artifact,
)
from .phase_b_source_closure import validate_phase_b_source_closure

MAXIMUM_GPU_ALLOCATED_BYTES = int(6.5 * 1024**3)
MAXIMUM_RSS_BYTES = 11 * 1024**3
MAXIMUM_FOLD_WALL_SECONDS = 6 * 60 * 60
MAXIMUM_SEED_WALL_SECONDS = 30 * 60 * 60
MAXIMUM_NEW_DISK_BYTES = 30 * 1024**3
WORKER_TIMEOUT_SECONDS = 6 * 60 * 60


class PhaseBControllerError(RuntimeError):
    """Raised when a worker, artifact, or seed-level resource gate fails."""


def directory_size_bytes(path: Path) -> int:
    total = 0
    for root, _directories, files in os.walk(path):
        for filename in files:
            try:
                total += (Path(root) / filename).stat().st_size
            except OSError as error:
                raise PhaseBControllerError("cannot measure Phase-B output directory") from error
    return total


def _worker_receipt_commitment(payload: Mapping[str, Any]) -> str:
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"runtime_commitment", "artifact_numeric_commitment"}
    }
    return hashlib.sha256(
        b"axon_loop175_arm_fold_worker_receipt_v1\0" + canonical_json_bytes(core)
    ).hexdigest()


def validate_worker_receipt(
    path: Path,
    *,
    arm: str,
    fold: int,
    protocol_sha256: str,
    cache_sha256: str,
) -> dict[str, Any]:
    payload = load_json_object(path)
    if (
        payload.get("schema") != "axon_loop175_arm_fold_worker_receipt_v1"
        or payload.get("arm") != arm
        or payload.get("fold") != fold
        or payload.get("seed") != 41
        or payload.get("fit_rows") != 16_000
        or payload.get("holdout_rows") != 4_000
        or payload.get("protocol_sha256") != protocol_sha256
        or payload.get("cache_sha256") != cache_sha256
        or payload.get("raw_rows_opened") != 0
        or payload.get("val_test_or_full_rows_opened") != 0
        or payload.get("decision") != "arm_fold_outer_oof_complete"
    ):
        raise PhaseBControllerError("arm-fold worker receipt scope drifted")
    commitment = payload.get("runtime_commitment")
    if commitment != _worker_receipt_commitment(payload):
        raise PhaseBControllerError("arm-fold worker runtime commitment drifted")
    checkpoint = Path(str(payload.get("checkpoint_path")))
    if not checkpoint.is_file() or sha256_file(checkpoint) != payload.get("model_commitment"):
        raise PhaseBControllerError("arm-fold checkpoint commitment drifted")
    resources = payload.get("resources")
    if not isinstance(resources, Mapping):
        raise PhaseBControllerError("arm-fold worker lacks resource evidence")
    for field in ("wall_seconds", "rss_bytes", "gpu_allocated_bytes", "new_disk_bytes"):
        if not isinstance(resources.get(field), int | float) or resources[field] < 0:
            raise PhaseBControllerError(f"arm-fold resource field is invalid: {field}")
    return payload


def validate_pilot_receipt(
    path: Path,
    *,
    arm: str,
    protocol_sha256: str,
    cache_sha256: str,
) -> dict[str, Any]:
    payload = load_json_object(path)
    if (
        payload.get("schema") != "axon_loop175_epoch_pilot_receipt_v1"
        or payload.get("arm") != arm
        or payload.get("seed") != 41
        or payload.get("outer_fold_never_read") != 0
        or payload.get("pilot_fit_folds") != [2, 3, 4]
        or payload.get("inner_selection_fold") != 1
        or payload.get("pilot_fit_rows") != 12_000
        or payload.get("selection_rows") != 4_000
        or payload.get("protocol_sha256") != protocol_sha256
        or payload.get("cache_sha256") != cache_sha256
        or payload.get("raw_rows_opened") != 0
        or payload.get("val_test_or_full_rows_opened") != 0
        or payload.get("decision") != "pilot_pass_freeze_epoch_for_all_outer_folds"
    ):
        raise PhaseBControllerError("epoch pilot receipt scope drifted")
    selected_epoch = payload.get("selected_epoch")
    losses = payload.get("selection_losses")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or not 1 <= selected_epoch <= 12
        or not isinstance(losses, list)
        or len(losses) != 12
    ):
        raise PhaseBControllerError("epoch pilot selection result drifted")
    return payload


def _run_command(command: Sequence[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    try:
        return subprocess.run(
            list(command),
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise PhaseBControllerError("Loop175 worker exceeded its hard timeout") from error


def _failure_type(path: Path) -> str:
    if not path.is_file():
        return "missing_failure_receipt"
    failure = load_json_object(path)
    return str(failure.get("error_type", "unknown"))


def _run_with_oom_retry(
    base_command: list[str],
    *,
    first_failure: Path,
    second_failure: Path,
    success_exists: Callable[[], bool],
) -> None:
    if success_exists():
        return
    first_stderr = ""
    if not first_failure.exists():
        first = _run_command(
            [
                *base_command,
                "--failure-receipt",
                str(first_failure),
                "--microbatch",
                "2",
                "--gradient-accumulation",
                "16",
            ],
            timeout=WORKER_TIMEOUT_SECONDS,
        )
        first_stderr = first.stderr
        if first.returncode == 0 and success_exists():
            return
    if _failure_type(first_failure) != "OutOfMemoryError":
        raise PhaseBControllerError(
            f"Loop175 worker failed without an OOM retry allowance: {first_stderr[-2000:]}"
        )
    if second_failure.exists():
        raise PhaseBControllerError("Loop175 worker already consumed its sole OOM retry")
    second = _run_command(
        [
            *base_command,
            "--failure-receipt",
            str(second_failure),
            "--microbatch",
            "1",
            "--gradient-accumulation",
            "32",
        ],
        timeout=WORKER_TIMEOUT_SECONDS,
    )
    if second.returncode != 0 or not success_exists():
        raise PhaseBControllerError(
            f"Loop175 worker failed its sole OOM retry: {second.stderr[-2000:]}"
        )


def run_seed41_controller(
    *,
    project_root: Path,
    source_closure_path: Path,
    cache_receipt_path: Path,
    output_directory: Path,
    final_receipt_path: Path,
) -> dict[str, Any]:
    protocol = load_phase_b_protocol(project_root)
    source_closure = load_json_object(source_closure_path)
    validate_phase_b_source_closure(project_root, source_closure)
    cache_receipt = load_json_object(cache_receipt_path)
    if cache_receipt.get("decision") != "full_train_region_cache_gate_pass_seed41_pilot_may_begin":
        raise PhaseBControllerError("full-Train region cache did not pass its gate")
    cache_sha256 = str(cache_receipt["cache"]["sha256"])
    if final_receipt_path.is_file():
        existing = load_json_object(final_receipt_path)
        if (
            existing.get("schema") != "axon_loop175_seed41_oof_receipt_v1"
            or existing.get("protocol_commitment") != protocol.sha256
        ):
            raise PhaseBControllerError("existing seed41 receipt scope drifted")
        return existing
    output_directory.mkdir(parents=True, exist_ok=True)
    pilot_directory = output_directory / "pilots"
    artifact_directory = output_directory / "oof"
    worker_directory = output_directory / "workers"
    checkpoint_directory = output_directory / "checkpoints"
    failure_directory = output_directory / "failures"
    for directory in (
        pilot_directory,
        artifact_directory,
        worker_directory,
        checkpoint_directory,
        failure_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    worker_script = project_root / "scripts/run_loop175_phase_b_worker.py"

    pilot_receipts: dict[str, dict[str, Any]] = {}
    for arm in ("B", "C", "D", "E"):
        pilot_path = pilot_directory / f"arm_{arm}.json"
        if not pilot_path.is_file():
            base = [
                sys.executable,
                str(worker_script),
                "--execute",
                "--mode",
                "pilot",
                "--arm",
                arm,
                "--source-closure",
                str(source_closure_path),
                "--cache-receipt",
                str(cache_receipt_path),
                "--output",
                str(pilot_path),
            ]
            _run_with_oom_retry(
                base,
                first_failure=failure_directory / f"pilot_{arm}_attempt1.json",
                second_failure=failure_directory / f"pilot_{arm}_attempt2.json",
                success_exists=pilot_path.is_file,
            )
        pilot_receipts[arm] = validate_pilot_receipt(
            pilot_path,
            arm=arm,
            protocol_sha256=protocol.sha256,
            cache_sha256=cache_sha256,
        )
        if time.monotonic() - started > MAXIMUM_SEED_WALL_SECONDS:
            raise PhaseBControllerError("seed41 wall limit exceeded during epoch pilots")

    workers: list[dict[str, Any]] = []
    for fold in range(5):
        fold_started = time.monotonic()
        for arm in ARM_NAMES:
            worker_receipt = worker_directory / f"arm_{arm}_fold_{fold}.json"
            numeric_path, metadata_path = artifact_paths(artifact_directory, arm=arm, fold=fold)
            checkpoint_suffix = ".joblib" if arm == "A" else ".pt"
            checkpoint = checkpoint_directory / f"arm_{arm}_fold_{fold}{checkpoint_suffix}"
            if not (worker_receipt.is_file() and numeric_path.is_file() and metadata_path.is_file()):
                base = [
                    sys.executable,
                    str(worker_script),
                    "--execute",
                    "--mode",
                    "outer",
                    "--arm",
                    arm,
                    "--fold",
                    str(fold),
                    "--source-closure",
                    str(source_closure_path),
                    "--cache-receipt",
                    str(cache_receipt_path),
                    "--artifact-directory",
                    str(artifact_directory),
                    "--worker-receipt",
                    str(worker_receipt),
                    "--checkpoint",
                    str(checkpoint),
                ]
                if arm != "A":
                    base.extend(["--pilot-receipt", str(pilot_directory / f"arm_{arm}.json")])
                _run_with_oom_retry(
                    base,
                    first_failure=failure_directory / f"arm_{arm}_fold_{fold}_attempt1.json",
                    second_failure=failure_directory / f"arm_{arm}_fold_{fold}_attempt2.json",
                    success_exists=lambda: (
                        worker_receipt.is_file()
                        and numeric_path.is_file()
                        and metadata_path.is_file()
                    ),
                )
            worker = validate_worker_receipt(
                worker_receipt,
                arm=arm,
                fold=fold,
                protocol_sha256=protocol.sha256,
                cache_sha256=cache_sha256,
            )
            artifact = load_arm_fold_artifact(numeric_path, metadata_path)
            if artifact.metadata["runtime_commitment"] != worker["runtime_commitment"]:
                raise PhaseBControllerError("worker and OOF runtime commitments differ")
            if artifact.metadata["numeric_commitment"] != worker["artifact_numeric_commitment"]:
                raise PhaseBControllerError("worker and OOF numeric commitments differ")
            workers.append(worker)
        fold_wall = sum(
            float(worker["resources"]["wall_seconds"])
            for worker in workers
            if worker["fold"] == fold
        )
        if fold_wall > MAXIMUM_FOLD_WALL_SECONDS or time.monotonic() - fold_started > MAXIMUM_FOLD_WALL_SECONDS:
            raise PhaseBControllerError("one outer fold exceeded the six-hour aggregate wall limit")
        if time.monotonic() - started > MAXIMUM_SEED_WALL_SECONDS:
            raise PhaseBControllerError("seed41 exceeded its thirty-hour wall limit")

    pilot_resources = [pilot["resources"] for pilot in pilot_receipts.values()]
    accounted_seed_wall = sum(
        float(resource["wall_seconds"])
        for resource in [*pilot_resources, *(worker["resources"] for worker in workers)]
    )
    maximum_gpu = max(
        int(resource["gpu_allocated_bytes"])
        for resource in [*pilot_resources, *(worker["resources"] for worker in workers)]
    )
    maximum_rss = max(
        int(resource["rss_bytes"])
        for resource in [*pilot_resources, *(worker["resources"] for worker in workers)]
    )
    maximum_fold_wall = max(
        sum(
            float(worker["resources"]["wall_seconds"])
            for worker in workers
            if worker["fold"] == fold
        )
        for fold in range(5)
    )
    output_disk_bytes = directory_size_bytes(output_directory)
    cache_disk_bytes = int(cache_receipt["resources"]["maximum_new_disk_bytes"])
    new_disk_bytes = cache_disk_bytes + output_disk_bytes
    if (
        maximum_gpu > MAXIMUM_GPU_ALLOCATED_BYTES
        or maximum_rss > MAXIMUM_RSS_BYTES
        or maximum_fold_wall > MAXIMUM_FOLD_WALL_SECONDS
        or accounted_seed_wall > MAXIMUM_SEED_WALL_SECONDS
        or new_disk_bytes > MAXIMUM_NEW_DISK_BYTES
    ):
        raise PhaseBControllerError("seed41 aggregate resource contract failed")
    aligned = load_aligned_phase_b_data(project_root)
    coverage = cache_receipt["coverage"]
    runtime = {
        "coverage": coverage["coverage"],
        "class_coverage_gap": coverage["class_coverage_gap"],
        "silent_drops": coverage["silent_drops"],
        "oom": False,
        "timeout": False,
        "nonfinite": False,
        "gpu_allocated_bytes": maximum_gpu,
        "rss_bytes": maximum_rss,
        "new_disk_bytes": new_disk_bytes,
        "maximum_fold_wall_seconds": maximum_fold_wall,
        "seed_wall_seconds": accounted_seed_wall,
    }
    return aggregate_seed41_receipt(
        artifact_directory,
        labels=aligned.labels,
        folds=aligned.folds,
        component_ids=aligned.component_ids,
        protocol_sha256=protocol.sha256,
        runtime=runtime,
        output=final_receipt_path,
    )


__all__ = [
    "PhaseBControllerError",
    "directory_size_bytes",
    "run_seed41_controller",
    "validate_pilot_receipt",
    "validate_worker_receipt",
]
