"""Immutable arm-fold OOF artifacts and seed-41 aggregation for Loop175."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.loop167_phase_b.contracts import canonical_json_bytes

from .phase_b_evaluation import (
    ARM_NAMES,
    DEFAULT_ROWS,
    DEFAULT_ROWS_PER_FOLD,
    evaluate_phase_b_oof,
)

ARM_FOLD_SCHEMA = "axon_loop175_arm_fold_oof_v1"
SEED41_RECEIPT_SCHEMA = "axon_loop175_seed41_oof_receipt_v1"
FOLD_COUNT = 5
SEED41 = 41
METADATA_FIELDS = frozenset(
    {
        "schema",
        "arm",
        "fold",
        "seed",
        "row_count",
        "fit_count",
        "holdout_count",
        "protocol_commitment",
        "cache_commitment",
        "config_commitment",
        "model_commitment",
        "runtime_commitment",
        "numeric_commitment",
    }
)
COMMITMENT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
NUMERIC_DOMAIN = b"axon_loop175_arm_fold_numeric_v1\0"


class PhaseBReceiptError(ValueError):
    """Raised when an arm-fold artifact or aggregate receipt is malformed."""


@dataclass(frozen=True, slots=True)
class ArmFoldArtifact:
    npz_path: Path
    metadata_path: Path
    arm: str
    fold: int
    seed: int
    holdout_indices: np.ndarray
    scores: np.ndarray
    metadata: Mapping[str, Any]


def _require_commitment(value: object, *, field: str) -> str:
    if not isinstance(value, str) or COMMITMENT_PATTERN.fullmatch(value) is None:
        raise PhaseBReceiptError(f"{field} must be a lowercase 64-character commitment")
    return value


def _require_regular_file(path: Path, *, label: str) -> Path:
    try:
        stat_result = path.lstat()
    except OSError as error:
        raise PhaseBReceiptError(f"{label} is missing or inaccessible") from error
    if path.is_symlink() or not path.is_file() or stat_result.st_size < 0:
        raise PhaseBReceiptError(f"{label} must be a regular file")
    return path


def _exclusive_write(path: Path, content: bytes, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise PhaseBReceiptError(f"refusing to overwrite {label}") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _normalize_indices(values: Sequence[Any], *, row_count: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype == np.dtype(bool):
        raise PhaseBReceiptError("holdout_indices must be a non-empty integer vector")
    if not np.issubdtype(array.dtype, np.integer):
        raise PhaseBReceiptError("holdout_indices must be an integer vector")
    normalized = np.ascontiguousarray(array, dtype=np.int64)
    if np.any(normalized < 0) or np.any(normalized >= row_count):
        raise PhaseBReceiptError("holdout_indices contain an out-of-range row")
    if np.unique(normalized).size != normalized.size:
        raise PhaseBReceiptError("holdout_indices contain duplicate rows")
    if normalized.size > 1 and np.any(np.diff(normalized) <= 0):
        raise PhaseBReceiptError("holdout_indices must be strictly increasing")
    return normalized


def _normalize_scores(values: Sequence[Any], *, rows: int) -> np.ndarray:
    try:
        scores = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise PhaseBReceiptError("scores must be a floating-point vector") from error
    if scores.ndim != 1 or scores.shape != (rows,):
        raise PhaseBReceiptError("scores must contain one value per holdout row")
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise PhaseBReceiptError("scores must be finite probabilities in [0, 1]")
    return np.ascontiguousarray(scores, dtype=np.float64)


def _numeric_commitment(holdout_indices: np.ndarray, scores: np.ndarray) -> str:
    digest = hashlib.sha256(NUMERIC_DOMAIN)
    for name, values in (("holdout_indices", holdout_indices), ("scores", scores)):
        normalized = np.ascontiguousarray(values)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(normalized.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(normalized.shape, dtype="<i8").tobytes())
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def artifact_paths(directory: Path | str, *, arm: str, fold: int) -> tuple[Path, Path]:
    if arm not in ARM_NAMES:
        raise PhaseBReceiptError("arm is not one of A-E")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(FOLD_COUNT):
        raise PhaseBReceiptError("fold must be an integer in 0..4")
    root = Path(directory)
    stem = f"arm_{arm}_fold_{fold}"
    return root / f"{stem}.npz", root / f"{stem}.json"


def _metadata(
    *,
    arm: str,
    fold: int,
    seed: int,
    row_count: int,
    fit_count: int,
    holdout_count: int,
    protocol_commitment: str,
    cache_commitment: str,
    config_commitment: str,
    model_commitment: str,
    runtime_commitment: str,
    numeric_commitment: str,
) -> dict[str, Any]:
    if arm not in ARM_NAMES:
        raise PhaseBReceiptError("arm is not one of A-E")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(FOLD_COUNT):
        raise PhaseBReceiptError("fold must be an integer in 0..4")
    if seed != SEED41:
        raise PhaseBReceiptError("arm-fold artifacts are frozen to seed 41")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise PhaseBReceiptError("row_count must be a positive integer")
    if isinstance(fit_count, bool) or not isinstance(fit_count, int) or fit_count < 0:
        raise PhaseBReceiptError("fit_count must be a non-negative integer")
    if isinstance(holdout_count, bool) or not isinstance(holdout_count, int) or holdout_count <= 0:
        raise PhaseBReceiptError("holdout_count must be a positive integer")
    if fit_count + holdout_count != row_count:
        raise PhaseBReceiptError("fit_count and holdout_count must cover row_count")
    commitments = {
        "protocol_commitment": protocol_commitment,
        "cache_commitment": cache_commitment,
        "config_commitment": config_commitment,
        "model_commitment": model_commitment,
        "runtime_commitment": runtime_commitment,
        "numeric_commitment": numeric_commitment,
    }
    for field, value in commitments.items():
        _require_commitment(value, field=field)
    return {
        "schema": ARM_FOLD_SCHEMA,
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "row_count": row_count,
        "fit_count": fit_count,
        "holdout_count": holdout_count,
        **commitments,
    }


def write_arm_fold_artifact(
    directory: Path | str,
    *,
    arm: str,
    fold: int,
    holdout_indices: Sequence[Any],
    scores: Sequence[Any],
    fit_count: int,
    seed: int = SEED41,
    row_count: int = DEFAULT_ROWS,
    protocol_commitment: str,
    cache_commitment: str,
    config_commitment: str,
    model_commitment: str,
    runtime_commitment: str,
) -> ArmFoldArtifact:
    """Write one immutable numeric arm-fold artifact and canonical sidecar."""

    indices = _normalize_indices(holdout_indices, row_count=row_count)
    probabilities = _normalize_scores(scores, rows=indices.size)
    metadata = _metadata(
        arm=arm,
        fold=fold,
        seed=seed,
        row_count=row_count,
        fit_count=fit_count,
        holdout_count=int(indices.size),
        protocol_commitment=protocol_commitment,
        cache_commitment=cache_commitment,
        config_commitment=config_commitment,
        model_commitment=model_commitment,
        runtime_commitment=runtime_commitment,
        numeric_commitment=_numeric_commitment(indices, probabilities),
    )
    npz_path, metadata_path = artifact_paths(directory, arm=arm, fold=fold)
    if npz_path.exists() or npz_path.is_symlink() or metadata_path.exists() or metadata_path.is_symlink():
        raise PhaseBReceiptError("refusing to overwrite arm-fold artifact")
    buffer = io.BytesIO()
    np.savez(
        buffer,
        holdout_indices=indices,
        scores=probabilities,
    )
    _exclusive_write(npz_path, buffer.getvalue(), label="arm-fold numeric artifact")
    _exclusive_write(metadata_path, canonical_json_bytes(metadata), label="arm-fold metadata")
    return ArmFoldArtifact(
        npz_path=npz_path,
        metadata_path=metadata_path,
        arm=arm,
        fold=fold,
        seed=seed,
        holdout_indices=indices,
        scores=probabilities,
        metadata=metadata,
    )


def _parse_metadata(path: Path) -> dict[str, Any]:
    raw = _require_regular_file(path, label="arm-fold metadata").read_bytes()
    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseBReceiptError("arm-fold metadata is not valid canonical JSON") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise PhaseBReceiptError("arm-fold metadata is not canonical JSON")
    if set(payload) != METADATA_FIELDS:
        raise PhaseBReceiptError("arm-fold metadata fields drifted")
    if any("path" in str(key).casefold() or "sha" in str(key).casefold() for key in payload):
        raise PhaseBReceiptError("arm-fold metadata must not contain path or SHA fields")
    return payload


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise PhaseBReceiptError(f"metadata repeats key: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite(value: str) -> object:
    raise PhaseBReceiptError(f"metadata contains non-finite value: {value}")


def load_arm_fold_artifact(
    npz_path: Path | str,
    metadata_path: Path | str,
    *,
    expected_row_count: int = DEFAULT_ROWS,
) -> ArmFoldArtifact:
    """Load and validate one numeric artifact/metadata pair."""

    numeric_path = _require_regular_file(Path(npz_path), label="arm-fold numeric artifact")
    metadata_file = Path(metadata_path)
    metadata = _parse_metadata(metadata_file)
    if metadata["schema"] != ARM_FOLD_SCHEMA:
        raise PhaseBReceiptError("arm-fold metadata schema drifted")
    if metadata["row_count"] != expected_row_count:
        raise PhaseBReceiptError("arm-fold row_count drifted")
    arm = metadata["arm"]
    fold = metadata["fold"]
    seed = metadata["seed"]
    if (
        not isinstance(arm, str)
        or arm not in ARM_NAMES
        or isinstance(fold, bool)
        or not isinstance(fold, int)
        or fold not in range(FOLD_COUNT)
    ):
        raise PhaseBReceiptError("arm-fold arm or fold is invalid")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != SEED41:
        raise PhaseBReceiptError("arm-fold seed drifted")
    if (
        isinstance(metadata["row_count"], bool)
        or not isinstance(metadata["row_count"], int)
        or metadata["row_count"] <= 0
    ):
        raise PhaseBReceiptError("arm-fold row_count is invalid")
    if any(
        isinstance(metadata[field], bool)
        or not isinstance(metadata[field], int)
        or metadata[field] < 0
        for field in ("fit_count", "holdout_count")
    ):
        raise PhaseBReceiptError("arm-fold fit/holdout counts are invalid")
    if metadata["fit_count"] + metadata["holdout_count"] != expected_row_count:
        raise PhaseBReceiptError("arm-fold counts do not cover the denominator")
    for field in (
        "protocol_commitment",
        "cache_commitment",
        "config_commitment",
        "model_commitment",
        "runtime_commitment",
        "numeric_commitment",
    ):
        _require_commitment(metadata[field], field=field)
    try:
        with np.load(numeric_path, allow_pickle=False) as archive:
            if set(archive.files) != {"holdout_indices", "scores"}:
                raise PhaseBReceiptError("arm-fold numeric members drifted")
            indices = _normalize_indices(archive["holdout_indices"], row_count=expected_row_count)
            probabilities = _normalize_scores(archive["scores"], rows=indices.size)
    except PhaseBReceiptError:
        raise
    except (OSError, ValueError, EOFError) as error:
        raise PhaseBReceiptError("arm-fold numeric artifact cannot be read") from error
    if indices.size != metadata["holdout_count"]:
        raise PhaseBReceiptError("arm-fold holdout_count does not match numeric rows")
    if metadata["numeric_commitment"] != _numeric_commitment(indices, probabilities):
        raise PhaseBReceiptError("arm-fold numeric commitment drifted")
    indices.setflags(write=False)
    probabilities.setflags(write=False)
    return ArmFoldArtifact(
        npz_path=numeric_path,
        metadata_path=metadata_file,
        arm=arm,
        fold=fold,
        seed=seed,
        holdout_indices=indices,
        scores=probabilities,
        metadata=metadata,
    )


def _normalize_labels_and_folds(
    labels: Sequence[Any],
    folds: Sequence[Any],
    *,
    expected_rows: int,
    expected_rows_per_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels)
    folds_array = np.asarray(folds)
    if (
        labels_array.shape != (expected_rows,)
        or labels_array.dtype == np.dtype(bool)
        or not np.issubdtype(labels_array.dtype, np.integer)
        or not np.isin(labels_array, (0, 1)).all()
    ):
        raise PhaseBReceiptError("labels do not match the frozen denominator")
    if (
        folds_array.shape != (expected_rows,)
        or folds_array.dtype == np.dtype(bool)
        or not np.issubdtype(folds_array.dtype, np.integer)
        or not np.isin(folds_array, np.arange(FOLD_COUNT)).all()
    ):
        raise PhaseBReceiptError("fold authority is invalid")
    normalized_folds = np.ascontiguousarray(folds_array, dtype=np.int8)
    if not np.all(np.bincount(normalized_folds, minlength=FOLD_COUNT) == expected_rows_per_fold):
        raise PhaseBReceiptError("fold authority counts drifted")
    return np.ascontiguousarray(labels_array, dtype=np.uint8), normalized_folds


def _commitment_set(artifacts: Sequence[ArmFoldArtifact], field: str) -> str:
    values = {str(artifact.metadata[field]) for artifact in artifacts}
    if len(values) != 1:
        raise PhaseBReceiptError(f"arm-fold {field} commitments are inconsistent")
    return values.pop()


def aggregate_seed41_receipt(
    directory: Path | str,
    *,
    labels: Sequence[Any],
    folds: Sequence[Any],
    component_ids: Sequence[Any],
    protocol_sha256: str,
    runtime: Mapping[str, Any],
    output: Path | str,
    expected_rows: int = DEFAULT_ROWS,
    expected_rows_per_fold: int = DEFAULT_ROWS_PER_FOLD,
    bootstrap_replicates: int = 200_000,
) -> dict[str, Any]:
    """Aggregate exactly one A-E outer-OOF vector and write one receipt."""

    labels_array, folds_array = _normalize_labels_and_folds(
        labels,
        folds,
        expected_rows=expected_rows,
        expected_rows_per_fold=expected_rows_per_fold,
    )
    if not COMMITMENT_PATTERN.fullmatch(protocol_sha256):
        raise PhaseBReceiptError("protocol_sha256 must be a lowercase 64-character commitment")
    artifact_list: list[ArmFoldArtifact] = []
    scores_by_arm: dict[str, np.ndarray] = {}
    for arm in ARM_NAMES:
        scores = np.full(expected_rows, np.nan, dtype=np.float64)
        seen = np.zeros(expected_rows, dtype=bool)
        for fold in range(FOLD_COUNT):
            numeric_path, metadata_path = artifact_paths(directory, arm=arm, fold=fold)
            artifact = load_arm_fold_artifact(
                numeric_path,
                metadata_path,
                expected_row_count=expected_rows,
            )
            expected_indices = np.flatnonzero(folds_array == fold).astype(np.int64)
            if artifact.arm != arm or artifact.fold != fold or artifact.seed != SEED41:
                raise PhaseBReceiptError("arm-fold identity does not match its authority slot")
            if not np.array_equal(artifact.holdout_indices, expected_indices):
                raise PhaseBReceiptError("arm-fold holdout_indices do not match fold authority")
            if artifact.metadata["protocol_commitment"] != protocol_sha256:
                raise PhaseBReceiptError("arm-fold protocol commitment drifted")
            if np.any(seen[artifact.holdout_indices]):
                raise PhaseBReceiptError("arm OOF vector contains duplicate rows")
            scores[artifact.holdout_indices] = artifact.scores
            seen[artifact.holdout_indices] = True
            artifact_list.append(artifact)
        if not np.all(seen) or not np.isfinite(scores).all():
            raise PhaseBReceiptError("arm OOF vector has missing or non-finite rows")
        scores.setflags(write=False)
        scores_by_arm[arm] = scores

    evaluation = evaluate_phase_b_oof(
        labels_array,
        folds_array,
        component_ids,
        scores_by_arm,
        protocol_sha256=protocol_sha256,
        expected_rows=expected_rows,
        expected_rows_per_fold=expected_rows_per_fold,
        bootstrap_replicates=bootstrap_replicates,
        runtime=runtime,
    )
    receipt: dict[str, Any] = {
        "schema": SEED41_RECEIPT_SCHEMA,
        "loop_id": "Loop175",
        "claim_scope": "train_only_outer_oof_not_val_test10k_or_full_test",
        "decision": evaluation["decision"],
        "seed": SEED41,
        "rows": expected_rows,
        "folds": FOLD_COUNT,
        "arms": list(ARM_NAMES),
        "protocol_commitment": protocol_sha256,
        "cache_commitment": _commitment_set(artifact_list, "cache_commitment"),
        "config_commitment": _commitment_set(artifact_list, "config_commitment"),
        "model_commitments": {
            f"{artifact.arm}_fold_{artifact.fold}": artifact.metadata["model_commitment"]
            for artifact in artifact_list
        },
        "runtime_commitments": {
            f"{artifact.arm}_fold_{artifact.fold}": artifact.metadata["runtime_commitment"]
            for artifact in artifact_list
        },
        "artifact_count": len(artifact_list),
        "artifact_numeric_commitments": {
            f"{artifact.arm}_fold_{artifact.fold}": artifact.metadata["numeric_commitment"]
            for artifact in artifact_list
        },
        "evaluation": evaluation,
        "val_rows_opened": 0,
        "test10k_rows_opened": 0,
        "full_test_rows_opened": 0,
        "val_test_or_full_rows_opened": 0,
    }
    output_path = Path(output)
    _exclusive_write(output_path, canonical_json_bytes(receipt), label="seed41 receipt")
    return receipt


aggregate_phase_b_oof = aggregate_seed41_receipt
write_seed41_receipt = aggregate_seed41_receipt
write_arm_fold_oof = write_arm_fold_artifact


__all__ = [
    "ARM_FOLD_SCHEMA",
    "ARM_NAMES",
    "ArmFoldArtifact",
    "PhaseBReceiptError",
    "SEED41_RECEIPT_SCHEMA",
    "aggregate_phase_b_oof",
    "aggregate_seed41_receipt",
    "artifact_paths",
    "load_arm_fold_artifact",
    "write_arm_fold_oof",
    "write_arm_fold_artifact",
    "write_seed41_receipt",
]
