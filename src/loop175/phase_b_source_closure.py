"""Canonical static source closure for Loop175 Phase B."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from src.loop167_phase_b.contracts import canonical_json_bytes

from .phase_b_contract import (
    PROTOCOL_RELATIVE_PATH,
    PhaseBContractError,
    load_json_object,
    load_phase_b_protocol,
    validate_bound_evidence,
)

SOURCE_CLOSURE_SCHEMA = "axon_loop175_phase_b_source_closure_v1"
SOURCE_CLOSURE_SCOPE = (
    "static_aggregate_closure_no_raw_pe_val_test_full_prediction_fit_or_training_access"
)
LOWERCASE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
REPARSE_POINT_ATTRIBUTE = 0x0400
MAXIMUM_BOUND_FILE_BYTES = 30 * 1024**3
FULL_SHAPE_GPU_SMOKE_RELATIVE_PATH = (
    "reports/roadmap_9997/loop175/phase_b_full_shape_gpu_smoke.json"
)
SOURCE_RELATIVE_PATHS = (
    "src/loop175/phase_b_contract.py",
    "src/loop175/phase_b_data.py",
    "src/loop175/phase_b_cache_builder.py",
    "src/loop175/model.py",
    "src/loop175/phase_b_training.py",
    "src/loop175/phase_b_engine.py",
    "src/loop175/phase_b_evaluation.py",
    "src/loop175/phase_b_receipt.py",
    "src/loop175/phase_b_worker.py",
    "src/loop175/phase_b_controller.py",
    "src/loop175/resource_guard.py",
    "src/loop175/phase_b_source_closure.py",
    "scripts/run_loop175_full_shape_gpu_smoke.py",
    "scripts/seal_loop175_phase_b_source_closure.py",
    "scripts/run_loop175_phase_b_region_cache.py",
    "scripts/run_loop175_phase_b_worker.py",
    "scripts/run_loop175_seed41_controller.py",
    "tests/test_loop175_phase_b_contract.py",
    "tests/test_loop175_phase_b_data.py",
    "tests/test_loop175_phase_b_cache_builder.py",
    "tests/test_loop175_phase_b_training.py",
    "tests/test_loop175_phase_b_engine.py",
    "tests/test_loop175_phase_b_evaluation.py",
    "tests/test_loop175_phase_b_receipt.py",
    "tests/test_loop175_phase_b_controller.py",
    "tests/test_loop175_phase_b_source_closure.py",
)


class PhaseBSourceClosureError(RuntimeError):
    """Raised when the static Phase-B source closure cannot be proven."""


@dataclass(frozen=True, slots=True)
class SourceClosureReceipt:
    path: Path
    sha256: str
    size_bytes: int
    source_count: int


def _require_lowercase_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or LOWERCASE_SHA256.fullmatch(value) is None:
        raise PhaseBSourceClosureError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _has_reparse_point(path: Path) -> bool:
    try:
        return bool(int(getattr(path.lstat(), "st_file_attributes", 0)) & REPARSE_POINT_ATTRIBUTE)
    except OSError as error:
        raise PhaseBSourceClosureError(f"cannot inspect path: {path}") from error


def _safe_root(project_root: Path | str) -> Path:
    original = Path(project_root).absolute()
    try:
        if original.is_symlink() or _has_reparse_point(original):
            raise PhaseBSourceClosureError("project root must not be a link or reparse point")
        root = original.resolve(strict=True)
    except OSError as error:
        raise PhaseBSourceClosureError("project root is missing or inaccessible") from error
    if not root.is_dir():
        raise PhaseBSourceClosureError("project root must be a directory")
    return root


def _relative_parts(relative_path: str | Path) -> tuple[str, ...]:
    text = str(relative_path)
    if not text or "\\" in text:
        raise PhaseBSourceClosureError("closure paths must use nonempty POSIX-relative syntax")
    candidate = PurePosixPath(text)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise PhaseBSourceClosureError("closure path escapes or is not canonical")
    return candidate.parts


def _safe_project_file(project_root: Path | str, relative_path: str | Path) -> Path:
    root = _safe_root(project_root)
    parts = _relative_parts(relative_path)
    current = root
    for part in parts:
        current = current / part
        try:
            result = current.lstat()
        except OSError as error:
            raise PhaseBSourceClosureError(f"required closure file is missing: {relative_path}") from error
        if stat.S_ISLNK(result.st_mode) or _has_reparse_point(current):
            raise PhaseBSourceClosureError(f"closure path uses a link or reparse point: {relative_path}")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PhaseBSourceClosureError(f"closure path escapes project root: {relative_path}") from error
    if not resolved.is_file():
        raise PhaseBSourceClosureError(f"closure input is not a regular file: {relative_path}")
    return resolved


def _sha256_file(path: Path, *, maximum_bytes: int = MAXIMUM_BOUND_FILE_BYTES) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            if size > maximum_bytes:
                raise PhaseBSourceClosureError(f"closure input exceeds byte limit: {path}")
            digest.update(block)
    return digest.hexdigest(), size


def _binding(
    project_root: Path | str,
    relative_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    path = _safe_project_file(project_root, relative_path)
    observed_sha256, size_bytes = _sha256_file(path)
    if expected_sha256 is not None and observed_sha256 != _require_lowercase_sha256(
        expected_sha256,
        field=f"{relative_path}.sha256",
    ):
        raise PhaseBSourceClosureError(f"closure SHA-256 drifted: {relative_path}")
    return {
        "path": PurePosixPath(str(relative_path)).as_posix(),
        "sha256": observed_sha256,
        "bytes": size_bytes,
    }


def _prefix_binding(
    project_root: Path | str,
    relative_path: str | Path,
    *,
    prefix_bytes: int,
    expected_prefix_sha256: str,
) -> dict[str, object]:
    if isinstance(prefix_bytes, bool) or not isinstance(prefix_bytes, int) or prefix_bytes <= 0:
        raise PhaseBSourceClosureError("Train prefix byte count is invalid")
    expected_sha = _require_lowercase_sha256(
        expected_prefix_sha256,
        field="canonical_train_prefix.sha256",
    )
    path = _safe_project_file(project_root, relative_path)
    with path.open("rb") as handle:
        prefix = handle.read(prefix_bytes)
    if len(prefix) != prefix_bytes or hashlib.sha256(prefix).hexdigest() != expected_sha:
        raise PhaseBSourceClosureError("canonical Train prefix binding drifted")
    return {
        "path": PurePosixPath(str(relative_path)).as_posix(),
        "sha256": expected_sha,
        "bytes": prefix_bytes,
        "rows": 20_000,
        "read_scope": "exact_train_prefix_only_no_later_split_bytes",
    }


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PhaseBSourceClosureError(f"{field} must be an object")
    return value


def build_phase_b_source_closure(project_root: Path | str) -> dict[str, object]:
    """Hash only frozen static/Train inputs and produce an aggregate receipt payload."""

    root = _safe_root(project_root)
    try:
        protocol = load_phase_b_protocol(root)
        validate_bound_evidence(root, protocol)
    except PhaseBContractError as error:
        raise PhaseBSourceClosureError("Phase-B protocol or bound evidence drifted") from error
    protocol_payload = protocol.payload
    evidence = _mapping(protocol_payload.get("evidence_bindings"), field="evidence_bindings")
    inputs = _mapping(protocol_payload.get("inputs"), field="inputs")
    fold = _mapping(inputs.get("fold_manifest"), field="inputs.fold_manifest")
    canonical_split = _mapping(inputs.get("canonical_split"), field="inputs.canonical_split")
    b0_cache = _mapping(inputs.get("b0_cache"), field="inputs.b0_cache")

    evidence_bindings: dict[str, object] = {}
    for name in ("proposal", "phase0_receipt", "phase_a_receipt", "execution_plan"):
        binding = _mapping(evidence.get(name), field=f"evidence_bindings.{name}")
        if set(binding) != {"path", "sha256"}:
            raise PhaseBSourceClosureError(f"evidence binding fields drifted: {name}")
        evidence_bindings[name] = _binding(
            root,
            str(binding["path"]),
            expected_sha256=str(binding["sha256"]),
        )
    smoke_path = _safe_project_file(root, FULL_SHAPE_GPU_SMOKE_RELATIVE_PATH)
    try:
        smoke = load_json_object(smoke_path)
    except PhaseBContractError as error:
        raise PhaseBSourceClosureError("full-shape GPU smoke report is invalid") from error
    smoke_inputs = _mapping(smoke.get("inputs"), field="full_shape_gpu_smoke.inputs")
    smoke_gates = _mapping(smoke.get("gates"), field="full_shape_gpu_smoke.gates")
    if (
        smoke.get("schema") != "axon_loop175_full_shape_gpu_smoke_v1"
        or smoke.get("decision")
        != "full_shape_resource_gate_pass_phase_b_implementation_may_continue"
        or smoke_inputs.get("raw_rows_opened") != 0
        or smoke_inputs.get("val_test_or_full_rows_opened") != 0
        or smoke_gates.get("passed") is not True
    ):
        raise PhaseBSourceClosureError("full-shape GPU smoke evidence did not pass safely")
    evidence_bindings["full_shape_gpu_smoke"] = _binding(
        root,
        FULL_SHAPE_GPU_SMOKE_RELATIVE_PATH,
    )

    input_bindings = {
        "fold_manifest": {
            **_binding(root, str(fold["path"]), expected_sha256=str(fold["sha256"])),
            "rows": 20_000,
            "folds": 5,
            "split_role": "train",
        },
        "canonical_train_prefix": _prefix_binding(
            root,
            str(canonical_split["path"]),
            prefix_bytes=int(canonical_split["train_prefix_bytes"]),
            expected_prefix_sha256=str(canonical_split["train_prefix_sha256"]),
        ),
        "b0_cache": {
            **_binding(
                root,
                str(b0_cache["path"]),
                expected_sha256=str(b0_cache["sha256"]),
            ),
            "array": "b0_values",
            "dtype": "float32",
            "shape": [20_000, 571],
            "missing_indicators_used": False,
        },
    }
    source_files = [_binding(root, path) for path in SOURCE_RELATIVE_PATHS]
    if len({str(binding["path"]) for binding in source_files}) != len(SOURCE_RELATIVE_PATHS):
        raise PhaseBSourceClosureError("source closure repeats a source path")

    return {
        "schema": SOURCE_CLOSURE_SCHEMA,
        "loop_id": "Loop175",
        "claim_scope": SOURCE_CLOSURE_SCOPE,
        "protocol": {
            "path": PROTOCOL_RELATIVE_PATH.as_posix(),
            "sha256": protocol.sha256,
            "bytes": protocol.path.stat().st_size,
        },
        "evidence_bindings": evidence_bindings,
        "input_bindings": input_bindings,
        "source_files": source_files,
        "aggregate_boundaries": {
            "identity_fields_persisted": False,
            "raw_pe_files_opened": 0,
            "val_rows_opened": 0,
            "test10k_rows_opened": 0,
            "legacy_full_rows_opened": 0,
            "sealed_window_rows_opened": 0,
            "prediction_rows_opened": 0,
            "model_fits": 0,
            "training_runs": 0,
        },
        "ready_for": {
            "source_closure": True,
            "full_train_region_cache": False,
            "seed41_epoch_pilot": False,
            "seed41_outer_oof": False,
            "seed42_43": False,
            "val_test_full_or_promotion": False,
        },
        "decision": (
            "source_closure_pass_cache_resource_guard_authorization_and_lease_still_required"
        ),
    }


def validate_phase_b_source_closure(
    project_root: Path | str,
    payload: Mapping[str, object],
) -> None:
    """Require exact equality with a freshly rebuilt closure."""

    if not isinstance(payload, Mapping):
        raise PhaseBSourceClosureError("source closure payload must be an object")
    expected_fields = {
        "schema",
        "loop_id",
        "claim_scope",
        "protocol",
        "evidence_bindings",
        "input_bindings",
        "source_files",
        "aggregate_boundaries",
        "ready_for",
        "decision",
    }
    if set(payload) != expected_fields:
        raise PhaseBSourceClosureError("source closure top-level fields drifted")
    expected = build_phase_b_source_closure(project_root)
    if dict(payload) != expected:
        raise PhaseBSourceClosureError("source closure content or SHA bindings drifted")


def _safe_output_path(project_root: Path | str, relative_path: str | Path) -> Path:
    root = _safe_root(project_root)
    parts = _relative_parts(relative_path)
    parent = root.joinpath(*parts[:-1])
    try:
        parent_resolved = parent.resolve(strict=True)
    except OSError as error:
        raise PhaseBSourceClosureError("source closure output parent must already exist") from error
    try:
        parent_resolved.relative_to(root)
    except ValueError as error:
        raise PhaseBSourceClosureError("source closure output escapes project root") from error
    current = root
    for part in parts[:-1]:
        current = current / part
        if current.is_symlink() or _has_reparse_point(current):
            raise PhaseBSourceClosureError("source closure output parent uses a link")
    output = parent_resolved / parts[-1]
    if output.exists() or output.is_symlink():
        raise PhaseBSourceClosureError("refusing to overwrite source closure receipt")
    return output


def _write_exclusive_canonical(
    project_root: Path | str,
    relative_path: str | Path,
    payload: Mapping[str, object],
) -> SourceClosureReceipt:
    output = _safe_output_path(project_root, relative_path)
    content = canonical_json_bytes(dict(payload))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(output, flags, 0o600)
    except FileExistsError as error:
        raise PhaseBSourceClosureError("refusing to overwrite source closure receipt") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return SourceClosureReceipt(
        path=output,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        source_count=len(payload.get("source_files", [])),
    )


def seal_phase_b_source_closure(
    project_root: Path | str,
    output_relative_path: str | Path,
) -> SourceClosureReceipt:
    payload = build_phase_b_source_closure(project_root)
    validate_phase_b_source_closure(project_root, payload)
    return _write_exclusive_canonical(project_root, output_relative_path, payload)
