"""Static-only preflight validation for the future Loop167 Phase-B controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import PhaseBContractError, require_canonical_json, verify_file_binding
from .runtime_lock import validate_runtime_lock

PHASE_A_BINDING_NAMES = (
    "proposal",
    "authorization",
    "semantic_delta_mapping",
    "frozen_deduplicated_baseline_allowlist",
    "source_semantics_addendum",
    "source_closure",
    "static_decision",
)


@dataclass(frozen=True)
class StaticPreflightReceipt:
    protocol_sha256: str
    source_closure_sha256: str
    runtime_lock_sha256: str
    raw_open_attempts: int


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PhaseBContractError(f"{label} must be an object")
    return value


def _verify_phase_a_bindings(root: Path, bindings: object) -> None:
    values = _require_mapping(bindings, label="phase_a_bindings")
    if set(values) != set(PHASE_A_BINDING_NAMES):
        raise PhaseBContractError("Phase-A binding names drifted")
    for name in PHASE_A_BINDING_NAMES:
        verify_file_binding(root, values[name], label=f"phase_a.{name}")


def validate_static_preflight(
    root: Path,
    *,
    source_closure_binding: Mapping[str, str],
    controller_binding: Mapping[str, str],
    canonical_argv: Sequence[str],
) -> StaticPreflightReceipt:
    closure_path, closure_sha256 = verify_file_binding(root, dict(source_closure_binding), label="source_closure")
    closure = require_canonical_json(closure_path)
    expected_closure_keys = {
        "schema",
        "loop_id",
        "scope",
        "phase_a_bindings",
        "phase_b_protocol",
        "phase_b_protocol_addendum",
        "runtime_lock",
        "source_files",
        "static_preflight_ready",
        "phase_b_raw_execution_ready",
        "remaining_execution_blockers",
    }
    if set(closure) != expected_closure_keys:
        raise PhaseBContractError("Phase-B source closure fields drifted")
    if closure["schema"] != "axon_loop167_phase_b_source_closure_v1":
        raise PhaseBContractError("Phase-B source closure schema drifted")
    if closure["loop_id"] != "loop167_ember_v3_novel_delta":
        raise PhaseBContractError("Phase-B source closure loop id drifted")
    if closure["scope"] != "static_preflight_only_no_raw_checkpoint_prediction_or_fit_access":
        raise PhaseBContractError("Phase-B source closure scope drifted")
    if closure["static_preflight_ready"] is not True or closure["phase_b_raw_execution_ready"] is not False:
        raise PhaseBContractError("Phase-B source closure execution state drifted")
    _verify_phase_a_bindings(root, closure["phase_a_bindings"])

    protocol_path, protocol_sha256 = verify_file_binding(root, closure["phase_b_protocol"], label="phase_b_protocol")
    protocol = require_canonical_json(protocol_path)
    if protocol.get("schema") != "axon_loop167_phase_b_protocol_v1":
        raise PhaseBContractError("Phase-B protocol schema drifted")
    _verify_phase_a_bindings(root, protocol.get("phase_a_bindings"))
    if protocol["phase_a_bindings"] != closure["phase_a_bindings"]:
        raise PhaseBContractError("Phase-A bindings differ between protocol and closure")

    addendum_path, _ = verify_file_binding(
        root,
        closure["phase_b_protocol_addendum"],
        label="phase_b_protocol_addendum",
    )
    addendum = require_canonical_json(addendum_path)
    if addendum.get("schema") != "axon_loop167_phase_b_protocol_addendum_v1":
        raise PhaseBContractError("Phase-B protocol addendum schema drifted")
    if addendum.get("parent_phase_b_protocol") != closure["phase_b_protocol"]:
        raise PhaseBContractError("Phase-B protocol addendum parent binding drifted")

    runtime_lock_path, runtime_lock_sha256 = verify_file_binding(root, closure["runtime_lock"], label="runtime_lock")
    runtime_lock = require_canonical_json(runtime_lock_path)
    validate_runtime_lock(
        root,
        runtime_lock,
        controller_binding=controller_binding,
        canonical_argv=canonical_argv,
    )

    source_files = closure["source_files"]
    if not isinstance(source_files, list) or not source_files:
        raise PhaseBContractError("Phase-B source closure has no source files")
    observed_paths: set[str] = set()
    for source_binding in source_files:
        source_path, _ = verify_file_binding(root, source_binding, label="source_file")
        relative_path = source_path.relative_to(root.resolve(strict=True)).as_posix()
        if relative_path in observed_paths:
            raise PhaseBContractError("Phase-B source closure repeats a source file")
        observed_paths.add(relative_path)
    controller_path, controller_sha256 = verify_file_binding(root, dict(controller_binding), label="controller")
    if {
        "path": controller_path.relative_to(root.resolve(strict=True)).as_posix(),
        "sha256": controller_sha256,
    } not in source_files:
        raise PhaseBContractError("Controller is not bound by Phase-B source closure")
    return StaticPreflightReceipt(
        protocol_sha256=protocol_sha256,
        source_closure_sha256=closure_sha256,
        runtime_lock_sha256=runtime_lock_sha256,
        raw_open_attempts=0,
    )
