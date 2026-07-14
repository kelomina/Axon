"""Static-only preflight v2 after the Loop167 isolated-runtime correction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import PhaseBContractError, require_canonical_json, verify_file_binding
from .preflight import PHASE_A_BINDING_NAMES
from .runtime_lock_v2 import validate_runtime_lock


@dataclass(frozen=True)
class StaticPreflightV2Receipt:
    protocol_sha256: str
    source_closure_sha256: str
    runtime_lock_sha256: str
    raw_open_attempts: int


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PhaseBContractError(f"{label} must be an object")
    return value


def _verify_phase_a_bindings(root: Path, bindings: object) -> None:
    values = _mapping(bindings, label="phase_a_bindings")
    if set(values) != set(PHASE_A_BINDING_NAMES):
        raise PhaseBContractError("Phase-A binding names drifted")
    for name in PHASE_A_BINDING_NAMES:
        verify_file_binding(root, values[name], label=f"phase_a.{name}")


def validate_static_preflight_v2(
    root: Path,
    *,
    source_closure_binding: Mapping[str, str],
    controller_binding: Mapping[str, str],
    isolation_addendum_binding: Mapping[str, str],
    canonical_argv: Sequence[str],
) -> StaticPreflightV2Receipt:
    closure_path, closure_sha256 = verify_file_binding(root, dict(source_closure_binding), label="source_closure_v2")
    closure = require_canonical_json(closure_path)
    expected_keys = {
        "schema",
        "loop_id",
        "scope",
        "supersedes_source_closure_v1",
        "runtime_isolation_addendum",
        "phase_a_bindings",
        "phase_b_protocol",
        "phase_b_protocol_addendum",
        "runtime_lock_v2",
        "source_files",
        "static_preflight_ready",
        "phase_b_raw_execution_ready",
        "remaining_execution_blockers",
    }
    if set(closure) != expected_keys:
        raise PhaseBContractError("Phase-B source closure v2 fields drifted")
    if closure["schema"] != "axon_loop167_phase_b_source_closure_v2":
        raise PhaseBContractError("Phase-B source closure v2 schema drifted")
    if closure["loop_id"] != "loop167_ember_v3_novel_delta":
        raise PhaseBContractError("Phase-B source closure v2 loop id drifted")
    if closure["scope"] != "static_preflight_only_no_raw_checkpoint_prediction_or_fit_access":
        raise PhaseBContractError("Phase-B source closure v2 scope drifted")
    if closure["static_preflight_ready"] is not True or closure["phase_b_raw_execution_ready"] is not False:
        raise PhaseBContractError("Phase-B source closure v2 execution state drifted")
    verify_file_binding(root, closure["supersedes_source_closure_v1"], label="source_closure_v1")
    if closure["runtime_isolation_addendum"] != dict(isolation_addendum_binding):
        raise PhaseBContractError("Phase-B source closure v2 isolation-addendum binding drifted")
    verify_file_binding(root, closure["runtime_isolation_addendum"], label="runtime_isolation_addendum")
    _verify_phase_a_bindings(root, closure["phase_a_bindings"])

    protocol_path, protocol_sha256 = verify_file_binding(root, closure["phase_b_protocol"], label="phase_b_protocol")
    protocol = require_canonical_json(protocol_path)
    if protocol.get("schema") != "axon_loop167_phase_b_protocol_v1":
        raise PhaseBContractError("Phase-B protocol schema drifted")
    _verify_phase_a_bindings(root, protocol.get("phase_a_bindings"))
    if protocol["phase_a_bindings"] != closure["phase_a_bindings"]:
        raise PhaseBContractError("Phase-A bindings differ between protocol and closure v2")

    addendum_path, _ = verify_file_binding(
        root,
        closure["phase_b_protocol_addendum"],
        label="phase_b_protocol_addendum",
    )
    addendum = require_canonical_json(addendum_path)
    if addendum.get("parent_phase_b_protocol") != closure["phase_b_protocol"]:
        raise PhaseBContractError("Phase-B replay-addendum binding drifted")

    runtime_lock_path, runtime_lock_sha256 = verify_file_binding(root, closure["runtime_lock_v2"], label="runtime_lock_v2")
    runtime_lock = require_canonical_json(runtime_lock_path)
    validate_runtime_lock(
        root,
        runtime_lock,
        controller_binding=controller_binding,
        isolation_addendum_binding=isolation_addendum_binding,
        canonical_argv=canonical_argv,
    )

    source_files = closure["source_files"]
    if not isinstance(source_files, list) or not source_files:
        raise PhaseBContractError("Phase-B source closure v2 has no source files")
    observed_paths: set[str] = set()
    for source_binding in source_files:
        source_path, _ = verify_file_binding(root, source_binding, label="source_file_v2")
        relative_path = source_path.relative_to(root.resolve(strict=True)).as_posix()
        if relative_path in observed_paths:
            raise PhaseBContractError("Phase-B source closure v2 repeats a source file")
        observed_paths.add(relative_path)
    controller_path, controller_sha256 = verify_file_binding(root, dict(controller_binding), label="controller")
    expected_controller = {
        "path": controller_path.relative_to(root.resolve(strict=True)).as_posix(),
        "sha256": controller_sha256,
    }
    if expected_controller not in source_files:
        raise PhaseBContractError("Controller v2 is not bound by the source closure")
    return StaticPreflightV2Receipt(
        protocol_sha256=protocol_sha256,
        source_closure_sha256=closure_sha256,
        runtime_lock_sha256=runtime_lock_sha256,
        raw_open_attempts=0,
    )
