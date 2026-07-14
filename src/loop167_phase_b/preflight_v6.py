"""Raw-free static validation for the Loop167 Phase-B v6 control plane."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import PhaseBContractError, require_canonical_json, sha256_file
from .execution_contract_v6 import (
    CONTROLLER_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    LOOP166_WINDOWS_JOB_RELATIVE_PATH,
    LOOP166_WINDOWS_PROCESS_LINEAGE_RELATIVE_PATH,
    LOOP_ID,
    PARENT_V5_PRELEASE_ATTESTATION_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    SUPERVISOR_RELATIVE_PATH,
    assert_attested_child_prelease_surface_v6,
    assert_contained_child_prelease_surface_v6,
    assert_leased_child_pre_raw_surface_v6,
    assert_output_catalog_is_fresh_v6,
    verify_execution_contract_v6,
    verify_parent_v5_prelease_attestation_v6,
)
from .path_safety_v4 import (
    safe_project_path,
    safe_project_relative_path,
    safe_project_root,
    verify_safe_file_binding,
)

SOURCE_CLOSURE_SCHEMA = "axon_loop167_phase_b_source_closure_v6"
V6_SCOPE = "v6_typed_windows_job_abi_remediation_single_authorized_train_only_execution_no_heldout_access"
EXPECTED_DYNAMIC_GATES = {
    "fresh_assignment_capable_resource_guard_required": True,
    "run_authorization_required": True,
    "pre_resume_containment_receipt_required": True,
    "child_job_attestation_required": True,
    "one_shot_lease_required": True,
}
EXPECTED_BLOCKERS = [
    "fresh_resource_guard_v6_not_sealed",
    "run_authorization_v6_not_sealed",
    "contained_child_pre_resume_receipt_not_created",
    "child_job_attestation_not_created",
    "one_shot_lease_v6_not_consumed",
]
REQUIRED_SOURCE_PATHS = frozenset(
    {
        "src/loop167_phase_b/__init__.py",
        "src/loop167_phase_b/contracts.py",
        "src/loop167_phase_b/path_safety_v4.py",
        "src/loop167_phase_b/execution_contract_v5.py",
        "src/loop167_phase_b/execution_contract_v6.py",
        "src/loop167_phase_b/windows_job_v6.py",
        "src/loop167_phase_b/supervisor_v6.py",
        "src/loop167_phase_b/invocation_v6.py",
        "src/loop167_phase_b/runtime_lock_v6.py",
        "src/loop167_phase_b/preflight_v6.py",
        "src/loop167_phase_b/resource_guard_v6.py",
        "src/loop167_phase_b/execution_authorization_v6.py",
        "src/loop167_phase_b/lease_v6.py",
        "src/loop167_phase_b/child_attestation_v6.py",
        "src/loop167_phase_b/loop166_v6_bridge.py",
        "src/loop166/windows_job.py",
        "src/loop166/windows_process_lineage.py",
        "scripts/run_loop167_phase_b_supervisor_v6.py",
        "scripts/run_loop167_phase_b_controller_v6.py",
    }
)


@dataclass(frozen=True)
class StaticPreflightV6Receipt:
    source_closure_binding: Mapping[str, str]
    source_closure_sha256: str
    execution_contract_binding: Mapping[str, str]
    execution_contract_sha256: str
    runtime_lock_binding: Mapping[str, str]
    runtime_lock_sha256: str
    controller_binding: Mapping[str, str]
    supervisor_binding: Mapping[str, str]
    loop166_windows_job_binding: Mapping[str, str]
    loop166_windows_process_lineage_binding: Mapping[str, str]
    phase: str
    raw_open_attempts: int


def _binding(value: object, *, label: str, expected_path: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise PhaseBContractError(f"{label} must be a source binding")
    if value.get("path") != expected_path or not isinstance(value.get("sha256"), str):
        raise PhaseBContractError(f"{label} path drifted")
    if len(str(value["sha256"])) != 64:
        raise PhaseBContractError(f"{label} hash is invalid")
    return {"path": expected_path, "sha256": str(value["sha256"])}


def _verify_source_files(root: Path, source_files: object) -> None:
    if not isinstance(source_files, list) or not source_files:
        raise PhaseBContractError("v6 source closure has no source-file bindings")
    observed: set[str] = set()
    for binding in source_files:
        path, _ = verify_safe_file_binding(root, binding, label="v6_source_file")
        relative = safe_project_relative_path(root, path, require_exists=True, require_regular_file=True)
        if relative in observed:
            raise PhaseBContractError("v6 source closure repeats a source file")
        observed.add(relative)
    if not REQUIRED_SOURCE_PATHS.issubset(observed):
        raise PhaseBContractError("v6 source closure omits required control-plane sources")


def _validate_source_closure(
    root: Path,
    binding: Mapping[str, str],
    *,
    controller_binding: Mapping[str, str],
    supervisor_binding: Mapping[str, str],
) -> tuple[dict[str, str], Mapping[str, Any], str]:
    normalized = _binding(binding, label="source_closure", expected_path=SOURCE_CLOSURE_RELATIVE_PATH)
    path, digest = verify_safe_file_binding(root, normalized, label="source_closure")
    payload = require_canonical_json(path)
    expected_keys = {
        "schema",
        "loop_id",
        "scope",
        "parent_v5_prelease_attestation",
        "phase_b_execution_contract",
        "runtime_lock",
        "controller",
        "supervisor",
        "source_files",
        "static_preflight_ready",
        "phase_b_raw_execution_ready",
        "dynamic_execution_gates",
        "remaining_execution_blockers",
    }
    if set(payload) != expected_keys or payload.get("schema") != SOURCE_CLOSURE_SCHEMA:
        raise PhaseBContractError("v6 source closure schema drifted")
    if payload.get("loop_id") != LOOP_ID or payload.get("scope") != V6_SCOPE:
        raise PhaseBContractError("v6 source closure identity drifted")
    if payload.get("static_preflight_ready") is not True or payload.get("phase_b_raw_execution_ready") is not False:
        raise PhaseBContractError("v6 source closure readiness drifted")
    if payload.get("dynamic_execution_gates") != EXPECTED_DYNAMIC_GATES:
        raise PhaseBContractError("v6 source closure dynamic gates drifted")
    if payload.get("remaining_execution_blockers") != EXPECTED_BLOCKERS:
        raise PhaseBContractError("v6 source closure blockers drifted")
    _binding(payload.get("parent_v5_prelease_attestation"), label="parent_attestation", expected_path=PARENT_V5_PRELEASE_ATTESTATION_RELATIVE_PATH)
    _binding(payload.get("phase_b_execution_contract"), label="execution_contract", expected_path=EXECUTION_CONTRACT_RELATIVE_PATH)
    _binding(payload.get("runtime_lock"), label="runtime_lock", expected_path=RUNTIME_LOCK_RELATIVE_PATH)
    if _binding(payload.get("controller"), label="controller", expected_path=CONTROLLER_RELATIVE_PATH) != dict(controller_binding):
        raise PhaseBContractError("v6 source closure controller binding drifted")
    if _binding(payload.get("supervisor"), label="supervisor", expected_path=SUPERVISOR_RELATIVE_PATH) != dict(supervisor_binding):
        raise PhaseBContractError("v6 source closure supervisor binding drifted")
    _verify_source_files(root, payload.get("source_files"))
    return normalized, payload, digest


def _source_file_binding(
    root: Path,
    closure: Mapping[str, Any],
    *,
    expected_path: str,
    label: str,
) -> dict[str, str]:
    source_files = closure.get("source_files")
    if not isinstance(source_files, list):
        raise PhaseBContractError("v6 source closure has no source-file bindings")
    candidates = [binding for binding in source_files if isinstance(binding, Mapping) and binding.get("path") == expected_path]
    if len(candidates) != 1:
        raise PhaseBContractError(f"v6 source closure lacks a unique {label} binding")
    _path, digest = verify_safe_file_binding(root, candidates[0], label=label)
    return {"path": expected_path, "sha256": digest}


def validate_static_preflight_v6(
    root: Path | str,
    *,
    source_closure_binding: Mapping[str, str],
    controller_binding: Mapping[str, str],
    supervisor_binding: Mapping[str, str],
    phase: str,
) -> StaticPreflightV6Receipt:
    """Verify only static bindings and the explicit prelease state machine."""

    if phase not in {"prelaunch", "contained_child", "attested_child", "leased_child_pre_raw"}:
        raise PhaseBContractError("v6 static preflight phase is invalid")
    root_path = safe_project_root(root)
    normalized_controller = _binding(controller_binding, label="controller", expected_path=CONTROLLER_RELATIVE_PATH)
    normalized_supervisor = _binding(supervisor_binding, label="supervisor", expected_path=SUPERVISOR_RELATIVE_PATH)
    _controller_path = safe_project_path(root_path, CONTROLLER_RELATIVE_PATH, require_exists=True, require_regular_file=True)
    _supervisor_path = safe_project_path(root_path, SUPERVISOR_RELATIVE_PATH, require_exists=True, require_regular_file=True)
    if sha256_file(_controller_path) != normalized_controller["sha256"]:
        raise PhaseBContractError("v6 controller changed during static preflight")
    if sha256_file(_supervisor_path) != normalized_supervisor["sha256"]:
        raise PhaseBContractError("v6 supervisor changed during static preflight")
    closure_binding, closure, closure_sha = _validate_source_closure(
        root_path,
        source_closure_binding,
        controller_binding=normalized_controller,
        supervisor_binding=normalized_supervisor,
    )
    verify_parent_v5_prelease_attestation_v6(root_path, closure["parent_v5_prelease_attestation"])
    contract = verify_execution_contract_v6(root_path, closure["phase_b_execution_contract"])
    loop166_windows_job_binding = _source_file_binding(
        root_path,
        closure,
        expected_path=LOOP166_WINDOWS_JOB_RELATIVE_PATH,
        label="loop166_windows_job",
    )
    loop166_windows_process_lineage_binding = _source_file_binding(
        root_path,
        closure,
        expected_path=LOOP166_WINDOWS_PROCESS_LINEAGE_RELATIVE_PATH,
        label="loop166_windows_process_lineage",
    )
    runtime_lock_binding = _binding(closure["runtime_lock"], label="runtime_lock", expected_path=RUNTIME_LOCK_RELATIVE_PATH)
    runtime_lock_path, runtime_lock_sha = verify_safe_file_binding(root_path, runtime_lock_binding, label="runtime_lock")
    runtime_lock = require_canonical_json(runtime_lock_path)
    if runtime_lock.get("schema") != "axon_loop167_phase_b_runtime_lock_v6" or runtime_lock.get("loop_id") != LOOP_ID:
        raise PhaseBContractError("v6 runtime lock identity drifted")
    if runtime_lock.get("controller") != normalized_controller or runtime_lock.get("supervisor") != normalized_supervisor:
        raise PhaseBContractError("v6 runtime lock controller bindings drifted")
    if runtime_lock.get("execution_contract") != dict(closure["phase_b_execution_contract"]):
        raise PhaseBContractError("v6 runtime lock execution contract binding drifted")
    if phase == "prelaunch":
        assert_output_catalog_is_fresh_v6(root_path, contract.output_catalog)
    elif phase == "contained_child":
        assert_contained_child_prelease_surface_v6(root_path, contract.output_catalog)
    elif phase == "attested_child":
        assert_attested_child_prelease_surface_v6(root_path, contract.output_catalog)
    else:
        assert_leased_child_pre_raw_surface_v6(root_path, contract.output_catalog)
    return StaticPreflightV6Receipt(
        source_closure_binding=MappingProxyType(dict(closure_binding)),
        source_closure_sha256=closure_sha,
        execution_contract_binding=MappingProxyType(dict(closure["phase_b_execution_contract"])),
        execution_contract_sha256=contract.contract_sha256,
        runtime_lock_binding=MappingProxyType(dict(runtime_lock_binding)),
        runtime_lock_sha256=runtime_lock_sha,
        controller_binding=MappingProxyType(dict(normalized_controller)),
        supervisor_binding=MappingProxyType(dict(normalized_supervisor)),
        loop166_windows_job_binding=MappingProxyType(loop166_windows_job_binding),
        loop166_windows_process_lineage_binding=MappingProxyType(loop166_windows_process_lineage_binding),
        phase=phase,
        raw_open_attempts=0,
    )


__all__ = [
    "EXPECTED_BLOCKERS",
    "EXPECTED_DYNAMIC_GATES",
    "REQUIRED_SOURCE_PATHS",
    "SOURCE_CLOSURE_SCHEMA",
    "StaticPreflightV6Receipt",
    "V6_SCOPE",
    "validate_static_preflight_v6",
]
