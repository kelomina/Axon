"""Fail-closed authorization for one contained Loop167 Phase-B v9 execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .child_attestation_v9 import verify_child_job_attestation_v9
from .contracts import (
    PhaseBContractError,
    canonical_argv_sha256,
    canonical_json_bytes,
    require_canonical_json,
    sha256_bytes,
    sha256_file,
)
from .execution_contract_v5 import EXPECTED_FORBIDDEN
from .execution_contract_v9 import (
    AUTHORIZATION_CLAIM_SCOPE,
    CANONICAL_CONTROLLER_EXECUTE_ARGV,
    CANONICAL_SUPERVISOR_EXECUTE_ARGV,
    CONTROLLER_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    EXPECTED_LEASE,
    FIXED_OUTPUT_CATALOG,
    LOOP166_WINDOWS_JOB_RELATIVE_PATH,
    LOOP166_WINDOWS_PROCESS_LINEAGE_RELATIVE_PATH,
    LOOP_ID,
    PHASE_B_PROTOCOL_RELATIVE_PATH,
    RESOURCE_GUARD_RELATIVE_PATH,
    RUN_AUTHORIZATION_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    SUPERVISOR_RELATIVE_PATH,
    assert_attested_child_prelease_surface_v9,
    assert_contained_child_prelease_surface_v9,
    assert_leased_child_pre_raw_surface_v9,
    assert_output_catalog_is_fresh_v9,
    output_catalog_sha256,
    verify_execution_contract_v9,
)
from .path_safety_v4 import (
    safe_project_path,
    safe_project_relative_path,
    safe_project_root,
    verify_safe_file_binding,
)
from .resource_guard_v9 import RESOURCE_GUARD_SCHEMA, verify_resource_guard_v9
from .supervisor_v9 import validate_launch_receipt_v9

AUTHORIZATION_SCHEMA = "axon_loop167_phase_b_run_authorization_v9"
AUTHORIZATION_STATUS = "authorized_pending_one_shot_contained_child_lease"
SOURCE_CLOSURE_SCHEMA = "axon_loop167_phase_b_source_closure_v9"
RUNTIME_LOCK_SCHEMA = "axon_loop167_phase_b_runtime_lock_v9"
AUTHORIZED_READY_FOR = {
    "raw_access": True,
    "fit": True,
    "val": False,
    "test10k": False,
    "legacy_full_test": False,
    "promotion": False,
}


@dataclass(frozen=True)
class VerifiedExecutionAuthorizationV9:
    project_root: Path
    authorization_path: Path
    authorization_sha256: str
    execution_contract_binding: Mapping[str, str]
    protocol_binding: Mapping[str, str]
    source_closure_binding: Mapping[str, str]
    runtime_lock_binding: Mapping[str, str]
    controller_binding: Mapping[str, str]
    supervisor_binding: Mapping[str, str]
    loop166_windows_job_binding: Mapping[str, str]
    loop166_windows_process_lineage_binding: Mapping[str, str]
    resource_guard_binding: Mapping[str, str]
    output_paths: Mapping[str, Path]
    lease_marker_path: Path


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PhaseBContractError(f"{label} must be a UTC timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as error:
        raise PhaseBContractError(f"{label} is invalid") from error


def _binding(root: Path, value: object, *, label: str, expected_path: str, expected_schema: str | None = None) -> tuple[dict[str, str], Mapping[str, Any] | None]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"} or value.get("path") != expected_path:
        raise PhaseBContractError(f"v9 authorization {label} binding drifted")
    path, digest = verify_safe_file_binding(root, value, label=label)
    payload = require_canonical_json(path) if expected_schema is not None else None
    if expected_schema is not None and (payload.get("schema") != expected_schema or payload.get("loop_id") != LOOP_ID):
        raise PhaseBContractError(f"v9 authorization {label} schema drifted")
    return {"path": expected_path, "sha256": digest}, payload


def _current_source_binding(root: Path, relative_path: str, *, label: str) -> dict[str, str]:
    path = safe_project_path(root, relative_path, require_exists=True, require_regular_file=True)
    return {"path": relative_path, "sha256": sha256_file(path)}


def _static_bindings(
    root: Path,
    payload: Mapping[str, Any],
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    contract, _ = _binding(root, payload.get("phase_b_execution_contract"), label="execution_contract", expected_path=EXECUTION_CONTRACT_RELATIVE_PATH)
    source, _ = _binding(root, payload.get("source_closure"), label="source_closure", expected_path=SOURCE_CLOSURE_RELATIVE_PATH, expected_schema=SOURCE_CLOSURE_SCHEMA)
    runtime, _ = _binding(root, payload.get("runtime_lock"), label="runtime_lock", expected_path=RUNTIME_LOCK_RELATIVE_PATH, expected_schema=RUNTIME_LOCK_SCHEMA)
    controller, _ = _binding(root, payload.get("controller"), label="controller", expected_path=CONTROLLER_RELATIVE_PATH)
    supervisor, _ = _binding(root, payload.get("supervisor"), label="supervisor", expected_path=SUPERVISOR_RELATIVE_PATH)
    loop166_windows_job, _ = _binding(
        root,
        payload.get("loop166_windows_job"),
        label="loop166_windows_job",
        expected_path=LOOP166_WINDOWS_JOB_RELATIVE_PATH,
    )
    loop166_windows_process_lineage, _ = _binding(
        root,
        payload.get("loop166_windows_process_lineage"),
        label="loop166_windows_process_lineage",
        expected_path=LOOP166_WINDOWS_PROCESS_LINEAGE_RELATIVE_PATH,
    )
    return (
        contract,
        source,
        runtime,
        controller,
        supervisor,
        loop166_windows_job,
        loop166_windows_process_lineage,
    )


def _validate_payload(
    root: Path,
    payload: Mapping[str, Any],
    *,
    now_utc: datetime,
    phase: str,
    launch_id: str | None,
) -> VerifiedExecutionAuthorizationV9:
    if phase not in {"prelaunch", "contained_child", "attested_child", "leased_child_pre_raw"}:
        raise PhaseBContractError("v9 authorization phase is invalid")
    expected_keys = {
        "schema", "loop_id", "claim_scope", "status", "execution_authorization_granted",
        "phase_b_execution_contract", "phase_b_protocol", "source_closure", "runtime_lock", "controller",
        "supervisor", "loop166_windows_job", "loop166_windows_process_lineage", "resource_guard",
        "canonical_supervisor_execute_argv", "canonical_supervisor_execute_argv_sha256",
        "canonical_controller_execute_argv", "canonical_controller_execute_argv_sha256", "output_catalog",
        "output_catalog_sha256", "lease", "ready_for", "forbidden", "created_at_utc", "raw_open_attempts",
    }
    if set(payload) != expected_keys or payload.get("schema") != AUTHORIZATION_SCHEMA:
        raise PhaseBContractError("v9 authorization schema drifted")
    if payload.get("loop_id") != LOOP_ID or payload.get("claim_scope") != AUTHORIZATION_CLAIM_SCOPE:
        raise PhaseBContractError("v9 authorization identity drifted")
    if payload.get("status") != AUTHORIZATION_STATUS or payload.get("execution_authorization_granted") is not True:
        raise PhaseBContractError("v9 authorization is not granted")
    (
        contract_binding,
        source_binding,
        runtime_binding,
        controller_binding,
        supervisor_binding,
        loop166_windows_job_binding,
        loop166_windows_process_lineage_binding,
    ) = _static_bindings(root, payload)
    contract = verify_execution_contract_v9(root, contract_binding)
    protocol_binding, protocol = _binding(root, payload.get("phase_b_protocol"), label="phase_b_protocol", expected_path=PHASE_B_PROTOCOL_RELATIVE_PATH, expected_schema="axon_loop167_phase_b_protocol_v1")
    if protocol_binding != dict(contract.protocol_binding) or protocol.get("claim_scope") != "local_train_only_structural_delta_diagnostic_not_model_quality_promotion_or_full_test":
        raise PhaseBContractError("v9 authorization protocol binding drifted")
    runtime_path = safe_project_path(root, RUNTIME_LOCK_RELATIVE_PATH, require_exists=True, require_regular_file=True)
    runtime_payload = require_canonical_json(runtime_path)
    if runtime_payload.get("controller") != controller_binding or runtime_payload.get("supervisor") != supervisor_binding or runtime_payload.get("execution_contract") != contract_binding:
        raise PhaseBContractError("v9 authorization runtime lock bindings drifted")
    resource_guard_binding, _ = _binding(root, payload.get("resource_guard"), label="resource_guard", expected_path=RESOURCE_GUARD_RELATIVE_PATH, expected_schema=RESOURCE_GUARD_SCHEMA)
    verified_guard = verify_resource_guard_v9(
        root,
        resource_guard_binding,
        expected_execution_contract_binding=contract_binding,
        expected_source_closure_binding=source_binding,
        expected_runtime_lock_binding=runtime_binding,
        now_utc=now_utc,
    )
    if verified_guard.guard_sha256 != resource_guard_binding["sha256"]:
        raise PhaseBContractError("v9 authorization resource guard digest drifted")
    if payload.get("canonical_supervisor_execute_argv") != list(CANONICAL_SUPERVISOR_EXECUTE_ARGV):
        raise PhaseBContractError("v9 authorization supervisor argv drifted")
    if payload.get("canonical_supervisor_execute_argv_sha256") != canonical_argv_sha256(CANONICAL_SUPERVISOR_EXECUTE_ARGV):
        raise PhaseBContractError("v9 authorization supervisor argv hash drifted")
    if payload.get("canonical_controller_execute_argv") != list(CANONICAL_CONTROLLER_EXECUTE_ARGV):
        raise PhaseBContractError("v9 authorization controller argv drifted")
    if payload.get("canonical_controller_execute_argv_sha256") != canonical_argv_sha256(CANONICAL_CONTROLLER_EXECUTE_ARGV):
        raise PhaseBContractError("v9 authorization controller argv hash drifted")
    if payload.get("output_catalog") != [dict(entry) for entry in FIXED_OUTPUT_CATALOG] or payload.get("output_catalog_sha256") != output_catalog_sha256():
        raise PhaseBContractError("v9 authorization output catalog drifted")
    if payload.get("lease") != EXPECTED_LEASE or payload.get("ready_for") != AUTHORIZED_READY_FOR:
        raise PhaseBContractError("v9 authorization lease or ready state drifted")
    if payload.get("forbidden") != list(EXPECTED_FORBIDDEN):
        raise PhaseBContractError("v9 authorization forbidden scope drifted")
    if payload.get("raw_open_attempts") != 0:
        raise PhaseBContractError("v9 authorization records raw access")
    _parse_utc(payload.get("created_at_utc"), label="authorization.created_at_utc")
    if phase == "prelaunch":
        output_paths = assert_output_catalog_is_fresh_v9(root, payload["output_catalog"])
    else:
        if phase == "contained_child":
            output_paths = assert_contained_child_prelease_surface_v9(root, payload["output_catalog"])
        elif phase == "attested_child":
            output_paths = assert_attested_child_prelease_surface_v9(root, payload["output_catalog"])
        else:
            output_paths = assert_leased_child_pre_raw_surface_v9(root, payload["output_catalog"])
        if not isinstance(launch_id, str) or not launch_id:
            raise PhaseBContractError("v9 attested child authorization requires launch id")
        expected_static_bindings = {
            "source_closure": source_binding,
            "execution_contract": contract_binding,
            "runtime_lock": runtime_binding,
            "controller": controller_binding,
            "supervisor": supervisor_binding,
            "loop166_windows_job": loop166_windows_job_binding,
            "loop166_windows_process_lineage": loop166_windows_process_lineage_binding,
        }
        launch_path = output_paths["supervisor_launch_receipt"]
        launch = validate_launch_receipt_v9(
            root,
            launch_path,
            mode="execute",
            expected_bindings=expected_static_bindings,
            expected_launch_id=launch_id,
        )
        if phase in {"attested_child", "leased_child_pre_raw"}:
            verify_child_job_attestation_v9(
                root,
                expected_launch_receipt_path=launch_path,
                expected_launch_id=str(launch.payload["launch_id"]),
                expected_bindings=expected_static_bindings,
            )
    marker = safe_project_path(root, EXPECTED_LEASE["marker_path"], require_exists=False)
    return VerifiedExecutionAuthorizationV9(
        project_root=root,
        authorization_path=Path(),
        authorization_sha256="",
        execution_contract_binding=MappingProxyType(contract_binding),
        protocol_binding=MappingProxyType(protocol_binding),
        source_closure_binding=MappingProxyType(source_binding),
        runtime_lock_binding=MappingProxyType(runtime_binding),
        controller_binding=MappingProxyType(controller_binding),
        supervisor_binding=MappingProxyType(supervisor_binding),
        loop166_windows_job_binding=MappingProxyType(loop166_windows_job_binding),
        loop166_windows_process_lineage_binding=MappingProxyType(loop166_windows_process_lineage_binding),
        resource_guard_binding=MappingProxyType(resource_guard_binding),
        output_paths=MappingProxyType(dict(output_paths)),
        lease_marker_path=marker,
    )


def build_execution_authorization_payload_v9(
    root: Path | str,
    *,
    execution_contract_binding: Mapping[str, str],
    source_closure_binding: Mapping[str, str],
    runtime_lock_binding: Mapping[str, str],
    controller_binding: Mapping[str, str],
    supervisor_binding: Mapping[str, str],
    resource_guard_binding: Mapping[str, str],
    created_at_utc: str,
) -> dict[str, Any]:
    root_path = safe_project_root(root)
    created = _parse_utc(created_at_utc, label="authorization.created_at_utc")
    contract = verify_execution_contract_v9(root_path, execution_contract_binding)
    payload = {
        "schema": AUTHORIZATION_SCHEMA,
        "loop_id": LOOP_ID,
        "claim_scope": AUTHORIZATION_CLAIM_SCOPE,
        "status": AUTHORIZATION_STATUS,
        "execution_authorization_granted": True,
        "phase_b_execution_contract": dict(execution_contract_binding),
        "phase_b_protocol": dict(contract.protocol_binding),
        "source_closure": dict(source_closure_binding),
        "runtime_lock": dict(runtime_lock_binding),
        "controller": dict(controller_binding),
        "supervisor": dict(supervisor_binding),
        "loop166_windows_job": _current_source_binding(
            root_path,
            LOOP166_WINDOWS_JOB_RELATIVE_PATH,
            label="loop166_windows_job",
        ),
        "loop166_windows_process_lineage": _current_source_binding(
            root_path,
            LOOP166_WINDOWS_PROCESS_LINEAGE_RELATIVE_PATH,
            label="loop166_windows_process_lineage",
        ),
        "resource_guard": dict(resource_guard_binding),
        "canonical_supervisor_execute_argv": list(CANONICAL_SUPERVISOR_EXECUTE_ARGV),
        "canonical_supervisor_execute_argv_sha256": canonical_argv_sha256(CANONICAL_SUPERVISOR_EXECUTE_ARGV),
        "canonical_controller_execute_argv": list(CANONICAL_CONTROLLER_EXECUTE_ARGV),
        "canonical_controller_execute_argv_sha256": canonical_argv_sha256(CANONICAL_CONTROLLER_EXECUTE_ARGV),
        "output_catalog": [dict(entry) for entry in FIXED_OUTPUT_CATALOG],
        "output_catalog_sha256": output_catalog_sha256(),
        "lease": dict(EXPECTED_LEASE),
        "ready_for": dict(AUTHORIZED_READY_FOR),
        "forbidden": list(EXPECTED_FORBIDDEN),
        "created_at_utc": created.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "raw_open_attempts": 0,
    }
    _validate_payload(root_path, payload, now_utc=created, phase="prelaunch", launch_id=None)
    return payload


def validate_execution_authorization_v9(
    root: Path | str,
    authorization_path: Path | str,
    *,
    now_utc: datetime,
    phase: str,
    launch_id: str | None = None,
) -> VerifiedExecutionAuthorizationV9:
    root_path = safe_project_root(root)
    expected_path = safe_project_path(root_path, RUN_AUTHORIZATION_RELATIVE_PATH, require_exists=True, require_regular_file=True)
    if safe_project_relative_path(root_path, authorization_path, require_exists=True, require_regular_file=True) != RUN_AUTHORIZATION_RELATIVE_PATH:
        raise PhaseBContractError("v9 authorization path drifted")
    payload = require_canonical_json(expected_path)
    digest = sha256_bytes(canonical_json_bytes(payload))
    verified = _validate_payload(root_path, payload, now_utc=now_utc, phase=phase, launch_id=launch_id)
    if sha256_file(expected_path) != digest:
        raise PhaseBContractError("v9 authorization changed during validation")
    return VerifiedExecutionAuthorizationV9(
        project_root=root_path,
        authorization_path=expected_path,
        authorization_sha256=digest,
        execution_contract_binding=verified.execution_contract_binding,
        protocol_binding=verified.protocol_binding,
        source_closure_binding=verified.source_closure_binding,
        runtime_lock_binding=verified.runtime_lock_binding,
        controller_binding=verified.controller_binding,
        supervisor_binding=verified.supervisor_binding,
        loop166_windows_job_binding=verified.loop166_windows_job_binding,
        loop166_windows_process_lineage_binding=verified.loop166_windows_process_lineage_binding,
        resource_guard_binding=verified.resource_guard_binding,
        output_paths=verified.output_paths,
        lease_marker_path=verified.lease_marker_path,
    )


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "AUTHORIZATION_STATUS",
    "VerifiedExecutionAuthorizationV9",
    "build_execution_authorization_payload_v9",
    "validate_execution_authorization_v9",
]
