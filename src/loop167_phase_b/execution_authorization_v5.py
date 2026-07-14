"""Fail-closed run authorization for the sealed Loop167 Phase-B v5 execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import (
    PhaseBContractError,
    canonical_argv_sha256,
    canonical_json_bytes,
    require_canonical_json,
    sha256_bytes,
    sha256_file,
)
from .execution_contract_v5 import (
    AUTHORIZATION_CLAIM_SCOPE,
    CANONICAL_EXECUTE_ARGV,
    CONTROLLER_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    EXECUTION_CONTRACT_SCHEMA,
    EXPECTED_FORBIDDEN,
    EXPECTED_LEASE,
    FIXED_OUTPUT_CATALOG,
    LOOP_ID,
    PHASE_B_PROTOCOL_RELATIVE_PATH,
    RESOURCE_GUARD_RELATIVE_PATH,
    RUN_AUTHORIZATION_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    assert_output_catalog_is_fresh_v5,
    canonical_execute_argv_v5,
    output_catalog_sha256,
    verify_execution_contract_v5,
)
from .invocation_v5 import canonical_argv_hashes_v5
from .path_safety_v4 import (
    canonical_project_relative_path,
    safe_project_path,
    safe_project_relative_path,
    safe_project_root,
    verify_safe_file_binding,
)
from .resource_guard_v5 import RESOURCE_GUARD_SCHEMA, verify_resource_guard_v5

AUTHORIZATION_SCHEMA = "axon_loop167_phase_b_run_authorization_v5"
AUTHORIZATION_STATUS = "authorized_pending_one_shot_lease"
SOURCE_CLOSURE_SCHEMA = "axon_loop167_phase_b_source_closure_v5"
RUNTIME_LOCK_SCHEMA = "axon_loop167_phase_b_runtime_lock_v5"

AUTHORIZED_READY_FOR: dict[str, bool] = {
    "raw_access": True,
    "fit": True,
    "val": False,
    "test10k": False,
    "legacy_full_test": False,
    "promotion": False,
}


@dataclass(frozen=True)
class VerifiedExecutionAuthorizationV5:
    """The exact, currently valid inputs needed to burn the one-shot lease."""

    project_root: Path
    authorization_path: Path
    authorization_sha256: str
    execution_contract_binding: Mapping[str, str]
    protocol_binding: Mapping[str, str]
    source_closure_binding: Mapping[str, str]
    runtime_lock_binding: Mapping[str, str]
    controller_binding: Mapping[str, str]
    resource_guard_binding: Mapping[str, str]
    lease_marker_path: Path
    output_paths: Mapping[str, Path]
    canonical_execute_argv: tuple[str, ...]
    output_catalog_sha256: str


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PhaseBContractError(f"{label} must be a UTC Z timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as exc:
        raise PhaseBContractError(f"{label} is invalid") from exc


def _require_now(now_utc: datetime) -> datetime:
    if not isinstance(now_utc, datetime) or now_utc.tzinfo is None:
        raise PhaseBContractError("Authorization validation time must be timezone-aware")
    return now_utc.astimezone(UTC)


def _require_fixed_binding(
    root: Path,
    binding: object,
    *,
    label: str,
    expected_path: str,
) -> tuple[Path, dict[str, str]]:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise PhaseBContractError(f"{label} binding must contain exactly path and sha256")
    if binding["path"] != expected_path:
        raise PhaseBContractError(f"{label} path is outside the fixed v5 contract")
    path, digest = verify_safe_file_binding(root, binding, label=label)
    relative_path = safe_project_relative_path(
        root,
        path,
        require_exists=True,
        require_regular_file=True,
    )
    if relative_path != expected_path:
        raise PhaseBContractError(f"{label} path is not canonically pinned")
    return path, {"path": relative_path, "sha256": digest}


def _require_static_json_binding(
    root: Path,
    binding: object,
    *,
    label: str,
    expected_path: str,
    expected_schema: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    path, normalized_binding = _require_fixed_binding(
        root,
        binding,
        label=label,
        expected_path=expected_path,
    )
    payload = require_canonical_json(path)
    if payload.get("schema") != expected_schema or payload.get("loop_id") != LOOP_ID:
        raise PhaseBContractError(f"{label} schema or loop id drifted")
    return payload, normalized_binding


def _validate_runtime_lock_payload(
    payload: Mapping[str, Any],
    *,
    execution_contract_binding: Mapping[str, str],
    controller_binding: Mapping[str, str],
) -> None:
    if payload.get("schema") != RUNTIME_LOCK_SCHEMA or payload.get("loop_id") != LOOP_ID:
        raise PhaseBContractError("Runtime lock v5 identity drifted")
    if payload.get("controller") != dict(controller_binding):
        raise PhaseBContractError("Runtime lock v5 controller binding drifted")
    if payload.get("execution_contract") != dict(execution_contract_binding):
        raise PhaseBContractError("Runtime lock v5 execution-contract binding drifted")
    if payload.get("canonical_argv") != {
        "preflight": [
            "vnev/Scripts/python.exe",
            "-I",
            CONTROLLER_RELATIVE_PATH,
            "--preflight",
        ],
        "execute": list(CANONICAL_EXECUTE_ARGV),
    }:
        raise PhaseBContractError("Runtime lock v5 canonical argv drifted")
    if payload.get("canonical_argv_sha256") != canonical_argv_hashes_v5():
        raise PhaseBContractError("Runtime lock v5 canonical argv hashes drifted")
    if payload.get("isolated_python_required") is not True:
        raise PhaseBContractError("Runtime lock v5 must require isolated Python")
    if payload.get("network_fetch_allowed") is not False or payload.get("dependency_install_allowed") is not False:
        raise PhaseBContractError("Runtime lock v5 permits an unsafe dependency action")


def _assert_fresh_lease_marker(root: Path, marker_relative_path: object) -> Path:
    marker_relative = canonical_project_relative_path(marker_relative_path)
    if marker_relative != EXPECTED_LEASE["marker_path"]:
        raise PhaseBContractError("Execution authorization lease marker path drifted")
    marker_path = safe_project_path(root, marker_relative, require_exists=False)
    if marker_path.exists() or marker_path.is_symlink():
        raise PhaseBContractError("Execution authorization lease marker already exists or is unsafe")
    return marker_path


def _validate_authorization_payload(
    root: Path,
    payload: Mapping[str, Any],
    *,
    now_utc: datetime,
) -> VerifiedExecutionAuthorizationV5:
    expected_keys = {
        "schema",
        "loop_id",
        "claim_scope",
        "status",
        "execution_authorization_granted",
        "phase_b_execution_contract",
        "phase_b_protocol",
        "source_closure",
        "runtime_lock",
        "controller",
        "resource_guard",
        "canonical_execute_argv",
        "canonical_execute_argv_sha256",
        "output_catalog",
        "output_catalog_sha256",
        "lease",
        "ready_for",
        "forbidden",
        "created_at_utc",
        "raw_open_attempts",
    }
    if set(payload) != expected_keys or payload.get("schema") != AUTHORIZATION_SCHEMA:
        raise PhaseBContractError("Execution authorization v5 schema drifted")
    if payload.get("loop_id") != LOOP_ID or payload.get("claim_scope") != AUTHORIZATION_CLAIM_SCOPE:
        raise PhaseBContractError("Execution authorization v5 identity or scope drifted")
    if (
        payload.get("status") != AUTHORIZATION_STATUS
        or payload.get("execution_authorization_granted") is not True
    ):
        raise PhaseBContractError("Execution authorization v5 is not granted")
    _parse_utc(payload.get("created_at_utc"), label="execution_authorization.created_at_utc")
    if payload.get("raw_open_attempts") != 0:
        raise PhaseBContractError("Execution authorization records raw access")
    argv = canonical_execute_argv_v5(payload.get("canonical_execute_argv", ()))
    if payload.get("canonical_execute_argv_sha256") != canonical_argv_sha256(argv):
        raise PhaseBContractError("Execution authorization execute argv hash drifted")

    contract_path, execution_contract_binding = _require_fixed_binding(
        root,
        payload.get("phase_b_execution_contract"),
        label="execution_contract",
        expected_path=EXECUTION_CONTRACT_RELATIVE_PATH,
    )
    contract_payload = require_canonical_json(contract_path)
    if contract_payload.get("schema") != EXECUTION_CONTRACT_SCHEMA:
        raise PhaseBContractError("Execution authorization contract schema drifted")
    contract = verify_execution_contract_v5(root, execution_contract_binding)

    protocol_payload, protocol_binding = _require_static_json_binding(
        root,
        payload.get("phase_b_protocol"),
        label="phase_b_protocol",
        expected_path=PHASE_B_PROTOCOL_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_protocol_v1",
    )
    if protocol_binding != dict(contract_payload["phase_b_protocol"]):
        raise PhaseBContractError("Execution authorization protocol binding drifted from contract")
    if protocol_payload.get("claim_scope") != contract_payload.get("claim_scope"):
        raise PhaseBContractError("Execution contract protocol scope drifted")

    _, source_closure_binding = _require_static_json_binding(
        root,
        payload.get("source_closure"),
        label="source_closure",
        expected_path=SOURCE_CLOSURE_RELATIVE_PATH,
        expected_schema=SOURCE_CLOSURE_SCHEMA,
    )
    runtime_payload, runtime_lock_binding = _require_static_json_binding(
        root,
        payload.get("runtime_lock"),
        label="runtime_lock",
        expected_path=RUNTIME_LOCK_RELATIVE_PATH,
        expected_schema=RUNTIME_LOCK_SCHEMA,
    )
    _, controller_binding = _require_fixed_binding(
        root,
        payload.get("controller"),
        label="controller",
        expected_path=CONTROLLER_RELATIVE_PATH,
    )
    _validate_runtime_lock_payload(
        runtime_payload,
        execution_contract_binding=execution_contract_binding,
        controller_binding=controller_binding,
    )

    _, resource_guard_binding = _require_static_json_binding(
        root,
        payload.get("resource_guard"),
        label="resource_guard",
        expected_path=RESOURCE_GUARD_RELATIVE_PATH,
        expected_schema=RESOURCE_GUARD_SCHEMA,
    )
    verified_guard = verify_resource_guard_v5(
        root,
        resource_guard_binding,
        expected_execution_contract_binding=execution_contract_binding,
        expected_source_closure_binding=source_closure_binding,
        expected_runtime_lock_binding=runtime_lock_binding,
        canonical_execute_argv=argv,
        now_utc=now_utc,
    )
    if verified_guard.guard_sha256 != resource_guard_binding["sha256"]:
        raise PhaseBContractError("Execution authorization resource guard digest drifted")

    if payload.get("output_catalog") != [dict(entry) for entry in FIXED_OUTPUT_CATALOG]:
        raise PhaseBContractError("Execution authorization output catalog drifted")
    if payload.get("output_catalog_sha256") != output_catalog_sha256():
        raise PhaseBContractError("Execution authorization output catalog hash drifted")
    if tuple(dict(entry) for entry in contract.output_catalog) != FIXED_OUTPUT_CATALOG:
        raise PhaseBContractError("Execution contract output catalog drifted during authorization")
    output_paths = assert_output_catalog_is_fresh_v5(root, payload["output_catalog"])

    if payload.get("lease") != EXPECTED_LEASE:
        raise PhaseBContractError("Execution authorization lease contract drifted")
    lease_marker_path = _assert_fresh_lease_marker(root, payload["lease"]["marker_path"])
    if payload.get("ready_for") != AUTHORIZED_READY_FOR:
        raise PhaseBContractError("Execution authorization ready state drifted")
    if payload.get("forbidden") != EXPECTED_FORBIDDEN:
        raise PhaseBContractError("Execution authorization forbidden scope drifted")

    return VerifiedExecutionAuthorizationV5(
        project_root=safe_project_root(root),
        authorization_path=Path(),
        authorization_sha256="",
        execution_contract_binding=MappingProxyType(dict(execution_contract_binding)),
        protocol_binding=MappingProxyType(dict(protocol_binding)),
        source_closure_binding=MappingProxyType(dict(source_closure_binding)),
        runtime_lock_binding=MappingProxyType(dict(runtime_lock_binding)),
        controller_binding=MappingProxyType(dict(controller_binding)),
        resource_guard_binding=MappingProxyType(dict(resource_guard_binding)),
        lease_marker_path=lease_marker_path,
        output_paths=MappingProxyType(dict(output_paths)),
        canonical_execute_argv=argv,
        output_catalog_sha256=output_catalog_sha256(),
    )


def build_execution_authorization_payload_v5(
    root: Path | str,
    *,
    execution_contract_binding: Mapping[str, str],
    source_closure_binding: Mapping[str, str],
    runtime_lock_binding: Mapping[str, str],
    controller_binding: Mapping[str, str],
    resource_guard_binding: Mapping[str, str],
    created_at_utc: str,
) -> dict[str, Any]:
    """Build the only permitted authorization payload after a passing fresh guard."""

    root_path = safe_project_root(root)
    created_at = _parse_utc(created_at_utc, label="execution_authorization.created_at_utc")
    created_at_value = created_at.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    contract_path, contract_binding = _require_fixed_binding(
        root_path,
        execution_contract_binding,
        label="execution_contract",
        expected_path=EXECUTION_CONTRACT_RELATIVE_PATH,
    )
    contract = verify_execution_contract_v5(root_path, contract_binding)
    contract_payload = require_canonical_json(contract_path)
    protocol_binding = contract_payload["phase_b_protocol"]
    payload = {
        "schema": AUTHORIZATION_SCHEMA,
        "loop_id": LOOP_ID,
        "claim_scope": AUTHORIZATION_CLAIM_SCOPE,
        "status": AUTHORIZATION_STATUS,
        "execution_authorization_granted": True,
        "phase_b_execution_contract": contract_binding,
        "phase_b_protocol": dict(protocol_binding),
        "source_closure": dict(source_closure_binding),
        "runtime_lock": dict(runtime_lock_binding),
        "controller": dict(controller_binding),
        "resource_guard": dict(resource_guard_binding),
        "canonical_execute_argv": list(contract.canonical_execute_argv),
        "canonical_execute_argv_sha256": canonical_argv_sha256(contract.canonical_execute_argv),
        "output_catalog": [dict(entry) for entry in contract.output_catalog],
        "output_catalog_sha256": output_catalog_sha256(),
        "lease": dict(contract.lease),
        "ready_for": dict(AUTHORIZED_READY_FOR),
        "forbidden": list(EXPECTED_FORBIDDEN),
        "created_at_utc": created_at_value,
        "raw_open_attempts": 0,
    }
    _validate_authorization_payload(root_path, payload, now_utc=_parse_utc(created_at_value, label="created_at_utc"))
    return payload


def validate_execution_authorization_v5(
    root: Path | str,
    authorization_path: Path | str,
    *,
    now_utc: datetime,
) -> VerifiedExecutionAuthorizationV5:
    """Validate the fixed authorization path and every bound v5 prerequisite."""

    root_path = safe_project_root(root)
    expected_path = safe_project_path(
        root_path,
        RUN_AUTHORIZATION_RELATIVE_PATH,
        require_exists=True,
        require_regular_file=True,
    )
    observed_relative_path = safe_project_relative_path(
        root_path,
        authorization_path,
        require_exists=True,
        require_regular_file=True,
    )
    if observed_relative_path != RUN_AUTHORIZATION_RELATIVE_PATH:
        raise PhaseBContractError("Execution authorization path is outside the fixed Phase-A contract")
    current_time = _require_now(now_utc)
    payload = require_canonical_json(expected_path)
    authorization_sha256 = sha256_bytes(canonical_json_bytes(payload))
    verified = _validate_authorization_payload(root_path, payload, now_utc=current_time)
    if sha256_file(expected_path) != authorization_sha256:
        raise PhaseBContractError("Execution authorization changed during validation")
    return VerifiedExecutionAuthorizationV5(
        project_root=root_path,
        authorization_path=expected_path,
        authorization_sha256=authorization_sha256,
        execution_contract_binding=verified.execution_contract_binding,
        protocol_binding=verified.protocol_binding,
        source_closure_binding=verified.source_closure_binding,
        runtime_lock_binding=verified.runtime_lock_binding,
        controller_binding=verified.controller_binding,
        resource_guard_binding=verified.resource_guard_binding,
        lease_marker_path=verified.lease_marker_path,
        output_paths=verified.output_paths,
        canonical_execute_argv=verified.canonical_execute_argv,
        output_catalog_sha256=verified.output_catalog_sha256,
    )
