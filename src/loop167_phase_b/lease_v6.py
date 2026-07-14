"""Durable one-shot lease bound to v6 containment receipts before raw access."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .child_attestation_v6 import verify_child_job_attestation_v6
from .contracts import PhaseBContractError, canonical_json_bytes, sha256_file
from .execution_authorization_v6 import (
    VerifiedExecutionAuthorizationV6,
    validate_execution_authorization_v6,
)
from .execution_contract_v6 import EXPECTED_LEASE, FIXED_OUTPUT_CATALOG, LOOP_ID
from .path_safety_v4 import safe_project_path, safe_project_relative_path, safe_project_root
from .supervisor_v6 import _fsync_parent_directory, validate_launch_receipt_v6

LEASE_SCHEMA = "axon_loop167_phase_b_execution_lease_v6"
LEASE_STATUS = "consumed_after_containment_attestation_before_first_raw_open"


class ExecutionLeaseV6Error(PhaseBContractError):
    """The v6 lease cannot safely be consumed or revalidated."""


@dataclass(frozen=True)
class ConsumedExecutionLeaseV6:
    marker_path: Path
    marker_sha256: str
    authorization_sha256: str
    launch_receipt_sha256: str
    child_attestation_sha256: str
    payload: Mapping[str, Any]


def _timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ExecutionLeaseV6Error("v6 lease timestamp must be timezone-aware")
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ExecutionLeaseV6Error("v6 lease timestamp is invalid")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as error:
        raise ExecutionLeaseV6Error("v6 lease timestamp is invalid") from error


def _is_link_or_reparse(stat_result: os.stat_result) -> bool:
    return stat.S_ISLNK(stat_result.st_mode) or bool(int(getattr(stat_result, "st_file_attributes", 0)) & 0x0400)


def _ensure_parent(root: Path, relative_path: str) -> Path:
    cursor = safe_project_root(root)
    for component in relative_path.split("/")[:-1]:
        cursor = cursor / component
        try:
            cursor.mkdir(exist_ok=True)
            stat_result = cursor.lstat()
        except OSError as error:
            raise ExecutionLeaseV6Error("v6 lease parent is unavailable") from error
        if _is_link_or_reparse(stat_result) or not stat.S_ISDIR(stat_result.st_mode):
            raise ExecutionLeaseV6Error("v6 lease parent is unsafe")
    return cursor


def _write_marker(marker_path: Path, payload: Mapping[str, Any]) -> ConsumedExecutionLeaseV6:
    content = canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(marker_path, flags, 0o600)
    except FileExistsError as error:
        raise ExecutionLeaseV6Error("v6 execution lease is already consumed") from error
    except OSError as error:
        raise ExecutionLeaseV6Error("v6 execution lease cannot be created") from error
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_parent_directory(marker_path.parent)
    except Exception as error:
        raise ExecutionLeaseV6Error("v6 lease remains consumed after a durability failure") from error
    return ConsumedExecutionLeaseV6(
        marker_path=marker_path,
        marker_sha256=hashlib.sha256(content).hexdigest(),
        authorization_sha256=str(payload["run_authorization"]["sha256"]),
        launch_receipt_sha256=str(payload["pre_resume_launch_receipt"]["sha256"]),
        child_attestation_sha256=str(payload["child_job_attestation"]["sha256"]),
        payload=dict(payload),
    )


def build_execution_lease_payload_v6(
    authorization: VerifiedExecutionAuthorizationV6,
    *,
    launch_id: str,
    consumed_at_utc: datetime,
) -> dict[str, Any]:
    if not isinstance(authorization, VerifiedExecutionAuthorizationV6):
        raise TypeError("authorization must be a VerifiedExecutionAuthorizationV6")
    root = authorization.project_root
    launch_path = authorization.output_paths["supervisor_launch_receipt"]
    expected_bindings = {
        "source_closure": dict(authorization.source_closure_binding),
        "execution_contract": dict(authorization.execution_contract_binding),
        "runtime_lock": dict(authorization.runtime_lock_binding),
        "controller": dict(authorization.controller_binding),
        "supervisor": dict(authorization.supervisor_binding),
        "loop166_windows_job": dict(authorization.loop166_windows_job_binding),
        "loop166_windows_process_lineage": dict(authorization.loop166_windows_process_lineage_binding),
    }
    validated_launch = validate_launch_receipt_v6(
        root,
        launch_path,
        mode="execute",
        expected_bindings=expected_bindings,
        expected_launch_id=launch_id,
    )
    child_path, child_sha, child_payload = verify_child_job_attestation_v6(
        root,
        expected_launch_receipt_path=launch_path,
        expected_launch_id=launch_id,
        expected_bindings=expected_bindings,
    )
    launch_relative = safe_project_relative_path(root, launch_path, require_exists=True, require_regular_file=True)
    child_relative = safe_project_relative_path(root, child_path, require_exists=True, require_regular_file=True)
    authorization_relative = safe_project_relative_path(root, authorization.authorization_path, require_exists=True, require_regular_file=True)
    if authorization_relative != "manifests/roadmap_9997/loop167_ember_v3_novel_delta/phase_b_v6_windows_job_abi_remediation/phase_b_run_authorization_v6.json":
        raise ExecutionLeaseV6Error("v6 authorization path drifted before lease")
    expected_launch_binding = {
        "path": launch_relative,
        "sha256": validated_launch.canonical_sha256,
    }
    if child_payload.get("launch_receipt") != expected_launch_binding:
        raise ExecutionLeaseV6Error("v6 child attestation launch receipt digest drifted")
    return {
        "schema": LEASE_SCHEMA,
        "loop_id": LOOP_ID,
        "status": LEASE_STATUS,
        "consumed_at_utc": _timestamp(consumed_at_utc),
        "run_authorization": {"path": authorization_relative, "sha256": authorization.authorization_sha256},
        "phase_b_execution_contract": dict(authorization.execution_contract_binding),
        "phase_b_protocol": dict(authorization.protocol_binding),
        "source_closure": dict(authorization.source_closure_binding),
        "runtime_lock": dict(authorization.runtime_lock_binding),
        "controller": dict(authorization.controller_binding),
        "supervisor": dict(authorization.supervisor_binding),
        "loop166_windows_job": dict(authorization.loop166_windows_job_binding),
        "loop166_windows_process_lineage": dict(authorization.loop166_windows_process_lineage_binding),
        "resource_guard": dict(authorization.resource_guard_binding),
        "pre_resume_launch_receipt": expected_launch_binding,
        "child_job_attestation": {"path": child_relative, "sha256": child_sha},
        "child_launch_id": str(child_payload["launch_id"]),
        "output_catalog": [dict(entry) for entry in FIXED_OUTPUT_CATALOG],
        "lease": dict(EXPECTED_LEASE),
        "raw_open_attempts_before_consume": 0,
    }


def consume_execution_lease_v6(
    root: Path | str,
    authorization_path: Path | str,
    *,
    now_utc: datetime,
    launch_id: str,
) -> ConsumedExecutionLeaseV6:
    root_path = safe_project_root(root)
    authorization = validate_execution_authorization_v6(
        root_path,
        authorization_path,
        now_utc=now_utc,
        phase="attested_child",
        launch_id=launch_id,
    )
    if sha256_file(authorization.authorization_path) != authorization.authorization_sha256:
        raise ExecutionLeaseV6Error("v6 authorization changed before lease")
    marker_relative = EXPECTED_LEASE["marker_path"]
    marker_path = safe_project_path(root_path, marker_relative, require_exists=False)
    if marker_path.exists() or marker_path.is_symlink():
        raise ExecutionLeaseV6Error("v6 lease already exists or is unsafe")
    _ensure_parent(root_path, marker_relative)
    payload = build_execution_lease_payload_v6(authorization, launch_id=launch_id, consumed_at_utc=now_utc)
    return _write_marker(marker_path, payload)


def verify_consumed_execution_lease_v6(
    root: Path | str,
    authorization: VerifiedExecutionAuthorizationV6,
    *,
    launch_id: str,
    now_utc: datetime,
) -> ConsumedExecutionLeaseV6:
    root_path = safe_project_root(root)
    if not isinstance(authorization, VerifiedExecutionAuthorizationV6):
        raise TypeError("authorization must be a VerifiedExecutionAuthorizationV6")
    refreshed_authorization = validate_execution_authorization_v6(
        root_path,
        authorization.authorization_path,
        now_utc=now_utc,
        phase="leased_child_pre_raw",
        launch_id=launch_id,
    )
    if (
        refreshed_authorization.authorization_sha256 != authorization.authorization_sha256
        or dict(refreshed_authorization.execution_contract_binding) != dict(authorization.execution_contract_binding)
        or dict(refreshed_authorization.source_closure_binding) != dict(authorization.source_closure_binding)
        or dict(refreshed_authorization.runtime_lock_binding) != dict(authorization.runtime_lock_binding)
        or dict(refreshed_authorization.controller_binding) != dict(authorization.controller_binding)
        or dict(refreshed_authorization.supervisor_binding) != dict(authorization.supervisor_binding)
        or dict(refreshed_authorization.loop166_windows_job_binding)
        != dict(authorization.loop166_windows_job_binding)
        or dict(refreshed_authorization.loop166_windows_process_lineage_binding)
        != dict(authorization.loop166_windows_process_lineage_binding)
        or dict(refreshed_authorization.resource_guard_binding) != dict(authorization.resource_guard_binding)
    ):
        raise ExecutionLeaseV6Error("v6 authorization drifted after lease consumption")
    marker_path = safe_project_path(root_path, EXPECTED_LEASE["marker_path"], require_exists=True, require_regular_file=True)
    try:
        marker_bytes = marker_path.read_bytes()
        payload = json.loads(marker_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExecutionLeaseV6Error("v6 lease marker is unavailable") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != marker_bytes:
        raise ExecutionLeaseV6Error("v6 lease marker is not canonical")
    marker_sha256 = hashlib.sha256(marker_bytes).hexdigest()
    consumed_at = _parse_timestamp(payload.get("consumed_at_utc"))
    expected = build_execution_lease_payload_v6(
        refreshed_authorization,
        launch_id=launch_id,
        consumed_at_utc=consumed_at,
    )
    if payload != expected:
        raise ExecutionLeaseV6Error("v6 lease payload drifted")
    if sha256_file(marker_path) != marker_sha256:
        raise ExecutionLeaseV6Error("v6 lease marker changed during verification")
    return ConsumedExecutionLeaseV6(
        marker_path=marker_path,
        marker_sha256=marker_sha256,
        authorization_sha256=refreshed_authorization.authorization_sha256,
        launch_receipt_sha256=str(payload["pre_resume_launch_receipt"]["sha256"]),
        child_attestation_sha256=str(payload["child_job_attestation"]["sha256"]),
        payload=payload,
    )


__all__ = [
    "ConsumedExecutionLeaseV6",
    "ExecutionLeaseV6Error",
    "LEASE_SCHEMA",
    "LEASE_STATUS",
    "build_execution_lease_payload_v6",
    "consume_execution_lease_v6",
    "verify_consumed_execution_lease_v6",
]
