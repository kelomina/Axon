"""Fresh assignment-capable resource gate for the Loop167 Phase-B v9 route."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .contracts import PhaseBContractError, canonical_argv_sha256, require_canonical_json
from .execution_contract_v9 import (
    CANONICAL_CONTROLLER_EXECUTE_ARGV,
    LOOP_ID,
    RESOURCE_GUARD_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    verify_execution_contract_v9,
)
from .path_safety_v4 import safe_project_path, safe_project_root, verify_safe_file_binding
from .windows_job_v9 import WindowsJobAssignmentProbeV9, probe_windows_job_assignment_v9

RESOURCE_GUARD_SCHEMA = "axon_loop167_phase_b_resource_guard_v9"
MAXIMUM_GUARD_AGE_SECONDS = 300


@dataclass(frozen=True)
class SystemResourceSnapshotV9:
    total_memory_bytes: int
    available_memory_bytes: int
    cpu_count: int


@dataclass(frozen=True)
class VerifiedResourceGuardV9:
    guard_path: Path
    guard_sha256: str
    execution_contract_sha256: str
    source_closure_sha256: str
    runtime_lock_sha256: str


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PhaseBContractError(f"{label} must be a non-negative integer")
    return value


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PhaseBContractError(f"{label} must be a UTC timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as error:
        raise PhaseBContractError(f"{label} is invalid") from error


def current_system_snapshot_v9() -> SystemResourceSnapshotV9:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]
        kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise PhaseBContractError("Unable to query Windows memory status")
        return SystemResourceSnapshotV9(
            total_memory_bytes=int(status.ullTotalPhys),
            available_memory_bytes=int(status.ullAvailPhys),
            cpu_count=os.cpu_count() or 0,
        )
    return SystemResourceSnapshotV9(
        total_memory_bytes=int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")),
        available_memory_bytes=int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")),
        cpu_count=os.cpu_count() or 0,
    )


def _binding(root: Path, value: object, *, label: str, expected_path: str, expected_schema: str) -> tuple[dict[str, str], str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"} or value.get("path") != expected_path:
        raise PhaseBContractError(f"{label} binding drifted")
    path, digest = verify_safe_file_binding(root, value, label=label)
    payload = require_canonical_json(path)
    if payload.get("schema") != expected_schema or payload.get("loop_id") != LOOP_ID:
        raise PhaseBContractError(f"{label} schema drifted")
    return {"path": expected_path, "sha256": digest}, digest


def _probe_payload(probe: WindowsJobAssignmentProbeV9) -> dict[str, Any]:
    assignment = None if probe.assignment is None else dict(probe.assignment)
    return {
        "ready": probe.ready,
        "operation": probe.operation,
        "win32_error_code": probe.win32_error_code,
        "detail": probe.detail,
        "assignment": assignment,
    }


def _validate_probe_payload(payload: object) -> WindowsJobAssignmentProbeV9:
    if not isinstance(payload, Mapping) or set(payload) != {"ready", "operation", "win32_error_code", "detail", "assignment"}:
        raise PhaseBContractError("v9 Job assignment probe payload drifted")
    ready = payload["ready"]
    if not isinstance(ready, bool):
        raise PhaseBContractError("v9 Job assignment probe readiness is invalid")
    operation = payload["operation"]
    code = payload["win32_error_code"]
    detail = payload["detail"]
    assignment = payload["assignment"]
    if ready:
        if operation is not None or code is not None or detail is not None or not isinstance(assignment, dict):
            raise PhaseBContractError("v9 ready Job assignment probe drifted")
        if assignment.get("current_process_assigned") is not True or assignment.get("kill_on_job_close") is not False:
            raise PhaseBContractError("v9 ready Job assignment proof drifted")
        if assignment.get("job_limit_flags") != 0x100:
            raise PhaseBContractError("v9 guard Job assignment flags drifted")
    else:
        if not isinstance(operation, str) or not operation or not isinstance(code, int) or code < 0:
            raise PhaseBContractError("v9 failed Job assignment probe lacks a precise error")
        if not isinstance(detail, str) or not detail or assignment is not None:
            raise PhaseBContractError("v9 failed Job assignment probe detail drifted")
    return WindowsJobAssignmentProbeV9(
        ready=ready,
        operation=operation,
        win32_error_code=code,
        detail=detail,
        assignment=None if assignment is None else dict(assignment),
    )


def build_resource_guard_payload_v9(
    root: Path | str,
    *,
    execution_contract_binding: Mapping[str, str],
    source_closure_binding: Mapping[str, str],
    runtime_lock_binding: Mapping[str, str],
    snapshot: SystemResourceSnapshotV9,
    created_at_utc: str,
    probe: WindowsJobAssignmentProbeV9 | None = None,
) -> dict[str, Any]:
    root_path = safe_project_root(root)
    created = _parse_utc(created_at_utc, label="resource_guard.created_at_utc")
    contract = verify_execution_contract_v9(root_path, execution_contract_binding)
    source_closure, source_sha = _binding(
        root_path,
        source_closure_binding,
        label="source_closure",
        expected_path=SOURCE_CLOSURE_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_source_closure_v9",
    )
    runtime_lock, runtime_sha = _binding(
        root_path,
        runtime_lock_binding,
        label="runtime_lock",
        expected_path=RUNTIME_LOCK_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_runtime_lock_v9",
    )
    observed_probe = probe or probe_windows_job_assignment_v9(
        memory_limit_bytes=int(contract.resource_contract["maximum_training_peak_rss_bytes"])
    )
    total = _integer(snapshot.total_memory_bytes, label="snapshot.total_memory_bytes")
    available = _integer(snapshot.available_memory_bytes, label="snapshot.available_memory_bytes")
    cpu_count = _integer(snapshot.cpu_count, label="snapshot.cpu_count")
    failures: list[str] = []
    if available > total:
        failures.append("available_memory_exceeds_total_memory")
    if cpu_count < int(contract.resource_contract["worker_count"]):
        failures.append("cpu_count_below_worker_count")
    if not observed_probe.ready:
        failures.append("windows_job_assignment_probe_failed")
    return {
        "schema": RESOURCE_GUARD_SCHEMA,
        "loop_id": LOOP_ID,
        "phase_b_execution_contract": dict(execution_contract_binding),
        "source_closure": source_closure,
        "runtime_lock": runtime_lock,
        "canonical_controller_execute_argv": list(CANONICAL_CONTROLLER_EXECUTE_ARGV),
        "canonical_controller_execute_argv_sha256": canonical_argv_sha256(CANONICAL_CONTROLLER_EXECUTE_ARGV),
        "created_at_utc": created.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "maximum_age_seconds": MAXIMUM_GUARD_AGE_SECONDS,
        "snapshot": {
            "total_memory_bytes": total,
            "available_memory_bytes": available,
            "cpu_count": cpu_count,
        },
        "assignment_probe": _probe_payload(observed_probe),
        "guard_ready": not failures,
        "failures": failures,
        "decision": "pass" if not failures else "fail_closed",
        "raw_open_attempts": 0,
        "execution_contract_sha256": contract.contract_sha256,
        "source_closure_sha256": source_sha,
        "runtime_lock_sha256": runtime_sha,
    }


def validate_resource_guard_payload_v9(
    root: Path | str,
    payload: Mapping[str, Any],
    *,
    expected_execution_contract_binding: Mapping[str, str],
    expected_source_closure_binding: Mapping[str, str],
    expected_runtime_lock_binding: Mapping[str, str],
    now_utc: datetime,
) -> None:
    root_path = safe_project_root(root)
    expected_keys = {
        "schema", "loop_id", "phase_b_execution_contract", "source_closure", "runtime_lock",
        "canonical_controller_execute_argv", "canonical_controller_execute_argv_sha256", "created_at_utc",
        "maximum_age_seconds", "snapshot",
        "assignment_probe", "guard_ready", "failures", "decision", "raw_open_attempts",
        "execution_contract_sha256", "source_closure_sha256", "runtime_lock_sha256",
    }
    if set(payload) != expected_keys or payload.get("schema") != RESOURCE_GUARD_SCHEMA or payload.get("loop_id") != LOOP_ID:
        raise PhaseBContractError("v9 resource guard schema drifted")
    if payload.get("phase_b_execution_contract") != dict(expected_execution_contract_binding):
        raise PhaseBContractError("v9 resource guard contract binding drifted")
    if payload.get("source_closure") != dict(expected_source_closure_binding) or payload.get("runtime_lock") != dict(expected_runtime_lock_binding):
        raise PhaseBContractError("v9 resource guard static bindings drifted")
    contract = verify_execution_contract_v9(root_path, expected_execution_contract_binding)
    if payload.get("execution_contract_sha256") != contract.contract_sha256:
        raise PhaseBContractError("v9 resource guard contract digest drifted")
    _, source_sha = _binding(root_path, expected_source_closure_binding, label="source_closure", expected_path=SOURCE_CLOSURE_RELATIVE_PATH, expected_schema="axon_loop167_phase_b_source_closure_v9")
    _, runtime_sha = _binding(root_path, expected_runtime_lock_binding, label="runtime_lock", expected_path=RUNTIME_LOCK_RELATIVE_PATH, expected_schema="axon_loop167_phase_b_runtime_lock_v9")
    if payload.get("source_closure_sha256") != source_sha or payload.get("runtime_lock_sha256") != runtime_sha:
        raise PhaseBContractError("v9 resource guard static digest drifted")
    if payload.get("canonical_controller_execute_argv") != list(CANONICAL_CONTROLLER_EXECUTE_ARGV):
        raise PhaseBContractError("v9 resource guard argv drifted")
    if payload.get("canonical_controller_execute_argv_sha256") != canonical_argv_sha256(CANONICAL_CONTROLLER_EXECUTE_ARGV):
        raise PhaseBContractError("v9 resource guard argv hash drifted")
    if payload.get("maximum_age_seconds") != MAXIMUM_GUARD_AGE_SECONDS:
        raise PhaseBContractError("v9 resource guard maximum age drifted")
    if not isinstance(now_utc, datetime) or now_utc.tzinfo is None:
        raise PhaseBContractError("v9 resource guard validation time is invalid")
    age = (now_utc.astimezone(UTC) - _parse_utc(payload.get("created_at_utc"), label="resource_guard.created_at_utc")).total_seconds()
    if age < 0 or age > MAXIMUM_GUARD_AGE_SECONDS:
        raise PhaseBContractError("v9 resource guard is stale")
    snapshot_payload = payload.get("snapshot")
    if not isinstance(snapshot_payload, Mapping):
        raise PhaseBContractError("v9 resource guard snapshot drifted")
    snapshot = SystemResourceSnapshotV9(
        total_memory_bytes=_integer(snapshot_payload.get("total_memory_bytes"), label="snapshot.total_memory_bytes"),
        available_memory_bytes=_integer(snapshot_payload.get("available_memory_bytes"), label="snapshot.available_memory_bytes"),
        cpu_count=_integer(snapshot_payload.get("cpu_count"), label="snapshot.cpu_count"),
    )
    probe = _validate_probe_payload(payload.get("assignment_probe"))
    expected = build_resource_guard_payload_v9(
        root_path,
        execution_contract_binding=expected_execution_contract_binding,
        source_closure_binding=expected_source_closure_binding,
        runtime_lock_binding=expected_runtime_lock_binding,
        snapshot=snapshot,
        created_at_utc=str(payload["created_at_utc"]),
        probe=probe,
    )
    if dict(payload) != expected:
        raise PhaseBContractError("v9 resource guard payload drifted")
    if payload.get("raw_open_attempts") != 0 or payload.get("guard_ready") is not True:
        raise PhaseBContractError("v9 resource guard is not ready")


def verify_resource_guard_v9(
    root: Path | str,
    binding: Mapping[str, str],
    *,
    expected_execution_contract_binding: Mapping[str, str],
    expected_source_closure_binding: Mapping[str, str],
    expected_runtime_lock_binding: Mapping[str, str],
    now_utc: datetime,
) -> VerifiedResourceGuardV9:
    root_path = safe_project_root(root)
    path = safe_project_path(root_path, RESOURCE_GUARD_RELATIVE_PATH, require_exists=True, require_regular_file=True)
    if not isinstance(binding, Mapping) or binding.get("path") != RESOURCE_GUARD_RELATIVE_PATH:
        raise PhaseBContractError("v9 resource guard path drifted")
    guard_path, digest = verify_safe_file_binding(root_path, binding, label="resource_guard")
    if guard_path != path:
        raise PhaseBContractError("v9 resource guard canonical path drifted")
    payload = require_canonical_json(guard_path)
    validate_resource_guard_payload_v9(
        root_path,
        payload,
        expected_execution_contract_binding=expected_execution_contract_binding,
        expected_source_closure_binding=expected_source_closure_binding,
        expected_runtime_lock_binding=expected_runtime_lock_binding,
        now_utc=now_utc,
    )
    return VerifiedResourceGuardV9(
        guard_path=guard_path,
        guard_sha256=digest,
        execution_contract_sha256=str(payload["execution_contract_sha256"]),
        source_closure_sha256=str(payload["source_closure_sha256"]),
        runtime_lock_sha256=str(payload["runtime_lock_sha256"]),
    )


__all__ = [
    "MAXIMUM_GUARD_AGE_SECONDS",
    "RESOURCE_GUARD_SCHEMA",
    "SystemResourceSnapshotV9",
    "VerifiedResourceGuardV9",
    "build_resource_guard_payload_v9",
    "current_system_snapshot_v9",
    "validate_resource_guard_payload_v9",
    "verify_resource_guard_v9",
]
