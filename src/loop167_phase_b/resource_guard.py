"""Fail-closed resource budgeting for a future Loop167 Phase-B launch."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import PhaseBContractError, canonical_argv_sha256

RESOURCE_FIELDS = (
    "maximum_raw_open_attempts",
    "maximum_raw_bytes",
    "maximum_feature_cache_bytes",
    "maximum_extraction_peak_rss_bytes",
    "maximum_training_peak_rss_bytes",
    "maximum_extraction_wall_seconds",
    "maximum_training_wall_seconds",
    "reserved_seal_evaluation_wall_seconds",
    "maximum_total_wall_seconds",
    "worker_count",
    "thread_count",
    "maximum_gpu_allocated_bytes",
    "kill_conditions",
)


@dataclass(frozen=True)
class SystemResourceSnapshot:
    total_memory_bytes: int
    available_memory_bytes: int
    cpu_count: int


@dataclass(frozen=True)
class ResourceGuardResult:
    ready: bool
    failures: tuple[str, ...]
    minimum_available_memory_bytes: int


def _nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PhaseBContractError(f"{label} must be a non-negative integer")
    return value


def validate_resource_contract(resource_contract: Mapping[str, Any]) -> dict[str, Any]:
    if set(resource_contract) != set(RESOURCE_FIELDS):
        raise PhaseBContractError("Resource contract fields drifted")
    values = dict(resource_contract)
    for name in RESOURCE_FIELDS:
        if name == "kill_conditions":
            continue
        _nonnegative_integer(values[name], label=name)
    if not isinstance(values["kill_conditions"], list) or not values["kill_conditions"]:
        raise PhaseBContractError("Resource contract kill conditions drifted")
    if values["maximum_raw_open_attempts"] != 20000 or values["maximum_raw_bytes"] != 26843545600:
        raise PhaseBContractError("Resource contract raw budget drifted")
    if values["worker_count"] != 1 or values["thread_count"] != 1:
        raise PhaseBContractError("Resource contract concurrency budget drifted")
    if values["maximum_gpu_allocated_bytes"] != 0:
        raise PhaseBContractError("Resource contract must forbid GPU allocation")
    if (
        values["maximum_extraction_wall_seconds"]
        + values["maximum_training_wall_seconds"]
        + values["reserved_seal_evaluation_wall_seconds"]
        != values["maximum_total_wall_seconds"]
    ):
        raise PhaseBContractError("Resource contract wall-time budget is not closed")
    return values


def minimum_available_memory_bytes(resource_contract: Mapping[str, Any]) -> int:
    values = validate_resource_contract(resource_contract)
    return max(
        int(values["maximum_extraction_peak_rss_bytes"]),
        int(values["maximum_training_peak_rss_bytes"]),
    ) + 2 * 1024 * 1024 * 1024


def evaluate_resource_guard(
    snapshot: SystemResourceSnapshot,
    resource_contract: Mapping[str, Any],
) -> ResourceGuardResult:
    values = validate_resource_contract(resource_contract)
    total_memory = _nonnegative_integer(snapshot.total_memory_bytes, label="total_memory_bytes")
    available_memory = _nonnegative_integer(snapshot.available_memory_bytes, label="available_memory_bytes")
    cpu_count = _nonnegative_integer(snapshot.cpu_count, label="cpu_count")
    minimum_available = minimum_available_memory_bytes(values)
    failures: list[str] = []
    if available_memory > total_memory:
        failures.append("available_memory_exceeds_total_memory")
    if available_memory < minimum_available:
        failures.append("available_memory_below_launch_floor")
    if cpu_count < values["worker_count"]:
        failures.append("cpu_count_below_worker_count")
    return ResourceGuardResult(not failures, tuple(failures), minimum_available)


def current_system_snapshot() -> SystemResourceSnapshot:
    if os.name == "nt":
        class _MemoryStatus(ctypes.Structure):
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

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise PhaseBContractError("Unable to read Windows memory status")
        return SystemResourceSnapshot(int(status.ullTotalPhys), int(status.ullAvailPhys), os.cpu_count() or 0)
    page_size = os.sysconf("SC_PAGE_SIZE")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    total_pages = os.sysconf("SC_PHYS_PAGES")
    return SystemResourceSnapshot(
        int(total_pages * page_size),
        int(available_pages * page_size),
        os.cpu_count() or 0,
    )


def build_resource_guard_payload(
    *,
    source_closure_binding: Mapping[str, str],
    protocol_binding: Mapping[str, str],
    runtime_lock_binding: Mapping[str, str],
    controller_binding: Mapping[str, str],
    canonical_argv: tuple[str, ...],
    resource_contract: Mapping[str, Any],
    snapshot: SystemResourceSnapshot,
    created_at_utc: str,
) -> dict[str, Any]:
    if not isinstance(created_at_utc, str) or not created_at_utc.endswith("Z"):
        raise PhaseBContractError("Resource guard timestamp must be a UTC Z string")
    result = evaluate_resource_guard(snapshot, resource_contract)
    return {
        "schema": "axon_loop167_phase_b_resource_guard_v1",
        "loop_id": "loop167_ember_v3_novel_delta",
        "source_closure": dict(source_closure_binding),
        "phase_b_protocol": dict(protocol_binding),
        "runtime_lock": dict(runtime_lock_binding),
        "controller": dict(controller_binding),
        "canonical_argv": list(canonical_argv),
        "canonical_argv_sha256": canonical_argv_sha256(canonical_argv),
        "resource_contract": dict(resource_contract),
        "created_at_utc": created_at_utc,
        "maximum_age_seconds": 300,
        "snapshot": {
            "total_memory_bytes": snapshot.total_memory_bytes,
            "available_memory_bytes": snapshot.available_memory_bytes,
            "cpu_count": snapshot.cpu_count,
        },
        "minimum_available_memory_bytes": result.minimum_available_memory_bytes,
        "guard_ready": result.ready,
        "failures": list(result.failures),
        "decision": "pass" if result.ready else "fail_closed",
        "raw_open_attempts": 0,
    }
