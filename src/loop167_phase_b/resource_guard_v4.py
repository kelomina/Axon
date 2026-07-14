"""Fresh resource and Windows Job gate for the Loop167 Phase-B v4 execution."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import (
    PhaseBContractError,
    canonical_argv_sha256,
    require_canonical_json,
    verify_file_binding,
)
from .execution_contract_v4 import (
    CANONICAL_EXECUTE_ARGV,
    LOOP_ID,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    canonical_execute_argv_v4,
    verify_execution_contract_v4,
)

RESOURCE_GUARD_SCHEMA = "axon_loop167_phase_b_resource_guard_v4"
MAXIMUM_GUARD_AGE_SECONDS = 300
MEMORY_HEADROOM_BYTES = 2 * 1024 * 1024 * 1024

JobObjectProbe = Callable[[int], tuple[bool, str | None]]


@dataclass(frozen=True)
class SystemResourceSnapshotV4:
    total_memory_bytes: int
    available_memory_bytes: int
    cpu_count: int


@dataclass(frozen=True)
class ResourceGuardResultV4:
    ready: bool
    failures: tuple[str, ...]
    minimum_available_memory_bytes: int
    job_object_ready: bool
    job_object_detail: str | None


@dataclass(frozen=True)
class VerifiedResourceGuardV4:
    guard_path: Path
    guard_sha256: str
    execution_contract_sha256: str
    source_closure_sha256: str
    runtime_lock_sha256: str
    canonical_execute_argv: tuple[str, ...]


def _nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PhaseBContractError(f"{label} must be a non-negative integer")
    return value


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PhaseBContractError(f"{label} must be a UTC Z timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as exc:
        raise PhaseBContractError(f"{label} is invalid") from exc


def _require_fixed_binding_path(binding: object, *, label: str, expected_path: str) -> None:
    if not isinstance(binding, dict) or binding.get("path") != expected_path:
        raise PhaseBContractError(f"{label} path is outside the fixed v4 contract")


def _verify_static_json_binding(
    root: Path,
    binding: Mapping[str, str],
    *,
    label: str,
    expected_path: str,
    expected_schema: str,
) -> tuple[Path, str]:
    _require_fixed_binding_path(binding, label=label, expected_path=expected_path)
    path, digest = verify_file_binding(root, dict(binding), label=label)
    payload = require_canonical_json(path)
    if payload.get("schema") != expected_schema or payload.get("loop_id") != LOOP_ID:
        raise PhaseBContractError(f"{label} schema or loop id drifted")
    return path, digest


def minimum_available_memory_bytes_v4(resource_contract: Mapping[str, Any]) -> int:
    maximum_training_rss = _nonnegative_integer(
        resource_contract.get("maximum_training_peak_rss_bytes"),
        label="maximum_training_peak_rss_bytes",
    )
    maximum_extraction_rss = _nonnegative_integer(
        resource_contract.get("maximum_extraction_peak_rss_bytes"),
        label="maximum_extraction_peak_rss_bytes",
    )
    return max(maximum_training_rss, maximum_extraction_rss) + MEMORY_HEADROOM_BYTES


def _normalize_probe_result(result: object) -> tuple[bool, str | None]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise PhaseBContractError("Windows Job probe returned an invalid result")
    ready, detail = result
    if not isinstance(ready, bool):
        raise PhaseBContractError("Windows Job probe readiness must be boolean")
    if detail is not None and not isinstance(detail, str):
        raise PhaseBContractError("Windows Job probe detail must be a string or null")
    if ready and detail is not None:
        raise PhaseBContractError("A ready Windows Job probe may not include a failure detail")
    if not ready and (not isinstance(detail, str) or not detail or len(detail) > 512):
        raise PhaseBContractError("An unavailable Windows Job probe needs a bounded failure detail")
    return ready, detail


def default_windows_job_probe_v4(memory_limit_bytes: int) -> tuple[bool, str | None]:
    """Probe a disposable Job Object; production callers cannot inject readiness."""

    from .windows_job_v4 import probe_windows_job_ready

    return probe_windows_job_ready(memory_limit_bytes=memory_limit_bytes)


def evaluate_resource_guard_v4(
    snapshot: SystemResourceSnapshotV4,
    resource_contract: Mapping[str, Any],
    *,
    job_object_ready: bool,
    job_object_detail: str | None,
) -> ResourceGuardResultV4:
    """Evaluate only static resource facts and the non-assigning Job probe."""

    total_memory = _nonnegative_integer(snapshot.total_memory_bytes, label="total_memory_bytes")
    available_memory = _nonnegative_integer(
        snapshot.available_memory_bytes,
        label="available_memory_bytes",
    )
    cpu_count = _nonnegative_integer(snapshot.cpu_count, label="cpu_count")
    worker_count = _nonnegative_integer(resource_contract.get("worker_count"), label="worker_count")
    minimum_available = minimum_available_memory_bytes_v4(resource_contract)
    ready, detail = _normalize_probe_result((job_object_ready, job_object_detail))

    failures: list[str] = []
    if available_memory > total_memory:
        failures.append("available_memory_exceeds_total_memory")
    if available_memory < minimum_available:
        failures.append("available_memory_below_launch_floor")
    if cpu_count < worker_count:
        failures.append("cpu_count_below_worker_count")
    if not ready:
        failures.append("windows_job_unavailable")
    return ResourceGuardResultV4(
        ready=not failures,
        failures=tuple(failures),
        minimum_available_memory_bytes=minimum_available,
        job_object_ready=ready,
        job_object_detail=detail,
    )


def current_system_snapshot_v4() -> SystemResourceSnapshotV4:
    """Read host memory without opening a protected input or importing model packages."""

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
        return SystemResourceSnapshotV4(
            total_memory_bytes=int(status.ullTotalPhys),
            available_memory_bytes=int(status.ullAvailPhys),
            cpu_count=os.cpu_count() or 0,
        )

    page_size = os.sysconf("SC_PAGE_SIZE")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    total_pages = os.sysconf("SC_PHYS_PAGES")
    return SystemResourceSnapshotV4(
        total_memory_bytes=int(total_pages * page_size),
        available_memory_bytes=int(available_pages * page_size),
        cpu_count=os.cpu_count() or 0,
    )


def build_resource_guard_payload_v4(
    root: Path,
    *,
    execution_contract_binding: Mapping[str, str],
    source_closure_binding: Mapping[str, str],
    runtime_lock_binding: Mapping[str, str],
    canonical_execute_argv: Sequence[str],
    snapshot: SystemResourceSnapshotV4,
    created_at_utc: str,
    job_object_probe: JobObjectProbe | None = None,
) -> dict[str, Any]:
    """Build a guard from sealed inputs; a failed probe remains non-authorizing."""

    _parse_utc(created_at_utc, label="resource_guard.created_at_utc")
    argv = canonical_execute_argv_v4(canonical_execute_argv)
    contract = verify_execution_contract_v4(root, execution_contract_binding)
    if argv != contract.canonical_execute_argv:
        raise PhaseBContractError("Resource guard argv differs from the execution contract")
    _, source_closure_sha256 = _verify_static_json_binding(
        root,
        source_closure_binding,
        label="source_closure",
        expected_path=SOURCE_CLOSURE_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_source_closure_v4",
    )
    _, runtime_lock_sha256 = _verify_static_json_binding(
        root,
        runtime_lock_binding,
        label="runtime_lock",
        expected_path=RUNTIME_LOCK_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_runtime_lock_v4",
    )
    probe = default_windows_job_probe_v4 if job_object_probe is None else job_object_probe
    job_ready, job_detail = _normalize_probe_result(
        probe(int(contract.resource_contract["maximum_training_peak_rss_bytes"]))
    )
    result = evaluate_resource_guard_v4(
        snapshot,
        contract.resource_contract,
        job_object_ready=job_ready,
        job_object_detail=job_detail,
    )
    return {
        "schema": RESOURCE_GUARD_SCHEMA,
        "loop_id": LOOP_ID,
        "phase_b_execution_contract": dict(execution_contract_binding),
        "source_closure": dict(source_closure_binding),
        "runtime_lock": dict(runtime_lock_binding),
        "canonical_execute_argv": list(argv),
        "canonical_execute_argv_sha256": canonical_argv_sha256(argv),
        "created_at_utc": created_at_utc,
        "maximum_age_seconds": MAXIMUM_GUARD_AGE_SECONDS,
        "snapshot": {
            "total_memory_bytes": snapshot.total_memory_bytes,
            "available_memory_bytes": snapshot.available_memory_bytes,
            "cpu_count": snapshot.cpu_count,
        },
        "minimum_available_memory_bytes": result.minimum_available_memory_bytes,
        "job_object_ready": result.job_object_ready,
        "job_object_detail": result.job_object_detail,
        "guard_ready": result.ready,
        "failures": list(result.failures),
        "decision": "pass" if result.ready else "fail_closed",
        "raw_open_attempts": 0,
        "execution_contract_sha256": contract.contract_sha256,
        "source_closure_sha256": source_closure_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
    }


def validate_resource_guard_payload_v4(
    root: Path,
    payload: Mapping[str, Any],
    *,
    expected_execution_contract_binding: Mapping[str, str],
    expected_source_closure_binding: Mapping[str, str],
    expected_runtime_lock_binding: Mapping[str, str],
    canonical_execute_argv: Sequence[str],
    now_utc: datetime,
) -> None:
    """Require a fresh passing guard before authorization or lease consumption."""

    expected_keys = {
        "schema",
        "loop_id",
        "phase_b_execution_contract",
        "source_closure",
        "runtime_lock",
        "canonical_execute_argv",
        "canonical_execute_argv_sha256",
        "created_at_utc",
        "maximum_age_seconds",
        "snapshot",
        "minimum_available_memory_bytes",
        "job_object_ready",
        "job_object_detail",
        "guard_ready",
        "failures",
        "decision",
        "raw_open_attempts",
        "execution_contract_sha256",
        "source_closure_sha256",
        "runtime_lock_sha256",
    }
    if set(payload) != expected_keys or payload.get("schema") != RESOURCE_GUARD_SCHEMA:
        raise PhaseBContractError("Resource guard v4 schema drifted")
    if payload.get("loop_id") != LOOP_ID:
        raise PhaseBContractError("Resource guard v4 loop id drifted")
    argv = canonical_execute_argv_v4(canonical_execute_argv)
    if payload["canonical_execute_argv"] != list(argv):
        raise PhaseBContractError("Resource guard execute argv drifted")
    if payload["canonical_execute_argv_sha256"] != canonical_argv_sha256(argv):
        raise PhaseBContractError("Resource guard execute argv hash drifted")
    if payload["phase_b_execution_contract"] != dict(expected_execution_contract_binding):
        raise PhaseBContractError("Resource guard execution contract binding drifted")
    if payload["source_closure"] != dict(expected_source_closure_binding):
        raise PhaseBContractError("Resource guard source closure binding drifted")
    if payload["runtime_lock"] != dict(expected_runtime_lock_binding):
        raise PhaseBContractError("Resource guard runtime lock binding drifted")

    contract = verify_execution_contract_v4(root, expected_execution_contract_binding)
    if payload["execution_contract_sha256"] != contract.contract_sha256:
        raise PhaseBContractError("Resource guard execution contract digest drifted")
    _, source_closure_sha256 = _verify_static_json_binding(
        root,
        expected_source_closure_binding,
        label="source_closure",
        expected_path=SOURCE_CLOSURE_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_source_closure_v4",
    )
    _, runtime_lock_sha256 = _verify_static_json_binding(
        root,
        expected_runtime_lock_binding,
        label="runtime_lock",
        expected_path=RUNTIME_LOCK_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_runtime_lock_v4",
    )
    if payload["source_closure_sha256"] != source_closure_sha256:
        raise PhaseBContractError("Resource guard source closure digest drifted")
    if payload["runtime_lock_sha256"] != runtime_lock_sha256:
        raise PhaseBContractError("Resource guard runtime lock digest drifted")

    if isinstance(payload["maximum_age_seconds"], bool) or payload["maximum_age_seconds"] != MAXIMUM_GUARD_AGE_SECONDS:
        raise PhaseBContractError("Resource guard maximum age drifted")
    created_at = _parse_utc(payload["created_at_utc"], label="resource_guard.created_at_utc")
    if now_utc.tzinfo is None:
        raise PhaseBContractError("Resource guard validation time must be timezone-aware")
    current = now_utc.astimezone(UTC)
    age_seconds = (current - created_at).total_seconds()
    if age_seconds < 0 or age_seconds > MAXIMUM_GUARD_AGE_SECONDS:
        raise PhaseBContractError("Resource guard is stale or from the future")

    snapshot_payload = payload["snapshot"]
    if not isinstance(snapshot_payload, dict) or set(snapshot_payload) != {
        "total_memory_bytes",
        "available_memory_bytes",
        "cpu_count",
    }:
        raise PhaseBContractError("Resource guard snapshot drifted")
    snapshot = SystemResourceSnapshotV4(
        total_memory_bytes=_nonnegative_integer(
            snapshot_payload["total_memory_bytes"],
            label="snapshot.total_memory_bytes",
        ),
        available_memory_bytes=_nonnegative_integer(
            snapshot_payload["available_memory_bytes"],
            label="snapshot.available_memory_bytes",
        ),
        cpu_count=_nonnegative_integer(snapshot_payload["cpu_count"], label="snapshot.cpu_count"),
    )
    result = evaluate_resource_guard_v4(
        snapshot,
        contract.resource_contract,
        job_object_ready=payload["job_object_ready"],
        job_object_detail=payload["job_object_detail"],
    )
    if payload["minimum_available_memory_bytes"] != result.minimum_available_memory_bytes:
        raise PhaseBContractError("Resource guard memory floor drifted")
    if payload["failures"] != list(result.failures):
        raise PhaseBContractError("Resource guard failure list drifted")
    if payload["guard_ready"] is not result.ready:
        raise PhaseBContractError("Resource guard readiness drifted")
    if payload["decision"] != ("pass" if result.ready else "fail_closed"):
        raise PhaseBContractError("Resource guard decision drifted")
    if payload["raw_open_attempts"] != 0:
        raise PhaseBContractError("Resource guard recorded raw access")
    if not result.ready:
        raise PhaseBContractError("Resource guard is not ready")


def verify_resource_guard_v4(
    root: Path,
    guard_binding: Mapping[str, str],
    *,
    expected_execution_contract_binding: Mapping[str, str],
    expected_source_closure_binding: Mapping[str, str],
    expected_runtime_lock_binding: Mapping[str, str],
    canonical_execute_argv: Sequence[str] = CANONICAL_EXECUTE_ARGV,
    now_utc: datetime,
) -> VerifiedResourceGuardV4:
    """Verify a fresh passing guard and return only its safe binding facts."""

    _require_fixed_binding_path(
        guard_binding,
        label="resource_guard",
        expected_path="manifests/roadmap_9997/loop167_ember_v3_novel_delta/phase_b_resource_guard_v4.json",
    )
    guard_path, guard_sha256 = verify_file_binding(root, dict(guard_binding), label="resource_guard")
    payload = require_canonical_json(guard_path)
    validate_resource_guard_payload_v4(
        root,
        payload,
        expected_execution_contract_binding=expected_execution_contract_binding,
        expected_source_closure_binding=expected_source_closure_binding,
        expected_runtime_lock_binding=expected_runtime_lock_binding,
        canonical_execute_argv=canonical_execute_argv,
        now_utc=now_utc,
    )
    return VerifiedResourceGuardV4(
        guard_path=guard_path,
        guard_sha256=guard_sha256,
        execution_contract_sha256=str(payload["execution_contract_sha256"]),
        source_closure_sha256=str(payload["source_closure_sha256"]),
        runtime_lock_sha256=str(payload["runtime_lock_sha256"]),
        canonical_execute_argv=tuple(payload["canonical_execute_argv"]),
    )

