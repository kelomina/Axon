"""One-shot, durable execution lease for Loop167 Phase-B v5."""

from __future__ import annotations

import ctypes
import hashlib
import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    PhaseBContractError,
    canonical_argv_sha256,
    canonical_json_bytes,
    require_canonical_json,
    sha256_file,
)
from .execution_authorization_v5 import (
    VerifiedExecutionAuthorizationV5,
    validate_execution_authorization_v5,
)
from .execution_contract_v5 import (
    CONTROLLER_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    EXPECTED_LEASE,
    FIXED_OUTPUT_CATALOG,
    LOOP_ID,
    PHASE_B_PROTOCOL_RELATIVE_PATH,
    RESOURCE_GUARD_RELATIVE_PATH,
    RUN_AUTHORIZATION_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
)
from .path_safety_v4 import (
    WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT,
    canonical_project_relative_path,
    safe_project_path,
    safe_project_relative_path,
    safe_project_root,
)

LEASE_SCHEMA = "axon_loop167_phase_b_execution_lease_v5"
LEASE_STATUS = "consumed_before_first_raw_open"


class ExecutionLeaseError(PhaseBContractError):
    """Raised when the non-retryable v5 execution lease cannot be consumed safely."""


@dataclass(frozen=True)
class ConsumedExecutionLeaseV5:
    """Durable evidence that the sole raw-pass authority was consumed."""

    marker_path: Path
    marker_sha256: str
    authorization_sha256: str
    payload: Mapping[str, Any]


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ExecutionLeaseError(f"{label} must be a UTC Z timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as exc:
        raise ExecutionLeaseError(f"{label} is invalid") from exc


def _canonical_timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ExecutionLeaseError("Lease consumption time must be timezone-aware")
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_link_or_reparse(stat_result: os.stat_result) -> bool:
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    return stat.S_ISLNK(stat_result.st_mode) or bool(attributes & WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT)


def _ensure_safe_parent_directory(root: Path, marker_relative_path: str) -> Path:
    current = safe_project_root(root)
    parent_parts = marker_relative_path.split("/")[:-1]
    for part in parent_parts:
        current = current / part
        try:
            stat_result = current.lstat()
        except FileNotFoundError:
            try:
                os.mkdir(current, 0o700)
            except FileExistsError:
                pass
            except OSError as exc:
                raise ExecutionLeaseError("Lease parent directory cannot be created safely") from exc
            try:
                stat_result = current.lstat()
            except OSError as exc:
                raise ExecutionLeaseError("Lease parent directory cannot be inspected safely") from exc
        except OSError as exc:
            raise ExecutionLeaseError("Lease parent directory cannot be inspected safely") from exc
        if _is_link_or_reparse(stat_result) or not stat.S_ISDIR(stat_result.st_mode):
            raise ExecutionLeaseError("Lease parent directory is a symlink, reparse point, or non-directory")
    return current


def _fsync_parent_directory(parent: Path) -> None:
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateFileW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_void_p,
        ]
        kernel32.CreateFileW.restype = ctypes.c_void_p
        kernel32.FlushFileBuffers.argtypes = [ctypes.c_void_p]
        kernel32.FlushFileBuffers.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.CreateFileW(
            str(parent),
            0x80000000 | 0x40000000,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,
            0x02000000,
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle in (None, 0, invalid_handle):
            raise ExecutionLeaseError("Lease parent directory cannot be opened for durable sync")
        try:
            if not kernel32.FlushFileBuffers(ctypes.c_void_p(handle)):
                raise ExecutionLeaseError("Lease parent directory durable sync failed")
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(parent, flags)
    except OSError as exc:
        raise ExecutionLeaseError("Lease parent directory cannot be opened for durable sync") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise ExecutionLeaseError("Lease parent directory durable sync failed") from exc
    finally:
        os.close(descriptor)


def _write_marker_exclusive(marker_path: Path, payload: Mapping[str, Any]) -> ConsumedExecutionLeaseV5:
    content = canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(marker_path, flags, 0o600)
    except FileExistsError as exc:
        raise ExecutionLeaseError("Execution lease has already been consumed") from exc
    except OSError as exc:
        raise ExecutionLeaseError("Execution lease marker cannot be created safely") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_parent_directory(marker_path.parent)
    except Exception as exc:
        raise ExecutionLeaseError("Execution lease marker remains consumed after a durability failure") from exc
    return ConsumedExecutionLeaseV5(
        marker_path=marker_path,
        marker_sha256=hashlib.sha256(content).hexdigest(),
        authorization_sha256=str(payload["run_authorization"]["sha256"]),
        payload=dict(payload),
    )


def _verify_unchanged_bound_file(
    root: Path,
    binding: Mapping[str, str],
    *,
    label: str,
    expected_relative_path: str,
) -> None:
    if set(binding) != {"path", "sha256"} or binding["path"] != expected_relative_path:
        raise ExecutionLeaseError(f"{label} path drifted after lease consumption")
    if not isinstance(binding["sha256"], str) or len(binding["sha256"]) != 64:
        raise ExecutionLeaseError(f"{label} digest drifted after lease consumption")
    path = safe_project_path(root, expected_relative_path, require_exists=True, require_regular_file=True)
    if sha256_file(path) != binding["sha256"]:
        raise ExecutionLeaseError(f"{label} changed after lease consumption")


def build_execution_lease_payload_v5(
    authorization: VerifiedExecutionAuthorizationV5,
    *,
    consumed_at_utc: datetime,
) -> dict[str, Any]:
    """Bind a consumed marker to the exact authorization and sealed input graph."""

    if not isinstance(authorization, VerifiedExecutionAuthorizationV5):
        raise TypeError("authorization must be a VerifiedExecutionAuthorizationV5")
    authorization_relative_path = safe_project_relative_path(
        authorization.project_root,
        authorization.authorization_path,
        require_exists=True,
        require_regular_file=True,
    )
    if authorization_relative_path != RUN_AUTHORIZATION_RELATIVE_PATH:
        raise ExecutionLeaseError("Execution authorization path drifted before lease consumption")
    return {
        "schema": LEASE_SCHEMA,
        "loop_id": LOOP_ID,
        "status": LEASE_STATUS,
        "consumed_at_utc": _canonical_timestamp(consumed_at_utc),
        "run_authorization": {
            "path": RUN_AUTHORIZATION_RELATIVE_PATH,
            "sha256": authorization.authorization_sha256,
        },
        "phase_b_execution_contract": dict(authorization.execution_contract_binding),
        "phase_b_protocol": dict(authorization.protocol_binding),
        "source_closure": dict(authorization.source_closure_binding),
        "runtime_lock": dict(authorization.runtime_lock_binding),
        "controller": dict(authorization.controller_binding),
        "resource_guard": dict(authorization.resource_guard_binding),
        "canonical_execute_argv": list(authorization.canonical_execute_argv),
        "canonical_execute_argv_sha256": canonical_argv_sha256(authorization.canonical_execute_argv),
        "output_catalog": [dict(entry) for entry in FIXED_OUTPUT_CATALOG],
        "output_catalog_sha256": authorization.output_catalog_sha256,
        "lease": dict(EXPECTED_LEASE),
        "raw_open_attempts_before_consume": 0,
    }


def consume_execution_lease_v5(
    root: Path | str,
    authorization_path: Path | str,
    *,
    now_utc: datetime,
) -> ConsumedExecutionLeaseV5:
    """Revalidate, then atomically burn the one-shot lease before any raw open."""

    root_path = safe_project_root(root)
    authorization = validate_execution_authorization_v5(
        root_path,
        authorization_path,
        now_utc=now_utc,
    )
    if sha256_file(authorization.authorization_path) != authorization.authorization_sha256:
        raise ExecutionLeaseError("Execution authorization changed before lease consumption")
    marker_relative_path = canonical_project_relative_path(EXPECTED_LEASE["marker_path"])
    marker_path = safe_project_path(root_path, marker_relative_path, require_exists=False)
    if marker_path != authorization.lease_marker_path:
        raise ExecutionLeaseError("Execution authorization lease marker path drifted")
    _ensure_safe_parent_directory(root_path, marker_relative_path)
    marker_path = safe_project_path(root_path, marker_relative_path, require_exists=False)
    if marker_path.exists() or marker_path.is_symlink():
        raise ExecutionLeaseError("Execution lease has already been consumed or is unsafe")
    payload = build_execution_lease_payload_v5(authorization, consumed_at_utc=now_utc)
    return _write_marker_exclusive(marker_path, payload)


def verify_consumed_execution_lease_v5(
    root: Path | str,
    authorization: VerifiedExecutionAuthorizationV5,
) -> ConsumedExecutionLeaseV5:
    """Verify an already-burned marker before its holder opens a protected input."""

    if not isinstance(authorization, VerifiedExecutionAuthorizationV5):
        raise TypeError("authorization must be a VerifiedExecutionAuthorizationV5")
    root_path = safe_project_root(root)
    if root_path != authorization.project_root:
        raise ExecutionLeaseError("Execution lease verification root drifted")
    marker_relative_path = canonical_project_relative_path(EXPECTED_LEASE["marker_path"])
    marker_path = safe_project_path(
        root_path,
        marker_relative_path,
        require_exists=True,
        require_regular_file=True,
    )
    if marker_path != authorization.lease_marker_path:
        raise ExecutionLeaseError("Execution lease marker path drifted")
    if sha256_file(authorization.authorization_path) != authorization.authorization_sha256:
        raise ExecutionLeaseError("Execution authorization changed after lease consumption")
    _verify_unchanged_bound_file(
        root_path,
        authorization.execution_contract_binding,
        label="execution contract",
        expected_relative_path=EXECUTION_CONTRACT_RELATIVE_PATH,
    )
    _verify_unchanged_bound_file(
        root_path,
        authorization.protocol_binding,
        label="phase-B protocol",
        expected_relative_path=PHASE_B_PROTOCOL_RELATIVE_PATH,
    )
    _verify_unchanged_bound_file(
        root_path,
        authorization.source_closure_binding,
        label="source closure",
        expected_relative_path=SOURCE_CLOSURE_RELATIVE_PATH,
    )
    _verify_unchanged_bound_file(
        root_path,
        authorization.runtime_lock_binding,
        label="runtime lock",
        expected_relative_path=RUNTIME_LOCK_RELATIVE_PATH,
    )
    _verify_unchanged_bound_file(
        root_path,
        authorization.controller_binding,
        label="controller",
        expected_relative_path=CONTROLLER_RELATIVE_PATH,
    )
    _verify_unchanged_bound_file(
        root_path,
        authorization.resource_guard_binding,
        label="resource guard",
        expected_relative_path=RESOURCE_GUARD_RELATIVE_PATH,
    )
    payload = require_canonical_json(marker_path)
    timestamp = _parse_utc(payload.get("consumed_at_utc"), label="execution_lease.consumed_at_utc")
    expected = build_execution_lease_payload_v5(authorization, consumed_at_utc=timestamp)
    if payload != expected:
        raise ExecutionLeaseError("Execution lease payload binding drifted")
    return ConsumedExecutionLeaseV5(
        marker_path=marker_path,
        marker_sha256=sha256_file(marker_path),
        authorization_sha256=authorization.authorization_sha256,
        payload=payload,
    )
