"""Create the v10 dynamic launch artifacts in the supervisor invocation only."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

from .contracts import PhaseBContractError, canonical_json_bytes, sha256_file
from .execution_authorization_v10 import (
    VerifiedExecutionAuthorizationV10,
    build_execution_authorization_payload_v10,
    validate_execution_authorization_v10,
)
from .execution_contract_v10 import (
    CONTROLLER_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    RESOURCE_GUARD_RELATIVE_PATH,
    RUN_AUTHORIZATION_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    SUPERVISOR_RELATIVE_PATH,
    assert_output_catalog_is_fresh_v10,
    ensure_v10_static_artifact_parent,
)
from .path_safety_v4 import safe_project_path, safe_project_root
from .resource_guard_v10 import build_resource_guard_payload_v10, current_system_snapshot_v10

MAXIMUM_FRESH_AUTHORIZATION_CHAIN_SECONDS = 30


class FreshAuthorizationV9Error(PhaseBContractError):
    """The v10 dynamic authorization chain cannot prove a fresh launch state."""


@dataclass(frozen=True)
class FreshLaunchAuthorizationV10:
    authorization: VerifiedExecutionAuthorizationV10
    resource_guard_sha256: str
    authorization_created_at_utc: str


def _binding(root: Path, relative_path: str) -> dict[str, str]:
    path = safe_project_path(root, relative_path, require_exists=True, require_regular_file=True)
    return {"path": relative_path, "sha256": sha256_file(path)}


def _is_link_or_reparse(stat_result: os.stat_result) -> bool:
    return stat.S_ISLNK(stat_result.st_mode) or bool(int(getattr(stat_result, "st_file_attributes", 0)) & 0x0400)


def _fsync_parent_directory(parent: Path) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
        kernel32.FlushFileBuffers.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateFileW(str(parent), 0xC0000000, 0x00000007, None, 3, 0x02000000, None)
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, 0, invalid):
            raise FreshAuthorizationV9Error("v10 dynamic artifact parent is unavailable")
        try:
            if not kernel32.FlushFileBuffers(handle):
                raise FreshAuthorizationV9Error("v10 dynamic artifact parent durable sync failed")
        finally:
            kernel32.CloseHandle(handle)
        return
    descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(root: Path, relative_path: str, payload: Mapping[str, object]) -> str:
    output_path = ensure_v10_static_artifact_parent(root, relative_path)
    try:
        stat_result = output_path.lstat()
    except FileNotFoundError:
        stat_result = None
    if stat_result is not None or output_path.is_symlink():
        raise FreshAuthorizationV9Error("v10 dynamic artifact already exists or is unsafe")
    content = canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(output_path, flags, 0o600)
    except FileExistsError as error:
        raise FreshAuthorizationV9Error("v10 dynamic artifact already exists") from error
    except OSError as error:
        raise FreshAuthorizationV9Error("v10 dynamic artifact cannot be created") from error
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_parent_directory(output_path.parent)
    return hashlib.sha256(content).hexdigest()


def create_fresh_launch_authorization_v10(
    root: Path | str,
    *,
    now_utc: datetime,
) -> FreshLaunchAuthorizationV10:
    """Seal guard then authorization immediately before the suspended child exists."""

    if not isinstance(now_utc, datetime) or now_utc.tzinfo is None:
        raise FreshAuthorizationV9Error("v10 fresh authorization requires a UTC timestamp")
    root_path = safe_project_root(root)
    started = datetime.now(UTC)
    assert_output_catalog_is_fresh_v10(root_path)
    for relative_path in (RESOURCE_GUARD_RELATIVE_PATH, RUN_AUTHORIZATION_RELATIVE_PATH):
        path = safe_project_path(root_path, relative_path, require_exists=False)
        if path.exists() or path.is_symlink():
            raise FreshAuthorizationV9Error("v10 execute refuses pre-created dynamic artifacts")
    created_at = now_utc.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    guard_payload = build_resource_guard_payload_v10(
        root_path,
        execution_contract_binding=_binding(root_path, EXECUTION_CONTRACT_RELATIVE_PATH),
        source_closure_binding=_binding(root_path, SOURCE_CLOSURE_RELATIVE_PATH),
        runtime_lock_binding=_binding(root_path, RUNTIME_LOCK_RELATIVE_PATH),
        snapshot=current_system_snapshot_v10(),
        created_at_utc=created_at,
    )
    if guard_payload.get("guard_ready") is not True:
        raise FreshAuthorizationV9Error("v10 fresh resource guard failed closed")
    guard_sha = _write_new(root_path, RESOURCE_GUARD_RELATIVE_PATH, guard_payload)
    authorization_payload = build_execution_authorization_payload_v10(
        root_path,
        execution_contract_binding=_binding(root_path, EXECUTION_CONTRACT_RELATIVE_PATH),
        source_closure_binding=_binding(root_path, SOURCE_CLOSURE_RELATIVE_PATH),
        runtime_lock_binding=_binding(root_path, RUNTIME_LOCK_RELATIVE_PATH),
        controller_binding=_binding(root_path, CONTROLLER_RELATIVE_PATH),
        supervisor_binding=_binding(root_path, SUPERVISOR_RELATIVE_PATH),
        resource_guard_binding={"path": RESOURCE_GUARD_RELATIVE_PATH, "sha256": guard_sha},
        created_at_utc=created_at,
    )
    _write_new(root_path, RUN_AUTHORIZATION_RELATIVE_PATH, authorization_payload)
    if (datetime.now(UTC) - started).total_seconds() > MAXIMUM_FRESH_AUTHORIZATION_CHAIN_SECONDS:
        raise FreshAuthorizationV9Error("v10 dynamic authorization chain exceeded its launch budget")
    authorization = validate_execution_authorization_v10(
        root_path,
        safe_project_path(root_path, RUN_AUTHORIZATION_RELATIVE_PATH, require_exists=True, require_regular_file=True),
        now_utc=datetime.now(UTC),
        phase="prelaunch",
    )
    return FreshLaunchAuthorizationV10(authorization, guard_sha, created_at)


__all__ = [
    "FreshAuthorizationV9Error",
    "FreshLaunchAuthorizationV10",
    "MAXIMUM_FRESH_AUTHORIZATION_CHAIN_SECONDS",
    "create_fresh_launch_authorization_v10",
]
