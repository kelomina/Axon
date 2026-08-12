"""Raw-free suspended-child supervision for the Loop167 Phase-B v12 route."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .contracts import PhaseBContractError, canonical_json_bytes, sha256_file
from .path_safety_v4 import safe_project_path, safe_project_relative_path
from .windows_job_v12 import WindowsJobV12, WindowsJobV12Error

LAUNCH_RECEIPT_SCHEMA = "axon_loop167_phase_b_v9_pre_resume_launch_receipt"
EXIT_RECEIPT_SCHEMA = "axon_loop167_phase_b_v9_supervisor_exit_receipt"
FAILURE_RECEIPT_SCHEMA = "axon_loop167_phase_b_v9_supervisor_failure_receipt"
MAXIMUM_TIMEOUT_SECONDS = 28_800


class SupervisorV12Error(PhaseBContractError):
    """The raw-free controller supervisor cannot prove containment."""


@dataclass(frozen=True)
class SupervisorConfigV12:
    project_root: Path
    mode: str
    command: tuple[str, ...]
    launch_receipt: Path
    exit_receipt: Path
    failure_receipt: Path
    memory_limit_bytes: int
    timeout_seconds: int
    static_bindings: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class SupervisedRunResultV12:
    returncode: int
    launch_receipt_sha256: str
    exit_receipt_sha256: str
    job_audit: Mapping[str, Any]


@dataclass(frozen=True)
class ValidatedLaunchReceiptV12:
    """A launch receipt and the digest of the exact bytes that were validated."""

    receipt_path: Path
    payload: Mapping[str, Any]
    canonical_sha256: str


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(command))).hexdigest()


def _current_process_identity() -> dict[str, int | str]:
    if os.name != "nt":
        raise SupervisorV12Error("v12 supervisor identity is available only on Windows")
    import ctypes
    from ctypes import wintypes

    class FileTime(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    creation = FileTime()
    exit_time = FileTime()
    kernel_time = FileTime()
    user_time = FileTime()
    if not kernel32.GetProcessTimes(
        kernel32.GetCurrentProcess(),
        ctypes.byref(creation),
        ctypes.byref(exit_time),
        ctypes.byref(kernel_time),
        ctypes.byref(user_time),
    ):
        raise SupervisorV12Error(f"Supervisor GetProcessTimes failed: Win32 {ctypes.get_last_error()}")
    return {
        "pid": os.getpid(),
        "creation_time_filetime": (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime),
        "verification_scope": "current_supervisor_process_identity",
    }


def _is_link_or_reparse(stat_result: os.stat_result) -> bool:
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    return stat.S_ISLNK(stat_result.st_mode) or bool(attributes & 0x0400)


def _safe_output_path(root: Path, path: Path) -> Path:
    try:
        root_path = root.resolve(strict=True)
        relative_path = path.relative_to(root_path)
    except (OSError, ValueError) as error:
        raise SupervisorV12Error("Supervisor output is outside the project root") from error
    cursor = root_path
    for component in relative_path.parts[:-1]:
        cursor = cursor / component
        try:
            cursor.mkdir(exist_ok=True)
            stat_result = cursor.lstat()
        except OSError as error:
            raise SupervisorV12Error("Supervisor output parent is unavailable") from error
        if _is_link_or_reparse(stat_result) or not stat.S_ISDIR(stat_result.st_mode):
            raise SupervisorV12Error("Supervisor output parent is unsafe")
    return cursor / relative_path.name


def _fsync_parent_directory(parent: Path) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
        kernel32.FlushFileBuffers.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateFileW(
            str(parent),
            0x80000000 | 0x40000000,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,
            0x02000000,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, 0, invalid):
            raise SupervisorV12Error(
                f"Supervisor receipt parent cannot be opened for durable sync: Win32 {ctypes.get_last_error()}"
            )
        try:
            if not kernel32.FlushFileBuffers(handle):
                raise SupervisorV12Error(
                    f"Supervisor receipt parent durable sync failed: Win32 {ctypes.get_last_error()}"
                )
        finally:
            kernel32.CloseHandle(handle)
        return
    try:
        descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as error:
        raise SupervisorV12Error("Supervisor receipt parent cannot be opened for durable sync") from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise SupervisorV12Error("Supervisor receipt parent durable sync failed") from error
    finally:
        os.close(descriptor)


def _write_new_json(root: Path, path: Path, payload: Mapping[str, Any]) -> str:
    output_path = _safe_output_path(root, path)
    content = canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(output_path, flags, 0o600)
    except FileExistsError as error:
        raise SupervisorV12Error(f"Supervisor output already exists: {output_path}") from error
    except OSError as error:
        raise SupervisorV12Error("Supervisor receipt cannot be created") from error
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_parent_directory(output_path.parent)
    return hashlib.sha256(content).hexdigest()


def _require_binding(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise SupervisorV12Error(f"Supervisor static binding {name} is invalid")
    path = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path, str) or not path or not isinstance(digest, str) or len(digest) != 64:
        raise SupervisorV12Error(f"Supervisor static binding {name} is invalid")
    return {"path": path, "sha256": digest}


def _validate_config(config: SupervisorConfigV12) -> tuple[Path, dict[str, dict[str, str]]]:
    if config.mode not in {"preflight", "execute"}:
        raise SupervisorV12Error("Supervisor mode must be preflight or execute")
    if not config.command or any(not isinstance(item, str) or not item for item in config.command):
        raise SupervisorV12Error("Supervisor command is invalid")
    if not isinstance(config.timeout_seconds, int) or not 0 < config.timeout_seconds <= MAXIMUM_TIMEOUT_SECONDS:
        raise SupervisorV12Error("Supervisor timeout is invalid")
    if not isinstance(config.memory_limit_bytes, int) or config.memory_limit_bytes <= 0:
        raise SupervisorV12Error("Supervisor memory limit is invalid")
    try:
        root = config.project_root.resolve(strict=True)
        executable = Path(config.command[0]).resolve(strict=True)
        executable.relative_to(root)
    except (OSError, ValueError) as error:
        raise SupervisorV12Error("Supervisor executable is unavailable or outside the project") from error
    if Path.cwd().resolve(strict=True) != root:
        raise SupervisorV12Error("Supervisor must run from the canonical project root")
    bindings = {name: _require_binding(value, name=name) for name, value in config.static_bindings.items()}
    if not bindings:
        raise SupervisorV12Error("Supervisor requires static source bindings")
    controller_binding = bindings.get("controller")
    if controller_binding is not None:
        if len(config.command) != 4 or config.command[1] != "-I" or config.command[3] not in {"--preflight", "--execute"}:
            raise SupervisorV12Error("Supervisor controller command shape is invalid")
        controller_path = str(controller_binding["path"])
        if config.command[2] != controller_path or Path(config.command[2]).is_absolute() or ".." in Path(config.command[2]).parts:
            raise SupervisorV12Error("Supervisor controller command must use the sealed relative path")
        resolved_controller = safe_project_path(root, controller_path, require_exists=True, require_regular_file=True)
        if sha256_file(resolved_controller) != controller_binding["sha256"]:
            raise SupervisorV12Error("Supervisor controller binding drifted before child creation")
    output_paths = (config.launch_receipt, config.exit_receipt, config.failure_receipt)
    if len(set(output_paths)) != len(output_paths):
        raise SupervisorV12Error("Supervisor output paths overlap")
    for output_path in output_paths:
        candidate = _safe_output_path(root, output_path)
        if candidate.exists() or candidate.is_symlink():
            raise SupervisorV12Error("Supervisor output already exists or is unsafe")
    return root, bindings


def _launch_payload(
    config: SupervisorConfigV12,
    *,
    launch_id: str,
    bindings: Mapping[str, Mapping[str, str]],
    assignment_audit: Mapping[str, Any],
    supervisor_identity: Mapping[str, Any],
) -> dict[str, Any]:
    expected_flags = 0x00000100 | 0x00002000
    required = {
        "creation_mode": "create_process_suspended_assign_verify_resume",
        "assignment_api": "AssignProcessToJobObject",
        "membership_api": "IsProcessInJob",
        "job_limit_flags": expected_flags,
        "kill_on_job_close": True,
        "memory_limit_bytes": config.memory_limit_bytes,
        "assigned_before_resume": True,
        "process_resumed": False,
    }
    if any(assignment_audit.get(name) != expected for name, expected in required.items()):
        raise SupervisorV12Error("Pre-resume Job assignment audit drifted")
    if len(launch_id) != 64 or any(character not in "0123456789abcdef" for character in launch_id):
        raise SupervisorV12Error("Supervisor launch id is invalid")
    pid = assignment_audit.get("process_pid")
    creation_time = assignment_audit.get("process_creation_time_filetime")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(creation_time, bool)
        or not isinstance(creation_time, int)
        or creation_time <= 0
    ):
        raise SupervisorV12Error("Pre-resume Job assignment identity is invalid")
    if (
        supervisor_identity.get("pid") != os.getpid()
        or not isinstance(supervisor_identity.get("creation_time_filetime"), int)
        or int(supervisor_identity["creation_time_filetime"]) <= 0
    ):
        raise SupervisorV12Error("Supervisor cannot prove its process identity")
    return {
        "schema": LAUNCH_RECEIPT_SCHEMA,
        "loop_id": "loop167_ember_v3_novel_delta",
        "status": "assigned_and_verified_before_child_resume",
        "mode": config.mode,
        "launch_id": launch_id,
        "created_at_utc": _utc_now(),
        "command": list(config.command),
        "command_sha256": _command_sha256(config.command),
        "supervisor_identity": dict(supervisor_identity),
        "supervisor_executable": str(Path(sys.executable).resolve(strict=True)),
        "static_bindings": {name: dict(binding) for name, binding in bindings.items()},
        "pre_resume_assignment": dict(assignment_audit),
        "raw_open_attempts": 0,
    }


def _failure_payload(
    config: SupervisorConfigV12,
    *,
    bindings: Mapping[str, Mapping[str, str]],
    error: BaseException,
    stage: str,
) -> dict[str, Any]:
    if stage not in {"pre_resume", "post_resume"}:
        raise SupervisorV12Error("Supervisor failure stage is invalid")
    code = error.win32_error_code if isinstance(error, WindowsJobV12Error) else None
    operation = error.operation if isinstance(error, WindowsJobV12Error) else type(error).__name__
    return {
        "schema": FAILURE_RECEIPT_SCHEMA,
        "loop_id": "loop167_ember_v3_novel_delta",
        "status": (
            "failed_before_child_resume_no_lease_or_raw_access"
            if stage == "pre_resume"
            else "failed_after_child_resume_raw_access_not_attested_by_supervisor"
        ),
        "mode": config.mode,
        "stage": stage,
        "created_at_utc": _utc_now(),
        "operation": operation,
        "win32_error_code": code,
        "detail": str(error)[:512],
        "static_bindings": {name: dict(binding) for name, binding in bindings.items()},
        "supervisor_raw_open_attempts": 0,
        "child_raw_access": "not_started" if stage == "pre_resume" else "not_attested_by_supervisor",
    }


def run_supervised_v12(
    config: SupervisorConfigV12,
    *,
    environment: Mapping[str, str] | None = None,
    job_factory: Callable[..., WindowsJobV12] = WindowsJobV12.create,
    supervisor_identity_provider: Callable[[], Mapping[str, Any]] = _current_process_identity,
) -> SupervisedRunResultV12:
    """Hold a kill-on-close Job until the exact controller child exits."""

    root, bindings = _validate_config(config)
    launch_id = secrets.token_hex(32)
    supervisor_identity = dict(supervisor_identity_provider())
    child_environment = dict(os.environ if environment is None else environment)
    child_environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "AXON_LOOP167_V12_LAUNCH_ID": launch_id,
            "AXON_LOOP167_V12_LAUNCH_RECEIPT": str(config.launch_receipt),
            "AXON_LOOP167_V12_SUPERVISOR_PID": str(supervisor_identity.get("pid", "")),
        }
    )
    job: WindowsJobV12 | None = None
    child: Any | None = None
    launch_sha256: str | None = None
    try:
        job = job_factory(memory_limit_bytes=config.memory_limit_bytes, kill_on_close=True)
        child = job.spawn_suspended_assigned(
            config.command,
            cwd=root,
            environment=child_environment,
        )
        assignment_audit = job.assignment_audit(child)
        launch_payload = _launch_payload(
            config,
            launch_id=launch_id,
            bindings=bindings,
            assignment_audit=assignment_audit,
            supervisor_identity=supervisor_identity,
        )
        launch_sha256 = _write_new_json(root, config.launch_receipt, launch_payload)
        child.resume()
        try:
            returncode = child.wait(config.timeout_seconds)
        except subprocess.TimeoutExpired as error:
            active_after = job.terminate(exit_code=1)
            raise SupervisorV12Error(
                f"Controller exceeded {config.timeout_seconds} seconds; active_after={active_after}"
            ) from error
        active_after = job.active_processes()
        if active_after != 0:
            job.terminate(exit_code=1)
            raise SupervisorV12Error("Controller exited while contained descendants remained")
        exit_payload = {
            "schema": EXIT_RECEIPT_SCHEMA,
            "loop_id": "loop167_ember_v3_novel_delta",
            "status": (
                "controller_zero_exit_with_contained_tree_empty"
                if returncode == 0
                else "controller_nonzero_exit_with_contained_tree_empty"
            ),
            "mode": config.mode,
            "created_at_utc": _utc_now(),
            "launch_receipt": {
                "path": str(config.launch_receipt.relative_to(root)).replace("\\", "/"),
                "sha256": launch_sha256,
            },
            "controller_returncode": returncode,
            "active_processes_after": active_after,
            "supervisor_raw_open_attempts": 0,
            "child_raw_access": "not_attested_by_supervisor",
        }
        exit_sha256 = _write_new_json(root, config.exit_receipt, exit_payload)
        if returncode != 0:
            raise SupervisorV12Error(f"Contained controller exited nonzero: {returncode}")
        return SupervisedRunResultV12(
            returncode=returncode,
            launch_receipt_sha256=launch_sha256,
            exit_receipt_sha256=exit_sha256,
            job_audit=dict(assignment_audit),
        )
    except Exception as error:
        if job is not None:
            try:
                if job.active_processes() > 0:
                    job.terminate(exit_code=1)
            except Exception:
                pass
        if child is not None and getattr(child, "resumed", False):
            try:
                child.wait(30)
            except Exception:
                pass
        elif child is not None:
            try:
                child.terminate()
            except Exception:
                pass
        try:
            _write_new_json(
                root,
                config.failure_receipt,
                _failure_payload(
                    config,
                    bindings=bindings,
                    error=error,
                    stage="post_resume" if launch_sha256 is not None else "pre_resume",
                ),
            )
        except Exception:
            pass
        if isinstance(error, SupervisorV12Error):
            raise
        raise SupervisorV12Error("Supervisor failed before a contained controller completion") from error
    finally:
        if child is not None:
            try:
                child.close()
            except Exception:
                pass
        if job is not None and not job.closed:
            try:
                job.close()
            except Exception:
                pass


def validate_launch_receipt_v12(
    root: Path,
    path: Path,
    *,
    mode: str,
    expected_bindings: Mapping[str, Mapping[str, str]],
    expected_launch_id: str | None = None,
    expected_pid: int | None = None,
    expected_creation_time_filetime: int | None = None,
    expected_supervisor_pid: int | None = None,
) -> ValidatedLaunchReceiptV12:
    """Validate the pre-resume receipt before a child can reach a v12 lease."""

    try:
        relative_path = safe_project_relative_path(root, path, require_exists=True, require_regular_file=True)
        receipt_path = Path(root) / relative_path
        raw = receipt_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, PhaseBContractError) as error:
        raise SupervisorV12Error("Pre-resume launch receipt is unavailable") from error
    if not isinstance(payload, dict):
        raise SupervisorV12Error("Pre-resume launch receipt is not an object")
    if (
        payload.get("schema") != LAUNCH_RECEIPT_SCHEMA
        or payload.get("status") != "assigned_and_verified_before_child_resume"
        or payload.get("mode") != mode
        or payload.get("raw_open_attempts") != 0
    ):
        raise SupervisorV12Error("Pre-resume launch receipt identity drifted")
    if canonical_json_bytes(payload) != raw:
        raise SupervisorV12Error("Pre-resume launch receipt is not canonical")
    launch_id = payload.get("launch_id")
    if not isinstance(launch_id, str) or len(launch_id) != 64 or any(
        character not in "0123456789abcdef" for character in launch_id
    ):
        raise SupervisorV12Error("Pre-resume launch receipt launch id is invalid")
    if expected_launch_id is not None and launch_id != expected_launch_id:
        raise SupervisorV12Error("Pre-resume launch receipt launch id drifted")
    command = payload.get("command")
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise SupervisorV12Error("Pre-resume launch receipt command is invalid")
    if payload.get("command_sha256") != _command_sha256(command):
        raise SupervisorV12Error("Pre-resume launch receipt command hash drifted")
    supervisor_identity = payload.get("supervisor_identity")
    if (
        not isinstance(supervisor_identity, dict)
        or not isinstance(supervisor_identity.get("pid"), int)
        or supervisor_identity["pid"] <= 0
        or not isinstance(supervisor_identity.get("creation_time_filetime"), int)
        or supervisor_identity["creation_time_filetime"] <= 0
        or not isinstance(payload.get("supervisor_executable"), str)
        or not Path(payload["supervisor_executable"]).is_absolute()
    ):
        raise SupervisorV12Error("Pre-resume launch receipt supervisor identity drifted")
    if expected_supervisor_pid is not None and supervisor_identity["pid"] != expected_supervisor_pid:
        raise SupervisorV12Error("Pre-resume launch receipt supervisor PID drifted")
    bindings = payload.get("static_bindings")
    if not isinstance(bindings, dict) or bindings != {
        name: dict(binding) for name, binding in expected_bindings.items()
    }:
        raise SupervisorV12Error("Pre-resume launch receipt static bindings drifted")
    audit = payload.get("pre_resume_assignment")
    if not isinstance(audit, dict) or audit.get("assigned_before_resume") is not True:
        raise SupervisorV12Error("Pre-resume launch receipt lacks assignment proof")
    expected_assignment = {
        "creation_mode": "create_process_suspended_assign_verify_resume",
        "assignment_api": "AssignProcessToJobObject",
        "membership_api": "IsProcessInJob",
        "job_limit_flags": 0x2100,
        "kill_on_job_close": True,
        "assigned_before_resume": True,
        "process_resumed": False,
    }
    if any(audit.get(name) != value for name, value in expected_assignment.items()):
        raise SupervisorV12Error("Pre-resume launch receipt assignment state drifted")
    receipt_pid = audit.get("process_pid")
    receipt_creation = audit.get("process_creation_time_filetime")
    if (
        isinstance(receipt_pid, bool)
        or not isinstance(receipt_pid, int)
        or receipt_pid <= 0
        or isinstance(receipt_creation, bool)
        or not isinstance(receipt_creation, int)
        or receipt_creation <= 0
    ):
        raise SupervisorV12Error("Pre-resume launch receipt child identity is invalid")
    if expected_pid is not None and receipt_pid != expected_pid:
        raise SupervisorV12Error("Pre-resume launch receipt child PID drifted")
    if expected_creation_time_filetime is not None and receipt_creation != expected_creation_time_filetime:
        raise SupervisorV12Error("Pre-resume launch receipt child creation time drifted")
    canonical_sha256 = hashlib.sha256(raw).hexdigest()
    if sha256_file(receipt_path) != canonical_sha256:
        raise SupervisorV12Error("Pre-resume launch receipt changed during validation")
    return ValidatedLaunchReceiptV12(
        receipt_path=receipt_path,
        payload=MappingProxyType(payload),
        canonical_sha256=canonical_sha256,
    )


__all__ = [
    "EXIT_RECEIPT_SCHEMA",
    "FAILURE_RECEIPT_SCHEMA",
    "LAUNCH_RECEIPT_SCHEMA",
    "SupervisorConfigV12",
    "SupervisorV12Error",
    "SupervisedRunResultV12",
    "ValidatedLaunchReceiptV12",
    "run_supervised_v12",
    "validate_launch_receipt_v12",
]
