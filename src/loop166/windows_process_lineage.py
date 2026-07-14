from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path
from typing import Any, Mapping


class ProcessLineageError(RuntimeError):
    """Raised when a worker cannot prove its bounded parent lineage."""


def _normalized_existing_path(path: str | Path) -> str:
    try:
        candidate = Path(path)
        if not str(path) or not candidate.is_absolute():
            raise ProcessLineageError(f"Process image path is not absolute: {path}")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_file():
            raise ProcessLineageError(f"Process image path is not a file: {path}")
        return os.path.normcase(str(resolved))
    except ProcessLineageError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ProcessLineageError(f"Process image path is unavailable: {path}") from exc


def _validate_observed_lineage(
    *,
    expected_parent_pid: int,
    current_pid: int,
    direct_parent_pid: int,
    process_parents: Mapping[int, int],
    process_images: Mapping[int, str],
    launcher_executable: str | Path,
    base_executable: str | Path,
    is_windows: bool,
) -> dict[str, Any]:
    if expected_parent_pid <= 0 or current_pid <= 0 or direct_parent_pid <= 0:
        raise ProcessLineageError("Process lineage contains a non-positive PID")
    if current_pid == expected_parent_pid:
        raise ProcessLineageError("Worker PID unexpectedly equals its bound parent PID")
    if direct_parent_pid == expected_parent_pid:
        audit = {
            "mode": "direct_parent",
            "expected_parent_pid": expected_parent_pid,
            "current_pid": current_pid,
            "direct_parent_pid": direct_parent_pid,
            "redirector_pid": 0,
        }
        if not is_windows:
            return audit
        if process_parents.get(current_pid) != expected_parent_pid:
            raise ProcessLineageError(
                "Current process parent disagrees with the process snapshot"
            )
        normalized_base = _normalized_existing_path(base_executable)
        for pid in (current_pid, expected_parent_pid):
            observed_image = process_images.get(pid)
            if observed_image is None:
                raise ProcessLineageError(f"Process image is unavailable for PID {pid}")
            if _normalized_existing_path(observed_image) != normalized_base:
                raise ProcessLineageError(f"Process image drifted for PID {pid}")
        return {
            **audit,
            "current_image": str(Path(process_images[current_pid])),
            "expected_parent_image": str(Path(process_images[expected_parent_pid])),
        }
    if not is_windows:
        raise ProcessLineageError("Worker was not spawned by its bound direct parent")

    # Windows venv 启动器会常驻为一层 redirector；这里只允许这一层且逐个核对镜像路径。
    if process_parents.get(current_pid) != direct_parent_pid:
        raise ProcessLineageError("Current process parent disagrees with the process snapshot")
    if process_parents.get(direct_parent_pid) != expected_parent_pid:
        raise ProcessLineageError("Worker lineage is not exactly one venv redirector deep")
    normalized_launcher = _normalized_existing_path(launcher_executable)
    normalized_base = _normalized_existing_path(base_executable)
    if normalized_launcher == normalized_base:
        raise ProcessLineageError("Redirector fallback requires an active virtual environment")
    expected_images = {
        current_pid: normalized_base,
        direct_parent_pid: normalized_launcher,
        expected_parent_pid: normalized_base,
    }
    for pid, expected_image in expected_images.items():
        observed_image = process_images.get(pid)
        if observed_image is None:
            raise ProcessLineageError(f"Process image is unavailable for PID {pid}")
        if _normalized_existing_path(observed_image) != expected_image:
            raise ProcessLineageError(f"Process image drifted for PID {pid}")
    return {
        "mode": "windows_venv_redirector",
        "expected_parent_pid": expected_parent_pid,
        "current_pid": current_pid,
        "direct_parent_pid": direct_parent_pid,
        "redirector_pid": direct_parent_pid,
        "redirector_image": str(Path(process_images[direct_parent_pid])),
        "current_image": str(Path(process_images[current_pid])),
        "expected_parent_image": str(Path(process_images[expected_parent_pid])),
    }


def _windows_process_snapshot() -> dict[int, int]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    dword = ctypes.c_ulong

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", dword),
            ("cntUsage", dword),
            ("th32ProcessID", dword),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", dword),
            ("cntThreads", dword),
            ("th32ParentProcessID", dword),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", dword),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    create_snapshot = kernel32.CreateToolhelp32Snapshot
    create_snapshot.argtypes = [dword, dword]
    create_snapshot.restype = ctypes.c_void_p
    process_first = kernel32.Process32FirstW
    process_first.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry32W)]
    process_first.restype = ctypes.c_int
    process_next = kernel32.Process32NextW
    process_next.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry32W)]
    process_next.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int

    snapshot = create_snapshot(0x00000002, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot in {None, invalid_handle}:
        error = ctypes.get_last_error()
        raise ProcessLineageError(
            f"Unable to create a Windows process snapshot: error {error}"
        )
    parents: dict[int, int] = {}
    primary_error: Exception | None = None
    try:
        entry = ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        if not process_first(snapshot, ctypes.byref(entry)):
            error = ctypes.get_last_error()
            raise ProcessLineageError(
                f"Unable to read the first Windows process entry: error {error}"
            )
        while True:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            entry.dwSize = ctypes.sizeof(entry)
            if not process_next(snapshot, ctypes.byref(entry)):
                error = ctypes.get_last_error()
                if error != 18:
                    raise ProcessLineageError(
                        f"Windows process snapshot iteration failed with error {error}"
                    )
                break
    except Exception as exc:
        primary_error = exc
        raise
    finally:
        if not close_handle(snapshot):
            error = ctypes.get_last_error()
            if primary_error is not None:
                primary_error.add_note(
                    f"CloseHandle failed for process snapshot with error {error}"
                )
            else:
                raise ProcessLineageError(
                    f"Unable to close Windows process snapshot: error {error}"
                )
    return parents


def _windows_process_evidence(pids: set[int]) -> tuple[dict[int, int], dict[int, str]]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    open_process.restype = ctypes.c_void_p
    query_image = kernel32.QueryFullProcessImageNameW
    query_image.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    query_image.restype = ctypes.c_int
    wait_for_single_object = kernel32.WaitForSingleObject
    wait_for_single_object.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    wait_for_single_object.restype = ctypes.c_ulong
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int

    handles: dict[int, Any] = {}
    primary_error: Exception | None = None
    try:
        for pid in sorted(pids):
            handle = open_process(0x00100000 | 0x1000, 0, pid)
            if handle in {None, 0}:
                error = ctypes.get_last_error()
                raise ProcessLineageError(
                    f"Unable to open process image for PID {pid}: error {error}"
                )
            handles[pid] = handle
        for pid, handle in handles.items():
            wait_result = wait_for_single_object(handle, 0)
            if wait_result != 0x00000102:
                raise ProcessLineageError(
                    f"Bound process is no longer active for PID {pid}: wait {wait_result}"
                )
        parents = _windows_process_snapshot()
        images: dict[int, str] = {}
        for pid, handle in handles.items():
            capacity = 32768
            size = ctypes.c_ulong(capacity)
            buffer = ctypes.create_unicode_buffer(capacity)
            if not query_image(handle, 0, buffer, ctypes.byref(size)):
                error = ctypes.get_last_error()
                raise ProcessLineageError(
                    f"Unable to query process image for PID {pid}: error {error}"
                )
            if not 0 < size.value < capacity:
                raise ProcessLineageError(
                    f"Process image length is invalid for PID {pid}: {size.value}"
                )
            images[pid] = buffer[: size.value]
        for pid, handle in handles.items():
            wait_result = wait_for_single_object(handle, 0)
            if wait_result != 0x00000102:
                raise ProcessLineageError(
                    f"Bound process exited during lineage validation for PID {pid}"
                )
        return parents, images
    except Exception as exc:
        primary_error = exc
        raise
    finally:
        close_failures = []
        for pid, handle in handles.items():
            if not close_handle(handle):
                close_failures.append((pid, ctypes.get_last_error()))
        if close_failures:
            detail = ", ".join(f"PID {pid}: error {error}" for pid, error in close_failures)
            if primary_error is not None:
                primary_error.add_note(f"CloseHandle failures: {detail}")
            else:
                raise ProcessLineageError(f"Unable to close process handles: {detail}")


def validate_spawn_lineage(
    expected_parent_pid: int,
    *,
    launcher_executable: str | Path,
    base_executable: str | Path,
) -> dict[str, Any]:
    """Accept a direct parent or exactly one verified Windows venv redirector."""
    current_pid = os.getpid()
    direct_parent_pid = os.getppid()
    is_windows = platform.system().casefold() == "windows"
    if not is_windows and direct_parent_pid != expected_parent_pid:
        raise ProcessLineageError("Worker was not spawned by its bound direct parent")
    if is_windows:
        relevant_pids = {current_pid, direct_parent_pid, expected_parent_pid}
        process_parents, process_images = _windows_process_evidence(relevant_pids)
    else:
        process_parents, process_images = {}, {}
    return _validate_observed_lineage(
        expected_parent_pid=expected_parent_pid,
        current_pid=current_pid,
        direct_parent_pid=direct_parent_pid,
        process_parents=process_parents,
        process_images=process_images,
        launcher_executable=launcher_executable,
        base_executable=base_executable,
        is_windows=is_windows,
    )


__all__ = [
    "ProcessLineageError",
    "validate_spawn_lineage",
]
