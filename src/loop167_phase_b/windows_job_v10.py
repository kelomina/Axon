"""Typed Windows Job Object boundary for Loop167 Phase-B v10.

The v5 boundary used untyped ``ctypes.windll`` calls.  On 64-bit Windows that
can pass a truncated HANDLE and loses the actual last-error value.  This module
owns the ABI explicitly and keeps the probe job non-killing so a guard process
can persist its result after proving assignment.
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
CREATE_SUSPENDED = 0x00000004
CREATE_UNICODE_ENVIRONMENT = 0x00000400
WAIT_OBJECT_0 = 0x00000000
WAIT_TIMEOUT = 0x00000102
STILL_ACTIVE = 259


class WindowsJobV10Error(RuntimeError):
    """A Windows Job failure with the API operation and captured Win32 code."""

    def __init__(self, operation: str, win32_error_code: int | None = None) -> None:
        self.operation = operation
        self.win32_error_code = win32_error_code
        detail = "unavailable" if win32_error_code is None else str(win32_error_code)
        super().__init__(f"{operation} failed with Win32 error {detail}")


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _BasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class _FileTime(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


class _StartupInfoW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


def _require_memory_limit(memory_limit_bytes: int) -> int:
    if isinstance(memory_limit_bytes, bool) or not isinstance(memory_limit_bytes, int):
        raise ValueError("memory_limit_bytes must be a positive integer")
    if memory_limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be a positive integer")
    return memory_limit_bytes


def _handle_is_valid(handle: object) -> bool:
    value = handle if isinstance(handle, int) else getattr(handle, "value", None)
    return isinstance(value, int) and value not in (0, INVALID_HANDLE_VALUE)


def _current_process_handle_is_valid(handle: object) -> bool:
    """``GetCurrentProcess`` returns the valid pseudo-handle ``(HANDLE)-1``."""

    value = handle if isinstance(handle, int) else getattr(handle, "value", None)
    return isinstance(value, int) and value != 0


class _Kernel32JobApiV10:
    """Small, fully typed subset of kernel32 used by the v10 Job boundary."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise WindowsJobV10Error("Windows Job Objects are unavailable")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.IsProcessInJob.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        ]
        kernel32.IsProcessInJob.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CreateProcessW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.BOOL,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.LPCWSTR,
            ctypes.POINTER(_StartupInfoW),
            ctypes.POINTER(_ProcessInformation),
        ]
        kernel32.CreateProcessW.restype = wintypes.BOOL
        kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
        kernel32.ResumeThread.restype = wintypes.DWORD
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        self.kernel32 = kernel32

    @staticmethod
    def _failure(operation: str) -> WindowsJobV10Error:
        return WindowsJobV10Error(operation, int(ctypes.get_last_error()))

    def create_job(self) -> Any:
        ctypes.set_last_error(0)
        handle = self.kernel32.CreateJobObjectW(None, None)
        if not _handle_is_valid(handle):
            raise self._failure("CreateJobObjectW")
        return handle

    def configure_job(
        self,
        handle: Any,
        *,
        memory_limit_bytes: int,
        kill_on_close: bool,
    ) -> None:
        information = _ExtendedLimitInformation()
        flags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
        if kill_on_close:
            flags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        information.BasicLimitInformation.LimitFlags = flags
        information.ProcessMemoryLimit = memory_limit_bytes
        ctypes.set_last_error(0)
        if not self.kernel32.SetInformationJobObject(
            handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            raise self._failure("SetInformationJobObject")

    def limit_flags(self, handle: Any) -> int:
        information = _ExtendedLimitInformation()
        ctypes.set_last_error(0)
        if not self.kernel32.QueryInformationJobObject(
            handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
            None,
        ):
            raise self._failure("QueryInformationJobObject")
        return int(information.BasicLimitInformation.LimitFlags)

    def current_process(self) -> Any:
        handle = self.kernel32.GetCurrentProcess()
        if not _current_process_handle_is_valid(handle):
            raise self._failure("GetCurrentProcess")
        return handle

    def assign_process(self, job_handle: Any, process_handle: Any) -> None:
        ctypes.set_last_error(0)
        if not self.kernel32.AssignProcessToJobObject(job_handle, process_handle):
            raise self._failure("AssignProcessToJobObject")

    def is_process_in_job(self, process_handle: Any, job_handle: Any) -> bool:
        in_job = wintypes.BOOL()
        ctypes.set_last_error(0)
        if not self.kernel32.IsProcessInJob(process_handle, job_handle, ctypes.byref(in_job)):
            raise self._failure("IsProcessInJob")
        return bool(in_job.value)

    def close_handle(self, handle: Any) -> None:
        ctypes.set_last_error(0)
        if not self.kernel32.CloseHandle(handle):
            raise self._failure("CloseHandle")

    @staticmethod
    def _environment_block(environment: Mapping[str, str]) -> Any:
        entries: list[str] = []
        for key in sorted(environment, key=str.casefold):
            value = environment[key]
            if (
                not isinstance(key, str)
                or not key
                or "=" in key
                or "\x00" in key
                or not isinstance(value, str)
                or "\x00" in value
            ):
                raise WindowsJobV10Error("CreateProcessW environment is invalid")
            entries.append(f"{key}={value}")
        return ctypes.create_unicode_buffer("\x00".join(entries) + "\x00\x00")

    def create_suspended_process(
        self,
        command: Sequence[str],
        *,
        cwd: Path | str,
        environment: Mapping[str, str],
    ) -> "SuspendedProcessV10":
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise WindowsJobV10Error("CreateProcessW command is invalid")
        try:
            executable = Path(command[0]).resolve(strict=True)
            working_directory = Path(cwd).resolve(strict=True)
        except OSError as error:
            raise WindowsJobV10Error("CreateProcessW executable or cwd is unavailable") from error
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(list(command)))
        environment_block = self._environment_block(environment)
        startup = _StartupInfoW()
        startup.cb = ctypes.sizeof(startup)
        process_information = _ProcessInformation()
        ctypes.set_last_error(0)
        if not self.kernel32.CreateProcessW(
            str(executable),
            command_line,
            None,
            None,
            False,
            CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT,
            ctypes.cast(environment_block, ctypes.c_void_p),
            str(working_directory),
            ctypes.byref(startup),
            ctypes.byref(process_information),
        ):
            raise self._failure("CreateProcessW")
        if not _handle_is_valid(process_information.hProcess) or not _handle_is_valid(
            process_information.hThread
        ):
            if _handle_is_valid(process_information.hThread):
                self.close_handle(process_information.hThread)
            if _handle_is_valid(process_information.hProcess):
                self.close_handle(process_information.hProcess)
            raise WindowsJobV10Error("CreateProcessW returned invalid handles")
        return SuspendedProcessV10(
            api=self,
            process_handle=process_information.hProcess,
            thread_handle=process_information.hThread,
            pid=int(process_information.dwProcessId),
            creation_time_filetime=self.process_creation_time_filetime(process_information.hProcess),
        )

    def process_creation_time_filetime(self, process_handle: Any) -> int:
        creation = _FileTime()
        exit_time = _FileTime()
        kernel_time = _FileTime()
        user_time = _FileTime()
        ctypes.set_last_error(0)
        if not self.kernel32.GetProcessTimes(
            process_handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            raise self._failure("GetProcessTimes")
        return (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)

    def resume_thread(self, thread_handle: Any) -> None:
        ctypes.set_last_error(0)
        if self.kernel32.ResumeThread(thread_handle) == 0xFFFFFFFF:
            raise self._failure("ResumeThread")

    def wait_process(self, process_handle: Any, timeout_seconds: float) -> int:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise WindowsJobV10Error("WaitForSingleObject timeout is invalid")
        timeout_milliseconds = min(0xFFFFFFFE, max(1, math.ceil(timeout_seconds * 1000)))
        ctypes.set_last_error(0)
        result = self.kernel32.WaitForSingleObject(process_handle, timeout_milliseconds)
        if result == WAIT_TIMEOUT:
            raise subprocess.TimeoutExpired("suspended Windows child", timeout_seconds)
        if result != WAIT_OBJECT_0:
            raise self._failure("WaitForSingleObject")
        exit_code = wintypes.DWORD()
        ctypes.set_last_error(0)
        if not self.kernel32.GetExitCodeProcess(process_handle, ctypes.byref(exit_code)):
            raise self._failure("GetExitCodeProcess")
        if int(exit_code.value) == STILL_ACTIVE:
            raise WindowsJobV10Error("GetExitCodeProcess reported an active process after wait")
        return int(exit_code.value)

    def terminate_process(self, process_handle: Any, exit_code: int = 1) -> None:
        ctypes.set_last_error(0)
        if not self.kernel32.TerminateProcess(process_handle, wintypes.UINT(exit_code)):
            raise self._failure("TerminateProcess")

    def terminate_job(self, job_handle: Any, exit_code: int = 1) -> None:
        ctypes.set_last_error(0)
        if not self.kernel32.TerminateJobObject(job_handle, wintypes.UINT(exit_code)):
            raise self._failure("TerminateJobObject")

    def active_processes(self, job_handle: Any) -> int:
        accounting = _BasicAccountingInformation()
        ctypes.set_last_error(0)
        if not self.kernel32.QueryInformationJobObject(
            job_handle,
            1,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            None,
        ):
            raise self._failure("QueryInformationJobObject(accounting)")
        return int(accounting.ActiveProcesses)


@dataclass
class SuspendedProcessV10:
    """The primary thread remains suspended until the supervisor persists proof."""

    api: Any
    process_handle: Any
    thread_handle: Any
    pid: int
    creation_time_filetime: int
    resumed: bool = False
    returncode: int | None = None
    _closed: bool = False

    def resume(self) -> None:
        if self.resumed:
            raise WindowsJobV10Error("Suspended process was already resumed")
        self.api.resume_thread(self.thread_handle)
        self.api.close_handle(self.thread_handle)
        self.thread_handle = None
        self.resumed = True

    def wait(self, timeout_seconds: float) -> int:
        if not self.resumed:
            raise WindowsJobV10Error("Cannot wait for an unresumed child")
        if self.returncode is None:
            self.returncode = self.api.wait_process(self.process_handle, timeout_seconds)
        return self.returncode

    def terminate(self, exit_code: int = 1) -> None:
        if self.returncode is None:
            self.api.terminate_process(self.process_handle, exit_code)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.thread_handle:
            self.api.close_handle(self.thread_handle)
            self.thread_handle = None
        self.api.close_handle(self.process_handle)


@dataclass
class WindowsJobV10:
    """A typed, memory-limited Job whose handle remains live until explicit close."""

    handle: Any
    memory_limit_bytes: int
    kill_on_close: bool
    _api: Any
    _closed: bool = False

    @classmethod
    def create(
        cls,
        *,
        memory_limit_bytes: int,
        kill_on_close: bool,
        api: Any | None = None,
    ) -> "WindowsJobV10":
        memory_limit = _require_memory_limit(memory_limit_bytes)
        job_api = _Kernel32JobApiV10() if api is None else api
        handle = job_api.create_job()
        job = cls(
            handle=handle,
            memory_limit_bytes=memory_limit,
            kill_on_close=kill_on_close,
            _api=job_api,
        )
        try:
            job_api.configure_job(
                handle,
                memory_limit_bytes=memory_limit,
                kill_on_close=kill_on_close,
            )
            expected_flags = job.expected_limit_flags
            if job_api.limit_flags(handle) != expected_flags:
                raise WindowsJobV10Error("QueryInformationJobObject returned unexpected limit flags")
            return job
        except Exception:
            try:
                job.close()
            except Exception:
                pass
            raise

    @property
    def expected_limit_flags(self) -> int:
        flags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
        if self.kill_on_close:
            flags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        return flags

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise WindowsJobV10Error("Windows Job Object is closed")

    def assign_current_process(self) -> dict[str, int | bool | str]:
        """Assign and verify the exact active process before the v10 lease boundary."""

        self._require_open()
        process_handle = self._api.current_process()
        self._api.assign_process(self.handle, process_handle)
        if not self._api.is_process_in_job(process_handle, self.handle):
            raise WindowsJobV10Error("IsProcessInJob returned false after assignment")
        return {
            "assignment_api": "AssignProcessToJobObject",
            "membership_api": "IsProcessInJob",
            "job_limit_flags": self.expected_limit_flags,
            "kill_on_job_close": self.kill_on_close,
            "memory_limit_bytes": self.memory_limit_bytes,
            "current_process_assigned": True,
        }

    def spawn_suspended_assigned(
        self,
        command: Sequence[str],
        *,
        cwd: Path | str,
        environment: Mapping[str, str],
    ) -> SuspendedProcessV10:
        """Create a child, assign it, and verify membership before it can run."""

        self._require_open()
        process = self._api.create_suspended_process(command, cwd=cwd, environment=environment)
        try:
            self._api.assign_process(self.handle, process.process_handle)
            if not self._api.is_process_in_job(process.process_handle, self.handle):
                raise WindowsJobV10Error("IsProcessInJob returned false after child assignment")
            return process
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.close()
            except Exception:
                pass
            raise

    def assignment_audit(self, process: SuspendedProcessV10) -> dict[str, int | bool | str]:
        """Return only evidence that was true before the child was resumed."""

        self._require_open()
        if process.resumed:
            raise WindowsJobV10Error("Child was resumed before pre-resume audit")
        return {
            "creation_mode": "create_process_suspended_assign_verify_resume",
            "assignment_api": "AssignProcessToJobObject",
            "membership_api": "IsProcessInJob",
            "job_limit_flags": self.expected_limit_flags,
            "kill_on_job_close": self.kill_on_close,
            "memory_limit_bytes": self.memory_limit_bytes,
            "process_pid": process.pid,
            "process_creation_time_filetime": process.creation_time_filetime,
            "assigned_before_resume": True,
            "process_resumed": False,
        }

    def active_processes(self) -> int:
        self._require_open()
        active = self._api.active_processes(self.handle)
        if isinstance(active, bool) or not isinstance(active, int) or active < 0:
            raise WindowsJobV10Error("QueryInformationJobObject returned invalid active count")
        return active

    def terminate(self, *, exit_code: int = 1) -> int:
        """Stop the complete contained tree and return its final active count."""

        self._require_open()
        self._api.terminate_job(self.handle, exit_code)
        deadline = time.monotonic() + 30.0
        while True:
            active = self.active_processes()
            if active == 0:
                return active
            if time.monotonic() >= deadline:
                raise WindowsJobV10Error("TerminateJobObject left active child processes")
            time.sleep(0.05)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._api.close_handle(self.handle)

    def __enter__(self) -> "WindowsJobV10":
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class WindowsJobAssignmentProbeV10:
    """Raw-free evidence from a disposable non-kill Job assignment."""

    ready: bool
    operation: str | None
    win32_error_code: int | None
    detail: str | None
    assignment: dict[str, int | bool | str] | None


def probe_windows_job_assignment_v10(
    *,
    memory_limit_bytes: int,
    job_factory: Callable[..., WindowsJobV10] = WindowsJobV10.create,
) -> WindowsJobAssignmentProbeV10:
    """Assign the guard process to a disposable non-kill Job and verify it.

    The caller must be a short-lived guard-builder process.  Windows keeps a
    process associated with a Job until it exits, so a production controller is
    never used for this probe.
    """

    try:
        with job_factory(memory_limit_bytes=memory_limit_bytes, kill_on_close=False) as job:
            assignment = job.assign_current_process()
        return WindowsJobAssignmentProbeV10(
            ready=True,
            operation=None,
            win32_error_code=None,
            detail=None,
            assignment=assignment,
        )
    except WindowsJobV10Error as error:
        return WindowsJobAssignmentProbeV10(
            ready=False,
            operation=error.operation,
            win32_error_code=error.win32_error_code,
            detail=str(error),
            assignment=None,
        )


__all__ = [
    "INVALID_HANDLE_VALUE",
    "JOB_OBJECT_EXTENDED_LIMIT_INFORMATION",
    "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE",
    "JOB_OBJECT_LIMIT_PROCESS_MEMORY",
    "SuspendedProcessV10",
    "WindowsJobAssignmentProbeV10",
    "WindowsJobV10",
    "WindowsJobV10Error",
    "probe_windows_job_assignment_v10",
]
