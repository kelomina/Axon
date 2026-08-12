from __future__ import annotations

import ctypes
import math
import os
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Sequence


class WindowsJobError(RuntimeError):
    """Raised when a Windows Job Object cannot prove process-tree containment."""


class WindowsJobTimeoutError(subprocess.TimeoutExpired):
    """A subprocess timeout after the assigned Job Object tree was terminated."""

    def __init__(
        self,
        command: Sequence[str],
        timeout_seconds: float,
        *,
        termination: Mapping[str, Any],
    ) -> None:
        super().__init__(tuple(command), timeout_seconds)
        self.termination = dict(termination)


class _JobBasicLimitInformation(ctypes.Structure):
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


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JobBasicAccountingInformation(ctypes.Structure):
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


class _Kernel32JobApi:
    _EXTENDED_LIMIT_CLASS = 9
    _BASIC_ACCOUNTING_CLASS = 1
    _PROCESS_MEMORY_LIMIT = 0x00000100
    _KILL_ON_JOB_CLOSE = 0x00002000
    _CREATE_SUSPENDED = 0x00000004
    _CREATE_UNICODE_ENVIRONMENT = 0x00000400
    _STARTF_USESTDHANDLES = 0x00000100
    _DUPLICATE_SAME_ACCESS = 0x00000002
    _WAIT_OBJECT_0 = 0x00000000
    _WAIT_TIMEOUT = 0x00000102
    _STILL_ACTIVE = 259
    _CREATE_NEW_PROCESS_GROUP = 0x00000200
    _CREATE_NO_WINDOW = 0x08000000
    _CREATE_BREAKAWAY_FROM_JOB = 0x01000000
    _ALLOWED_CALLER_CREATION_FLAGS = _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW

    def __init__(self) -> None:
        if os.name != "nt":
            raise WindowsJobError("Windows Job Objects are unavailable off Windows")
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
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.DuplicateHandle.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.DuplicateHandle.restype = wintypes.BOOL
        kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel32.GetStdHandle.restype = wintypes.HANDLE
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
        kernel32.IsProcessInJob.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        ]
        kernel32.IsProcessInJob.restype = wintypes.BOOL
        kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
        kernel32.ResumeThread.restype = wintypes.DWORD
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        self.kernel32 = kernel32

    @staticmethod
    def _error(operation: str) -> WindowsJobError:
        return WindowsJobError(f"{operation} failed with Windows error {ctypes.get_last_error()}")

    @staticmethod
    def _handle_value(handle: Any) -> int:
        value = handle if isinstance(handle, int) else getattr(handle, "value", None)
        if not isinstance(value, int) or value <= 0:
            raise WindowsJobError("Windows API returned an invalid handle")
        return value

    def create_job(self) -> Any:
        handle = self.kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise self._error("CreateJobObjectW")
        return handle

    def enable_kill_on_close(self, handle: Any) -> None:
        limits = _JobExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = self._KILL_ON_JOB_CLOSE
        if not self.kernel32.SetInformationJobObject(
            handle,
            self._EXTENDED_LIMIT_CLASS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            raise self._error("SetInformationJobObject")

    def set_process_memory_limit(self, handle: Any, memory_limit_bytes: int) -> None:
        if (
            not isinstance(memory_limit_bytes, int)
            or isinstance(memory_limit_bytes, bool)
            or memory_limit_bytes <= 0
        ):
            raise WindowsJobError("Windows Job process memory limit is invalid")
        limits = _JobExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = (
            self._KILL_ON_JOB_CLOSE | self._PROCESS_MEMORY_LIMIT
        )
        limits.ProcessMemoryLimit = memory_limit_bytes
        if not self.kernel32.SetInformationJobObject(
            handle,
            self._EXTENDED_LIMIT_CLASS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            raise self._error("SetInformationJobObject(PROCESS_MEMORY_LIMIT)")

    def limit_flags(self, handle: Any) -> int:
        limits = _JobExtendedLimitInformation()
        if not self.kernel32.QueryInformationJobObject(
            handle,
            self._EXTENDED_LIMIT_CLASS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
            None,
        ):
            raise self._error("QueryInformationJobObject(extended limits)")
        return int(limits.BasicLimitInformation.LimitFlags)

    def process_memory_limit(self, handle: Any) -> int:
        limits = _JobExtendedLimitInformation()
        if not self.kernel32.QueryInformationJobObject(
            handle,
            self._EXTENDED_LIMIT_CLASS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
            None,
        ):
            raise self._error("QueryInformationJobObject(PROCESS_MEMORY_LIMIT)")
        return int(limits.ProcessMemoryLimit)

    def assign_process(self, handle: Any, process_handle: int) -> None:
        if not self.kernel32.AssignProcessToJobObject(handle, wintypes.HANDLE(process_handle)):
            raise self._error("AssignProcessToJobObject")

    def active_processes(self, handle: Any) -> int:
        accounting = _JobBasicAccountingInformation()
        if not self.kernel32.QueryInformationJobObject(
            handle,
            self._BASIC_ACCOUNTING_CLASS,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            None,
        ):
            raise self._error("QueryInformationJobObject")
        return int(accounting.ActiveProcesses)

    def terminate_job(self, handle: Any, exit_code: int) -> None:
        if not self.kernel32.TerminateJobObject(handle, wintypes.UINT(exit_code)):
            raise self._error("TerminateJobObject")

    def close_handle(self, handle: Any) -> None:
        if not self.kernel32.CloseHandle(handle):
            raise self._error("CloseHandle")

    def _duplicate_inheritable_handle(self, handle: int) -> Any:
        current = self.kernel32.GetCurrentProcess()
        duplicate = wintypes.HANDLE()
        if not self.kernel32.DuplicateHandle(
            current,
            wintypes.HANDLE(handle),
            current,
            ctypes.byref(duplicate),
            0,
            True,
            self._DUPLICATE_SAME_ACCESS,
        ):
            raise self._error("DuplicateHandle")
        return duplicate

    @staticmethod
    def _stream_handle(stream: BinaryIO) -> int:
        import msvcrt

        try:
            handle = int(msvcrt.get_osfhandle(stream.fileno()))
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise WindowsJobError("Subprocess stream lacks a Windows handle") from exc
        if handle <= 0:
            raise WindowsJobError("Subprocess stream has an invalid Windows handle")
        return handle

    def _nul_handle(self, *, writable: bool) -> Any:
        access = 0x40000000 if writable else 0x80000000
        handle = self.kernel32.CreateFileW(
            "NUL",
            access,
            0x00000001 | 0x00000002,
            None,
            3,
            0x00000080,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in {None, 0, invalid}:
            raise self._error("CreateFileW(NUL)")
        return handle

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
                raise WindowsJobError("Subprocess environment contains an invalid entry")
            entries.append(f"{key}={value}")
        return ctypes.create_unicode_buffer("\x00".join(entries) + "\x00\x00")

    def create_suspended_assigned_process(
        self,
        job_handle: Any,
        command: Sequence[str],
        *,
        cwd: str | Path,
        env: Mapping[str, str],
        stdout: BinaryIO | None,
        stderr: BinaryIO | None,
        creationflags: int,
    ) -> _NativeWindowsProcess:
        if stdout in {subprocess.PIPE, subprocess.STDOUT} or stderr in {
            subprocess.PIPE,
            subprocess.STDOUT,
        }:
            raise WindowsJobError("Atomic Job launch does not expose pipe handles")
        if (
            not isinstance(creationflags, int)
            or isinstance(creationflags, bool)
            or creationflags < 0
            or creationflags & self._CREATE_BREAKAWAY_FROM_JOB
            or creationflags & ~self._ALLOWED_CALLER_CREATION_FLAGS
        ):
            raise WindowsJobError("Subprocess creation flags violate Job containment")
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(list(command)))
        environment_block = self._environment_block(env)
        startup = _StartupInfoW()
        startup.cb = ctypes.sizeof(startup)
        process_info = _ProcessInformation()
        owned_handles: list[Any] = []
        process_created = False
        try:
            stdin_source = self._nul_handle(writable=False)
            owned_handles.append(stdin_source)
            stdin_handle = self._duplicate_inheritable_handle(self._handle_value(stdin_source))
            owned_handles.append(stdin_handle)
            if stdout is None:
                stdout_source = self._nul_handle(writable=True)
                owned_handles.append(stdout_source)
                stdout_numeric = self._handle_value(stdout_source)
            else:
                stdout_numeric = self._stream_handle(stdout)
            stdout_handle = self._duplicate_inheritable_handle(stdout_numeric)
            owned_handles.append(stdout_handle)
            if stderr is None:
                stderr_source = self._nul_handle(writable=True)
                owned_handles.append(stderr_source)
                stderr_numeric = self._handle_value(stderr_source)
            else:
                stderr_numeric = self._stream_handle(stderr)
            stderr_handle = self._duplicate_inheritable_handle(stderr_numeric)
            owned_handles.append(stderr_handle)
            startup.dwFlags = self._STARTF_USESTDHANDLES
            startup.hStdInput = stdin_handle
            startup.hStdOutput = stdout_handle
            startup.hStdError = stderr_handle
            flags = int(creationflags) | self._CREATE_SUSPENDED | self._CREATE_UNICODE_ENVIRONMENT
            if not self.kernel32.CreateProcessW(
                str(Path(command[0]).resolve(strict=True)),
                command_line,
                None,
                None,
                True,
                flags,
                ctypes.cast(environment_block, ctypes.c_void_p),
                str(Path(cwd).resolve(strict=True)),
                ctypes.byref(startup),
                ctypes.byref(process_info),
            ):
                raise self._error("CreateProcessW")
            process_created = True
            self.assign_process(
                job_handle,
                self._handle_value(process_info.hProcess),
            )
            in_job = wintypes.BOOL()
            if not self.kernel32.IsProcessInJob(
                process_info.hProcess,
                job_handle,
                ctypes.byref(in_job),
            ):
                raise self._error("IsProcessInJob")
            if not in_job.value:
                raise WindowsJobError("Suspended subprocess was not assigned to its Job Object")
            return _NativeWindowsProcess(
                api=self,
                handle=process_info.hProcess,
                thread_handle=process_info.hThread,
                pid=int(process_info.dwProcessId),
            )
        except Exception:
            if process_created and process_info.hProcess:
                self.kernel32.TerminateProcess(process_info.hProcess, 1)
            if process_info.hThread:
                try:
                    self.close_handle(process_info.hThread)
                except Exception:
                    pass
            if process_info.hProcess:
                try:
                    self.close_handle(process_info.hProcess)
                except Exception:
                    pass
            raise
        finally:
            for handle in reversed(owned_handles):
                try:
                    self.close_handle(handle)
                except Exception:
                    pass

    def poll_process(self, handle: Any) -> int | None:
        exit_code = wintypes.DWORD()
        if not self.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise self._error("GetExitCodeProcess")
        return None if exit_code.value == self._STILL_ACTIVE else int(exit_code.value)

    def wait_process(self, handle: Any, timeout_seconds: float) -> int:
        milliseconds = min(0xFFFFFFFE, max(1, math.ceil(timeout_seconds * 1000)))
        wait_result = self.kernel32.WaitForSingleObject(handle, milliseconds)
        if wait_result == self._WAIT_TIMEOUT:
            raise subprocess.TimeoutExpired("suspended-assigned Windows process", timeout_seconds)
        if wait_result != self._WAIT_OBJECT_0:
            raise self._error("WaitForSingleObject")
        exit_code = self.poll_process(handle)
        if exit_code is None:
            raise WindowsJobError("Signaled Windows process still reports active")
        return exit_code

    def terminate_process(self, handle: Any, exit_code: int) -> None:
        if not self.kernel32.TerminateProcess(handle, wintypes.UINT(exit_code)):
            raise self._error("TerminateProcess")

    def resume_process_thread(self, thread_handle: Any) -> None:
        if self.kernel32.ResumeThread(thread_handle) == 0xFFFFFFFF:
            raise self._error("ResumeThread")

    def process_creation_time_filetime(self, handle: Any) -> int:
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not self.kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise self._error("GetProcessTimes")
        return (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)

    def current_process_membership(self) -> tuple[bool, int]:
        current = self.kernel32.GetCurrentProcess()
        return self._process_membership(current)

    def _process_membership(self, process_handle: Any) -> tuple[bool, int]:
        in_job = wintypes.BOOL()
        if not self.kernel32.IsProcessInJob(
            process_handle,
            None,
            ctypes.byref(in_job),
        ):
            raise self._error("IsProcessInJob(process)")
        return bool(in_job.value), self.process_creation_time_filetime(process_handle)

    def process_membership(self, pid: int) -> tuple[bool, int, bool]:
        handle = self.kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not handle:
            raise self._error("OpenProcess(job membership audit)")
        try:
            wait_result = self.kernel32.WaitForSingleObject(handle, 0)
            if wait_result not in {self._WAIT_OBJECT_0, self._WAIT_TIMEOUT}:
                raise self._error("WaitForSingleObject(job membership audit)")
            in_job, creation_time = self._process_membership(handle)
            return in_job, creation_time, wait_result == self._WAIT_TIMEOUT
        finally:
            self.close_handle(handle)


class _NativeWindowsProcess:
    def __init__(
        self,
        *,
        api: _Kernel32JobApi,
        handle: Any,
        thread_handle: Any,
        pid: int,
    ) -> None:
        self._api = api
        self._handle = handle
        self._thread_handle = thread_handle
        self.pid = pid
        self.returncode: int | None = None
        self._closed = False
        self._resumed = False
        self.creation_time_filetime = self._api.process_creation_time_filetime(handle)

    @property
    def resumed(self) -> bool:
        return self._resumed

    def resume(self) -> None:
        if self._resumed:
            raise WindowsJobError("Suspended Windows process was already resumed")
        if not self._thread_handle:
            raise WindowsJobError("Suspended Windows process lacks its primary thread")
        self._api.resume_process_thread(self._thread_handle)
        self._api.close_handle(self._thread_handle)
        self._thread_handle = None
        self._resumed = True

    def poll(self) -> int | None:
        if self.returncode is None:
            self.returncode = self._api.poll_process(self._handle)
        return self.returncode

    def wait(self, timeout: float = 30.0) -> int:
        if not self._resumed:
            raise WindowsJobError("Cannot wait for a suspended Windows process")
        if self.returncode is None:
            self.returncode = self._api.wait_process(self._handle, timeout)
        return self.returncode

    def communicate(self, timeout: float = 30.0) -> tuple[None, None]:
        self.wait(timeout)
        return None, None

    def kill(self) -> None:
        if self.poll() is None:
            self._api.terminate_process(self._handle, 1)

    def close(self) -> None:
        if not self._closed:
            if self._thread_handle:
                self._api.close_handle(self._thread_handle)
                self._thread_handle = None
            self._api.close_handle(self._handle)
            self._closed = True


@dataclass(frozen=True)
class JobRunResult:
    args: tuple[str, ...]
    returncode: int
    stdout: bytes | str | None
    stderr: bytes | str | None
    job_audit: dict[str, Any]


class WindowsKillOnCloseJob:
    """Own a Job Object whose complete process tree dies when the handle closes."""

    def __init__(self, *, memory_limit_bytes: int | None = None, api: Any | None = None) -> None:
        if memory_limit_bytes is not None and (
            not isinstance(memory_limit_bytes, int)
            or isinstance(memory_limit_bytes, bool)
            or memory_limit_bytes <= 0
        ):
            raise WindowsJobError("Windows Job process memory limit is invalid")
        self._api = api if api is not None else _Kernel32JobApi()
        self._handle = self._api.create_job()
        self._closed = False
        self._assigned_pids: list[int] = []
        self._memory_limit_bytes = memory_limit_bytes
        try:
            self._api.enable_kill_on_close(self._handle)
            if memory_limit_bytes is not None:
                self._api.set_process_memory_limit(self._handle, memory_limit_bytes)
            self._limit_flags = self._api.limit_flags(self._handle)
            expected_flags = _Kernel32JobApi._KILL_ON_JOB_CLOSE
            if memory_limit_bytes is not None:
                expected_flags |= _Kernel32JobApi._PROCESS_MEMORY_LIMIT
            if self._limit_flags != expected_flags:
                raise WindowsJobError(
                    "Windows Job Object limit flags are not exactly the containment contract"
                )
            if memory_limit_bytes is not None and self._api.process_memory_limit(self._handle) != memory_limit_bytes:
                raise WindowsJobError("Windows Job Object process memory limit differs from contract")
        except Exception:
            self._api.close_handle(self._handle)
            self._closed = True
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def assigned_pids(self) -> tuple[int, ...]:
        return tuple(self._assigned_pids)

    @property
    def limit_flags(self) -> int:
        return self._limit_flags

    @property
    def memory_limit_bytes(self) -> int | None:
        return self._memory_limit_bytes

    def _require_open(self) -> None:
        if self._closed:
            raise WindowsJobError("Windows Job Object is already closed")

    def spawn_suspended(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        env: Mapping[str, str],
        stdout: BinaryIO | None,
        stderr: BinaryIO | None,
        creationflags: int = 0,
    ) -> _NativeWindowsProcess:
        self._require_open()
        process = self._api.create_suspended_assigned_process(
            self._handle,
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )
        self._assigned_pids.append(process.pid)
        return process

    def assignment_audit(self, process: _NativeWindowsProcess) -> dict[str, Any]:
        self._require_open()
        if process.pid not in self._assigned_pids or process.resumed:
            raise WindowsJobError("Process is not in the pre-resume assigned state")
        return {
            "creation_mode": "create_process_suspended_assign_verify_resume",
            "kill_on_job_close": True,
            "exact_limit_flags": self._limit_flags,
            "breakaway_allowed": False,
            "process_memory_limit_bytes": self._memory_limit_bytes,
            "process_pid": process.pid,
            "process_creation_time_filetime": process.creation_time_filetime,
            "assigned_before_resume": True,
            "process_resumed": False,
        }

    def active_processes(self) -> int:
        self._require_open()
        active = self._api.active_processes(self._handle)
        if not isinstance(active, int) or isinstance(active, bool) or active < 0:
            raise WindowsJobError("Windows Job Object returned an invalid active count")
        return active

    def wait_empty(
        self,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = 0.05,
    ) -> int:
        self._require_open()
        if (
            not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or not math.isfinite(poll_interval_seconds)
            or poll_interval_seconds <= 0
        ):
            raise WindowsJobError("Windows Job wait timeout is invalid")
        deadline = time.monotonic() + timeout_seconds
        while True:
            active = self.active_processes()
            if active == 0:
                return 0
            if time.monotonic() >= deadline:
                raise WindowsJobError(
                    f"Windows Job Object still contains {active} active processes"
                )
            time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))

    def terminate(
        self,
        *,
        exit_code: int = 1,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        self._require_open()
        if (
            not isinstance(exit_code, int)
            or isinstance(exit_code, bool)
            or not 0 <= exit_code <= 0xFFFFFFFF
        ):
            raise WindowsJobError("Windows Job termination exit code is invalid")
        active_before = self.active_processes()
        self._api.terminate_job(self._handle, exit_code)
        active_after = self.wait_empty(timeout_seconds=timeout_seconds)
        return {
            "method": "windows_job_object_terminate",
            "kill_on_job_close": True,
            "assigned_process_count": len(self._assigned_pids),
            "active_processes_before": active_before,
            "active_processes_after": active_after,
            "termination_exit_code": exit_code,
            "tree_termination_confirmed": active_after == 0,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._api.close_handle(self._handle)
        self._closed = True

    def __enter__(self) -> WindowsKillOnCloseJob:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_closed", True):
            return
        try:
            self.close()
        except Exception:
            pass


def run_subprocess_in_job(
    command: Sequence[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str],
    timeout_seconds: float,
    stdout: int | BinaryIO | None = None,
    stderr: int | BinaryIO | None = None,
    creationflags: int = 0,
    job_factory: Callable[[], Any] = WindowsKillOnCloseJob,
    before_resume: Callable[[Any, Mapping[str, Any]], None] | None = None,
    monitor_callback: Callable[[], None] | None = None,
    monitor_interval_seconds: float = 1.0,
) -> JobRunResult:
    """Run one subprocess and fail closed if it or any assigned descendant survives."""
    frozen_command = tuple(command)
    if (
        not frozen_command
        or any(not isinstance(item, str) or not item for item in frozen_command)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or not math.isfinite(monitor_interval_seconds)
        or monitor_interval_seconds <= 0
    ):
        raise WindowsJobError("Windows Job subprocess invocation is invalid")
    job = job_factory()
    process: Any | None = None
    try:
        process = job.spawn_suspended(
            frozen_command,
            cwd=cwd,
            env=dict(env),
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )
        assignment_audit = job.assignment_audit(process)
        if before_resume is not None:
            before_resume(process, assignment_audit)
        process.resume()
        try:
            deadline = time.monotonic() + timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(frozen_command, timeout_seconds)
                try:
                    process.wait(timeout=min(monitor_interval_seconds, remaining))
                    break
                except subprocess.TimeoutExpired:
                    if monitor_callback is not None:
                        monitor_callback()
            if monitor_callback is not None:
                monitor_callback()
            captured_stdout, captured_stderr = None, None
        except subprocess.TimeoutExpired as exc:
            termination = job.terminate(timeout_seconds=30.0)
            process.communicate(timeout=30.0)
            raise WindowsJobTimeoutError(
                frozen_command,
                timeout_seconds,
                termination=termination,
            ) from exc
        active_after_root = job.active_processes()
        if active_after_root != 0:
            termination = job.terminate(timeout_seconds=30.0)
            raise WindowsJobError(
                f"Subprocess root exited while assigned descendants remained: {termination}"
            )
        return JobRunResult(
            args=frozen_command,
            returncode=int(process.returncode),
            stdout=captured_stdout,
            stderr=captured_stderr,
            job_audit={
                **assignment_audit,
                "kill_on_job_close": True,
                "assigned_process_count": len(job.assigned_pids),
                "active_processes_after": active_after_root,
                "tree_empty_after_root_exit": True,
                "process_resumed": True,
            },
        )
    except Exception:
        if process is not None and process.poll() is None:
            try:
                job.terminate(timeout_seconds=30.0)
                process.wait(timeout=30.0)
            except Exception:
                pass
        raise
    finally:
        if process is not None and hasattr(process, "close"):
            process.close()
        job.close()


def audit_current_process_job_membership(
    expected_creation_time_filetime: int | None = None,
    *,
    expected_pid: int | None = None,
    api: Any | None = None,
) -> dict[str, Any]:
    """Prove the current PID, creation time, and active Windows Job membership."""
    if expected_creation_time_filetime is not None and (
        not isinstance(expected_creation_time_filetime, int)
        or isinstance(expected_creation_time_filetime, bool)
        or expected_creation_time_filetime <= 0
    ):
        raise WindowsJobError("Expected process creation FILETIME is invalid")
    if expected_pid is not None and (
        not isinstance(expected_pid, int) or isinstance(expected_pid, bool) or expected_pid <= 0
    ):
        raise WindowsJobError("Expected process PID is invalid")
    observed_pid = os.getpid()
    if expected_pid is not None and observed_pid != expected_pid:
        raise WindowsJobError("Current process PID differs from its launch receipt")
    observed_api = api if api is not None else _Kernel32JobApi()
    in_job, creation_time = observed_api.current_process_membership()
    if not in_job:
        raise WindowsJobError("Current process is not assigned to a Windows Job Object")
    if (
        expected_creation_time_filetime is not None
        and creation_time != expected_creation_time_filetime
    ):
        raise WindowsJobError("Current process creation FILETIME differs from its launch receipt")
    return {
        "pid": observed_pid,
        "creation_time_filetime": creation_time,
        "in_job": True,
        "verification_scope": (
            "current_process_membership_plus_pre_resume_supervisor_assignment_receipt"
        ),
    }


def audit_process_job_membership(
    pid: int,
    expected_creation_time_filetime: int,
    *,
    api: Any | None = None,
) -> dict[str, Any]:
    """Prove a live launcher PID still has its receipt-bound Job membership."""
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not isinstance(expected_creation_time_filetime, int)
        or isinstance(expected_creation_time_filetime, bool)
        or expected_creation_time_filetime <= 0
    ):
        raise WindowsJobError("Expected launcher process identity is invalid")
    observed_api = api if api is not None else _Kernel32JobApi()
    in_job, creation_time, active = observed_api.process_membership(pid)
    if not active:
        raise WindowsJobError("Receipt-bound launcher process is no longer active")
    if not in_job:
        raise WindowsJobError("Receipt-bound launcher process is not in a Windows Job")
    if creation_time != expected_creation_time_filetime:
        raise WindowsJobError("Receipt-bound launcher creation FILETIME differs from observation")
    return {
        "pid": pid,
        "creation_time_filetime": creation_time,
        "in_job": True,
        "active": True,
        "verification_scope": "receipt_bound_launcher_process_job_membership",
    }


__all__ = [
    "JobRunResult",
    "WindowsJobError",
    "WindowsJobTimeoutError",
    "WindowsKillOnCloseJob",
    "audit_current_process_job_membership",
    "audit_process_job_membership",
    "run_subprocess_in_job",
]
