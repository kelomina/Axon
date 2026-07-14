"""Windows Job Object readiness and one-process resource containment for Phase B v4."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass


class WindowsJobError(RuntimeError):
    """Raised when the required Windows Job Object boundary is unavailable."""


JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


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
        ("LimitFlags", ctypes.c_ulong),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_ulong),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_ulong),
        ("SchedulingClass", ctypes.c_ulong),
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


def _require_limit(limit_bytes: int) -> int:
    if isinstance(limit_bytes, bool) or not isinstance(limit_bytes, int) or limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be a positive integer")
    return limit_bytes


def _kernel32() -> object:
    if os.name != "nt":
        raise WindowsJobError("Loop167 Phase B v4 requires Windows Job Objects")
    return ctypes.windll.kernel32


def _last_error(label: str) -> WindowsJobError:
    return WindowsJobError(f"{label} failed with Win32 error {ctypes.get_last_error()}")


@dataclass
class WindowsJob:
    """A process job with a hard memory ceiling and kill-on-close semantics."""

    handle: int
    memory_limit_bytes: int
    _closed: bool = False

    @classmethod
    def create(cls, *, memory_limit_bytes: int) -> "WindowsJob":
        limit = _require_limit(memory_limit_bytes)
        kernel32 = _kernel32()
        handle = kernel32.CreateJobObjectW(None, None)
        if handle in (None, 0, INVALID_HANDLE_VALUE):
            raise _last_error("CreateJobObjectW")
        job = cls(int(handle), limit)
        try:
            info = _ExtendedLimitInformation()
            info.BasicLimitInformation.LimitFlags = (
                JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            info.ProcessMemoryLimit = limit
            if not kernel32.SetInformationJobObject(
                ctypes.c_void_p(job.handle),
                JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info),
                ctypes.sizeof(info),
            ):
                raise _last_error("SetInformationJobObject")
            return job
        except Exception:
            job.close()
            raise

    def assign_current_process(self) -> None:
        if self._closed:
            raise WindowsJobError("Windows Job Object is closed")
        kernel32 = _kernel32()
        if not kernel32.AssignProcessToJobObject(
            ctypes.c_void_p(self.handle),
            kernel32.GetCurrentProcess(),
        ):
            raise _last_error("AssignProcessToJobObject")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.handle:
            _kernel32().CloseHandle(ctypes.c_void_p(self.handle))

    def __enter__(self) -> "WindowsJob":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def probe_windows_job_ready(*, memory_limit_bytes: int) -> tuple[bool, str | None]:
    """Create and configure a disposable Job Object without assigning the caller."""

    try:
        with WindowsJob.create(memory_limit_bytes=memory_limit_bytes):
            return True, None
    except (ValueError, WindowsJobError) as error:
        return False, str(error)
