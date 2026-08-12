from __future__ import annotations

import ctypes

import pytest

from src.loop167_phase_b import windows_job_v10


class FakeApi:
    def __init__(self, *, limit_flags: int, assignment_error: int | None = None) -> None:
        self.limit_flags_value = limit_flags
        self.assignment_error = assignment_error
        self.calls: list[object] = []

    def create_job(self):
        self.calls.append("create_job")
        return 101

    def configure_job(self, handle, *, memory_limit_bytes, kill_on_close):
        self.calls.append(("configure_job", handle, memory_limit_bytes, kill_on_close))

    def limit_flags(self, handle):
        self.calls.append(("limit_flags", handle))
        return self.limit_flags_value

    def current_process(self):
        self.calls.append("current_process")
        return -1

    def assign_process(self, job_handle, process_handle):
        self.calls.append(("assign_process", job_handle, process_handle))
        if self.assignment_error is not None:
            raise windows_job_v10.WindowsJobV10Error(
                "AssignProcessToJobObject",
                self.assignment_error,
            )

    def is_process_in_job(self, process_handle, job_handle):
        self.calls.append(("is_process_in_job", process_handle, job_handle))
        return True

    def close_handle(self, handle):
        self.calls.append(("close_handle", handle))


class FakeSuspendedProcess:
    def __init__(self) -> None:
        self.process_handle = 222
        self.thread_handle = 333
        self.pid = 444
        self.creation_time_filetime = 555
        self.resumed = False
        self.terminated = False
        self.closed = False

    def resume(self) -> None:
        self.resumed = True

    def terminate(self, _exit_code: int = 1) -> None:
        self.terminated = True

    def close(self) -> None:
        self.closed = True


class ChildApi(FakeApi):
    def __init__(self, *, limit_flags: int, child_membership: bool = True) -> None:
        super().__init__(limit_flags=limit_flags)
        self.child_membership = child_membership
        self.child = FakeSuspendedProcess()

    def create_suspended_process(self, command, *, cwd, environment):
        self.calls.append(("create_suspended_process", tuple(command), cwd, dict(environment)))
        return self.child

    def is_process_in_job(self, process_handle, job_handle):
        self.calls.append(("is_process_in_job", process_handle, job_handle))
        return self.child_membership


def test_controller_job_uses_memory_and_kill_on_close_flags():
    expected_flags = (
        windows_job_v10.JOB_OBJECT_LIMIT_PROCESS_MEMORY
        | windows_job_v10.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    api = FakeApi(limit_flags=expected_flags)

    job = windows_job_v10.WindowsJobV10.create(
        memory_limit_bytes=4096,
        kill_on_close=True,
        api=api,
    )
    assignment = job.assign_current_process()
    job.close()

    assert assignment["current_process_assigned"] is True
    assert assignment["job_limit_flags"] == expected_flags
    assert api.calls == [
        "create_job",
        ("configure_job", 101, 4096, True),
        ("limit_flags", 101),
        "current_process",
        ("assign_process", 101, -1),
        ("is_process_in_job", -1, 101),
        ("close_handle", 101),
    ]


def test_nonkill_probe_performs_real_assignment_and_records_error_code():
    expected_flags = windows_job_v10.JOB_OBJECT_LIMIT_PROCESS_MEMORY
    api = FakeApi(limit_flags=expected_flags, assignment_error=6)

    result = windows_job_v10.probe_windows_job_assignment_v10(
        memory_limit_bytes=4096,
        job_factory=lambda **kwargs: windows_job_v10.WindowsJobV10.create(api=api, **kwargs),
    )

    assert result.ready is False
    assert result.operation == "AssignProcessToJobObject"
    assert result.win32_error_code == 6
    assert result.detail == "AssignProcessToJobObject failed with Win32 error 6"
    assert ("configure_job", 101, 4096, False) in api.calls
    assert ("close_handle", 101) in api.calls


def test_create_rejects_flag_drift_and_closes_handle():
    api = FakeApi(limit_flags=windows_job_v10.JOB_OBJECT_LIMIT_PROCESS_MEMORY)

    with pytest.raises(windows_job_v10.WindowsJobV10Error, match="unexpected limit flags"):
        windows_job_v10.WindowsJobV10.create(
            memory_limit_bytes=4096,
            kill_on_close=True,
            api=api,
        )

    assert ("close_handle", 101) in api.calls


def test_suspended_child_is_assigned_and_audited_before_resume(tmp_path):
    expected_flags = (
        windows_job_v10.JOB_OBJECT_LIMIT_PROCESS_MEMORY
        | windows_job_v10.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    api = ChildApi(limit_flags=expected_flags)
    job = windows_job_v10.WindowsJobV10.create(
        memory_limit_bytes=4096,
        kill_on_close=True,
        api=api,
    )

    child = job.spawn_suspended_assigned(
        ("C:/python.exe", "-I", "controller.py", "--probe"),
        cwd=tmp_path,
        environment={"SystemRoot": "C:/Windows"},
    )
    audit = job.assignment_audit(child)

    assert child.resumed is False
    assert audit["assigned_before_resume"] is True
    assert audit["process_resumed"] is False
    assert audit["process_pid"] == 444
    assert ("assign_process", 101, 222) in api.calls
    assert ("is_process_in_job", 222, 101) in api.calls
    job.close()


def test_failed_child_membership_terminates_without_resume(tmp_path):
    expected_flags = (
        windows_job_v10.JOB_OBJECT_LIMIT_PROCESS_MEMORY
        | windows_job_v10.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    api = ChildApi(limit_flags=expected_flags, child_membership=False)
    job = windows_job_v10.WindowsJobV10.create(
        memory_limit_bytes=4096,
        kill_on_close=True,
        api=api,
    )

    with pytest.raises(windows_job_v10.WindowsJobV10Error, match="returned false"):
        job.spawn_suspended_assigned(
            ("C:/python.exe", "-I", "controller.py", "--probe"),
            cwd=tmp_path,
            environment={"SystemRoot": "C:/Windows"},
        )

    assert api.child.terminated is True
    assert api.child.closed is True
    assert api.child.resumed is False
    job.close()


def test_kernel32_api_declares_handle_safe_signatures(monkeypatch):
    class Function:
        def __init__(self, result=1):
            self.result = result
            self.argtypes = None
            self.restype = None

        def __call__(self, *_args):
            return self.result

    class Kernel32:
        def __init__(self):
            self.CreateJobObjectW = Function(result=ctypes.c_void_p(123))
            self.SetInformationJobObject = Function()
            self.QueryInformationJobObject = Function()
            self.AssignProcessToJobObject = Function()
            self.IsProcessInJob = Function()
            self.GetCurrentProcess = Function(result=ctypes.c_void_p(-1).value)
            self.CloseHandle = Function()
            self.CreateProcessW = Function()
            self.ResumeThread = Function()
            self.WaitForSingleObject = Function()
            self.GetExitCodeProcess = Function()
            self.GetProcessTimes = Function()
            self.TerminateProcess = Function()
            self.TerminateJobObject = Function()

    kernel32 = Kernel32()
    observed = {}

    def fake_windll(name, *, use_last_error):
        observed["name"] = name
        observed["use_last_error"] = use_last_error
        return kernel32

    monkeypatch.setattr(windows_job_v10.os, "name", "nt")
    monkeypatch.setattr(windows_job_v10.ctypes, "WinDLL", fake_windll, raising=False)
    api = windows_job_v10._Kernel32JobApiV10()

    assert observed == {"name": "kernel32", "use_last_error": True}
    assert api.kernel32.GetCurrentProcess.argtypes == []
    assert api.kernel32.GetCurrentProcess.restype == windows_job_v10.wintypes.HANDLE
    assert api.kernel32.AssignProcessToJobObject.argtypes == [
        windows_job_v10.wintypes.HANDLE,
        windows_job_v10.wintypes.HANDLE,
    ]
    assert api.kernel32.AssignProcessToJobObject.restype == windows_job_v10.wintypes.BOOL


def test_current_process_accepts_the_valid_minus_one_pseudohandle(monkeypatch):
    class Kernel32:
        @staticmethod
        def GetCurrentProcess():  # noqa: N802
            return ctypes.c_void_p(-1).value

    api = object.__new__(windows_job_v10._Kernel32JobApiV10)
    api.kernel32 = Kernel32()

    assert api.current_process() == ctypes.c_void_p(-1).value
