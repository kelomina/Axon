from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import loop166.windows_job as windows_job  # noqa: E402


class FakeProcess:
    def __init__(self, *, pid: int = 41, timeout: bool = False) -> None:
        self.pid = pid
        self.creation_time_filetime = 123456789
        self.resumed = False
        self.returncode = None
        self.closed = False
        self.timeout = timeout
        self.on_exit = lambda: None

    def resume(self) -> None:
        assert not self.resumed
        self.resumed = True

    def poll(self):
        return self.returncode

    def communicate(self, timeout: float):
        assert self.resumed
        if self.timeout and self.returncode is None:
            raise subprocess.TimeoutExpired(("synthetic",), timeout)
        self.returncode = 0 if self.returncode is None else self.returncode
        self.on_exit()
        return None, None

    def wait(self, timeout: float):
        if self.timeout and self.returncode is None:
            raise subprocess.TimeoutExpired(("synthetic",), timeout)
        self.returncode = 0 if self.returncode is None else self.returncode
        self.on_exit()
        return self.returncode

    def close(self) -> None:
        self.closed = True


class FakeApi:
    def __init__(self, *, limit_flags: int = 0x2000) -> None:
        self.calls = []
        self.limit_flag_value = limit_flags
        self.process = FakeProcess()
        self.process.on_exit = lambda: setattr(self, "active", 0)
        self.active = 0
        self.memory_limit = None

    def create_job(self):
        self.calls.append("create_job")
        return 101

    def enable_kill_on_close(self, handle):
        self.calls.append(("enable_kill_on_close", handle))

    def set_process_memory_limit(self, handle, memory_limit_bytes):
        self.calls.append(("set_process_memory_limit", handle, memory_limit_bytes))
        self.limit_flag_value |= 0x100
        self.memory_limit = memory_limit_bytes

    def process_memory_limit(self, handle):
        self.calls.append(("process_memory_limit", handle))
        return self.memory_limit

    def limit_flags(self, handle):
        self.calls.append(("limit_flags", handle))
        return self.limit_flag_value

    def create_suspended_assigned_process(self, handle, command, **kwargs):
        self.calls.append(("create_suspended_assigned", handle, tuple(command)))
        self.active = 1
        return self.process

    def active_processes(self, handle):
        return self.active

    def terminate_job(self, handle, exit_code):
        self.calls.append(("terminate_job", handle, exit_code))
        self.process.returncode = exit_code
        self.active = 0

    def close_handle(self, handle):
        self.calls.append(("close_handle", handle))


def test_job_configures_exact_kill_on_close_before_suspended_spawn(tmp_path: Path):
    api = FakeApi()
    job = windows_job.WindowsKillOnCloseJob(api=api)

    process = job.spawn_suspended(
        (str(tmp_path / "python.exe"), "worker.py"),
        cwd=tmp_path,
        env={},
        stdout=None,
        stderr=None,
    )
    audit = job.assignment_audit(process)

    assert api.calls[:4] == [
        "create_job",
        ("enable_kill_on_close", 101),
        ("limit_flags", 101),
        (
            "create_suspended_assigned",
            101,
            (str(tmp_path / "python.exe"), "worker.py"),
        ),
    ]
    assert audit["assigned_before_resume"] is True
    assert audit["process_resumed"] is False
    assert audit["exact_limit_flags"] == 0x2000
    assert process.resumed is False
    job.terminate(timeout_seconds=1)
    job.close()


def test_job_rejects_any_limit_flag_drift_and_closes_handle():
    api = FakeApi(limit_flags=0x2000 | 0x1000)

    with pytest.raises(windows_job.WindowsJobError, match="not exactly"):
        windows_job.WindowsKillOnCloseJob(api=api)

    assert ("close_handle", 101) in api.calls


def test_job_records_verified_process_memory_limit():
    api = FakeApi()
    job = windows_job.WindowsKillOnCloseJob(memory_limit_bytes=1024, api=api)

    assert job.memory_limit_bytes == 1024
    assert ("set_process_memory_limit", 101, 1024) in api.calls
    assert ("process_memory_limit", 101) in api.calls
    job.close()


def test_run_callback_happens_after_assignment_and_before_resume(tmp_path: Path):
    api = FakeApi()
    job = windows_job.WindowsKillOnCloseJob(api=api)
    events = []

    result = windows_job.run_subprocess_in_job(
        (str(tmp_path / "python.exe"), "worker.py"),
        cwd=tmp_path,
        env={},
        timeout_seconds=1,
        job_factory=lambda: job,
        before_resume=lambda process, audit: events.append(
            (process.resumed, audit["assigned_before_resume"])
        ),
    )

    assert events == [(False, True)]
    assert result.returncode == 0
    assert result.job_audit["process_resumed"] is True
    assert api.process.closed is True
    assert job.closed is True


def test_pre_resume_callback_failure_terminates_suspended_tree(tmp_path: Path):
    api = FakeApi()
    job = windows_job.WindowsKillOnCloseJob(api=api)

    def fail_before_resume(process, audit):
        raise RuntimeError("receipt failed")

    with pytest.raises(RuntimeError, match="receipt failed"):
        windows_job.run_subprocess_in_job(
            (str(tmp_path / "python.exe"), "worker.py"),
            cwd=tmp_path,
            env={},
            timeout_seconds=1,
            job_factory=lambda: job,
            before_resume=fail_before_resume,
        )

    assert api.process.resumed is False
    assert ("terminate_job", 101, 1) in api.calls
    assert api.process.closed is True


def test_timeout_terminates_entire_assigned_job(tmp_path: Path):
    api = FakeApi()
    api.process.timeout = True
    job = windows_job.WindowsKillOnCloseJob(api=api)

    with pytest.raises(windows_job.WindowsJobTimeoutError) as captured:
        windows_job.run_subprocess_in_job(
            (str(tmp_path / "python.exe"), "worker.py"),
            cwd=tmp_path,
            env={},
            timeout_seconds=1,
            job_factory=lambda: job,
        )

    assert captured.value.termination["tree_termination_confirmed"] is True
    assert captured.value.termination["active_processes_after"] == 0
    assert api.process.closed is True


def test_breakaway_creation_flag_is_rejected_before_winapi_use(tmp_path: Path):
    api = object.__new__(windows_job._Kernel32JobApi)

    with pytest.raises(windows_job.WindowsJobError, match="creation flags"):
        api.create_suspended_assigned_process(
            1,
            (str(tmp_path / "python.exe"),),
            cwd=tmp_path,
            env={},
            stdout=None,
            stderr=None,
            creationflags=0x01000000,
        )


def test_current_process_membership_binds_pid_and_creation_time(monkeypatch):
    class MembershipApi:
        @staticmethod
        def current_process_membership():
            return True, 99887766

    monkeypatch.setattr(windows_job.os, "getpid", lambda: 1234)
    audit = windows_job.audit_current_process_job_membership(
        99887766,
        expected_pid=1234,
        api=MembershipApi(),
    )

    assert audit["pid"] == 1234
    assert audit["creation_time_filetime"] == 99887766
    assert audit["in_job"] is True

    with pytest.raises(windows_job.WindowsJobError, match="creation FILETIME"):
        windows_job.audit_current_process_job_membership(
            99887767,
            expected_pid=1234,
            api=MembershipApi(),
        )


def test_current_process_membership_rejects_no_job(monkeypatch):
    class MembershipApi:
        @staticmethod
        def current_process_membership():
            return False, 99887766

    monkeypatch.setattr(windows_job.os, "getpid", lambda: 1234)
    with pytest.raises(windows_job.WindowsJobError, match="not assigned"):
        windows_job.audit_current_process_job_membership(api=MembershipApi())


def test_receipt_bound_launcher_membership_binds_pid_creation_and_liveness():
    class MembershipApi:
        @staticmethod
        def process_membership(pid):
            assert pid == 4321
            return True, 11223344, True

    audit = windows_job.audit_process_job_membership(
        4321,
        11223344,
        api=MembershipApi(),
    )

    assert audit == {
        "pid": 4321,
        "creation_time_filetime": 11223344,
        "in_job": True,
        "active": True,
        "verification_scope": "receipt_bound_launcher_process_job_membership",
    }

    class DeadMembershipApi:
        @staticmethod
        def process_membership(pid):
            return True, 11223344, False

    with pytest.raises(windows_job.WindowsJobError, match="no longer active"):
        windows_job.audit_process_job_membership(
            4321,
            11223344,
            api=DeadMembershipApi(),
        )


def _windows_pid_is_active(pid: int) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.WaitForSingleObject.restype = ctypes.c_ulong
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(0x00100000, 0, pid)
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == 0x00000102
    finally:
        kernel32.CloseHandle(handle)


@pytest.mark.skipif(platform.system().casefold() != "windows", reason="Windows-only")
def test_real_windows_timeout_kills_immediate_grandchild(tmp_path: Path):
    pid_file = tmp_path / "pids.json"
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    probe = (
        "import json,os,pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys._base_executable,'-c','import time;time.sleep(300)']);"
        f"pathlib.Path({str(pid_file)!r}).write_text(json.dumps([os.getpid(),child.pid]));"
        "time.sleep(300)"
    )
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        with pytest.raises(windows_job.WindowsJobTimeoutError) as captured:
            windows_job.run_subprocess_in_job(
                (sys.executable, "-u", "-c", probe),
                cwd=tmp_path,
                env=dict(os.environ, PYTHONUNBUFFERED="1"),
                timeout_seconds=3,
                stdout=stdout,
                stderr=stderr,
            )

    assert captured.value.termination["tree_termination_confirmed"] is True
    pids = __import__("json").loads(pid_file.read_text(encoding="utf-8"))
    assert len(pids) == 2
    deadline = time.monotonic() + 5
    while any(_windows_pid_is_active(int(pid)) for pid in pids):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    assert all(not _windows_pid_is_active(int(pid)) for pid in pids)


@pytest.mark.skipif(platform.system().casefold() != "windows", reason="Windows-only")
def test_real_windows_close_handle_kills_immediate_grandchild(tmp_path: Path):
    pid_file = tmp_path / "close-pids.json"
    probe = (
        "import json,os,pathlib,subprocess,sys,time;"
        "child=subprocess.Popen([sys._base_executable,'-c','import time;time.sleep(300)']);"
        f"pathlib.Path({str(pid_file)!r}).write_text(json.dumps([os.getpid(),child.pid]));"
        "time.sleep(300)"
    )
    stdout_path = tmp_path / "close-stdout.log"
    stderr_path = tmp_path / "close-stderr.log"
    job = windows_job.WindowsKillOnCloseJob()
    process = None
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = job.spawn_suspended(
                (sys.executable, "-u", "-c", probe),
                cwd=tmp_path,
                env=dict(os.environ, PYTHONUNBUFFERED="1"),
                stdout=stdout,
                stderr=stderr,
            )
            audit = job.assignment_audit(process)
            assert audit["exact_limit_flags"] == 0x2000
            process.resume()
            deadline = time.monotonic() + 5
            while not pid_file.exists():
                if time.monotonic() >= deadline:
                    raise AssertionError("Synthetic grandchild did not publish its PIDs")
                time.sleep(0.05)
            job.close()
            process.wait(timeout=5)
        pids = __import__("json").loads(pid_file.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 5
        while any(_windows_pid_is_active(int(pid)) for pid in pids):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        assert all(not _windows_pid_is_active(int(pid)) for pid in pids)
    finally:
        if not job.closed:
            job.close()
        if process is not None:
            process.close()


@pytest.mark.skipif(platform.system().casefold() != "windows", reason="Windows-only")
def test_real_windows_child_can_audit_its_job_membership(tmp_path: Path):
    stdout_path = tmp_path / "membership.json"
    stderr_path = tmp_path / "membership.stderr"
    identity_path = tmp_path / "launcher-identity.json"
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=str(SRC_DIR))
    probe = (
        "import json,pathlib;"
        "from loop166.windows_job import audit_current_process_job_membership,audit_process_job_membership;"
        f"identity=json.loads(pathlib.Path({str(identity_path)!r}).read_text());"
        "print(json.dumps({'current':audit_current_process_job_membership(),"
        "'launcher':audit_process_job_membership(identity['pid'],identity['creation_time_filetime'])}))"
    )

    def persist_launcher_identity(process, audit):
        identity_path.write_text(
            __import__("json").dumps(
                {
                    "pid": process.pid,
                    "creation_time_filetime": audit["process_creation_time_filetime"],
                }
            ),
            encoding="utf-8",
        )

    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        result = windows_job.run_subprocess_in_job(
            (sys.executable, "-u", "-c", probe),
            cwd=tmp_path,
            env=environment,
            timeout_seconds=30,
            stdout=stdout,
            stderr=stderr,
            before_resume=persist_launcher_identity,
        )

    assert result.returncode == 0
    audit = __import__("json").loads(stdout_path.read_text(encoding="utf-8"))
    assert audit["current"]["in_job"] is True
    assert audit["launcher"]["in_job"] is True
    assert audit["launcher"]["active"] is True
    assert audit["current"]["pid"] != audit["launcher"]["pid"]
