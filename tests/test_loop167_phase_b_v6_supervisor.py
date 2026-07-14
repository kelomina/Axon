from __future__ import annotations

import ctypes
import json
import os
import subprocess
from pathlib import Path

import pytest

import src.loop167_phase_b.supervisor_v6 as supervisor_v6
from src.loop167_phase_b.contracts import canonical_json_bytes
from src.loop167_phase_b.supervisor_v6 import (
    LAUNCH_RECEIPT_SCHEMA,
    SupervisorConfigV6,
    SupervisorV6Error,
    run_supervised_v6,
    validate_launch_receipt_v6,
)


class FakeChild:
    def __init__(self, launch_path: Path) -> None:
        self.launch_path = launch_path
        self.resumed = False
        self.closed = False
        self.terminated = False
        self.launch_exists_when_resumed = False

    def resume(self) -> None:
        self.launch_exists_when_resumed = self.launch_path.exists()
        self.resumed = True

    def wait(self, _timeout: int) -> int:
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def close(self) -> None:
        self.closed = True


class FakeJob:
    def __init__(self, launch_path: Path) -> None:
        self.child = FakeChild(launch_path)
        self.closed = False
        self.terminated = False

    def spawn_suspended_assigned(self, _command, *, cwd, environment):
        assert cwd.is_dir()
        assert environment["PYTHONUNBUFFERED"] == "1"
        return self.child

    def assignment_audit(self, _child):
        return {
            "creation_mode": "create_process_suspended_assign_verify_resume",
            "assignment_api": "AssignProcessToJobObject",
            "membership_api": "IsProcessInJob",
            "job_limit_flags": 0x2100,
            "kill_on_job_close": True,
            "memory_limit_bytes": 8192,
            "process_pid": 41,
            "process_creation_time_filetime": 99,
            "assigned_before_resume": True,
            "process_resumed": False,
        }

    def active_processes(self) -> int:
        return 0

    def terminate(self, *, exit_code: int) -> int:
        assert exit_code == 1
        self.terminated = True
        return 0

    def close(self) -> None:
        self.closed = True


class TimeoutChild(FakeChild):
    def wait(self, timeout: int) -> int:
        raise subprocess.TimeoutExpired("synthetic", timeout)


class NonzeroChild(FakeChild):
    def wait(self, _timeout: int) -> int:
        return 7


class InvalidAuditJob(FakeJob):
    def assignment_audit(self, child):
        audit = super().assignment_audit(child)
        audit["job_limit_flags"] = 0
        return audit


def _config(root: Path) -> SupervisorConfigV6:
    executable = root / "vnev" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"synthetic")
    controller = root / "scripts" / "controller.py"
    controller.parent.mkdir(parents=True)
    controller.write_text("print('synthetic')\n", encoding="ascii")
    output = root / "reports" / "v6"
    return SupervisorConfigV6(
        project_root=root,
        mode="preflight",
        command=(str(executable), "-I", str(controller), "--preflight"),
        launch_receipt=output / "launch.json",
        exit_receipt=output / "exit.json",
        failure_receipt=output / "failure.json",
        memory_limit_bytes=8192,
        timeout_seconds=30,
        static_bindings={"controller": {"path": "scripts/controller.py", "sha256": "a" * 64}},
    )


def _supervisor_identity() -> dict[str, int | bool | str]:
    return {
        "pid": os.getpid(),
        "creation_time_filetime": 18,
        "in_job": True,
        "verification_scope": "test",
    }


def test_supervisor_persists_receipt_before_resume(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.loop167_phase_b.supervisor_v6._fsync_parent_directory", lambda _parent: None)
    job = FakeJob(config.launch_receipt)

    result = run_supervised_v6(
        config,
        job_factory=lambda **_kwargs: job,
        supervisor_identity_provider=_supervisor_identity,
    )
    launch = json.loads(config.launch_receipt.read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert job.child.launch_exists_when_resumed is True
    assert launch["schema"] == LAUNCH_RECEIPT_SCHEMA
    assert launch["pre_resume_assignment"]["assigned_before_resume"] is True
    assert json.loads(config.exit_receipt.read_text(encoding="utf-8"))["active_processes_after"] == 0
    validated = validate_launch_receipt_v6(
        tmp_path,
        config.launch_receipt,
        mode="preflight",
        expected_bindings=config.static_bindings,
    )
    assert validated.payload["pre_resume_assignment"]["process_pid"] == 41
    assert validated.canonical_sha256 == supervisor_v6.hashlib.sha256(config.launch_receipt.read_bytes()).hexdigest()


def test_supervisor_does_not_resume_when_launch_receipt_cannot_be_written(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.loop167_phase_b.supervisor_v6._fsync_parent_directory", lambda _parent: None)
    config.launch_receipt.parent.mkdir(parents=True)
    config.launch_receipt.write_text("existing", encoding="ascii")
    job = FakeJob(config.launch_receipt)

    with pytest.raises(SupervisorV6Error, match="output already exists"):
        run_supervised_v6(
            config,
            job_factory=lambda **_kwargs: job,
            supervisor_identity_provider=_supervisor_identity,
        )

    assert job.child.resumed is False
    assert job.child.terminated is False
    assert job.closed is False


def test_launch_receipt_rejects_static_binding_tampering(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.loop167_phase_b.supervisor_v6._fsync_parent_directory", lambda _parent: None)
    job = FakeJob(config.launch_receipt)
    run_supervised_v6(
        config,
        job_factory=lambda **_kwargs: job,
        supervisor_identity_provider=_supervisor_identity,
    )
    launch = json.loads(config.launch_receipt.read_text(encoding="utf-8"))
    launch["static_bindings"]["controller"]["sha256"] = "b" * 64
    config.launch_receipt.write_bytes(canonical_json_bytes(launch))

    with pytest.raises(SupervisorV6Error, match="static bindings drifted"):
        validate_launch_receipt_v6(
            tmp_path,
            config.launch_receipt,
            mode="preflight",
            expected_bindings=config.static_bindings,
        )


def test_supervisor_does_not_resume_when_parent_sync_fails(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "src.loop167_phase_b.supervisor_v6._fsync_parent_directory",
        lambda _parent: (_ for _ in ()).throw(SupervisorV6Error("sync failed")),
    )
    job = FakeJob(config.launch_receipt)

    with pytest.raises(SupervisorV6Error, match="sync failed"):
        run_supervised_v6(
            config,
            job_factory=lambda **_kwargs: job,
            supervisor_identity_provider=_supervisor_identity,
        )

    assert job.child.resumed is False


def test_windows_parent_sync_returns_without_posix_fallback(tmp_path: Path, monkeypatch):
    class Function:
        def __init__(self, result):
            self.result = result
            self.argtypes = None
            self.restype = None

        def __call__(self, *_args):
            return self.result

    class Kernel32:
        def __init__(self):
            self.CreateFileW = Function(ctypes.c_void_p(7))
            self.FlushFileBuffers = Function(1)
            self.CloseHandle = Function(1)

    monkeypatch.setattr(supervisor_v6.os, "name", "nt")
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: Kernel32(), raising=False)
    monkeypatch.setattr(supervisor_v6.os, "open", lambda *_args, **_kwargs: pytest.fail("POSIX fallback"))

    supervisor_v6._fsync_parent_directory(tmp_path)


def test_launch_receipt_rejects_a_change_after_its_exact_bytes_are_read(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.loop167_phase_b.supervisor_v6._fsync_parent_directory", lambda _parent: None)
    job = FakeJob(config.launch_receipt)
    run_supervised_v6(
        config,
        job_factory=lambda **_kwargs: job,
        supervisor_identity_provider=_supervisor_identity,
    )
    monkeypatch.setattr(supervisor_v6, "sha256_file", lambda _path: "0" * 64)

    with pytest.raises(SupervisorV6Error, match="changed during validation"):
        validate_launch_receipt_v6(
            tmp_path,
            config.launch_receipt,
            mode="preflight",
            expected_bindings=config.static_bindings,
        )


def test_supervisor_terminates_and_closes_the_job_after_child_timeout(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.loop167_phase_b.supervisor_v6._fsync_parent_directory", lambda _parent: None)
    job = FakeJob(config.launch_receipt)
    job.child = TimeoutChild(config.launch_receipt)

    with pytest.raises(SupervisorV6Error, match="exceeded"):
        run_supervised_v6(
            config,
            job_factory=lambda **_kwargs: job,
            supervisor_identity_provider=_supervisor_identity,
        )

    failure = json.loads(config.failure_receipt.read_text(encoding="utf-8"))
    assert job.terminated is True
    assert job.child.resumed is True
    assert job.child.closed is True
    assert job.closed is True
    assert failure["stage"] == "post_resume"
    assert failure["supervisor_raw_open_attempts"] == 0


def test_supervisor_records_nonzero_exit_and_closes_the_job(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.loop167_phase_b.supervisor_v6._fsync_parent_directory", lambda _parent: None)
    job = FakeJob(config.launch_receipt)
    job.child = NonzeroChild(config.launch_receipt)

    with pytest.raises(SupervisorV6Error, match="nonzero: 7"):
        run_supervised_v6(
            config,
            job_factory=lambda **_kwargs: job,
            supervisor_identity_provider=_supervisor_identity,
        )

    exit_receipt = json.loads(config.exit_receipt.read_text(encoding="utf-8"))
    failure = json.loads(config.failure_receipt.read_text(encoding="utf-8"))
    assert exit_receipt["controller_returncode"] == 7
    assert failure["stage"] == "post_resume"
    assert job.child.closed is True
    assert job.closed is True


def test_supervisor_terminates_a_suspended_child_when_pre_resume_audit_fails(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.loop167_phase_b.supervisor_v6._fsync_parent_directory", lambda _parent: None)
    job = InvalidAuditJob(config.launch_receipt)

    with pytest.raises(SupervisorV6Error, match="assignment audit drifted"):
        run_supervised_v6(
            config,
            job_factory=lambda **_kwargs: job,
            supervisor_identity_provider=_supervisor_identity,
        )

    failure = json.loads(config.failure_receipt.read_text(encoding="utf-8"))
    assert job.child.resumed is False
    assert job.child.terminated is True
    assert job.child.closed is True
    assert job.closed is True
    assert failure["stage"] == "pre_resume"
