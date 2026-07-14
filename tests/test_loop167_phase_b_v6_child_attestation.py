from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path

import pytest

import src.loop167_phase_b.child_attestation_v6 as child_attestation_v6
from src.loop167_phase_b.child_attestation_v6 import (
    _verify_live_process_identity,
    build_child_job_attestation_payload_v6,
)
from src.loop167_phase_b.contracts import PhaseBContractError
from src.loop167_phase_b.supervisor_v6 import ValidatedLaunchReceiptV6

LAUNCH_ID = "a" * 64
SUPERVISOR_PID = 701
SUPERVISOR_FILETIME = 702
CHILD_FILETIME = 703


def _bindings() -> dict[str, dict[str, str]]:
    return {
        "source_closure": {"path": "manifests/source.json", "sha256": "1" * 64},
        "execution_contract": {"path": "manifests/contract.json", "sha256": "2" * 64},
        "runtime_lock": {"path": "manifests/runtime.json", "sha256": "3" * 64},
        "controller": {"path": "scripts/controller.py", "sha256": "4" * 64},
        "supervisor": {"path": "scripts/supervisor.py", "sha256": "5" * 64},
        "loop166_windows_job": {"path": "src/loop166/windows_job.py", "sha256": "6" * 64},
        "loop166_windows_process_lineage": {
            "path": "src/loop166/windows_process_lineage.py",
            "sha256": "7" * 64,
        },
    }


def _validated_launch(root: Path, *, launcher_pid: int) -> ValidatedLaunchReceiptV6:
    receipt_path = root / "reports" / "launch.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(b"synthetic launch receipt")
    payload = {
        "launch_id": LAUNCH_ID,
        "command": [str(root / "vnev" / "Scripts" / "python.exe"), "-I", "controller.py", "--execute"],
        "supervisor_identity": {
            "pid": SUPERVISOR_PID,
            "creation_time_filetime": SUPERVISOR_FILETIME,
        },
        "pre_resume_assignment": {
            "process_pid": launcher_pid,
            "process_creation_time_filetime": CHILD_FILETIME,
            "assigned_before_resume": True,
        },
    }
    return ValidatedLaunchReceiptV6(
        receipt_path=receipt_path,
        payload=payload,
        canonical_sha256="d" * 64,
    )


def _install_common_attestation_mocks(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    launcher_pid: int,
    lineage: dict[str, object],
) -> dict[str, object]:
    validated = _validated_launch(root, launcher_pid=launcher_pid)
    observed: dict[str, object] = {}
    monkeypatch.setattr(child_attestation_v6, "validate_launch_receipt_v6", lambda *_args, **_kwargs: validated)
    monkeypatch.setattr(
        child_attestation_v6,
        "audit_current_process_job_membership",
        lambda _root, _binding, creation_time_filetime, *, expected_pid: {
            "pid": expected_pid,
            "creation_time_filetime": creation_time_filetime,
            "in_expected_job": True,
        },
    )
    monkeypatch.setattr(
        child_attestation_v6,
        "audit_process_job_membership",
        lambda _root, _binding, pid, creation_time_filetime: {
            "pid": pid,
            "creation_time_filetime": creation_time_filetime,
            "in_expected_job": True,
        },
    )

    def verify_supervisor(pid: int, creation_time_filetime: int) -> dict[str, object]:
        observed["supervisor"] = (pid, creation_time_filetime)
        return {
            "pid": pid,
            "creation_time_filetime": creation_time_filetime,
            "active": True,
            "verification_scope": "synthetic",
        }

    monkeypatch.setattr(child_attestation_v6, "_verify_live_process_identity", verify_supervisor)
    monkeypatch.setattr(child_attestation_v6, "validate_spawn_lineage", lambda *_args, **_kwargs: lineage)
    monkeypatch.setenv("AXON_LOOP167_V6_SUPERVISOR_PID", str(SUPERVISOR_PID))
    return observed


def test_child_attestation_binds_the_validator_exact_bytes_digest_and_direct_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_pid = os.getpid()
    observed = _install_common_attestation_mocks(
        monkeypatch,
        tmp_path,
        launcher_pid=child_pid,
        lineage={
            "mode": "direct_parent",
            "expected_parent_pid": SUPERVISOR_PID,
            "current_pid": child_pid,
            "direct_parent_pid": SUPERVISOR_PID,
            "redirector_pid": 0,
        },
    )

    payload = build_child_job_attestation_payload_v6(
        tmp_path,
        launch_receipt_path=tmp_path / "reports" / "launch.json",
        launch_id=LAUNCH_ID,
        expected_bindings=_bindings(),
    )

    assert payload["launch_receipt"]["sha256"] == "d" * 64
    assert payload["launch_receipt"]["sha256"] != hashlib.sha256(
        (tmp_path / "reports" / "launch.json").read_bytes()
    ).hexdigest()
    assert payload["lineage"]["mode"] == "direct_parent"
    assert payload["launcher_membership"]["pid"] == child_pid
    assert observed["supervisor"] == (SUPERVISOR_PID, SUPERVISOR_FILETIME)


def test_child_attestation_accepts_only_the_receipted_redirector_as_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redirector_pid = 991
    _install_common_attestation_mocks(
        monkeypatch,
        tmp_path,
        launcher_pid=redirector_pid,
        lineage={
            "mode": "windows_venv_redirector",
            "expected_parent_pid": SUPERVISOR_PID,
            "current_pid": os.getpid(),
            "direct_parent_pid": redirector_pid,
            "redirector_pid": redirector_pid,
        },
    )

    payload = build_child_job_attestation_payload_v6(
        tmp_path,
        launch_receipt_path=tmp_path / "reports" / "launch.json",
        launch_id=LAUNCH_ID,
        expected_bindings=_bindings(),
    )

    assert payload["lineage"]["mode"] == "windows_venv_redirector"
    assert payload["launcher_membership"]["pid"] == redirector_pid


def test_child_attestation_rejects_a_receipt_launcher_that_does_not_match_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_common_attestation_mocks(
        monkeypatch,
        tmp_path,
        launcher_pid=991,
        lineage={
            "mode": "direct_parent",
            "expected_parent_pid": SUPERVISOR_PID,
            "current_pid": os.getpid(),
            "direct_parent_pid": SUPERVISOR_PID,
            "redirector_pid": 0,
        },
    )

    with pytest.raises(PhaseBContractError, match="launcher identity differs"):
        build_child_job_attestation_payload_v6(
            tmp_path,
            launch_receipt_path=tmp_path / "reports" / "launch.json",
            launch_id=LAUNCH_ID,
            expected_bindings=_bindings(),
        )


def test_child_attestation_rejects_a_dead_or_unverifiable_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_common_attestation_mocks(
        monkeypatch,
        tmp_path,
        launcher_pid=os.getpid(),
        lineage={
            "mode": "direct_parent",
            "expected_parent_pid": SUPERVISOR_PID,
            "current_pid": os.getpid(),
            "direct_parent_pid": SUPERVISOR_PID,
            "redirector_pid": 0,
        },
    )
    monkeypatch.setattr(
        child_attestation_v6,
        "_verify_live_process_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PhaseBContractError("supervisor is dead")),
    )

    with pytest.raises(PhaseBContractError, match="launcher lineage and liveness"):
        build_child_job_attestation_payload_v6(
            tmp_path,
            launch_receipt_path=tmp_path / "reports" / "launch.json",
            launch_id=LAUNCH_ID,
            expected_bindings=_bindings(),
        )


class _Function:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


def _install_identity_kernel(monkeypatch: pytest.MonkeyPatch, *, wait_result: int, creation: int) -> None:
    class Kernel32:
        def __init__(self) -> None:
            self.OpenProcess = _Function(lambda *_args: ctypes.c_void_p(17))
            self.WaitForSingleObject = _Function(lambda *_args: wait_result)

            def get_process_times(_handle, creation_time, *_args):
                target = creation_time._obj
                target.dwLowDateTime = creation & 0xFFFFFFFF
                target.dwHighDateTime = creation >> 32
                return 1

            self.GetProcessTimes = _Function(get_process_times)
            self.CloseHandle = _Function(lambda *_args: 1)

    monkeypatch.setattr(child_attestation_v6.os, "name", "nt")
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: Kernel32(), raising=False)


def test_live_supervisor_identity_checks_pid_liveness_and_creation_time(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_identity_kernel(monkeypatch, wait_result=0x00000102, creation=SUPERVISOR_FILETIME)

    identity = _verify_live_process_identity(SUPERVISOR_PID, SUPERVISOR_FILETIME)

    assert identity["pid"] == SUPERVISOR_PID
    assert identity["active"] is True
    with pytest.raises(PhaseBContractError, match="creation time drifted"):
        _verify_live_process_identity(SUPERVISOR_PID, SUPERVISOR_FILETIME + 1)


def test_live_supervisor_identity_rejects_a_dead_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_identity_kernel(monkeypatch, wait_result=0, creation=SUPERVISOR_FILETIME)

    with pytest.raises(PhaseBContractError, match="no longer active"):
        _verify_live_process_identity(SUPERVISOR_PID, SUPERVISOR_FILETIME)
