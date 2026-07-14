from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SUPERVISOR_PATH = (
    PROJECT_ROOT / "scripts" / "run_loop166_phase_b1_step4096_recovery_v2_supervisor.py"
)
POWERSHELL_PATH = (
    PROJECT_ROOT / "scripts" / "run_loop166_phase_b1_step4096_recovery_v2_detached.ps1"
)


def _load_supervisor():
    spec = importlib.util.spec_from_file_location("loop166_v2_supervisor", SUPERVISOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_config(module, root: Path):
    python = root / "vnev" / "Scripts" / "python.exe"
    controller = root / "scripts" / "controller.py"
    source = root / "src" / "source.py"
    for path, payload in (
        (python, b"synthetic-python"),
        (controller, b"print('synthetic')\n"),
        (source, b"SOURCE = True\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    output = root / "reports"
    output.mkdir()
    return module.SupervisorConfig(
        project_root=root,
        command=(str(python), "-u", str(controller), "--fixed"),
        launch_receipt=output / "launch.json",
        exit_receipt=output / "exit.json",
        stdout_log=output / "stdout.log",
        stderr_log=output / "stderr.log",
        source_bindings=(("controller", controller), ("source", source)),
        timeout_seconds=30,
        maximum_combined_log_bytes=1024 * 1024,
    )


def test_supervisor_writes_mutually_bound_receipts_before_resume(
    tmp_path: Path,
    monkeypatch,
):
    module = _load_supervisor()
    config = _synthetic_config(module, tmp_path)
    monkeypatch.chdir(tmp_path)
    observations = {}

    def fake_runner(command, **kwargs):
        process = SimpleNamespace(
            pid=4242,
            resumed=False,
            creation_time_filetime=987654321,
        )
        audit = {
            "creation_mode": "create_process_suspended_assign_verify_resume",
            "kill_on_job_close": True,
            "exact_limit_flags": 0x2000,
            "breakaway_allowed": False,
            "process_pid": process.pid,
            "process_creation_time_filetime": process.creation_time_filetime,
            "assigned_before_resume": True,
            "process_resumed": False,
        }
        kwargs["before_resume"](process, audit)
        observations["launch_exists_before_resume"] = config.launch_receipt.exists()
        observations["environment"] = dict(kwargs["env"])
        process.resumed = True
        kwargs["stdout"].write(b"synthetic stdout\n")
        kwargs["stderr"].write(b"synthetic stderr\n")
        return SimpleNamespace(
            args=tuple(command),
            returncode=0,
            stdout=None,
            stderr=None,
            job_audit={**audit, "process_resumed": True, "active_processes_after": 0},
        )

    monkeypatch.setattr(module, "run_subprocess_in_job", fake_runner)
    exit_payload = module.run_supervised(
        config,
        environment={
            "PATH": os.environ.get("PATH", ""),
            "AXON_B1_RECOVERY_V2_NONCE": "must-not-survive",
        },
    )
    launch, observed_exit = module.validate_receipt_pair(
        config.launch_receipt,
        config.exit_receipt,
        expected_command=config.command,
    )

    assert observations["launch_exists_before_resume"] is True
    assert observations["environment"]["PYTHONUNBUFFERED"] == "1"
    assert "AXON_B1_RECOVERY_V2_NONCE" not in observations["environment"]
    assert len(observations["environment"]["AXON_B1_RECOVERY_V2_SUPERVISOR_LAUNCH_ID"]) == 64
    assert (
        observations["environment"]["AXON_B1_RECOVERY_V2_SUPERVISOR_LAUNCH_ID"]
        == launch["launch_id"]
    )
    assert launch["python_unbuffered"] is True
    assert launch["controller_launcher_pid"] == 4242
    assert launch["controller_launcher_creation_time_filetime"] == 987654321
    assert exit_payload == observed_exit
    assert exit_payload["decision"] == ("supervisor_controller_zero_exit_and_closure_verified")
    assert exit_payload["pre_resume_assignment_audit"]["process_pid"] == 4242
    combined = config.launch_receipt.read_text() + config.exit_receipt.read_text()
    assert "must-not-survive" not in combined


def test_supervisor_timeout_receipt_is_fail_closed(tmp_path: Path, monkeypatch):
    module = _load_supervisor()
    config = _synthetic_config(module, tmp_path)
    monkeypatch.chdir(tmp_path)

    def fake_timeout(command, **kwargs):
        process = SimpleNamespace(pid=77, resumed=False, creation_time_filetime=88)
        audit = {
            "creation_mode": "create_process_suspended_assign_verify_resume",
            "kill_on_job_close": True,
            "exact_limit_flags": 0x2000,
            "breakaway_allowed": False,
            "process_pid": 77,
            "process_creation_time_filetime": 88,
            "assigned_before_resume": True,
            "process_resumed": False,
        }
        kwargs["before_resume"](process, audit)
        raise module.WindowsJobTimeoutError(
            command,
            config.timeout_seconds,
            termination={
                "tree_termination_confirmed": True,
                "active_processes_after": 0,
            },
        )

    monkeypatch.setattr(module, "run_subprocess_in_job", fake_timeout)
    result = module.run_supervised(config, environment={})

    assert result["decision"] == "supervisor_timeout_killed_job_tree_fail_closed"
    assert result["controller_returncode"] is None
    assert result["timeout_termination"]["tree_termination_confirmed"] is True
    module.validate_receipt_pair(config.launch_receipt, config.exit_receipt)


def test_supervisor_enforces_combined_log_cap(tmp_path: Path, monkeypatch):
    module = _load_supervisor()
    config = replace(
        _synthetic_config(module, tmp_path),
        maximum_combined_log_bytes=16,
    )
    monkeypatch.chdir(tmp_path)

    def fake_runner(command, **kwargs):
        process = SimpleNamespace(pid=90, resumed=False, creation_time_filetime=91)
        audit = {
            "exact_limit_flags": 0x2000,
            "breakaway_allowed": False,
            "process_pid": 90,
            "process_creation_time_filetime": 91,
            "assigned_before_resume": True,
            "process_resumed": False,
        }
        kwargs["before_resume"](process, audit)
        kwargs["stdout"].write(b"x" * 17)
        with pytest.raises(module.SupervisorError, match="log cap"):
            kwargs["monitor_callback"]()
        return SimpleNamespace(
            returncode=0,
            job_audit={**audit, "process_resumed": True, "active_processes_after": 0},
        )

    monkeypatch.setattr(module, "run_subprocess_in_job", fake_runner)
    result = module.run_supervised(config, environment={})

    assert result["decision"] == "supervisor_log_cap_exceeded_fail_closed"
    assert result["combined_log_bytes"] == 17
    assert result["logs_within_cap"] is False


def test_receipt_pair_rejects_tampered_launch(tmp_path: Path, monkeypatch):
    module = _load_supervisor()
    config = _synthetic_config(module, tmp_path)
    monkeypatch.chdir(tmp_path)

    def fake_runner(command, **kwargs):
        process = SimpleNamespace(pid=5, resumed=False, creation_time_filetime=6)
        audit = {
            "exact_limit_flags": 0x2000,
            "breakaway_allowed": False,
            "process_pid": 5,
            "process_creation_time_filetime": 6,
            "assigned_before_resume": True,
            "process_resumed": False,
        }
        kwargs["before_resume"](process, audit)
        return SimpleNamespace(
            args=tuple(command),
            returncode=0,
            stdout=None,
            stderr=None,
            job_audit=audit,
        )

    monkeypatch.setattr(module, "run_subprocess_in_job", fake_runner)
    module.run_supervised(config, environment={})
    launch = json.loads(config.launch_receipt.read_text(encoding="utf-8"))
    launch["timeout_seconds"] = 1
    config.launch_receipt.write_text(json.dumps(launch), encoding="utf-8")

    with pytest.raises(module.SupervisorError, match="not mutually bound"):
        module.validate_receipt_pair(config.launch_receipt, config.exit_receipt)


def test_powershell_launcher_is_fixed_and_scrubs_internal_environment():
    payload = POWERSHELL_PATH.read_text(encoding="utf-8")

    assert "param(" not in payload.casefold()
    assert "E:\\Project\\python\\Axon_v2.6Exp" in payload
    assert "vnev\\Scripts\\python.exe" in payload
    assert "PYTHONUNBUFFERED" in payload
    assert "AXON_B1_RECOVERY_V2_" in payload
    assert ".EnvironmentVariables.Remove" in payload
    assert ".WorkingDirectory = $ProjectRoot" in payload
    assert "$LaunchReceipt" in payload and "$ExitReceipt" in payload
    assert "NONCE" not in payload


@pytest.mark.skipif(platform.system().casefold() != "windows", reason="Windows-only")
def test_real_windows_supervisor_receipt_smoke():
    module = _load_supervisor()
    work_root = PROJECT_ROOT / "artifacts" / "loop166_supervisor_synthetic" / uuid.uuid4().hex
    work_root.mkdir(parents=True)
    try:
        source = work_root / "source.py"
        source.write_text("SYNTHETIC = True\n", encoding="ascii")
        config = module.SupervisorConfig(
            project_root=PROJECT_ROOT,
            command=(
                sys.executable,
                "-u",
                "-c",
                "import json,os,pathlib;"
                "from loop166.windows_job import audit_current_process_job_membership,audit_process_job_membership;"
                f"launch=json.loads(pathlib.Path({str(work_root / 'launch.json')!r}).read_text());"
                "print(json.dumps({'unbuffered':os.environ.get('PYTHONUNBUFFERED'),"
                "'unexpected_internal':any(k.startswith('AXON_B1_RECOVERY_V2_') and 'SUPERVISOR' not in k for k in os.environ),"
                "'current':audit_current_process_job_membership(),"
                "'launcher':audit_process_job_membership(launch['controller_launcher_pid'],"
                "launch['controller_launcher_creation_time_filetime'])}))",
            ),
            launch_receipt=work_root / "launch.json",
            exit_receipt=work_root / "exit.json",
            stdout_log=work_root / "stdout.log",
            stderr_log=work_root / "stderr.log",
            source_bindings=(("synthetic_source", source),),
            timeout_seconds=30,
            maximum_combined_log_bytes=1024 * 1024,
        )
        result = module.run_supervised(
            config,
            environment=dict(
                os.environ,
                AXON_B1_RECOVERY_V2_NONCE="must-not-survive",
                PYTHONPATH=str(SRC_DIR),
            ),
        )

        assert result["decision"] == ("supervisor_controller_zero_exit_and_closure_verified")
        child_audit = json.loads(config.stdout_log.read_text(encoding="utf-8"))
        assert child_audit["unbuffered"] == "1"
        assert child_audit["unexpected_internal"] is False
        assert child_audit["current"]["in_job"] is True
        assert child_audit["launcher"]["in_job"] is True
        assert child_audit["launcher"]["active"] is True
        assert child_audit["current"]["pid"] != child_audit["launcher"]["pid"]
        module.validate_receipt_pair(
            config.launch_receipt,
            config.exit_receipt,
            expected_command=config.command,
        )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


@pytest.mark.skipif(platform.system().casefold() != "windows", reason="Windows-only")
def test_real_windows_detached_launch_and_exit_receipt_smoke():
    module = _load_supervisor()
    work_root = PROJECT_ROOT / "artifacts" / "loop166_supervisor_synthetic" / uuid.uuid4().hex
    work_root.mkdir(parents=True)
    helper = work_root / "detached_probe.py"
    source = work_root / "source.py"
    source.write_text("SYNTHETIC = True\n", encoding="ascii")
    launch = work_root / "launch.json"
    exit_path = work_root / "exit.json"
    stdout = work_root / "stdout.log"
    stderr = work_root / "stderr.log"
    helper.write_text(
        "import importlib.util, pathlib, sys\n"
        f"path=pathlib.Path({str(SUPERVISOR_PATH)!r})\n"
        "spec=importlib.util.spec_from_file_location('detached_supervisor',path)\n"
        "module=importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name]=module\n"
        "spec.loader.exec_module(module)\n"
        "config=module.SupervisorConfig(\n"
        f" project_root=pathlib.Path({str(PROJECT_ROOT)!r}),\n"
        f" command=({sys.executable!r},'-u','-c','print(\"detached synthetic\")'),\n"
        f" launch_receipt=pathlib.Path({str(launch)!r}),\n"
        f" exit_receipt=pathlib.Path({str(exit_path)!r}),\n"
        f" stdout_log=pathlib.Path({str(stdout)!r}),\n"
        f" stderr_log=pathlib.Path({str(stderr)!r}),\n"
        f' source_bindings=(("helper",pathlib.Path({str(helper)!r})),("source",pathlib.Path({str(source)!r}))),\n'
        " timeout_seconds=30,maximum_combined_log_bytes=1048576)\n"
        "result=module.run_supervised(config)\n"
        "raise SystemExit(0 if result['decision']=='supervisor_controller_zero_exit_and_closure_verified' else 1)\n",
        encoding="ascii",
    )
    escaped_python = str(Path(sys.executable)).replace("'", "''")
    escaped_helper = str(helper).replace("'", "''")
    escaped_root = str(PROJECT_ROOT).replace("'", "''")
    escaped_launch = str(launch).replace("'", "''")
    powershell = (
        "$s=[System.Diagnostics.ProcessStartInfo]::new();"
        f"$s.FileName='{escaped_python}';"
        f"$s.Arguments='-u \"{escaped_helper}\"';"
        f"$s.WorkingDirectory='{escaped_root}';"
        "$s.UseShellExecute=$false;$s.CreateNoWindow=$true;"
        "$p=[System.Diagnostics.Process]::Start($s);"
        f"while(-not (Test-Path -LiteralPath '{escaped_launch}')){{Start-Sleep -Milliseconds 25}};"
        f"$r=Get-Content -LiteralPath '{escaped_launch}' -Raw|ConvertFrom-Json;"
        "[pscustomobject]@{launcher_pid=$p.Id;supervisor_pid=[int64]$r.supervisor_pid}|ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", powershell],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        launch_processes = json.loads(completed.stdout)
        supervisor_pid = int(launch_processes["supervisor_pid"])
        deadline = time.monotonic() + 30
        while not exit_path.exists():
            if time.monotonic() >= deadline:
                raise AssertionError("Detached synthetic supervisor did not write exit receipt")
            time.sleep(0.05)
        observed_launch, observed_exit = module.validate_receipt_pair(
            launch,
            exit_path,
            expected_command=(
                sys.executable,
                "-u",
                "-c",
                'print("detached synthetic")',
            ),
        )

        assert observed_launch["supervisor_pid"] == supervisor_pid
        assert int(launch_processes["launcher_pid"]) > 0
        assert observed_exit["decision"] == ("supervisor_controller_zero_exit_and_closure_verified")
        assert stdout.read_text(encoding="utf-8").strip() == "detached synthetic"
    finally:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                shutil.rmtree(work_root)
                break
            except OSError:
                time.sleep(0.05)
        else:
            shutil.rmtree(work_root, ignore_errors=True)
