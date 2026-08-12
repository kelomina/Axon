from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from src.loop167_phase_b.supervisor_v8 import (
    EXIT_RECEIPT_SCHEMA,
    LAUNCH_RECEIPT_SCHEMA,
    SupervisorConfigV8,
    run_supervised_v8,
    validate_launch_receipt_v8,
)

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects require Windows")


def test_windows_supervisor_real_job_path_uses_only_temporary_artifacts():
    system_root = os.environ.get("SystemRoot")
    if not system_root:
        pytest.skip("SystemRoot is unavailable")
    source_cmd = Path(system_root) / "System32" / "cmd.exe"
    if not source_cmd.is_file():
        pytest.skip("Windows cmd.exe is unavailable")

    temporary_root: Path | None = None
    with tempfile.TemporaryDirectory(prefix="axon-loop167-v8-job-containment-") as temporary_directory:
        temporary_root = Path(temporary_directory).resolve()
        child_executable = temporary_root / "runner" / "cmd.exe"
        child_executable.parent.mkdir(parents=True)
        shutil.copy2(source_cmd, child_executable)

        marker = temporary_root / "synthetic" / "containment_marker.txt"
        marker.parent.mkdir(parents=True)
        marker.write_text("temporary containment integration test", encoding="ascii")
        bindings = {
            "synthetic_marker": {
                "path": "synthetic/containment_marker.txt",
                "sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
            }
        }
        receipt_directory = temporary_root / "receipts"
        command = (str(child_executable), "/d", "/c", "exit", "0")
        config = SupervisorConfigV8(
            project_root=temporary_root,
            mode="preflight",
            command=command,
            launch_receipt=receipt_directory / "launch.json",
            exit_receipt=receipt_directory / "exit.json",
            failure_receipt=receipt_directory / "failure.json",
            memory_limit_bytes=128 * 1024 * 1024,
            timeout_seconds=30,
            static_bindings=bindings,
        )

        previous_working_directory = Path.cwd()
        try:
            os.chdir(temporary_root)
            # 复制后的 cmd.exe 只执行内建 exit，真实 Job 路径不会读取项目 raw 数据。
            result = run_supervised_v8(config)
            validated = validate_launch_receipt_v8(
                temporary_root,
                config.launch_receipt,
                mode="preflight",
                expected_bindings=bindings,
                expected_pid=int(result.job_audit["process_pid"]),
                expected_creation_time_filetime=int(
                    result.job_audit["process_creation_time_filetime"]
                ),
                expected_supervisor_pid=os.getpid(),
            )
            launch = validated.payload
            exit_receipt = json.loads(config.exit_receipt.read_text(encoding="utf-8"))

            assert result.returncode == 0
            assert validated.receipt_path == config.launch_receipt
            assert validated.canonical_sha256 == result.launch_receipt_sha256
            assert validated.canonical_sha256 == hashlib.sha256(
                config.launch_receipt.read_bytes()
            ).hexdigest()
            assert launch["schema"] == LAUNCH_RECEIPT_SCHEMA
            assert launch["status"] == "assigned_and_verified_before_child_resume"
            assert launch["raw_open_attempts"] == 0
            assert launch["command"] == list(command)
            assert launch["static_bindings"] == bindings
            assert launch["supervisor_identity"]["pid"] == os.getpid()

            pre_resume = launch["pre_resume_assignment"]
            assert pre_resume["creation_mode"] == "create_process_suspended_assign_verify_resume"
            assert pre_resume["assignment_api"] == "AssignProcessToJobObject"
            assert pre_resume["membership_api"] == "IsProcessInJob"
            assert pre_resume["job_limit_flags"] == 0x2100
            assert pre_resume["kill_on_job_close"] is True
            assert pre_resume["memory_limit_bytes"] == config.memory_limit_bytes
            assert pre_resume["assigned_before_resume"] is True
            assert pre_resume["process_resumed"] is False
            assert pre_resume["process_pid"] == result.job_audit["process_pid"]
            assert (
                pre_resume["process_creation_time_filetime"]
                == result.job_audit["process_creation_time_filetime"]
            )

            assert exit_receipt["schema"] == EXIT_RECEIPT_SCHEMA
            assert exit_receipt["status"] == "controller_zero_exit_with_contained_tree_empty"
            assert exit_receipt["controller_returncode"] == 0
            assert exit_receipt["active_processes_after"] == 0
            assert exit_receipt["supervisor_raw_open_attempts"] == 0
            assert exit_receipt["child_raw_access"] == "not_attested_by_supervisor"
            assert exit_receipt["launch_receipt"] == {
                "path": "receipts/launch.json",
                "sha256": result.launch_receipt_sha256,
            }
            assert not config.failure_receipt.exists()
            assert sorted(path.name for path in receipt_directory.iterdir()) == [
                "exit.json",
                "launch.json",
            ]
        finally:
            os.chdir(previous_working_directory)

    assert temporary_root is not None
    assert not temporary_root.exists()
