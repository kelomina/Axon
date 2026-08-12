from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR_ENTRY_PATH = PROJECT_ROOT / "scripts" / "run_loop167_phase_b_supervisor_v7.py"


def _load_supervisor_entry():
    specification = importlib.util.spec_from_file_location(
        "loop167_phase_b_supervisor_v7_entry_test",
        SUPERVISOR_ENTRY_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_v7_supervisor_entry_passes_sealed_relative_controller_argv(monkeypatch) -> None:
    supervisor_entry = _load_supervisor_entry()
    binding = {"path": "sealed.json", "sha256": "a" * 64}
    static_receipt = SimpleNamespace(
        source_closure_binding=binding,
        execution_contract_binding=binding,
        runtime_lock_binding=binding,
        controller_binding=binding,
        supervisor_binding=binding,
        loop166_windows_job_binding=binding,
        loop166_windows_process_lineage_binding=binding,
    )
    authorization = SimpleNamespace(
        source_closure_binding=binding,
        execution_contract_binding=binding,
        runtime_lock_binding=binding,
        controller_binding=binding,
        supervisor_binding=binding,
        loop166_windows_job_binding=binding,
        loop166_windows_process_lineage_binding=binding,
        output_paths={
            "supervisor_launch_receipt": PROJECT_ROOT / "reports" / "synthetic-launch.json",
            "supervisor_exit_receipt": PROJECT_ROOT / "reports" / "synthetic-exit.json",
            "supervisor_failure_receipt": PROJECT_ROOT / "reports" / "synthetic-failure.json",
        },
    )
    captured: dict[str, object] = {}

    def fake_run_supervised(config):
        captured["config"] = config
        return SimpleNamespace(
            returncode=0,
            launch_receipt_sha256="b" * 64,
            exit_receipt_sha256="c" * 64,
        )

    monkeypatch.setattr(supervisor_entry, "_static_preflight", lambda _mode: static_receipt)
    monkeypatch.setattr(
        supervisor_entry,
        "validate_execution_authorization_v7",
        lambda *_args, **_kwargs: authorization,
    )
    monkeypatch.setattr(
        supervisor_entry,
        "verify_execution_contract_v7",
        lambda *_args, **_kwargs: SimpleNamespace(
            resource_contract={
                "maximum_training_peak_rss_bytes": 8192,
                "maximum_total_wall_seconds": 30,
            }
        ),
    )
    monkeypatch.setattr(
        supervisor_entry,
        "run_supervised_v7",
        fake_run_supervised,
    )

    supervisor_entry.run_execute()

    command = captured["config"].command
    assert command[2] == supervisor_entry.CONTROLLER_RELATIVE_PATH
    assert not Path(command[2]).is_absolute()
    assert ".." not in Path(command[2]).parts
