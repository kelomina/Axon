from __future__ import annotations

from pathlib import Path

import pytest

from src.loop167_phase_b.contracts import PhaseBContractError
from src.loop167_phase_b.invocation_v5 import (
    CONTROLLER_V5_RELATIVE_PATH,
    THREAD_ENVIRONMENT_V5,
    VNEV_PYTHON_RELATIVE_PATH,
    bootstrap_thread_environment_v5,
    canonical_argv_v5,
    canonical_controller_argv_v5,
    validate_current_runtime_invocation_v5,
    validate_launch_argv_v5,
)


def _synthetic_runtime_layout(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    executable = root / VNEV_PYTHON_RELATIVE_PATH
    controller = root / CONTROLLER_V5_RELATIVE_PATH
    executable.parent.mkdir(parents=True)
    controller.parent.mkdir(parents=True, exist_ok=True)
    executable.write_bytes(b"synthetic-python")
    controller.write_text("print('synthetic controller')\n", encoding="ascii")
    return root, executable


def test_canonical_v5_invocations_are_mode_specific() -> None:
    assert canonical_argv_v5("preflight") == (
        "vnev/Scripts/python.exe",
        "-I",
        "scripts/run_loop167_phase_b_controller_v5.py",
        "--preflight",
    )
    assert canonical_argv_v5("execute")[-1] == "--execute"
    validate_launch_argv_v5(canonical_argv_v5("execute"), mode="execute")
    with pytest.raises(PhaseBContractError):
        validate_launch_argv_v5(canonical_argv_v5("execute"), mode="preflight")


def test_thread_bootstrap_requires_no_prior_external_runtime_import() -> None:
    environment: dict[str, str] = {}

    assert bootstrap_thread_environment_v5(environment=environment, modules={}) == THREAD_ENVIRONMENT_V5
    assert environment == THREAD_ENVIRONMENT_V5
    with pytest.raises(PhaseBContractError):
        bootstrap_thread_environment_v5(environment={}, modules={"numpy": object()})


def test_current_runtime_invocation_checks_windows_vnev_isolation_cwd_and_controller_argv(tmp_path: Path) -> None:
    root, executable = _synthetic_runtime_layout(tmp_path)
    nested_cwd = root / "nested"
    nested_cwd.mkdir()
    alternate_executable = root / "vnev" / "Scripts" / "alternate.exe"
    alternate_executable.write_bytes(b"alternate-python")
    receipt = validate_current_runtime_invocation_v5(
        root,
        mode="preflight",
        executable=executable,
        cwd=root,
        isolated=True,
        os_name="nt",
        process_argv=canonical_controller_argv_v5("preflight"),
        environment=dict(THREAD_ENVIRONMENT_V5),
    )

    assert receipt.executable == executable
    assert receipt.cwd == root
    assert receipt.canonical_argv == canonical_argv_v5("preflight")
    with pytest.raises(PhaseBContractError):
        validate_current_runtime_invocation_v5(
            root,
            mode="preflight",
            executable=executable,
            cwd=root,
            isolated=False,
            os_name="nt",
            process_argv=canonical_controller_argv_v5("preflight"),
            environment=dict(THREAD_ENVIRONMENT_V5),
        )
    with pytest.raises(PhaseBContractError):
        validate_current_runtime_invocation_v5(
            root,
            mode="preflight",
            executable=alternate_executable,
            cwd=root,
            isolated=True,
            os_name="nt",
            process_argv=canonical_controller_argv_v5("preflight"),
            environment=dict(THREAD_ENVIRONMENT_V5),
        )
    with pytest.raises(PhaseBContractError):
        validate_current_runtime_invocation_v5(
            root,
            mode="preflight",
            executable=executable,
            cwd=nested_cwd,
            isolated=True,
            os_name="nt",
            process_argv=canonical_controller_argv_v5("preflight"),
            environment=dict(THREAD_ENVIRONMENT_V5),
        )
    with pytest.raises(PhaseBContractError):
        validate_current_runtime_invocation_v5(
            root,
            mode="execute",
            executable=executable,
            cwd=root,
            isolated=True,
            os_name="nt",
            process_argv=canonical_controller_argv_v5("preflight"),
            environment=dict(THREAD_ENVIRONMENT_V5),
        )
