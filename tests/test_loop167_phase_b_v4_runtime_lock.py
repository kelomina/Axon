from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from src.loop167_phase_b import runtime_lock_v4
from src.loop167_phase_b.contracts import PhaseBContractError
from src.loop167_phase_b.invocation_v4 import (
    CONTROLLER_V4_RELATIVE_PATH,
    EXECUTION_CONTRACT_V4_RELATIVE_PATH,
    THREAD_ENVIRONMENT_V4,
    VNEV_PYTHON_RELATIVE_PATH,
)


def _write(path: Path, content: bytes) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": path.as_posix(), "sha256": hashlib.sha256(content).hexdigest()}


def _synthetic_bindings(root: Path) -> tuple[dict[str, str], dict[str, str], Path]:
    executable = root / VNEV_PYTHON_RELATIVE_PATH
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"synthetic-python")
    controller = root / CONTROLLER_V4_RELATIVE_PATH
    execution_contract = root / EXECUTION_CONTRACT_V4_RELATIVE_PATH
    controller_binding = _write(controller, b"print('controller')\n")
    execution_binding = _write(execution_contract, b"{}\n")
    controller_binding["path"] = controller.relative_to(root).as_posix()
    execution_binding["path"] = execution_contract.relative_to(root).as_posix()
    return controller_binding, execution_binding, executable


def test_runtime_lock_binds_controller_execution_contract_and_both_invocations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    controller_binding, execution_binding, executable = _synthetic_bindings(root)
    python_binding = {
        "relative_path": VNEV_PYTHON_RELATIVE_PATH,
        "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "implementation": "CPython",
        "version": "synthetic",
    }
    package_bindings = [
        {
            "distribution": "synthetic-package",
            "module": "synthetic_module",
            "relative_path": "vnev/Lib/site-packages/synthetic_module.py",
            "sha256": "0" * 64,
            "version": "1.0",
        }
    ]
    monkeypatch.setattr(runtime_lock_v4, "validate_runtime_envelope_v4", lambda root: (executable, root))
    monkeypatch.setattr(runtime_lock_v4, "_require_clean_external_import_state", lambda: None)
    monkeypatch.setattr(runtime_lock_v4, "_python_binding", lambda root: python_binding)
    monkeypatch.setattr(runtime_lock_v4, "_runtime_packages", lambda root: package_bindings)
    for name, value in THREAD_ENVIRONMENT_V4.items():
        monkeypatch.setenv(name, value)

    payload = runtime_lock_v4.build_runtime_lock_payload_v4(
        root,
        controller_binding=controller_binding,
        execution_contract_binding=execution_binding,
    )
    receipt = runtime_lock_v4.validate_runtime_lock_v4(
        root,
        payload,
        controller_binding=controller_binding,
        execution_contract_binding=execution_binding,
        mode="execute",
        verify_current_runtime=False,
    )

    assert payload["controller"] == controller_binding
    assert payload["execution_contract"] == execution_binding
    assert payload["canonical_argv"]["preflight"][-1] == "--preflight"
    assert payload["canonical_argv"]["execute"][-1] == "--execute"
    assert receipt.canonical_argv_sha256 == payload["canonical_argv_sha256"]["execute"]

    drifted = copy.deepcopy(payload)
    drifted["canonical_argv"]["execute"][-1] = "--wrong"
    with pytest.raises(PhaseBContractError):
        runtime_lock_v4.validate_runtime_lock_v4(
            root,
            drifted,
            controller_binding=controller_binding,
            execution_contract_binding=execution_binding,
            mode="execute",
            verify_current_runtime=False,
        )


def test_runtime_lock_rejects_an_execution_contract_at_an_unpinned_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    controller_binding, _, executable = _synthetic_bindings(root)
    wrong_contract = root / "manifests" / "other-contract.json"
    wrong_binding = _write(wrong_contract, b"{}\n")
    wrong_binding["path"] = wrong_contract.relative_to(root).as_posix()
    monkeypatch.setattr(runtime_lock_v4, "validate_runtime_envelope_v4", lambda root: (executable, root))
    monkeypatch.setattr(runtime_lock_v4, "_require_clean_external_import_state", lambda: None)

    with pytest.raises(PhaseBContractError):
        runtime_lock_v4.build_runtime_lock_payload_v4(
            root,
            controller_binding=controller_binding,
            execution_contract_binding=wrong_binding,
        )


def test_lock_builder_pins_threads_before_importing_project_runtime_modules() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_loop167_phase_b_runtime_lock_v4.py"
    source = script_path.read_text(encoding="utf-8")

    assert source.index("os.environ.update(THREAD_ENVIRONMENT)") < source.index(
        "from loop167_phase_b.contracts"
    )
