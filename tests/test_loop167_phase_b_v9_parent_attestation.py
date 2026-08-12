from __future__ import annotations

from pathlib import Path

import pytest

import src.loop167_phase_b.execution_contract_v9 as contract_v9
from src.loop167_phase_b.contracts import PhaseBContractError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v9_parent_attestation_binds_v8_static_control_plane_and_absence_surface() -> None:
    payload = contract_v9.build_parent_v8_prelease_attestation_payload_v9(PROJECT_ROOT)

    assert payload["v8_resource_guard_absent"] is True
    assert payload["v8_run_authorization_absent"] is True
    assert payload["v8_launch_receipt_absent"] is True
    assert payload["v8_child_attestation_absent"] is True
    assert payload["v8_lease_absent"] is True
    assert payload["v8_data_outputs_absent"] is True
    assert payload["parent_v8_execution_contract"]["path"] == contract_v9.PARENT_V8_EXECUTION_CONTRACT_RELATIVE_PATH


def test_v9_parent_attestation_rejects_a_parent_contract_with_a_different_output_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "output_catalog": [{"path": path} for path in contract_v9.PARENT_V8_OUTPUT_PATHS[:-1]],
        "lease": dict(contract_v9.PARENT_V8_EXPECTED_LEASE),
        "canonical_controller_execute_argv": [
            contract_v9.VNEV_PYTHON_RELATIVE_PATH,
            "-I",
            "scripts/run_loop167_phase_b_controller_v8.py",
            "--execute",
        ],
    }

    with pytest.raises(PhaseBContractError, match="output catalog drifted"):
        contract_v9._validate_parent_v8_execution_surface(payload)


def test_v9_parent_attestation_rejects_a_dynamic_v8_artifact(tmp_path: Path) -> None:
    forbidden_path = tmp_path / "forbidden.json"
    forbidden_path.parent.mkdir(parents=True, exist_ok=True)
    forbidden_path.write_text("{}", encoding="utf-8")
    with pytest.raises(PhaseBContractError, match="dynamic surface exists"):
        contract_v9._assert_absent(tmp_path, ("forbidden.json",), label="dynamic surface")
