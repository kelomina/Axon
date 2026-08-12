from __future__ import annotations

from pathlib import Path

import pytest

import src.loop167_phase_b.execution_contract_v7 as contract_v7
from src.loop167_phase_b.contracts import PhaseBContractError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v7_parent_attestation_binds_the_v6_control_plane_and_absence_surface() -> None:
    payload = contract_v7.build_parent_v6_prelease_attestation_payload_v7(PROJECT_ROOT)

    assert payload["v6_launch_receipt_absent"] is True
    assert payload["v6_child_attestation_absent"] is True
    assert payload["v6_lease_absent"] is True
    assert payload["v6_data_outputs_absent"] is True
    assert payload["parent_v6_execution_contract"]["path"] == contract_v7.PARENT_V6_EXECUTION_CONTRACT_RELATIVE_PATH


def test_v7_parent_attestation_rejects_a_parent_contract_with_a_different_output_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "output_catalog": [{"path": path} for path in contract_v7.PARENT_V6_OUTPUT_PATHS[:-1]],
        "lease": dict(contract_v7.PARENT_V6_EXPECTED_LEASE),
        "canonical_controller_execute_argv": [
            contract_v7.VNEV_PYTHON_RELATIVE_PATH,
            "-I",
            "scripts/run_loop167_phase_b_controller_v6.py",
            "--execute",
        ],
    }

    with pytest.raises(PhaseBContractError, match="output catalog drifted"):
        contract_v7._validate_parent_v6_execution_surface(payload)
