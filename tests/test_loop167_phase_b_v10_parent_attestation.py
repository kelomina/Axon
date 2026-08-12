from __future__ import annotations

from pathlib import Path

import pytest

import src.loop167_phase_b.execution_contract_v10 as contract_v10
from src.loop167_phase_b.contracts import PhaseBContractError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v10_parent_attestation_binds_v9_postlaunch_failure_before_lineage() -> None:
    payload = contract_v10.build_parent_v9_prelease_attestation_payload_v10(PROJECT_ROOT)

    assert payload["v9_pre_resume_child_pid"] > 0
    assert payload["v9_failure_stage"] == "post_resume_pid_before_lineage_attestation"
    assert payload["v9_child_attestation_absent"] is True
    assert payload["v9_lease_absent"] is True
    assert payload["v9_data_outputs_absent"] is True
    assert payload["parent_v9_execution_contract"]["path"] == contract_v10.PARENT_V9_EXECUTION_CONTRACT_RELATIVE_PATH


def test_v10_parent_attestation_rejects_a_parent_contract_with_a_different_output_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "output_catalog": [{"path": path} for path in contract_v10.PARENT_V9_OUTPUT_PATHS[:-1]],
        "lease": dict(contract_v10.PARENT_V9_EXPECTED_LEASE),
        "canonical_controller_execute_argv": [
            contract_v10.VNEV_PYTHON_RELATIVE_PATH,
            "-I",
            "scripts/run_loop167_phase_b_controller_v9.py",
            "--execute",
        ],
    }

    with pytest.raises(PhaseBContractError, match="output catalog drifted"):
        contract_v10._validate_parent_v9_execution_surface(payload)


def test_v10_parent_attestation_rejects_a_required_absence_violation(tmp_path: Path) -> None:
    forbidden_path = tmp_path / "forbidden.json"
    forbidden_path.parent.mkdir(parents=True, exist_ok=True)
    forbidden_path.write_text("{}", encoding="utf-8")
    with pytest.raises(PhaseBContractError, match="dynamic surface exists"):
        contract_v10._assert_absent(tmp_path, ("forbidden.json",), label="dynamic surface")
