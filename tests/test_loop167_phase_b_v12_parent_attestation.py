from __future__ import annotations

from pathlib import Path

import pytest

import src.loop167_phase_b.execution_contract_v12 as contract_v12
from src.loop167_phase_b.contracts import PhaseBContractError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v12_parent_attestation_binds_v10_launcher_controller_identity_mismatch() -> None:
    payload = contract_v12.build_parent_v10_prelease_attestation_payload_v12(PROJECT_ROOT)

    assert payload["v10_pre_resume_child_pid"] > 0
    assert payload["v10_failure_stage"] == (
        "post_resume_controller_creation_time_checked_against_launcher_receipt_before_child_attestation"
    )
    assert payload["v10_failure_cause"] == (
        "controller_current_process_creation_time_compared_to_launcher_receipt_creation_time"
    )
    assert payload["v10_launcher_creation_time_filetime"] > 0
    assert payload["v10_child_attestation_absent"] is True
    assert payload["v10_lease_absent"] is True
    assert payload["v10_data_outputs_absent"] is True
    assert payload["parent_v10_execution_contract"]["path"] == contract_v12.PARENT_V10_EXECUTION_CONTRACT_RELATIVE_PATH
    assert payload["v10_child_attestation_source"]["path"] == "src/loop167_phase_b/child_attestation_v10.py"


def test_v12_parent_attestation_rejects_a_parent_contract_with_a_different_output_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "output_catalog": [{"path": path} for path in contract_v12.PARENT_V10_OUTPUT_PATHS[:-1]],
        "lease": dict(contract_v12.PARENT_V10_EXPECTED_LEASE),
        "canonical_controller_execute_argv": [
            contract_v12.VNEV_PYTHON_RELATIVE_PATH,
            "-I",
            "scripts/run_loop167_phase_b_controller_v10.py",
            "--execute",
        ],
    }

    with pytest.raises(PhaseBContractError, match="output catalog drifted"):
        contract_v12._validate_parent_v10_execution_surface(payload)


def test_v12_parent_attestation_rejects_a_required_absence_violation(tmp_path: Path) -> None:
    forbidden_path = tmp_path / "forbidden.json"
    forbidden_path.parent.mkdir(parents=True, exist_ok=True)
    forbidden_path.write_text("{}", encoding="utf-8")
    with pytest.raises(PhaseBContractError, match="dynamic surface exists"):
        contract_v12._assert_absent(tmp_path, ("forbidden.json",), label="dynamic surface")
