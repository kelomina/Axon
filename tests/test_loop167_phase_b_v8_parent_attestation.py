from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.loop167_phase_b.execution_contract_v8 as contract_v8
from src.loop167_phase_b.contracts import PhaseBContractError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v8_parent_attestation_binds_the_v7_control_plane_and_absence_surface() -> None:
    payload = contract_v8.build_parent_v7_prelease_attestation_payload_v8(PROJECT_ROOT)

    assert payload["v7_launch_receipt_absent"] is True
    assert payload["v7_child_attestation_absent"] is True
    assert payload["v7_lease_absent"] is True
    assert payload["v7_data_outputs_absent"] is True
    assert payload["v7_prelaunch_rejection"] == "resource_guard_age_exceeded_before_supervisor_launch"
    assert payload["v7_guard_maximum_age_seconds"] == 300
    assert payload["parent_v7_execution_contract"]["path"] == contract_v8.PARENT_V7_EXECUTION_CONTRACT_RELATIVE_PATH


def test_v8_parent_attestation_rejects_a_parent_contract_with_a_different_output_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "output_catalog": [{"path": path} for path in contract_v8.PARENT_V7_OUTPUT_PATHS[:-1]],
        "lease": dict(contract_v8.PARENT_V7_EXPECTED_LEASE),
        "canonical_controller_execute_argv": [
            contract_v8.VNEV_PYTHON_RELATIVE_PATH,
            "-I",
            "scripts/run_loop167_phase_b_controller_v7.py",
            "--execute",
        ],
    }

    with pytest.raises(PhaseBContractError, match="output catalog drifted"):
        contract_v8._validate_parent_v7_execution_surface(payload)


def test_v8_parent_attestation_rejects_any_change_to_the_stale_guard_fact() -> None:
    guard_path = PROJECT_ROOT / contract_v8.PARENT_V7_RESOURCE_GUARD_RELATIVE_PATH
    authorization_path = PROJECT_ROOT / contract_v8.PARENT_V7_RUN_AUTHORIZATION_RELATIVE_PATH
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    guard["created_at_utc"] = "2026-07-14T10:28:17Z"

    with pytest.raises(PhaseBContractError, match="resource guard facts drifted"):
        contract_v8._validate_parent_v7_stale_guard_surface(
            guard,
            authorization,
            resource_guard_binding=contract_v8._binding(PROJECT_ROOT, contract_v8.PARENT_V7_RESOURCE_GUARD_RELATIVE_PATH),
        )
