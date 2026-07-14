from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

import src.loop167_phase_b.execution_authorization_v6 as execution_authorization_v6
from src.loop167_phase_b.contracts import PhaseBContractError, canonical_json_bytes, sha256_file
from src.loop167_phase_b.execution_contract_v6 import (
    CONTROLLER_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    FIXED_OUTPUT_CATALOG,
    LOOP_ID,
    LOOP166_WINDOWS_JOB_RELATIVE_PATH,
    LOOP166_WINDOWS_PROCESS_LINEAGE_RELATIVE_PATH,
    PHASE_B_PROTOCOL_RELATIVE_PATH,
    RESOURCE_GUARD_RELATIVE_PATH,
    RUN_AUTHORIZATION_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    SUPERVISOR_RELATIVE_PATH,
)
from src.loop167_phase_b.resource_guard_v6 import RESOURCE_GUARD_SCHEMA


NOW = datetime(2026, 7, 14, 0, 0, tzinfo=UTC)
LAUNCH_ID = "a" * 64


def _write_bytes(root: Path, relative_path: str, content: bytes) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_json(root: Path, relative_path: str, payload: dict[str, object]) -> Path:
    return _write_bytes(root, relative_path, canonical_json_bytes(payload))


def _binding(root: Path, relative_path: str) -> dict[str, str]:
    return {"path": relative_path, "sha256": sha256_file(root / relative_path)}


def _output_paths(root: Path) -> dict[str, Path]:
    return {entry["name"]: root / entry["path"] for entry in FIXED_OUTPUT_CATALOG}


def _authorization_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "project"
    root.mkdir()
    protocol_path = _write_json(
        root,
        PHASE_B_PROTOCOL_RELATIVE_PATH,
        {
            "schema": "axon_loop167_phase_b_protocol_v1",
            "loop_id": LOOP_ID,
            "claim_scope": "local_train_only_structural_delta_diagnostic_not_model_quality_promotion_or_full_test",
        },
    )
    contract_path = _write_json(root, EXECUTION_CONTRACT_RELATIVE_PATH, {"schema": "synthetic_contract"})
    controller_path = _write_bytes(root, CONTROLLER_RELATIVE_PATH, b"print('synthetic controller')\n")
    supervisor_path = _write_bytes(root, SUPERVISOR_RELATIVE_PATH, b"print('synthetic supervisor')\n")
    _write_bytes(root, LOOP166_WINDOWS_JOB_RELATIVE_PATH, b"synthetic windows job proof\n")
    _write_bytes(root, LOOP166_WINDOWS_PROCESS_LINEAGE_RELATIVE_PATH, b"synthetic lineage proof\n")

    protocol_binding = _binding(root, PHASE_B_PROTOCOL_RELATIVE_PATH)
    contract_binding = _binding(root, EXECUTION_CONTRACT_RELATIVE_PATH)
    controller_binding = _binding(root, CONTROLLER_RELATIVE_PATH)
    supervisor_binding = _binding(root, SUPERVISOR_RELATIVE_PATH)
    _write_json(
        root,
        SOURCE_CLOSURE_RELATIVE_PATH,
        {"schema": "axon_loop167_phase_b_source_closure_v6", "loop_id": LOOP_ID},
    )
    source_closure_binding = _binding(root, SOURCE_CLOSURE_RELATIVE_PATH)
    _write_json(
        root,
        RUNTIME_LOCK_RELATIVE_PATH,
        {
            "schema": "axon_loop167_phase_b_runtime_lock_v6",
            "loop_id": LOOP_ID,
            "controller": controller_binding,
            "supervisor": supervisor_binding,
            "execution_contract": contract_binding,
        },
    )
    runtime_lock_binding = _binding(root, RUNTIME_LOCK_RELATIVE_PATH)
    _write_json(
        root,
        RESOURCE_GUARD_RELATIVE_PATH,
        {"schema": RESOURCE_GUARD_SCHEMA, "loop_id": LOOP_ID},
    )
    resource_guard_binding = _binding(root, RESOURCE_GUARD_RELATIVE_PATH)

    context: dict[str, object] = {
        "protocol_binding": protocol_binding,
        "contract_binding": contract_binding,
        "source_closure_binding": source_closure_binding,
        "runtime_lock_binding": runtime_lock_binding,
        "controller_binding": controller_binding,
        "supervisor_binding": supervisor_binding,
        "resource_guard_binding": resource_guard_binding,
        "protocol_path": protocol_path,
        "contract_path": contract_path,
        "controller_path": controller_path,
        "supervisor_path": supervisor_path,
    }

    def verify_contract(_root: Path, binding: object) -> SimpleNamespace:
        assert binding == context["contract_binding"]
        return SimpleNamespace(protocol_binding=MappingProxyType(dict(protocol_binding)))

    def verify_guard(_root: Path, binding: object, **_kwargs: object) -> SimpleNamespace:
        assert binding == context["resource_guard_binding"]
        return SimpleNamespace(guard_sha256=resource_guard_binding["sha256"])

    monkeypatch.setattr(execution_authorization_v6, "verify_execution_contract_v6", verify_contract)
    monkeypatch.setattr(execution_authorization_v6, "verify_resource_guard_v6", verify_guard)
    authorization_payload = execution_authorization_v6.build_execution_authorization_payload_v6(
        root,
        execution_contract_binding=contract_binding,
        source_closure_binding=source_closure_binding,
        runtime_lock_binding=runtime_lock_binding,
        controller_binding=controller_binding,
        supervisor_binding=supervisor_binding,
        resource_guard_binding=resource_guard_binding,
        created_at_utc="2026-07-14T00:00:00Z",
    )
    authorization_path = _write_json(root, RUN_AUTHORIZATION_RELATIVE_PATH, authorization_payload)
    context["authorization_path"] = authorization_path
    context["authorization_payload"] = authorization_payload
    return root, context


def _validate_prelaunch(root: Path, context: dict[str, object]) -> object:
    return execution_authorization_v6.validate_execution_authorization_v6(
        root,
        context["authorization_path"],
        now_utc=NOW,
        phase="prelaunch",
    )


def test_v6_authorization_prelaunch_and_attested_child_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, context = _authorization_fixture(tmp_path, monkeypatch)
    prelaunch = _validate_prelaunch(root, context)
    assert set(prelaunch.output_paths) == {entry["name"] for entry in FIXED_OUTPUT_CATALOG}

    outputs = _output_paths(root)
    outputs["supervisor_launch_receipt"].parent.mkdir(parents=True, exist_ok=True)
    outputs["supervisor_launch_receipt"].write_bytes(b"synthetic launch receipt")
    with pytest.raises(PhaseBContractError, match="output already exists"):
        _validate_prelaunch(root, context)

    with pytest.raises(PhaseBContractError, match="attested child requires"):
        execution_authorization_v6.validate_execution_authorization_v6(
            root,
            context["authorization_path"],
            now_utc=NOW,
            phase="attested_child",
            launch_id=LAUNCH_ID,
        )

    outputs["child_job_attestation"].write_bytes(b"synthetic child attestation")
    with pytest.raises(PhaseBContractError, match="requires launch id"):
        execution_authorization_v6.validate_execution_authorization_v6(
            root,
            context["authorization_path"],
            now_utc=NOW,
            phase="attested_child",
            launch_id=None,
        )

    observed: dict[str, object] = {}

    def validate_launch(_root: Path, path: Path, **kwargs: object) -> SimpleNamespace:
        observed["launch_path"] = path
        observed["launch_kwargs"] = kwargs
        return SimpleNamespace(payload={"launch_id": LAUNCH_ID})

    def verify_attestation(_root: Path, **kwargs: object) -> None:
        observed["attestation_kwargs"] = kwargs

    monkeypatch.setattr(execution_authorization_v6, "validate_launch_receipt_v6", validate_launch)
    monkeypatch.setattr(execution_authorization_v6, "verify_child_job_attestation_v6", verify_attestation)
    verified = execution_authorization_v6.validate_execution_authorization_v6(
        root,
        context["authorization_path"],
        now_utc=NOW,
        phase="attested_child",
        launch_id=LAUNCH_ID,
    )

    assert verified.authorization_path == context["authorization_path"]
    assert observed["launch_path"] == outputs["supervisor_launch_receipt"]
    assert observed["launch_kwargs"]["expected_launch_id"] == LAUNCH_ID
    assert observed["attestation_kwargs"]["expected_launch_id"] == LAUNCH_ID


def test_v6_authorization_rejects_guard_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, context = _authorization_fixture(tmp_path, monkeypatch)
    _write_json(
        root,
        RESOURCE_GUARD_RELATIVE_PATH,
        {"schema": RESOURCE_GUARD_SCHEMA, "loop_id": LOOP_ID, "drift": True},
    )

    with pytest.raises(PhaseBContractError, match="resource_guard hash mismatch"):
        _validate_prelaunch(root, context)


def test_v6_authorization_rejects_controller_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, context = _authorization_fixture(tmp_path, monkeypatch)
    controller_path = context["controller_path"]
    assert isinstance(controller_path, Path)
    controller_path.write_bytes(b"print('drifted controller')\n")

    with pytest.raises(PhaseBContractError, match="controller hash mismatch"):
        _validate_prelaunch(root, context)
