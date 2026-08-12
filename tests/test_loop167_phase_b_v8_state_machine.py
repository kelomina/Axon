from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.loop167_phase_b.preflight_v8 as preflight_v8
from src.loop167_phase_b.contracts import PhaseBContractError, canonical_json_bytes
from src.loop167_phase_b.execution_contract_v8 import (
    CONTROLLER_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    EXPECTED_LEASE,
    FIXED_OUTPUT_CATALOG,
    LOOP_ID,
    PARENT_V7_PRELEASE_ATTESTATION_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    SUPERVISOR_RELATIVE_PATH,
    assert_attested_child_prelease_surface_v8,
    assert_leased_child_pre_raw_surface_v8,
    assert_output_catalog_is_fresh_v8,
)
from src.loop167_phase_b.preflight_v8 import (
    EXPECTED_BLOCKERS,
    EXPECTED_DYNAMIC_GATES,
    REQUIRED_SOURCE_PATHS,
    SOURCE_CLOSURE_SCHEMA,
    V8_SCOPE,
)


def _output_path(root: Path, name: str) -> Path:
    return root / next(entry["path"] for entry in FIXED_OUTPUT_CATALOG if entry["name"] == name)


def _write(root: Path, relative_path: str, content: bytes) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _binding(root: Path, relative_path: str) -> dict[str, str]:
    path = root / relative_path
    return {"path": relative_path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _preflight_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    source_paths = set(REQUIRED_SOURCE_PATHS) | {
        "src/loop167_phase_b/raw_worker.py",
        "src/loop167_phase_b/fit_worker.py",
    }
    for relative_path in source_paths:
        _write(root, relative_path, f"synthetic:{relative_path}\n".encode("ascii"))
    _write(root, PARENT_V7_PRELEASE_ATTESTATION_RELATIVE_PATH, canonical_json_bytes({"parent": "synthetic"}))
    _write(root, EXECUTION_CONTRACT_RELATIVE_PATH, canonical_json_bytes({"contract": "synthetic"}))
    controller_binding = _binding(root, CONTROLLER_RELATIVE_PATH)
    supervisor_binding = _binding(root, SUPERVISOR_RELATIVE_PATH)
    contract_binding = _binding(root, EXECUTION_CONTRACT_RELATIVE_PATH)
    _write(
        root,
        RUNTIME_LOCK_RELATIVE_PATH,
        canonical_json_bytes(
            {
                "schema": "axon_loop167_phase_b_runtime_lock_v8",
                "loop_id": LOOP_ID,
                "controller": controller_binding,
                "supervisor": supervisor_binding,
                "execution_contract": contract_binding,
            }
        ),
    )
    runtime_binding = _binding(root, RUNTIME_LOCK_RELATIVE_PATH)
    source_payload = {
        "schema": SOURCE_CLOSURE_SCHEMA,
        "loop_id": LOOP_ID,
        "scope": V8_SCOPE,
        "parent_v7_prelease_attestation": _binding(root, PARENT_V7_PRELEASE_ATTESTATION_RELATIVE_PATH),
        "phase_b_execution_contract": contract_binding,
        "runtime_lock": runtime_binding,
        "controller": controller_binding,
        "supervisor": supervisor_binding,
        "source_files": [_binding(root, path) for path in sorted(source_paths)],
        "static_preflight_ready": True,
        "phase_b_raw_execution_ready": False,
        "dynamic_execution_gates": dict(EXPECTED_DYNAMIC_GATES),
        "remaining_execution_blockers": list(EXPECTED_BLOCKERS),
    }
    _write(root, SOURCE_CLOSURE_RELATIVE_PATH, canonical_json_bytes(source_payload))
    launch = _output_path(root, "supervisor_launch_receipt")
    child = _output_path(root, "child_job_attestation")
    launch.parent.mkdir(parents=True)
    launch.write_bytes(canonical_json_bytes({"launch": "synthetic"}))
    child.write_bytes(canonical_json_bytes({"child": "synthetic"}))
    monkeypatch.setattr(preflight_v8, "verify_parent_v7_prelease_attestation_v8", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        preflight_v8,
        "verify_execution_contract_v8",
        lambda *_args, **_kwargs: SimpleNamespace(
            output_catalog=FIXED_OUTPUT_CATALOG,
            contract_sha256=contract_binding["sha256"],
        ),
    )
    return {
        "source_closure": _binding(root, SOURCE_CLOSURE_RELATIVE_PATH),
        "controller": controller_binding,
        "supervisor": supervisor_binding,
    }


def test_v7_prelaunch_attested_and_leased_pre_raw_surfaces_transition_without_raw_access(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()

    assert_output_catalog_is_fresh_v8(root)
    launch = _output_path(root, "supervisor_launch_receipt")
    child_attestation = _output_path(root, "child_job_attestation")
    launch.parent.mkdir(parents=True)
    launch.write_bytes(canonical_json_bytes({"launch": "synthetic"}))
    child_attestation.write_bytes(canonical_json_bytes({"attestation": "synthetic"}))

    assert_attested_child_prelease_surface_v8(root)
    with pytest.raises(PhaseBContractError, match="Project path is missing"):
        assert_leased_child_pre_raw_surface_v8(root)

    lease_marker = root / EXPECTED_LEASE["marker_path"]
    lease_marker.parent.mkdir(parents=True, exist_ok=True)
    lease_marker.write_bytes(canonical_json_bytes({"lease": "synthetic"}))

    assert_leased_child_pre_raw_surface_v8(root)
    with pytest.raises(PhaseBContractError, match="execution lease already exists"):
        assert_attested_child_prelease_surface_v8(root)


@pytest.mark.parametrize(
    "mutated_path",
    ("src/loop167_phase_b/raw_worker.py", "src/loop167_phase_b/fit_worker.py"),
)
def test_v7_post_lease_guard_rejects_data_plane_source_drift_before_any_raw_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_path: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    bindings = _preflight_fixture(root, monkeypatch)

    initial = preflight_v8.validate_static_preflight_v8(
        root,
        source_closure_binding=bindings["source_closure"],
        controller_binding=bindings["controller"],
        supervisor_binding=bindings["supervisor"],
        phase="attested_child",
    )
    assert initial.raw_open_attempts == 0
    lease_marker = root / EXPECTED_LEASE["marker_path"]
    lease_marker.parent.mkdir(parents=True, exist_ok=True)
    lease_marker.write_bytes(canonical_json_bytes({"lease": "synthetic"}))
    _write(root, mutated_path, b"tampered after initial preflight\n")

    with pytest.raises(PhaseBContractError, match="hash mismatch"):
        preflight_v8.validate_static_preflight_v8(
            root,
            source_closure_binding=bindings["source_closure"],
            controller_binding=bindings["controller"],
            supervisor_binding=bindings["supervisor"],
            phase="leased_child_pre_raw",
        )
