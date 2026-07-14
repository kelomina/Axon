from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.loop167_phase_b.contracts import PhaseBContractError, canonical_json_bytes, sha256_file
from src.loop167_phase_b.execution_authorization_v5 import (
    build_execution_authorization_payload_v5,
    validate_execution_authorization_v5,
)
from src.loop167_phase_b.execution_contract_v5 import (
    CANONICAL_EXECUTE_ARGV,
    CONTROLLER_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    FIXED_OUTPUT_CATALOG,
    PARENT_ARTIFACT_DIRECTORY,
    PARENT_V4_PRELEASE_ATTESTATION_RELATIVE_PATH,
    PHASE_B_PROTOCOL_RELATIVE_PATH,
    RESOURCE_GUARD_RELATIVE_PATH,
    RUN_AUTHORIZATION_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    build_execution_contract_payload_v5,
    build_parent_v4_prelease_attestation_payload_v5,
)
from src.loop167_phase_b.invocation_v5 import canonical_argv_hashes_v5
from src.loop167_phase_b.resource_guard_v5 import (
    SystemResourceSnapshotV5,
    build_resource_guard_payload_v5,
)

NOW = datetime(2026, 7, 14, 0, 0, tzinfo=UTC)


def _write_bytes(root: Path, relative_path: str, content: bytes) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_json(root: Path, relative_path: str, payload: dict) -> Path:
    return _write_bytes(root, relative_path, canonical_json_bytes(payload))


def _binding(root: Path, relative_path: str) -> dict[str, str]:
    return {"path": relative_path, "sha256": sha256_file(root / relative_path)}


def _synthetic_v5_root(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "project"
    root.mkdir()
    repository_root = Path(__file__).resolve().parents[1]
    protocol_content = (
        repository_root / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta" / "phase_b_protocol.json"
    ).read_bytes()
    _write_bytes(root, PHASE_B_PROTOCOL_RELATIVE_PATH, protocol_content)
    protocol_binding = _binding(root, PHASE_B_PROTOCOL_RELATIVE_PATH)
    for filename in (
        "phase_b_execution_contract_v4.json",
        "phase_b_runtime_lock_v4.json",
        "phase_b_source_closure_v4.json",
        "phase_b_resource_guard_v4.json",
        "phase_b_run_authorization.json",
    ):
        source_path = repository_root / PARENT_ARTIFACT_DIRECTORY / filename
        _write_bytes(root, f"{PARENT_ARTIFACT_DIRECTORY}/{filename}", source_path.read_bytes())
    parent_closure = json.loads(
        (repository_root / PARENT_ARTIFACT_DIRECTORY / "phase_b_source_closure_v4.json").read_text(encoding="ascii")
    )
    for source_binding in parent_closure["source_files"]:
        source_path = repository_root / source_binding["path"]
        _write_bytes(root, source_binding["path"], source_path.read_bytes())
    parent_attestation_payload = build_parent_v4_prelease_attestation_payload_v5(root)
    _write_json(root, PARENT_V4_PRELEASE_ATTESTATION_RELATIVE_PATH, parent_attestation_payload)
    parent_attestation_binding = _binding(root, PARENT_V4_PRELEASE_ATTESTATION_RELATIVE_PATH)
    contract_payload = build_execution_contract_payload_v5(
        root,
        protocol_binding=protocol_binding,
        parent_v4_prelease_attestation_binding=parent_attestation_binding,
    )
    _write_json(root, EXECUTION_CONTRACT_RELATIVE_PATH, contract_payload)
    controller = _write_bytes(root, CONTROLLER_RELATIVE_PATH, b"print('synthetic controller')\n")
    controller_binding = {"path": CONTROLLER_RELATIVE_PATH, "sha256": sha256_file(controller)}
    contract_binding = _binding(root, EXECUTION_CONTRACT_RELATIVE_PATH)
    _write_json(
        root,
        SOURCE_CLOSURE_RELATIVE_PATH,
        {"schema": "axon_loop167_phase_b_source_closure_v5", "loop_id": "loop167_ember_v3_novel_delta"},
    )
    source_closure_binding = _binding(root, SOURCE_CLOSURE_RELATIVE_PATH)
    _write_json(
        root,
        RUNTIME_LOCK_RELATIVE_PATH,
        {
            "schema": "axon_loop167_phase_b_runtime_lock_v5",
            "loop_id": "loop167_ember_v3_novel_delta",
            "controller": controller_binding,
            "execution_contract": contract_binding,
            "canonical_argv": {
                "preflight": [
                    "vnev/Scripts/python.exe",
                    "-I",
                    CONTROLLER_RELATIVE_PATH,
                    "--preflight",
                ],
                "execute": list(CANONICAL_EXECUTE_ARGV),
            },
            "canonical_argv_sha256": canonical_argv_hashes_v5(),
            "isolated_python_required": True,
            "network_fetch_allowed": False,
            "dependency_install_allowed": False,
        },
    )
    runtime_lock_binding = _binding(root, RUNTIME_LOCK_RELATIVE_PATH)
    guard_payload = build_resource_guard_payload_v5(
        root,
        execution_contract_binding=contract_binding,
        source_closure_binding=source_closure_binding,
        runtime_lock_binding=runtime_lock_binding,
        canonical_execute_argv=CANONICAL_EXECUTE_ARGV,
        snapshot=SystemResourceSnapshotV5(
            total_memory_bytes=32 * 1024**3,
            available_memory_bytes=12 * 1024**3,
            cpu_count=4,
        ),
        created_at_utc="2026-07-14T00:00:00Z",
        job_object_probe=lambda _limit: (True, None),
    )
    assert guard_payload["guard_ready"] is True
    _write_json(root, RESOURCE_GUARD_RELATIVE_PATH, guard_payload)
    guard_binding = _binding(root, RESOURCE_GUARD_RELATIVE_PATH)
    authorization_payload = build_execution_authorization_payload_v5(
        root,
        execution_contract_binding=contract_binding,
        source_closure_binding=source_closure_binding,
        runtime_lock_binding=runtime_lock_binding,
        controller_binding=controller_binding,
        resource_guard_binding=guard_binding,
        created_at_utc="2026-07-14T00:00:00Z",
    )
    authorization_path = _write_json(root, RUN_AUTHORIZATION_RELATIVE_PATH, authorization_payload)
    return root, {"authorization_path": authorization_path, "payload": authorization_payload}


def test_v5_authorization_binds_all_inputs_and_the_fixed_output_catalog(tmp_path: Path) -> None:
    root, context = _synthetic_v5_root(tmp_path)

    verified = validate_execution_authorization_v5(root, context["authorization_path"], now_utc=NOW)

    assert verified.authorization_path == context["authorization_path"]
    assert verified.authorization_sha256 == hashlib.sha256(context["authorization_path"].read_bytes()).hexdigest()
    assert set(verified.output_paths) == {
        "feature_cache",
        "raw_progress_ledger",
        "fit_progress_ledger",
        "execution_receipt",
    }
    assert not verified.lease_marker_path.exists()


def test_v5_authorization_rejects_drifted_catalog_and_nonfixed_auth_path(tmp_path: Path) -> None:
    root, context = _synthetic_v5_root(tmp_path)
    wrong_path = root / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta" / "other.json"
    wrong_path.write_bytes(context["authorization_path"].read_bytes())
    with pytest.raises(PhaseBContractError, match="fixed Phase-A contract"):
        validate_execution_authorization_v5(root, wrong_path, now_utc=NOW)

    drifted = copy.deepcopy(context["payload"])
    drifted["output_catalog"][0]["path"] = "reports/roadmap_9997/loop167/not_allowed.npz"
    context["authorization_path"].write_bytes(canonical_json_bytes(drifted))
    with pytest.raises(PhaseBContractError, match="output catalog"):
        validate_execution_authorization_v5(root, context["authorization_path"], now_utc=NOW)


def test_v5_authorization_refuses_an_existing_sealed_output(tmp_path: Path) -> None:
    root, context = _synthetic_v5_root(tmp_path)
    output = root / FIXED_OUTPUT_CATALOG[0]["path"]
    output.parent.mkdir(parents=True)
    output.write_bytes(b"prior output")

    with pytest.raises(PhaseBContractError, match="sealed v5 output"):
        validate_execution_authorization_v5(root, context["authorization_path"], now_utc=NOW)


def test_v5_authorization_refuses_a_stale_resource_guard(tmp_path: Path) -> None:
    root, context = _synthetic_v5_root(tmp_path)

    with pytest.raises(PhaseBContractError, match="stale"):
        validate_execution_authorization_v5(
            root,
            context["authorization_path"],
            now_utc=NOW + timedelta(seconds=301),
        )


def test_v5_authorization_rechecks_every_parent_v4_source_hash(tmp_path: Path) -> None:
    root, context = _synthetic_v5_root(tmp_path)
    parent_raw_worker = root / "src/loop167_phase_b/raw_worker.py"
    parent_raw_worker.write_text("# tampered static source\n", encoding="ascii")

    with pytest.raises(PhaseBContractError, match="hash mismatch"):
        validate_execution_authorization_v5(root, context["authorization_path"], now_utc=NOW)


def test_resource_guard_builder_never_writes_the_fixed_guard_when_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_loop167_phase_b_resource_guard_v5.py"
    specification = importlib.util.spec_from_file_location("loop167_guard_builder_v5_test", script_path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(module, "PROJECT_ROOT", root)
    monkeypatch.setattr(
        module,
        "build_payload",
        lambda *, created_at_utc: {
            "decision": "fail_closed",
            "failures": ["available_memory_below_launch_floor"],
            "guard_ready": False,
        },
    )
    monkeypatch.setattr(sys, "argv", [str(script_path), "--write"])

    assert module.main() == 2
    assert not (root / RESOURCE_GUARD_RELATIVE_PATH).exists()
