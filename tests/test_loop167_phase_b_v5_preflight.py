from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.loop167_phase_b.contracts import PhaseBContractError, canonical_json_bytes
from src.loop167_phase_b.execution_contract_v5 import (
    EXECUTION_CONTRACT_RELATIVE_PATH,
    PARENT_V4_PRELEASE_ATTESTATION_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    build_execution_contract_payload_v5,
    build_parent_v4_prelease_attestation_payload_v5,
)
from src.loop167_phase_b.invocation_v5 import (
    THREAD_ENVIRONMENT_V5,
    canonical_argv_hashes_v5,
    canonical_argv_v5,
)
from src.loop167_phase_b.preflight_v5 import (
    EXPECTED_BLOCKERS,
    EXPECTED_DYNAMIC_GATES,
    SOURCE_CLOSURE_V4_RELATIVE_PATH,
    V5_SCOPE,
    validate_static_preflight_v5,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = "manifests/roadmap_9997/loop167_ember_v3_novel_delta"


def _write_canonical(path: Path, payload: dict) -> dict[str, str]:
    content = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": path.as_posix(), "sha256": hashlib.sha256(content).hexdigest()}


def _copy_bound_file(root: Path, binding: dict[str, str]) -> dict[str, str]:
    source = PROJECT_ROOT / binding["path"]
    destination = root / binding["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    return {"path": binding["path"], "sha256": binding["sha256"]}


def _fixture(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    root.mkdir()
    protocol_path = PROJECT_ROOT / ARTIFACT_DIRECTORY / "phase_b_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="ascii"))
    protocol_binding = _copy_bound_file(
        root,
        {"path": f"{ARTIFACT_DIRECTORY}/phase_b_protocol.json", "sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()},
    )
    for binding in protocol["phase_a_bindings"].values():
        _copy_bound_file(root, binding)
    addendum_path = PROJECT_ROOT / ARTIFACT_DIRECTORY / "phase_b_protocol_addendum.json"
    _copy_bound_file(
        root,
        {"path": f"{ARTIFACT_DIRECTORY}/phase_b_protocol_addendum.json", "sha256": hashlib.sha256(addendum_path.read_bytes()).hexdigest()},
    )
    parent_v4_artifacts = (
        "phase_b_execution_contract_v4.json",
        "phase_b_runtime_lock_v4.json",
        "phase_b_source_closure_v4.json",
        "phase_b_resource_guard_v4.json",
        "phase_b_run_authorization.json",
    )
    for filename in parent_v4_artifacts:
        path = PROJECT_ROOT / ARTIFACT_DIRECTORY / filename
        _copy_bound_file(
            root,
            {
                "path": f"{ARTIFACT_DIRECTORY}/{filename}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        )
    prior_path = PROJECT_ROOT / SOURCE_CLOSURE_V4_RELATIVE_PATH
    _copy_bound_file(
        root,
        {"path": SOURCE_CLOSURE_V4_RELATIVE_PATH, "sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest()},
    )
    parent_closure = json.loads(prior_path.read_text(encoding="ascii"))
    for source_binding in parent_closure["source_files"]:
        _copy_bound_file(root, source_binding)

    controller_path = root / "scripts/run_loop167_phase_b_controller_v5.py"
    controller_path.parent.mkdir(parents=True, exist_ok=True)
    controller_path.write_text("print('controller')\n", encoding="ascii")
    controller_binding = {
        "path": "scripts/run_loop167_phase_b_controller_v5.py",
        "sha256": hashlib.sha256(controller_path.read_bytes()).hexdigest(),
    }
    for relative_path in ("src/loop167_phase_b/preflight_v5.py",):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic static source\n", encoding="ascii")

    parent_attestation_path = root / PARENT_V4_PRELEASE_ATTESTATION_RELATIVE_PATH
    parent_attestation_binding = _write_canonical(
        parent_attestation_path,
        build_parent_v4_prelease_attestation_payload_v5(root),
    )
    parent_attestation_binding["path"] = PARENT_V4_PRELEASE_ATTESTATION_RELATIVE_PATH

    execution_contract_path = root / EXECUTION_CONTRACT_RELATIVE_PATH
    execution_contract_binding = _write_canonical(
        execution_contract_path,
        build_execution_contract_payload_v5(
            root,
            protocol_binding=protocol_binding,
            parent_v4_prelease_attestation_binding=parent_attestation_binding,
        ),
    )
    execution_contract_binding["path"] = EXECUTION_CONTRACT_RELATIVE_PATH

    runtime_lock_path = root / RUNTIME_LOCK_RELATIVE_PATH
    runtime_lock_payload = {
        "schema": "axon_loop167_phase_b_runtime_lock_v5",
        "loop_id": "loop167_ember_v3_novel_delta",
        "runtime_platform": "windows",
        "cwd_contract": "project_root_without_symlink_or_reparse",
        "project_root_no_symlink_or_reparse_required": True,
        "python": {
            "relative_path": "vnev/Scripts/python.exe",
            "sha256": "0" * 64,
            "implementation": "CPython",
            "version": "synthetic",
        },
        "packages": [
            {
                "distribution": distribution,
                "module": module,
                "relative_path": f"vnev/Lib/site-packages/{module}.py",
                "sha256": "0" * 64,
                "version": "synthetic",
            }
            for distribution, module in (
                ("numpy", "numpy"),
                ("scipy", "scipy"),
                ("scikit-learn", "sklearn"),
                ("pefile", "pefile"),
                ("threadpoolctl", "threadpoolctl"),
            )
        ],
        "controller": controller_binding,
        "execution_contract": execution_contract_binding,
        "canonical_argv": {
            "preflight": list(canonical_argv_v5("preflight")),
            "execute": list(canonical_argv_v5("execute")),
        },
        "canonical_argv_sha256": canonical_argv_hashes_v5(),
        "thread_environment": dict(THREAD_ENVIRONMENT_V5),
        "thread_environment_bootstrap_before_external_imports_required": True,
        "isolated_python_required": True,
        "network_fetch_allowed": False,
        "dependency_install_allowed": False,
    }
    runtime_lock_binding = _write_canonical(runtime_lock_path, runtime_lock_payload)
    runtime_lock_binding["path"] = RUNTIME_LOCK_RELATIVE_PATH

    source_files = [
        controller_binding,
        {
            "path": "src/loop167_phase_b/__init__.py",
            "sha256": hashlib.sha256((root / "src/loop167_phase_b/__init__.py").read_bytes()).hexdigest(),
        },
        {
            "path": "src/loop167_phase_b/preflight_v5.py",
            "sha256": hashlib.sha256((root / "src/loop167_phase_b/preflight_v5.py").read_bytes()).hexdigest(),
        },
    ]
    closure_payload = {
        "schema": "axon_loop167_phase_b_source_closure_v5",
        "loop_id": "loop167_ember_v3_novel_delta",
        "scope": V5_SCOPE,
        "supersedes_source_closure_v4": {
            "path": SOURCE_CLOSURE_V4_RELATIVE_PATH,
            "sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        },
        "parent_v4_prelease_attestation": parent_attestation_binding,
        "phase_a_bindings": protocol["phase_a_bindings"],
        "phase_b_protocol": protocol_binding,
        "phase_b_protocol_addendum": {
            "path": f"{ARTIFACT_DIRECTORY}/phase_b_protocol_addendum.json",
            "sha256": hashlib.sha256(addendum_path.read_bytes()).hexdigest(),
        },
        "phase_b_execution_contract": execution_contract_binding,
        "runtime_lock_v5": runtime_lock_binding,
        "source_files": source_files,
        "static_preflight_ready": True,
        "phase_b_raw_execution_ready": False,
        "dynamic_execution_gates": dict(EXPECTED_DYNAMIC_GATES),
        "remaining_execution_blockers": list(EXPECTED_BLOCKERS),
    }
    closure_path = root / SOURCE_CLOSURE_RELATIVE_PATH
    closure_binding = _write_canonical(closure_path, closure_payload)
    closure_binding["path"] = SOURCE_CLOSURE_RELATIVE_PATH
    return closure_binding, controller_binding


def test_v5_static_preflight_validates_only_static_bound_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "project"
    closure_binding, controller_binding = _fixture(root)

    receipt = validate_static_preflight_v5(
        root,
        source_closure_binding=closure_binding,
        controller_binding=controller_binding,
        canonical_preflight_argv=canonical_argv_v5("preflight"),
    )

    assert receipt.raw_open_attempts == 0
    assert receipt.source_closure_binding == closure_binding
    assert receipt.controller_binding == controller_binding


def test_v5_static_preflight_rejects_a_closure_that_claims_raw_readiness(tmp_path: Path) -> None:
    root = tmp_path / "project"
    closure_binding, controller_binding = _fixture(root)
    closure_path = root / closure_binding["path"]
    closure = json.loads(closure_path.read_text(encoding="ascii"))
    closure["phase_b_raw_execution_ready"] = True
    closure_binding = _write_canonical(closure_path, closure)
    closure_binding["path"] = SOURCE_CLOSURE_RELATIVE_PATH

    with pytest.raises(PhaseBContractError, match="execution state"):
        validate_static_preflight_v5(
            root,
            source_closure_binding=closure_binding,
            controller_binding=controller_binding,
            canonical_preflight_argv=canonical_argv_v5("preflight"),
        )
