from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.loop167_phase_b.contracts import PhaseBContractError, canonical_json_bytes
from src.loop167_phase_b.execution_contract_v4 import build_execution_contract_payload_v4
from src.loop167_phase_b.invocation_v4 import (
    THREAD_ENVIRONMENT_V4,
    canonical_argv_hashes_v4,
    canonical_argv_v4,
)
from src.loop167_phase_b.preflight_v4 import (
    EXPECTED_BLOCKERS,
    EXPECTED_DYNAMIC_GATES,
    SOURCE_CLOSURE_V3_RELATIVE_PATH,
    V4_SCOPE,
    validate_static_preflight_v4,
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
    prior_path = PROJECT_ROOT / SOURCE_CLOSURE_V3_RELATIVE_PATH
    _copy_bound_file(
        root,
        {"path": SOURCE_CLOSURE_V3_RELATIVE_PATH, "sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest()},
    )

    controller_path = root / "scripts/run_loop167_phase_b_controller_v4.py"
    controller_path.parent.mkdir(parents=True, exist_ok=True)
    controller_path.write_text("print('controller')\n", encoding="ascii")
    controller_binding = {
        "path": "scripts/run_loop167_phase_b_controller_v4.py",
        "sha256": hashlib.sha256(controller_path.read_bytes()).hexdigest(),
    }
    for relative_path in (
        "src/loop167_phase_b/__init__.py",
        "src/loop167_phase_b/preflight_v4.py",
    ):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic static source\n", encoding="ascii")

    execution_contract_path = root / ARTIFACT_DIRECTORY / "phase_b_execution_contract_v4.json"
    execution_contract_binding = _write_canonical(
        execution_contract_path,
        build_execution_contract_payload_v4(root, protocol_binding=protocol_binding),
    )
    execution_contract_binding["path"] = f"{ARTIFACT_DIRECTORY}/phase_b_execution_contract_v4.json"

    runtime_lock_path = root / ARTIFACT_DIRECTORY / "phase_b_runtime_lock_v4.json"
    runtime_lock_payload = {
        "schema": "axon_loop167_phase_b_runtime_lock_v4",
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
            "preflight": list(canonical_argv_v4("preflight")),
            "execute": list(canonical_argv_v4("execute")),
        },
        "canonical_argv_sha256": canonical_argv_hashes_v4(),
        "thread_environment": dict(THREAD_ENVIRONMENT_V4),
        "thread_environment_bootstrap_before_external_imports_required": True,
        "isolated_python_required": True,
        "network_fetch_allowed": False,
        "dependency_install_allowed": False,
    }
    runtime_lock_binding = _write_canonical(runtime_lock_path, runtime_lock_payload)
    runtime_lock_binding["path"] = f"{ARTIFACT_DIRECTORY}/phase_b_runtime_lock_v4.json"

    source_files = [
        controller_binding,
        {
            "path": "src/loop167_phase_b/__init__.py",
            "sha256": hashlib.sha256((root / "src/loop167_phase_b/__init__.py").read_bytes()).hexdigest(),
        },
        {
            "path": "src/loop167_phase_b/preflight_v4.py",
            "sha256": hashlib.sha256((root / "src/loop167_phase_b/preflight_v4.py").read_bytes()).hexdigest(),
        },
    ]
    closure_payload = {
        "schema": "axon_loop167_phase_b_source_closure_v4",
        "loop_id": "loop167_ember_v3_novel_delta",
        "scope": V4_SCOPE,
        "supersedes_source_closure_v3": {
            "path": SOURCE_CLOSURE_V3_RELATIVE_PATH,
            "sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        },
        "phase_a_bindings": protocol["phase_a_bindings"],
        "phase_b_protocol": protocol_binding,
        "phase_b_protocol_addendum": {
            "path": f"{ARTIFACT_DIRECTORY}/phase_b_protocol_addendum.json",
            "sha256": hashlib.sha256(addendum_path.read_bytes()).hexdigest(),
        },
        "phase_b_execution_contract": execution_contract_binding,
        "runtime_lock_v4": runtime_lock_binding,
        "source_files": source_files,
        "static_preflight_ready": True,
        "phase_b_raw_execution_ready": False,
        "dynamic_execution_gates": dict(EXPECTED_DYNAMIC_GATES),
        "remaining_execution_blockers": list(EXPECTED_BLOCKERS),
    }
    closure_path = root / ARTIFACT_DIRECTORY / "phase_b_source_closure_v4.json"
    closure_binding = _write_canonical(closure_path, closure_payload)
    closure_binding["path"] = f"{ARTIFACT_DIRECTORY}/phase_b_source_closure_v4.json"
    return closure_binding, controller_binding


def test_v4_static_preflight_validates_only_static_bound_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "project"
    closure_binding, controller_binding = _fixture(root)

    receipt = validate_static_preflight_v4(
        root,
        source_closure_binding=closure_binding,
        controller_binding=controller_binding,
        canonical_preflight_argv=canonical_argv_v4("preflight"),
    )

    assert receipt.raw_open_attempts == 0
    assert receipt.source_closure_binding == closure_binding
    assert receipt.controller_binding == controller_binding


def test_v4_static_preflight_rejects_a_closure_that_claims_raw_readiness(tmp_path: Path) -> None:
    root = tmp_path / "project"
    closure_binding, controller_binding = _fixture(root)
    closure_path = root / closure_binding["path"]
    closure = json.loads(closure_path.read_text(encoding="ascii"))
    closure["phase_b_raw_execution_ready"] = True
    closure_binding = _write_canonical(closure_path, closure)
    closure_binding["path"] = f"{ARTIFACT_DIRECTORY}/phase_b_source_closure_v4.json"

    with pytest.raises(PhaseBContractError, match="execution state"):
        validate_static_preflight_v4(
            root,
            source_closure_binding=closure_binding,
            controller_binding=controller_binding,
            canonical_preflight_argv=canonical_argv_v4("preflight"),
        )
