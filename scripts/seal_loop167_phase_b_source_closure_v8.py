#!/usr/bin/env python3
"""Seal the complete v8 source closure for the Loop167 Phase-B execution route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loop167_phase_b.contracts import (  # noqa: E402
    canonical_json_bytes,
    require_canonical_json,
    sha256_file,
)
from loop167_phase_b.execution_contract_v8 import (  # noqa: E402
    CONTROLLER_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    LOOP_ID,
    PARENT_V7_PRELEASE_ATTESTATION_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    SUPERVISOR_RELATIVE_PATH,
    ensure_v8_static_artifact_parent,
    verify_execution_contract_v8,
    verify_parent_v7_prelease_attestation_v8,
)
from loop167_phase_b.path_safety_v4 import safe_project_path, safe_project_root  # noqa: E402
from loop167_phase_b.preflight_v8 import (  # noqa: E402
    EXPECTED_BLOCKERS,
    EXPECTED_DYNAMIC_GATES,
    REQUIRED_SOURCE_PATHS,
    SOURCE_CLOSURE_SCHEMA,
    V8_SCOPE,
)

V8_SOURCE_PATHS = (
    "src/loop167_phase_b/__init__.py",
    "src/loop167_phase_b/contracts.py",
    "src/loop167_phase_b/path_safety_v4.py",
    "src/loop167_phase_b/execution_contract_v4.py",
    "src/loop167_phase_b/execution_contract_v5.py",
    "src/loop167_phase_b/execution_contract_v8.py",
    "src/loop167_phase_b/invocation_v8.py",
    "src/loop167_phase_b/runtime_lock_v8.py",
    "src/loop167_phase_b/preflight_v8.py",
    "src/loop167_phase_b/resource_guard_v8.py",
    "src/loop167_phase_b/execution_authorization_v8.py",
    "src/loop167_phase_b/launch_authorization_v8.py",
    "src/loop167_phase_b/supervisor_v8.py",
    "src/loop167_phase_b/child_attestation_v8.py",
    "src/loop167_phase_b/lease_v8.py",
    "src/loop167_phase_b/loop166_v8_bridge.py",
    "src/loop167_phase_b/windows_job_v4.py",
    "src/loop167_phase_b/windows_job_v8.py",
    "src/loop167_phase_b/raw_manifest_adapter_v4.py",
    "src/loop167_phase_b/raw_worker.py",
    "src/loop167_phase_b/feature_cache_v4.py",
    "src/loop167_phase_b/fit_worker.py",
    "src/loop167_phase_b/evaluation_v4.py",
    "src/loop167_phase_b/progress_ledger.py",
    "src/loop167_phase_b/arm_contract.py",
    "src/loop167_phase_b/counterfactual.py",
    "src/loop167_phase_b/b0_projector.py",
    "src/loop167_phase_b/ember_controls.py",
    "src/loop167_phase_b/authenticode.py",
    "src/loop167_phase_b/one_pass_reader.py",
    "src/loop167_phase_b/raw_context.py",
    "src/loop166/__init__.py",
    "src/loop166/windows_job.py",
    "src/loop166/windows_process_lineage.py",
    "src/loop167/__init__.py",
    "src/loop167/ember_v3_native.py",
    "src/loop167/semantic_mapping.py",
    "src/loop167/semantic_schema.py",
    "src/kvd_features/__init__.py",
    "src/kvd_features/content_pe_v1.py",
    "src/kvd_features/extractor.py",
    "src/kvd_features/schema_names.py",
    "scripts/build_loop167_phase_b_execution_contract_v8.py",
    "scripts/build_loop167_phase_b_runtime_lock_v8.py",
    "scripts/run_loop167_phase_b_supervisor_v8.py",
    "scripts/run_loop167_phase_b_controller_v8.py",
    "scripts/probe_loop167_phase_b_v8_suspended_child.py",
    "scripts/seal_loop167_phase_b_source_closure_v8.py",
    "tests/test_loop167_phase_b_package_import.py",
    "tests/test_loop167_phase_b_v8_windows_job.py",
    "tests/test_loop167_phase_b_v8_supervisor.py",
    "tests/test_loop167_phase_b_v8_child_attestation.py",
    "tests/test_loop167_phase_b_v8_loop166_bridge.py",
    "tests/test_loop167_phase_b_v8_path_safety.py",
    "tests/test_loop167_phase_b_v8_resource_guard.py",
    "tests/test_loop167_phase_b_v8_execution_authorization.py",
    "tests/test_loop167_phase_b_lease_v8.py",
    "tests/test_loop167_phase_b_v8_controller.py",
    "tests/test_loop167_phase_b_v8_source_closure.py",
    "tests/test_loop167_phase_b_v8_state_machine.py",
    "tests/test_loop167_phase_b_v8_windows_integration.py",
    "tests/test_loop167_phase_b_v4_path_safety.py",
    "tests/test_loop167_phase_b_v4_manifest_cache.py",
    "tests/test_loop167_phase_b_windows_job_v4.py",
    "tests/test_loop167_phase_b_evaluation_v4.py",
    "tests/test_loop167_phase_b_contracts.py",
    "tests/test_loop167_phase_b_protocol.py",
    "tests/test_loop167_phase_b_protocol_addendum.py",
    "tests/test_loop167_phase_b_arm_contract.py",
    "tests/test_loop167_phase_b_counterfactual.py",
    "tests/test_loop167_phase_b_progress_ledger.py",
    "tests/test_loop167_phase_b_raw_worker.py",
    "tests/test_loop167_phase_b_fit_worker.py",
    "tests/test_loop167_phase_b_context.py",
    "tests/test_loop167_phase_b_authenticode.py",
    "tests/test_loop167_phase_b_minimal_pe.py",
    "tests/test_loop167_phase_b_reader.py",
)


def _binding(relative_path: str) -> dict[str, str]:
    path = safe_project_path(PROJECT_ROOT, relative_path, require_exists=True, require_regular_file=True)
    return {"path": relative_path, "sha256": sha256_file(path)}


def _validate_static_v8_artifacts() -> None:
    parent_attestation = _binding(PARENT_V7_PRELEASE_ATTESTATION_RELATIVE_PATH)
    verify_parent_v7_prelease_attestation_v8(PROJECT_ROOT, parent_attestation)
    contract_binding = _binding(EXECUTION_CONTRACT_RELATIVE_PATH)
    verify_execution_contract_v8(PROJECT_ROOT, contract_binding)
    runtime_lock = require_canonical_json(PROJECT_ROOT / RUNTIME_LOCK_RELATIVE_PATH)
    if runtime_lock.get("schema") != "axon_loop167_phase_b_runtime_lock_v8" or runtime_lock.get("loop_id") != LOOP_ID:
        raise ValueError("Phase-B runtime lock v8 schema drifted")
    if runtime_lock.get("controller") != _binding(CONTROLLER_RELATIVE_PATH):
        raise ValueError("Phase-B runtime lock v8 controller binding drifted")
    if runtime_lock.get("supervisor") != _binding(SUPERVISOR_RELATIVE_PATH):
        raise ValueError("Phase-B runtime lock v8 supervisor binding drifted")
    if runtime_lock.get("execution_contract") != contract_binding:
        raise ValueError("Phase-B runtime lock v8 execution-contract binding drifted")


def build_source_closure_v8_payload() -> dict[str, Any]:
    _validate_static_v8_artifacts()
    if len(V8_SOURCE_PATHS) != len(set(V8_SOURCE_PATHS)):
        raise ValueError("v8 source closure repeats a path")
    if not REQUIRED_SOURCE_PATHS.issubset(set(V8_SOURCE_PATHS)):
        raise ValueError("v8 source closure omits a required control-plane path")
    return {
        "schema": SOURCE_CLOSURE_SCHEMA,
        "loop_id": LOOP_ID,
        "scope": V8_SCOPE,
        "parent_v7_prelease_attestation": _binding(PARENT_V7_PRELEASE_ATTESTATION_RELATIVE_PATH),
        "phase_b_execution_contract": _binding(EXECUTION_CONTRACT_RELATIVE_PATH),
        "runtime_lock": _binding(RUNTIME_LOCK_RELATIVE_PATH),
        "controller": _binding(CONTROLLER_RELATIVE_PATH),
        "supervisor": _binding(SUPERVISOR_RELATIVE_PATH),
        "source_files": [_binding(relative_path) for relative_path in V8_SOURCE_PATHS],
        "static_preflight_ready": True,
        "phase_b_raw_execution_ready": False,
        "dynamic_execution_gates": dict(EXPECTED_DYNAMIC_GATES),
        "remaining_execution_blockers": list(EXPECTED_BLOCKERS),
    }


def _write_new(path: Path, payload: dict[str, Any]) -> str:
    content = canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(content).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.write) == bool(args.check):
        raise SystemExit("Specify exactly one of --write or --check")
    root = safe_project_root(PROJECT_ROOT)
    payload = build_source_closure_v8_payload()
    expected = canonical_json_bytes(payload)
    output_path = safe_project_path(root, SOURCE_CLOSURE_RELATIVE_PATH, require_exists=False)
    if args.write:
        output_path = ensure_v8_static_artifact_parent(root, SOURCE_CLOSURE_RELATIVE_PATH)
        digest = _write_new(output_path, payload)
    else:
        if not output_path.is_file() or output_path.read_bytes() != expected:
            raise SystemExit("Phase-B source closure v8 is missing or drifted")
        digest = sha256_file(output_path)
    print(json.dumps({"path": SOURCE_CLOSURE_RELATIVE_PATH, "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
