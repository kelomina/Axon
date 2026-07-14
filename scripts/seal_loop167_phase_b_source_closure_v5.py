#!/usr/bin/env python3
"""Seal the active v5-only source closure for Loop167 Phase B."""

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
    verify_file_binding,
)
from loop167_phase_b.execution_contract_v5 import (  # noqa: E402
    EXECUTION_CONTRACT_RELATIVE_PATH,
    PARENT_V4_PRELEASE_ATTESTATION_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    ensure_v5_static_artifact_parent,
    verify_execution_contract_v5,
    verify_parent_v4_prelease_attestation_v5,
)
from loop167_phase_b.invocation_v5 import CONTROLLER_V5_RELATIVE_PATH  # noqa: E402
from loop167_phase_b.preflight_v5 import (  # noqa: E402
    EXPECTED_BLOCKERS,
    EXPECTED_DYNAMIC_GATES,
    PHASE_A_BINDING_NAMES,
    PROTOCOL_ADDITION_RELATIVE_PATH,
    SOURCE_CLOSURE_V4_RELATIVE_PATH,
    V5_SCOPE,
)

ARTIFACT_ROOT = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta"
PROTOCOL_PATH = ARTIFACT_ROOT / "phase_b_protocol.json"
REPLAY_ADDENDUM_PATH = PROJECT_ROOT / PROTOCOL_ADDITION_RELATIVE_PATH
SOURCE_CLOSURE_V4_PATH = PROJECT_ROOT / SOURCE_CLOSURE_V4_RELATIVE_PATH
PARENT_V4_PRELEASE_ATTESTATION_PATH = PROJECT_ROOT / PARENT_V4_PRELEASE_ATTESTATION_RELATIVE_PATH
EXECUTION_CONTRACT_PATH = PROJECT_ROOT / EXECUTION_CONTRACT_RELATIVE_PATH
RUNTIME_LOCK_V5_PATH = PROJECT_ROOT / RUNTIME_LOCK_RELATIVE_PATH
CLOSURE_V5_PATH = PROJECT_ROOT / SOURCE_CLOSURE_RELATIVE_PATH
CONTROLLER_PATH = PROJECT_ROOT / CONTROLLER_V5_RELATIVE_PATH

# 只列入当前 v5 运行链及其直接/递归导入；v4 只作为受哈希保护的父证据或共享数据平面。
V5_SOURCE_PATHS = (
    "src/loop167_phase_b/__init__.py",
    "src/loop167_phase_b/contracts.py",
    "src/loop167_phase_b/path_safety_v4.py",
    "src/loop167_phase_b/execution_contract_v4.py",
    "src/loop167_phase_b/invocation_v5.py",
    "src/loop167_phase_b/execution_contract_v5.py",
    "src/loop167_phase_b/runtime_lock_v5.py",
    "src/loop167_phase_b/preflight_v5.py",
    "src/loop167_phase_b/resource_guard_v5.py",
    "src/loop167_phase_b/execution_authorization_v5.py",
    "src/loop167_phase_b/lease_v5.py",
    "src/loop167_phase_b/windows_job_v4.py",
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
    "src/loop167/__init__.py",
    "src/loop167/ember_v3_native.py",
    "src/loop167/semantic_mapping.py",
    "src/loop167/semantic_schema.py",
    "src/kvd_features/__init__.py",
    "src/kvd_features/content_pe_v1.py",
    "src/kvd_features/extractor.py",
    "src/kvd_features/schema_names.py",
    "scripts/build_loop167_phase_b_execution_contract_v5.py",
    "scripts/build_loop167_phase_b_runtime_lock_v5.py",
    "scripts/build_loop167_phase_b_resource_guard_v5.py",
    "scripts/build_loop167_phase_b_execution_authorization_v5.py",
    "scripts/run_loop167_phase_b_controller_v5.py",
    "scripts/seal_loop167_phase_b_source_closure_v5.py",
    "tests/test_loop167_phase_b_package_import.py",
    "tests/test_loop167_phase_b_v5_controller.py",
    "tests/test_loop167_phase_b_v5_preflight.py",
    "tests/test_loop167_phase_b_v5_source_closure.py",
    "tests/test_loop167_phase_b_v5_execution_contract.py",
    "tests/test_loop167_phase_b_v5_execution_authorization.py",
    "tests/test_loop167_phase_b_lease_v5.py",
    "tests/test_loop167_phase_b_v5_invocation.py",
    "tests/test_loop167_phase_b_v4_path_safety.py",
    "tests/test_loop167_phase_b_v5_runtime_lock.py",
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


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Required v5 closure path is missing or unsafe: {path}")
    return {"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(path)}


def _validate_provenance() -> tuple[dict[str, Any], dict[str, str]]:
    protocol = require_canonical_json(PROTOCOL_PATH)
    if protocol.get("schema") != "axon_loop167_phase_b_protocol_v1":
        raise ValueError("Phase-B protocol schema drifted")
    phase_a_bindings = protocol.get("phase_a_bindings")
    if not isinstance(phase_a_bindings, dict) or set(phase_a_bindings) != set(PHASE_A_BINDING_NAMES):
        raise ValueError("Phase-B protocol lacks complete Phase-A bindings")
    for name in PHASE_A_BINDING_NAMES:
        verify_file_binding(PROJECT_ROOT, phase_a_bindings[name], label=f"phase_a.{name}")
    replay_addendum = require_canonical_json(REPLAY_ADDENDUM_PATH)
    if replay_addendum.get("parent_phase_b_protocol") != _binding(PROTOCOL_PATH):
        raise ValueError("Phase-B replay addendum binding drifted")
    prior_closure = require_canonical_json(SOURCE_CLOSURE_V4_PATH)
    if prior_closure.get("schema") != "axon_loop167_phase_b_source_closure_v4":
        raise ValueError("Phase-B v4 provenance schema drifted")
    return protocol, phase_a_bindings


def _validate_static_v5_artifacts(protocol_binding: dict[str, str]) -> None:
    verify_parent_v4_prelease_attestation_v5(
        PROJECT_ROOT,
        _binding(PARENT_V4_PRELEASE_ATTESTATION_PATH),
    )
    verify_execution_contract_v5(
        PROJECT_ROOT,
        _binding(EXECUTION_CONTRACT_PATH),
        expected_protocol_binding=protocol_binding,
    )
    runtime_lock = require_canonical_json(RUNTIME_LOCK_V5_PATH)
    if runtime_lock.get("schema") != "axon_loop167_phase_b_runtime_lock_v5":
        raise ValueError("Phase-B runtime lock v5 schema drifted")
    if runtime_lock.get("controller") != _binding(CONTROLLER_PATH):
        raise ValueError("Phase-B runtime lock v5 controller binding drifted")
    if runtime_lock.get("execution_contract") != _binding(EXECUTION_CONTRACT_PATH):
        raise ValueError("Phase-B runtime lock v5 execution-contract binding drifted")


def build_source_closure_v5_payload() -> dict[str, Any]:
    protocol, phase_a_bindings = _validate_provenance()
    protocol_binding = _binding(PROTOCOL_PATH)
    _validate_static_v5_artifacts(protocol_binding)
    source_files = [_binding(PROJECT_ROOT / relative_path) for relative_path in V5_SOURCE_PATHS]
    return {
        "schema": "axon_loop167_phase_b_source_closure_v5",
        "loop_id": "loop167_ember_v3_novel_delta",
        "scope": V5_SCOPE,
        "supersedes_source_closure_v4": _binding(SOURCE_CLOSURE_V4_PATH),
        "parent_v4_prelease_attestation": _binding(PARENT_V4_PRELEASE_ATTESTATION_PATH),
        "phase_a_bindings": phase_a_bindings,
        "phase_b_protocol": protocol_binding,
        "phase_b_protocol_addendum": _binding(REPLAY_ADDENDUM_PATH),
        "phase_b_execution_contract": _binding(EXECUTION_CONTRACT_PATH),
        "runtime_lock_v5": _binding(RUNTIME_LOCK_V5_PATH),
        "source_files": source_files,
        "static_preflight_ready": True,
        "phase_b_raw_execution_ready": False,
        "dynamic_execution_gates": dict(EXPECTED_DYNAMIC_GATES),
        "remaining_execution_blockers": list(EXPECTED_BLOCKERS),
    }


def write_new(path: Path, payload: dict[str, Any]) -> str:
    content = canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
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
    payload = build_source_closure_v5_payload()
    expected = canonical_json_bytes(payload)
    if args.write:
        closure_path = ensure_v5_static_artifact_parent(PROJECT_ROOT, SOURCE_CLOSURE_RELATIVE_PATH)
        digest = write_new(closure_path, payload)
    else:
        if not CLOSURE_V5_PATH.is_file() or CLOSURE_V5_PATH.read_bytes() != expected:
            raise SystemExit("Phase-B source closure v5 is missing or drifted")
        digest = sha256_file(CLOSURE_V5_PATH)
    print(json.dumps({"path": SOURCE_CLOSURE_RELATIVE_PATH, "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
