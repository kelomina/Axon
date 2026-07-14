#!/usr/bin/env python3
"""Seal or verify the static-only source closure for Loop167 Phase B."""

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

ARTIFACT_ROOT = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta"
PROTOCOL_PATH = ARTIFACT_ROOT / "phase_b_protocol.json"
ADDENDUM_PATH = ARTIFACT_ROOT / "phase_b_protocol_addendum.json"
RUNTIME_LOCK_PATH = ARTIFACT_ROOT / "phase_b_runtime_lock.json"
CLOSURE_PATH = ARTIFACT_ROOT / "phase_b_source_closure.json"
SOURCE_PATHS = (
    "pyproject.toml",
    "src/config.py",
    "src/kvd_features/__init__.py",
    "src/kvd_features/content_pe_v1.py",
    "src/kvd_features/extractor.py",
    "src/kvd_features/schema_names.py",
    "src/loop167/__init__.py",
    "src/loop167/ember_v3_native.py",
    "src/loop167/semantic_mapping.py",
    "src/loop167/semantic_schema.py",
    "src/loop167_phase_b/__init__.py",
    "src/loop167_phase_b/arm_contract.py",
    "src/loop167_phase_b/authenticode.py",
    "src/loop167_phase_b/b0_projector.py",
    "src/loop167_phase_b/contracts.py",
    "src/loop167_phase_b/counterfactual.py",
    "src/loop167_phase_b/ember_controls.py",
    "src/loop167_phase_b/lease.py",
    "src/loop167_phase_b/one_pass_reader.py",
    "src/loop167_phase_b/preflight.py",
    "src/loop167_phase_b/progress_ledger.py",
    "src/loop167_phase_b/raw_context.py",
    "src/loop167_phase_b/resource_guard.py",
    "src/loop167_phase_b/runtime_lock.py",
    "scripts/build_loop167_phase_b_protocol.py",
    "scripts/build_loop167_phase_b_resource_guard.py",
    "scripts/build_loop167_phase_b_runtime_lock.py",
    "scripts/run_loop167_phase_b_controller.py",
    "scripts/seal_loop167_phase_b_protocol_addendum.py",
    "scripts/seal_loop167_phase_b_source_closure.py",
    "tests/test_loop167_phase_b_arm_contract.py",
    "tests/test_loop167_phase_b_authenticode.py",
    "tests/test_loop167_phase_b_context.py",
    "tests/test_loop167_phase_b_contracts.py",
    "tests/test_loop167_phase_b_counterfactual.py",
    "tests/test_loop167_phase_b_lease.py",
    "tests/test_loop167_phase_b_minimal_pe.py",
    "tests/test_loop167_phase_b_preflight.py",
    "tests/test_loop167_phase_b_progress_ledger.py",
    "tests/test_loop167_phase_b_protocol.py",
    "tests/test_loop167_phase_b_protocol_addendum.py",
    "tests/test_loop167_phase_b_reader.py",
    "tests/test_loop167_phase_b_resource_guard.py",
    "tests/test_loop167_phase_b_runtime_lock.py",
    "tests/test_loop167_phase_b_source_closure.py",
)


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Required source closure path is missing or unsafe: {path}")
    return {"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(path)}


def build_source_closure_payload() -> dict[str, Any]:
    protocol = require_canonical_json(PROTOCOL_PATH)
    addendum = require_canonical_json(ADDENDUM_PATH)
    if protocol.get("schema") != "axon_loop167_phase_b_protocol_v1":
        raise ValueError("Phase-B protocol schema drifted")
    if addendum.get("parent_phase_b_protocol") != _binding(PROTOCOL_PATH):
        raise ValueError("Phase-B protocol addendum binding drifted")
    phase_a_bindings = protocol.get("phase_a_bindings")
    if not isinstance(phase_a_bindings, dict) or len(phase_a_bindings) != 7:
        raise ValueError("Phase-B protocol lacks complete Phase-A bindings")
    source_files = [_binding(PROJECT_ROOT / relative_path) for relative_path in SOURCE_PATHS]
    return {
        "schema": "axon_loop167_phase_b_source_closure_v1",
        "loop_id": "loop167_ember_v3_novel_delta",
        "scope": "static_preflight_only_no_raw_checkpoint_prediction_or_fit_access",
        "phase_a_bindings": phase_a_bindings,
        "phase_b_protocol": _binding(PROTOCOL_PATH),
        "phase_b_protocol_addendum": _binding(ADDENDUM_PATH),
        "runtime_lock": _binding(RUNTIME_LOCK_PATH),
        "source_files": source_files,
        "static_preflight_ready": True,
        "phase_b_raw_execution_ready": False,
        "remaining_execution_blockers": [
            "raw_worker_input_manifest_scope_adapter_and_cache_seal_not_implemented",
            "fit_worker_isolation_and_75_fit_execution_not_implemented",
            "fresh_resource_guard_and_run_authorization_not_sealed",
        ],
    }


def write_new(path: Path, payload: dict[str, Any]) -> str:
    content = canonical_json_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
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
    payload = build_source_closure_payload()
    expected = canonical_json_bytes(payload)
    if args.write:
        digest = write_new(CLOSURE_PATH, payload)
    else:
        if not CLOSURE_PATH.is_file() or CLOSURE_PATH.read_bytes() != expected:
            raise SystemExit("Phase-B source closure is missing or drifted")
        digest = sha256_file(CLOSURE_PATH)
    print(json.dumps({"path": CLOSURE_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
