#!/usr/bin/env python3
"""Seal the corrected static-only source closure for Loop167 Phase B."""

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

from seal_loop167_phase_b_source_closure import SOURCE_PATHS as V1_SOURCE_PATHS  # noqa: E402

from loop167_phase_b.contracts import (  # noqa: E402
    canonical_json_bytes,
    require_canonical_json,
    sha256_file,
)

ARTIFACT_ROOT = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta"
PROTOCOL_PATH = ARTIFACT_ROOT / "phase_b_protocol.json"
REPLAY_ADDENDUM_PATH = ARTIFACT_ROOT / "phase_b_protocol_addendum.json"
ISOLATION_ADDENDUM_PATH = ARTIFACT_ROOT / "phase_b_runtime_isolation_addendum.json"
SOURCE_CLOSURE_V1_PATH = ARTIFACT_ROOT / "phase_b_source_closure.json"
RUNTIME_LOCK_V2_PATH = ARTIFACT_ROOT / "phase_b_runtime_lock_v2.json"
CLOSURE_V2_PATH = ARTIFACT_ROOT / "phase_b_source_closure_v2.json"
V2_SOURCE_PATHS = V1_SOURCE_PATHS + (
    "src/loop167_phase_b/preflight_v2.py",
    "src/loop167_phase_b/runtime_lock_v2.py",
    "scripts/build_loop167_phase_b_runtime_lock_v2.py",
    "scripts/run_loop167_phase_b_controller_v2.py",
    "scripts/seal_loop167_phase_b_runtime_isolation_addendum.py",
    "scripts/seal_loop167_phase_b_source_closure_v2.py",
    "tests/test_loop167_phase_b_v2_preflight.py",
    "tests/test_loop167_phase_b_v2_runtime_isolation.py",
    "tests/test_loop167_phase_b_v2_runtime_lock.py",
    "tests/test_loop167_phase_b_v2_source_closure.py",
)


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Required source closure v2 path is missing or unsafe: {path}")
    return {"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(path)}


def build_source_closure_v2_payload() -> dict[str, Any]:
    protocol = require_canonical_json(PROTOCOL_PATH)
    replay_addendum = require_canonical_json(REPLAY_ADDENDUM_PATH)
    isolation_addendum = require_canonical_json(ISOLATION_ADDENDUM_PATH)
    if replay_addendum.get("parent_phase_b_protocol") != _binding(PROTOCOL_PATH):
        raise ValueError("Phase-B replay addendum binding drifted")
    if isolation_addendum.get("parent_phase_b_protocol") != _binding(PROTOCOL_PATH):
        raise ValueError("Phase-B isolation addendum protocol binding drifted")
    if isolation_addendum.get("parent_replay_addendum") != _binding(REPLAY_ADDENDUM_PATH):
        raise ValueError("Phase-B isolation addendum replay binding drifted")
    if isolation_addendum.get("preserved_source_closure_v1") != _binding(SOURCE_CLOSURE_V1_PATH):
        raise ValueError("Phase-B isolation addendum source closure binding drifted")
    phase_a_bindings = protocol.get("phase_a_bindings")
    if not isinstance(phase_a_bindings, dict) or len(phase_a_bindings) != 7:
        raise ValueError("Phase-B protocol lacks complete Phase-A bindings")
    source_files = [_binding(PROJECT_ROOT / relative_path) for relative_path in V2_SOURCE_PATHS]
    return {
        "schema": "axon_loop167_phase_b_source_closure_v2",
        "loop_id": "loop167_ember_v3_novel_delta",
        "scope": "static_preflight_only_no_raw_checkpoint_prediction_or_fit_access",
        "supersedes_source_closure_v1": _binding(SOURCE_CLOSURE_V1_PATH),
        "runtime_isolation_addendum": _binding(ISOLATION_ADDENDUM_PATH),
        "phase_a_bindings": phase_a_bindings,
        "phase_b_protocol": _binding(PROTOCOL_PATH),
        "phase_b_protocol_addendum": _binding(REPLAY_ADDENDUM_PATH),
        "runtime_lock_v2": _binding(RUNTIME_LOCK_V2_PATH),
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
    payload = build_source_closure_v2_payload()
    expected = canonical_json_bytes(payload)
    if args.write:
        digest = write_new(CLOSURE_V2_PATH, payload)
    else:
        if not CLOSURE_V2_PATH.is_file() or CLOSURE_V2_PATH.read_bytes() != expected:
            raise SystemExit("Phase-B source closure v2 is missing or drifted")
        digest = sha256_file(CLOSURE_V2_PATH)
    print(json.dumps({"path": CLOSURE_V2_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
