#!/usr/bin/env python3
"""Validate and optionally seal Loop167 Phase-A source closure without raw access."""

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

from loop167.semantic_mapping import (  # noqa: E402
    EMBER2024_FEATURES_SHA256,
    EMBER2024_WARNINGS_SHA256,
    build_frozen_baseline_allowlist,
    build_semantic_delta_mapping,
)

ARTIFACT_ROOT = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta"
MAPPING_PATH = ARTIFACT_ROOT / "semantic_delta_mapping.json"
ALLOWLIST_PATH = ARTIFACT_ROOT / "frozen_deduplicated_baseline_allowlist.json"
ADDENDUM_PATH = ARTIFACT_ROOT / "phase_a_source_semantics_addendum.json"
CLOSURE_PATH = ARTIFACT_ROOT / "phase_a_source_closure.json"
SOURCE_PATHS = (
    "src/loop167/__init__.py",
    "src/loop167/semantic_schema.py",
    "src/loop167/semantic_mapping.py",
    "src/loop167/ember_v3_native.py",
    "scripts/build_loop167_semantic_delta_mapping.py",
    "scripts/seal_loop167_phase_a_addendum.py",
    "scripts/validate_loop167_phase_a_closure.py",
    "tests/test_loop167_semantic_mapping.py",
    "tests/test_loop167_native_features.py",
)


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_canonical(path: Path, expected_payload: dict[str, Any]) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase-A artifact: {path}")
    if path.read_bytes() != canonical_json_bytes(expected_payload):
        raise ValueError(f"Phase-A artifact drift: {path}")
    return sha256_path(path)


def build_source_closure_payload() -> dict[str, Any]:
    mapping_sha256 = _load_canonical(MAPPING_PATH, build_semantic_delta_mapping())
    allowlist_sha256 = _load_canonical(ALLOWLIST_PATH, build_frozen_baseline_allowlist())
    if not ADDENDUM_PATH.is_file():
        raise FileNotFoundError("Phase-A source-semantics addendum must be sealed first")
    source_bindings = []
    for relative_path in SOURCE_PATHS:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing source-closure file: {relative_path}")
        source_bindings.append({"path": relative_path, "sha256": sha256_path(path)})
    return {
        "schema": "axon_loop167_phase_a_source_closure_v1",
        "scope": "static_only_no_raw_checkpoint_prediction_training_or_fitting",
        "phase_b_execution_ready": False,
        "external_source_binding": {
            "features_sha256": EMBER2024_FEATURES_SHA256,
            "warnings_sha256": EMBER2024_WARNINGS_SHA256,
            "reference_execution_allowed": False,
        },
        "artifacts": {
            "semantic_delta_mapping": {
                "path": str(MAPPING_PATH.relative_to(PROJECT_ROOT)),
                "sha256": mapping_sha256,
            },
            "frozen_deduplicated_baseline_allowlist": {
                "path": str(ALLOWLIST_PATH.relative_to(PROJECT_ROOT)),
                "sha256": allowlist_sha256,
            },
            "source_semantics_addendum": {
                "path": str(ADDENDUM_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(ADDENDUM_PATH),
            },
        },
        "source_files": source_bindings,
        "remaining_phase_b_blockers": [
            "one_pass_raw_context_and_native_overlap_control_extractor",
            "pinned_native_authenticode_contract",
            "phase_b_controller_and_fail_closed_lease",
            "runtime_lock_and_fresh_resource_guard",
            "revised_seed_independence_or_deterministic_replay_policy",
            "new_phase_b_run_authorization",
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
    parser.add_argument("--write-closure", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.write_closure) == bool(args.check):
        raise SystemExit("Specify exactly one of --write-closure or --check")
    payload = build_source_closure_payload()
    if args.write_closure:
        digest = write_new(CLOSURE_PATH, payload)
    else:
        if not CLOSURE_PATH.is_file() or CLOSURE_PATH.read_bytes() != canonical_json_bytes(payload):
            raise SystemExit("Phase-A source closure is missing or drifted")
        digest = sha256_path(CLOSURE_PATH)
    print(json.dumps({"path": str(CLOSURE_PATH.relative_to(PROJECT_ROOT)), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
