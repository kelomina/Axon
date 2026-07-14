#!/usr/bin/env python3
"""Seal the deterministic-replay clarification for Loop167 Phase B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta"
PROTOCOL_PATH = ARTIFACT_ROOT / "phase_b_protocol.json"
ADDENDUM_PATH = ARTIFACT_ROOT / "phase_b_protocol_addendum.json"


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_addendum_payload() -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError("Phase-B protocol must be sealed before its addendum")
    return {
        "schema": "axon_loop167_phase_b_protocol_addendum_v1",
        "loop_id": "loop167_ember_v3_novel_delta",
        "parent_phase_b_protocol": {
            "path": PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "scope": "deterministic_replay_semantics_only_no_raw_checkpoint_prediction_or_fit_access",
        "deterministic_replay": {
            "run_labels": [41, 42, 43],
            "canonical_replay_seed": 41,
            "counterfactual_permutation_seed": 41,
            "all_arm_feature_matrix_hashes_must_match_across_run_labels": True,
            "all_arm_prediction_hashes_must_match_across_run_labels": True,
            "run_labels_are_not_independent_statistical_trials": True,
            "select_best_run_label_allowed": False,
        },
        "decision": "clarify_phase_b_three_run_rule_as_identical_deterministic_replays",
        "ready_for": {
            "raw_access": False,
            "fit": False,
            "val": False,
            "test10k": False,
            "legacy_full_test": False,
        },
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
    payload = build_addendum_payload()
    expected = canonical_json_bytes(payload)
    if args.write:
        digest = write_new(ADDENDUM_PATH, payload)
    else:
        if not ADDENDUM_PATH.is_file() or ADDENDUM_PATH.read_bytes() != expected:
            raise SystemExit("Phase-B protocol addendum is missing or drifted")
        digest = sha256_file(ADDENDUM_PATH)
    print(json.dumps({"path": ADDENDUM_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
