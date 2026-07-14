#!/usr/bin/env python3
"""Seal the source-semantics addendum after Loop167 mapping artifacts exist."""

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
    CATEGORY_EXACT,
    CATEGORY_FORBIDDEN,
    CATEGORY_NOVEL,
    CATEGORY_PARTIAL,
    build_frozen_baseline_allowlist,
    build_semantic_delta_mapping,
)

ARTIFACT_ROOT = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta"
MAPPING_PATH = ARTIFACT_ROOT / "semantic_delta_mapping.json"
ALLOWLIST_PATH = ARTIFACT_ROOT / "frozen_deduplicated_baseline_allowlist.json"
ADDENDUM_PATH = ARTIFACT_ROOT / "phase_a_source_semantics_addendum.json"
PROPOSAL_PATH = ARTIFACT_ROOT / "proposal.json"
AUTHORIZATION_PATH = ARTIFACT_ROOT / "authorization.json"


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_canonical(path: Path, payload: dict[str, Any]) -> str:
    if not path.is_file() or path.read_bytes() != canonical_json_bytes(payload):
        raise ValueError(f"Missing or drifted Phase-A artifact: {path}")
    return sha256_path(path)


def build_addendum_payload() -> dict[str, Any]:
    mapping = build_semantic_delta_mapping()
    allowlist = build_frozen_baseline_allowlist()
    mapping_sha256 = _require_canonical(MAPPING_PATH, mapping)
    allowlist_sha256 = _require_canonical(ALLOWLIST_PATH, allowlist)
    return {
        "schema": "axon_loop167_phase_a_source_semantics_addendum_v1",
        "scope": "static_semantic_freeze_only",
        "proposal_binding": {
            "path": str(PROPOSAL_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(PROPOSAL_PATH),
        },
        "authorization_binding": {
            "path": str(AUTHORIZATION_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_path(AUTHORIZATION_PATH),
        },
        "phase_a_artifact_bindings": {
            "semantic_delta_mapping": {
                "path": str(MAPPING_PATH.relative_to(PROJECT_ROOT)),
                "sha256": mapping_sha256,
            },
            "frozen_deduplicated_baseline_allowlist": {
                "path": str(ALLOWLIST_PATH.relative_to(PROJECT_ROOT)),
                "sha256": allowlist_sha256,
            },
        },
        "actual_source_order": [
            {"group": "general", "range": [0, 7]},
            {"group": "histogram", "range": [7, 263]},
            {"group": "byteentropy", "range": [263, 519]},
            {"group": "strings", "range": [519, 696]},
            {"group": "header", "range": [696, 770]},
            {"group": "section", "range": [770, 994]},
            {"group": "imports", "range": [994, 2276]},
            {"group": "exports", "range": [2276, 2405]},
            {"group": "datadirectories", "range": [2405, 2439]},
            {"group": "richheader", "range": [2439, 2472]},
            {"group": "authenticode", "range": [2472, 2480]},
            {"group": "pefilewarnings", "range": [2480, 2568]},
        ],
        "category_counts": {
            CATEGORY_EXACT: mapping["category_counts"][CATEGORY_EXACT],
            CATEGORY_PARTIAL: mapping["category_counts"][CATEGORY_PARTIAL],
            CATEGORY_NOVEL: mapping["category_counts"][CATEGORY_NOVEL],
            CATEGORY_FORBIDDEN: mapping["category_counts"][CATEGORY_FORBIDDEN],
        },
        "semantic_decisions": {
            "ordered_start_bytes": "Novel because they are absent from the frozen 572-dimensional B0 inventory.",
            "byteentropy": "Novel 16x16 local entropy/byte joint histogram with pinned nonempty semantics.",
            "header_and_dos": "Only the 31 field-level uncovered header/DOS values are novel.",
            "rich_pair_count": "Novel deterministic scalar; its 32 FeatureHasher columns remain forbidden.",
            "data_directories_reserved_pair": "Official columns 2435 and 2436 are dead due to the pinned loop bound and remain forbidden.",
            "exports_sentinel": "Official column 2276 is literal hash-vector length sentinel, not an export count.",
            "empty_input": "Native extractor returns finite zero values plus empty_input rather than claiming reference parity.",
        },
        "baseline_boundary": {
            "source_inventory_dimension": allowlist["source_inventory_dimension"],
            "frozen_allowlist_dimension": allowlist["frozen_allowlist_dimension"],
            "ordered_start_bytes_added_to_B0": False,
            "deduplication_policy": allowlist["deduplication_policy"],
        },
        "phase_b_blockers": [
            "No native one-pass overlap-control extractor or RawFeatureContext exists yet.",
            "Authenticode controls require a pinned native parser/missing contract; signify is not an Axon dependency lock.",
            "Frozen HGB configuration produces bit-identical seed 41/42/43 outputs on the audited synthetic matrix; three runs cannot be called independent robustness until a revised authorization resolves this.",
            "No Phase-B controller, immutable lease, runtime lock, or fresh resource guard is source-closed.",
        ],
        "authority_preserved": {
            "phase_a_only": True,
            "raw_opens": 0,
            "training": False,
            "fitting": False,
            "public_key_required": False,
            "phase_b_execution_ready": False,
        },
    }


def write_new(payload: dict[str, Any]) -> str:
    content = canonical_json_bytes(payload)
    descriptor = os.open(ADDENDUM_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(content).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    payload = build_addendum_payload()
    if parse_args().check:
        if not ADDENDUM_PATH.is_file() or ADDENDUM_PATH.read_bytes() != canonical_json_bytes(payload):
            raise SystemExit("Loop167 source-semantics addendum is missing or drifted")
        digest = sha256_path(ADDENDUM_PATH)
    else:
        digest = write_new(payload)
    print(json.dumps({"path": str(ADDENDUM_PATH.relative_to(PROJECT_ROOT)), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
