#!/usr/bin/env python3
"""Build deterministic, raw-free Loop167 Phase-A semantic artifacts."""

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
    build_frozen_baseline_allowlist,
    build_semantic_delta_mapping,
)

ARTIFACT_ROOT = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta"
MAPPING_PATH = ARTIFACT_ROOT / "semantic_delta_mapping.json"
ALLOWLIST_PATH = ARTIFACT_ROOT / "frozen_deduplicated_baseline_allowlist.json"


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_new_canonical_json(path: Path, payload: dict[str, Any]) -> str:
    """Create an immutable artifact; an existing path is verification-only."""

    content = canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return sha256_bytes(content)


def check_canonical_artifact(path: Path, payload: dict[str, Any]) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Loop167 Phase-A artifact: {path}")
    content = path.read_bytes()
    expected = canonical_json_bytes(payload)
    if content != expected:
        raise ValueError(f"Loop167 artifact drift or non-canonical encoding: {path}")
    return sha256_bytes(content)


def build_or_check(*, check_only: bool) -> dict[str, str]:
    payloads = {
        MAPPING_PATH: build_semantic_delta_mapping(),
        ALLOWLIST_PATH: build_frozen_baseline_allowlist(),
    }
    operation = check_canonical_artifact if check_only else write_new_canonical_json
    return {str(path.relative_to(PROJECT_ROOT)): operation(path, payload) for path, payload in payloads.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify existing immutable artifacts instead of creating them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_or_check(check_only=bool(args.check))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
