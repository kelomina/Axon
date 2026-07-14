#!/usr/bin/env python3
"""Build or verify runtime-lock v2 for the corrected isolated controller."""

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

from loop167_phase_b.contracts import canonical_json_bytes, sha256_file  # noqa: E402
from loop167_phase_b.runtime_lock_v2 import build_runtime_lock_payload  # noqa: E402

ARTIFACT_ROOT = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta"
RUNTIME_LOCK_PATH = ARTIFACT_ROOT / "phase_b_runtime_lock_v2.json"
ISOLATION_ADDENDUM_PATH = ARTIFACT_ROOT / "phase_b_runtime_isolation_addendum.json"
CONTROLLER_PATH = PROJECT_ROOT / "scripts" / "run_loop167_phase_b_controller_v2.py"
CANONICAL_ARGV = (
    "vnev/Scripts/python.exe",
    "-I",
    "scripts/run_loop167_phase_b_controller_v2.py",
    "--preflight",
)


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required runtime-lock input is missing: {path}")
    return {"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(path)}


def build_payload() -> dict[str, Any]:
    return build_runtime_lock_payload(
        PROJECT_ROOT,
        controller_binding=_binding(CONTROLLER_PATH),
        isolation_addendum_binding=_binding(ISOLATION_ADDENDUM_PATH),
        canonical_argv=CANONICAL_ARGV,
    )


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
    payload = build_payload()
    expected = canonical_json_bytes(payload)
    if args.write:
        digest = write_new(RUNTIME_LOCK_PATH, payload)
    else:
        if not RUNTIME_LOCK_PATH.is_file() or RUNTIME_LOCK_PATH.read_bytes() != expected:
            raise SystemExit("Phase-B runtime lock v2 is missing or drifted")
        digest = sha256_file(RUNTIME_LOCK_PATH)
    print(json.dumps({"path": RUNTIME_LOCK_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
