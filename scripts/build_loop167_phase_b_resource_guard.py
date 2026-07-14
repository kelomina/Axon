#!/usr/bin/env python3
"""Build or verify a fresh, static-only resource guard for Loop167 Phase B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
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
from loop167_phase_b.resource_guard import (  # noqa: E402
    build_resource_guard_payload,
    current_system_snapshot,
)

ARTIFACT_ROOT = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta"
PROTOCOL_PATH = ARTIFACT_ROOT / "phase_b_protocol.json"
RUNTIME_LOCK_PATH = ARTIFACT_ROOT / "phase_b_runtime_lock.json"
SOURCE_CLOSURE_PATH = ARTIFACT_ROOT / "phase_b_source_closure.json"
RESOURCE_GUARD_PATH = ARTIFACT_ROOT / "phase_b_resource_guard.json"
CONTROLLER_PATH = PROJECT_ROOT / "scripts" / "run_loop167_phase_b_controller.py"
CANONICAL_ARGV = (
    "vnev/Scripts/python.exe",
    "-I",
    "scripts/run_loop167_phase_b_controller.py",
    "--preflight",
)


def _binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required static artifact is missing: {path}")
    return {"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(path)}


def build_payload(*, created_at_utc: str) -> dict[str, Any]:
    protocol = require_canonical_json(PROTOCOL_PATH)
    return build_resource_guard_payload(
        source_closure_binding=_binding(SOURCE_CLOSURE_PATH),
        protocol_binding=_binding(PROTOCOL_PATH),
        runtime_lock_binding=_binding(RUNTIME_LOCK_PATH),
        controller_binding=_binding(CONTROLLER_PATH),
        canonical_argv=CANONICAL_ARGV,
        resource_contract=protocol["resource_contract"],
        snapshot=current_system_snapshot(),
        created_at_utc=created_at_utc,
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
    parser.add_argument("--created-at-utc")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.write) == bool(args.check):
        raise SystemExit("Specify exactly one of --write or --check")
    if args.check and args.created_at_utc is None:
        if not RESOURCE_GUARD_PATH.is_file():
            raise SystemExit("Phase-B resource guard is missing")
        payload = require_canonical_json(RESOURCE_GUARD_PATH)
        print(json.dumps({"path": RESOURCE_GUARD_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(RESOURCE_GUARD_PATH)}, sort_keys=True))
        return 0
    created_at_utc = args.created_at_utc or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload = build_payload(created_at_utc=created_at_utc)
    if args.write:
        digest = write_new(RESOURCE_GUARD_PATH, payload)
    else:
        if not RESOURCE_GUARD_PATH.is_file() or RESOURCE_GUARD_PATH.read_bytes() != canonical_json_bytes(payload):
            raise SystemExit("Phase-B resource guard is missing or drifted")
        digest = sha256_file(RESOURCE_GUARD_PATH)
    print(json.dumps({"path": RESOURCE_GUARD_PATH.relative_to(PROJECT_ROOT).as_posix(), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
