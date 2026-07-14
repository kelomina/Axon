#!/usr/bin/env python3
"""Build or verify the immutable Loop167 Phase-B v4 execution contract."""

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
from loop167_phase_b.execution_contract_v4 import (  # noqa: E402
    EXECUTION_CONTRACT_RELATIVE_PATH,
    PHASE_B_PROTOCOL_RELATIVE_PATH,
    build_execution_contract_payload_v4,
)
from loop167_phase_b.path_safety_v4 import safe_project_path, safe_project_root  # noqa: E402


def _binding(relative_path: str) -> dict[str, str]:
    root = safe_project_root(PROJECT_ROOT)
    path = safe_project_path(root, relative_path, require_exists=True, require_regular_file=True)
    return {"path": relative_path, "sha256": sha256_file(path)}


def build_payload() -> dict[str, Any]:
    return build_execution_contract_payload_v4(
        safe_project_root(PROJECT_ROOT),
        protocol_binding=_binding(PHASE_B_PROTOCOL_RELATIVE_PATH),
    )


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
    output_path = safe_project_path(root, EXECUTION_CONTRACT_RELATIVE_PATH, require_exists=False)
    payload = build_payload()
    expected = canonical_json_bytes(payload)
    if args.write:
        digest = _write_new(output_path, payload)
    else:
        if not output_path.is_file() or output_path.read_bytes() != expected:
            raise SystemExit("Phase-B execution contract v4 is missing or drifted")
        digest = sha256_file(output_path)
    print(json.dumps({"path": EXECUTION_CONTRACT_RELATIVE_PATH, "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
