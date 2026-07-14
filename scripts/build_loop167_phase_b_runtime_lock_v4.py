#!/usr/bin/env python3
"""Build or verify the v4 lock for the future isolated Phase-B controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
os.environ.update(THREAD_ENVIRONMENT)

PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loop167_phase_b.contracts import canonical_json_bytes, sha256_file  # noqa: E402
from loop167_phase_b.invocation_v4 import (  # noqa: E402
    CONTROLLER_V4_RELATIVE_PATH,
    EXECUTION_CONTRACT_V4_RELATIVE_PATH,
)
from loop167_phase_b.path_safety_v4 import safe_project_path  # noqa: E402
from loop167_phase_b.runtime_lock_v4 import build_runtime_lock_payload_v4  # noqa: E402

RUNTIME_LOCK_RELATIVE_PATH = "manifests/roadmap_9997/loop167_ember_v3_novel_delta/phase_b_runtime_lock_v4.json"
CONTROLLER_PATH = PROJECT_ROOT / CONTROLLER_V4_RELATIVE_PATH
EXECUTION_CONTRACT_PATH = PROJECT_ROOT / EXECUTION_CONTRACT_V4_RELATIVE_PATH


def _binding(path: Path) -> dict[str, str]:
    relative_path = path.relative_to(PROJECT_ROOT).as_posix()
    safe_path = safe_project_path(
        PROJECT_ROOT,
        relative_path,
        require_exists=True,
        require_regular_file=True,
    )
    return {"path": relative_path, "sha256": sha256_file(safe_path)}


def build_payload() -> dict[str, Any]:
    return build_runtime_lock_payload_v4(
        PROJECT_ROOT,
        controller_binding=_binding(CONTROLLER_PATH),
        execution_contract_binding=_binding(EXECUTION_CONTRACT_PATH),
    )


def write_new(path: Path, payload: dict[str, Any]) -> str:
    content = canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
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
    payload = build_payload()
    expected = canonical_json_bytes(payload)
    safe_output = safe_project_path(
        PROJECT_ROOT,
        RUNTIME_LOCK_RELATIVE_PATH,
        require_exists=False,
    )
    if args.write:
        digest = write_new(safe_output, payload)
    else:
        if not safe_output.is_file() or safe_output.read_bytes() != expected:
            raise SystemExit("Phase-B runtime lock v4 is missing or drifted")
        digest = sha256_file(safe_output)
    print(json.dumps({"path": RUNTIME_LOCK_RELATIVE_PATH, "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
