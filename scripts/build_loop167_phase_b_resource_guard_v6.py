#!/usr/bin/env python3
"""Build a fresh passing Loop167 Phase-B v6 resource guard without raw access."""

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

from loop167_phase_b.contracts import canonical_json_bytes, sha256_file  # noqa: E402
from loop167_phase_b.execution_contract_v6 import (  # noqa: E402
    EXECUTION_CONTRACT_RELATIVE_PATH,
    RESOURCE_GUARD_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    ensure_v6_static_artifact_parent,
)
from loop167_phase_b.path_safety_v4 import safe_project_path, safe_project_root  # noqa: E402
from loop167_phase_b.resource_guard_v6 import (  # noqa: E402
    build_resource_guard_payload_v6,
    current_system_snapshot_v6,
    verify_resource_guard_v6,
)


def _binding(relative_path: str) -> dict[str, str]:
    root = safe_project_root(PROJECT_ROOT)
    path = safe_project_path(root, relative_path, require_exists=True, require_regular_file=True)
    return {"path": relative_path, "sha256": sha256_file(path)}


def build_payload(*, created_at_utc: str) -> dict[str, Any]:
    return build_resource_guard_payload_v6(
        safe_project_root(PROJECT_ROOT),
        execution_contract_binding=_binding(EXECUTION_CONTRACT_RELATIVE_PATH),
        source_closure_binding=_binding(SOURCE_CLOSURE_RELATIVE_PATH),
        runtime_lock_binding=_binding(RUNTIME_LOCK_RELATIVE_PATH),
        snapshot=current_system_snapshot_v6(),
        created_at_utc=created_at_utc,
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
    output_path = safe_project_path(root, RESOURCE_GUARD_RELATIVE_PATH, require_exists=False)
    if args.check:
        verified = verify_resource_guard_v6(
            root,
            _binding(RESOURCE_GUARD_RELATIVE_PATH),
            expected_execution_contract_binding=_binding(EXECUTION_CONTRACT_RELATIVE_PATH),
            expected_source_closure_binding=_binding(SOURCE_CLOSURE_RELATIVE_PATH),
            expected_runtime_lock_binding=_binding(RUNTIME_LOCK_RELATIVE_PATH),
            now_utc=datetime.now(UTC),
        )
        print(json.dumps({"path": RESOURCE_GUARD_RELATIVE_PATH, "sha256": verified.guard_sha256}, sort_keys=True))
        return 0

    created_at_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    payload = build_payload(created_at_utc=created_at_utc)
    if payload["guard_ready"] is not True:
        print(
            json.dumps(
                {
                    "decision": payload["decision"],
                    "failures": payload["failures"],
                    "guard_ready": False,
                    "path_not_written": RESOURCE_GUARD_RELATIVE_PATH,
                },
                sort_keys=True,
            )
        )
        return 2
    output_path = ensure_v6_static_artifact_parent(root, RESOURCE_GUARD_RELATIVE_PATH)
    digest = _write_new(output_path, payload)
    print(json.dumps({"path": RESOURCE_GUARD_RELATIVE_PATH, "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
