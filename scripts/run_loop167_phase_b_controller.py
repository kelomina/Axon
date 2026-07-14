#!/usr/bin/env python3
"""Run only the static preflight for the future Loop167 Phase-B execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loop167_phase_b.contracts import sha256_file  # noqa: E402
from loop167_phase_b.preflight import validate_static_preflight  # noqa: E402

ARTIFACT_ROOT = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta"
SOURCE_CLOSURE_PATH = ARTIFACT_ROOT / "phase_b_source_closure.json"
CANONICAL_ARGV = (
    "vnev/Scripts/python.exe",
    "-I",
    "scripts/run_loop167_phase_b_controller.py",
    "--preflight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.preflight:
        raise SystemExit("Only --preflight is implemented; raw access and fitting remain unavailable")
    receipt = validate_static_preflight(
        PROJECT_ROOT,
        source_closure_binding={
            "path": SOURCE_CLOSURE_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(SOURCE_CLOSURE_PATH),
        },
        controller_binding={
            "path": Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        canonical_argv=CANONICAL_ARGV,
    )
    print(
        json.dumps(
            {
                "schema": "axon_loop167_phase_b_static_preflight_receipt_v1",
                "decision": "pass_static_preflight_raw_open_attempts_zero",
                "protocol_sha256": receipt.protocol_sha256,
                "source_closure_sha256": receipt.source_closure_sha256,
                "runtime_lock_sha256": receipt.runtime_lock_sha256,
                "raw_open_attempts": receipt.raw_open_attempts,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
