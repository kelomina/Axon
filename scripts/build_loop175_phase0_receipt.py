#!/usr/bin/env python3
"""Build a source-bound Loop175 Phase-0 receipt after focused validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    Path("docs/phase3_loop175_section_region_moe_proposal.md"),
    Path("manifests/roadmap_9997/loop175_section_region_moe/proposal.json"),
    Path("src/loop175/__init__.py"),
    Path("src/loop175/region_extractor.py"),
    Path("src/loop175/model.py"),
    Path("src/loop175/protocol.py"),
    Path("src/loop175/resource_guard.py"),
    Path("scripts/build_loop175_phase0_receipt.py"),
    Path("scripts/run_loop175_region_coverage_probe.py"),
    Path("tests/test_loop175_region_extractor.py"),
    Path("tests/test_loop175_region_model.py"),
    Path("tests/test_loop175_protocol.py"),
    Path("tests/test_run_loop175_region_coverage_probe.py"),
)
TEST_PATHS = (
    "tests/test_loop175_region_extractor.py",
    "tests/test_loop175_region_model.py",
    "tests/test_loop175_protocol.py",
    "tests/test_run_loop175_region_coverage_probe.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_validation() -> tuple[subprocess.CompletedProcess[str], subprocess.CompletedProcess[str]]:
    pytest_result = subprocess.run(
        [sys.executable, "-m", "pytest", *TEST_PATHS, "-q"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    ruff_result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src/loop175", *TEST_PATHS],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return pytest_result, ruff_result


def build_receipt(output: Path) -> dict[str, object]:
    missing = [str(path) for path in SOURCE_PATHS if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"Loop175 source closure is incomplete: {missing}")
    pytest_result, ruff_result = run_validation()
    if pytest_result.returncode != 0 or ruff_result.returncode != 0:
        raise RuntimeError(
            "Loop175 Phase-0 validation failed\n"
            f"pytest:\n{pytest_result.stdout}\n{pytest_result.stderr}\n"
            f"ruff:\n{ruff_result.stdout}\n{ruff_result.stderr}"
        )
    sources = [
        {
            "path": path.as_posix(),
            "sha256": sha256(PROJECT_ROOT / path),
            "bytes": (PROJECT_ROOT / path).stat().st_size,
        }
        for path in SOURCE_PATHS
    ]
    payload: dict[str, object] = {
        "schema": "axon_loop175_phase0_receipt_v1",
        "loop_id": "Loop175",
        "claim_scope": "synthetic_and_static_validation_only_no_real_sample_access_or_model_quality",
        "sources": sources,
        "validation": {
            "pytest_command": [sys.executable, "-m", "pytest", *TEST_PATHS, "-q"],
            "pytest_returncode": pytest_result.returncode,
            "pytest_summary": pytest_result.stdout.strip().splitlines()[-1],
            "ruff_returncode": ruff_result.returncode,
            "ruff_summary": ruff_result.stdout.strip().splitlines()[-1],
        },
        "raw_opens": 0,
        "training_runs": 0,
        "decision": "phase0_pass_phase_a_256_train_coverage_probe_ready",
    }
    write_exclusive_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = build_receipt(arguments.output.resolve())
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
