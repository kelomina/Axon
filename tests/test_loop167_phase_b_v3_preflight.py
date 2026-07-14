from __future__ import annotations

from pathlib import Path

import pytest

from src.loop167_phase_b.contracts import sha256_file
from src.loop167_phase_b.preflight_v3 import validate_static_preflight_v3
from src.loop167_phase_b.runtime_lock_v2 import REQUIRED_ENVIRONMENT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta"
SOURCE_CLOSURE_PATH = ARTIFACT_ROOT / "phase_b_source_closure_v3.json"
ISOLATION_ADDENDUM_PATH = ARTIFACT_ROOT / "phase_b_runtime_isolation_addendum.json"
BOOTSTRAP_ADDENDUM_PATH = ARTIFACT_ROOT / "phase_b_runtime_bootstrap_addendum.json"
CONTROLLER_PATH = PROJECT_ROOT / "scripts" / "run_loop167_phase_b_controller_v3.py"
CANONICAL_ARGV = (
    "vnev/Scripts/python.exe",
    "-I",
    "scripts/run_loop167_phase_b_controller_v3.py",
    "--preflight",
)


def test_v3_static_preflight_opens_no_raw_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    if not SOURCE_CLOSURE_PATH.is_file():
        pytest.skip("Phase-B source closure v3 is not sealed yet")
    for name, value in REQUIRED_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    receipt = validate_static_preflight_v3(
        PROJECT_ROOT,
        source_closure_binding={
            "path": SOURCE_CLOSURE_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(SOURCE_CLOSURE_PATH),
        },
        controller_binding={
            "path": CONTROLLER_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(CONTROLLER_PATH),
        },
        isolation_addendum_binding={
            "path": ISOLATION_ADDENDUM_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(ISOLATION_ADDENDUM_PATH),
        },
        bootstrap_addendum_binding={
            "path": BOOTSTRAP_ADDENDUM_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(BOOTSTRAP_ADDENDUM_PATH),
        },
        canonical_argv=CANONICAL_ARGV,
    )

    assert receipt.raw_open_attempts == 0
    assert len(receipt.runtime_lock_sha256) == 64
