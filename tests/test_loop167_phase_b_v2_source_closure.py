from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from seal_loop167_phase_b_source_closure_v2 import V2_SOURCE_PATHS  # noqa: E402


def test_v2_source_closure_includes_the_isolated_runtime_correction_once() -> None:
    assert len(V2_SOURCE_PATHS) == len(set(V2_SOURCE_PATHS))
    assert "src/loop167_phase_b/runtime_lock_v2.py" in V2_SOURCE_PATHS
    assert "src/loop167_phase_b/preflight_v2.py" in V2_SOURCE_PATHS
    assert "scripts/run_loop167_phase_b_controller_v2.py" in V2_SOURCE_PATHS
    assert "tests/test_loop167_phase_b_v2_preflight.py" in V2_SOURCE_PATHS
