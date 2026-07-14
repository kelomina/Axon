from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from seal_loop167_phase_b_source_closure_v3 import V3_SOURCE_PATHS  # noqa: E402


def test_v3_source_closure_binds_the_self_bootstrapping_controller_once() -> None:
    assert len(V3_SOURCE_PATHS) == len(set(V3_SOURCE_PATHS))
    assert "src/loop167_phase_b/preflight_v3.py" in V3_SOURCE_PATHS
    assert "scripts/run_loop167_phase_b_controller_v3.py" in V3_SOURCE_PATHS
    assert "tests/test_loop167_phase_b_v3_preflight.py" in V3_SOURCE_PATHS
