from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from seal_loop167_phase_b_source_closure import SOURCE_PATHS  # noqa: E402


def test_phase_b_source_closure_inventory_includes_all_static_components_once() -> None:
    assert len(SOURCE_PATHS) == len(set(SOURCE_PATHS))
    assert "src/loop167_phase_b/progress_ledger.py" in SOURCE_PATHS
    assert "src/loop167_phase_b/arm_contract.py" in SOURCE_PATHS
    assert "src/loop167_phase_b/resource_guard.py" in SOURCE_PATHS
    assert "scripts/run_loop167_phase_b_controller.py" in SOURCE_PATHS
    assert all("loop164" not in path for path in SOURCE_PATHS)
