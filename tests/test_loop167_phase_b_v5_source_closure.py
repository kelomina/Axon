from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from seal_loop167_phase_b_source_closure_v5 import V5_SOURCE_PATHS  # noqa: E402


def test_v5_source_closure_contains_only_active_v5_execution_sources() -> None:
    assert len(V5_SOURCE_PATHS) == len(set(V5_SOURCE_PATHS))
    assert "scripts/run_loop167_phase_b_controller_v5.py" in V5_SOURCE_PATHS
    assert "src/loop167_phase_b/preflight_v5.py" in V5_SOURCE_PATHS
    assert "src/loop167_phase_b/execution_authorization_v5.py" in V5_SOURCE_PATHS
    assert "src/loop167_phase_b/lease_v5.py" in V5_SOURCE_PATHS
    assert "src/loop167_phase_b/raw_manifest_adapter_v4.py" in V5_SOURCE_PATHS
    assert "src/loop167_phase_b/__init__.py" in V5_SOURCE_PATHS
    assert "tests/test_loop167_phase_b_v5_controller.py" in V5_SOURCE_PATHS
    assert "tests/test_loop167_phase_b_v5_preflight.py" in V5_SOURCE_PATHS
    assert "tests/test_loop167_phase_b_v5_execution_authorization.py" in V5_SOURCE_PATHS
    assert "tests/test_loop167_phase_b_v4_manifest_cache.py" in V5_SOURCE_PATHS
    assert "tests/test_loop167_phase_b_raw_worker.py" in V5_SOURCE_PATHS
    assert "tests/test_loop167_phase_b_fit_worker.py" in V5_SOURCE_PATHS
    assert "tests/test_loop167_phase_b_arm_contract.py" in V5_SOURCE_PATHS
    assert "tests/test_loop167_phase_b_progress_ledger.py" in V5_SOURCE_PATHS
    assert all("controller_v3" not in path for path in V5_SOURCE_PATHS)
    assert all("source_closure_v4.py" not in path for path in V5_SOURCE_PATHS)
    assert all(
        not path.endswith(("phase_b_resource_guard_v5.json", "phase_b_run_authorization_v5.json"))
        for path in V5_SOURCE_PATHS
    )


def test_v5_source_closure_inventory_points_only_to_existing_regular_files() -> None:
    project_root = Path(__file__).resolve().parents[1]
    assert all((project_root / relative_path).is_file() for relative_path in V5_SOURCE_PATHS)
    assert all(not (project_root / relative_path).is_symlink() for relative_path in V5_SOURCE_PATHS)
