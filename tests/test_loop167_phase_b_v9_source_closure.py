from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from seal_loop167_phase_b_source_closure_v9 import V9_SOURCE_PATHS  # noqa: E402

from loop167_phase_b.preflight_v9 import REQUIRED_SOURCE_PATHS  # noqa: E402


def test_v7_source_closure_contains_the_complete_control_and_data_plane() -> None:
    source_paths = set(V9_SOURCE_PATHS)

    assert len(V9_SOURCE_PATHS) == len(source_paths)
    assert REQUIRED_SOURCE_PATHS.issubset(source_paths)
    assert {
        "src/loop167_phase_b/windows_job_v9.py",
        "src/loop167_phase_b/supervisor_v9.py",
        "src/loop167_phase_b/child_attestation_v9.py",
        "src/loop167_phase_b/lease_v9.py",
        "src/loop167_phase_b/loop166_v8_bridge.py",
        "src/loop166/windows_job.py",
        "src/loop166/windows_process_lineage.py",
        "scripts/run_loop167_phase_b_supervisor_v9.py",
        "scripts/run_loop167_phase_b_controller_v9.py",
        "scripts/probe_loop167_phase_b_v8_suspended_child.py",
        "tests/test_loop167_phase_b_v8_child_attestation.py",
        "tests/test_loop167_phase_b_v8_loop166_bridge.py",
        "tests/test_loop167_phase_b_v8_path_safety.py",
        "tests/test_loop167_phase_b_v8_resource_guard.py",
        "tests/test_loop167_phase_b_v8_execution_authorization.py",
        "tests/test_loop167_phase_b_lease_v9.py",
        "tests/test_loop167_phase_b_v8_controller.py",
        "tests/test_loop167_phase_b_v8_windows_integration.py",
        "tests/test_loop167_phase_b_v8_state_machine.py",
    }.issubset(source_paths)
    assert {
        "src/loop167_phase_b/raw_manifest_adapter_v4.py",
        "src/loop167_phase_b/raw_worker.py",
        "src/loop167_phase_b/feature_cache_v4.py",
        "src/loop167_phase_b/fit_worker.py",
        "src/loop167_phase_b/evaluation_v4.py",
        "src/loop167_phase_b/progress_ledger.py",
        "src/loop167/ember_v3_native.py",
        "src/kvd_features/extractor.py",
    }.issubset(source_paths)


def test_v7_source_closure_inventory_points_only_to_regular_workspace_files() -> None:
    assert all((PROJECT_ROOT / relative_path).is_file() for relative_path in V9_SOURCE_PATHS)
    assert all(not (PROJECT_ROOT / relative_path).is_symlink() for relative_path in V9_SOURCE_PATHS)
