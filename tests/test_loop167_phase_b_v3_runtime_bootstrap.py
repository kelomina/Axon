from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from seal_loop167_phase_b_runtime_bootstrap_addendum import build_addendum_payload  # noqa: E402


def test_runtime_bootstrap_addendum_requires_in_process_thread_locking() -> None:
    payload = build_addendum_payload()
    replacement = payload["replacement_runtime_contract"]

    assert payload["finding"]["isolated_windows_launcher_can_clear_parent_thread_environment"] is True
    assert replacement["controller_sets_thread_environment_before_native_package_import"] is True
    assert replacement["external_environment_inheritance_required"] is False
