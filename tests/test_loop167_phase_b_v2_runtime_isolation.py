from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from seal_loop167_phase_b_runtime_isolation_addendum import build_addendum_payload  # noqa: E402


def test_runtime_isolation_addendum_preserves_isolated_python_without_hash_seed_env() -> None:
    payload = build_addendum_payload()
    replacement = payload["replacement_runtime_contract"]

    assert payload["finding"]["pythonhashseed_environment_is_ignored"] is True
    assert replacement["isolated_python_required"] is True
    assert replacement["pythonhashseed_environment_required"] is False
    assert replacement["determinism_must_not_depend_on_python_hash_seed"] is True
