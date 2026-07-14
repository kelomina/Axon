from __future__ import annotations

from pathlib import Path

import pytest

from src.loop167_phase_b.contracts import PhaseBContractError
from src.loop167_phase_b.execution_contract_v4 import (
    B1_SAMPLING_INDICATORS_CONTRACT,
    FIXED_OUTPUT_CATALOG,
    RAW_ROOT_RELATIVE_PATH,
    resolve_output_catalog_v4,
)


def test_v4_contract_freezes_the_train_raw_root_and_b1_audit_only_indicators() -> None:
    assert RAW_ROOT_RELATIVE_PATH == "data/random_20w_worktree"
    assert B1_SAMPLING_INDICATORS_CONTRACT == {
        "dimension": 3,
        "role": "audit_only_not_in_fit_cache",
        "receipt_key": "sampling_audit",
    }


def test_v4_output_catalog_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "reports").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform")

    with pytest.raises(PhaseBContractError):
        resolve_output_catalog_v4(root, list(FIXED_OUTPUT_CATALOG))
