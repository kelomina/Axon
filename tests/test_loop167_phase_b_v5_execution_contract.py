from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from src.loop167_phase_b.contracts import PhaseBContractError
from src.loop167_phase_b.execution_contract_v5 import (
    B1_SAMPLING_INDICATORS_CONTRACT,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    FIXED_OUTPUT_CATALOG,
    RAW_ROOT_RELATIVE_PATH,
    _fixed_binding_snapshot,
    resolve_output_catalog_v5,
)


def test_v5_contract_freezes_the_train_raw_root_and_b1_audit_only_indicators() -> None:
    assert RAW_ROOT_RELATIVE_PATH == "data/random_20w_worktree"
    assert B1_SAMPLING_INDICATORS_CONTRACT == {
        "dimension": 3,
        "role": "audit_only_not_in_fit_cache",
        "receipt_key": "sampling_audit",
    }


def test_v5_contract_snapshots_an_immutable_authorization_binding() -> None:
    binding = MappingProxyType(
        {
            "path": EXECUTION_CONTRACT_RELATIVE_PATH,
            "sha256": "a" * 64,
        }
    )

    snapshot = _fixed_binding_snapshot(
        binding,
        label="execution_contract",
        expected_path=EXECUTION_CONTRACT_RELATIVE_PATH,
    )

    assert snapshot == dict(binding)
    assert snapshot is not binding


@pytest.mark.parametrize(
    "binding",
    (
        object(),
        {"path": EXECUTION_CONTRACT_RELATIVE_PATH},
        {"path": "manifests/roadmap_9997/other.json", "sha256": "a" * 64},
    ),
)
def test_v5_contract_keeps_the_mapping_boundary_fail_closed(binding: object) -> None:
    with pytest.raises(PhaseBContractError):
        _fixed_binding_snapshot(
            binding,
            label="execution_contract",
            expected_path=EXECUTION_CONTRACT_RELATIVE_PATH,
        )


def test_v5_output_catalog_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "reports").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform")

    with pytest.raises(PhaseBContractError):
        resolve_output_catalog_v5(root, list(FIXED_OUTPUT_CATALOG))
