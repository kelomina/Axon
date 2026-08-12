from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.protocol import (  # noqa: E402
    CoverageAccounting,
    assert_coverage_gate,
    deterministic_region_permutation,
)
from src.loop175.region_extractor import (  # noqa: E402
    Region,
    RegionExtractionResult,
    RegionKind,
)
from src.loop175.resource_guard import ResourceGuard, ResourceLimits  # noqa: E402


def _result(status: str = "ok") -> RegionExtractionResult:
    supported = status == "ok"
    return RegionExtractionResult(
        status=status,
        file_size=4,
        bytes_read=4,
        parse_ok=supported,
        regions=(Region(RegionKind.DOS_PE_HEADER, 0, b"MZ"),),
    )


def test_counterfactual_permutation_is_deterministic_and_role_isolated() -> None:
    fit = deterministic_region_permutation(20, seed=41, fold=2, role="fit")
    assert fit == deterministic_region_permutation(20, seed=41, fold=2, role="fit")
    assert fit != deterministic_region_permutation(20, seed=41, fold=2, role="holdout")
    assert sorted(fit) == list(range(20))


def test_coverage_accounting_keeps_failures_and_enforces_gate() -> None:
    accounting = CoverageAccounting()
    for label in (0, 1):
        for _ in range(100):
            accounting.observe(_result(), label=label)
    assert assert_coverage_gate(accounting)["attempted"] == 200

    failing = CoverageAccounting()
    failing.observe(_result("pe_parse_failure"), label=0)
    with pytest.raises(RuntimeError, match="coverage"):
        assert_coverage_gate(failing)


def test_resource_guard_fails_closed_on_disk_limit() -> None:
    guard = ResourceGuard(ResourceLimits(maximum_new_disk_bytes=1))
    guard.start()
    with pytest.raises(RuntimeError, match="disk"):
        guard.snapshot(new_disk_bytes=2)
