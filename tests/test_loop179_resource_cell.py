"""Loop179 resource_cell 资源门测试。"""

from __future__ import annotations

import pytest
import torch

from src.loop179.resource_cell import (
    IntegrityGateResult,
    ResourceCell,
    ResourceSample,
    assert_bitwise_deterministic,
    assert_budget_invariants,
    check_integrity,
)
from src.loop179.contracts import PHASE_A_GATE


def test_budget_invariants_pass() -> None:
    """Phase A 预算自检必须通过。"""

    assert_budget_invariants()


def test_resource_cell_accepts_within_budget_sample() -> None:
    """资源门必须接受预算内的采样。"""

    cell = ResourceCell()
    cell.inject_sample(ResourceSample(
        wall_seconds=1000.0,
        gpu_allocated_bytes=6_000_000_000,  # 6 GB < 6.5 GB
        rss_bytes=10_000_000_000,  # 10 GB < 11 GB
        epoch=1,
        step=100,
    ))
    assert cell.passed()
    assert len(cell.violations) == 0


def test_resource_cell_detects_gpu_over() -> None:
    """资源门必须检测 GPU 超限。"""

    cell = ResourceCell()
    cell.inject_sample(ResourceSample(
        wall_seconds=1000.0,
        gpu_allocated_bytes=PHASE_A_GATE.gpu_allocated_bytes + 1,
        rss_bytes=0,
        epoch=1,
        step=100,
    ))
    assert not cell.passed()
    kinds = {v.kind for v in cell.violations}
    assert "gpu_over" in kinds


def test_resource_cell_detects_rss_over() -> None:
    """资源门必须检测 RSS 超限。"""

    cell = ResourceCell()
    cell.inject_sample(ResourceSample(
        wall_seconds=1000.0,
        gpu_allocated_bytes=0,
        rss_bytes=PHASE_A_GATE.rss_bytes + 1,
        epoch=1,
        step=100,
    ))
    assert not cell.passed()
    kinds = {v.kind for v in cell.violations}
    assert "rss_over" in kinds


def test_resource_cell_detects_wall_over() -> None:
    """资源门必须检测 wall clock 超限。"""

    cell = ResourceCell()
    cell.inject_sample(ResourceSample(
        wall_seconds=float(PHASE_A_GATE.wall_seconds + 1),
        gpu_allocated_bytes=0,
        rss_bytes=0,
        epoch=12,
        step=10000,
    ))
    assert not cell.passed()
    kinds = {v.kind for v in cell.violations}
    assert "wall_over" in kinds


def test_resource_cell_records_integrity_violations() -> None:
    """资源门必须记录完整性违规。"""

    cell = ResourceCell()
    cell.record_integrity(kind="oom", detail="CUDA out of memory at epoch 3")
    cell.record_integrity(kind="nonfinite", detail="loss became NaN at step 500")
    assert not cell.passed()
    kinds = {v.kind for v in cell.violations}
    assert "oom" in kinds
    assert "nonfinite" in kinds


def test_resource_cell_build_receipt_has_correct_schema() -> None:
    """资源门 receipt 必须有正确的 schema。"""

    cell = ResourceCell()
    cell.inject_sample(ResourceSample(
        wall_seconds=500.0,
        gpu_allocated_bytes=5_000_000_000,
        rss_bytes=9_000_000_000,
        epoch=1,
        step=50,
    ))
    receipt = cell.build_receipt()
    assert receipt["loop_id"] == "Loop179"
    assert receipt["phase"] == "A"
    assert receipt["sample_count"] == 1
    assert receipt["violation_count"] == 0
    assert receipt["passed"] is True
    assert "budget" in receipt
    assert "violations" in receipt


def test_check_integrity_passes_on_clean_run() -> None:
    """完整性门必须接受干净的运行。"""

    result = check_integrity(rows_input=12000, rows_output=12000)
    assert result.passed()
    assert result.silent_drop_rows == 0
    assert result.all_rows_accounted


def test_check_integrity_detects_silent_drop() -> None:
    """完整性门必须检测 silent drop。"""

    result = check_integrity(rows_input=12000, rows_output=11999)
    assert not result.passed()
    assert result.silent_drop_rows == 1
    assert not result.all_rows_accounted


def test_check_integrity_detects_oom_and_timeout() -> None:
    """完整性门必须检测 OOM 和 timeout。"""

    result = check_integrity(
        rows_input=12000,
        rows_output=12000,
        oom=True,
        timeout=True,
    )
    assert not result.passed()
    assert result.oom
    assert result.timeout


def test_assert_bitwise_deterministic_passes_on_identical_tensors() -> None:
    """确定性验证必须接受完全相同的张量。"""

    torch.manual_seed(41)
    first = torch.randn(4, 2)
    second = first.clone()
    assert_bitwise_deterministic(first, second, label="test_logits")


def test_assert_bitwise_deterministic_rejects_different_tensors() -> None:
    """确定性验证必须拒绝不同的张量。"""

    torch.manual_seed(41)
    first = torch.randn(4, 2)
    torch.manual_seed(42)
    second = torch.randn(4, 2)
    with pytest.raises(AssertionError, match="determinism"):
        assert_bitwise_deterministic(first, second, label="test_logits")


def test_resource_cell_multiple_violations_all_recorded() -> None:
    """资源门必须记录所有违规，不只第一个。"""

    cell = ResourceCell()
    cell.inject_sample(ResourceSample(
        wall_seconds=float(PHASE_A_GATE.wall_seconds + 1000),
        gpu_allocated_bytes=PHASE_A_GATE.gpu_allocated_bytes + 1_000_000,
        rss_bytes=PHASE_A_GATE.rss_bytes + 1_000_000,
        epoch=12,
        step=99999,
    ))
    assert not cell.passed()
    kinds = {v.kind for v in cell.violations}
    assert "gpu_over" in kinds
    assert "rss_over" in kinds
    assert "wall_over" in kinds
