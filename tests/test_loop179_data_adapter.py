"""Loop179 data_adapter 数据适配器测试。"""

from __future__ import annotations

import pytest
import torch

from src.loop179.data_adapter import (
    PHASE_A_FIT_FOLDS,
    PHASE_A_FOLD_SPLIT,
    PHASE_A_FORBIDDEN_FOLDS,
    PHASE_A_SELECTION_FOLD,
    PhaseADataLoader,
    make_synthetic_batch,
    validate_b0_features_shape,
    validate_metadata_ranges,
    validate_metadata_shape,
    validate_region_tokens_shape,
    validate_token_ranges,
)


def test_phase_a_fold_split_is_frozen() -> None:
    """Phase A fold 划分必须冻结为 fits=2/3/4, selection=1, forbidden=0。"""

    assert PHASE_A_FIT_FOLDS == (2, 3, 4)
    assert PHASE_A_SELECTION_FOLD == 1
    assert PHASE_A_FORBIDDEN_FOLDS == (0,)
    PHASE_A_FOLD_SPLIT.validate()


def test_phase_a_fold_split_rejects_forbidden_overlap() -> None:
    """fold 划分若 fit 与 forbidden 重叠必须报错。"""

    from src.loop179.data_adapter import FoldSplit

    with pytest.raises(ValueError, match="forbidden"):
        FoldSplit(
            fit_folds=(0, 1, 2),
            selection_fold=3,
            forbidden_folds=(0,),
            fit_rows_expected=12_000,
            selection_rows_expected=4_000,
        ).validate()


def test_synthetic_batch_has_frozen_abi() -> None:
    """合成 batch 必须符合 [B, 16, 8192] ABI。"""

    batch = make_synthetic_batch(batch_size=4, seed=41)
    assert batch["region_tokens"].shape == (4, 16, 8192)
    assert batch["region_lengths"].shape == (4, 16)
    assert batch["region_types"].shape == (4, 16)
    assert batch["offset_buckets"].shape == (4, 16)
    assert batch["length_buckets"].shape == (4, 16)
    assert batch["b0_features"].shape == (4, 571)
    assert batch["labels"].shape == (4,)
    assert batch["row_ids"].shape == (4,)


def test_synthetic_batch_respects_padding_contract() -> None:
    """length=0 的 region 必须 type=0, offset=0, length_bucket=0。"""

    batch = make_synthetic_batch(batch_size=8, seed=42)
    lengths = batch["region_lengths"]
    types = batch["region_types"]
    offsets = batch["offset_buckets"]
    length_buckets = batch["length_buckets"]

    missing = lengths == 0
    assert torch.all(types[missing] == 0), "missing regions must have type=0"
    assert torch.all(offsets[missing] == 0), "missing regions must have offset=0"
    assert torch.all(length_buckets[missing] == 0), "missing regions must have length_bucket=0"


def test_synthetic_batch_pads_beyond_length() -> None:
    """length 以外的 token 必须 = 256 (padding_token)。"""

    from src.loop179.contracts import PADDING_TOKEN

    batch = make_synthetic_batch(batch_size=2, seed=43)
    tokens = batch["region_tokens"]
    lengths = batch["region_lengths"]

    positions = torch.arange(tokens.shape[-1]).view(1, 1, -1)
    beyond = positions >= lengths.unsqueeze(-1)
    assert torch.all(tokens[beyond] == PADDING_TOKEN), "beyond-length tokens must be padding"


def test_validate_region_tokens_rejects_wrong_shape() -> None:
    """形状验证必须拒绝错误的 region_tokens。"""

    wrong = torch.zeros(2, 8, 8192, dtype=torch.int64)
    with pytest.raises(ValueError, match="dim 1 must be 16"):
        validate_region_tokens_shape(wrong)

    wrong_dtype = torch.zeros(2, 16, 8192, dtype=torch.float32)
    with pytest.raises(ValueError, match="integer dtype"):
        validate_region_tokens_shape(wrong_dtype)


def test_validate_metadata_rejects_wrong_shape() -> None:
    """形状验证必须拒绝错误的 metadata。"""

    wrong = torch.zeros(2, 8, dtype=torch.int64)
    with pytest.raises(ValueError, match="dim 1 must be 16"):
        validate_metadata_shape(wrong, "region_lengths")


def test_validate_b0_rejects_wrong_dim() -> None:
    """B0 特征必须 = 571 维。"""

    wrong = torch.zeros(2, 100, dtype=torch.float32)
    with pytest.raises(ValueError, match="dim 1 must be 571"):
        validate_b0_features_shape(wrong)


def test_validate_token_ranges_rejects_out_of_range() -> None:
    """token 范围验证必须拒绝 > 256 的值。"""

    tokens = torch.zeros(1, 16, 8192, dtype=torch.int64)
    tokens[0, 0, 0] = 257
    with pytest.raises(ValueError, match="out-of-range"):
        validate_token_ranges(tokens)


def test_validate_metadata_ranges_rejects_out_of_range() -> None:
    """metadata 范围验证必须拒绝越界值。"""

    types = torch.zeros(1, 16, dtype=torch.int64)
    types[0, 0] = 6  # REGION_TYPE_COUNT = 6, 越界
    offsets = torch.zeros(1, 16, dtype=torch.int64)
    length_buckets = torch.zeros(1, 16, dtype=torch.int64)
    with pytest.raises(ValueError, match="out-of-range"):
        validate_metadata_ranges(types, offsets, length_buckets)


def test_phase_a_data_loader_rejects_fold0() -> None:
    """PhaseADataLoader 必须拒绝 fold0。"""

    loader = PhaseADataLoader()
    with pytest.raises(ValueError, match="forbidden"):
        loader.assert_not_forbidden_fold(0)


def test_phase_a_data_loader_validates_synthetic_batch() -> None:
    """PhaseADataLoader.validate_batch 必须接受合法合成 batch。"""

    loader = PhaseADataLoader()
    batch = make_synthetic_batch(batch_size=2, seed=41)
    loader.validate_batch(batch)  # 不抛异常


def test_phase_a_data_loader_load_rows_not_implemented() -> None:
    """PhaseADataLoader._load_region_cache_rows 在 Phase 0 必须抛 NotImplementedError。"""

    loader = PhaseADataLoader()
    with pytest.raises(NotImplementedError, match="A2"):
        loader._load_region_cache_rows((2, 3, 4))


def test_synthetic_batch_without_b0() -> None:
    """make_synthetic_batch(include_b0=False) 必须 b0=None。"""

    batch = make_synthetic_batch(batch_size=1, include_b0=False)
    assert batch["b0_features"] is None
