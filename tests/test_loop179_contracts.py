"""Loop179 contracts 冻结常量一致性测试。"""

from __future__ import annotations

import pytest

from src.loop179.contracts import (
    B0_FEATURE_DIM,
    BUCKET_COUNT,
    DROPOUT,
    EXPECTED_REGION_BYTES,
    EXPECTED_REGIONS,
    HGCONV_BLOCKS,
    HGCONV_FILTER_LENGTH,
    MODEL_DIM,
    PADDING_TOKEN,
    PATCH_SEQUENCE_LENGTH,
    PATCH_SIZE,
    PHASE0_SOURCE_WHITELIST,
    PHASE_A_GATE,
    PHASE_B_GATE,
    REGION_TYPE_COUNT,
    TRANSFORMER_FFN_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
    VOCABULARY_SIZE,
    assert_contract_invariants,
)


def test_contract_invariants_pass() -> None:
    """冻结常量自检必须通过。"""

    assert_contract_invariants()


def test_abi_shapes_match_model_default() -> None:
    """contracts 常量必须与 HGConvRegionConfig 默认值一致。"""

    from src.loop179.model import HGConvRegionConfig

    config = HGConvRegionConfig()
    assert config.expected_regions == EXPECTED_REGIONS
    assert config.expected_region_bytes == EXPECTED_REGION_BYTES
    assert config.vocabulary_size == VOCABULARY_SIZE
    assert config.padding_token == PADDING_TOKEN
    assert config.region_type_count == REGION_TYPE_COUNT
    assert config.bucket_count == BUCKET_COUNT
    assert config.b0_feature_dim == B0_FEATURE_DIM
    assert config.model_dim == MODEL_DIM
    assert config.hgconv_blocks == HGCONV_BLOCKS
    assert config.hgconv_filter_length == HGCONV_FILTER_LENGTH
    assert config.patch_size == PATCH_SIZE
    assert config.transformer_layers == TRANSFORMER_LAYERS
    assert config.transformer_heads == TRANSFORMER_HEADS
    assert config.transformer_ffn_dim == TRANSFORMER_FFN_DIM
    assert config.dropout == DROPOUT


def test_patch_sequence_length_is_512() -> None:
    """8192 / 16 = 512，patch 序列长度必须为 512。"""

    assert PATCH_SEQUENCE_LENGTH == 512
    assert PATCH_SEQUENCE_LENGTH == EXPECTED_REGION_BYTES // PATCH_SIZE


def test_phase_a_budget_covers_16000_train_rows() -> None:
    """Phase A fit + selection 必须覆盖 16000 Train 行，不触碰 fold0。"""

    assert PHASE_A_GATE.fit_rows == 12_000
    assert PHASE_A_GATE.selection_rows == 4_000
    assert PHASE_A_GATE.fit_rows + PHASE_A_GATE.selection_rows == 16_000
    assert PHASE_A_GATE.fold0_model_rows == 0


def test_phase_a_batch_contract() -> None:
    """effective_batch = microbatch * accumulation。"""

    assert PHASE_A_GATE.effective_batch == PHASE_A_GATE.microbatch * PHASE_A_GATE.accumulation
    assert PHASE_A_GATE.effective_batch == 32


def test_phase_a_resource_limits_within_loop175_bounds() -> None:
    """Phase A 资源上限必须在 Loop175 seed41 合同之内（留 5% 安全边际）。"""

    # Loop175 上限：GPU 6.98 GiB, RSS 11.81 GiB, wall 21600s
    assert PHASE_A_GATE.gpu_allocated_bytes <= 6_979_321_856
    assert PHASE_A_GATE.rss_bytes <= 11_811_160_064
    assert PHASE_A_GATE.wall_seconds <= 21_600


def test_phase_b_j_gate_requires_strict_causal_gain() -> None:
    """Phase B J 臂晋级门必须严格要求因果收益。"""

    assert PHASE_B_GATE.j_net_fewer_errors_vs_a >= 30
    assert PHASE_B_GATE.j_repairs_vs_a >= 50
    assert PHASE_B_GATE.j_override_precision >= 0.80
    assert PHASE_B_GATE.j_net_positive_folds >= 4
    assert PHASE_B_GATE.fp_relative_worsening <= 0.05
    assert PHASE_B_GATE.fn_relative_worsening <= 0.05


def test_phase_b_k_gate_requires_shuffle_harm() -> None:
    """Phase B K 臂必须证明收益来自区域归属（shuffle 后变差）。"""

    assert PHASE_B_GATE.k_more_errors_vs_j >= 30


def test_source_whitelist_contains_seven_phase0_files() -> None:
    """Phase 0 源码白名单必须包含 7 个文件。"""

    assert len(PHASE0_SOURCE_WHITELIST) == 7
    assert "src/loop179/contracts.py" in PHASE0_SOURCE_WHITELIST
    assert "src/loop179/hgconv.py" in PHASE0_SOURCE_WHITELIST
    assert "src/loop179/model.py" in PHASE0_SOURCE_WHITELIST
    assert "src/loop179/source_closure.py" in PHASE0_SOURCE_WHITELIST
    assert "src/loop179/data_adapter.py" in PHASE0_SOURCE_WHITELIST
    assert "src/loop179/resource_cell.py" in PHASE0_SOURCE_WHITELIST
    assert "src/loop179/__init__.py" in PHASE0_SOURCE_WHITELIST


def test_padding_token_is_last_vocab_item() -> None:
    """padding token 必须是词表最后一项。"""

    assert PADDING_TOKEN == VOCABULARY_SIZE - 1
    assert PADDING_TOKEN == 256
    assert VOCABULARY_SIZE == 257
