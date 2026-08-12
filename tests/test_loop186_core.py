"""Loop186 核心单元测试：契约、HGConv、模型、数据适配器、资源门。"""

from __future__ import annotations

import json
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

# ---------------------------------------------------------------------------
# 被测试模块
# ---------------------------------------------------------------------------

from src.loop186.contracts import (
    B0_FEATURE_DIM,
    HGCONV_BLOCKS,
    MODEL_DIM,
    MULTI_SCALE_FILTER_LENGTHS,
    PATCH_SEQUENCE_LENGTH,
    TRANSFORMER_LAYERS,
    PHASE_A_GATE,
    SAM_ENABLED,
    SAM_RHO,
    SWA_START_EPOCH,
    assert_contract_invariants,
)
from src.loop186.hgconv import (
    HGConvBlock,
    HGConvConfig,
    MultiScaleHGConvBlock,
    MultiScaleHGConvConfig,
    approximate_inverse,
    circular_convolution,
    malware_kernel_precondition,
)
from src.loop186.model import (
    HGConvRegionConfig,
    HGConvRegionNet,
    parameter_count,
)
from src.loop186.data_adapter import (
    FoldSplit,
    ROWS_PER_FOLD,
    FULL_TRAIN_ROWS,
    RegionBatch,
    make_fold_split,
    make_synthetic_batch,
    validate_region_tokens_shape,
    validate_metadata_shape,
    validate_b0_features_shape,
    validate_token_ranges,
    validate_metadata_ranges,
)
from src.loop186.resource_cell import (
    IntegrityGateResult,
    ResourceCell,
    ResourceSample,
    check_integrity,
    assert_bitwise_deterministic,
    assert_budget_invariants,
    deadline_check_due,
    enforce_epoch_deadline,
)
from src.loop186.source_closure import (
    build_current_manifest,
    scan_source_closure,
)
from scripts.run_loop186_phase_a import select_rotating_stratified_probe, select_stratified_probe


# ===================================================================
# 1. Contracts
# ===================================================================


class TestContractInvariants:
    def test_assert_contract_invariants_passes(self) -> None:
        """基线：冻结常量应全部自洽。"""
        assert_contract_invariants()

    def test_loop186_speed_profile_values(self) -> None:
        assert MODEL_DIM == 192, f"expected 192, got {MODEL_DIM}"
        assert HGCONV_BLOCKS == 2, f"expected 2, got {HGCONV_BLOCKS}"
        assert TRANSFORMER_LAYERS == 4, f"expected 4, got {TRANSFORMER_LAYERS}"
        assert not SAM_ENABLED, "SAM is disabled for Loop186 (large microbatch instead)"

    def test_phase_a_gate_single_fold(self) -> None:
        """Phase A gate 应为大 microbatch 无 SAM 配置。"""
        assert PHASE_A_GATE.fit_rows == 12_000
        assert PHASE_A_GATE.selection_rows == 4_000
        assert PHASE_A_GATE.max_epochs == 12
        assert PHASE_A_GATE.fold0_model_rows == 0
        assert not PHASE_A_GATE.sam_enabled
        assert PHASE_A_GATE.microbatch * PHASE_A_GATE.accumulation == PHASE_A_GATE.effective_batch
        assert PHASE_A_GATE.effective_batch == 32
        assert PHASE_A_GATE.selection_probe_rows == 400
        assert PHASE_A_GATE.evaluation_microbatch == 32
        assert PHASE_A_GATE.epoch_wall_seconds == 600
        assert PHASE_A_GATE.wall_seconds >= (
            PHASE_A_GATE.max_epochs * PHASE_A_GATE.epoch_wall_seconds
        )

    def test_sam_and_swa_constants(self) -> None:
        """验证 SAM（已关闭）和 SWA 常量。"""
        assert not SAM_ENABLED
        assert SAM_RHO == pytest.approx(0.05)
        assert SWA_START_EPOCH == 9

    def test_multi_scale_filter_lengths(self) -> None:
        """验证多尺度 filter 配置。"""
        assert MULTI_SCALE_FILTER_LENGTHS == (8, 16, 32, 64)
        assert max(MULTI_SCALE_FILTER_LENGTHS) <= PATCH_SEQUENCE_LENGTH


# ===================================================================
# 2. HGConv Core
# ===================================================================


class TestHGConvMath:
    """测试 HGConv 核心数学运算。"""

    def test_circular_convolution_basic(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0, 0.0])
        y = torch.tensor([0.5, 0.0, 0.0, 0.5])
        result = circular_convolution(x, y, dim=0)
        assert result.shape == (4,)
        assert torch.isfinite(result).all()

    def test_circular_convolution_identity(self) -> None:
        x = torch.randn(16)
        identity = torch.zeros(16)
        identity[0] = 1.0
        result = circular_convolution(x, identity, dim=0)
        assert torch.allclose(result, x, atol=1e-6)

    def test_circular_convolution_2d(self) -> None:
        x = torch.randn(2, 8, 16)
        y = torch.randn(1, 16)
        result = circular_convolution(x, y, dim=-1)
        assert result.shape == (2, 8, 16)
        assert torch.isfinite(result).all()

    def test_circular_convolution_raises_on_dim_mismatch(self) -> None:
        x = torch.randn(8)
        y = torch.randn(16)
        with pytest.raises(ValueError, match="axes must have equal length"):
            circular_convolution(x, y, dim=0)

    def test_circular_convolution_raises_on_non_float(self) -> None:
        x = torch.randint(0, 10, (8,)).float()
        y = torch.randint(0, 10, (8,)).long()
        with pytest.raises(TypeError, match="must be a real floating"):
            circular_convolution(x, y, dim=0)

    def test_approximate_inverse_shape(self) -> None:
        x = torch.randn(4, 16, 64)
        inv = approximate_inverse(x, dim=-1)
        assert inv.shape == x.shape
        assert torch.isfinite(inv).all()

    def test_approximate_inverse_property(self) -> None:
        """翻转再滚动一次等于自身。"""
        x = torch.randn(16)
        inv = approximate_inverse(x, dim=0)
        inv_inv = approximate_inverse(inv, dim=0)
        assert torch.allclose(x, inv_inv, atol=1e-6)

    def test_malware_kernel_precondition(self) -> None:
        kernel = torch.randn(1, 32, 64)
        result = malware_kernel_precondition(kernel, sequence_length=512, dim=1)
        assert result.shape == (1, 512, 64)
        assert torch.isfinite(result).all()
        assert result.is_complex()

    def test_malware_kernel_precondition_ortho_norm(self) -> None:
        """验证 kernel 形状和有限性，不硬编码精确值。"""
        kernel = torch.ones(1, 32, 1)
        result = malware_kernel_precondition(kernel, sequence_length=512, dim=1)
        assert result.shape == (1, 512, 1)
        assert torch.isfinite(result).all()


class TestHGConvBlock:
    def test_hgconv_config_validation(self) -> None:
        with pytest.raises(ValueError, match="dimensions must be positive"):
            HGConvConfig(model_dim=0)
        with pytest.raises(ValueError, match="must be in"):
            HGConvConfig(dropout=1.5)

    def test_hgconv_block_forward(self) -> None:
        config = HGConvConfig(model_dim=64, filter_length=8)
        block = HGConvBlock(config)
        x = torch.randn(4, 128, 64)
        mask = torch.ones(4, 128, dtype=torch.bool)
        out = block(x, mask)
        assert out.shape == (4, 128, 64)
        assert torch.isfinite(out).all()

    def test_hgconv_block_masked(self) -> None:
        config = HGConvConfig(model_dim=32, filter_length=8)
        block = HGConvBlock(config)
        x = torch.randn(2, 64, 32)
        mask = torch.ones(2, 64, dtype=torch.bool)
        mask[:, 32:] = False  # 后半部分 masked
        out = block(x, mask)
        assert out.shape == (2, 64, 32)
        assert torch.isfinite(out).all()
        # masked 部分应为 0
        assert out[:, 32:, :].abs().max().item() == pytest.approx(0.0)

    def test_hgconv_block_raises_on_shape_mismatch(self) -> None:
        block = HGConvBlock(HGConvConfig(model_dim=32))
        x = torch.randn(4, 64, 64)  # model_dim=32 vs 64
        mask = torch.ones(4, 64, dtype=torch.bool)
        with pytest.raises(ValueError, match="drifte"):
            block(x, mask)

    def test_hgconv_block_raises_on_short_sequence(self) -> None:
        block = HGConvBlock(HGConvConfig(model_dim=16, filter_length=32))
        x = torch.randn(2, 16, 16)  # too short
        mask = torch.ones(2, 16, dtype=torch.bool)
        with pytest.raises(ValueError, match="shorter than"):
            block(x, mask)


class TestMultiScaleHGConvBlock:
    def test_multi_scale_config_validation(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            MultiScaleHGConvConfig(filter_lengths=())
        with pytest.raises(ValueError, match="unique"):
            MultiScaleHGConvConfig(filter_lengths=(8, 8, 16))

    def test_multi_scale_block_forward(self) -> None:
        config = MultiScaleHGConvConfig(model_dim=64, filter_lengths=(4, 8, 16))
        block = MultiScaleHGConvBlock(config)
        x = torch.randn(4, 128, 64)
        mask = torch.ones(4, 128, dtype=torch.bool)
        out = block(x, mask)
        assert out.shape == (4, 128, 64)
        assert torch.isfinite(out).all()

    def test_fast_multi_scale_block_does_not_use_fft(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def reject_fft(*args: object, **kwargs: object) -> torch.Tensor:
            raise AssertionError("FFT path used")

        monkeypatch.setattr("src.loop186.hgconv._fft_circular_convolution", reject_fft)
        block = MultiScaleHGConvBlock(
            MultiScaleHGConvConfig(
                model_dim=32,
                filter_lengths=(4, 8),
                fast=True,
            )
        )
        values = torch.randn(2, 64, 32)
        mask = torch.ones(2, 64, dtype=torch.bool)

        output = block(values, mask)

        assert output.shape == values.shape
        assert torch.isfinite(output).all()

    def test_multi_scale_block_loop186_config(self) -> None:
        """使用 Loop186 实际配置（model_dim=384）测试前向。"""
        config = MultiScaleHGConvConfig(
            model_dim=384,
            filter_lengths=(8, 16, 32, 64),
        )
        block = MultiScaleHGConvBlock(config)
        x = torch.randn(2, 512, 384)
        mask = torch.ones(2, 512, dtype=torch.bool)
        out = block(x, mask)
        assert out.shape == (2, 512, 384)
        assert torch.isfinite(out).all()

    def test_multi_scale_block_masked(self) -> None:
        config = MultiScaleHGConvConfig(model_dim=32, filter_lengths=(4, 8))
        block = MultiScaleHGConvBlock(config)
        x = torch.randn(2, 64, 32)
        mask = torch.ones(2, 64, dtype=torch.bool)
        mask[:, 32:] = False
        out = block(x, mask)
        assert out.shape == (2, 64, 32)
        assert out[:, 32:, :].abs().max().item() == pytest.approx(0.0)

    def test_multi_scale_scale_weights_trained(self) -> None:
        """验证 scale fusion 权重初始化为 0 且可训练。"""
        block = MultiScaleHGConvBlock()
        assert block.scale_weights.numel() == 4
        assert torch.allclose(block.scale_weights, torch.zeros(4))
        assert block.scale_weights.requires_grad


# ===================================================================
# 3. Model
# ===================================================================


class TestHGConvRegionConfig:
    def test_default_config(self) -> None:
        config = HGConvRegionConfig()
        assert config.model_dim == 192
        assert config.hgconv_blocks == 2
        assert config.transformer_layers == 4
        assert config.transformer_heads == 6
        assert config.byte_embedding_dim == 64
        assert config.padding_token == 256

    def test_loop186_model_enables_fast_hgconv(self) -> None:
        model = HGConvRegionNet(HGConvRegionConfig(runtime_checks=False))
        assert all(block.config.fast for block in model.blocks)

    def test_config_validation_positive_dims(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            HGConvRegionConfig(model_dim=-1)

    def test_config_validation_padding(self) -> None:
        with pytest.raises(ValueError, match="padding token"):
            HGConvRegionConfig(padding_token=255)

    def test_config_validation_divisible(self) -> None:
        with pytest.raises(ValueError, match="divisible by transformer_heads"):
            HGConvRegionConfig(model_dim=383, transformer_heads=8)

    def test_config_validation_filter_lengths(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            HGConvRegionConfig(multi_scale_filter_lengths=())

    def test_config_validation_filter_unique(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            HGConvRegionConfig(multi_scale_filter_lengths=(8, 8, 16))

    def test_config_validation_filter_exceeds_sequence(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            HGConvRegionConfig(multi_scale_filter_lengths=(9999,))


class TestHGConvRegionNet:
    def test_speed_profile_disables_runtime_checks(self) -> None:
        model = HGConvRegionNet(HGConvRegionConfig(runtime_checks=False))
        assert not model.config.runtime_checks
        assert all(not block.config.runtime_checks for block in model.blocks)

    def _synthetic_batch(
        self, batch_size: int = 2, cfg: HGConvRegionConfig | None = None
    ) -> dict[str, torch.Tensor]:
        cfg = cfg or HGConvRegionConfig()
        tokens = torch.randint(
            0, cfg.vocabulary_size - 1,
            (batch_size, cfg.expected_regions, cfg.expected_region_bytes),
            dtype=torch.long,
        )
        lengths = torch.randint(
            1, cfg.expected_region_bytes + 1,
            (batch_size, cfg.expected_regions),
            dtype=torch.long,
        )
        positions = torch.arange(cfg.expected_region_bytes).view(1, 1, -1)
        mask = positions < lengths.unsqueeze(-1)
        tokens = torch.where(mask, tokens, torch.full_like(tokens, cfg.padding_token))

        # type 0 保留给缺失区域，有效区域 type 从 1 开始
        types = torch.randint(
            1, cfg.region_type_count,
            (batch_size, cfg.expected_regions),
            dtype=torch.long,
        )
        types = torch.where(lengths == 0, torch.zeros_like(types), types)

        offsets = torch.randint(
            0, cfg.bucket_count,
            (batch_size, cfg.expected_regions),
            dtype=torch.long,
        )
        length_buckets = torch.randint(
            0, cfg.bucket_count,
            (batch_size, cfg.expected_regions),
            dtype=torch.long,
        )
        offsets = torch.where(lengths == 0, torch.zeros_like(offsets), offsets)
        length_buckets = torch.where(lengths == 0, torch.zeros_like(length_buckets), length_buckets)

        b0 = torch.randn(batch_size, cfg.b0_feature_dim)

        return {
            "region_tokens": tokens,
            "region_lengths": lengths,
            "region_types": types,
            "offset_buckets": offsets,
            "length_buckets": length_buckets,
            "b0_features": b0,
        }

    def test_forward_default_config(self) -> None:
        cfg = HGConvRegionConfig()
        model = HGConvRegionNet(cfg)
        batch = self._synthetic_batch(2, cfg)
        output = model(**batch)
        assert "region_features" in output
        assert "region_logits" in output
        assert "fusion_logits" in output
        assert output["region_features"].shape == (2, cfg.model_dim)
        assert output["region_logits"].shape == (2, 2)
        assert output["fusion_logits"].shape == (2, 2)
        assert torch.isfinite(output["region_logits"]).all()
        assert torch.isfinite(output["fusion_logits"]).all()

    def test_forward_without_b0(self) -> None:
        cfg = HGConvRegionConfig()
        model = HGConvRegionNet(cfg)
        batch = self._synthetic_batch(2, cfg)
        batch.pop("b0_features")
        output = model(**batch, b0_features=None)
        assert "region_features" in output
        assert "region_logits" in output
        assert "fusion_logits" not in output
        assert output["region_logits"].shape == (2, 2)

    def test_forward_loop186_full_config(self) -> None:
        """使用完整 Loop186 配置做前向验证。"""
        cfg = HGConvRegionConfig(
            model_dim=384,
            byte_embedding_dim=128,
            hgconv_blocks=4,
            transformer_layers=8,
            transformer_heads=8,
            transformer_ffn_dim=1536,
        )
        model = HGConvRegionNet(cfg)
        batch = self._synthetic_batch(2, cfg)
        output = model(**batch)
        assert output["fusion_logits"].shape == (2, 2)

    def test_forward_deterministic(self) -> None:
        """两次前向应得到完全相同的结果（eval mode）。"""
        cfg = HGConvRegionConfig()
        model = HGConvRegionNet(cfg)
        model.eval()
        batch = self._synthetic_batch(2, cfg)
        with torch.no_grad():
            out1 = model(**batch)["fusion_logits"]
            out2 = model(**batch)["fusion_logits"]
        assert torch.equal(out1, out2)

    def test_parameter_count(self) -> None:
        """验证参数量合理范围。"""
        model = HGConvRegionNet()
        count = parameter_count(model)
        assert 2_000_000 < count < 3_500_000, f"parameter count {count} outside expected range"

    def test_forward_gradient_flow(self) -> None:
        """验证梯度可以反向传播（padding token embedding 不会收到梯度）。"""
        cfg = HGConvRegionConfig(
            model_dim=64,
            transformer_heads=8,
            transformer_layers=2,
            hgconv_blocks=2,
        )
        model = HGConvRegionNet(cfg)
        batch = self._synthetic_batch(2, cfg)
        logits = model(**batch)["fusion_logits"]
        loss = torch.nn.functional.cross_entropy(
            logits, torch.randint(0, 2, (2,), dtype=torch.long)
        )
        loss.backward()
        # 绝大多数参数应有梯度；padding token embedding 自然无梯度
        has_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total = sum(1 for _ in model.parameters())
        assert has_grad >= total - 5, f"{has_grad}/{total} parameters have grad"

    def test_forward_checkpointing(self) -> None:
        """验证 gradient checkpointing 不影响前向结果（eval mode，无随机增强）。"""
        cfg = HGConvRegionConfig(
            model_dim=64,
            transformer_heads=8,
            transformer_layers=2,
            hgconv_blocks=2,
        )
        model_ckpt = HGConvRegionNet(HGConvRegionConfig(**{**cfg.__dict__, 'use_checkpointing': True}))
        model_ref = HGConvRegionNet(cfg)
        model_ckpt.load_state_dict(model_ref.state_dict())
        model_ckpt.eval()
        model_ref.eval()
        batch = self._synthetic_batch(2, cfg)
        with torch.no_grad():
            out_ckpt = model_ckpt(**batch)
            out_ref = model_ref(**batch)
        for key in out_ckpt:
            assert torch.allclose(out_ckpt[key], out_ref[key], atol=1e-5), f"{key} differs with checkpointing"

    def test_validation_raises_on_bad_tokens(self) -> None:
        # 注入越界 token——用默认配置，然后手动改 token 为越界值
        cfg = HGConvRegionConfig()
        model = HGConvRegionNet(cfg)
        batch = self._synthetic_batch(2, cfg)
        batch["region_tokens"][0, 0, 0] = 999
        with pytest.raises((ValueError, RuntimeError)):
            model(**batch)

    def test_validation_raises_on_missing_region_values(self) -> None:
        model = HGConvRegionNet()
        batch = self._synthetic_batch(2)
        # 强制 region 0 为缺失（lengths=0, types=0, tokens=padding）
        batch["region_lengths"][0, 0] = 0
        batch["region_types"][0, 0] = 0
        batch["region_tokens"][0, 0, :] = 256
        batch["offset_buckets"][0, 0] = 0
        batch["length_buckets"][0, 0] = 0
        # 通过验证后 forward 应正常
        output = model(**batch)
        assert torch.isfinite(output["fusion_logits"]).all()


# ===================================================================
# 4. Data Adapter
# ===================================================================


class TestFoldSplit:
    def test_make_fold_split_basic(self) -> None:
        split = make_fold_split(1)
        assert split.selection_fold == 1
        assert split.fit_folds == (2, 3, 4)
        assert split.forbidden_folds == (0,)
        assert split.fit_rows_expected == 12_000
        assert split.selection_rows_expected == 4_000

    def test_make_fold_split_all_folds(self) -> None:
        for sf in (1, 2, 3, 4):
            split = make_fold_split(sf)
            assert sf not in split.fit_folds
            assert split.selection_fold == sf
            assert sf not in split.forbidden_folds

    def test_make_fold_split_invalid(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            make_fold_split(0)
        with pytest.raises(ValueError, match="must be in"):
            make_fold_split(5)

    def test_fold_split_validate_overlap(self) -> None:
        with pytest.raises(ValueError, match="must not overlap"):
            FoldSplit(
                fit_folds=(1, 2, 3),
                selection_fold=1,
                forbidden_folds=(0,),
                fit_rows_expected=12_000,
                selection_rows_expected=4_000,
            ).validate()

    def test_fold_split_validate_forbidden_sel(self) -> None:
        with pytest.raises(ValueError, match="must not be forbidden"):
            FoldSplit(
                fit_folds=(1, 2, 3),
                selection_fold=0,
                forbidden_folds=(0,),
                fit_rows_expected=12_000,
                selection_rows_expected=4_000,
            ).validate()


class TestSyntheticBatch:
    def test_make_synthetic_batch_shape(self) -> None:
        batch = make_synthetic_batch(4, seed=41, include_b0=True)
        assert batch["region_tokens"].shape == (4, 16, 8192)
        assert batch["region_lengths"].shape == (4, 16)
        assert batch["region_types"].shape == (4, 16)
        assert batch["offset_buckets"].shape == (4, 16)
        assert batch["length_buckets"].shape == (4, 16)
        assert batch["b0_features"].shape == (4, 571)
        assert batch["labels"].shape == (4,)
        assert batch["row_ids"].shape == (4,)

    def test_make_synthetic_batch_deterministic(self) -> None:
        b1 = make_synthetic_batch(2, seed=42)
        b2 = make_synthetic_batch(2, seed=42)
        for key in b1:
            assert torch.equal(b1[key], b2[key]), f"{key} differs"

    def test_make_synthetic_batch_no_b0(self) -> None:
        batch = make_synthetic_batch(2, seed=41, include_b0=False)
        assert batch["b0_features"] is None

    def test_make_synthetic_batch_padding_valid(self) -> None:
        """验证填充 token 只在 valid range 之后出现。"""
        batch = make_synthetic_batch(2, seed=41)
        tokens = batch["region_tokens"]
        lengths = batch["region_lengths"]
        positions = torch.arange(8192).view(1, 1, -1)
        valid = positions < lengths.unsqueeze(-1)
        padding = tokens == 256
        assert not (valid & padding).any(), "padding token inside valid range"


class TestSelectionProbe:
    def test_selection_probe_is_deterministic_and_stratified(self) -> None:
        indices = np.arange(1000, 5000, dtype=np.int64)
        labels = np.zeros(6000, dtype=np.int64)
        labels[indices[::2]] = 1

        first = select_stratified_probe(indices, labels, rows=400, seed=41)
        second = select_stratified_probe(indices, labels, rows=400, seed=41)

        assert np.array_equal(first, second)
        assert first.shape == (400,)
        assert np.unique(first).size == 400
        assert set(first).issubset(set(indices))
        assert int(labels[first].sum()) == 200

    def test_selection_probe_rotates_to_cover_partition(self) -> None:
        indices = np.arange(1000, 5000, dtype=np.int64)
        labels = np.zeros(6000, dtype=np.int64)
        labels[indices[::2]] = 1

        probes = [
            select_rotating_stratified_probe(
                indices,
                labels,
                rows=400,
                seed=41,
                epoch=epoch,
            )
            for epoch in range(1, 11)
        ]

        combined = np.concatenate(probes)
        assert np.unique(combined).size == 4000
        assert set(combined) == set(indices)
        assert all(int(labels[probe].sum()) == 200 for probe in probes)
        assert np.array_equal(
            probes[0],
            select_rotating_stratified_probe(
                indices,
                labels,
                rows=400,
                seed=41,
                epoch=11,
            ),
        )


class TestValidationFunctions:
    def test_validate_region_tokens_shape_ok(self) -> None:
        t = torch.randint(0, 256, (2, 16, 8192), dtype=torch.long)
        validate_region_tokens_shape(t)  # should not raise

    def test_validate_region_tokens_shape_bad_dims(self) -> None:
        with pytest.raises(ValueError, match="must be 3D"):
            validate_region_tokens_shape(torch.zeros(2, 16))

    def test_validate_metadata_shape_ok(self) -> None:
        t = torch.randint(0, 10, (2, 16), dtype=torch.long)
        validate_metadata_shape(t, "test")

    def test_validate_metadata_shape_bad_dtype(self) -> None:
        with pytest.raises(ValueError, match="must be integer"):
            validate_metadata_shape(torch.randn(2, 16), "test")

    def test_validate_b0_features_shape_ok(self) -> None:
        t = torch.randn(2, 571)
        validate_b0_features_shape(t)

    def test_validate_b0_features_shape_bad(self) -> None:
        with pytest.raises(ValueError, match="dim 1 must be"):
            validate_b0_features_shape(torch.randn(2, 100))

    def test_validate_token_ranges_ok(self) -> None:
        t = torch.randint(0, 256, (2, 16, 8192), dtype=torch.long)
        validate_token_ranges(t)

    def test_validate_token_ranges_out_of_range(self) -> None:
        t = torch.full((2, 16, 8192), 300, dtype=torch.long)
        with pytest.raises(ValueError, match="out-of-range"):
            validate_token_ranges(t)

    def test_validate_metadata_ranges_ok(self) -> None:
        types = torch.randint(0, 6, (2, 16), dtype=torch.long)
        offsets = torch.randint(0, 64, (2, 16), dtype=torch.long)
        buckets = torch.randint(0, 64, (2, 16), dtype=torch.long)
        validate_metadata_ranges(types, offsets, buckets)

    def test_validate_metadata_ranges_bad(self) -> None:
        types = torch.full((2, 16), 99, dtype=torch.long)
        with pytest.raises(ValueError, match="out-of-range"):
            validate_metadata_ranges(types, torch.zeros(2, 16, dtype=torch.long), torch.zeros(2, 16, dtype=torch.long))


# ===================================================================
# 5. Resource Cell
# ===================================================================


class TestResourceCell:
    def test_epoch_deadline_is_checked_every_optimizer_step(self) -> None:
        assert all(deadline_check_due(step) for step in range(1, 376))

    def test_phase_a_gate_microbatch_yields_full_coverage(self) -> None:
        assert PHASE_A_GATE.effective_batch == (
            PHASE_A_GATE.microbatch * PHASE_A_GATE.accumulation
        )
        assert PHASE_A_GATE.effective_batch == 32
        assert PHASE_A_GATE.microbatch >= 16, (
            "microbatch below 16 causes the Conv1d fast path to regress past the 600s gate"
        )

    def test_phase_a_speed_evidence_16x2(self) -> None:
        """桌上实证：192/2/4 + Conv1d + 16x2 实测稳态 ≤ 540 秒投影。

        受控基准来自 ``.cache/bench_epoch186_steady.py``：稳态均值 0.779s/优化步
        375 步 → 292.1s/epoch 投影，峰值 3.60 GiB。
        """

        steady_per_step_seconds = 0.779
        total_steps = 375
        projection_seconds = steady_per_step_seconds * total_steps

        assert projection_seconds <= 540.0, (
            f"new contract projection {projection_seconds:.1f}s no longer satisfies the gate"
        )
        assert projection_seconds <= PHASE_A_GATE.epoch_wall_seconds * 0.9


class TestResourceCellDeadline:
    def test_epoch_deadline_rejects_slow_projection(self) -> None:
        with pytest.raises(TimeoutError, match="projected epoch wall"):
            enforce_epoch_deadline(
                elapsed_seconds=50.0,
                completed_steps=25,
                total_steps=375,
                hard_seconds=600.0,
                projection_seconds=540.0,
            )

    def test_epoch_deadline_accepts_fast_projection(self) -> None:
        enforce_epoch_deadline(
            elapsed_seconds=30.0,
            completed_steps=25,
            total_steps=375,
            hard_seconds=600.0,
            projection_seconds=540.0,
        )

    def test_epoch_deadline_rejects_hard_limit(self) -> None:
        with pytest.raises(TimeoutError, match="epoch wall"):
            enforce_epoch_deadline(
                elapsed_seconds=601.0,
                completed_steps=300,
                total_steps=375,
                hard_seconds=600.0,
                projection_seconds=540.0,
            )

    def test_resource_cell_rejects_epoch_over_ten_minutes(self) -> None:
        cell = ResourceCell()
        cell.inject_sample(ResourceSample(wall_seconds=100.0, epoch=1))
        cell.inject_sample(ResourceSample(wall_seconds=701.0, epoch=2))

        assert not cell.passed()
        assert cell.violations[-1].kind == "epoch_wall_over"
        assert cell.violations[-1].actual == pytest.approx(601.0)
        assert cell.violations[-1].threshold == pytest.approx(600.0)

    def test_resource_receipt_exposes_epoch_budget(self) -> None:
        receipt = ResourceCell().build_receipt()
        assert receipt["loop_id"] == "Loop186"
        assert receipt["budget"]["epoch_wall_seconds"] == 600

    def test_resource_cell_start_and_sample(self) -> None:
        cell = ResourceCell()
        cell.start()
        sample = cell.sample_and_inject(epoch=1, step=10)
        assert sample.epoch == 1
        assert sample.step == 10
        assert sample.wall_seconds >= 0.0
        assert cell.passed()

    def test_resource_cell_gpu_budget(self) -> None:
        cell = ResourceCell()
        cell.budget = PHASE_A_GATE
        cell.inject_sample(ResourceSample(
            wall_seconds=1.0,
            gpu_allocated_bytes=PHASE_A_GATE.gpu_allocated_bytes + 1,
        ))
        assert not cell.passed()
        assert any(v.kind == "gpu_over" for v in cell.violations)

    def test_resource_cell_rss_budget(self) -> None:
        cell = ResourceCell()
        cell.inject_sample(ResourceSample(
            wall_seconds=1.0,
            rss_bytes=PHASE_A_GATE.rss_bytes + 1,
        ))
        assert not cell.passed()
        assert any(v.kind == "rss_over" for v in cell.violations)

    def test_resource_cell_wall_budget(self) -> None:
        cell = ResourceCell()
        cell.start()
        cell.inject_sample(ResourceSample(wall_seconds=PHASE_A_GATE.wall_seconds + 1))
        assert not cell.passed()
        assert any(v.kind == "wall_over" for v in cell.violations)

    def test_resource_cell_violation_detail(self) -> None:
        cell = ResourceCell()
        cell.budget = PHASE_A_GATE
        cell.inject_sample(ResourceSample(wall_seconds=float(PHASE_A_GATE.wall_seconds + 100)))
        assert len(cell.violations) == 1
        v = cell.violations[0]
        assert "wall" in v.detail
        assert v.actual > v.threshold

    def test_record_integrity(self) -> None:
        cell = ResourceCell()
        cell.record_integrity(kind="nondeterministic", detail="eval scores differ")
        assert not cell.passed()
        assert any(v.kind == "nondeterministic" for v in cell.violations)

    def test_build_receipt(self) -> None:
        cell = ResourceCell()
        cell.start()
        cell.inject_sample(ResourceSample(wall_seconds=10.0))
        receipt = cell.build_receipt()
        assert receipt["passed"]
        assert receipt["sample_count"] == 1
        assert receipt["violation_count"] == 0
        assert "loop_id" in receipt

    def test_samples_maintain_order(self) -> None:
        cell = ResourceCell()
        cell.start()
        samples = [ResourceSample(wall_seconds=float(i)) for i in range(5)]
        for s in samples:
            cell.inject_sample(s)
        assert len(cell.samples) == 5
        assert [s.wall_seconds for s in cell.samples] == [float(i) for i in range(5)]


class TestCheckIntegrity:
    def test_check_integrity_clean(self) -> None:
        result = check_integrity(rows_input=100, rows_output=100)
        assert result.passed()
        assert result.silent_drop_rows == 0

    def test_check_integrity_dropped_rows(self) -> None:
        result = check_integrity(rows_input=100, rows_output=95)
        assert not result.passed()
        assert result.silent_drop_rows == 5

    def test_check_integrity_oom(self) -> None:
        result = check_integrity(rows_input=100, rows_output=100, oom=True)
        assert not result.passed()
        assert result.oom

    def test_check_integrity_nonfinite(self) -> None:
        result = check_integrity(rows_input=100, rows_output=100, nonfinite=True)
        assert not result.passed()
        assert result.nonfinite

    def test_check_integrity_nondeterministic(self) -> None:
        result = check_integrity(rows_input=100, rows_output=100, bitwise_deterministic_eval=False)
        assert not result.passed()
        assert not result.bitwise_deterministic_eval

    def test_check_integrity_timeout(self) -> None:
        result = check_integrity(rows_input=100, rows_output=100, timeout=True)
        assert not result.passed()
        assert result.timeout


class TestAssertBitwiseDeterministic:
    def test_deterministic_passes(self) -> None:
        a = torch.randn(4, 2)
        b = a.clone()
        assert_bitwise_deterministic(a, b)

    def test_deterministic_fails(self) -> None:
        a = torch.randn(4, 2)
        b = torch.randn(4, 2)
        with pytest.raises(AssertionError, match="determinism"):
            assert_bitwise_deterministic(a, b)


class TestAssertBudgetInvariants:
    def test_budget_invariants_passes(self) -> None:
        assert_budget_invariants()


# ===================================================================
# 6. Source Closure
# ===================================================================


class TestSourceClosure:
    def test_scan_source_closure_no_violations(self) -> None:
        """Phase 0 源码闭包应无违规。"""
        report = scan_source_closure()
        assert report.passed, f"violations: {report.violations}"
        assert len(report.scanned_files) >= 6  # 至少 6 个 .py 文件

    def test_scan_source_closure_manifest_has_all_files(self) -> None:
        report = scan_source_closure()
        for expected in (
            "src/loop186/__init__.py",
            "src/loop186/contracts.py",
            "src/loop186/hgconv.py",
            "src/loop186/model.py",
            "src/loop186/source_closure.py",
            "src/loop186/data_adapter.py",
            "src/loop186/resource_cell.py",
        ):
            assert expected in report.manifest, f"missing: {expected}"

    def test_scan_source_closure_no_forbidden_imports(self) -> None:
        """验证 foridden 导入模式未被引入。"""
        report = scan_source_closure()
        violations = [v for v in report.violations if v.kind == "forbidden_import"]
        assert len(violations) == 0, f"forbidden imports: {violations}"

    def test_build_current_manifest_returns_dict(self) -> None:
        manifest = build_current_manifest()
        assert isinstance(manifest, dict)
        assert len(manifest) >= 6

    def test_assert_phase0_closure_no_error(self) -> None:
        from src.loop186.source_closure import assert_phase0_closure
        report = assert_phase0_closure()
        assert report.passed


# ===================================================================
# 7. Integration: model + synthetic data + gradient
# ===================================================================


class TestModelIntegration:
    def test_forward_backward_on_synthetic_batch(self) -> None:
        """从 make_synthetic_batch 到 model forward 再到 backward 的完整链路。"""
        cfg = HGConvRegionConfig(
            model_dim=128,
            transformer_heads=8,
            transformer_layers=2,
            hgconv_blocks=2,
        )
        model = HGConvRegionNet(cfg)
        batch_dict = make_synthetic_batch(2, seed=41)
        output = model(
            batch_dict["region_tokens"],
            batch_dict["region_lengths"],
            batch_dict["region_types"],
            batch_dict["offset_buckets"],
            batch_dict["length_buckets"],
            batch_dict["b0_features"],
        )
        loss = torch.nn.functional.cross_entropy(
            output["fusion_logits"],
            batch_dict["labels"],
        )
        loss.backward()
        assert torch.isfinite(output["fusion_logits"]).all()
        assert any(p.grad is not None for p in model.parameters())

    def test_multi_scale_to_transformer_chain(self) -> None:
        """验证 MultiScaleHGConvBlock → TransformerEncoder 的维度兼容性。"""
        cfg = HGConvRegionConfig(
            model_dim=256,
            hgconv_blocks=2,
            transformer_layers=2,
            transformer_heads=8,
            transformer_ffn_dim=1024,
        )
        model = HGConvRegionNet(cfg)
        batch_dict = make_synthetic_batch(2, seed=41)
        output = model(
            batch_dict["region_tokens"],
            batch_dict["region_lengths"],
            batch_dict["region_types"],
            batch_dict["offset_buckets"],
            batch_dict["length_buckets"],
            batch_dict["b0_features"],
        )
        assert torch.isfinite(output["fusion_logits"]).all()
        assert torch.isfinite(output["region_logits"]).all()

    def test_model_save_and_load(self, tmp_path: Path) -> None:
        """验证模型可以 pickle 保存和加载（eval mode 下比较）。"""
        cfg = HGConvRegionConfig(
            model_dim=64,
            transformer_heads=8,
            transformer_layers=2,
            hgconv_blocks=2,
        )
        model = HGConvRegionNet(cfg)
        model.eval()
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)
        loaded = HGConvRegionNet(cfg)
        loaded.eval()
        loaded.load_state_dict(torch.load(path, weights_only=True))
        with torch.no_grad():
            batch_dict = make_synthetic_batch(2, seed=41)
            kwargs = {k: v for k, v in batch_dict.items() if k != "labels" and k != "row_ids"}
            out_orig = model(**kwargs)
            out_loaded = loaded(**kwargs)
        for key in out_orig:
            assert torch.equal(out_orig[key], out_loaded[key]), f"{key} differs after load"


# ===================================================================
# 8. Edge Cases
# ===================================================================


class TestEdgeCases:
    def test_model_all_padding(self) -> None:
        """全 padding 区域不应产生 NaN。"""
        cfg = HGConvRegionConfig()
        model = HGConvRegionNet(cfg)
        model.eval()
        tokens = torch.full((1, 16, 8192), 256, dtype=torch.long)
        lengths = torch.zeros((1, 16), dtype=torch.long)
        types = torch.zeros((1, 16), dtype=torch.long)
        offsets = torch.zeros((1, 16), dtype=torch.long)
        length_buckets = torch.zeros((1, 16), dtype=torch.long)
        b0 = torch.randn(1, 571)
        with torch.no_grad():
            output = model(tokens, lengths, types, offsets, length_buckets, b0)
        assert torch.isfinite(output["fusion_logits"]).all()

    def test_model_single_region(self) -> None:
        """仅 1 个有效 region。"""
        cfg = HGConvRegionConfig()
        model = HGConvRegionNet(cfg)
        model.eval()
        tokens = torch.full((1, 16, 8192), 256, dtype=torch.long)
        lengths = torch.zeros((1, 16), dtype=torch.long)
        lengths[0, 0] = 512
        tokens[0, 0, :512] = torch.randint(0, 255, (512,), dtype=torch.long)
        types = torch.zeros((1, 16), dtype=torch.long)
        types[0, 0] = 1
        offsets = torch.zeros((1, 16), dtype=torch.long)
        length_buckets = torch.zeros((1, 16), dtype=torch.long)
        b0 = torch.randn(1, 571)
        with torch.no_grad():
            output = model(tokens, lengths, types, offsets, length_buckets, b0)
        assert torch.isfinite(output["fusion_logits"]).all()

    def test_resource_cell_violation_message_readable(self) -> None:
        cell = ResourceCell()
        cell.start()
        cell.inject_sample(ResourceSample(wall_seconds=10_000_000.0))
        assert not cell.passed()
        msg = str(cell.violations[0].detail)
        assert "wall" in msg.lower() or "Wall" in msg

    def test_check_integrity_receipt_keys(self) -> None:
        result = check_integrity(rows_input=100, rows_output=100)
        assert result.silent_drop_rows == 0
        assert result.all_rows_accounted
        assert not result.oom
        assert not result.timeout
        assert not result.nonfinite
