from __future__ import annotations

import pytest
import torch

from src.loop179.model import HGConvRegionConfig, HGConvRegionNet, parameter_count


def tiny_config() -> HGConvRegionConfig:
    return HGConvRegionConfig(
        byte_embedding_dim=8,
        model_dim=24,
        patch_size=8,
        hgconv_filter_length=3,
        transformer_layers=1,
        transformer_heads=3,
        transformer_ffn_dim=48,
        b0_feature_dim=7,
        expected_regions=3,
        expected_region_bytes=32,
        dropout=0.0,
    )


def batch(config: HGConvRegionConfig, rows: int = 2) -> tuple[torch.Tensor, ...]:
    tokens = torch.full(
        (rows, config.expected_regions, config.expected_region_bytes),
        config.padding_token,
        dtype=torch.int64,
    )
    lengths = torch.tensor([[32, 17, 0]] * rows, dtype=torch.int64)
    for row in range(rows):
        tokens[row, 0] = row + 1
        tokens[row, 1, :17] = row + 11
    types = torch.tensor([[1, 3, 0]] * rows, dtype=torch.int64)
    offsets = torch.tensor([[0, 17, 0]] * rows, dtype=torch.int64)
    length_buckets = torch.tensor([[63, 34, 0]] * rows, dtype=torch.int64)
    b0 = torch.zeros(rows, config.b0_feature_dim)
    return tokens, lengths, types, offsets, length_buckets, b0


def test_model_preserves_loop175_tensor_abi() -> None:
    config = tiny_config()
    model = HGConvRegionNet(config).eval()
    with torch.no_grad():
        output = model(*batch(config))
    assert output["region_features"].shape == (2, config.model_dim)
    assert output["region_logits"].shape == (2, 2)
    assert output["b0_features"].shape == (2, 128)
    assert output["fusion_logits"].shape == (2, 2)
    assert all(torch.isfinite(values).all() for values in output.values())
    assert parameter_count(model) > 0


def test_model_is_deterministic_in_eval_for_fixed_state() -> None:
    config = tiny_config()
    torch.manual_seed(41)
    model = HGConvRegionNet(config).eval()
    values = batch(config)
    with torch.no_grad():
        first = model(*values)
        second = model(*values)
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0.0, atol=0.0)


def test_model_rejects_padding_and_b0_repairs() -> None:
    config = tiny_config()
    model = HGConvRegionNet(config).eval()
    values = list(batch(config))
    values[0] = values[0].clone()
    values[0][0, 1, 20] = 7
    with pytest.raises(ValueError, match="padding bytes"):
        model(*values)

    values = list(batch(config))
    values[5] = values[5].clone()
    values[5][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(*values)


def test_all_missing_regions_remain_finite() -> None:
    config = tiny_config()
    model = HGConvRegionNet(config).eval()
    tokens = torch.full(
        (1, config.expected_regions, config.expected_region_bytes),
        config.padding_token,
        dtype=torch.int64,
    )
    metadata = torch.zeros(1, config.expected_regions, dtype=torch.int64)
    with torch.no_grad():
        output = model(tokens, metadata, metadata, metadata, metadata)
    assert output["region_logits"].shape == (1, 2)
    assert torch.isfinite(output["region_logits"]).all()


def test_default_config_accepts_full_synthetic_shape() -> None:
    config = HGConvRegionConfig(dropout=0.0)
    model = HGConvRegionNet(config).eval()
    tokens = torch.full(
        (1, config.expected_regions, config.expected_region_bytes),
        config.padding_token,
        dtype=torch.int64,
    )
    lengths = torch.zeros(1, config.expected_regions, dtype=torch.int64)
    types = torch.zeros_like(lengths)
    buckets = torch.zeros_like(lengths)
    tokens[0, 0, : config.patch_size] = 1
    lengths[0, 0] = config.patch_size
    types[0, 0] = 1
    with torch.no_grad():
        output = model(tokens, lengths, types, buckets, buckets)
    assert output["region_logits"].shape == (1, 2)
    assert torch.isfinite(output["region_logits"]).all()
