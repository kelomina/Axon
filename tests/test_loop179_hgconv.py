from __future__ import annotations

import math

import pytest
import torch

from src.loop179.hgconv import (
    HGConvBlock,
    HGConvConfig,
    approximate_inverse,
    circular_convolution,
    malware_kernel_precondition,
)


def explicit_circular_convolution(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    length = left.shape[-1]
    return torch.stack([
        sum(left[..., index] * right[(output - index) % length] for index in range(length))
        for output in range(length)
    ], dim=-1)


def test_feature_binding_matches_explicit_modulo_sum() -> None:
    left = torch.tensor([[[1.0, 2.0, -1.0, 0.5], [0.0, 3.0, 2.0, -2.0]]], dtype=torch.float64)
    right = torch.tensor([0.25, -1.0, 2.0, 0.5], dtype=torch.float64)
    actual = circular_convolution(left, right, dim=-1)
    expected = explicit_circular_convolution(left, right)
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_sequence_convolution_matches_explicit_modulo_sum() -> None:
    left = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]], dtype=torch.float64)
    right = torch.tensor([[[2.0], [-1.0], [0.5], [0.0], [0.0]]], dtype=torch.float64)
    actual = circular_convolution(left, right, dim=1).squeeze(0).squeeze(-1)
    values = left.squeeze(0).squeeze(-1)
    kernel = right.squeeze(0).squeeze(-1)
    expected = torch.stack([
        sum(values[index] * kernel[(output - index) % values.numel()] for index in range(values.numel()))
        for output in range(values.numel())
    ])
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_approximate_inverse_is_flip_then_roll() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(
        approximate_inverse(values, dim=-1),
        torch.tensor([1.0, 4.0, 3.0, 2.0], dtype=torch.float64),
    )


def test_malware_kernel_precondition_freezes_ortho_scale() -> None:
    values = torch.tensor([[[1.0], [2.0]]], dtype=torch.float64)
    kernel = malware_kernel_precondition(values, sequence_length=5, dim=1)
    expected = torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float64) * math.sqrt(5)
    torch.testing.assert_close(kernel.real.squeeze(), expected, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(kernel.imag, torch.zeros_like(kernel.imag), rtol=0.0, atol=1.0e-12)


def test_block_preserves_shape_mask_and_gradients() -> None:
    torch.manual_seed(7)
    block = HGConvBlock(HGConvConfig(model_dim=8, filter_length=3, dropout=0.0)).double()
    patches = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False], [False, False, False, False, False]])
    changed = patches.detach().clone()
    changed[0, 3:] = 1000.0
    changed[1] = -1000.0
    changed.requires_grad_(True)

    first = block(patches, mask)
    second = block(changed, mask)
    assert first.shape == patches.shape
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first[~mask], torch.zeros_like(first[~mask]), rtol=0.0, atol=0.0)
    first.square().sum().backward()
    assert patches.grad is not None and torch.isfinite(patches.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in block.parameters())


def test_block_rejects_nonfinite_and_shape_drift() -> None:
    block = HGConvBlock(HGConvConfig(model_dim=4, filter_length=2, dropout=0.0))
    patches = torch.zeros(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    invalid = patches.clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        block(invalid, mask)
    with pytest.raises(ValueError, match="patch_mask"):
        block(patches, torch.ones(1, 3))
