"""PyTorch HGConv core for the isolated Loop179 lineage."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional


def _normalize_dim(values: torch.Tensor, dim: int) -> int:
    normalized = dim if dim >= 0 else values.ndim + dim
    if not 0 <= normalized < values.ndim:
        raise ValueError("FFT dimension is outside the tensor rank")
    return normalized


def _require_real_floating(values: torch.Tensor, *, name: str) -> None:
    if not torch.is_floating_point(values) or torch.is_complex(values):
        raise TypeError(f"{name} must be a real floating tensor")
    if not torch.isfinite(values).all().item():
        raise ValueError(f"{name} must be finite")


def _compute_dtype(values: torch.Tensor) -> torch.dtype:
    return torch.float64 if values.dtype == torch.float64 else torch.float32


def _fft_circular_convolution(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    normalized_dim = _normalize_dim(left, dim)
    compute_dtype = _compute_dtype(left)
    left_values = left.to(dtype=compute_dtype)
    if torch.is_complex(right):
        complex_dtype = torch.complex128 if compute_dtype == torch.float64 else torch.complex64
        right_values = right.to(dtype=complex_dtype)
    else:
        right_values = right.to(dtype=compute_dtype)
    right_dim = normalized_dim - (left.ndim - right.ndim)
    if not 0 <= right_dim < right.ndim:
        raise ValueError("right tensor does not expose the convolution axis")
    left_spectrum = torch.fft.fft(left_values, dim=normalized_dim)
    right_spectrum = torch.fft.fft(right_values, dim=right_dim)
    output = torch.fft.ifft(left_spectrum * right_spectrum, dim=normalized_dim).real
    if not torch.isfinite(output).all().item():
        raise FloatingPointError("circular convolution produced a non-finite output")
    return output


def circular_convolution(left: torch.Tensor, right: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Return real circular convolution along one equal-length broadcastable axis."""

    _require_real_floating(left, name="left")
    _require_real_floating(right, name="right")
    normalized_dim = _normalize_dim(left, dim)
    right_dim = normalized_dim - (left.ndim - right.ndim)
    if not 0 <= right_dim < right.ndim:
        raise ValueError("right tensor does not expose the convolution axis")
    if left.shape[normalized_dim] != right.shape[right_dim]:
        raise ValueError("circular convolution axes must have equal length")
    return _fft_circular_convolution(left, right, dim=normalized_dim)


def approximate_inverse(values: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Return the HRR flip-and-roll approximate inverse used by the reference code."""

    _require_real_floating(values, name="values")
    normalized_dim = _normalize_dim(values, dim)
    return torch.roll(torch.flip(values, dims=(normalized_dim,)), shifts=1, dims=normalized_dim)


def malware_kernel_precondition(
    filter_values: torch.Tensor,
    *,
    sequence_length: int,
    dim: int,
) -> torch.Tensor:
    """Apply the official malware-task FFT/IFFT-ortho kernel precondition."""

    _require_real_floating(filter_values, name="filter_values")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    normalized_dim = _normalize_dim(filter_values, dim)
    if filter_values.shape[normalized_dim] > sequence_length:
        raise ValueError("filter length cannot exceed the sequence length")
    compute_dtype = _compute_dtype(filter_values)
    spectrum = torch.fft.fft(
        filter_values.to(dtype=compute_dtype),
        n=sequence_length,
        dim=normalized_dim,
    )
    kernel = torch.fft.ifft(
        spectrum,
        n=sequence_length,
        dim=normalized_dim,
        norm="ortho",
    )
    if not torch.isfinite(kernel).all().item():
        raise FloatingPointError("kernel precondition produced a non-finite output")
    return kernel


@dataclass(frozen=True)
class HGConvConfig:
    model_dim: int = 192
    filter_length: int = 32
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.model_dim <= 0 or self.filter_length <= 0:
            raise ValueError("HGConv dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class HGConvBlock(nn.Module):
    """Feature bind, sequence convolution, approximate unbind, and GLU residual."""

    def __init__(self, config: HGConvConfig | None = None) -> None:
        super().__init__()
        self.config = config or HGConvConfig()
        dim = self.config.model_dim
        self.bind_filter = nn.Parameter(torch.empty(dim))
        self.convolution_filter = nn.Parameter(torch.empty(1, self.config.filter_length, dim))
        self.unbind_filter = nn.Parameter(torch.empty(dim))
        self.bias_weight = nn.Parameter(torch.empty(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.glu_projection = nn.Linear(dim, dim * 2)
        self.dropout = nn.Dropout(self.config.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.bind_filter)
        nn.init.normal_(self.convolution_filter)
        nn.init.normal_(self.unbind_filter)
        nn.init.normal_(self.bias_weight)
        self.norm.reset_parameters()
        self.glu_projection.reset_parameters()

    def forward(self, patches: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        _require_real_floating(patches, name="patches")
        if patches.ndim != 3:
            raise ValueError("patches must have shape [batch_regions, sequence, model_dim]")
        if patches.shape[-1] != self.config.model_dim:
            raise ValueError("patch model dimension drifted")
        if patch_mask.dtype != torch.bool or patch_mask.shape != patches.shape[:2]:
            raise ValueError("patch_mask must be boolean with shape [batch_regions, sequence]")
        if patches.shape[1] < self.config.filter_length:
            raise ValueError("sequence length is shorter than the convolution filter")

        public_dtype = patches.dtype
        mask = patch_mask.unsqueeze(-1)
        values = patches * mask
        residual = values
        bound = circular_convolution(values, self.bind_filter, dim=-1)
        kernel = malware_kernel_precondition(
            self.convolution_filter,
            sequence_length=patches.shape[1],
            dim=1,
        )
        convolved = _fft_circular_convolution(bound, kernel, dim=1)
        convolved = functional.gelu(convolved + bound * self.bias_weight.to(dtype=bound.dtype))
        inverse = approximate_inverse(self.unbind_filter, dim=-1)
        unbound = circular_convolution(convolved, inverse, dim=-1)
        gated = functional.glu(self.glu_projection(self.norm(unbound)), dim=-1)
        output = (residual.to(dtype=gated.dtype) + self.dropout(gated)) * mask
        if not torch.isfinite(output).all().item():
            raise FloatingPointError("HGConv block produced a non-finite output")
        return output.to(dtype=public_dtype)
