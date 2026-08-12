"""PyTorch Multi-Scale HGConv core for the isolated Loop182 lineage.

Loop182 核心改进：在 HGConvBlock 中并行使用多个 filter_length，
通过可学习权重融合多尺度特征，捕捉不同粒度的字节模式。
"""

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
    """单尺度 HGConv 配置（与 Loop179 一致，用于回退对比）。"""

    model_dim: int = 192
    filter_length: int = 32
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.model_dim <= 0 or self.filter_length <= 0:
            raise ValueError("HGConv dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class MultiScaleHGConvConfig:
    """Loop182 核心改进：多尺度 HGConv 配置。

    并行使用多个 filter_length，每个尺度独立 bind → convolve → unbind，
    通过可学习权重融合多尺度特征。
    """

    model_dim: int = 192
    filter_lengths: tuple[int, ...] = (8, 16, 32, 64)
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if not self.filter_lengths:
            raise ValueError("filter_lengths must not be empty")
        if any(fl <= 0 for fl in self.filter_lengths):
            raise ValueError("all filter_lengths must be positive")
        if len(set(self.filter_lengths)) != len(self.filter_lengths):
            raise ValueError("filter_lengths must be unique")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class HGConvBlock(nn.Module):
    """单尺度 HGConvBlock（与 Loop179 一致，用于回退对比）。"""

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


class MultiScaleHGConvBlock(nn.Module):
    """Loop182 核心改进：多尺度并行 HGConvBlock。

    对每个 filter_length 独立执行：
        bind → convolve(filter_i) → bias+gelu → unbind → norm
    产生 num_scales 个 [B, S, D] 张量，通过可学习的门控权重融合：
        fused = sum_i(softmax(weights)_i * scale_i_output)
    然后通过 GLU 残差连接。

    相比单尺度 HGConvBlock：
    - 参数量增加：每个尺度增加 filter_length_i * model_dim 个参数
    - 表达能力：捕捉不同尺度的字节模式（短模式 vs 长模式）
    - 融合方式：可学习权重 + softmax 归一化，让模型自动选择尺度重要性
    """

    def __init__(self, config: MultiScaleHGConvConfig | None = None) -> None:
        super().__init__()
        self.config = config or MultiScaleHGConvConfig()
        dim = self.config.model_dim
        self.num_scales = len(self.config.filter_lengths)

        # 共享的 bind/unbind filter（跨尺度共享，减少参数）
        self.bind_filter = nn.Parameter(torch.empty(dim))
        self.unbind_filter = nn.Parameter(torch.empty(dim))

        # 每个尺度独立的 convolution filter
        self.convolution_filters = nn.ParameterList(
            nn.Parameter(torch.empty(1, fl, dim)) for fl in self.config.filter_lengths
        )

        # 每个尺度独立的 bias weight
        self.bias_weights = nn.ParameterList(
            nn.Parameter(torch.empty(1, 1, dim)) for _ in self.config.filter_lengths
        )

        # 可学习的尺度融合权重（softmax 归一化）
        self.scale_weights = nn.Parameter(torch.zeros(self.num_scales))

        self.norm = nn.LayerNorm(dim)
        self.glu_projection = nn.Linear(dim, dim * 2)
        self.dropout = nn.Dropout(self.config.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.bind_filter)
        nn.init.normal_(self.unbind_filter)
        for conv_filter in self.convolution_filters:
            nn.init.normal_(conv_filter)
        for bias in self.bias_weights:
            nn.init.normal_(bias)
        # scale_weights 初始化为 0 → softmax 后均匀分布
        nn.init.zeros_(self.scale_weights)
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
        sequence_length = patches.shape[1]
        for fl in self.config.filter_lengths:
            if sequence_length < fl:
                raise ValueError(
                    f"sequence length {sequence_length} is shorter than filter length {fl}"
                )

        public_dtype = patches.dtype
        mask = patch_mask.unsqueeze(-1)
        values = patches * mask
        residual = values

        # 共享 bind
        bound = circular_convolution(values, self.bind_filter, dim=-1)

        # 多尺度并行卷积
        scale_outputs: list[torch.Tensor] = []
        for i, (conv_filter, bias_weight) in enumerate(
            zip(self.convolution_filters, self.bias_weights)
        ):
            kernel = malware_kernel_precondition(
                conv_filter,
                sequence_length=sequence_length,
                dim=1,
            )
            convolved = _fft_circular_convolution(bound, kernel, dim=1)
            convolved = functional.gelu(
                convolved + bound * bias_weight.to(dtype=bound.dtype)
            )
            scale_outputs.append(convolved)

        # 可学习权重融合（softmax 归一化）
        scale_probs = functional.softmax(self.scale_weights, dim=0)
        fused = sum(
            prob * output
            for prob, output in zip(scale_probs, scale_outputs)
        )

        # 共享 unbind
        inverse = approximate_inverse(self.unbind_filter, dim=-1)
        unbound = circular_convolution(fused, inverse, dim=-1)

        # GLU 残差
        gated = functional.glu(self.glu_projection(self.norm(unbound)), dim=-1)
        output = (residual.to(dtype=gated.dtype) + self.dropout(gated)) * mask
        if not torch.isfinite(output).all().item():
            raise FloatingPointError("MultiScaleHGConvBlock produced a non-finite output")
        return output.to(dtype=public_dtype)
