"""Independent Multi-Scale HGConv-Region model for Loop185.

Loop185 架构（与 Loop184 一致）：
- byte_embedding → patchify (Conv1d) → 2 × MultiScaleHGConvBlocks → attention pooling
- → 4 层 region_transformer → fusion_head

参数量：2,610,573（与 Loop184 一致）
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .hgconv import MultiScaleHGConvBlock, MultiScaleHGConvConfig


@dataclass(frozen=True)
class HGConvRegionConfig:
    """Loop185 模型配置（与 Loop184 一致）。"""

    vocabulary_size: int = 257
    padding_token: int = 256
    byte_embedding_dim: int = 64
    model_dim: int = 192
    patch_size: int = 16
    hgconv_blocks: int = 2
    multi_scale_filter_lengths: tuple[int, ...] = (8, 16, 32, 64)
    region_type_count: int = 6
    bucket_count: int = 64
    transformer_layers: int = 4
    transformer_heads: int = 6
    transformer_ffn_dim: int = 768
    b0_feature_dim: int = 571
    expected_regions: int = 16
    expected_region_bytes: int = 8192
    dropout: float = 0.2

    def __post_init__(self) -> None:
        positive = (
            self.vocabulary_size,
            self.byte_embedding_dim,
            self.model_dim,
            self.patch_size,
            self.hgconv_blocks,
            self.region_type_count,
            self.bucket_count,
            self.transformer_layers,
            self.transformer_heads,
            self.transformer_ffn_dim,
            self.b0_feature_dim,
            self.expected_regions,
            self.expected_region_bytes,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("model dimensions must be positive")
        if self.padding_token != self.vocabulary_size - 1:
            raise ValueError("padding token must be the last vocabulary item")
        if self.model_dim % self.transformer_heads:
            raise ValueError("model_dim must be divisible by transformer_heads")
        if self.expected_region_bytes % self.patch_size:
            raise ValueError("region bytes must be divisible by patch size")
        patch_sequence = self.expected_region_bytes // self.patch_size
        if not self.multi_scale_filter_lengths:
            raise ValueError("multi_scale_filter_lengths must not be empty")
        if len(set(self.multi_scale_filter_lengths)) != len(self.multi_scale_filter_lengths):
            raise ValueError("multi_scale_filter_lengths must be unique")
        if max(self.multi_scale_filter_lengths) > patch_sequence:
            raise ValueError("max filter length must not exceed patch sequence length")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


def _require_integer(values: torch.Tensor, *, name: str) -> None:
    if values.dtype == torch.bool or torch.is_floating_point(values) or torch.is_complex(values):
        raise TypeError(f"{name} must be an integer tensor")


class HGConvRegionNet(nn.Module):
    """Loop185 Multi-Scale HGConv-Region 模型（与 Loop184 架构一致）。"""

    def __init__(self, config: HGConvRegionConfig | None = None) -> None:
        super().__init__()
        self.config = config or HGConvRegionConfig()
        cfg = self.config
        self.byte_embedding = nn.Embedding(
            cfg.vocabulary_size,
            cfg.byte_embedding_dim,
            padding_idx=cfg.padding_token,
        )
        self.patchify = nn.Conv1d(
            cfg.byte_embedding_dim,
            cfg.model_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        block_config = MultiScaleHGConvConfig(
            model_dim=cfg.model_dim,
            filter_lengths=cfg.multi_scale_filter_lengths,
            dropout=cfg.dropout,
        )
        self.blocks = nn.ModuleList(
            MultiScaleHGConvBlock(block_config) for _ in range(cfg.hgconv_blocks)
        )
        self.attention_score = nn.Linear(cfg.model_dim, 1)
        self.pool_gate = nn.Linear(cfg.model_dim * 2, cfg.model_dim)
        self.pool_candidate = nn.Linear(cfg.model_dim * 2, cfg.model_dim)
        self.region_type_embedding = nn.Embedding(cfg.region_type_count, cfg.model_dim)
        self.offset_embedding = nn.Embedding(cfg.bucket_count, cfg.model_dim)
        self.length_embedding = nn.Embedding(cfg.bucket_count, cfg.model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.region_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.transformer_layers,
            norm=nn.LayerNorm(cfg.model_dim),
            enable_nested_tensor=False,
        )
        self.region_head = nn.Sequential(
            nn.Linear(cfg.model_dim, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 2),
        )
        self.b0_projector = nn.Sequential(
            nn.Linear(cfg.b0_feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 128),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(cfg.model_dim + 128, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 2),
        )
        nn.init.normal_(self.cls_token, std=0.02)

    def _validate_inputs(
        self,
        region_tokens: torch.Tensor,
        region_lengths: torch.Tensor,
        region_types: torch.Tensor,
        offset_buckets: torch.Tensor,
        length_buckets: torch.Tensor,
        b0_features: torch.Tensor | None,
    ) -> None:
        cfg = self.config
        _require_integer(region_tokens, name="region_tokens")
        if region_tokens.ndim != 3:
            raise ValueError("region_tokens must have shape [batch, regions, bytes]")
        batch_size, region_count, region_bytes = region_tokens.shape
        if region_count != cfg.expected_regions or region_bytes != cfg.expected_region_bytes:
            raise ValueError("region_tokens drifted from the frozen shape")
        expected_shape = (batch_size, region_count)
        metadata = {
            "region_lengths": region_lengths,
            "region_types": region_types,
            "offset_buckets": offset_buckets,
            "length_buckets": length_buckets,
        }
        for name, values in metadata.items():
            _require_integer(values, name=name)
            if tuple(values.shape) != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")
        if torch.any(region_tokens < 0).item() or torch.any(region_tokens >= cfg.vocabulary_size).item():
            raise ValueError("region_tokens contain an out-of-range token")
        if torch.any(region_lengths < 0).item() or torch.any(region_lengths > region_bytes).item():
            raise ValueError("region_lengths are outside the frozen range")
        if torch.any(region_types < 0).item() or torch.any(region_types >= cfg.region_type_count).item():
            raise ValueError("region_types are outside the frozen range")
        for name, values in {
            "offset_buckets": offset_buckets,
            "length_buckets": length_buckets,
        }.items():
            if torch.any(values < 0).item() or torch.any(values >= cfg.bucket_count).item():
                raise ValueError(f"{name} are outside the frozen range")
        positions = torch.arange(region_bytes, device=region_tokens.device).view(1, 1, -1)
        valid = positions < region_lengths.unsqueeze(-1)
        if torch.any(region_tokens[valid] == cfg.padding_token).item():
            raise ValueError("padding token appeared inside a valid region span")
        if torch.any(region_tokens[~valid] != cfg.padding_token).item():
            raise ValueError("region padding bytes must use the frozen padding token")
        missing = region_lengths == 0
        if torch.any(missing != (region_types == 0)).item():
            raise ValueError("zero-length regions and missing region types disagree")
        if torch.any(offset_buckets[missing] != 0).item() or torch.any(length_buckets[missing] != 0).item():
            raise ValueError("missing regions must use zero metadata buckets")
        if b0_features is not None:
            if b0_features.shape != (batch_size, cfg.b0_feature_dim):
                raise ValueError("b0_features drifted from the frozen shape")
            if not torch.is_floating_point(b0_features) or not torch.isfinite(b0_features).all().item():
                raise ValueError("b0_features must be finite floating values")

    def _encode_patches(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        embedded = self.byte_embedding(tokens.long()).transpose(1, 2)
        patches = self.patchify(embedded).transpose(1, 2)
        patch_positions = torch.arange(patches.shape[1], device=tokens.device).unsqueeze(0)
        patch_mask = patch_positions * cfg.patch_size < lengths.unsqueeze(1)
        patches = patches * patch_mask.unsqueeze(-1)
        for block in self.blocks:
            patches = block(patches, patch_mask)

        scores = self.attention_score(patches).squeeze(-1).masked_fill(~patch_mask, -1.0e4)
        weights = torch.softmax(scores, dim=1) * patch_mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        attentive = torch.sum(patches * weights.unsqueeze(-1), dim=1)
        maximum = patches.masked_fill(~patch_mask.unsqueeze(-1), -1.0e4).amax(dim=1)
        has_content = patch_mask.any(dim=1)
        maximum = torch.where(has_content.unsqueeze(-1), maximum, torch.zeros_like(maximum))
        pooled = torch.cat([attentive, maximum], dim=-1)
        gate = torch.sigmoid(self.pool_gate(pooled))
        candidate = torch.tanh(self.pool_candidate(pooled))
        output = gate * candidate + (1.0 - gate) * attentive
        output = output * has_content.unsqueeze(-1)
        if not torch.isfinite(output).all().item():
            raise FloatingPointError("patch encoder produced a non-finite output")
        return output

    def forward(
        self,
        region_tokens: torch.Tensor,
        region_lengths: torch.Tensor,
        region_types: torch.Tensor,
        offset_buckets: torch.Tensor,
        length_buckets: torch.Tensor,
        b0_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(
            region_tokens,
            region_lengths,
            region_types,
            offset_buckets,
            length_buckets,
            b0_features,
        )
        cfg = self.config
        batch_size, region_count, token_count = region_tokens.shape
        flattened = region_tokens.reshape(batch_size * region_count, token_count)
        flat_lengths = region_lengths.reshape(batch_size * region_count)
        region_values = self._encode_patches(flattened, flat_lengths).reshape(
            batch_size, region_count, cfg.model_dim
        )
        region_values = (
            region_values
            + self.region_type_embedding(region_types)
            + self.offset_embedding(offset_buckets)
            + self.length_embedding(length_buckets)
        )
        region_mask = region_lengths > 0
        region_values = region_values * region_mask.unsqueeze(-1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls, region_values], dim=1)
        padding_mask = torch.cat(
            [torch.zeros(batch_size, 1, dtype=torch.bool, device=region_mask.device), ~region_mask],
            dim=1,
        )
        encoded = self.region_transformer(sequence, src_key_padding_mask=padding_mask)
        region_features = encoded[:, 0]
        output = {
            "region_features": region_features,
            "region_logits": self.region_head(region_features),
        }
        if b0_features is not None:
            b0_representation = self.b0_projector(b0_features)
            output["b0_features"] = b0_representation
            output["fusion_logits"] = self.fusion_head(
                torch.cat([region_features, b0_representation], dim=-1)
            )
        if any(not torch.isfinite(values).all().item() for values in output.values()):
            raise FloatingPointError("HGConvRegionNet produced a non-finite output")
        return output


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
