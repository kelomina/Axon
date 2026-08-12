"""Compact Section/Region-MoE model frozen by the Loop175 proposal."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class RegionNetConfig:
    vocabulary_size: int = 257
    padding_token: int = 256
    byte_embedding_dim: int = 64
    model_dim: int = 192
    patch_size: int = 16
    block_count: int = 6
    block_expansion: int = 6
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    region_type_count: int = 6
    bucket_count: int = 64
    transformer_layers: int = 2
    transformer_heads: int = 6
    transformer_ffn_dim: int = 768
    b0_feature_dim: int = 571
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.block_count != len(self.dilations):
            raise ValueError("one frozen dilation is required per residual block")
        if self.model_dim % self.transformer_heads:
            raise ValueError("model_dim must be divisible by transformer_heads")


class _DilatedGluBlock(nn.Module):
    def __init__(self, dim: int, expansion: int, dilation: int, dropout: float) -> None:
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            padding=3 * dilation,
            dilation=dilation,
            groups=dim,
        )
        self.expand = nn.Linear(dim, hidden * 2)
        self.project = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.norm(values)
        values = self.depthwise(values.transpose(1, 2)).transpose(1, 2)
        values = functional.glu(self.expand(values), dim=-1)
        values = self.dropout(self.project(values))
        values = (values + residual) * patch_mask.unsqueeze(-1)
        return values


class RegionNet(nn.Module):
    def __init__(self, config: RegionNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or RegionNetConfig()
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
        self.blocks = nn.ModuleList(
            _DilatedGluBlock(cfg.model_dim, cfg.block_expansion, dilation, cfg.dropout)
            for dilation in cfg.dilations
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

    def _encode_patches(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_regions, token_count = tokens.shape
        usable_tokens = token_count - token_count % self.config.patch_size
        if usable_tokens < self.config.patch_size:
            raise ValueError("region token length must contain at least one complete patch")
        tokens = tokens[:, :usable_tokens]
        lengths = lengths.clamp(min=0, max=usable_tokens)
        positions = torch.arange(usable_tokens, device=tokens.device).unsqueeze(0)
        valid_tokens = positions < lengths.unsqueeze(1)
        tokens = torch.where(
            valid_tokens,
            tokens,
            torch.full_like(tokens, self.config.padding_token),
        )
        embedded = self.byte_embedding(tokens.long()).transpose(1, 2)
        patches = self.patchify(embedded).transpose(1, 2)
        patch_positions = torch.arange(patches.shape[1], device=tokens.device).unsqueeze(0)
        patch_mask = patch_positions * self.config.patch_size < lengths.unsqueeze(1)
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
        return output * has_content.unsqueeze(-1)

    def forward(
        self,
        region_tokens: torch.Tensor,
        region_lengths: torch.Tensor,
        region_types: torch.Tensor,
        offset_buckets: torch.Tensor,
        length_buckets: torch.Tensor,
        b0_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if region_tokens.ndim != 3:
            raise ValueError("region_tokens must have shape [batch, regions, bytes]")
        batch_size, region_count, token_count = region_tokens.shape
        expected_shape = (batch_size, region_count)
        for name, tensor in {
            "region_lengths": region_lengths,
            "region_types": region_types,
            "offset_buckets": offset_buckets,
            "length_buckets": length_buckets,
        }.items():
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}")

        flattened = region_tokens.reshape(batch_size * region_count, token_count)
        flat_lengths = region_lengths.reshape(batch_size * region_count)
        region_values = self._encode_patches(flattened, flat_lengths).reshape(
            batch_size, region_count, self.config.model_dim
        )
        region_values = (
            region_values
            + self.region_type_embedding(region_types.clamp(0, self.config.region_type_count - 1))
            + self.offset_embedding(offset_buckets.clamp(0, self.config.bucket_count - 1))
            + self.length_embedding(length_buckets.clamp(0, self.config.bucket_count - 1))
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
            if b0_features.shape != (batch_size, self.config.b0_feature_dim):
                raise ValueError(
                    f"b0_features must have shape {(batch_size, self.config.b0_feature_dim)}"
                )
            b0_features = torch.nan_to_num(b0_features.float())
            b0_representation = self.b0_projector(b0_features)
            fused_features = torch.cat([region_features, b0_representation], dim=-1)
            output["b0_features"] = b0_representation
            output["fusion_logits"] = self.fusion_head(fused_features)
        return output


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
