"""Loop197: Hard Example Mining Specialist Network.

Specialized neural network focusing specifically on hard-negative (high-confidence FP)
and hard-positive (deep-miss FN) samples using Focal Loss and Soft Label Smoothing.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for hard example weighting."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class Loop197HardMiningExpert(nn.Module):
    """Hard Example Mining Neural Specialist."""

    def __init__(
        self,
        dsra_dim: int = 192,
        ember_dim: int = 292,
        kvd_dim: int = 571,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.dsra_proj = nn.Sequential(
            nn.Linear(dsra_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
        )

        self.struct_proj = nn.Sequential(
            nn.Linear(ember_dim + kvd_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        dsra_repr: torch.Tensor,
        ember_feat: torch.Tensor,
        kvd_feat: torch.Tensor,
    ) -> torch.Tensor:
        h_dsra = self.dsra_proj(dsra_repr)
        struct_feat = torch.cat([ember_feat, kvd_feat], dim=-1)
        h_struct = self.struct_proj(struct_feat)

        fused = torch.cat([h_dsra, h_struct], dim=-1)
        logits = self.classifier(fused)
        return logits
