"""Loop207: Deep Contrastive Hard Example Projection Network.

Applies Supervised Contrastive Loss (SupCon) to project PE structural and chunk features
into a 128-dim normalized embedding space for hard-example separation.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """features: [B, D] normalized, labels: [B]."""
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()

        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        loss = -mean_log_prob_pos.mean()
        return loss


class Loop207ContrastiveProjection(nn.Module):
    """Contrastive Projection Encoder."""

    def __init__(self, in_dim: int = 256, proj_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.Linear(192, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return F.normalize(feat, dim=-1)
