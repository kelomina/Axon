"""Loop190: Precision Gated Rescue Adapter with Cosine Similarity Masking.

Dual-gated rescue mechanism combining:
1. Probability Uncertainty Window (default 0.30 <= p <= 0.35 near PRIMARY_THR = 0.31)
2. Cosine Similarity Masking (default sim > 0.80) to filter out low-relevance retrieval noise.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loop190PrecisionGatedRescueAdapter(nn.Module):
    """Dual-gated precision rescue adapter."""

    def __init__(
        self,
        low_bound: float = 0.30,
        high_bound: float = 0.35,
        sim_threshold: float = 0.80,
        alpha: float = 0.80,
    ) -> None:
        super().__init__()
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.sim_threshold = sim_threshold
        self.alpha = alpha
        self.dsra_adapter = nn.Sequential(
            nn.Linear(192, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        stage2_prob: torch.Tensor,
        dsra_repr: torch.Tensor,
        retrieved_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dual-gated forward pass.

        1. Probability window mask
        2. Cosine similarity mask
        """
        dsra_score = self.dsra_adapter(dsra_repr).squeeze(-1)

        # 1. Uncertainty window mask
        prob_mask = (stage2_prob >= self.low_bound) & (stage2_prob <= self.high_bound)

        # 2. Similarity mask
        if retrieved_k is not None:
            # Flatten heads and head_dim to match d_model=192
            if retrieved_k.dim() == 4:
                # [B, H, R, d] -> mean across tokens R -> [B, H, d] -> reshape [B, H*d]
                k_mean = retrieved_k.mean(dim=2).reshape(retrieved_k.shape[0], -1)
            else:
                k_mean = retrieved_k.reshape(retrieved_k.shape[0], -1)

            cos_sim = F.cosine_similarity(dsra_repr, k_mean, dim=-1)
            sim_mask = cos_sim >= self.sim_threshold
        else:
            sim_mask = torch.ones_like(prob_mask, dtype=torch.bool)

        dual_mask = prob_mask & sim_mask

        adjusted_prob = self.alpha * stage2_prob + (1.0 - self.alpha) * dsra_score
        final_prob = torch.where(dual_mask, adjusted_prob, stage2_prob)
        return final_prob
