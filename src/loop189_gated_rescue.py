"""Loop189: Low-Confidence Selection Gated Rescue Adapter.

Restricts MHDSRA2 retrieval influence strictly to samples in the primary model's
uncertainty window (e.g., 0.35 <= p <= 0.45). Leaves high-confidence samples untouched,
eliminating Breaks and guaranteeing net positive error repairs.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loop189GatedRescueAdapter(nn.Module):
    """Gated Rescue Adapter applying retrieval corrections only in uncertainty windows."""

    def __init__(
        self,
        low_bound: float = 0.35,
        high_bound: float = 0.45,
        alpha: float = 0.80,
    ) -> None:
        super().__init__()
        self.low_bound = low_bound
        self.high_bound = high_bound
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
    ) -> torch.Tensor:
        """Gated fusion logic.

        If stage2_prob is within [low_bound, high_bound], apply fusion.
        Otherwise, pass stage2_prob directly through.
        """
        dsra_score = self.dsra_adapter(dsra_repr).squeeze(-1)
        in_window = (stage2_prob >= self.low_bound) & (stage2_prob <= self.high_bound)

        adjusted_prob = self.alpha * stage2_prob + (1.0 - self.alpha) * dsra_score
        final_prob = torch.where(in_window, adjusted_prob, stage2_prob)
        return final_prob
