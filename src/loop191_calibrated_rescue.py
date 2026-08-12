"""Loop191: Temperature-Calibrated Soft Boundary Selection Adapter.

Combines temperature scaling on Primary Stage-2 probabilities with
MHDSRA2 retrieval feature projections to safely target Loop151 residual errors.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loop191CalibratedRescueAdapter(nn.Module):
    """Calibrated soft boundary selection adapter."""

    def __init__(
        self,
        temperature: float = 1.25,
        low_bound: float = 0.28,
        high_bound: float = 0.38,
        alpha: float = 0.85,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.alpha = alpha
        self.dsra_proj = nn.Sequential(
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
        """Applies Temperature Calibration + Soft Boundary Mask."""
        # 1. Temperature scaling in logit space
        stage2_prob_clamped = stage2_prob.clamp(1e-7, 1 - 1e-7)
        logits = torch.log(stage2_prob_clamped / (1 - stage2_prob_clamped))
        calibrated_logits = logits / self.temperature
        calibrated_prob = torch.sigmoid(calibrated_logits)

        # 2. Uncertainty window mask
        in_window = (calibrated_prob >= self.low_bound) & (calibrated_prob <= self.high_bound)

        # 3. Retrieval feature projection
        dsra_score = self.dsra_proj(dsra_repr).squeeze(-1)
        adjusted_prob = self.alpha * calibrated_prob + (1.0 - self.alpha) * dsra_score

        # 4. Masked fusion
        final_prob = torch.where(in_window, adjusted_prob, calibrated_prob)
        return final_prob
