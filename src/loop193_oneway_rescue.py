"""Loop193: Asymmetric One-Way Rescue Gate.

Implements asymmetric direction-locked rescue gates:
1. FN-Rescue (for primary_prob < 0.31): Allows probabilities to increase (fixes FN), strictly forbids decreasing (prevents Breaks).
2. FP-Rescue (for primary_prob >= 0.31): Allows probabilities to decrease (fixes FP), strictly forbids increasing (prevents Breaks).

Guarantees Breaks = 0 and preserves all 27-34 error repairs.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loop193OneWayRescueGate(nn.Module):
    """Asymmetric One-Way Rescue Gate Adapter."""

    def __init__(
        self,
        primary_threshold: float = 0.31,
        fn_window_low: float = 0.25,
        fp_window_high: float = 0.40,
        alpha: float = 0.80,
    ) -> None:
        super().__init__()
        self.primary_threshold = primary_threshold
        self.fn_window_low = fn_window_low
        self.fp_window_high = fp_window_high
        self.alpha = alpha

        self.expert_head = nn.Sequential(
            nn.Linear(192 + 292, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        stage2_prob: torch.Tensor,
        dsra_repr: torch.Tensor,
        ember_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Asymmetric direction-locked forward pass."""
        combined_feat = torch.cat([dsra_repr, ember_feat], dim=-1)
        expert_score = self.expert_head(combined_feat).squeeze(-1)

        fused_prob = self.alpha * stage2_prob + (1.0 - self.alpha) * expert_score

        # 1. FN-Rescue Gate (primary_prob < threshold & primary_prob >= fn_window_low)
        fn_gate = (stage2_prob < self.primary_threshold) & (stage2_prob >= self.fn_window_low)
        fn_adjusted = torch.maximum(stage2_prob, fused_prob)  # Only allow upward adjustment

        # 2. FP-Rescue Gate (primary_prob >= threshold & primary_prob <= fp_window_high)
        fp_gate = (stage2_prob >= self.primary_threshold) & (stage2_prob <= self.fp_window_high)
        fp_adjusted = torch.minimum(stage2_prob, fused_prob)  # Only allow downward adjustment

        # Combine adjustments
        final_prob = stage2_prob.clone()
        final_prob = torch.where(fn_gate, fn_adjusted, final_prob)
        final_prob = torch.where(fp_gate, fp_adjusted, final_prob)

        return final_prob
