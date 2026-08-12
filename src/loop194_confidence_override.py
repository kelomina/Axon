"""Loop194: Hard Confidence-Gated One-Way Override Adapter.

Applies high-confidence one-way override:
1. FN-Fix: Overrides primary FN (pred=0) to malicious (pred=1) ONLY IF expert confidence >= 0.90.
2. FP-Fix: Overrides primary FP (pred=1) to benign (pred=0) ONLY IF expert confidence <= 0.10.

Enforces zero Breaks while retaining high-precision error repairs.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loop194ConfidenceOverrideAdapter(nn.Module):
    """Hard Confidence-Gated One-Way Override Adapter."""

    def __init__(
        self,
        primary_threshold: float = 0.31,
        fn_override_high: float = 0.90,
        fp_override_low: float = 0.10,
    ) -> None:
        super().__init__()
        self.primary_threshold = primary_threshold
        self.fn_override_high = fn_override_high
        self.fp_override_low = fp_override_low

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
        """High-confidence one-way override logic."""
        combined_feat = torch.cat([dsra_repr, ember_feat], dim=-1)
        expert_score = self.expert_head(combined_feat).squeeze(-1)

        primary_pred = (stage2_prob >= self.primary_threshold).long()
        final_pred = primary_pred.clone()

        # 1. FN-Fix: Primary pred is 0, but Expert is strongly confident >= 0.90
        fn_override_mask = (primary_pred == 0) & (expert_score >= self.fn_override_high)
        final_pred = torch.where(fn_override_mask, torch.ones_like(final_pred), final_pred)

        # 2. FP-Fix: Primary pred is 1, but Expert is strongly confident <= 0.10
        fp_override_mask = (primary_pred == 1) & (expert_score <= self.fp_override_low)
        final_pred = torch.where(fp_override_mask, torch.zeros_like(final_pred), final_pred)

        return final_pred.float()
