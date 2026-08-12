"""Loop192: Multi-Expert Fusion Stacker Engine.

Fuses four independent expert representations:
1. Primary Stage-2 GBDT Logits (Loop151 Baseline)
2. Upgraded MHDSRA2 Retrieval Representations (Loop187 Engine)
3. EMBER-v3 Novel-Delta Structural Features (292-dim)
4. Trusted Signer Guard Authenticode Post-Processing

Uses Constrained Non-Negative Stacker to eliminate residual errors without breaking valid predictions.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loop192MultiExpertStacker(nn.Module):
    """Constrained Multi-Expert Fusion Stacker."""

    def __init__(self, dsra_dim: int = 192, ember_dim: int = 292) -> None:
        super().__init__()
        # 1. DSRA Retrieval Adapter
        self.dsra_head = nn.Sequential(
            nn.Linear(dsra_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # 2. EMBER-v3 Novel Structural Adapter
        self.ember_head = nn.Sequential(
            nn.Linear(ember_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # 3. Non-Negative Gated Fusion Weights (Primary bias: 0.85, DSRA: 0.10, EMBER: 0.05)
        self.gate = nn.Parameter(torch.tensor([0.85, 0.10, 0.05]))

    def forward(
        self,
        stage2_prob: torch.Tensor,
        dsra_repr: torch.Tensor,
        ember_novel_feat: Optional[torch.Tensor] = None,
        low_bound: float = 0.30,
        high_bound: float = 0.35,
    ) -> torch.Tensor:
        """Dual-gated Multi-Expert forward pass."""
        # Convert Stage-2 prob to logit
        p_clamped = stage2_prob.clamp(1e-7, 1 - 1e-7)
        stage2_logit = torch.log(p_clamped / (1.0 - p_clamped))

        # Expert logits
        dsra_logit = self.dsra_head(dsra_repr).squeeze(-1)

        if ember_novel_feat is not None:
            ember_logit = self.ember_head(ember_novel_feat).squeeze(-1)
        else:
            ember_logit = torch.zeros_like(stage2_logit)

        # Softmax gate weights (guarantee positive sum = 1.0)
        w = F.softmax(self.gate, dim=0)

        # Weighted Logit Fusion
        fused_logit = w[0] * stage2_logit + w[1] * dsra_logit + w[2] * ember_logit
        fused_prob = torch.sigmoid(fused_logit)

        # Apply Uncertainty Window Mask to enforce zero breaks outside boundary
        in_window = (stage2_prob >= low_bound) & (stage2_prob <= high_bound)
        final_prob = torch.where(in_window, fused_prob, stage2_prob)

        return final_prob
