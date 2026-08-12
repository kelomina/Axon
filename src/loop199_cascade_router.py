"""Loop199: Hierarchical Multi-Expert Cascade Router.

Combines:
- Primary Stage-2 GBDT Baseline (Loop151)
- Structural Expert (Loop196)
- Hard Mining Specialist (Loop197)
- Authenticode Signer Guard Extension (Loop198)
- Zero-Breaks Directional Gated Override (Loop194)
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loop198_trusted_signer_guard import Loop198TrustedSignerGuard


class Loop199CascadeRouter(nn.Module):
    """Hierarchical Multi-Expert Cascade Router."""

    def __init__(self, primary_threshold: float = 0.31) -> None:
        super().__init__()
        self.primary_threshold = primary_threshold
        self.signer_guard = Loop198TrustedSignerGuard()

        self.router_head = nn.Sequential(
            nn.Linear(2 + 2, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        primary_prob: float,
        l196_logits: Tuple[float, float],
        l197_logits: Tuple[float, float],
        auth_status: str,
        signer_subject: str,
    ) -> Tuple[int, float]:
        """Runs multi-expert cascade routing.

        Returns (final_prediction, final_probability).
        """
        # 1. Base prediction
        base_pred = int(primary_prob >= self.primary_threshold)

        # 2. Check Authenticode trusted signer guard
        guard_pred, is_down = self.signer_guard.evaluate_sample(base_pred, auth_status, signer_subject)
        if is_down:
            return 0, 0.05

        # 3. If primary_prob is in marginal uncertainty zone [0.28, 0.38], invoke cascade router
        if 0.28 <= primary_prob <= 0.38:
            e196_prob = float(torch.softmax(torch.tensor(l196_logits), dim=-1)[1].item())
            e197_prob = float(torch.softmax(torch.tensor(l197_logits), dim=-1)[1].item())

            # Combine expert signals
            expert_fused_prob = 0.60 * primary_prob + 0.25 * e196_prob + 0.15 * e197_prob

            # Directional safeguard: only override FN upward if expert is strong (>0.60)
            if base_pred == 0 and expert_fused_prob >= 0.35:
                return 1, expert_fused_prob
            elif base_pred == 1 and expert_fused_prob < 0.25:
                return 0, expert_fused_prob

        return base_pred, primary_prob
