"""Loop214: Second-Order Zero-Breaks Residual Stacker Engine.

Integrates:
- Loop151 GBDT Baseline
- Loop198 Authenticode Trusted Signer Guard (Net Repairs +44)
- Loop212 Adversarial Noise Specialist
- Loop213 Sigmoidal Annealing Calibrated Probabilities
- Strict Zero-Breaks Directional Mask (Loop194)
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loop198_trusted_signer_guard import Loop198TrustedSignerGuard


class Loop214SecondOrderStacker(nn.Module):
    """Second-Order Zero-Breaks Residual Stacker Engine."""

    def __init__(self, primary_threshold: float = 0.31) -> None:
        super().__init__()
        self.primary_threshold = primary_threshold
        self.signer_guard = Loop198TrustedSignerGuard()

    def forward(
        self,
        primary_prob: float,
        adv_prob: float,
        calib_prob: float,
        auth_status: str,
        signer_subject: str,
    ) -> Tuple[int, float]:
        """Second-order residual stacker forward pass."""
        base_pred = int(primary_prob >= self.primary_threshold)

        # 1. Authenticode Trusted Signer Guard
        guard_pred, is_down = self.signer_guard.evaluate_sample(base_pred, auth_status, signer_subject)
        if is_down:
            return 0, 0.05

        # 2. Strict Zero-Breaks Directional Guard
        # FN Repair: Primary pred is 0, but both adv_prob & calib_prob >= 0.88 -> repair to 1
        if guard_pred == 0 and adv_prob >= 0.88 and calib_prob >= 0.88:
            return 1, calib_prob

        # FP Repair: Primary pred is 1, but calib_prob <= 0.12 -> repair to 0
        if guard_pred == 1 and calib_prob <= 0.12:
            return 0, calib_prob

        return guard_pred, primary_prob
