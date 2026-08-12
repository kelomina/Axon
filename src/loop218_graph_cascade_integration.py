"""Loop218: High-Order Graph Neural Network Cascade Integration.

Integrates:
- Primary Stage-2 GBDT Baseline (Loop151)
- Heterogeneous Feature Graph Expert (Loop216, Val F1 = 0.6618)
- Authenticode Trusted Signer Guard (Loop198, Net Repairs +44)
- Dynamic Cubic Polynomial Spline Annealer (Loop217, Breaks = 0)
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loop198_trusted_signer_guard import Loop198TrustedSignerGuard
from src.loop217_spline_annealer import Loop217SplineAnnealer


class Loop218GraphCascadeIntegration(nn.Module):
    """Graph Neural Network Cascade Integration Engine."""

    def __init__(self, primary_threshold: float = 0.31) -> None:
        super().__init__()
        self.primary_threshold = primary_threshold
        self.signer_guard = Loop198TrustedSignerGuard()
        self.spline_annealer = Loop217SplineAnnealer()

    def forward(
        self,
        primary_prob: float,
        graph_prob: float,
        auth_status: str,
        signer_subject: str,
    ) -> Tuple[int, float]:
        """Graph cascade forward pass."""
        base_pred = int(primary_prob >= self.primary_threshold)

        # 1. Authenticode Trusted Signer Guard
        guard_pred, is_down = self.signer_guard.evaluate_sample(base_pred, auth_status, signer_subject)
        if is_down:
            return 0, 0.05

        # 2. Cubic Spline Annealing
        annealed_prob = self.spline_annealer.anneal(primary_prob)

        # 3. Directional High Confidence Gated Override via Graph Expert
        if guard_pred == 0 and graph_prob >= 0.88:
            return 1, graph_prob

        if guard_pred == 1 and graph_prob <= 0.12:
            return 0, graph_prob

        return guard_pred, annealed_prob
