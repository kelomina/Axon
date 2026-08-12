"""Loop204: Deep Ultra-High Capacity Stacker Engine.

Combines:
- Primary Stage-2 GBDT Baseline (Loop151)
- Whole-File Multi-Chunk Streaming Encoder (Loop202, Val F1 = 0.6742)
- Focal Loss Hard Mining Specialist (Loop197, Val F1 = 0.6028)
- Micro-Section Anomaly Features (Loop203)
- Authenticode Trusted Signer Guard Extension (Loop198)
- Zero-Breaks Directional Mask (Loop194)
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loop198_trusted_signer_guard import Loop198TrustedSignerGuard


class Loop204UltraCapacityStacker(nn.Module):
    """Deep Ultra-High Capacity Stacker Engine."""

    def __init__(self, primary_threshold: float = 0.31) -> None:
        super().__init__()
        self.primary_threshold = primary_threshold
        self.signer_guard = Loop198TrustedSignerGuard()

        self.fused_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        primary_prob: float,
        l202_prob: float,
        l197_prob: float,
        l203_entropy: float,
        auth_status: str,
        signer_subject: str,
    ) -> Tuple[int, float]:
        """Deep Ultra-Capacity forward pass."""
        # 1. Base prediction
        base_pred = int(primary_prob >= self.primary_threshold)

        # 2. Check Authenticode trusted signer guard
        guard_pred, is_down = self.signer_guard.evaluate_sample(base_pred, auth_status, signer_subject)
        if is_down:
            return 0, 0.05

        # 3. High-capacity fusion
        input_tensor = torch.tensor([[primary_prob, l202_prob, l197_prob, l203_entropy]], dtype=torch.float32)
        fused_prob = self.fused_net(input_tensor).item()

        # 4. Strict Zero-Breaks Directional Guard
        # FN Repair: Primary pred is 0, but fused prob >= 0.85 -> repair to 1
        if base_pred == 0 and fused_prob >= 0.85:
            return 1, fused_prob

        # FP Repair: Primary pred is 1, but fused prob <= 0.15 -> repair to 0
        if base_pred == 1 and fused_prob <= 0.15:
            return 0, fused_prob

        return guard_pred, primary_prob
