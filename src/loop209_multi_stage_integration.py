"""Loop209: Strict Zero-Breaks Multi-Stage Cascade Integration.

Integrates:
- Loop151 GBDT Baseline
- Loop198 Authenticode Signer Guard Extension (Net Repairs +44)
- Loop206 Certificate Serial Fingerprint Guard
- Loop202 Whole-File Multi-Chunk Streaming (Val F1 = 0.6742)
- Loop207 Contrastive Projection Embeddings
- Loop208 Rich Header Fusion Classifier
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loop198_trusted_signer_guard import Loop198TrustedSignerGuard
from src.loop206_signer_fingerprint_guard import Loop206SignerFingerprintGuard


class Loop209MultiStageIntegration(nn.Module):
    """Multi-Stage Cascade Integration Engine."""

    def __init__(self, primary_threshold: float = 0.31) -> None:
        super().__init__()
        self.primary_threshold = primary_threshold
        self.signer_guard = Loop198TrustedSignerGuard()
        self.fingerprint_guard = Loop206SignerFingerprintGuard()

    def forward(
        self,
        primary_prob: float,
        l202_prob: float,
        l207_sim: float,
        l208_prob: float,
        auth_status: str,
        signer_subject: str,
        cert_serial: str,
    ) -> Tuple[int, float]:
        """Multi-stage cascade integration forward pass."""
        base_pred = int(primary_prob >= self.primary_threshold)

        # 1. Certificate Fingerprint Guard
        fp_pred, fp_down = self.fingerprint_guard.evaluate_sample(base_pred, auth_status, cert_serial)
        if fp_down:
            return 0, 0.05

        # 2. Authenticode Trusted Signer Guard
        guard_pred, is_down = self.signer_guard.evaluate_sample(fp_pred, auth_status, signer_subject)
        if is_down:
            return 0, 0.05

        # 3. High-Confidence Directional Safeguard
        # FN Fix: base is 0, but whole-file streamer (l202) is very confident >= 0.88
        if guard_pred == 0 and l202_prob >= 0.88 and l207_sim >= 0.80:
            return 1, l202_prob

        # FP Fix: base is 1, but rich header & section fusion (l208) is very confident <= 0.12
        if guard_pred == 1 and l208_prob <= 0.12:
            return 0, l208_prob

        return guard_pred, primary_prob
