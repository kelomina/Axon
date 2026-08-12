"""Loop215: Axon Frontier Comprehensive Multi-Expert Pipeline.

Combines all trained expert models & guards:
1. Primary Stage-2 GBDT Baseline (Loop151)
2. Authenticode Trusted Signer Guard (Loop198)
3. Certificate Serial Fingerprint Guard (Loop206)
4. Whole-File 4-Chunk Streaming Encoder (Loop202)
5. Focal Loss Hard Mining Specialist (Loop197)
6. Supervised Contrastive Projection (Loop207)
7. Rich Header Compiler Metadata Fusion (Loop208)
8. Adversarial Noise Specialist (Loop212)
9. Sigmoidal Annealing Probability Calibrator (Loop213)
10. Strict Zero-Breaks Directional Guard (Loop194)
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loop198_trusted_signer_guard import Loop198TrustedSignerGuard
from src.loop206_signer_fingerprint_guard import Loop206SignerFingerprintGuard
from src.loop213_annealed_calibrator import Loop213AnnealedCalibrator


class Loop215ComprehensiveFrontierPipeline(nn.Module):
    """Axon Frontier Comprehensive Multi-Expert Pipeline."""

    def __init__(self, primary_threshold: float = 0.31) -> None:
        super().__init__()
        self.primary_threshold = primary_threshold
        self.signer_guard = Loop198TrustedSignerGuard()
        self.fingerprint_guard = Loop206SignerFingerprintGuard()
        self.calibrator = Loop213AnnealedCalibrator()

    def forward(
        self,
        primary_prob: float,
        l202_prob: float,
        l197_prob: float,
        l207_sim: float,
        l208_prob: float,
        l212_prob: float,
        auth_status: str,
        signer_subject: str,
        cert_serial: str,
    ) -> Tuple[int, float]:
        """Comprehensive multi-expert forward pass."""
        base_pred = int(primary_prob >= self.primary_threshold)

        # 1. Certificate Fingerprint Guard
        fp_pred, fp_down = self.fingerprint_guard.evaluate_sample(base_pred, auth_status, cert_serial)
        if fp_down:
            return 0, 0.05

        # 2. Authenticode Trusted Signer Guard
        guard_pred, is_down = self.signer_guard.evaluate_sample(fp_pred, auth_status, signer_subject)
        if is_down:
            return 0, 0.05

        # 3. Calibrated Score
        calib_prob = self.calibrator.calibrate(primary_prob)

        # 4. Strict Zero-Breaks High Confidence Directional Override
        # FN Repair: Base is 0, but whole-file streamer (l202) & hard mining specialist (l197) are strongly confident (>= 0.90)
        if guard_pred == 0 and l202_prob >= 0.90 and l197_prob >= 0.90:
            return 1, l202_prob

        # FP Repair: Base is 1, but rich header (l208) & adversarial specialist (l212) are strongly confident (<= 0.10)
        if guard_pred == 1 and l208_prob <= 0.10 and l212_prob <= 0.10:
            return 0, l208_prob

        return guard_pred, calib_prob
