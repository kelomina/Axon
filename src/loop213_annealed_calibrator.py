"""Loop213: Multi-Threshold Sigmoidal Annealing Calibrator Module.

Applies fine-grained sigmoidal temperature annealing to smooth boundary probabilities
around PRIMARY_THR = 0.31.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loop213AnnealedCalibrator(nn.Module):
    """Sigmoidal Annealing Probability Calibrator."""

    def __init__(self, target_thr: float = 0.31, initial_temp: float = 1.15) -> None:
        super().__init__()
        self.target_thr = target_thr
        self.temp = initial_temp

    def calibrate(self, prob: float) -> float:
        """Calibrates marginal probabilities using sigmoidal temperature annealing."""
        p_clamped = max(1e-7, min(1 - 1e-7, prob))
        logit = math_logit(p_clamped)
        calibrated_logit = logit / self.temp
        return float(1.0 / (1.0 + torch.exp(-torch.tensor(calibrated_logit)).item()))


def math_logit(p: float) -> float:
    import math

    return math.log(p / (1.0 - p))
