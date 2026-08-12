"""Loop217: Dynamic Cubic Polynomial Spline Annealer Module.

Smooths marginal probabilities near PRIMARY_THR = 0.31 using a cubic Hermite spline
to eliminate sharp threshold quantization artifacts.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loop217SplineAnnealer(nn.Module):
    """Cubic Hermite Spline Probability Annealer."""

    def __init__(self, low: float = 0.29, high: float = 0.33) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def anneal(self, p: float) -> float:
        """Applies cubic smoothstep annealing in [low, high]."""
        if p <= self.low:
            return p
        if p >= self.high:
            return p
        # Normalized t in [0, 1]
        t = (p - self.low) / (self.high - self.low)
        # Cubic smoothstep: 3t^2 - 2t^3
        smooth_t = 3 * (t**2) - 2 * (t**3)
        return self.low + smooth_t * (self.high - self.low)
