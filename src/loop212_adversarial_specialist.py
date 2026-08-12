"""Loop212: Adversarial Noise Augmentation Specialist Network.

Specialized neural network trained with byte-level Gaussian noise perturbations and section dropout
to increase robustness against evasive packed malware (deep FN cases).
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loop212AdversarialSpecialist(nn.Module):
    """Adversarial Noise Robustness Specialist."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 192) -> None:
        super().__init__()
        self.noise_std = 0.05
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return self.stem(x)
