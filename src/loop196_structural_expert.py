"""Loop196: EMBER-292 + KVD Structural Hybrid Expert Module.

Integrates:
- 292-dim EMBER-v3 Novel Structural Features (PE Header, Entropy, Section, Export/Import RVAs)
- KVD Structural Representations
- Multi-Layer MLP & Tree Hybrid Classifier for Hard-Example Disambiguation
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loop196StructuralExpert(nn.Module):
    """High-Order Structural Hybrid Expert Model."""

    def __init__(self, ember_dim: int = 292, kvd_dim: int = 571, hidden_dim: int = 256) -> None:
        super().__init__()
        self.ember_stem = nn.Sequential(
            nn.Linear(ember_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
        )

        self.kvd_stem = nn.Sequential(
            nn.Linear(kvd_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, ember_feat: torch.Tensor, kvd_feat: torch.Tensor) -> torch.Tensor:
        """Returns logits for binary malware classification [B, 2]."""
        h_ember = self.ember_stem(ember_feat)
        h_kvd = self.kvd_stem(kvd_feat)
        fused = torch.cat([h_ember, h_kvd], dim=-1)
        logits = self.fusion_head(fused)
        return logits
