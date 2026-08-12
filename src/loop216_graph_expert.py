"""Loop216: Heterogeneous Feature Graph Network (HFGN) Module.

Constructs PE structural nodes (Header, Section Entropy, Import/Export RVAs)
and computes Graph Neural Network embeddings for hard-example representation.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loop216GraphExpert(nn.Module):
    """Heterogeneous Feature Graph Expert Module."""

    def __init__(self, node_dim: int = 128, hidden_dim: int = 192) -> None:
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.gnn_layer = nn.Linear(hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 2),
        )

    def forward(self, node_feats: torch.Tensor) -> torch.Tensor:
        """node_feats: [B, num_nodes, node_dim]."""
        # Project nodes
        h = self.node_proj(node_feats)

        # Simple GNN aggregation over graph nodes
        h_graph = self.gnn_layer(h).mean(dim=1)

        logits = self.classifier(h_graph)
        return logits
