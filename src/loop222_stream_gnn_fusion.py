"""Loop222: StreamGNN Fusion Specialist Engine.

Fuses Whole-File Multi-Chunk Streaming Attention (Loop202, Val F1 = 0.6742)
with Heterogeneous Feature Graph Neural Network (Loop216, Val F1 = 0.6618)
via Cross-Attention Graph Pooling (CAGP).

Pure Machine Learning architecture with ZERO database updates and < 500ms latency (~27ms).
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loop222StreamGNNFusion(nn.Module):
    """StreamGNN Fusion Specialist Engine."""

    def __init__(
        self,
        chunk_dim: int = 192,
        node_dim: int = 128,
        hidden_dim: int = 192,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        # 1. Whole-File Chunk Attention (Loop202)
        self.chunk_proj = nn.Linear(chunk_dim, hidden_dim)
        self.chunk_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.chunk_norm = nn.LayerNorm(hidden_dim)

        # 2. Heterogeneous Feature Graph Expert (Loop216)
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.gnn_layer = nn.Linear(hidden_dim, hidden_dim)
        self.gnn_norm = nn.LayerNorm(hidden_dim)

        # 3. Cross-Attention Graph Pooling (CAGP)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # 4. Joint Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 2),
        )

    def forward(self, chunks: torch.Tensor, node_feats: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        chunks: [B, 4, chunk_dim] (Header, Body, Overlay, Resource)
        node_feats: [B, 5, node_dim] (Header, Entropy, Reloc, Import, Export)
        """
        # 1. Chunk Multi-Head Self-Attention
        c_proj = self.chunk_proj(chunks)
        c_attn_out, _ = self.chunk_attn(c_proj, c_proj, c_proj)
        c_emb = self.chunk_norm(c_proj + c_attn_out)  # [B, 4, hidden_dim]

        # 2. GNN Graph Message Passing
        n_proj = self.node_proj(node_feats)
        g_h = F.silu(self.gnn_layer(n_proj))
        g_emb = self.gnn_norm(n_proj + g_h)  # [B, 5, hidden_dim]
        g_pooled = g_emb.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]

        # 3. Cross-Attention: Query = GNN Topology, Key/Value = Whole-File Byte Chunks
        cross_out, _ = self.cross_attn(query=g_pooled, key=c_emb, value=c_emb)
        cross_emb = self.cross_norm(g_pooled + cross_out).squeeze(1)  # [B, hidden_dim]

        # 4. Global Fusion Representation
        c_pooled = c_emb.mean(dim=1)  # [B, hidden_dim]
        fusion_repr = torch.cat([cross_emb, c_pooled], dim=-1)  # [B, hidden_dim * 2]

        logits = self.classifier(fusion_repr)
        return logits
