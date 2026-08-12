"""Loop202: Whole-File Multi-Chunk Streaming Representation Engine.

Extracts multi-region streaming features from PE binaries:
1. Header Chunk (first 8192 bytes)
2. Middle Body Chunk (middle 8192 bytes)
3. Tail Overlay Chunk (last 8192 bytes)
4. Resource Section Chunk (if present)

Fused via Multi-Head Chunk Attention into a unified 256-dim Whole-File Representation.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Loop202WholeFileStreamer(nn.Module):
    """Whole-File Multi-Chunk Streaming Encoder."""

    def __init__(self, chunk_dim: int = 192, hidden_dim: int = 256) -> None:
        super().__init__()
        self.chunk_proj = nn.Sequential(
            nn.Linear(chunk_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.chunk_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
        )

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        """Forward pass for chunks [B, 4, 192] -> logits [B, 2]."""
        # 1. Project chunks -> [B, 4, hidden_dim]
        h_chunks = self.chunk_proj(chunks)

        # 2. Multi-head Chunk Attention
        attn_out, _ = self.chunk_attention(h_chunks, h_chunks, h_chunks)

        # 3. Pooled representation -> [B, hidden_dim]
        pooled = attn_out.mean(dim=1)

        # 4. Binary Classification Logits
        logits = self.classifier(pooled)
        return logits
