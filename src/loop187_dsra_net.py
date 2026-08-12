"""Loop187: Neural network architecture integrating upgraded MHDSRA2 with multi-layer retrieval.

Extends the Axon deep model family with:
- Upgraded MultiHeadDSRA2 (featuring retrieval quality adapter and projection aux)
- CPU-side PagedExactMemory for evidence retrieval
- Hybrid Byte-level Attention + Feature Fusion Head
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F

from src.dsra.mhdsra2 import MultiHeadDSRA2, MHDSRA2Config, MHDSRA2State
from src.dsra.mhdsra2.paged_exact_memory import PagedExactMemory


@dataclass(frozen=True)
class Loop187Config:
    vocabulary_size: int = 257
    padding_token: int = 256
    byte_embedding_dim: int = 64
    model_dim: int = 192
    num_heads: int = 8
    num_slots: int = 128
    read_topk: int = 8
    write_topk: int = 4
    local_window: int = 512
    use_retrieval: bool = True
    b0_feature_dim: int = 571
    dropout: float = 0.2
    num_classes: int = 2


class Loop187DSRANet(nn.Module):
    """Loop187 Neural Network with Upgraded MHDSRA2 Retrieval Branch."""

    def __init__(self, config: Optional[Loop187Config] = None) -> None:
        super().__init__()
        self.config = config or Loop187Config()
        cfg = self.config

        # 1. Byte Embedding Layer
        self.byte_embedding = nn.Embedding(cfg.vocabulary_size, cfg.byte_embedding_dim, padding_idx=cfg.padding_token)
        self.byte_proj = nn.Linear(cfg.byte_embedding_dim, cfg.model_dim)

        # 2. Upgraded MHDSRA2 Core Module
        mhdsra_cfg = MHDSRA2Config(
            dim=cfg.model_dim,
            heads=cfg.num_heads,
            slots=cfg.num_slots,
            read_topk=cfg.read_topk,
            write_topk=cfg.write_topk,
            local_window=cfg.local_window,
            use_retrieval=cfg.use_retrieval,
        )
        self.mhdsra = MultiHeadDSRA2(mhdsra_cfg)

        # 3. Static B0 Feature Stem
        self.b0_proj = nn.Sequential(
            nn.Linear(cfg.b0_feature_dim, cfg.model_dim),
            nn.LayerNorm(cfg.model_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
        )

        # 4. Fusion & Classification Head
        self.fusion_head = nn.Sequential(
            nn.Linear(cfg.model_dim * 2, cfg.model_dim),
            nn.LayerNorm(cfg.model_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.model_dim, cfg.num_classes),
        )

    def forward(
        self,
        x_bytes: torch.Tensor,
        b0_features: Optional[torch.Tensor] = None,
        retrieved_k: Optional[torch.Tensor] = None,
        retrieved_v: Optional[torch.Tensor] = None,
        retrieved_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, seq_len = x_bytes.shape

        # Byte Embedding
        emb = self.byte_embedding(x_bytes.long())
        h_bytes = self.byte_proj(emb)

        # MHDSRA2 Attention Layer with optional Multi-Layer Retrieval
        if return_aux:
            out_bytes, state, aux = self.mhdsra(
                h_bytes,
                retrieved_k=retrieved_k,
                retrieved_v=retrieved_v,
                retrieved_mask=retrieved_mask,
                return_aux=True,
                return_projection_aux=True,
            )
        else:
            out_bytes, state = self.mhdsra(
                h_bytes,
                retrieved_k=retrieved_k,
                retrieved_v=retrieved_v,
                retrieved_mask=retrieved_mask,
            )
            aux = {}

        # Global Mean Pooling on Sequence Dimension
        byte_repr = out_bytes.mean(dim=1)

        # Process B0 Features
        if b0_features is None:
            b0_features = torch.zeros(batch_size, self.config.b0_feature_dim, device=x_bytes.device, dtype=h_bytes.dtype)
        b0_repr = self.b0_proj(b0_features)

        # Fusion
        fused = torch.cat([byte_repr, b0_repr], dim=-1)
        logits = self.fusion_head(fused)

        if return_aux:
            return logits, aux
        return logits
