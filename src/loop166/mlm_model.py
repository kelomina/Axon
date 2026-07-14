"""Resource-bounded local-attention masked language model for Loop166."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class TinyMLMConfig:
    vocab_size: int = 1029
    sequence_tokens: int = 512
    layers: int = 6
    hidden_dim: int = 384
    heads: int = 6
    ffn_dim: int = 1536
    local_attention_window: int = 128
    global_token_index: int = 0
    dropout: float = 0.1
    activation: str = "gelu"
    gradient_checkpointing: bool = True
    tied_input_output_embeddings: bool = True
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        positive_fields = {
            "vocab_size": self.vocab_size,
            "sequence_tokens": self.sequence_tokens,
            "layers": self.layers,
            "hidden_dim": self.hidden_dim,
            "heads": self.heads,
            "ffn_dim": self.ffn_dim,
            "local_attention_window": self.local_attention_window,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_dim % self.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        if not 0 <= self.global_token_index < self.sequence_tokens:
            raise ValueError("global_token_index must fall inside sequence_tokens")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.activation not in {"gelu", "relu"}:
            raise ValueError("activation must be gelu or relu")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must fall inside vocab_size")


def build_cls_global_local_attention_mask(
    sequence_length: int,
    local_window: int,
    *,
    global_token_index: int = 0,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Build a boolean visibility mask where CLS is global and other tokens are local."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if local_window <= 0:
        raise ValueError("local_window must be positive")
    if not 0 <= global_token_index < sequence_length:
        raise ValueError("global_token_index must fall inside sequence_length")

    positions = torch.arange(sequence_length, device=device)
    distance = (positions[:, None] - positions[None, :]).abs()
    visibility = distance <= local_window
    visibility[global_token_index, :] = True
    visibility[:, global_token_index] = True
    return visibility


class TinyMaskedLanguageModel(nn.Module):
    """Six-layer encoder with tied embeddings and resource-safe local attention."""

    def __init__(self, config: TinyMLMConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_dim,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(config.sequence_tokens, config.hidden_dim)
        self.embedding_norm = nn.LayerNorm(config.hidden_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_dim,
                    nhead=config.heads,
                    dim_feedforward=config.ffn_dim,
                    dropout=config.dropout,
                    activation=config.activation,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=True)
        if config.tied_input_output_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        self.register_buffer(
            "_local_attention_mask",
            build_cls_global_local_attention_mask(
                config.sequence_tokens,
                config.local_attention_window,
                global_token_index=config.global_token_index,
            ),
            persistent=False,
        )

    def _run_encoder_layer(
        self,
        layer: nn.TransformerEncoderLayer,
        hidden_states: Tensor,
        attention_mask: Tensor,
        padding_mask: Optional[Tensor],
    ) -> Tensor:
        if self.config.gradient_checkpointing and self.training:
            def checkpointed_layer(states: Tensor) -> Tensor:
                return layer(
                    states,
                    src_mask=attention_mask,
                    src_key_padding_mask=padding_mask,
                )

            return checkpoint(checkpointed_layer, hidden_states, use_reentrant=False)
        return layer(
            hidden_states,
            src_mask=attention_mask,
            src_key_padding_mask=padding_mask,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.sequence_tokens:
            raise ValueError(
                f"sequence length {sequence_length} exceeds {self.config.sequence_tokens}"
            )
        if sequence_length <= self.config.global_token_index:
            raise ValueError("input sequence does not contain the configured global token")

        if attention_mask is None:
            valid_tokens = input_ids.ne(self.config.pad_token_id)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids")
            valid_tokens = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        padding_mask = ~valid_tokens if not bool(valid_tokens.all()) else None

        positions = torch.arange(sequence_length, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        hidden_states = self.embedding_dropout(self.embedding_norm(hidden_states))
        # TransformerEncoderLayer uses True for blocked pairs, opposite to the public visibility mask.
        local_mask = ~self._local_attention_mask[:sequence_length, :sequence_length]

        for layer in self.encoder_layers:
            hidden_states = self._run_encoder_layer(
                layer,
                hidden_states,
                local_mask,
                padding_mask,
            )
        logits = self.lm_head(self.final_norm(hidden_states))
        output = {"logits": logits}

        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids")
            output["loss"] = functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return output


TinyCodeMLM = TinyMaskedLanguageModel


def count_parameters(model: nn.Module) -> int:
    """Count unique model parameters, including frozen parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count unique parameters that participate in optimization."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
