"""Exact low-memory Global Channel Gating primitives for Loop164.

The module accepts already-authorized byte tokens only.  It deliberately has
no filesystem or cache access: a future loader supplies one fresh iterable of
output-position chunks for each whole-file pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Protocol, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as functional

PAD_TOKEN = 0
RAW_BYTE_OFFSET = 1
VOCAB_SIZE = 257


@dataclass(frozen=True)
class OutputPartition:
    """A half-open interval in convolution output coordinates."""

    start: int
    end: int

    @property
    def output_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class OutputChunk:
    """Tokens needed to compute one contiguous range of convolution outputs."""

    output_start: int
    output_count: int
    tokens: torch.Tensor


@dataclass(frozen=True)
class WinnerSelection:
    """No-grad winner locations and their independent receptive-field windows."""

    values: torch.Tensor
    positions: torch.Tensor
    windows: torch.Tensor


class WholeFileByteSource(Protocol):
    """Supplies a fresh, contiguous chunk stream for each sequence pass."""

    def expected_output_count(
        self,
        *,
        receptive_field_bytes: int,
        output_stride_bytes: int,
    ) -> int:
        """Return the exact number of output positions in every pass."""

    def iter_output_chunks(
        self,
        *,
        receptive_field_bytes: int,
        output_stride_bytes: int,
        max_outputs_per_chunk: int,
    ) -> Iterable[OutputChunk]:
        """Yield output-position chunks from position zero through the padded tail."""


def output_partitions(output_count: int, max_outputs_per_chunk: int) -> Iterator[OutputPartition]:
    """Yield complete output-position partitions without dropping the tail."""

    if output_count < 1:
        raise ValueError("output_count must be positive")
    if max_outputs_per_chunk < 1:
        raise ValueError("max_outputs_per_chunk must be positive")
    for start in range(0, output_count, max_outputs_per_chunk):
        yield OutputPartition(start=start, end=min(start + max_outputs_per_chunk, output_count))


def padded_length_for_valid_length(
    valid_length: int,
    receptive_field_bytes: int,
    output_stride_bytes: int,
) -> int:
    """Return the right-padded length that gives the final byte a valid window."""

    if valid_length < 1:
        raise ValueError("valid_length must be positive")
    if receptive_field_bytes < 1:
        raise ValueError("receptive_field_bytes must be positive")
    if output_stride_bytes < 1:
        raise ValueError("output_stride_bytes must be positive")
    if valid_length <= receptive_field_bytes:
        return receptive_field_bytes
    excess = valid_length - receptive_field_bytes
    output_steps = (excess + output_stride_bytes - 1) // output_stride_bytes
    return receptive_field_bytes + output_steps * output_stride_bytes


def encode_raw_bytes(
    raw_bytes: Sequence[int] | torch.Tensor,
    *,
    padded_length: Optional[int] = None,
) -> tuple[torch.Tensor, int]:
    """Map raw ``0..255`` bytes to ``1..256`` while reserving zero for padding."""

    raw_tensor = torch.as_tensor(raw_bytes)
    if raw_tensor.ndim != 1:
        raise ValueError("raw_bytes must be one-dimensional")
    if raw_tensor.numel() < 1:
        raise ValueError("raw_bytes must not be empty")
    if raw_tensor.is_floating_point() or raw_tensor.is_complex():
        raise ValueError("raw_bytes must use an integral dtype")
    raw_tensor = raw_tensor.to(dtype=torch.long)
    if raw_tensor.lt(0).any() or raw_tensor.gt(255).any():
        raise ValueError("raw_bytes values must be in [0, 255]")

    valid_length = int(raw_tensor.numel())
    target_length = valid_length if padded_length is None else int(padded_length)
    if target_length < valid_length:
        raise ValueError("padded_length cannot truncate raw bytes")
    tokens = torch.zeros(target_length, dtype=torch.long, device=raw_tensor.device)
    tokens[:valid_length] = raw_tensor + RAW_BYTE_OFFSET
    return tokens, valid_length


class InMemoryByteSource:
    """Synthetic/test source with the same output-position interface as a stream loader."""

    def __init__(self, tokens: torch.Tensor, valid_length: int):
        if tokens.ndim != 1:
            raise ValueError("tokens must be one-dimensional")
        if tokens.is_floating_point() or tokens.is_complex():
            raise ValueError("tokens must use an integral dtype")
        if valid_length < 1 or valid_length > tokens.numel():
            raise ValueError("valid_length must select at least one provided token")
        token_values = tokens.detach().clone().to(dtype=torch.long)
        if token_values.lt(PAD_TOKEN).any() or token_values.ge(VOCAB_SIZE).any():
            raise ValueError("tokens must be in [0, 256]")
        if token_values[:valid_length].eq(PAD_TOKEN).any():
            raise ValueError("valid byte tokens must not use the reserved pad token")
        if token_values[valid_length:].ne(PAD_TOKEN).any():
            raise ValueError("tokens after valid_length must be padding")
        self._tokens = token_values
        self._valid_length = int(valid_length)

    @classmethod
    def from_raw_bytes(cls, raw_bytes: Sequence[int] | torch.Tensor) -> "InMemoryByteSource":
        tokens, valid_length = encode_raw_bytes(raw_bytes)
        return cls(tokens, valid_length)

    @property
    def valid_length(self) -> int:
        return self._valid_length

    @property
    def padded_tokens(self) -> torch.Tensor:
        return self._tokens

    def expected_output_count(
        self,
        *,
        receptive_field_bytes: int,
        output_stride_bytes: int,
    ) -> int:
        padded_length = padded_length_for_valid_length(
            self._valid_length,
            receptive_field_bytes,
            output_stride_bytes,
        )
        return (padded_length - receptive_field_bytes) // output_stride_bytes + 1

    def iter_output_chunks(
        self,
        *,
        receptive_field_bytes: int,
        output_stride_bytes: int,
        max_outputs_per_chunk: int,
    ) -> Iterable[OutputChunk]:
        padded_length = padded_length_for_valid_length(
            self._valid_length,
            receptive_field_bytes,
            output_stride_bytes,
        )
        if self._tokens.numel() < padded_length:
            padded_tokens = functional.pad(
                self._tokens,
                (0, padded_length - self._tokens.numel()),
                value=PAD_TOKEN,
            )
        else:
            padded_tokens = self._tokens[:padded_length]
        output_count = self.expected_output_count(
            receptive_field_bytes=receptive_field_bytes,
            output_stride_bytes=output_stride_bytes,
        )
        for partition in output_partitions(output_count, max_outputs_per_chunk):
            byte_start = partition.start * output_stride_bytes
            byte_end = (partition.end - 1) * output_stride_bytes + receptive_field_bytes
            yield OutputChunk(
                output_start=partition.start,
                output_count=partition.output_count,
                tokens=padded_tokens[byte_start:byte_end],
            )


class _ByteConvEncoder(nn.Module):
    """One MalConv2 GLU block with position-wise channel sharing."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        channels: int,
        receptive_field_bytes: int,
        output_stride_bytes: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=PAD_TOKEN)
        self.conv = nn.Conv1d(
            embedding_dim,
            channels * 2,
            kernel_size=receptive_field_bytes,
            stride=output_stride_bytes,
        )
        self.channel_share = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("encoder tokens must have shape [batch, bytes]")
        encoded = self.embedding(tokens).transpose(1, 2)
        activations = functional.glu(self.conv(encoded), dim=1)
        return functional.leaky_relu(self.channel_share(activations), negative_slope=0.01)


class _GlobalChannelGatedEncoder(_ByteConvEncoder):
    """Apply a learned global channel gate to chunk-local byte activations."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        channels: int,
        receptive_field_bytes: int,
        output_stride_bytes: int,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            channels=channels,
            receptive_field_bytes=receptive_field_bytes,
            output_stride_bytes=output_stride_bytes,
        )
        self.context_projection = nn.Linear(channels, channels)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        activations = super().forward(tokens)
        if context.ndim != 2 or context.shape[0] != activations.shape[0]:
            raise ValueError("context must have shape [batch, channels]")
        if context.shape[1] != activations.shape[1]:
            raise ValueError("context channel count must match byte activations")
        projected_context = torch.tanh(self.context_projection(context)).unsqueeze(-1)
        gates = torch.sigmoid((activations * projected_context).sum(dim=1, keepdim=True))
        return activations * gates


class WholeFileGCGClassifier(nn.Module):
    """Two-pass GCG classifier with exact chunk winner reconstruction.

    Each pass scans all output positions without gradients, stores only one
    receptive-field window per channel, then recomputes those windows with
    gradients.  This preserves first-winner max semantics without concatenating
    noncontiguous byte regions into a synthetic sequence.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 8,
        channels: int = 64,
        receptive_field_bytes: int = 512,
        output_stride_bytes: int = 256,
        max_outputs_per_chunk: int = 255,
        num_classes: int = 2,
    ):
        super().__init__()
        for name, value in {
            "embedding_dim": embedding_dim,
            "channels": channels,
            "receptive_field_bytes": receptive_field_bytes,
            "output_stride_bytes": output_stride_bytes,
            "max_outputs_per_chunk": max_outputs_per_chunk,
            "num_classes": num_classes,
        }.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if int(output_stride_bytes) > int(receptive_field_bytes):
            raise ValueError("output_stride_bytes cannot exceed receptive_field_bytes")
        self.channels = int(channels)
        self.receptive_field_bytes = int(receptive_field_bytes)
        self.output_stride_bytes = int(output_stride_bytes)
        self.max_outputs_per_chunk = int(max_outputs_per_chunk)
        self.context_encoder = _ByteConvEncoder(
            embedding_dim=int(embedding_dim),
            channels=self.channels,
            receptive_field_bytes=self.receptive_field_bytes,
            output_stride_bytes=self.output_stride_bytes,
        )
        self.gated_encoder = _GlobalChannelGatedEncoder(
            embedding_dim=int(embedding_dim),
            channels=self.channels,
            receptive_field_bytes=self.receptive_field_bytes,
            output_stride_bytes=self.output_stride_bytes,
        )
        self.feature_projection = nn.Linear(self.channels, self.channels)
        self.classifier = nn.Linear(self.channels, int(num_classes))

    @property
    def device(self) -> torch.device:
        return self.context_encoder.embedding.weight.device

    def _validate_chunk(
        self,
        chunk: OutputChunk,
        *,
        expected_output_start: int,
    ) -> None:
        if chunk.output_start != expected_output_start:
            raise ValueError("output chunks must be contiguous and begin at zero")
        if chunk.output_count < 1:
            raise ValueError("output chunks must contain at least one output")
        if chunk.output_count > self.max_outputs_per_chunk:
            raise ValueError("output chunk exceeds max_outputs_per_chunk")
        if chunk.tokens.ndim != 1:
            raise ValueError("output chunk tokens must be one-dimensional")
        required_length = (
            (chunk.output_count - 1) * self.output_stride_bytes + self.receptive_field_bytes
        )
        if chunk.tokens.numel() != required_length:
            raise ValueError("output chunk does not contain exact receptive-field coverage")
        if chunk.tokens.is_floating_point() or chunk.tokens.is_complex():
            raise ValueError("output chunk tokens must use an integral dtype")
        if chunk.tokens.lt(PAD_TOKEN).any() or chunk.tokens.ge(VOCAB_SIZE).any():
            raise ValueError("output chunk tokens must be in [0, 256]")

    def _scan_winners(
        self,
        source: WholeFileByteSource,
        encoder: nn.Module,
        *,
        context: Optional[torch.Tensor] = None,
    ) -> WinnerSelection:
        device = self.device
        best_values: Optional[torch.Tensor] = None
        best_positions: Optional[torch.Tensor] = None
        best_windows: Optional[torch.Tensor] = None
        expected_output_start = 0
        chunk_count = 0
        expected_output_count = source.expected_output_count(
            receptive_field_bytes=self.receptive_field_bytes,
            output_stride_bytes=self.output_stride_bytes,
        )
        if expected_output_count < 1:
            raise ValueError("whole-file source must declare at least one output")

        # 扫描阶段不保留整文件激活图，只保存每个通道当前最佳的独立感受野窗口。
        with torch.no_grad():
            for chunk in source.iter_output_chunks(
                receptive_field_bytes=self.receptive_field_bytes,
                output_stride_bytes=self.output_stride_bytes,
                max_outputs_per_chunk=self.max_outputs_per_chunk,
            ):
                self._validate_chunk(chunk, expected_output_start=expected_output_start)
                expected_output_start += chunk.output_count
                if expected_output_start > expected_output_count:
                    raise ValueError("output chunks exceed declared whole-file coverage")
                chunk_count += 1
                token_batch = chunk.tokens.to(device=device, dtype=torch.long).unsqueeze(0)
                if context is None:
                    activations = encoder(token_batch)
                else:
                    activations = encoder(token_batch, context.detach().unsqueeze(0))
                if activations.shape != (1, self.channels, chunk.output_count):
                    raise ValueError("encoder output shape does not match output chunk coordinates")
                if not torch.isfinite(activations).all():
                    raise ValueError("encoder produced non-finite activations")

                chunk_values, local_positions = activations.squeeze(0).max(dim=-1)
                global_positions = local_positions + chunk.output_start
                window_offsets = (
                    local_positions.unsqueeze(1) * self.output_stride_bytes
                    + torch.arange(self.receptive_field_bytes, device=device).unsqueeze(0)
                )
                candidate_windows = token_batch.squeeze(0)[window_offsets]

                if best_values is None:
                    best_values = torch.full_like(chunk_values, float("-inf"))
                    best_positions = torch.zeros_like(global_positions)
                    best_windows = torch.zeros(
                        (self.channels, self.receptive_field_bytes),
                        dtype=torch.long,
                        device=device,
                    )
                replace = chunk_values > best_values
                best_values = torch.where(replace, chunk_values, best_values)
                best_positions = torch.where(replace, global_positions, best_positions)
                best_windows = torch.where(replace.unsqueeze(1), candidate_windows, best_windows)

        if chunk_count == 0 or best_values is None or best_positions is None or best_windows is None:
            raise ValueError("whole-file source did not provide any output chunks")
        if expected_output_start != expected_output_count:
            raise ValueError("output chunks do not cover the declared whole file")
        return WinnerSelection(values=best_values, positions=best_positions, windows=best_windows)

    def _recompute_winners(
        self,
        selection: WinnerSelection,
        encoder: nn.Module,
        *,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        windows = selection.windows.to(device=self.device, dtype=torch.long)
        if context is None:
            activations = encoder(windows)
        else:
            repeated_context = context.unsqueeze(0).expand(self.channels, -1)
            activations = encoder(windows, repeated_context)
        if activations.shape != (self.channels, self.channels, 1):
            raise ValueError("winner-window reconstruction must produce exactly one output per window")
        diagonal = torch.arange(self.channels, device=self.device)
        return activations[diagonal, diagonal, 0]

    def find_context_winners(self, source: WholeFileByteSource) -> WinnerSelection:
        """Expose context winners for exactness tests and future loader integration."""

        return self._scan_winners(source, self.context_encoder)

    def _forward_single_source(self, source: WholeFileByteSource) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_selection = self._scan_winners(source, self.context_encoder)
        context = self._recompute_winners(context_selection, self.context_encoder)
        gated_selection = self._scan_winners(source, self.gated_encoder, context=context)
        pooled = self._recompute_winners(gated_selection, self.gated_encoder, context=context)
        features = functional.leaky_relu(self.feature_projection(pooled), negative_slope=0.01)
        return self.classifier(features), features, context

    def forward_from_source(
        self,
        source: WholeFileByteSource,
        *,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Classify one source through exactly two fresh output-chunk scans."""

        logits, features, context = self._forward_single_source(source)
        result = {"logits": logits.unsqueeze(0)}
        if return_features:
            result["features"] = features.unsqueeze(0)
            result["context"] = context.unsqueeze(0)
        return result

    @staticmethod
    def _normalize_valid_lengths(
        tokens: torch.Tensor,
        valid_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, bytes]")
        if valid_lengths is None:
            valid_lengths = tokens.ne(PAD_TOKEN).sum(dim=1)
        valid_lengths = torch.as_tensor(valid_lengths, device=tokens.device, dtype=torch.long)
        if valid_lengths.ndim != 1 or valid_lengths.numel() != tokens.shape[0]:
            raise ValueError("valid_lengths must have one value per batch row")
        if valid_lengths.lt(1).any() or valid_lengths.gt(tokens.shape[1]).any():
            raise ValueError("valid_lengths must select at least one available token")
        for row_index, valid_length in enumerate(valid_lengths.tolist()):
            if tokens[row_index, :valid_length].eq(PAD_TOKEN).any():
                raise ValueError("valid byte tokens must not use the reserved pad token")
            if tokens[row_index, valid_length:].ne(PAD_TOKEN).any():
                raise ValueError("tokens after valid_length must be padding")
        return valid_lengths

    def _sources_from_batch(
        self,
        tokens: torch.Tensor,
        valid_lengths: Optional[torch.Tensor],
    ) -> list[InMemoryByteSource]:
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        lengths = self._normalize_valid_lengths(tokens, valid_lengths)
        if tokens.is_floating_point() or tokens.is_complex():
            raise ValueError("tokens must use an integral dtype")
        if tokens.lt(PAD_TOKEN).any() or tokens.ge(VOCAB_SIZE).any():
            raise ValueError("tokens must be in [0, 256]")
        return [
            InMemoryByteSource(tokens[row_index], int(valid_length))
            for row_index, valid_length in enumerate(lengths.tolist())
        ]

    def forward(
        self,
        tokens: torch.Tensor,
        valid_lengths: Optional[torch.Tensor] = None,
        *,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Classify padded reserved-token batches using per-row whole-file scans."""

        rows = [
            self.forward_from_source(source, return_features=return_features)
            for source in self._sources_from_batch(tokens, valid_lengths)
        ]
        result = {"logits": torch.cat([row["logits"] for row in rows], dim=0)}
        if return_features:
            result["features"] = torch.cat([row["features"] for row in rows], dim=0)
            result["context"] = torch.cat([row["context"] for row in rows], dim=0)
        return result

    def _forward_dense_single(
        self,
        source: InMemoryByteSource,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded_length = padded_length_for_valid_length(
            source.valid_length,
            self.receptive_field_bytes,
            self.output_stride_bytes,
        )
        tokens = source.padded_tokens
        if tokens.numel() < padded_length:
            tokens = functional.pad(tokens, (0, padded_length - tokens.numel()), value=PAD_TOKEN)
        else:
            tokens = tokens[:padded_length]
        token_batch = tokens.to(device=self.device, dtype=torch.long).unsqueeze(0)
        context = self.context_encoder(token_batch).max(dim=-1).values.squeeze(0)
        pooled = self.gated_encoder(token_batch, context.unsqueeze(0)).max(dim=-1).values.squeeze(0)
        features = functional.leaky_relu(self.feature_projection(pooled), negative_slope=0.01)
        return self.classifier(features), features, context

    def forward_dense(
        self,
        tokens: torch.Tensor,
        valid_lengths: Optional[torch.Tensor] = None,
        *,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Dense in-memory reference used only by synthetic equivalence tests."""

        rows = [self._forward_dense_single(source) for source in self._sources_from_batch(tokens, valid_lengths)]
        result = {"logits": torch.stack([row[0] for row in rows], dim=0)}
        if return_features:
            result["features"] = torch.stack([row[1] for row in rows], dim=0)
            result["context"] = torch.stack([row[2] for row in rows], dim=0)
        return result
