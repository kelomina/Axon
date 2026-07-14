from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loop164.whole_file_gcg import (  # noqa: E402
    PAD_TOKEN,
    InMemoryByteSource,
    WholeFileGCGClassifier,
    encode_raw_bytes,
    padded_length_for_valid_length,
)


def _model() -> WholeFileGCGClassifier:
    torch.manual_seed(164)
    return WholeFileGCGClassifier(
        embedding_dim=3,
        channels=4,
        receptive_field_bytes=4,
        output_stride_bytes=2,
        max_outputs_per_chunk=2,
    ).double()


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    first, first_length = encode_raw_bytes([0, 255, 3, 7, 10, 4, 1, 9, 12])
    second, second_length = encode_raw_bytes([9, 1, 0, 2, 8, 4])
    width = max(first.numel(), second.numel())
    batch = torch.zeros((2, width), dtype=torch.long)
    batch[0, : first.numel()] = first
    batch[1, : second.numel()] = second
    return batch, torch.tensor([first_length, second_length], dtype=torch.long)


def _loss(result: dict[str, torch.Tensor]) -> torch.Tensor:
    return result["logits"].square().sum() + result["features"].square().sum()


def test_chunked_model_matches_dense_logits_features_context_and_gradients():
    chunked_model = _model()
    dense_model = copy.deepcopy(chunked_model)
    tokens, valid_lengths = _batch()

    chunked = chunked_model(tokens, valid_lengths, return_features=True)
    dense = dense_model.forward_dense(tokens, valid_lengths, return_features=True)
    chunked_loss = _loss(chunked)
    dense_loss = _loss(dense)
    chunked_loss.backward()
    dense_loss.backward()

    assert torch.allclose(chunked_loss, dense_loss, rtol=1e-10, atol=1e-10)
    for name in ("logits", "features", "context"):
        assert torch.allclose(chunked[name], dense[name], rtol=1e-10, atol=1e-10)
    for (chunked_name, chunked_parameter), (dense_name, dense_parameter) in zip(
        chunked_model.named_parameters(), dense_model.named_parameters()
    ):
        assert chunked_name == dense_name
        assert chunked_parameter.grad is not None
        assert dense_parameter.grad is not None
        assert torch.allclose(chunked_parameter.grad, dense_parameter.grad, rtol=1e-10, atol=1e-10)


def test_reserved_pad_keeps_raw_zero_distinct_and_covers_tail():
    source = InMemoryByteSource.from_raw_bytes([0, 255, 3, 7, 1, 2, 4])
    model = _model()
    chunks = list(
        source.iter_output_chunks(
            receptive_field_bytes=model.receptive_field_bytes,
            output_stride_bytes=model.output_stride_bytes,
            max_outputs_per_chunk=model.max_outputs_per_chunk,
        )
    )

    assert source.padded_tokens[0].item() == 1
    assert source.padded_tokens[1].item() == 256
    assert PAD_TOKEN not in source.padded_tokens[: source.valid_length]
    assert padded_length_for_valid_length(7, 4, 2) == 8
    assert [(chunk.output_start, chunk.output_count) for chunk in chunks] == [(0, 2), (2, 1)]
    assert chunks[-1].tokens[-1].item() == PAD_TOKEN


def test_first_tie_winner_and_all_negative_activations_are_preserved():
    model = WholeFileGCGClassifier(
        embedding_dim=1,
        channels=1,
        receptive_field_bytes=1,
        output_stride_bytes=1,
        max_outputs_per_chunk=2,
    ).double()
    with torch.no_grad():
        model.context_encoder.embedding.weight.zero_()
        model.context_encoder.embedding.weight[1, 0] = -2.0
        model.context_encoder.embedding.weight[2, 0] = -1.0
        model.context_encoder.conv.weight.zero_()
        model.context_encoder.conv.bias.zero_()
        model.context_encoder.conv.weight[0, 0, 0] = 1.0
        model.context_encoder.conv.bias[1] = 10.0
        model.context_encoder.channel_share.weight.fill_(1.0)
        model.context_encoder.channel_share.bias.zero_()

    selection = model.find_context_winners(InMemoryByteSource.from_raw_bytes([0, 1, 1, 0]))

    assert selection.positions.tolist() == [1]
    assert selection.values.item() < 0.0
    assert selection.windows.shape == (1, 1)
    assert selection.windows.item() == 2


def test_tied_context_and_gated_winners_match_dense_parameter_gradients():
    chunked_model = _model()
    dense_model = copy.deepcopy(chunked_model)
    tokens, valid_length = encode_raw_bytes([7] * 10)

    chunked = chunked_model(tokens, torch.tensor([valid_length]), return_features=True)
    dense = dense_model.forward_dense(
        tokens, torch.tensor([valid_length]), return_features=True
    )
    _loss(chunked).backward()
    _loss(dense).backward()

    assert torch.equal(chunked["logits"], dense["logits"])
    for (chunked_name, chunked_parameter), (dense_name, dense_parameter) in zip(
        chunked_model.named_parameters(), dense_model.named_parameters()
    ):
        assert chunked_name == dense_name
        assert torch.allclose(
            chunked_parameter.grad, dense_parameter.grad, rtol=1e-10, atol=1e-10
        )


def test_source_requires_contiguous_exact_output_coordinates():
    class BrokenSource:
        def expected_output_count(self, **_kwargs):
            return 1

        def iter_output_chunks(self, **_kwargs):
            yield type(
                "Chunk",
                (),
                {"output_start": 1, "output_count": 1, "tokens": torch.tensor([1, 2, 3, 4])},
            )()

    model = _model()
    with pytest.raises(ValueError, match="contiguous"):
        model.forward_from_source(BrokenSource())


def test_forward_from_source_uses_exactly_two_fresh_chunk_passes():
    class CountingSource:
        def __init__(self):
            self.inner = InMemoryByteSource.from_raw_bytes([0, 1, 2, 3, 4, 5, 6])
            self.pass_count = 0

        def iter_output_chunks(self, **kwargs):
            self.pass_count += 1
            return self.inner.iter_output_chunks(**kwargs)

        def expected_output_count(self, **kwargs):
            return self.inner.expected_output_count(**kwargs)

    source = CountingSource()
    model = _model()

    result = model.forward_from_source(source, return_features=True)

    assert source.pass_count == 2
    assert result["logits"].shape == (1, 2)
    assert result["features"].shape == (1, 4)


def test_source_rejects_truncated_declared_coverage():
    class TruncatedSource:
        def expected_output_count(self, **_kwargs):
            return 3

        def iter_output_chunks(self, **_kwargs):
            yield type(
                "Chunk",
                (),
                {"output_start": 0, "output_count": 2, "tokens": torch.tensor([1, 2, 3, 4, 5, 6])},
            )()

    with pytest.raises(ValueError, match="declared whole file"):
        _model().forward_from_source(TruncatedSource())


def test_source_rejects_chunk_larger_than_memory_contract():
    class OversizedChunkSource:
        def expected_output_count(self, **_kwargs):
            return 3

        def iter_output_chunks(self, **_kwargs):
            yield type(
                "Chunk",
                (),
                {
                    "output_start": 0,
                    "output_count": 3,
                    "tokens": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
                },
            )()

    with pytest.raises(ValueError, match="max_outputs_per_chunk"):
        _model().forward_from_source(OversizedChunkSource())


def test_model_rejects_stride_that_would_leave_unscanned_bytes():
    with pytest.raises(ValueError, match="cannot exceed"):
        WholeFileGCGClassifier(receptive_field_bytes=4, output_stride_bytes=5)
