from __future__ import annotations

import torch
import torch.nn.functional as functional


def _output_partitions(output_count: int, max_outputs_per_chunk: int) -> list[tuple[int, int]]:
    if output_count < 1 or max_outputs_per_chunk < 1:
        raise ValueError("Output partitions require positive lengths")
    return [
        (start, min(start + max_outputs_per_chunk, output_count))
        for start in range(0, output_count, max_outputs_per_chunk)
    ]


def _dense_global_max(
    byte_features: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return functional.conv1d(byte_features, kernel, bias).max(dim=-1).values


def _exact_chunked_global_max(
    byte_features: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor,
    *,
    max_outputs_per_chunk: int,
) -> torch.Tensor:
    # 统一使用首 winner 语义，避免 amax/maximum 在 tie 时分摊出不同梯度。
    values, _positions = _exact_chunked_global_max_with_positions(
        byte_features,
        kernel,
        bias,
        max_outputs_per_chunk=max_outputs_per_chunk,
    )
    return values


def _exact_chunked_global_max_with_positions(
    byte_features: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor,
    *,
    max_outputs_per_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    receptive_field = kernel.shape[-1]
    output_count = byte_features.shape[-1] - receptive_field + 1
    best_values: torch.Tensor | None = None
    best_positions: torch.Tensor | None = None
    for output_start, output_end in _output_partitions(output_count, max_outputs_per_chunk):
        input_region = byte_features[..., output_start : output_end + receptive_field - 1]
        region_values, region_positions = functional.conv1d(input_region, kernel, bias).max(dim=-1)
        region_positions = region_positions + output_start
        if best_values is None:
            best_values = region_values
            best_positions = region_positions
            continue
        replace = region_values > best_values
        best_values = torch.where(replace, region_values, best_values)
        best_positions = torch.where(replace, region_positions, best_positions)
    if best_values is None or best_positions is None:
        raise AssertionError("At least one convolution output is required")
    return best_values, best_positions


def _dense_two_pass_gcg_toy(
    byte_features: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    scores = functional.conv1d(byte_features, kernel, bias)
    gate = torch.sigmoid(scores.max(dim=-1).values).unsqueeze(-1)
    return (scores * gate).max(dim=-1).values


def _chunked_two_pass_gcg_toy(
    byte_features: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor,
    *,
    max_outputs_per_chunk: int,
) -> torch.Tensor:
    receptive_field = kernel.shape[-1]
    output_count = byte_features.shape[-1] - receptive_field + 1
    gate = torch.sigmoid(
        _exact_chunked_global_max(
            byte_features, kernel, bias, max_outputs_per_chunk=max_outputs_per_chunk
        )
    ).unsqueeze(-1)
    pooled: torch.Tensor | None = None
    for output_start, output_end in _output_partitions(output_count, max_outputs_per_chunk):
        input_region = byte_features[..., output_start : output_end + receptive_field - 1]
        region_max = (functional.conv1d(input_region, kernel, bias) * gate).max(dim=-1).values
        pooled = region_max if pooled is None else torch.where(region_max > pooled, region_max, pooled)
    if pooled is None:
        raise AssertionError("At least one convolution output is required")
    return pooled


def _encode_bytes_with_reserved_pad(raw_bytes: list[int], padded_length: int) -> tuple[torch.Tensor, int]:
    if padded_length < len(raw_bytes):
        raise ValueError("Padding capacity cannot truncate a supported file")
    tokens = torch.zeros(padded_length, dtype=torch.long)
    if raw_bytes:
        tokens[: len(raw_bytes)] = torch.tensor(raw_bytes, dtype=torch.long) + 1
    return tokens, len(raw_bytes)


def test_dense_reference_equivalence():
    byte_features = torch.tensor([[[2.0, -1.0, 3.0, 0.0, 4.0, -2.0, 5.0]]])
    kernel = torch.tensor([[[1.0, -2.0, 0.5]]])
    bias = torch.tensor([-0.75])

    dense = _dense_global_max(byte_features, kernel, bias)
    chunked = _exact_chunked_global_max(
        byte_features, kernel, bias, max_outputs_per_chunk=2
    )

    assert torch.equal(dense, chunked)


def test_gradient_reference_equivalence():
    generator = torch.Generator().manual_seed(164)
    dense_input = torch.randn((2, 2, 29), generator=generator, dtype=torch.float64).requires_grad_()
    dense_kernel = torch.randn((3, 2, 5), generator=generator, dtype=torch.float64).requires_grad_()
    dense_bias = torch.randn((3,), generator=generator, dtype=torch.float64).requires_grad_()
    chunked_input = dense_input.detach().clone().requires_grad_()
    chunked_kernel = dense_kernel.detach().clone().requires_grad_()
    chunked_bias = dense_bias.detach().clone().requires_grad_()

    dense_loss = _dense_global_max(dense_input, dense_kernel, dense_bias).square().sum()
    chunked_loss = _exact_chunked_global_max(
        chunked_input, chunked_kernel, chunked_bias, max_outputs_per_chunk=4
    ).square().sum()
    dense_loss.backward()
    chunked_loss.backward()

    assert torch.allclose(dense_loss, chunked_loss, rtol=1e-10, atol=1e-10)
    assert torch.allclose(dense_input.grad, chunked_input.grad, rtol=1e-10, atol=1e-10)
    assert torch.allclose(dense_kernel.grad, chunked_kernel.grad, rtol=1e-10, atol=1e-10)
    assert torch.allclose(dense_bias.grad, chunked_bias.grad, rtol=1e-10, atol=1e-10)


def test_first_winner_tie_gradient_equivalence():
    dense_input = torch.ones((1, 1, 5), dtype=torch.float64, requires_grad=True)
    dense_kernel = torch.ones((1, 1, 1), dtype=torch.float64, requires_grad=True)
    dense_bias = torch.zeros((1,), dtype=torch.float64, requires_grad=True)
    chunked_input = dense_input.detach().clone().requires_grad_()
    chunked_kernel = dense_kernel.detach().clone().requires_grad_()
    chunked_bias = dense_bias.detach().clone().requires_grad_()

    dense = _dense_global_max(dense_input, dense_kernel, dense_bias)
    chunked = _exact_chunked_global_max(
        chunked_input, chunked_kernel, chunked_bias, max_outputs_per_chunk=1
    )
    dense.sum().backward()
    chunked.sum().backward()

    assert torch.equal(dense, chunked)
    assert torch.equal(dense_input.grad, chunked_input.grad)
    assert torch.equal(dense_kernel.grad, chunked_kernel.grad)
    assert torch.equal(dense_bias.grad, chunked_bias.grad)


def test_chunk_boundary_equivalence():
    byte_features = torch.tensor([[[2.0, -1.0, 3.0, 0.0, 4.0, -2.0, 5.0, 1.0, -3.0]]])
    kernel = torch.tensor([[[1.0, -2.0, 0.5]]])
    bias = torch.tensor([-0.75])

    dense = _dense_global_max(byte_features, kernel, bias)
    chunked = _exact_chunked_global_max(
        byte_features, kernel, bias, max_outputs_per_chunk=2
    )

    assert torch.equal(dense, chunked)


def test_winner_global_position_equivalence():
    byte_features = torch.tensor([[[1.0, 4.0, 4.0, 2.0, -1.0]]])
    kernel = torch.tensor([[[1.0]]])
    bias = torch.tensor([0.0])
    dense_values, dense_positions = functional.conv1d(byte_features, kernel, bias).max(dim=-1)
    chunked_values, chunked_positions = _exact_chunked_global_max_with_positions(
        byte_features, kernel, bias, max_outputs_per_chunk=2
    )

    assert torch.equal(dense_values, chunked_values)
    assert torch.equal(dense_positions, chunked_positions)
    assert dense_positions.item() == 1


def test_tail_coverage():
    output_count = 8
    partitions = _output_partitions(output_count, max_outputs_per_chunk=3)
    covered_positions = [position for start, end in partitions for position in range(start, end)]
    byte_features = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]]])
    kernel = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    bias = torch.tensor([0.0])
    dense_values, dense_positions = functional.conv1d(byte_features, kernel, bias).max(dim=-1)
    chunked_values, chunked_positions = _exact_chunked_global_max_with_positions(
        byte_features, kernel, bias, max_outputs_per_chunk=3
    )

    assert partitions[-1] == (6, 8)
    assert covered_positions == list(range(output_count))
    assert torch.equal(dense_values, chunked_values)
    assert torch.equal(dense_positions, chunked_positions)
    assert dense_positions.item() == 7


def test_all_negative_activation_winner():
    byte_features = torch.tensor([[[2.0, 3.0, 4.0, 5.0, 6.0]]])
    kernel = torch.tensor([[[-1.0, -1.0]]])
    bias = torch.tensor([0.0])

    dense = _dense_global_max(byte_features, kernel, bias)
    chunked = _exact_chunked_global_max(
        byte_features, kernel, bias, max_outputs_per_chunk=2
    )

    assert dense.item() == -5.0
    assert torch.equal(dense, chunked)


def test_zero_byte_and_eof_semantics():
    tokens, valid_length = _encode_bytes_with_reserved_pad([0, 255, 3], padded_length=5)

    assert valid_length == 3
    assert tokens.tolist() == [1, 256, 4, 0, 0]
    assert tokens[:valid_length].ne(0).all()
    assert tokens[valid_length:].eq(0).all()


def test_two_pass_context_equivalence():
    generator = torch.Generator().manual_seed(264)
    dense_input = torch.randn((2, 2, 23), generator=generator, dtype=torch.float64).requires_grad_()
    dense_kernel = torch.randn((3, 2, 4), generator=generator, dtype=torch.float64).requires_grad_()
    dense_bias = torch.randn((3,), generator=generator, dtype=torch.float64).requires_grad_()
    chunked_input = dense_input.detach().clone().requires_grad_()
    chunked_kernel = dense_kernel.detach().clone().requires_grad_()
    chunked_bias = dense_bias.detach().clone().requires_grad_()

    dense_loss = _dense_two_pass_gcg_toy(dense_input, dense_kernel, dense_bias).square().sum()
    chunked_loss = _chunked_two_pass_gcg_toy(
        chunked_input, chunked_kernel, chunked_bias, max_outputs_per_chunk=3
    ).square().sum()
    dense_loss.backward()
    chunked_loss.backward()

    assert torch.allclose(dense_loss, chunked_loss, rtol=1e-10, atol=1e-10)
    assert torch.allclose(dense_input.grad, chunked_input.grad, rtol=1e-10, atol=1e-10)
    assert torch.allclose(dense_kernel.grad, chunked_kernel.grad, rtol=1e-10, atol=1e-10)
    assert torch.allclose(dense_bias.grad, chunked_bias.grad, rtol=1e-10, atol=1e-10)


def test_deterministic_repeated_run():
    byte_features = torch.tensor([[[1.0, 4.0, -3.0, 2.0, 5.0, -1.0, 6.0]]])
    kernel = torch.tensor([[[0.25, -1.0, 0.5]]])
    bias = torch.tensor([0.125])

    first = _exact_chunked_global_max(byte_features, kernel, bias, max_outputs_per_chunk=3)
    second = _exact_chunked_global_max(byte_features, kernel, bias, max_outputs_per_chunk=3)

    assert torch.equal(first, second)


def test_missingness_denominator():
    reason_counts = {
        "timeout": 1,
        "unsupported": 1,
        "read_failure": 0,
        "parse_failure": 1,
        "oversize": 0,
    }
    successful_rows = 7
    denominator_rows = 10

    assert set(reason_counts) == {
        "timeout",
        "unsupported",
        "read_failure",
        "parse_failure",
        "oversize",
    }
    assert successful_rows + sum(reason_counts.values()) == denominator_rows


def test_noncontiguous_winner_independence():
    byte_features = torch.tensor([[[10.0, -10.0, 0.0, 0.0, 20.0, -20.0]]])
    kernel = torch.tensor([[[-1.0, 1.0]]])
    bias = torch.tensor([0.0])
    dense = _dense_global_max(byte_features, kernel, bias)
    concatenated_winners = torch.cat((byte_features[..., :2], byte_features[..., 4:]), dim=-1)
    unsafe = _dense_global_max(concatenated_winners, kernel, bias)

    assert dense.item() == 20.0
    assert unsafe.item() == 30.0
