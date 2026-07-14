"""Deterministic, exact-once scheduling primitives for the Loop166 B1 cell."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isclose
from operator import index
from struct import pack
from typing import Any, Iterator, NamedTuple, Sequence


@dataclass(frozen=True)
class MicrobatchSlice:
    cursor_start: int
    cursor_end: int
    indices: tuple[int, ...]
    loss_weight: float

    @property
    def sequence_count(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class OptimizerGroup:
    optimizer_step_index: int
    cursor_start: int
    cursor_end: int
    microbatches: tuple[MicrobatchSlice, ...]

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(index_value for batch in self.microbatches for index_value in batch.indices)

    @property
    def sequence_count(self) -> int:
        return self.cursor_end - self.cursor_start


class MaskedContentBatch(NamedTuple):
    masked_input_ids: Any
    labels: Any
    masked_token_count: int


def _strict_index(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} cannot be boolean")
    try:
        return index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _positive_index(value: object, *, name: str) -> int:
    normalized = _strict_index(value, name=name)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def deterministic_permutation(sequence_count: int, seed: int) -> tuple[int, ...]:
    """Return a runtime-independent SHA-256 ordering of all sequence indices."""
    normalized_count = _positive_index(sequence_count, name="sequence_count")
    normalized_seed = _strict_index(seed, name="seed")
    if not 0 <= normalized_seed < 1 << 64:
        raise ValueError("seed must fit uint64")
    seed_bytes = pack("<Q", normalized_seed)
    return tuple(
        sorted(
            range(normalized_count),
            key=lambda sequence_index: (
                sha256(
                    b"axon_loop166_b1_permutation_v1\x00"
                    + seed_bytes
                    + pack("<Q", sequence_index)
                ).digest(),
                sequence_index,
            ),
        )
    )


def _normalize_permutation(
    permutation: Sequence[int],
    *,
    sequence_count: int | None = None,
) -> tuple[int, ...]:
    if not permutation:
        raise ValueError("permutation cannot be empty")
    expected_count = len(permutation) if sequence_count is None else _positive_index(
        sequence_count,
        name="sequence_count",
    )
    if len(permutation) != expected_count:
        raise ValueError("permutation length does not equal sequence_count")

    normalized: list[int] = []
    seen = bytearray(expected_count)
    for position, raw_index in enumerate(permutation):
        sequence_index = _strict_index(raw_index, name=f"permutation[{position}]")
        if not 0 <= sequence_index < expected_count:
            raise ValueError(f"permutation index out of range: {sequence_index}")
        if seen[sequence_index]:
            raise ValueError(f"duplicate permutation index: {sequence_index}")
        seen[sequence_index] = 1
        normalized.append(sequence_index)
    if not all(seen):
        raise ValueError("permutation does not cover every sequence exactly once")
    return tuple(normalized)


def validate_permutation(
    permutation: Sequence[int],
    sequence_count: int | None = None,
) -> None:
    """Fail closed unless the permutation covers exactly ``range(sequence_count)``."""
    _normalize_permutation(permutation, sequence_count=sequence_count)


def permutation_commitment_sha256(permutation: Sequence[int]) -> str:
    normalized = _normalize_permutation(permutation)
    digest = sha256()
    digest.update(b"axon_loop166_b1_permutation_commitment_v1\x00")
    digest.update(pack("<Q", len(normalized)))
    for sequence_index in normalized:
        digest.update(pack("<Q", sequence_index))
    return digest.hexdigest()


def _normalize_schedule_parameters(
    microbatch_size: int,
    gradient_accumulation_steps: int,
) -> tuple[int, int, int]:
    normalized_microbatch = _positive_index(microbatch_size, name="microbatch_size")
    normalized_accumulation = _positive_index(
        gradient_accumulation_steps,
        name="gradient_accumulation_steps",
    )
    return (
        normalized_microbatch,
        normalized_accumulation,
        normalized_microbatch * normalized_accumulation,
    )


def _group_from_validated_permutation(
    permutation: tuple[int, ...],
    cursor: int,
    *,
    microbatch_size: int,
    gradient_accumulation_steps: int,
) -> OptimizerGroup | None:
    microbatch, _accumulation, group_capacity = _normalize_schedule_parameters(
        microbatch_size,
        gradient_accumulation_steps,
    )
    normalized_cursor = _strict_index(cursor, name="cursor")
    if not 0 <= normalized_cursor <= len(permutation):
        raise ValueError("cursor is outside permutation boundaries")
    if normalized_cursor != len(permutation) and normalized_cursor % group_capacity != 0:
        raise ValueError("cursor must point to an optimizer-group boundary")
    if normalized_cursor == len(permutation):
        return None

    group_end = min(normalized_cursor + group_capacity, len(permutation))
    group_size = group_end - normalized_cursor
    microbatches: list[MicrobatchSlice] = []
    for batch_start in range(normalized_cursor, group_end, microbatch):
        batch_end = min(batch_start + microbatch, group_end)
        batch_indices = permutation[batch_start:batch_end]
        microbatches.append(
            MicrobatchSlice(
                cursor_start=batch_start,
                cursor_end=batch_end,
                indices=batch_indices,
                loss_weight=len(batch_indices) / group_size,
            )
        )
    return OptimizerGroup(
        optimizer_step_index=normalized_cursor // group_capacity,
        cursor_start=normalized_cursor,
        cursor_end=group_end,
        microbatches=tuple(microbatches),
    )


def optimizer_group_from_cursor(
    permutation: Sequence[int],
    cursor: int,
    *,
    microbatch_size: int,
    gradient_accumulation_steps: int,
) -> OptimizerGroup | None:
    """Build one optimizer group; the final group is shorter without repetition."""
    normalized_permutation = _normalize_permutation(permutation)
    return _group_from_validated_permutation(
        normalized_permutation,
        cursor,
        microbatch_size=microbatch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


def iter_optimizer_groups(
    permutation: Sequence[int],
    *,
    start_cursor: int = 0,
    microbatch_size: int,
    gradient_accumulation_steps: int,
) -> Iterator[OptimizerGroup]:
    """Yield contiguous optimizer groups from a validated resume cursor."""
    normalized_permutation = _normalize_permutation(permutation)
    cursor = _strict_index(start_cursor, name="start_cursor")
    while True:
        group = _group_from_validated_permutation(
            normalized_permutation,
            cursor,
            microbatch_size=microbatch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        if group is None:
            return
        yield group
        cursor = group.cursor_end


def validate_exact_once_schedule(
    permutation: Sequence[int],
    groups: Sequence[OptimizerGroup],
    sequence_count: int,
) -> None:
    """Validate optimizer and microbatch boundaries plus exact-once coverage."""
    normalized_permutation = _normalize_permutation(
        permutation,
        sequence_count=sequence_count,
    )
    if not groups:
        raise ValueError("groups cannot be empty")

    expected_cursor = 0
    visited: list[int] = []
    for expected_step, group in enumerate(groups):
        if not isinstance(group, OptimizerGroup):
            raise ValueError("groups must contain OptimizerGroup values")
        if group.optimizer_step_index != expected_step:
            raise ValueError("optimizer step indices are not contiguous")
        if group.cursor_start != expected_cursor or group.cursor_end <= group.cursor_start:
            raise ValueError("optimizer group cursor boundaries are not contiguous")
        if group.cursor_end > len(normalized_permutation):
            raise ValueError("optimizer group exceeds permutation boundary")
        if not group.microbatches:
            raise ValueError("optimizer group cannot have zero microbatches")

        micro_cursor = group.cursor_start
        weight_sum = 0.0
        for microbatch in group.microbatches:
            if microbatch.cursor_start != micro_cursor:
                raise ValueError("microbatch cursor boundaries are not contiguous")
            if microbatch.cursor_end <= microbatch.cursor_start:
                raise ValueError("microbatch cannot be empty")
            expected_indices = normalized_permutation[
                microbatch.cursor_start : microbatch.cursor_end
            ]
            if microbatch.indices != expected_indices:
                raise ValueError("microbatch indices diverge from permutation")
            expected_weight = len(expected_indices) / group.sequence_count
            if not isclose(microbatch.loss_weight, expected_weight, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError("microbatch loss weight is not sequence-count normalized")
            weight_sum += microbatch.loss_weight
            visited.extend(microbatch.indices)
            micro_cursor = microbatch.cursor_end
        if micro_cursor != group.cursor_end:
            raise ValueError("microbatch boundaries do not close optimizer group")
        if not isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("optimizer group loss weights do not sum to one")
        expected_cursor = group.cursor_end

    if expected_cursor != len(normalized_permutation):
        raise ValueError("schedule does not consume the full permutation")
    if tuple(visited) != normalized_permutation:
        raise ValueError("schedule repeats, skips, or reorders sequences")


def _validate_mask_id(value: object, *, name: str) -> int:
    normalized = _strict_index(value, name=name)
    if normalized < 0:
        raise ValueError(f"{name} cannot be negative")
    return normalized


def mask_content_batch(
    torch_module: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    cls_id: int,
    sep_id: int,
    mask_token_id: int,
    mask_ratio: float,
    generator: Any,
) -> MaskedContentBatch:
    """Mask only content positions strictly between validated CLS and SEP tokens."""
    if generator is None:
        raise ValueError("an explicit generator is required")
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must be rank two")
    if input_ids.shape != attention_mask.shape or input_ids.shape[0] == 0:
        raise ValueError("input_ids and attention_mask must have the same non-empty shape")
    if input_ids.shape[1] < 3:
        raise ValueError("framed sequences must contain CLS, content, and SEP")
    if input_ids.dtype != torch_module.long or attention_mask.dtype != torch_module.bool:
        raise ValueError("input_ids must be long and attention_mask must be bool")
    if input_ids.device.type != "cpu" or attention_mask.device.type != "cpu":
        raise ValueError("B1 masking must run on CPU before device transfer")
    if isinstance(mask_ratio, bool) or not isinstance(mask_ratio, (int, float)):
        raise ValueError("mask_ratio must be numeric")
    normalized_ratio = float(mask_ratio)
    if not 0.0 < normalized_ratio <= 1.0:
        raise ValueError("mask_ratio must be in (0, 1]")

    normalized_cls_id = _validate_mask_id(cls_id, name="cls_id")
    normalized_sep_id = _validate_mask_id(sep_id, name="sep_id")
    normalized_mask_id = _validate_mask_id(mask_token_id, name="mask_token_id")
    if len({normalized_cls_id, normalized_sep_id, normalized_mask_id}) != 3:
        raise ValueError("CLS, SEP, and MASK ids must be distinct")

    masked_input_ids = input_ids.clone()
    labels = torch_module.full_like(input_ids, -100)
    total_masked = 0
    for row_index in range(input_ids.shape[0]):
        valid_count = int(attention_mask[row_index].sum().item())
        if valid_count < 3:
            raise ValueError("each row must contain at least one content token")
        if not bool(attention_mask[row_index, :valid_count].all().item()) or bool(
            attention_mask[row_index, valid_count:].any().item()
        ):
            raise ValueError("attention_mask must describe one contiguous valid prefix")
        if int(input_ids[row_index, 0].item()) != normalized_cls_id:
            raise ValueError("first valid token must be CLS")
        if int(input_ids[row_index, valid_count - 1].item()) != normalized_sep_id:
            raise ValueError("last valid token must be SEP")
        content = input_ids[row_index, 1 : valid_count - 1]
        if bool(
            (
                (content == normalized_cls_id)
                | (content == normalized_sep_id)
                | (content == normalized_mask_id)
            )
            .any()
            .item()
        ):
            raise ValueError("unmasked content cannot contain CLS, SEP, or MASK")

        content_count = valid_count - 2
        masked_count = max(1, int(content_count * normalized_ratio + 0.5))
        selected = torch_module.randperm(content_count, generator=generator)[:masked_count] + 1
        labels[row_index, selected] = input_ids[row_index, selected]
        masked_input_ids[row_index, selected] = normalized_mask_id
        total_masked += masked_count

    return MaskedContentBatch(masked_input_ids, labels, total_masked)
