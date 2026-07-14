"""Compact in-memory storage for lossless Loop166 BPE sequences."""

from __future__ import annotations

import sys
from array import array
from hashlib import sha256
from operator import index
from struct import pack
from typing import Any, NamedTuple, Sequence

from .byte_bpe import LosslessTokenChunk

UINT16_MAX = (1 << 16) - 1
MAX_CONTENT_TOKENS = 510
MAX_ORIGINAL_WINDOW_BYTES = 512


class FramedBatch(NamedTuple):
    input_ids: Any
    attention_mask: Any
    original_byte_lengths: Any


def _strict_index(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} cannot be boolean")
    try:
        return index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _little_endian_bytes(values: array) -> bytes:
    if sys.byteorder == "little" or values.itemsize == 1:
        return values.tobytes()
    normalized = array(values.typecode, values)
    normalized.byteswap()
    return normalized.tobytes()


class CompactSequenceCorpus:
    """Store variable-length token chunks without persistent padded Python rows."""

    __slots__ = (
        "_flat_token_ids",
        "_lengths",
        "_max_content_tokens",
        "_offsets",
        "_original_byte_lengths",
        "_vocab_size",
    )

    def __init__(self, vocab_size: int, max_content_tokens: int):
        normalized_vocab_size = _strict_index(vocab_size, name="vocab_size")
        normalized_max_tokens = _strict_index(
            max_content_tokens,
            name="max_content_tokens",
        )
        if not 0 < normalized_vocab_size <= UINT16_MAX + 1:
            raise ValueError("vocab_size must be in [1, 65536]")
        if not 0 < normalized_max_tokens <= MAX_CONTENT_TOKENS:
            raise ValueError("max_content_tokens must be in [1, 510]")

        self._vocab_size = normalized_vocab_size
        self._max_content_tokens = normalized_max_tokens
        self._flat_token_ids = array("H")
        self._offsets = array("Q", [0])
        self._lengths = array("H")
        self._original_byte_lengths = array("H")

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def max_content_tokens(self) -> int:
        return self._max_content_tokens

    @property
    def total_tokens(self) -> int:
        return len(self._flat_token_ids)

    @property
    def total_original_bytes(self) -> int:
        return sum(self._original_byte_lengths)

    @property
    def estimated_storage_bytes(self) -> int:
        arrays = (
            self._flat_token_ids,
            self._offsets,
            self._lengths,
            self._original_byte_lengths,
        )
        return sum(len(values) * values.itemsize for values in arrays)

    def __len__(self) -> int:
        return len(self._lengths)

    def append(self, chunk: LosslessTokenChunk) -> None:
        if not isinstance(chunk, LosslessTokenChunk):
            raise ValueError("chunk must be a LosslessTokenChunk")
        if not chunk.token_ids:
            raise ValueError("chunk token_ids cannot be empty")
        if len(chunk.token_ids) > self._max_content_tokens:
            raise ValueError("chunk exceeds max_content_tokens")

        original_byte_length = _strict_index(
            chunk.original_byte_length,
            name="original_byte_length",
        )
        if not 0 < original_byte_length <= MAX_ORIGINAL_WINDOW_BYTES:
            raise ValueError("original_byte_length must be in [1, 512]")

        normalized_ids: list[int] = []
        for position, raw_token_id in enumerate(chunk.token_ids):
            token_id = _strict_index(raw_token_id, name=f"token_ids[{position}]")
            if not 0 <= token_id < self._vocab_size:
                raise ValueError(f"token id is outside vocabulary: {token_id}")
            normalized_ids.append(token_id)

        next_offset = len(self._flat_token_ids) + len(normalized_ids)
        if next_offset >= 1 << 64:
            raise OverflowError("compact corpus token offset exceeds uint64")

        # 先完成全部校验再更新四个数组，避免失败 append 留下半行状态。
        self._flat_token_ids.extend(normalized_ids)
        self._offsets.append(next_offset)
        self._lengths.append(len(normalized_ids))
        self._original_byte_lengths.append(original_byte_length)

    def _normalize_row_index(self, row_index: int) -> int:
        normalized = _strict_index(row_index, name="row_index")
        if not 0 <= normalized < len(self):
            raise IndexError(f"compact corpus row index out of range: {normalized}")
        return normalized

    def get(self, row_index: int) -> LosslessTokenChunk:
        normalized = self._normalize_row_index(row_index)
        start = self._offsets[normalized]
        end = self._offsets[normalized + 1]
        if end - start != self._lengths[normalized]:
            raise RuntimeError("compact corpus offset and length accounting diverged")
        return LosslessTokenChunk(
            tuple(self._flat_token_ids[start:end]),
            self._original_byte_lengths[normalized],
        )

    def __getitem__(self, row_index: int) -> LosslessTokenChunk:
        return self.get(row_index)

    def commitment_sha256(self) -> str:
        digest = sha256()
        digest.update(b"axon_loop166_compact_sequence_corpus_v1\x00")
        digest.update(pack("<QQQ", self._vocab_size, self._max_content_tokens, len(self)))
        for name, values in (
            (b"flat_token_ids", self._flat_token_ids),
            (b"offsets", self._offsets),
            (b"lengths", self._lengths),
            (b"original_byte_lengths", self._original_byte_lengths),
        ):
            digest.update(pack("<Q", len(name)))
            digest.update(name)
            digest.update(pack("<QQ", len(values), values.itemsize))
            digest.update(_little_endian_bytes(values))
        return digest.hexdigest()


def _validate_framing_id(corpus: CompactSequenceCorpus, value: object, *, name: str) -> int:
    token_id = _strict_index(value, name=name)
    if not 0 <= token_id < corpus.vocab_size:
        raise ValueError(f"{name} is outside corpus vocabulary")
    return token_id


def materialize_framed_batch(
    corpus: CompactSequenceCorpus,
    indices: Sequence[int],
    *,
    pad_id: int,
    cls_id: int,
    sep_id: int,
    sequence_tokens: int,
    torch_module: Any,
) -> FramedBatch:
    """Materialize padding only for the current batch of compact corpus rows."""
    if not isinstance(corpus, CompactSequenceCorpus):
        raise ValueError("corpus must be a CompactSequenceCorpus")
    if not indices:
        raise ValueError("indices cannot be empty")
    normalized_sequence_tokens = _strict_index(sequence_tokens, name="sequence_tokens")
    if normalized_sequence_tokens < 3:
        raise ValueError("sequence_tokens must fit CLS, content, and SEP")
    if corpus.max_content_tokens > normalized_sequence_tokens - 2:
        raise ValueError("sequence_tokens cannot fit configured content plus framing")

    normalized_pad_id = _validate_framing_id(corpus, pad_id, name="pad_id")
    normalized_cls_id = _validate_framing_id(corpus, cls_id, name="cls_id")
    normalized_sep_id = _validate_framing_id(corpus, sep_id, name="sep_id")
    framing_ids = {normalized_pad_id, normalized_cls_id, normalized_sep_id}
    if len(framing_ids) != 3:
        raise ValueError("PAD, CLS, and SEP ids must be distinct")

    chunks = [corpus.get(row_index) for row_index in indices]
    input_ids = torch_module.full(
        (len(chunks), normalized_sequence_tokens),
        normalized_pad_id,
        dtype=torch_module.long,
    )
    attention_mask = torch_module.zeros_like(input_ids, dtype=torch_module.bool)
    original_byte_lengths = torch_module.empty(len(chunks), dtype=torch_module.long)

    for batch_row, chunk in enumerate(chunks):
        if any(token_id in framing_ids for token_id in chunk.token_ids):
            raise ValueError("content chunk contains a framing special token id")
        valid_tokens = len(chunk.token_ids) + 2
        if valid_tokens > normalized_sequence_tokens:
            raise ValueError("content chunk exceeds framed sequence capacity")
        input_ids[batch_row, 0] = normalized_cls_id
        input_ids[batch_row, 1 : valid_tokens - 1] = torch_module.tensor(
            chunk.token_ids,
            dtype=torch_module.long,
        )
        input_ids[batch_row, valid_tokens - 1] = normalized_sep_id
        attention_mask[batch_row, :valid_tokens] = True
        original_byte_lengths[batch_row] = chunk.original_byte_length

    return FramedBatch(input_ids, attention_mask, original_byte_lengths)
