"""In-memory byte-bijective BPE helpers for the Loop166 outer-fit corpus."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from operator import index
from typing import Iterable, Sequence

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = ("[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]")


@dataclass(frozen=True)
class LosslessTokenChunk:
    token_ids: tuple[int, ...]
    original_byte_length: int


@lru_cache(maxsize=1)
def _byte_unicode_pairs() -> tuple[tuple[int, str], ...]:
    """Return a reversible private-use alphabet that cannot collide with specials."""
    return tuple((byte_value, chr(0xE000 + byte_value)) for byte_value in range(256))


def bytes_to_unicode() -> dict[int, str]:
    """Return a fresh byte-to-Unicode bijection for all 256 byte values."""
    return dict(_byte_unicode_pairs())


def unicode_to_bytes() -> dict[str, int]:
    """Return the inverse of :func:`bytes_to_unicode`."""
    return {character: byte_value for byte_value, character in _byte_unicode_pairs()}


def encode_byte_string(data: bytes) -> str:
    """Map raw bytes to the lossless Unicode alphabet consumed by BPE."""
    alphabet = bytes_to_unicode()
    return "".join(alphabet[byte_value] for byte_value in data)


def decode_byte_string(encoded: str) -> bytes:
    """Invert :func:`encode_byte_string`, rejecting non-alphabet characters."""
    alphabet = unicode_to_bytes()
    try:
        return bytes(alphabet[character] for character in encoded)
    except KeyError as error:
        raise ValueError(f"Character is outside the byte alphabet: {error.args[0]!r}") from error


def select_even_window_indices(total_windows: int, maximum_windows: int) -> list[int]:
    """Select deterministic, endpoint-preserving indices across non-overlapping windows."""
    if total_windows < 0:
        raise ValueError("total_windows cannot be negative")
    if maximum_windows <= 0:
        raise ValueError("maximum_windows must be positive")
    if total_windows <= maximum_windows:
        return list(range(total_windows))
    if maximum_windows == 1:
        return [total_windows // 2]

    denominator = maximum_windows - 1
    span = total_windows - 1
    return [
        (selection_index * span + denominator // 2) // denominator
        for selection_index in range(maximum_windows)
    ]


def select_even_windows(
    data: bytes,
    *,
    window_bytes: int,
    max_windows: int,
) -> list[bytes]:
    """Split into non-overlapping windows and retain at most evenly spaced samples."""
    if window_bytes <= 0:
        raise ValueError("window_bytes must be positive")
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")
    windows = [data[start : start + window_bytes] for start in range(0, len(data), window_bytes)]
    indices = select_even_window_indices(len(windows), max_windows)
    return [windows[index] for index in indices]


def _validate_special_tokens(special_tokens: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(token) for token in special_tokens)
    if not normalized:
        raise ValueError("special_tokens cannot be empty")
    if any(not token for token in normalized):
        raise ValueError("special_tokens cannot contain empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("special_tokens must be unique")
    if "[UNK]" not in normalized:
        raise ValueError("special_tokens must include [UNK]")
    return normalized


def train_byte_bpe_tokenizer(
    windows: Sequence[bytes] | Iterable[bytes],
    *,
    vocab_size: int,
    special_tokens: Sequence[str] = SPECIAL_TOKENS,
) -> Tokenizer:
    """Fit a tokenizer only from explicitly supplied outer-fit byte windows."""
    normalized_special_tokens = _validate_special_tokens(special_tokens)
    minimum_vocabulary = 256 + len(normalized_special_tokens)
    if vocab_size < minimum_vocabulary:
        raise ValueError(
            f"vocab_size must cover 256 bytes and specials: {vocab_size} < {minimum_vocabulary}"
        )

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        show_progress=False,
        special_tokens=list(normalized_special_tokens),
        initial_alphabet=list(bytes_to_unicode().values()),
    )
    tokenizer.train_from_iterator(
        (encode_byte_string(bytes(window)) for window in windows),
        trainer=trainer,
    )
    actual_vocabulary = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocabulary != vocab_size:
        raise ValueError(
            "Outer-fit corpus cannot support the requested BPE vocabulary: "
            f"{actual_vocabulary} != {vocab_size}"
        )
    return tokenizer


def encode_bytes(tokenizer: Tokenizer, data: bytes) -> list[int]:
    """Encode raw bytes without injecting framing or identity metadata."""
    return [token.id for token in tokenizer.model.tokenize(encode_byte_string(data))]


def _special_token_ids(tokenizer: Tokenizer) -> set[int]:
    special_ids = {
        token_id
        for token in SPECIAL_TOKENS
        if (token_id := tokenizer.token_to_id(token)) is not None
    }
    for token_id, added_token in tokenizer.get_added_tokens_decoder().items():
        if added_token.special:
            special_ids.add(int(token_id))
    return special_ids


def _decode_content_token_ids(tokenizer: Tokenizer, token_ids: Sequence[int]) -> bytes:
    if not token_ids:
        raise ValueError("Content token ids cannot be empty")
    special_ids = _special_token_ids(tokenizer)
    pieces: list[str] = []
    for raw_token_id in token_ids:
        if isinstance(raw_token_id, bool):
            raise ValueError("Boolean values are not valid content token ids")
        try:
            token_id = index(raw_token_id)
        except TypeError as error:
            raise ValueError(f"Invalid content token id: {raw_token_id!r}") from error
        if token_id in special_ids:
            raise ValueError(f"Special token id is forbidden in byte content: {token_id}")
        token = tokenizer.id_to_token(token_id)
        if token is None:
            raise ValueError(f"Unknown tokenizer id: {token_id}")
        pieces.append(token)
    decoded = decode_byte_string("".join(pieces))
    if not decoded:
        raise ValueError("Content token ids decoded to no original bytes")
    return decoded


def token_ids_original_byte_length(tokenizer: Tokenizer, token_ids: Sequence[int]) -> int:
    """Return the exact original-byte length represented by content-only token ids."""
    return len(_decode_content_token_ids(tokenizer, token_ids))


def chunk_token_ids_losslessly(
    tokenizer: Tokenizer,
    data: bytes,
    *,
    max_content_tokens: int,
) -> list[LosslessTokenChunk]:
    """Split encoded bytes without dropping tokens or original bytes."""
    if max_content_tokens <= 0:
        raise ValueError("max_content_tokens must be positive")
    if not data:
        raise ValueError("data cannot be empty")

    token_ids = encode_bytes(tokenizer, data)
    if not token_ids:
        raise ValueError("Non-empty data produced no content token ids")
    chunks: list[LosslessTokenChunk] = []
    recovered_parts: list[bytes] = []
    for start in range(0, len(token_ids), max_content_tokens):
        chunk_ids = tuple(token_ids[start : start + max_content_tokens])
        recovered = _decode_content_token_ids(tokenizer, chunk_ids)
        chunks.append(LosslessTokenChunk(chunk_ids, len(recovered)))
        recovered_parts.append(recovered)

    # 每段独立可逆，且拼接后必须与原窗口逐字节一致，避免高熵尾部被静默丢弃。
    recovered_data = b"".join(recovered_parts)
    if recovered_data != data:
        raise ValueError("Token chunks do not reconstruct the original bytes exactly")
    if sum(chunk.original_byte_length for chunk in chunks) != len(data):
        raise ValueError("Token chunk original-byte accounting is not conserved")
    return chunks


def decode_token_ids(
    tokenizer: Tokenizer,
    token_ids: Sequence[int],
    *,
    skip_special_tokens: bool = True,
) -> bytes:
    """Decode BPE ids directly through token strings and the byte bijection."""
    special_ids = _special_token_ids(tokenizer)
    pieces: list[str] = []
    for token_id in token_ids:
        normalized_id = int(token_id)
        if skip_special_tokens and normalized_id in special_ids:
            continue
        token = tokenizer.id_to_token(normalized_id)
        if token is None:
            raise ValueError(f"Unknown tokenizer id: {normalized_id}")
        pieces.append(token)
    return decode_byte_string("".join(pieces))


def tokenizer_vocab_size(tokenizer: Tokenizer) -> int:
    """Return the complete vocabulary size, including special tokens."""
    return int(tokenizer.get_vocab_size(with_added_tokens=True))
