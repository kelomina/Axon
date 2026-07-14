"""Bounded same-stream byte reader for the future Loop167 raw worker."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import BinaryIO

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class StreamReadResult:
    """The only handoff from a raw stream to RawFeatureContext."""

    result: str
    bytez: bytes | None
    declared_size: int
    bytes_read: int
    observed_sha256: str | None

    @property
    def available(self) -> bool:
        return self.result == "available" and self.bytez is not None


def _validate_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def read_verified_bytes(
    stream: BinaryIO,
    *,
    declared_size: int,
    expected_sha256: str,
    maximum_source_file_bytes: int,
    chunk_bytes: int = 1024 * 1024,
) -> StreamReadResult:
    """Read exactly one bounded stream while computing its SHA-256 commitment."""

    _validate_nonnegative_integer(declared_size, "declared_size")
    _validate_nonnegative_integer(maximum_source_file_bytes, "maximum_source_file_bytes")
    if maximum_source_file_bytes <= 0:
        raise ValueError("maximum_source_file_bytes must be positive")
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    if not isinstance(expected_sha256, str) or not SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError("expected_sha256 must be lowercase SHA-256")
    if declared_size > maximum_source_file_bytes:
        return StreamReadResult("oversize_declared", None, declared_size, 0, None)

    digest = hashlib.sha256()
    remaining = declared_size
    bytes_read = 0
    payload = bytearray(declared_size)
    cursor = 0
    try:
        while remaining:
            request_bytes = min(chunk_bytes, remaining)
            target = memoryview(payload)[cursor : cursor + request_bytes]
            reader = getattr(stream, "readinto", None)
            if not callable(reader):
                return StreamReadResult("read_failure", None, declared_size, bytes_read, None)
            count = reader(target)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0 or count > request_bytes:
                return StreamReadResult("read_failure", None, declared_size, bytes_read, None)
            if count == 0:
                return StreamReadResult("read_truncated", None, declared_size, bytes_read, None)
            digest.update(target[:count])
            bytes_read += count
            cursor += count
            remaining -= count
        extra = bytearray(1)
        reader = getattr(stream, "readinto", None)
        if not callable(reader):
            return StreamReadResult("read_failure", None, declared_size, bytes_read, None)
        extra_count = reader(memoryview(extra))
        if isinstance(extra_count, bool) or not isinstance(extra_count, int) or extra_count < 0 or extra_count > 1:
            return StreamReadResult("read_failure", None, declared_size, bytes_read, None)
        if extra_count:
            return StreamReadResult("declared_size_mismatch", None, declared_size, bytes_read + extra_count, None)
    except Exception:
        return StreamReadResult("read_failure", None, declared_size, bytes_read, None)

    observed_sha256 = digest.hexdigest()
    if observed_sha256 != expected_sha256:
        return StreamReadResult("sha256_mismatch", None, declared_size, bytes_read, observed_sha256)
    return StreamReadResult("available", bytes(payload), declared_size, bytes_read, observed_sha256)
