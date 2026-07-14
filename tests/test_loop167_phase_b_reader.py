from __future__ import annotations

import hashlib
import inspect
from io import BytesIO

from src.loop167_phase_b.one_pass_reader import read_verified_bytes


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_reader_returns_verified_bytes_from_one_stream() -> None:
    data = b"one-pass-synthetic-bytes"
    result = read_verified_bytes(
        BytesIO(data),
        declared_size=len(data),
        expected_sha256=_sha256(data),
        maximum_source_file_bytes=1024,
        chunk_bytes=3,
    )
    assert result.available is True
    assert result.bytez == data
    assert result.bytes_read == len(data)
    assert result.observed_sha256 == _sha256(data)


def test_reader_fails_closed_for_declared_size_and_hash_drift() -> None:
    data = b"abcdef"
    truncated = read_verified_bytes(
        BytesIO(data[:-1]),
        declared_size=len(data),
        expected_sha256=_sha256(data),
        maximum_source_file_bytes=1024,
    )
    assert truncated.result == "read_truncated"
    assert truncated.bytez is None

    expanded = read_verified_bytes(
        BytesIO(data + b"x"),
        declared_size=len(data),
        expected_sha256=_sha256(data),
        maximum_source_file_bytes=1024,
    )
    assert expanded.result == "declared_size_mismatch"
    assert expanded.bytez is None

    mismatched = read_verified_bytes(
        BytesIO(data),
        declared_size=len(data),
        expected_sha256="0" * 64,
        maximum_source_file_bytes=1024,
    )
    assert mismatched.result == "sha256_mismatch"
    assert mismatched.bytez is None
    assert mismatched.observed_sha256 == _sha256(data)


def test_reader_rejects_oversize_before_stream_read_and_has_no_path_surface() -> None:
    class _NoReadStream:
        def read(self, _: int) -> bytes:
            raise AssertionError("oversize declared input must not be read")

    result = read_verified_bytes(
        _NoReadStream(),
        declared_size=6,
        expected_sha256="0" * 64,
        maximum_source_file_bytes=5,
    )
    assert result.result == "oversize_declared"
    assert result.bytes_read == 0
    assert result.bytez is None
    signature = inspect.signature(read_verified_bytes)
    assert tuple(signature.parameters) == (
        "stream",
        "declared_size",
        "expected_sha256",
        "maximum_source_file_bytes",
        "chunk_bytes",
    )
    source = inspect.getsource(read_verified_bytes)
    for forbidden in ("Path", ".open(", "label", "score", "row_index"):
        assert forbidden not in source


def test_reader_rejects_a_nonconforming_readinto_count_without_materializing_bytes() -> None:
    class _InvalidReadIntoStream:
        def readinto(self, target: memoryview) -> int:
            return len(target) + 1

    result = read_verified_bytes(
        _InvalidReadIntoStream(),
        declared_size=4,
        expected_sha256=_sha256(b"test"),
        maximum_source_file_bytes=1024,
    )

    assert result.result == "read_failure"
    assert result.bytez is None
    assert result.bytes_read == 0
