from __future__ import annotations

import hashlib
import shutil
import struct
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from identity_feature_guard import identity_feature_violations  # noqa: E402
import train_loop55_overlay_boundary as loop55  # noqa: E402
from train_loop55_overlay_boundary import (  # noqa: E402
    OVERLAY_BOUNDARY_FEATURE_NAMES,
    OverlayBoundaryConfig,
    _subtract_span,
    build_overlay_boundary_matrix,
    build_overlay_boundary_cache,
    overlay_boundary_features_from_path,
)


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _put_u16(buffer: bytearray, offset: int, value: int) -> None:
    buffer[offset : offset + 2] = struct.pack("<H", value)


def _put_u32(buffer: bytearray, offset: int, value: int) -> None:
    buffer[offset : offset + 4] = struct.pack("<I", value)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _minimal_signed_overlay_pe() -> bytes:
    data = bytearray(b"\0" * 0x1000)
    data[0:2] = b"MZ"
    _put_u32(data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    coff = 0x84
    optional = coff + 20
    section_table = optional + 224

    _put_u16(data, coff, 0x14C)
    _put_u16(data, coff + 2, 1)
    _put_u16(data, coff + 16, 224)
    _put_u16(data, coff + 18, 0x010F)

    _put_u16(data, optional, 0x10B)
    _put_u32(data, optional + 16, 0x1000)
    _put_u32(data, optional + 28, 0x400000)
    _put_u32(data, optional + 32, 0x1000)
    _put_u32(data, optional + 36, 0x200)
    _put_u16(data, optional + 68, 3)
    _put_u32(data, optional + 80, 0x2000)
    _put_u32(data, optional + 84, 0x200)
    _put_u32(data, optional + 92, 16)
    _put_u32(data, optional + 96 + 4 * 8, 0x800)  # security directory file offset
    _put_u32(data, optional + 96 + 4 * 8 + 4, 0x100)

    text = section_table
    data[text : text + 8] = b".text\0\0\0"
    _put_u32(data, text + 8, 0x1000)
    _put_u32(data, text + 12, 0x1000)
    _put_u32(data, text + 16, 0x400)
    _put_u32(data, text + 20, 0x400)
    _put_u32(data, text + 36, 0x60000020)

    data[0x400:0x800] = b"\x90\xCC" * 512
    data[0x800:0x900] = b"CERT" * 64
    data[0x900:0xA00] = bytes(range(256))
    return bytes(data[:0xA00])


def test_subtract_span_leaves_payload_after_security():
    assert _subtract_span([(100, 260)], (100, 180)) == [(180, 260)]
    assert _subtract_span([(100, 260)], (140, 180)) == [(100, 140), (180, 260)]
    assert _subtract_span([(100, 260)], (90, 300)) == []


def test_overlay_boundary_features_ignore_filename(tmp_path: Path):
    payload = _minimal_signed_overlay_pe()
    first = tmp_path / "benign-looking.exe"
    second = tmp_path / "malicious-looking-no-extension"
    first.write_bytes(payload)
    second.write_bytes(payload)

    first_features = overlay_boundary_features_from_path(first)
    second_features = overlay_boundary_features_from_path(second)

    assert first_features.shape == (len(OVERLAY_BOUNDARY_FEATURE_NAMES),)
    assert np.isfinite(first_features).all()
    np.testing.assert_array_equal(first_features, second_features)


def test_overlay_boundary_detects_payload_after_security(tmp_path: Path):
    sample = tmp_path / "sample.bin"
    sample.write_bytes(_minimal_signed_overlay_pe())

    features = overlay_boundary_features_from_path(sample)
    by_name = {name: features[index] for index, name in enumerate(OVERLAY_BOUNDARY_FEATURE_NAMES)}

    assert by_name["overlay_boundary_security_present"] == 1.0
    assert by_name["overlay_boundary_overlay_present"] == 1.0
    assert by_name["overlay_boundary_payload_present"] == 1.0
    assert by_name["overlay_boundary_payload_after_security"] == 1.0
    assert by_name["overlay_boundary_security_starts_at_overlay"] == 1.0
    assert by_name["overlay_boundary_payload_after_cert_log_size"] > 0.0
    assert by_name["overlay_boundary_payload_entropy"] > 0.0


def test_overlay_boundary_feature_names_are_identity_safe():
    assert identity_feature_violations(OVERLAY_BOUNDARY_FEATURE_NAMES) == []


def test_overlay_boundary_cache_builder_reuses_content_features(tmp_path: Path):
    sample = tmp_path / "sample.bin"
    payload = _minimal_signed_overlay_pe()
    sample.write_bytes(payload)
    source_sha = _sha256_bytes(payload)
    rows = [{"source_path": str(sample), "source_sha256": source_sha}]

    report = build_overlay_boundary_cache(rows, cache_dir=tmp_path / "cache", workers=1)

    assert report["processed"] == 1
    assert report["zero_features"] == 0
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1


def test_overlay_boundary_cache_builder_rejects_multiprocess_workers(tmp_path: Path):
    with pytest.raises(ValueError, match="single-process"):
        build_overlay_boundary_cache([], cache_dir=tmp_path / "cache", workers=2)


def test_build_overlay_boundary_matrix_preallocates_stable_width(monkeypatch):
    def fake_features(row, _cache_dir):
        return np.full(len(OVERLAY_BOUNDARY_FEATURE_NAMES), float(row["value"]), dtype=np.float32)

    monkeypatch.setattr(loop55, "overlay_boundary_features_for_row", fake_features)

    matrix = build_overlay_boundary_matrix(
        [{"source_sha256": "a" * 64, "value": 1}, {"source_sha256": "b" * 64, "value": 2}],
        OverlayBoundaryConfig(cache_dir=None),
    )

    assert matrix.shape == (2, len(OVERLAY_BOUNDARY_FEATURE_NAMES))
    assert matrix.dtype == np.float32
    assert matrix[0, 0] == 1.0
    assert matrix[1, 0] == 2.0


def test_read_span_prefix_reads_only_requested_bytes(tmp_path: Path, monkeypatch):
    sample = tmp_path / "large-section.bin"
    sample.write_bytes(b"A" * 8192)
    seen = {}
    original_open = Path.open

    def tracking_open(self, *args, **kwargs):
        handle = original_open(self, *args, **kwargs)
        original_read = handle.read

        def tracking_read(size=-1):
            seen["size"] = size
            return original_read(size)

        handle.read = tracking_read
        return handle

    monkeypatch.setattr(Path, "open", tracking_open)

    payload = loop55._read_span_prefix(sample, (0, 8192), 4096)

    assert len(payload) == 4096
    assert seen["size"] == 4096


def test_overlay_boundary_cache_path_rejects_invalid_source_sha256(tmp_path: Path):
    with pytest.raises(ValueError, match="invalid source_sha256"):
        loop55._overlay_cache_path(
            {"source_path": str(tmp_path / "sample.exe"), "source_sha256": "../escape"},
            str(tmp_path / "cache"),
        )

    assert not (tmp_path / "escape.npz").exists()


def test_overlay_boundary_cache_path_is_namespaced(tmp_path: Path):
    source_sha = "a" * 64
    cache_path = loop55._overlay_cache_path(
        {"source_path": str(tmp_path / "sample.exe"), "source_sha256": source_sha},
        str(tmp_path / "cache"),
    )

    assert cache_path is not None
    assert cache_path.name == f"overlay_boundary_v1_{source_sha}.npz"


def test_overlay_boundary_features_reject_source_sha256_mismatch_before_writing(
    tmp_path: Path,
    monkeypatch,
):
    source_path = tmp_path / "sample.exe"
    source_path.write_bytes(b"actual-content")
    wrong_sha = _sha256_bytes(b"different-content")
    cache_dir = tmp_path / "cache"

    def fail_if_extractor_is_called(_path):
        raise AssertionError("extractor should not run when source_sha256 mismatches source_path bytes")

    monkeypatch.setattr(loop55, "overlay_boundary_features_from_path", fail_if_extractor_is_called)

    with pytest.raises(ValueError, match="source_sha256_mismatch"):
        loop55.overlay_boundary_features_for_row(
            {"source_path": str(source_path), "source_sha256": wrong_sha},
            str(cache_dir),
        )

    assert not (cache_dir / f"{wrong_sha}.npz").exists()


def test_overlay_boundary_features_reject_existing_cache_when_source_sha256_mismatches(
    tmp_path: Path,
    monkeypatch,
):
    source_path = tmp_path / "sample.exe"
    source_path.write_bytes(b"actual-content")
    wrong_sha = _sha256_bytes(b"different-content")
    cache_dir = tmp_path / "cache"
    row = {"source_path": str(source_path), "source_sha256": wrong_sha}
    cache_path = loop55._overlay_cache_path(row, str(cache_dir))
    assert cache_path is not None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=np.ones(len(OVERLAY_BOUNDARY_FEATURE_NAMES), dtype=np.float32))

    def fail_if_extractor_is_called(_path):
        raise AssertionError("extractor should not run when source_sha256 mismatches source_path bytes")

    monkeypatch.setattr(loop55, "overlay_boundary_features_from_path", fail_if_extractor_is_called)

    with pytest.raises(ValueError, match="source_sha256_mismatch"):
        loop55.overlay_boundary_features_for_row(row, str(cache_dir))
