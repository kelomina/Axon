from __future__ import annotations

import shutil
import struct
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from identity_feature_guard import identity_feature_violations  # noqa: E402
from train_loop55_overlay_boundary import (  # noqa: E402
    OVERLAY_BOUNDARY_FEATURE_NAMES,
    _subtract_span,
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
    sample.write_bytes(_minimal_signed_overlay_pe())
    rows = [{"source_path": str(sample), "source_sha256": ""}]

    report = build_overlay_boundary_cache(rows, cache_dir=tmp_path / "cache", workers=1)

    assert report["processed"] == 1
    assert report["zero_features"] == 0
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
