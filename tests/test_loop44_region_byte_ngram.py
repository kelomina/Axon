from __future__ import annotations

import shutil
import struct
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_loop44_region_byte_ngram import (  # noqa: E402
    REGION_SCALAR_FEATURE_NAMES,
    RegionHashConfig,
    _hashed_region_ngram_features,
    _overlay_payload_region,
    region_slices_from_path,
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


def _minimal_pe_bytes() -> bytes:
    data = bytearray(b"\0" * 0x1800)
    data[0:2] = b"MZ"
    _put_u32(data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    coff = 0x84
    optional = coff + 20
    section_table = optional + 224

    _put_u16(data, coff, 0x14C)
    _put_u16(data, coff + 2, 2)
    _put_u16(data, coff + 16, 224)
    _put_u16(data, coff + 18, 0x010F)

    _put_u16(data, optional, 0x10B)
    _put_u32(data, optional + 16, 0x1000)
    _put_u32(data, optional + 28, 0x400000)
    _put_u32(data, optional + 32, 0x1000)
    _put_u32(data, optional + 36, 0x200)
    _put_u16(data, optional + 68, 3)
    _put_u32(data, optional + 80, 0x3000)
    _put_u32(data, optional + 84, 0x200)
    _put_u32(data, optional + 92, 16)
    _put_u32(data, optional + 96 + 1 * 8, 0x2100)  # import directory RVA
    _put_u32(data, optional + 96 + 1 * 8 + 4, 0x40)
    _put_u32(data, optional + 96 + 2 * 8, 0x2000)  # resource directory RVA
    _put_u32(data, optional + 96 + 2 * 8 + 4, 0x80)
    _put_u32(data, optional + 96 + 4 * 8, 0xD00)  # security directory file offset
    _put_u32(data, optional + 96 + 4 * 8 + 4, 0x80)

    text = section_table
    data[text : text + 8] = b".text\0\0\0"
    _put_u32(data, text + 8, 0x1000)
    _put_u32(data, text + 12, 0x1000)
    _put_u32(data, text + 16, 0x400)
    _put_u32(data, text + 20, 0x400)
    _put_u32(data, text + 36, 0x60000020)

    rsrc = section_table + 40
    data[rsrc : rsrc + 8] = b".rsrc\0\0\0"
    _put_u32(data, rsrc + 8, 0x1000)
    _put_u32(data, rsrc + 12, 0x2000)
    _put_u32(data, rsrc + 16, 0x400)
    _put_u32(data, rsrc + 20, 0x800)
    _put_u32(data, rsrc + 36, 0x40000040)

    data[0x400:0x420] = b"\x90\x90\xCC\xCC" * 8
    data[0x800:0x840] = b"RSRC" * 16
    data[0xC00:0xC20] = b"OVERLAY_REGION_CONTENT_123456"
    data[0xD00:0xD80] = b"CERT" * 32
    data[0xD80:0xDC0] = b"PAYLOAD" * 10
    return bytes(data[:0xDC0])


def _config() -> RegionHashConfig:
    return RegionHashConfig(
        n_features=4096,
        prefix_len=128,
        ngram_min=2,
        ngram_max=3,
        ngram_stride=1,
        include_prefix_features=False,
        include_full_ngram_features=False,
        include_region_ngram_features=True,
        include_region_scalar_features=True,
        include_byte_hist=False,
        include_cache_features=False,
        region_window=128,
        tail_window=128,
        max_byte_length=512,
        pe_feature_dim=256,
        stat_feature_dim=49,
        lightweight_feature_dim=256,
    )


def test_region_slices_are_derived_from_pe_content_not_filename():
    with _case_dir("loop44_regions") as tmp_path:
        first = tmp_path / "benign-looking-name.bin"
        second = tmp_path / "malicious-looking-name.exe"
        payload = _minimal_pe_bytes()
        first.write_bytes(payload)
        second.write_bytes(payload)

        first_regions = region_slices_from_path(first, region_window=128, tail_window=128)
        second_regions = region_slices_from_path(second, region_window=128, tail_window=128)

    first_tuples = [(region.name, region.start, region.size) for region in first_regions]
    second_tuples = [(region.name, region.start, region.size) for region in second_regions]
    assert first_tuples == second_tuples
    assert "entrypoint" in {region.name for region in first_regions}
    assert "resource_directory" in {region.name for region in first_regions}
    assert "security_directory" in {region.name for region in first_regions}
    assert "overlay_payload" in {region.name for region in first_regions}
    assert "tail" in {region.name for region in first_regions}


def test_region_hashes_change_by_region_salt_for_same_bytes():
    config = _config()
    same_data = b"ABCDABCD"
    head_cols = _hashed_region_ngram_features(
        [("head", same_data)],
        ngram_min=config.ngram_min,
        ngram_max=config.ngram_max,
        stride=config.ngram_stride,
        n_features=config.n_features,
    )
    overlay_cols = _hashed_region_ngram_features(
        [("overlay", same_data)],
        ngram_min=config.ngram_min,
        ngram_max=config.ngram_max,
        stride=config.ngram_stride,
        n_features=config.n_features,
    )

    assert head_cols.size > 0
    assert overlay_cols.size == head_cols.size
    assert not np.array_equal(head_cols, overlay_cols)


def test_region_scalar_feature_names_are_identity_safe():
    assert REGION_SCALAR_FEATURE_NAMES
    assert not any("path" in name or "filename" in name or "extension" in name for name in REGION_SCALAR_FEATURE_NAMES)


def test_overlay_payload_excludes_security_blob_when_they_overlap():
    region = _overlay_payload_region(
        overlay_offset=100,
        file_size=260,
        window=512,
        security_span=(100, 180),
    )

    assert region is not None
    assert region.name == "overlay_payload"
    assert region.start == 180
    assert region.size == 80


def test_region_slices_without_pefile_still_have_head_tail(monkeypatch: pytest.MonkeyPatch):
    import train_loop44_region_byte_ngram as loop44

    with _case_dir("loop44_no_pefile") as tmp_path:
        sample = tmp_path / "sample"
        sample.write_bytes(b"A" * 4096)
        monkeypatch.setattr(loop44, "PEFILE_AVAILABLE", False)
        regions = region_slices_from_path(sample, region_window=256, tail_window=128)

    assert [(region.name, region.start, region.size) for region in regions] == [
        ("head", 0, 256),
        ("tail", 4096 - 128, 128),
    ]
