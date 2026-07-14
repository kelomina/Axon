from __future__ import annotations

import struct

import numpy as np

from src.loop167_phase_b.b0_projector import extract_b0_projection
from src.loop167_phase_b.ember_controls import extract_context_features
from src.loop167_phase_b.raw_context import RawFeatureContext


def _minimal_pe32() -> bytes:
    dos = bytearray(0x80)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x80)
    coff = struct.pack("<HHIIIHH", 0x14C, 1, 0, 0, 0, 0xE0, 0x0102)
    optional = bytearray(0xE0)
    struct.pack_into("<H", optional, 0, 0x10B)
    struct.pack_into("<I", optional, 4, 0x200)
    struct.pack_into("<I", optional, 8, 0x200)
    struct.pack_into("<I", optional, 16, 0x1000)
    struct.pack_into("<I", optional, 20, 0x1000)
    struct.pack_into("<I", optional, 24, 0x2000)
    struct.pack_into("<I", optional, 28, 0x400000)
    struct.pack_into("<I", optional, 32, 0x1000)
    struct.pack_into("<I", optional, 36, 0x200)
    struct.pack_into("<H", optional, 40, 6)
    struct.pack_into("<H", optional, 48, 6)
    struct.pack_into("<I", optional, 56, 0x2000)
    struct.pack_into("<I", optional, 60, 0x200)
    struct.pack_into("<H", optional, 68, 3)
    struct.pack_into("<H", optional, 70, 0x140)
    struct.pack_into("<I", optional, 72, 0x100000)
    struct.pack_into("<I", optional, 76, 0x1000)
    struct.pack_into("<I", optional, 80, 0x100000)
    struct.pack_into("<I", optional, 84, 0x1000)
    struct.pack_into("<I", optional, 92, 16)
    section = struct.pack("<8sIIIIIIHHI", b".text\x00\x00\x00", 0x10, 0x1000, 0x200, 0x200, 0, 0, 0, 0, 0x60000020)
    headers = b"PE\x00\x00" + coff + bytes(optional) + section
    return bytes(dos) + headers + bytes(0x200 - len(headers) - len(dos)) + b"\x90" * 0x200


def test_minimal_pe_uses_real_in_memory_parser_and_produces_finite_bundles() -> None:
    context = RawFeatureContext.from_bytes(_minimal_pe32(), maximum_input_bytes=1024 * 1024)
    assert context.pe_parse_succeeded is True
    assert context.directory_parse_reason is None

    b0 = extract_b0_projection(context)
    controls_and_novel = extract_context_features(context)
    assert b0.values.shape == (571,)
    assert b0.missing_indicators.tolist() == [0.0] * 6
    assert controls_and_novel.controls.values.shape == (536,)
    assert controls_and_novel.novel.values.shape == (292,)
    assert np.isfinite(b0.values).all()
    assert np.isfinite(controls_and_novel.controls.values).all()
    assert np.isfinite(controls_and_novel.novel.values).all()
