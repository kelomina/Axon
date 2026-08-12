from __future__ import annotations

import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.region_extractor import (  # noqa: E402
    RegionExtractionConfig,
    RegionKind,
    extract_regions_from_bytes,
)


def _synthetic_pe() -> bytes:
    data = bytearray(0x7000)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    file_header = 0x84
    struct.pack_into("<H", data, file_header, 0x14C)
    struct.pack_into("<H", data, file_header + 2, 3)
    struct.pack_into("<H", data, file_header + 16, 0xE0)
    optional = file_header + 20
    struct.pack_into("<H", data, optional, 0x10B)
    struct.pack_into("<I", data, optional + 16, 0x1100)
    struct.pack_into("<I", data, optional + 60, 0x400)
    section_table = optional + 0xE0
    sections = [
        (b".text", 0x1800, 0x1000, 0x1800, 0x400, 0x60000020),
        (b".rsrc", 0x1000, 0x3000, 0x1000, 0x1C00, 0x40000040),
        (b".data", 0x1000, 0x4000, 0x1000, 0x2C00, 0xC0000040),
    ]
    for index, (name, virtual_size, virtual_address, raw_size, raw_offset, characteristics) in enumerate(sections):
        offset = section_table + index * 40
        data[offset : offset + len(name)] = name
        struct.pack_into("<I", data, offset + 8, virtual_size)
        struct.pack_into("<I", data, offset + 12, virtual_address)
        struct.pack_into("<I", data, offset + 16, raw_size)
        struct.pack_into("<I", data, offset + 20, raw_offset)
        struct.pack_into("<I", data, offset + 36, characteristics)
    for index in range(0x400, 0x3400):
        data[index] = (index * 17 + index // 7) & 0xFF
    data[0x3400:] = b"OVERLAY" * ((len(data) - 0x3400) // 7)
    return bytes(data)


def test_region_priority_budget_and_overlay_are_deterministic() -> None:
    config = RegionExtractionConfig(maximum_region_bytes=512, maximum_total_region_bytes=8192)
    first = extract_regions_from_bytes(_synthetic_pe(), config)
    second = extract_regions_from_bytes(_synthetic_pe(), config)

    assert first == second
    assert first.supported
    assert len(first.regions) == 16
    assert first.regions[0].kind is RegionKind.DOS_PE_HEADER
    assert first.regions[1].kind is RegionKind.ENTRYPOINT
    assert any(region.kind is RegionKind.EXECUTABLE_SECTION for region in first.regions)
    assert any(region.kind is RegionKind.SEMANTIC_SECTION for region in first.regions)
    assert any(region.kind is RegionKind.OVERLAY for region in first.regions)
    assert first.model_region_bytes <= config.maximum_total_region_bytes
    keys = [(region.kind, region.start, region.length) for region in first.regions if region.length]
    assert len(keys) == len(set(keys))


def test_parse_failure_and_empty_inputs_remain_in_the_denominator() -> None:
    invalid = extract_regions_from_bytes(b"not-a-pe")
    empty = extract_regions_from_bytes(b"")

    assert invalid.status == "pe_parse_failure"
    assert not invalid.supported
    assert len(invalid.regions) == 16
    assert empty.status == "empty"
    assert not empty.supported
    assert len(empty.regions) == 16
    assert all(region.kind is RegionKind.MISSING for region in empty.regions)
