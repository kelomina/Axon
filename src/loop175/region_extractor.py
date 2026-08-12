"""Deterministic content-only region extraction for Loop175."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable, Iterable


class RegionKind(IntEnum):
    MISSING = 0
    DOS_PE_HEADER = 1
    ENTRYPOINT = 2
    EXECUTABLE_SECTION = 3
    SEMANTIC_SECTION = 4
    OVERLAY = 5


@dataclass(frozen=True)
class RegionExtractionConfig:
    maximum_regions: int = 16
    maximum_region_bytes: int = 8192
    maximum_total_region_bytes: int = 131072
    maximum_file_bytes: int = 1024 * 1024 * 1024
    header_parse_bytes: int = 1024 * 1024
    maximum_executable_sections: int = 4
    maximum_semantic_sections: int = 3

    def __post_init__(self) -> None:
        if self.maximum_regions <= 0 or self.maximum_region_bytes <= 0:
            raise ValueError("region limits must be positive")
        if self.maximum_regions * self.maximum_region_bytes > self.maximum_total_region_bytes:
            raise ValueError("region slots exceed the total byte budget")


@dataclass(frozen=True)
class Region:
    kind: RegionKind
    start: int
    data: bytes
    missing_reason: str = ""

    @property
    def length(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class RegionExtractionResult:
    status: str
    file_size: int
    bytes_read: int
    parse_ok: bool
    regions: tuple[Region, ...]

    @property
    def supported(self) -> bool:
        return self.status == "ok" and self.parse_ok

    @property
    def model_region_bytes(self) -> int:
        return sum(region.length for region in self.regions)


@dataclass(frozen=True)
class _Section:
    index: int
    name: str
    virtual_size: int
    virtual_address: int
    raw_size: int
    raw_offset: int
    characteristics: int

    @property
    def executable(self) -> bool:
        return bool(self.characteristics & 0x20000000)

    @property
    def resource_like(self) -> bool:
        return self.name.casefold() in {".rsrc", "rsrc", ".resource"}


@dataclass(frozen=True)
class _ParsedPe:
    entrypoint_rva: int
    size_of_headers: int
    sections: tuple[_Section, ...]

    def rva_to_offset(self, rva: int, file_size: int) -> int | None:
        if rva < 0:
            return None
        if rva < self.size_of_headers and rva < file_size:
            return rva
        for section in self.sections:
            span = max(section.virtual_size, section.raw_size)
            if span <= 0:
                continue
            if section.virtual_address <= rva < section.virtual_address + span:
                offset = section.raw_offset + (rva - section.virtual_address)
                if 0 <= offset < file_size:
                    return offset
        return None


def _u16(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 2 > len(data):
        raise ValueError("truncated PE field")
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        raise ValueError("truncated PE field")
    return struct.unpack_from("<I", data, offset)[0]


def _parse_pe_header(header: bytes, file_size: int) -> _ParsedPe:
    if len(header) < 64 or header[:2] != b"MZ":
        raise ValueError("missing DOS header")
    pe_offset = _u32(header, 0x3C)
    if pe_offset + 24 > len(header) or header[pe_offset : pe_offset + 4] != b"PE\0\0":
        raise ValueError("missing PE header")
    file_header = pe_offset + 4
    section_count = _u16(header, file_header + 2)
    optional_size = _u16(header, file_header + 16)
    if section_count <= 0 or section_count > 96:
        raise ValueError("unsupported section count")
    optional_offset = file_header + 20
    section_table = optional_offset + optional_size
    if optional_size < 64 or section_table + section_count * 40 > len(header):
        raise ValueError("truncated optional header or section table")
    magic = _u16(header, optional_offset)
    if magic not in {0x10B, 0x20B}:
        raise ValueError("unsupported optional header")
    entrypoint_rva = _u32(header, optional_offset + 16)
    size_of_headers = _u32(header, optional_offset + 60)
    sections: list[_Section] = []
    for index in range(section_count):
        offset = section_table + index * 40
        raw_name = header[offset : offset + 8].split(b"\0", 1)[0]
        sections.append(
            _Section(
                index=index,
                name=raw_name.decode("ascii", errors="ignore"),
                virtual_size=_u32(header, offset + 8),
                virtual_address=_u32(header, offset + 12),
                raw_size=_u32(header, offset + 16),
                raw_offset=_u32(header, offset + 20),
                characteristics=_u32(header, offset + 36),
            )
        )
    if size_of_headers <= 0:
        size_of_headers = min(file_size, section_table + section_count * 40)
    return _ParsedPe(entrypoint_rva, size_of_headers, tuple(sections))


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for value in data:
        counts[value] += 1
    length = float(len(data))
    return -sum((count / length) * math.log2(count / length) for count in counts if count)


def _window_start(center: int, file_size: int, window_size: int) -> int:
    if file_size <= window_size:
        return 0
    return min(max(center - window_size // 2, 0), file_size - window_size)


def _missing_regions(config: RegionExtractionConfig, reason: str) -> tuple[Region, ...]:
    return tuple(Region(RegionKind.MISSING, 0, b"", reason) for _ in range(config.maximum_regions))


def _extract_with_reader(
    *,
    file_size: int,
    header: bytes,
    read_at: Callable[[int, int], bytes],
    bytes_read: Callable[[], int],
    config: RegionExtractionConfig,
) -> RegionExtractionResult:
    if file_size == 0:
        return RegionExtractionResult("empty", 0, bytes_read(), False, _missing_regions(config, "empty"))
    if file_size > config.maximum_file_bytes:
        return RegionExtractionResult(
            "oversize", file_size, bytes_read(), False, _missing_regions(config, "oversize")
        )

    candidates: list[Region] = []

    def append(kind: RegionKind, start: int, requested: int) -> None:
        start = min(max(int(start), 0), file_size)
        requested = min(max(int(requested), 0), config.maximum_region_bytes, file_size - start)
        if requested <= 0:
            return
        payload = read_at(start, requested)
        if not payload:
            return
        candidates.append(Region(kind, start, payload[:requested]))

    append(RegionKind.DOS_PE_HEADER, 0, min(config.maximum_region_bytes, file_size))
    try:
        parsed = _parse_pe_header(header, file_size)
    except ValueError:
        regions = candidates[: config.maximum_regions]
        regions.extend(
            Region(RegionKind.MISSING, 0, b"", "pe_parse_failure")
            for _ in range(config.maximum_regions - len(regions))
        )
        return RegionExtractionResult(
            "pe_parse_failure", file_size, bytes_read(), False, tuple(regions)
        )

    entrypoint_offset = parsed.rva_to_offset(parsed.entrypoint_rva, file_size)
    if entrypoint_offset is not None:
        append(
            RegionKind.ENTRYPOINT,
            _window_start(entrypoint_offset, file_size, config.maximum_region_bytes),
            min(config.maximum_region_bytes, file_size),
        )

    executable_sections = [
        section for section in parsed.sections if section.executable and section.raw_size > 0
    ][: config.maximum_executable_sections]
    for section in executable_sections:
        section_end = min(file_size, section.raw_offset + section.raw_size)
        append(RegionKind.EXECUTABLE_SECTION, section.raw_offset, section_end - section.raw_offset)
        append(
            RegionKind.EXECUTABLE_SECTION,
            max(section.raw_offset, section_end - config.maximum_region_bytes),
            section_end - max(section.raw_offset, section_end - config.maximum_region_bytes),
        )

    executable_indices = {section.index for section in executable_sections}
    semantic_candidates: list[tuple[tuple[int, float, int, int], _Section]] = []
    for section in parsed.sections:
        if section.index in executable_indices or section.raw_size <= 0 or section.raw_offset >= file_size:
            continue
        sample = read_at(
            section.raw_offset,
            min(section.raw_size, config.maximum_region_bytes, file_size - section.raw_offset),
        )
        rank = (int(section.resource_like), _entropy(sample), section.raw_size, -section.index)
        semantic_candidates.append((rank, section))
    semantic_candidates.sort(key=lambda item: item[0], reverse=True)
    for _rank, section in semantic_candidates[: config.maximum_semantic_sections]:
        append(
            RegionKind.SEMANTIC_SECTION,
            section.raw_offset,
            min(section.raw_size, file_size - section.raw_offset),
        )

    overlay_start = min(
        file_size,
        max(
            [parsed.size_of_headers]
            + [
                min(file_size, section.raw_offset + section.raw_size)
                for section in parsed.sections
                if section.raw_offset < file_size
            ],
        ),
    )
    if overlay_start < file_size:
        append(RegionKind.OVERLAY, overlay_start, file_size - overlay_start)
        append(
            RegionKind.OVERLAY,
            max(overlay_start, file_size - config.maximum_region_bytes),
            file_size - max(overlay_start, file_size - config.maximum_region_bytes),
        )

    # 只按冻结的内容语义键去重，保留不同 region type 对同一物理窗口的独立含义。
    unique: list[Region] = []
    seen: set[tuple[int, int, int]] = set()
    for region in candidates:
        key = (int(region.kind), region.start, region.length)
        if key in seen:
            continue
        seen.add(key)
        unique.append(region)
        if len(unique) >= config.maximum_regions:
            break
    if sum(region.length for region in unique) > config.maximum_total_region_bytes:
        raise RuntimeError("region byte accounting exceeded the frozen budget")
    unique.extend(
        Region(RegionKind.MISSING, 0, b"", "unused_slot")
        for _ in range(config.maximum_regions - len(unique))
    )
    return RegionExtractionResult("ok", file_size, bytes_read(), True, tuple(unique))


def extract_regions_from_bytes(
    data: bytes | bytearray | memoryview,
    config: RegionExtractionConfig | None = None,
) -> RegionExtractionResult:
    config = config or RegionExtractionConfig()
    payload = bytes(data)
    read_total = 0

    def read_at(start: int, length: int) -> bytes:
        nonlocal read_total
        chunk = payload[start : start + length]
        read_total += len(chunk)
        return chunk

    header = read_at(0, min(len(payload), config.header_parse_bytes))
    return _extract_with_reader(
        file_size=len(payload),
        header=header,
        read_at=read_at,
        bytes_read=lambda: read_total,
        config=config,
    )


def extract_regions_from_path(
    path: str | Path,
    config: RegionExtractionConfig | None = None,
) -> RegionExtractionResult:
    config = config or RegionExtractionConfig()
    source = Path(path)
    try:
        file_size = source.stat().st_size
    except OSError:
        return RegionExtractionResult(
            "read_failure", 0, 0, False, _missing_regions(config, "read_failure")
        )
    if file_size > config.maximum_file_bytes:
        return RegionExtractionResult(
            "oversize", file_size, 0, False, _missing_regions(config, "oversize")
        )

    read_total = 0
    try:
        with source.open("rb") as handle:

            def read_at(start: int, length: int) -> bytes:
                nonlocal read_total
                handle.seek(start)
                chunk = handle.read(length)
                read_total += len(chunk)
                return chunk

            header = read_at(0, min(file_size, config.header_parse_bytes))
            return _extract_with_reader(
                file_size=file_size,
                header=header,
                read_at=read_at,
                bytes_read=lambda: read_total,
                config=config,
            )
    except OSError:
        return RegionExtractionResult(
            "read_failure", file_size, read_total, False, _missing_regions(config, "read_failure")
        )


def region_kind_counts(regions: Iterable[Region]) -> dict[str, int]:
    counts = {kind.name.casefold(): 0 for kind in RegionKind}
    for region in regions:
        counts[region.kind.name.casefold()] += 1
    return counts
