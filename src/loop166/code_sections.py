"""Deterministic extraction of raw bytes from PE executable sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import pefile

IMAGE_SCN_MEM_EXECUTE = 0x20000000
MISSING_REASONS = (
    "parse_failure",
    "no_executable_section",
    "zero_raw_executable_section",
    "invalid_executable_section_span",
)


@dataclass(frozen=True)
class SpanPlan:
    spans: tuple[tuple[int, int], ...]
    declared_executable_sections: int
    declared_raw_bytes: int
    overlap_bytes_removed: int
    missing_reason: Optional[str]


@dataclass(frozen=True)
class CodeSectionExtraction:
    code_bytes: bytes
    spans: tuple[tuple[int, int], ...]
    declared_executable_sections: int
    declared_raw_bytes: int
    overlap_bytes_removed: int
    parser_warning_count: int
    missing_reason: Optional[str]

    @property
    def available(self) -> bool:
        return self.missing_reason is None


def plan_executable_spans(sections: Sequence[object], *, file_size: int) -> SpanPlan:
    if file_size < 0:
        raise ValueError("file_size cannot be negative")
    declared_executable_sections = 0
    declared_raw_bytes = 0
    spans: list[tuple[int, int]] = []
    invalid_span = False
    for section in sections:
        characteristics = int(getattr(section, "Characteristics", 0) or 0)
        if characteristics & IMAGE_SCN_MEM_EXECUTE == 0:
            continue
        declared_executable_sections += 1
        start = int(getattr(section, "PointerToRawData", 0) or 0)
        size = int(getattr(section, "SizeOfRawData", 0) or 0)
        if start < 0 or size < 0:
            invalid_span = True
            continue
        if size == 0:
            continue
        end = start + size
        declared_raw_bytes += size
        if start > file_size or end > file_size or end <= start:
            invalid_span = True
            continue
        spans.append((start, end))

    if declared_executable_sections == 0:
        return SpanPlan((), 0, 0, 0, "no_executable_section")
    if invalid_span:
        return SpanPlan(
            (),
            declared_executable_sections,
            declared_raw_bytes,
            0,
            "invalid_executable_section_span",
        )
    if not spans:
        return SpanPlan(
            (),
            declared_executable_sections,
            declared_raw_bytes,
            0,
            "zero_raw_executable_section",
        )

    # 按文件偏移合并重叠区，确保同一原始字节不会因异常 section table 被重复学习。
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    merged_spans = tuple((start, end) for start, end in merged)
    deduplicated_bytes = sum(end - start for start, end in merged_spans)
    return SpanPlan(
        merged_spans,
        declared_executable_sections,
        declared_raw_bytes,
        declared_raw_bytes - deduplicated_bytes,
        None,
    )


def _default_pe_factory(*, data: bytes, fast_load: bool) -> Any:
    return pefile.PE(data=data, fast_load=fast_load)


def extract_executable_code(
    bytez: bytes,
    *,
    pe_factory: Callable[..., Any] = _default_pe_factory,
) -> CodeSectionExtraction:
    parsed = None
    try:
        parsed = pe_factory(data=bytez, fast_load=True)
        warnings = getattr(parsed, "get_warnings", lambda: [])()
        warning_count = len(warnings or [])
        plan = plan_executable_spans(getattr(parsed, "sections", ()), file_size=len(bytez))
        if plan.missing_reason is not None:
            return CodeSectionExtraction(
                b"",
                plan.spans,
                plan.declared_executable_sections,
                plan.declared_raw_bytes,
                plan.overlap_bytes_removed,
                warning_count,
                plan.missing_reason,
            )
        code_bytes = b"".join(bytez[start:end] for start, end in plan.spans)
        if not code_bytes:
            raise ValueError("Available span plan produced no code bytes")
        return CodeSectionExtraction(
            code_bytes,
            plan.spans,
            plan.declared_executable_sections,
            plan.declared_raw_bytes,
            plan.overlap_bytes_removed,
            warning_count,
            None,
        )
    except (pefile.PEFormatError, AttributeError, TypeError, ValueError):
        return CodeSectionExtraction(b"", (), 0, 0, 0, 0, "parse_failure")
    finally:
        close = getattr(parsed, "close", None)
        if callable(close):
            close()
