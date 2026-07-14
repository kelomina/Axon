"""Pure in-memory native extraction for Loop167's genuinely novel columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .semantic_mapping import HEADER_NOVEL_LOCAL_INDICES, novel_indices

try:
    import pefile
except ImportError:  # pragma: no cover - the project dependency is normally present.
    pefile = None


BYTE_ENTROPY_WINDOW = 2048
BYTE_ENTROPY_STEP = 1024
HEADER_GLOBAL_START = 696
BYTE_ENTROPY_GLOBAL_START = 263
RICH_PAIR_COUNT_INDEX = 2439
GENERAL_START_BYTE_INDICES = (3, 4, 5, 6)


@dataclass(frozen=True)
class NativeNovelDelta:
    """Finite novel-vector result with source-independent missingness evidence."""

    values: np.ndarray
    original_indices: tuple[int, ...]
    missing_reasons: tuple[str, ...]
    pe_parse_succeeded: bool


def _default_pe_factory(*, data: bytes, fast_load: bool) -> Any:
    if pefile is None:
        raise RuntimeError("pefile is unavailable")
    return pefile.PE(data=data, fast_load=fast_load)


def _integer_attr(value: object, attribute: str) -> int:
    try:
        return int(getattr(value, attribute, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _dos_value(pe: object, member: str) -> int:
    dos_header = getattr(pe, "DOS_HEADER", None)
    dump_dict = getattr(dos_header, "dump_dict", None)
    if callable(dump_dict):
        try:
            raw_value = dump_dict().get(member, {}).get("Value", 0)
            return int(raw_value or 0)
        except (AttributeError, TypeError, ValueError):
            return 0
    return _integer_attr(dos_header, member)


def _byte_entropy_vector(bytez: bytes) -> np.ndarray:
    """Reproduce the pinned nonempty 16x16 byte/entropy histogram semantics."""

    output = np.zeros((16, 16), dtype=np.int32)
    values = np.frombuffer(bytez, dtype=np.uint8)
    if not values.size:
        return output.reshape(-1).astype(np.float32)

    def add_block(block: np.ndarray) -> None:
        counts = np.bincount(block >> 4, minlength=16).astype(np.int32)
        probabilities = counts.astype(np.float32) / float(BYTE_ENTROPY_WINDOW)
        nonzero = np.where(counts)[0]
        entropy = np.sum(-probabilities[nonzero] * np.log2(probabilities[nonzero])) * 2.0
        entropy_bin = min(15, max(0, int(float(entropy) * 2.0)))
        output[entropy_bin, :] += counts

    if values.size < BYTE_ENTROPY_WINDOW:
        add_block(values)
    else:
        for start in range(0, values.size - BYTE_ENTROPY_WINDOW + 1, BYTE_ENTROPY_STEP):
            add_block(values[start : start + BYTE_ENTROPY_WINDOW])

    total = float(output.sum())
    if total > 0.0:
        return (output.astype(np.float32) / total).reshape(-1)
    return output.reshape(-1).astype(np.float32)


def _header_novel_values(pe: object | None) -> dict[int, float]:
    values = {HEADER_GLOBAL_START + local: 0.0 for local in HEADER_NOVEL_LOCAL_INDICES}
    if pe is None:
        return values
    file_header = getattr(pe, "FILE_HEADER", None)
    optional_header = getattr(pe, "OPTIONAL_HEADER", None)
    direct_values = {
        2: _integer_attr(file_header, "NumberOfSymbols"),
        4: _integer_attr(file_header, "PointerToSymbolTable"),
        7: _integer_attr(optional_header, "MajorImageVersion"),
        8: _integer_attr(optional_header, "MinorImageVersion"),
        11: _integer_attr(optional_header, "MajorOperatingSystemVersion"),
        12: _integer_attr(optional_header, "MinorOperatingSystemVersion"),
        13: _integer_attr(optional_header, "MajorSubsystemVersion"),
        14: _integer_attr(optional_header, "MinorSubsystemVersion"),
        20: _integer_attr(optional_header, "SizeOfStackReserve"),
        21: _integer_attr(optional_header, "SizeOfStackCommit"),
        22: _integer_attr(optional_header, "SizeOfHeapReserve"),
        23: _integer_attr(optional_header, "SizeOfHeapCommit"),
        25: _integer_attr(optional_header, "BaseOfCode"),
        29: _integer_attr(optional_header, "NumberOfRvaAndSizes"),
    }
    for local, value in direct_values.items():
        values[HEADER_GLOBAL_START + local] = float(value)
    dos_members = (
        "e_magic",
        "e_cblp",
        "e_cp",
        "e_crlc",
        "e_cparhdr",
        "e_minalloc",
        "e_maxalloc",
        "e_ss",
        "e_sp",
        "e_csum",
        "e_ip",
        "e_cs",
        "e_lfarlc",
        "e_ovno",
        "e_oemid",
        "e_oeminfo",
        "e_lfanew",
    )
    for offset, member in enumerate(dos_members, start=57):
        values[HEADER_GLOBAL_START + offset] = float(_dos_value(pe, member))
    return values


def _rich_pair_count(pe: object | None) -> float:
    rich_header = getattr(pe, "RICH_HEADER", None) if pe is not None else None
    raw_values = getattr(rich_header, "values", None)
    if not raw_values:
        return 0.0
    try:
        return float(len(raw_values) // 2)
    except TypeError:
        return 0.0


def _parse_pe(
    bytez: bytes,
    pe_factory: Callable[..., Any],
) -> tuple[object | None, str | None]:
    if not bytez:
        return None, "empty_input"
    try:
        return pe_factory(data=bytez, fast_load=True), None
    except Exception:
        return None, "pe_parse_failure"


def extract_novel_delta(
    bytez: bytes | bytearray | memoryview,
    *,
    pe_factory: Callable[..., Any] = _default_pe_factory,
) -> NativeNovelDelta:
    """Extract only the Phase-A novel columns from already-opened byte content."""

    if isinstance(bytez, memoryview):
        bytez = bytez.tobytes()
    elif isinstance(bytez, bytearray):
        bytez = bytes(bytez)
    if not isinstance(bytez, bytes):
        raise TypeError("Loop167 native extraction accepts bytes-like content only")

    parsed, parse_reason = _parse_pe(bytez, pe_factory)
    try:
        values_by_index: dict[int, float] = {}
        for offset, index in enumerate(GENERAL_START_BYTE_INDICES):
            values_by_index[index] = float(bytez[offset]) if offset < len(bytez) else 0.0
        byte_entropy = _byte_entropy_vector(bytez)
        for offset, value in enumerate(byte_entropy):
            values_by_index[BYTE_ENTROPY_GLOBAL_START + offset] = float(value)
        values_by_index.update(_header_novel_values(parsed))
        values_by_index[RICH_PAIR_COUNT_INDEX] = _rich_pair_count(parsed)

        indices = novel_indices()
        vector = np.asarray([values_by_index[index] for index in indices], dtype=np.float32)
        if vector.shape != (len(indices),) or not np.isfinite(vector).all():
            raise ValueError("Loop167 native novel vector must be finite and source ordered")
        return NativeNovelDelta(
            values=vector,
            original_indices=indices,
            missing_reasons=(() if parse_reason is None else (parse_reason,)),
            pe_parse_succeeded=parse_reason is None,
        )
    finally:
        close = getattr(parsed, "close", None)
        if callable(close):
            close()
