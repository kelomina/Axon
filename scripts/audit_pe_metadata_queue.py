#!/usr/bin/env python3
"""Extract lightweight PE metadata for an error review queue."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Optional, Sequence


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for value in data:
        counts[value] += 1
    total = float(len(data))
    return -sum((count / total) * math.log2(count / total) for count in counts if count)


def _safe_unpack(fmt: str, data: bytes, offset: int):
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        return None
    return struct.unpack_from(fmt, data, offset)


def _section_name(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("ascii", errors="replace")


def _bucket_number(value: int | float | None, buckets: Sequence[int]) -> str:
    if value is None:
        return "<unknown>"
    for bucket in buckets:
        if value <= bucket:
            return f"<= {bucket}"
    return f"> {buckets[-1]}"


def parse_pe(path: Path) -> dict:
    result = {
        "exists": path.exists(),
        "file_size": None,
        "is_mz": False,
        "is_pe": False,
        "parse_error": "",
        "machine": "",
        "number_of_sections": None,
        "time_date_stamp": None,
        "characteristics": None,
        "optional_magic": "",
        "subsystem": "",
        "dll_characteristics": "",
        "entry_point": None,
        "image_base": None,
        "size_of_image": None,
        "section_names": "",
        "max_section_entropy": None,
        "avg_section_entropy": None,
        "high_entropy_section_count": 0,
        "executable_section_count": 0,
        "writable_executable_section_count": 0,
        "raw_to_virtual_ratio_max": None,
        "overlay_size": None,
        "overlay_entropy": None,
        "whole_file_entropy": None,
    }
    if not path.exists():
        result["parse_error"] = "source_missing"
        return result
    data = path.read_bytes()
    result["file_size"] = len(data)
    result["whole_file_entropy"] = _entropy(data[: min(len(data), 1024 * 1024)])
    if len(data) < 0x40 or data[:2] != b"MZ":
        result["parse_error"] = "not_mz"
        return result
    result["is_mz"] = True
    e_lfanew_tuple = _safe_unpack("<I", data, 0x3C)
    if e_lfanew_tuple is None:
        result["parse_error"] = "missing_lfanew"
        return result
    pe_offset = int(e_lfanew_tuple[0])
    if pe_offset + 24 > len(data) or data[pe_offset:pe_offset + 4] != b"PE\x00\x00":
        result["parse_error"] = "missing_pe_signature"
        return result
    result["is_pe"] = True
    file_header = _safe_unpack("<HHIIIHH", data, pe_offset + 4)
    if file_header is None:
        result["parse_error"] = "missing_file_header"
        return result
    machine, number_sections, timestamp, _ptr_sym, _num_sym, opt_size, characteristics = file_header
    result["machine"] = f"0x{int(machine):04x}"
    result["number_of_sections"] = int(number_sections)
    result["time_date_stamp"] = int(timestamp)
    result["characteristics"] = f"0x{int(characteristics):04x}"

    opt_offset = pe_offset + 24
    magic_tuple = _safe_unpack("<H", data, opt_offset)
    if magic_tuple is None:
        result["parse_error"] = "missing_optional_header"
        return result
    magic = int(magic_tuple[0])
    result["optional_magic"] = f"0x{magic:04x}"
    if magic == 0x10B:
        entry = _safe_unpack("<I", data, opt_offset + 16)
        image_base = _safe_unpack("<I", data, opt_offset + 28)
        size_image = _safe_unpack("<I", data, opt_offset + 56)
        subsystem = _safe_unpack("<H", data, opt_offset + 68)
        dll_chars = _safe_unpack("<H", data, opt_offset + 70)
    elif magic == 0x20B:
        entry = _safe_unpack("<I", data, opt_offset + 16)
        image_base = _safe_unpack("<Q", data, opt_offset + 24)
        size_image = _safe_unpack("<I", data, opt_offset + 56)
        subsystem = _safe_unpack("<H", data, opt_offset + 68)
        dll_chars = _safe_unpack("<H", data, opt_offset + 70)
    else:
        entry = image_base = size_image = subsystem = dll_chars = None
    if entry is not None:
        result["entry_point"] = int(entry[0])
    if image_base is not None:
        result["image_base"] = int(image_base[0])
    if size_image is not None:
        result["size_of_image"] = int(size_image[0])
    if subsystem is not None:
        result["subsystem"] = f"0x{int(subsystem[0]):04x}"
    if dll_chars is not None:
        result["dll_characteristics"] = f"0x{int(dll_chars[0]):04x}"

    section_offset = opt_offset + int(opt_size)
    sections = []
    max_raw_end = 0
    for index in range(int(number_sections)):
        offset = section_offset + index * 40
        if offset + 40 > len(data):
            break
        raw = data[offset:offset + 40]
        name = _section_name(raw[:8])
        fields = struct.unpack_from("<IIIIIIHHI", raw, 8)
        virtual_size, virtual_address, raw_size, raw_ptr, _reloc_ptr, _line_ptr, _reloc_count, _line_count, chars = fields
        raw_end = min(len(data), int(raw_ptr) + int(raw_size)) if raw_ptr and raw_size else 0
        max_raw_end = max(max_raw_end, raw_end)
        section_data = data[int(raw_ptr):raw_end] if raw_end > int(raw_ptr) else b""
        entropy = _entropy(section_data)
        executable = bool(int(chars) & 0x20000000)
        writable = bool(int(chars) & 0x80000000)
        sections.append(
            {
                "name": name,
                "virtual_size": int(virtual_size),
                "virtual_address": int(virtual_address),
                "raw_size": int(raw_size),
                "raw_ptr": int(raw_ptr),
                "chars": int(chars),
                "entropy": entropy,
                "executable": executable,
                "writable": writable,
                "ratio": float(raw_size / max(int(virtual_size), 1)),
            }
        )

    result["section_names"] = "|".join(section["name"] for section in sections)
    entropies = [section["entropy"] for section in sections]
    if entropies:
        result["max_section_entropy"] = max(entropies)
        result["avg_section_entropy"] = mean(entropies)
    result["high_entropy_section_count"] = sum(1 for section in sections if section["entropy"] >= 7.2)
    result["executable_section_count"] = sum(1 for section in sections if section["executable"])
    result["writable_executable_section_count"] = sum(
        1 for section in sections if section["executable"] and section["writable"]
    )
    ratios = [section["ratio"] for section in sections if section["virtual_size"] > 0]
    result["raw_to_virtual_ratio_max"] = max(ratios) if ratios else None
    overlay_size = max(0, len(data) - max_raw_end) if max_raw_end else 0
    result["overlay_size"] = overlay_size
    result["overlay_entropy"] = _entropy(data[max_raw_end:]) if overlay_size else 0.0
    return result


def _data_dir(path_text: str) -> str:
    parts = [part for part in re.split(r"[\\/]+", path_text) if part]
    lowered = [part.casefold() for part in parts]
    if "data" in lowered:
        index = lowered.index("data")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "<unknown>"


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_audit(queue_csv: Path, max_priority: int, output_csv: Path, output_json: Path) -> dict:
    with queue_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        queue_rows = list(csv.DictReader(handle))
    selected = [row for row in queue_rows if int(row.get("priority", 999)) <= max_priority]
    output_rows = []
    for row in selected:
        metadata = parse_pe(Path(row["source_path"]))
        out = {
            **{key: row.get(key, "") for key in [
                "priority",
                "reason",
                "error_type",
                "source_path",
                "source_sha256",
                "label",
                "prediction",
                "prob_malicious",
                "distance_to_threshold",
                "split",
                "sample_index",
                "extension",
                "month",
            ]},
            "data_dir": _data_dir(row.get("source_path", "")),
            **metadata,
        }
        output_rows.append(out)

    numeric_keys = [
        "file_size",
        "number_of_sections",
        "max_section_entropy",
        "avg_section_entropy",
        "high_entropy_section_count",
        "executable_section_count",
        "writable_executable_section_count",
        "overlay_size",
        "overlay_entropy",
        "whole_file_entropy",
    ]
    groups = defaultdict(list)
    for row in output_rows:
        groups[(row["error_type"], row["reason"])].append(row)

    grouped = []
    for (error_type, reason), rows in sorted(groups.items()):
        item = {
            "error_type": error_type,
            "reason": reason,
            "count": len(rows),
            "is_pe_count": sum(1 for row in rows if row["is_pe"]),
            "parse_error_counts": dict(sorted(Counter(str(row["parse_error"]) for row in rows).items())),
            "extension_counts": dict(sorted(Counter(str(row["extension"]) for row in rows).items())),
        }
        for key in numeric_keys:
            values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
            item[f"{key}_avg"] = mean(values) if values else None
            item[f"{key}_max"] = max(values) if values else None
        grouped.append(item)

    fieldnames = [
        "priority",
        "reason",
        "error_type",
        "source_path",
        "source_sha256",
        "label",
        "prediction",
        "prob_malicious",
        "distance_to_threshold",
        "split",
        "sample_index",
        "data_dir",
        "extension",
        "month",
        "exists",
        "file_size",
        "is_mz",
        "is_pe",
        "parse_error",
        "machine",
        "number_of_sections",
        "time_date_stamp",
        "characteristics",
        "optional_magic",
        "subsystem",
        "dll_characteristics",
        "entry_point",
        "image_base",
        "size_of_image",
        "section_names",
        "max_section_entropy",
        "avg_section_entropy",
        "high_entropy_section_count",
        "executable_section_count",
        "writable_executable_section_count",
        "raw_to_virtual_ratio_max",
        "overlay_size",
        "overlay_entropy",
        "whole_file_entropy",
    ]
    _write_csv(output_csv, output_rows, fieldnames)
    summary = {
        "schema": "axon_pe_metadata_queue_audit_v1",
        "queue_csv": str(queue_csv),
        "max_priority": max_priority,
        "selected_rows": len(output_rows),
        "is_pe_count": sum(1 for row in output_rows if row["is_pe"]),
        "parse_error_counts": dict(sorted(Counter(str(row["parse_error"]) for row in output_rows).items())),
        "file_size_buckets": dict(sorted(Counter(_bucket_number(row["file_size"], [65536, 262144, 1048576, 10485760]) for row in output_rows).items())),
        "high_entropy_section_count": sum(1 for row in output_rows if row["high_entropy_section_count"]),
        "writable_executable_section_sample_count": sum(1 for row in output_rows if row["writable_executable_section_count"]),
        "overlay_gt_1mb_count": sum(1 for row in output_rows if isinstance(row["overlay_size"], int) and row["overlay_size"] > 1024 * 1024),
        "grouped": grouped,
        "examples": output_rows[:20],
        "outputs": {
            "metadata_csv": str(output_csv),
            "summary_json": str(output_json),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract PE metadata for a review queue.")
    parser.add_argument("--queue-csv", type=Path, required=True)
    parser.add_argument("--max-priority", type=int, default=1)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build_audit(args.queue_csv, args.max_priority, args.output_csv, args.output_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
