#!/usr/bin/env python3
"""Build a read-only content-evidence package for Loop65 review rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence


IDENTITY_COLUMNS = [
    "source_path",
    "cache_path",
    "source_sha256",
    "sample_index",
    "split",
]

MODEL_CONTEXT_COLUMNS = [
    "loop57_final_prob",
    "loop57_base_prob",
    "loop57_candidate_prob",
    "loop57_gate_prob",
    "loop39_corrected_by_any_compared_model",
]

MANUAL_COLUMNS = [
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
]

OUTPUT_FIELDS = [
    "review_batch_rank",
    "review_category",
    "review_priority_rank",
    "loop57_error_type",
    "label",
    "review_tags",
    "content_evidence_fields",
    "source_exists",
    "source_size_bytes",
    "source_sha256_match",
    "source_sha256_actual",
    "file_entropy",
    "file_entropy_bytes",
    "file_entropy_truncated",
    "mz_signature",
    "pe_parse_status",
    "pe_machine",
    "pe_timestamp",
    "pe_characteristics",
    "pe_optional_magic",
    "pe_subsystem",
    "pe_dll_characteristics",
    "pe_number_of_sections",
    "pe_section_names",
    "pe_section_raw_size_total",
    "pe_section_raw_end_max",
    "pe_has_export_directory",
    "pe_export_directory_size",
    "pe_has_import_directory",
    "pe_import_directory_size",
    "pe_has_resource_directory",
    "pe_resource_directory_size",
    "pe_has_security_directory",
    "pe_security_directory_file_offset",
    "pe_security_directory_size",
    "overlay_size",
    "overlay_entropy",
    "overlay_entropy_bytes",
    "overlay_entropy_truncated",
    "overlay_after_security_size",
    "overlay_after_security_entropy",
    "overlay_after_security_entropy_bytes",
    "overlay_after_security_entropy_truncated",
    "cache_exists",
    "cache_size_bytes",
    "duplicate_manifest_sha_group",
    "manifest_duplicate_group_id",
    "manifest_duplicate_group_size",
    "manifest_duplicate_group_focus_rows",
    "objective_issue_count",
    "objective_issue_flags",
    "identity_columns_are_not_evidence",
    "model_score_columns_are_not_verdict_evidence",
    *IDENTITY_COLUMNS,
    *MODEL_CONTEXT_COLUMNS,
    *MANUAL_COLUMNS,
]

CONTENT_EVIDENCE_FIELDS = [
    "source_size_bytes",
    "source_sha256_match",
    "file_entropy",
    "mz_signature",
    "pe_parse_status",
    "pe_machine",
    "pe_characteristics",
    "pe_optional_magic",
    "pe_subsystem",
    "pe_dll_characteristics",
    "pe_number_of_sections",
    "pe_section_names",
    "pe_section_raw_size_total",
    "pe_has_export_directory",
    "pe_has_import_directory",
    "pe_has_resource_directory",
    "pe_has_security_directory",
    "overlay_size",
    "overlay_entropy",
    "overlay_after_security_size",
    "overlay_after_security_entropy",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _float_text(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def entropy_from_counts(counts: Sequence[int], total: int) -> Optional[float]:
    if total <= 0:
        return None
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def file_digest_and_entropy(path: Path, *, max_entropy_bytes: int, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    digest = hashlib.sha256()
    counts = [0] * 256
    entropy_bytes = 0
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
            size += len(chunk)
            remaining = max_entropy_bytes - entropy_bytes
            if remaining > 0:
                entropy_chunk = chunk[:remaining]
                for byte in entropy_chunk:
                    counts[byte] += 1
                entropy_bytes += len(entropy_chunk)
    return {
        "source_size_bytes": size,
        "source_sha256_actual": digest.hexdigest(),
        "file_entropy": entropy_from_counts(counts, entropy_bytes),
        "file_entropy_bytes": entropy_bytes,
        "file_entropy_truncated": size > entropy_bytes,
    }


def range_entropy(
    path: Path,
    *,
    start: int,
    size: int,
    max_entropy_bytes: int,
    chunk_size: int = 1024 * 1024,
) -> dict[str, Any]:
    if size <= 0 or start < 0:
        return {"entropy": None, "bytes": 0, "truncated": False}

    counts = [0] * 256
    entropy_bytes = 0
    target = min(size, max_entropy_bytes)
    with path.open("rb") as handle:
        handle.seek(start)
        while entropy_bytes < target:
            chunk = handle.read(min(chunk_size, target - entropy_bytes))
            if not chunk:
                break
            for byte in chunk:
                counts[byte] += 1
            entropy_bytes += len(chunk)
    return {
        "entropy": entropy_from_counts(counts, entropy_bytes),
        "bytes": entropy_bytes,
        "truncated": size > entropy_bytes,
    }


def _read_at(handle, offset: int, size: int) -> bytes:
    handle.seek(offset)
    return handle.read(size)


def parse_pe(path: Path, *, file_size: int, max_entropy_bytes: int) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "mz_signature": False,
        "pe_parse_status": "not_parsed",
    }
    try:
        with path.open("rb") as handle:
            dos = _read_at(handle, 0, 64)
            if len(dos) < 64 or dos[:2] != b"MZ":
                facts["pe_parse_status"] = "missing_mz"
                return facts
            facts["mz_signature"] = True
            pe_offset = struct.unpack_from("<I", dos, 0x3C)[0]
            if pe_offset <= 0 or pe_offset > max(file_size - 24, 0):
                facts["pe_parse_status"] = "bad_pe_offset"
                return facts

            coff = _read_at(handle, pe_offset, 24)
            if len(coff) < 24 or coff[:4] != b"PE\x00\x00":
                facts["pe_parse_status"] = "missing_pe_signature"
                return facts

            (
                machine,
                section_count,
                timestamp,
                _ptr_symbols,
                _num_symbols,
                optional_size,
                characteristics,
            ) = struct.unpack_from("<HHIIIHH", coff, 4)
            optional_offset = pe_offset + 24
            optional = _read_at(handle, optional_offset, optional_size)
            if len(optional) < min(optional_size, 2):
                facts["pe_parse_status"] = "truncated_optional_header"
                return facts
            optional_magic = struct.unpack_from("<H", optional, 0)[0]
            if optional_magic == 0x10B:
                data_dir_count_offset = 92
                data_dir_offset = 96
            elif optional_magic == 0x20B:
                data_dir_count_offset = 108
                data_dir_offset = 112
            else:
                data_dir_count_offset = 0
                data_dir_offset = 0

            subsystem = ""
            dll_characteristics = ""
            if len(optional) >= 72:
                subsystem = struct.unpack_from("<H", optional, 68)[0]
                dll_characteristics = struct.unpack_from("<H", optional, 70)[0]

            data_dirs: list[tuple[int, int]] = []
            if data_dir_offset and len(optional) >= data_dir_count_offset + 4:
                declared_dirs = struct.unpack_from("<I", optional, data_dir_count_offset)[0]
                dir_count = min(declared_dirs, 16, max((len(optional) - data_dir_offset) // 8, 0))
                for idx in range(dir_count):
                    rva_or_offset, size = struct.unpack_from("<II", optional, data_dir_offset + idx * 8)
                    data_dirs.append((rva_or_offset, size))

            section_offset = optional_offset + optional_size
            sections = []
            section_raw_end_max = 0
            section_raw_size_total = 0
            for idx in range(section_count):
                raw = _read_at(handle, section_offset + idx * 40, 40)
                if len(raw) < 40:
                    break
                name_raw = raw[:8].split(b"\x00", 1)[0]
                name = name_raw.decode("ascii", errors="replace")
                virtual_size, virtual_address, raw_size, raw_ptr = struct.unpack_from("<IIII", raw, 8)
                characteristics_section = struct.unpack_from("<I", raw, 36)[0]
                section_raw_size_total += raw_size
                if raw_ptr and raw_size:
                    section_raw_end_max = max(section_raw_end_max, raw_ptr + raw_size)
                sections.append(
                    {
                        "name": name,
                        "virtual_size": virtual_size,
                        "virtual_address": virtual_address,
                        "raw_size": raw_size,
                        "raw_ptr": raw_ptr,
                        "characteristics": characteristics_section,
                    }
                )

        def dir_size(index: int) -> int:
            if index >= len(data_dirs):
                return 0
            return int(data_dirs[index][1])

        def dir_offset(index: int) -> int:
            if index >= len(data_dirs):
                return 0
            return int(data_dirs[index][0])

        security_offset = dir_offset(4)
        security_size = dir_size(4)
        overlay_start = min(section_raw_end_max, file_size) if section_raw_end_max else file_size
        overlay_size = max(file_size - overlay_start, 0)
        security_end = security_offset + security_size if security_offset and security_size else overlay_start
        overlay_after_security_start = max(overlay_start, security_end)
        overlay_after_security_size = max(file_size - overlay_after_security_start, 0)
        overlay_entropy = range_entropy(
            path,
            start=overlay_start,
            size=overlay_size,
            max_entropy_bytes=max_entropy_bytes,
        )
        after_security_entropy = range_entropy(
            path,
            start=overlay_after_security_start,
            size=overlay_after_security_size,
            max_entropy_bytes=max_entropy_bytes,
        )

        facts.update(
            {
                "pe_parse_status": "ok",
                "pe_machine": machine,
                "pe_timestamp": timestamp,
                "pe_characteristics": characteristics,
                "pe_optional_magic": hex(optional_magic),
                "pe_subsystem": subsystem,
                "pe_dll_characteristics": dll_characteristics,
                "pe_number_of_sections": len(sections),
                "pe_section_names": "|".join(section["name"] for section in sections),
                "pe_section_raw_size_total": section_raw_size_total,
                "pe_section_raw_end_max": section_raw_end_max,
                "pe_has_export_directory": dir_size(0) > 0,
                "pe_export_directory_size": dir_size(0),
                "pe_has_import_directory": dir_size(1) > 0,
                "pe_import_directory_size": dir_size(1),
                "pe_has_resource_directory": dir_size(2) > 0,
                "pe_resource_directory_size": dir_size(2),
                "pe_has_security_directory": security_size > 0,
                "pe_security_directory_file_offset": security_offset,
                "pe_security_directory_size": security_size,
                "overlay_size": overlay_size,
                "overlay_entropy": overlay_entropy["entropy"],
                "overlay_entropy_bytes": overlay_entropy["bytes"],
                "overlay_entropy_truncated": overlay_entropy["truncated"],
                "overlay_after_security_size": overlay_after_security_size,
                "overlay_after_security_entropy": after_security_entropy["entropy"],
                "overlay_after_security_entropy_bytes": after_security_entropy["bytes"],
                "overlay_after_security_entropy_truncated": after_security_entropy["truncated"],
            }
        )
        return facts
    except OSError as exc:
        facts["pe_parse_status"] = f"io_error:{type(exc).__name__}"
        return facts
    except struct.error as exc:
        facts["pe_parse_status"] = f"parse_error:{type(exc).__name__}"
        return facts


def build_review_tags(facts: dict[str, Any]) -> list[str]:
    tags = []
    if not facts.get("source_exists"):
        tags.append("source_missing")
    if facts.get("source_sha256_match") is False:
        tags.append("source_sha256_mismatch")
    if facts.get("pe_parse_status") != "ok":
        tags.append(f"pe_{facts.get('pe_parse_status', 'unknown')}")
    if facts.get("pe_has_import_directory") is False:
        tags.append("no_import_directory")
    if facts.get("pe_has_resource_directory") is True:
        tags.append("has_resource_directory")
    if facts.get("pe_has_security_directory") is True:
        tags.append("has_security_directory")
    if _safe_int(facts.get("overlay_size"), 0) > 0:
        tags.append("overlay_present")
    if _safe_int(facts.get("overlay_after_security_size"), 0) > 0:
        tags.append("overlay_after_security_present")
    if (facts.get("file_entropy") or 0.0) >= 7.2:
        tags.append("high_file_entropy")
    if (facts.get("overlay_entropy") or 0.0) >= 7.2:
        tags.append("high_overlay_entropy")
    if _safe_int(facts.get("pe_number_of_sections"), 0) >= 8:
        tags.append("many_sections")
    return tags


def build_evidence_package(
    *,
    review_csv: Path,
    output_csv: Path,
    output_json: Path,
    max_entropy_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    rows = read_rows(review_csv)
    output_rows: list[dict[str, Any]] = []
    tag_counts: Counter[str] = Counter()
    parse_status_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    error_type_counts: Counter[str] = Counter()

    for row in rows:
        source_path_text = str(row.get("source_path", "")).strip()
        cache_path_text = str(row.get("cache_path", "")).strip()
        source_path = Path(source_path_text) if source_path_text else Path("__missing_source_path__")
        cache_path = Path(cache_path_text) if cache_path_text else Path("__missing_cache_path__")
        expected_sha = str(row.get("source_sha256", "")).strip().lower()
        facts: dict[str, Any] = {
            "source_exists": source_path.is_file(),
            "cache_exists": cache_path.is_file(),
            "cache_size_bytes": cache_path.stat().st_size if cache_path.is_file() else "",
        }
        if source_path.is_file():
            digest_facts = file_digest_and_entropy(source_path, max_entropy_bytes=max_entropy_bytes)
            facts.update(digest_facts)
            facts["source_sha256_match"] = bool(expected_sha and digest_facts["source_sha256_actual"].lower() == expected_sha)
            facts.update(
                parse_pe(
                    source_path,
                    file_size=int(digest_facts["source_size_bytes"]),
                    max_entropy_bytes=max_entropy_bytes,
                )
            )
        else:
            facts.update(
                {
                    "source_size_bytes": "",
                    "source_sha256_actual": "",
                    "source_sha256_match": False,
                    "file_entropy": None,
                    "file_entropy_bytes": 0,
                    "file_entropy_truncated": False,
                    "mz_signature": False,
                    "pe_parse_status": "source_missing",
                }
            )

        tags = build_review_tags(facts)
        tag_counts.update(tags)
        parse_status_counts[str(facts.get("pe_parse_status", ""))] += 1
        category_counts[str(row.get("review_category", ""))] += 1
        error_type_counts[str(row.get("loop57_error_type", ""))] += 1

        output = {field: "" for field in OUTPUT_FIELDS}
        for field in [
            "review_batch_rank",
            "review_category",
            "review_priority_rank",
            "loop57_error_type",
            "label",
            "duplicate_manifest_sha_group",
            "manifest_duplicate_group_id",
            "manifest_duplicate_group_size",
            "manifest_duplicate_group_focus_rows",
            "objective_issue_count",
            "objective_issue_flags",
            *IDENTITY_COLUMNS,
            *MODEL_CONTEXT_COLUMNS,
            *MANUAL_COLUMNS,
        ]:
            output[field] = row.get(field, "")

        output["review_tags"] = "|".join(tags)
        output["content_evidence_fields"] = "|".join(CONTENT_EVIDENCE_FIELDS)
        output["identity_columns_are_not_evidence"] = "true"
        output["model_score_columns_are_not_verdict_evidence"] = "true"

        for key, value in facts.items():
            if key in {
                "file_entropy",
                "overlay_entropy",
                "overlay_after_security_entropy",
            }:
                output[key] = _float_text(value)
            elif isinstance(value, bool):
                output[key] = _bool_text(value)
            else:
                output[key] = value
        output_rows.append(output)

    write_rows(output_csv, output_rows)
    manual_blank = all(not row.get(column) for row in output_rows for column in MANUAL_COLUMNS)
    summary = {
        "schema": "axon_loop86_review_evidence_package_v1",
        "protocol": (
            "read-only content evidence package for manual/external review; no training, no threshold tuning, "
            "no automatic relabeling, no split/cache mutation"
        ),
        "review_csv": str(review_csv),
        "rows": len(output_rows),
        "category_counts": dict(sorted(category_counts.items())),
        "error_type_counts": dict(sorted(error_type_counts.items())),
        "manual_fields_blank": manual_blank,
        "source_exists_count": sum(1 for row in output_rows if row.get("source_exists") == "true"),
        "source_sha256_mismatch_count": sum(1 for row in output_rows if row.get("source_sha256_match") == "false"),
        "cache_exists_count": sum(1 for row in output_rows if row.get("cache_exists") == "true"),
        "pe_parse_status_counts": dict(sorted(parse_status_counts.items())),
        "review_tag_counts": dict(sorted(tag_counts.items())),
        "identity_feature_policy": {
            "identity_columns": IDENTITY_COLUMNS,
            "model_context_columns": MODEL_CONTEXT_COLUMNS,
            "allowed_identity_uses": [
                "loading",
                "alignment",
                "cache audit",
                "duplicate detection",
                "manual/external review indexing",
            ],
            "forbidden_identity_uses": [
                "training features",
                "fusion features",
                "threshold shortcuts",
                "automatic relabel evidence",
                "production inference evidence",
                "replacement sampling by filename/path/directory similarity",
            ],
            "content_evidence_fields": CONTENT_EVIDENCE_FIELDS,
        },
        "decisions": {
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed_from_this_package": False,
            "test10k_allowed_from_this_package": False,
            "replacement_rule": (
                "Only an independent manual/external verdict can trigger quarantine plus fresh redraw from the "
                "locked-manifest original-label pool. This package alone does not change labels or counts."
            ),
        },
        "outputs": {
            "evidence_csv": str(output_csv),
            "summary_json": str(output_json),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop86 content evidence package from Loop65 review rows.")
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-entropy-bytes", type=int, default=64 * 1024 * 1024)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_evidence_package(
        review_csv=args.review_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        max_entropy_bytes=args.max_entropy_bytes,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
