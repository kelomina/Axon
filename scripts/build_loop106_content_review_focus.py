#!/usr/bin/env python3
"""Build identity-safe content-priority focus batches from Loop96 blinded review CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence


PROTOCOL = (
    "read-only Loop106 content focus builder; no model fitting, no threshold selection, "
    "no automatic verdict, no private-map read, no split/cache mutation"
)
IDENTITY_POLICY = (
    "The input must be the Loop96 blinded reviewer CSV. Filename, path, extension, directory, "
    "hash, source_sha256, sample_index, split, row order, and model scores are forbidden. "
    "Only blinded content fields may drive focus ranking."
)
FORBIDDEN_COLUMN_TOKENS = [
    "filename",
    "file_name",
    "source_path",
    "cache_path",
    "path",
    "directory",
    "extension",
    "source_sha",
    "sha256",
    "hash",
    "sample_index",
    "split",
    "row_order",
    "review_priority_rank",
    "review_batch_rank",
    "loop57",
    "loop39",
    "prob",
    "score",
    "prediction",
    "threshold",
]
BASE_OUTPUT_FIELDS = [
    "blind_review_id",
    "current_label",
    "loop106_focus_rank",
    "loop106_focus_score",
    "loop106_focus_bucket",
    "loop106_focus_reasons",
    "review_tags",
    "content_evidence_fields",
    "source_size_bytes",
    "file_entropy",
    "pe_parse_status",
    "pe_number_of_sections",
    "pe_section_names",
    "pe_has_import_directory",
    "pe_import_directory_size",
    "pe_has_resource_directory",
    "pe_resource_directory_size",
    "pe_has_security_directory",
    "pe_security_directory_size",
    "overlay_size",
    "overlay_entropy",
    "overlay_after_security_size",
    "overlay_after_security_entropy",
    "duplicate_manifest_sha_group",
    "manifest_duplicate_group_size",
    "objective_issue_count",
    "objective_issue_flags",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
]
CONTENT_EVIDENCE_DENYLIST = {
    "source_sha256_match",
    "source_sha256_actual",
    "source_sha256",
    "cache_path",
    "source_path",
    "sample_index",
}


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_text(value: object) -> str:
    return str(value or "").strip()


def parse_float(row: dict[str, str], field: str, default: float = 0.0) -> float:
    try:
        value = normalize_text(row.get(field))
        if not value:
            return default
        parsed = float(value)
        if not math.isfinite(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def parse_bool(row: dict[str, str], field: str) -> bool:
    return normalize_text(row.get(field)).casefold() in {"1", "true", "yes", "y"}


def parse_int(row: dict[str, str], field: str, default: int = 0) -> int:
    try:
        value = normalize_text(row.get(field))
        if not value:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def sanitize_content_evidence_fields(value: object) -> str:
    kept = []
    for item in normalize_text(value).split("|"):
        name = item.strip()
        if not name:
            continue
        if name.casefold() in CONTENT_EVIDENCE_DENYLIST:
            continue
        kept.append(name)
    return "|".join(kept)


def forbidden_columns(fieldnames: Sequence[str]) -> list[str]:
    found: list[str] = []
    for field in fieldnames:
        lowered = field.casefold()
        if field in {"file_entropy", "file_entropy_bytes", "file_entropy_truncated"}:
            continue
        if lowered.startswith("pe_") and "directory" in lowered:
            continue
        if any(token in lowered for token in FORBIDDEN_COLUMN_TOKENS):
            found.append(field)
    return sorted(set(found))


def score_content_row(row: dict[str, str]) -> tuple[float, list[str], str]:
    reasons: list[str] = []
    score = 0.0
    label = normalize_text(row.get("current_label"))
    overlay_size = parse_float(row, "overlay_size")
    overlay_entropy = parse_float(row, "overlay_entropy")
    overlay_after_security_size = parse_float(row, "overlay_after_security_size")
    overlay_after_security_entropy = parse_float(row, "overlay_after_security_entropy")
    file_entropy = parse_float(row, "file_entropy")
    section_count = parse_int(row, "pe_number_of_sections")
    import_size = parse_float(row, "pe_import_directory_size")
    resource_size = parse_float(row, "pe_resource_directory_size")
    security_size = parse_float(row, "pe_security_directory_size")
    source_size = parse_float(row, "source_size_bytes")
    duplicate_group_size = parse_int(row, "manifest_duplicate_group_size")
    objective_issue_count = parse_int(row, "objective_issue_count")
    tags = set(filter(None, normalize_text(row.get("review_tags")).split("|")))

    if objective_issue_count > 0:
        score += 100.0 + objective_issue_count * 5.0
        reasons.append("objective_issue_present")
    if duplicate_group_size > 1 or parse_bool(row, "duplicate_manifest_sha_group"):
        score += 45.0 + duplicate_group_size
        reasons.append("duplicate_content_group")
    if overlay_size > 0:
        score += min(30.0, math.log10(overlay_size + 1.0) * 6.0)
        reasons.append("overlay_present")
    if overlay_entropy >= 7.0:
        score += 18.0
        reasons.append("high_overlay_entropy")
    if overlay_after_security_size > 0:
        score += min(20.0, math.log10(overlay_after_security_size + 1.0) * 4.0)
        reasons.append("post_security_overlay_present")
    if overlay_after_security_entropy >= 7.0:
        score += 12.0
        reasons.append("high_post_security_overlay_entropy")
    if file_entropy >= 7.0:
        score += 15.0
        reasons.append("high_file_entropy")
    if section_count >= 8:
        score += min(14.0, float(section_count))
        reasons.append("many_sections")
    if parse_bool(row, "pe_has_security_directory"):
        score += 8.0
        reasons.append("has_security_directory")
    if security_size > 0 and overlay_after_security_size > 0:
        score += 10.0
        reasons.append("security_directory_with_extra_overlay")
    if parse_bool(row, "pe_has_import_directory") and import_size == 0:
        score += 8.0
        reasons.append("import_directory_declared_zero_size")
    if parse_bool(row, "pe_has_resource_directory") and resource_size > max(source_size * 0.15, 2_000_000.0):
        score += 8.0
        reasons.append("large_resource_directory")
    if "overlay_present" in tags and overlay_size <= 0:
        score += 5.0
        reasons.append("tag_overlay_size_mismatch")

    if label == "1":
        if overlay_size <= 0 and file_entropy < 5.0 and import_size > 0:
            score += 8.0
            reasons.append("malicious_label_benign_like_static_shape")
        bucket = "malicious_label_content_review"
    elif label == "0":
        if overlay_size > 0 or file_entropy >= 7.0 or section_count >= 8:
            score += 8.0
            reasons.append("benign_label_malware_like_static_shape")
        bucket = "benign_label_content_review"
    else:
        score += 20.0
        reasons.append("blank_or_unknown_current_label")
        bucket = "unknown_label_content_review"

    if not reasons:
        reasons.append("baseline_content_sample")
    return round(score, 6), reasons, bucket


def build_focus(
    *,
    blinded_csv: Path,
    output_csv: Path,
    output_json: Path,
    max_rows: int = 240,
    require_expected_rows: Optional[int] = 1868,
) -> dict[str, Any]:
    rows, fieldnames = read_csv_rows(blinded_csv)
    blockers: list[str] = []
    if require_expected_rows is not None and len(rows) != require_expected_rows:
        blockers.append("input_row_count_mismatch_expected")
    if "blind_review_id" not in fieldnames:
        blockers.append("missing_blind_review_id")
    if "current_label" not in fieldnames:
        blockers.append("missing_current_label")
    forbidden = forbidden_columns(fieldnames)
    if forbidden:
        blockers.append("input_contains_identity_or_model_columns")

    scored_rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for row in rows:
        score, reasons, bucket = score_content_row(row)
        label_counts[normalize_text(row.get("current_label")) or "blank"] += 1
        bucket_counts[bucket] += 1
        for reason in reasons:
            reason_counts[reason] += 1
        item = dict(row)
        item["content_evidence_fields"] = sanitize_content_evidence_fields(row.get("content_evidence_fields"))
        item["loop106_focus_score"] = f"{score:.6f}"
        item["loop106_focus_bucket"] = bucket
        item["loop106_focus_reasons"] = "|".join(reasons)
        scored_rows.append(item)

    selected = sorted(
        scored_rows,
        key=lambda row: (
            -parse_float(row, "loop106_focus_score"),
            normalize_text(row.get("blind_review_id")),
        ),
    )[:max_rows]
    for rank, row in enumerate(selected, start=1):
        row["loop106_focus_rank"] = str(rank)

    output_fieldnames = [field for field in BASE_OUTPUT_FIELDS if field in set(fieldnames) | {
        "loop106_focus_rank",
        "loop106_focus_score",
        "loop106_focus_bucket",
        "loop106_focus_reasons",
    }]
    write_csv_rows(output_csv, selected, output_fieldnames)

    summary = {
        "schema": "axon_loop106_content_review_focus_v1",
        "protocol": PROTOCOL,
        "identity_policy": IDENTITY_POLICY,
        "input_csv": str(blinded_csv),
        "rows": len(rows),
        "expected_rows": require_expected_rows,
        "selected_rows": len(selected),
        "max_rows": max_rows,
        "blockers": blockers,
        "forbidden_input_columns": forbidden,
        "label_counts": dict(sorted(label_counts.items())),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "selected_bucket_counts": dict(sorted(Counter(row["loop106_focus_bucket"] for row in selected).items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "top_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in selected
                    for reason in normalize_text(row.get("loop106_focus_reasons")).split("|")
                    if reason
                ).items()
            )
        ),
        "decisions": {
            "automatic_verdict_allowed": False,
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "ready_for_independent_content_review": not blockers,
        },
        "outputs": {
            "focus_csv": str(output_csv),
            "summary_json": str(output_json),
        },
        "notes": [
            "Ranking uses blinded content fields only. It is a review prioritization aid, not a verdict.",
            "Reviewers must still cite content or external evidence in manual_verdict_note before Loop87 can accept an actionable row.",
            "The focus CSV intentionally omits private alignment fields, hashes, paths, model scores, split, and row order.",
        ],
    }
    write_json(output_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop106 identity-safe content review focus batch.")
    parser.add_argument("--blinded-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=240)
    parser.add_argument("--expected-rows", type=int, default=1868)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_focus(
        blinded_csv=args.blinded_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        max_rows=args.max_rows,
        require_expected_rows=args.expected_rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
