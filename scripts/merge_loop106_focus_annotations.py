#!/usr/bin/env python3
"""Merge Loop106 focus annotations back into the full Loop96 blinded CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence


PROTOCOL = (
    "read-only focus annotation merger; no private-map read, no model fitting, no threshold selection, "
    "no automatic verdict, no split/cache mutation"
)
IDENTITY_POLICY = (
    "Focus annotations are keyed only by blind_review_id. Filename, path, extension, directory, hash, "
    "source_sha256, sample_index, split, row order, and model scores are forbidden in the focus CSV."
)
MANUAL_FIELDS = ["manual_label_verdict", "manual_verdict_note", "recommended_action"]
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


def forbidden_columns(fieldnames: Sequence[str]) -> list[str]:
    allowed_focus_fields = {
        "blind_review_id",
        "current_label",
        "loop106_focus_rank",
        "loop106_focus_score",
        "loop106_focus_bucket",
        "loop106_focus_reasons",
    }
    found: list[str] = []
    for field in fieldnames:
        lowered = field.casefold()
        if field in allowed_focus_fields or field in MANUAL_FIELDS:
            continue
        if field in {"file_entropy", "file_entropy_bytes", "file_entropy_truncated"}:
            continue
        if lowered.startswith("pe_") and "directory" in lowered:
            continue
        if any(token in lowered for token in FORBIDDEN_COLUMN_TOKENS):
            found.append(field)
    return sorted(set(found))


def _rows_by_blind_id(rows: Sequence[dict[str, str]], *, source_name: str) -> tuple[dict[str, dict[str, str]], list[str]]:
    by_id: dict[str, dict[str, str]] = {}
    issues: list[str] = []
    counts: Counter[str] = Counter()
    for row in rows:
        blind_id = normalize_text(row.get("blind_review_id"))
        if not blind_id:
            issues.append(f"{source_name}_missing_blind_review_id")
            continue
        counts[blind_id] += 1
        by_id[blind_id] = row
    for blind_id, count in sorted(counts.items()):
        if count > 1:
            issues.append(f"{source_name}_duplicate_blind_review_id:{blind_id}")
    return by_id, issues


def _manual_has_content(row: dict[str, str]) -> bool:
    return any(normalize_text(row.get(field)) for field in MANUAL_FIELDS)


def merge_focus_annotations(
    *,
    full_blinded_csv: Path,
    focus_annotations_csv: Path,
    output_csv: Path,
    output_json: Path,
    expected_full_rows: Optional[int] = 1868,
    expected_focus_rows: Optional[int] = None,
) -> dict[str, Any]:
    full_rows, full_fieldnames = read_csv_rows(full_blinded_csv)
    focus_rows, focus_fieldnames = read_csv_rows(focus_annotations_csv)
    blockers: list[str] = []
    warnings: list[str] = []

    if expected_full_rows is not None and len(full_rows) != expected_full_rows:
        blockers.append("full_blinded_row_count_mismatch_expected")
    if expected_focus_rows is not None and len(focus_rows) != expected_focus_rows:
        blockers.append("focus_row_count_mismatch_expected")
    missing_full_manual = [field for field in MANUAL_FIELDS if field not in full_fieldnames]
    missing_focus_manual = [field for field in MANUAL_FIELDS if field not in focus_fieldnames]
    if "blind_review_id" not in full_fieldnames:
        blockers.append("full_missing_blind_review_id")
    if "blind_review_id" not in focus_fieldnames:
        blockers.append("focus_missing_blind_review_id")
    if missing_full_manual:
        blockers.append("full_missing_manual_fields")
    if missing_focus_manual:
        blockers.append("focus_missing_manual_fields")

    forbidden_focus = forbidden_columns(focus_fieldnames)
    if forbidden_focus:
        blockers.append("focus_contains_identity_or_model_columns")

    full_by_id, full_issues = _rows_by_blind_id(full_rows, source_name="full")
    focus_by_id, focus_issues = _rows_by_blind_id(focus_rows, source_name="focus")
    blockers.extend(full_issues)
    blockers.extend(focus_issues)
    unknown_focus_ids = sorted(set(focus_by_id) - set(full_by_id))
    if unknown_focus_ids:
        blockers.append("focus_ids_missing_from_full_blinded_csv")

    annotated_focus_rows = sum(1 for row in focus_rows if _manual_has_content(row))
    output_rows: list[dict[str, Any]] = []
    merged_rows = 0
    for full_row in full_rows:
        item = dict(full_row)
        blind_id = normalize_text(full_row.get("blind_review_id"))
        focus_row = focus_by_id.get(blind_id)
        if focus_row is not None:
            for field in MANUAL_FIELDS:
                value = normalize_text(focus_row.get(field))
                item[field] = value
            if _manual_has_content(focus_row):
                merged_rows += 1
        output_rows.append(item)

    existing_full_manual_rows = sum(1 for row in full_rows if _manual_has_content(row))
    if existing_full_manual_rows:
        warnings.append("full_blinded_csv_already_has_manual_annotations")

    write_csv_rows(output_csv, output_rows, full_fieldnames)
    summary = {
        "schema": "axon_loop107_focus_annotation_merge_v1",
        "protocol": PROTOCOL,
        "identity_policy": IDENTITY_POLICY,
        "inputs": {
            "full_blinded_csv": str(full_blinded_csv),
            "focus_annotations_csv": str(focus_annotations_csv),
        },
        "rows": {
            "full_blinded": len(full_rows),
            "expected_full": expected_full_rows,
            "focus": len(focus_rows),
            "expected_focus": expected_focus_rows,
            "annotated_focus_rows": annotated_focus_rows,
            "merged_annotated_rows": merged_rows,
        },
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "forbidden_focus_columns": forbidden_focus,
        "unknown_focus_id_count": len(unknown_focus_ids),
        "unknown_focus_id_examples": unknown_focus_ids[:20],
        "decisions": {
            "ready_for_loop96_unblind": not blockers,
            "automatic_verdict_allowed": False,
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
        },
        "outputs": {
            "merged_blinded_csv": str(output_csv),
            "summary_json": str(output_json),
        },
        "notes": [
            "Only manual_label_verdict, manual_verdict_note, and recommended_action are merged from focus rows.",
            "Unannotated full rows remain blank and will be treated as no_decision by Loop87 after unblinding.",
            "This script does not read the private map. Loop96 unblind and Loop87 remain the strict downstream gates.",
        ],
    }
    write_json(output_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge Loop106 focus annotations into the full Loop96 blinded CSV.")
    parser.add_argument("--full-blinded-csv", type=Path, required=True)
    parser.add_argument("--focus-annotations-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-full-rows", type=int, default=1868)
    parser.add_argument("--expected-focus-rows", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = merge_focus_annotations(
        full_blinded_csv=args.full_blinded_csv,
        focus_annotations_csv=args.focus_annotations_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        expected_full_rows=args.expected_full_rows,
        expected_focus_rows=args.expected_focus_rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
