#!/usr/bin/env python3
"""Import external Loop106 focus annotations through a strict blinded gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from apply_manual_review_verdicts import normalize_text  # noqa: E402
from merge_loop106_focus_annotations import (  # noqa: E402
    FORBIDDEN_COLUMN_TOKENS,
    MANUAL_FIELDS,
    forbidden_columns,
)
from preflight_loop106_focus_annotations import preflight_focus_annotations  # noqa: E402


PROTOCOL = (
    "strict Loop111 external focus annotation import; blind_review_id keyed only; no private-map read, "
    "no unblind, no model fitting, no threshold selection, no automatic verdict, no split/cache mutation"
)
IDENTITY_POLICY = (
    "External annotations may update only blind_review_id plus manual_label_verdict, manual_verdict_note, "
    "and recommended_action. Filename/path/directory/extension/hash/source_sha256/sample_index/split/"
    "row order/model score fields are forbidden as import columns and are not verdict evidence."
)
ALLOWED_EXTERNAL_FIELDS = ["blind_review_id", *MANUAL_FIELDS]
OUTPUT_EXTRAS = [
    "loop111_import_status",
    "loop111_import_reason",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str], list[str]]:
    issues: list[str] = []
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows: list[dict[str, str]] = []
        for row in reader:
            if None in row:
                issues.append("csv_row_has_extra_unheaded_values")
            rows.append({str(key): str(value or "") for key, value in row.items() if key is not None})
    duplicate_fields = [field for field, count in Counter(fieldnames).items() if count > 1]
    if duplicate_fields:
        issues.append("csv_header_has_duplicate_fields")
    return rows, fieldnames, issues


def read_jsonl_rows(path: Path) -> tuple[list[dict[str, str]], list[str], list[str]]:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    issues: list[str] = []
    with resolve_path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                issues.append(f"jsonl_invalid_json_line:{line_number}")
                continue
            if not isinstance(payload, dict):
                issues.append(f"jsonl_row_not_object:{line_number}")
                continue
            row = {str(key): "" if value is None else str(value) for key, value in payload.items()}
            rows.append(row)
            for field in row:
                if field not in fieldnames:
                    fieldnames.append(field)
    return rows, fieldnames, issues


def read_external_rows(path: Path, input_format: str) -> tuple[list[dict[str, str]], list[str], list[str], str]:
    resolved = resolve_path(path)
    fmt = input_format
    if fmt == "auto":
        fmt = "jsonl" if resolved.suffix.casefold() in {".jsonl", ".ndjson"} else "csv"
    if fmt == "jsonl":
        rows, fieldnames, issues = read_jsonl_rows(path)
    elif fmt == "csv":
        rows, fieldnames, issues = read_csv_rows(path)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")
    return rows, fieldnames, issues, fmt


def write_csv_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_output_fieldnames(source_fieldnames: Sequence[str]) -> list[str]:
    fieldnames = list(source_fieldnames)
    for field in OUTPUT_EXTRAS:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames


def field_has_forbidden_token(field: str) -> bool:
    lowered = field.casefold()
    return any(token in lowered for token in FORBIDDEN_COLUMN_TOKENS)


def external_forbidden_fields(fieldnames: Sequence[str]) -> list[str]:
    allowed = set(ALLOWED_EXTERNAL_FIELDS)
    return sorted({field for field in fieldnames if field not in allowed})


def external_identity_or_model_fields(fieldnames: Sequence[str]) -> list[str]:
    return sorted({field for field in fieldnames if field not in ALLOWED_EXTERNAL_FIELDS and field_has_forbidden_token(field)})


def row_has_manual_content(row: dict[str, str]) -> bool:
    return any(normalize_text(row.get(field)) for field in MANUAL_FIELDS)


def row_manual_values(row: dict[str, str]) -> dict[str, str]:
    return {field: normalize_text(row.get(field)) for field in MANUAL_FIELDS}


def rows_by_blind_id(rows: Sequence[dict[str, str]], *, source_name: str) -> tuple[dict[str, dict[str, str]], list[str], int, int]:
    by_id: dict[str, dict[str, str]] = {}
    counts: Counter[str] = Counter()
    missing = 0
    for row in rows:
        blind_id = normalize_text(row.get("blind_review_id"))
        if not blind_id:
            missing += 1
            continue
        counts[blind_id] += 1
        by_id[blind_id] = row
    issues = [f"{source_name}_duplicate_blind_review_id:{blind_id}" for blind_id, count in sorted(counts.items()) if count > 1]
    duplicate_count = int(sum(count - 1 for count in counts.values() if count > 1))
    return by_id, issues, missing, duplicate_count


def _same_manual_values(left: dict[str, str], right: dict[str, str]) -> bool:
    return row_manual_values(left) == row_manual_values(right)


def import_focus_external_annotations(
    *,
    focus_csv: Path,
    external_annotations: Path,
    output_csv: Path,
    output_json: Path,
    preflight_output_csv: Path,
    preflight_output_json: Path,
    expected_focus_rows: Optional[int] = 240,
    input_format: str = "auto",
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    focus_rows, focus_fieldnames, focus_read_issues = read_csv_rows(focus_csv)
    external_rows, external_fieldnames, external_read_issues, resolved_format = read_external_rows(
        external_annotations,
        input_format,
    )

    blockers: list[str] = []
    warnings: list[str] = []
    if focus_read_issues:
        blockers.append("focus_csv_read_issues")
    if external_read_issues:
        blockers.append("external_annotation_read_issues")
    if expected_focus_rows is not None and len(focus_rows) != expected_focus_rows:
        blockers.append("unexpected_focus_row_count")
    missing_focus_columns = [field for field in ["blind_review_id", *MANUAL_FIELDS] if field not in focus_fieldnames]
    if missing_focus_columns:
        blockers.append("focus_missing_required_columns")
    forbidden_focus_columns = forbidden_columns(focus_fieldnames)
    if forbidden_focus_columns:
        blockers.append("focus_contains_identity_or_model_columns")

    missing_external_fields = [field for field in ALLOWED_EXTERNAL_FIELDS if field not in external_fieldnames]
    if missing_external_fields:
        blockers.append("external_missing_required_fields")
    extra_external_fields = external_forbidden_fields(external_fieldnames)
    if extra_external_fields:
        blockers.append("external_contains_unapproved_fields")
    identity_external_fields = external_identity_or_model_fields(external_fieldnames)
    if identity_external_fields:
        blockers.append("external_contains_identity_or_model_fields")

    focus_by_id, focus_id_issues, focus_missing_id_rows, focus_duplicate_id_rows = rows_by_blind_id(
        focus_rows,
        source_name="focus",
    )
    external_by_id, external_id_issues, external_missing_id_rows, external_duplicate_id_rows = rows_by_blind_id(
        external_rows,
        source_name="external",
    )
    if focus_id_issues:
        blockers.append("focus_duplicate_blind_review_id")
    if external_id_issues:
        blockers.append("external_duplicate_blind_review_id")
    if focus_missing_id_rows:
        blockers.append("focus_missing_blind_review_id")
    if external_missing_id_rows:
        blockers.append("external_missing_blind_review_id")
    unknown_external_ids = sorted(set(external_by_id) - set(focus_by_id))
    if unknown_external_ids:
        blockers.append("external_ids_missing_from_focus_csv")

    blank_external_annotation_rows = sum(1 for row in external_rows if not row_has_manual_content(row))
    if blank_external_annotation_rows:
        blockers.append("external_annotation_rows_have_blank_manual_fields")

    overwrite_conflict_ids: list[str] = []
    for blind_id, external_row in sorted(external_by_id.items()):
        focus_row = focus_by_id.get(blind_id)
        if focus_row is None or not row_has_manual_content(focus_row):
            continue
        if not _same_manual_values(focus_row, external_row) and not allow_overwrite:
            overwrite_conflict_ids.append(blind_id)
    if overwrite_conflict_ids:
        blockers.append("external_would_overwrite_existing_focus_annotations")

    output_rows: list[dict[str, Any]] = []
    imported_rows = 0
    if blockers:
        for row in focus_rows:
            item = dict(row)
            item["loop111_import_status"] = "not_imported"
            item["loop111_import_reason"] = "blocked_before_import"
            output_rows.append(item)
        preflight_summary: Optional[dict[str, Any]] = None
    else:
        for row in focus_rows:
            item = dict(row)
            blind_id = normalize_text(row.get("blind_review_id"))
            external_row = external_by_id.get(blind_id)
            if external_row is None:
                item["loop111_import_status"] = "not_targeted"
                item["loop111_import_reason"] = "no_external_annotation"
            else:
                for field in MANUAL_FIELDS:
                    item[field] = normalize_text(external_row.get(field))
                item["loop111_import_status"] = "imported"
                item["loop111_import_reason"] = "external_annotation_applied"
                imported_rows += 1
            output_rows.append(item)
        preflight_summary = None

    write_csv_rows(output_csv, output_rows, build_output_fieldnames(focus_fieldnames))

    if not blockers:
        preflight_summary = preflight_focus_annotations(
            focus_annotations_csv=output_csv,
            output_csv=preflight_output_csv,
            output_json=preflight_output_json,
            expected_rows=expected_focus_rows,
        )
        if not preflight_summary.get("ready_for_focus_merge"):
            blockers.append("post_import_focus_preflight_not_ready")

    if blockers:
        decision = "blocked_invalid_external_annotations"
        ready_for_loop110 = False
    elif imported_rows == 0:
        decision = "ready_noop_no_external_annotations"
        ready_for_loop110 = True
    else:
        decision = "ready_for_focus_verdict_pipeline"
        ready_for_loop110 = True

    if preflight_summary and preflight_summary.get("warnings"):
        warnings.extend(preflight_summary.get("warnings", []))

    summary = {
        "schema": "axon_loop111_focus_external_annotation_import_v1",
        "protocol": PROTOCOL,
        "identity_policy": IDENTITY_POLICY,
        "inputs": {
            "focus_csv": str(focus_csv),
            "external_annotations": str(external_annotations),
            "input_format": resolved_format,
            "expected_focus_rows": expected_focus_rows,
            "allow_overwrite": allow_overwrite,
        },
        "decision": decision,
        "ready_for_loop110_focus_pipeline": ready_for_loop110,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "focus": {
            "rows": len(focus_rows),
            "field_count": len(focus_fieldnames),
            "missing_required_columns": missing_focus_columns,
            "forbidden_columns": forbidden_focus_columns,
            "read_issues": focus_read_issues,
            "missing_blind_review_id_rows": focus_missing_id_rows,
            "duplicate_blind_review_id_rows": focus_duplicate_id_rows,
        },
        "external": {
            "rows": len(external_rows),
            "fieldnames": external_fieldnames,
            "allowed_fields": ALLOWED_EXTERNAL_FIELDS,
            "missing_required_fields": missing_external_fields,
            "unapproved_fields": extra_external_fields,
            "identity_or_model_fields": identity_external_fields,
            "read_issues": external_read_issues,
            "missing_blind_review_id_rows": external_missing_id_rows,
            "duplicate_blind_review_id_rows": external_duplicate_id_rows,
            "unknown_blind_review_id_count": len(unknown_external_ids),
            "unknown_blind_review_id_examples": unknown_external_ids[:20],
            "blank_manual_field_rows": blank_external_annotation_rows,
            "overwrite_conflict_count": len(overwrite_conflict_ids),
            "overwrite_conflict_examples": overwrite_conflict_ids[:20],
        },
        "counts": {
            "imported_rows": imported_rows,
            "post_preflight_annotated_rows": None if preflight_summary is None else preflight_summary.get("annotated_rows"),
            "post_preflight_actionable_rows": None if preflight_summary is None else preflight_summary.get("actionable_rows"),
            "post_preflight_invalid_rows": None if preflight_summary is None else preflight_summary.get("invalid_rows"),
        },
        "post_import_preflight": preflight_summary,
        "decisions": {
            "ready_for_loop110_focus_pipeline": ready_for_loop110,
            "automatic_verdict_allowed": False,
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "full_test_allowed": False,
            "next_allowed_step": (
                "run Loop110 focus verdict pipeline"
                if ready_for_loop110 and imported_rows > 0
                else "collect independent content/external focus annotations"
                if ready_for_loop110
                else "fix external annotation import blockers"
            ),
        },
        "outputs": {
            "annotated_focus_csv": str(output_csv),
            "summary_json": str(output_json),
            "preflight_validated_csv": str(preflight_output_csv),
            "preflight_summary_json": str(preflight_output_json),
        },
        "notes": [
            "The importer never reads the private map and never sees source_path/source_sha256/sample_index/split.",
            "External reviewers must provide independent content or external evidence in manual_verdict_note.",
            "Loop109 preflight remains the post-import quality gate before Loop110 merge/unblind/import.",
            "This script does not train, evaluate, redraw, or mutate any split/cache file.",
        ],
    }
    write_json(output_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import external Loop106 focus annotations through a strict gate.")
    parser.add_argument("--focus-csv", type=Path, required=True)
    parser.add_argument("--external-annotations", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--preflight-output-csv", type=Path, required=True)
    parser.add_argument("--preflight-output-json", type=Path, required=True)
    parser.add_argument("--expected-focus-rows", type=int, default=240)
    parser.add_argument("--input-format", choices=["auto", "csv", "jsonl"], default="auto")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = import_focus_external_annotations(
        focus_csv=args.focus_csv,
        external_annotations=args.external_annotations,
        output_csv=args.output_csv,
        output_json=args.output_json,
        preflight_output_csv=args.preflight_output_csv,
        preflight_output_json=args.preflight_output_json,
        expected_focus_rows=args.expected_focus_rows,
        input_format=args.input_format,
        allow_overwrite=bool(args.allow_overwrite),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["ready_for_loop110_focus_pipeline"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
