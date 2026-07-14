#!/usr/bin/env python3
"""Export an identity-safe external annotation package for Loop156.

The package contains a reviewer context CSV and a header-only annotation
template. It never reads the private map, never unblinds identities, and never
turns review priority, model scores, paths, hashes, or row order into evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANUAL_FIELDS = ["manual_label_verdict", "manual_verdict_note", "recommended_action"]
ANNOTATION_FIELDS = ["review_focus_id", *MANUAL_FIELDS]
CONTEXT_FIELDS = [
    "review_focus_id",
    "priority_band",
    "current_label",
    "error_type",
    "review_lane",
    "content_signal_count",
    "content_tags",
    "content_is_dll",
    "content_export_count_log",
    "content_dir_export_log_size",
    "content_dir_security_log_size",
    "content_overlay_log_size",
    "content_resource_entry_count_log",
    "content_resource_type_count_log",
    "content_dir_resource_size_ratio",
    "content_dir_resource_log_size",
    "content_overlay_entropy",
    "content_import_api_count_log",
    "content_avg_imports_per_dll",
    "content_image_base_log",
    "v2_resource_data_entry_count_log",
    "v2_resource_type_icon_count_log",
    "v2_resource_type_version_count_log",
    "v2_resource_type_manifest_count_log",
    "v2_resource_type_dialog_count_log",
    "v2_last_section_entropy",
    "v2_section_max_virtual_raw_ratio_log",
    "v2_api_file_mutation_ratio",
    "v2_import_dll_version_api_ratio",
    "string_benign_vendor_count_log",
    "string_version_resource_count_log",
    "string_script_exec_count_log",
    "string_script_exec_present",
]
FORBIDDEN_HEADER_TOKENS = [
    "source",
    "sha",
    "hash",
    "cache",
    "path",
    "sample_index",
    "filename",
    "directory",
    "extension",
    "score",
    "prob",
    "threshold",
    "prediction",
    "neighbor",
    "similarity",
    "rank",
    "row_order",
]
FORBIDDEN_VALUE_TOKENS = [
    "source_path",
    "cache_path",
    "source_sha",
    "sha256",
    "hash",
    "sample_index",
    "filename",
    "file name",
    "source directory",
    "file directory",
    "folder",
    "extension",
    "row order",
    "model score",
    "probability",
    "prob_malicious",
    "prediction",
    "threshold",
    "neighbor similarity",
]
PROTOCOL = (
    "Loop157 external annotation package for all Loop151 current-best Val errors; context CSV plus "
    "annotation template only; no private-map read, no unblind, no automatic verdict, no training, "
    "no threshold selection, and no split/cache mutation"
)
IDENTITY_POLICY = (
    "External reviewers receive only review_focus_id and content-derived/context fields. source_path, cache_path, "
    "source_sha256, sample_index, filename, directory, extension, split, row order, model score, probability, "
    "prediction, threshold, neighbor labels, and similarity fields are withheld and are not verdict evidence."
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize(value: object) -> str:
    return str(value or "").strip()


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


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


def _tokenized(field: str) -> set[str]:
    return {part for part in field.casefold().replace("-", "_").split("_") if part}


def header_violations(fieldnames: Sequence[str]) -> list[str]:
    violations: list[str] = []
    for field in fieldnames:
        parts = _tokenized(field)
        if any(token in parts for token in FORBIDDEN_HEADER_TOKENS):
            violations.append(field)
    return sorted(set(violations))


def value_violations(rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        review_id = normalize(row.get("review_focus_id"))
        for field in fieldnames:
            value = normalize(row.get(field)).casefold()
            if not value:
                continue
            hits = [token for token in FORBIDDEN_VALUE_TOKENS if token in value]
            if hits:
                violations.append(
                    {
                        "row_number": row_number,
                        "review_focus_id": review_id,
                        "field": field,
                        "tokens": sorted(set(hits)),
                    }
                )
    return violations


def review_focus_id_counts(rows: Sequence[dict[str, str]]) -> tuple[int, int]:
    seen: set[str] = set()
    missing = 0
    duplicate = 0
    for row in rows:
        review_id = normalize(row.get("review_focus_id"))
        if not review_id:
            missing += 1
            continue
        if review_id in seen:
            duplicate += 1
        seen.add(review_id)
    return missing, duplicate


def build_context_rows(rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> tuple[list[dict[str, Any]], list[str]]:
    context_fields = [field for field in CONTEXT_FIELDS if field in fieldnames]
    context_rows = [{field: normalize(row.get(field)) for field in context_fields} for row in rows]
    return context_rows, context_fields


def export_loop157_package(
    *,
    review_csv: Path,
    context_csv: Path,
    annotation_template_csv: Path,
    reviewer_guide_json: Path,
    output_json: Path,
    expected_rows: Optional[int] = 162,
) -> dict[str, Any]:
    rows, fieldnames = read_csv_rows(review_csv)
    blockers: list[str] = []
    warnings: list[str] = []

    required_columns = ["review_focus_id", "current_label", "error_type", *MANUAL_FIELDS]
    missing_required_columns = [field for field in required_columns if field not in fieldnames]
    if missing_required_columns:
        blockers.append("review_csv_missing_required_columns")
    if expected_rows is not None and len(rows) != expected_rows:
        blockers.append("unexpected_review_row_count")

    input_forbidden = header_violations(fieldnames)
    # The source review CSV contains focus_rank and manual fields by design; they
    # must not be exported as reviewer context, but they are not blockers here.
    tolerated_input_columns = {"focus_rank", *MANUAL_FIELDS}
    input_forbidden = [field for field in input_forbidden if field not in tolerated_input_columns]
    if input_forbidden:
        blockers.append("review_csv_contains_identity_or_model_columns")

    missing_review_id_rows, duplicate_review_id_rows = review_focus_id_counts(rows)
    if missing_review_id_rows:
        blockers.append("missing_review_focus_id")
    if duplicate_review_id_rows:
        blockers.append("duplicate_review_focus_id")

    context_rows, context_fields = build_context_rows(rows, fieldnames)
    missing_context_fields = [field for field in CONTEXT_FIELDS if field not in fieldnames]
    if missing_context_fields:
        warnings.append("some_context_fields_missing_from_review_csv")

    context_header_violations = header_violations(context_fields)
    if context_header_violations:
        blockers.append("context_output_header_contains_forbidden_fields")
    context_value_violations = value_violations(context_rows, context_fields)
    if context_value_violations:
        blockers.append("context_output_values_reference_identity_or_model_terms")

    write_csv_rows(context_csv, context_rows, context_fields)
    write_csv_rows(annotation_template_csv, [], ANNOTATION_FIELDS)
    guide = {
        "schema": "axon_loop157_external_annotation_guide_v1",
        "protocol": PROTOCOL,
        "identity_policy": IDENTITY_POLICY,
        "annotation_fields": ANNOTATION_FIELDS,
        "allowed_manual_label_verdict_examples": [
            "label_correct",
            "label_wrong",
            "feature_broken",
            "out_of_scope",
            "uncertain",
            "needs_more_evidence",
        ],
        "allowed_recommended_action_examples": [
            "",
            "keep_label",
            "model_blindspot",
            "keep_sample",
            "needs_more_evidence",
            "quarantine_for_more_evidence",
            "replace_with_fresh_same_label_candidate",
        ],
        "required_note_rule": "Actionable verdicts must cite content or external evidence, not identity fields or model scores.",
        "submission_rule": "Submit only rows with filled manual fields; blank rows should be omitted from the returned annotation file.",
        "forbidden_evidence": [
            "filename/path/directory/extension",
            "hash/source_sha256/sample_index/split/row order",
            "model score/probability/prediction/threshold",
            "neighbor labels/similarity or row order",
        ],
    }
    write_json(reviewer_guide_json, guide)

    summary = {
        "schema": "axon_loop157_current_best_val_external_annotation_package_v1",
        "protocol": PROTOCOL,
        "identity_policy": IDENTITY_POLICY,
        "inputs": {
            "review_csv": str(resolve_path(review_csv)),
            "expected_rows": expected_rows,
        },
        "decision": "blocked_invalid_external_package" if blockers else "ready_for_external_content_annotation",
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "rows": len(rows),
        "label_counts": dict(sorted(Counter(normalize(row.get("current_label")) or "blank" for row in rows).items())),
        "error_counts": dict(sorted(Counter(normalize(row.get("error_type")) or "blank" for row in rows).items())),
        "field_audit": {
            "review_field_count": len(fieldnames),
            "context_field_count": len(context_fields),
            "missing_required_columns": missing_required_columns,
            "missing_context_fields": missing_context_fields,
            "forbidden_input_columns": input_forbidden,
            "context_header_violations": context_header_violations,
            "context_value_violation_count": len(context_value_violations),
            "context_value_violation_examples": context_value_violations[:20],
            "missing_review_focus_id_rows": missing_review_id_rows,
            "duplicate_review_focus_id_rows": duplicate_review_id_rows,
        },
        "decisions": {
            "ready_for_external_content_annotation": not blockers,
            "automatic_verdict_allowed": False,
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "full_test_allowed": False,
            "next_allowed_step": (
                "fill annotation template with independent content/external verdicts, then run Loop126 preflight and redraw readiness"
                if not blockers
                else "fix package blockers before external annotation"
            ),
        },
        "outputs": {
            "context_csv": str(resolve_path(context_csv)),
            "annotation_template_csv": str(resolve_path(annotation_template_csv)),
            "reviewer_guide_json": str(resolve_path(reviewer_guide_json)),
        },
    }
    write_json(output_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Loop157 external annotation package.")
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--context-csv", type=Path, required=True)
    parser.add_argument("--annotation-template-csv", type=Path, required=True)
    parser.add_argument("--reviewer-guide-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=162)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = export_loop157_package(
        review_csv=args.review_csv,
        context_csv=args.context_csv,
        annotation_template_csv=args.annotation_template_csv,
        reviewer_guide_json=args.reviewer_guide_json,
        output_json=args.output_json,
        expected_rows=args.expected_rows,
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "rows": summary["rows"],
                "blockers": summary["blockers"],
                "context_field_count": summary["field_audit"]["context_field_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if not summary["blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
