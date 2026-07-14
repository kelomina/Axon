#!/usr/bin/env python3
"""Export an identity-safe external annotation package from Loop106 focus rows."""

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

from apply_manual_review_verdicts import (  # noqa: E402
    EXCLUDE_ACTIONS,
    EXCLUDE_VERDICTS,
    KEEP_VERDICTS,
    UNCERTAIN_VERDICTS,
    normalize_text,
)
from merge_loop106_focus_annotations import (  # noqa: E402
    MANUAL_FIELDS,
    forbidden_columns,
)


PROTOCOL = (
    "Loop113 external focus annotation package export; content context plus Loop111 submission template only; "
    "no private-map read, no unblind, no model fitting, no threshold selection, no automatic verdict, "
    "no split/cache mutation"
)
IDENTITY_POLICY = (
    "The external context package may expose blind_review_id and content-derived fields only. Filename/path/"
    "directory/extension/hash/source_sha256/sample_index/split/row order/model score/rank fields are not "
    "exported and are not verdict evidence."
)
ANNOTATION_FIELDS = ["blind_review_id", *MANUAL_FIELDS]
CONTEXT_FIELDS = [
    "blind_review_id",
    "current_label",
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
]
FORBIDDEN_OUTPUT_HEADER_TOKENS = [
    "filename",
    "file_name",
    "source_path",
    "cache_path",
    "source_sha",
    "sha256",
    "hash",
    "sample_index",
    "split",
    "row_order",
    "review_priority_rank",
    "review_batch_rank",
    "rank",
    "loop57",
    "loop39",
    "prob",
    "score",
    "prediction",
    "threshold",
]
FORBIDDEN_CONTENT_TOKENS = [
    "filename",
    "file_name",
    "source_path",
    "cache_path",
    "source_sha",
    "sha256",
    "sample_index",
    "row_order",
    "review_priority_rank",
    "review_batch_rank",
    "loop57",
    "loop39",
    "probability",
    "prob_malicious",
    "final_prob",
    "gate_prob",
    "score",
    "prediction",
    "threshold",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def output_header_violations(fieldnames: Sequence[str]) -> list[str]:
    violations: list[str] = []
    for field in fieldnames:
        lowered = field.casefold()
        if field == "blind_review_id":
            continue
        if lowered.startswith("pe_") and "directory" in lowered:
            continue
        if any(token in lowered for token in FORBIDDEN_OUTPUT_HEADER_TOKENS):
            violations.append(field)
    return sorted(set(violations))


def content_value_violations(rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        blind_id = normalize_text(row.get("blind_review_id"))
        for field in fieldnames:
            text = normalize_text(row.get(field)).casefold()
            if not text:
                continue
            hits = [token for token in FORBIDDEN_CONTENT_TOKENS if token in text]
            if hits:
                violations.append(
                    {
                        "row_number": row_number,
                        "blind_review_id": blind_id,
                        "field": field,
                        "tokens": sorted(set(hits)),
                    }
                )
    return violations


def rows_by_blind_id(rows: Sequence[dict[str, str]]) -> tuple[int, int]:
    seen: set[str] = set()
    missing = 0
    duplicate = 0
    for row in rows:
        blind_id = normalize_text(row.get("blind_review_id"))
        if not blind_id:
            missing += 1
            continue
        if blind_id in seen:
            duplicate += 1
        seen.add(blind_id)
    return missing, duplicate


def build_context_rows(rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> tuple[list[dict[str, Any]], list[str]]:
    output_fields = [field for field in CONTEXT_FIELDS if field in fieldnames]
    context_rows = [{field: normalize_text(row.get(field)) for field in output_fields} for row in rows]
    return context_rows, output_fields


def export_external_focus_annotation_package(
    *,
    focus_csv: Path,
    context_csv: Path,
    annotation_template_csv: Path,
    reviewer_guide_json: Path,
    output_json: Path,
    expected_focus_rows: Optional[int] = 240,
) -> dict[str, Any]:
    rows, fieldnames = read_csv_rows(focus_csv)
    blockers: list[str] = []
    warnings: list[str] = []
    if expected_focus_rows is not None and len(rows) != expected_focus_rows:
        blockers.append("unexpected_focus_row_count")

    required_columns = ["blind_review_id", "current_label", *MANUAL_FIELDS]
    missing_required_columns = [field for field in required_columns if field not in fieldnames]
    if missing_required_columns:
        blockers.append("focus_missing_required_columns")

    forbidden_focus_columns = forbidden_columns(fieldnames)
    if forbidden_focus_columns:
        blockers.append("focus_contains_identity_or_model_columns")

    missing_blind_id_rows, duplicate_blind_id_rows = rows_by_blind_id(rows)
    if missing_blind_id_rows:
        blockers.append("missing_blind_review_id")
    if duplicate_blind_id_rows:
        blockers.append("duplicate_blind_review_id")

    context_rows, context_fields = build_context_rows(rows, fieldnames)
    missing_context_fields = [field for field in CONTEXT_FIELDS if field not in fieldnames]
    if missing_context_fields:
        warnings.append("some_context_fields_missing_from_focus_csv")

    header_violations = output_header_violations(context_fields)
    if header_violations:
        blockers.append("context_output_header_contains_forbidden_fields")
    value_violations = content_value_violations(context_rows, context_fields)
    if value_violations:
        blockers.append("context_output_values_reference_identity_or_model_terms")

    write_csv_rows(context_csv, context_rows, context_fields)
    write_csv_rows(annotation_template_csv, [], ANNOTATION_FIELDS)
    guide = {
        "schema": "axon_loop113_external_focus_annotation_guide_v1",
        "protocol": PROTOCOL,
        "identity_policy": IDENTITY_POLICY,
        "annotation_fields": ANNOTATION_FIELDS,
        "allowed_manual_label_verdict_examples": sorted(KEEP_VERDICTS | EXCLUDE_VERDICTS | UNCERTAIN_VERDICTS),
        "allowed_recommended_action_examples": sorted(set(EXCLUDE_ACTIONS) | {"", "keep_label", "model_blindspot", "keep_sample", "needs_more_evidence", "quarantine_for_more_evidence", "replace_with_fresh_same_label_candidate"}),
        "required_note_rule": "Actionable verdicts must cite content or external evidence, not identity fields or model scores.",
        "submission_rule": "Submit only rows with filled manual fields. Blank rows should be omitted from the Loop111 input file.",
        "forbidden_evidence": [
            "filename/path/directory/extension",
            "hash/source_sha256/sample_index/split/row order",
            "model score/probability/prediction/threshold",
            "loop106 rank/score or review ordering",
        ],
    }
    write_json(reviewer_guide_json, guide)

    label_counts = Counter(normalize_text(row.get("current_label")) or "blank" for row in rows)
    summary = {
        "schema": "axon_loop113_external_focus_annotation_package_v1",
        "protocol": PROTOCOL,
        "identity_policy": IDENTITY_POLICY,
        "inputs": {
            "focus_csv": str(focus_csv),
            "expected_focus_rows": expected_focus_rows,
        },
        "decision": "blocked_invalid_focus_package" if blockers else "ready_for_external_content_annotation",
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "rows": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "field_audit": {
            "focus_field_count": len(fieldnames),
            "context_field_count": len(context_fields),
            "missing_required_columns": missing_required_columns,
            "missing_context_fields": missing_context_fields,
            "forbidden_focus_columns": forbidden_focus_columns,
            "context_header_violations": header_violations,
            "context_value_violation_count": len(value_violations),
            "context_value_violation_examples": value_violations[:20],
            "missing_blind_review_id_rows": missing_blind_id_rows,
            "duplicate_blind_review_id_rows": duplicate_blind_id_rows,
        },
        "decisions": {
            "ready_for_external_content_annotation": not blockers,
            "loop111_input_ready_as_noop_template": not blockers,
            "automatic_verdict_allowed": False,
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "full_test_allowed": False,
            "next_allowed_step": (
                "fill annotation template with independent content/external verdicts, then run Loop112"
                if not blockers
                else "fix focus package blockers before external annotation"
            ),
        },
        "outputs": {
            "context_csv": str(context_csv),
            "annotation_template_csv": str(annotation_template_csv),
            "reviewer_guide_json": str(reviewer_guide_json),
            "summary_json": str(output_json),
        },
        "notes": [
            "The context CSV intentionally omits loop106 rank/score, source paths, hashes, sample indices, split, and model scores.",
            "The annotation template is header-only so it is a valid no-op Loop111 input until real rows are added.",
            "Blank annotation rows are not accepted by Loop111; reviewers should submit only filled verdict rows.",
        ],
    }
    write_json(output_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Loop113 external focus annotation package.")
    parser.add_argument("--focus-csv", type=Path, required=True)
    parser.add_argument("--context-csv", type=Path, required=True)
    parser.add_argument("--annotation-template-csv", type=Path, required=True)
    parser.add_argument("--reviewer-guide-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-focus-rows", type=int, default=240)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = export_external_focus_annotation_package(
        focus_csv=args.focus_csv,
        context_csv=args.context_csv,
        annotation_template_csv=args.annotation_template_csv,
        reviewer_guide_json=args.reviewer_guide_json,
        output_json=args.output_json,
        expected_focus_rows=args.expected_focus_rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
