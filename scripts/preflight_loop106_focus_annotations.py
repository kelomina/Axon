#!/usr/bin/env python3
"""Preflight Loop106 focus annotations before merge/unblind."""

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
    RELABEL_VERDICTS,
    UNCERTAIN_VERDICTS,
    normalize_text,
)
from import_loop87_review_evidence_verdicts import (  # noqa: E402
    CONTENT_OR_EXTERNAL_EVIDENCE_TERMS,
    IDENTITY_NOTE_TERMS,
    MODEL_SCORE_NOTE_TERMS,
    action_kind,
    evidence_note_issues,
    note_has_any,
    verdict_kind,
)
from merge_loop106_focus_annotations import (  # noqa: E402
    MANUAL_FIELDS,
    forbidden_columns,
)


PROTOCOL = (
    "read-only Loop106 focus annotation preflight; no private-map read, no unblind, no model fitting, "
    "no threshold selection, no automatic verdict, no split/cache mutation"
)
IDENTITY_POLICY = (
    "Focus manual fields must cite content or external evidence. Filename/path/directory/extension/hash/"
    "source_sha256/sample_index/split/row order/review rank/model score terms are not sufficient evidence."
)
REQUIRED_COLUMNS = ["blind_review_id", *MANUAL_FIELDS]
OUTPUT_EXTRAS = [
    "loop109_row_number",
    "loop109_status",
    "loop109_reason",
    "loop109_issue_flags",
]
EXCLUDE_ACTIONS_STRICT = set(EXCLUDE_ACTIONS) | {"replace_with_fresh_same_label_candidate"}
UNCERTAIN_ACTIONS = {"", "needs_more_evidence", "quarantine_for_more_evidence"}
KEEP_ACTIONS = {"", "keep_label", "model_blindspot", "keep_sample"}


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


def build_fieldnames(source_fieldnames: Sequence[str]) -> list[str]:
    fieldnames = list(source_fieldnames)
    for field in OUTPUT_EXTRAS:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames


def row_has_manual_content(row: dict[str, str]) -> bool:
    return any(normalize_text(row.get(field)) for field in MANUAL_FIELDS)


def manual_text(row: dict[str, str]) -> str:
    return " ".join(normalize_text(row.get(field)) for field in MANUAL_FIELDS if normalize_text(row.get(field)))


def row_identity_or_score_mentions(row: dict[str, str]) -> list[str]:
    text = manual_text(row)
    terms = sorted(IDENTITY_NOTE_TERMS | MODEL_SCORE_NOTE_TERMS)
    return [term for term in terms if term in text]


def validate_focus_annotation_row(row: dict[str, str]) -> tuple[str, str, list[str]]:
    verdict = normalize_text(row.get("manual_label_verdict"))
    action = normalize_text(row.get("recommended_action"))
    note = normalize_text(row.get("manual_verdict_note"))
    issues: list[str] = []

    if not verdict and not action and not note:
        return "no_decision", "blank_manual_fields", []

    vkind = verdict_kind(verdict)
    akind = action_kind(action)
    if vkind == "invalid":
        issues.append("invalid_manual_label_verdict")
    if akind == "invalid":
        issues.append("invalid_recommended_action")
    if issues:
        return "invalid", "invalid_verdict_or_action", issues

    if vkind == "uncertain":
        if action not in UNCERTAIN_ACTIONS:
            issues.append("uncertain_verdict_requires_more_evidence_or_blank_action")
        if note:
            issues.extend(evidence_note_issues(note))
        return ("uncertain_no_action" if not issues else "invalid", "manual_uncertain", issues)

    if not note:
        issues.append("actionable_verdict_requires_manual_verdict_note")
    else:
        issues.extend(evidence_note_issues(note))

    if vkind == "keep":
        if action not in KEEP_ACTIONS:
            issues.append("label_correct_requires_keep_or_model_blindspot_action")
        return ("label_correct_model_blindspot" if not issues else "invalid", "manual_label_kept_model_error", issues)

    if vkind in {"label_wrong", "exclude"}:
        if action not in EXCLUDE_ACTIONS_STRICT:
            if vkind == "label_wrong":
                issues.append("label_wrong_requires_replace_or_quarantine_action")
            else:
                issues.append("feature_or_scope_issue_requires_replace_or_quarantine_action")
        return (f"{vkind}_replace" if not issues else "invalid", "manual_bad_row_replacement_required", issues)

    return "invalid", "unhandled_verdict_pair", ["unhandled_verdict_pair"]


def preflight_focus_annotations(
    *,
    focus_annotations_csv: Path,
    output_csv: Path,
    output_json: Path,
    expected_rows: Optional[int] = 240,
) -> dict[str, Any]:
    rows, fieldnames = read_csv_rows(focus_annotations_csv)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    forbidden_focus_columns = forbidden_columns(fieldnames)
    blockers: list[str] = []
    warnings: list[str] = []
    if missing_columns:
        blockers.append("missing_required_columns")
    if forbidden_focus_columns:
        blockers.append("focus_contains_identity_or_model_columns")
    if expected_rows is not None and len(rows) != expected_rows:
        blockers.append("unexpected_row_count")

    seen_blind_ids: set[str] = set()
    duplicate_blind_id_rows = 0
    missing_blind_id_rows = 0
    output_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    identity_or_score_mention_rows = 0
    content_or_external_note_rows = 0

    for row_number, row in enumerate(rows, start=1):
        blind_id = str(row.get("blind_review_id") or "").strip()
        if not blind_id:
            missing_blind_id_rows += 1
        elif blind_id in seen_blind_ids:
            duplicate_blind_id_rows += 1
        else:
            seen_blind_ids.add(blind_id)

        verdict_counts[normalize_text(row.get("manual_label_verdict")) or "blank"] += 1
        action_counts[normalize_text(row.get("recommended_action")) or "blank"] += 1
        status, reason, issues = validate_focus_annotation_row(row)
        status_counts[status] += 1
        reason_counts[reason] += 1
        for issue in issues:
            issue_counts[issue] += 1

        note = normalize_text(row.get("manual_verdict_note"))
        if note_has_any(note, CONTENT_OR_EXTERNAL_EVIDENCE_TERMS):
            content_or_external_note_rows += 1
        if row_identity_or_score_mentions(row):
            identity_or_score_mention_rows += 1

        item = dict(row)
        item["loop109_row_number"] = str(row_number)
        item["loop109_status"] = status
        item["loop109_reason"] = reason
        item["loop109_issue_flags"] = "|".join(issues)
        output_rows.append(item)

    if duplicate_blind_id_rows:
        blockers.append("duplicate_blind_review_id")
    if missing_blind_id_rows:
        blockers.append("missing_blind_review_id")
    if issue_counts:
        blockers.append("invalid_focus_annotation_rows")

    annotated_rows = sum(1 for row in rows if row_has_manual_content(row))
    actionable_rows = sum(
        count
        for status, count in status_counts.items()
        if status not in {"no_decision", "uncertain_no_action"}
    )
    invalid_rows = int(sum(issue_counts.values()))
    if blockers:
        decision = "blocked_invalid_focus_annotations"
        ready_for_merge = False
    elif annotated_rows == 0:
        decision = "ready_noop_no_focus_annotations"
        ready_for_merge = True
    else:
        decision = "ready_for_focus_merge"
        ready_for_merge = True
    if identity_or_score_mention_rows:
        warnings.append("manual_fields_reference_identity_or_model_terms")

    write_csv_rows(output_csv, output_rows, build_fieldnames(fieldnames))
    summary = {
        "schema": "axon_loop109_focus_annotation_preflight_v1",
        "protocol": PROTOCOL,
        "identity_policy": IDENTITY_POLICY,
        "inputs": {
            "focus_annotations_csv": str(focus_annotations_csv),
        },
        "rows": len(rows),
        "expected_rows": expected_rows,
        "decision": decision,
        "ready_for_focus_merge": ready_for_merge,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "missing_required_columns": missing_columns,
        "forbidden_focus_columns": forbidden_focus_columns,
        "duplicate_blind_review_id_rows": duplicate_blind_id_rows,
        "missing_blind_review_id_rows": missing_blind_id_rows,
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "row_issue_counts": dict(sorted(issue_counts.items())),
        "annotated_rows": annotated_rows,
        "actionable_rows": actionable_rows,
        "invalid_rows": invalid_rows,
        "manual_quality": {
            "blank_verdict_rows": int(verdict_counts.get("blank", 0)),
            "content_or_external_note_rows": content_or_external_note_rows,
            "identity_or_model_term_mention_rows": identity_or_score_mention_rows,
            "actionable_verdict_missing_note_rows": int(
                issue_counts.get("actionable_verdict_requires_manual_verdict_note", 0)
            ),
            "evidence_note_missing_content_or_external_rows": int(
                issue_counts.get("manual_verdict_note_missing_content_or_external_evidence", 0)
            ),
            "evidence_note_identity_or_score_only_rows": int(
                issue_counts.get("manual_verdict_note_identity_or_score_only", 0)
            ),
        },
        "decisions": {
            "ready_for_focus_merge": ready_for_merge,
            "automatic_verdict_allowed": False,
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "next_allowed_step": (
                "merge focus annotations into full blinded CSV"
                if ready_for_merge
                else "fix invalid focus annotations before merge/unblind"
            ),
        },
        "outputs": {
            "validated_focus_csv": str(output_csv),
            "summary_json": str(output_json),
        },
        "notes": [
            "This preflight keeps the review blinded and does not read the private map.",
            "Blank manual fields are allowed as no-op review state; they do not create actionable verdicts.",
            "Actionable focus annotations must cite content or external evidence before merge/unblind.",
            "Identity fields and model scores alone are blocked before Loop107 merge and Loop87 import.",
        ],
    }
    write_json(output_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight Loop106 focus annotations before merge/unblind.")
    parser.add_argument("--focus-annotations-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=240)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = preflight_focus_annotations(
        focus_annotations_csv=args.focus_annotations_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["ready_for_focus_merge"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
