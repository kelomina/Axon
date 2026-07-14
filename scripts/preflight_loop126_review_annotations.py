#!/usr/bin/env python3
"""Preflight Loop126 blinded review annotations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANUAL_FIELDS = ["manual_label_verdict", "manual_verdict_note", "recommended_action"]
REQUIRED_COLUMNS = ["review_focus_id", *MANUAL_FIELDS]
KEEP_VERDICTS = {"label_correct", "correct", "keep"}
BAD_ROW_VERDICTS = {"label_wrong", "wrong", "mislabeled", "feature_broken", "invalid_pe", "out_of_scope", "bad_feature", "corrupt"}
UNCERTAIN_VERDICTS = {"", "uncertain", "needs_more_evidence", "unknown", "review_later"}
KEEP_ACTIONS = {"", "keep_label", "model_blindspot", "keep_sample"}
REPLACE_ACTIONS = {"replace_with_fresh_same_label_candidate", "quarantine_for_more_evidence"}
UNCERTAIN_ACTIONS = {"", "needs_more_evidence", "quarantine_for_more_evidence"}
FORBIDDEN_INPUT_FIELD_TOKENS = [
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
]
CONTENT_OR_EXTERNAL_EVIDENCE_TERMS = {
    "api",
    "authenticode",
    "behavior",
    "bytes",
    "certificate",
    "content",
    "corrupt",
    "dynamic",
    "entropy",
    "evidence",
    "export",
    "external",
    "extraction",
    "feature",
    "field",
    "header",
    "import",
    "invalid",
    "mismatch",
    "multi-engine",
    "npz",
    "overlay",
    "packer",
    "parse",
    "pe",
    "permission",
    "rwx",
    "sandbox",
    "section",
    "security directory",
    "signature",
    "static",
    "virustotal",
    "vt",
    "yara",
}
IDENTITY_OR_SCORE_NOTE_TERMS = {
    "filename",
    "file name",
    "path",
    "directory",
    "folder",
    "extension",
    "source_path",
    "cache_path",
    "source_sha256",
    "sha256",
    "hash",
    "sample_index",
    "split",
    "row order",
    "rank",
    "model score",
    "probability",
    "prob_malicious",
    "prediction",
    "threshold",
    "confidence",
    "severity",
}
OUTPUT_EXTRAS = [
    "loop126_row_number",
    "loop126_status",
    "loop126_reason",
    "loop126_issue_flags",
    "loop126_plan_action",
    "loop126_replacement_required",
    "loop126_replacement_label",
    "loop126_training_policy_allowed",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_text(value: object) -> str:
    return str(value or "").strip().casefold()


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


def _forbidden_input_columns(fieldnames: Sequence[str]) -> list[str]:
    allowed = {
        "review_focus_id",
        "focus_rank",
        "priority_band",
        "split",
        "current_label",
        "error_type",
        "error_transition",
        "triage_confidence_bucket",
        "review_lane",
        "content_signal_count",
        "content_tags",
        "recommended_review_action",
        "pe_schema_version",
        *MANUAL_FIELDS,
    }
    found: list[str] = []
    for fieldname in fieldnames:
        if fieldname in allowed:
            continue
        if any(_field_has_forbidden_token(fieldname, token) for token in FORBIDDEN_INPUT_FIELD_TOKENS):
            found.append(fieldname)
    return sorted(set(found))


def _field_has_forbidden_token(fieldname: str, token: str) -> bool:
    folded = fieldname.casefold()
    parts = [part for part in folded.replace("-", "_").split("_") if part]
    return token in parts


def _has_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _note_issues(note: str) -> list[str]:
    text = normalize_text(note)
    if not text:
        return []
    has_content = _has_any(text, CONTENT_OR_EXTERNAL_EVIDENCE_TERMS)
    has_identity_or_score = _has_any(text, IDENTITY_OR_SCORE_NOTE_TERMS)
    issues: list[str] = []
    if not has_content:
        issues.append("manual_verdict_note_missing_content_or_external_evidence")
    if has_identity_or_score and not has_content:
        issues.append("manual_verdict_note_identity_or_score_only")
    return issues


def _validate_row(row: dict[str, str]) -> tuple[str, str, list[str], dict[str, str]]:
    verdict = normalize_text(row.get("manual_label_verdict"))
    action = normalize_text(row.get("recommended_action"))
    note = normalize_text(row.get("manual_verdict_note"))
    label = normalize_text(row.get("current_label"))
    issues: list[str] = []
    plan = {
        "plan_action": "no_action",
        "replacement_required": "false",
        "replacement_label": "",
        "training_policy_allowed": "false",
    }

    if not verdict and not action and not note:
        return "no_decision", "blank_manual_fields", [], plan

    if verdict not in KEEP_VERDICTS and verdict not in BAD_ROW_VERDICTS and verdict not in UNCERTAIN_VERDICTS:
        issues.append("invalid_manual_label_verdict")
    if action not in KEEP_ACTIONS and action not in REPLACE_ACTIONS and action not in UNCERTAIN_ACTIONS:
        issues.append("invalid_recommended_action")
    if issues:
        return "invalid", "invalid_verdict_or_action", issues, plan

    if verdict in UNCERTAIN_VERDICTS:
        if action not in UNCERTAIN_ACTIONS:
            issues.append("uncertain_verdict_requires_more_evidence_or_blank_action")
        if note:
            issues.extend(_note_issues(note))
        plan["plan_action"] = "needs_more_evidence"
        return ("uncertain_no_action" if not issues else "invalid", "manual_uncertain", issues, plan)

    if not note:
        issues.append("actionable_verdict_requires_manual_verdict_note")
    else:
        issues.extend(_note_issues(note))

    if verdict in KEEP_VERDICTS:
        if action not in KEEP_ACTIONS:
            issues.append("label_correct_requires_keep_or_model_blindspot_action")
        plan["plan_action"] = "keep_sample_or_model_blindspot"
        return ("label_correct_model_blindspot" if not issues else "invalid", "manual_label_kept_model_error", issues, plan)

    if verdict in BAD_ROW_VERDICTS:
        if action not in REPLACE_ACTIONS:
            issues.append("bad_row_verdict_requires_quarantine_or_fresh_replacement_action")
        if label not in {"0", "1"}:
            issues.append("replacement_requires_current_label_0_or_1")
        plan["plan_action"] = "quarantine_and_replace_fresh_same_original_label"
        plan["replacement_required"] = "true"
        plan["replacement_label"] = label if label in {"0", "1"} else ""
        return ("bad_row_replacement_required" if not issues else "invalid", "manual_bad_row_replacement_required", issues, plan)

    return "invalid", "unhandled_verdict_pair", ["unhandled_verdict_pair"], plan


def _build_fieldnames(fieldnames: Sequence[str]) -> list[str]:
    output = list(fieldnames)
    for field in OUTPUT_EXTRAS:
        if field not in output:
            output.append(field)
    return output


def preflight_loop126_review_annotations(
    *,
    annotations_csv: Path,
    output_csv: Path,
    output_json: Path,
    expected_rows: Optional[int] = None,
) -> dict[str, Any]:
    rows, fieldnames = read_csv_rows(annotations_csv)
    blockers: list[str] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    forbidden_columns = _forbidden_input_columns(fieldnames)
    if missing_columns:
        blockers.append("missing_required_columns")
    if forbidden_columns:
        blockers.append("annotations_contain_identity_or_model_columns")
    if expected_rows is not None and len(rows) != expected_rows:
        blockers.append("row_count_mismatch_expected")

    seen_ids: set[str] = set()
    missing_id_rows = 0
    duplicate_id_rows = 0
    status_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    replacement_label_counts: Counter[str] = Counter()
    output_rows: list[dict[str, Any]] = []

    for row_number, row in enumerate(rows, start=1):
        review_id = str(row.get("review_focus_id") or "").strip()
        if not review_id:
            missing_id_rows += 1
        elif review_id in seen_ids:
            duplicate_id_rows += 1
        else:
            seen_ids.add(review_id)

        status, reason, issues, plan = _validate_row(row)
        status_counts[status] += 1
        verdict_counts[normalize_text(row.get("manual_label_verdict")) or "blank"] += 1
        action_counts[normalize_text(row.get("recommended_action")) or "blank"] += 1
        for issue in issues:
            issue_counts[issue] += 1
        if plan["replacement_required"] == "true":
            replacement_label_counts[plan["replacement_label"]] += 1

        item = dict(row)
        item["loop126_row_number"] = str(row_number)
        item["loop126_status"] = status
        item["loop126_reason"] = reason
        item["loop126_issue_flags"] = "|".join(issues)
        item["loop126_plan_action"] = plan["plan_action"]
        item["loop126_replacement_required"] = plan["replacement_required"]
        item["loop126_replacement_label"] = plan["replacement_label"]
        item["loop126_training_policy_allowed"] = plan["training_policy_allowed"]
        output_rows.append(item)

    if missing_id_rows:
        blockers.append("missing_review_focus_id")
    if duplicate_id_rows:
        blockers.append("duplicate_review_focus_id")
    if issue_counts:
        blockers.append("manual_annotation_quality_issues")

    annotated_rows = sum(1 for row in rows if any(normalize_text(row.get(field)) for field in MANUAL_FIELDS))
    actionable_rows = int(status_counts["label_correct_model_blindspot"] + status_counts["bad_row_replacement_required"])
    replacement_required_rows = int(status_counts["bad_row_replacement_required"])
    ready = not blockers and annotated_rows > 0

    write_csv_rows(output_csv, output_rows, _build_fieldnames(fieldnames))
    payload = {
        "schema": "axon_loop126_review_preflight_v1",
        "protocol": "strict Loop126 annotation preflight; no private-map read, no unblind, no automatic relabel, no split/cache mutation",
        "identity_policy": (
            "review_focus_id is the only public row key. Source path, cache path, source hash, sample index, "
            "filename, directory, extension, model score, probability, threshold, and prediction fields are not allowed."
        ),
        "annotations_csv": str(resolve_path(annotations_csv)),
        "output_csv": str(resolve_path(output_csv)),
        "expected_rows": expected_rows,
        "rows": len(rows),
        "annotated_rows": annotated_rows,
        "actionable_rows": actionable_rows,
        "replacement_required_rows": replacement_required_rows,
        "training_policy_rows": 0,
        "ready_for_private_mapping": ready,
        "blockers": sorted(set(blockers)),
        "missing_columns": missing_columns,
        "forbidden_columns": forbidden_columns,
        "missing_review_focus_id_rows": missing_id_rows,
        "duplicate_review_focus_id_rows": duplicate_id_rows,
        "status_counts": dict(sorted(status_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "replacement_label_counts": dict(sorted(replacement_label_counts.items())),
        "automatic_relabel_allowed": False,
        "automatic_replacement_allowed": False,
        "test10k_allowed": False,
        "full_test_allowed": False,
    }
    resolved_json = resolve_path(output_json)
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight Loop126 blinded review annotations.")
    parser.add_argument("--annotations-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = preflight_loop126_review_annotations(
        annotations_csv=args.annotations_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        expected_rows=args.expected_rows,
    )
    print(
        json.dumps(
            {
                "ready_for_private_mapping": payload["ready_for_private_mapping"],
                "rows": payload["rows"],
                "annotated_rows": payload["annotated_rows"],
                "blockers": payload["blockers"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if payload["ready_for_private_mapping"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
