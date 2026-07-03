#!/usr/bin/env python3
"""Strictly validate Loop86 review-evidence verdicts before redraw planning."""

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


IDENTITY_FEATURE_POLICY = (
    "source_path/cache_path/source_sha256/sample_index/split/review rank/model score columns are loading, "
    "alignment, priority, and manual-review fields only; they are not verdict evidence, model evidence, "
    "replacement sampling keys, or threshold/fusion inputs"
)
REPLACEMENT_RULE = (
    "Confirmed label_wrong/feature_broken/out_of_scope rows must be quarantined and replaced by a fresh valid "
    "sample from the locked-manifest original-label pool. Never self-fill from the bad row."
)
REQUIRED_COLUMNS = [
    "review_batch_rank",
    "source_path",
    "source_sha256",
    "sample_index",
    "split",
    "label",
    "loop57_error_type",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
]
VALID_LABEL_TEXT = {
    "0": 0,
    "benign": 0,
    "white": 0,
    "clean": 0,
    "1": 1,
    "malicious": 1,
    "black": 1,
    "malware": 1,
}
UNCERTAIN_ACTIONS = {"", "needs_more_evidence", "quarantine_for_more_evidence"}
KEEP_ACTIONS = {"", "keep_label", "model_blindspot", "keep_sample"}
EXCLUDE_ACTIONS_STRICT = set(EXCLUDE_ACTIONS) | {"replace_with_fresh_same_label_candidate"}
IDENTITY_NOTE_TERMS = {
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
    "review rank",
    "review_batch_rank",
}
MODEL_SCORE_NOTE_TERMS = {
    "model score",
    "probability",
    "prob_malicious",
    "final_prob",
    "loop57",
    "loop28",
    "prediction",
    "threshold",
    "gate_prob",
    "candidate_prob",
}
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
    "provenance",
    "publisher",
    "resource",
    "sandbox",
    "section",
    "security directory",
    "signature",
    "signer",
    "static",
    "vendor",
    "virustotal",
    "vt",
    "yara",
}
STRICT_OUTPUT_EXTRAS = [
    "loop87_row_number",
    "loop87_status",
    "loop87_reason",
    "loop87_issue_flags",
    "loop87_plan_action",
    "loop87_replacement_required",
    "loop87_replacement_label",
    "loop87_training_policy_allowed",
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


def parse_label(value: object) -> Optional[int]:
    return VALID_LABEL_TEXT.get(normalize_text(value))


def verdict_kind(verdict: str) -> str:
    if verdict in KEEP_VERDICTS:
        return "keep"
    if verdict in RELABEL_VERDICTS:
        return "label_wrong"
    if verdict in EXCLUDE_VERDICTS:
        return "exclude"
    if verdict in UNCERTAIN_VERDICTS:
        return "uncertain"
    return "invalid"


def action_kind(action: str) -> str:
    if action in KEEP_ACTIONS:
        return "keep"
    if action in EXCLUDE_ACTIONS_STRICT:
        return "exclude"
    if action in UNCERTAIN_ACTIONS:
        return "uncertain"
    return "invalid"


def note_has_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def evidence_note_issues(note: str) -> list[str]:
    text = normalize_text(note)
    if not text:
        return []
    has_content_or_external = note_has_any(text, CONTENT_OR_EXTERNAL_EVIDENCE_TERMS)
    has_identity_or_score = note_has_any(text, IDENTITY_NOTE_TERMS) or note_has_any(text, MODEL_SCORE_NOTE_TERMS)
    issues: list[str] = []
    if not has_content_or_external:
        issues.append("manual_verdict_note_missing_content_or_external_evidence")
    if has_identity_or_score and not has_content_or_external:
        issues.append("manual_verdict_note_identity_or_score_only")
    return issues


def row_has_manual_content(row: dict[str, str]) -> bool:
    return any(normalize_text(row.get(field)) for field in ["manual_label_verdict", "recommended_action", "manual_verdict_note"])


def validate_row(row: dict[str, str]) -> tuple[str, str, list[str], dict[str, Any]]:
    verdict = normalize_text(row.get("manual_label_verdict"))
    action = normalize_text(row.get("recommended_action"))
    note = normalize_text(row.get("manual_verdict_note"))
    label = parse_label(row.get("label"))
    issues: list[str] = []
    plan = {
        "plan_action": "no_action",
        "replacement_required": "false",
        "replacement_label": "",
        "training_policy_allowed": "false",
    }

    if not verdict and not action and not note:
        return "no_decision", "blank_manual_fields", [], plan

    vkind = verdict_kind(verdict)
    akind = action_kind(action)
    if vkind == "invalid":
        issues.append("invalid_manual_label_verdict")
    if akind == "invalid":
        issues.append("invalid_recommended_action")
    if issues:
        return "invalid", "invalid_verdict_or_action", issues, plan

    if vkind != "uncertain" and not note:
        issues.append("actionable_verdict_requires_manual_verdict_note")
    elif vkind != "uncertain":
        issues.extend(evidence_note_issues(note))

    if vkind == "uncertain":
        if akind not in {"uncertain"}:
            issues.append("uncertain_verdict_requires_more_evidence_or_blank_action")
        if note:
            issues.extend(evidence_note_issues(note))
        plan["plan_action"] = "needs_more_evidence"
        return ("uncertain_no_action" if not issues else "invalid", "manual_uncertain", issues, plan)

    if vkind == "keep":
        if akind != "keep":
            issues.append("label_correct_requires_keep_or_model_blindspot_action")
        plan["plan_action"] = "keep_sample_or_model_blindspot"
        return ("label_correct_model_blindspot" if not issues else "invalid", "manual_label_kept_model_error", issues, plan)

    if vkind in {"label_wrong", "exclude"}:
        if akind != "exclude":
            if vkind == "label_wrong":
                issues.append("label_wrong_requires_replace_or_quarantine_action")
            else:
                issues.append("feature_or_scope_issue_requires_replace_or_quarantine_action")
        if label is None:
            issues.append("invalid_original_label")
        plan["plan_action"] = "quarantine_and_fresh_redraw"
        plan["replacement_required"] = "true"
        plan["replacement_label"] = "" if label is None else str(label)
        return (
            f"{vkind}_replace" if not issues else "invalid",
            "manual_bad_row_replacement_required",
            issues,
            plan,
        )

    return "invalid", "unhandled_verdict_pair", ["unhandled_verdict_pair"], plan


def build_fieldnames(source_fieldnames: Sequence[str]) -> list[str]:
    fieldnames = list(source_fieldnames)
    for field in STRICT_OUTPUT_EXTRAS:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames


def validate_loop86_verdicts(
    *,
    evidence_csv: Path,
    output_csv: Path,
    output_json: Path,
    expected_rows: Optional[int] = 62,
) -> dict[str, Any]:
    rows, source_fieldnames = read_csv_rows(evidence_csv)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in source_fieldnames]
    output_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    duplicate_sample_index = 0
    duplicate_review_rank = 0
    seen_sample_index: set[str] = set()
    seen_review_rank: set[str] = set()
    replacement_counts: Counter[str] = Counter()
    training_policy_rows = 0

    for row_number, row in enumerate(rows, start=1):
        sample_index = normalize_text(row.get("sample_index"))
        review_rank = normalize_text(row.get("review_batch_rank"))
        if sample_index:
            if sample_index in seen_sample_index:
                duplicate_sample_index += 1
            seen_sample_index.add(sample_index)
        if review_rank:
            if review_rank in seen_review_rank:
                duplicate_review_rank += 1
            seen_review_rank.add(review_rank)

        verdict_counts[normalize_text(row.get("manual_label_verdict")) or "blank"] += 1
        action_counts[normalize_text(row.get("recommended_action")) or "blank"] += 1
        status, reason, issues, plan = validate_row(row)
        status_counts[status] += 1
        reason_counts[reason] += 1
        for issue in issues:
            issue_counts[issue] += 1
        if plan["replacement_required"] == "true":
            replacement_counts[str(plan["replacement_label"])] += 1
        if plan["training_policy_allowed"] == "true":
            training_policy_rows += 1

        item = dict(row)
        item["loop87_row_number"] = str(row_number)
        item["loop87_status"] = status
        item["loop87_reason"] = reason
        item["loop87_issue_flags"] = "|".join(issues)
        item["loop87_plan_action"] = plan["plan_action"]
        item["loop87_replacement_required"] = plan["replacement_required"]
        item["loop87_replacement_label"] = plan["replacement_label"]
        item["loop87_training_policy_allowed"] = plan["training_policy_allowed"]
        output_rows.append(item)

    blocking_issues = []
    if missing_columns:
        blocking_issues.append("missing_required_columns")
    if expected_rows is not None and len(rows) != expected_rows:
        blocking_issues.append("unexpected_row_count")
    if duplicate_sample_index:
        blocking_issues.append("duplicate_sample_index")
    if duplicate_review_rank:
        blocking_issues.append("duplicate_review_batch_rank")
    if issue_counts:
        blocking_issues.append("invalid_verdict_rows")

    actionable_rows = sum(count for status, count in status_counts.items() if status not in {"no_decision", "uncertain_no_action"})
    invalid_rows = int(sum(issue_counts.values()))
    if blocking_issues:
        decision = "blocked_invalid_verdicts"
        import_ready = False
    elif actionable_rows == 0:
        decision = "ready_noop_no_actionable_verdicts"
        import_ready = True
    else:
        decision = "ready_for_redraw_plan_review_only"
        import_ready = True

    write_csv_rows(output_csv, output_rows, build_fieldnames(source_fieldnames))
    summary = {
        "schema": "axon_loop87_review_evidence_verdict_import_v1",
        "protocol": (
            "strict manual/external verdict gate for Loop86 evidence package; no model fitting, no threshold "
            "selection, no automatic relabeling, no split/cache mutation"
        ),
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "replacement_rule": REPLACEMENT_RULE,
        "evidence_csv": str(evidence_csv),
        "rows": len(rows),
        "expected_rows": expected_rows,
        "import_ready": import_ready,
        "decision": decision,
        "blocking_issues": blocking_issues,
        "missing_required_columns": missing_columns,
        "duplicate_sample_index_rows": duplicate_sample_index,
        "duplicate_review_batch_rank_rows": duplicate_review_rank,
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "row_issue_counts": dict(sorted(issue_counts.items())),
        "invalid_rows": invalid_rows,
        "actionable_rows": actionable_rows,
        "replacement_required_rows": sum(replacement_counts.values()),
        "replacement_counts_by_original_label": dict(sorted(replacement_counts.items())),
        "training_policy_rows": training_policy_rows,
        "manual_quality": {
            "blank_verdict_rows": int(verdict_counts.get("blank", 0)),
            "actionable_verdict_missing_note_rows": int(issue_counts.get("actionable_verdict_requires_manual_verdict_note", 0)),
            "evidence_note_missing_content_or_external_rows": int(
                issue_counts.get("manual_verdict_note_missing_content_or_external_evidence", 0)
            ),
            "evidence_note_identity_or_score_only_rows": int(
                issue_counts.get("manual_verdict_note_identity_or_score_only", 0)
            ),
        },
        "decisions": {
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "next_allowed_step": (
                "no-op" if decision == "ready_noop_no_actionable_verdicts" else
                "fix invalid verdicts" if not import_ready else
                "build non-destructive redraw plan for confirmed bad rows"
            ),
        },
        "outputs": {
            "validated_csv": str(output_csv),
            "summary_json": str(output_json),
        },
        "notes": [
            "Manual verdict notes must cite content or external evidence. Identity fields and model scores alone are blocked.",
            "A confirmed bad row creates a redraw request from the locked-manifest original-label pool; it does not self-fill counts.",
            "This gate validates verdict ingress only. It does not build a corrected split or authorize Train/Val/Test evaluation.",
        ],
    }
    write_json(output_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Loop86 manual/external verdicts.")
    parser.add_argument("--evidence-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=62)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate_loop86_verdicts(
        evidence_csv=args.evidence_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["import_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
