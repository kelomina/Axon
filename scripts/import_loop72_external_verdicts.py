#!/usr/bin/env python3
"""Validate filled Loop72 external verdicts before any split adjustment.

This is the strict ingress gate between an external/manual review spreadsheet
and the non-destructive adjustment planner. It validates row identity, verdict
schema, action pairs, held-out test policy, and target-gap accounting. It does
not train, tune thresholds, mutate the split, or use identity fields as model
evidence.
"""

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
    KEEP_ACTIONS,
    KEEP_VERDICTS,
    RELABEL_ACTIONS,
    RELABEL_VERDICTS,
    UNCERTAIN_VERDICTS,
    build_plan,
    normalize_text,
    write_csv_rows,
)


IDENTITY_FEATURE_POLICY = (
    "filename/path/extension/directory/source hash/cache_path/sample_index/split/row order are loading, "
    "alignment, cache-audit, duplicate-review, and manual-review fields only; they are not model evidence "
    "and must not drive thresholds, relabeling, feature engineering, or production inference"
)
REPLACEMENT_RULE = (
    "Confirmed label_wrong/feature_broken/out_of_scope rows must trigger fresh same-original-label redraw. "
    "Do not fill from the bad rows, and preserve exactly 200000 split rows."
)
STRICT_SPLIT_COUNTS = {"train": 20000, "val": 20000, "test": 160000}
REQUIRED_COLUMNS = [
    "source_path",
    "source_sha256",
    "sample_index",
    "split",
    "label",
    "loop57_error_type",
    "loop57_prediction",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
    "corrected_label",
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
UNCERTAIN_ACTIONS = {"", "needs_more_evidence"}
KEEP_STRICT_ACTIONS = {"keep_label", "model_blindspot"}
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
    "review_rank",
    "wave",
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
    "review rank",
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
    "signature",
    "signer",
    "static",
    "vendor",
    "virustotal",
    "vt",
    "yara",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_int(value: object, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def parse_label(value: object) -> Optional[int]:
    text = normalize_text(value)
    return VALID_LABEL_TEXT.get(text)


def load_split(split_csv: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, Any]]:
    rows = read_csv_rows(split_csv)
    by_sample_index: dict[str, dict[str, str]] = {}
    duplicate_sample_index = 0
    for row in rows:
        sample_index = normalize_text(row.get("sample_index"))
        if not sample_index:
            continue
        if sample_index in by_sample_index:
            duplicate_sample_index += 1
            continue
        by_sample_index[sample_index] = row
    split_counts = Counter(row.get("split", "") for row in rows)
    label_split_counts = Counter(f"{row.get('split', '')}:{row.get('label', '')}" for row in rows)
    summary = {
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "label_split_counts": dict(sorted(label_split_counts.items())),
        "duplicate_sample_index_rows": duplicate_sample_index,
    }
    return rows, by_sample_index, summary


def expected_rows_from_target_gap(target_gap: dict[str, Any]) -> Optional[int]:
    value = (
        target_gap.get("review_sources", {})
        .get("loop63", {})
        .get("error_rows")
    )
    parsed = parse_int(value)
    if parsed is not None:
        return parsed
    return parse_int(target_gap.get("current_best", {}).get("errors"))


def current_best_counts(target_gap: dict[str, Any]) -> dict[str, int]:
    current = target_gap.get("current_best", {})
    return {
        "tp": int(current.get("tp", 0) or 0),
        "tn": int(current.get("tn", 0) or 0),
        "fp": int(current.get("fp", 0) or 0),
        "fn": int(current.get("fn", 0) or 0),
    }


def current_best_summary(target_gap: dict[str, Any]) -> dict[str, Any]:
    current = target_gap.get("current_best", {})
    return {
        "source": current.get("source", ""),
        "samples": int(current.get("samples", 0) or 0),
        "f1": float(current.get("f1", 0.0) or 0.0),
        "errors": int(current.get("errors", 0) or 0),
        "fp": int(current.get("fp", 0) or 0),
        "fn": int(current.get("fn", 0) or 0),
        "tp": int(current.get("tp", 0) or 0),
        "tn": int(current.get("tn", 0) or 0),
    }


def target_summary(target_gap: dict[str, Any]) -> dict[str, Any]:
    best_case = target_gap.get("target_gap_best_case", {})
    reduction = target_gap.get("error_reduction_needed_ratio_of_current_errors", 0.0)
    return {
        "target_f1": float(target_gap.get("target_f1", 0.999) or 0.999),
        "minimum_fixed_errors_best_case": int(
            best_case.get("minimum_fixed_errors_best_case", target_gap.get("error_reduction_needed_best_case", 0)) or 0
        ),
        "required_error_reduction_ratio": float(reduction or 0.0),
    }


def f1_from_counts(*, tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def verdict_category(verdict: str) -> str:
    if verdict in KEEP_VERDICTS:
        return "keep"
    if verdict in RELABEL_VERDICTS:
        return "relabel"
    if verdict in EXCLUDE_VERDICTS:
        return "exclude"
    if verdict in UNCERTAIN_VERDICTS:
        return "uncertain"
    return "invalid"


def action_category(action: str) -> str:
    if action in KEEP_STRICT_ACTIONS:
        return "keep"
    if action in RELABEL_ACTIONS:
        return "relabel"
    if action in EXCLUDE_ACTIONS:
        return "exclude"
    if action in UNCERTAIN_ACTIONS:
        return "uncertain"
    return "invalid"


def row_has_manual_content(row: dict[str, str]) -> bool:
    return any(
        normalize_text(row.get(field))
        for field in ["manual_label_verdict", "recommended_action", "corrected_label", "manual_verdict_note"]
    )


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


def validate_verdict_pair(row: dict[str, str]) -> tuple[str, str, list[str], Optional[int]]:
    verdict = normalize_text(row.get("manual_label_verdict"))
    action = normalize_text(row.get("recommended_action"))
    corrected = normalize_text(row.get("corrected_label"))
    note = normalize_text(row.get("manual_verdict_note"))
    original_label = parse_label(row.get("label"))
    prediction = parse_label(row.get("loop57_prediction"))
    corrected_label = parse_label(corrected) if corrected else None
    issues: list[str] = []

    if not verdict and not action and not corrected:
        return "no_decision", "blank_manual_fields", [], None

    verdict_kind = verdict_category(verdict)
    action_kind = action_category(action)
    if verdict_kind == "invalid":
        issues.append("invalid_manual_label_verdict")
    if action_kind == "invalid":
        issues.append("invalid_recommended_action")
    if corrected and corrected_label is None:
        issues.append("invalid_corrected_label")

    if issues:
        return "invalid", "invalid_verdict_or_action", issues, corrected_label

    if verdict_kind != "uncertain" and not note:
        issues.append("actionable_verdict_requires_manual_verdict_note")
    elif verdict_kind != "uncertain":
        issues.extend(evidence_note_issues(note))

    if verdict_kind == "uncertain":
        if action_kind not in {"uncertain"}:
            issues.append("uncertain_verdict_requires_needs_more_evidence_or_blank_action")
        if corrected:
            issues.append("uncertain_verdict_must_not_have_corrected_label")
        status = "uncertain_no_action" if not issues else "invalid"
        return status, "manual_uncertain", issues, corrected_label

    if verdict_kind == "keep":
        if action not in KEEP_STRICT_ACTIONS:
            issues.append("label_correct_requires_keep_label_or_model_blindspot")
        if corrected:
            issues.append("label_correct_must_not_have_corrected_label")
        status = "label_correct_model_blindspot" if not issues else "invalid"
        return status, "manual_label_kept_model_error", issues, corrected_label

    if verdict_kind == "relabel":
        if action_kind != "exclude":
            issues.append("label_wrong_requires_replace_or_quarantine_action")
        if corrected_label is None:
            issues.append("label_wrong_requires_corrected_label")
        elif original_label is None:
            issues.append("invalid_original_label")
        elif corrected_label == original_label:
            issues.append("corrected_label_must_differ_from_original_label")
        if prediction is None:
            issues.append("invalid_loop57_prediction")
        status = "label_wrong_replace" if not issues else "invalid"
        return status, "manual_label_wrong_replacement_required", issues, corrected_label

    if verdict_kind == "exclude":
        if action_kind != "exclude":
            issues.append("feature_or_scope_issue_requires_replace_or_quarantine_action")
        if corrected:
            issues.append("exclude_verdict_must_not_have_corrected_label")
        status = "exclude_and_replace" if not issues else "invalid"
        return status, "manual_exclude_or_replace", issues, corrected_label

    return "invalid", "unhandled_verdict_pair", ["unhandled_verdict_pair"], corrected_label


def metric_effect(row: dict[str, str], status: str, corrected_label: Optional[int]) -> str:
    label = parse_label(row.get("label"))
    prediction = parse_label(row.get("loop57_prediction"))
    error_type = normalize_text(row.get("loop57_error_type")).upper()
    if status == "label_wrong_replace":
        if corrected_label is not None and prediction == corrected_label:
            if error_type in {"FP", "FN"}:
                return f"label_wrong_fixes_current_{error_type.lower()}"
            return "label_wrong_fixes_current_error"
        return "label_wrong_does_not_fix_current_prediction"
    if status == "exclude_and_replace":
        if label is None:
            return "replacement_required_unknown_label"
        return f"replacement_required_same_original_label_{label}"
    if status == "label_correct_model_blindspot":
        return "confirmed_model_error_not_data_noise"
    if status == "uncertain_no_action":
        return "uncertain_no_metric_change"
    return "no_metric_change"


def strict_plan_fieldnames(source_fieldnames: Sequence[str]) -> list[str]:
    extras = [
        "import_row_number",
        "normalized_manual_label_verdict",
        "normalized_recommended_action",
        "normalized_corrected_label",
        "strict_import_status",
        "strict_import_reason",
        "strict_metric_effect",
        "strict_issue_flags",
    ]
    fieldnames = list(source_fieldnames)
    for field in extras:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames


def apply_relabel_metric_delta(counts: dict[str, int], row: dict[str, str], corrected_label: Optional[int]) -> bool:
    label = parse_label(row.get("label"))
    prediction = parse_label(row.get("loop57_prediction"))
    if corrected_label is None or prediction != corrected_label:
        return False
    if label == 0 and prediction == 1 and corrected_label == 1:
        counts["fp"] -= 1
        counts["tp"] += 1
        return True
    if label == 1 and prediction == 0 and corrected_label == 0:
        counts["fn"] -= 1
        counts["tn"] += 1
        return True
    return False


def optimistic_replacement_delta(counts: dict[str, int], row: dict[str, str]) -> bool:
    label = parse_label(row.get("label"))
    prediction = parse_label(row.get("loop57_prediction"))
    if label == 0 and prediction == 1:
        counts["fp"] -= 1
        counts["tn"] += 1
        return True
    if label == 1 and prediction == 0:
        counts["fn"] -= 1
        counts["tp"] += 1
        return True
    return False


def target_gap_metrics(
    *,
    target_gap: dict[str, Any],
    validated_rows: Sequence[dict[str, str]],
    corrected_labels: Sequence[Optional[int]],
) -> dict[str, Any]:
    base_counts = current_best_counts(target_gap)
    if not any(base_counts.values()):
        return {}
    label_wrong_counts = dict(base_counts)
    optimistic_counts = dict(base_counts)
    label_wrong_fixed = 0
    optimistic_replacement_fixed = 0
    replacement_required = 0
    model_blindspot = 0
    uncertain = 0

    for row, corrected_label in zip(validated_rows, corrected_labels):
        status = row["strict_import_status"]
        if status == "label_wrong_replace":
            replacement_required += 1
            if apply_relabel_metric_delta(label_wrong_counts, row, corrected_label):
                label_wrong_fixed += 1
                apply_relabel_metric_delta(optimistic_counts, row, corrected_label)
        elif status == "exclude_and_replace":
            replacement_required += 1
            if optimistic_replacement_delta(optimistic_counts, row):
                optimistic_replacement_fixed += 1
        elif status == "label_correct_model_blindspot":
            model_blindspot += 1
        elif status == "uncertain_no_action":
            uncertain += 1

    target_f1 = float(target_gap.get("target_f1", 0.999) or 0.999)
    label_wrong_f1 = f1_from_counts(tp=label_wrong_counts["tp"], fp=label_wrong_counts["fp"], fn=label_wrong_counts["fn"])
    optimistic_f1 = f1_from_counts(tp=optimistic_counts["tp"], fp=optimistic_counts["fp"], fn=optimistic_counts["fn"])
    current_f1 = f1_from_counts(tp=base_counts["tp"], fp=base_counts["fp"], fn=base_counts["fn"])
    return {
        "target_f1": target_f1,
        "current_counts": base_counts,
        "current_f1": current_f1,
        "strict_label_correction_only": {
            "counts": label_wrong_counts,
            "fixed_current_errors": label_wrong_fixed,
            "f1": label_wrong_f1,
            "target_reached": label_wrong_f1 >= target_f1,
        },
        "redraw_required_best_case": {
            "counts": optimistic_counts,
            "fixed_current_errors": label_wrong_fixed + optimistic_replacement_fixed,
            "replacement_rows_assumed_correct_after_redraw": optimistic_replacement_fixed,
            "f1": optimistic_f1,
            "target_reached": optimistic_f1 >= target_f1,
        },
        "replacement_required_rows": replacement_required,
        "post_redraw_requires_new_eval": replacement_required > 0,
        "confirmed_model_blindspot_rows": model_blindspot,
        "uncertain_rows": uncertain,
        "notes": [
            "Confirmed label_wrong rows are counted only as target-feasibility evidence; split repair still requires fresh same-original-label redraw.",
            "Replacement rows are not counted as achieved fixes until fresh same-original-label redraw, cache extraction, and full re-evaluation complete.",
            "Held-out test verdicts are target-feasibility evidence by default, not train/val policy or threshold-selection input.",
        ],
    }


def validate_loop72_external_verdicts(
    *,
    review_csv: Path,
    split_csv: Path,
    target_gap_json: Optional[Path],
    output_csv: Path,
    output_json: Path,
    plan_csv: Optional[Path] = None,
    plan_json: Optional[Path] = None,
    allow_partial: bool = False,
    expected_rows: Optional[int] = None,
    enforce_20w_split: bool = True,
    allow_test_actions: bool = False,
) -> dict[str, Any]:
    review_rows = read_csv_rows(review_csv)
    target_gap = read_json(target_gap_json)
    _split_rows, split_by_sample_index, split_summary = load_split(split_csv)
    source_fieldnames = list(review_rows[0].keys()) if review_rows else []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in source_fieldnames]
    expected_rows = expected_rows if expected_rows is not None else expected_rows_from_target_gap(target_gap)

    row_issue_counts: Counter[str] = Counter()
    import_status_counts: Counter[str] = Counter()
    metric_effect_counts: Counter[str] = Counter()
    split_action_counts: Counter[str] = Counter()
    verdict_counts: Counter[str] = Counter()
    recommended_action_counts: Counter[str] = Counter()
    confirmed_bad_by_error_type: Counter[str] = Counter()
    blocking_issues: list[str] = []
    seen_sample_index: Counter[str] = Counter()
    validated_rows: list[dict[str, str]] = []
    corrected_labels: list[Optional[int]] = []
    sample_index_match_count = 0
    sha_match_count = 0
    rows_with_split_match = 0

    if missing_columns:
        blocking_issues.append("missing_required_columns")
    if expected_rows is not None and len(review_rows) != expected_rows and not allow_partial:
        blocking_issues.append("review_row_count_mismatch")
    if enforce_20w_split:
        if split_summary["rows"] != 200000:
            blocking_issues.append("split_row_count_not_200000")
        if split_summary["split_counts"] != STRICT_SPLIT_COUNTS:
            blocking_issues.append("split_counts_not_20000_20000_160000")
    if split_summary["duplicate_sample_index_rows"]:
        blocking_issues.append("split_duplicate_sample_index")

    for row_number, row in enumerate(review_rows, start=2):
        row = dict(row)
        issues: list[str] = []
        sample_index = normalize_text(row.get("sample_index"))
        verdict_counts[normalize_text(row.get("manual_label_verdict")) or "blank"] += 1
        recommended_action_counts[normalize_text(row.get("recommended_action")) or "blank"] += 1
        if not sample_index:
            issues.append("missing_sample_index")
        else:
            seen_sample_index[sample_index] += 1
            split_row = split_by_sample_index.get(sample_index)
            if split_row is None:
                issues.append("sample_index_not_found_in_split")
            else:
                sample_index_match_count += 1
                if normalize_text(row.get("split")) != normalize_text(split_row.get("split")):
                    issues.append("split_mismatch_for_sample_index")
                else:
                    rows_with_split_match += 1
                if parse_label(row.get("label")) != parse_label(split_row.get("label")):
                    issues.append("label_mismatch_for_sample_index")
                review_sha = normalize_text(row.get("source_sha256"))
                split_sha = normalize_text(split_row.get("source_sha256"))
                if review_sha and split_sha and review_sha == split_sha:
                    sha_match_count += 1

        if normalize_text(row.get("split")) not in {"train", "val", "test"}:
            issues.append("invalid_split")
        if parse_label(row.get("label")) is None:
            issues.append("invalid_label")
        if normalize_text(row.get("loop57_error_type")).upper() not in {"FP", "FN"}:
            issues.append("invalid_loop57_error_type")
        if parse_label(row.get("loop57_prediction")) is None:
            issues.append("invalid_loop57_prediction")

        status, reason, verdict_issues, corrected_label = validate_verdict_pair(row)
        issues.extend(verdict_issues)
        effect = metric_effect(row, status, corrected_label)

        for issue in issues:
            row_issue_counts[issue] += 1
        if issues:
            status = "invalid"
            reason = "strict_validation_failed"

        row["import_row_number"] = str(row_number)
        row["normalized_manual_label_verdict"] = normalize_text(row.get("manual_label_verdict"))
        row["normalized_recommended_action"] = normalize_text(row.get("recommended_action"))
        row["normalized_corrected_label"] = normalize_text(row.get("corrected_label"))
        row["strict_import_status"] = status
        row["strict_import_reason"] = reason
        row["strict_metric_effect"] = effect
        row["strict_issue_flags"] = "|".join(sorted(set(issues)))
        validated_rows.append(row)
        corrected_labels.append(corrected_label)
        import_status_counts[status] += 1
        metric_effect_counts[effect] += 1
        split_action_counts[f"{row.get('split', '')}:{status}"] += 1
        if status in {"label_wrong_replace", "exclude_and_replace"}:
            confirmed_bad_by_error_type[normalize_text(row.get("loop57_error_type")).upper() or "unknown"] += 1

    duplicate_review_sample_index_rows = int(sum(count - 1 for count in seen_sample_index.values() if count > 1))
    duplicate_review_sample_index_examples = [
        {"sample_index": key, "count": count}
        for key, count in sorted(seen_sample_index.items())
        if count > 1
    ][:10]
    if duplicate_review_sample_index_rows:
        blocking_issues.append("duplicate_review_sample_index")
        row_issue_counts["duplicate_review_sample_index"] += duplicate_review_sample_index_rows

    invalid_rows = int(import_status_counts.get("invalid", 0))
    label_wrong_rows = int(import_status_counts.get("label_wrong_replace", 0))
    exclude_rows = int(import_status_counts.get("exclude_and_replace", 0))
    confirmed_bad_rows = label_wrong_rows + exclude_rows
    no_op_rows = int(import_status_counts.get("no_decision", 0)) + int(import_status_counts.get("uncertain_no_action", 0))
    actionable_rows = sum(
        import_status_counts.get(status, 0)
        for status in ["label_wrong_replace", "exclude_and_replace", "label_correct_model_blindspot", "uncertain_no_action"]
    )
    manual_content_rows = sum(1 for row in review_rows if row_has_manual_content(row))
    import_ready = not blocking_issues and invalid_rows == 0
    target_info = target_summary(target_gap)
    target_min_fixed = int(target_info.get("minimum_fixed_errors_best_case", 0) or 0)
    target_gap_coverage = float(confirmed_bad_rows / target_min_fixed) if target_min_fixed else 0.0

    output_fieldnames = strict_plan_fieldnames(source_fieldnames)
    write_csv_rows(output_csv, validated_rows, output_fieldnames)

    adjustment_summary: Optional[dict[str, Any]] = None
    if import_ready and plan_csv is not None and plan_json is not None:
        plan_rows, adjustment_summary = build_plan(
            review_csv=output_csv,
            split_csv=split_csv,
            allow_test_actions=allow_test_actions,
        )
        plan_fieldnames = [
            "source_path",
            "source_sha256",
            "sample_index",
            "split",
            "original_label",
            "planned_label",
            "plan_action",
            "reason",
            "manual_label_verdict",
            "recommended_action",
            "manual_verdict_note",
            "replacement_required",
            "replacement_label",
            "usable_for_training_policy",
        ]
        write_csv_rows(plan_csv, plan_rows, plan_fieldnames)
        adjustment_summary["outputs"] = {
            "csv": str(resolve_path(plan_csv)),
            "json": str(resolve_path(plan_json)),
        }
        write_json(plan_json, adjustment_summary)

    if not import_ready:
        decision = "blocked_invalid_verdicts"
    elif confirmed_bad_rows == 0 and int(import_status_counts.get("label_correct_model_blindspot", 0)) == 0:
        decision = "ready_noop_no_actionable_verdicts"
    elif target_min_fixed and confirmed_bad_rows >= target_min_fixed:
        decision = "target_feasibility_covered_but_requires_redraw_eval"
    elif confirmed_bad_rows > 0:
        decision = "ready_for_fresh_redraw_target_gap_not_covered"
    else:
        decision = "target_gap_not_covered"

    held_out_test_verdict_only_rows = 0
    training_policy_rows = 0
    replacement_counts_by_original_label: dict[str, int] = {}
    if adjustment_summary is not None:
        held_out_test_verdict_only_rows = int(adjustment_summary.get("action_counts", {}).get("held_out_test_verdict_only", 0))
        training_policy_rows = int(adjustment_summary.get("training_policy_rows", 0))
        replacement_counts_by_original_label = dict(adjustment_summary.get("replacement_counts_by_original_label", {}))

    summary = {
        "schema": "axon_loop74_external_verdict_import_v1",
        "protocol": "strict external/manual verdict import; no model fitting, no threshold selection, no automatic split mutation",
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "replacement_rule": REPLACEMENT_RULE,
        "decision": decision,
        "current_best": current_best_summary(target_gap),
        "target": target_info,
        "review_csv": str(resolve_path(review_csv)),
        "split_csv": str(resolve_path(split_csv)),
        "target_gap_json": str(resolve_path(target_gap_json)) if target_gap_json is not None else None,
        "output_csv": str(resolve_path(output_csv)),
        "output_json": str(resolve_path(output_json)),
        "plan_csv": str(resolve_path(plan_csv)) if plan_csv is not None else None,
        "plan_json": str(resolve_path(plan_json)) if plan_json is not None else None,
        "allow_partial": bool(allow_partial),
        "expected_rows": expected_rows,
        "allow_test_actions": bool(allow_test_actions),
        "enforce_20w_split": bool(enforce_20w_split),
        "import_ready": import_ready,
        "blocking_issues": sorted(set(blocking_issues)),
        "missing_required_columns": missing_columns,
        "review_rows": len(review_rows),
        "input_alignment": {
            "review_rows": len(review_rows),
            "expected_rows": expected_rows,
            "matched_loop72_rows": len(review_rows) if expected_rows is None or len(review_rows) == expected_rows else min(len(review_rows), expected_rows),
            "missing_loop72_rows": max(0, int(expected_rows or len(review_rows)) - len(review_rows)),
            "missing_split_rows": int(row_issue_counts.get("sample_index_not_found_in_split", 0)),
            "duplicate_review_rows": duplicate_review_sample_index_rows,
            "sample_index_match_count": sample_index_match_count,
            "sha_match_count": sha_match_count,
            "split_match_count": rows_with_split_match,
        },
        "manual_content_rows": manual_content_rows,
        "blank_or_no_decision_rows": int(import_status_counts.get("no_decision", 0)),
        "invalid_rows": invalid_rows,
        "actionable_or_reviewed_rows": actionable_rows,
        "duplicate_review_sample_index_rows": duplicate_review_sample_index_rows,
        "duplicate_review_sample_index_examples": duplicate_review_sample_index_examples,
        "split_summary": split_summary,
        "split_counts": split_summary["split_counts"],
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "recommended_action_counts": dict(sorted(recommended_action_counts.items())),
        "manual_quality": {
            "blank_verdict_rows": int(verdict_counts.get("blank", 0)),
            "invalid_verdict_rows": int(row_issue_counts.get("invalid_manual_label_verdict", 0)),
            "blank_action_rows": int(recommended_action_counts.get("blank", 0)),
            "invalid_action_rows": int(row_issue_counts.get("invalid_recommended_action", 0)),
            "manual_fields_inconsistent_rows": int(
                sum(
                    count
                    for issue, count in row_issue_counts.items()
                    if issue
                    in {
                        "uncertain_verdict_requires_needs_more_evidence_or_blank_action",
                        "label_correct_requires_keep_label_or_model_blindspot",
                        "label_wrong_requires_replace_or_quarantine_action",
                        "feature_or_scope_issue_requires_replace_or_quarantine_action",
                    }
                )
            ),
            "label_wrong_missing_corrected_label_rows": int(row_issue_counts.get("label_wrong_requires_corrected_label", 0)),
            "actionable_verdict_missing_note_rows": int(row_issue_counts.get("actionable_verdict_requires_manual_verdict_note", 0)),
            "evidence_note_missing_content_or_external_rows": int(
                row_issue_counts.get("manual_verdict_note_missing_content_or_external_evidence", 0)
            ),
            "evidence_note_identity_or_score_only_rows": int(
                row_issue_counts.get("manual_verdict_note_identity_or_score_only", 0)
            ),
        },
        "actionable_counts": {
            "actionable_rows": actionable_rows,
            "no_op_rows": no_op_rows,
            "held_out_test_verdict_only_rows": held_out_test_verdict_only_rows,
            "replacement_required": int((adjustment_summary or {}).get("replacement_required", confirmed_bad_rows)),
        },
        "replacement_counts_by_original_label": replacement_counts_by_original_label,
        "confirmed_bad_rows": {
            "total": confirmed_bad_rows,
            "label_wrong": label_wrong_rows,
            "feature_broken_or_out_of_scope": exclude_rows,
            "by_loop57_error_type": dict(sorted(confirmed_bad_by_error_type.items())),
        },
        "confirmed_model_blindspot_rows": int(import_status_counts.get("label_correct_model_blindspot", 0)),
        "target_feasibility": {
            "confirmed_bad_rows": confirmed_bad_rows,
            "minimum_fixed_errors_best_case": target_min_fixed,
            "confirmed_bad_rows_to_required_ratio": target_gap_coverage,
            "target_gap_covered_by_confirmed_bad_rows": bool(target_min_fixed and confirmed_bad_rows >= target_min_fixed),
        },
        "training_policy_rows": training_policy_rows,
        "import_status_counts": dict(sorted(import_status_counts.items())),
        "metric_effect_counts": dict(sorted(metric_effect_counts.items())),
        "split_action_counts": dict(sorted(split_action_counts.items())),
        "row_issue_counts": dict(sorted(row_issue_counts.items())),
        "target_gap_metrics": target_gap_metrics(
            target_gap=target_gap,
            validated_rows=validated_rows,
            corrected_labels=corrected_labels,
        ),
        "adjustment_plan_summary": adjustment_summary,
        "notes": [
            "sample_index is required to preserve row-level identity for duplicate content; it remains an audit field only.",
            "Paths, hashes, split names, and row order are not accepted as label evidence or model features.",
            "If import_ready=false, do not use the output CSV for split adjustment or training policy.",
            "Test split verdicts remain held out of train/val policy unless --allow-test-actions is explicitly set.",
        ],
    }
    write_json(output_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strictly validate filled Loop72 external verdicts.")
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--target-gap-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--plan-csv", type=Path, default=None)
    parser.add_argument("--plan-json", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--skip-20w-split-check", action="store_true")
    parser.add_argument(
        "--allow-test-actions",
        action="store_true",
        help="Allow test verdict actions in the adjustment plan. Training policy remains false for test rows.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.plan_csv is None) != (args.plan_json is None):
        raise SystemExit("--plan-csv and --plan-json must be provided together.")
    summary = validate_loop72_external_verdicts(
        review_csv=args.review_csv,
        split_csv=args.split_csv,
        target_gap_json=args.target_gap_json,
        output_csv=args.output_csv,
        output_json=args.output_json,
        plan_csv=args.plan_csv,
        plan_json=args.plan_json,
        allow_partial=bool(args.allow_partial),
        expected_rows=args.expected_rows,
        enforce_20w_split=not bool(args.skip_20w_split_check),
        allow_test_actions=bool(args.allow_test_actions),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["import_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
