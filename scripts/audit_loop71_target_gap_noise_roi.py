#!/usr/bin/env python3
"""Quantify target gap and review ROI after Loop70.

This is a read-only planning audit. It uses full-test errors only for target
feasibility and manual-review prioritization, never for model fitting,
threshold selection, feature engineering, or automatic relabeling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence


REPLACEMENT_RULE = (
    "Confirmed label_wrong/feature_broken/out_of_scope rows must trigger fresh same-original-label redraw. "
    "Do not fill from the bad rows, and preserve exactly 200000 split rows."
)

IDENTITY_FEATURE_POLICY = (
    "filename/path/extension/directory/source hash/cache_path/sample_index/split/row order are loading, "
    "alignment, cache-audit, duplicate-review, and manual-review fields only; they are not model evidence "
    "and must not drive thresholds, relabeling, feature engineering, or production inference"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def _scenario_metrics(tp: int, fp: int, fn: int, fixed_errors: int, *, prefer_fix_fn: bool) -> dict[str, Any]:
    fixed = max(0, min(int(fixed_errors), int(fp + fn)))
    if prefer_fix_fn:
        fix_fn = min(fn, fixed)
        fix_fp = min(fp, fixed - fix_fn)
    else:
        fix_fp = min(fp, fixed)
        fix_fn = min(fn, fixed - fix_fp)
    new_tp = tp + fix_fn
    new_fp = fp - fix_fp
    new_fn = fn - fix_fn
    return {
        "fixed_errors": int(fixed),
        "fixed_fp": int(fix_fp),
        "fixed_fn": int(fix_fn),
        "remaining_errors": int(new_fp + new_fn),
        "fp": int(new_fp),
        "fn": int(new_fn),
        "tp": int(new_tp),
        "f1": f1_from_counts(new_tp, new_fp, new_fn),
    }


def _minimum_fixed_for_target(tp: int, fp: int, fn: int, target_f1: float) -> dict[str, Any]:
    total_errors = fp + fn
    best = None
    for fixed in range(total_errors + 1):
        # FN repairs raise TP, so they are the best possible case for F1.
        scenario = _scenario_metrics(tp, fp, fn, fixed, prefer_fix_fn=True)
        if scenario["f1"] >= target_f1:
            best = scenario
            break
    if best is None:
        best = _scenario_metrics(tp, fp, fn, total_errors, prefer_fix_fn=True)
    best["target_f1"] = float(target_f1)
    best["minimum_fixed_errors_best_case"] = int(best["fixed_errors"])
    return best


def _counter(rows: Sequence[dict], column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(column, "") or "")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _review_roi(
    *,
    review_rows: int,
    current_tp: int,
    current_fp: int,
    current_fn: int,
    target_f1: float,
) -> dict[str, Any]:
    scenarios = {}
    for rate in (0.05, 0.10, 0.25, 0.50, 1.0):
        fixed = math.floor(review_rows * rate)
        scenarios[f"{int(rate * 100)}pct_confirmed_best_case"] = _scenario_metrics(
            current_tp,
            current_fp,
            current_fn,
            fixed,
            prefer_fix_fn=True,
        )
    minimum = _minimum_fixed_for_target(current_tp, current_fp, current_fn, target_f1)
    return {
        "review_rows": int(review_rows),
        "scenarios": scenarios,
        "minimum_fixed_for_target": minimum,
        "review_rows_cover_target_gap": int(review_rows) >= int(minimum["minimum_fixed_errors_best_case"]),
    }


def build_audit(
    *,
    loop57_eval_json: Path,
    loop63_summary_json: Path,
    loop63_queue_csv: Path,
    loop65_summary_json: Path,
    loop65_batch_csv: Path,
    loop50_summary_json: Path,
    loop64_summary_json: Path,
    output_json: Path,
    target_f1: float,
) -> dict[str, Any]:
    loop57 = read_json(loop57_eval_json)
    metrics = loop57["metrics"]
    tp = int(metrics["true_positive"])
    fp = int(metrics["false_positive"])
    fn = int(metrics["false_negative"])
    errors = fp + fn
    samples = int(metrics["samples"])
    current_f1 = float(metrics["f1"])

    loop63 = read_json(loop63_summary_json)
    loop65 = read_json(loop65_summary_json)
    loop50 = read_json(loop50_summary_json)
    loop64 = read_json(loop64_summary_json)
    queue_rows = read_rows(loop63_queue_csv)
    batch_rows = read_rows(loop65_batch_csv)

    target_gap = _minimum_fixed_for_target(tp, fp, fn, float(target_f1))
    remaining_after_review = {
        "loop65_all_selected_best_case": _scenario_metrics(tp, fp, fn, int(loop65["selected_rows"]), prefer_fix_fn=True),
        "loop63_all_errors_best_case": _scenario_metrics(tp, fp, fn, int(loop63["loop57_error_rows"]), prefer_fix_fn=True),
        "loop63_A_lane_best_case": _scenario_metrics(
            tp,
            fp,
            fn,
            int(loop63["review_lane_counts"].get("A_persistent_error_in_high_conflict_queue", 0)),
            prefer_fix_fn=True,
        ),
        "loop50_objective_issues_best_case": _scenario_metrics(
            tp,
            fp,
            fn,
            int(loop50.get("objective_issue_row_count", 0)),
            prefer_fix_fn=True,
        ),
        "loop64_duplicate_detail_rows_best_case": _scenario_metrics(
            tp,
            fp,
            fn,
            int(loop64.get("focus_duplicate_detail_rows", 0)),
            prefer_fix_fn=True,
        ),
    }

    report = {
        "schema": "axon_loop71_target_gap_noise_roi_v1",
        "protocol": (
            "read-only target-gap and noise-review ROI audit; no model fitting, no threshold selection, "
            "no automatic relabeling, no split mutation, no Test-derived feature engineering"
        ),
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "target_f1": float(target_f1),
        "current_best": {
            "source": str(loop57_eval_json),
            "samples": samples,
            "f1": current_f1,
            "errors": errors,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "tn": int(metrics["true_negative"]),
        },
        "target_gap_best_case": target_gap,
        "error_reduction_needed_best_case": int(target_gap["minimum_fixed_errors_best_case"]),
        "error_reduction_needed_ratio_of_current_errors": (
            float(target_gap["minimum_fixed_errors_best_case"] / errors) if errors else 0.0
        ),
        "review_sources": {
            "loop63": {
                "error_rows": int(loop63["loop57_error_rows"]),
                "review_lane_counts": loop63["review_lane_counts"],
                "error_type_counts": loop63["error_type_counts"],
                "loop39_intersection_rows": int(loop63["loop39_intersection_rows"]),
                "manual_blank": int(loop63["manual_label_verdict_blank_count"]),
            },
            "loop65": {
                "selected_rows": int(loop65["selected_rows"]),
                "category_counts": loop65["category_counts"],
                "error_type_counts": loop65["error_type_counts"],
                "manual_fields_blank_output": bool(loop65["manual_fields_blank_output"]),
            },
            "loop50": {
                "rows": int(loop50["rows"]),
                "objective_issue_row_count": int(loop50["objective_issue_row_count"]),
                "issue_counts": loop50.get("issue_counts", {}),
            },
            "loop64": {
                "duplicate_groups": int(loop64["duplicate_groups"]),
                "focus_duplicate_detail_rows": int(loop64["focus_duplicate_detail_rows"]),
                "cross_label_groups": int(loop64["cross_label_groups"]),
                "cross_split_groups": int(loop64["cross_split_groups"]),
            },
        },
        "review_roi": {
            "loop65_selected_batch": _review_roi(
                review_rows=int(loop65["selected_rows"]),
                current_tp=tp,
                current_fp=fp,
                current_fn=fn,
                target_f1=float(target_f1),
            ),
            "loop63_A_lane": _review_roi(
                review_rows=int(loop63["review_lane_counts"].get("A_persistent_error_in_high_conflict_queue", 0)),
                current_tp=tp,
                current_fp=fp,
                current_fn=fn,
                target_f1=float(target_f1),
            ),
            "loop63_all_current_errors": _review_roi(
                review_rows=int(loop63["loop57_error_rows"]),
                current_tp=tp,
                current_fp=fp,
                current_fn=fn,
                target_f1=float(target_f1),
            ),
        },
        "best_case_after_review_sources": remaining_after_review,
        "queue_breakdowns": {
            "loop63_review_lane_counts_from_csv": _counter(queue_rows, "review_lane"),
            "loop63_priority_reason_counts_from_csv": _counter(queue_rows, "priority_reason"),
            "loop65_review_category_counts_from_csv": _counter(batch_rows, "review_category"),
        },
        "decision": {
            "model_search_implication": (
                "Current score/content-stack routes are insufficient. The target requires removing a large majority "
                "of current errors or proving a large label/data-quality issue rate."
            ),
            "noise_implication": (
                "Existing objective audits found too few automatic issues to close the gap. Manual/external review "
                "is still important, but the 62-row Loop65 batch alone cannot materially approach 99.9%."
            ),
            "replacement_rule": REPLACEMENT_RULE,
            "next_action": (
                "Scale external-evidence adjudication for high-conflict rows or introduce a genuinely independent "
                "detector/view; do not continue thin threshold/stacking variants."
            ),
        },
        "artifacts": {
            "output_json": str(output_json),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Loop71 target gap and noise review ROI.")
    parser.add_argument("--loop57-eval-json", type=Path, required=True)
    parser.add_argument("--loop63-summary-json", type=Path, required=True)
    parser.add_argument("--loop63-queue-csv", type=Path, required=True)
    parser.add_argument("--loop65-summary-json", type=Path, required=True)
    parser.add_argument("--loop65-batch-csv", type=Path, required=True)
    parser.add_argument("--loop50-summary-json", type=Path, required=True)
    parser.add_argument("--loop64-summary-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--target-f1", type=float, default=0.999)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_audit(
        loop57_eval_json=args.loop57_eval_json,
        loop63_summary_json=args.loop63_summary_json,
        loop63_queue_csv=args.loop63_queue_csv,
        loop65_summary_json=args.loop65_summary_json,
        loop65_batch_csv=args.loop65_batch_csv,
        loop50_summary_json=args.loop50_summary_json,
        loop64_summary_json=args.loop64_summary_json,
        output_json=args.output_json,
        target_f1=args.target_f1,
    )
    print(
        json.dumps(
            {
                "current_f1": report["current_best"]["f1"],
                "current_errors": report["current_best"]["errors"],
                "target_f1": report["target_f1"],
                "minimum_fixed_errors_best_case": report["error_reduction_needed_best_case"],
                "decision": report["decision"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
