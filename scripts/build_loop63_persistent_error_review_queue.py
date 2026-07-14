#!/usr/bin/env python3
"""Build a review queue for persistent current-best full-test errors.

This is read-only data/noise triage. Full-test rows are used only to prioritize
manual or external-evidence review and target-feasibility analysis. They must
not be used for model fitting, threshold selection, or feature engineering.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


ALLOWED_VERDICTS = "label_correct|label_wrong|feature_broken|out_of_scope|uncertain"
ALLOWED_ACTIONS = "keep_sample|replace_with_fresh_same_label_candidate|quarantine_for_more_evidence|model_blindspot"
REPLACEMENT_RULE = (
    "If feature_broken/out_of_scope/label_wrong is confirmed, do not fill from this row. "
    "Re-sample one fresh valid candidate from the same original-label pool and preserve the exact 200000-row split."
)

FIELDNAMES = [
    "review_priority_rank",
    "review_lane",
    "priority_reason",
    "exchange_group",
    "in_loop39_conflict_queue",
    "loop39_review_lane",
    "loop39_conflict_bucket",
    "loop39_review_priority_rank",
    "loop39_corrected_by_any_compared_model",
    "allowed_manual_label_verdicts",
    "allowed_recommended_actions",
    "replacement_rule",
    "source_path",
    "cache_path",
    "source_sha256",
    "sample_index",
    "split",
    "label",
    "loop57_error_type",
    "loop57_final_prob",
    "loop57_base_prob",
    "loop57_candidate_prob",
    "loop57_gate_prob",
    "loop57_prediction",
    "loop57_fn_override",
    "loop28_prob",
    "loop28_prediction",
    "loop28_correct",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
]


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _key(row: dict) -> tuple[str, str]:
    sha = str(row.get("source_sha256") or "").strip().casefold()
    if sha:
        return "source_sha256", sha
    sample_index = str(row.get("sample_index") or "").strip()
    if sample_index:
        return "sample_index", sample_index
    path = str(row.get("source_path") or "").strip().replace("\\", "/").casefold()
    if path:
        return "source_path", path
    return "", ""


def _index(rows: Sequence[dict]) -> dict[tuple[str, str], dict]:
    indexed = {}
    for row in rows:
        key = _key(row)
        if key[0]:
            indexed[key] = row
    return indexed


def _prob(row: dict, *columns: str) -> float:
    for column in columns:
        if column in row and str(row.get(column, "")).strip() != "":
            return _float(row[column])
    return 0.0


def _error_type(label: int, prediction: int) -> str:
    if label == 0 and prediction == 1:
        return "FP"
    if label == 1 and prediction == 0:
        return "FN"
    return ""


def _priority_reason(error_type: str, prob: float) -> tuple[int, str]:
    if error_type == "FN":
        if prob <= 0.01:
            return 0, "severe_fn_prob_le_0.01"
        if prob <= 0.05:
            return 1, "high_fn_prob_le_0.05"
        if prob <= 0.15:
            return 2, "medium_fn_prob_le_0.15"
        return 3, "lower_confidence_fn"
    if prob >= 0.99:
        return 0, "severe_fp_prob_ge_0.99"
    if prob >= 0.95:
        return 1, "high_fp_prob_ge_0.95"
    if prob >= 0.85:
        return 2, "medium_fp_prob_ge_0.85"
    return 3, "lower_confidence_fp"


def _exchange_group(loop28_correct: Optional[bool], loop57_correct: bool) -> str:
    if loop28_correct is None:
        return "loop28_missing"
    if not loop28_correct and not loop57_correct:
        return "loop28_loop57_both_error"
    if loop28_correct and not loop57_correct:
        return "loop57_new_error"
    if not loop28_correct and loop57_correct:
        return "loop57_repaired_loop28_error"
    return "both_correct"


def _review_lane(exchange_group: str, in_conflict_queue: bool) -> str:
    if exchange_group == "loop28_loop57_both_error" and in_conflict_queue:
        return "A_persistent_error_in_high_conflict_queue"
    if exchange_group == "loop28_loop57_both_error":
        return "B_persistent_error"
    if exchange_group == "loop57_new_error" and in_conflict_queue:
        return "C_loop57_new_error_in_high_conflict_queue"
    if exchange_group == "loop57_new_error":
        return "D_loop57_new_error"
    return "E_other"


def _lane_order(lane: str) -> int:
    order = {
        "A_persistent_error_in_high_conflict_queue": 0,
        "B_persistent_error": 1,
        "C_loop57_new_error_in_high_conflict_queue": 2,
        "D_loop57_new_error": 3,
        "E_other": 9,
    }
    return order.get(lane, 9)


def _confidence_for_sort(error_type: str, prob: float) -> float:
    return (1.0 - prob) if error_type == "FN" else prob


def build_queue(
    *,
    loop57_predictions: Path,
    loop28_predictions: Path,
    loop39_conflict_queue: Path,
    output_csv: Path,
    output_json: Path,
    max_examples: int,
) -> dict:
    loop57_rows = read_rows(loop57_predictions)
    loop28_by_key = _index(read_rows(loop28_predictions))
    conflict_by_key = _index(read_rows(loop39_conflict_queue))

    output_rows = []
    missing_loop28 = 0
    for row in loop57_rows:
        label = _int(row.get("label"))
        prediction = _int(row.get("prediction"))
        if label == prediction:
            continue

        key = _key(row)
        row28 = loop28_by_key.get(key)
        conflict = conflict_by_key.get(key)
        if row28 is None:
            missing_loop28 += 1
        loop28_correct = None if row28 is None else _bool(row28.get("correct"))
        loop57_correct = _bool(row.get("correct"))
        exchange_group = _exchange_group(loop28_correct, loop57_correct)
        final_prob = _prob(row, "final_prob_malicious", "stage2_prob_malicious", "prob_malicious")
        error_type = _error_type(label, prediction)
        priority_value, reason = _priority_reason(error_type, final_prob)
        in_conflict = conflict is not None
        lane = _review_lane(exchange_group, in_conflict)

        output_rows.append(
            {
                "review_lane": lane,
                "priority_reason": reason,
                "priority_value": priority_value,
                "exchange_group": exchange_group,
                "in_loop39_conflict_queue": str(in_conflict),
                "loop39_review_lane": "" if conflict is None else conflict.get("review_lane", ""),
                "loop39_conflict_bucket": "" if conflict is None else conflict.get("conflict_bucket", ""),
                "loop39_review_priority_rank": "" if conflict is None else conflict.get("review_priority_rank", ""),
                "loop39_corrected_by_any_compared_model": ""
                if conflict is None
                else conflict.get("corrected_by_any_compared_model", ""),
                "allowed_manual_label_verdicts": ALLOWED_VERDICTS,
                "allowed_recommended_actions": ALLOWED_ACTIONS,
                "replacement_rule": REPLACEMENT_RULE,
                "source_path": row.get("source_path", ""),
                "cache_path": row.get("cache_path", ""),
                "source_sha256": row.get("source_sha256", ""),
                "sample_index": row.get("sample_index", ""),
                "split": row.get("split", ""),
                "label": str(label),
                "loop57_error_type": error_type,
                "loop57_final_prob": f"{final_prob:.10f}",
                "loop57_base_prob": f"{_prob(row, 'base_prob_malicious'):.10f}",
                "loop57_candidate_prob": f"{_prob(row, 'candidate_prob_malicious'):.10f}",
                "loop57_gate_prob": f"{_prob(row, 'gate_prob_override'):.10f}",
                "loop57_prediction": str(prediction),
                "loop57_fn_override": str(_bool(row.get("fn_override"))),
                "loop28_prob": "" if row28 is None else f"{_prob(row28, 'stage2_prob_malicious'):.10f}",
                "loop28_prediction": "" if row28 is None else str(_int(row28.get("prediction"))),
                "loop28_correct": "" if row28 is None else str(loop28_correct),
                "manual_label_verdict": "",
                "manual_verdict_note": "",
                "recommended_action": "",
            }
        )

    output_rows.sort(
        key=lambda row: (
            _lane_order(row["review_lane"]),
            int(row["priority_value"]),
            0 if row["loop57_error_type"] == "FN" else 1,
            -_confidence_for_sort(row["loop57_error_type"], _float(row["loop57_final_prob"])),
            _int(row.get("loop39_review_priority_rank"), 999999),
            row.get("source_sha256", ""),
        )
    )
    for rank, row in enumerate(output_rows, start=1):
        row["review_priority_rank"] = str(rank)

    write_rows(output_csv, output_rows)
    summary = {
        "schema": "axon_loop63_persistent_error_review_queue_v1",
        "protocol": (
            "read-only full-test error triage; no model fitting, no threshold selection, "
            "no automatic relabeling, and no split mutation"
        ),
        "identity_feature_policy": (
            "source_path/source_sha256/cache_path/sample_index/split are review, alignment, and cache fields only; "
            "they are not model evidence"
        ),
        "loop57_predictions": str(loop57_predictions),
        "loop28_predictions": str(loop28_predictions),
        "loop39_conflict_queue": str(loop39_conflict_queue),
        "rows_total": len(loop57_rows),
        "loop57_error_rows": len(output_rows),
        "missing_loop28_alignment": missing_loop28,
        "review_lane_counts": dict(sorted(Counter(row["review_lane"] for row in output_rows).items())),
        "exchange_group_counts": dict(sorted(Counter(row["exchange_group"] for row in output_rows).items())),
        "error_type_counts": dict(sorted(Counter(row["loop57_error_type"] for row in output_rows).items())),
        "priority_reason_counts": dict(sorted(Counter(row["priority_reason"] for row in output_rows).items())),
        "loop39_intersection_rows": sum(1 for row in output_rows if row["in_loop39_conflict_queue"] == "True"),
        "manual_label_verdict_blank_count": sum(1 for row in output_rows if not row["manual_label_verdict"]),
        "recommended_action_blank_count": sum(1 for row in output_rows if not row["recommended_action"]),
        "replacement_rule": REPLACEMENT_RULE,
        "outputs": {
            "queue_csv": str(output_csv),
            "summary_json": str(output_json),
        },
        "examples": output_rows[: max(0, int(max_examples))],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build Loop63 persistent current-best error review queue.")
    parser.add_argument("--loop57-predictions", type=Path, required=True)
    parser.add_argument("--loop28-predictions", type=Path, required=True)
    parser.add_argument("--loop39-conflict-queue", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=30)
    args = parser.parse_args(argv)
    summary = build_queue(
        loop57_predictions=args.loop57_predictions,
        loop28_predictions=args.loop28_predictions,
        loop39_conflict_queue=args.loop39_conflict_queue,
        output_csv=args.output_csv,
        output_json=args.output_json,
        max_examples=args.max_examples,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
