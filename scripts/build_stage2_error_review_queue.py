#!/usr/bin/env python3
"""Build a prioritized error-review queue from frozen Stage2 predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "priority",
        "error_type",
        "reason",
        "source_path",
        "cache_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "stage2_prob_malicious",
        "prediction",
        "manual_label_verdict",
        "manual_verdict_note",
        "recommended_action",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def priority_for_error(label: int, prob: float) -> tuple[int, str]:
    if label == 1:
        if prob <= 0.05:
            return 0, "severe_fn_prob_le_0.05"
        if prob <= 0.15:
            return 1, "high_confidence_fn_prob_le_0.15"
        if prob <= 0.35:
            return 2, "mid_confidence_fn_prob_le_0.35"
        return 3, "near_threshold_fn"
    if prob >= 0.95:
        return 0, "severe_fp_prob_ge_0.95"
    if prob >= 0.85:
        return 1, "high_confidence_fp_prob_ge_0.85"
    if prob >= 0.65:
        return 2, "mid_confidence_fp_prob_ge_0.65"
    return 3, "near_threshold_fp"


def build_queue(predictions_csv: Path, output_csv: Path, output_json: Path, max_examples: int) -> dict:
    rows = read_rows(predictions_csv)
    errors: list[dict] = []
    priority_counts: Counter[str] = Counter()
    error_type_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for row in rows:
        label = int(row["label"])
        prediction = int(row["prediction"])
        if label == prediction:
            continue
        prob = float(row["stage2_prob_malicious"])
        priority, reason = priority_for_error(label, prob)
        error_type = "FN" if label == 1 else "FP"
        priority_counts[str(priority)] += 1
        error_type_counts[error_type] += 1
        reason_counts[reason] += 1
        errors.append(
            {
                **row,
                "priority": priority,
                "error_type": error_type,
                "reason": reason,
                "manual_label_verdict": "",
                "manual_verdict_note": "",
                "recommended_action": "",
            }
        )

    errors.sort(
        key=lambda row: (
            int(row["priority"]),
            0 if row["error_type"] == "FN" else 1,
            float(row["stage2_prob_malicious"]) if row["error_type"] == "FN" else -float(row["stage2_prob_malicious"]),
            row.get("source_path", ""),
        )
    )
    write_csv(output_csv, errors)
    summary = {
        "schema": "axon_stage2_error_review_queue_v1",
        "predictions_csv": str(predictions_csv),
        "rows_total": len(rows),
        "errors_total": len(errors),
        "error_type_counts": dict(sorted(error_type_counts.items())),
        "priority_counts": dict(sorted(priority_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "output_csv": str(output_csv),
        "examples": errors[:max(0, int(max_examples))],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a prioritized Stage2 error-review queue.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=30)
    args = parser.parse_args(argv)
    summary = build_queue(args.predictions_csv, args.output_csv, args.output_json, args.max_examples)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
