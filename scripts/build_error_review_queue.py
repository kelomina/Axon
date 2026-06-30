#!/usr/bin/env python3
"""Build a prioritized review queue from validation prediction errors."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

PROBABILITY_COLUMNS = [
    "prob_malicious",
    "stage2_prob_malicious",
    "calibrator_prob_malicious",
    "baseline_prob_malicious",
]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _probability(row: dict) -> float:
    for column in PROBABILITY_COLUMNS:
        value = row.get(column)
        if value not in (None, ""):
            return _safe_float(value)
    return 0.0


def _extension(path_text: str) -> str:
    return Path(path_text).suffix.casefold() or "<none>"


def _month(path_text: str) -> str:
    match = re.search(r"20\d{2}-\d{2}", path_text)
    return match.group(0) if match else "<none>"


def _data_dir(path_text: str) -> str:
    parts = [part for part in re.split(r"[\\/]+", path_text) if part]
    lowered = [part.casefold() for part in parts]
    if "data" in lowered:
        index = lowered.index("data")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "<unknown>"


def _priority(label: int, prob: float, threshold: float) -> tuple[int, str]:
    if label == 0 and prob >= 0.99:
        return 0, "severe_fp_label0_prob_ge_0.99"
    if label == 1 and prob <= 0.01:
        return 0, "severe_fn_label1_prob_le_0.01"
    if label == 0 and prob >= 0.95:
        return 1, "high_fp_label0_prob_ge_0.95"
    if label == 1 and prob <= 0.05:
        return 1, "high_fn_label1_prob_le_0.05"
    if abs(prob - threshold) <= 0.05:
        return 3, "near_threshold_error_le_0.05"
    if abs(prob - threshold) <= 0.10:
        return 4, "near_threshold_error_le_0.10"
    return 2, "mid_confidence_structural_error"


def build_review_queue(predictions: Path, threshold: float, output_csv: Path, output_json: Path) -> dict:
    with predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    queue = []
    for row in rows:
        label = _safe_int(row.get("label"))
        prob = _probability(row)
        pred = int(prob >= threshold)
        if pred == label:
            continue
        error_type = "FP" if label == 0 else "FN"
        priority, reason = _priority(label, prob, threshold)
        source_path = row.get("source_path", "")
        path = Path(source_path)
        file_size = path.stat().st_size if path.exists() else ""
        queue.append(
            {
                "priority": priority,
                "reason": reason,
                "error_type": error_type,
                "source_path": source_path,
                "cache_path": row.get("cache_path", ""),
                "source_sha256": row.get("source_sha256", ""),
                "label": label,
                "prediction": pred,
                "prob_malicious": f"{prob:.10f}",
                "distance_to_threshold": f"{abs(prob - threshold):.10f}",
                "split": row.get("split", ""),
                "sample_index": row.get("sample_index", ""),
                "data_dir": _data_dir(source_path),
                "extension": _extension(source_path),
                "month": _month(source_path),
                "file_size": file_size,
            }
        )

    queue.sort(
        key=lambda row: (
            int(row["priority"]),
            row["error_type"],
            -abs(float(row["prob_malicious"]) - threshold),
            row["source_path"],
        )
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "priority",
        "reason",
        "error_type",
        "source_path",
        "cache_path",
        "source_sha256",
        "label",
        "prediction",
        "prob_malicious",
        "distance_to_threshold",
        "split",
        "sample_index",
        "data_dir",
        "extension",
        "month",
        "file_size",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(queue)

    priority_counts = Counter(str(row["priority"]) for row in queue)
    reason_counts = Counter(row["reason"] for row in queue)
    error_counts = Counter(row["error_type"] for row in queue)
    summary = {
        "schema": "axon_error_review_queue_v1",
        "predictions": str(predictions),
        "threshold": threshold,
        "total_predictions": len(rows),
        "error_count": len(queue),
        "error_type_counts": dict(sorted(error_counts.items())),
        "priority_counts": dict(sorted(priority_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "top20": queue[:20],
        "outputs": {
            "queue_csv": str(output_csv),
            "summary_json": str(output_json),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a prioritized review queue from prediction errors.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build_review_queue(args.predictions, args.threshold, args.output_csv, args.output_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
