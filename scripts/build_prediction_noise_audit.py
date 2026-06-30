#!/usr/bin/env python3
"""Build a noise and hard-example audit from exported prediction rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

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


def _prediction(row: dict, threshold: float) -> int:
    return int(_probability(row) >= threshold)


def _error_type(row: dict, threshold: float) -> str:
    label = _safe_int(row.get("label"))
    pred = _prediction(row, threshold)
    if label == 0 and pred == 1:
        return "FP"
    if label == 1 and pred == 0:
        return "FN"
    return ""


def _data_dir(row: dict) -> str:
    parts = [part for part in re.split(r"[\\/]+", str(row.get("source_path", ""))) if part]
    lowered = [part.casefold() for part in parts]
    if "data" in lowered:
        index = lowered.index("data")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "<unknown>"


def _month(row: dict) -> str:
    match = re.search(r"20\d{2}-\d{2}", str(row.get("source_path", "")))
    return match.group(0) if match else "<none>"


def _extension(row: dict) -> str:
    return Path(str(row.get("source_path", ""))).suffix.casefold() or "<none>"


def _bucket_counts(rows: Iterable[dict], threshold: float) -> list[dict]:
    dimensions = {
        "data_dir": _data_dir,
        "extension": _extension,
        "month": _month,
        "error_type": lambda row: row.get("error_type") or "correct",
        "noise_bucket": lambda row: row.get("noise_bucket") or "none",
    }
    output = []
    for dimension, key_fn in dimensions.items():
        groups = defaultdict(list)
        for row in rows:
            groups[str(key_fn(row))].append(row)
        for value, items in groups.items():
            probs = [_probability(row) for row in items]
            output.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "count": len(items),
                    "fp_count": sum(1 for row in items if row.get("error_type") == "FP"),
                    "fn_count": sum(1 for row in items if row.get("error_type") == "FN"),
                    "avg_prob_malicious": mean(probs) if probs else None,
                    "min_prob_malicious": min(probs) if probs else None,
                    "max_prob_malicious": max(probs) if probs else None,
                }
            )
    output.sort(key=lambda row: (row["dimension"], -int(row["count"]), row["value"]))
    return output


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_audit(predictions: Path, output_dir: Path, threshold: float) -> dict:
    rows = read_rows(predictions)
    enriched = []
    for row in rows:
        prob = _probability(row)
        label = _safe_int(row.get("label"))
        error_type = _error_type(row, threshold)
        distance = abs(prob - threshold)
        noise_bucket = "none"
        if label == 0 and prob >= 0.99:
            noise_bucket = "severe_fp_conflict_prob_ge_0.99"
        elif label == 1 and prob <= 0.01:
            noise_bucket = "severe_fn_conflict_prob_le_0.01"
        elif label == 0 and prob >= 0.95:
            noise_bucket = "high_fp_conflict_prob_ge_0.95"
        elif label == 1 and prob <= 0.05:
            noise_bucket = "high_fn_conflict_prob_le_0.05"
        elif error_type and distance <= 0.05:
            noise_bucket = "near_threshold_error_le_0.05"
        elif error_type and distance <= 0.10:
            noise_bucket = "near_threshold_error_le_0.10"

        item = dict(row)
        item["error_type"] = error_type
        item["prob_malicious"] = f"{prob:.10f}"
        item["prediction_at_threshold"] = _prediction(row, threshold)
        item["distance_to_threshold"] = f"{distance:.10f}"
        item["noise_bucket"] = noise_bucket
        enriched.append(item)

    errors = [row for row in enriched if row["error_type"]]
    suspected = [row for row in enriched if row["noise_bucket"] != "none"]
    bucket_counter = Counter(row["noise_bucket"] for row in enriched)
    fp_probs = [_probability(row) for row in errors if row["error_type"] == "FP"]
    fn_probs = [_probability(row) for row in errors if row["error_type"] == "FN"]
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "noise_bucket",
        "error_type",
        "source_path",
        "cache_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "prob_malicious",
        "prediction_at_threshold",
        "distance_to_threshold",
    ]
    write_csv(output_dir / "suspected_noise_and_hard_examples.csv", suspected, fieldnames)
    write_csv(
        output_dir / "noise_breakdown.csv",
        _bucket_counts(enriched, threshold),
        [
            "dimension",
            "value",
            "count",
            "fp_count",
            "fn_count",
            "avg_prob_malicious",
            "min_prob_malicious",
            "max_prob_malicious",
        ],
    )

    summary = {
        "schema": "axon_prediction_noise_audit_v1",
        "predictions": str(predictions),
        "threshold": threshold,
        "samples": len(enriched),
        "errors": len(errors),
        "false_positive_count": sum(1 for row in errors if row["error_type"] == "FP"),
        "false_negative_count": sum(1 for row in errors if row["error_type"] == "FN"),
        "fp_prob": {
            "avg": mean(fp_probs) if fp_probs else None,
            "min": min(fp_probs) if fp_probs else None,
            "max": max(fp_probs) if fp_probs else None,
        },
        "fn_prob": {
            "avg": mean(fn_probs) if fn_probs else None,
            "min": min(fn_probs) if fn_probs else None,
            "max": max(fn_probs) if fn_probs else None,
        },
        "noise_bucket_counts": dict(sorted(bucket_counter.items())),
        "suspected_count": len(suspected),
        "outputs": {
            "suspected_csv": str(output_dir / "suspected_noise_and_hard_examples.csv"),
            "breakdown_csv": str(output_dir / "noise_breakdown.csv"),
            "summary_json": str(output_dir / "noise_audit_summary.json"),
        },
    }
    (output_dir / "noise_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build prediction noise and hard-example audit reports.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build_audit(args.predictions, args.output_dir, args.threshold)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
