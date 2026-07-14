#!/usr/bin/env python3
"""Summarize high-similarity opposite-label neighbor conflicts."""

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


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def threshold_key(similarity: float, opposite_ratio: float) -> str:
    sim_bucket = "sim_lt_0.90"
    for threshold in (0.90, 0.95, 0.98, 0.99, 0.995):
        if similarity >= threshold:
            sim_bucket = f"sim_ge_{threshold:g}"
    ratio_bucket = "opp_lt_0.80"
    for threshold in (0.80, 0.90, 1.00):
        if opposite_ratio >= threshold:
            ratio_bucket = f"opp_ge_{threshold:g}"
    return f"{sim_bucket}__{ratio_bucket}"


def probability_value(row: dict) -> str:
    return row.get("prob_malicious") or row.get("stage2_prob_malicious") or ""


def score_column_value(row: dict) -> str:
    if row.get("score_column"):
        return row["score_column"]
    if row.get("stage2_prob_malicious"):
        return "stage2_prob_malicious"
    if row.get("prob_malicious"):
        return "prob_malicious"
    return ""


def summarize(input_csv: Path, output_json: Path, output_csv: Path, max_priority: int) -> dict:
    rows = [row for row in read_rows(input_csv) if as_int(row.get("priority", "999"), 999) <= int(max_priority)]
    enriched = []
    bucket_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    support_counts: Counter[str] = Counter()
    high_conflict_rows = []
    for row in rows:
        nearest_similarity = as_float(row.get("nearest_similarity", "0"))
        opposite_ratio = as_float(row.get("opposite_label_ratio", "0"))
        key = threshold_key(nearest_similarity, opposite_ratio)
        bucket_counts[key] += 1
        error_counts[row.get("error_type", "")] += 1
        support_counts[row.get("support_bucket", "")] += 1
        out = {
            **row,
            "prob_malicious": probability_value(row),
            "stage2_prob_malicious": row.get("stage2_prob_malicious") or probability_value(row),
            "score_column": score_column_value(row),
            "conflict_bucket": key,
            "high_similarity_opposite_label_conflict": (
                nearest_similarity >= 0.95
                and opposite_ratio >= 0.8
                and row.get("support_bucket") == "neighbors_support_model_prediction"
            ),
        }
        enriched.append(out)
        if out["high_similarity_opposite_label_conflict"]:
            high_conflict_rows.append(out)

    high_conflict_rows.sort(
        key=lambda row: (
            row.get("error_type", ""),
            -as_float(row.get("nearest_similarity", "0")),
            -as_float(row.get("opposite_label_ratio", "0")),
            row.get("source_path", ""),
        )
    )
    fieldnames = [
        "conflict_bucket",
        "high_similarity_opposite_label_conflict",
        "support_bucket",
        "priority",
        "reason",
        "error_type",
        "source_path",
        "source_sha256",
        "label",
        "prediction",
        "prob_malicious",
        "score_column",
        "stage2_prob_malicious",
        "base_prob_malicious",
        "neighbor_label_counts",
        "opposite_label_ratio",
        "nearest_similarity",
        "top5_neighbor_labels",
        "top5_neighbor_similarities",
        "top5_neighbor_sha256",
        "top5_neighbor_paths",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(high_conflict_rows)

    summary = {
        "schema": "axon_neighbor_label_conflict_summary_v1",
        "input_csv": str(input_csv),
        "max_priority": int(max_priority),
        "rows_selected": len(rows),
        "error_type_counts": dict(sorted(error_counts.items())),
        "support_bucket_counts": dict(sorted(support_counts.items())),
        "conflict_bucket_counts": dict(sorted(bucket_counts.items())),
        "high_similarity_opposite_label_conflict": {
            "definition": "nearest_similarity >= 0.95 and opposite_label_ratio >= 0.8 and support_bucket == neighbors_support_model_prediction",
            "count": len(high_conflict_rows),
            "error_type_counts": dict(sorted(Counter(row.get("error_type", "") for row in high_conflict_rows).items())),
        },
        "output_csv": str(output_csv),
        "examples": high_conflict_rows[:20],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize nearest-neighbor label conflicts.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-priority", type=int, default=1)
    args = parser.parse_args(argv)
    summary = summarize(args.input_csv, args.output_json, args.output_csv, args.max_priority)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
