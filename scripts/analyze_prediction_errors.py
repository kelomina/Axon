#!/usr/bin/env python3
"""Summarize false positives and false negatives from exported predictions."""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from raw_group_tools import read_csv_rows, resolve_path, write_csv  # noqa: E402


ERROR_COLUMNS = [
    "error_type",
    "source_path",
    "cache_path",
    "sample_index",
    "group_id",
    "source_group_id",
    "group_size",
    "sample_weight",
    "hard_family_role",
    "is_rare_group",
    "group_source",
    "label",
    "split",
    "prob_malicious",
    "prediction",
    "margin_to_threshold",
]

PROBABILITY_COLUMNS = [
    "blend_prob_malicious",
    "prob_malicious",
    "stage2_prob_malicious",
    "calibrator_prob_malicious",
    "baseline_prob_malicious",
]

GROUP_COLUMNS = [
    "group_id",
    "error_count",
    "fp_count",
    "fn_count",
    "label_counts",
    "avg_prob_malicious",
    "min_prob_malicious",
    "max_prob_malicious",
    "group_size",
    "is_rare_group",
    "group_source",
    "hard_family_roles",
    "example_paths",
]


BREAKDOWN_COLUMNS = [
    "dimension",
    "value",
    "error_count",
    "fp_count",
    "fn_count",
    "avg_prob_malicious",
    "min_prob_malicious",
    "max_prob_malicious",
]


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0) -> int:
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


def _prediction_at_threshold(row: dict, threshold: float) -> int:
    return 1 if _probability(row) >= threshold else 0


def _error_type(row: dict, threshold: float) -> str:
    label = _safe_int(row.get("label"))
    pred = _prediction_at_threshold(row, threshold)
    if label == 0 and pred == 1:
        return "FP"
    if label == 1 and pred == 0:
        return "FN"
    return ""


def _counter_to_text(counter: Counter) -> str:
    return "|".join(f"{key}:{counter[key]}" for key in sorted(counter))


def _path_parts(row: dict) -> list[str]:
    source_path = str(row.get("source_path", ""))
    return [part for part in re.split(r"[\\/]+", source_path) if part]


def _extension_bucket(row: dict) -> str:
    suffix = Path(str(row.get("source_path", ""))).suffix.casefold()
    return suffix or "<none>"


def _data_dir_bucket(row: dict) -> str:
    parts = _path_parts(row)
    lowered = [part.casefold() for part in parts]
    if "data" in lowered:
        index = lowered.index("data")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "<unknown>"


def _month_bucket(row: dict) -> str:
    match = re.search(r"20\d{2}-\d{2}", str(row.get("source_path", "")))
    return match.group(0) if match else "<none>"


def _group_size_bucket(row: dict) -> str:
    size = _safe_int(row.get("group_size"), 0)
    if size <= 0:
        return "<unknown>"
    if size == 1:
        return "1"
    if size == 2:
        return "2"
    if size <= 5:
        return "3-5"
    if size <= 10:
        return "6-10"
    return ">10"


def _prob_bucket(row: dict, threshold: float) -> str:
    prob = _probability(row)
    if row["error_type"] == "FP":
        if prob >= 0.90:
            return "fp:>=0.90"
        if prob >= 0.75:
            return "fp:0.75-0.90"
        return f"fp:{threshold:.2f}-0.75"
    if prob < 0.10:
        return "fn:<0.10"
    if prob < 0.30:
        return "fn:0.10-0.30"
    if prob < 0.45:
        return "fn:0.30-0.45"
    return f"fn:0.45-{threshold:.2f}"


def _build_breakdown(error_rows: Sequence[dict], threshold: float) -> list[dict]:
    dimensions = {
        "extension": _extension_bucket,
        "data_dir": _data_dir_bucket,
        "group_source": lambda row: row.get("group_source") or "<unknown>",
        "is_rare_group": lambda row: row.get("is_rare_group") or "<unknown>",
        "group_size_bucket": _group_size_bucket,
        "hard_family_role": lambda row: row.get("hard_family_role") or "<unknown>",
        "month": _month_bucket,
        "prob_bucket": lambda row: _prob_bucket(row, threshold),
    }
    rows = []
    for dimension, key_fn in dimensions.items():
        grouped = defaultdict(list)
        for row in error_rows:
            grouped[str(key_fn(row))].append(row)
        for value, items in grouped.items():
            probs = [_probability(row) for row in items]
            rows.append({
                "dimension": dimension,
                "value": value,
                "error_count": len(items),
                "fp_count": sum(1 for row in items if row["error_type"] == "FP"),
                "fn_count": sum(1 for row in items if row["error_type"] == "FN"),
                "avg_prob_malicious": mean(probs) if probs else "",
                "min_prob_malicious": min(probs) if probs else "",
                "max_prob_malicious": max(probs) if probs else "",
            })
    rows.sort(key=lambda row: (row["dimension"], -int(row["error_count"]), row["value"]))
    return rows


def analyze_errors(predictions_path: Path, output_dir: Path, threshold: float) -> dict:
    rows = read_csv_rows(resolve_path(predictions_path))
    error_rows = []
    for row in rows:
        error_type = _error_type(row, threshold)
        if not error_type:
            continue
        prob = _probability(row)
        enriched = dict(row)
        enriched["error_type"] = error_type
        enriched["prob_malicious"] = prob
        enriched["prediction"] = _prediction_at_threshold(row, threshold)
        enriched["margin_to_threshold"] = abs(prob - threshold)
        error_rows.append(enriched)

    fp_rows = [row for row in error_rows if row["error_type"] == "FP"]
    fn_rows = [row for row in error_rows if row["error_type"] == "FN"]

    def sort_key(row: dict):
        prob = _probability(row)
        if row["error_type"] == "FP":
            return (-prob, row.get("source_path", ""))
        return (prob, row.get("source_path", ""))

    fp_rows = sorted(fp_rows, key=sort_key)
    fn_rows = sorted(fn_rows, key=sort_key)
    error_rows = sorted(error_rows, key=lambda row: (row["error_type"], sort_key(row)))

    group_map = defaultdict(list)
    for row in error_rows:
        group_map[str(row.get("group_id") or "unknown")].append(row)

    group_rows = []
    for group_id, items in group_map.items():
        probs = [_probability(row) for row in items]
        label_counts = Counter(str(row.get("label", "")) for row in items)
        roles = Counter(str(row.get("hard_family_role", "")) for row in items if row.get("hard_family_role"))
        fp_count = sum(1 for row in items if row["error_type"] == "FP")
        fn_count = sum(1 for row in items if row["error_type"] == "FN")
        first = items[0]
        group_rows.append({
            "group_id": group_id,
            "error_count": len(items),
            "fp_count": fp_count,
            "fn_count": fn_count,
            "label_counts": _counter_to_text(label_counts),
            "avg_prob_malicious": mean(probs) if probs else "",
            "min_prob_malicious": min(probs) if probs else "",
            "max_prob_malicious": max(probs) if probs else "",
            "group_size": first.get("group_size", ""),
            "is_rare_group": first.get("is_rare_group", ""),
            "group_source": first.get("group_source", ""),
            "hard_family_roles": _counter_to_text(roles),
            "example_paths": " | ".join(row.get("source_path", "") for row in items[:3]),
        })
    group_rows.sort(key=lambda row: (-int(row["error_count"]), -int(row["fn_count"]), -int(row["fp_count"]), row["group_id"]))
    breakdown_rows = _build_breakdown(error_rows, threshold)

    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_csv = output_dir / "prediction_errors.csv"
    fp_csv = output_dir / "false_positives.csv"
    fn_csv = output_dir / "false_negatives.csv"
    group_csv = output_dir / "error_groups.csv"
    breakdown_csv = output_dir / "error_breakdown.csv"
    summary_json = output_dir / "prediction_error_summary.json"

    write_csv(errors_csv, error_rows, ERROR_COLUMNS)
    write_csv(fp_csv, fp_rows, ERROR_COLUMNS)
    write_csv(fn_csv, fn_rows, ERROR_COLUMNS)
    write_csv(group_csv, group_rows, GROUP_COLUMNS)
    write_csv(breakdown_csv, breakdown_rows, BREAKDOWN_COLUMNS)

    prob_fp = [_probability(row) for row in fp_rows]
    prob_fn = [_probability(row) for row in fn_rows]
    summary = {
        "predictions": str(resolve_path(predictions_path)),
        "threshold": threshold,
        "total_predictions": len(rows),
        "error_count": len(error_rows),
        "false_positive_count": len(fp_rows),
        "false_negative_count": len(fn_rows),
        "fp_prob": {
            "avg": mean(prob_fp) if prob_fp else None,
            "min": min(prob_fp) if prob_fp else None,
            "max": max(prob_fp) if prob_fp else None,
        },
        "fn_prob": {
            "avg": mean(prob_fn) if prob_fn else None,
            "min": min(prob_fn) if prob_fn else None,
            "max": max(prob_fn) if prob_fn else None,
        },
        "top_error_groups": group_rows[:20],
        "top_breakdowns": {
            dimension: [
                row for row in breakdown_rows
                if row["dimension"] == dimension
            ][:10]
            for dimension in sorted({row["dimension"] for row in breakdown_rows})
        },
        "outputs": {
            "errors_csv": str(errors_csv),
            "false_positives_csv": str(fp_csv),
            "false_negatives_csv": str(fn_csv),
            "error_groups_csv": str(group_csv),
            "error_breakdown_csv": str(breakdown_csv),
            "summary_json": str(summary_json),
        },
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Summarize prediction FP/FN errors.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = analyze_errors(args.predictions, args.output_dir, args.threshold)
    print("=" * 60)
    print("Prediction Error Analysis")
    print("=" * 60)
    print(f"Total predictions: {summary['total_predictions']}")
    print(f"False positives: {summary['false_positive_count']}")
    print(f"False negatives: {summary['false_negative_count']}")
    print(f"Report: {summary['outputs']['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
