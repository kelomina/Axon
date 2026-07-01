#!/usr/bin/env python3
"""Compare FP/FN overlap across multiple prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_prediction(text: str) -> tuple[str, Path, str, float]:
    parts = text.split("=", 3)
    if len(parts) != 4:
        raise ValueError(f"Expected name=path=score_column=threshold, got: {text}")
    return parts[0], Path(parts[1]), parts[2], float(parts[3])


def parse_key_columns(text: str) -> list[str]:
    columns = [item.strip() for item in text.split(",") if item.strip()]
    if not columns:
        raise ValueError("At least one key column is required")
    return columns


def make_key(row: dict, key_columns: Sequence[str]) -> str:
    values = [str(row.get(column) or "").casefold() for column in key_columns]
    if all(values):
        return "\x1f".join(values)
    for fallback in ("sample_index", "source_sha256", "source_path"):
        value = str(row.get(fallback) or "")
        if value:
            return value.casefold()
    raise ValueError("Prediction row has no usable key")


def error_type(label: int, prediction: int) -> str:
    if label == 0 and prediction == 1:
        return "FP"
    if label == 1 and prediction == 0:
        return "FN"
    return ""


def read_prediction_csv(path: Path, score_column: str, threshold: float, key_columns: Sequence[str]) -> dict:
    rows: dict[str, dict] = {}
    duplicate_keys: Counter[str] = Counter()
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = make_key(row, key_columns)
            if key in rows:
                duplicate_keys[key] += 1
                continue
            score = float(row[score_column])
            label = int(row["label"])
            prediction = int(score >= threshold)
            rows[key] = {
                **row,
                "label": label,
                "score": score,
                "prediction": prediction,
                "error_type": error_type(label, prediction),
            }
    return {
        "path": str(resolve_path(path)),
        "score_column": score_column,
        "threshold": threshold,
        "rows": rows,
        "duplicate_key_count": sum(duplicate_keys.values()),
        "duplicate_keys": len(duplicate_keys),
    }


def metrics_for_rows(rows: dict[str, dict]) -> dict:
    labels = np.asarray([row["label"] for row in rows.values()], dtype=np.int64)
    scores = np.asarray([row["score"] for row in rows.values()], dtype=np.float32)
    predictions = np.asarray([row["prediction"] for row in rows.values()], dtype=np.int64)
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return {
        "rows": int(len(rows)),
        "accuracy": float(accuracy_score(labels, predictions)) if len(rows) else None,
        "precision": float(precision_score(labels, predictions, zero_division=0)) if len(rows) else None,
        "recall": float(recall_score(labels, predictions, zero_division=0)) if len(rows) else None,
        "f1": float(f1_score(labels, predictions, zero_division=0)) if len(rows) else None,
        "auc": float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else None,
        "false_positive": fp,
        "false_negative": fn,
        "errors": fp + fn,
    }


def compare_predictions(
    predictions: Sequence[tuple[str, Path, str, float]],
    *,
    key_columns: Sequence[str],
) -> tuple[dict, list[dict]]:
    loaded = {
        name: read_prediction_csv(path, score_column, threshold, key_columns)
        for name, path, score_column, threshold in predictions
    }
    names = list(loaded)
    key_sets = {name: set(item["rows"]) for name, item in loaded.items()}
    common_keys = set.intersection(*(key_sets[name] for name in names))
    union_keys = set.union(*(key_sets[name] for name in names))

    single = {}
    for name, item in loaded.items():
        single[name] = {
            **metrics_for_rows(item["rows"]),
            "threshold": item["threshold"],
            "path": item["path"],
            "score_column": item["score_column"],
            "duplicate_keys": item["duplicate_keys"],
            "duplicate_key_count": item["duplicate_key_count"],
            "missing_from_common": len(key_sets[name] - common_keys),
        }

    error_sets = {
        name: {key for key in common_keys if loaded[name]["rows"][key]["error_type"]}
        for name in names
    }
    pairwise = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_errors = error_sets[left]
            right_errors = error_sets[right]
            shared = left_errors & right_errors
            union = left_errors | right_errors
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "left_errors_on_common": len(left_errors),
                    "right_errors_on_common": len(right_errors),
                    "shared_errors": len(shared),
                    "left_only_errors": len(left_errors - right_errors),
                    "right_only_errors": len(right_errors - left_errors),
                    "union_errors": len(union),
                    "jaccard": float(len(shared) / len(union)) if union else 0.0,
                }
            )

    baseline_name = names[0]
    baseline_errors = error_sets[baseline_name]
    versus_baseline = []
    for name in names[1:]:
        candidate_errors = error_sets[name]
        versus_baseline.append(
            {
                "baseline": baseline_name,
                "candidate": name,
                "fixed_baseline_errors": len(baseline_errors - candidate_errors),
                "new_candidate_errors": len(candidate_errors - baseline_errors),
                "shared_errors": len(baseline_errors & candidate_errors),
                "net_error_delta_candidate_minus_baseline": len(candidate_errors) - len(baseline_errors),
            }
        )

    detail_rows = []
    pattern_counts: Counter[str] = Counter()
    for key in sorted(union_keys):
        present = [name for name in names if key in loaded[name]["rows"]]
        if len(present) != len(names):
            pattern = "missing:" + ",".join(name for name in names if name not in present)
            pattern_counts[pattern] += 1
            continue
        error_names = [name for name in names if loaded[name]["rows"][key]["error_type"]]
        if not error_names:
            pattern_counts["all_correct"] += 1
            continue
        pattern = "|".join(error_names)
        pattern_counts[pattern] += 1
        base_row = loaded[present[0]]["rows"][key]
        detail = {
            "key": key,
            "source_path": base_row.get("source_path", ""),
            "source_sha256": base_row.get("source_sha256", ""),
            "sample_index": base_row.get("sample_index", ""),
            "label": base_row["label"],
            "error_pattern": pattern,
        }
        for name in names:
            row = loaded[name]["rows"][key]
            detail[f"{name}_score"] = row["score"]
            detail[f"{name}_prediction"] = row["prediction"]
            detail[f"{name}_error_type"] = row["error_type"]
        detail_rows.append(detail)

    report = {
        "schema": "axon_prediction_error_overlap_v1",
        "key_columns": list(key_columns),
        "inputs": names,
        "rows": {
            "union": len(union_keys),
            "common": len(common_keys),
            "all_correct_on_common": pattern_counts.get("all_correct", 0),
            "any_error_on_common": len(detail_rows),
        },
        "single": single,
        "pairwise": pairwise,
        "versus_baseline": versus_baseline,
        "pattern_counts": dict(sorted(pattern_counts.items())),
    }
    return report, detail_rows


def write_detail_csv(path: Path, rows: Sequence[dict]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0])
    else:
        fieldnames = ["key", "source_path", "source_sha256", "sample_index", "label", "error_pattern"]
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare FP/FN overlap across prediction CSVs.")
    parser.add_argument("--prediction", action="append", required=True, help="name=path=score_column=threshold")
    parser.add_argument("--key-columns", default="source_sha256,source_path")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report, rows = compare_predictions(
        [parse_prediction(item) for item in args.prediction],
        key_columns=parse_key_columns(args.key_columns),
    )
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_detail_csv(args.output_csv, rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
