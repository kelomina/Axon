#!/usr/bin/env python3
"""Evaluate model predictions by raw similarity groups."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

from raw_group_tools import (
    normalize_path_text,
    parse_pipe_paths,
    read_csv_rows,
    resolve_path,
    write_csv,
    write_json,
)


GROUP_EVAL_COLUMNS = [
    "group_id",
    "source",
    "size",
    "labels",
    "splits",
    "is_rare_group",
    "is_singleton",
    "has_leakage",
    "train_count",
    "predicted_samples",
    "correct_count",
    "error_count",
    "accuracy",
    "malicious_count",
    "malicious_recall",
    "benign_count",
    "benign_recall",
    "avg_prob_malicious",
    "max_prob_malicious",
    "min_prob_malicious",
    "top_error_paths",
]


def _to_bool(value) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _prediction_by_path(prediction_rows):
    return {normalize_path_text(row["source_path"]): row for row in prediction_rows}


def _evaluate_group(group_row: dict, group_members: Sequence[dict], predictions_by_path: dict) -> dict:
    paths = [member["source_path"] for member in group_members]
    if not paths:
        paths = parse_pipe_paths(group_row.get("source_paths", ""))
    matched = []
    for path in paths:
        prediction = predictions_by_path.get(normalize_path_text(path))
        if prediction is not None:
            matched.append(prediction)

    correct_count = sum(1 for row in matched if _to_bool(row.get("correct")))
    error_rows = [row for row in matched if not _to_bool(row.get("correct"))]
    label_counts = Counter(int(row["label"]) for row in matched)
    correct_by_label = Counter(int(row["label"]) for row in matched if _to_bool(row.get("correct")))
    probs = [_safe_float(row.get("prob_malicious")) for row in matched]
    predicted_samples = len(matched)
    malicious_count = label_counts.get(1, 0)
    benign_count = label_counts.get(0, 0)

    return {
        "group_id": group_row["group_id"],
        "source": group_row.get("source", ""),
        "size": int(group_row.get("size", 0)),
        "labels": group_row.get("labels", ""),
        "splits": group_row.get("splits", ""),
        "is_rare_group": group_row.get("is_rare_group", ""),
        "is_singleton": group_row.get("is_singleton", ""),
        "has_leakage": group_row.get("has_leakage", ""),
        "train_count": int(group_row.get("train_count", 0)),
        "predicted_samples": predicted_samples,
        "correct_count": correct_count,
        "error_count": len(error_rows),
        "accuracy": (correct_count / predicted_samples) if predicted_samples else 0.0,
        "malicious_count": malicious_count,
        "malicious_recall": (correct_by_label.get(1, 0) / malicious_count) if malicious_count else "",
        "benign_count": benign_count,
        "benign_recall": (correct_by_label.get(0, 0) / benign_count) if benign_count else "",
        "avg_prob_malicious": (sum(probs) / len(probs)) if probs else "",
        "max_prob_malicious": max(probs) if probs else "",
        "min_prob_malicious": min(probs) if probs else "",
        "top_error_paths": "|".join(row["source_path"] for row in error_rows[:10]),
    }


def _aggregate(rows: Sequence[dict], predicate) -> dict:
    selected = [row for row in rows if predicate(row)]
    predicted = sum(int(row["predicted_samples"]) for row in selected)
    correct = sum(int(row["correct_count"]) for row in selected)
    errors = sum(int(row["error_count"]) for row in selected)
    return {
        "groups": len(selected),
        "predicted_samples": predicted,
        "correct_count": correct,
        "error_count": errors,
        "accuracy": (correct / predicted) if predicted else 0.0,
    }


def _load_members_for_groups(groups_path: Path) -> dict:
    members_path = groups_path.with_name("group_members.csv")
    if not members_path.exists():
        return {}
    by_group = defaultdict(list)
    for row in read_csv_rows(members_path):
        by_group[str(row["group_id"])].append(row)
    return by_group


def evaluate_groups(groups_path: Path, predictions_path: Path, output_dir: Path) -> dict:
    group_rows = read_csv_rows(resolve_path(groups_path))
    prediction_rows = read_csv_rows(resolve_path(predictions_path))
    predictions_by_path = _prediction_by_path(prediction_rows)
    members_by_group = _load_members_for_groups(resolve_path(groups_path))
    eval_rows = [
        _evaluate_group(row, members_by_group.get(str(row["group_id"]), []), predictions_by_path)
        for row in group_rows
    ]

    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_csv = output_dir / "group_evaluation.csv"
    summary_json = output_dir / "group_evaluation_summary.json"

    write_csv(eval_csv, eval_rows, GROUP_EVAL_COLUMNS)
    summary = {
        "groups": str(resolve_path(groups_path)),
        "predictions": str(resolve_path(predictions_path)),
        "outputs": {
            "group_evaluation_csv": str(eval_csv),
            "group_evaluation_summary_json": str(summary_json),
        },
        "overall": _aggregate(eval_rows, lambda _row: True),
        "rare_groups": _aggregate(eval_rows, lambda row: _to_bool(row.get("is_rare_group"))),
        "singleton_groups": _aggregate(eval_rows, lambda row: _to_bool(row.get("is_singleton"))),
        "leakage_groups": _aggregate(eval_rows, lambda row: _to_bool(row.get("has_leakage"))),
        "train_too_small_groups": _aggregate(eval_rows, lambda row: int(row.get("train_count", 0)) <= 1),
        "worst_groups": sorted(
            [
                {
                    "group_id": row["group_id"],
                    "size": row["size"],
                    "is_rare_group": row["is_rare_group"],
                    "error_count": row["error_count"],
                    "accuracy": row["accuracy"],
                    "top_error_paths": row["top_error_paths"],
                }
                for row in eval_rows
                if int(row["predicted_samples"]) > 0 and int(row["error_count"]) > 0
            ],
            key=lambda row: (-int(row["error_count"]), float(row["accuracy"]), int(row["size"])),
        )[:30],
    }
    write_json(summary_json, summary)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Evaluate model predictions by raw similarity groups.")
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/raw_group_diagnostics"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = evaluate_groups(args.groups, args.predictions, args.output_dir)
    print("=" * 60)
    print("Group Evaluation")
    print("=" * 60)
    print(f"Overall accuracy: {summary['overall']['accuracy']:.4f}")
    print(f"Rare-group accuracy: {summary['rare_groups']['accuracy']:.4f}")
    print(f"Singleton accuracy: {summary['singleton_groups']['accuracy']:.4f}")
    print(f"Worst groups reported: {len(summary['worst_groups'])}")
    print(f"Report: {summary['outputs']['group_evaluation_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
