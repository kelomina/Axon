#!/usr/bin/env python3
"""Compare threshold-sweep metrics across exported prediction CSV files."""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence


SUMMARY_COLUMNS = [
    "model",
    "prediction_file",
    "sample_count",
    "best_threshold",
    "best_accuracy",
    "best_precision",
    "best_recall",
    "best_f1",
    "best_fp",
    "best_fn",
    "best_errors",
]

THRESHOLD_COLUMNS = [
    "model",
    "threshold",
    "sample_count",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tp",
    "tn",
    "fp",
    "fn",
    "errors",
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


def parse_thresholds(text: str) -> list[float]:
    thresholds = []
    for item in text.split(","):
        item = item.strip()
        if item:
            thresholds.append(float(item))
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return thresholds


def parse_prediction_arg(text: str) -> tuple[str, Path]:
    if "=" not in text:
        path = Path(text)
        return path.stem, path
    name, path = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"prediction name is empty: {text}")
    return name, Path(path.strip())


def read_prediction_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    missing = {"label", "prob_malicious"} - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def compute_metrics(rows: Sequence[dict], threshold: float) -> dict:
    tp = tn = fp = fn = 0
    for row in rows:
        label = _safe_int(row.get("label"))
        prob = _safe_float(row.get("prob_malicious"))
        pred = 1 if prob >= threshold else 0
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 1 and pred == 0:
            fn += 1

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "threshold": float(threshold),
        "sample_count": int(total),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "errors": int(fp + fn),
    }


def summarize_prediction(name: str, path: Path, thresholds: Sequence[float]) -> dict:
    rows = read_prediction_rows(path)
    threshold_metrics = [compute_metrics(rows, threshold) for threshold in thresholds]
    best = max(
        threshold_metrics,
        key=lambda item: (item["f1"], -item["errors"], item["precision"], item["recall"]),
    )
    return {
        "model": name,
        "prediction_file": str(path),
        "sample_count": int(best["sample_count"]),
        "best_threshold": best["threshold"],
        "best_accuracy": best["accuracy"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "best_f1": best["f1"],
        "best_fp": best["fp"],
        "best_fn": best["fn"],
        "best_errors": best["errors"],
        "threshold_metrics": threshold_metrics,
    }


def compare_predictions(predictions: Sequence[tuple[str, Path]], thresholds: Sequence[float]) -> dict:
    summaries = [summarize_prediction(name, path, thresholds) for name, path in predictions]
    ranked = sorted(summaries, key=lambda item: (item["best_f1"], -item["best_errors"]), reverse=True)
    return {
        "thresholds": [float(item) for item in thresholds],
        "summary": ranked,
        "threshold_metrics": {
            item["model"]: item["threshold_metrics"] for item in summaries
        },
    }


def write_csv_rows(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_outputs(report: dict, output_json: Optional[Path], output_csv: Optional[Path]) -> None:
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if output_csv:
        summary_rows = [
            {key: item.get(key, "") for key in SUMMARY_COLUMNS}
            for item in report["summary"]
        ]
        write_csv_rows(output_csv, summary_rows, SUMMARY_COLUMNS)

        threshold_rows = []
        for model, rows in report["threshold_metrics"].items():
            for row in rows:
                threshold_rows.append({"model": model, **row})
        threshold_csv = output_csv.with_name(f"{output_csv.stem}_thresholds{output_csv.suffix}")
        write_csv_rows(threshold_csv, threshold_rows, THRESHOLD_COLUMNS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare val/test prediction CSV files across decision thresholds."
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Prediction CSV path, optionally named as model=path. Repeat for multiple models.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.45,0.46,0.47,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60,0.61,0.62,0.63,0.64,0.65",
        help="Comma-separated decision thresholds.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = parse_thresholds(args.thresholds)
    predictions = [parse_prediction_arg(item) for item in args.prediction]
    report = compare_predictions(predictions, thresholds)
    write_outputs(report, args.output_json, args.output_csv)

    print("model | best_threshold | best_f1 | fp | fn | errors")
    for row in report["summary"]:
        print(
            f"{row['model']} | {row['best_threshold']:.3f} | "
            f"{row['best_f1']:.6f} | {row['best_fp']} | "
            f"{row['best_fn']} | {row['best_errors']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
