#!/usr/bin/env python3
"""Evaluate a frozen union of prediction overrides.

The script treats every input model decision as already frozen. It never uses
path, filename, hash, sample id, split, or row order as model evidence; those
fields are only row-alignment and audit metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "axon_prediction_override_union_eval_v1"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_key_columns(text: str) -> tuple[str, ...]:
    columns = tuple(item.strip() for item in text.split(",") if item.strip())
    if not columns:
        raise ValueError("At least one key column is required")
    return columns


def parse_override(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Expected name=path, got: {text}")
    name, path = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Override name must not be empty: {text}")
    return name, Path(path)


def row_key(row: dict[str, str], key_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")).strip().casefold() for column in key_columns)


def read_rows(path: Path, key_columns: Sequence[str]) -> dict[tuple[str, ...], dict[str, str]]:
    resolved = resolve_path(path)
    rows: dict[tuple[str, ...], dict[str, str]] = {}
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row_key(row, key_columns)
            if any(not part for part in key):
                raise ValueError(f"Missing key column in {resolved}: key={key}")
            if key in rows:
                raise ValueError(f"Duplicate key in {resolved}: key={key}")
            rows[key] = row
    return rows


def _score(row: dict[str, str]) -> float:
    for column in ("stage2_prob_malicious", "prob_malicious", "blend_prob_malicious"):
        value = str(row.get(column, "")).strip()
        if value:
            return float(value)
    return float(int(row["prediction"]))


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "errors": int(fp + fn),
    }


def evaluate_override_union(
    *,
    baseline_csv: Path,
    overrides: Sequence[tuple[str, Path]],
    key_columns: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    baseline = read_rows(baseline_csv, key_columns)
    override_rows = [(name, read_rows(path, key_columns), path) for name, path in overrides]
    baseline_keys = set(baseline)
    for name, rows, path in override_rows:
        if set(rows) != baseline_keys:
            missing = sorted(baseline_keys - set(rows))[:10]
            extra = sorted(set(rows) - baseline_keys)[:10]
            raise ValueError(
                f"Override {name} does not cover the same rows as baseline {baseline_csv}; "
                f"missing={missing}, extra={extra}, path={resolve_path(path)}"
            )

    ordered_keys = sorted(
        baseline_keys,
        key=lambda key: tuple(int(part) if part.isdigit() else part for part in key),
    )
    output_rows: list[dict[str, object]] = []
    labels = np.empty(len(ordered_keys), dtype=np.int64)
    baseline_predictions = np.empty(len(ordered_keys), dtype=np.int64)
    final_predictions = np.empty(len(ordered_keys), dtype=np.int64)
    override_counts = {name: 0 for name, _path in overrides}
    override_label_counts = {name: {"0": 0, "1": 0} for name, _path in overrides}
    conflicts: list[dict[str, object]] = []

    for index, key in enumerate(ordered_keys):
        base_row = baseline[key]
        label = int(base_row["label"])
        base_prediction = int(base_row["prediction"])
        selected_prediction = base_prediction
        selected_score = _score(base_row)
        selected_override = ""
        selected_override_score = ""

        for name, rows, _path in override_rows:
            row = rows[key]
            if int(row["label"]) != label:
                raise ValueError(f"Label mismatch for key={key} override={name}")
            if str(row.get("split", "")) != str(base_row.get("split", "")):
                raise ValueError(f"Split mismatch for key={key} override={name}")
            override_prediction = int(row["prediction"])
            if override_prediction == base_prediction:
                continue
            if selected_override and override_prediction != selected_prediction:
                conflicts.append(
                    {
                        "key": key,
                        "first_override": selected_override,
                        "second_override": name,
                        "first_prediction": int(selected_prediction),
                        "second_prediction": int(override_prediction),
                    }
                )
                continue
            selected_prediction = override_prediction
            selected_score = _score(row)
            selected_override = name
            selected_override_score = f"{selected_score:.10f}"

        labels[index] = label
        baseline_predictions[index] = base_prediction
        final_predictions[index] = selected_prediction
        if selected_override:
            override_counts[selected_override] += 1
            override_label_counts[selected_override][str(label)] += 1

        output_rows.append(
            {
                "source_path": base_row.get("source_path", ""),
                "cache_path": base_row.get("cache_path", ""),
                "source_sha256": base_row.get("source_sha256", ""),
                "label": label,
                "split": base_row.get("split", ""),
                "sample_index": base_row.get("sample_index", ""),
                "baseline_prediction": base_prediction,
                "baseline_prob_malicious": f"{_score(base_row):.10f}",
                "accepted_override": selected_override,
                "accepted_override_prob_malicious": selected_override_score,
                "stage2_prob_malicious": f"{selected_score:.10f}",
                "prediction": int(selected_prediction),
                "correct": int(selected_prediction) == label,
            }
        )

    if conflicts:
        raise ValueError(f"Conflicting override predictions found: {conflicts[:10]}")

    baseline_metrics = _metrics(labels, baseline_predictions)
    final_metrics = _metrics(labels, final_predictions)
    report = {
        "schema": SCHEMA,
        "protocol": "frozen prediction union: baseline is kept unless a frozen override changes its decision",
        "identity_feature_policy": (
            "source_path/cache_path/source_sha256/sample_index/split/row order are alignment and audit fields only; "
            "they are not model evidence"
        ),
        "baseline_csv": str(resolve_path(baseline_csv)),
        "overrides": [
            {"name": name, "path": str(resolve_path(path))}
            for name, path in overrides
        ],
        "key_columns": list(key_columns),
        "rows": int(len(ordered_keys)),
        "accepted_override_counts": override_counts,
        "accepted_override_label_counts": override_label_counts,
        "changed_rows": int(sum(override_counts.values())),
        "baseline_metrics": baseline_metrics,
        "metrics": final_metrics,
        "delta_vs_baseline": {
            "errors": int(final_metrics["errors"]) - int(baseline_metrics["errors"]),
            "false_positive": int(final_metrics["false_positive"]) - int(baseline_metrics["false_positive"]),
            "false_negative": int(final_metrics["false_negative"]) - int(baseline_metrics["false_negative"]),
            "f1": float(final_metrics["f1"]) - float(baseline_metrics["f1"]),
        },
    }
    return output_rows, report


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "cache_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "baseline_prediction",
        "baseline_prob_malicious",
        "accepted_override",
        "accepted_override_prob_malicious",
        "stage2_prob_malicious",
        "prediction",
        "correct",
    ]
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a frozen prediction override union.")
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--override", action="append", required=True, help="Format: name=path")
    parser.add_argument("--key-columns", default="sample_index,source_sha256")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows, report = evaluate_override_union(
        baseline_csv=args.baseline_csv,
        overrides=[parse_override(item) for item in args.override],
        key_columns=parse_key_columns(args.key_columns),
    )
    write_rows(args.output_csv, rows)
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
