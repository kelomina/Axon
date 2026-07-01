#!/usr/bin/env python3
"""Evaluate a frozen weighted blend of prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_prediction(text: str) -> tuple[str, Path, str, float]:
    parts = text.split("=", 3)
    if len(parts) != 4:
        raise ValueError(f"Expected name=path=score_column=weight, got: {text}")
    return parts[0], Path(parts[1]), parts[2], float(parts[3])


def read_rows(path: Path, score_column: str, key_column: str) -> dict[str, dict]:
    rows = {}
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get(key_column) or row.get("sample_index") or row.get("source_sha256") or row.get("source_path")
            if not key:
                continue
            rows[str(key)] = {
                **row,
                "label": int(row["label"]),
                "score": float(row[score_column]),
            }
    return rows


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "errors": int(fp + fn),
    }


def evaluate_blend(
    predictions: Sequence[tuple[str, Path, str, float]],
    *,
    threshold: float,
    key_column: str = "sample_index",
) -> tuple[list[dict], dict]:
    loaded = [(name, read_rows(path, score_column, key_column), weight) for name, path, score_column, weight in predictions]
    common_keys = set(loaded[0][1])
    for _name, rows, _weight in loaded[1:]:
        common_keys &= set(rows)
    ordered_keys = sorted(common_keys, key=lambda key: int(loaded[0][1][key].get("sample_index") or 0))
    if not ordered_keys:
        raise ValueError("No common rows across prediction inputs")

    labels = np.asarray([loaded[0][1][key]["label"] for key in ordered_keys], dtype=np.int64)
    alignment_mismatches = []
    reference_rows = loaded[0][1]
    for key in ordered_keys:
        reference = reference_rows[key]
        for name, rows, _weight in loaded[1:]:
            row = rows[key]
            mismatch_fields = []
            if int(row["label"]) != int(reference["label"]):
                mismatch_fields.append("label")
            if row.get("source_sha256") and reference.get("source_sha256"):
                if row["source_sha256"].strip().lower() != reference["source_sha256"].strip().lower():
                    mismatch_fields.append("source_sha256")
            elif row.get("source_path") and reference.get("source_path"):
                if row["source_path"] != reference["source_path"]:
                    mismatch_fields.append("source_path")
            if mismatch_fields:
                alignment_mismatches.append(
                    {
                        "key": key,
                        "model": name,
                        "fields": mismatch_fields,
                    }
                )
                if len(alignment_mismatches) >= 10:
                    break
        if len(alignment_mismatches) >= 10:
            break
    if alignment_mismatches:
        raise ValueError(
            "Prediction inputs are not aligned on the requested key. "
            f"First mismatches: {alignment_mismatches}"
        )

    weights = np.asarray([weight for _name, _rows, weight in loaded], dtype=np.float32)
    weights = weights / weights.sum()
    stacked = np.vstack([
        np.asarray([rows[key]["score"] for key in ordered_keys], dtype=np.float32)
        for _name, rows, _weight in loaded
    ])
    scores = (stacked * weights[:, None]).sum(axis=0)
    selected_metrics = metrics(scores, labels, threshold)
    out_rows = []
    for row_index, key in enumerate(ordered_keys):
        base_row = loaded[0][1][key]
        score = float(scores[row_index])
        prediction = int(score >= threshold)
        out_rows.append(
            {
                "source_path": base_row.get("source_path", ""),
                "source_sha256": base_row.get("source_sha256", ""),
                "label": int(labels[row_index]),
                "split": base_row.get("split", ""),
                "sample_index": base_row.get("sample_index", ""),
                "blend_prob_malicious": f"{score:.10f}",
                "prediction": prediction,
                "correct": prediction == int(labels[row_index]),
            }
        )

    report = {
        "schema": "axon_prediction_blend_eval_v1",
        "rows": len(ordered_keys),
        "key_column": key_column,
        "alignment_audit": {
            "checked": True,
            "audit_only_fields": ["label", "source_sha256", "source_path"],
            "mismatches": 0,
        },
        "inputs": [
            {"name": name, "path": str(resolve_path(path)), "score_column": score_column, "weight": float(weight)}
            for name, path, score_column, weight in predictions
        ],
        "normalized_weights": {name: float(weight) for (name, _rows, _raw_weight), weight in zip(loaded, weights)},
        "metrics": selected_metrics,
    }
    return out_rows, report


def write_predictions(path: Path, rows: Sequence[dict]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "blend_prob_malicious",
        "prediction",
        "correct",
    ]
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a frozen weighted prediction blend.")
    parser.add_argument("--prediction", action="append", required=True, help="name=path=score_column=weight")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--key-column", default="sample_index")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows, report = evaluate_blend(
        [parse_prediction(item) for item in args.prediction],
        threshold=float(args.threshold),
        key_column=args.key_column,
    )
    write_predictions(args.output_csv, rows)
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
