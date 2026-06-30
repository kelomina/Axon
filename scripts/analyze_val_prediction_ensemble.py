#!/usr/bin/env python3
"""Analyze validation prediction complementarity without touching test data."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_prediction_csv(path: Path, score_column: str, key_column: str) -> dict[str, dict]:
    rows = {}
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get(key_column) or row.get("sample_index") or row.get("source_sha256") or row.get("source_path")
            if not key:
                continue
            rows[str(key)] = {
                "label": int(row["label"]),
                "score": float(row[score_column]),
                "sample_index": row.get("sample_index", ""),
                "source_path": row.get("source_path", ""),
            }
    return rows


def parse_input(text: str) -> tuple[str, Path, str]:
    parts = text.split("=", 2)
    if len(parts) != 3:
        raise ValueError(f"Expected name=path=score_column, got: {text}")
    return parts[0], Path(parts[1]), parts[2]


def metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "errors": int((pred != labels).sum()),
        "fp": int(((pred == 1) & (labels == 0)).sum()),
        "fn": int(((pred == 0) & (labels == 1)).sum()),
    }


def best_threshold(scores: np.ndarray, labels: np.ndarray, thresholds: Sequence[float]) -> dict:
    rows = [metrics(scores, labels, threshold) for threshold in thresholds]
    rows.sort(key=lambda row: (row["f1"], -row["errors"], row["threshold"]), reverse=True)
    return rows[0]


def parse_thresholds(text: str) -> list[float]:
    if ":" in text:
        start_text, stop_text, step_text = text.split(":")
        start = float(start_text)
        stop = float(stop_text)
        step = float(step_text)
        count = int(np.floor((stop - start) / step)) + 1
        return [round(start + step * index, 10) for index in range(count)]
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_weighted_blend(text: str) -> tuple[list[str], list[float]]:
    names = []
    weights = []
    for item in text.split(","):
        if not item.strip():
            continue
        name, weight = item.split(":", 1)
        names.append(name.strip())
        weights.append(float(weight))
    if len(names) < 2:
        raise ValueError(f"Weighted blend needs at least two members: {text}")
    return names, weights


def analyze(
    inputs: Sequence[tuple[str, Path, str]],
    thresholds: Sequence[float],
    weighted_blends: Sequence[tuple[list[str], list[float]]] = (),
    key_column: str = "sample_index",
) -> dict:
    loaded = [(name, read_prediction_csv(path, score_column, key_column)) for name, path, score_column in inputs]
    common_keys = set(loaded[0][1])
    for _name, rows in loaded[1:]:
        common_keys &= set(rows)
    ordered_keys = sorted(common_keys, key=lambda key: int(loaded[0][1][key].get("sample_index") or 0))
    if not ordered_keys:
        raise ValueError("No common validation rows across prediction inputs")

    labels = np.asarray([loaded[0][1][key]["label"] for key in ordered_keys], dtype=np.int64)
    result = {
        "schema": "axon_val_prediction_ensemble_analysis_v1",
        "rows": len(ordered_keys),
        "key_column": key_column,
        "inputs": [name for name, _path, _score_column in inputs],
        "single_models": [],
        "pairwise_error_overlap": [],
        "average_ensembles": [],
        "weighted_ensembles": [],
    }
    score_by_name = {}
    error_mask_by_name = {}
    for name, rows in loaded:
        scores = np.asarray([rows[key]["score"] for key in ordered_keys], dtype=np.float32)
        selected = best_threshold(scores, labels, thresholds)
        error_mask = (scores >= selected["threshold"]).astype(np.int64) != labels
        score_by_name[name] = scores
        error_mask_by_name[name] = error_mask
        result["single_models"].append({"name": name, **selected})

    for left, right in combinations(score_by_name, 2):
        left_errors = error_mask_by_name[left]
        right_errors = error_mask_by_name[right]
        intersection = int((left_errors & right_errors).sum())
        union = int((left_errors | right_errors).sum())
        result["pairwise_error_overlap"].append(
            {
                "left": left,
                "right": right,
                "left_errors": int(left_errors.sum()),
                "right_errors": int(right_errors.sum()),
                "shared_errors": intersection,
                "union_errors": union,
                "jaccard": float(intersection / union) if union else 0.0,
            }
        )

    names = list(score_by_name)
    for size in range(2, min(4, len(names)) + 1):
        for group in combinations(names, size):
            scores = np.mean([score_by_name[name] for name in group], axis=0)
            selected = best_threshold(scores, labels, thresholds)
            result["average_ensembles"].append({"names": list(group), **selected})

    for blend_names, blend_weights in weighted_blends:
        missing = [name for name in blend_names if name not in score_by_name]
        if missing:
            raise ValueError(f"Weighted blend references unknown model(s): {missing}")
        weights = np.asarray(blend_weights, dtype=np.float32)
        weights = weights / weights.sum()
        stacked = np.vstack([score_by_name[name] for name in blend_names])
        scores = (stacked * weights[:, None]).sum(axis=0)
        selected = best_threshold(scores, labels, thresholds)
        result["weighted_ensembles"].append(
            {
                "names": blend_names,
                "weights": [float(weight) for weight in weights],
                **selected,
            }
        )

    result["single_models"].sort(key=lambda row: (row["f1"], -row["errors"]), reverse=True)
    result["average_ensembles"].sort(key=lambda row: (row["f1"], -row["errors"]), reverse=True)
    result["weighted_ensembles"].sort(key=lambda row: (row["f1"], -row["errors"]), reverse=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Val prediction complementarity.")
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Format: name=path=score_column",
    )
    parser.add_argument("--thresholds", default="0.05:0.95:0.005")
    parser.add_argument("--weighted-blend", action="append", default=[], help="Format: name:weight,name:weight")
    parser.add_argument("--key-column", default="sample_index")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze(
        [parse_input(item) for item in args.prediction],
        parse_thresholds(args.thresholds),
        [parse_weighted_blend(item) for item in args.weighted_blend],
        key_column=args.key_column,
    )
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
