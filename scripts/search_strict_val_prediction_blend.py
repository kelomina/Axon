#!/usr/bin/env python3
"""Search a two-model prediction blend on Val using source_sha256 alignment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DENIED_SCORE_COLUMNS = {
    "cache_path",
    "correct",
    "directory",
    "dir",
    "extension",
    "file_name",
    "filename",
    "label",
    "path",
    "prediction",
    "sample_index",
    "source_path",
    "source_sha256",
    "split",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def is_valid_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def validate_score_column(score_column: str) -> None:
    normalized = str(score_column or "").strip().casefold()
    if not normalized:
        raise ValueError("Score column must not be empty")
    if normalized in DENIED_SCORE_COLUMNS:
        raise ValueError(f"Refusing to use identity/leakage column as a score: {score_column!r}")


def row_identity_key(row: dict[str, str], path: Path) -> tuple[str, str]:
    source_sha = str(row.get("source_sha256") or "").strip().casefold()
    sample_index = str(row.get("sample_index") or "").strip()
    if not is_valid_sha256(source_sha):
        raise ValueError(f"Prediction row has invalid source_sha256 in {path}: {source_sha!r}")
    if not sample_index:
        raise ValueError(f"Prediction row missing sample_index in {path}")
    return source_sha, sample_index


def read_predictions(path: Path, score_column: str) -> dict[str, dict]:
    validate_score_column(score_column)
    rows = {}
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row_identity_key(row, path)
            if key in rows:
                raise ValueError(f"Prediction file contains duplicate source_sha256/sample_index key {key!r}: {path}")
            if score_column not in row:
                raise ValueError(f"Prediction file missing score column {score_column!r}: {path}")
            rows[key] = {
                **row,
                "source_sha256": key[0],
                "sample_index": key[1],
                "label": int(row["label"]),
                "score": float(row[score_column]),
            }
    return rows


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int64)
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "errors": fp + fn,
    }


def parse_float_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def search_blend(
    *,
    first_csv: Path,
    second_csv: Path,
    first_score_column: str,
    second_score_column: str,
    weights: Sequence[float],
    thresholds: Sequence[float],
) -> dict:
    first = read_predictions(first_csv, first_score_column)
    second = read_predictions(second_csv, second_score_column)
    first_keys = set(first)
    second_keys = set(second)
    if first_keys != second_keys:
        missing_from_first = sorted(second_keys - first_keys)[:10]
        missing_from_second = sorted(first_keys - second_keys)[:10]
        raise ValueError(
            "Prediction files must cover the exact same source_sha256/sample_index set; "
            f"missing_from_first={missing_from_first}, missing_from_second={missing_from_second}"
        )
    common_keys = sorted(first_keys & second_keys, key=lambda key: int(first[key].get("sample_index", 0) or 0))
    if not common_keys:
        raise ValueError("No source_sha256/sample_index overlap between prediction files")

    alignment_issues = []
    for key in common_keys:
        if int(first[key]["label"]) != int(second[key]["label"]):
            alignment_issues.append({"source_sha256": key, "issue": "label_mismatch"})
        if first[key].get("sample_index") != second[key].get("sample_index"):
            alignment_issues.append({"source_sha256": key, "issue": "sample_index_mismatch"})
        if first[key].get("split") != second[key].get("split"):
            alignment_issues.append({"source_sha256": key, "issue": "split_mismatch"})
        if len(alignment_issues) >= 20:
            break
    if alignment_issues:
        raise ValueError(f"Prediction files are not aligned: {alignment_issues[:5]}")

    labels = np.asarray([first[key]["label"] for key in common_keys], dtype=np.int64)
    first_scores = np.asarray([first[key]["score"] for key in common_keys], dtype=np.float64)
    second_scores = np.asarray([second[key]["score"] for key in common_keys], dtype=np.float64)

    rows = []
    for second_weight in weights:
        second_weight = float(second_weight)
        first_weight = 1.0 - second_weight
        blended = first_weight * first_scores + second_weight * second_scores
        for threshold in thresholds:
            item = metrics(labels, blended, float(threshold))
            item["first_weight"] = float(first_weight)
            item["second_weight"] = float(second_weight)
            rows.append(item)
    rows.sort(key=lambda item: (-item["f1"], item["errors"], item["threshold"], item["second_weight"]))
    return {
        "schema": "axon_strict_val_prediction_blend_search_v1",
        "identity_feature_policy": (
            "source_sha256 is the primary row identity; sample_index disambiguates duplicate file-content rows. "
            "Neither identity field, nor path/name/directory/extension, is used for scoring or threshold selection."
        ),
        "first_csv": str(resolve_path(first_csv)),
        "second_csv": str(resolve_path(second_csv)),
        "rows": len(common_keys),
        "missing_from_first": len(set(second) - set(first)),
        "missing_from_second": len(set(first) - set(second)),
        "duplicate_source_sha256_rows": len(common_keys) - len({key[0] for key in common_keys}),
        "grid_size": len(rows),
        "best": rows[0],
        "top10": rows[:10],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search strict Val blend grid.")
    parser.add_argument("--first-csv", type=Path, required=True)
    parser.add_argument("--second-csv", type=Path, required=True)
    parser.add_argument("--first-score-column", default="prob_malicious")
    parser.add_argument("--second-score-column", default="prob_malicious")
    parser.add_argument("--weights", required=True, help="Comma separated second-model weights")
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = search_blend(
        first_csv=args.first_csv,
        second_csv=args.second_csv,
        first_score_column=args.first_score_column,
        second_score_column=args.second_score_column,
        weights=parse_float_grid(args.weights),
        thresholds=parse_float_grid(args.thresholds),
    )
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["best"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
