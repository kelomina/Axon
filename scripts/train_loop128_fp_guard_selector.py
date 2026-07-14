#!/usr/bin/env python3
"""Train/evaluate a narrow FP guard selector between two frozen predictors.

The guard never uses filename, path, hash, split, row order, or sample id as
model evidence. Those fields are used only to align rows and write auditable
prediction CSVs. The only model inputs are the two frozen model probabilities
and their derived score differences.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for item in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402


FP_GUARD_FEATURE_NAMES = [
    "primary_prob_malicious",
    "conservative_prob_malicious",
    "prob_delta_primary_minus_conservative",
    "primary_logit",
    "conservative_logit",
    "logit_delta_primary_minus_conservative",
    "prob_spread_abs",
]


@dataclass(frozen=True)
class AlignedPredictions:
    rows: list[dict]
    labels: np.ndarray
    primary_prob: np.ndarray
    conservative_prob: np.ndarray
    primary_pred: np.ndarray
    conservative_pred: np.ndarray


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_prediction_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict, key_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")) for column in key_columns)


def align_predictions(primary_path: Path, conservative_path: Path, key_columns: Sequence[str]) -> AlignedPredictions:
    primary_rows = read_prediction_rows(primary_path)
    conservative_rows = read_prediction_rows(conservative_path)
    conservative_by_key = {row_key(row, key_columns): row for row in conservative_rows}
    if len(conservative_by_key) != len(conservative_rows):
        raise ValueError("Conservative predictions contain duplicate alignment keys")

    rows = []
    labels = []
    primary_prob = []
    conservative_prob = []
    primary_pred = []
    conservative_pred = []
    missing = []
    for primary_row in primary_rows:
        key = row_key(primary_row, key_columns)
        conservative_row = conservative_by_key.get(key)
        if conservative_row is None:
            missing.append(key)
            continue
        label = int(primary_row["label"])
        if label != int(conservative_row["label"]):
            raise ValueError(f"Label mismatch for key={key}")
        rows.append(primary_row)
        labels.append(label)
        primary_prob.append(float(primary_row["stage2_prob_malicious"]))
        conservative_prob.append(float(conservative_row["stage2_prob_malicious"]))
        primary_pred.append(int(primary_row["prediction"]))
        conservative_pred.append(int(conservative_row["prediction"]))
    if missing:
        preview = ", ".join(str(item) for item in missing[:5])
        raise ValueError(f"Missing {len(missing)} conservative rows; first keys: {preview}")
    return AlignedPredictions(
        rows=rows,
        labels=np.asarray(labels, dtype=np.int64),
        primary_prob=np.asarray(primary_prob, dtype=np.float32),
        conservative_prob=np.asarray(conservative_prob, dtype=np.float32),
        primary_pred=np.asarray(primary_pred, dtype=np.int64),
        conservative_pred=np.asarray(conservative_pred, dtype=np.int64),
    )


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float32, copy=False), 1.0e-6, 1.0 - 1.0e-6)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32, copy=False)


def build_guard_features(primary_prob: np.ndarray, conservative_prob: np.ndarray) -> np.ndarray:
    assert_no_identity_feature_names(FP_GUARD_FEATURE_NAMES, context="Loop128 FP guard selector")
    primary_logit = logit(primary_prob)
    conservative_logit = logit(conservative_prob)
    matrix = np.column_stack(
        [
            primary_prob,
            conservative_prob,
            primary_prob - conservative_prob,
            primary_logit,
            conservative_logit,
            primary_logit - conservative_logit,
            np.abs(primary_prob - conservative_prob),
        ]
    ).astype(np.float32, copy=False)
    if matrix.shape[1] != len(FP_GUARD_FEATURE_NAMES):
        raise ValueError("Guard feature shape mismatch")
    return matrix


def possible_flip_mask(aligned: AlignedPredictions) -> np.ndarray:
    return (aligned.primary_pred == 1) & (aligned.conservative_pred == 0)


def metrics_at_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict:
    labels = labels.astype(np.int64, copy=False)
    predictions = predictions.astype(np.int64, copy=False)
    tp = int(np.count_nonzero((labels == 1) & (predictions == 1)))
    tn = int(np.count_nonzero((labels == 0) & (predictions == 0)))
    fp = int(np.count_nonzero((labels == 0) & (predictions == 1)))
    fn = int(np.count_nonzero((labels == 1) & (predictions == 0)))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "accuracy": float((tp + tn) / labels.shape[0]) if labels.shape[0] else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "errors": fp + fn,
    }


def model_candidates(seed: int) -> list[tuple[str, object]]:
    return [
        (
            "guard_logreg_l2_c0.1",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="liblinear", C=0.1)),
        ),
        (
            "guard_logreg_balanced_c0.1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, solver="liblinear", C=0.1, class_weight="balanced"),
            ),
        ),
        (
            "guard_hgb_leaf3",
            HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, max_leaf_nodes=3, l2_regularization=0.01),
        ),
        (
            "guard_extra_trees_100_leaf2",
            ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1, class_weight="balanced"),
        ),
    ]


def score_model(model, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float32)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(matrix), dtype=np.float32)
        return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)
    return np.asarray(model.predict(matrix), dtype=np.float32)


def apply_guard(aligned: AlignedPredictions, guard_scores: np.ndarray, allow_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    flip = possible_flip_mask(aligned) & (guard_scores >= float(allow_threshold))
    predictions = aligned.primary_pred.copy()
    predictions[flip] = 0
    return predictions, flip


def parse_thresholds(text: str) -> list[float]:
    if ":" in text:
        start, stop, step = (float(part) for part in text.split(":"))
        values = []
        current = start
        while current <= stop + 1.0e-12:
            values.append(round(float(current), 10))
            current += step
        return values
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def write_predictions(path: Path, aligned: AlignedPredictions, guard_scores: np.ndarray, predictions: np.ndarray, flip: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(aligned.rows[0].keys()) + ["primary_prob_malicious", "conservative_prob_malicious", "guard_score", "guard_flip"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(aligned.rows):
            output = dict(row)
            output["primary_prob_malicious"] = f"{float(aligned.primary_prob[index]):.10f}"
            output["conservative_prob_malicious"] = f"{float(aligned.conservative_prob[index]):.10f}"
            output["stage2_prob_malicious"] = f"{float(aligned.primary_prob[index]):.10f}"
            output["guard_score"] = f"{float(guard_scores[index]):.10f}"
            output["guard_flip"] = "1" if bool(flip[index]) else "0"
            output["prediction"] = str(int(predictions[index]))
            output["correct"] = "True" if int(predictions[index]) == int(aligned.labels[index]) else "False"
            writer.writerow(output)


def train_command(args: argparse.Namespace) -> int:
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    key_columns = tuple(item.strip() for item in args.key_columns.split(",") if item.strip())
    train = align_predictions(args.primary_train_predictions, args.conservative_train_predictions, key_columns)
    val = align_predictions(args.primary_val_predictions, args.conservative_val_predictions, key_columns)

    train_features = build_guard_features(train.primary_prob, train.conservative_prob)
    val_features = build_guard_features(val.primary_prob, val.conservative_prob)
    train_possible = possible_flip_mask(train)
    val_possible = possible_flip_mask(val)
    train_targets = (train.labels[train_possible] == 0).astype(np.int64)
    if train_targets.size < 4 or len(set(int(item) for item in train_targets.tolist())) < 2:
        raise ValueError(
            f"Not enough mixed possible flip rows to train guard: rows={train_targets.size}, classes={sorted(set(train_targets.tolist()))}"
        )

    thresholds = parse_thresholds(args.allow_thresholds)
    primary_val_metrics = metrics_at_predictions(val.labels, val.primary_pred)
    conservative_val_metrics = metrics_at_predictions(val.labels, val.conservative_pred)
    candidates = []
    selected = None
    selected_model = None
    selected_val_scores = None
    selected_predictions = None
    selected_flip = None
    best_key = None
    start_all = time.perf_counter()
    for name, prototype in model_candidates(int(args.seed)):
        model = clone(prototype)
        start = time.perf_counter()
        model.fit(train_features[train_possible], train_targets)
        fit_sec = time.perf_counter() - start
        train_scores_possible = score_model(model, train_features[train_possible])
        val_scores = np.zeros(val.labels.shape[0], dtype=np.float32)
        if int(np.count_nonzero(val_possible)):
            val_scores[val_possible] = score_model(model, val_features[val_possible])
        for threshold in thresholds:
            predictions, flip = apply_guard(val, val_scores, threshold)
            metrics = metrics_at_predictions(val.labels, predictions)
            result = {
                "name": name,
                "allow_threshold": float(threshold),
                "fit_sec": float(fit_sec),
                "val": metrics,
                "val_flips": int(np.count_nonzero(flip)),
                "val_flipped_label0": int(np.count_nonzero(flip & (val.labels == 0))),
                "val_flipped_label1": int(np.count_nonzero(flip & (val.labels == 1))),
                "train_possible_rows": int(train_targets.size),
                "train_possible_label0": int(np.count_nonzero(train_targets == 1)),
                "train_possible_label1": int(np.count_nonzero(train_targets == 0)),
                "train_guard_score_mean": float(np.mean(train_scores_possible)) if train_scores_possible.size else 0.0,
            }
            candidates.append(result)
            key = (int(metrics["errors"]) * -1, float(metrics["f1"]), int(metrics["false_negative"]) * -1)
            if best_key is None or key > best_key:
                best_key = key
                selected = result
                selected_model = model
                selected_val_scores = val_scores.copy()
                selected_predictions = predictions.copy()
                selected_flip = flip.copy()
        if selected_model is not model:
            del model

    if selected is None or selected_model is None or selected_val_scores is None or selected_predictions is None or selected_flip is None:
        raise ValueError("No FP guard candidate was selected")

    payload = {
        "schema": "axon_loop128_fp_guard_selector_payload_v1",
        "protocol": "primary predictor is default; guard may only flip primary 1 to 0 when conservative predictor is 0",
        "model": selected_model,
        "selected": selected,
        "feature_names": FP_GUARD_FEATURE_NAMES,
        "key_columns": key_columns,
        "primary_prediction_role": "with-logreg current best",
        "conservative_prediction_role": "no-logreg low-FP reference",
        "identity_feature_policy": (
            "source_path/source_sha256/cache_path/sample_index/split/filename/extension/directory are alignment or audit fields only"
        ),
    }
    model_path = output_dir / "loop128_fp_guard_selector.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle)
    write_predictions(output_dir / "loop128_fp_guard_val_predictions.csv", val, selected_val_scores, selected_predictions, selected_flip)

    report = {
        "schema": "axon_loop128_fp_guard_selector_v1",
        "protocol": payload["protocol"],
        "primary_train_predictions": str(resolve_path(args.primary_train_predictions)),
        "conservative_train_predictions": str(resolve_path(args.conservative_train_predictions)),
        "primary_val_predictions": str(resolve_path(args.primary_val_predictions)),
        "conservative_val_predictions": str(resolve_path(args.conservative_val_predictions)),
        "model_path": str(model_path),
        "feature_names": FP_GUARD_FEATURE_NAMES,
        "primary_val": primary_val_metrics,
        "conservative_val": conservative_val_metrics,
        "possible_flip_summary": {
            "train": int(np.count_nonzero(train_possible)),
            "train_beneficial_label0": int(np.count_nonzero(train_possible & (train.labels == 0))),
            "train_harmful_label1": int(np.count_nonzero(train_possible & (train.labels == 1))),
            "val": int(np.count_nonzero(val_possible)),
            "val_beneficial_label0": int(np.count_nonzero(val_possible & (val.labels == 0))),
            "val_harmful_label1": int(np.count_nonzero(val_possible & (val.labels == 1))),
        },
        "selected_by_val": selected,
        "candidates": candidates,
        "elapsed_sec": float(time.perf_counter() - start_all),
    }
    report_path = output_dir / "loop128_fp_guard_selector_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected_by_val": selected, "primary_val": primary_val_metrics, "conservative_val": conservative_val_metrics}, ensure_ascii=False, indent=2))
    print(f"JSON: {report_path}")
    return 0


def eval_command(args: argparse.Namespace) -> int:
    with resolve_path(args.model).open("rb") as handle:
        payload = pickle.load(handle)
    key_columns = tuple(payload["key_columns"])
    aligned = align_predictions(args.primary_predictions, args.conservative_predictions, key_columns)
    features = build_guard_features(aligned.primary_prob, aligned.conservative_prob)
    possible = possible_flip_mask(aligned)
    guard_scores = np.zeros(aligned.labels.shape[0], dtype=np.float32)
    if int(np.count_nonzero(possible)):
        guard_scores[possible] = score_model(payload["model"], features[possible])
    threshold = float(args.allow_threshold) if args.allow_threshold is not None else float(payload["selected"]["allow_threshold"])
    predictions, flip = apply_guard(aligned, guard_scores, threshold)
    metrics = metrics_at_predictions(aligned.labels, predictions)
    write_predictions(resolve_path(args.output_predictions_csv), aligned, guard_scores, predictions, flip)
    report = {
        "schema": "axon_loop128_fp_guard_selector_frozen_eval_v1",
        "protocol": "frozen FP guard selector only; no fitting and no threshold sweep",
        "model": str(resolve_path(args.model)),
        "primary_predictions": str(resolve_path(args.primary_predictions)),
        "conservative_predictions": str(resolve_path(args.conservative_predictions)),
        "output_predictions_csv": str(resolve_path(args.output_predictions_csv)),
        "allow_threshold": threshold,
        "selected_from_val": payload["selected"],
        "records": {"total": int(aligned.labels.shape[0]), "kept": int(aligned.labels.shape[0])},
        "possible_flip_rows": int(np.count_nonzero(possible)),
        "flips": int(np.count_nonzero(flip)),
        "flipped_label0": int(np.count_nonzero(flip & (aligned.labels == 0))),
        "flipped_label1": int(np.count_nonzero(flip & (aligned.labels == 1))),
        "metrics": metrics,
    }
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"JSON: {output_json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate Loop128 FP guard selector.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--primary-train-predictions", type=Path, required=True)
    train.add_argument("--conservative-train-predictions", type=Path, required=True)
    train.add_argument("--primary-val-predictions", type=Path, required=True)
    train.add_argument("--conservative-val-predictions", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--allow-thresholds", default="0.05:0.95:0.01")
    train.add_argument("--key-columns", default="sample_index,source_sha256")
    train.add_argument("--seed", type=int, default=128)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--primary-predictions", type=Path, required=True)
    evaluate.add_argument("--conservative-predictions", type=Path, required=True)
    evaluate.add_argument("--output-json", type=Path, required=True)
    evaluate.add_argument("--output-predictions-csv", type=Path, required=True)
    evaluate.add_argument("--allow-threshold", type=float, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        return train_command(args)
    if args.command == "eval":
        return eval_command(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
