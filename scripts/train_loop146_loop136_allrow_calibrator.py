#!/usr/bin/env python3
"""Train/evaluate a Loop136-aware all-row calibrator.

This is a narrow Phase-3 candidate runner. It uses only frozen prediction
scores plus numeric PE/content features; identity columns are kept only for row
alignment, sidecar lookup, and auditable output.
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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from train_loop135_pairwise_selector import build_content_features  # noqa: E402


SCHEMA = "axon_loop146_loop136_allrow_calibrator_v1"

SCORE_FEATURE_NAMES = [
    "loop136_final_prob_malicious",
    "loop136_final_logit",
    "loop136_final_abs_margin_from_half",
    "loop136_final_prediction_malicious",
    "r5_baseline_prob_malicious",
    "r5_baseline_logit",
    "r5_baseline_abs_margin_from_half",
    "oof_candidate_prob_malicious",
    "oof_candidate_logit",
    "oof_candidate_abs_margin_from_half",
    "oof_minus_r5_prob_delta",
    "loop136_minus_r5_prob_delta",
    "loop136_minus_oof_prob_delta",
    "loop136_selector_score",
    "loop136_selector_accept_candidate",
    "loop136_r5_oof_prob_spread",
    "approx_direction_r5_zero_oof_one",
    "approx_direction_r5_one_oof_zero",
    "approx_r5_prediction_malicious",
    "approx_oof_prediction_malicious",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path, max_rows: Optional[int] = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if max_rows is not None and len(rows) >= int(max_rows):
                break
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def _float(row: dict[str, str], column: str, default: float = 0.0) -> float:
    value = str(row.get(column, "")).strip()
    return float(value) if value else float(default)


def _int(row: dict[str, str], column: str, default: int = 0) -> int:
    value = str(row.get(column, "")).strip()
    return int(value) if value else int(default)


def _clip_prob(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(np.float32, copy=False), 1.0e-6, 1.0 - 1.0e-6)


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = _clip_prob(values)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32, copy=False)


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
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
        "accuracy": float((tp + tn) / max(labels.shape[0], 1)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "errors": int(fp + fn),
    }


def parse_thresholds(text: str) -> list[float]:
    if ":" not in text:
        return [float(item.strip()) for item in text.split(",") if item.strip()]
    start_text, stop_text, step_text = text.split(":", 2)
    start = float(start_text)
    stop = float(stop_text)
    step = float(step_text)
    count = int(np.floor((stop - start) / step)) + 1
    return [round(start + step * index, 10) for index in range(count)]


def score_matrix(rows: Sequence[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    assert_no_identity_feature_names(SCORE_FEATURE_NAMES, context="Loop146 all-row score features")
    final_prob = np.asarray([_float(row, "stage2_prob_malicious") for row in rows], dtype=np.float32)
    baseline_prob = np.asarray([_float(row, "baseline_prob_malicious", _float(row, "stage2_prob_malicious")) for row in rows], dtype=np.float32)
    candidate_prob = np.asarray([_float(row, "candidate_prob_malicious", _float(row, "stage2_prob_malicious")) for row in rows], dtype=np.float32)
    selector_score = np.asarray([_float(row, "selector_score") for row in rows], dtype=np.float32)
    accept_candidate = np.asarray([_float(row, "selector_accept_candidate") for row in rows], dtype=np.float32)
    final_pred = np.asarray([_int(row, "prediction") for row in rows], dtype=np.float32)

    approx_baseline_pred = (baseline_prob >= 0.5).astype(np.float32)
    approx_candidate_pred = (candidate_prob >= 0.5).astype(np.float32)
    matrix = np.column_stack(
        [
            final_prob,
            _logit(final_prob),
            np.abs(final_prob - 0.5),
            final_pred,
            baseline_prob,
            _logit(baseline_prob),
            np.abs(baseline_prob - 0.5),
            candidate_prob,
            _logit(candidate_prob),
            np.abs(candidate_prob - 0.5),
            candidate_prob - baseline_prob,
            final_prob - baseline_prob,
            final_prob - candidate_prob,
            selector_score,
            accept_candidate,
            np.abs(candidate_prob - baseline_prob),
            ((approx_baseline_pred == 0) & (approx_candidate_pred == 1)).astype(np.float32),
            ((approx_baseline_pred == 1) & (approx_candidate_pred == 0)).astype(np.float32),
            approx_baseline_pred,
            approx_candidate_pred,
        ]
    ).astype(np.float32, copy=False)
    labels = np.asarray([_int(row, "label") for row in rows], dtype=np.int64)
    baseline_predictions = np.asarray([_int(row, "prediction") for row in rows], dtype=np.int64)
    return matrix, labels, baseline_predictions, list(SCORE_FEATURE_NAMES)


def full_feature_matrix(
    rows: Sequence[dict[str, str]],
    *,
    content_pe_cache_dir: Optional[Path] = None,
    content_pe_v2_cache_dir: Optional[Path] = None,
    content_string_cache_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    matrix, labels, baseline_predictions, names = score_matrix(rows)
    content_matrix, content_names = build_content_features(
        list(rows),
        np.arange(len(rows), dtype=np.int64),
        content_pe_cache_dir=content_pe_cache_dir,
        content_pe_v2_cache_dir=content_pe_v2_cache_dir,
        content_string_cache_dir=content_string_cache_dir,
    )
    if content_matrix.shape[1]:
        matrix = np.concatenate([matrix, content_matrix.astype(np.float32, copy=False)], axis=1)
        names.extend(content_names)
    assert_no_identity_feature_names(names, context="Loop146 all-row full features")
    return matrix.astype(np.float32, copy=False), labels, baseline_predictions, names


def model_candidates(seed: int) -> list[tuple[str, object]]:
    return [
        (
            "logreg_l2_c0.1",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="liblinear", C=0.1)),
        ),
        (
            "logreg_l2_c1",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="liblinear", C=1.0)),
        ),
        (
            "logreg_balanced_c0.1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, solver="liblinear", C=0.1, class_weight="balanced"),
            ),
        ),
        (
            "hgb_leaf7_l2",
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_leaf_nodes=7,
                l2_regularization=1.0e-2,
                max_iter=120,
                random_state=seed,
            ),
        ),
        (
            "extra_trees_300_leaf2",
            ExtraTreesClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=1,
                random_state=seed,
            ),
        ),
        (
            "random_forest_300_leaf2",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=1,
                class_weight="balanced_subsample",
                random_state=seed,
            ),
        ),
    ]


def predict_scores(model: object, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(matrix)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(matrix)
        raw = np.clip(raw, -50.0, 50.0)
        scores = 1.0 / (1.0 + np.exp(-raw))
    else:
        scores = model.predict(matrix)
    return np.asarray(scores, dtype=np.float32)


def select_threshold(labels: np.ndarray, scores: np.ndarray, thresholds: Sequence[float]) -> dict[str, object]:
    rows = []
    for threshold in thresholds:
        predictions = (scores >= float(threshold)).astype(np.int64)
        rows.append({"threshold": float(threshold), **metrics(labels, predictions)})
    rows.sort(key=lambda item: (float(item["f1"]), -int(item["errors"]), float(item["threshold"])), reverse=True)
    return rows[0]


@dataclass(frozen=True)
class Gate:
    max_errors: Optional[int]
    max_fp: Optional[int]
    max_fn: Optional[int]
    min_f1_exclusive: Optional[float]

    def passes(self, row: dict[str, object]) -> bool:
        if self.max_errors is not None and int(row["errors"]) > self.max_errors:
            return False
        if self.max_fp is not None and int(row["false_positive"]) > self.max_fp:
            return False
        if self.max_fn is not None and int(row["false_negative"]) > self.max_fn:
            return False
        if self.min_f1_exclusive is not None and float(row["f1"]) <= self.min_f1_exclusive:
            return False
        return True


def _fit_model(model: object, matrix: np.ndarray, labels: np.ndarray) -> object:
    fitted = clone(model)
    fitted.fit(matrix, labels)
    return fitted


def train(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_rows(args.train_predictions, args.max_train_rows)
    val_rows = read_rows(args.val_predictions, args.max_val_rows)
    train_x, train_y, train_baseline_pred, feature_names = full_feature_matrix(
        train_rows,
        content_pe_cache_dir=args.content_pe_cache_dir,
        content_pe_v2_cache_dir=args.content_pe_v2_cache_dir,
        content_string_cache_dir=args.content_string_cache_dir,
    )
    val_x, val_y, val_baseline_pred, val_feature_names = full_feature_matrix(
        val_rows,
        content_pe_cache_dir=args.content_pe_cache_dir,
        content_pe_v2_cache_dir=args.content_pe_v2_cache_dir,
        content_string_cache_dir=args.content_string_cache_dir,
    )
    if feature_names != val_feature_names:
        raise ValueError("Train and Val feature names differ")

    thresholds = parse_thresholds(args.thresholds)
    gate = Gate(
        max_errors=args.gate_max_errors,
        max_fp=args.gate_max_fp,
        max_fn=args.gate_max_fn,
        min_f1_exclusive=args.gate_min_f1,
    )
    baseline_val = metrics(val_y, val_baseline_pred)
    reports = []
    for name, prototype in model_candidates(int(args.seed)):
        fitted = _fit_model(prototype, train_x, train_y)
        val_scores = predict_scores(fitted, val_x)
        selected = select_threshold(val_y, val_scores, thresholds)
        selected["name"] = name
        selected["passes_gate"] = gate.passes(selected)
        reports.append({"name": name, "model": fitted, "selected": selected, "val_scores": val_scores})
        print(
            f"[val] {name} f1={float(selected['f1']):.10f} "
            f"errors={selected['errors']} fp={selected['false_positive']} fn={selected['false_negative']} "
            f"threshold={selected['threshold']:.4f} passes_gate={selected['passes_gate']}",
            flush=True,
        )

    reports.sort(
        key=lambda item: (
            bool(item["selected"]["passes_gate"]),
            float(item["selected"]["f1"]),
            -int(item["selected"]["errors"]),
            -int(item["selected"]["false_negative"]),
        ),
        reverse=True,
    )
    best = reports[0]
    selected = dict(best["selected"])
    selected_model = best["model"]
    val_scores = best["val_scores"]
    val_predictions = (val_scores >= float(selected["threshold"])).astype(np.int64)

    model_path = output_dir / "loop146_allrow_calibrator.pkl"
    payload = {
        "schema": SCHEMA,
        "protocol": "Train rows fit all-row calibrator; Val selects model and threshold; no test used",
        "model": selected_model,
        "selected": selected,
        "feature_names": feature_names,
        "train_predictions": str(resolve_path(args.train_predictions)),
        "val_predictions": str(resolve_path(args.val_predictions)),
        "content_pe_cache_dir": str(resolve_path(args.content_pe_cache_dir)) if args.content_pe_cache_dir else None,
        "content_pe_v2_cache_dir": str(resolve_path(args.content_pe_v2_cache_dir)) if args.content_pe_v2_cache_dir else None,
        "content_string_cache_dir": str(resolve_path(args.content_string_cache_dir)) if args.content_string_cache_dir else None,
    }
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle)

    val_predictions_csv = output_dir / "loop146_allrow_calibrator_val_predictions.csv"
    write_predictions(val_predictions_csv, val_rows, val_scores, val_predictions, baseline_predictions=val_baseline_pred)
    report = {
        "schema": SCHEMA,
        "protocol": payload["protocol"],
        "model_path": str(model_path),
        "train_predictions": str(resolve_path(args.train_predictions)),
        "val_predictions": str(resolve_path(args.val_predictions)),
        "output_predictions_csv": str(val_predictions_csv),
        "records": {"train": len(train_rows), "val": len(val_rows)},
        "feature_dim": int(train_x.shape[1]),
        "feature_names": feature_names,
        "baseline_val": baseline_val,
        "selected_by_val": selected,
        "val_metrics": metrics(val_y, val_predictions),
        "candidates": [item["selected"] for item in reports],
        "gate": {
            "max_errors": args.gate_max_errors,
            "max_fp": args.gate_max_fp,
            "max_fn": args.gate_max_fn,
            "min_f1_exclusive": args.gate_min_f1,
        },
        "elapsed_sec": float(time.perf_counter() - start),
    }
    report_path = output_dir / "loop146_allrow_calibrator_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"JSON: {report_path}")
    return 0


def write_predictions(
    path: Path,
    rows: Sequence[dict[str, str]],
    scores: np.ndarray,
    predictions: np.ndarray,
    *,
    baseline_predictions: np.ndarray,
) -> None:
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
        "calibrated_prob_malicious",
        "stage2_prob_malicious",
        "prediction",
        "correct",
    ]
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row, score, prediction, baseline_prediction in zip(rows, scores, predictions, baseline_predictions):
            label = _int(row, "label")
            writer.writerow(
                {
                    "source_path": row.get("source_path", ""),
                    "cache_path": row.get("cache_path", ""),
                    "source_sha256": row.get("source_sha256", ""),
                    "label": label,
                    "split": row.get("split", ""),
                    "sample_index": row.get("sample_index", ""),
                    "baseline_prediction": int(baseline_prediction),
                    "baseline_prob_malicious": f"{_float(row, 'stage2_prob_malicious'):.10f}",
                    "calibrated_prob_malicious": f"{float(score):.10f}",
                    "stage2_prob_malicious": f"{float(score):.10f}",
                    "prediction": int(prediction),
                    "correct": "True" if int(prediction) == label else "False",
                }
            )


def evaluate(args: argparse.Namespace) -> int:
    model_path = resolve_path(args.model)
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    rows = read_rows(args.predictions, args.max_rows)
    matrix, labels, baseline_predictions, feature_names = full_feature_matrix(
        rows,
        content_pe_cache_dir=args.content_pe_cache_dir,
        content_pe_v2_cache_dir=args.content_pe_v2_cache_dir,
        content_string_cache_dir=args.content_string_cache_dir,
    )
    if feature_names != list(payload["feature_names"]):
        raise ValueError("Feature names do not match frozen calibrator payload")
    threshold = float(args.threshold if args.threshold is not None else payload["selected"]["threshold"])
    scores = predict_scores(payload["model"], matrix)
    predictions = (scores >= threshold).astype(np.int64)
    output_csv = resolve_path(args.output_predictions_csv)
    write_predictions(output_csv, rows, scores, predictions, baseline_predictions=baseline_predictions)
    report = {
        "schema": "axon_loop146_loop136_allrow_calibrator_eval_v1",
        "protocol": "Frozen all-row calibrator evaluation; no fitting and no threshold sweep",
        "model": str(model_path),
        "predictions": str(resolve_path(args.predictions)),
        "output_predictions_csv": str(output_csv),
        "threshold": threshold,
        "records": {"total": len(rows), "kept": len(rows)},
        "baseline_metrics": metrics(labels, baseline_predictions),
        "metrics": metrics(labels, predictions),
        "selected_from_val": payload["selected"],
    }
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"JSON: {output_json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Loop136-aware all-row calibrator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train-predictions", type=Path, required=True)
    train_parser.add_argument("--val-predictions", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--content-pe-cache-dir", type=Path, default=None)
    train_parser.add_argument("--content-pe-v2-cache-dir", type=Path, default=None)
    train_parser.add_argument("--content-string-cache-dir", type=Path, default=None)
    train_parser.add_argument("--thresholds", default="0.05:0.95:0.005")
    train_parser.add_argument("--max-train-rows", type=int, default=None)
    train_parser.add_argument("--max-val-rows", type=int, default=None)
    train_parser.add_argument("--seed", type=int, default=146)
    train_parser.add_argument("--gate-max-errors", type=int, default=None)
    train_parser.add_argument("--gate-max-fp", type=int, default=None)
    train_parser.add_argument("--gate-max-fn", type=int, default=None)
    train_parser.add_argument("--gate-min-f1", type=float, default=None)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--model", type=Path, required=True)
    eval_parser.add_argument("--predictions", type=Path, required=True)
    eval_parser.add_argument("--output-json", type=Path, required=True)
    eval_parser.add_argument("--output-predictions-csv", type=Path, required=True)
    eval_parser.add_argument("--content-pe-cache-dir", type=Path, default=None)
    eval_parser.add_argument("--content-pe-v2-cache-dir", type=Path, default=None)
    eval_parser.add_argument("--content-string-cache-dir", type=Path, default=None)
    eval_parser.add_argument("--threshold", type=float, default=None)
    eval_parser.add_argument("--max-rows", type=int, default=None)
    eval_parser.set_defaults(func=evaluate)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
