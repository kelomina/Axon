#!/usr/bin/env python3
"""Train a strict OOF residual gate for controlled candidate overrides.

Loop42 is a validation-only experiment. It trains base/candidate predictors with
out-of-fold train scores, then trains a small gate to decide when a candidate
prediction should override the base prediction. Identity fields are used only to
align rows and audit labels, never as model features.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from config import AxonExperimentConfig  # noqa: E402
from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402
from train_byte_ngram_sgd import (  # noqa: E402
    ByteHashConfig,
    predict_scores as predict_byte_scores,
    train_candidate as train_byte_candidate,
)
from train_loop44_region_byte_ngram import (  # noqa: E402
    RegionHashConfig,
    predict_scores as predict_region_scores,
    train_candidate as train_region_candidate,
)
from train_stage2_cache_matrix import (  # noqa: E402
    FeatureConfig,
    append_feature_columns,
    assert_stage2_feature_names_safe,
    build_matrix,
    clean_slice_metrics,
    filter_model_candidates,
    metrics_at_threshold,
    model_candidates,
    parse_content_pe_v2_groups,
    parse_thresholds,
    predict_scores,
    read_prediction_rows,
    resolve_path,
    select_best_threshold,
    write_predictions,
)


LOOP28_VAL_F1 = 0.9919048570857486
LOOP28_VAL_ERRORS = 162
LOOP42_TEST10K_ERROR_GATE = 152


def _safe_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32, copy=False), 1.0e-6, 1.0 - 1.0e-6)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32, copy=False)


def build_gate_score_features(base_scores: np.ndarray, candidate_scores: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Build non-identity gate features from OOF/frozen model scores only."""

    base = np.clip(base_scores.astype(np.float32, copy=False), 1.0e-6, 1.0 - 1.0e-6)
    candidate = np.clip(candidate_scores.astype(np.float32, copy=False), 1.0e-6, 1.0 - 1.0e-6)
    base_logit = _safe_logit(base)
    candidate_logit = _safe_logit(candidate)
    features = np.empty((base.shape[0], 9), dtype=np.float32)
    features[:, 0] = base
    features[:, 1] = candidate
    features[:, 2] = candidate - base
    features[:, 3] = np.abs(candidate - base)
    features[:, 4] = np.abs(base - 0.5) * 2.0
    features[:, 5] = np.abs(candidate - 0.5) * 2.0
    features[:, 6] = base_logit
    features[:, 7] = candidate_logit
    features[:, 8] = candidate_logit - base_logit
    names = [
        "gate_base_score",
        "gate_candidate_score",
        "gate_score_delta",
        "gate_abs_score_delta",
        "gate_base_confidence",
        "gate_candidate_confidence",
        "gate_base_logit",
        "gate_candidate_logit",
        "gate_logit_delta",
    ]
    assert_no_identity_feature_names(names, context="Loop42 gate score features")
    return features, names


def build_gate_matrix(
    content_matrix: np.ndarray,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    include_content_features: bool,
) -> tuple[np.ndarray, list[str]]:
    score_features, score_names = build_gate_score_features(base_scores, candidate_scores)
    if not include_content_features:
        return score_features, score_names
    content_names = [f"gate_content_feature_{index}" for index in range(content_matrix.shape[1])]
    assert_no_identity_feature_names(content_names, context="Loop42 gate content feature aliases")
    matrix = append_feature_columns(score_features, content_matrix)
    return matrix, score_names + content_names


def gate_model_candidates(seed: int) -> list[tuple[str, object]]:
    return [
        (
            "gate_logreg_balanced_c0.25",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=5000,
                    solver="liblinear",
                    C=0.25,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ),
        (
            "gate_logreg_balanced_c1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=5000,
                    solver="liblinear",
                    C=1.0,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ),
        (
            "gate_hgb_leaf7",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_leaf_nodes=7,
                l2_regularization=1.0e-3,
                max_iter=160,
                random_state=seed,
            ),
        ),
        (
            "gate_hgb_leaf15",
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=1.0e-3,
                max_iter=180,
                random_state=seed,
            ),
        ),
    ]


def fit_with_optional_weights(model, matrix: np.ndarray, labels: np.ndarray, weights: Optional[np.ndarray] = None):
    if weights is None:
        model.fit(matrix, labels)
        return model
    if hasattr(model, "steps") and getattr(model, "steps"):
        final_step_name = model.steps[-1][0]
        model.fit(matrix, labels, **{f"{final_step_name}__sample_weight": weights})
        return model
    try:
        model.fit(matrix, labels, sample_weight=weights)
    except (TypeError, ValueError):
        model.fit(matrix, labels)
    return model


def oof_stage2_scores(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    specs: Sequence[tuple[str, object]],
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[object], list[dict]]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros((train_x.shape[0], len(specs)), dtype=np.float32)
    val = np.zeros((val_x.shape[0], len(specs)), dtype=np.float32)
    fitted = []
    reports = []
    for model_index, (name, prototype) in enumerate(specs):
        start = time.perf_counter()
        fold_reports = []
        for fold_index, (fit_idx, holdout_idx) in enumerate(splitter.split(train_x, train_y), start=1):
            fold_model = clone(prototype)
            fit_with_optional_weights(fold_model, train_x[fit_idx], train_y[fit_idx])
            fold_scores = predict_scores(fold_model, train_x[holdout_idx])
            oof[holdout_idx, model_index] = fold_scores
            fold_best = select_best_threshold(fold_scores, train_y[holdout_idx], [0.5])
            fold_reports.append({"fold": fold_index, "rows": int(holdout_idx.shape[0]), "at_0_5": fold_best})
            print(
                f"[stage2-oof] {name} fold={fold_index}/{folds} "
                f"errors@0.5={fold_best['errors']}",
                flush=True,
            )

        full_model = clone(prototype)
        fit_with_optional_weights(full_model, train_x, train_y)
        val[:, model_index] = predict_scores(full_model, val_x)
        fitted.append(full_model)
        reports.append(
            {
                "name": name,
                "fit_sec": time.perf_counter() - start,
                "folds": fold_reports,
            }
        )
    return oof, val, fitted, reports


def oof_byte_ngram_scores(
    *,
    train_records: Sequence[dict],
    train_y: np.ndarray,
    val_records: Sequence[dict],
    config: ByteHashConfig,
    alpha: float,
    l1_ratio: float,
    epochs: int,
    batch_size: int,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, object, dict]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    train_records_list = list(train_records)
    oof = np.zeros(train_y.shape[0], dtype=np.float32)
    fold_reports = []
    start = time.perf_counter()
    for fold_index, (fit_idx, holdout_idx) in enumerate(splitter.split(np.zeros_like(train_y), train_y), start=1):
        fold_train_records = [train_records_list[int(index)] for index in fit_idx]
        fold_holdout_records = [train_records_list[int(index)] for index in holdout_idx]
        fold_model = train_byte_candidate(
            fold_train_records,
            config,
            alpha=alpha,
            l1_ratio=l1_ratio,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + fold_index,
        )
        holdout_labels, holdout_scores = predict_byte_scores(fold_model, fold_holdout_records, config, batch_size)
        if not np.array_equal(holdout_labels, train_y[holdout_idx]):
            raise ValueError(f"Byte n-gram fold label alignment failed at fold {fold_index}")
        oof[holdout_idx] = holdout_scores.astype(np.float32, copy=False)
        fold_reports.append({"fold": fold_index, "rows": int(holdout_idx.shape[0])})
        print(f"[byte-oof] fold={fold_index}/{folds} rows={holdout_idx.shape[0]}", flush=True)

    full_model = train_byte_candidate(
        train_records_list,
        config,
        alpha=alpha,
        l1_ratio=l1_ratio,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    val_labels, val_scores = predict_byte_scores(full_model, val_records, config, batch_size)
    report = {
        "name": "byte_ngram_sgd",
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "fit_sec": time.perf_counter() - start,
        "folds": fold_reports,
    }
    return oof, val_scores.astype(np.float32, copy=False), full_model, {"labels": val_labels, **report}


def oof_region_ngram_scores(
    *,
    train_records: Sequence[dict],
    train_y: np.ndarray,
    val_records: Sequence[dict],
    config: RegionHashConfig,
    alpha: float,
    l1_ratio: float,
    epochs: int,
    batch_size: int,
    folds: int,
    seed: int,
    allow_missing_source: bool,
) -> tuple[np.ndarray, np.ndarray, object, dict]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    train_records_list = list(train_records)
    oof = np.zeros(train_y.shape[0], dtype=np.float32)
    fold_reports = []
    start = time.perf_counter()
    for fold_index, (fit_idx, holdout_idx) in enumerate(splitter.split(np.zeros_like(train_y), train_y), start=1):
        fold_train_records = [train_records_list[int(index)] for index in fit_idx]
        fold_holdout_records = [train_records_list[int(index)] for index in holdout_idx]
        fold_model = train_region_candidate(
            fold_train_records,
            config,
            alpha=alpha,
            l1_ratio=l1_ratio,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + fold_index,
            allow_missing_source=allow_missing_source,
        )
        holdout_labels, holdout_scores = predict_region_scores(
            fold_model,
            fold_holdout_records,
            config,
            batch_size,
            allow_missing_source=allow_missing_source,
        )
        if not np.array_equal(holdout_labels, train_y[holdout_idx]):
            raise ValueError(f"Region n-gram fold label alignment failed at fold {fold_index}")
        oof[holdout_idx] = holdout_scores.astype(np.float32, copy=False)
        fold_reports.append({"fold": fold_index, "rows": int(holdout_idx.shape[0])})
        print(f"[region-oof] fold={fold_index}/{folds} rows={holdout_idx.shape[0]}", flush=True)

    full_model = train_region_candidate(
        train_records_list,
        config,
        alpha=alpha,
        l1_ratio=l1_ratio,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        allow_missing_source=allow_missing_source,
    )
    val_labels, val_scores = predict_region_scores(
        full_model,
        val_records,
        config,
        batch_size,
        allow_missing_source=allow_missing_source,
    )
    report = {
        "name": "region_byte_ngram_sgd",
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "allow_missing_source": bool(allow_missing_source),
        "fit_sec": time.perf_counter() - start,
        "folds": fold_reports,
    }
    return oof, val_scores.astype(np.float32, copy=False), full_model, {"labels": val_labels, **report}


def prediction_metrics(labels: np.ndarray, predictions: np.ndarray, scores: Optional[np.ndarray] = None) -> dict:
    tn, fp, fn, tp = confusion_matrix(labels, predictions.astype(np.int64), labels=[0, 1]).ravel()
    result = {
        "samples": int(labels.shape[0]),
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
    if scores is not None and len(np.unique(labels)) == 2:
        result["auc"] = float(roc_auc_score(labels, scores))
    return result


def override_predictions(
    *,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    gate_scores: np.ndarray,
    base_threshold: float,
    candidate_threshold: float,
    gate_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_pred = (base_scores >= base_threshold).astype(np.int64)
    candidate_pred = (candidate_scores >= candidate_threshold).astype(np.int64)
    override = gate_scores >= gate_threshold
    final_pred = np.where(override, candidate_pred, base_pred).astype(np.int64)
    final_scores = np.where(override, candidate_scores, base_scores).astype(np.float32, copy=False)
    return final_pred, final_scores, override


def gate_training_targets(
    labels: np.ndarray,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    base_threshold: float,
    candidate_threshold: float,
    neutral_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    base_pred = (base_scores >= base_threshold).astype(np.int64)
    candidate_pred = (candidate_scores >= candidate_threshold).astype(np.int64)
    base_correct = base_pred == labels
    candidate_correct = candidate_pred == labels
    beneficial = (~base_correct) & candidate_correct
    harmful = base_correct & (~candidate_correct)
    neutral = ~(beneficial | harmful)
    targets = beneficial.astype(np.int64)
    weights = np.full(labels.shape[0], float(neutral_weight), dtype=np.float32)
    weights[beneficial | harmful] = 1.0
    summary = {
        "beneficial_overrides": int(beneficial.sum()),
        "harmful_overrides": int(harmful.sum()),
        "neutral_rows": int(neutral.sum()),
        "base_errors": int((~base_correct).sum()),
        "candidate_errors": int((~candidate_correct).sum()),
        "weighted_rows": float(weights.sum()),
    }
    return targets, weights, summary


def select_gate_threshold(
    *,
    labels: np.ndarray,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    gate_scores: np.ndarray,
    base_threshold: float,
    candidate_threshold: float,
    gate_thresholds: Sequence[float],
) -> dict:
    rows = []
    for gate_threshold in gate_thresholds:
        predictions, final_scores, override = override_predictions(
            base_scores=base_scores,
            candidate_scores=candidate_scores,
            gate_scores=gate_scores,
            base_threshold=base_threshold,
            candidate_threshold=candidate_threshold,
            gate_threshold=float(gate_threshold),
        )
        metrics = prediction_metrics(labels, predictions, final_scores)
        metrics["gate_threshold"] = float(gate_threshold)
        metrics["override_count"] = int(override.sum())
        metrics["override_ratio"] = float(override.mean())
        rows.append(metrics)
    rows.sort(key=lambda row: (row["f1"], -row["errors"], -row["gate_threshold"]), reverse=True)
    return rows[0]


def _prediction_key(row: dict, key_column: str) -> str:
    value = row.get(key_column, "")
    if value == "":
        raise ValueError(f"Missing alignment key column {key_column}")
    return str(value)


def align_external_scores(
    *,
    rows: Sequence[dict],
    prediction_path: Path,
    probability_column: str,
    key_column: str,
) -> tuple[np.ndarray, dict]:
    needed_keys = {_prediction_key(row, key_column) for row in rows}
    by_key: dict[str, dict] = {}
    scanned_rows = 0
    if not needed_keys:
        return np.empty(0, dtype=np.float32), {
            "path": str(resolve_path(prediction_path)),
            "probability_column": probability_column,
            "key_column": key_column,
            "rows": 0,
            "external_rows": 0,
            "external_rows_scanned": 0,
            "matched_external_rows": 0,
            "sha_checked": 0,
        }
    with resolve_path(prediction_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for external_row in reader:
            scanned_rows += 1
            key = _prediction_key(external_row, key_column)
            if key not in needed_keys:
                continue
            if key in by_key:
                raise ValueError(f"External prediction duplicate key {key!r} in {prediction_path}")
            by_key[key] = external_row
            if len(by_key) == len(needed_keys):
                break
    scores = np.empty(len(rows), dtype=np.float32)
    sha_checked = 0
    for index, row in enumerate(rows):
        key = _prediction_key(row, key_column)
        other = by_key.get(key)
        if other is None:
            raise ValueError(f"External prediction missing key {key!r} from {prediction_path}")
        if int(other["label"]) != int(row["label"]):
            raise ValueError(f"External prediction label mismatch for key {key!r}")
        left_sha = str(row.get("source_sha256") or "").casefold()
        right_sha = str(other.get("source_sha256") or "").casefold()
        if left_sha and right_sha:
            sha_checked += 1
            if left_sha != right_sha:
                raise ValueError(f"External prediction SHA mismatch for key {key!r}")
        scores[index] = float(other[probability_column])
    return scores, {
        "path": str(resolve_path(prediction_path)),
        "probability_column": probability_column,
        "key_column": key_column,
        "rows": len(rows),
        "external_rows": scanned_rows,
        "external_rows_scanned": scanned_rows,
        "matched_external_rows": len(by_key),
        "sha_checked": sha_checked,
    }


def write_gate_predictions(
    path: Path,
    rows: Sequence[dict],
    labels: np.ndarray,
    *,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    gate_scores: np.ndarray,
    final_scores: np.ndarray,
    final_predictions: np.ndarray,
    override_mask: np.ndarray,
    selected_candidate: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "cache_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "base_prob_malicious",
        "candidate_prob_malicious",
        "gate_prob_override",
        "final_prob_malicious",
        "prediction",
        "correct",
        "override",
        "selected_candidate",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row, label, base, candidate, gate, final_score, prediction, override in zip(
            rows,
            labels,
            base_scores,
            candidate_scores,
            gate_scores,
            final_scores,
            final_predictions,
            override_mask,
        ):
            writer.writerow(
                {
                    "source_path": row.get("source_path", ""),
                    "cache_path": row.get("cache_path", ""),
                    "source_sha256": row.get("source_sha256", ""),
                    "label": int(label),
                    "split": row.get("split", ""),
                    "sample_index": row.get("sample_index", ""),
                    "base_prob_malicious": f"{float(base):.10f}",
                    "candidate_prob_malicious": f"{float(candidate):.10f}",
                    "gate_prob_override": f"{float(gate):.10f}",
                    "final_prob_malicious": f"{float(final_score):.10f}",
                    "prediction": int(prediction),
                    "correct": int(prediction) == int(label),
                    "override": bool(override),
                    "selected_candidate": selected_candidate,
                }
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a strict OOF residual gate for Loop42.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--thresholds", default="0.05:0.95:0.005")
    parser.add_argument("--gate-thresholds", default="0.50:0.95:0.01")
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--chunk-count", type=int, default=16)
    parser.add_argument("--feature-set", choices=["tabular", "extended"], default="extended")
    parser.add_argument("--content-pe-features", action="store_true")
    parser.add_argument("--content-pe-cache-dir", type=Path, default=None)
    parser.add_argument("--content-pe-v2-features", action="store_true")
    parser.add_argument("--content-pe-v2-cache-dir", type=Path, default=None)
    parser.add_argument("--content-pe-v2-groups", default="all")
    parser.add_argument("--content-string-features", action="store_true")
    parser.add_argument("--content-string-cache-dir", type=Path, default=None)
    parser.add_argument("--content-cert-features", action="store_true")
    parser.add_argument("--content-cert-cache-dir", type=Path, default=None)
    parser.add_argument("--drop-base-prob-features", action="store_true")
    parser.add_argument("--gate-content-features", action="store_true")
    parser.add_argument("--base-model-candidate", default="hgb_lr0.06_leaf31_l2_0")
    parser.add_argument(
        "--candidate-model-candidates",
        default="hgb_lr0.08_leaf31_l2_1e-3,extra_trees_300_leaf1,extra_trees_500_leaf2,rf_300_leaf2,logreg_l2_c1",
    )
    parser.add_argument("--gate-model-candidates", default="")
    parser.add_argument("--include-byte-ngram", action="store_true")
    parser.add_argument("--byte-ngram-n-features", type=int, default=2**21)
    parser.add_argument("--byte-ngram-prefix-len", type=int, default=4096)
    parser.add_argument("--byte-ngram-min", type=int, default=2)
    parser.add_argument("--byte-ngram-max", type=int, default=5)
    parser.add_argument("--byte-ngram-stride", type=int, default=2)
    parser.add_argument("--byte-ngram-alpha", type=float, default=3.0e-6)
    parser.add_argument("--byte-ngram-l1-ratio", type=float, default=0.0)
    parser.add_argument("--byte-ngram-epochs", type=int, default=3)
    parser.add_argument("--byte-ngram-batch-size", type=int, default=256)
    parser.add_argument("--byte-ngram-include-byte-hist", action="store_true")
    parser.add_argument("--byte-ngram-include-cache-features", action="store_true")
    parser.add_argument("--include-region-ngram", action="store_true")
    parser.add_argument("--region-ngram-n-features", type=int, default=2**21)
    parser.add_argument("--region-ngram-prefix-len", type=int, default=4096)
    parser.add_argument("--region-ngram-window", type=int, default=1024)
    parser.add_argument("--region-ngram-tail-window", type=int, default=1024)
    parser.add_argument("--region-ngram-min", type=int, default=2)
    parser.add_argument("--region-ngram-max", type=int, default=5)
    parser.add_argument("--region-ngram-stride", type=int, default=2)
    parser.add_argument("--region-ngram-alpha", type=float, default=3.0e-6)
    parser.add_argument("--region-ngram-l1-ratio", type=float, default=0.0)
    parser.add_argument("--region-ngram-epochs", type=int, default=2)
    parser.add_argument("--region-ngram-batch-size", type=int, default=256)
    parser.add_argument("--region-ngram-include-prefix-features", action="store_true")
    parser.add_argument("--region-ngram-include-full-ngram-features", action="store_true")
    parser.add_argument("--region-ngram-include-byte-hist", action="store_true")
    parser.add_argument("--region-ngram-include-cache-features", action="store_true")
    parser.add_argument("--region-ngram-no-region-scalar-features", action="store_true")
    parser.add_argument("--region-ngram-allow-missing-source", action="store_true")
    parser.add_argument("--neutral-weight", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-val-predictions", type=Path, default=None)
    parser.add_argument("--baseline-probability-column", default="stage2_prob_malicious")
    parser.add_argument("--alignment-key-column", default="sample_index")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    checkpoint = load_safe_checkpoint(resolve_path(args.checkpoint), map_location="cpu")
    checkpoint_config = AxonExperimentConfig.from_dict(dict(checkpoint["config"]))
    del checkpoint
    gc.collect()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    content_pe_cache_dir = None
    if args.content_pe_features:
        content_pe_cache_dir = resolve_path(args.content_pe_cache_dir or (output_dir / "content_pe_cache_v1"))
    content_pe_v2_cache_dir = None
    if args.content_pe_v2_features:
        content_pe_v2_cache_dir = resolve_path(args.content_pe_v2_cache_dir or (output_dir / "content_pe_v2_cache"))
    content_string_cache_dir = None
    if args.content_string_features:
        content_string_cache_dir = resolve_path(args.content_string_cache_dir or (output_dir / "content_string_cache_v1"))
    content_cert_cache_dir = None
    if args.content_cert_features:
        content_cert_cache_dir = resolve_path(args.content_cert_cache_dir or (output_dir / "content_cert_cache_v1"))

    feature_config = FeatureConfig(
        prefix_len=max(0, int(args.prefix_len)),
        chunk_count=max(1, int(args.chunk_count)),
        include_pe=True,
        include_stat=True,
        include_lightweight=args.feature_set == "extended",
        include_byte_summary=args.feature_set == "extended",
        include_content_pe=bool(args.content_pe_features),
        content_cache_dir=str(content_pe_cache_dir) if content_pe_cache_dir is not None else None,
        include_content_pe_v2=bool(args.content_pe_v2_features),
        content_pe_v2_cache_dir=str(content_pe_v2_cache_dir) if content_pe_v2_cache_dir is not None else None,
        content_pe_v2_groups=parse_content_pe_v2_groups(args.content_pe_v2_groups),
        include_content_string=bool(args.content_string_features),
        content_string_cache_dir=str(content_string_cache_dir) if content_string_cache_dir is not None else None,
        include_content_cert=bool(args.content_cert_features),
        content_cert_cache_dir=str(content_cert_cache_dir) if content_cert_cache_dir is not None else None,
    )
    safe_feature_name_groups = assert_stage2_feature_names_safe(feature_config, checkpoint_config=checkpoint_config)

    train_rows = read_prediction_rows(args.train_predictions, args.max_train_rows)
    val_rows = read_prediction_rows(args.val_predictions, args.max_val_rows)
    train_x, train_y, train_base_exported, train_kept_rows, train_counts = build_matrix(
        train_rows, checkpoint_config, feature_config
    )
    val_x, val_y, val_base_exported, val_kept_rows, val_counts = build_matrix(
        val_rows, checkpoint_config, feature_config
    )
    del train_rows
    del val_rows
    gc.collect()
    dropped_feature_count = 0
    if args.drop_base_prob_features:
        dropped_feature_count = 6
        train_x = train_x[:, dropped_feature_count:].astype(np.float32, copy=False)
        val_x = val_x[:, dropped_feature_count:].astype(np.float32, copy=False)

    print(f"[matrix] train={train_x.shape} val={val_x.shape}", flush=True)
    thresholds = parse_thresholds(args.thresholds)
    gate_thresholds = parse_thresholds(args.gate_thresholds)
    folds = min(max(2, int(args.folds)), int(np.bincount(train_y).min()))

    base_specs = filter_model_candidates(model_candidates(int(args.seed)), args.base_model_candidate)
    if len(base_specs) != 1:
        raise ValueError(f"Expected exactly one base model candidate, got {[name for name, _ in base_specs]}")
    candidate_specs = filter_model_candidates(model_candidates(int(args.seed)), args.candidate_model_candidates)
    if not candidate_specs and not args.include_byte_ngram and not args.include_region_ngram:
        raise ValueError("No candidate override models selected")

    stage2_specs = base_specs + candidate_specs
    stage2_oof, stage2_val, fitted_stage2_models, stage2_reports = oof_stage2_scores(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        specs=stage2_specs,
        folds=folds,
        seed=int(args.seed),
    )

    base_name = stage2_specs[0][0]
    base_oof_scores = stage2_oof[:, 0]
    base_val_scores = stage2_val[:, 0]
    base_train_best = select_best_threshold(base_oof_scores, train_y, thresholds)
    base_val_at_train_threshold = metrics_at_threshold(base_val_scores, val_y, float(base_train_best["threshold"]))
    baseline_external = None
    baseline_external_alignment = None
    if args.baseline_val_predictions is not None:
        external_scores, baseline_external_alignment = align_external_scores(
            rows=val_kept_rows,
            prediction_path=args.baseline_val_predictions,
            probability_column=args.baseline_probability_column,
            key_column=args.alignment_key_column,
        )
        external_best = select_best_threshold(external_scores, val_y, thresholds)
        external_locked = metrics_at_threshold(external_scores, val_y, 0.5)
        baseline_external = {
            "best_sweep": external_best,
            "locked_threshold_0_5": external_locked,
        }

    candidate_score_sets: list[dict] = []
    for index, (name, _model) in enumerate(candidate_specs, start=1):
        candidate_score_sets.append(
            {
                "name": name,
                "kind": "stage2",
                "train_oof_scores": stage2_oof[:, index],
                "val_scores": stage2_val[:, index],
                "model": fitted_stage2_models[index],
            }
        )

    byte_report = None
    byte_model = None
    if args.include_byte_ngram:
        byte_config = ByteHashConfig(
            n_features=int(args.byte_ngram_n_features),
            prefix_len=int(args.byte_ngram_prefix_len),
            ngram_min=int(args.byte_ngram_min),
            ngram_max=int(args.byte_ngram_max),
            ngram_stride=int(args.byte_ngram_stride),
            include_byte_hist=bool(args.byte_ngram_include_byte_hist),
            include_cache_features=bool(args.byte_ngram_include_cache_features),
            max_byte_length=checkpoint_config.max_byte_length,
            pe_feature_dim=checkpoint_config.pe_feature_dim,
            stat_feature_dim=checkpoint_config.stat_feature_dim,
            lightweight_feature_dim=checkpoint_config.lightweight_feature_dim,
        )
        byte_oof, byte_val_scores, byte_model, byte_report = oof_byte_ngram_scores(
            train_records=train_kept_rows,
            train_y=train_y,
            val_records=val_kept_rows,
            config=byte_config,
            alpha=float(args.byte_ngram_alpha),
            l1_ratio=float(args.byte_ngram_l1_ratio),
            epochs=int(args.byte_ngram_epochs),
            batch_size=int(args.byte_ngram_batch_size),
            folds=folds,
            seed=int(args.seed),
        )
        if not np.array_equal(byte_report.pop("labels"), val_y):
            raise ValueError("Byte n-gram validation labels are not aligned with Stage-2 validation rows")
        candidate_score_sets.append(
            {
                "name": "byte_ngram_sgd",
                "kind": "byte_ngram",
                "train_oof_scores": byte_oof,
                "val_scores": byte_val_scores,
                "model": byte_model,
            }
        )

    region_report = None
    region_model = None
    if args.include_region_ngram:
        region_config = RegionHashConfig(
            n_features=int(args.region_ngram_n_features),
            prefix_len=int(args.region_ngram_prefix_len),
            ngram_min=int(args.region_ngram_min),
            ngram_max=int(args.region_ngram_max),
            ngram_stride=int(args.region_ngram_stride),
            include_prefix_features=bool(args.region_ngram_include_prefix_features),
            include_full_ngram_features=bool(args.region_ngram_include_full_ngram_features),
            include_region_ngram_features=True,
            include_region_scalar_features=not bool(args.region_ngram_no_region_scalar_features),
            include_byte_hist=bool(args.region_ngram_include_byte_hist),
            include_cache_features=bool(args.region_ngram_include_cache_features),
            region_window=max(1, int(args.region_ngram_window)),
            tail_window=max(1, int(args.region_ngram_tail_window)),
            max_byte_length=checkpoint_config.max_byte_length,
            pe_feature_dim=checkpoint_config.pe_feature_dim,
            stat_feature_dim=checkpoint_config.stat_feature_dim,
            lightweight_feature_dim=checkpoint_config.lightweight_feature_dim,
        )
        region_oof, region_val_scores, region_model, region_report = oof_region_ngram_scores(
            train_records=train_kept_rows,
            train_y=train_y,
            val_records=val_kept_rows,
            config=region_config,
            alpha=float(args.region_ngram_alpha),
            l1_ratio=float(args.region_ngram_l1_ratio),
            epochs=int(args.region_ngram_epochs),
            batch_size=int(args.region_ngram_batch_size),
            folds=folds,
            seed=int(args.seed),
            allow_missing_source=bool(args.region_ngram_allow_missing_source),
        )
        if not np.array_equal(region_report.pop("labels"), val_y):
            raise ValueError("Region n-gram validation labels are not aligned with Stage-2 validation rows")
        region_report["hash_config"] = region_config.__dict__
        candidate_score_sets.append(
            {
                "name": "region_byte_ngram_sgd",
                "kind": "region_byte_ngram",
                "train_oof_scores": region_oof,
                "val_scores": region_val_scores,
                "model": region_model,
            }
        )

    selected_gate_candidates = gate_model_candidates(int(args.seed))
    selected_gate_names = [item.strip() for item in args.gate_model_candidates.split(",") if item.strip()]
    if selected_gate_names:
        selected_gate_candidates = [
            (name, model) for name, model in selected_gate_candidates if name in set(selected_gate_names)
        ]
    if not selected_gate_candidates:
        raise ValueError("No gate model candidates selected")

    best_key = None
    selected = None
    selected_gate_model = None
    selected_gate_scores = None
    selected_candidate = None
    selected_gate_feature_names = None
    candidate_reports = []
    include_content_for_gate = bool(args.gate_content_features)
    for candidate in candidate_score_sets:
        candidate_name = candidate["name"]
        candidate_train_scores = candidate["train_oof_scores"]
        candidate_val_scores = candidate["val_scores"]
        candidate_train_best = select_best_threshold(candidate_train_scores, train_y, thresholds)
        candidate_val_at_train_threshold = metrics_at_threshold(
            candidate_val_scores,
            val_y,
            float(candidate_train_best["threshold"]),
        )
        targets, weights, target_summary = gate_training_targets(
            train_y,
            base_oof_scores,
            candidate_train_scores,
            base_threshold=float(base_train_best["threshold"]),
            candidate_threshold=float(candidate_train_best["threshold"]),
            neutral_weight=float(args.neutral_weight),
        )
        if int(targets.sum()) == 0:
            print(f"[gate-skip] {candidate_name}: no beneficial train overrides", flush=True)
            continue
        train_gate_x, gate_feature_names = build_gate_matrix(
            train_x,
            base_oof_scores,
            candidate_train_scores,
            include_content_features=include_content_for_gate,
        )
        val_gate_x, _ = build_gate_matrix(
            val_x,
            base_val_scores,
            candidate_val_scores,
            include_content_features=include_content_for_gate,
        )
        gate_model_reports = []
        for gate_name, gate_prototype in selected_gate_candidates:
            start = time.perf_counter()
            gate_model = clone(gate_prototype)
            fit_with_optional_weights(gate_model, train_gate_x, targets, weights)
            gate_train_scores = predict_scores(gate_model, train_gate_x)
            gate_val_scores = predict_scores(gate_model, val_gate_x)
            train_gate_best = select_gate_threshold(
                labels=train_y,
                base_scores=base_oof_scores,
                candidate_scores=candidate_train_scores,
                gate_scores=gate_train_scores,
                base_threshold=float(base_train_best["threshold"]),
                candidate_threshold=float(candidate_train_best["threshold"]),
                gate_thresholds=gate_thresholds,
            )
            val_gate_best = select_gate_threshold(
                labels=val_y,
                base_scores=base_val_scores,
                candidate_scores=candidate_val_scores,
                gate_scores=gate_val_scores,
                base_threshold=float(base_train_best["threshold"]),
                candidate_threshold=float(candidate_train_best["threshold"]),
                gate_thresholds=gate_thresholds,
            )
            report_row = {
                "candidate": candidate_name,
                "candidate_kind": candidate["kind"],
                "gate_model": gate_name,
                "fit_sec": time.perf_counter() - start,
                "base_train_threshold": float(base_train_best["threshold"]),
                "candidate_train_threshold": float(candidate_train_best["threshold"]),
                "target_summary": target_summary,
                "train_gate_best": train_gate_best,
                "val_gate_best": val_gate_best,
                "delta_val_errors_vs_loop42_base": int(val_gate_best["errors"])
                - int(base_val_at_train_threshold["errors"]),
                "delta_val_errors_vs_loop28_locked": int(val_gate_best["errors"]) - LOOP28_VAL_ERRORS,
                "delta_val_f1_vs_loop28_locked": float(val_gate_best["f1"]) - LOOP28_VAL_F1,
            }
            gate_model_reports.append(report_row)
            candidate_key = (float(val_gate_best["f1"]), -int(val_gate_best["errors"]))
            if best_key is None or candidate_key > best_key:
                if selected_gate_model is not None:
                    del selected_gate_model
                if selected_gate_scores is not None:
                    del selected_gate_scores
                best_key = candidate_key
                selected = report_row
                selected_gate_model = gate_model
                selected_gate_scores = gate_val_scores.astype(np.float32, copy=True)
                selected_candidate = candidate
                selected_gate_feature_names = list(gate_feature_names)
            else:
                del gate_model
                del gate_val_scores
            del gate_train_scores
            gc.collect()
            print(
                f"[gate-val] candidate={candidate_name} gate={gate_name} "
                f"f1={val_gate_best['f1']:.6f} errors={val_gate_best['errors']} "
                f"overrides={val_gate_best['override_count']}",
                flush=True,
            )

        candidate_reports.append(
            {
                "name": candidate_name,
                "kind": candidate["kind"],
                "train_best": candidate_train_best,
                "val_at_train_threshold": candidate_val_at_train_threshold,
                "target_summary": target_summary,
                "gate_models": sorted(
                    gate_model_reports,
                    key=lambda row: (row["val_gate_best"]["f1"], -row["val_gate_best"]["errors"]),
                    reverse=True,
                ),
            }
        )
        del train_gate_x
        del val_gate_x
        gc.collect()

    if selected is None or selected_gate_model is None or selected_gate_scores is None or selected_candidate is None:
        raise ValueError("No gate candidate was fitted")
    if selected_gate_feature_names is None:
        raise ValueError("Selected gate feature names are missing")
    gate_feature_names = selected_gate_feature_names
    base_model = fitted_stage2_models[0]
    selected_candidate_model = selected_candidate.get("model")
    for candidate in candidate_score_sets:
        if candidate is not selected_candidate and "model" in candidate:
            candidate["model"] = None
    candidate_score_sets = [selected_candidate]
    byte_model = selected_candidate_model if selected_candidate["kind"] == "byte_ngram" else None
    region_model = selected_candidate_model if selected_candidate["kind"] == "region_byte_ngram" else None
    if selected_candidate["kind"] == "stage2":
        fitted_stage2_models = [base_model, selected_candidate_model]
    else:
        fitted_stage2_models = [base_model]
    gc.collect()

    final_predictions, final_scores, override_mask = override_predictions(
        base_scores=base_val_scores,
        candidate_scores=selected_candidate["val_scores"],
        gate_scores=selected_gate_scores,
        base_threshold=float(selected["base_train_threshold"]),
        candidate_threshold=float(selected["candidate_train_threshold"]),
        gate_threshold=float(selected["val_gate_best"]["gate_threshold"]),
    )
    val_predictions_path = output_dir / "loop42_oof_residual_gate_val_predictions.csv"
    write_gate_predictions(
        val_predictions_path,
        val_kept_rows,
        val_y,
        base_scores=base_val_scores,
        candidate_scores=selected_candidate["val_scores"],
        gate_scores=selected_gate_scores,
        final_scores=final_scores,
        final_predictions=final_predictions,
        override_mask=override_mask,
        selected_candidate=selected_candidate["name"],
    )

    selected_model_path = output_dir / "loop42_oof_residual_gate_selected_model.pkl"
    with selected_model_path.open("wb") as handle:
        pickle.dump(
            {
                "schema": "axon_loop42_oof_residual_gate_payload_v1",
                "base_name": base_name,
                "base_model": base_model,
                "selected_candidate": selected_candidate["name"],
                "selected_candidate_kind": selected_candidate["kind"],
                "selected_candidate_model": selected_candidate_model,
                "gate_model": selected_gate_model,
                "selected": selected,
                "feature_config": feature_config,
                "drop_base_prob_features": bool(args.drop_base_prob_features),
                "dropped_feature_count": int(dropped_feature_count),
                "gate_content_features": include_content_for_gate,
                "gate_feature_names": gate_feature_names,
                "checkpoint_config": checkpoint_config.to_dict(),
                "byte_ngram_model": byte_model,
                "region_ngram_model": region_model,
                "identity_feature_policy": (
                    "source_path/source_sha256/cache_path/sample_index/split/filename/extension/directory "
                    "are audit or alignment fields only and are forbidden as model features"
                ),
            },
            handle,
        )

    val_kept_count = int(val_counts.get("kept", 0)) if isinstance(val_counts, dict) else int(len(val_y))
    if val_kept_count < 20000:
        test_gate_decision = "smoke_only_not_eligible_for_test10k"
    elif int(selected["val_gate_best"]["errors"]) <= LOOP42_TEST10K_ERROR_GATE:
        test_gate_decision = "eligible_for_test10k"
    else:
        test_gate_decision = "reject_val_margin_too_small"

    report = {
        "schema": "axon_loop42_oof_residual_gate_v1",
        "protocol": (
            "base and candidate train scores are out-of-fold; gate trains only on train OOF signals; "
            "Val selects gate model/threshold; no Test-10k or full-test used"
        ),
        "checkpoint": str(resolve_path(args.checkpoint)),
        "train_predictions": str(resolve_path(args.train_predictions)),
        "val_predictions": str(resolve_path(args.val_predictions)),
        "identity_feature_policy": (
            "filename/path/extension/directory/source hash/sample id/split/row order are alignment/audit "
            "fields only and are not model features"
        ),
        "records": {"train": train_counts, "val": val_counts},
        "feature_config": feature_config.__dict__,
        "feature_name_groups": safe_feature_name_groups,
        "drop_base_prob_features": bool(args.drop_base_prob_features),
        "dropped_feature_count": int(dropped_feature_count),
        "gate_content_features": include_content_for_gate,
        "folds": folds,
        "base": {
            "name": base_name,
            "train_oof_best": base_train_best,
            "val_at_train_threshold": base_val_at_train_threshold,
        },
        "loop28_reference": {
            "locked_val_f1": LOOP28_VAL_F1,
            "locked_val_errors": LOOP28_VAL_ERRORS,
            "external_baseline": baseline_external,
            "external_alignment": baseline_external_alignment,
        },
        "stage2_reports": stage2_reports,
        "byte_ngram_report": byte_report,
        "region_ngram_report": region_report,
        "candidates": sorted(
            candidate_reports,
            key=lambda row: (
                row["gate_models"][0]["val_gate_best"]["f1"] if row["gate_models"] else -1.0,
                -(row["gate_models"][0]["val_gate_best"]["errors"] if row["gate_models"] else 10**9),
            ),
            reverse=True,
        ),
        "selected_by_val": selected,
        "model_path": str(selected_model_path),
        "val_predictions_csv": str(val_predictions_path),
        "test_ran": False,
        "test_gate_decision": test_gate_decision,
        "test10k_error_gate": LOOP42_TEST10K_ERROR_GATE,
    }
    report_path = output_dir / "loop42_oof_residual_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected_by_val": selected, "test_gate_decision": report["test_gate_decision"]}, indent=2))
    print(f"JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
