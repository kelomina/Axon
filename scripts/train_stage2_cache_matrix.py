#!/usr/bin/env python3
"""Run cache-backed stage-2 validation matrix for Axon predictions.

The script is intentionally cache-first: it consumes exported prediction CSVs
and feature-cache NPZ files, then runs many cheap train/val candidates before
optionally confirming the best val candidate on a fixed test CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig  # noqa: E402
from dataset import _load_cached_feature_npz  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_thresholds(text: str) -> list[float]:
    if ":" in text:
        start_text, stop_text, step_text = text.split(":")
        start = float(start_text)
        stop = float(stop_text)
        step = float(step_text)
        count = int(math.floor((stop - start) / step)) + 1
        return [round(start + step * index, 10) for index in range(count)]
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError(f"All values must be positive: {values}")
    return sorted(set(values))


def read_prediction_rows(path: Path, max_rows: Optional[int] = None) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1.0e-6, 1.0 - 1.0e-6)
    return np.log(clipped / (1.0 - clipped))


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum() / 8.0)


def _byte_summary_features(byte_seq: np.ndarray, prefix_len: int, chunk_count: int) -> np.ndarray:
    byte_values = byte_seq.astype(np.uint8, copy=False)
    counts = np.bincount(byte_values, minlength=256).astype(np.float32)
    hist = counts / max(float(byte_values.shape[0]), 1.0)
    log_hist = np.log1p(counts) / np.log1p(max(float(byte_values.shape[0]), 1.0))

    prefix = byte_values[:prefix_len].astype(np.float32) / 255.0
    if prefix.shape[0] < prefix_len:
        prefix = np.pad(prefix, (0, prefix_len - prefix.shape[0]))

    chunks = np.array_split(byte_values, max(1, chunk_count))
    chunk_features = []
    for chunk in chunks:
        if chunk.size == 0:
            chunk_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        chunk_counts = np.bincount(chunk, minlength=256).astype(np.float32)
        chunk_features.extend(
            [
                float(np.mean(chunk) / 255.0),
                float(np.std(chunk) / 255.0),
                _entropy_from_counts(chunk_counts),
                float(np.count_nonzero(chunk) / max(chunk.size, 1)),
                float(np.max(chunk_counts) / max(chunk.size, 1)),
            ]
        )

    scalar = np.asarray(
        [
            _entropy_from_counts(counts),
            float(np.count_nonzero(byte_values) / max(byte_values.shape[0], 1)),
            float(np.mean(byte_values) / 255.0),
            float(np.std(byte_values) / 255.0),
            float(np.max(counts) / max(byte_values.shape[0], 1)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([hist, log_hist, prefix, np.asarray(chunk_features, dtype=np.float32), scalar])


@dataclass(frozen=True)
class FeatureConfig:
    prefix_len: int
    chunk_count: int
    include_pe: bool
    include_stat: bool
    include_lightweight: bool
    include_byte_summary: bool


def build_matrix(
    rows: Sequence[dict],
    checkpoint_config: AxonExperimentConfig,
    feature_config: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    features = []
    labels = []
    base_probs = []
    kept_rows = []
    skipped_missing_cache = 0
    for row in rows:
        cache_path = Path(row["cache_path"])
        if not cache_path.exists():
            skipped_missing_cache += 1
            continue
        label = int(row["label"])
        byte_seq, pe_feat, stat_feat, lightweight_feat, cached_label = _load_cached_feature_npz(
            cache_path,
            checkpoint_config.max_byte_length,
            checkpoint_config.pe_feature_dim,
            checkpoint_config.stat_feature_dim,
            checkpoint_config.lightweight_feature_dim,
            expected_label=label,
        )
        if cached_label != label:
            raise ValueError(f"Cache label mismatch: {cache_path}")

        prob = float(row["prob_malicious"])
        prob_arr = np.asarray(
            [
                prob,
                prob * prob,
                abs(prob - 0.5),
                math.log(max(prob, 1.0e-6)),
                math.log(max(1.0 - prob, 1.0e-6)),
                float(_safe_logit(np.asarray([prob]))[0]),
            ],
            dtype=np.float32,
        )
        parts = [prob_arr]
        if feature_config.include_stat:
            parts.append(np.nan_to_num(stat_feat.astype(np.float32, copy=False), copy=False))
        if feature_config.include_pe:
            parts.append(np.nan_to_num(pe_feat.astype(np.float32, copy=False), copy=False))
        if feature_config.include_lightweight:
            parts.append(np.nan_to_num(lightweight_feat.astype(np.float32, copy=False), copy=False))
        if feature_config.include_byte_summary:
            parts.append(_byte_summary_features(byte_seq, feature_config.prefix_len, feature_config.chunk_count))
        features.append(np.concatenate(parts).astype(np.float32, copy=False))
        labels.append(label)
        base_probs.append(prob)
        kept_rows.append(row)
    if not features:
        raise ValueError("No usable rows were loaded")
    return (
        np.vstack(features),
        np.asarray(labels, dtype=np.int64),
        np.asarray(base_probs, dtype=np.float32),
        kept_rows,
        {"total": len(rows), "kept": len(labels), "skipped_missing_cache": skipped_missing_cache},
    )


def _fit_standard_l2_reference(matrix: np.ndarray) -> dict:
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    centered = (matrix.astype(np.float32, copy=False) - mean) / std
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norms, 1.0e-8)
    return {"mean": mean, "std": std, "normalized": normalized.astype(np.float32, copy=False)}


def _normalize_with_reference(matrix: np.ndarray, reference: dict) -> np.ndarray:
    centered = (matrix.astype(np.float32, copy=False) - reference["mean"]) / reference["std"]
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return (centered / np.maximum(norms, 1.0e-8)).astype(np.float32, copy=False)


def _knn_feature_names(top_ks: Sequence[int]) -> list[str]:
    names = []
    for top_k in top_ks:
        names.extend(
            [
                f"knn{top_k}_mal_ratio",
                f"knn{top_k}_benign_ratio",
                f"knn{top_k}_label_margin",
                f"knn{top_k}_weighted_mal_ratio",
                f"knn{top_k}_mean_similarity",
                f"knn{top_k}_min_similarity",
            ]
        )
    names.extend(["knn_top1_label", "knn_top1_similarity", "knn_top1_top2_gap"])
    return names


def _knn_support_features_from_norm(
    query_norm: np.ndarray,
    memory_norm: np.ndarray,
    memory_labels: np.ndarray,
    top_ks: Sequence[int],
    *,
    batch_size: int,
) -> np.ndarray:
    if memory_norm.shape[0] == 0:
        raise ValueError("kNN memory is empty")
    top_ks = [min(int(top_k), int(memory_norm.shape[0])) for top_k in top_ks]
    max_k = max(top_ks)
    feature_dim = len(_knn_feature_names(top_ks))
    features = np.empty((query_norm.shape[0], feature_dim), dtype=np.float32)
    memory_labels = memory_labels.astype(np.float32, copy=False)
    batch_size = max(1, int(batch_size))

    for start in range(0, query_norm.shape[0], batch_size):
        stop = min(start + batch_size, query_norm.shape[0])
        similarities = query_norm[start:stop] @ memory_norm.T
        top_unsorted = np.argpartition(-similarities, max_k - 1, axis=1)[:, :max_k]
        top_sim_unsorted = np.take_along_axis(similarities, top_unsorted, axis=1)
        top_order = np.argsort(-top_sim_unsorted, axis=1)
        top_idx = np.take_along_axis(top_unsorted, top_order, axis=1)
        top_sim = np.take_along_axis(similarities, top_idx, axis=1).astype(np.float32, copy=False)
        top_labels = memory_labels[top_idx]

        batch_features = np.empty((stop - start, feature_dim), dtype=np.float32)
        column = 0
        for top_k in top_ks:
            labels_k = top_labels[:, :top_k]
            sim_k = top_sim[:, :top_k]
            mal_ratio = labels_k.mean(axis=1)
            weights = np.clip((sim_k + 1.0) * 0.5, 1.0e-6, None)
            weighted_mal_ratio = (labels_k * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1.0e-6)
            batch_features[:, column] = mal_ratio
            batch_features[:, column + 1] = 1.0 - mal_ratio
            batch_features[:, column + 2] = 2.0 * mal_ratio - 1.0
            batch_features[:, column + 3] = weighted_mal_ratio
            batch_features[:, column + 4] = sim_k.mean(axis=1)
            batch_features[:, column + 5] = sim_k[:, -1]
            column += 6

        top2_index = 1 if top_sim.shape[1] > 1 else 0
        batch_features[:, column] = top_labels[:, 0]
        batch_features[:, column + 1] = top_sim[:, 0]
        batch_features[:, column + 2] = top_sim[:, 0] - top_sim[:, top2_index]
        features[start:stop] = batch_features

    return features


def build_oof_knn_features(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    top_ks: Sequence[int],
    folds: int,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    if folds < 2:
        raise ValueError("OOF kNN requires at least 2 folds")
    folds = min(int(folds), int(np.bincount(labels).min()))
    if folds < 2:
        raise ValueError("Not enough samples per class for OOF kNN")

    reference = _fit_standard_l2_reference(matrix)
    normalized = reference["normalized"]
    features = np.empty((matrix.shape[0], len(_knn_feature_names(top_ks))), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_sizes = []
    for fold_index, (memory_idx, query_idx) in enumerate(splitter.split(matrix, labels)):
        fold_sizes.append(int(query_idx.shape[0]))
        features[query_idx] = _knn_support_features_from_norm(
            normalized[query_idx],
            normalized[memory_idx],
            labels[memory_idx],
            top_ks,
            batch_size=batch_size,
        )
        print(
            f"[knn-oof] fold={fold_index + 1}/{folds} query={query_idx.shape[0]} memory={memory_idx.shape[0]}",
            flush=True,
        )
    return features, {"folds": folds, "fold_sizes": fold_sizes}


def build_frozen_knn_reference(matrix: np.ndarray, labels: np.ndarray) -> dict:
    reference = _fit_standard_l2_reference(matrix)
    return {
        "mean": reference["mean"],
        "std": reference["std"],
        "memory_norm": reference["normalized"],
        "memory_labels": labels.astype(np.int64, copy=False),
    }


def append_frozen_knn_features(
    matrix: np.ndarray,
    frozen_reference: dict,
    top_ks: Sequence[int],
    *,
    batch_size: int,
) -> np.ndarray:
    query_norm = _normalize_with_reference(
        matrix,
        {
            "mean": frozen_reference["mean"],
            "std": frozen_reference["std"],
        },
    )
    knn_features = _knn_support_features_from_norm(
        query_norm,
        frozen_reference["memory_norm"],
        frozen_reference["memory_labels"],
        top_ks,
        batch_size=batch_size,
    )
    return np.hstack([matrix, knn_features]).astype(np.float32, copy=False)


def metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "errors": int(fp + fn),
    }


def select_best_threshold(scores: np.ndarray, labels: np.ndarray, thresholds: Sequence[float]) -> dict:
    rows = [metrics_at_threshold(scores, labels, threshold) for threshold in thresholds]
    rows.sort(key=lambda row: (row["f1"], -row["errors"], row["threshold"]), reverse=True)
    return rows[0]


def suspected_noise_mask(labels: np.ndarray, base_probs: np.ndarray, *, low: float = 0.05, high: float = 0.95) -> np.ndarray:
    return ((labels == 1) & (base_probs <= low)) | ((labels == 0) & (base_probs >= high))


def sample_weights(labels: np.ndarray, base_probs: np.ndarray, mode: str) -> np.ndarray:
    weights = np.ones(labels.shape[0], dtype=np.float32)
    if mode == "none":
        return weights
    if mode == "soft_conflict_downweight":
        severe = suspected_noise_mask(labels, base_probs, low=0.05, high=0.95)
        medium = ((labels == 1) & (base_probs <= 0.15)) | ((labels == 0) & (base_probs >= 0.85))
        weights[medium] = 0.5
        weights[severe] = 0.15
        return weights
    if mode == "trim_extreme_conflict":
        severe = suspected_noise_mask(labels, base_probs, low=0.03, high=0.97)
        weights[severe] = 0.0
        return weights
    raise ValueError(f"Unknown noise mode: {mode}")


def model_candidates(seed: int) -> list[tuple[str, object]]:
    return [
        (
            "hgb_lr0.04_leaf15_l2_0",
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=0.0,
                max_iter=320,
                random_state=seed,
            ),
        ),
        (
            "hgb_lr0.06_leaf31_l2_0",
            HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.0,
                max_iter=260,
                random_state=seed,
            ),
        ),
        (
            "hgb_lr0.08_leaf31_l2_1e-3",
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=1.0e-3,
                max_iter=220,
                random_state=seed,
            ),
        ),
        (
            "hgb_lr0.10_leaf63_l2_1e-3",
            HistGradientBoostingClassifier(
                learning_rate=0.10,
                max_leaf_nodes=63,
                l2_regularization=1.0e-3,
                max_iter=180,
                random_state=seed,
            ),
        ),
        (
            "extra_trees_300_leaf1",
            ExtraTreesClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=seed,
                class_weight=None,
            ),
        ),
        (
            "extra_trees_500_leaf2",
            ExtraTreesClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
                class_weight=None,
            ),
        ),
        (
            "rf_300_leaf2",
            RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=seed,
                class_weight=None,
            ),
        ),
        (
            "logreg_l2_c1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, solver="liblinear", C=1.0),
            ),
        ),
    ]


def predict_scores(model, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(matrix)[:, 1].astype(np.float32, copy=False)
    scores = model.decision_function(matrix)
    scores = np.clip(scores, -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-scores))).astype(np.float32, copy=False)


def write_predictions(path: Path, rows: Sequence[dict], labels: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "cache_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "stage2_prob_malicious",
        "prediction",
        "correct",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row, label, score in zip(rows, labels, scores):
            prediction = int(score >= threshold)
            writer.writerow(
                {
                    "source_path": row.get("source_path", ""),
                    "cache_path": row.get("cache_path", ""),
                    "source_sha256": row.get("source_sha256", ""),
                    "label": int(label),
                    "split": row.get("split", ""),
                    "sample_index": row.get("sample_index", ""),
                    "stage2_prob_malicious": f"{float(score):.10f}",
                    "prediction": prediction,
                    "correct": prediction == int(label),
                }
            )


def summarize_noise(labels: np.ndarray, base_probs: np.ndarray) -> dict:
    severe = suspected_noise_mask(labels, base_probs, low=0.05, high=0.95)
    medium = ((labels == 1) & (base_probs <= 0.15)) | ((labels == 0) & (base_probs >= 0.85))
    return {
        "medium_conflict_count": int(medium.sum()),
        "severe_conflict_count": int(severe.sum()),
        "medium_conflict_ratio": float(medium.mean()),
        "severe_conflict_ratio": float(severe.mean()),
        "label0_severe": int((severe & (labels == 0)).sum()),
        "label1_severe": int((severe & (labels == 1)).sum()),
    }


def clean_slice_metrics(scores: np.ndarray, labels: np.ndarray, base_probs: np.ndarray, threshold: float) -> dict:
    severe = suspected_noise_mask(labels, base_probs, low=0.05, high=0.95)
    clean = ~severe
    if clean.sum() == 0:
        return {"samples": 0}
    result = metrics_at_threshold(scores[clean], labels[clean], threshold)
    result["samples"] = int(clean.sum())
    result["excluded_suspected_noise"] = int(severe.sum())
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run cache-backed stage-2 validation matrix.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.05:0.95:0.005")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--chunk-count", type=int, default=16)
    parser.add_argument("--feature-set", choices=["tabular", "extended"], default="extended")
    parser.add_argument("--noise-modes", default="none,soft_conflict_downweight,trim_extreme_conflict")
    parser.add_argument("--test-val-f1-gate", type=float, default=0.980)
    parser.add_argument("--knn-features", action="store_true", help="Append train-only kNN label-support features.")
    parser.add_argument("--knn-top-k", default="5,10,25,50")
    parser.add_argument("--knn-folds", type=int, default=5)
    parser.add_argument("--knn-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    checkpoint = load_safe_checkpoint(resolve_path(args.checkpoint), map_location="cpu")
    checkpoint_config = AxonExperimentConfig.from_dict(dict(checkpoint["config"]))
    feature_config = FeatureConfig(
        prefix_len=max(0, int(args.prefix_len)),
        chunk_count=max(1, int(args.chunk_count)),
        include_pe=True,
        include_stat=True,
        include_lightweight=args.feature_set == "extended",
        include_byte_summary=args.feature_set == "extended",
    )
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_prediction_rows(args.train_predictions, args.max_train_rows)
    val_rows = read_prediction_rows(args.val_predictions, args.max_val_rows)
    print(f"[load] train rows={len(train_rows)} val rows={len(val_rows)}", flush=True)
    train_x, train_y, train_base, train_kept_rows, train_counts = build_matrix(train_rows, checkpoint_config, feature_config)
    val_x, val_y, val_base, val_kept_rows, val_counts = build_matrix(val_rows, checkpoint_config, feature_config)
    print(f"[matrix] train={train_x.shape} val={val_x.shape}", flush=True)

    base_feature_dim = int(train_x.shape[1])
    knn_config = {
        "enabled": bool(args.knn_features),
        "top_ks": parse_int_list(args.knn_top_k),
        "folds": int(args.knn_folds),
        "batch_size": int(args.knn_batch_size),
        "feature_names": [],
        "oof": None,
    }
    frozen_knn_reference = None
    if args.knn_features:
        top_ks = knn_config["top_ks"]
        knn_config["feature_names"] = _knn_feature_names(top_ks)
        print(
            f"[knn] building OOF train features top_k={top_ks} folds={args.knn_folds} batch={args.knn_batch_size}",
            flush=True,
        )
        train_knn, oof_info = build_oof_knn_features(
            train_x,
            train_y,
            top_ks=top_ks,
            folds=int(args.knn_folds),
            seed=int(args.seed),
            batch_size=int(args.knn_batch_size),
        )
        frozen_knn_reference = build_frozen_knn_reference(train_x, train_y)
        val_x = append_frozen_knn_features(
            val_x,
            frozen_knn_reference,
            top_ks,
            batch_size=int(args.knn_batch_size),
        )
        train_x = np.hstack([train_x, train_knn]).astype(np.float32, copy=False)
        knn_config["oof"] = oof_info
        print(f"[knn] augmented train={train_x.shape} val={val_x.shape}", flush=True)

    thresholds = parse_thresholds(args.thresholds)
    baseline_val_best = select_best_threshold(val_base, val_y, thresholds)
    results = []
    fitted = []
    noise_modes = [item.strip() for item in args.noise_modes.split(",") if item.strip()]
    for noise_mode in noise_modes:
        weights = sample_weights(train_y, train_base, noise_mode)
        effective_train_rows = int(np.count_nonzero(weights > 0.0))
        for model_name, model in model_candidates(int(args.seed)):
            start = time.perf_counter()
            fit_kwargs = {}
            if not isinstance(model, type(make_pipeline(StandardScaler(), LogisticRegression()))):
                fit_kwargs["sample_weight"] = weights
            try:
                model.fit(train_x, train_y, **fit_kwargs)
            except TypeError:
                model.fit(train_x, train_y)
            fit_sec = time.perf_counter() - start
            val_scores = predict_scores(model, val_x)
            val_best = select_best_threshold(val_scores, val_y, thresholds)
            clean_val = clean_slice_metrics(val_scores, val_y, val_base, float(val_best["threshold"]))
            result = {
                "name": f"{model_name}__noise_{noise_mode}",
                "base_model": model_name,
                "noise_mode": noise_mode,
                "fit_sec": fit_sec,
                "effective_train_rows": effective_train_rows,
                "val_best": val_best,
                "clean_val_at_val_threshold": clean_val,
                "delta_val_f1_vs_baseline": val_best["f1"] - baseline_val_best["f1"],
            }
            results.append(result)
            fitted.append((val_best["f1"], -val_best["errors"], result, model, val_scores))
            print(
                f"[val] {result['name']} f1={val_best['f1']:.6f} errors={val_best['errors']} "
                f"threshold={val_best['threshold']:.4f} fit_sec={fit_sec:.1f}",
                flush=True,
            )

    fitted.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected_f1, _neg_errors, selected, selected_model, selected_val_scores = fitted[0]
    report = {
        "schema": "axon_stage2_cache_matrix_v1",
        "protocol": "train predictions/cache fit candidates; val selects model/noise mode/threshold; test10k only if val gate passes",
        "checkpoint": str(resolve_path(args.checkpoint)),
        "train_predictions": str(resolve_path(args.train_predictions)),
        "val_predictions": str(resolve_path(args.val_predictions)),
        "test_predictions": str(resolve_path(args.test_predictions)) if args.test_predictions else None,
        "feature_config": feature_config.__dict__,
        "records": {"train": train_counts, "val": val_counts},
        "base_feature_dim": base_feature_dim,
        "feature_dim": int(train_x.shape[1]),
        "knn_config": knn_config,
        "noise_summary": {
            "train": summarize_noise(train_y, train_base),
            "val": summarize_noise(val_y, val_base),
        },
        "baseline_val_best": baseline_val_best,
        "models": sorted(results, key=lambda row: (row["val_best"]["f1"], -row["val_best"]["errors"]), reverse=True),
        "selected_by_val": selected,
    }

    selected_threshold = float(selected["val_best"]["threshold"])
    write_predictions(output_dir / "stage2_val_predictions.csv", val_kept_rows, val_y, selected_val_scores, selected_threshold)

    test_ran = False
    if args.test_predictions is not None and selected_f1 >= float(args.test_val_f1_gate):
        test_rows = read_prediction_rows(args.test_predictions, args.max_test_rows)
        test_x, test_y, test_base, test_kept_rows, test_counts = build_matrix(test_rows, checkpoint_config, feature_config)
        if args.knn_features:
            test_x = append_frozen_knn_features(
                test_x,
                frozen_knn_reference,
                knn_config["top_ks"],
                batch_size=int(args.knn_batch_size),
            )
        test_scores = predict_scores(selected_model, test_x)
        test_metrics = metrics_at_threshold(test_scores, test_y, selected_threshold)
        report["records"]["test"] = test_counts
        report["test_at_val_threshold"] = test_metrics
        report["clean_test_at_val_threshold"] = clean_slice_metrics(test_scores, test_y, test_base, selected_threshold)
        report["noise_summary"]["test"] = summarize_noise(test_y, test_base)
        write_predictions(output_dir / "stage2_test_predictions.csv", test_kept_rows, test_y, test_scores, selected_threshold)
        test_ran = True
    else:
        report["test_skipped"] = {
            "reason": "selected val F1 below gate or no test predictions provided",
            "selected_val_f1": float(selected_f1),
            "gate": float(args.test_val_f1_gate),
        }

    model_path = output_dir / "stage2_selected_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "model": selected_model,
                "feature_config": feature_config,
                "threshold": selected_threshold,
                "selected": selected,
                "checkpoint_config": checkpoint_config.to_dict(),
                "knn": {
                    "enabled": bool(args.knn_features),
                    "top_ks": knn_config["top_ks"],
                    "batch_size": int(args.knn_batch_size),
                    "feature_names": knn_config["feature_names"],
                    "reference": frozen_knn_reference,
                },
            },
            handle,
        )
    report["model_path"] = str(model_path)
    report["test_ran"] = test_ran
    report_path = output_dir / "stage2_cache_matrix_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected_by_val": selected, "test": report.get("test_at_val_threshold")}, indent=2, ensure_ascii=False))
    print(f"JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
