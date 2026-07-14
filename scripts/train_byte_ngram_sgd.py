#!/usr/bin/env python3
"""Train a streaming byte n-gram SGD classifier from feature-cache rows.

This is an experiment runner, not the production Axon model path. It keeps the
protocol strict: train split fits models, val split selects hyperparameters and
thresholds, and test split is evaluated only after selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import joblib
import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils import shuffle as sklearn_shuffle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig  # noqa: E402
from dataset import _load_cached_feature_npz  # noqa: E402


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_path(path_text: str) -> str:
    return str(Path(path_text)).replace("/", "\\").casefold()


def project_relative_path_key(path_text: str) -> Optional[str]:
    if not path_text:
        return None
    normalized = normalize_path(path_text)
    root = normalize_path(str(PROJECT_ROOT)).rstrip("\\")
    prefix = root + "\\"
    if normalized.startswith(prefix):
        return normalized[len(prefix):]
    return None


def source_path_keys(path_text: str) -> list[str]:
    if not path_text:
        return []
    keys = {normalize_path(path_text)}
    path = Path(path_text)
    if not path.is_absolute():
        keys.add(normalize_path(str(PROJECT_ROOT / path)))
    relative_key = project_relative_path_key(path_text)
    if relative_key:
        keys.add(relative_key)
    keys.add(path.name.casefold())
    return list(keys)


def source_sha_from_path(path_text: str) -> Optional[str]:
    stem = Path(path_text).stem.casefold()
    if len(stem) == 64 and all(char in "0123456789abcdef" for char in stem):
        return stem
    return None


def load_manifest_lookup(manifest_path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_source: dict[str, dict] = {}
    by_sha: dict[str, dict] = {}
    for row in payload.get("samples", []):
        source_path = row.get("source_path")
        if source_path:
            for key in source_path_keys(source_path):
                by_source.setdefault(key, row)
        source_sha256 = str(row.get("source_sha256") or "").casefold()
        if source_sha256:
            by_sha.setdefault(source_sha256, row)
    return by_source, by_sha


def lookup_manifest_sample(row: dict, by_source: dict[str, dict], by_sha: dict[str, dict]) -> tuple[Optional[dict], str]:
    candidate_paths = [
        row.get("source_path", ""),
        row.get("original_source_path", ""),
    ]
    for path_text in candidate_paths:
        for key in source_path_keys(path_text):
            sample = by_source.get(key)
            if sample is not None:
                return sample, "source_path"
    for path_text in candidate_paths:
        source_sha = source_sha_from_path(path_text)
        if source_sha:
            sample = by_sha.get(source_sha)
            if sample is not None:
                return sample, "source_sha256_from_path"
    row_sha = str(row.get("source_sha256") or "").casefold()
    if row_sha:
        sample = by_sha.get(row_sha)
        if sample is not None:
            return sample, "source_sha256"
    return None, "missing"


def load_checkpoint_config(checkpoint_path: Path) -> AxonExperimentConfig:
    try:
        checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise ValueError(f"Checkpoint missing config: {checkpoint_path}")
    return AxonExperimentConfig.from_dict(dict(checkpoint["config"]))


def sigmoid(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(scores, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-scores))


def read_split_rows(split_csv: Path, split: str, max_rows: Optional[int]) -> list[dict]:
    with resolve_path(split_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == split]
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def load_records(split_csv: Path, manifest_path: Path, split: str, max_rows: Optional[int]) -> list[dict]:
    rows = read_split_rows(split_csv, split, max_rows)
    by_source, by_sha = load_manifest_lookup(resolve_path(manifest_path))
    cache_dir = resolve_path(manifest_path).parent
    records = []
    missing = 0
    for row in rows:
        sample, reason = lookup_manifest_sample(row, by_source, by_sha)
        if sample is None:
            missing += 1
            continue
        label = int(row["label"])
        sample_label = int(sample["label"])
        if sample_label != label:
            raise ValueError(
                f"Manifest label mismatch for {row.get('source_path')}: split={label}, cache={sample_label}"
            )
        sample_sha = str(sample.get("source_sha256") or "").casefold()
        row_sha = str(row.get("source_sha256") or "").casefold()
        if row_sha and sample_sha and row_sha != sample_sha:
            raise ValueError(
                f"Manifest source SHA mismatch for {row.get('source_path')}: split={row_sha}, cache={sample_sha}"
            )
        cache_path = Path(sample["cache_path"])
        if not cache_path.is_absolute():
            cache_path = cache_dir / cache_path.name
        records.append(
            {
                "source_path": row["source_path"],
                "original_source_path": row.get("original_source_path", ""),
                "cache_path": str(cache_path),
                "source_sha256": sample_sha or row_sha,
                "label": label,
                "split": split,
                "sample_index": row.get("sample_index", ""),
                "match_reason": reason,
            }
        )
    if missing:
        raise ValueError(f"{split} has {missing} rows missing from manifest")
    return records


@dataclass(frozen=True)
class ByteHashConfig:
    n_features: int
    prefix_len: int
    ngram_min: int
    ngram_max: int
    ngram_stride: int
    include_byte_hist: bool
    include_cache_features: bool
    max_byte_length: int
    pe_feature_dim: int
    stat_feature_dim: int
    lightweight_feature_dim: int


def _hashed_position_byte_features(byte_seq: np.ndarray, prefix_len: int, n_features: int) -> np.ndarray:
    limit = min(prefix_len, byte_seq.shape[0])
    if limit <= 0:
        return np.empty(0, dtype=np.int64)
    positions = np.arange(limit, dtype=np.uint64)
    values = byte_seq[:limit].astype(np.uint64, copy=False)
    cols = (positions * np.uint64(1009) + values * np.uint64(9176) + np.uint64(17)) % np.uint64(n_features)
    return cols.astype(np.int64, copy=False)


def _hashed_ngram_features(
    byte_seq: np.ndarray,
    ngram_min: int,
    ngram_max: int,
    stride: int,
    n_features: int,
) -> np.ndarray:
    if ngram_max < ngram_min:
        return np.empty(0, dtype=np.int64)
    stride = max(1, int(stride))
    parts = []
    arr = byte_seq.astype(np.uint64, copy=False)
    for n in range(ngram_min, ngram_max + 1):
        if arr.shape[0] < n:
            continue
        windows = arr[: arr.shape[0] - n + 1 : stride]
        if windows.size == 0:
            continue
        hashes = windows.copy()
        for offset in range(1, n):
            hashes = hashes * np.uint64(257) + arr[offset : arr.shape[0] - n + 1 + offset : stride]
        hashes = (hashes * np.uint64(11400714819323198485) + np.uint64(n * 104729)) % np.uint64(n_features)
        parts.append(hashes.astype(np.int64, copy=False))
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def _hashed_byte_hist_features(byte_seq: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(byte_seq.astype(np.uint8, copy=False), minlength=256).astype(np.float32)
    nonzero = np.flatnonzero(counts)
    cols = (nonzero.astype(np.uint64) * np.uint64(65537) + np.uint64(424242)) % np.uint64(n_features)
    values = np.log1p(counts[nonzero])
    return cols.astype(np.int64, copy=False), values.astype(np.float32, copy=False)


def _hashed_dense_features(
    pe_feat: np.ndarray,
    stat_feat: np.ndarray,
    lightweight_feat: np.ndarray,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(
        [
            np.nan_to_num(pe_feat.astype(np.float32, copy=False), copy=False),
            np.nan_to_num(stat_feat.astype(np.float32, copy=False), copy=False),
            np.nan_to_num(lightweight_feat.astype(np.float32, copy=False), copy=False),
        ]
    )
    values = np.clip(values, -1.0e6, 1.0e6)
    values = np.sign(values) * np.log1p(np.abs(values))
    nonzero = np.flatnonzero(values)
    cols = (nonzero.astype(np.uint64) * np.uint64(1315423911) + np.uint64(7777777)) % np.uint64(n_features)
    return cols.astype(np.int64, copy=False), values[nonzero].astype(np.float32, copy=False)


def transform_batch(records: Sequence[dict], config: ByteHashConfig) -> tuple[sparse.csr_matrix, np.ndarray]:
    row_indices: list[np.ndarray] = []
    col_indices: list[np.ndarray] = []
    data_values: list[np.ndarray] = []
    labels = np.empty(len(records), dtype=np.int64)

    for row_idx, record in enumerate(records):
        byte_seq, pe_feat, stat_feat, lightweight_feat, label = _load_cached_feature_npz(
            Path(record["cache_path"]),
            config.max_byte_length,
            config.pe_feature_dim,
            config.stat_feature_dim,
            config.lightweight_feature_dim,
            expected_label=int(record["label"]),
            expected_source_sha256=record.get("source_sha256") or None,
            allow_missing_source_sha256=False,
        )
        labels[row_idx] = label

        cols_parts = [
            _hashed_position_byte_features(byte_seq, config.prefix_len, config.n_features),
            _hashed_ngram_features(
                byte_seq,
                config.ngram_min,
                config.ngram_max,
                config.ngram_stride,
                config.n_features,
            ),
        ]
        values_parts = [np.ones(part.shape[0], dtype=np.float32) for part in cols_parts]
        if config.include_byte_hist:
            hist_cols, hist_values = _hashed_byte_hist_features(byte_seq, config.n_features)
            cols_parts.append(hist_cols)
            values_parts.append(hist_values)
        if config.include_cache_features:
            dense_cols, dense_values = _hashed_dense_features(pe_feat, stat_feat, lightweight_feat, config.n_features)
            cols_parts.append(dense_cols)
            values_parts.append(dense_values)

        cols = np.concatenate(cols_parts) if cols_parts else np.empty(0, dtype=np.int64)
        values = np.concatenate(values_parts) if values_parts else np.empty(0, dtype=np.float32)
        if cols.size:
            cols, inverse = np.unique(cols, return_inverse=True)
            summed = np.zeros(cols.shape[0], dtype=np.float32)
            np.add.at(summed, inverse, values)
            norm = float(np.linalg.norm(summed))
            if norm > 0:
                summed /= norm
            row_indices.append(np.full(cols.shape[0], row_idx, dtype=np.int32))
            col_indices.append(cols.astype(np.int32, copy=False))
            data_values.append(summed)

    if row_indices:
        rows = np.concatenate(row_indices)
        cols = np.concatenate(col_indices)
        data = np.concatenate(data_values)
    else:
        rows = np.empty(0, dtype=np.int32)
        cols = np.empty(0, dtype=np.int32)
        data = np.empty(0, dtype=np.float32)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(records), config.n_features), dtype=np.float32)
    return matrix, labels


def batched(records: Sequence[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(records), batch_size):
        yield list(records[start : start + batch_size])


def train_candidate(
    train_records: Sequence[dict],
    config: ByteHashConfig,
    *,
    alpha: float,
    l1_ratio: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> SGDClassifier:
    model = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet" if l1_ratio > 0 else "l2",
        alpha=alpha,
        l1_ratio=l1_ratio,
        learning_rate="optimal",
        average=True,
        class_weight=None,
        random_state=seed,
    )
    classes = np.asarray([0, 1], dtype=np.int64)
    first_batch = True
    rng = np.random.default_rng(seed)
    records = list(train_records)
    for epoch in range(epochs):
        order = rng.permutation(len(records))
        shuffled = [records[int(i)] for i in order]
        processed = 0
        for batch in batched(shuffled, batch_size):
            x_batch, y_batch = transform_batch(batch, config)
            if first_batch:
                model.partial_fit(x_batch, y_batch, classes=classes)
                first_batch = False
            else:
                model.partial_fit(x_batch, y_batch)
            processed += len(batch)
        print(f"[train] alpha={alpha:g} epoch={epoch + 1}/{epochs} rows={processed}", flush=True)
    return model


def predict_scores(model: SGDClassifier, records: Sequence[dict], config: ByteHashConfig, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    scores = []
    for batch in batched(records, batch_size):
        x_batch, y_batch = transform_batch(batch, config)
        batch_scores = sigmoid(model.decision_function(x_batch))
        labels.append(y_batch)
        scores.append(batch_scores.astype(np.float32, copy=False))
    return np.concatenate(labels), np.concatenate(scores)


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "samples": int(labels.shape[0]),
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


def select_threshold(labels: np.ndarray, scores: np.ndarray, thresholds: Sequence[float]) -> dict:
    candidates = [metrics_at_threshold(labels, scores, threshold) for threshold in thresholds]
    candidates.sort(key=lambda row: (row["f1"], -row["errors"], row["threshold"]), reverse=True)
    return candidates[0]


def write_predictions(path: Path, records: Sequence[dict], labels: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "original_source_path",
        "cache_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "prob_malicious",
        "prediction",
        "correct",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record, label, score in zip(records, labels, scores):
            prediction = int(score >= threshold)
            writer.writerow(
                {
                    "source_path": record["source_path"],
                    "original_source_path": record.get("original_source_path", ""),
                    "cache_path": record["cache_path"],
                    "source_sha256": record.get("source_sha256", ""),
                    "label": int(label),
                    "split": record["split"],
                    "sample_index": record.get("sample_index", ""),
                    "prob_malicious": f"{float(score):.10f}",
                    "prediction": prediction,
                    "correct": prediction == int(label),
                }
            )


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_thresholds(text: str) -> list[float]:
    if ":" in text:
        start_text, stop_text, step_text = text.split(":")
        start = float(start_text)
        stop = float(stop_text)
        step = float(step_text)
        count = int(math.floor((stop - start) / step)) + 1
        return [round(start + i * step, 10) for i in range(count)]
    return parse_float_list(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train a byte n-gram SGD classifier from cache rows.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-features", type=int, default=2 ** 20)
    parser.add_argument("--prefix-len", type=int, default=2048)
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--ngram-stride", type=int, default=4)
    parser.add_argument("--alphas", default="1e-5,3e-5,1e-4")
    parser.add_argument("--l1-ratio", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--thresholds", default="0.05:0.95:0.01")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=10000)
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--include-byte-hist", action="store_true")
    parser.add_argument("--include-cache-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    checkpoint_config = load_checkpoint_config(args.checkpoint)
    hash_config = ByteHashConfig(
        n_features=int(args.n_features),
        prefix_len=int(args.prefix_len),
        ngram_min=int(args.ngram_min),
        ngram_max=int(args.ngram_max),
        ngram_stride=int(args.ngram_stride),
        include_byte_hist=bool(args.include_byte_hist),
        include_cache_features=bool(args.include_cache_features),
        max_byte_length=checkpoint_config.max_byte_length,
        pe_feature_dim=checkpoint_config.pe_feature_dim,
        stat_feature_dim=checkpoint_config.stat_feature_dim,
        lightweight_feature_dim=checkpoint_config.lightweight_feature_dim,
    )

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records = load_records(args.split_csv, args.manifest, "train", args.max_train_rows)
    val_records = load_records(args.split_csv, args.manifest, "val", args.max_val_rows)
    test_records = []
    if not args.skip_test_eval:
        test_records = load_records(args.split_csv, args.manifest, "test", args.max_test_rows)
    train_records = list(sklearn_shuffle(train_records, random_state=args.seed))

    thresholds = parse_thresholds(args.thresholds)
    alpha_values = parse_float_list(args.alphas)
    candidates = []
    best = None
    best_model = None
    for alpha in alpha_values:
        model = train_candidate(
            train_records,
            hash_config,
            alpha=alpha,
            l1_ratio=float(args.l1_ratio),
            epochs=max(1, int(args.epochs)),
            batch_size=max(1, int(args.batch_size)),
            seed=int(args.seed),
        )
        val_labels, val_scores = predict_scores(model, val_records, hash_config, max(1, int(args.batch_size)))
        val_best = select_threshold(val_labels, val_scores, thresholds)
        candidate = {"alpha": float(alpha), "val": val_best}
        candidates.append(candidate)
        print(f"[val] alpha={alpha:g} f1={val_best['f1']:.6f} errors={val_best['errors']} threshold={val_best['threshold']:.4f}", flush=True)
        if best is None or (val_best["f1"], -val_best["errors"]) > (best["val"]["f1"], -best["val"]["errors"]):
            best = candidate
            best_model = model

    if best is None or best_model is None:
        raise ValueError("No candidate model was trained")

    threshold = float(best["val"]["threshold"])
    val_labels, val_scores = predict_scores(best_model, val_records, hash_config, max(1, int(args.batch_size)))
    val_metrics = metrics_at_threshold(val_labels, val_scores, threshold)
    test_metrics = None
    test_predictions = None
    if not args.skip_test_eval:
        test_labels, test_scores = predict_scores(best_model, test_records, hash_config, max(1, int(args.batch_size)))
        test_metrics = metrics_at_threshold(test_labels, test_scores, threshold)

    model_path = output_dir / "byte_ngram_sgd_selected_model.joblib"
    joblib.dump(
        {
            "model": best_model,
            "hash_config": hash_config,
            "threshold": threshold,
            "selected": best,
            "checkpoint_config": checkpoint_config.to_dict(),
        },
        model_path,
    )
    val_predictions = output_dir / "byte_ngram_sgd_val_predictions.csv"
    write_predictions(val_predictions, val_records, val_labels, val_scores, threshold)
    if not args.skip_test_eval:
        test_predictions = output_dir / "byte_ngram_sgd_test_predictions.csv"
        write_predictions(test_predictions, test_records, test_labels, test_scores, threshold)

    report = {
        "schema": "axon_byte_ngram_sgd_report_v1",
        "protocol": "train split fits SGD candidates; val split selects alpha and threshold; test split is fixed confirmation only",
        "split_csv": str(resolve_path(args.split_csv)),
        "manifest": str(resolve_path(args.manifest)),
        "checkpoint": str(resolve_path(args.checkpoint)),
        "records": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "hash_config": hash_config.__dict__,
        "candidate_alphas": alpha_values,
        "candidates": candidates,
        "selected": best,
        "selected_threshold": threshold,
        "val_metrics_at_selected_threshold": val_metrics,
        "test_metrics_at_selected_threshold": test_metrics,
        "model_path": str(model_path),
        "val_predictions": str(val_predictions),
        "test_predictions": str(test_predictions) if test_predictions is not None else None,
    }
    report_path = output_dir / "byte_ngram_sgd_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
