#!/usr/bin/env python3
"""Fit an exportable native HGB student from fixed feature-cache rows."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import AxonExperimentConfig  # noqa: E402
from dataset import _load_cached_feature_npz  # noqa: E402
from kvd_features.extractor import ExtractionConfig, extract_pe_features  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402

from export_stage2_hgb_json import FeatureConfig  # noqa: E402
from train_stage2_cache_matrix import _byte_summary_features  # noqa: E402


SELECTED_FEATURE_INDICES = list(range(6, 6 + 49 + 256 + 256)) + list(range(1315, 1420))


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _runtime_prefix_pe_features(
    source_path: Path,
    config: AxonExperimentConfig,
    prefix_bytes: int,
    scratch_dir: Path,
) -> np.ndarray:
    snapshot_path = scratch_dir / f"{source_path.stem}.prefix{source_path.suffix}"
    with source_path.open("rb") as source:
        snapshot_path.write_bytes(source.read(prefix_bytes))
    extraction_config = ExtractionConfig.from_axon_config(
        config,
        max_file_size=prefix_bytes,
        pe_feature_dim=config.pe_feature_dim,
    )
    features = extract_pe_features(
        str(snapshot_path),
        config=extraction_config,
        axon_config=config,
        allow_fallback=False,
    )
    snapshot_path.unlink(missing_ok=True)
    if features is None:
        raise ValueError(f"Runtime-prefix PE extraction failed: {source_path}")
    return np.asarray(features, dtype=np.float32)


def feature_vector(
    row: dict[str, str],
    config: AxonExperimentConfig,
    *,
    runtime_prefix_bytes: Optional[int] = None,
    scratch_dir: Optional[Path] = None,
) -> tuple[np.ndarray, int]:
    byte_sequence, pe_features, stat_features, lightweight_features, label = _load_cached_feature_npz(
        Path(row["cache_path"]),
        config.max_byte_length,
        config.pe_feature_dim,
        config.stat_feature_dim,
        config.lightweight_feature_dim,
        expected_label=int(row["label"]),
        expected_source_sha256=row["source_sha256"],
    )
    if runtime_prefix_bytes is not None:
        if scratch_dir is None:
            raise ValueError("scratch_dir is required for runtime-prefix feature extraction")
        pe_features = _runtime_prefix_pe_features(
            Path(row["source_path"]), config, runtime_prefix_bytes, scratch_dir
        )
    byte_summary = _byte_summary_features(byte_sequence, 256, 16)
    full = np.concatenate(
        [
            np.zeros(6, dtype=np.float32),
            stat_features,
            pe_features,
            lightweight_features,
            byte_summary,
            np.zeros(100, dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)
    return full[SELECTED_FEATURE_INDICES], label


def matrix(
    rows: Sequence[dict[str, str]],
    config: AxonExperimentConfig,
    *,
    runtime_prefix_bytes: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    with tempfile.TemporaryDirectory(prefix="axon_native_student_") as scratch:
        scratch_dir = Path(scratch)
        for row in rows:
            values, label = feature_vector(
                row,
                config,
                runtime_prefix_bytes=runtime_prefix_bytes,
                scratch_dir=scratch_dir,
            )
            features.append(values)
            labels.append(label)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "threshold": threshold,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "errors": int(false_positive + false_negative),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-rows", type=Path, required=True)
    parser.add_argument("--val-rows", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--runtime-prefix-bytes", type=int)
    args = parser.parse_args(argv)

    checkpoint = load_safe_checkpoint(args.checkpoint, map_location="cpu")
    config = AxonExperimentConfig.from_dict(dict(checkpoint["config"]))
    if args.runtime_prefix_bytes is not None and args.runtime_prefix_bytes < config.max_byte_length:
        parser.error("--runtime-prefix-bytes must be at least max_byte_length")
    train_features, train_labels = matrix(
        load_rows(args.train_rows), config, runtime_prefix_bytes=args.runtime_prefix_bytes
    )
    val_features, val_labels = matrix(
        load_rows(args.val_rows), config, runtime_prefix_bytes=args.runtime_prefix_bytes
    )
    candidates = [
        HistGradientBoostingClassifier(learning_rate=0.04, max_leaf_nodes=15, max_iter=320, random_state=args.seed),
        HistGradientBoostingClassifier(learning_rate=0.06, max_leaf_nodes=31, max_iter=260, random_state=args.seed),
        HistGradientBoostingClassifier(learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0e-3, max_iter=220, random_state=args.seed),
    ]
    thresholds = np.arange(0.05, 0.951, 0.005)
    results = []
    selected = None
    for model in candidates:
        model.fit(train_features, train_labels)
        probabilities = model.predict_proba(val_features)[:, 1]
        candidate_metrics = [metrics(val_labels, probabilities, float(threshold)) for threshold in thresholds]
        best = max(candidate_metrics, key=lambda row: (row["f1"], -row["errors"], row["threshold"]))
        result = {"params": model.get_params(), "val": best}
        results.append(result)
        key = (best["f1"], -best["errors"], best["threshold"])
        if selected is None or key > selected[0]:
            selected = (key, model, best)
    assert selected is not None
    _, selected_model, selected_metrics = selected
    feature_config = FeatureConfig(
        prefix_len=256,
        chunk_count=16,
        include_pe=True,
        include_stat=True,
        include_lightweight=True,
        include_byte_summary=True,
        include_content_pe=True,
    )
    payload = {
        "model": selected_model,
        "feature_config": feature_config,
        "threshold": selected_metrics["threshold"],
        "selected": {"val": selected_metrics, "selected_feature_indices": SELECTED_FEATURE_INDICES},
        "checkpoint_config": config.to_dict(),
        "runtime_prefix_bytes": args.runtime_prefix_bytes,
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    with args.output_model.open("wb") as handle:
        pickle.dump(payload, handle)
    report = {
        "schema": "axon_native_student_fit_v1",
        "train_rows": int(train_features.shape[0]),
        "val_rows": int(val_features.shape[0]),
        "feature_dim": int(train_features.shape[1]),
        "runtime_prefix_bytes": args.runtime_prefix_bytes,
        "selected_feature_indices": SELECTED_FEATURE_INDICES,
        "candidates": results,
        "selected_val": selected_metrics,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
