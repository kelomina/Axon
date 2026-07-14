#!/usr/bin/env python3
"""Val-only content-feature separability probe for calibrator rescue rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from config import AxonExperimentConfig  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402
from train_stage2_cache_matrix import (  # noqa: E402
    FeatureConfig,
    assert_stage2_feature_names_safe,
    build_matrix,
    read_prediction_rows,
    resolve_path,
)


RESCUE_GROUP = "calibrator_only_correct"
REGRESSION_GROUP = "loop57_only_correct"
DROPPED_PROBABILITY_FEATURE_COUNT = 6

FORBIDDEN_IDENTITY_EVIDENCE = [
    "filename",
    "path",
    "extension",
    "directory",
    "hash",
    "source_sha256",
    "sample_index",
    "split",
    "row_order",
]


def read_overlap_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_checkpoint_config(checkpoint_path: Path) -> AxonExperimentConfig:
    checkpoint = load_safe_checkpoint(resolve_path(checkpoint_path), map_location="cpu")
    return AxonExperimentConfig.from_dict(checkpoint["config"])


def index_prediction_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = read_prediction_rows(path)
    indexed = {}
    duplicates = []
    for row in rows:
        key = str(row.get("source_sha256", "")).strip()
        if not key:
            raise ValueError(f"{path} contains a row without source_sha256")
        if key in indexed:
            duplicates.append(key)
        indexed[key] = row
    if duplicates:
        raise ValueError(f"{path} contains duplicate source_sha256 values: {duplicates[:5]}")
    return indexed


def selector_candidates(seed: int) -> list[tuple[str, Any]]:
    return [
        (
            "logreg_balanced_c0.10",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=5000,
                    solver="liblinear",
                    C=0.10,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ),
        (
            "logreg_balanced_c1",
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
            "hgb_leaf3",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_leaf_nodes=3,
                l2_regularization=1.0e-2,
                max_iter=120,
                random_state=seed,
            ),
        ),
        (
            "extra_trees_100_leaf3",
            ExtraTreesClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
        ),
    ]


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(labels.shape[0], 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else None,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "errors": int(fp + fn),
    }


def selector_report(
    *,
    matrix: np.ndarray,
    labels: np.ndarray,
    folds: int,
    seed: int,
) -> list[dict[str, Any]]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    reports = []
    for name, model in selector_candidates(seed):
        scores = cross_val_predict(model, matrix, labels, cv=splitter, method="predict_proba")[:, 1]
        report = classification_metrics(labels, scores)
        report["model"] = name
        reports.append(report)
    reports.sort(key=lambda item: (item["f1"], item["auc"] or 0.0, -item["errors"]), reverse=True)
    return reports


def build_focus_rows(
    *,
    overlap_rows: Sequence[dict[str, str]],
    base_predictions_by_sha: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], np.ndarray, dict[str, Any]]:
    focus_rows = []
    selector_labels = []
    skipped = Counter()
    for row in overlap_rows:
        group = row.get("overlap_group", "")
        if group not in {RESCUE_GROUP, REGRESSION_GROUP}:
            continue
        sha = str(row.get("join_key", "")).strip()
        base_row = base_predictions_by_sha.get(sha)
        if base_row is None:
            skipped["missing_base_prediction"] += 1
            continue
        item = dict(base_row)
        item["_loop82_overlap_group"] = group
        item["_loop82_join_key"] = sha
        focus_rows.append(item)
        selector_labels.append(1 if group == RESCUE_GROUP else 0)
    summary = {
        "focus_rows": len(focus_rows),
        "selector_labels": dict(Counter(str(value) for value in selector_labels)),
        "skipped": dict(skipped),
    }
    return focus_rows, np.asarray(selector_labels, dtype=np.int64), summary


def build_summary(
    *,
    overlap_csv: Path,
    base_predictions: Path,
    checkpoint: Path,
    content_cache_dir: Path,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    overlap_rows = read_overlap_rows(overlap_csv)
    base_by_sha = index_prediction_rows(base_predictions)
    focus_rows, selector_labels, focus_summary = build_focus_rows(
        overlap_rows=overlap_rows,
        base_predictions_by_sha=base_by_sha,
    )
    checkpoint_config = load_checkpoint_config(checkpoint)
    feature_config = FeatureConfig(
        prefix_len=256,
        chunk_count=16,
        include_pe=True,
        include_stat=True,
        include_lightweight=True,
        include_byte_summary=True,
        include_content_pe=True,
        content_cache_dir=str(resolve_path(content_cache_dir)),
    )
    feature_name_groups = assert_stage2_feature_names_safe(feature_config, checkpoint_config=checkpoint_config)
    matrix, labels, _base_probs, kept_rows, matrix_counts = build_matrix(focus_rows, checkpoint_config, feature_config)
    if DROPPED_PROBABILITY_FEATURE_COUNT:
        matrix = matrix[:, DROPPED_PROBABILITY_FEATURE_COUNT:].astype(np.float32, copy=False)
    kept_sha = [str(row.get("_loop82_join_key", "")) for row in kept_rows]
    kept_selector_labels = []
    selector_by_sha = {
        str(row.get("_loop82_join_key", "")): int(label)
        for row, label in zip(focus_rows, selector_labels)
    }
    for sha in kept_sha:
        kept_selector_labels.append(selector_by_sha[sha])
    kept_selector_labels_arr = np.asarray(kept_selector_labels, dtype=np.int64)

    blockers = []
    if len(overlap_rows) != 20000:
        blockers.append("Expected Loop82 complete 20000-row overlap")
    if focus_summary["focus_rows"] != 519:
        blockers.append("Expected 56 rescue + 463 regression focus rows")
    if matrix_counts["skipped_missing_cache"]:
        blockers.append("Focus rows have missing cache entries")
    if len(np.unique(kept_selector_labels_arr)) < 2:
        blockers.append("Selector focus labels contain fewer than two classes")

    selector_reports = [] if blockers else selector_report(
        matrix=matrix,
        labels=kept_selector_labels_arr,
        folds=min(folds, int(np.bincount(kept_selector_labels_arr).min())),
        seed=seed,
    )
    best = selector_reports[0] if selector_reports else None
    return {
        "schema": "axon_loop84_content_rescue_separability_v1",
        "protocol": "Val-only separability diagnostic; no Test/Test-10k access, no identity evidence",
        "overlap_csv": str(overlap_csv),
        "base_predictions": str(base_predictions),
        "checkpoint": str(checkpoint),
        "content_cache_dir": str(content_cache_dir),
        "rows": {
            "overlap": len(overlap_rows),
            "focus": focus_summary,
            "matrix": matrix_counts,
        },
        "blockers": blockers,
        "feature_config": feature_config.__dict__,
        "feature_name_groups": feature_name_groups,
        "dropped_probability_feature_count": DROPPED_PROBABILITY_FEATURE_COUNT,
        "content_matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "selector_cv": {
            "folds_requested": folds,
            "seed": seed,
            "reports": selector_reports,
            "best": best,
        },
        "interpretation": {
            "rescue_label": 1,
            "regression_label": 0,
            "is_promising": bool(best and best["auc"] is not None and best["auc"] >= 0.70 and best["recall"] >= 0.50),
            "gate_for_next_step": (
                "Only build a Val-only fusion selector if CV AUC/recall show real separation "
                "between calibrator-only-correct and Loop57-only-correct rows."
            ),
        },
        "identity_feature_policy": {
            "forbidden_as_model_evidence": FORBIDDEN_IDENTITY_EVIDENCE,
            "allowed_identity_use": "source_sha256 only maps Loop82 overlap rows back to cache-backed Val rows",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Loop84 content-feature rescue separability probe.")
    parser.add_argument("--overlap-csv", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--content-cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=8401)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_summary(
        overlap_csv=args.overlap_csv,
        base_predictions=args.base_predictions,
        checkpoint=args.checkpoint,
        content_cache_dir=args.content_cache_dir,
        folds=args.folds,
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
