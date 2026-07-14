#!/usr/bin/env python3
"""Materialize nested train OOF predictions for a Loop61-style override pipeline.

This script is a protocol-enabling step, not a model candidate. It rebuilds the
override-only pipeline inside outer train folds so every exported train-row
score comes from models that did not fit that row. Identity columns are written
only for alignment and audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from config import AxonExperimentConfig  # noqa: E402
from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402
from train_loop42_oof_residual_gate import (  # noqa: E402
    fit_with_optional_weights,
    oof_stage2_scores,
    prediction_metrics,
)
from train_loop55_overlay_boundary import (  # noqa: E402
    OVERLAY_BOUNDARY_FEATURE_NAMES,
    OverlayBoundaryConfig,
    build_overlay_boundary_matrix,
)
from train_loop57_fn_overlay_gate import build_fn_gate_matrix  # noqa: E402
from train_loop61_override_classifier import (  # noqa: E402
    _filter_override_candidates,
    override_classifier_predictions,
    override_model_candidates,
    override_target_summary,
    possible_override_mask,
    select_override_allow_threshold,
)
from train_stage2_cache_matrix import (  # noqa: E402
    FeatureConfig,
    assert_stage2_feature_names_safe,
    build_matrix,
    filter_model_candidates,
    metrics_at_threshold,
    model_candidates,
    parse_thresholds,
    predict_scores,
    read_prediction_rows,
    resolve_path,
    select_best_threshold,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize Loop69 nested OOF override predictions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--thresholds", default="0.35:0.65:0.005")
    parser.add_argument("--allow-thresholds", default="0.05:0.99:0.005")
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--chunk-count", type=int, default=16)
    parser.add_argument("--feature-set", choices=["tabular", "extended"], default="extended")
    parser.add_argument("--content-pe-cache-dir", type=Path, required=True)
    parser.add_argument("--overlay-boundary-cache-dir", type=Path, required=True)
    parser.add_argument("--drop-base-prob-features", action="store_true")
    parser.add_argument("--no-override-overlay-features", action="store_true")
    parser.add_argument("--override-content-features", action="store_true")
    parser.add_argument("--base-model-candidate", default="hgb_lr0.06_leaf31_l2_0")
    parser.add_argument("--candidate-model-candidate", default="extra_trees_300_leaf1")
    parser.add_argument("--override-model-candidate", default="override_logreg_balanced_c1")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--base-threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def _empty_float(size: int) -> np.ndarray:
    return np.full(size, np.nan, dtype=np.float32)


def _empty_int(size: int, fill: int = -1) -> np.ndarray:
    return np.full(size, fill, dtype=np.int64)


def write_nested_oof_predictions(
    path: Path,
    rows: Sequence[dict],
    labels: np.ndarray,
    *,
    oof_fold: np.ndarray,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    allow_scores: np.ndarray,
    final_scores: np.ndarray,
    final_predictions: np.ndarray,
    override_mask: np.ndarray,
    possible_mask: np.ndarray,
    candidate_thresholds: np.ndarray,
    allow_thresholds: np.ndarray,
    selected_candidate: str,
    selected_override_model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "cache_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "oof_fold",
        "base_oof_prob_malicious",
        "candidate_oof_prob_malicious",
        "allow_oof_prob",
        "final_oof_prob_malicious",
        "final_oof_prediction",
        "oof_override_flag",
        "possible_override_flag",
        "candidate_threshold",
        "allow_threshold",
        "selected_candidate",
        "selected_override_model",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, (row, label) in enumerate(zip(rows, labels)):
            writer.writerow(
                {
                    "source_path": row.get("source_path", ""),
                    "cache_path": row.get("cache_path", ""),
                    "source_sha256": row.get("source_sha256", ""),
                    "label": int(label),
                    "split": row.get("split", ""),
                    "sample_index": row.get("sample_index", ""),
                    "oof_fold": int(oof_fold[index]),
                    "base_oof_prob_malicious": f"{float(base_scores[index]):.10f}",
                    "candidate_oof_prob_malicious": f"{float(candidate_scores[index]):.10f}",
                    "allow_oof_prob": f"{float(allow_scores[index]):.10f}",
                    "final_oof_prob_malicious": f"{float(final_scores[index]):.10f}",
                    "final_oof_prediction": int(final_predictions[index]),
                    "oof_override_flag": bool(override_mask[index]),
                    "possible_override_flag": bool(possible_mask[index]),
                    "candidate_threshold": f"{float(candidate_thresholds[index]):.10f}",
                    "allow_threshold": f"{float(allow_thresholds[index]):.10f}",
                    "selected_candidate": selected_candidate,
                    "selected_override_model": selected_override_model,
                }
            )


def _fit_override_classifier(model, matrix: np.ndarray, labels: np.ndarray):
    if matrix.shape[0] == 0:
        raise ValueError("No possible override rows are available for fitting")
    if len(np.unique(labels)) < 2:
        raise ValueError("Possible override training rows contain only one class")
    return fit_with_optional_weights(model, matrix, labels)


def materialize_nested_oof(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_overlay: np.ndarray,
    base_spec: tuple[str, object],
    candidate_spec: tuple[str, object],
    override_spec: tuple[str, object],
    thresholds: Sequence[float],
    allow_thresholds: Sequence[float],
    base_threshold: float,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    include_override_overlay_features: bool,
    include_override_content_features: bool,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    class_counts = np.bincount(train_y)
    outer_folds = min(max(2, int(outer_folds)), int(class_counts.min()))
    inner_folds = min(max(2, int(inner_folds)), int(class_counts.min()))
    splitter = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    row_count = int(train_y.shape[0])

    oof_fold = _empty_int(row_count)
    base_scores = _empty_float(row_count)
    candidate_scores = _empty_float(row_count)
    allow_scores = _empty_float(row_count)
    final_scores = _empty_float(row_count)
    final_predictions = _empty_int(row_count)
    override_mask = np.zeros(row_count, dtype=bool)
    possible_mask = np.zeros(row_count, dtype=bool)
    candidate_thresholds = _empty_float(row_count)
    allow_threshold_values = _empty_float(row_count)
    fold_reports: list[dict] = []

    for fold_index, (fit_idx, holdout_idx) in enumerate(splitter.split(train_x, train_y), start=1):
        start = time.perf_counter()
        fit_labels = train_y[fit_idx]
        fold_inner = min(inner_folds, int(np.bincount(fit_labels).min()))
        if fold_inner < 2:
            raise ValueError(f"Outer fold {fold_index} does not have enough class support for inner OOF")

        fit_x = train_x[fit_idx]
        holdout_x = train_x[holdout_idx]
        fit_overlay = train_overlay[fit_idx]
        holdout_overlay = train_overlay[holdout_idx]
        fit_candidate_x = np.hstack([fit_x, fit_overlay]).astype(np.float32, copy=False)
        holdout_candidate_x = np.hstack([holdout_x, holdout_overlay]).astype(np.float32, copy=False)

        base_oof, base_holdout, _base_models, base_reports = oof_stage2_scores(
            train_x=fit_x,
            train_y=fit_labels,
            val_x=holdout_x,
            specs=[base_spec],
            folds=fold_inner,
            seed=seed + 1000 + fold_index,
        )
        candidate_oof, candidate_holdout, _candidate_models, candidate_reports = oof_stage2_scores(
            train_x=fit_candidate_x,
            train_y=fit_labels,
            val_x=holdout_candidate_x,
            specs=[candidate_spec],
            folds=fold_inner,
            seed=seed + 2000 + fold_index,
        )
        base_fit_scores = base_oof[:, 0]
        base_holdout_scores = base_holdout[:, 0]
        candidate_fit_scores = candidate_oof[:, 0]
        candidate_holdout_scores = candidate_holdout[:, 0]
        candidate_best = select_best_threshold(candidate_fit_scores, fit_labels, thresholds)
        candidate_threshold = float(candidate_best["threshold"])

        fit_possible = possible_override_mask(
            base_scores=base_fit_scores,
            candidate_scores=candidate_fit_scores,
            base_threshold=base_threshold,
            candidate_threshold=candidate_threshold,
        )
        fit_gate_x, gate_feature_names = build_fn_gate_matrix(
            fit_x,
            fit_overlay,
            base_fit_scores,
            candidate_fit_scores,
            include_overlay_features=include_override_overlay_features,
            include_content_features=include_override_content_features,
        )
        holdout_gate_x, _ = build_fn_gate_matrix(
            holdout_x,
            holdout_overlay,
            base_holdout_scores,
            candidate_holdout_scores,
            include_overlay_features=include_override_overlay_features,
            include_content_features=include_override_content_features,
        )
        assert_no_identity_feature_names(gate_feature_names, context="Loop69 override feature names")

        override_model = clone(override_spec[1])
        _fit_override_classifier(override_model, fit_gate_x[fit_possible], fit_labels[fit_possible])
        fit_allow_scores = predict_scores(override_model, fit_gate_x)
        holdout_allow_scores = predict_scores(override_model, holdout_gate_x)
        train_best = select_override_allow_threshold(
            labels=fit_labels,
            base_scores=base_fit_scores,
            candidate_scores=candidate_fit_scores,
            allow_scores=fit_allow_scores,
            base_threshold=base_threshold,
            candidate_threshold=candidate_threshold,
            allow_thresholds=allow_thresholds,
        )
        allow_threshold = float(train_best["allow_threshold"])
        holdout_pred, holdout_final_scores, holdout_override = override_classifier_predictions(
            base_scores=base_holdout_scores,
            candidate_scores=candidate_holdout_scores,
            allow_scores=holdout_allow_scores,
            base_threshold=base_threshold,
            candidate_threshold=candidate_threshold,
            allow_threshold=allow_threshold,
        )
        holdout_possible = possible_override_mask(
            base_scores=base_holdout_scores,
            candidate_scores=candidate_holdout_scores,
            base_threshold=base_threshold,
            candidate_threshold=candidate_threshold,
        )

        oof_fold[holdout_idx] = fold_index
        base_scores[holdout_idx] = base_holdout_scores
        candidate_scores[holdout_idx] = candidate_holdout_scores
        allow_scores[holdout_idx] = holdout_allow_scores
        final_scores[holdout_idx] = holdout_final_scores
        final_predictions[holdout_idx] = holdout_pred
        override_mask[holdout_idx] = holdout_override
        possible_mask[holdout_idx] = holdout_possible
        candidate_thresholds[holdout_idx] = candidate_threshold
        allow_threshold_values[holdout_idx] = allow_threshold

        fold_report = {
            "outer_fold": fold_index,
            "fit_rows": int(fit_idx.shape[0]),
            "holdout_rows": int(holdout_idx.shape[0]),
            "inner_folds": int(fold_inner),
            "candidate_threshold": candidate_threshold,
            "allow_threshold": allow_threshold,
            "fit_possible_summary": override_target_summary(fit_labels, fit_possible),
            "holdout_possible_summary": override_target_summary(train_y[holdout_idx], holdout_possible),
            "candidate_train_best": candidate_best,
            "override_train_best": train_best,
            "holdout_metrics": prediction_metrics(train_y[holdout_idx], holdout_pred, holdout_final_scores),
            "holdout_base_metrics": metrics_at_threshold(base_holdout_scores, train_y[holdout_idx], base_threshold),
            "holdout_candidate_metrics": metrics_at_threshold(
                candidate_holdout_scores,
                train_y[holdout_idx],
                candidate_threshold,
            ),
            "base_reports": base_reports,
            "candidate_reports": candidate_reports,
            "fit_sec": time.perf_counter() - start,
        }
        fold_reports.append(fold_report)
        print(
            f"[loop69-oof] fold={fold_index}/{outer_folds} "
            f"holdout_errors={fold_report['holdout_metrics']['errors']} "
            f"overrides={int(holdout_override.sum())}",
            flush=True,
        )

    if np.isnan(final_scores).any() or np.any(oof_fold < 0):
        raise ValueError("Nested OOF materialization left unfilled train rows")

    arrays = {
        "oof_fold": oof_fold,
        "base_scores": base_scores,
        "candidate_scores": candidate_scores,
        "allow_scores": allow_scores,
        "final_scores": final_scores,
        "final_predictions": final_predictions,
        "override_mask": override_mask,
        "possible_mask": possible_mask,
        "candidate_thresholds": candidate_thresholds,
        "allow_thresholds": allow_threshold_values,
    }
    return arrays, fold_reports


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    checkpoint = load_safe_checkpoint(resolve_path(args.checkpoint), map_location="cpu")
    checkpoint_config = AxonExperimentConfig.from_dict(dict(checkpoint["config"]))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_config = FeatureConfig(
        prefix_len=max(0, int(args.prefix_len)),
        chunk_count=max(1, int(args.chunk_count)),
        include_pe=True,
        include_stat=True,
        include_lightweight=args.feature_set == "extended",
        include_byte_summary=args.feature_set == "extended",
        include_content_pe=True,
        content_cache_dir=str(resolve_path(args.content_pe_cache_dir)),
    )
    safe_feature_name_groups = assert_stage2_feature_names_safe(feature_config, checkpoint_config=checkpoint_config)
    assert_no_identity_feature_names(OVERLAY_BOUNDARY_FEATURE_NAMES, context="Loop69 overlay boundary features")

    rows = read_prediction_rows(args.train_predictions, args.max_train_rows)
    train_x, train_y, _exported_probs, kept_rows, counts = build_matrix(rows, checkpoint_config, feature_config)
    dropped_feature_count = 0
    if args.drop_base_prob_features:
        dropped_feature_count = 6
        train_x = train_x[:, dropped_feature_count:].astype(np.float32, copy=False)

    overlay_config = OverlayBoundaryConfig(cache_dir=str(resolve_path(args.overlay_boundary_cache_dir)))
    train_overlay = build_overlay_boundary_matrix(kept_rows, overlay_config)

    base_specs = filter_model_candidates(model_candidates(int(args.seed)), args.base_model_candidate)
    if len(base_specs) != 1:
        raise ValueError(f"Expected exactly one base model candidate, got {[name for name, _ in base_specs]}")
    candidate_specs = filter_model_candidates(model_candidates(int(args.seed)), args.candidate_model_candidate)
    if len(candidate_specs) != 1:
        raise ValueError(f"Expected exactly one candidate model, got {[name for name, _ in candidate_specs]}")
    override_specs = _filter_override_candidates(override_model_candidates(int(args.seed)), args.override_model_candidate)
    if len(override_specs) != 1:
        raise ValueError(f"Expected exactly one override model, got {[name for name, _ in override_specs]}")

    arrays, fold_reports = materialize_nested_oof(
        train_x=train_x,
        train_y=train_y,
        train_overlay=train_overlay,
        base_spec=base_specs[0],
        candidate_spec=candidate_specs[0],
        override_spec=override_specs[0],
        thresholds=parse_thresholds(args.thresholds),
        allow_thresholds=parse_thresholds(args.allow_thresholds),
        base_threshold=float(args.base_threshold),
        outer_folds=int(args.outer_folds),
        inner_folds=int(args.inner_folds),
        seed=int(args.seed),
        include_override_overlay_features=not bool(args.no_override_overlay_features),
        include_override_content_features=bool(args.override_content_features),
    )

    predictions_path = output_dir / "loop69_nested_oof_override_train_predictions.csv"
    write_nested_oof_predictions(
        predictions_path,
        kept_rows,
        train_y,
        oof_fold=arrays["oof_fold"],
        base_scores=arrays["base_scores"],
        candidate_scores=arrays["candidate_scores"],
        allow_scores=arrays["allow_scores"],
        final_scores=arrays["final_scores"],
        final_predictions=arrays["final_predictions"],
        override_mask=arrays["override_mask"],
        possible_mask=arrays["possible_mask"],
        candidate_thresholds=arrays["candidate_thresholds"],
        allow_thresholds=arrays["allow_thresholds"],
        selected_candidate=base_specs[0][0] + " -> " + candidate_specs[0][0],
        selected_override_model=override_specs[0][0],
    )

    final_metrics = prediction_metrics(train_y, arrays["final_predictions"], arrays["final_scores"])
    base_metrics = metrics_at_threshold(arrays["base_scores"], train_y, float(args.base_threshold))
    candidate_threshold_mean = float(np.mean(arrays["candidate_thresholds"]))
    candidate_metrics = metrics_at_threshold(arrays["candidate_scores"], train_y, candidate_threshold_mean)
    report = {
        "schema": "axon_loop69_nested_oof_override_v1",
        "protocol": (
            "train-only nested OOF materialization for Loop61-style override pipeline; no Val selection, "
            "no Test-10k/full-test, no third-layer training"
        ),
        "identity_feature_policy": (
            "source_path/cache_path/source_sha256/sample_index/split are alignment/cache/audit fields only. "
            "They are written to the CSV for traceability and are forbidden as model features."
        ),
        "checkpoint": str(resolve_path(args.checkpoint)),
        "train_predictions": str(resolve_path(args.train_predictions)),
        "records": counts,
        "feature_config": feature_config.__dict__,
        "feature_name_groups": safe_feature_name_groups,
        "dropped_feature_count": dropped_feature_count,
        "outer_folds": int(args.outer_folds),
        "inner_folds": int(args.inner_folds),
        "base_model": base_specs[0][0],
        "candidate_model": candidate_specs[0][0],
        "override_model": override_specs[0][0],
        "base_threshold": float(args.base_threshold),
        "include_override_overlay_features": not bool(args.no_override_overlay_features),
        "include_override_content_features": bool(args.override_content_features),
        "metrics": final_metrics,
        "base_metrics": base_metrics,
        "candidate_metrics_at_mean_fold_threshold": candidate_metrics,
        "candidate_threshold_summary": {
            "min": float(np.min(arrays["candidate_thresholds"])),
            "mean": candidate_threshold_mean,
            "max": float(np.max(arrays["candidate_thresholds"])),
        },
        "allow_threshold_summary": {
            "min": float(np.min(arrays["allow_thresholds"])),
            "mean": float(np.mean(arrays["allow_thresholds"])),
            "max": float(np.max(arrays["allow_thresholds"])),
        },
        "override_summary": {
            "possible_override_count": int(arrays["possible_mask"].sum()),
            "override_count": int(arrays["override_mask"].sum()),
            "override_label1_count": int(((train_y == 1) & arrays["override_mask"]).sum()),
            "override_label0_count": int(((train_y == 0) & arrays["override_mask"]).sum()),
        },
        "fold_reports": fold_reports,
        "artifacts": {
            "train_oof_predictions": str(predictions_path),
        },
    }
    report_path = output_dir / "loop69_nested_oof_override_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics": final_metrics, "predictions": str(predictions_path)}, indent=2))
    print(f"JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
