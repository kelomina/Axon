#!/usr/bin/env python3
"""Train an out-of-fold Stage-2 stacker from cache-backed Axon features."""

from __future__ import annotations

import argparse
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
from security import load_safe_checkpoint  # noqa: E402
from train_stage2_cache_matrix import (  # noqa: E402
    CONTENT_CERT_FEATURE_NAMES,
    CONTENT_PE_FEATURE_NAMES,
    CONTENT_STRING_FEATURE_NAMES,
    FeatureConfig,
    build_matrix,
    clean_slice_metrics,
    content_pe_v2_selected_feature_names,
    filter_model_candidates,
    assert_stage2_feature_names_safe,
    metrics_at_threshold,
    model_candidates,
    parse_content_pe_v2_groups,
    parse_thresholds,
    predict_scores,
    read_prediction_rows,
    resolve_path,
    sample_weights,
    select_best_threshold,
    summarize_noise,
    summarize_weights,
    write_predictions,
)

STAGE2_PROB_FEATURE_COUNT = 6


def fit_model(model, matrix: np.ndarray, labels: np.ndarray, weights: np.ndarray):
    try:
        model.fit(matrix, labels, sample_weight=weights)
    except TypeError:
        model.fit(matrix, labels)
    return model


def build_stack_features(base_scores: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if base_scores.ndim != 2:
        raise ValueError(f"base_scores must be 2-D, got shape={base_scores.shape}")
    clipped = np.clip(base_scores.astype(np.float32, copy=False), 1.0e-6, 1.0 - 1.0e-6)
    columns = [clipped]
    names = [f"base_model_{index}_score" for index in range(clipped.shape[1])]

    mean = clipped.mean(axis=1, keepdims=True)
    std = clipped.std(axis=1, keepdims=True)
    minimum = clipped.min(axis=1, keepdims=True)
    maximum = clipped.max(axis=1, keepdims=True)
    spread = maximum - minimum
    median = np.median(clipped, axis=1, keepdims=True)
    logit_mean = np.log(clipped / (1.0 - clipped)).mean(axis=1, keepdims=True)
    columns.extend([mean, std, minimum, maximum, spread, median, logit_mean])
    names.extend(
        [
            "base_score_mean",
            "base_score_std",
            "base_score_min",
            "base_score_max",
            "base_score_spread",
            "base_score_median",
            "base_score_logit_mean",
        ]
    )
    return np.hstack(columns).astype(np.float32, copy=False), names


def meta_model_candidates(seed: int) -> list[tuple[str, object]]:
    return [
        (
            "meta_logreg_l2_c0.1",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="liblinear", C=0.1)),
        ),
        (
            "meta_logreg_l2_c1",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="liblinear", C=1.0)),
        ),
        (
            "meta_logreg_balanced_c0.1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, solver="liblinear", C=0.1, class_weight="balanced"),
            ),
        ),
        (
            "meta_hgb_leaf7",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_leaf_nodes=7,
                l2_regularization=1.0e-3,
                max_iter=160,
                random_state=seed,
            ),
        ),
        (
            "meta_hgb_leaf15",
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=1.0e-3,
                max_iter=180,
                random_state=seed,
            ),
        ),
    ]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OOF stacker for Stage-2 cache features.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.05:0.95:0.005")
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
    parser.add_argument("--base-model-candidates", default="")
    parser.add_argument("--noise-modes", default="none")
    parser.add_argument(
        "--drop-base-prob-features",
        action="store_true",
        help="Remove the first six exported base-probability features from the Stage-2 matrix.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    checkpoint = load_safe_checkpoint(resolve_path(args.checkpoint), map_location="cpu")
    checkpoint_config = AxonExperimentConfig.from_dict(dict(checkpoint["config"]))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    content_cache_dir = None
    if args.content_pe_features:
        content_cache_dir = resolve_path(args.content_pe_cache_dir or (output_dir / "content_pe_cache_v1"))
    content_pe_v2_cache_dir = None
    if args.content_pe_v2_features:
        content_pe_v2_cache_dir = resolve_path(args.content_pe_v2_cache_dir or (output_dir / "content_pe_v2_cache"))
    content_string_cache_dir = None
    if args.content_string_features:
        content_string_cache_dir = resolve_path(args.content_string_cache_dir or (output_dir / "content_string_cache_v1"))
    content_cert_cache_dir = None
    if args.content_cert_features:
        content_cert_cache_dir = resolve_path(args.content_cert_cache_dir or (output_dir / "content_cert_cache_v1"))
    content_pe_v2_groups = parse_content_pe_v2_groups(args.content_pe_v2_groups)

    feature_config = FeatureConfig(
        prefix_len=max(0, int(args.prefix_len)),
        chunk_count=max(1, int(args.chunk_count)),
        include_pe=True,
        include_stat=True,
        include_lightweight=args.feature_set == "extended",
        include_byte_summary=args.feature_set == "extended",
        include_content_pe=bool(args.content_pe_features),
        content_cache_dir=str(content_cache_dir) if content_cache_dir is not None else None,
        include_content_pe_v2=bool(args.content_pe_v2_features),
        content_pe_v2_cache_dir=str(content_pe_v2_cache_dir) if content_pe_v2_cache_dir is not None else None,
        content_pe_v2_groups=content_pe_v2_groups,
        include_content_string=bool(args.content_string_features),
        content_string_cache_dir=str(content_string_cache_dir) if content_string_cache_dir is not None else None,
        include_content_cert=bool(args.content_cert_features),
        content_cert_cache_dir=str(content_cert_cache_dir) if content_cert_cache_dir is not None else None,
    )

    train_rows = read_prediction_rows(args.train_predictions)
    val_rows = read_prediction_rows(args.val_predictions)
    print(f"[load] train rows={len(train_rows)} val rows={len(val_rows)}", flush=True)
    train_x, train_y, train_base, train_kept_rows, train_counts = build_matrix(train_rows, checkpoint_config, feature_config)
    val_x, val_y, val_base, val_kept_rows, val_counts = build_matrix(val_rows, checkpoint_config, feature_config)
    dropped_feature_count = 0
    if args.drop_base_prob_features:
        dropped_feature_count = STAGE2_PROB_FEATURE_COUNT
        train_x = train_x[:, STAGE2_PROB_FEATURE_COUNT:].astype(np.float32, copy=False)
        val_x = val_x[:, STAGE2_PROB_FEATURE_COUNT:].astype(np.float32, copy=False)
    print(f"[matrix] train={train_x.shape} val={val_x.shape}", flush=True)

    thresholds = parse_thresholds(args.thresholds)
    baseline_val_best = select_best_threshold(val_base, val_y, thresholds)
    noise_modes = [item.strip() for item in args.noise_modes.split(",") if item.strip()]
    base_candidates = filter_model_candidates(model_candidates(int(args.seed)), args.base_model_candidates)
    base_specs = [(model_name, noise_mode, model) for noise_mode in noise_modes for model_name, model in base_candidates]
    if not base_specs:
        raise ValueError("No base model candidates selected")

    folds = min(max(2, int(args.folds)), int(np.bincount(train_y).min()))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(args.seed))
    oof_scores = np.zeros((train_x.shape[0], len(base_specs)), dtype=np.float32)
    val_base_scores = np.zeros((val_x.shape[0], len(base_specs)), dtype=np.float32)
    fitted_base_models = []
    base_reports = []

    for base_index, (model_name, noise_mode, prototype) in enumerate(base_specs):
        weights = sample_weights(train_y, train_base, noise_mode)
        weight_summary = summarize_weights(train_y, weights)
        fold_reports = []
        start = time.perf_counter()
        for fold_index, (fit_idx, holdout_idx) in enumerate(splitter.split(train_x, train_y), start=1):
            fold_model = clone(prototype)
            fit_model(fold_model, train_x[fit_idx], train_y[fit_idx], weights[fit_idx])
            fold_scores = predict_scores(fold_model, train_x[holdout_idx])
            oof_scores[holdout_idx, base_index] = fold_scores
            fold_metrics = select_best_threshold(fold_scores, train_y[holdout_idx], thresholds)
            fold_reports.append({"fold": fold_index, "rows": int(holdout_idx.shape[0]), "best": fold_metrics})
            print(
                f"[oof] {model_name} noise={noise_mode} fold={fold_index}/{folds} "
                f"errors={fold_metrics['errors']} f1={fold_metrics['f1']:.6f}",
                flush=True,
            )

        full_model = clone(prototype)
        fit_model(full_model, train_x, train_y, weights)
        val_scores = predict_scores(full_model, val_x)
        val_base_scores[:, base_index] = val_scores
        val_best = select_best_threshold(val_scores, val_y, thresholds)
        fit_sec = time.perf_counter() - start
        fitted_base_models.append(full_model)
        base_reports.append(
            {
                "name": f"{model_name}__noise_{noise_mode}",
                "base_model": model_name,
                "noise_mode": noise_mode,
                "fit_sec": fit_sec,
                "weight_summary": weight_summary,
                "folds": fold_reports,
                "val_best_full_base": val_best,
            }
        )
        print(
            f"[base-val] {model_name} noise={noise_mode} f1={val_best['f1']:.6f} "
            f"errors={val_best['errors']} threshold={val_best['threshold']:.4f}",
            flush=True,
        )

    stack_train, stack_feature_names = build_stack_features(oof_scores)
    stack_val, _ = build_stack_features(val_base_scores)
    safe_feature_name_groups = assert_stage2_feature_names_safe(
        feature_config,
        checkpoint_config=checkpoint_config,
    )
    if args.drop_base_prob_features:
        safe_feature_name_groups = dict(safe_feature_name_groups)
        safe_feature_name_groups["base_probability_features"] = []
    assert_stage2_feature_names_safe(
        feature_config,
        stack_feature_names,
        checkpoint_config,
    )
    meta_results = []
    fitted_meta = []
    for meta_name, meta_model in meta_model_candidates(int(args.seed)):
        start = time.perf_counter()
        meta_model.fit(stack_train, train_y)
        fit_sec = time.perf_counter() - start
        train_scores = predict_scores(meta_model, stack_train)
        val_scores = predict_scores(meta_model, stack_val)
        train_best = select_best_threshold(train_scores, train_y, thresholds)
        val_best = select_best_threshold(val_scores, val_y, thresholds)
        clean_val = clean_slice_metrics(val_scores, val_y, val_base, float(val_best["threshold"]))
        result = {
            "name": meta_name,
            "fit_sec": fit_sec,
            "train_oof_best": train_best,
            "val_best": val_best,
            "clean_val_at_val_threshold": clean_val,
            "delta_val_f1_vs_loop28": val_best["f1"] - 0.9919048570857486,
            "delta_val_errors_vs_loop28": val_best["errors"] - 162,
        }
        meta_results.append(result)
        fitted_meta.append((val_best["f1"], -val_best["errors"], result, meta_model, val_scores))
        print(
            f"[meta-val] {meta_name} f1={val_best['f1']:.6f} errors={val_best['errors']} "
            f"threshold={val_best['threshold']:.4f}",
            flush=True,
        )

    fitted_meta.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _selected_f1, _neg_errors, selected, selected_meta_model, selected_val_scores = fitted_meta[0]
    selected_threshold = float(selected["val_best"]["threshold"])
    write_predictions(output_dir / "stage2_oof_stacker_val_predictions.csv", val_kept_rows, val_y, selected_val_scores, selected_threshold)

    payload = {
        "schema": "axon_stage2_oof_stacker_payload_v1",
        "protocol": "base learners train OOF on train; meta trains only on OOF predictions; val selects meta model and threshold; no test used",
        "base_models": fitted_base_models,
        "meta_model": selected_meta_model,
        "threshold": selected_threshold,
        "feature_config": feature_config,
        "drop_base_prob_features": bool(args.drop_base_prob_features),
        "dropped_feature_count": int(dropped_feature_count),
        "checkpoint_config": checkpoint_config.to_dict(),
        "identity_feature_policy": (
            "source_path/source_sha256/cache_path/sample_index/split/filename/extension/directory are identity "
            "or audit fields only and are forbidden as model features"
        ),
        "feature_name_groups": safe_feature_name_groups,
        "base_specs": [
            {"name": report["name"], "base_model": report["base_model"], "noise_mode": report["noise_mode"]}
            for report in base_reports
        ],
        "stack_feature_names": stack_feature_names,
        "selected": selected,
        "content_pe_feature_names": CONTENT_PE_FEATURE_NAMES if feature_config.include_content_pe else [],
        "content_pe_v2_feature_names": (
            content_pe_v2_selected_feature_names(feature_config.content_pe_v2_groups)
            if feature_config.include_content_pe_v2
            else []
        ),
        "content_string_feature_names": CONTENT_STRING_FEATURE_NAMES if feature_config.include_content_string else [],
        "content_cert_feature_names": CONTENT_CERT_FEATURE_NAMES if feature_config.include_content_cert else [],
    }
    model_path = output_dir / "stage2_oof_stacker_selected_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle)

    report = {
        "schema": "axon_stage2_oof_stacker_v1",
        "protocol": payload["protocol"],
        "checkpoint": str(resolve_path(args.checkpoint)),
        "train_predictions": str(resolve_path(args.train_predictions)),
        "val_predictions": str(resolve_path(args.val_predictions)),
        "feature_config": feature_config.__dict__,
        "identity_feature_policy": payload["identity_feature_policy"],
        "feature_name_groups": safe_feature_name_groups,
        "drop_base_prob_features": bool(args.drop_base_prob_features),
        "dropped_feature_count": int(dropped_feature_count),
        "records": {"train": train_counts, "val": val_counts},
        "feature_dim": int(train_x.shape[1]),
        "folds": folds,
        "stack_feature_names": stack_feature_names,
        "base_reports": base_reports,
        "baseline_val_best": baseline_val_best,
        "noise_summary": {
            "train": summarize_noise(train_y, train_base),
            "val": summarize_noise(val_y, val_base),
        },
        "meta_models": sorted(meta_results, key=lambda row: (row["val_best"]["f1"], -row["val_best"]["errors"]), reverse=True),
        "selected_by_val": selected,
        "model_path": str(model_path),
    }
    report_path = output_dir / "stage2_oof_stacker_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected_by_val": selected}, indent=2, ensure_ascii=False))
    print(f"JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
