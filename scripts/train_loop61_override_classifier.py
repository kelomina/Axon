#!/usr/bin/env python3
"""Train an override-only classifier for Loop57-style FN repairs.

Loop61 narrows the second-stage question to rows where the locked/base model
predicts benign and an overlay-aware candidate predicts malicious. The override
classifier is trained only on those possible 0->1 override rows and decides
whether the override should be allowed. Identity fields are alignment/cache
fields only, never model features.
"""

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
from train_loop57_fn_overlay_gate import (  # noqa: E402
    align_external_scores,
    build_fn_gate_matrix,
    write_fn_gate_predictions,
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
    summarize_noise,
)


LOOP57_VAL_F1 = 0.9926635723910765
LOOP57_VAL_ERRORS = 147


def possible_override_mask(
    *,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    base_threshold: float,
    candidate_threshold: float,
) -> np.ndarray:
    """Rows where a candidate can make a strict benign->malicious override."""

    base_pred = (base_scores >= base_threshold).astype(np.int64)
    candidate_pred = (candidate_scores >= candidate_threshold).astype(np.int64)
    return (base_pred == 0) & (candidate_pred == 1)


def override_classifier_predictions(
    *,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    allow_scores: np.ndarray,
    base_threshold: float,
    candidate_threshold: float,
    allow_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply only candidate-approved 0->1 overrides."""

    base_pred = (base_scores >= base_threshold).astype(np.int64)
    possible = possible_override_mask(
        base_scores=base_scores,
        candidate_scores=candidate_scores,
        base_threshold=base_threshold,
        candidate_threshold=candidate_threshold,
    )
    override = possible & (allow_scores >= allow_threshold)
    final_pred = np.where(override, 1, base_pred).astype(np.int64)
    final_scores = np.where(override, candidate_scores, base_scores).astype(np.float32, copy=False)
    return final_pred, final_scores, override


def override_target_summary(labels: np.ndarray, possible: np.ndarray) -> dict:
    return {
        "possible_overrides": int(possible.sum()),
        "beneficial_fn_repairs": int(((labels == 1) & possible).sum()),
        "harmful_new_fp": int(((labels == 0) & possible).sum()),
        "label1_ratio_in_possible": (
            float(((labels == 1) & possible).sum() / possible.sum()) if int(possible.sum()) else None
        ),
    }


def select_override_allow_threshold(
    *,
    labels: np.ndarray,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    allow_scores: np.ndarray,
    base_threshold: float,
    candidate_threshold: float,
    allow_thresholds: Sequence[float],
) -> dict:
    rows = []
    possible = possible_override_mask(
        base_scores=base_scores,
        candidate_scores=candidate_scores,
        base_threshold=base_threshold,
        candidate_threshold=candidate_threshold,
    )
    for allow_threshold in allow_thresholds:
        predictions, final_scores, override = override_classifier_predictions(
            base_scores=base_scores,
            candidate_scores=candidate_scores,
            allow_scores=allow_scores,
            base_threshold=base_threshold,
            candidate_threshold=candidate_threshold,
            allow_threshold=float(allow_threshold),
        )
        metrics = prediction_metrics(labels, predictions, final_scores)
        metrics["allow_threshold"] = float(allow_threshold)
        metrics["possible_override_count"] = int(possible.sum())
        metrics["override_count"] = int(override.sum())
        metrics["override_ratio"] = float(override.mean())
        metrics["override_label1_count"] = int(((labels == 1) & override).sum())
        metrics["override_label0_count"] = int(((labels == 0) & override).sum())
        metrics["blocked_possible_count"] = int(possible.sum() - override.sum())
        metrics["blocked_label1_count"] = int(((labels == 1) & possible & ~override).sum())
        metrics["blocked_label0_count"] = int(((labels == 0) & possible & ~override).sum())
        rows.append(metrics)
    rows.sort(
        key=lambda row: (
            row["f1"],
            -row["errors"],
            -row["override_label0_count"],
            row["override_label1_count"],
            -row["override_count"],
            row["allow_threshold"],
        ),
        reverse=True,
    )
    return rows[0]


def override_model_candidates(seed: int) -> list[tuple[str, object]]:
    """Small models only; possible override rows are intentionally sparse."""

    return [
        (
            "override_logreg_balanced_c0.10",
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
            "override_logreg_balanced_c0.25",
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
            "override_logreg_balanced_c1",
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
            "override_hgb_leaf3",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_leaf_nodes=3,
                l2_regularization=1.0e-2,
                max_iter=120,
                random_state=seed,
            ),
        ),
        (
            "override_hgb_leaf7",
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_leaf_nodes=7,
                l2_regularization=1.0e-2,
                max_iter=140,
                random_state=seed,
            ),
        ),
    ]


def _filter_override_candidates(candidates: list[tuple[str, object]], names: str) -> list[tuple[str, object]]:
    selected_names = [name.strip() for name in names.split(",") if name.strip()]
    if not selected_names:
        return candidates
    selected = [(name, model) for name, model in candidates if name in selected_names]
    missing = sorted(set(selected_names) - {name for name, _model in selected})
    if missing:
        available = ", ".join(name for name, _model in candidates)
        raise ValueError(f"Unknown override classifier candidate(s): {missing}. Available: {available}")
    return selected


def _fit_override_classifier(model, matrix: np.ndarray, labels: np.ndarray):
    if matrix.shape[0] == 0:
        raise ValueError("No possible override rows are available for fitting")
    if len(np.unique(labels)) < 2:
        raise ValueError("Possible override training rows contain only one class")
    return fit_with_optional_weights(model, matrix, labels)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Loop61 override-only classifier.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--baseline-val-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
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
    parser.add_argument(
        "--candidate-model-candidates",
        default="hgb_lr0.06_leaf31_l2_0,hgb_lr0.08_leaf31_l2_1e-3,extra_trees_300_leaf1",
    )
    parser.add_argument("--override-model-candidates", default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--baseline-probability-column", default="stage2_prob_malicious")
    parser.add_argument("--alignment-key-column", default="sample_index")
    parser.add_argument("--base-threshold", type=float, default=0.5)
    parser.add_argument("--reference-val-errors", type=int, default=LOOP57_VAL_ERRORS)
    parser.add_argument("--reference-val-f1", type=float, default=LOOP57_VAL_F1)
    parser.add_argument("--min-val-error-improvement", type=int, default=2)
    return parser.parse_args(argv)


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
    assert_no_identity_feature_names(OVERLAY_BOUNDARY_FEATURE_NAMES, context="Loop61 overlay boundary features")

    train_rows = read_prediction_rows(args.train_predictions, args.max_train_rows)
    val_rows = read_prediction_rows(args.val_predictions, args.max_val_rows)
    train_x, train_y, train_base_exported, train_kept_rows, train_counts = build_matrix(
        train_rows,
        checkpoint_config,
        feature_config,
    )
    val_x, val_y, val_base_exported, val_kept_rows, val_counts = build_matrix(
        val_rows,
        checkpoint_config,
        feature_config,
    )
    dropped_feature_count = 0
    if args.drop_base_prob_features:
        dropped_feature_count = 6
        train_x = train_x[:, dropped_feature_count:].astype(np.float32, copy=False)
        val_x = val_x[:, dropped_feature_count:].astype(np.float32, copy=False)

    overlay_config = OverlayBoundaryConfig(cache_dir=str(resolve_path(args.overlay_boundary_cache_dir)))
    train_overlay = build_overlay_boundary_matrix(train_kept_rows, overlay_config)
    val_overlay = build_overlay_boundary_matrix(val_kept_rows, overlay_config)
    candidate_train_x = np.hstack([train_x, train_overlay]).astype(np.float32, copy=False)
    candidate_val_x = np.hstack([val_x, val_overlay]).astype(np.float32, copy=False)
    print(
        f"[matrix] base_train={train_x.shape} candidate_train={candidate_train_x.shape} val={val_x.shape}",
        flush=True,
    )

    thresholds = parse_thresholds(args.thresholds)
    allow_thresholds = parse_thresholds(args.allow_thresholds)
    folds = min(max(2, int(args.folds)), int(np.bincount(train_y).min()))

    base_specs = filter_model_candidates(model_candidates(int(args.seed)), args.base_model_candidate)
    if len(base_specs) != 1:
        raise ValueError(f"Expected exactly one base model candidate, got {[name for name, _ in base_specs]}")
    candidate_specs = filter_model_candidates(model_candidates(int(args.seed)), args.candidate_model_candidates)
    if not candidate_specs:
        raise ValueError("No candidate override models selected")

    base_oof, base_val_internal, _fitted_base_models, base_reports = oof_stage2_scores(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        specs=base_specs,
        folds=folds,
        seed=int(args.seed),
    )
    candidate_oof, candidate_val, fitted_candidate_models, candidate_reports_raw = oof_stage2_scores(
        train_x=candidate_train_x,
        train_y=train_y,
        val_x=candidate_val_x,
        specs=candidate_specs,
        folds=folds,
        seed=int(args.seed) + 100,
    )

    base_name = base_specs[0][0]
    base_train_scores = base_oof[:, 0]
    internal_base_val_scores = base_val_internal[:, 0]
    base_threshold = float(args.base_threshold)
    internal_base_val_at_threshold = metrics_at_threshold(internal_base_val_scores, val_y, base_threshold)

    external_base_scores, external_alignment = align_external_scores(
        rows=val_kept_rows,
        prediction_path=args.baseline_val_predictions,
        probability_column=args.baseline_probability_column,
        key_column=args.alignment_key_column,
    )
    external_base_at_threshold = metrics_at_threshold(external_base_scores, val_y, base_threshold)

    include_overlay_for_override = not bool(args.no_override_overlay_features)
    include_content_for_override = bool(args.override_content_features)
    override_candidates = _filter_override_candidates(override_model_candidates(int(args.seed)), args.override_model_candidates)
    if not override_candidates:
        raise ValueError("No override classifier candidates selected")

    fitted_results = []
    candidate_reports = []
    for candidate_index, (candidate_name, _prototype) in enumerate(candidate_specs):
        candidate_train_scores = candidate_oof[:, candidate_index]
        candidate_val_scores = candidate_val[:, candidate_index]
        candidate_train_best = select_best_threshold(candidate_train_scores, train_y, thresholds)
        candidate_threshold = float(candidate_train_best["threshold"])
        candidate_val_at_train_threshold = metrics_at_threshold(candidate_val_scores, val_y, candidate_threshold)
        train_possible = possible_override_mask(
            base_scores=base_train_scores,
            candidate_scores=candidate_train_scores,
            base_threshold=base_threshold,
            candidate_threshold=candidate_threshold,
        )
        val_possible = possible_override_mask(
            base_scores=external_base_scores,
            candidate_scores=candidate_val_scores,
            base_threshold=base_threshold,
            candidate_threshold=candidate_threshold,
        )
        train_target_summary = override_target_summary(train_y, train_possible)
        val_possible_summary = override_target_summary(val_y, val_possible)
        if int(train_possible.sum()) == 0 or len(np.unique(train_y[train_possible])) < 2:
            print(
                f"[override-skip] {candidate_name}: possible={int(train_possible.sum())} "
                f"classes={np.unique(train_y[train_possible]).tolist()}",
                flush=True,
            )
            candidate_reports.append(
                {
                    "candidate": candidate_name,
                    "candidate_train_best": candidate_train_best,
                    "candidate_val_at_train_threshold": candidate_val_at_train_threshold,
                    "train_target_summary": train_target_summary,
                    "val_possible_summary": val_possible_summary,
                    "override_models": [],
                    "skip_reason": "no_two_class_possible_override_training_rows",
                }
            )
            continue

        train_override_x, override_feature_names = build_fn_gate_matrix(
            train_x,
            train_overlay,
            base_train_scores,
            candidate_train_scores,
            include_overlay_features=include_overlay_for_override,
            include_content_features=include_content_for_override,
        )
        val_override_x, _ = build_fn_gate_matrix(
            val_x,
            val_overlay,
            external_base_scores,
            candidate_val_scores,
            include_overlay_features=include_overlay_for_override,
            include_content_features=include_content_for_override,
        )
        train_override_possible_x = train_override_x[train_possible]
        train_override_possible_y = train_y[train_possible]

        override_model_reports = []
        for override_name, override_prototype in override_candidates:
            start = time.perf_counter()
            override_model = clone(override_prototype)
            try:
                _fit_override_classifier(override_model, train_override_possible_x, train_override_possible_y)
            except ValueError as exc:
                override_model_reports.append(
                    {
                        "override_model": override_name,
                        "skip_reason": str(exc),
                    }
                )
                print(f"[override-skip] candidate={candidate_name} model={override_name}: {exc}", flush=True)
                continue

            allow_train_scores = predict_scores(override_model, train_override_x)
            allow_val_scores = predict_scores(override_model, val_override_x)
            train_best = select_override_allow_threshold(
                labels=train_y,
                base_scores=base_train_scores,
                candidate_scores=candidate_train_scores,
                allow_scores=allow_train_scores,
                base_threshold=base_threshold,
                candidate_threshold=candidate_threshold,
                allow_thresholds=allow_thresholds,
            )
            val_best = select_override_allow_threshold(
                labels=val_y,
                base_scores=external_base_scores,
                candidate_scores=candidate_val_scores,
                allow_scores=allow_val_scores,
                base_threshold=base_threshold,
                candidate_threshold=candidate_threshold,
                allow_thresholds=allow_thresholds,
            )
            report_row = {
                "candidate": candidate_name,
                "override_model": override_name,
                "fit_sec": time.perf_counter() - start,
                "base_name": base_name,
                "base_threshold": base_threshold,
                "candidate_train_threshold": candidate_threshold,
                "train_target_summary": train_target_summary,
                "val_possible_summary": val_possible_summary,
                "candidate_val_at_train_threshold": candidate_val_at_train_threshold,
                "train_best": train_best,
                "val_best": val_best,
                "delta_val_errors_vs_external_base": int(val_best["errors"])
                - int(external_base_at_threshold["errors"]),
                "delta_val_errors_vs_loop57_reference": int(val_best["errors"]) - int(args.reference_val_errors),
                "delta_val_f1_vs_loop57_reference": float(val_best["f1"]) - float(args.reference_val_f1),
            }
            override_model_reports.append(report_row)
            fitted_results.append(
                (
                    float(val_best["f1"]),
                    -int(val_best["errors"]),
                    -int(val_best["override_label0_count"]),
                    int(val_best["override_label1_count"]),
                    report_row,
                    override_model,
                    allow_val_scores,
                    candidate_index,
                    candidate_val_scores,
                    override_feature_names,
                )
            )
            print(
                f"[override-val] candidate={candidate_name} model={override_name} "
                f"f1={val_best['f1']:.6f} errors={val_best['errors']} "
                f"overrides={val_best['override_count']} "
                f"label1={val_best['override_label1_count']} label0={val_best['override_label0_count']}",
                flush=True,
            )
        candidate_reports.append(
            {
                "candidate": candidate_name,
                "candidate_train_best": candidate_train_best,
                "candidate_val_at_train_threshold": candidate_val_at_train_threshold,
                "train_target_summary": train_target_summary,
                "val_possible_summary": val_possible_summary,
                "override_models": override_model_reports,
            }
        )

    if not fitted_results:
        raise ValueError("No fitted override classifier results were produced")
    fitted_results.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    (
        _best_f1,
        _neg_errors,
        _neg_label0_overrides,
        _label1_overrides,
        selected,
        selected_override_model,
        selected_allow_val_scores,
        selected_candidate_index,
        selected_candidate_val_scores,
        override_feature_names,
    ) = fitted_results[0]
    selected_candidate_name = selected["candidate"]
    selected_candidate_threshold = float(selected["candidate_train_threshold"])
    selected_allow_threshold = float(selected["val_best"]["allow_threshold"])
    final_predictions, final_scores, override_mask = override_classifier_predictions(
        base_scores=external_base_scores,
        candidate_scores=selected_candidate_val_scores,
        allow_scores=selected_allow_val_scores,
        base_threshold=base_threshold,
        candidate_threshold=selected_candidate_threshold,
        allow_threshold=selected_allow_threshold,
    )

    val_predictions_path = output_dir / "loop61_override_classifier_val_predictions.csv"
    write_fn_gate_predictions(
        val_predictions_path,
        val_kept_rows,
        val_y,
        base_scores=external_base_scores,
        candidate_scores=selected_candidate_val_scores,
        gate_scores=selected_allow_val_scores,
        final_scores=final_scores,
        final_predictions=final_predictions,
        override_mask=override_mask,
        selected_candidate=selected_candidate_name,
    )

    model_path = output_dir / "loop61_override_classifier_selected_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "schema": "axon_loop61_override_classifier_payload_v1",
                # Keep Loop57-compatible names so the frozen evaluator can score Test-10k if Val passes.
                "gate_model": selected_override_model,
                "candidate_model": fitted_candidate_models[selected_candidate_index],
                "selected": selected,
                "base_threshold": base_threshold,
                "candidate_threshold": selected_candidate_threshold,
                "gate_threshold": selected_allow_threshold,
                "feature_config": feature_config,
                "checkpoint_config": checkpoint_config.to_dict(),
                "dropped_feature_count": dropped_feature_count,
                "gate_feature_names": override_feature_names,
                "include_gate_overlay_features": include_overlay_for_override,
                "include_gate_content_features": include_content_for_override,
                "overlay_boundary_feature_names": OVERLAY_BOUNDARY_FEATURE_NAMES,
                "identity_feature_policy": (
                    "source_path/source_sha256/cache_path/sample_index/split/filename/extension/directory "
                    "are alignment or loading fields only and are forbidden as model features"
                ),
            },
            handle,
        )

    val_kept_count = int(val_counts.get("kept", 0)) if isinstance(val_counts, dict) else int(len(val_y))
    selected_errors = int(selected["val_best"]["errors"])
    required_errors = int(args.reference_val_errors) - int(args.min_val_error_improvement)
    if val_kept_count < 20000:
        test_gate_decision = "smoke_only_not_eligible_for_test10k"
    elif selected_errors <= required_errors:
        test_gate_decision = "eligible_for_test10k"
    else:
        test_gate_decision = "reject_val_margin_too_small"

    report = {
        "schema": "axon_loop61_override_classifier_v1",
        "protocol": (
            "base/candidate train scores are strict OOF; classifier is trained only on possible 0->1 "
            "override rows; Val selects classifier and allow threshold; no Test-10k/full-test used"
        ),
        "identity_feature_policy": (
            "filename/path/extension/directory/source hash/sample id/split/row order are "
            "alignment/cache/audit fields only and are not model features"
        ),
        "checkpoint": str(resolve_path(args.checkpoint)),
        "train_predictions": str(resolve_path(args.train_predictions)),
        "val_predictions": str(resolve_path(args.val_predictions)),
        "baseline_val_predictions": str(resolve_path(args.baseline_val_predictions)),
        "records": {"train": train_counts, "val": val_counts},
        "feature_config": feature_config.__dict__,
        "feature_name_groups": safe_feature_name_groups,
        "overlay_boundary_feature_names": OVERLAY_BOUNDARY_FEATURE_NAMES,
        "override_feature_names": override_feature_names,
        "include_override_overlay_features": include_overlay_for_override,
        "include_override_content_features": include_content_for_override,
        "dropped_feature_count": dropped_feature_count,
        "folds": folds,
        "base_model": base_name,
        "base_reports": base_reports,
        "candidate_reports_raw": candidate_reports_raw,
        "external_base_alignment": external_alignment,
        "internal_base_val_at_threshold": internal_base_val_at_threshold,
        "external_base_at_threshold": external_base_at_threshold,
        "train_exported_base_noise_summary": summarize_noise(train_y, train_base_exported),
        "val_exported_base_noise_summary": summarize_noise(val_y, val_base_exported),
        "candidate_reports": candidate_reports,
        "selected_by_val": selected,
        "reference": {
            "name": "Loop57 FN overlay gate",
            "val_f1": float(args.reference_val_f1),
            "val_errors": int(args.reference_val_errors),
            "min_val_error_improvement": int(args.min_val_error_improvement),
            "required_errors_for_test10k": int(required_errors),
        },
        "test_gate_decision": test_gate_decision,
        "artifacts": {
            "val_predictions": str(val_predictions_path),
            "selected_model": str(model_path),
        },
    }
    report_path = output_dir / "loop61_override_classifier_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected_by_val": selected, "test_gate_decision": test_gate_decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
