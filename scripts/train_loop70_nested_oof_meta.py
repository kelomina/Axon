#!/usr/bin/env python3
"""Train a Val-only meta layer from Loop69 nested OOF override scores.

Loop70 uses Loop69 train-only nested OOF predictions as the meta-model training
input, then builds frozen Val scores from models fitted on all train rows. It
does not touch Test-10k/full-test.
"""

from __future__ import annotations

import argparse
import csv
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
from train_loop42_oof_residual_gate import fit_with_optional_weights, prediction_metrics  # noqa: E402
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


LOOP57_VAL_ERRORS = 147


def _safe_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float32, copy=False), 1.0e-6, 1.0 - 1.0e-6)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32, copy=False)


def read_oof_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_array(rows: Sequence[dict], column: str) -> np.ndarray:
    return np.asarray([float(row[column]) for row in rows], dtype=np.float32)


def _int_array(rows: Sequence[dict], column: str) -> np.ndarray:
    return np.asarray([int(row[column]) for row in rows], dtype=np.int64)


def _bool_array(rows: Sequence[dict], column: str) -> np.ndarray:
    return np.asarray([str(row[column]).strip().lower() in {"1", "true", "yes"} for row in rows], dtype=bool)


def build_meta_score_features(
    *,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    allow_scores: np.ndarray,
    final_scores: np.ndarray,
    final_predictions: np.ndarray,
    override_mask: np.ndarray,
    possible_mask: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build deployable score-only meta features.

    Fold ids, row ids, paths, hashes, and labels are deliberately excluded.
    """

    base = np.clip(base_scores.astype(np.float32, copy=False), 1.0e-6, 1.0 - 1.0e-6)
    candidate = np.clip(candidate_scores.astype(np.float32, copy=False), 1.0e-6, 1.0 - 1.0e-6)
    allow = np.clip(allow_scores.astype(np.float32, copy=False), 1.0e-6, 1.0 - 1.0e-6)
    final = np.clip(final_scores.astype(np.float32, copy=False), 1.0e-6, 1.0 - 1.0e-6)
    base_logit = _safe_logit(base)
    candidate_logit = _safe_logit(candidate)
    allow_logit = _safe_logit(allow)
    final_logit = _safe_logit(final)
    matrix = np.column_stack(
        [
            base,
            candidate,
            allow,
            final,
            candidate - base,
            allow - base,
            final - base,
            np.abs(base - 0.5) * 2.0,
            np.abs(candidate - 0.5) * 2.0,
            np.abs(allow - 0.5) * 2.0,
            np.abs(final - 0.5) * 2.0,
            base_logit,
            candidate_logit,
            allow_logit,
            final_logit,
            candidate_logit - base_logit,
            allow_logit - base_logit,
            final_logit - base_logit,
            final_predictions.astype(np.float32, copy=False),
            override_mask.astype(np.float32, copy=False),
            possible_mask.astype(np.float32, copy=False),
        ]
    ).astype(np.float32, copy=False)
    names = [
        "meta_base_score",
        "meta_candidate_score",
        "meta_allow_score",
        "meta_final_score",
        "meta_candidate_minus_base",
        "meta_allow_minus_base",
        "meta_final_minus_base",
        "meta_base_confidence",
        "meta_candidate_confidence",
        "meta_allow_confidence",
        "meta_final_confidence",
        "meta_base_logit",
        "meta_candidate_logit",
        "meta_allow_logit",
        "meta_final_logit",
        "meta_candidate_logit_minus_base",
        "meta_allow_logit_minus_base",
        "meta_final_logit_minus_base",
        "meta_previous_prediction",
        "meta_previous_override_flag",
        "meta_possible_override_flag",
    ]
    assert_no_identity_feature_names(names, context="Loop70 meta score features")
    return matrix, names


def meta_model_candidates(seed: int) -> list[tuple[str, object]]:
    return [
        (
            "meta_logreg_balanced_c0.1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, solver="liblinear", C=0.1, class_weight="balanced", random_state=seed),
            ),
        ),
        (
            "meta_logreg_balanced_c1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, solver="liblinear", C=1.0, class_weight="balanced", random_state=seed),
            ),
        ),
        (
            "meta_hgb_leaf7",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_leaf_nodes=7,
                l2_regularization=1.0e-3,
                max_iter=120,
                random_state=seed,
            ),
        ),
        (
            "meta_hgb_leaf15",
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=1.0e-3,
                max_iter=140,
                random_state=seed,
            ),
        ),
    ]


def write_loop70_predictions(
    path: Path,
    rows: Sequence[dict],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    allow_scores: np.ndarray,
    previous_scores: np.ndarray,
    previous_predictions: np.ndarray,
    previous_override: np.ndarray,
    selected_meta_model: str,
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
        "allow_prob",
        "previous_final_prob_malicious",
        "previous_prediction",
        "previous_override",
        "loop70_prob_malicious",
        "prediction",
        "correct",
        "selected_meta_model",
    ]
    predictions = (scores >= threshold).astype(np.int64)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row, label, base, candidate, allow, prev_score, prev_pred, prev_override, score, pred in zip(
            rows,
            labels,
            base_scores,
            candidate_scores,
            allow_scores,
            previous_scores,
            previous_predictions,
            previous_override,
            scores,
            predictions,
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
                    "allow_prob": f"{float(allow):.10f}",
                    "previous_final_prob_malicious": f"{float(prev_score):.10f}",
                    "previous_prediction": int(prev_pred),
                    "previous_override": bool(prev_override),
                    "loop70_prob_malicious": f"{float(score):.10f}",
                    "prediction": int(pred),
                    "correct": int(pred) == int(label),
                    "selected_meta_model": selected_meta_model,
                }
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Loop70 nested-OOF meta layer.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--train-oof-predictions", type=Path, required=True)
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
    parser.add_argument("--candidate-model-candidate", default="extra_trees_300_leaf1")
    parser.add_argument("--override-model-candidate", default="override_logreg_balanced_c1")
    parser.add_argument("--meta-model-candidates", default="")
    parser.add_argument("--base-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=70)
    parser.add_argument("--reference-val-errors", type=int, default=LOOP57_VAL_ERRORS)
    parser.add_argument("--min-val-error-improvement", type=int, default=10)
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
    assert_no_identity_feature_names(OVERLAY_BOUNDARY_FEATURE_NAMES, context="Loop70 overlay boundary features")

    train_rows = read_prediction_rows(args.train_predictions, args.max_train_rows)
    val_rows = read_prediction_rows(args.val_predictions, args.max_val_rows)
    train_x, train_y, _train_exported, train_kept_rows, train_counts = build_matrix(
        train_rows,
        checkpoint_config,
        feature_config,
    )
    val_x, val_y, _val_exported, val_kept_rows, val_counts = build_matrix(
        val_rows,
        checkpoint_config,
        feature_config,
    )
    dropped_feature_count = 0
    if args.drop_base_prob_features:
        dropped_feature_count = 6
        train_x = train_x[:, dropped_feature_count:].astype(np.float32, copy=False)
        val_x = val_x[:, dropped_feature_count:].astype(np.float32, copy=False)

    oof_rows = read_oof_rows(args.train_oof_predictions)
    if args.max_train_rows is not None:
        oof_rows = oof_rows[: int(args.max_train_rows)]
    if len(oof_rows) != len(train_kept_rows):
        raise ValueError(f"OOF row count mismatch: {len(oof_rows)} != {len(train_kept_rows)}")
    oof_labels = _int_array(oof_rows, "label")
    if not np.array_equal(oof_labels, train_y):
        raise ValueError("OOF labels do not align with train rows")
    for left, right in zip(oof_rows, train_kept_rows):
        if str(left.get("sample_index", "")) != str(right.get("sample_index", "")):
            raise ValueError("OOF sample_index order does not align with train rows")

    train_base_oof = _float_array(oof_rows, "base_oof_prob_malicious")
    train_candidate_oof = _float_array(oof_rows, "candidate_oof_prob_malicious")
    train_allow_oof = _float_array(oof_rows, "allow_oof_prob")
    thresholds = parse_thresholds(args.thresholds)
    allow_thresholds = parse_thresholds(args.allow_thresholds)
    candidate_best = select_best_threshold(train_candidate_oof, train_y, thresholds)
    candidate_threshold = float(candidate_best["threshold"])
    allow_best = select_override_allow_threshold(
        labels=train_y,
        base_scores=train_base_oof,
        candidate_scores=train_candidate_oof,
        allow_scores=train_allow_oof,
        base_threshold=float(args.base_threshold),
        candidate_threshold=candidate_threshold,
        allow_thresholds=allow_thresholds,
    )
    allow_threshold = float(allow_best["allow_threshold"])
    train_prev_pred, train_prev_scores, train_override = override_classifier_predictions(
        base_scores=train_base_oof,
        candidate_scores=train_candidate_oof,
        allow_scores=train_allow_oof,
        base_threshold=float(args.base_threshold),
        candidate_threshold=candidate_threshold,
        allow_threshold=allow_threshold,
    )
    train_possible = possible_override_mask(
        base_scores=train_base_oof,
        candidate_scores=train_candidate_oof,
        base_threshold=float(args.base_threshold),
        candidate_threshold=candidate_threshold,
    )

    overlay_config = OverlayBoundaryConfig(cache_dir=str(resolve_path(args.overlay_boundary_cache_dir)))
    train_overlay = build_overlay_boundary_matrix(train_kept_rows, overlay_config)
    val_overlay = build_overlay_boundary_matrix(val_kept_rows, overlay_config)
    train_candidate_x = np.hstack([train_x, train_overlay]).astype(np.float32, copy=False)
    val_candidate_x = np.hstack([val_x, val_overlay]).astype(np.float32, copy=False)

    base_specs = filter_model_candidates(model_candidates(int(args.seed)), args.base_model_candidate)
    candidate_specs = filter_model_candidates(model_candidates(int(args.seed)), args.candidate_model_candidate)
    override_specs = _filter_override_candidates(override_model_candidates(int(args.seed)), args.override_model_candidate)
    if len(base_specs) != 1 or len(candidate_specs) != 1 or len(override_specs) != 1:
        raise ValueError("Loop70 expects exactly one base, candidate, and override spec")

    start = time.perf_counter()
    base_model = clone(base_specs[0][1])
    fit_with_optional_weights(base_model, train_x, train_y)
    candidate_model = clone(candidate_specs[0][1])
    fit_with_optional_weights(candidate_model, train_candidate_x, train_y)
    val_base_scores = predict_scores(base_model, val_x)
    val_candidate_scores = predict_scores(candidate_model, val_candidate_x)

    train_gate_x, gate_feature_names = build_fn_gate_matrix(
        train_x,
        train_overlay,
        train_base_oof,
        train_candidate_oof,
        include_overlay_features=not bool(args.no_override_overlay_features),
        include_content_features=bool(args.override_content_features),
    )
    val_gate_x, _ = build_fn_gate_matrix(
        val_x,
        val_overlay,
        val_base_scores,
        val_candidate_scores,
        include_overlay_features=not bool(args.no_override_overlay_features),
        include_content_features=bool(args.override_content_features),
    )
    assert_no_identity_feature_names(gate_feature_names, context="Loop70 override gate features")
    override_model = clone(override_specs[0][1])
    possible_train_x = train_gate_x[train_possible]
    possible_train_y = train_y[train_possible]
    if possible_train_x.shape[0] == 0 or len(np.unique(possible_train_y)) < 2:
        raise ValueError("Not enough two-class possible override rows for Loop70 override model")
    fit_with_optional_weights(override_model, possible_train_x, possible_train_y)
    val_allow_scores = predict_scores(override_model, val_gate_x)
    val_prev_pred, val_prev_scores, val_override = override_classifier_predictions(
        base_scores=val_base_scores,
        candidate_scores=val_candidate_scores,
        allow_scores=val_allow_scores,
        base_threshold=float(args.base_threshold),
        candidate_threshold=candidate_threshold,
        allow_threshold=allow_threshold,
    )
    val_possible = possible_override_mask(
        base_scores=val_base_scores,
        candidate_scores=val_candidate_scores,
        base_threshold=float(args.base_threshold),
        candidate_threshold=candidate_threshold,
    )
    upstream_fit_sec = time.perf_counter() - start

    train_meta_x, meta_feature_names = build_meta_score_features(
        base_scores=train_base_oof,
        candidate_scores=train_candidate_oof,
        allow_scores=train_allow_oof,
        final_scores=train_prev_scores,
        final_predictions=train_prev_pred,
        override_mask=train_override,
        possible_mask=train_possible,
    )
    val_meta_x, _ = build_meta_score_features(
        base_scores=val_base_scores,
        candidate_scores=val_candidate_scores,
        allow_scores=val_allow_scores,
        final_scores=val_prev_scores,
        final_predictions=val_prev_pred,
        override_mask=val_override,
        possible_mask=val_possible,
    )

    selected_meta_names = [name.strip() for name in args.meta_model_candidates.split(",") if name.strip()]
    meta_candidates = meta_model_candidates(int(args.seed))
    if selected_meta_names:
        meta_candidates = [(name, model) for name, model in meta_candidates if name in set(selected_meta_names)]
    if not meta_candidates:
        raise ValueError("No Loop70 meta candidates selected")

    fitted_results = []
    meta_reports = []
    for meta_name, prototype in meta_candidates:
        start = time.perf_counter()
        model = clone(prototype)
        model.fit(train_meta_x, train_y)
        train_scores = predict_scores(model, train_meta_x)
        val_scores = predict_scores(model, val_meta_x)
        train_best = select_best_threshold(train_scores, train_y, thresholds)
        val_best = select_best_threshold(val_scores, val_y, thresholds)
        row = {
            "name": meta_name,
            "fit_sec": time.perf_counter() - start,
            "train_oof_best": train_best,
            "val_best": val_best,
            "delta_val_errors_vs_reference": int(val_best["errors"]) - int(args.reference_val_errors),
        }
        meta_reports.append(row)
        fitted_results.append((float(val_best["f1"]), -int(val_best["errors"]), row, model, val_scores))
        print(
            f"[loop70-val] {meta_name} f1={val_best['f1']:.6f} errors={val_best['errors']} "
            f"threshold={val_best['threshold']:.4f}",
            flush=True,
        )

    fitted_results.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _best_f1, _neg_errors, selected, selected_model, selected_val_scores = fitted_results[0]
    selected_threshold = float(selected["val_best"]["threshold"])
    output_predictions = output_dir / "loop70_nested_oof_meta_val_predictions.csv"
    write_loop70_predictions(
        output_predictions,
        val_kept_rows,
        val_y,
        selected_val_scores,
        selected_threshold,
        base_scores=val_base_scores,
        candidate_scores=val_candidate_scores,
        allow_scores=val_allow_scores,
        previous_scores=val_prev_scores,
        previous_predictions=val_prev_pred,
        previous_override=val_override,
        selected_meta_model=str(selected["name"]),
    )

    required_errors = int(args.reference_val_errors) - int(args.min_val_error_improvement)
    selected_errors = int(selected["val_best"]["errors"])
    if int(val_counts.get("kept", 0)) < 20000:
        test_gate_decision = "smoke_only_not_eligible_for_test10k"
    elif selected_errors <= required_errors:
        test_gate_decision = "eligible_for_test10k_after_frozen_protocol_review"
    else:
        test_gate_decision = "reject_val_margin_too_small"

    model_path = output_dir / "loop70_nested_oof_meta_selected_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "schema": "axon_loop70_nested_oof_meta_payload_v1",
                "meta_model": selected_model,
                "threshold": selected_threshold,
                "candidate_threshold": candidate_threshold,
                "allow_threshold": allow_threshold,
                "meta_feature_names": meta_feature_names,
                "identity_feature_policy": "identity fields are alignment/audit only and are forbidden as model features",
                "selected": selected,
            },
            handle,
        )

    report = {
        "schema": "axon_loop70_nested_oof_meta_v1",
        "protocol": (
            "train meta model on Loop69 train nested OOF scores; fit upstream models on train only; "
            "Val selects meta model and threshold; no Test-10k/full-test used"
        ),
        "identity_feature_policy": (
            "filename/path/extension/directory/source hash/sample id/split/row order are alignment/cache/audit "
            "fields only and are not model features"
        ),
        "checkpoint": str(resolve_path(args.checkpoint)),
        "train_predictions": str(resolve_path(args.train_predictions)),
        "val_predictions": str(resolve_path(args.val_predictions)),
        "train_oof_predictions": str(resolve_path(args.train_oof_predictions)),
        "records": {"train": train_counts, "val": val_counts},
        "feature_config": feature_config.__dict__,
        "feature_name_groups": safe_feature_name_groups,
        "dropped_feature_count": dropped_feature_count,
        "base_model": base_specs[0][0],
        "candidate_model": candidate_specs[0][0],
        "override_model": override_specs[0][0],
        "upstream_fit_sec": upstream_fit_sec,
        "base_threshold": float(args.base_threshold),
        "candidate_threshold_from_train_oof": candidate_best,
        "allow_threshold_from_train_oof": allow_best,
        "previous_layer_train_oof_metrics": prediction_metrics(train_y, train_prev_pred, train_prev_scores),
        "previous_layer_val_metrics": prediction_metrics(val_y, val_prev_pred, val_prev_scores),
        "previous_layer_val_base_metrics": metrics_at_threshold(val_base_scores, val_y, float(args.base_threshold)),
        "meta_feature_names": meta_feature_names,
        "meta_models": sorted(meta_reports, key=lambda row: (row["val_best"]["f1"], -row["val_best"]["errors"]), reverse=True),
        "selected_by_val": selected,
        "reference": {
            "name": "Loop57 Val reference",
            "val_errors": int(args.reference_val_errors),
            "min_val_error_improvement": int(args.min_val_error_improvement),
            "required_errors_for_test10k": int(required_errors),
        },
        "test_gate_decision": test_gate_decision,
        "artifacts": {
            "val_predictions": str(output_predictions),
            "selected_model": str(model_path),
        },
    }
    report_path = output_dir / "loop70_nested_oof_meta_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected_by_val": selected, "test_gate_decision": test_gate_decision}, indent=2))
    print(f"JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
