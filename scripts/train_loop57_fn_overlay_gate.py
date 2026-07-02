#!/usr/bin/env python3
"""Train a strict FN-specific overlay residual gate.

Loop57 only allows 0->1 overrides: if the locked/base model predicts benign,
an overlay-aware candidate may flip the sample to malicious when the gate is
confident. Identity fields are alignment/cache fields only, never features.
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
    build_gate_score_features,
    fit_with_optional_weights,
    gate_model_candidates,
    oof_stage2_scores,
    prediction_metrics,
)
from train_loop55_overlay_boundary import (  # noqa: E402
    OVERLAY_BOUNDARY_FEATURE_NAMES,
    OverlayBoundaryConfig,
    build_overlay_boundary_matrix,
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


LOOP28_VAL_F1 = 0.9919048570857486
LOOP28_VAL_ERRORS = 162
LOOP57_TEST10K_ERROR_GATE = 152


def build_fn_gate_matrix(
    overlay_matrix: np.ndarray,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    include_overlay_features: bool,
) -> tuple[np.ndarray, list[str]]:
    """Build non-identity gate features for FN-specific override decisions."""

    score_features, score_names = build_gate_score_features(base_scores, candidate_scores)
    if not include_overlay_features:
        return score_features, score_names
    overlay_names = [f"gate_{name}" for name in OVERLAY_BOUNDARY_FEATURE_NAMES]
    assert_no_identity_feature_names(overlay_names, context="Loop57 gate overlay feature aliases")
    matrix = np.hstack([score_features, overlay_matrix.astype(np.float32, copy=False)])
    return matrix.astype(np.float32, copy=False), score_names + overlay_names


def fn_override_predictions(
    *,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    gate_scores: np.ndarray,
    base_threshold: float,
    candidate_threshold: float,
    gate_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply only benign->malicious overrides.

    The candidate cannot change a base malicious prediction to benign.
    """

    base_pred = (base_scores >= base_threshold).astype(np.int64)
    candidate_pred = (candidate_scores >= candidate_threshold).astype(np.int64)
    allowed = (base_pred == 0) & (candidate_pred == 1)
    override = allowed & (gate_scores >= gate_threshold)
    final_pred = np.where(override, 1, base_pred).astype(np.int64)
    final_scores = np.where(override, candidate_scores, base_scores).astype(np.float32, copy=False)
    return final_pred, final_scores, override


def fn_gate_training_targets(
    labels: np.ndarray,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    base_threshold: float,
    candidate_threshold: float,
    neutral_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create targets for the 0->1 override gate.

    Beneficial rows are base FN repaired by candidate. Harmful rows are base TN
    that candidate would turn into FP. Other rows are neutral and get a small
    weight only to stabilize score calibration.
    """

    base_pred = (base_scores >= base_threshold).astype(np.int64)
    candidate_pred = (candidate_scores >= candidate_threshold).astype(np.int64)
    possible = (base_pred == 0) & (candidate_pred == 1)
    beneficial = possible & (labels == 1)
    harmful = possible & (labels == 0)
    neutral = ~possible
    targets = beneficial.astype(np.int64)
    weights = np.full(labels.shape[0], float(neutral_weight), dtype=np.float32)
    weights[possible] = 1.0
    summary = {
        "possible_overrides": int(possible.sum()),
        "beneficial_fn_repairs": int(beneficial.sum()),
        "harmful_new_fp": int(harmful.sum()),
        "neutral_rows": int(neutral.sum()),
        "base_predicted_benign": int((base_pred == 0).sum()),
        "candidate_predicted_malicious": int((candidate_pred == 1).sum()),
        "base_errors": int((base_pred != labels).sum()),
        "candidate_errors": int((candidate_pred != labels).sum()),
        "weighted_rows": float(weights.sum()),
    }
    return targets, weights, summary


def select_fn_gate_threshold(
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
        predictions, final_scores, override = fn_override_predictions(
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
        metrics["override_label1_count"] = int(((labels == 1) & override).sum())
        metrics["override_label0_count"] = int(((labels == 0) & override).sum())
        rows.append(metrics)
    rows.sort(key=lambda row: (row["f1"], -row["errors"], -row["override_count"], row["gate_threshold"]), reverse=True)
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
    with resolve_path(prediction_path).open("r", encoding="utf-8-sig", newline="") as handle:
        external_rows = list(csv.DictReader(handle))
    by_key = {_prediction_key(row, key_column): row for row in external_rows}
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
        "external_rows": len(external_rows),
        "sha_checked": sha_checked,
    }


def write_fn_gate_predictions(
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
        "fn_override",
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
                    "fn_override": bool(override),
                    "selected_candidate": selected_candidate,
                }
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Loop57 FN-specific overlay residual gate.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--baseline-val-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--thresholds", default="0.35:0.65:0.005")
    parser.add_argument("--gate-thresholds", default="0.50:0.99:0.005")
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--chunk-count", type=int, default=16)
    parser.add_argument("--feature-set", choices=["tabular", "extended"], default="extended")
    parser.add_argument("--content-pe-cache-dir", type=Path, required=True)
    parser.add_argument("--overlay-boundary-cache-dir", type=Path, required=True)
    parser.add_argument("--drop-base-prob-features", action="store_true")
    parser.add_argument("--no-gate-overlay-features", action="store_true")
    parser.add_argument("--base-model-candidate", default="hgb_lr0.06_leaf31_l2_0")
    parser.add_argument(
        "--candidate-model-candidates",
        default="hgb_lr0.06_leaf31_l2_0,hgb_lr0.08_leaf31_l2_1e-3,extra_trees_300_leaf1",
    )
    parser.add_argument("--gate-model-candidates", default="")
    parser.add_argument("--neutral-weight", type=float, default=0.02)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--baseline-probability-column", default="stage2_prob_malicious")
    parser.add_argument("--alignment-key-column", default="sample_index")
    parser.add_argument("--base-threshold", type=float, default=0.5)
    parser.add_argument("--baseline-val-errors", type=int, default=LOOP28_VAL_ERRORS)
    parser.add_argument("--baseline-val-f1", type=float, default=LOOP28_VAL_F1)
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
    assert_no_identity_feature_names(OVERLAY_BOUNDARY_FEATURE_NAMES, context="Loop57 overlay boundary features")

    train_rows = read_prediction_rows(args.train_predictions, args.max_train_rows)
    val_rows = read_prediction_rows(args.val_predictions, args.max_val_rows)
    train_x, train_y, _train_base_exported, train_kept_rows, train_counts = build_matrix(
        train_rows, checkpoint_config, feature_config
    )
    val_x, val_y, _val_base_exported, val_kept_rows, val_counts = build_matrix(
        val_rows, checkpoint_config, feature_config
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
    gate_thresholds = parse_thresholds(args.gate_thresholds)
    folds = min(max(2, int(args.folds)), int(np.bincount(train_y).min()))

    base_specs = filter_model_candidates(model_candidates(int(args.seed)), args.base_model_candidate)
    if len(base_specs) != 1:
        raise ValueError(f"Expected exactly one base model candidate, got {[name for name, _ in base_specs]}")
    candidate_specs = filter_model_candidates(model_candidates(int(args.seed)), args.candidate_model_candidates)
    if not candidate_specs:
        raise ValueError("No candidate override models selected")

    base_oof, base_val_internal, fitted_base_models, base_reports = oof_stage2_scores(
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

    selected_gate_candidates = gate_model_candidates(int(args.seed))
    selected_gate_names = [item.strip() for item in args.gate_model_candidates.split(",") if item.strip()]
    if selected_gate_names:
        selected_gate_candidates = [
            (name, model) for name, model in selected_gate_candidates if name in set(selected_gate_names)
        ]
    if not selected_gate_candidates:
        raise ValueError("No gate model candidates selected")

    fitted_results = []
    candidate_reports = []
    include_overlay_for_gate = not bool(args.no_gate_overlay_features)
    for candidate_index, (candidate_name, _prototype) in enumerate(candidate_specs):
        candidate_train_scores = candidate_oof[:, candidate_index]
        candidate_val_scores = candidate_val[:, candidate_index]
        candidate_train_best = select_best_threshold(candidate_train_scores, train_y, thresholds)
        candidate_val_at_train_threshold = metrics_at_threshold(
            candidate_val_scores,
            val_y,
            float(candidate_train_best["threshold"]),
        )
        targets, weights, target_summary = fn_gate_training_targets(
            train_y,
            base_train_scores,
            candidate_train_scores,
            base_threshold=base_threshold,
            candidate_threshold=float(candidate_train_best["threshold"]),
            neutral_weight=float(args.neutral_weight),
        )
        if int(targets.sum()) == 0:
            print(f"[gate-skip] {candidate_name}: no beneficial FN repairs in train OOF", flush=True)
            continue
        train_gate_x, gate_feature_names = build_fn_gate_matrix(
            train_overlay,
            base_train_scores,
            candidate_train_scores,
            include_overlay_features=include_overlay_for_gate,
        )
        val_gate_x, _ = build_fn_gate_matrix(
            val_overlay,
            external_base_scores,
            candidate_val_scores,
            include_overlay_features=include_overlay_for_gate,
        )
        gate_model_reports = []
        for gate_name, gate_prototype in selected_gate_candidates:
            start = time.perf_counter()
            gate_model = clone(gate_prototype)
            fit_with_optional_weights(gate_model, train_gate_x, targets, weights)
            gate_train_scores = predict_scores(gate_model, train_gate_x)
            gate_val_scores = predict_scores(gate_model, val_gate_x)
            train_gate_best = select_fn_gate_threshold(
                labels=train_y,
                base_scores=base_train_scores,
                candidate_scores=candidate_train_scores,
                gate_scores=gate_train_scores,
                base_threshold=base_threshold,
                candidate_threshold=float(candidate_train_best["threshold"]),
                gate_thresholds=gate_thresholds,
            )
            val_gate_best = select_fn_gate_threshold(
                labels=val_y,
                base_scores=external_base_scores,
                candidate_scores=candidate_val_scores,
                gate_scores=gate_val_scores,
                base_threshold=base_threshold,
                candidate_threshold=float(candidate_train_best["threshold"]),
                gate_thresholds=gate_thresholds,
            )
            report_row = {
                "candidate": candidate_name,
                "gate_model": gate_name,
                "fit_sec": time.perf_counter() - start,
                "base_name": base_name,
                "base_threshold": base_threshold,
                "candidate_train_threshold": float(candidate_train_best["threshold"]),
                "target_summary": target_summary,
                "candidate_val_at_train_threshold": candidate_val_at_train_threshold,
                "train_gate_best": train_gate_best,
                "val_gate_best": val_gate_best,
                "delta_val_errors_vs_external_base": int(val_gate_best["errors"])
                - int(external_base_at_threshold["errors"]),
                "delta_val_errors_vs_loop28_locked": int(val_gate_best["errors"]) - int(args.baseline_val_errors),
                "delta_val_f1_vs_loop28_locked": float(val_gate_best["f1"]) - float(args.baseline_val_f1),
            }
            gate_model_reports.append(report_row)
            fitted_results.append(
                (
                    float(val_gate_best["f1"]),
                    -int(val_gate_best["errors"]),
                    report_row,
                    gate_model,
                    gate_val_scores,
                    candidate_index,
                    candidate_val_scores,
                    gate_feature_names,
                )
            )
            print(
                f"[fn-gate-val] candidate={candidate_name} gate={gate_name} "
                f"f1={val_gate_best['f1']:.6f} errors={val_gate_best['errors']} "
                f"overrides={val_gate_best['override_count']} "
                f"label1={val_gate_best['override_label1_count']} label0={val_gate_best['override_label0_count']}",
                flush=True,
            )
        candidate_reports.append(
            {
                "candidate": candidate_name,
                "candidate_train_best": candidate_train_best,
                "candidate_val_at_train_threshold": candidate_val_at_train_threshold,
                "gate_models": gate_model_reports,
            }
        )

    if not fitted_results:
        raise ValueError("No fitted gate results were produced")
    fitted_results.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _best_f1, _neg_errors, selected, selected_gate_model, selected_gate_val_scores, selected_candidate_index, selected_candidate_val_scores, gate_feature_names = fitted_results[0]
    selected_candidate_name = selected["candidate"]
    selected_candidate_threshold = float(selected["candidate_train_threshold"])
    selected_gate_threshold = float(selected["val_gate_best"]["gate_threshold"])
    final_predictions, final_scores, override_mask = fn_override_predictions(
        base_scores=external_base_scores,
        candidate_scores=selected_candidate_val_scores,
        gate_scores=selected_gate_val_scores,
        base_threshold=base_threshold,
        candidate_threshold=selected_candidate_threshold,
        gate_threshold=selected_gate_threshold,
    )
    val_predictions_path = output_dir / "loop57_fn_overlay_gate_val_predictions.csv"
    write_fn_gate_predictions(
        val_predictions_path,
        val_kept_rows,
        val_y,
        base_scores=external_base_scores,
        candidate_scores=selected_candidate_val_scores,
        gate_scores=selected_gate_val_scores,
        final_scores=final_scores,
        final_predictions=final_predictions,
        override_mask=override_mask,
        selected_candidate=selected_candidate_name,
    )

    model_path = output_dir / "loop57_fn_overlay_gate_selected_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "schema": "axon_loop57_fn_overlay_gate_payload_v1",
                "gate_model": selected_gate_model,
                "candidate_model": fitted_candidate_models[selected_candidate_index],
                "selected": selected,
                "base_threshold": base_threshold,
                "candidate_threshold": selected_candidate_threshold,
                "gate_threshold": selected_gate_threshold,
                "feature_config": feature_config,
                "checkpoint_config": checkpoint_config.to_dict(),
                "dropped_feature_count": dropped_feature_count,
                "gate_feature_names": gate_feature_names,
                "overlay_boundary_feature_names": OVERLAY_BOUNDARY_FEATURE_NAMES,
                "identity_feature_policy": (
                    "source_path/source_sha256/cache_path/sample_index/split/filename/extension/directory "
                    "are alignment or loading fields only and are forbidden as model features"
                ),
            },
            handle,
        )

    val_kept_count = int(val_counts.get("kept", 0)) if isinstance(val_counts, dict) else int(len(val_y))
    selected_errors = int(selected["val_gate_best"]["errors"])
    if val_kept_count < 20000:
        test_gate_decision = "smoke_only_not_eligible_for_test10k"
    elif selected_errors <= LOOP57_TEST10K_ERROR_GATE:
        test_gate_decision = "eligible_for_test10k"
    else:
        test_gate_decision = "reject_val_margin_too_small"

    report = {
        "schema": "axon_loop57_fn_overlay_gate_v1",
        "protocol": (
            "base/candidate train scores are strict OOF; gate only permits 0->1 overrides; "
            "Val selects gate model and threshold; no Test-10k/full-test used"
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
        "gate_feature_names": gate_feature_names,
        "dropped_feature_count": dropped_feature_count,
        "folds": folds,
        "base_model": base_name,
        "base_reports": base_reports,
        "candidate_reports_raw": candidate_reports_raw,
        "external_base_alignment": external_alignment,
        "internal_base_val_at_threshold": internal_base_val_at_threshold,
        "external_base_at_threshold": external_base_at_threshold,
        "candidate_reports": candidate_reports,
        "selected_by_val": selected,
        "test_gate_error_threshold": LOOP57_TEST10K_ERROR_GATE,
        "test_gate_decision": test_gate_decision,
        "artifacts": {
            "val_predictions": str(val_predictions_path),
            "selected_model": str(model_path),
        },
    }
    report_path = output_dir / "loop57_fn_overlay_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected_by_val": selected, "test_gate_decision": test_gate_decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
