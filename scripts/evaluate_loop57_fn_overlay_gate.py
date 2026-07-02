#!/usr/bin/env python3
"""Evaluate a frozen Loop57 FN-specific overlay gate payload."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from config import AxonExperimentConfig  # noqa: E402
from train_loop42_oof_residual_gate import prediction_metrics  # noqa: E402
from train_loop55_overlay_boundary import OverlayBoundaryConfig, build_overlay_boundary_matrix  # noqa: E402
from train_loop57_fn_overlay_gate import (  # noqa: E402
    align_external_scores,
    build_fn_gate_matrix,
    fn_override_predictions,
    write_fn_gate_predictions,
)
from train_stage2_cache_matrix import (  # noqa: E402
    FeatureConfig,
    build_matrix,
    metrics_at_threshold,
    predict_scores,
    read_prediction_rows,
    resolve_path,
    summarize_noise,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Loop57 FN overlay gate.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--overlay-boundary-cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-predictions-csv", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--baseline-probability-column", default="stage2_prob_malicious")
    parser.add_argument("--alignment-key-column", default="sample_index")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    with resolve_path(args.model).open("rb") as handle:
        payload = pickle.load(handle)

    feature_config = payload["feature_config"]
    if not isinstance(feature_config, FeatureConfig):
        feature_config = FeatureConfig(**dict(feature_config))
    checkpoint_config = AxonExperimentConfig.from_dict(dict(payload["checkpoint_config"]))
    dropped_feature_count = int(payload.get("dropped_feature_count") or 0)
    base_threshold = float(payload["base_threshold"])
    candidate_threshold = float(payload["candidate_threshold"])
    gate_threshold = float(payload["gate_threshold"])

    rows = read_prediction_rows(args.predictions, args.max_rows)
    matrix, labels, base_probs_from_input, kept_rows, counts = build_matrix(rows, checkpoint_config, feature_config)
    if dropped_feature_count:
        matrix = matrix[:, dropped_feature_count:].astype(np.float32, copy=False)

    overlay_config = OverlayBoundaryConfig(cache_dir=str(resolve_path(args.overlay_boundary_cache_dir)))
    overlay = build_overlay_boundary_matrix(kept_rows, overlay_config)
    candidate_matrix = np.hstack([matrix, overlay]).astype(np.float32, copy=False)
    candidate_scores = predict_scores(payload["candidate_model"], candidate_matrix)
    base_scores, external_alignment = align_external_scores(
        rows=kept_rows,
        prediction_path=args.baseline_predictions,
        probability_column=args.baseline_probability_column,
        key_column=args.alignment_key_column,
    )
    gate_matrix, gate_feature_names = build_fn_gate_matrix(
        overlay,
        base_scores,
        candidate_scores,
        include_overlay_features=True,
    )
    gate_scores = predict_scores(payload["gate_model"], gate_matrix)
    final_predictions, final_scores, override_mask = fn_override_predictions(
        base_scores=base_scores,
        candidate_scores=candidate_scores,
        gate_scores=gate_scores,
        base_threshold=base_threshold,
        candidate_threshold=candidate_threshold,
        gate_threshold=gate_threshold,
    )
    direct_metrics = prediction_metrics(labels, final_predictions, final_scores)
    direct_metrics["threshold"] = base_threshold
    base_metrics = metrics_at_threshold(base_scores, labels, base_threshold)
    candidate_metrics = metrics_at_threshold(candidate_scores, labels, candidate_threshold)

    output_predictions = resolve_path(args.output_predictions_csv)
    write_fn_gate_predictions(
        output_predictions,
        kept_rows,
        labels,
        base_scores=base_scores,
        candidate_scores=candidate_scores,
        gate_scores=gate_scores,
        final_scores=final_scores,
        final_predictions=final_predictions,
        override_mask=override_mask,
        selected_candidate=str((payload.get("selected") or {}).get("candidate", "")),
    )

    report = {
        "schema": "axon_loop57_fn_overlay_gate_frozen_eval_v1",
        "protocol": "frozen Loop57 payload only; no fitting and no threshold sweep",
        "identity_feature_policy": (
            "filename/path/extension/directory/source hash/sample id/split/row order are "
            "alignment/cache/audit fields only and are not model features"
        ),
        "model": str(resolve_path(args.model)),
        "predictions": str(resolve_path(args.predictions)),
        "baseline_predictions": str(resolve_path(args.baseline_predictions)),
        "output_predictions_csv": str(output_predictions),
        "selected_from_val": payload.get("selected"),
        "base_threshold": base_threshold,
        "candidate_threshold": candidate_threshold,
        "gate_threshold": gate_threshold,
        "feature_config": feature_config.__dict__,
        "dropped_feature_count": dropped_feature_count,
        "records": counts,
        "feature_dim": int(matrix.shape[1]),
        "candidate_feature_dim": int(candidate_matrix.shape[1]),
        "gate_feature_dim": int(gate_matrix.shape[1]),
        "gate_feature_names": gate_feature_names,
        "external_base_alignment": external_alignment,
        "base_metrics": base_metrics,
        "candidate_metrics": candidate_metrics,
        "metrics": direct_metrics,
        "override_summary": {
            "override_count": int(override_mask.sum()),
            "override_ratio": float(override_mask.mean()),
            "override_label1_count": int(((labels == 1) & override_mask).sum()),
            "override_label0_count": int(((labels == 0) & override_mask).sum()),
        },
        "noise_summary": summarize_noise(labels, base_probs_from_input),
    }
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"JSON: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
