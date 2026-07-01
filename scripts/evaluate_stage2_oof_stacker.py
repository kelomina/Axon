#!/usr/bin/env python3
"""Evaluate a frozen Stage-2 OOF stacker payload."""

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
from train_stage2_cache_matrix import (  # noqa: E402
    FeatureConfig,
    build_matrix,
    metrics_at_threshold,
    predict_scores,
    read_prediction_rows,
    resolve_path,
    summarize_noise,
    write_predictions,
)
from train_stage2_oof_stacker import build_stack_features  # noqa: E402
from train_stage2_oof_stacker import STAGE2_PROB_FEATURE_COUNT  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen OOF stage-2 stacker.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-predictions-csv", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    with resolve_path(args.model).open("rb") as handle:
        payload = pickle.load(handle)

    feature_config = payload["feature_config"]
    if not isinstance(feature_config, FeatureConfig):
        feature_config = FeatureConfig(**dict(feature_config))
    checkpoint_config = AxonExperimentConfig.from_dict(dict(payload["checkpoint_config"]))
    threshold = float(args.threshold if args.threshold is not None else payload["threshold"])

    rows = read_prediction_rows(args.predictions, args.max_rows)
    matrix, labels, base_probs, kept_rows, counts = build_matrix(rows, checkpoint_config, feature_config)
    dropped_feature_count = int(payload.get("dropped_feature_count") or 0)
    if payload.get("drop_base_prob_features"):
        dropped_feature_count = STAGE2_PROB_FEATURE_COUNT
        matrix = matrix[:, STAGE2_PROB_FEATURE_COUNT:].astype(np.float32, copy=False)
    base_scores = np.column_stack([predict_scores(model, matrix) for model in payload["base_models"]]).astype(
        np.float32,
        copy=False,
    )
    stack_matrix, stack_feature_names = build_stack_features(base_scores)
    scores = predict_scores(payload["meta_model"], stack_matrix)
    metrics = metrics_at_threshold(scores, labels, threshold)
    output_predictions = resolve_path(args.output_predictions_csv)
    write_predictions(output_predictions, kept_rows, labels, scores, threshold)

    report = {
        "schema": "axon_stage2_oof_stacker_frozen_eval_v1",
        "protocol": "frozen OOF stacker only; no fitting and no threshold sweep",
        "model": str(resolve_path(args.model)),
        "predictions": str(resolve_path(args.predictions)),
        "output_predictions_csv": str(output_predictions),
        "threshold": threshold,
        "selected_from_val": payload.get("selected"),
        "feature_config": feature_config.__dict__,
        "drop_base_prob_features": bool(payload.get("drop_base_prob_features")),
        "dropped_feature_count": dropped_feature_count,
        "records": counts,
        "feature_dim": int(matrix.shape[1]),
        "base_specs": payload.get("base_specs"),
        "stack_feature_names": stack_feature_names,
        "metrics": metrics,
        "noise_summary": summarize_noise(labels, base_probs),
    }
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"JSON: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
