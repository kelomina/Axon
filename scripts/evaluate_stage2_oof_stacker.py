#!/usr/bin/env python3
"""Evaluate a frozen Stage-2 OOF stacker on exported Axon predictions."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import replace
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
from train_stage2_oof_stacker import build_stack_features, drop_stage2_probability_features  # noqa: E402


def _override_feature_config(
    feature_config: FeatureConfig,
    *,
    content_pe_cache_dir: Optional[Path],
    content_pe_v2_cache_dir: Optional[Path],
    content_string_cache_dir: Optional[Path],
    content_cert_cache_dir: Optional[Path],
) -> FeatureConfig:
    updates = {}
    if content_pe_cache_dir is not None:
        updates["content_cache_dir"] = str(resolve_path(content_pe_cache_dir))
    if content_pe_v2_cache_dir is not None:
        updates["content_pe_v2_cache_dir"] = str(resolve_path(content_pe_v2_cache_dir))
    if content_string_cache_dir is not None:
        updates["content_string_cache_dir"] = str(resolve_path(content_string_cache_dir))
    if content_cert_cache_dir is not None:
        updates["content_cert_cache_dir"] = str(resolve_path(content_cert_cache_dir))
    return replace(feature_config, **updates) if updates else feature_config


def score_oof_payload(payload: dict, matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    """Score rows with frozen base models and the frozen meta model."""

    scoring_matrix = matrix
    dropped_feature_count = 0
    if bool(payload.get("drop_base_prob_features", False)):
        scoring_matrix = drop_stage2_probability_features(scoring_matrix)
        dropped_feature_count = int(matrix.shape[1] - scoring_matrix.shape[1])

    base_models = list(payload["base_models"])
    if not base_models:
        raise ValueError("OOF stacker payload has no base models")
    base_scores = np.zeros((scoring_matrix.shape[0], len(base_models)), dtype=np.float32)
    for index, base_model in enumerate(base_models):
        base_scores[:, index] = predict_scores(base_model, scoring_matrix)

    stack_matrix, stack_feature_names = build_stack_features(base_scores)
    meta_scores = predict_scores(payload["meta_model"], stack_matrix)
    info = {
        "input_feature_dim": int(matrix.shape[1]),
        "scoring_feature_dim": int(scoring_matrix.shape[1]),
        "dropped_feature_count": dropped_feature_count,
        "base_model_count": len(base_models),
        "stack_feature_dim": int(stack_matrix.shape[1]),
        "stack_feature_names": stack_feature_names,
    }
    return meta_scores.astype(np.float32, copy=False), info


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Stage-2 OOF stacker.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-predictions-csv", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--content-pe-cache-dir", type=Path, default=None)
    parser.add_argument("--content-pe-v2-cache-dir", type=Path, default=None)
    parser.add_argument("--content-string-cache-dir", type=Path, default=None)
    parser.add_argument("--content-cert-cache-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    model_path = resolve_path(args.model)
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)

    feature_config = payload["feature_config"]
    if not isinstance(feature_config, FeatureConfig):
        feature_config = FeatureConfig(**dict(feature_config))
    feature_config = _override_feature_config(
        feature_config,
        content_pe_cache_dir=args.content_pe_cache_dir,
        content_pe_v2_cache_dir=args.content_pe_v2_cache_dir,
        content_string_cache_dir=args.content_string_cache_dir,
        content_cert_cache_dir=args.content_cert_cache_dir,
    )
    checkpoint_config = AxonExperimentConfig.from_dict(dict(payload["checkpoint_config"]))
    threshold = float(args.threshold if args.threshold is not None else payload["threshold"])

    rows = read_prediction_rows(args.predictions, args.max_rows)
    matrix, labels, base_probs, kept_rows, counts = build_matrix(rows, checkpoint_config, feature_config)
    scores, scoring_info = score_oof_payload(payload, matrix)
    eval_metrics = metrics_at_threshold(scores, labels, threshold)

    output_predictions = resolve_path(args.output_predictions_csv)
    write_predictions(output_predictions, kept_rows, labels, scores, threshold)
    report = {
        "schema": "axon_stage2_oof_stacker_frozen_eval_v1",
        "protocol": "frozen Stage-2 OOF stacker only; no fitting and no threshold sweep",
        "model": str(model_path),
        "predictions": str(resolve_path(args.predictions)),
        "output_predictions_csv": str(output_predictions),
        "selected_from_val": payload.get("selected"),
        "threshold": threshold,
        "feature_config": feature_config.__dict__,
        "records": counts,
        "scoring_info": scoring_info,
        "metrics": eval_metrics,
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
