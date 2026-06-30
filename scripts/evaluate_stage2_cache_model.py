#!/usr/bin/env python3
"""Evaluate a frozen stage-2 cache model on exported Axon predictions."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from config import AxonExperimentConfig  # noqa: E402
from train_stage2_cache_matrix import (  # noqa: E402
    FeatureConfig,
    append_frozen_knn_features,
    build_matrix,
    metrics_at_threshold,
    predict_scores,
    read_prediction_rows,
    resolve_path,
    summarize_noise,
    write_predictions,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen stage-2 model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-predictions-csv", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--knn-batch-size", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    with resolve_path(args.model).open("rb") as handle:
        payload = pickle.load(handle)

    model = payload["model"]
    feature_config = payload["feature_config"]
    if not isinstance(feature_config, FeatureConfig):
        feature_config = FeatureConfig(**dict(feature_config))
    checkpoint_config = AxonExperimentConfig.from_dict(dict(payload["checkpoint_config"]))
    threshold = float(args.threshold if args.threshold is not None else payload["threshold"])

    rows = read_prediction_rows(args.predictions, args.max_rows)
    matrix, labels, base_probs, kept_rows, counts = build_matrix(rows, checkpoint_config, feature_config)
    base_feature_dim = int(matrix.shape[1])
    knn_payload = payload.get("knn") or {}
    if knn_payload.get("enabled"):
        batch_size = int(args.knn_batch_size or knn_payload.get("batch_size") or 2048)
        matrix = append_frozen_knn_features(
            matrix,
            knn_payload["reference"],
            knn_payload["top_ks"],
            batch_size=batch_size,
        )
    scores = predict_scores(model, matrix)
    metrics = metrics_at_threshold(scores, labels, threshold)
    output_predictions = resolve_path(args.output_predictions_csv)
    write_predictions(output_predictions, kept_rows, labels, scores, threshold)

    report = {
        "schema": "axon_stage2_frozen_eval_v1",
        "protocol": "frozen stage2 model only; no fitting and no threshold sweep",
        "model": str(resolve_path(args.model)),
        "predictions": str(resolve_path(args.predictions)),
        "output_predictions_csv": str(output_predictions),
        "selected_from_val": payload.get("selected"),
        "threshold": threshold,
        "feature_config": feature_config.__dict__,
        "records": counts,
        "base_feature_dim": base_feature_dim,
        "feature_dim": int(matrix.shape[1]),
        "knn_config": {
            "enabled": bool(knn_payload.get("enabled")),
            "top_ks": knn_payload.get("top_ks"),
            "feature_names": knn_payload.get("feature_names"),
        },
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
