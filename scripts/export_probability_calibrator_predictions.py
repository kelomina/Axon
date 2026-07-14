#!/usr/bin/env python3
"""Export per-sample predictions from a trained probability calibrator."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from evaluate_probability_calibrator import _load_prediction_features


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export calibrator probabilities for an existing prediction CSV.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--missing-cache-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with args.model.open("rb") as handle:
        payload = pickle.load(handle)

    model = payload["model"]
    blend_weight = float(payload["blend_model_weight"])
    threshold = float(args.threshold if args.threshold is not None else payload["val_selected_threshold"])
    features, labels, baseline_probs, kept_rows, _counts = _load_prediction_features(
        args.predictions,
        missing_cache_output=args.missing_cache_output,
    )
    calibrator_probs = model.predict_proba(features)[:, 1]
    blended_probs = blend_weight * calibrator_probs + (1.0 - blend_weight) * baseline_probs

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "original_source_path",
        "cache_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "baseline_prob_malicious",
        "prob_malicious",
        "prediction",
        "correct",
    ]
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row, label, baseline_prob, score in zip(kept_rows, labels, baseline_probs, blended_probs):
            prediction = int(float(score) >= threshold)
            output = dict(row)
            output["baseline_prob_malicious"] = f"{float(baseline_prob):.10f}"
            output["prob_malicious"] = f"{float(score):.10f}"
            output["prediction"] = prediction
            output["correct"] = prediction == int(label)
            writer.writerow(output)

    print(f"Exported calibrator predictions: {args.output_csv}")
    print(f"Rows: {len(kept_rows)} threshold={threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
