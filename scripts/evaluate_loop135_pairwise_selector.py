#!/usr/bin/env python3
"""Evaluate a frozen Loop135 pairwise selector."""

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
for item in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from train_loop135_pairwise_selector import (  # noqa: E402
    align_pair,
    apply_selector,
    apply_selector_directional,
    build_eval_support_feature_block,
    build_selector_features,
    disagreement_mask,
    metrics,
    predict_scores,
    resolve_path,
    write_predictions,
)
from train_stage2_cache_matrix import FeatureConfig  # noqa: E402,F401


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen Loop135 pairwise selector.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-predictions-csv", type=Path, required=True)
    parser.add_argument("--content-pe-cache-dir", type=Path, default=None)
    parser.add_argument("--content-pe-v2-cache-dir", type=Path, default=None)
    parser.add_argument("--content-string-cache-dir", type=Path, default=None)
    parser.add_argument("--support-predictions", type=Path, default=None)
    parser.add_argument("--support-knn-batch-size", type=int, default=None)
    parser.add_argument("--support-knn-similarity-memory-mib", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-0to1", type=float, default=None)
    parser.add_argument("--threshold-1to0", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    model_path = resolve_path(args.model)
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    key_columns = tuple(payload["key_columns"])
    pair = align_pair(
        args.baseline_predictions,
        args.candidate_predictions,
        key_columns=key_columns,
        baseline_score_column=payload["baseline_score_column"],
        candidate_score_column=payload["candidate_score_column"],
    )
    diff_indices = np.flatnonzero(disagreement_mask(pair))
    support_matrix = None
    support_names = []
    eval_support_info = None
    support_info = payload.get("support_info")
    if support_info is not None:
        if args.support_predictions is None:
            raise ValueError("--support-predictions is required for support-aware selector payloads")
        support_matrix, support_names, eval_support_info = build_eval_support_feature_block(
            support_stage2_model=Path(support_info["support_stage2_model"]),
            support_predictions=args.support_predictions,
            pair=pair,
            support_key_columns=tuple(support_info.get("support_key_columns") or key_columns),
            top_ks=[int(item) for item in support_info["top_ks"]],
            batch_size=int(
                args.support_knn_batch_size
                if args.support_knn_batch_size is not None
                else support_info.get("batch_size", 256)
            ),
            similarity_memory_mib=float(
                args.support_knn_similarity_memory_mib
                if args.support_knn_similarity_memory_mib is not None
                else support_info.get("similarity_memory_mib", 128.0)
            ),
        )
    matrix, feature_names = build_selector_features(
        pair,
        diff_indices,
        content_pe_cache_dir=args.content_pe_cache_dir,
        content_pe_v2_cache_dir=args.content_pe_v2_cache_dir,
        content_string_cache_dir=args.content_string_cache_dir,
        support_matrix=support_matrix,
        support_names=support_names,
    )
    if list(feature_names) != list(payload["feature_names"]):
        raise ValueError("Loop135 selector feature names do not match frozen payload")
    diff_scores = predict_scores(payload["model"], matrix) if diff_indices.size else np.asarray([], dtype=np.float32)
    selected = payload["selected"]
    threshold_mode = str(payload.get("threshold_mode") or selected.get("threshold_mode") or "global")
    full_scores = np.zeros(pair.labels.shape[0], dtype=np.float32)
    full_scores[diff_indices] = diff_scores
    if threshold_mode == "directional":
        thresholds_by_direction = selected.get("thresholds_by_direction") or {}
        default_0to1 = selected.get("threshold_0to1", thresholds_by_direction.get("baseline0_candidate1"))
        default_1to0 = selected.get("threshold_1to0", thresholds_by_direction.get("baseline1_candidate0"))
        if default_0to1 is None or default_1to0 is None:
            raise ValueError("Directional selector payload is missing direction thresholds")
        global_override = args.threshold
        threshold_0to1 = float(
            args.threshold_0to1
            if args.threshold_0to1 is not None
            else global_override
            if global_override is not None
            else default_0to1
        )
        threshold_1to0 = float(
            args.threshold_1to0
            if args.threshold_1to0 is not None
            else global_override
            if global_override is not None
            else default_1to0
        )
        threshold = None
        predictions, accept_candidate = apply_selector_directional(
            pair,
            diff_indices,
            diff_scores,
            threshold_0to1,
            threshold_1to0,
        )
    else:
        threshold = float(args.threshold if args.threshold is not None else selected["threshold"])
        threshold_0to1 = threshold
        threshold_1to0 = threshold
        predictions, accept_candidate = apply_selector(pair, diff_indices, diff_scores, threshold)
    output_predictions = resolve_path(args.output_predictions_csv)
    write_predictions(output_predictions, pair, full_scores, predictions, accept_candidate)
    eval_metrics = metrics(pair.labels, predictions)
    report = {
        "schema": "axon_loop135_pairwise_selector_frozen_eval_v1",
        "protocol": "frozen pairwise selector only; no fitting and no threshold sweep",
        "model": str(model_path),
        "baseline_predictions": str(resolve_path(args.baseline_predictions)),
        "candidate_predictions": str(resolve_path(args.candidate_predictions)),
        "output_predictions_csv": str(output_predictions),
        "selected_from_val": selected,
        "support_info": support_info,
        "eval_support_info": eval_support_info,
        "threshold_mode": threshold_mode,
        "threshold": threshold,
        "threshold_0to1": threshold_0to1,
        "threshold_1to0": threshold_1to0,
        "thresholds_by_direction": {
            "baseline0_candidate1": threshold_0to1,
            "baseline1_candidate0": threshold_1to0,
        },
        "records": {"total": int(pair.labels.shape[0]), "kept": int(pair.labels.shape[0])},
        "disagreements": int(diff_indices.size),
        "accepted_candidate": int(np.count_nonzero(accept_candidate)),
        "accepted_candidate_label0": int(np.count_nonzero(accept_candidate & (pair.labels == 0))),
        "accepted_candidate_label1": int(np.count_nonzero(accept_candidate & (pair.labels == 1))),
        "baseline_metrics": metrics(pair.labels, pair.baseline_pred),
        "candidate_metrics": metrics(pair.labels, pair.candidate_pred),
        "metrics": eval_metrics,
    }
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"JSON: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
