#!/usr/bin/env python3
"""Evaluate a frozen stage-2 cache model on exported Axon predictions."""

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
    CONTENT_PE_FEATURE_NAMES,
    CONTENT_PE_V2_FEATURE_NAMES,
    FeatureConfig,
    append_feature_columns,
    append_frozen_knn_features,
    build_matrix,
    content_cache_path_for_row,
    load_stage2_knn_reference_from_payload,
    load_valid_feature_npz,
    metrics_at_threshold,
    predict_scores,
    read_prediction_rows,
    resolve_path,
    summarize_noise,
    write_predictions,
)
from train_loop43_content_cross import (  # noqa: E402
    CONTENT_CROSS_FEATURE_NAMES,
    CrossConfig,
    build_content_cross_matrix,
    content_cross_features_from_arrays,
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
    parser.add_argument("--content-pe-cache-dir", type=Path, default=None)
    parser.add_argument("--content-pe-v2-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-content-sidecar-build",
        action="store_true",
        help=(
            "Allow missing Loop43 content sidecars to be built during eval. "
            "Default is cache-only frozen evaluation."
        ),
    )
    parser.add_argument(
        "--sidecar-progress-interval",
        type=int,
        default=5000,
        help="Print cache-only sidecar progress every N rows; set <=0 to disable.",
    )
    parser.add_argument(
        "--eval-chunk-size",
        type=int,
        default=50000,
        help="Rows per prediction chunk; set <=0 to predict in one call.",
    )
    return parser.parse_args(argv)


def _load_required_sidecar_features(row: dict, cache_dir: str | Path, expected_dim: int, name: str) -> np.ndarray:
    cache_path = content_cache_path_for_row(row, cache_dir)
    if cache_path is None:
        raise ValueError(f"{name}_cache_dir_missing")
    if not cache_path.is_file():
        sha = str(row.get("source_sha256") or "")
        raise FileNotFoundError(f"{name}_sidecar_missing: sha={sha} path={cache_path}")
    features = load_valid_feature_npz(cache_path, expected_dim)
    if features is None:
        sha = str(row.get("source_sha256") or "")
        raise ValueError(f"{name}_sidecar_invalid: sha={sha} path={cache_path} expected_dim={expected_dim}")
    return features


def build_content_cross_matrix_from_sidecars(
    rows: Sequence[dict],
    *,
    content_pe_cache_dir: str | Path,
    content_pe_v2_cache_dir: str | Path,
    progress_interval: int = 5000,
) -> np.ndarray:
    """Build Loop43 cross features from existing sidecars only.

    Frozen evaluation must not mutate the content cache. Missing or invalid
    sidecars should be materialized by an explicit preparation step first.
    """

    if not rows:
        raise ValueError("No content cross rows were loaded")
    matrix = np.empty((len(rows), len(CONTENT_CROSS_FEATURE_NAMES)), dtype=np.float32)
    for index, row in enumerate(rows):
        pe1 = _load_required_sidecar_features(
            row,
            content_pe_cache_dir,
            len(CONTENT_PE_FEATURE_NAMES),
            "content_pe_v1",
        )
        pe2 = _load_required_sidecar_features(
            row,
            content_pe_v2_cache_dir,
            len(CONTENT_PE_V2_FEATURE_NAMES),
            "content_pe_v2",
        )
        matrix[index] = content_cross_features_from_arrays(pe1, pe2)
        processed = index + 1
        if progress_interval > 0 and processed % progress_interval == 0:
            print(f"[content-cross-cache-only] processed={processed}/{len(rows)}")
    return matrix


def predict_scores_chunked(model, matrix: np.ndarray, chunk_size: Optional[int]) -> np.ndarray:
    if chunk_size is None or chunk_size <= 0 or chunk_size >= int(matrix.shape[0]):
        return predict_scores(model, matrix)
    chunks: list[np.ndarray] = []
    for start in range(0, int(matrix.shape[0]), chunk_size):
        end = min(start + chunk_size, int(matrix.shape[0]))
        print(f"[frozen-eval] scoring rows {start}:{end}")
        chunks.append(predict_scores(model, matrix[start:end]))
    return np.concatenate(chunks).astype(np.float32, copy=False)


def assert_expected_feature_dim(model, matrix: np.ndarray) -> None:
    expected = getattr(model, "n_features_in_", None)
    if expected is None:
        return
    expected = int(expected)
    actual = int(matrix.shape[1])
    if actual != expected:
        raise ValueError(f"Frozen model feature dimension mismatch: matrix has {actual}, model expects {expected}")


def append_payload_extra_features(
    matrix,
    kept_rows: Sequence[dict],
    payload: dict,
    feature_config: FeatureConfig,
    args: argparse.Namespace,
):
    """Append payload-specific features that are not part of base Stage-2 config."""

    extra_feature_names = []
    if payload.get("schema") == "axon_loop43_content_cross_payload_v1":
        content_pe_cache_dir = args.content_pe_cache_dir or getattr(feature_config, "content_cache_dir", None)
        content_pe_v2_cache_dir = args.content_pe_v2_cache_dir or payload.get("content_pe_v2_cache_dir")
        if content_pe_cache_dir is None or content_pe_v2_cache_dir is None:
            raise ValueError(
                "Loop43 content-cross payload requires --content-pe-cache-dir and --content-pe-v2-cache-dir "
                "to rebuild the same frozen feature matrix."
            )
        if bool(getattr(args, "allow_content_sidecar_build", False)):
            cross_matrix = build_content_cross_matrix(
                kept_rows,
                CrossConfig(
                    content_pe_cache_dir=str(resolve_path(Path(content_pe_cache_dir))),
                    content_pe_v2_cache_dir=str(resolve_path(Path(content_pe_v2_cache_dir))),
                ),
            )
        else:
            cross_matrix = build_content_cross_matrix_from_sidecars(
                kept_rows,
                content_pe_cache_dir=resolve_path(Path(content_pe_cache_dir)),
                content_pe_v2_cache_dir=resolve_path(Path(content_pe_v2_cache_dir)),
                progress_interval=int(getattr(args, "sidecar_progress_interval", 5000)),
            )
        matrix = append_feature_columns(matrix, cross_matrix)
        extra_feature_names.extend(str(name) for name in payload.get("content_cross_feature_names", []))
    return matrix, extra_feature_names


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    model_path = resolve_path(args.model)
    with model_path.open("rb") as handle:
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
    matrix, extra_feature_names = append_payload_extra_features(matrix, kept_rows, payload, feature_config, args)
    payload_feature_dim = int(matrix.shape[1])
    knn_payload = payload.get("knn") or {}
    if knn_payload.get("enabled"):
        batch_size = int(args.knn_batch_size or knn_payload.get("batch_size") or 2048)
        knn_reference = load_stage2_knn_reference_from_payload(model_path, knn_payload)
        matrix = append_frozen_knn_features(
            matrix,
            knn_reference,
            knn_payload["top_ks"],
            batch_size=batch_size,
        )
    assert_expected_feature_dim(model, matrix)
    scores = predict_scores_chunked(model, matrix, int(args.eval_chunk_size))
    metrics = metrics_at_threshold(scores, labels, threshold)
    output_predictions = resolve_path(args.output_predictions_csv)
    write_predictions(output_predictions, kept_rows, labels, scores, threshold)

    report = {
        "schema": "axon_stage2_frozen_eval_v1",
        "protocol": "frozen stage2 model only; no fitting and no threshold sweep",
        "model": str(model_path),
        "predictions": str(resolve_path(args.predictions)),
        "output_predictions_csv": str(output_predictions),
        "selected_from_val": payload.get("selected"),
        "threshold": threshold,
        "feature_config": feature_config.__dict__,
        "records": counts,
        "base_feature_dim": base_feature_dim,
        "payload_feature_dim": payload_feature_dim,
        "payload_extra_feature_names": extra_feature_names,
        "feature_dim": int(matrix.shape[1]),
        "content_sidecar_eval": {
            "cache_only": not bool(args.allow_content_sidecar_build),
            "content_pe_cache_dir": str(resolve_path(args.content_pe_cache_dir)) if args.content_pe_cache_dir else None,
            "content_pe_v2_cache_dir": str(resolve_path(args.content_pe_v2_cache_dir)) if args.content_pe_v2_cache_dir else None,
        },
        "eval_chunk_size": int(args.eval_chunk_size),
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
