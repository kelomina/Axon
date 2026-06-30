#!/usr/bin/env python3
"""Audit validation errors by nearest train neighbors in the frozen stage-2 feature space."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter
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
    read_prediction_rows,
    resolve_path,
)


def _read_csv(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _normalize(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    scaled = (matrix.astype(np.float32, copy=False) - mean) / std
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    return scaled / np.maximum(norms, 1.0e-8)


def _compact_counter(values: Sequence[int]) -> str:
    counter = Counter(int(value) for value in values)
    return "|".join(f"{key}:{counter[key]}" for key in sorted(counter))


def _review_key(row: dict) -> str:
    return str(row.get("source_sha256") or row.get("source_path") or row.get("sample_index"))


def _load_stage2_payload(path: Path) -> tuple[FeatureConfig, AxonExperimentConfig]:
    with resolve_path(path).open("rb") as handle:
        payload = pickle.load(handle)
    feature_config = payload["feature_config"]
    if not isinstance(feature_config, FeatureConfig):
        feature_config = FeatureConfig(**dict(feature_config))
    checkpoint_config = AxonExperimentConfig.from_dict(dict(payload["checkpoint_config"]))
    return feature_config, checkpoint_config


def audit_neighbors(
    *,
    stage2_model: Path,
    train_predictions: Path,
    eval_base_predictions: Path,
    review_queue: Path,
    max_priority: int,
    top_k: int,
    output_json: Path,
    output_csv: Path,
) -> dict:
    feature_config, checkpoint_config = _load_stage2_payload(stage2_model)
    queue_rows_all = _read_csv(review_queue)
    queue_rows = [row for row in queue_rows_all if int(row.get("priority", 999)) <= max_priority]
    queue_by_key = {_review_key(row).casefold(): row for row in queue_rows}
    if not queue_by_key:
        raise ValueError("No review rows selected for the requested priority range")

    train_rows = read_prediction_rows(train_predictions)
    eval_rows_all = read_prediction_rows(eval_base_predictions)
    eval_rows = [
        row for row in eval_rows_all
        if _review_key(row).casefold() in queue_by_key
    ]
    found = {_review_key(row).casefold() for row in eval_rows}
    missing = sorted(set(queue_by_key) - found)[:10]
    if missing:
        raise ValueError(f"Could not match every review row to base predictions; missing examples={missing}")

    train_x, train_y, _train_base, train_kept_rows, train_counts = build_matrix(
        train_rows,
        checkpoint_config,
        feature_config,
    )
    eval_x, eval_y, _eval_base, eval_kept_rows, eval_counts = build_matrix(
        eval_rows,
        checkpoint_config,
        feature_config,
    )

    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    train_norm = _normalize(train_x, mean, std)
    eval_norm = _normalize(eval_x, mean, std)
    similarities = eval_norm @ train_norm.T

    top_k = min(int(top_k), train_norm.shape[0])
    output_rows = []
    support_counts: Counter[str] = Counter()
    for index, eval_row in enumerate(eval_kept_rows):
        key = _review_key(eval_row).casefold()
        review = queue_by_key[key]
        top_idx_unsorted = np.argpartition(-similarities[index], top_k - 1)[:top_k]
        top_idx = top_idx_unsorted[np.argsort(-similarities[index, top_idx_unsorted])]
        neighbor_labels = train_y[top_idx].astype(int)
        same_label_count = int((neighbor_labels == int(eval_y[index])).sum())
        opposite_label_count = int(top_k - same_label_count)
        opposite_ratio = float(opposite_label_count / top_k)
        same_ratio = float(same_label_count / top_k)
        if opposite_ratio >= 0.8:
            support_bucket = "neighbors_support_model_prediction"
        elif same_ratio >= 0.8:
            support_bucket = "neighbors_support_dataset_label"
        else:
            support_bucket = "neighbors_mixed"
        support_counts[support_bucket] += 1

        neighbor_paths = [train_kept_rows[int(i)].get("source_path", "") for i in top_idx[:5]]
        neighbor_shas = [train_kept_rows[int(i)].get("source_sha256", "") for i in top_idx[:5]]
        row = {
            "source_path": eval_row.get("source_path", ""),
            "source_sha256": eval_row.get("source_sha256", ""),
            "label": int(eval_y[index]),
            "error_type": review.get("error_type", ""),
            "priority": review.get("priority", ""),
            "reason": review.get("reason", ""),
            "stage2_prob_malicious": review.get("prob_malicious", ""),
            "support_bucket": support_bucket,
            "top_k": top_k,
            "neighbor_label_counts": _compact_counter(neighbor_labels),
            "same_label_count": same_label_count,
            "opposite_label_count": opposite_label_count,
            "opposite_label_ratio": f"{opposite_ratio:.6f}",
            "nearest_similarity": f"{float(similarities[index, top_idx[0]]):.8f}",
            "top5_neighbor_labels": "|".join(str(int(label)) for label in neighbor_labels[:5]),
            "top5_neighbor_similarities": "|".join(f"{float(similarities[index, i]):.8f}" for i in top_idx[:5]),
            "top5_neighbor_sha256": " | ".join(neighbor_shas),
            "top5_neighbor_paths": " | ".join(neighbor_paths),
        }
        output_rows.append(row)

    output_rows.sort(
        key=lambda row: (
            row["support_bucket"],
            int(row["priority"]),
            -float(row["opposite_label_ratio"]),
            row["source_path"],
        )
    )
    fieldnames = [
        "support_bucket",
        "priority",
        "reason",
        "error_type",
        "source_path",
        "source_sha256",
        "label",
        "stage2_prob_malicious",
        "top_k",
        "neighbor_label_counts",
        "same_label_count",
        "opposite_label_count",
        "opposite_label_ratio",
        "nearest_similarity",
        "top5_neighbor_labels",
        "top5_neighbor_similarities",
        "top5_neighbor_sha256",
        "top5_neighbor_paths",
    ]
    _write_csv(output_csv, output_rows, fieldnames)

    summary = {
        "schema": "axon_error_neighbor_audit_v1",
        "stage2_model": str(resolve_path(stage2_model)),
        "train_predictions": str(resolve_path(train_predictions)),
        "eval_base_predictions": str(resolve_path(eval_base_predictions)),
        "review_queue": str(resolve_path(review_queue)),
        "max_priority": max_priority,
        "top_k": top_k,
        "feature_config": feature_config.__dict__,
        "feature_dim": int(train_x.shape[1]),
        "train_records": train_counts,
        "eval_records": eval_counts,
        "review_rows_total": len(queue_rows_all),
        "review_rows_selected": len(queue_rows),
        "support_bucket_counts": dict(sorted(support_counts.items())),
        "examples": output_rows[:20],
        "outputs": {
            "neighbors_csv": str(resolve_path(output_csv)),
            "summary_json": str(resolve_path(output_json)),
        },
    }
    output_json = resolve_path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit validation errors by nearest train neighbors.")
    parser.add_argument("--stage2-model", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--eval-base-predictions", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--max-priority", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = audit_neighbors(
        stage2_model=args.stage2_model,
        train_predictions=args.train_predictions,
        eval_base_predictions=args.eval_base_predictions,
        review_queue=args.review_queue,
        max_priority=args.max_priority,
        top_k=args.top_k,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
