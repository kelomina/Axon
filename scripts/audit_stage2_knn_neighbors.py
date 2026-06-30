#!/usr/bin/env python3
"""Audit Stage2 kNN errors against the frozen train-memory neighbors."""

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
from train_stage2_cache_matrix import FeatureConfig, build_matrix, read_prediction_rows, resolve_path  # noqa: E402


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


def _review_key(row: dict) -> str:
    return str(row.get("source_sha256") or row.get("sample_index") or row.get("source_path")).casefold()


def _normalize(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    centered = (matrix.astype(np.float32, copy=False) - mean) / std
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return (centered / np.maximum(norms, 1.0e-8)).astype(np.float32, copy=False)


def _compact_counter(values: Sequence[int]) -> str:
    counter = Counter(int(value) for value in values)
    return "|".join(f"{key}:{counter[key]}" for key in sorted(counter))


def _load_payload(model_path: Path) -> tuple[FeatureConfig, AxonExperimentConfig, dict]:
    with resolve_path(model_path).open("rb") as handle:
        payload = pickle.load(handle)
    feature_config = payload["feature_config"]
    if not isinstance(feature_config, FeatureConfig):
        feature_config = FeatureConfig(**dict(feature_config))
    checkpoint_config = AxonExperimentConfig.from_dict(dict(payload["checkpoint_config"]))
    knn_payload = payload.get("knn") or {}
    if not knn_payload.get("enabled"):
        raise ValueError("The selected Stage2 model does not contain enabled kNN memory")
    return feature_config, checkpoint_config, knn_payload


def audit_neighbors(
    *,
    stage2_model: Path,
    train_predictions: Path,
    eval_base_predictions: Path,
    review_queue: Path,
    max_priority: int,
    top_k: int,
    batch_size: int,
    output_json: Path,
    output_csv: Path,
) -> dict:
    feature_config, checkpoint_config, knn_payload = _load_payload(stage2_model)
    reference = knn_payload["reference"]
    memory_norm = reference["memory_norm"].astype(np.float32, copy=False)
    memory_labels = reference["memory_labels"].astype(np.int64, copy=False)
    mean = reference["mean"].astype(np.float32, copy=False)
    std = reference["std"].astype(np.float32, copy=False)

    train_rows = read_prediction_rows(train_predictions)
    if len(train_rows) != memory_norm.shape[0]:
        raise ValueError(
            f"Train rows and frozen memory disagree: rows={len(train_rows)} memory={memory_norm.shape[0]}"
        )

    queue_rows_all = _read_csv(review_queue)
    queue_rows = [row for row in queue_rows_all if int(row.get("priority", 999)) <= int(max_priority)]
    queue_by_key = {_review_key(row): row for row in queue_rows}
    if not queue_by_key:
        raise ValueError("No review rows selected for the requested priority")

    eval_rows_all = read_prediction_rows(eval_base_predictions)
    eval_by_key = {_review_key(row): row for row in eval_rows_all}
    selected_base_rows = []
    missing = []
    for row in queue_rows:
        key = _review_key(row)
        base_row = eval_by_key.get(key)
        if base_row is None:
            missing.append(key)
            continue
        merged = dict(base_row)
        merged["_review_key"] = key
        selected_base_rows.append(merged)
    if missing:
        raise ValueError(f"Could not match every review row to base predictions; first missing={missing[:5]}")

    eval_x, eval_y, _eval_base, eval_kept_rows, eval_counts = build_matrix(
        selected_base_rows,
        checkpoint_config,
        feature_config,
    )
    eval_norm = _normalize(eval_x, mean, std)
    top_k = min(int(top_k), int(memory_norm.shape[0]))
    batch_size = max(1, int(batch_size))

    output_rows = []
    support_counts: Counter[str] = Counter()
    for start in range(0, eval_norm.shape[0], batch_size):
        stop = min(start + batch_size, eval_norm.shape[0])
        similarities = eval_norm[start:stop] @ memory_norm.T
        top_unsorted = np.argpartition(-similarities, top_k - 1, axis=1)[:, :top_k]
        top_sim_unsorted = np.take_along_axis(similarities, top_unsorted, axis=1)
        top_order = np.argsort(-top_sim_unsorted, axis=1)
        top_idx_batch = np.take_along_axis(top_unsorted, top_order, axis=1)
        top_sim_batch = np.take_along_axis(similarities, top_idx_batch, axis=1)

        for local_index in range(stop - start):
            eval_index = start + local_index
            eval_row = eval_kept_rows[eval_index]
            review = queue_by_key[eval_row["_review_key"]]
            top_idx = top_idx_batch[local_index]
            top_sim = top_sim_batch[local_index]
            neighbor_labels = memory_labels[top_idx].astype(int)
            label = int(eval_y[eval_index])
            same_label_count = int((neighbor_labels == label).sum())
            opposite_label_count = int(top_k - same_label_count)
            same_ratio = float(same_label_count / top_k)
            opposite_ratio = float(opposite_label_count / top_k)
            if opposite_ratio >= 0.8:
                support_bucket = "neighbors_support_model_prediction"
            elif same_ratio >= 0.8:
                support_bucket = "neighbors_support_dataset_label"
            else:
                support_bucket = "neighbors_mixed"
            support_counts[support_bucket] += 1

            top5_indices = [int(index) for index in top_idx[:5]]
            row = {
                "support_bucket": support_bucket,
                "priority": review.get("priority", ""),
                "reason": review.get("reason", ""),
                "error_type": review.get("error_type", ""),
                "source_path": review.get("source_path", ""),
                "source_sha256": review.get("source_sha256", ""),
                "label": label,
                "prediction": review.get("prediction", ""),
                "stage2_prob_malicious": review.get("stage2_prob_malicious", ""),
                "base_prob_malicious": eval_row.get("prob_malicious", ""),
                "top_k": top_k,
                "neighbor_label_counts": _compact_counter(neighbor_labels),
                "same_label_count": same_label_count,
                "opposite_label_count": opposite_label_count,
                "same_label_ratio": f"{same_ratio:.6f}",
                "opposite_label_ratio": f"{opposite_ratio:.6f}",
                "nearest_similarity": f"{float(top_sim[0]):.8f}",
                "top5_neighbor_labels": "|".join(str(int(memory_labels[index])) for index in top5_indices),
                "top5_neighbor_similarities": "|".join(f"{float(top_sim[i]):.8f}" for i in range(min(5, len(top_sim)))),
                "top5_neighbor_sha256": " | ".join(train_rows[index].get("source_sha256", "") for index in top5_indices),
                "top5_neighbor_paths": " | ".join(train_rows[index].get("source_path", "") for index in top5_indices),
            }
            output_rows.append(row)

    output_rows.sort(
        key=lambda row: (
            row["support_bucket"],
            int(row["priority"]),
            -float(row["opposite_label_ratio"]),
            row["error_type"],
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
        "prediction",
        "stage2_prob_malicious",
        "base_prob_malicious",
        "top_k",
        "neighbor_label_counts",
        "same_label_count",
        "opposite_label_count",
        "same_label_ratio",
        "opposite_label_ratio",
        "nearest_similarity",
        "top5_neighbor_labels",
        "top5_neighbor_similarities",
        "top5_neighbor_sha256",
        "top5_neighbor_paths",
    ]
    _write_csv(output_csv, output_rows, fieldnames)
    summary = {
        "schema": "axon_stage2_knn_neighbor_audit_v1",
        "stage2_model": str(resolve_path(stage2_model)),
        "train_predictions": str(resolve_path(train_predictions)),
        "eval_base_predictions": str(resolve_path(eval_base_predictions)),
        "review_queue": str(resolve_path(review_queue)),
        "max_priority": int(max_priority),
        "top_k": int(top_k),
        "feature_config": feature_config.__dict__,
        "base_feature_dim": int(eval_x.shape[1]),
        "memory_rows": int(memory_norm.shape[0]),
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
    parser = argparse.ArgumentParser(description="Audit errors against frozen Stage2 kNN train memory.")
    parser.add_argument("--stage2-model", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--eval-base-predictions", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--max-priority", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
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
        batch_size=args.batch_size,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
