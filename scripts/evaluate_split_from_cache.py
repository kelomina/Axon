#!/usr/bin/env python3
"""Evaluate a checkpoint on rows from a split CSV using feature cache only."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.config import AxonExperimentConfig
from src.dataset import _iter_manifest_sample_entries, _load_cached_feature_npz
from src.model import AxonMalwareModel
from src.security import load_safe_checkpoint


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_path(path_text: str) -> str:
    return str(Path(path_text)).replace("/", "\\").casefold()


def project_relative_path_key(path_text: str) -> Optional[str]:
    """Return a cheap project-relative key without touching the filesystem."""
    if not path_text:
        return None
    normalized = normalize_path(path_text)
    root = normalize_path(str(PROJECT_ROOT)).rstrip("\\")
    prefix = root + "\\"
    if normalized.startswith(prefix):
        return normalized[len(prefix):]
    return None


def source_path_keys(path_text: str) -> list[str]:
    if not path_text:
        return []
    keys = {normalize_path(path_text)}
    path = Path(path_text)
    if not path.is_absolute():
        keys.add(normalize_path(str(PROJECT_ROOT / path)))
    relative_key = project_relative_path_key(path_text)
    if relative_key:
        keys.add(relative_key)
    keys.add(path.name.casefold())
    return list(keys)


def source_sha_from_path(path_text: str) -> Optional[str]:
    stem = Path(path_text).stem.casefold()
    if len(stem) == 64 and all(char in "0123456789abcdef" for char in stem):
        return stem
    return None


def cache_eval_num_workers(num_workers: int) -> int:
    value = max(0, int(num_workers))
    if os.name == "nt" and value > 0:
        raise ValueError(
            "Cache split evaluation keeps an in-memory record index; on Windows "
            "num_workers > 0 would spawn worker copies of that index. Use 0."
        )
    return value


def load_manifest_lookup(manifest_path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    by_source: dict[str, dict] = {}
    by_sha: dict[str, dict] = {}
    for row in _iter_manifest_sample_entries(manifest_path):
        row = {
            "source_path": row.get("source_path", ""),
            "cache_path": row.get("cache_path", ""),
            "label": row.get("label", ""),
            "source_sha256": str(row.get("source_sha256") or "").casefold(),
        }
        source_path = row.get("source_path")
        if source_path:
            for key in source_path_keys(source_path):
                by_source.setdefault(key, row)
        source_sha256 = str(row.get("source_sha256") or "").casefold()
        if source_sha256:
            by_sha.setdefault(source_sha256, row)
    return by_source, by_sha


MISSING_CACHE_FIELDNAMES = ["source_path", "original_source_path", "label", "split", "sample_index", "reason"]

PREDICTION_FIELDNAMES = [
    "source_path",
    "original_source_path",
    "cache_path",
    "source_sha256",
    "label",
    "split",
    "sample_index",
    "prob_malicious",
    "prediction",
    "correct",
]


def iter_split_rows(split_csv: Path, split: Optional[str], max_rows: Optional[int] = None):
    emitted = 0
    with split_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if split and split != "all" and row.get("split") != split:
                continue
            yield row
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                break


def lookup_manifest_sample(row: dict, by_source: dict[str, dict], by_sha: dict[str, dict]) -> tuple[Optional[dict], str]:
    candidate_paths = [
        row.get("source_path", ""),
        row.get("original_source_path", ""),
    ]
    for path_text in candidate_paths:
        for key in source_path_keys(path_text):
            sample = by_source.get(key)
            if sample is not None:
                return sample, "source_path"
    for path_text in candidate_paths:
        source_sha = source_sha_from_path(path_text)
        if source_sha:
            sample = by_sha.get(source_sha)
            if sample is not None:
                return sample, "source_sha256_from_path"
    return None, "missing"


def write_missing_cache_rows(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MISSING_CACHE_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class CachedSplitRowsDataset(Dataset):
    """Load split rows directly from feature-cache NPZ files."""

    def __init__(self, records: list[dict], checkpoint_config: AxonExperimentConfig):
        self.records = records
        self.config = checkpoint_config

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        byte_seq, pe_feat, stat_feat, _lightweight_feat, label = _load_cached_feature_npz(
            Path(record["cache_path"]),
            self.config.max_byte_length,
            self.config.pe_feature_dim,
            self.config.stat_feature_dim,
            self.config.lightweight_feature_dim,
            expected_label=int(record["label"]),
            expected_source_sha256=record.get("source_sha256"),
        )
        return (
            torch.from_numpy(byte_seq),
            torch.from_numpy(pe_feat).float(),
            torch.from_numpy(stat_feat).float(),
            int(label),
            index,
        )


def compute_metrics(labels: list[int], probs: list[float], threshold: float) -> dict:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    auc = None
    if len(set(y_true.tolist())) == 2:
        auc = float(roc_auc_score(y_true, y_prob))
    return {
        "threshold": float(threshold),
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "errors": fp + fn,
    }


def write_prediction_rows(path: Path, rows: Sequence[dict], threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            probability = float(row["prob_malicious"])
            prediction = int(probability >= threshold)
            label = int(row["label"])
            writer.writerow(
                {
                    **row,
                    "prob_malicious": probability,
                    "prediction": prediction,
                    "correct": prediction == label,
                }
            )


def evaluate_from_cache(
    *,
    checkpoint_path: Path,
    config_path: Path,
    split_csv: Path,
    manifest_path: Path,
    output_json: Path,
    split: Optional[str],
    threshold: float,
    sweep_thresholds: Optional[Sequence[float]],
    batch_size: int,
    num_workers: int,
    max_rows: Optional[int],
    device_name: str,
    missing_cache_output: Optional[Path],
    output_predictions_csv: Optional[Path],
) -> dict:
    checkpoint = load_safe_checkpoint(resolve_path(checkpoint_path), map_location="cpu")
    checkpoint_config = AxonExperimentConfig()
    if isinstance(checkpoint.get("config"), dict):
        checkpoint_config = AxonExperimentConfig.from_dict(checkpoint["config"])

    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = AxonMalwareModel(checkpoint_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    checkpoint = None
    gc.collect()
    model.to(device)
    model.eval()

    manifest_by_source, manifest_by_sha = load_manifest_lookup(manifest_path)
    cache_dir = resolve_path(manifest_path).parent

    labels: list[int] = []
    probs: list[float] = []
    records: list[dict] = []
    match_counts: dict[str, int] = {}
    raw_row_count = 0
    missing_cache_count = 0

    missing_handle = None
    missing_writer = None
    if missing_cache_output is not None:
        missing_output_path = resolve_path(missing_cache_output)
        missing_output_path.parent.mkdir(parents=True, exist_ok=True)
        missing_handle = missing_output_path.open("w", encoding="utf-8-sig", newline="")
        missing_writer = csv.DictWriter(missing_handle, fieldnames=MISSING_CACHE_FIELDNAMES, extrasaction="ignore")
        missing_writer.writeheader()
    try:
        for row in iter_split_rows(split_csv, split, max_rows=max_rows):
            raw_row_count += 1
            sample, match_reason = lookup_manifest_sample(row, manifest_by_source, manifest_by_sha)
            if sample is None:
                missing_cache_count += 1
                if missing_writer is not None:
                    missing_writer.writerow({**row, "reason": match_reason})
                continue
            cache_path = Path(sample["cache_path"])
            if not cache_path.is_absolute():
                cache_path = cache_dir / cache_path.name
            if not cache_path.exists():
                missing_cache_count += 1
                if missing_writer is not None:
                    missing_writer.writerow({**row, "reason": "cache_file_missing"})
                continue
            match_counts[match_reason] = match_counts.get(match_reason, 0) + 1
            records.append(
                {
                    "cache_path": str(cache_path),
                    "label": int(row["label"]),
                    "source_sha256": sample.get("source_sha256"),
                    "source_path": row.get("source_path", ""),
                    "original_source_path": row.get("original_source_path", ""),
                    "split": row.get("split", ""),
                    "sample_index": row.get("sample_index", ""),
                }
            )
    finally:
        if missing_handle is not None:
            missing_handle.close()

    dataset = CachedSplitRowsDataset(records, checkpoint_config)
    worker_count = cache_eval_num_workers(num_workers)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=(device.type == "cuda"),
    )
    prediction_handle = None
    prediction_writer = None
    if output_predictions_csv is not None:
        predictions_output_path = resolve_path(output_predictions_csv)
        predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_handle = predictions_output_path.open("w", encoding="utf-8-sig", newline="")
        prediction_writer = csv.DictWriter(prediction_handle, fieldnames=PREDICTION_FIELDNAMES, extrasaction="ignore")
        prediction_writer.writeheader()
    try:
        with torch.inference_mode():
            for byte_seq, pe_features, stat_features, batch_labels, batch_indices in loader:
                byte_seq = byte_seq.to(device, non_blocking=True)
                pe_features = pe_features.to(device, non_blocking=True)
                stat_features = stat_features.to(device, non_blocking=True)
                logits = model(byte_seq, pe_features, stat_features=stat_features)["logits"]
                batch_probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                batch_label_values = [int(value) for value in batch_labels.numpy().tolist()]
                batch_index_values = [int(value) for value in batch_indices.numpy().tolist()]
                labels.extend(batch_label_values)
                probs.extend(float(value) for value in batch_probs)
                if prediction_writer is not None:
                    for record_index, label, probability in zip(batch_index_values, batch_label_values, batch_probs):
                        record = records[record_index]
                        probability = float(probability)
                        prediction = int(probability >= threshold)
                        prediction_writer.writerow(
                            {
                                "source_path": record.get("source_path", ""),
                                "original_source_path": record.get("original_source_path", ""),
                                "cache_path": record["cache_path"],
                                "source_sha256": record.get("source_sha256") or "",
                                "label": label,
                                "split": record.get("split", ""),
                                "sample_index": record.get("sample_index", ""),
                                "prob_malicious": probability,
                                "prediction": prediction,
                                "correct": prediction == int(label),
                            }
                        )
    finally:
        if prediction_handle is not None:
            prediction_handle.close()

    thresholds = [float(threshold)]
    if sweep_thresholds:
        thresholds = sorted({*thresholds, *(float(value) for value in sweep_thresholds)})
    sweep_rows = [compute_metrics(labels, probs, item) for item in thresholds]
    primary_metrics = next(row for row in sweep_rows if row["threshold"] == float(threshold))

    payload = {
        "schema": "axon_split_cache_eval_v1",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "split_csv": str(split_csv),
        "manifest": str(manifest_path),
        "split": split or "all",
        "raw_rows": raw_row_count,
        "predicted_samples": len(labels),
        "missing_cache_samples": missing_cache_count,
        "missing_cache_output": str(resolve_path(missing_cache_output)) if missing_cache_output is not None else None,
        "predictions_csv": str(resolve_path(output_predictions_csv)) if output_predictions_csv is not None else None,
        "manifest_match_counts": dict(sorted(match_counts.items())),
        "batch_size": batch_size,
        "num_workers": worker_count,
        "max_rows": max_rows,
        "device": str(device),
        "metrics": primary_metrics,
        "threshold_sweep": sweep_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate split CSV rows from feature cache.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep-thresholds", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--missing-cache-output", type=Path, default=None)
    parser.add_argument("--output-predictions-csv", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = evaluate_from_cache(
        checkpoint_path=resolve_path(args.checkpoint),
        config_path=resolve_path(args.config),
        split_csv=resolve_path(args.split_csv),
        manifest_path=resolve_path(args.manifest),
        output_json=resolve_path(args.output_json),
        split=args.split,
        threshold=args.threshold,
        sweep_thresholds=(
            [float(item.strip()) for item in args.sweep_thresholds.split(",") if item.strip()]
            if args.sweep_thresholds
            else None
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_rows=args.max_rows,
        device_name=args.device,
        missing_cache_output=args.missing_cache_output,
        output_predictions_csv=args.output_predictions_csv,
    )
    print(f"JSON: {args.output_json}")
    print(json.dumps(payload["metrics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
