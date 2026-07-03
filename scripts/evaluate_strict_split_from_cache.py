#!/usr/bin/env python3
"""Evaluate split rows from cache with source_sha256-only alignment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig  # noqa: E402
from dataset import _load_cached_feature_npz, _resolve_manifest_cache_path  # noqa: E402
from feature_mask import apply_feature_mask_to_tensors, load_feature_mask_tensors, summarize_feature_mask  # noqa: E402
from model import AxonMalwareModel  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402


VALID_SPLITS = {"train", "val", "test", "test10k", "all"}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def is_valid_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def read_split_rows(path: Path, split: str) -> list[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"source_path", "source_sha256", "label", "sample_index", "split"}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"Split CSV missing strict columns: {missing}")
        rows = [dict(row) for row in reader]
    if split != "all":
        rows = [row for row in rows if str(row.get("split", "")).strip() == split]
    return rows


def read_manifest_by_sha(path: Path) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Manifest must contain a samples list")
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    issue_counts: Counter = Counter()
    for sample in samples:
        source_sha = str(sample.get("source_sha256") or "").strip().casefold()
        if not is_valid_sha256(source_sha):
            issue_counts["manifest_invalid_source_sha256"] += 1
            continue
        label = str(sample.get("label", "")).strip()
        if label not in {"0", "1"}:
            issue_counts["manifest_invalid_label"] += 1
            continue
        normalized = dict(sample)
        normalized["source_sha256"] = source_sha
        normalized["label"] = label
        by_sha[source_sha].append(normalized)
    return by_sha, issue_counts


def select_manifest_sample(row: dict[str, str], by_sha: dict[str, list[dict[str, Any]]]) -> tuple[Optional[dict[str, Any]], list[str]]:
    issues: list[str] = []
    source_sha = str(row.get("source_sha256") or "").strip().casefold()
    label = str(row.get("label", "")).strip()
    if not is_valid_sha256(source_sha):
        return None, ["split_invalid_source_sha256"]
    if label not in {"0", "1"}:
        return None, ["split_label_invalid"]

    matches = by_sha.get(source_sha, [])
    if not matches:
        return None, ["manifest_missing_source_sha256"]
    manifest_labels = {str(sample.get("label", "")).strip() for sample in matches}
    if len(manifest_labels) > 1:
        issues.append("manifest_conflicting_labels_for_source_sha256")
    label_matches = [sample for sample in matches if str(sample.get("label", "")).strip() == label]
    if not label_matches:
        issues.append("label_mismatch_split_manifest")
        return None, issues
    if issues:
        return None, issues
    return label_matches[0], []


def collect_strict_records(
    *,
    split_csv: Path,
    manifest_json: Path,
    split: str,
    max_rows: Optional[int] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_split_rows(split_csv, split)
    if max_rows is not None:
        rows = rows[:max_rows]
    manifest_by_sha, manifest_issue_counts = read_manifest_by_sha(manifest_json)
    cache_dir = resolve_path(manifest_json).parent

    records: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    issue_counts: Counter = Counter(manifest_issue_counts)
    label_counts: Counter = Counter()
    match_counts: Counter = Counter()

    for row_index, row in enumerate(rows):
        issues: list[str] = []
        sample, row_issues = select_manifest_sample(row, manifest_by_sha)
        issues.extend(row_issues)
        label = str(row.get("label", "")).strip()
        if label in {"0", "1"}:
            label_counts[label] += 1
        if sample is not None:
            try:
                cache_path = _resolve_manifest_cache_path(str(sample.get("cache_path", "")), cache_dir)
            except Exception as exc:
                cache_path = None
                issues.append(f"cache_path_invalid:{type(exc).__name__}")
            if cache_path is not None:
                match_counts["source_sha256"] += 1
                records.append(
                    {
                        "cache_path": str(cache_path),
                        "source_path": row.get("source_path", ""),
                        "source_sha256": str(row.get("source_sha256", "")).strip().casefold(),
                        "label": int(label),
                        "split": row.get("split", ""),
                        "sample_index": row.get("sample_index", ""),
                    }
                )
        if issues:
            for issue in issues:
                issue_counts[issue.split(":", 1)[0]] += 1
            issue_rows.append(
                {
                    "row_index": row_index,
                    "sample_index": row.get("sample_index", ""),
                    "split": row.get("split", ""),
                    "label": row.get("label", ""),
                    "source_sha256": row.get("source_sha256", ""),
                    "source_path": row.get("source_path", ""),
                    "issues": issues,
                }
            )

    summary = {
        "raw_rows": len(rows),
        "records": len(records),
        "label_counts": dict(sorted(label_counts.items())),
        "manifest_match_counts": dict(sorted(match_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_rows": len(issue_rows),
        "issue_examples": issue_rows[:50],
    }
    return records, summary


class StrictCachedSplitDataset(Dataset):
    def __init__(self, records: Sequence[dict[str, Any]], config: AxonExperimentConfig):
        self.records = list(records)
        self.config = config

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        byte_seq, pe_features, stat_features, _lightweight_features, label = _load_cached_feature_npz(
            Path(record["cache_path"]),
            self.config.max_byte_length,
            self.config.pe_feature_dim,
            self.config.stat_feature_dim,
            self.config.lightweight_feature_dim,
            expected_label=int(record["label"]),
            expected_source_sha256=str(record["source_sha256"]),
        )
        return (
            torch.from_numpy(byte_seq).long(),
            torch.from_numpy(pe_features).float(),
            torch.from_numpy(stat_features).float(),
            int(label),
            int(index),
        )


def compute_metrics(labels: Sequence[int], probs: Sequence[float], threshold: float) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) == 2 else None
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


def parse_thresholds(value: Optional[str], *, base_threshold: float) -> list[float]:
    thresholds = {float(base_threshold)}
    if value:
        thresholds.update(float(item.strip()) for item in value.split(",") if item.strip())
    return sorted(thresholds)


def write_prediction_rows(path: Path, records: Sequence[dict[str, Any]], probs: Sequence[float], threshold: float) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "source_sha256",
        "cache_path",
        "label",
        "split",
        "sample_index",
        "prob_malicious",
        "prediction",
        "correct",
    ]
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for record, probability in zip(records, probs):
            prediction = int(float(probability) >= threshold)
            label = int(record["label"])
            writer.writerow(
                {
                    **record,
                    "prob_malicious": float(probability),
                    "prediction": prediction,
                    "correct": prediction == label,
                }
            )


def evaluate_strict_split_from_cache(
    *,
    checkpoint: Path,
    split_csv: Path,
    manifest_json: Path,
    output_json: Path,
    split: str = "val",
    threshold: float = 0.5,
    sweep_thresholds: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    max_rows: Optional[int] = None,
    device_name: str = "cuda",
    output_predictions_csv: Optional[Path] = None,
    feature_mask_path: Optional[Path] = None,
) -> dict[str, Any]:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")
    records, record_summary = collect_strict_records(
        split_csv=split_csv,
        manifest_json=manifest_json,
        split=split,
        max_rows=max_rows,
    )
    if record_summary["issue_rows"]:
        payload = {
            "schema": "axon_strict_split_cache_eval_v1",
            "decision": "blocked_strict_record_alignment",
            "identity_feature_policy": (
                "source_sha256 is the only manifest/cache alignment key; path/name/extension/directory are never lookup keys or model evidence."
            ),
            "checkpoint": str(resolve_path(checkpoint)),
            "split_csv": str(resolve_path(split_csv)),
            "manifest_json": str(resolve_path(manifest_json)),
            "split": split,
            "record_summary": record_summary,
            "ready_for": {"train_val_only": False, "test10k": False, "full_test": False},
        }
        resolved_output = resolve_path(output_json)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    checkpoint_payload = load_safe_checkpoint(resolve_path(checkpoint), map_location="cpu")
    config = AxonExperimentConfig.from_dict(dict(checkpoint_payload["config"]))
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = AxonMalwareModel(config)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.to(device)
    model.eval()
    feature_mask = load_feature_mask_tensors(feature_mask_path, config, device) if feature_mask_path else None

    labels: list[int] = []
    probs: list[float] = []
    dataset = StrictCachedSplitDataset(records, config)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )
    with torch.inference_mode():
        for byte_seq, pe_features, stat_features, batch_labels, _batch_indices in loader:
            byte_seq = byte_seq.to(device, non_blocking=True)
            pe_features = pe_features.to(device, non_blocking=True)
            stat_features = stat_features.to(device, non_blocking=True)
            pe_features, stat_features = apply_feature_mask_to_tensors(pe_features, stat_features, feature_mask)
            logits = model(byte_seq, pe_features, stat_features=stat_features)["logits"]
            batch_probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            labels.extend(int(value) for value in batch_labels.numpy().tolist())
            probs.extend(float(value) for value in batch_probs)

    thresholds = parse_thresholds(sweep_thresholds, base_threshold=threshold)
    sweep = [compute_metrics(labels, probs, item) for item in thresholds]
    primary = next(item for item in sweep if item["threshold"] == float(threshold))
    best_by_f1 = max(sweep, key=lambda item: (item["f1"], -item["errors"], item["threshold"]))
    if output_predictions_csv is not None:
        write_prediction_rows(resolve_path(output_predictions_csv), records, probs, threshold)

    payload = {
        "schema": "axon_strict_split_cache_eval_v1",
        "decision": "evaluated",
        "identity_feature_policy": (
            "source_sha256 is the only manifest/cache alignment key; path/name/extension/directory are never lookup keys or model evidence."
        ),
        "checkpoint": str(resolve_path(checkpoint)),
        "feature_mask": str(resolve_path(feature_mask_path)) if feature_mask_path is not None else None,
        "feature_mask_summary": summarize_feature_mask(feature_mask[2]) if feature_mask is not None else None,
        "checkpoint_config": {
            "max_byte_length": config.max_byte_length,
            "pe_feature_dim": config.pe_feature_dim,
            "stat_feature_dim": config.stat_feature_dim,
            "pe_schema_version": config.pe_schema_version,
            "pe_fixed_section_slots": config.pe_fixed_section_slots,
        },
        "split_csv": str(resolve_path(split_csv)),
        "manifest_json": str(resolve_path(manifest_json)),
        "split": split,
        "record_summary": record_summary,
        "path_used_for_lookup": False,
        "predicted_samples": len(labels),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "max_rows": max_rows,
        "device": str(device),
        "metrics": primary,
        "threshold_sweep": sweep,
        "best_threshold_by_val_f1": best_by_f1,
        "predictions_csv": str(resolve_path(output_predictions_csv)) if output_predictions_csv is not None else None,
        "ready_for": {
            "train_val_only": True,
            "test10k": False,
            "full_test": False,
        },
        "memory_leak_profile": {
            "uses_cuda": device.type == "cuda",
            "uses_inference_mode": True,
            "persistent_workers": False,
            "cache_alignment": "source_sha256_only",
        },
    }
    resolved_output = resolve_path(output_json)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a split from cache with source_sha256-only alignment.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep-thresholds", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-predictions-csv", type=Path, default=None)
    parser.add_argument("--feature-mask", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = evaluate_strict_split_from_cache(
        checkpoint=args.checkpoint,
        split_csv=args.split_csv,
        manifest_json=args.manifest_json,
        output_json=args.output_json,
        split=args.split,
        threshold=float(args.threshold),
        sweep_thresholds=args.sweep_thresholds,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_rows=args.max_rows,
        device_name=args.device,
        output_predictions_csv=args.output_predictions_csv,
        feature_mask_path=args.feature_mask,
    )
    print(json.dumps({key: payload.get(key) for key in ["decision", "split", "predicted_samples", "metrics", "best_threshold_by_val_f1"]}, indent=2, ensure_ascii=False))
    if args.strict and payload.get("decision") != "evaluated":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
