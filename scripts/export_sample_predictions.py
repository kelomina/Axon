#!/usr/bin/env python3
"""Export per-sample model predictions with source paths and split labels."""

import argparse
import csv
import dataclasses
import json
import sys
import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig, TrainingConfig  # noqa: E402
from dataset import _load_cached_feature_npz, _resolve_manifest_cache_path  # noqa: E402
from feature_mask import apply_feature_mask_to_tensors, load_feature_mask_tensors, summarize_feature_mask  # noqa: E402
from model import AxonMalwareModel  # noqa: E402
from raw_group_tools import RawSample, normalize_path_text, read_csv_rows, resolve_path, scan_raw_sample_records  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402


PREDICTION_FIELDNAMES = [
    "source_path",
    "cache_path",
    "sample_index",
    "group_id",
    "source_group_id",
    "group_size",
    "sample_weight",
    "hard_family_role",
    "is_rare_group",
    "group_source",
    "label",
    "split",
    "prob_malicious",
    "prediction",
    "correct",
]

MISSING_CACHE_FIELDNAMES = ["source_path", "label", "split"]


def _read_toml_config(config_path: Optional[Path]) -> dict:
    if config_path is None:
        return {}
    with config_path.open("rb") as f:
        return tomllib.load(f)


def _dataclass_from_sections(cls, *sections: dict):
    merged = {}
    for section in sections:
        if section:
            merged.update(section)
    field_names = {field.name for field in dataclasses.fields(cls)}
    return cls(**{key: value for key, value in merged.items() if key in field_names})


def resolve_config(config_path: Optional[Path]):
    raw_config = _read_toml_config(resolve_path(config_path) if config_path else None)
    config = _dataclass_from_sections(
        AxonExperimentConfig,
        raw_config.get("experiment", {}),
        raw_config.get("model", {}),
        raw_config.get("data", {}),
        raw_config.get("device", {}),
    )
    train_config = _dataclass_from_sections(TrainingConfig, raw_config.get("training", {}))
    if "name" in raw_config.get("experiment", {}):
        config.experiment_name = raw_config["experiment"]["name"]
    return raw_config, config, train_config


def load_checkpoint_config(checkpoint_path: Path, batch_size: int):
    checkpoint = load_safe_checkpoint(checkpoint_path, map_location="cpu")
    config = AxonExperimentConfig.from_dict(checkpoint["config"])
    saved_train_config = checkpoint.get("train_config", {})
    if saved_train_config:
        saved_train_config["batch_size"] = batch_size
    try:
        train_config = TrainingConfig(**saved_train_config) if saved_train_config else TrainingConfig(batch_size=batch_size)
    except ValueError:
        # Prediction export only needs runtime batching and thresholding. Some
        # older checkpoints contain training-only options that no longer pass
        # current validation, so fall back to a fresh runtime-safe config.
        train_config = TrainingConfig(batch_size=batch_size)
        if isinstance(saved_train_config, dict) and "decision_threshold" in saved_train_config:
            train_config.decision_threshold = float(saved_train_config["decision_threshold"])
    train_config.batch_size = batch_size
    train_config.num_workers = 0
    return checkpoint, config, train_config


def load_manifest_samples(data_dir: Path, config: AxonExperimentConfig) -> tuple[List[dict], str]:
    cache_dir = data_dir / ".cache"
    if not cache_dir.exists():
        raise FileNotFoundError(f"Feature cache directory not found: {cache_dir}")
    manifest_name = None
    # Reuse the same cache hash logic indirectly through available manifest metadata.
    candidates = []
    for manifest_path in cache_dir.glob("manifest_*.json"):
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            continue
        if int(manifest.get("pe_feature_dim", -1)) != int(config.pe_feature_dim):
            continue
        if manifest.get("pe_schema_version", "legacy_dynamic") != config.pe_schema_version:
            continue
        byte_length_match = int(manifest.get("max_byte_length", -1)) == int(config.max_byte_length)
        candidates.append((
            1 if byte_length_match else 0,
            len(manifest.get("samples", [])),
            manifest_path.stat().st_mtime,
            manifest_path,
            manifest,
        ))
    if not candidates:
        suffix = "_38672ba0.npz" if (
            config.max_byte_length == 8192
            and config.stat_feature_dim == 49
            and config.pe_feature_dim == 256
            and getattr(config, "lightweight_feature_dim", 256) == 256
            and config.strict_pe_parsing
            and not config.allow_pe_fallback
            and config.pe_schema_version == "fixed_v2"
            and config.pe_fixed_section_slots == 32
        ) else ".npz"
        cache_files = sorted(cache_dir.glob(f"*{suffix}"))
        samples = [
            {
                "source_path": "",
                "cache_path": str(path),
                "label": None,
                "_manifest_path": "",
            }
            for path in cache_files
        ]
        return samples, "cache_scan_sorted"
    candidates.sort(reverse=True)
    _byte_match, _count, _mtime, manifest_path, manifest = candidates[0]
    manifest_name = str(manifest_path)
    samples = []
    for sample in manifest.get("samples", []):
        cache_path = _resolve_manifest_cache_path(sample.get("cache_path", ""), cache_dir)
        label = int(sample["label"])
        checked = dict(sample)
        checked["cache_path"] = str(cache_path)
        checked["label"] = label
        checked["_manifest_path"] = manifest_name
        samples.append(checked)
    return samples, "manifest"


def map_cache_samples_by_source(samples: Sequence[dict]) -> Dict[str, dict]:
    mapped = {}
    for sample in samples:
        source_path = sample.get("source_path")
        if source_path:
            for key in source_path_keys(source_path):
                mapped[key] = sample
    return mapped


def source_path_keys(source_path: str) -> List[str]:
    path = Path(source_path)
    keys = {normalize_path_text(source_path)}
    if not path.is_absolute():
        abs_path = (PROJECT_ROOT / path).resolve()
        keys.add(normalize_path_text(str(abs_path)))
    else:
        try:
            keys.add(normalize_path_text(str(path.resolve().relative_to(PROJECT_ROOT))))
        except ValueError:
            pass
    keys.add(path.name.casefold())
    return list(keys)


def lookup_cache_sample(cache_by_source: Dict[str, dict], source_path: str) -> Optional[dict]:
    for key in source_path_keys(source_path):
        sample = cache_by_source.get(key)
        if sample is not None:
            return sample
    return None


def load_sample_records_from_csv(samples_path: Path) -> List[RawSample]:
    rows = read_csv_rows(resolve_path(samples_path))
    records = []
    for fallback_index, row in enumerate(rows):
        record = RawSample(
            index=int(row.get("sample_index") or fallback_index),
            source_path=row["source_path"],
            label=int(row["label"]),
            split=row.get("split", "unknown"),
        )
        record.metadata = dict(row)
        records.append(record)
    return records


def _resolve_cache_path(cache_path: str, cache_dir: Path) -> Path:
    return _resolve_manifest_cache_path(cache_path, cache_dir)


def export_predictions(
    checkpoint_path: Path,
    config_path: Optional[Path],
    data_dir: Path,
    output_path: Path,
    batch_size: int,
    device_name: str,
    max_samples: Optional[int] = None,
    samples_path: Optional[Path] = None,
    split: Optional[str] = None,
    decision_threshold: Optional[float] = None,
    feature_mask_path: Optional[Path] = None,
) -> dict:
    checkpoint, checkpoint_config, train_config = load_checkpoint_config(checkpoint_path, batch_size)
    if decision_threshold is not None:
        train_config.decision_threshold = decision_threshold
    raw_config, current_config, _current_train = resolve_config(config_path)
    data_dir = resolve_path(data_dir or Path(current_config.data_dir or "data"))

    if samples_path is not None:
        raw_records = load_sample_records_from_csv(samples_path)
    else:
        raw_records = scan_raw_sample_records(current_config, data_dir)
    if split and split != "all":
        raw_records = [record for record in raw_records if record.split == split]
    if max_samples is not None and max_samples > 0:
        raw_records = raw_records[:max_samples]

    cache_dir = data_dir / ".cache"
    cache_samples, cache_source = load_manifest_samples(data_dir, checkpoint_config)
    cache_by_source = map_cache_samples_by_source(cache_samples)
    if cache_source == "cache_scan_sorted":
        for record, sample in zip(raw_records, cache_samples):
            sample["source_path"] = record.source_path
            sample["label"] = record.label
            for key in source_path_keys(record.source_path):
                cache_by_source[key] = sample

    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = AxonMalwareModel(checkpoint_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    feature_mask = None
    feature_mask_payload = None
    if feature_mask_path is not None:
        feature_mask = load_feature_mask_tensors(resolve_path(feature_mask_path), checkpoint_config, device)
        if feature_mask is not None:
            feature_mask_payload = feature_mask[2]

    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_path = output_path.with_name(output_path.stem + "_missing_cache.csv")
    predicted_count = 0
    missing_count = 0
    batch_items = []

    def flush_batch(prediction_writer):
        nonlocal predicted_count
        if not batch_items:
            return
        byte_seq = torch.stack([item["byte_seq"] for item in batch_items]).to(device)
        pe_features = torch.stack([item["pe_features"] for item in batch_items]).to(device)
        stat_features = torch.stack([item["stat_features"] for item in batch_items]).to(device)
        pe_features, stat_features = apply_feature_mask_to_tensors(pe_features, stat_features, feature_mask)
        with torch.no_grad():
            logits = model(byte_seq, pe_features, stat_features=stat_features)["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = (probs >= train_config.decision_threshold).astype(np.int64)
        for item, prob, pred in zip(batch_items, probs, preds):
            label = int(item["label"])
            metadata = item.get("metadata", {})
            prediction_writer.writerow({
                "source_path": item["source_path"],
                "cache_path": item["cache_path"],
                "sample_index": item["sample_index"],
                "group_id": metadata.get("group_id", ""),
                "source_group_id": metadata.get("source_group_id", ""),
                "group_size": metadata.get("group_size", ""),
                "sample_weight": metadata.get("sample_weight", ""),
                "hard_family_role": metadata.get("hard_family_role", ""),
                "is_rare_group": metadata.get("is_rare_group", ""),
                "group_source": metadata.get("group_source", ""),
                "label": label,
                "split": item["split"],
                "prob_malicious": float(prob),
                "prediction": int(pred),
                "correct": bool(int(pred) == label),
            })
            predicted_count += 1
        batch_items.clear()

    with (
        output_path.open("w", newline="", encoding="utf-8-sig") as prediction_handle,
        missing_path.open("w", newline="", encoding="utf-8-sig") as missing_handle,
    ):
        prediction_writer = csv.DictWriter(prediction_handle, fieldnames=PREDICTION_FIELDNAMES)
        missing_writer = csv.DictWriter(missing_handle, fieldnames=MISSING_CACHE_FIELDNAMES)
        prediction_writer.writeheader()
        missing_writer.writeheader()

        for record in raw_records:
            sample = lookup_cache_sample(cache_by_source, record.source_path)
            if sample is None:
                missing_writer.writerow({"source_path": record.source_path, "label": record.label, "split": record.split})
                missing_count += 1
                continue
            cache_path = _resolve_cache_path(sample["cache_path"], cache_dir)
            byte_seq, pe_feat, stat_feat, _lightweight_feat, label = _load_cached_feature_npz(
                cache_path,
                checkpoint_config.max_byte_length,
                checkpoint_config.pe_feature_dim,
                checkpoint_config.stat_feature_dim,
                checkpoint_config.lightweight_feature_dim,
                expected_label=int(record.label),
                expected_source_sha256=sample.get("source_sha256"),
            )
            batch_items.append({
                "source_path": record.source_path,
                "cache_path": str(cache_path),
                "sample_index": record.index,
                "label": label,
                "split": record.split,
                "metadata": getattr(record, "metadata", {}),
                "byte_seq": torch.from_numpy(byte_seq).long(),
                "pe_features": torch.from_numpy(pe_feat).float(),
                "stat_features": torch.from_numpy(stat_feat).float(),
            })
            if len(batch_items) >= batch_size:
                flush_batch(prediction_writer)
        flush_batch(prediction_writer)

    summary = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(data_dir),
        "output": str(output_path),
        "missing_cache_output": str(missing_path),
        "raw_samples": len(raw_records),
        "predicted_samples": predicted_count,
        "missing_cache_samples": missing_count,
        "decision_threshold": train_config.decision_threshold,
        "device": str(device),
        "cache_source": cache_source,
        "samples_source": str(resolve_path(samples_path)) if samples_path else "data_dir_scan",
        "split": split or "all",
        "feature_mask": str(resolve_path(feature_mask_path)) if feature_mask_path else None,
        "feature_mask_summary": summarize_feature_mask(feature_mask_payload) if feature_mask_payload else None,
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Export per-sample predictions for raw dataset diagnostics.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--config", type=Path, default=Path("config/default_config.toml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=Path, default=None,
                        help="Optional CSV with source_path,label,split,sample_index columns, e.g. group_members.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split", type=str, default=None,
                        help="Optional split filter when --samples contains split labels, e.g. train, val, test, all")
    parser.add_argument("--decision-threshold", type=float, default=None,
                        help="Override checkpoint decision threshold for exported predictions")
    parser.add_argument("--feature-mask", type=Path, default=None,
                        help="Optional exported PE/stat feature mask JSON to apply before prediction")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = export_predictions(
        checkpoint_path=resolve_path(args.checkpoint),
        config_path=args.config,
        data_dir=args.data_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        device_name=args.device,
        max_samples=args.max_samples,
        samples_path=args.samples,
        split=args.split,
        decision_threshold=args.decision_threshold,
        feature_mask_path=args.feature_mask,
    )
    print("=" * 60)
    print("Sample Predictions Export")
    print("=" * 60)
    print(f"Predicted samples: {summary['predicted_samples']}")
    print(f"Missing cache samples: {summary['missing_cache_samples']}")
    print(f"Output: {summary['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
