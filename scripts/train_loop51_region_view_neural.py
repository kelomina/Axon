#!/usr/bin/env python3
"""Train the Loop51 region-view neural candidate on Train/Val only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for item in (PROJECT_ROOT, SRC_DIR, SCRIPTS_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dataset import FeatureCacheDataset, SubDataset  # noqa: E402
from main import _make_train_generator, _resolve_config, _seed_worker, _set_training_seed  # noqa: E402
from model import AxonMalwareModel  # noqa: E402
from trainer import AxonTrainer  # noqa: E402


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def split_indices_from_file(dataset: FeatureCacheDataset, split_csv: Path) -> dict[str, list[int]]:
    assignments = {}
    with split_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            assignments[(row["source_path"], str(row["label"]))] = row["split"]

    indices = {"train": [], "val": []}
    for index, sample in enumerate(dataset.samples):
        split = assignments.get((sample["source_path"], str(sample["label"])))
        if split in indices:
            indices[split].append(index)
    return indices


def balanced_limit(indices: list[int], labels: list[int], limit: Optional[int]) -> list[int]:
    if limit is None or limit <= 0 or len(indices) <= limit:
        return indices
    by_label: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        by_label[int(labels[index])].append(index)
    per_label = max(1, limit // max(1, len(by_label)))
    selected = []
    for label in sorted(by_label):
        selected.extend(by_label[label][:per_label])
    remaining = limit - len(selected)
    if remaining > 0:
        selected_set = set(selected)
        selected.extend(index for index in indices if index not in selected_set and len(selected) < limit)
    return sorted(selected[:limit])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/random_20w_region_view_8192_seed51.toml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--split-file", default="reports/random_20w_split/loop27_corrected_split.csv")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--summary-json", default="reports/random_20w_split/loop51_region_view_neural_smoke_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, train_config = _resolve_config(args)
    if args.epochs is not None:
        train_config.max_epochs = args.epochs
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device
    if args.output_dir is not None:
        config.model_save_dir = Path(args.output_dir)
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if train_config.lr_scheduler == "cosine" and train_config.warmup_epochs >= train_config.max_epochs:
        train_config.warmup_epochs = max(0, train_config.max_epochs - 1)

    data_dir = args.data_dir or config.data_dir or "data"
    cache_dir = args.cache_dir or getattr(config, "cache_dir", None) or "data/.cache_loop51_region_view_8192"

    _set_training_seed(config.seed)
    train_generator = _make_train_generator(config.seed)
    worker_init_fn = _seed_worker if train_config.num_workers > 0 else None

    dataset = FeatureCacheDataset(
        data_dir=data_dir,
        cache_dir=cache_dir,
        max_byte_length=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        axon_config=config,
        require_manifest=True,
    )
    indices = split_indices_from_file(dataset, resolve_path(args.split_file))
    train_indices = balanced_limit(indices["train"], dataset.label_list, args.max_train_samples)
    val_indices = balanced_limit(indices["val"], dataset.label_list, args.max_val_samples)
    if not train_indices or not val_indices:
        raise ValueError(f"Loop51 requires non-empty train/val splits, got train={len(train_indices)} val={len(val_indices)}")

    train_dataset = SubDataset(dataset, train_indices)
    val_dataset = SubDataset(dataset, val_indices)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        generator=train_generator,
        worker_init_fn=worker_init_fn,
        pin_memory=train_config.pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
    )

    print("[Loop51] Region-view Train/Val only")
    print(f"[Loop51] cache_dir={cache_dir}")
    print(f"[Loop51] train={len(train_dataset)} val={len(val_dataset)} test=not_loaded")

    model = AxonMalwareModel(config)
    trainer = AxonTrainer(model, config, train_config)
    results = trainer.train(train_loader, val_loader=val_loader, test_loader=None, fast_mode=False)

    summary = {
        "schema": "axon_loop51_region_view_neural_train_v1",
        "config": str(resolve_path(args.config)),
        "cache_dir": str(resolve_path(cache_dir)),
        "split_file": str(resolve_path(args.split_file)),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": 0,
        "best_f1": float(trainer.best_f1),
        "best_epoch": int(trainer.best_epoch),
        "epochs": int(train_config.max_epochs),
        "model_dir": str(config.model_save_dir),
        "test_policy": "not loaded or evaluated before Val gate",
        "val_history": [
            {"epoch": int(metric.epoch), "f1": float(metric.f1), "fp": int(metric.false_positive), "fn": int(metric.false_negative)}
            for metric in results.get("val", [])
        ],
    }
    output_path = resolve_path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Loop51] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
