#!/usr/bin/env python3
"""在 data/.cache 上做监督 baseline 与 Pro RL 的小规模对照。

这个脚本只读取已经提取好的缓存，不扫描原始 PE 文件，不重新提取特征。
默认使用 fixed PE256 主配置在 fast 模式下的缓存口径：max_byte_length=8192。
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import statistics
import sys
import tomllib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PRO_DIR = PROJECT_ROOT / "Pro"
sys.path.insert(0, str(PRO_DIR))
sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig, DSRAArchitectureConfig  # noqa: E402
from dataset import FeatureCacheDataset, create_stratified_split  # noqa: E402
from model import AxonMalwareModel  # noqa: E402
from rl_axon import (  # noqa: E402
    AxonPolicyAgent,
    MalwareBanditEnv,
    PolicyGradientTrainer,
    RLTrainingConfig,
    RewardConfig,
)


Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class TensorCacheDataset(Dataset):
    """把本次对照用到的缓存样本预载入内存。

    data/.cache 里的每个样本都是一个 npz 文件。大规模训练时，如果每个 epoch
    都重新打开几万次 npz，会被磁盘 I/O 卡住。这个包装器只在开始时读取一次，
    后续训练直接从内存张量切片。
    """

    def __init__(self, samples: List[Batch]):
        byte_seq, pe_features, stat_features, labels = zip(*samples)
        self.byte_seq = torch.stack([item.long() for item in byte_seq])
        self.pe_features = torch.stack([item.float() for item in pe_features])
        self.stat_features = torch.stack([item.float() for item in stat_features])
        self.labels = torch.stack([item.long() for item in labels])

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Batch:
        return (
            self.byte_seq[idx],
            self.pe_features[idx],
            self.stat_features[idx],
            self.labels[idx],
        )


class IndexedDataset(Dataset):
    """用固定索引复用同一份 base dataset。"""

    def __init__(self, base_dataset: Dataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = [int(idx) for idx in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Batch:
        return self.base_dataset[self.indices[idx]]


def read_toml_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


def dataclass_from_sections(cls, *sections: dict):
    merged = {}
    for section in sections:
        if section:
            merged.update(section)
    field_names = {field.name for field in dataclasses.fields(cls)}
    return cls(**{key: value for key, value in merged.items() if key in field_names})


def resolve_config(config_path: str, fast_cache: bool) -> AxonExperimentConfig:
    raw_config = read_toml_config(config_path)
    experiment_section = raw_config.get("experiment", {})
    model_section = raw_config.get("model", {})
    dsra_section = raw_config.get("dsra", {})
    data_section = raw_config.get("data", {})
    device_section = raw_config.get("device", {})

    config = dataclass_from_sections(
        AxonExperimentConfig,
        experiment_section,
        model_section,
        data_section,
        device_section,
    )
    if "name" in experiment_section:
        config.experiment_name = experiment_section["name"]
    if "device" in device_section:
        config.device = device_section["device"]
    if "output_dir" in data_section:
        config.model_save_dir = Path(data_section["output_dir"])
    if "log_dir" in data_section:
        config.log_dir = Path(data_section["log_dir"])

    if dsra_section:
        config.dsra_arch_config = dataclass_from_sections(
            DSRAArchitectureConfig,
            {
                "dim": config.dsra_dim,
                "heads": config.dsra_heads,
                "slots": config.dsra_slots,
                "read_topk": config.dsra_read_topk,
                "write_topk": config.dsra_write_topk,
                "local_window": config.dsra_local_window,
            },
            dsra_section,
        )

    if fast_cache:
        config.fast_mode = True
        config.max_byte_length = config.fast_mode_byte_length
    return config


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
    )


def dataset_indices(dataset: Dataset) -> List[int]:
    if hasattr(dataset, "indices"):
        return [int(idx) for idx in dataset.indices]
    return list(range(len(dataset)))


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_metadata(args, config: AxonExperimentConfig, dataset: Dataset) -> Dict[str, object]:
    return {
        "version": 1,
        "config_path": args.config,
        "data_dir": args.data_dir,
        "cache_dir": args.cache_dir,
        "samples_per_class": args.samples_per_class,
        "seed": args.seed,
        "max_byte_length": config.max_byte_length,
        "pe_feature_dim": config.pe_feature_dim,
        "stat_feature_dim": config.stat_feature_dim,
        "pe_schema_version": config.pe_schema_version,
        "pe_fixed_section_slots": config.pe_fixed_section_slots,
        "dataset_size": len(dataset),
    }


def load_or_create_split(
    dataset: Dataset,
    args,
    config: AxonExperimentConfig,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, object]]:
    split_path = Path(args.split_file) if args.split_file else None
    if split_path and split_path.exists():
        split = load_json(split_path)
        indices = split["indices"]
        print(f"[Split] Loaded fixed split: {split_path}")
        return (
            IndexedDataset(dataset, indices["train"]),
            IndexedDataset(dataset, indices["val"]),
            IndexedDataset(dataset, indices["test"]),
            split,
        )

    train_dataset, val_dataset, test_dataset = create_stratified_split(dataset, axon_config=config)
    split = split_metadata(args, config, dataset)
    split["indices"] = {
        "train": dataset_indices(train_dataset),
        "val": dataset_indices(val_dataset),
        "test": dataset_indices(test_dataset),
    }
    split["sizes"] = {
        "train": len(split["indices"]["train"]),
        "val": len(split["indices"]["val"]),
        "test": len(split["indices"]["test"]),
    }
    if split_path:
        save_json(split_path, split)
        print(f"[Split] Saved fixed split: {split_path}")
    return train_dataset, val_dataset, test_dataset, split


def preload_dataset(dataset: Dataset, name: str) -> TensorCacheDataset:
    print(f"Preloading {name} dataset into memory ({len(dataset)} samples)...")
    samples = [dataset[idx] for idx in range(len(dataset))]
    loaded = TensorCacheDataset(samples)
    mb = (
        loaded.byte_seq.numel() * loaded.byte_seq.element_size()
        + loaded.pe_features.numel() * loaded.pe_features.element_size()
        + loaded.stat_features.numel() * loaded.stat_features.element_size()
        + loaded.labels.numel() * loaded.labels.element_size()
    ) / (1024 * 1024)
    print(f"  {name} memory: {mb:.1f} MB")
    return loaded


def move_batch(batch: Batch, device: torch.device) -> Batch:
    byte_seq, pe_features, stat_features, labels = batch
    return (
        byte_seq.to(device),
        pe_features.to(device),
        stat_features.to(device),
        labels.to(device),
    )


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int, total_reward: float, total: int) -> Dict[str, float]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "reward": total_reward / max(1, total),
        "accuracy": (tp + tn) / max(1, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / max(1, fp + tn),
        "false_negative_rate": fn / max(1, fn + tp),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "samples": float(total),
    }


@torch.no_grad()
def evaluate_supervised(
    model: AxonMalwareModel,
    loader: Iterable[Batch],
    env: MalwareBanditEnv,
    device: torch.device,
    decision_threshold: float,
) -> Dict[str, float]:
    model.eval()
    tp = tn = fp = fn = total = 0
    total_reward = 0.0
    total_loss = 0.0
    batches = 0

    for batch in loader:
        byte_seq, pe_features, stat_features, labels = move_batch(batch, device)
        logits = model(byte_seq, pe_features, stat_features)["logits"]
        loss = F.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        actions = (probs >= decision_threshold).long()
        rewards = env.reward_for_actions(actions, labels)

        tp += int(((actions == 1) & (labels == 1)).sum().item())
        tn += int(((actions == 0) & (labels == 0)).sum().item())
        fp += int(((actions == 1) & (labels == 0)).sum().item())
        fn += int(((actions == 0) & (labels == 1)).sum().item())
        total += int(labels.numel())
        total_reward += float(rewards.sum().item())
        total_loss += float(loss.item())
        batches += 1

    metrics = metrics_from_counts(tp, tn, fp, fn, total_reward, total)
    metrics["loss"] = total_loss / max(1, batches)
    metrics["decision_threshold"] = decision_threshold
    return metrics


def train_supervised(
    model: AxonMalwareModel,
    train_loader: Iterable[Batch],
    val_loader: Iterable[Batch],
    env: MalwareBanditEnv,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    gradient_clip: float,
    decision_threshold: float,
) -> Dict[str, Dict[str, float]]:
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: Dict[str, Dict[str, float]] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            byte_seq, pe_features, stat_features, labels = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(byte_seq, pe_features, stat_features)["logits"]
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite supervised loss at epoch={epoch}")
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1

        val_metrics = evaluate_supervised(model, val_loader, env, device, decision_threshold)
        val_metrics["train_loss"] = total_loss / max(1, batches)
        history[f"epoch_{epoch}"] = val_metrics
        print(
            f"CE  | Epoch {epoch} | train_loss={val_metrics['train_loss']:.4f} "
            f"reward={val_metrics['reward']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} "
            f"fp_rate={val_metrics['false_positive_rate']:.4f}"
        )

    return history


def save_supervised_checkpoint(
    path: Path,
    model: AxonMalwareModel,
    config: AxonExperimentConfig,
    args,
    history: Dict[str, Dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "run_args": vars(args),
            "history": history,
        },
        path,
    )


def save_rl_checkpoint(
    path: Path,
    trainer: PolicyGradientTrainer,
    config: AxonExperimentConfig,
    args,
    history: Dict[str, Dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": trainer.agent.base_model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "config": config.to_dict(),
            "rl_config": dataclasses.asdict(trainer.config),
            "run_args": vars(args),
            "history": history,
        },
        path,
    )


@torch.no_grad()
def collect_supervised_outputs(
    model: AxonMalwareModel,
    loader: Iterable[Batch],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    batches = 0
    for batch in loader:
        byte_seq, pe_features, stat_features, labels = move_batch(batch, device)
        logits = model(byte_seq, pe_features, stat_features)["logits"]
        loss = F.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())
        total_loss += float(loss.item())
        batches += 1
    return torch.cat(all_probs), torch.cat(all_labels), total_loss / max(1, batches)


@torch.no_grad()
def collect_rl_outputs(
    trainer: PolicyGradientTrainer,
    loader: Iterable[Batch],
) -> Tuple[torch.Tensor, torch.Tensor]:
    trainer.agent.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        byte_seq, pe_features, stat_features, labels = trainer._move_batch(batch)
        probs = trainer.agent.probabilities(byte_seq, pe_features, stat_features)[:, 1]
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())
    return torch.cat(all_probs), torch.cat(all_labels)


def metrics_from_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    env: MalwareBanditEnv,
    threshold: float,
) -> Dict[str, float]:
    actions = (probs >= threshold).long()
    rewards = env.reward_for_actions(actions, labels)
    tp = int(((actions == 1) & (labels == 1)).sum().item())
    tn = int(((actions == 0) & (labels == 0)).sum().item())
    fp = int(((actions == 1) & (labels == 0)).sum().item())
    fn = int(((actions == 0) & (labels == 1)).sum().item())
    metrics = metrics_from_counts(tp, tn, fp, fn, float(rewards.sum().item()), int(labels.numel()))
    metrics["decision_threshold"] = float(threshold)
    return metrics


def threshold_sweep_from_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    env: MalwareBanditEnv,
    thresholds: List[float],
    loss: float | None = None,
) -> Dict[str, Dict[str, float]]:
    rows = {}
    for threshold in thresholds:
        metrics = metrics_from_probs(probs, labels, env, threshold)
        if loss is not None:
            metrics["loss"] = loss
        rows[f"{threshold:.3f}"] = metrics
    return rows


def build_run_config(args, config: AxonExperimentConfig, device: torch.device, thresholds: List[float]) -> Dict[str, object]:
    return {
        "path": args.config,
        "data_dir": args.data_dir,
        "cache_dir": args.cache_dir,
        "samples_per_class": args.samples_per_class,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "decision_threshold": args.decision_threshold,
        "sweep_thresholds": thresholds,
        "seed": args.seed,
        "mode": args.mode,
        "split_file": args.split_file,
        "run_dir": args.run_dir,
        "max_byte_length": config.max_byte_length,
        "pe_feature_dim": config.pe_feature_dim,
        "stat_feature_dim": config.stat_feature_dim,
        "pe_schema_version": config.pe_schema_version,
        "pe_fixed_section_slots": config.pe_fixed_section_slots,
    }


def build_method_report(
    method: str,
    run_config: Dict[str, object],
    split_info: Dict[str, object],
    history: Dict[str, Dict[str, float]],
    test_metrics: Dict[str, float],
    threshold_sweep: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    return {
        "method": method,
        "config": run_config,
        "split": split_info,
        "history": history,
        "test": test_metrics,
        "threshold_sweep": threshold_sweep,
    }


def metric_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def summarize_runs(summary_dirs: List[str]) -> Dict[str, object]:
    metrics = ["reward", "accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate"]
    rows = []
    grouped = {
        "ce": {metric: [] for metric in metrics},
        "rl": {metric: [] for metric in metrics},
        "delta_rl_minus_ce": {metric: [] for metric in metrics},
    }

    for raw_dir in summary_dirs:
        run_dir = Path(raw_dir)
        compare_path = run_dir / "compare_report.json"
        ce_path = run_dir / "ce_report.json"
        rl_path = run_dir / "rl_report.json"
        if compare_path.exists():
            report = load_json(compare_path)
            ce = report["test"]["ce"]
            rl = report["test"]["rl"]
            seed = report["config"].get("seed")
        else:
            ce = None
            rl = None
            seed = None
        if ce is None or rl is None:
            if not ce_path.exists() or not rl_path.exists():
                raise FileNotFoundError(f"Missing compare or method reports in: {run_dir}")
            ce_report = load_json(ce_path)
            rl_report = load_json(rl_path)
            ce = ce_report["test"]
            rl = rl_report["test"]
            seed = ce_report["config"].get("seed", rl_report["config"].get("seed"))
        row = {"run_dir": str(run_dir), "seed": seed, "ce": ce, "rl": rl, "delta_rl_minus_ce": {}}
        for metric in metrics:
            delta = float(rl[metric]) - float(ce[metric])
            row["delta_rl_minus_ce"][metric] = delta
            grouped["ce"][metric].append(float(ce[metric]))
            grouped["rl"][metric].append(float(rl[metric]))
            grouped["delta_rl_minus_ce"][metric].append(delta)
        rows.append(row)

    aggregate = {
        group: {metric: metric_summary(values) for metric, values in metric_map.items()}
        for group, metric_map in grouped.items()
    }
    rl_lower_fp = sum(1 for row in rows if row["delta_rl_minus_ce"]["false_positive_rate"] < 0)
    reward_not_worse = sum(1 for row in rows if row["delta_rl_minus_ce"]["reward"] >= -0.02)
    fn_not_much_worse = sum(1 for row in rows if row["delta_rl_minus_ce"]["false_negative_rate"] <= 0.05)
    decision = {
        "rl_lower_fp_count": rl_lower_fp,
        "reward_not_worse_count": reward_not_worse,
        "fn_not_much_worse_count": fn_not_much_worse,
        "seed_count": len(rows),
        "continue_to_larger_scale": bool(
            rl_lower_fp >= 2
            and reward_not_worse >= len(rows)
            and fn_not_much_worse >= len(rows)
        ),
    }
    return {
        "runs": rows,
        "aggregate": aggregate,
        "decision": decision,
    }


def parse_thresholds(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CE baseline and Pro RL on data/.cache.")
    parser.add_argument("--mode", choices=["ce", "rl", "both", "summary"], default="both")
    parser.add_argument("--config", default="config/default_config.toml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="data/.cache")
    parser.add_argument("--samples-per-class", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--decision-threshold", type=float, default=0.80)
    parser.add_argument("--sweep-thresholds", default="0.50,0.60,0.65,0.70,0.80,0.85,0.90")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-cache", action="store_true", default=True)
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--summary-dirs", default="")
    parser.add_argument("--output-json", default="reports/pro_rl_cache_compare.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "summary":
        summary_dirs = [item.strip() for item in args.summary_dirs.split(",") if item.strip()]
        if not summary_dirs:
            raise ValueError("--summary-dirs is required for --mode summary")
        summary = summarize_runs(summary_dirs)
        output_path = Path(args.output_json)
        save_json(output_path, summary)
        print(f"Summary saved to: {output_path}")
        print(json.dumps(summary["decision"], indent=2))
        return 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = resolve_config(args.config, fast_cache=args.fast_cache)
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    thresholds = parse_thresholds(args.sweep_thresholds)
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.output_json).with_suffix("")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Axon Pro RL Cache Compare")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Cache: {args.cache_dir}")
    print(f"Samples per class: {args.samples_per_class}")
    print(f"Epochs: {args.epochs}")
    print(f"Mode: {args.mode}")
    print(f"Run dir: {run_dir}")
    print(f"Decision threshold: {args.decision_threshold}")
    print(f"Feature shape: byte={config.max_byte_length}, pe={config.pe_feature_dim}, stat={config.stat_feature_dim}")

    dataset = FeatureCacheDataset(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        max_byte_length=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        max_samples_per_class=args.samples_per_class,
        axon_config=config,
    )
    train_dataset, val_dataset, test_dataset, split_doc = load_or_create_split(dataset, args, config)
    save_json(run_dir / "split.json", split_doc)
    split_info = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
        "split_file": args.split_file,
    }
    print(
        f"Split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    if not args.no_preload:
        train_dataset = preload_dataset(train_dataset, "train")
        val_dataset = preload_dataset(val_dataset, "val")
        test_dataset = preload_dataset(test_dataset, "test")

    train_loader = make_loader(train_dataset, args.batch_size, True, args.seed)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.seed)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.seed)

    env = MalwareBanditEnv(
        RewardConfig(
            true_negative_reward=1.0,
            true_positive_reward=1.0,
            false_positive_penalty=-2.0,
            false_negative_penalty=-1.2,
        )
    )

    base_config = copy.deepcopy(config)
    run_config = build_run_config(args, config, device, thresholds)
    ce_history = {}
    rl_history = {}
    ce_test = None
    rl_test = None
    ce_sweep = None
    rl_sweep = None

    if args.mode in {"ce", "both"}:
        ce_model = AxonMalwareModel(copy.deepcopy(base_config))
        print("\nTraining supervised CE baseline...")
        ce_history = train_supervised(
            ce_model,
            train_loader,
            val_loader,
            env,
            device,
            epochs=args.epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            decision_threshold=args.decision_threshold,
        )
        ce_probs, test_labels, ce_loss = collect_supervised_outputs(ce_model, test_loader, device)
        ce_test = metrics_from_probs(ce_probs, test_labels, env, args.decision_threshold)
        ce_test["loss"] = ce_loss
        ce_sweep = threshold_sweep_from_probs(ce_probs, test_labels, env, thresholds, loss=ce_loss)
        ce_report = build_method_report("ce", run_config, split_info, ce_history, ce_test, ce_sweep)
        save_json(run_dir / "ce_history.json", ce_history)
        save_json(run_dir / "ce_report.json", ce_report)
        save_supervised_checkpoint(run_dir / "ce_model.pt", ce_model, config, args, ce_history)

    if args.mode in {"rl", "both"}:
        rl_model = AxonMalwareModel(copy.deepcopy(base_config))
        print("\nTraining Pro RL policy...")
        rl_agent = AxonPolicyAgent(rl_model)
        rl_config = RLTrainingConfig(
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            entropy_coef=args.entropy_coef,
            decision_threshold=args.decision_threshold,
            device=args.device,
            log_interval=max(1, len(train_loader) // 2),
        )
        rl_trainer = PolicyGradientTrainer(rl_agent, env, rl_config)
        rl_history = rl_trainer.train(train_loader, val_loader)
        rl_probs, rl_labels = collect_rl_outputs(rl_trainer, test_loader)
        rl_test = metrics_from_probs(rl_probs, rl_labels, env, args.decision_threshold)
        rl_sweep = threshold_sweep_from_probs(rl_probs, rl_labels, env, thresholds)
        rl_report = build_method_report("rl", run_config, split_info, rl_history, rl_test, rl_sweep)
        save_json(run_dir / "rl_history.json", rl_history)
        save_json(run_dir / "rl_report.json", rl_report)
        save_rl_checkpoint(run_dir / "rl_model.pt", rl_trainer, config, args, rl_history)

    print("\n" + "=" * 60)
    print("Test Compare")
    print("=" * 60)
    print("method | reward | accuracy | precision | recall | f1 | fp_rate | fn_rate")
    for name, metrics in [("CE", ce_test), ("RL", rl_test)]:
        if metrics is None:
            continue
        print(
            f"{name} | {metrics['reward']:.4f} | {metrics['accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | "
            f"{metrics['false_positive_rate']:.4f} | {metrics['false_negative_rate']:.4f}"
        )

    print("\nThreshold Sweep")
    print("method | threshold | reward | accuracy | precision | recall | fp_rate | fn_rate")
    for method_name, sweep in [("CE", ce_sweep), ("RL", rl_sweep)]:
        if sweep is None:
            continue
        for threshold, metrics in sweep.items():
            print(
                f"{method_name} | {threshold} | {metrics['reward']:.4f} | "
                f"{metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
                f"{metrics['recall']:.4f} | {metrics['false_positive_rate']:.4f} | "
                f"{metrics['false_negative_rate']:.4f}"
            )

    report = {
        "config": run_config,
        "split": split_info,
        "ce_history": ce_history,
        "rl_history": rl_history,
        "test": {
            "ce": ce_test,
            "rl": rl_test,
        },
        "threshold_sweep": {
            "ce": ce_sweep,
            "rl": rl_sweep,
        },
    }

    output_path = Path(args.output_json)
    save_json(output_path, report)
    save_json(run_dir / "compare_report.json", report)
    print(f"\nReport saved to: {output_path}")
    print(f"Run artifacts saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
