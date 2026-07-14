#!/usr/bin/env python3
"""Axon Pro RL smoke test.

这个脚本只验证 Pro 强化学习分支是否能跑通：
1. 构建一个很小的 Axon 模型；
2. 生成合成 byte/PE/stat 输入；
3. 用 contextual bandit 奖励做轻量训练；
4. 确认前向、奖励、反向传播和评估都没有数值错误。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PRO_DIR = PROJECT_ROOT / "Pro"
sys.path.insert(0, str(PRO_DIR))
sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig  # noqa: E402
from model import AxonMalwareModel  # noqa: E402
from rl_axon import (  # noqa: E402
    AxonPolicyAgent,
    MalwareBanditEnv,
    PolicyGradientTrainer,
    RLTrainingConfig,
    RewardConfig,
    SyntheticMalwareDataset,
)


def build_smoke_model(device: str) -> AxonMalwareModel:
    """构建一个小模型，保证 smoke 测试几秒内能跑完。"""
    config = AxonExperimentConfig(
        max_byte_length=128,
        pe_feature_dim=256,
        stat_feature_dim=49,
        byte_embedding_dim=32,
        dsra_dim=32,
        dsra_heads=4,
        dsra_slots=32,
        dsra_read_topk=4,
        dsra_write_topk=2,
        dsra_local_window=32,
        dsra_chunk_size=64,
        pe_projection_dim=32,
        pe_projector_hidden_dim=64,
        classifier_hidden_dim=32,
        fusion_type="concat",
        dropout=0.05,
        pe_schema_version="fixed_v2",
        pe_fixed_section_slots=32,
        strict_pe_parsing=True,
        allow_pe_fallback=False,
        device=device,
    )
    return AxonMalwareModel(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Axon Pro RL smoke test.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decision-threshold", type=float, default=0.65)
    parser.add_argument("--sweep-thresholds", default="0.50,0.60,0.65,0.70,0.80")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    model = build_smoke_model(args.device)
    agent = AxonPolicyAgent(model)
    env = MalwareBanditEnv(
        RewardConfig(
            true_negative_reward=1.0,
            true_positive_reward=1.0,
            false_positive_penalty=-2.0,
            false_negative_penalty=-1.2,
        )
    )
    train_config = RLTrainingConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=5e-4,
        entropy_coef=0.005,
        decision_threshold=args.decision_threshold,
        device=args.device,
        log_interval=4,
    )

    dataset = SyntheticMalwareDataset(
        num_samples=args.samples,
        max_byte_length=model.config.max_byte_length,
        pe_feature_dim=model.config.pe_feature_dim,
        stat_feature_dim=model.config.stat_feature_dim,
        seed=args.seed,
    )
    train_size = int(len(dataset) * 0.75)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    trainer = PolicyGradientTrainer(agent, env, train_config)

    print("=" * 60)
    print("Axon Pro RL Smoke Test")
    print("=" * 60)
    print(f"Device: {trainer.device}")
    print(f"Samples: train={len(train_dataset)}, val={len(val_dataset)}")
    print(f"Reward config: {env.to_dict()}")
    print(f"Decision threshold: {train_config.decision_threshold:.2f}")
    print("=" * 60)

    history = trainer.train(train_loader, val_loader)
    final_metrics = trainer.evaluate(val_loader)
    sweep_thresholds = [
        float(item.strip())
        for item in args.sweep_thresholds.split(",")
        if item.strip()
    ]
    sweep = trainer.threshold_sweep(val_loader, sweep_thresholds)

    if not history:
        raise RuntimeError("RL history is empty")
    if not all(torch.isfinite(torch.tensor(value)) for value in final_metrics.values()):
        raise FloatingPointError(f"Non-finite final metrics: {final_metrics}")

    print("=" * 60)
    print("Final RL smoke metrics")
    print("=" * 60)
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")
    print()
    print("Threshold sweep")
    print("threshold | reward | accuracy | precision | recall | fp_rate | fn_rate")
    for threshold, metrics in sweep.items():
        print(
            f"{threshold} | {metrics['reward']:.4f} | {metrics['accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['false_positive_rate']:.4f} | {metrics['false_negative_rate']:.4f}"
        )
    print("[OK] RL smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
