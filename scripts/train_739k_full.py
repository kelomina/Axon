#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Axon v2.6 - 全量 739k 缓存数据集重训练脚本。

数据划分: 训练集 70% / 验证集 10% / 测试集 20% (即 7:1:2)
数据源:   data/.cache/ 中的全部 .npz 特征矩阵文件
数据加载: 8 进程并行读取/解压 NPZ 缓存 (persistent_workers=True, pin_memory=True)
训练序列: 截断到 TRUNCATE_BYTE_LENGTH (4096)，实测单步 9.2s → 0.5s (约 18 倍加速)
"""

from __future__ import annotations

import dataclasses
import os

# 必须在 import numpy/torch 之前设置：Windows 下 8 个 spawn worker 各自初始化
# 多线程 OpenBLAS（每进程默认 32 线程）会导致线程栈内存分配失败并卡死训练。
# 数据加载是 I/O 密集，worker 内不需要多线程 BLAS，单线程足够。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── 路径设置 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import AxonExperimentConfig, TrainingConfig  # noqa: E402
from dataset import FeatureCacheDataset, create_stratified_split  # noqa: E402
from model import AxonMalwareModel  # noqa: E402
from trainer import AxonTrainer  # noqa: E402

# ── 超参数 ────────────────────────────────────────────────────────────────────
VAL_RATIO   = 0.10   # 验证集 10%
TEST_RATIO  = 0.20   # 测试集 20%
TRAIN_RATIO = 0.70   # 训练集 70% (剩余)

SEED        = 42
BATCH_SIZE  = 64          # 实测 RTX 4070 Laptop (8GB) 训练步峰值显存 ~2.3GB，64 留足余量
NUM_WORKERS = 8           # 8 个 CPU 进程并行预加载和解压 .npz 缓存
TRUNCATE_BYTE_LENGTH = 4096  # 训练序列长度：65536→4096 单步 9.2s→0.5s；保留 PE 头与入口代码上下文
MAX_EPOCHS  = 20          # 实测单步 ~0.5s，1 epoch ≈ 1.6h，20 epochs 上限 ≈ 32h，早停通常提前
LR          = 8e-5        # 对齐主链路 default_config.toml，降低 logits 溢出 NaN 风险
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 8
DEVICE      = "cuda"   # 若无 GPU 自动回退 CPU

OUTPUT_DIR  = PROJECT_ROOT / "models" / "full_739k"
REPORT_DIR  = PROJECT_ROOT / "reports" / "full_739k"

# ── 随机种子固定 ──────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class _TruncatedByteDataset(torch.utils.data.Dataset):
    """把 byte_seq 截断到固定长度的包装数据集。

    函数名称：_TruncatedByteDataset
    函数作用：包装任意返回 (byte_seq, pe, stat, label[, weight]) 元组的数据集，
              将 byte_seq 截断到 max_len。AxonMalwareModel 支持变长序列输入，
              截断后 DSRA 的 chunk 数从 128 (65536/512) 降到 8 (4096/512)，
              训练单步耗时实测从 9.2s 降到 0.5s（约 18 倍）。
    调用方：train_739k_full.main() 在分层划分之后对三个子数据集各包装一次。
    被调用方：_build_dataloaders 的 DataLoader 迭代（Windows spawn 下可 pickle：
              仅持有 base 数据集引用与整数常量，均为可序列化状态）。

    参数说明：
        - base: torch.utils.data.Dataset，被包装数据集；其 __getitem__ 返回的
          第一个元素必须是可索引的字节序列张量。
        - max_len: int，截断后的字节序列长度，默认 TRUNCATE_BYTE_LENGTH (4096)。

    返回值说明：
        - __getitem__: 与 base 返回结构一致，仅 byte_seq 被截断到 max_len；
          pe/stat/label（及可选的 sample_weight）原样透传。

    错误处理：base 样本第一元素不可切片时由张量切片操作抛出，不吞异常。
    副作用：无。并发与幂等：包装无共享可变状态，同一索引恒返回同一截断结果。
    """

    def __init__(self, base, max_len: int = TRUNCATE_BYTE_LENGTH):
        self.base = base
        self.max_len = int(max_len)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        byte_seq = item[0]
        if byte_seq.shape[0] > self.max_len:
            byte_seq = byte_seq[: self.max_len]
        return (byte_seq,) + tuple(item[1:])


def _build_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    seed: int = SEED,
):
    """构建训练/验证/测试三个 DataLoader。

    函数名称：_build_dataloaders
    函数作用：统一构造三个 DataLoader，启用多进程并行读取 NPZ 缓存（num_workers）、
              固定内存页（pin_memory）与常驻 worker（persistent_workers），
              避免每个 epoch 重复 spawn 进程带来的 I/O 阻塞。
    调用方：main()；单元测试 tests/test_train_739k_full.py 直接调用验证参数。
    被调用方：torch.utils.data.DataLoader（PyTorch 数据加载器）。

    参数说明：
        - train_dataset: torch.utils.data.Dataset，训练子数据集（SubDataset 包装），不可为空。
        - val_dataset: torch.utils.data.Dataset，验证子数据集。
        - test_dataset: torch.utils.data.Dataset，测试子数据集。
        - batch_size: int，批大小，默认 64。
        - num_workers: int，数据加载进程数，默认 8；Windows spawn 模式下数据集状态
          必须可 pickle（FeatureCacheDataset/SubDataset 均满足）。
        - pin_memory: bool，是否固定内存页；GPU 训练时应为 True，默认 True。
        - persistent_workers: bool，是否让 worker 跨 epoch 常驻，默认 True；
          仅在 num_workers > 0 时生效。
        - seed: int，训练 DataLoader 的 shuffle 随机种子，保证可复现。

    返回值说明：
        - Tuple[DataLoader, DataLoader, DataLoader]，依次为 (train_loader, val_loader,
          test_loader)；train_loader 带 shuffle=True 与 drop_last=True。

    错误处理：num_workers > 0 且 persistent_workers=True 时若数据集不可 pickle，
        子进程启动阶段会抛出 PicklingError，由 PyTorch 向上传播，不吞异常。

    副作用：创建三个 DataLoader 对象，无文件写入、无网络调用。
    并发与幂等：num_workers > 0 时 DataLoader 内部多进程并发读取磁盘 NPZ，
        各进程无共享可变状态；同一 seed 下 train_loader 的洗牌顺序可复现。
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=gen,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, test_loader


def _jsonable(value):
    """把 trainer 返回的指标对象递归转换成 JSON 可序列化结构。

    TrainingMetrics 是 dataclass（无 to_dict），直接 json.dump 会抛
    "Object of type TrainingMetrics is not JSON serializable"。
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def main():
    set_seed(SEED)

    print("=" * 70)
    print("Axon v2.6 - Full 739k Dataset Retrain  (70% Train / 10% Val / 20% Test)")
    print("=" * 70)

    # ── 配置 ──────────────────────────────────────────────────────────────────
    config = AxonExperimentConfig(
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
        device=DEVICE,
        model_save_dir=str(OUTPUT_DIR),
        experiment_name="axon_v26_full_739k",
    )

    train_config = TrainingConfig(
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        early_stopping_patience=EARLY_STOP_PATIENCE,
        lr_scheduler="cosine",
        warmup_epochs=3,
        # 数值稳定与类别不平衡 (benign:malware ≈ 1:3.45) 配置，对齐 config/default_config.toml：
        # 全量训练实测 loss 降到 ~0.08 后 logits 溢出 NaN，需 label_smoothing + focal + 保守梯度裁剪
        gradient_clip=0.75,
        label_smoothing=0.03,
        focal_gamma=1.0,
        focal_alpha=0.55,
        diversity_loss_weight=0.03,
        mixed_precision=False,  # DSRA 注意力机制在 FP16 模式下溢出产生 NaN，必须使用 FP32 纯单精度
        enable_swanlab=False,
        best_model_filename="best_model_739k.pt",
        final_model_filename="final_model_739k.pt",
    )

    device = torch.device("cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 加载全量 NPZ 缓存数据集 ───────────────────────────────────────────────
    cache_dir = PROJECT_ROOT / "data" / ".cache"
    data_dir  = PROJECT_ROOT / "data"

    print(f"\n[Dataset] Scanning feature cache: {cache_dir}")
    dataset = FeatureCacheDataset(
        data_dir=str(data_dir),
        cache_dir=str(cache_dir),
        max_byte_length=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        max_samples_per_class=None,   # 全量，不限制每类样本数
        axon_config=config,
    )
    print(f"[Dataset] Total samples loaded: {len(dataset):,}")

    label_arr = np.array(dataset.label_list)
    n_benign  = int((label_arr == 0).sum())
    n_malware = int((label_arr == 1).sum())
    print(f"[Dataset] Benign:  {n_benign:,}")
    print(f"[Dataset] Malware: {n_malware:,}")

    # ── 7:1:2 分层划分 ────────────────────────────────────────────────────────
    print(f"\n[Split] Stratified split: {int(TRAIN_RATIO*100)}% Train / "
          f"{int(VAL_RATIO*100)}% Val / {int(TEST_RATIO*100)}% Test")

    train_dataset, val_dataset, test_dataset = create_stratified_split(
        dataset,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
        axon_config=config,
    )

    print(f"[Split] Train samples: {len(train_dataset):,}")
    print(f"[Split] Val   samples: {len(val_dataset):,}")
    print(f"[Split] Test  samples: {len(test_dataset):,}")

    # ── 序列截断（加速训练）───────────────────────────────────────────────────
    print(f"[Truncate] Byte sequence truncated to {TRUNCATE_BYTE_LENGTH} bytes for training")
    train_dataset = _TruncatedByteDataset(train_dataset, TRUNCATE_BYTE_LENGTH)
    val_dataset = _TruncatedByteDataset(val_dataset, TRUNCATE_BYTE_LENGTH)
    test_dataset = _TruncatedByteDataset(test_dataset, TRUNCATE_BYTE_LENGTH)

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = _build_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        seed=SEED,
    )

    # ── 模型 ──────────────────────────────────────────────────────────────────
    print("\n[Model] Initializing AxonMalwareModel...")
    model = AxonMalwareModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters:     {total_params:,}")
    print(f"[Model] Trainable parameters: {trainable_params:,}")

    # ── 输出目录 ──────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 训练 ──────────────────────────────────────────────────────────────────
    print(f"\n[Trainer] Starting training for up to {MAX_EPOCHS} epochs...")
    print(f"[Trainer] LR={LR}, WD={WEIGHT_DECAY}, Batch={BATCH_SIZE}, EarlyStop={EARLY_STOP_PATIENCE}")
    print(f"[Trainer] AMP={train_config.mixed_precision}")
    print()

    trainer = AxonTrainer(model, config, train_config)
    t0 = time.time()

    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        fast_mode=False,
    )

    elapsed = time.time() - t0

    # ── 结果摘要 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"  Total time: {elapsed/3600:.2f} h  ({elapsed:.0f} s)")

    best_val = results.get("best_val_f1", results.get("val_f1", "N/A"))
    test_f1  = results.get("test_f1", "N/A")
    test_acc = results.get("test_accuracy", "N/A")

    print(f"  Best Val F1:  {best_val}")
    print(f"  Test F1:      {test_f1}")
    print(f"  Test Acc:     {test_acc}")

    # ── 保存结果报告 ──────────────────────────────────────────────────────────
    receipt = {
        "script":        "train_739k_full.py",
        "total_samples": len(dataset),
        "n_benign":      n_benign,
        "n_malware":     n_malware,
        "train_samples": len(train_dataset),
        "val_samples":   len(val_dataset),
        "test_samples":  len(test_dataset),
        "val_ratio":     VAL_RATIO,
        "test_ratio":    TEST_RATIO,
        "seed":          SEED,
        "batch_size":    BATCH_SIZE,
        "num_workers":   NUM_WORKERS,
        "truncate_byte_length": TRUNCATE_BYTE_LENGTH,
        "max_epochs":    MAX_EPOCHS,
        "learning_rate": LR,
        "elapsed_sec":   elapsed,
        "results":       _jsonable(results),
    }

    receipt_path = REPORT_DIR / "train_739k_receipt.json"
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    print(f"\n  Receipt saved: {receipt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
