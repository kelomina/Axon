#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基座难例重训：良性难例加权微调（从 best checkpoint 加载权重，不恢复优化器）。

背景（Stage-2 侧已近最优，FP 976→928）：基座系统性把 ~7% 真良性判恶（base_prob>=0.5），
误判画像=导入密集的合法 Windows 程序。基座字节模型对这些难例给 0.98-0.99。
本脚本在基座训练时对"基座判恶的真良性"施加更高 loss 权重（w = 1 + K*base_prob for 良性），
强迫基座学会这些难例是良性。

设计：
- 复用 train_739k_full 的 set_seed/_build_dataloaders/_TruncatedByteDataset/_jsonable；
- 难例权重 = 全量 base_prob（reports/full_739k_benign/base_prob/，index 对齐 dataset）；
  benign: w = 1 + K*base_prob；malware: w = 1；
- 注入：create_stratified_split 后用 SubDataset(train.base, train.indices, sample_weights) 重包装；
- 从 models/full_739k_benign/best_model_739k.pt 加载 model_state_dict（不恢复优化器/scheduler）；
- 微调级超参（LR 降、epochs 少、patience 大），输出到 full_739k_benign_hardneg/。

冒烟：python scripts/train_739k_benign_hardneg.py --smoke  （fast_mode 小样本验证管线）
正式：python -u scripts/train_739k_benign_hardneg.py [--k 5.0] [--epochs 8]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import glob
import time

import numpy as np
import torch

import train_739k_full as T  # noqa: E402

from config import AxonExperimentConfig, TrainingConfig  # noqa: E402
from dataset import FeatureCacheDataset, create_stratified_split, SubDataset  # noqa: E402
from model import AxonMalwareModel  # noqa: E402
from trainer import AxonTrainer  # noqa: E402

PRETRAINED = PROJECT_ROOT / "models" / "full_739k_benign" / "best_model_739k.pt"
BASE_PROB_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "base_prob"
OUTPUT_DIR = PROJECT_ROOT / "models" / "full_739k_benign_hardneg"
REPORT_DIR = PROJECT_ROOT / "reports" / "full_739k_benign_hardneg"

K_DEFAULT = 5.0
MAX_EPOCHS_DEFAULT = 8
LR_DEFAULT = 3e-5


def load_base_prob() -> np.ndarray:
    return np.concatenate(
        [np.load(f) for f in sorted(glob.glob(str(BASE_PROB_DIR / "chunk_*.npy")))]
    ).astype(np.float32)


def main() -> None:
    args = sys.argv[1:]
    smoke = "--smoke" in args
    do_compile = "--compile" in args
    compile_mode = "reduce-overhead"  # CUDA graph 捕获；benchmark_step_faithful 实测快于 dynamic
    k = K_DEFAULT
    epochs = MAX_EPOCHS_DEFAULT
    warmup = None  # None → 沿用 TrainingConfig 默认（1）；探针(1 epoch)时需 0 绕过校验
    batch_size = None  # None → 沿用 T.BATCH_SIZE（64）
    for a in args:
        if a.startswith("--k="):
            k = float(a.split("=", 1)[1])
        if a.startswith("--epochs="):
            epochs = int(a.split("=", 1)[1])
        if a.startswith("--mode="):
            compile_mode = a.split("=", 1)[1]
        if a.startswith("--warmup="):
            warmup = int(a.split("=", 1)[1])
        if a.startswith("--batch-size="):
            batch_size = int(a.split("=", 1)[1])
    print(f"[Args] k={k} max_epochs={epochs} smoke={smoke} compile={do_compile} "
          f"mode={compile_mode}  pretrained={PRETRAINED.name}")

    T.set_seed(T.SEED)

    config = AxonExperimentConfig(
        val_ratio=T.VAL_RATIO,
        test_ratio=T.TEST_RATIO,
        seed=T.SEED,
        device=T.DEVICE,
        model_save_dir=str(OUTPUT_DIR),
        experiment_name="axon_v26_full_739k_benign_hardneg",
    )
    train_config = TrainingConfig(
        learning_rate=LR_DEFAULT,
        weight_decay=T.WEIGHT_DECAY,
        max_epochs=epochs,
        batch_size=T.BATCH_SIZE if batch_size is None else batch_size,
        early_stopping_patience=6,
        lr_scheduler="cosine",
        warmup_epochs=1 if warmup is None else warmup,
        gradient_clip=0.75,
        label_smoothing=0.03,
        focal_gamma=1.0,
        focal_alpha=0.55,
        diversity_loss_weight=0.03,
        mixed_precision=False,
        enable_swanlab=False,
        best_model_filename="best_model_739k_hardneg.pt",
        final_model_filename="final_model_739k_hardneg.pt",
        best_metric="goal",
        best_metric_beta=5.0,
    )

    device = torch.device("cuda" if T.DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_dir = PROJECT_ROOT / "data" / ".cache"
    data_dir = PROJECT_ROOT / "data"
    dataset = FeatureCacheDataset(
        data_dir=str(data_dir),
        cache_dir=str(cache_dir),
        max_byte_length=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        max_samples_per_class=None,
        axon_config=config,
    )
    total = len(dataset)
    label_arr = np.array(dataset.label_list)
    n_benign = int((label_arr == 0).sum())
    print(f"[Dataset] {total:,} samples  benign={n_benign:,} malware={int((label_arr==1).sum()):,}")

    # 难例权重（index 对齐 dataset 顺序；base_prob 由同一 dataset 顺序生成）
    base_prob = load_base_prob()
    assert len(base_prob) == total, f"base_prob {len(base_prob)} != dataset {total}"
    w = np.ones(total, dtype=np.float32)
    ben = label_arr == 0
    w[ben] = 1.0 + k * base_prob[ben]
    n_weighted = int((w > 1.0).sum())
    w_max = float(w.max())
    print(f"[Weight] benign weighted (k={k}): {n_weighted:,}  max_w={w_max:.2f}")

    train_ds, val_ds, test_ds = create_stratified_split(
        dataset, val_ratio=T.VAL_RATIO, test_ratio=T.TEST_RATIO, seed=T.SEED, axon_config=config,
    )
    train_ds = SubDataset(train_ds.base_dataset, train_ds.indices, sample_weights=w[train_ds.indices])
    print(f"[Split] train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")

    train_ds = T._TruncatedByteDataset(train_ds, T.TRUNCATE_BYTE_LENGTH)
    val_ds = T._TruncatedByteDataset(val_ds, T.TRUNCATE_BYTE_LENGTH)
    test_ds = T._TruncatedByteDataset(test_ds, T.TRUNCATE_BYTE_LENGTH)

    if smoke:
        n_smoke = 200
        train_ds = SubDataset(train_ds, list(range(n_smoke)))
        val_ds = SubDataset(val_ds, list(range(n_smoke)))
        test_ds = SubDataset(test_ds, list(range(n_smoke)))
        print(f"[Smoke] subset: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader, val_loader, test_loader = T._build_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=T.BATCH_SIZE if batch_size is None else batch_size, num_workers=T.NUM_WORKERS,
        pin_memory=device.type == "cuda", persistent_workers=True, seed=T.SEED,
    )

    print("[Model] initializing...")
    model = AxonMalwareModel(config)
    if PRETRAINED.exists():
        ckpt = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        n_loaded = sum(p.numel() for p in model.parameters())
        print(f"[Model] loaded pretrained weights from {PRETRAINED.name} ({n_loaded:,} params)")
    else:
        print(f"[WARN] pretrained not found: {PRETRAINED} —— 从头训练")
    model.to(device)

    if do_compile:
        if compile_mode == "reduce-overhead":
            model = torch.compile(model, mode="reduce-overhead")
            print("[Model] torch.compile(mode=reduce-overhead / CUDA graph) enabled")
        elif compile_mode == "default":
            model = torch.compile(model)
            print("[Model] torch.compile(default static) enabled")
        else:
            model = torch.compile(model, dynamic=True)
            print("[Model] torch.compile(dynamic=True) enabled")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = AxonTrainer(model, config, train_config)
    t0 = time.time()
    print(f"[Trainer] max_epochs={epochs} lr={LR_DEFAULT} k={k} hardneg weighting")
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        fast_mode=False,
    )
    elapsed = time.time() - t0

    print("=" * 60)
    print("Training Complete")
    print(f"  elapsed={elapsed/3600:.2f} h")
    best_val = results.get("best_val_f1", results.get("val_f1", "N/A"))
    test_f1 = results.get("test_f1", "N/A")
    print(f"  Best Val F1: {best_val}")
    print(f"  Test F1:     {test_f1}")

    receipt = {
        "script": "train_739k_benign_hardneg.py",
        "k": k, "max_epochs": epochs, "lr": LR_DEFAULT,
        "pretrained": PRETRAINED.name,
        "n_total": total, "n_benign": n_benign,
        "n_weighted_benign": n_weighted, "max_w": w_max,
        "elapsed_sec": elapsed,
        "results": T._jsonable(results),
    }
    (REPORT_DIR / "train_hardneg_receipt.json").write_text(
        __import__("json").dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Receipt: {REPORT_DIR / 'train_hardneg_receipt.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
