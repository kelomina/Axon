#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据加载测速（CPU 端，不占 GPU）。

与 train_739k_benign_hardneg.py 完全相同的 dataset/dataloader 构造，
只迭代取 batch（不做 forward），测每 batch 纯加载耗时。

若加载耗时 ≈ 训练单步耗时(0.47s)，则瓶颈在数据层而非算子；
若明显小于 0.47s，则瓶颈在 GPU 算子/kernel 启动。

用法: python scripts/profile_loader_speed.py [--batch 64] [--workers 8] [--n 200]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from config import AxonExperimentConfig  # noqa: E402
from dataset import FeatureCacheDataset, create_stratified_split  # noqa: E402

TRUNCATE_BYTE_LENGTH = 4096
CKPT = PROJECT_ROOT / "models" / "full_739k_benign" / "best_model_739k.pt"


class _TruncatedByteDataset(torch.utils.data.Dataset):
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


def main() -> None:
    args = [a for a in sys.argv[1:]]
    batch_size = 64
    num_workers = 8
    n = 200
    for a in args:
        if a.startswith("--batch="):
            batch_size = int(a.split("=", 1)[1])
        if a.startswith("--workers="):
            num_workers = int(a.split("=", 1)[1])
        if a.startswith("--n="):
            n = int(a.split("=", 1)[1])

    print(f"[profile] batch={batch_size} workers={num_workers} n={n}")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    raw_cfg = ckpt["config"]
    config = AxonExperimentConfig.from_dict(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg

    dataset = FeatureCacheDataset(
        data_dir=str(PROJECT_ROOT / "data"),
        cache_dir=str(PROJECT_ROOT / "data" / ".cache"),
        max_byte_length=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        max_samples_per_class=None,
        axon_config=config,
    )
    print(f"[dataset] {len(dataset):,} samples  max_byte_length={config.max_byte_length}")

    train_ds, _, _ = create_stratified_split(
        dataset, val_ratio=0.10, test_ratio=0.20, seed=42, axon_config=config,
    )
    train_ds = _TruncatedByteDataset(train_ds, TRUNCATE_BYTE_LENGTH)
    print(f"[train] {len(train_ds):,} samples -> truncate {TRUNCATE_BYTE_LENGTH}")

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        persistent_workers=True, pin_memory=False, drop_last=True,
    )

    # warmup
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i == 5:
            t_warm = time.time() - t0
            print(f"[warmup] first {i+1} batches: {t_warm:.2f}s")
            t0 = time.time()
            break

    times = []
    t0 = time.time()
    for i, (byte_seq, pe, stat, label) in enumerate(loader):
        times.append(time.time() - t0)
        t0 = time.time()
        if i >= n - 1:
            break
    arr = [t for t in times[1:] if t > 0]
    if arr:
        import numpy as np
        a = np.array(arr)
        print(f"[result] {len(arr)} batches: mean={a.mean():.4f}s  "
              f"p50={np.median(a):.4f}  p90={np.percentile(a, 90):.4f}  "
              f"max={a.max():.4f}")
        print(f"[estimate] per-epoch loader time = {a.mean()*569170/batch_size/3600:.2f} h")
        print(f"[compare] training step is ~0.47s; loader-only {a.mean():.3f}s "
              f"= {a.mean()/0.47*100:.0f}% of step")
    print("[done]")


if __name__ == "__main__":
    main()
