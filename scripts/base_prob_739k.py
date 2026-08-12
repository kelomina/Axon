#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""739k 全量 cache 基座模型概率导出（Step-2，Stage-2 的 base_prob 输入）。

对 data/.cache 全部 738,983 样本按 manifest 顺序跑 best_model_739k.pt，
输出每样本 p_malicious（与训练同语义：4096 截断、threshold 无关）。
顺序与 content_pe_v1 提取（extract_content_739k.py）对齐 —— 二者都用 manifest 索引，
Stage-2 直接按 index 拼接 [base_prob, content_100]。

输出（reports/full_739k/base_prob/）：
  - chunk_{i:06d}.npy   每块 50,000 行 float32 = p_malicious（顺序对齐 manifest）
  - meta.csv            index, cache_path, source_sha256, label, prob
可断点续跑：已存在的 chunk 跳过。
GPU（约 1.7-2h @ batch64/workers8）。
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import AxonExperimentConfig  # noqa: E402
from dataset import FeatureCacheDataset  # noqa: E402
from model import AxonMalwareModel  # noqa: E402

CHECKPOINT = PROJECT_ROOT / "models" / "full_739k" / "best_model_739k.pt"
CACHE_DIR = PROJECT_ROOT / "data" / ".cache"
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "reports" / "full_739k" / "base_prob"
TRUNCATE_BYTE_LENGTH = 4096
BATCH_SIZE = 64
NUM_WORKERS = 8
CHUNK_SIZE = 50000


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


class _IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        return (*item, idx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None, help="冒烟测试：仅前 N 个")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} (cuda_count={torch.cuda.device_count()})")

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    raw_cfg = ckpt["config"]
    config = AxonExperimentConfig.from_dict(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    try:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass

    print("[dataset] loading full cache...")
    dataset = FeatureCacheDataset(
        data_dir=str(DATA_DIR),
        cache_dir=str(CACHE_DIR),
        max_byte_length=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        max_samples_per_class=None,
        axon_config=config,
    )
    total = len(dataset)
    if args.max_samples is not None:
        total = min(total, args.max_samples)
    print(f"[dataset] {total:,} samples (max_samples={args.max_samples})")

    idx_ds = _IndexedDataset(_TruncatedByteDataset(dataset, TRUNCATE_BYTE_LENGTH))
    if args.max_samples is not None:
        from torch.utils.data import Subset
        idx_ds = Subset(idx_ds, list(range(total)))
    loader = torch.utils.data.DataLoader(
        idx_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

    print("[model] loading checkpoint...")
    model = AxonMalwareModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    done_chunks = sorted(int(p.name.split("_")[1].split(".")[0]) for p in OUT_DIR.glob("chunk_*.npy"))
    done_set = set(done_chunks)
    n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"[chunks] total={n_chunks}, already_done={len(done_set)}")

    # 用 manifest 对齐的 cache_path/sha（dataset 顺序 == manifest 顺序）
    base_cache = dataset.cache_path_list
    base_sha = dataset.source_sha256_list
    base_label = dataset.label_list

    # 预分配 + 分块落盘
    probs = np.zeros(total, dtype=np.float32)
    chunk_data = {ci: [] for ci in range(n_chunks) if ci not in done_set}
    batch_t0 = time.time()
    n_scored = 0
    with torch.no_grad():
        for step, (byte_seq, pe, stat, label, sub_idx) in enumerate(loader):
            byte_seq = byte_seq.to(device)
            pe = pe.to(device)
            stat = stat.to(device)
            out = model(byte_seq, pe, stat)
            p = torch.softmax(out["logits"], dim=1)[:, 1].cpu().numpy()
            for b in range(len(sub_idx)):
                gi = int(sub_idx[b].item())
                probs[gi] = float(p[b])
                chunk_data[gi // CHUNK_SIZE].append((gi, float(p[b])))
                n_scored += 1
            if step % 100 == 0 and step > 0:
                print(f"  [{step} batches] {n_scored:,}/{total:,} "
                      f"({(time.time()-batch_t0):.0f}s elapsed)")

    # 写未完成的 chunk
    for ci, rows in chunk_data.items():
        arr = np.full((CHUNK_SIZE,), np.nan, dtype=np.float32)
        for gi, p in rows:
            arr[gi - ci * CHUNK_SIZE] = p
        if ci == n_chunks - 1:
            arr = arr[: total - ci * CHUNK_SIZE]
        np.save(OUT_DIR / f"chunk_{ci:06d}.npy", arr)
        print(f"[chunk {ci}/{n_chunks}] saved {len(rows):,} rows")

    # meta
    meta_path = OUT_DIR / "meta.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "cache_path", "source_sha256", "label", "prob"])
        for i in range(total):
            w.writerow([i, base_cache[i], base_sha[i], base_label[i], float(probs[i])])
    print(f"[meta] saved {meta_path}")

    # 校验
    full = np.concatenate([np.load(OUT_DIR / f"chunk_{ci:06d}.npy") for ci in range(n_chunks)])
    finite = np.isfinite(full).all()
    mean_p = float(full.mean())
    print(f"[verify] shape={full.shape} (expected {(total,)}), all_finite={finite}, mean_p={mean_p:.4f}")
    print(f"[done] {(time.time()-t_start)/3600:.2f} h")


if __name__ == "__main__":
    main()
