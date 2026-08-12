#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""813k 全量 cache 基座模型概率导出（良性扩充重训后）。

与 base_prob_739k.py 同逻辑，仅改 checkpoint 与输出目录：
- 模型：models/full_739k_benign/best_model_739k.pt（epoch 19, Test F1 0.9765）
- 数据：data/.cache 全部 813,098 样本（含新增 ~74k 良性），manifest 顺序
- 输出：reports/full_739k_benign/base_prob/

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

CHECKPOINT = PROJECT_ROOT / "models" / "full_739k_benign" / "best_model_739k.pt"
CACHE_DIR = PROJECT_ROOT / "data" / ".cache"
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "base_prob"
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
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT), help="模型 checkpoint 路径")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR), help="base_prob 输出目录（新目录可断点续跑）")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} (cuda_count={torch.cuda.device_count()})")
    print(f"[checkpoint] {ckpt_path.name}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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

    done_chunks = sorted(int(p.name.split("_")[1].split(".")[0]) for p in out_dir.glob("chunk_*.npy"))
    done_set = set(done_chunks)
    n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    first_undone = min((ci for ci in range(n_chunks) if ci not in done_set), default=n_chunks)
    print(f"[chunks] total={n_chunks}, already_done={len(done_set)}, first_undone={first_undone}")

    base_cache = dataset.cache_path_list
    base_sha = dataset.source_sha256_list
    base_label = dataset.label_list

    probs = np.zeros(total, dtype=np.float32)
    # 分块累积；已完成 chunk 立即落盘（防后台任务被杀丢失全部进度）
    chunk_data = {ci: [] for ci in range(n_chunks) if ci not in done_set}
    last_flushed = max(done_set) if done_set else -1

    def flush_upto(complete_chunk: int):
        nonlocal last_flushed
        for ci in range(last_flushed + 1, complete_chunk + 1):
            if ci not in done_set and chunk_data.get(ci) is not None:
                arr = np.full((CHUNK_SIZE,), np.nan, dtype=np.float32)
                for gi, p in chunk_data[ci]:
                    arr[gi - ci * CHUNK_SIZE] = p
                if ci == n_chunks - 1:
                    arr = arr[: total - ci * CHUNK_SIZE]
                np.save(out_dir / f"chunk_{ci:06d}.npy", arr)
                print(f"[chunk {ci}/{n_chunks}] saved {len(chunk_data[ci]):,} rows")
                chunk_data[ci] = None  # 释放内存
            last_flushed = ci

    batch_t0 = time.time()
    n_scored = 0
    with torch.no_grad():
        for step, (byte_seq, pe, stat, label, sub_idx) in enumerate(loader):
            # 整个 batch 都落在已完成 chunk 内 → 跳过前向（resume 智能续跑，不重算旧样本）
            if int(sub_idx[-1].item()) < first_undone * CHUNK_SIZE:
                continue
            byte_seq = byte_seq.to(device)
            pe = pe.to(device)
            stat = stat.to(device)
            out = model(byte_seq, pe, stat)
            p = torch.softmax(out["logits"], dim=1)[:, 1].cpu().numpy()
            max_gi = -1
            for b in range(len(sub_idx)):
                gi = int(sub_idx[b].item())
                probs[gi] = float(p[b])
                bucket = chunk_data.get(gi // CHUNK_SIZE)
                if bucket is not None:
                    bucket.append((gi, float(p[b])))
                max_gi = max(max_gi, gi)
                n_scored += 1
            # max_gi 之后的所有完整 chunk 立即落盘
            complete = (max_gi + 1) // CHUNK_SIZE - 1
            if complete > last_flushed:
                flush_upto(complete)
            if step % 100 == 0 and step > 0:
                print(f"  [{step} batches] {n_scored:,}/{total:,} "
                      f"({(time.time()-batch_t0):.0f}s elapsed)")

    # 收尾：保存剩余未完成的 chunk（含最后一个截断 chunk）
    flush_upto(n_chunks - 1)

    # 回填已完成 chunk 的 probs（resume 时被跳过的样本从磁盘取回），保证 meta.csv 完整
    for ci in sorted(done_set):
        arr = np.load(out_dir / f"chunk_{ci:06d}.npy")
        probs[ci * CHUNK_SIZE: ci * CHUNK_SIZE + len(arr)] = arr

    meta_path = out_dir / "meta.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "cache_path", "source_sha256", "label", "prob"])
        for i in range(total):
            w.writerow([i, base_cache[i], base_sha[i], base_label[i], float(probs[i])])
    print(f"[meta] saved {meta_path}")

    full = np.concatenate([np.load(out_dir / f"chunk_{ci:06d}.npy") for ci in range(n_chunks)])
    finite = np.isfinite(full).all()
    mean_p = float(full.mean())
    print(f"[verify] shape={full.shape} (expected {(total,)}), all_finite={finite}, mean_p={mean_p:.4f}")
    print(f"[done] {(time.time()-t_start)/3600:.2f} h")


if __name__ == "__main__":
    main()
