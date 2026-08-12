#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""739k 测试集逐样本评分 + FP 提取。

用途：对 full_739k 训练产出的 best 检查点，在 test 划分（7:1:2 的 20%）上
复现训练时评估，输出逐样本概率，并提取阈值 0.50 下的 FP（true_label=0, pred=1），
供后续 FP 拆解（Authenticode / 跨树冲突 / 噪声）使用。

数据管线与 scripts/train_739k_full.py 完全一致：
  FeatureCacheDataset(全量 cache) -> create_stratified_split(0.10/0.20, seed=42)
  -> _TruncatedByteDataset(4096) -> DataLoader(batch 64, workers 8, pin_memory)
GPU 占用注意：与其它 GPU 任务共存时请用小 batch 或等待空闲。
"""

from __future__ import annotations

import os
import sys

# Windows spawn 多进程加载必须限制 BLAS 线程（必须在 import numpy/torch 之前）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import AxonExperimentConfig  # noqa: E402
from dataset import FeatureCacheDataset, create_stratified_split  # noqa: E402
from model import AxonMalwareModel  # noqa: E402

CHECKPOINT = PROJECT_ROOT / "models" / "full_739k" / "best_model_739k.pt"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "full_739k"
CACHE_DIR = PROJECT_ROOT / "data" / ".cache"
DATA_DIR = PROJECT_ROOT / "data"

TRUNCATE_BYTE_LENGTH = 4096
BATCH_SIZE = 64
NUM_WORKERS = 8
SEED = 42
DECISION_THRESHOLD = 0.50


class _TruncatedByteDataset(torch.utils.data.Dataset):
    """把 byte_seq 截断到固定长度的包装数据集（与 train_739k_full.py 同构）。"""

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
    """给样本追加子索引，便于映射回 base 的 cache_path/source_sha256。"""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        return (*item, idx)


def load_checkpoint_config(ckpt) -> AxonExperimentConfig:
    raw = ckpt["config"]
    if isinstance(raw, dict):
        return AxonExperimentConfig.from_dict(raw)
    return raw


def main() -> None:
    print("=" * 70)
    print("739k test split per-sample scoring + FP extraction")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  (cuda count={torch.cuda.device_count()})")

    t0 = time.time()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = load_checkpoint_config(ckpt)
    try:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    print(f"Checkpoint config: pe={config.pe_feature_dim} stat={config.stat_feature_dim} "
          f"max_byte={config.max_byte_length} schema={config.pe_schema_version}")

    # ---- 数据管线（与训练一致）----
    print("\n[Dataset] Scanning feature cache...")
    dataset = FeatureCacheDataset(
        data_dir=str(DATA_DIR),
        cache_dir=str(CACHE_DIR),
        max_byte_length=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        max_samples_per_class=None,
        axon_config=config,
    )
    print(f"[Dataset] Total samples: {len(dataset):,}")

    _, _, test_ds = create_stratified_split(
        dataset, val_ratio=0.10, test_ratio=0.20, seed=SEED, axon_config=config
    )
    test_base_idx = np.asarray(test_ds.indices, dtype=np.int64)
    base_cache = dataset.cache_path_list
    base_sha = dataset.source_sha256_list
    base_label = dataset.label_list
    print(f"[Split] Test samples: {len(test_ds):,}")

    test_dataset = _IndexedDataset(_TruncatedByteDataset(test_ds, TRUNCATE_BYTE_LENGTH))
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

    # ---- 模型 ----
    print("\n[Model] Loading checkpoint...")
    model = AxonMalwareModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"[Model] params={sum(p.numel() for p in model.parameters()):,}")

    # ---- 推理 ----
    rows = []
    fp_rows = []
    n = len(loader)
    batch_t0 = time.time()
    with torch.no_grad():
        for step, (byte_seq, pe, stat, label, sub_idx) in enumerate(loader):
            byte_seq = byte_seq.to(device)
            pe = pe.to(device)
            stat = stat.to(device)
            out = model(byte_seq, pe, stat)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            lbl = label.numpy()
            for b in range(len(sub_idx)):
                si = int(sub_idx[b].item())
                bi = int(test_base_idx[si])
                p = float(probs[b])
                pred = 1 if p >= DECISION_THRESHOLD else 0
                true = int(lbl[b])
                rec = {
                    "cache_path": str(base_cache[bi]),
                    "source_sha256": str(base_sha[bi]),
                    "true_label": true,
                    "prob_malicious": p,
                    "pred": pred,
                }
                rows.append(rec)
                if true == 0 and pred == 1:
                    fp_rows.append(rec)
            if step % 50 == 0 and step > 0:
                el = time.time() - batch_t0
                print(f"  [{step}/{n}] {el:.0f}s elapsed ({len(rows):,}/{len(test_ds):,} scored)")
    print(f"[Eval] scored {len(rows):,} in {time.time() - batch_t0:.0f}s")

    # ---- 指标复算（应复现 test F1≈0.9795）----
    tp = sum(1 for r in rows if r["pred"] == 1 and r["true_label"] == 1)
    tn = sum(1 for r in rows if r["pred"] == 0 and r["true_label"] == 0)
    fp = sum(1 for r in rows if r["pred"] == 1 and r["true_label"] == 0)
    fn = sum(1 for r in rows if r["pred"] == 0 and r["true_label"] == 1)
    acc = (tp + tn) / len(rows)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"\n[Check] TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"[Check] Acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}  (reported: F1 0.9795)")

    # ---- 保存 ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = OUTPUT_DIR / "test739k_scores.csv"
    fp_path = OUTPUT_DIR / "test739k_fp_list.csv"
    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(fp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fp_rows[0].keys())
        w.writeheader()
        w.writerows(fp_rows)
    print(f"[Saved] {scores_path} ({len(rows):,} rows)")
    print(f"[Saved] {fp_path} ({len(fp_rows):,} FPs)")
    print(f"Total wall: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
