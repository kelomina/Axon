#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基座 base_prob 全量分布体检：良性判恶是否系统性问题。

1) 全量 813k：label=0（良性）的 base_prob 分布（>0.5/>0.7/>0.9 计数）；
2) 按 train/val/test 分区看良性 base_prob 分布 → 判断是训练内错判（能力问题）
   还是 test 特有（分布外）；
3) 抽查 data/.cache 缓存 npz 的 label 现状（train 71 冲突 sha 是否已同步新标签）。
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

BASE_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "base_prob"
CONTENT_V1_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_pe_v1"
V2_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_v2string"


def load_chunks(d: Path) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "chunk_*.npy")))]).astype(np.float32)


def main() -> None:
    bp = load_chunks(BASE_DIR)
    with open(CONTENT_V1_DIR / "meta.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    label = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    print(f"[all] {len(bp)} rows, benign={int((label==0).sum())}, malware={int((label==1).sum())}")

    with open(V2_DIR / "meta.json", encoding="utf-8") as f:
        m = json.load(f)
    val_idx = set(int(x) for x in m["val_indices"])
    test_idx = set(int(x) for x in m["test_indices"])
    split = np.empty(len(bp), dtype=np.int64)
    split[:] = -1  # train
    for i in val_idx:
        split[i] = 0
    for i in test_idx:
        split[i] = 1

    # ---- 1) 全量良性 base_prob 分布 ----
    ben = (label == 0)
    bp_b = bp[ben]
    print(f"\n=== 全量良性 base_prob 分布（n={len(bp_b)}） ===")
    for lo, hi in [(.5, .7), (.7, .9), (.9, 1.0)]:
        n = int(((bp_b >= lo) & (bp_b < hi)).sum())
        print(f"  base_prob in [{lo:.1f},{hi:.1f}): {n}  ({n/max(len(bp_b),1)*100:.3f}%)")
    n50 = int((bp_b >= 0.5).sum())
    n90 = int((bp_b >= 0.9).sum())
    print(f"  >=0.5: {n50} ({n50/max(len(bp_b),1)*100:.3f}%)   >=0.9: {n90}")

    # ---- 2) 分区良性 base_prob ----
    names = ["train", "val", "test"]
    print(f"\n=== 分区良性 base_prob（label=0 样本） ===")
    for s, nm in [(0, "val"), (1, "test"), (-1, "train")]:
        mask = (label == 0) & (split == s)
        if mask.sum() == 0:
            continue
        b = bp[mask]
        gt5 = int((b >= 0.5).sum())
        gt9 = int((b >= 0.9).sum())
        print(f"  {nm:<6} n={int(mask.sum()):>7}  mean={b.mean():.4f}  p90={np.percentile(b,90):.4f}  "
              f">=0.5: {gt5} ({gt5/mask.sum()*100:.2f}%)  >=0.9: {gt9}")

    # ---- 3) data/.cache npz 抽查 ----
    print(f"\n=== data/.cache 缓存 label 抽查 ===")
    caches = sorted(glob.glob(str(PROJECT_ROOT / "data" / ".cache" / "*.npz")))
    if not caches:
        caches = sorted(glob.glob(str(PROJECT_ROOT / "data" / "**" / "*.npz"), recursive=True))
    print(f"  npz count: {len(caches)}")
    # 只列目录
    dirs = sorted(set(Path(c).parent for c in caches))
    for d in dirs[:20]:
        print(f"    {d}")


if __name__ == "__main__":
    main()
