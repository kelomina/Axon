#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""向 cache manifest 追加新增良性样本（保持旧顺序，不重扫全量 npz）。

旧 manifest 738,983 条（顺序不变，既有特征矩阵仍对齐）；新样本从 npz 元数据读取追加。
用法: python append_manifest_benign.py
输出: data/.cache/manifest_a807341e.json（旧 + 新）
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / ".cache"
MANIFEST = CACHE_DIR / "manifest_a807341e.json"


def read_meta(name):
    """模块顶层 worker：读单个 npz 元数据（Windows spawn 需可 pickle）。"""
    p = CACHE_DIR / name
    try:
        d = np.load(p, allow_pickle=False)
        label = int(d["label"])
        sha = str(d["source_sha256"])
    except Exception:
        return None
    return {
        "source_path": str(p),
        "cache_path": str(p),
        "label": label,
        "source_sha256": sha,
        "allow_missing_source_sha256": False,
    }


def main():
    t0 = time.time()
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    old_samples = manifest["samples"]
    old_shas32 = {s["source_sha256"][:32] for s in old_samples}
    print(f"旧 manifest: {len(old_samples):,} 条")

    # 列出 cache 中全部 npz（文件名 = <sha32>_a807341e.npz）
    npz_names = [fn for fn in os.listdir(CACHE_DIR) if fn.endswith(".npz")]
    print(f"cache npz: {len(npz_names):,}")

    # 新 npz = 前缀不在旧 sha32 中
    new_names = [fn for fn in npz_names if fn.split("_")[0] not in old_shas32]
    print(f"新 npz: {len(new_names):,}")

    from multiprocessing import Pool
    new_samples = []
    with Pool(min(16, os.cpu_count() or 1)) as pool:
        for k, meta in enumerate(pool.imap(read_meta, new_names, chunksize=256)):
            if meta is not None:
                new_samples.append(meta)
            if (k + 1) % 10000 == 0:
                print(f"  read {k+1:,}/{len(new_names):,} ({(time.time()-t0)/60:.1f}min)", flush=True)
    print(f"新样本读取: {len(new_samples):,}")

    # 追加 + 写回
    manifest["samples"] = old_samples + new_samples
    manifest["total_samples"] = len(manifest["samples"])
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    print(f"[done] manifest 现 {len(manifest['samples']):,} 条 ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
