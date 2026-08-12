#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 data/.cache npz 结构与 index→cache_path 映射（只读）。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BASE_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "base_prob"
MOVE_PLAN = PROJECT_ROOT / "reports" / "full_739k_benign" / "label_governance" / "move_plan_preview.csv"


def main() -> None:
    with open(BASE_DIR / "meta.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[base_prob meta] {len(rows)} rows")
    print(f"  header: {list(rows[0].keys())}")
    for i in [0, 1, 100, len(rows) - 1]:
        r = rows[i]
        print(f"  idx={r['index']} label={r['label']} prob={float(r['prob']):.4f} cache={r['cache_path'][:80]}")

    # 冲突 sha 的 index
    with open(MOVE_PLAN, encoding="utf-8-sig") as f:
        mv = list(csv.DictReader(f))
    print(f"\n[move_plan] {len(mv)} conflicts")
    print(f"  header: {list(mv[0].keys())}")
    conflict_idx = [int(r["index"]) for r in mv if r["index"].strip()]
    print(f"  conflict indices: {len(conflict_idx)}  sample={conflict_idx[:5]}")

    # 检查 cache npz 结构（取第一个冲突 index 对应的 npz）
    idx_by_text = {r["index"]: r for r in rows}
    sample = None
    for i in conflict_idx:
        r = idx_by_text.get(str(i))
        if r:
            sample = r
            break
    if sample:
        p = Path(sample["cache_path"])
        if not p.is_absolute():
            p = PROJECT_ROOT / "data" / ".cache" / p.name
        print(f"\n[sample npz] idx={sample['index']} path={p}")
        print(f"  exists={p.exists()}")
        if p.exists():
            with np.load(p, allow_pickle=False) as d:
                print(f"  keys: {list(d.files)}")
                for k in d.files:
                    a = d[k]
                    print(f"    {k}: shape={a.shape} dtype={a.dtype}"
                          + (f" val={a.flat[0]}" if a.ndim == 0 or a.size == 1 else ""))
        else:
            print("  MISSING!")

    # cache 目录文件命名样例
    cache_dir = PROJECT_ROOT / "data" / ".cache"
    import glob
    fs = sorted(glob.glob(str(cache_dir / "*.npz")))[:3]
    for f_ in fs:
        print(f"[cache file] {Path(f_).name}")


if __name__ == "__main__":
    main()
