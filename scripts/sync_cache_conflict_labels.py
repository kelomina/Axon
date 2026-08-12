#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同步 data/.cache 冲突 sha 的 label 为最终判定值。

基座训练（train_739k_full.py）直接读 data/.cache/<hash>.npz 的 label 字段。
100 冲突 sha 物理归位后 meta.csv 已更新，但 cache npz 仍是旧 label —— 基座重训前必须同步，
否则训练数据仍把同一 sha 标成污染旧值。

依据：
  - reports/full_739k_benign/base_prob/meta.csv   index -> cache_path
  - move_plan_preview.csv                         index -> new_label（最终判定）

只改 label 字段，其他数组原样保留；文件名（sha+config hash）不变；tmp+os.replace 原子写。
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_META = PROJECT_ROOT / "reports" / "full_739k_benign" / "base_prob" / "meta.csv"
MOVE_PLAN = PROJECT_ROOT / "reports" / "full_739k_benign" / "label_governance" / "move_plan_preview.csv"


def main() -> None:
    with open(BASE_META, encoding="utf-8") as f:
        meta = {r["index"]: r for r in csv.DictReader(f)}
    with open(MOVE_PLAN, encoding="utf-8-sig") as f:
        plan = {r["index"]: r for r in csv.DictReader(f)}

    synced = []
    skipped_missing = 0
    for idx in sorted(plan, key=int):
        row = plan[idx]
        new_label = int(row["new_label"])
        if idx not in meta:
            print(f"[SKIP] idx={idx} not in base_prob meta")
            continue
        p = meta[idx]["cache_path"]
        if not os.path.exists(p):
            print(f"[MISSING] idx={idx} {p}")
            skipped_missing += 1
            continue
        with np.load(p, allow_pickle=False) as d:
            data = {k: d[k] for k in d.files}
        old = int(data["label"])
        if old == new_label:
            print(f"[same] idx={idx} label={old}")
            continue
        data["label"] = np.asarray(new_label, dtype=np.int64)
        tmp = p + ".tmp.npz"  # np.savez 会为无 .npz 后缀路径补后缀，故 tmp 本身用 .npz
        np.savez(tmp, **data)
        os.replace(tmp, p)
        synced.append((idx, old, new_label))
        print(f"[sync] idx={idx} label {old}->{new_label}  {Path(p).name[:40]}")

    print(f"\n=== summary ===")
    print(f"synced={len(synced)}  already_ok={len(plan)-len(synced)-skipped_missing}  missing={skipped_missing}")
    if synced:
        n_1to0 = sum(1 for _, o, n in synced if o == 1 and n == 0)
        n_0to1 = sum(1 for _, o, n in synced if o == 0 and n == 1)
        print(f"  1->0 (恶->良): {n_1to0}   0->1 (良->恶): {n_0to1}")


if __name__ == "__main__":
    main()
