#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩表：把语料未入 cache 的良性文件 + C 盘 EXE（Avast 已扫可信）加入 cache。

- 收集: (a) 从 name_index 推导的良性根目录里的全部文件（去重由内容 sha 完成）
        (b) C 盘常见位置的 .exe
- 每个候选: MZ 校验 + 大小 1KB-50MB → 内容 sha256 → 若 npz 已存在则跳过 → 提取特征写 npz(label=0)
- torch-free extractor → 16+ 进程 Pool。
- 完成后需重建 manifest（见 regenerate_manifest 步骤）。
"""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CACHE_DIR = PROJECT_ROOT / "data" / ".cache"

# 与 rebuild_raw_cache_parallel.py 相同的 cache 配置 hash
_CACHE_CONFIG_HASH = hashlib.md5(b"65536_49_1500_256_True_False").hexdigest()[:8]

# 可访问的良性根（E: 盘；G:/H: 盘已断开不可用）
BENIGN_ROOTS = [r"E:\Project\python\KoloVirusDetector_ML_V2-main\benign_samples\待加入白名单"]

C_EXE_ROOTS = [
    r"C:\Windows\System32",
    r"C:\Windows\SysWOW64",
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"C:\Windows",
]
C_SKIP_DIRS = {"winsxs", "servicing", "assembly", "installer", "prefetch", "temp", "winxs",
               "system32", "syswow64", "WinSxS", "Installer", "cache", "logs", "Resources"}


def _worker(args):
    fpath_str, label, cache_dir_str = args
    try:
        fpath = Path(fpath_str)
        if not fpath.exists() or fpath.stat().st_size < 1024 or fpath.stat().st_size > 50 * 1024 * 1024:
            return None
        with open(fpath, "rb") as f:
            if f.read(2) != b"MZ":
                return None
            f.seek(0)
            bdata = f.read()
        sha256 = hashlib.sha256(bdata).hexdigest().lower()
        target = Path(cache_dir_str) / f"{sha256[:32]}_{_CACHE_CONFIG_HASH}.npz"
        if target.exists():
            return ("dup", sha256)
        from kvd_features.extractor import ExtractionConfig, extract_all_features
        res = extract_all_features(str(fpath), config=ExtractionConfig())
        if res is not None and len(res) >= 4 and res[1] is not None and res[2] is not None:
            np.savez_compressed(
                target,
                byte_sequence=res[0],
                pe_features=res[1].astype(np.float32),
                stat_features=res[2].astype(np.float32),
                lightweight_features=res[3].astype(np.float32),
                label=int(label),
                source_sha256=sha256,
                raw_source_path=str(fpath),
            )
            return ("new", sha256)
        return ("fail", sha256)
    except Exception:
        return ("fail", "")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--smoke", type=int, default=None, help="冒烟：只处理前 N 个")
    args = parser.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 1) 良性根目录（E: 盘可访问；G:/H: 已断开）
    print(f"[roots] 良性根 {len(BENIGN_ROOTS)} 个")
    candidates = []
    for rp in BENIGN_ROOTS:
        if not os.path.isdir(rp):
            print(f"[warn] 良性根不可访问: {rp}")
            continue
        for dirpath, _d, files in os.walk(rp):
            for fn in files:
                candidates.append(os.path.join(dirpath, fn))
    print(f"[collect] 语料良性候选: {len(candidates):,}")

    # 3) C 盘 EXE
    c_exe = 0
    c_paths = []
    for root in C_EXE_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in C_SKIP_DIRS]
            for fn in files:
                if fn.lower().endswith(".exe"):
                    c_paths.append(os.path.join(dirpath, fn))
                    c_exe += 1
    print(f"[collect] C 盘 EXE 候选: {c_exe:,}")
    candidates.extend(c_paths)

    if args.smoke:
        candidates = candidates[: args.smoke]
    print(f"[total] 候选: {len(candidates):,}")

    # 4) 并行提取
    tasks = [(p, 0, str(CACHE_DIR)) for p in candidates]
    stats = Counter()
    with mp.Pool(processes=args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=64)):
            if r:
                stats[r[0]] += 1
            if (i + 1) % 2000 == 0 or (i + 1) == len(tasks):
                print(f"  {i+1:,}/{len(tasks):,}  {dict(stats)}  "
                      f"({(time.time()-t_start)/60:.1f}min)", flush=True)
    print(f"\n[done] {dict(stats)} 总耗时 {(time.time()-t_start)/60:.1f} min")
    print(f"新写入 npz: {stats.get('new', 0):,}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
