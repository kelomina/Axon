#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""813k cache 全量 content PE v1 特征提取（良性扩充重训后）。

与 extract_content_739k.py 同逻辑，仅改 OUT_DIR 到 benign 目录。
manifest_a807341e.json 现为 813,098 样本（新增 ~74k 良性追加在尾部）；
前 738,983 行与旧缓存顺序一致，故可复用旧 reports/full_739k/content_pe_v1/
的 chunk_000000~000013，仅需提取新增 chunk（14~16）。

输出（reports/full_739k_benign/content_pe_v1/）：
  - chunk_{i:06d}.npy   每块 50,000 x 100 float32（顺序对齐 manifest 索引；未定位行=全零）
  - meta.csv            index, cache_path, source_sha256, label, located, raw_path
可断点续跑：已存在的 chunk 文件自动跳过。
CPU-only。
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvd_features.content_pe_v1 import (  # noqa: E402
    CONTENT_PE_V1_FEATURE_NAMES,
    extract_content_pe_v1_features,
)

MANIFEST = PROJECT_ROOT / "data" / ".cache" / "manifest_a807341e.json"
OUT_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_pe_v1"
INDEX_PKL = PROJECT_ROOT / "reports" / "full_739k" / "name_index.pkl"
CHUNK_SIZE = 50000

BENIGN_ROOTS = [
    r"E:\Project\python\KoloVirusDetector_ML_V2-main\benign_samples\待加入白名单",
    r"G:\私人\良性文件\待加入白名单",
    r"H:\私人\良性文件",
]
MALWARE_ROOTS = [
    r"E:\Project\python\KoloVirusDetector_ML_V2-main\malicious_samples\待拉黑",
    r"G:\私人\恶意\MB\unziped",
    r"H:\私人\恶意\MB\unziped",
]
ALL_ROOTS = BENIGN_ROOTS + MALWARE_ROOTS

EXTENSIONS = ("", ".exe", ".dll", ".sys", ".bin")


def build_name_index(roots, progress_every=200_000):
    idx: dict = {}
    for root in roots:
        if not os.path.isdir(root):
            print(f"[index] skip (missing): {root}")
            continue
        t0 = time.time()
        n = 0
        for dirpath, _d, files in os.walk(root):
            for fn in files:
                idx.setdefault(fn.casefold(), os.path.join(dirpath, fn))
                n += 1
                if n % progress_every == 0:
                    print(f"  [{root}] {n:,} files, {len(idx):,} names, {time.time()-t0:.0f}s")
        print(f"[index] {root}: {n:,} files in {time.time()-t0:.0f}s")
    return idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--max-samples", type=int, default=None,
                        help="仅处理前 N 个样本（冒烟测试）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if INDEX_PKL.exists():
        print(f"[index] loading cached name index: {INDEX_PKL}")
        with open(INDEX_PKL, "rb") as f:
            name_idx = pickle.load(f)
        print(f"[index] loaded {len(name_idx):,} names")
    else:
        print("[index] building name index over raw trees...")
        name_idx = build_name_index(ALL_ROOTS)
        with open(INDEX_PKL, "wb") as f:
            pickle.dump(name_idx, f, protocol=4)
        print(f"[index] saved {len(name_idx):,} names to {INDEX_PKL}")

    import json
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    samples = manifest["samples"]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    total = len(samples)
    print(f"[manifest] {total:,} samples (max_samples={args.max_samples})")

    def locate(sha: str) -> str:
        for ext in EXTENSIONS:
            p = name_idx.get((sha + ext).casefold())
            if p:
                return p
        return ""

    n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    done_chunks = sorted(
        int(p.name.split("_")[1].split(".")[0]) for p in OUT_DIR.glob("chunk_*.npy")
    )
    done_set = set(done_chunks)
    print(f"[chunks] total={n_chunks}, already_done={len(done_set)}")

    from multiprocessing import Pool

    located_total = 0
    for ci in range(n_chunks):
        if ci in done_set:
            print(f"[chunk {ci}/{n_chunks}] skip (done)")
            continue
        lo = ci * CHUNK_SIZE
        hi = min(lo + CHUNK_SIZE, total)
        block = samples[lo:hi]
        t0 = time.time()

        raws = []
        located = 0
        for s in block:
            p = locate(s["source_sha256"])
            raws.append(p)
            if p:
                located += 1
        located_total += located

        mat = np.zeros((len(block), len(CONTENT_PE_V1_FEATURE_NAMES)), dtype=np.float32)
        if located:
            paths = [p for p in raws if p]
            feats = []
            with Pool(args.workers) as pool:
                for fv in pool.imap(extract_content_pe_v1_features, paths, chunksize=64):
                    feats.append(fv)
            j = 0
            for i, p in enumerate(raws):
                if p:
                    mat[i] = feats[j]
                    j += 1

        np.save(OUT_DIR / f"chunk_{ci:06d}.npy", mat)
        print(f"[chunk {ci}/{n_chunks}] {len(block):,} rows, located {located:,} "
              f"({located/len(block)*100:.1f}%), {time.time()-t0:.0f}s, "
              f"elapsed {(time.time()-t_start)/3600:.1f}h")

    meta_path = OUT_DIR / "meta.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "cache_path", "source_sha256", "label", "located", "raw_path"])
        for i, s in enumerate(samples):
            p = locate(s["source_sha256"])
            w.writerow([i, s["cache_path"], s["source_sha256"], s["label"], 1 if p else 0, p])
    print(f"[meta] saved {meta_path}")

    total_mat = np.zeros((0, len(CONTENT_PE_V1_FEATURE_NAMES)), dtype=np.float32)
    for ci in range(n_chunks):
        total_mat = np.concatenate([total_mat, np.load(OUT_DIR / f"chunk_{ci:06d}.npy")])
    print(f"[verify] matrix {total_mat.shape} (expected {(total, len(CONTENT_PE_V1_FEATURE_NAMES))}), "
          f"locate_rate={located_total/total*100:.2f}%")
    print(f"[done] total {(time.time()-t_start)/3600:.1f} h")


if __name__ == "__main__":
    main()
