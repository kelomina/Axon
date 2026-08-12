#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2/string 提取编排（内存安全版）：manifest + 名字索引本进程独占一次，
解析路径后写小切片文件，错峰启动 N×2 个 worker，拼接，写 meta。

用法: python run_v2string_parallel.py [--parts N]   (N 默认 8 → 16 个 worker)
输出: reports/full_739k/content_v2string/{val,test}.npy + meta.json
本脚本不 import torch（仅 numpy + 文件处理）。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"E:/Project/python/Axon_v2.6Exp")
PY = str(PROJECT_ROOT / "vnev" / "Scripts" / "python.exe")
PART_SCRIPT = str(PROJECT_ROOT / "scripts" / "extract_v2string_part.py")
OUT = PROJECT_ROOT / "reports" / "full_739k" / "content_v2string"
MANIFEST = PROJECT_ROOT / "data" / ".cache" / "manifest_a807341e.json"
NAME_IDX = PROJECT_ROOT / "reports" / "full_739k" / "name_index.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=int, default=8)
    args = parser.parse_args()
    N = args.parts
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 本进程独占加载 ----
    print("[orchestrator] loading manifest...")
    with open(MANIFEST, encoding="utf-8") as f:
        samples = json.load(f)["samples"]
    print(f"[orchestrator] {len(samples):,} samples")
    print("[orchestrator] loading name index...")
    import pickle
    with open(NAME_IDX, "rb") as f:
        name_idx = pickle.load(f)
    print(f"[orchestrator] {len(name_idx):,} names")

    # ---- 划分 ----
    labels = [s["label"] for s in samples]
    rng = np.random.RandomState(42)
    val_idx, test_idx = [], []
    for lab in (0, 1):
        idx = [i for i, l in enumerate(labels) if l == lab]
        rng.shuffle(idx)
        n_val = int(len(idx) * 0.10)
        n_test = int(len(idx) * 0.20)
        val_idx += idx[:n_val]
        test_idx += idx[n_val:n_val + n_test]
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    def resolve(indices):
        """为每个索引解析原始路径（空串=未定位）。"""
        out = []
        for i in indices:
            sha = samples[i]["source_sha256"]
            p = ""
            for ext in ("", ".exe", ".dll", ".sys"):
                hit = name_idx.get((sha + ext).casefold())
                if hit:
                    p = hit
                    break
            out.append(p)
        return out

    # ---- 写切片文件 + 启动 worker ----
    procs = []
    t0 = time.time()
    for kind, indices in (("val", val_idx), ("test", test_idx)):
        n = len(indices)
        for k in range(N):
            lo = n * k // N
            hi = n * (k + 1) // N
            part_paths = resolve(indices[lo:hi])
            slice_file = OUT / f"slice_{kind}_{k:02d}.txt"
            with open(slice_file, "w", encoding="utf-8") as f:
                f.write("\n".join(part_paths))
            located = sum(1 for p in part_paths if p)
            out_npy = OUT / f"{kind}_part{k:02d}.npy"
            log = open(OUT / f"{kind}_part{k:02d}.log", "w", encoding="utf-8")
            p = subprocess.Popen(
                [PY, "-u", PART_SCRIPT, "--slice", str(slice_file), "--out", str(out_npy)],
                stdout=log, stderr=subprocess.STDOUT)
            procs.append((kind, k, p, log))
            print(f"[launch] {kind} part {k}: {len(part_paths)} rows, located {located}, "
                  f"({time.time()-t0:.0f}s)", flush=True)
            time.sleep(3)

    # ---- 等待 ----
    for kind, k, p, log in procs:
        p.wait()
        log.close()
        print(f"[{kind} part {k}] exit={p.returncode}", flush=True)

    # ---- 拼接 ----
    for kind in ("val", "test"):
        parts = [np.load(OUT / f"{kind}_part{k:02d}.npy") for k in range(N)]
        arr = np.concatenate(parts)
        np.save(OUT / f"{kind}.npy", arr)
        print(f"[assemble] {kind}.npy {arr.shape}")

    with open(OUT / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"val_indices": val_idx.tolist(), "test_indices": test_idx.tolist(),
                   "v2_dim": 182, "string_dim": 43}, f)
    print(f"[meta] saved {OUT / 'meta.json'}")
    print("[done]")


if __name__ == "__main__":
    main()
