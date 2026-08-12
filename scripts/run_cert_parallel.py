#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""content_cert 特征提取编排：复用 v2/string 的切片文件，错峰启动 N×2 个 cert worker，拼接。

用法: python run_cert_parallel.py [--parts N]   (需先有 slice_val_*/slice_test_* 切片文件)
输出: reports/full_739k/content_cert/{val,test}.npy
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
PART_SCRIPT = str(PROJECT_ROOT / "scripts" / "extract_cert_part.py")
OUT = PROJECT_ROOT / "reports" / "full_739k" / "content_cert"
V2OUT = PROJECT_ROOT / "reports" / "full_739k" / "content_v2string"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", type=int, default=8)
    args = parser.parse_args()
    N = args.parts
    OUT.mkdir(parents=True, exist_ok=True)

    # 校验切片文件存在（由 run_v2string_parallel.py 产生）
    for kind in ("val", "test"):
        if not (V2OUT / f"slice_{kind}_00.txt").exists():
            raise SystemExit(f"slice_{kind}_00.txt missing; run run_v2string_parallel.py first")

    procs = []
    t0 = time.time()
    for kind in ("val", "test"):
        for k in range(N):
            slice_file = V2OUT / f"slice_{kind}_{k:02d}.txt"
            out_npy = OUT / f"{kind}_part{k:02d}.npy"
            log = open(OUT / f"{kind}_part{k:02d}.log", "w", encoding="utf-8")
            p = subprocess.Popen(
                [PY, "-u", PART_SCRIPT, "--slice", str(slice_file), "--out", str(out_npy)],
                stdout=log, stderr=subprocess.STDOUT)
            procs.append((kind, k, p, log))
            print(f"[launch] {kind} part {k} ({time.time()-t0:.0f}s)", flush=True)
            time.sleep(3)

    for kind, k, p, log in procs:
        p.wait()
        log.close()
        print(f"[{kind} part {k}] exit={p.returncode}", flush=True)

    for kind in ("val", "test"):
        parts = [np.load(OUT / f"{kind}_part{k:02d}.npy") for k in range(N)]
        arr = np.concatenate(parts)
        np.save(OUT / f"{kind}.npy", arr)
        print(f"[assemble] {kind}.npy {arr.shape}")

    print("[done]")


if __name__ == "__main__":
    main()
