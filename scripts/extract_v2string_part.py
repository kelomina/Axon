#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2/string 特征提取单 part worker（内存安全版）。

用法: python extract_v2string_part.py --slice <paths.txt> --out <out.npy>
  paths.txt：每行一个原始文件路径；空行 = 未定位（输出零向量）。行数 = 本 part 样本数。
worker 不加载 manifest / 名字索引；主线程 import torch 一次（无 multiprocessing）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", required=True, help="paths.txt（一行一路径，空行=未定位）")
    parser.add_argument("--out", required=True, help="输出 .npy")
    args = parser.parse_args()
    t_start = time.time()

    with open(args.slice, encoding="utf-8") as f:
        paths = [line.rstrip("\n") for line in f]
    print(f"[slice] {len(paths)} rows from {args.slice}")

    from train_stage2_cache_matrix import (  # noqa: E402
        CONTENT_PE_V2_FEATURE_NAMES,
        CONTENT_STRING_FEATURE_NAMES,
        _content_pe_v2_features_from_path,
        _content_string_features_from_path,
    )
    DIM = len(CONTENT_PE_V2_FEATURE_NAMES) + len(CONTENT_STRING_FEATURE_NAMES)

    mat = np.zeros((len(paths), DIM), dtype=np.float32)
    for k, p in enumerate(paths):
        if p:
            mat[k] = np.concatenate([
                _content_pe_v2_features_from_path(Path(p)),
                _content_string_features_from_path(Path(p)),
            ]).astype(np.float32)
        if (k + 1) % 1000 == 0:
            print(f"  {k+1}/{len(paths)} ({time.time()-t_start:.0f}s)")
    np.save(args.out, mat)
    print(f"[saved] {args.out} {mat.shape}  ({(time.time()-t_start)/60:.1f} min)")


if __name__ == "__main__":
    main()
