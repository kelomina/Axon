#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""content_pe_v2(182) + content_string(43) 特征提取（仅 val+test，供 Stage-2 扩展）。

顺序与 create_stratified_split(seed42) 的 val_ds.indices / test_ds.indices 对齐。
输出 reports/full_739k/content_v2string/：
  - val.npy (73897, 225) / test.npy (147796, 225)   [v2+string 拼接]
  - meta.json: {"val_indices": [...], "test_indices": [...], "v2_dim":182, "string_dim":43}
CPU-only；可断点续跑（val/test 各一次，完成即跳过）。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_stage2_cache_matrix import (  # noqa: E402
    CONTENT_PE_V2_FEATURE_NAMES,
    CONTENT_STRING_FEATURE_NAMES,
    _content_pe_v2_features_from_path,
    _content_string_features_from_path,
)

MANIFEST = PROJECT_ROOT / "data" / ".cache" / "manifest_a807341e.json"
NAME_IDX = PROJECT_ROOT / "reports" / "full_739k" / "name_index.pkl"
OUT_DIR = PROJECT_ROOT / "reports" / "full_739k" / "content_v2string"
CHECKPOINT = PROJECT_ROOT / "models" / "full_739k" / "best_model_739k.pt"


def extract_v2string(path: str) -> np.ndarray:
    """模块顶层 worker：v2 + string 拼接（Windows spawn 必须可 pickle）。"""
    return np.concatenate([
        _content_pe_v2_features_from_path(Path(path)),
        _content_string_features_from_path(Path(path)),
    ]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    with open(NAME_IDX, "rb") as f:
        name_idx = pickle.load(f)
    import json as _json
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = _json.load(f)
    samples = manifest["samples"]

    # 划分（与训练一致）
    import torch
    from config import AxonExperimentConfig
    from dataset import FeatureCacheDataset, create_stratified_split
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    raw = ckpt["config"]
    config = AxonExperimentConfig.from_dict(raw) if isinstance(raw, dict) else raw
    ds = FeatureCacheDataset(
        data_dir=str(PROJECT_ROOT / "data"), cache_dir=str(PROJECT_ROOT / "data" / ".cache"),
        max_byte_length=config.max_byte_length, pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim, max_samples_per_class=None, axon_config=config)
    _, val_ds, test_ds = create_stratified_split(ds, val_ratio=0.10, test_ratio=0.20, seed=42, axon_config=config)
    val_idx = np.asarray(val_ds.indices, dtype=np.int64)
    test_idx = np.asarray(test_ds.indices, dtype=np.int64)

    V2DIM = len(CONTENT_PE_V2_FEATURE_NAMES)
    STRDIM = len(CONTENT_STRING_FEATURE_NAMES)
    DIM = V2DIM + STRDIM
    print(f"v2={V2DIM} string={STRDIM} total={DIM}, val={len(val_idx):,} test={len(test_idx):,}")

    def locate(sha):
        for ext in ("", ".exe", ".dll", ".sys"):
            p = name_idx.get((sha + ext).casefold())
            if p:
                return p
        return ""

    def process(indices, out_npy):
        if out_npy.exists():
            print(f"[skip] {out_npy} exists")
            return
        rows = [(i, locate(samples[i]["source_sha256"])) for i in indices]
        located = sum(1 for _, p in rows if p)
        print(f"[locate] {located}/{len(rows)}")
        mat = np.zeros((len(rows), DIM), dtype=np.float32)
        paths = [p for _, p in rows if p]
        from multiprocessing import Pool

        if paths:
            feats = []
            with Pool(args.workers) as pool:
                for k, fv in enumerate(pool.imap(extract_v2string, paths, chunksize=64)):
                    feats.append(fv)
                    if (k + 1) % 20000 == 0:
                        print(f"  extracted {k+1:,}/{len(paths):,} ({time.time()-t_start:.0f}s)")
            j = 0
            for k, (_, p) in enumerate(rows):
                if p:
                    mat[k] = feats[j]
                    j += 1
        np.save(out_npy, mat)
        print(f"[saved] {out_npy} {mat.shape}")

    process(val_idx, OUT_DIR / "val.npy")
    process(test_idx, OUT_DIR / "test.npy")

    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "val_indices": val_idx.tolist(), "test_indices": test_idx.tolist(),
            "v2_dim": V2DIM, "string_dim": STRDIM, "v2_names": CONTENT_PE_V2_FEATURE_NAMES,
            "string_names": CONTENT_STRING_FEATURE_NAMES,
        }, f)
    print(f"[meta] saved {OUT_DIR / 'meta.json'}")
    print(f"[done] {(time.time()-t_start)/3600:.2f} h")


if __name__ == "__main__":
    main()
