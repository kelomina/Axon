#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""739k Stage-2 叠加层：base_prob + content_pe_v1 -> GBM -> 最终判定。

依赖（顺序对齐 manifest 索引）：
  - base_prob/ : base_prob_739k.py 输出（chunk_*.npy + meta.csv）
  - content_pe_v1/ : extract_content_739k.py 输出（chunk_*.npy + meta.csv）
划分：create_stratified_split(seed=42, 0.1/0.2) 与训练完全一致。
协议：Stage-2 GBM 在 VAL(OOF) 上训练+选阈值，在 TEST(OOF) 上评估一次（与 main.py 阈值策略一致）。

用法：& vnev/Scripts/python.exe -u scripts/stage2_739k.py
输出：reports/full_739k/stage2_report.json
CPU-only。
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

BASE_DIR = PROJECT_ROOT / "reports" / "full_739k" / "base_prob"
CONTENT_DIR = PROJECT_ROOT / "reports" / "full_739k" / "content_pe_v1"
OUT_JSON = PROJECT_ROOT / "reports" / "full_739k" / "stage2_report.json"
CHECKPOINT = PROJECT_ROOT / "models" / "full_739k" / "best_model_739k.pt"


def load_chunked(dir_path: Path) -> np.ndarray:
    files = sorted(glob.glob(str(dir_path / "chunk_*.npy")))
    if not files:
        raise SystemExit(f"no chunks in {dir_path}")
    return np.concatenate([np.load(f) for f in files])


def main() -> None:
    t0 = time.time()
    print("=== Stage-2 739k: base_prob + content_pe_v1 ===")

    if not (BASE_DIR / "chunk_000000.npy").exists():
        raise SystemExit("base_prob not ready; run base_prob_739k.py first")

    # ---- 加载 ----
    print("[load] base_prob...")
    base_prob = load_chunked(BASE_DIR).astype(np.float32)
    print(f"[load] base_prob {base_prob.shape}")
    print("[load] content...")
    content = load_chunked(CONTENT_DIR).astype(np.float32)
    print(f"[load] content {content.shape}")

    with open(CONTENT_DIR / "meta.csv", encoding="utf-8") as f:
        import csv
        labels = np.asarray([int(r["label"]) for r in csv.DictReader(f)], dtype=np.int64)
    total = len(labels)
    print(f"[load] labels {labels.shape}, positive_rate={labels.mean():.3f}")

    # ---- 划分（与训练一致，重建 FeatureCacheDataset + create_stratified_split）----
    from config import AxonExperimentConfig
    from dataset import FeatureCacheDataset, create_stratified_split

    print("[split] rebuilding stratified split (seed 42, 0.1/0.2)...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    raw_cfg = ckpt["config"]
    config = AxonExperimentConfig.from_dict(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    ds = FeatureCacheDataset(
        data_dir=str(PROJECT_ROOT / "data"),
        cache_dir=str(PROJECT_ROOT / "data" / ".cache"),
        max_byte_length=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        max_samples_per_class=None,
        axon_config=config,
    )
    _, val_ds, test_ds = create_stratified_split(
        ds, val_ratio=0.10, test_ratio=0.20, seed=42, axon_config=config
    )
    val_idx = np.asarray(val_ds.indices, dtype=np.int64)
    test_idx = np.asarray(test_ds.indices, dtype=np.int64)
    print(f"[split] val={len(val_idx):,} test={len(test_idx):,}")

    # ---- Stage-2 特征 ----
    def feats(idx):
        return np.column_stack([base_prob[idx], content[idx]]).astype(np.float32)

    Xv, yv = feats(val_idx), labels[val_idx]
    Xt, yt = feats(test_idx), labels[test_idx]
    print(f"[feat] Xv={Xv.shape} Xt={Xt.shape}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score

    def train_and_eval(Xtr, ytr, Xval, yval, Xtest, ytest, name):
        clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, random_state=0,
        )
        clf.fit(Xtr, ytr)
        pv = clf.predict_proba(Xval)[:, 1]
        # VAL 上选阈值
        best = (0.0, 0.5)
        for t in [x / 100 for x in range(20, 90)]:
            f1 = f1_score(yval, (pv >= t).astype(int))
            if f1 > best[0]:
                best = (f1, t)
        thr = best[1]
        pt = clf.predict_proba(Xtest)[:, 1]
        pred = (pt >= thr).astype(int)
        tp = int(((pred == 1) & (ytest == 1)).sum())
        fp = int(((pred == 1) & (ytest == 0)).sum())
        fn = int(((pred == 0) & (ytest == 1)).sum())
        tn = int(((pred == 0) & (ytest == 0)).sum())
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1t = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        print(f"[{name}] val_bestF1={best[0]:.5f}@t={thr:.2f} | TEST: "
              f"F1={f1t:.5f} P={p_:.4f} R={r_:.4f} errors={fp+fn} (TP={tp} FP={fp} FN={fn} TN={tn})")
        return {"name": name, "threshold": thr, "val_best_f1": best[0],
                "test_f1": f1t, "precision": p_, "recall": r_,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn, "errors": fp + fn}

    # 一次性算好特征矩阵
    base_v = base_prob[val_idx].reshape(-1, 1)
    base_t = base_prob[test_idx].reshape(-1, 1)
    Xv, Xt = feats(val_idx), feats(test_idx)

    results = []
    # 参考：base_prob 单特征（重放纯阈值）
    results.append(train_and_eval(base_v, yv, base_v, yv, base_t, yt, "base_prob only"))
    # 主模型：base_prob + content
    results.append(train_and_eval(Xv, yv, Xv, yv, Xt, yt, "base_prob + content(100)"))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"split": {"val": len(val_idx), "test": len(test_idx)},
                   "results": results,
                   "elapsed_sec": time.time() - t0}, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {OUT_JSON}")
    print(f"[done] {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
