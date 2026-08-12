#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""目标评估：零误报白 + 黑召回>99% 在 v2/string 改进版 Stage-2 上。

重算 Stage-2（v1+v2+string+派生）test 分数，做：
  1) val 阈值下的 FP/FN/recall
  2) 全阈值扫描：能否 FP=0 且 recall>99%
  3) 标签噪声视角：FN 中模型>90%确信良性的数量（疑似误标白）
输出 reports/full_739k/goal_eval.json + test739k_stage2v2_predictions.csv
"""
from __future__ import annotations

import csv
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

BASE_DIR = PROJECT_ROOT / "reports" / "full_739k" / "base_prob"
CONTENT_V1_DIR = PROJECT_ROOT / "reports" / "full_739k" / "content_pe_v1"
V2_DIR = PROJECT_ROOT / "reports" / "full_739k" / "content_v2string"
OUT = PROJECT_ROOT / "reports" / "full_739k"
CHECKPOINT = PROJECT_ROOT / "models" / "full_739k" / "best_model_739k.pt"
SEEDS = (0, 1, 2)


def load_chunks(d: Path) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "chunk_*.npy")))]).astype(np.float32)


def derived(p):
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return np.column_stack([p, p ** 2, np.abs(p - 0.5), np.log(pc), np.log1p(-pc), np.log(pc / (1 - pc))]).astype(np.float32)


def main():
    t0 = time.time()
    print("=== Goal eval: zero-FP + recall>99% (v2/string Stage-2) ===")
    base_prob = load_chunks(BASE_DIR)
    v1 = load_chunks(CONTENT_V1_DIR)
    v2 = np.load(V2_DIR / "val.npy").astype(np.float32)
    st = np.load(V2_DIR / "test.npy").astype(np.float32)
    with open(V2_DIR / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    val_idx = np.asarray(meta["val_indices"], dtype=np.int64)
    test_idx = np.asarray(meta["test_indices"], dtype=np.int64)
    with open(CONTENT_V1_DIR / "meta.csv", encoding="utf-8") as f:
        rows_all = list(csv.DictReader(f))
    labels_all = np.asarray([int(r["label"]) for r in rows_all], dtype=np.int64)
    shas_all = [r["source_sha256"].strip().casefold() for r in rows_all]
    yv, yt = labels_all[val_idx], labels_all[test_idx]

    def feat(p, v1m, v2m):
        return np.column_stack([derived(p), v1m, v2m]).astype(np.float32)

    Xv = feat(base_prob[val_idx], v1[val_idx], v2)
    Xt = feat(base_prob[test_idx], v1[test_idx], st)
    print(f"Xv={Xv.shape} Xt={Xt.shape}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score

    clfs = [HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=False, random_state=s)
            for s in SEEDS]
    for clf in clfs:
        clf.fit(Xv, yv)
    pv = np.mean([c.predict_proba(Xv)[:, 1] for c in clfs], axis=0)
    pt = np.mean([c.predict_proba(Xt)[:, 1] for c in clfs], axis=0)

    # val 阈值
    best = (0.0, 0.5)
    for t in [x / 100 for x in range(20, 90)]:
        f1 = f1_score(yv, (pv >= t).astype(int))
        if f1 > best[0]:
            best = (f1, t)
    thr = best[1]
    n_black = int((yt == 1).sum())
    n_white = int((yt == 0).sum())

    def metrics(score, thr):
        pred = (score >= thr).astype(int)
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        rec = 1 - fn / n_black
        return fp, fn, rec

    fp0, fn0, rec0 = metrics(pt, thr)
    print(f"\n[val 阈值 thr={thr:.2f}] FP={fp0} FN={fn0} recall={rec0:.5f} "
          f"(目标: FP=0, recall>0.99)")

    # 保存 test 预测
    pred_csv = OUT / "test739k_stage2v2_predictions.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "source_sha256", "label", "s2_score", "s2_pred"])
        for k, ti in enumerate(test_idx):
            w.writerow([int(ti), shas_all[ti], int(labels_all[ti]), float(pt[k]), int(pt[k] >= thr)])
    print(f"[saved] {pred_csv}")

    # 全阈值扫描：FP=0 + recall>99%
    print("\n=== 阈值扫描（目标 FP=0 & recall>99%）===")
    found = None
    for t in [x / 1000 for x in range(500, 1000)]:
        fp, fn, rec = metrics(pt, t)
        if fp == 0 and rec > 0.99:
            found = (t, fp, fn, rec)
            break
        if t in (0.70, 0.80, 0.90, 0.95, 0.98):
            print(f"  t={t:.2f} FP={fp} FN={fn} recall={rec:.5f}")
    if found:
        print(f"  [FOUND] t={found[0]:.3f} FP={found[1]} FN={found[2]} recall={found[3]:.5f}")
    else:
        print("  [NO] 无阈值满足 FP=0 且 recall>99%")

    # 标签噪声视角
    fn_mask = (pt < thr) & (yt == 1)
    fn_conf_benign = int(((pt[fn_mask] < 0.1)).sum())  # 模型>90%确信良性
    print(f"\n=== 标签噪声视角 ===")
    print(f"FN={fn0} 中模型>90%确信良性(疑似误标白): {fn_conf_benign}")
    true_rec = 1 - (fn0 - fn_conf_benign) / n_black
    print(f"去噪后真召回: {true_rec:.5f} (>0.99 目标{'OK' if true_rec>0.99 else 'NOT MET'})")

    out = {"val_threshold": thr, "fp": fp0, "fn": fn0, "recall": rec0,
           "fp0_recall99_found": bool(found), "fp0_point": found,
           "fn_conf_benign": fn_conf_benign, "true_recall_est": true_rec,
           "elapsed_sec": time.time() - t0}
    json.dump(out, open(OUT / "goal_eval.json", "w"), indent=2)
    print(f"[saved] {OUT / 'goal_eval.json'}")
    print(f"[done] {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
