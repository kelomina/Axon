#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 模型侧改进对比实验：如何让 Stage-2 在高 base_prob 区间利用 content 特征纠正基座误判。

背景归因（diagnose_stage2_fp.py）：
  - 66% 的 test FP 在 base_prob>0.7（基座把真良性判恶，系统性，train 分布内即如此）；
  - content_v1 能区分 FP vs TP（FP 特征更像 TN）→ 内容有纠正信息但被 base_derived 支配。

变体（均 3-seed HGB，VAL 训练 + VAL 选 F1 阈值，TEST 一次评估）：
  1. baseline_331       复现当前（FP≈976）
  2. hardneg_w5         val 中 base>=0.5 的良性样本 sample_weight=5（难例加权）
  3. interact_high      加 content×mask(base>=0.5) 交互特征，让 HGB 学到 base 高时的 content 模式
  4. both               hardneg_w5 + interact_high
输出：控制台 + reports/full_739k_benign/stage2_v3_experiments.json
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

BASE_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "base_prob"
CONTENT_V1_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_pe_v1"
V2_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_v2string"
REPORT_JSON = PROJECT_ROOT / "reports" / "full_739k_benign" / "stage2_v3_experiments.json"
SEEDS = (0, 1, 2)


def load_chunks(d: Path) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "chunk_*.npy")))]).astype(np.float32)


def derived(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return np.column_stack([p, p ** 2, np.abs(p - 0.5), np.log(pc), np.log1p(-pc), np.log(pc / (1 - pc))]).astype(np.float32)


def train_eval(Xv, yv, Xt, yt, sample_weight=None):
    """3-seed HGB，val 选 F1 阈值，test 报告。返回 (metrics, pt)。"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    clfs = []
    for seed in SEEDS:
        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, random_state=seed)
        clf.fit(Xv, yv, sample_weight=sample_weight)
        clfs.append(clf)
    pv = np.mean([c.predict_proba(Xv)[:, 1] for c in clfs], axis=0)
    pt = np.mean([c.predict_proba(Xt)[:, 1] for c in clfs], axis=0)

    # val 选 F1 阈值
    best_t, best_f1 = 0.5, -1.0
    for t in [x / 100 for x in range(1, 100)]:
        pv_pred = (pv >= t).astype(int)
        tp = ((pv_pred == 1) & (yv == 1)).sum()
        fp = ((pv_pred == 1) & (yv == 0)).sum()
        fn = ((pv_pred == 0) & (yv == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t

    pred = (pt >= best_t).astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "val_thr": round(float(best_t), 2), "val_f1": round(float(best_f1), 5),
        "test_tp": tp, "test_fp": fp, "test_fn": fn, "test_tn": tn,
        "test_f1": round(float(f1), 5), "test_recall": round(float(recall), 5),
        "test_precision": round(float(precision), 5), "test_errors": fp + fn,
    }, pt


def main() -> None:
    t0 = time.time()
    base_prob = load_chunks(BASE_DIR)
    content_v1 = load_chunks(CONTENT_V1_DIR)
    v2 = np.load(V2_DIR / "val.npy").astype(np.float32)
    st = np.load(V2_DIR / "test.npy").astype(np.float32)
    with open(V2_DIR / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    val_idx = np.asarray(meta["val_indices"], dtype=np.int64)
    test_idx = np.asarray(meta["test_indices"], dtype=np.int64)
    with open(CONTENT_V1_DIR / "meta.csv", encoding="utf-8") as f:
        labels_all = np.asarray([int(r["label"]) for r in csv.DictReader(f)], dtype=np.int64)
    yv = labels_all[val_idx]
    yt = labels_all[test_idx]

    bp_v = base_prob[val_idx]
    bp_t = base_prob[test_idx]
    d6_v = derived(bp_v)
    d6_t = derived(bp_t)
    c1_v = content_v1[val_idx]
    c1_t = content_v1[test_idx]

    Xv_base = np.column_stack([d6_v, c1_v, v2]).astype(np.float32)
    Xt_base = np.column_stack([d6_t, c1_t, st]).astype(np.float32)
    print(f"[base] Xv={Xv_base.shape} Xt={Xt_base.shape}  (val_thr 协议：val 最优 F1)")

    results = {}

    # ---- 1) baseline 复现 ----
    m, pt1 = train_eval(Xv_base, yv, Xt_base, yt)
    results["baseline_331"] = m
    print(f"\n[1] baseline_331  thr={m['val_thr']}  TEST FP={m['test_fp']} FN={m['test_fn']} "
          f"recall={m['test_recall']} F1={m['test_f1']}")

    # ---- 2) hardneg 加权：val 中 base>=0.5 的良性样本 w=5 ----
    w = np.ones(len(yv), dtype=np.float64)
    hard_v = (yv == 0) & (bp_v >= 0.5)
    w[hard_v] = 5.0
    print(f"[2] hardneg_w5: {int(hard_v.sum())} val benign base>=0.5 weighted x5")
    m, _ = train_eval(Xv_base, yv, Xt_base, yt, sample_weight=w)
    results["hardneg_w5"] = m
    print(f"    hardneg_w5  thr={m['val_thr']}  TEST FP={m['test_fp']} FN={m['test_fn']} "
          f"recall={m['test_recall']} F1={m['test_f1']}")

    # ---- 3) 交互特征：content × mask(base>=0.5) ----
    maskv = (bp_v >= 0.5).astype(np.float32).reshape(-1, 1)
    maskt = (bp_t >= 0.5).astype(np.float32).reshape(-1, 1)
    Xv3 = np.column_stack([Xv_base, c1_v * maskv, v2 * maskv]).astype(np.float32)
    Xt3 = np.column_stack([Xt_base, c1_t * maskt, st * maskt]).astype(np.float32)
    print(f"[3] interact_high: +282 dims (content×base>=0.5) -> {Xv3.shape[1]}")
    m, _ = train_eval(Xv3, yv, Xt3, yt)
    results["interact_high"] = m
    print(f"    interact_high  thr={m['val_thr']}  TEST FP={m['test_fp']} FN={m['test_fn']} "
          f"recall={m['test_recall']} F1={m['test_f1']}")

    # ---- 4) both ----
    w4 = np.ones(len(yv), dtype=np.float64)
    w4[hard_v] = 5.0
    m, _ = train_eval(Xv3, yv, Xt3, yt, sample_weight=w4)
    results["both"] = m
    print(f"[4] both  thr={m['val_thr']}  TEST FP={m['test_fp']} FN={m['test_fn']} "
          f"recall={m['test_recall']} F1={m['test_f1']}")

    REPORT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {REPORT_JSON}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
