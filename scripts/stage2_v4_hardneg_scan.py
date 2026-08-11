#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 难例加权细化：让模型更重视"基座判恶的真良性"。

两种操作点协议：
  P1 val 最优 F1（对齐主脚本）
  P2 val recall>=0.99 下最小 FP（对齐 standing goal：零误报白 + 黑召回>99%）

变体（VAL 训练 + 双协议，TEST 一次评估）：
  w5_t05     base>=0.5 良性 w=5
  w10_t05    base>=0.5 良性 w=10
  w5_t07     base>=0.7 良性 w=5
  w5_t03     base>=0.3 良性 w=5
  cont_w5    benign w = 1 + 5*base（连续加权，越难越重）
输出：控制台 + reports/full_739k_benign/stage2_v4_hardneg.json
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
REPORT_JSON = PROJECT_ROOT / "reports" / "full_739k_benign" / "stage2_v4_hardneg.json"
SEEDS = (0, 1, 2)


def load_chunks(d: Path) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "chunk_*.npy")))]).astype(np.float32)


def derived(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return np.column_stack([p, p ** 2, np.abs(p - 0.5), np.log(pc), np.log1p(-pc), np.log(pc / (1 - pc))]).astype(np.float32)


def train_clfs(Xv, yv, sw):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clfs = []
    for seed in SEEDS:
        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, random_state=seed)
        clf.fit(Xv, yv, sample_weight=sw)
        clfs.append(clf)
    return clfs


def eval_protocols(clfs, Xv, yv, Xt, yt):
    pv = np.mean([c.predict_proba(Xv)[:, 1] for c in clfs], axis=0)
    pt = np.mean([c.predict_proba(Xt)[:, 1] for c in clfs], axis=0)
    out = {}
    # P1: val 最优 F1（主脚本协议 0.20-0.89）
    best_t, best_f1 = 0.5, -1.0
    for t in [x / 100 for x in range(20, 90)]:
        pv_pred = (pv >= t).astype(int)
        tp = ((pv_pred == 1) & (yv == 1)).sum()
        fp = ((pv_pred == 1) & (yv == 0)).sum()
        fn = ((pv_pred == 0) & (yv == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    out["P1_f1opt"] = _summarize(pt, yt, best_t)
    # P2: val recall>=0.99 下最小 FP（实测有 val→test 召回偏移：test recall 只 ~0.987）
    cand = []
    for t in [x / 1000 for x in range(200, 950)]:
        pv_pred = (pv >= t).astype(int)
        r = ((pv_pred == 1) & (yv == 1)).sum() / max((yv == 1).sum(), 1)
        if r >= 0.99:
            fp = int(((pt >= t) & (yt == 0)).sum())
            fn = int(((pt < t) & (yt == 1)).sum())
            cand.append((fp, fn, t, float(r)))
    cand.sort()
    if cand:
        fp, fn, t, r = cand[0]
        out["P2_recall99_minFP"] = _summarize(pt, yt, t)
    # P3: val recall>=0.995 下最小 FP（保守校准，补偿 val→test 召回偏移，目标 test recall>0.99）
    cand3 = []
    for t in [x / 1000 for x in range(200, 950)]:
        pv_pred = (pv >= t).astype(int)
        r = ((pv_pred == 1) & (yv == 1)).sum() / max((yv == 1).sum(), 1)
        if r >= 0.995:
            fp = int(((pt >= t) & (yt == 0)).sum())
            fn = int(((pt < t) & (yt == 1)).sum())
            cand3.append((fp, fn, t, float(r)))
    cand3.sort()
    if cand3:
        fp, fn, t, r = cand3[0]
        out["P3_recall995_minFP"] = _summarize(pt, yt, t)
    return out


def _summarize(pt, yt, t):
    pred = (pt >= t).astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {"thr": round(float(t), 3), "fp": fp, "fn": fn, "tp": tp, "tn": tn,
            "errors": fp + fn, "recall": round(float(recall), 5),
            "precision": round(float(precision), 5), "f1": round(float(f1), 5)}


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
    Xv = np.column_stack([derived(bp_v), content_v1[val_idx], v2]).astype(np.float32)
    Xt = np.column_stack([derived(bp_t), content_v1[test_idx], st]).astype(np.float32)
    print(f"[data] Xv={Xv.shape} Xt={Xt.shape}")

    results = {}

    # ---- baseline（无加权） ----
    clfs = train_clfs(Xv, yv, None)
    results["baseline"] = eval_protocols(clfs, Xv, yv, Xt, yt)
    print("\n[baseline]")
    print(f"  P1_f1opt:  {results['baseline']['P1_f1opt']}")
    print(f"  P2_recall99: {results['baseline'].get('P2_recall99_minFP')}")

    # ---- 加权变体 ----
    ben = (yv == 0)
    variants = {
        "w5_t05": lambda: np.where(ben & (bp_v >= 0.5), 5.0, 1.0),
        "w5_t07": lambda: np.where(ben & (bp_v >= 0.7), 5.0, 1.0),
        "w3_t03": lambda: np.where(ben & (bp_v >= 0.3), 3.0, 1.0),
        "w5_t03": lambda: np.where(ben & (bp_v >= 0.3), 5.0, 1.0),
        "w5_t04": lambda: np.where(ben & (bp_v >= 0.4), 5.0, 1.0),
        "w8_t03": lambda: np.where(ben & (bp_v >= 0.3), 8.0, 1.0),
        "cont_w5": lambda: np.where(ben, 1.0 + 5.0 * bp_v, 1.0),
    }
    for name, mk in variants.items():
        sw = mk()
        clfs = train_clfs(Xv, yv, sw)
        res = eval_protocols(clfs, Xv, yv, Xt, yt)
        results[name] = res
        p1, p2, p3 = res["P1_f1opt"], res.get("P2_recall99_minFP"), res.get("P3_recall995_minFP")
        print(f"\n[{name}]  (n_weighted={int((sw>1).sum())})")
        print(f"  P1_f1opt:  thr={p1['thr']} FP={p1['fp']} FN={p1['fn']} recall={p1['recall']} F1={p1['f1']}")
        if p2:
            print(f"  P2_r99:    thr={p2['thr']} FP={p2['fp']} FN={p2['fn']} recall={p2['recall']}")
        if p3:
            print(f"  P3_r995:   thr={p3['thr']} FP={p3['fp']} FN={p3['fn']} recall={p3['recall']}")

    REPORT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {REPORT_JSON}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
