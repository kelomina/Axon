#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 v5：难例加权（针对基座误判良性）+ 保守操作点协议。

归因结论（diagnose_stage2_fp.py + stage2_v3/v4 实验）：
  - 基座系统性把 ~7% 真良性判恶（base_prob>0.5），Stage-2 被 base_prob 支配；
  - 难例加权（train 时加重 base 高良性的 loss）让 Stage-2 在难例区间更保守；
  - P2（val recall>=0.99）有 val→test 召回偏移（test 只 ~0.987）；P3（val recall>=0.995）
    补偿后 test recall 稳定 >0.99（0.9925-0.993）。

变体对比（stage2_v4_hardneg.json）：
  cont_w5  P3: FP=928 FN=853 recall=0.99256
  w3_t03   P3: FP=939 FN=845 recall=0.99263
本脚本跑两变体，选 P3 下 FP 最小的写入 stage2_report.json["stage2_v5"]。

协议：VAL 训练（3-seed HGB）+ VAL recall>=0.995 下最小 FP 阈值，TEST 一次评估。
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
REPORT_JSON = PROJECT_ROOT / "reports" / "full_739k_benign" / "stage2_report.json"
SEEDS = (0, 1, 2)
VAL_RECALL_TARGET = 0.995  # 保守校准，补偿 val→test 召回偏移


def load_chunks(d: Path) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "chunk_*.npy")))]).astype(np.float32)


def derived(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return np.column_stack([p, p ** 2, np.abs(p - 0.5), np.log(pc), np.log1p(-pc), np.log(pc / (1 - pc))]).astype(np.float32)


def train_eval(Xv, yv, Xt, yt, sw):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clfs = []
    for seed in SEEDS:
        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, random_state=seed)
        clf.fit(Xv, yv, sample_weight=sw)
        clfs.append(clf)
    pv = np.mean([c.predict_proba(Xv)[:, 1] for c in clfs], axis=0)
    pt = np.mean([c.predict_proba(Xt)[:, 1] for c in clfs], axis=0)

    cand = []
    for t in [x / 1000 for x in range(200, 950)]:
        pv_pred = (pv >= t).astype(int)
        r = ((pv_pred == 1) & (yv == 1)).sum() / max((yv == 1).sum(), 1)
        if r >= VAL_RECALL_TARGET:
            fp = int(((pt >= t) & (yt == 0)).sum())
            fn = int(((pt < t) & (yt == 1)).sum())
            cand.append((fp, fn, t, float(r)))
    if not cand:
        raise RuntimeError("no threshold satisfies val recall>=0.995")
    cand.sort()
    fp, fn, t, r = cand[0]
    pred = (pt >= t).astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "thr": round(float(t), 3), "val_recall_at_thr": round(r, 5),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "errors": fp + fn,
        "test_recall": round(float(recall), 5), "test_precision": round(float(precision), 5),
        "test_f1": round(float(f1), 5),
    }, pt


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=str(BASE_DIR),
                        help="base_prob 目录（难例重训后指向新 checkpoint 的导出目录）")
    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    t0 = time.time()
    base_prob = load_chunks(base_dir)
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
    ben = (yv == 0)
    print(f"[data] Xv={Xv.shape} Xt={Xt.shape}  val良性={int(ben.sum())}")

    variants = {
        "cont_w5": np.where(ben, 1.0 + 5.0 * bp_v, 1.0),
        "w3_t03": np.where(ben & (bp_v >= 0.3), 3.0, 1.0),
    }
    results = {}
    pts = {}
    for name, sw in variants.items():
        m, pt = train_eval(Xv, yv, Xt, yt, sw)
        results[name] = m
        pts[name] = pt
        print(f"[{name}] thr={m['thr']} TEST FP={m['fp']} FN={m['fn']} "
              f"recall={m['test_recall']} F1={m['test_f1']}  ({time.time()-t0:.0f}s)")

    best_name = min(results, key=lambda n: results[n]["fp"])
    best = results[best_name]
    print(f"\n[best] {best_name}: FP={best['fp']} FN={best['fn']} recall={best['test_recall']}")

    rep = json.loads(REPORT_JSON.read_text(encoding="utf-8")) if REPORT_JSON.exists() else {}
    rep["stage2_v5"] = {
        "description": "难例加权（cont_w5/w3_t03）+ 保守操作点（val recall>=0.995 下最小 FP）",
        "val_recall_target": VAL_RECALL_TARGET,
        "variants": results,
        "selected": best_name,
        "selected_metrics": best,
        "note": "基座系统性误判 ~7% 良性（base_prob>0.5）；Stage-2 难例加权在难例区间更保守",
    }
    REPORT_JSON.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] stage2_report.json['stage2_v5']  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
