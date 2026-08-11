#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 content 特征独立判别力：去掉 base_derived 6 维，仅用 content_v1+v2string。

如果 content-only 在 recall>0.99 下 FP 更低 → 基座误判可被内容特征纠正（弱化 base）；
如果 recall 大降 → base 不可弃，只能难例加权。

变体（cont_w5 难例加权保持，P3 协议）：
  full331+cont_w5     复现（FP≈928）
  content_only        d6 去掉
  content_interact    content_only + content×mask(base>=0.5) 交互
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
SEEDS = (0, 1, 2)
VAL_RECALL_TARGET = 0.995


def load_chunks(d: Path) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "chunk_*.npy")))]).astype(np.float32)


def derived(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return np.column_stack([p, p ** 2, np.abs(p - 0.5), np.log(pc), np.log1p(-pc), np.log(pc / (1 - pc))]).astype(np.float32)


def train_eval(Xv, yv, Xt, yt, sw, tag):
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
        print(f"[{tag}] NO threshold satisfies val recall>=0.995  "
              f"(max val recall={max(((pv>=t).astype(int)==1)&(yv==1)).sum()/max((yv==1).sum(),1):.4f}@t={min([x/1000 for x in range(200,950)]):.2f})")
        return None
    cand.sort()
    fp, fn, t, r = cand[0]
    pred = (pt >= t).astype(int)
    tp = int(((pred == 1) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return {"thr": round(float(t), 3), "fp": fp, "fn": fn, "tp": tp, "tn": tn,
            "recall": round(float(recall), 5), "precision": round(float(precision), 5),
            "f1": round(float(f1), 5)}


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
    d6_v, d6_t = derived(bp_v), derived(bp_t)
    c1_v, c1_t = content_v1[val_idx], content_v1[test_idx]
    ben = (yv == 0)
    sw = np.where(ben, 1.0 + 5.0 * bp_v, 1.0)

    full_v = np.column_stack([d6_v, c1_v, v2]).astype(np.float32)
    full_t = np.column_stack([d6_t, c1_t, st]).astype(np.float32)
    co_v = np.column_stack([c1_v, v2]).astype(np.float32)
    co_t = np.column_stack([c1_t, st]).astype(np.float32)

    r1 = train_eval(full_v, yv, full_t, yt, sw, "full331")
    print(f"[full331]      {r1}")
    r2 = train_eval(co_v, yv, co_t, yt, sw, "content_only")
    print(f"[content_only] {r2}")
    maskv = (bp_v >= 0.5).astype(np.float32).reshape(-1, 1)
    maskt = (bp_t >= 0.5).astype(np.float32).reshape(-1, 1)
    ci_v = np.column_stack([co_v, c1_v * maskv, v2 * maskv]).astype(np.float32)
    ci_t = np.column_stack([co_t, c1_t * maskt, st * maskt]).astype(np.float32)
    r3 = train_eval(ci_v, yv, ci_t, yt, sw, "content_interact")
    print(f"[content_inter] {r3}")

    out = {"full331_cont_w5": r1, "content_only": r2, "content_interact": r3}
    (PROJECT_ROOT / "reports" / "full_739k_benign" / "stage2_v6_content_only.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
