#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 模型侧 FP 归因：test 976 个误报的特征画像。

对 test 真良性样本（yt=0）按预测分类拆成 FP（pred=1，误报）vs TN（pred=0，正确），分析：
  1) base_prob 分布 → 基座是否已判恶（决定改进基座 or Stage-2 特征）；
  2) FP vs TN 的 top 判别特征（AUC，content_v1 / v2string / base derived）；
  3) FP 与 TP（真恶意）的特征相似度 → 模型为何混淆；
  4) FP 的 raw_path 目录分布 → 误报文件类型。
"""
from __future__ import annotations

import csv
import glob
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

BASE_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "base_prob"
CONTENT_V1_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_pe_v1"
V2_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_v2string"
META = CONTENT_V1_DIR / "meta.csv"
THR = 0.55
SEEDS = (0, 1, 2)


def load_chunks(d: Path) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "chunk_*.npy")))]).astype(np.float32)


def derived(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return np.column_stack([p, p ** 2, np.abs(p - 0.5), np.log(pc), np.log1p(-pc), np.log(pc / (1 - pc))]).astype(np.float32)


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """二分类 AUC（scores 高 → 判正）。"""
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    r = np.concatenate([pos, neg])
    o = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(r, kind="stable")
    r, o = r[order], o[order]
    ranks = np.arange(len(r), dtype=np.float64)
    # 处理并列：用平均秩
    sums, counts = np.zeros(len(r)), np.ones(len(r))
    i = 0
    while i < len(r):
        j = i
        while j < len(r) and r[j] == r[i]:
            j += 1
        m = j - i
        if m > 1:
            avg = (2 * i + m - 1) / 2.0
            for k in range(i, j):
                sums[k] = avg
            i = j
        else:
            sums[i] = float(i)
            i += 1
    n_pos = len(pos)
    n_neg = len(neg)
    auc = (sums[o == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def main() -> None:
    import json
    import time
    t0 = time.time()
    base_prob = load_chunks(BASE_DIR)
    content_v1 = load_chunks(CONTENT_V1_DIR)
    v2 = np.load(V2_DIR / "val.npy").astype(np.float32)
    st = np.load(V2_DIR / "test.npy").astype(np.float32)
    with open(V2_DIR / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    val_idx = np.asarray(meta["val_indices"], dtype=np.int64)
    test_idx = np.asarray(meta["test_indices"], dtype=np.int64)

    with open(META, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels_all = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    paths_all = [r.get("raw_path", "") for r in rows]
    yv = labels_all[val_idx]
    yt = labels_all[test_idx]

    def feat(p_vec, v1, v2s):
        return np.column_stack([derived(p_vec), v1, v2s]).astype(np.float32)

    Xv = feat(base_prob[val_idx], content_v1[val_idx], v2)
    Xt = feat(base_prob[test_idx], content_v1[test_idx], st)

    from sklearn.ensemble import HistGradientBoostingClassifier
    clfs = []
    for seed in SEEDS:
        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, random_state=seed)
        clf.fit(Xv, yv)
        clfs.append(clf)
    pt = np.mean([c.predict_proba(Xt)[:, 1] for c in clfs], axis=0)

    pred = (pt >= THR).astype(int)
    fp_mask = (yt == 0) & (pred == 1)
    tn_mask = (yt == 0) & (pred == 0)
    tp_mask = (yt == 1) & (pred == 1)
    n_fp = int(fp_mask.sum())
    n_tn = int(tn_mask.sum())
    print(f"[thr={THR}] TEST: FP={n_fp}  TN={n_tn}  (TP={int(tp_mask.sum())})  ({time.time()-t0:.0f}s loaded/trained)")

    bp_test = base_prob[test_idx]
    bpf = bp_test[fp_mask]
    print(f"\n=== 1) FP base_prob 分布（基座是否已判恶） ===")
    print(f"  FP   base_prob: mean={bpf.mean():.4f}  p50={np.median(bpf):.4f}  p90={np.percentile(bpf,90):.4f}  "
          f"p99={np.percentile(bpf,99):.4f}  max={bpf.max():.4f}")
    bptn = bp_test[tn_mask]
    print(f"  TN   base_prob: mean={bptn.mean():.4f}  p50={np.median(bptn):.4f}  p90={np.percentile(bptn,90):.4f}")
    for lo, hi in [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.0)]:
        n = int(((bpf >= lo) & (bpf < hi)).sum())
        print(f"    base_prob in [{lo:.1f},{hi:.1f}): {n}  ({n/max(n_fp,1)*100:.1f}%)")

    # ---- 2) FP vs TN 判别特征（content_v1: dims 0-99, v2string: dims 100-281, string=43? 实际 6+100+182+43=331）----
    c1 = content_v1[test_idx]
    v2s = st
    x_bp = derived(bp_test)
    groups = [
        ("base_derived", x_bp, 6),
        ("content_v1", c1, 100),
        ("v2string", v2s, 182),
    ]
    print(f"\n=== 2) FP vs TN 判别特征（AUC 高=能分开误报与正确良性） ===")
    top = []
    for gi, (name, Xg, n_dim) in enumerate(groups):
        for j in range(n_dim):
            y_ft = np.zeros(n_fp + n_tn, dtype=np.int64)
            y_ft[:n_fp] = 1
            a = auc(np.concatenate([Xg[fp_mask, j], Xg[tn_mask, j]]), y_ft)
            top.append((a, gi, j, name))
    top.sort(reverse=True)
    print(f"  {'AUC':>6} {'gi':>3} {'dim':>5} {'group':<12}  mean(FP)  mean(TN)")
    for a, gi, j, name in top[:20]:
        Xg = [x_bp, c1, v2s][gi]
        mf = Xg[fp_mask, j].mean(); mt = Xg[tn_mask, j].mean()
        print(f"  {a:>6.3f} {gi:>3} {j:>5} {name:<12}  {mf:>8.4f}  {mt:>8.4f}")

    # ---- 3) FP vs TP 特征相似度：FP 是否像真恶意 ----
    print(f"\n=== 3) FP vs TP 特征距离（小=FP 与真恶意混淆） ===")
    # 用 base_prob 分布
    bptp = bp_test[tp_mask]
    print(f"  base_prob: FP mean={bpf.mean():.4f}  TP mean={bptp.mean():.4f}  TN mean={bptn.mean():.4f}")
    # 内容 v1 的均值距离
    for name, Xg, n_dim in groups:
        Xf = np.stack([Xg[fp_mask, j].mean() for j in range(n_dim)])
        Xt2 = np.stack([Xg[tp_mask, j].mean() for j in range(n_dim)])
        Xn = np.stack([Xg[tn_mask, j].mean() for j in range(n_dim)])
        d_fp_tp = float(np.linalg.norm(Xf - Xt2) / np.sqrt(n_dim))
        d_fp_tn = float(np.linalg.norm(Xf - Xn) / np.sqrt(n_dim))
        print(f"  {name:<12} ||FP-TP||/√d={d_fp_tp:.4f}   ||FP-TN||/√d={d_fp_tn:.4f}   "
              f"({'FP更像TP' if d_fp_tp < d_fp_tn else 'FP更像TN'})")

    # ---- 4) FP 的 raw_path 目录分布 ----
    print(f"\n=== 4) FP 样本目录分布（前 25） ===")
    dirs = Counter()
    for k, idx in enumerate(test_idx[fp_mask]):
        p = paths_all[idx]
        # 归一化盘符
        d = "/".join(Path(p).parts[2:3]) if p and ":" in p else Path(p).parent.name if p else "(no-path)"
        dirs[d] += 1
    for d, n in dirs.most_common(25):
        print(f"  {n:>5}  {d}")

    # ---- 5) 典型强误报样本 ----
    print(f"\n=== 5) 典型强误报（FP 且 pt 最高 10 个） ===")
    idx_fp = np.where(fp_mask)[0]
    pt_fp = pt[fp_mask]
    order = np.argsort(-pt_fp)[:10]
    for o in order:
        k = int(idx_fp[o])
        print(f"  idx={int(test_idx[k])}  pt={pt_fp[o]:.4f}  base={bp_test[k]:.4f}  path={paths_all[int(test_idx[k])][:100]}")
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
