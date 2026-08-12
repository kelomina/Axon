#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FP/TN 特征间隙诊断（良性扩充重训后）：模型没利用的判别信号在哪。

对 test 中良性样本（label=0），对比被误报（Stage-2 pred=1，FP）与判对（TN）在
331 维特征（derived6 + content_v1 100 + content_v2string 225）上的逐个特征区分度
（rank-AUC）。AUC 高 → 该特征能区分 FP 与正常白，模型却没完全用上 → 改进候选。
同时给 FN vs TP（恶意侧）对照。
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
BASE_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "base_prob"
CONTENT_V1_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_pe_v1"
V2_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_v2string"
PRED_CSV = PROJECT_ROOT / "reports" / "full_739k_benign" / "test739k_benign_stage2v2_predictions.csv"
OUT = PROJECT_ROOT / "reports" / "full_739k_benign" / "fp_feature_gap.json"


def load_chunks(d: Path) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "chunk_*.npy")))]).astype(np.float32)


def derived(p):
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return np.column_stack([p, p ** 2, np.abs(p - 0.5), np.log(pc), np.log1p(-pc), np.log(pc / (1 - pc))]).astype(np.float32)


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """单个特征的 pos-vs-neg 区分度（rank AUC，1.0=完美可分）。"""
    if pos.size == 0 or neg.size == 0:
        return 0.5
    vals = np.concatenate([pos, neg])
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(vals) + 1)
    # 处理平局：取平均秩
    sorted_vals = vals[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    pos_ranks = ranks[:pos.size]
    auc = (pos_ranks.sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def main() -> None:
    t0 = time.time()
    print("=== FP/TN feature gap diagnostic ===")
    base_prob = load_chunks(BASE_DIR)
    v1 = load_chunks(CONTENT_V1_DIR)
    v2 = np.load(V2_DIR / "val.npy").astype(np.float32)
    st = np.load(V2_DIR / "test.npy").astype(np.float32)
    with open(V2_DIR / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    test_idx = np.asarray(meta["test_indices"], dtype=np.int64)
    with open(CONTENT_V1_DIR / "meta.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels_all = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    yt = labels_all[test_idx]
    d6 = derived(base_prob[test_idx])
    X = np.column_stack([d6, v1[test_idx], st]).astype(np.float32)  # 331
    print(f"X={X.shape}")

    pred = {}
    with open(PRED_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pred[int(r["index"])] = int(r["s2_pred"])
    pred_vals = np.asarray([pred[i] for i in test_idx], dtype=np.int64)

    # 良性侧：FP vs TN
    fp_mask = (yt == 0) & (pred_vals == 1)
    tn_mask = (yt == 0) & (pred_vals == 0)
    # 恶意侧：FN vs TP
    fn_mask = (yt == 1) & (pred_vals == 0)
    tp_mask = (yt == 1) & (pred_vals == 1)
    print(f"FP={fp_mask.sum()} TN={tn_mask.sum()} FN={fn_mask.sum()} TP={tp_mask.sum()}")

    feature_names = (
        ["d6_p", "d6_p2", "d6_abs_pm05", "d6_logp", "d6_log1mp", "d6_logit"]
        + meta["v2_names"] + meta["string_names"]
    )
    # content_v1 名字从模块拿（这里简化：v1 100 维名字在 stage2 脚本里，用占位）
    # 加载真实 v1 名字
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        from train_stage2_cache_matrix import CONTENT_PE_V1_FEATURE_NAMES as V1N
        feature_names = (["d6_p", "d6_p2", "d6_abs_pm05", "d6_logp", "d6_log1mp", "d6_logit"]
                         + list(V1N) + list(meta["v2_names"]) + list(meta["string_names"]))
        assert len(feature_names) == X.shape[1]
    except Exception as e:
        print(f"[warn] v1 names unavailable ({e}); using placeholder names for v1 block")
        feature_names = (["d6_p", "d6_p2", "d6_abs_pm05", "d6_logp", "d6_log1mp", "d6_logit"]
                         + [f"v1_{i}" for i in range(100)]
                         + list(meta["v2_names"]) + list(meta["string_names"]))
        assert len(feature_names) == X.shape[1]

    def top_k(mask_pos, mask_neg, tag):
        pos = X[mask_pos]
        neg = X[mask_neg]
        aucs = np.asarray([rank_auc(pos[:, j], neg[:, j]) for j in range(X.shape[1])])
        order = np.argsort(-np.abs(aucs - 0.5))
        print(f"\n=== {tag}: 区分度 Top 15 (AUC, 1.0=FP 特征值更大) ===")
        out = []
        for j in order[:15]:
            direction = "FP>TN" if aucs[j] > 0.5 else "FP<TN"
            print(f"  {feature_names[j]:42s} AUC={aucs[j]:.4f}  {direction}")
            out.append({"feature": feature_names[j], "auc": round(float(aucs[j]), 4),
                        "mean_pos": round(float(pos[:, j].mean()), 5),
                        "mean_neg": round(float(neg[:, j].mean()), 5)})
        return out

    fp_gap = top_k(fp_mask, tn_mask, "FP vs TN (良性误报 vs 判对白)")
    fn_gap = top_k(fn_mask, tp_mask, "FN vs TP (恶意漏报 vs 判对黑)")

    rep = {
        "n_fp": int(fp_mask.sum()), "n_tn": int(tn_mask.sum()),
        "n_fn": int(fn_mask.sum()), "n_tp": int(tp_mask.sum()),
        "fp_vs_tn_top": fp_gap,
        "fn_vs_tp_top": fn_gap,
        "elapsed_sec": time.time() - t0,
    }
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
