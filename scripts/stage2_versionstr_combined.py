#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组合实验：版本字符串特征 + 重平衡，能否再压 FP。

X = derived6 + v1 100 + v2string 225 [+ versionstr 8] = 331 或 339 维。
协议：VAL 训练 + VAL recall>=0.99 约束下扫阈值（min FPR），TEST 一次性评估。
Config：
  A  base331_w1          无版本特征, 无重平衡（= sweep baseline）
  B  base331_benignx3    无版本特征, benign_all×3（sweep 最优重平衡）
  C  ver339_w1           有版本特征, 无重平衡
  D  ver339_benignx3     有版本特征, benign_all×3
输出 reports/full_739k_benign/stage2_versionstr_combined.json
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

BASE_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "base_prob"
CONTENT_V1_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_pe_v1"
V2_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_v2string"
VER_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_versionstr"
OUT = PROJECT_ROOT / "reports" / "full_739k_benign" / "stage2_versionstr_combined.json"
SEEDS = (0, 1, 2)


def load_chunks(d: Path) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "chunk_*.npy")))]).astype(np.float32)


def derived(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return np.column_stack([p, p ** 2, np.abs(p - 0.5), np.log(pc), np.log1p(-pc), np.log(pc / (1 - pc))]).astype(np.float32)


def main() -> None:
    t0 = time.time()
    print("=== Stage-2 combined: versionstr features + rebalance ===")

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

    ver_v = np.load(VER_DIR / "val.npy").astype(np.float32)
    ver_t = np.load(VER_DIR / "test.npy").astype(np.float32)
    print(f"versionstr val={ver_v.shape} test={ver_t.shape}")

    from kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES
    j_dll = list(CONTENT_PE_V1_FEATURE_NAMES).index("content_is_dll")
    benign_mask = yv == 0
    exe_benign = benign_mask & (content_v1[val_idx, j_dll] <= 0)

    base_feat_v = np.column_stack([derived(base_prob[val_idx]), content_v1[val_idx], v2]).astype(np.float32)
    base_feat_t = np.column_stack([derived(base_prob[test_idx]), content_v1[test_idx], st]).astype(np.float32)
    Xv331, Xt331 = base_feat_v, base_feat_t
    Xv339 = np.column_stack([base_feat_v, ver_v]).astype(np.float32)
    Xt339 = np.column_stack([base_feat_t, ver_t]).astype(np.float32)
    print(f"Xv331={Xv331.shape} Xv339={Xv339.shape}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score

    configs = [
        {"name": "base331_w1", "Xv": Xv331, "Xt": Xt331, "rebalance": False},
        {"name": "base331_benignx3", "Xv": Xv331, "Xt": Xt331, "rebalance": True},
        {"name": "ver339_w1", "Xv": Xv339, "Xt": Xt339, "rebalance": False},
        {"name": "ver339_benignx3", "Xv": Xv339, "Xt": Xt339, "rebalance": True},
    ]

    results = []
    for cfg in configs:
        sw = np.ones(len(yv), dtype=np.float64)
        if cfg["rebalance"]:
            sw[benign_mask] = 3.0
        clfs = []
        for seed in SEEDS:
            clf = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, early_stopping=False, random_state=seed)
            clf.fit(cfg["Xv"], yv, sample_weight=sw)
            clfs.append(clf)
        pv = np.mean([c.predict_proba(cfg["Xv"])[:, 1] for c in clfs], axis=0)
        pt = np.mean([c.predict_proba(cfg["Xt"])[:, 1] for c in clfs], axis=0)

        best_f1, best_t = 0.0, 0.5
        goal_t, goal_fpr = 0.5, 1.0
        for t in [x / 100 for x in range(20, 95)]:
            predv = (pv >= t).astype(int)
            f1 = f1_score(yv, predv)
            if f1 > best_f1:
                best_f1, best_t = f1, t
            tp = int(((predv == 1) & (yv == 1)).sum()); fp = int(((predv == 1) & (yv == 0)).sum())
            fn = int(((predv == 0) & (yv == 1)).sum())
            rec = tp / (tp + fn) if tp + fn else 0.0
            fpr = fp / (yv == 0).sum()
            if rec >= 0.99 and fpr < goal_fpr:
                goal_fpr, goal_t = fpr, t

        pt_g = (pt >= goal_t).astype(int)
        tp = int(((pt_g == 1) & (yt == 1)).sum()); fp = int(((pt_g == 1) & (yt == 0)).sum())
        fn = int(((pt_g == 0) & (yt == 1)).sum()); tn = int(((pt_g == 0) & (yt == 0)).sum())
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1t = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        print(f"\n[{cfg['name']}] val_bestF1={best_f1:.5f} val_fpr@rec99={goal_fpr:.4f}@t={goal_t:.2f}")
        print(f"  [TEST] F1={f1t:.5f} P={p_:.4f} R={r_:.4f} FP={fp} FN={fn} errors={fp+fn}")

        # FP breakdown: for ver configs, what fraction of TEST FP have version strings
        order = {int(g): k for k, g in enumerate(test_idx.tolist())}
        fp_break = None
        if "ver" in cfg["name"]:
            fp_pos = np.where((pt_g == 1) & (yt == 0))[0]
            fp_has_ver = (ver_t[fp_pos].sum(axis=1) > 0).mean()
            fp_break = {"n_fp": int(len(fp_pos)), "fp_has_verstr": round(float(fp_has_ver), 3)}

        results.append({
            "config": cfg["name"], "rebalance": cfg["rebalance"], "dim": cfg["Xv"].shape[1],
            "val_best_f1": round(best_f1, 5), "val_goal_fpr": round(goal_fpr, 4), "val_goal_t": round(goal_t, 3),
            "test_f1": round(f1t, 5), "precision": round(p_, 4), "recall": round(r_, 4),
            "fp": fp, "fn": fn, "errors": fp + fn,
            "fp_break": fp_break,
        })
        OUT.write_text(json.dumps({"results": results, "elapsed_sec": time.time() - t0},
                                  indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[saved] {OUT} ({(time.time()-t0)/60:.1f} min)")
    best = min(results, key=lambda r: r["val_goal_fpr"])
    print(f"[candidate] {best['config']} TEST FP={best['fp']} FN={best['fn']} R={best['recall']:.4f}")


if __name__ == "__main__":
    main()
