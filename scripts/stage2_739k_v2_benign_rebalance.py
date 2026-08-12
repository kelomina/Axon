#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 重平衡 sweep：对良性 non-DLL（FP 高危人群）过采样，看能否降 FP。

假设（基于归因）：Stage-2 训练良性 86.9% 是 DLL，而 FP 人群 74% 是 non-DLL 良性 EXE
→ 模型把 non-DLL 学成"≈恶意"，良性 EXE 流形欠学习。对良性 non-DLL 加权可纠正。

协议：VAL 训练 + VAL 扫阈值（沿用原协议），TEST 对每个 config 评一次（诊断表）。
选择标准（目标导向）：VAL recall>=0.99 约束下 VAL FPR 最小。
只改 sample_weight，不碰特征——纯模型侧，符合"不用白名单"约束。

输出：reports/full_739k_benign/stage2_rebalance_sweep.json
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
OUT = PROJECT_ROOT / "reports" / "full_739k_benign" / "stage2_rebalance_sweep.json"
SEEDS = (0, 1, 2)


def load_chunks(d: Path) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "chunk_*.npy")))]).astype(np.float32)


def derived(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return np.column_stack([p, p ** 2, np.abs(p - 0.5), np.log(pc), np.log1p(-pc), np.log(pc / (1 - pc))]).astype(np.float32)


def main() -> None:
    t0 = time.time()
    print("=== Stage-2 rebalance sweep (benign non-DLL oversampling) ===")

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

    Xv = np.column_stack([derived(base_prob[val_idx]), content_v1[val_idx], v2]).astype(np.float32)
    Xt = np.column_stack([derived(base_prob[test_idx]), content_v1[test_idx], st]).astype(np.float32)
    print(f"Xv={Xv.shape} Xt={Xt.shape}")

    # benign non-DLL mask from v1 content_is_dll (column 21 of the 100 v1 features, after derived6)
    v1_names = None
    from kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES
    v1_names = list(CONTENT_PE_V1_FEATURE_NAMES)
    j_dll = 6 + v1_names.index("content_is_dll")
    benign_mask = yv == 0
    exe_benign = benign_mask & (content_v1[val_idx, j_dll - 6] <= 0)
    print(f"val benign={benign_mask.sum()} benign_non_dll={exe_benign.sum()} "
          f"(exe fraction of benign={exe_benign.sum() / max(benign_mask.sum(), 1):.3f})")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score

    configs = [
        {"name": "baseline_w1", "w_exe": 1.0, "w_all_benign": 1.0},
        {"name": "w_exe3", "w_exe": 3.0, "w_all_benign": 1.0},
        {"name": "w_exe6", "w_exe": 6.0, "w_all_benign": 1.0},
        {"name": "w_exe10", "w_exe": 10.0, "w_all_benign": 1.0},
        {"name": "benign_all_x3", "w_exe": 3.0, "w_all_benign": 3.0},
    ]

    results = []
    for cfg in configs:
        sw = np.ones(len(yv), dtype=np.float64)
        sw[exe_benign] = cfg["w_exe"]
        if cfg["w_all_benign"] > 1.0:
            sw[benign_mask] = cfg["w_all_benign"]
        clfs = []
        for seed in SEEDS:
            clf = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, early_stopping=False, random_state=seed)
            clf.fit(Xv, yv, sample_weight=sw)
            clfs.append(clf)
        pv = np.mean([c.predict_proba(Xv)[:, 1] for c in clfs], axis=0)
        pt = np.mean([c.predict_proba(Xt)[:, 1] for c in clfs], axis=0)

        # VAL threshold scan: best F1 + goal-aligned (recall>=0.99 -> min FPR)
        best_f1, best_t = 0.0, 0.5
        goal_t, goal_fpr, goal_rec = 0.5, 1.0, 0.0
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
                goal_fpr, goal_rec, goal_t = fpr, rec, t
        print(f"\n[{cfg['name']}] val_bestF1={best_f1:.5f}@t={best_t:.2f} | "
              f"val_recall99: t={goal_t:.2f} fpr={goal_fpr:.4f} rec={goal_rec:.4f}")

        # TEST eval at goal-aligned threshold (the decision threshold per protocol)
        pt_t = (pt >= goal_t).astype(int)
        tp = int(((pt_t == 1) & (yt == 1)).sum()); fp = int(((pt_t == 1) & (yt == 0)).sum())
        fn = int(((pt_t == 0) & (yt == 1)).sum()); tn = int(((pt_t == 0) & (yt == 0)).sum())
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f1t = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        print(f"  [TEST @val-recall99-t] F1={f1t:.5f} P={p_:.4f} R={r_:.4f} FP={fp} FN={fn} "
              f"(errors={fp + fn}) | baseline FP=1047 FN=807")

        # also TEST at best-F1 threshold for reference
        pt_b = (pt >= best_t).astype(int)
        fp_b = int(((pt_b == 1) & (yt == 0)).sum()); fn_b = int(((pt_b == 0) & (yt == 1)).sum())

        results.append({
            "config": cfg["name"], "w_exe": cfg["w_exe"], "w_all_benign": cfg["w_all_benign"],
            "val_best_f1": round(best_f1, 5), "val_best_t": round(best_t, 3),
            "val_goal_t": round(goal_t, 3), "val_goal_fpr": round(goal_fpr, 4), "val_goal_rec": round(goal_rec, 4),
            "test_at_goal": {"f1": round(f1t, 5), "p": round(p_, 4), "r": round(r_, 4),
                             "fp": fp, "fn": fn, "errors": fp + fn},
            "test_at_bestf1": {"fp": fp_b, "fn": fn_b, "errors": fp_b + fn_b},
        })
        # save incremental so a kill doesn't lose everything
        OUT.write_text(json.dumps({"results": results, "elapsed_sec": time.time() - t0},
                                  indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[saved] {OUT}  ({(time.time()-t0)/60:.1f} min)")
    # mark best by goal criterion
    goal_cand = min(results, key=lambda r: r["val_goal_fpr"])
    print(f"[candidate by goal criterion (min val fpr@rec99)]: {goal_cand['config']} "
          f"-> TEST FP={goal_cand['test_at_goal']['fp']} FN={goal_cand['test_at_goal']['fn']} "
          f"R={goal_cand['test_at_goal']['r']:.4f}")


if __name__ == "__main__":
    main()
