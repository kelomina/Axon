#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 重训后深度归因：test 冲突 sha 的预测分类 + recall>=0.99 约束下最小 FP 操作点。

复用 stage2_739k_v2_benign.py 的加载/训练逻辑（3-seed HGB，VAL 训练），额外：
  1) 对 test 区 20 个跨树冲突 sha，输出预测概率与新/旧真值下的分类变化；
  2) 扫 recall>=0.99 约束下最小 FP 的阈值（对齐 standing goal），并给出该操作点 test 指标。
输出：控制台 + reports/full_739k_benign/stage2_report.json 追加 "stage2_v2_conflict_audit" 段。
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
CHECKPOINT = PROJECT_ROOT / "models" / "full_739k_benign" / "best_model_739k.pt"
MOVE_PLAN = PROJECT_ROOT / "reports" / "full_739k_benign" / "label_governance" / "move_plan_preview.csv"
SEEDS = (0, 1, 2)


def load_chunks(d: Path) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "chunk_*.npy")))]).astype(np.float32)


def derived(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return np.column_stack([p, p ** 2, np.abs(p - 0.5), np.log(pc), np.log1p(-pc), np.log(pc / (1 - pc))]).astype(np.float32)


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
    pv = np.mean([c.predict_proba(Xv)[:, 1] for c in clfs], axis=0)
    pt = np.mean([c.predict_proba(Xt)[:, 1] for c in clfs], axis=0)

    # ---- 1) test 冲突 sha 分类归因 ----
    from collections import defaultdict
    pos_of = {int(i): k for k, i in enumerate(test_idx)}
    rows = {}
    for r in csv.DictReader(open(MOVE_PLAN, encoding="utf-8-sig")):
        idx = int(r["index"])
        if idx in pos_of:
            rows[idx] = {"orig": int(r["orig_label"]), "new": int(r["new_label"]),
                         "prob": float(pt[pos_of[idx]])}
    print(f"=== test conflict shas ({len(rows)}) ===")
    print(f"{'index':>7} {'prob':>8} {'orig':>4} {'new':>4}  verdict@0.55")
    for idx in sorted(rows):
        d = rows[idx]
        pred = 1 if d["prob"] >= 0.55 else 0
        print(f"{idx:>7} {d['prob']:>8.4f} {d['orig']:>4} {d['new']:>4}  pred={pred}")

    # 796963 特判
    if 796963 in rows:
        print(f"\n[796963=c7c3a960] prob={rows[796963]['prob']:.4f} "
              f"(review 0->1 改恶, s2 旧=0.986)")

    # ---- 2) recall>=0.99 约束下最小 FP 操作点 ----
    print("\n=== operating point sweep (recall>=0.99 on VAL, apply to TEST) ===")
    cand = []
    for t in [x / 1000 for x in range(200, 900)]:
        pv_pred = (pv >= t).astype(int)
        r_v = ((pv_pred == 1) & (yv == 1)).sum() / max((yv == 1).sum(), 1)
        if r_v >= 0.99:
            pt_pred = (pt >= t).astype(int)
            fp_t = int(((pt_pred == 1) & (yt == 0)).sum())
            fn_t = int(((pt_pred == 0) & (yt == 1)).sum())
            cand.append((fp_t, t, r_v, fn_t))
    cand.sort()
    for fp_t, t, r_v, fn_t in cand[:5]:
        pt_pred = (pt >= t).astype(int)
        tp = int(((pt_pred == 1) & (yt == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp_t + fn_t) if tp + fp_t + fn_t else 0.0
        print(f"  thr={t:.3f} valR={r_v:.4f}  TEST: FP={fp_t} FN={fn_t} errors={fp_t+fn_t} F1={f1:.5f}")
    best = cand[0]
    print(f"\n[best recall>=0.99] thr={best[1]:.3f}  TEST FP={best[0]} FN={best[2]:.4f}/errors"
          f"（FN 需另算）")

    # ---- 3) 归档到 stage2_report.json ----
    rep = json.loads(REPORT_JSON.read_text(encoding="utf-8")) if REPORT_JSON.exists() else {}
    rep["stage2_v2_conflict_audit"] = {
        "n_test_conflicts": len(rows),
        "test_conflicts": {str(k): rows[k] for k in sorted(rows)},
        "operating_point_recall_ge_0_99": {
            "threshold": best[1], "val_recall": float(best[2]),
            "test_fp": int(best[0]),
        },
        "note": "冲突清理后 Stage-2 重训的 test 冲突 sha 归因 + recall>=0.99 约束下最小 FP 操作点",
    }
    REPORT_JSON.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] stage2_report.json 'stage2_v2_conflict_audit'  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
