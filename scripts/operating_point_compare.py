#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""操作点对比：ver339 vs base331，在匹配 recall 下比 FP。

部署模型 stage2_v2 操作点：TEST recall 0.9930（FP 1047）。
ver339_benignx3 在 val-recall99 阈值下 recall 0.9869（FP 610）。
到底哪个好？在 TEST 上扫阈值，报告每个 config 在指定 recall 档位的 FP。

输出 reports/full_739k_benign/operating_point_compare.json
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
OUT = PROJECT_ROOT / "reports" / "full_739k_benign" / "operating_point_compare.json"
SEEDS = (0, 1, 2)


def load_chunks(d: Path) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "chunk_*.npy")))]).astype(np.float32)


def derived(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return np.column_stack([p, p ** 2, np.abs(p - 0.5), np.log(pc), np.log1p(-pc), np.log(pc / (1 - pc))]).astype(np.float32)


def main() -> None:
    t0 = time.time()
    print("=== operating-point compare: base331 vs ver339 ===")
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
    from kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES
    j_dll = list(CONTENT_PE_V1_FEATURE_NAMES).index("content_is_dll")
    benign_mask = yv == 0

    base_v = np.column_stack([derived(base_prob[val_idx]), content_v1[val_idx], v2]).astype(np.float32)
    base_t = np.column_stack([derived(base_prob[test_idx]), content_v1[test_idx], st]).astype(np.float32)
    feats = {
        "base331": (base_v, base_t),
        "ver339": (np.column_stack([base_v, ver_v]).astype(np.float32),
                   np.column_stack([base_t, ver_t]).astype(np.float32)),
    }

    from sklearn.ensemble import HistGradientBoostingClassifier

    preds = {}
    for name, (Xv, Xt) in feats.items():
        clfs = []
        for seed in SEEDS:
            clf = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, early_stopping=False, random_state=seed)
            clf.fit(Xv, yv, sample_weight=np.where(benign_mask, 3.0, 1.0))
            clfs.append(clf)
        preds[name] = np.mean([c.predict_proba(Xt)[:, 1] for c in clfs], axis=0)
        print(f"[{name}] trained")

    n_ben_test = int((yt == 0).sum())
    n_mal_test = int((yt == 1).sum())
    report = {"n_ben_test": n_ben_test, "n_mal_test": n_mal_test, "configs": {}}
    for name, pt in preds.items():
        rows = []
        for t in [x / 1000 for x in range(250, 651)]:
            p = (pt >= t).astype(int)
            fp = int(((p == 1) & (yt == 0)).sum())
            fn = int(((p == 0) & (yt == 1)).sum())
            rec = 1 - fn / n_mal_test
            rows.append({"t": round(t, 3), "fp": fp, "fn": fn, "recall": round(rec, 5), "fpr": round(fp / n_ben_test, 5)})
        # pick rows at specified recall bands
        def row_at_recall(target):
            best = min(rows, key=lambda r: abs(r["recall"] - target))
            return best
        at_09930 = row_at_recall(0.9930)   # deployed recall
        at_09900 = row_at_recall(0.9900)
        at_09870 = row_at_recall(0.9870)   # val-recall99 op
        # also: max recall at which FP minimal
        report["configs"][name] = {
            "at_recall_0.9930": at_09930,
            "at_recall_0.9900": at_09900,
            "at_recall_0.9870": at_09870,
        }
        print(f"\n[{name}]")
        print(f"  @recall~0.9930: t={at_09930['t']} FP={at_09930['fp']} FN={at_09930['fn']} (deployed op)")
        print(f"  @recall~0.9900: t={at_09900['t']} FP={at_09900['fp']} FN={at_09900['fn']}")
        print(f"  @recall~0.9870: t={at_09870['t']} FP={at_09870['fp']} FN={at_09870['fn']} (val-recall99 op)")

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
