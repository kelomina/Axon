#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标签治理复核队列（良性扩充重训后 test+val 集）。

识别疑似标签噪声样本，产出人工复核队列：
  1) FN 侧（疑似误标黑为白）：label=恶意 但 Stage-2 s2<0.1（模型>90% 确信良性）
  2) FP 侧（疑似误标白为黑）：label=良性 但基座 base_prob>0.9（>90% 确信恶意）
  3) 跨树 sha 冲突：同一 sha 同时出现在良性树与恶意树（用 dir_index.pkl 判定）
  4) 不可定位：raw_path 前缀交换后仍无文件（数据质量问题）

辅助证据：base_prob / s2_score / s2_pred / content_is_dll / 版本资源位 / 原始路径（G:/H:→F:）。
定位用 content_versionstr/dir_index.pkl（listdir 一次建集），不逐文件 stat。
输出 reports/full_739k_benign/label_governance/{review_queue.csv, summary.json}
"""
from __future__ import annotations

import csv
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BASE = PROJECT_ROOT / "reports" / "full_739k_benign"
PRED_CSV = BASE / "test739k_benign_stage2v2_predictions.csv"
META_CSV = BASE / "content_pe_v1" / "meta.csv"
V2_META = BASE / "content_v2string" / "meta.json"
BASE_PROB_DIR = BASE / "base_prob"
V1_DIR = BASE / "content_pe_v1"
V2_DIR = BASE / "content_v2string"
DIR_INDEX = BASE / "content_versionstr" / "dir_index.pkl"
OUT_DIR = BASE / "label_governance"

FN_S2_CUT = 0.1   # FN 候选：s2<0.1 即模型>90% 确信良性
FP_BASE_CUT = 0.9  # FP 候选：base_prob>0.9 即基座>90% 确信恶意


def load_chunks(d: Path) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(d / "chunk_*.npy")))]).astype(np.float32)


def tree_of_dir(d: str) -> str:
    """按目录路径判定所属语料树：benign / malware / unknown。"""
    low = d.casefold()
    if "恶意" in low or "malicious" in low or "待拉黑" in low:
        return "malware"
    if "良性" in low or "benign" in low or "待加入白名单" in low:
        return "benign"
    return "unknown"


def main() -> None:
    import time
    t0 = time.time()
    print("=== label governance review queue (test+val) ===")

    # ---- 数据 ----
    with open(V2_META, encoding="utf-8") as f:
        m = json.load(f)
    val_idx = [int(i) for i in m["val_indices"]]
    test_idx = [int(i) for i in m["test_indices"]]

    meta = {}
    for r in csv.DictReader(open(META_CSV, encoding="utf-8")):
        i = int(r["index"])
        meta[i] = {"sha": r["source_sha256"].strip().casefold(),
                   "raw_path": r["raw_path"], "label": int(r["label"])}
    shas = np.asarray([meta[i]["sha"] for i in range(len(meta))])
    # content 特征（DLL / 版本资源位）
    v1 = load_chunks(V1_DIR)
    from kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES as N1
    j_dll = list(N1).index("content_is_dll")
    v2 = np.load(V2_DIR / "test.npy").astype(np.float32)
    with open(V2_DIR / "meta.json", encoding="utf-8") as f:
        v2meta = json.load(f)
    v2_test_order = {int(g): k for k, g in enumerate(v2meta["test_indices"])}
    v2_val_order = {int(g): k for k, g in enumerate(v2meta["val_indices"])}
    j_ver = v2meta["v2_names"].index("v2_resource_type_version_present")
    v2_val = np.load(V2_DIR / "val.npy").astype(np.float32)

    # 预测（test）
    pred = {}
    for r in csv.DictReader(open(PRED_CSV, encoding="utf-8")):
        i = int(r["index"])
        pred[i] = {"label": int(r["label"]), "s2": float(r["s2_score"]),
                   "s2_pred": int(r["s2_pred"])}

    # 基座概率（全 manifest 顺序，按 index 取）
    base_prob = load_chunks(BASE_PROB_DIR)
    print(f"base_prob shape={base_prob.shape}  (n={len(base_prob)})")

    # ---- 目录索引 → 良/恶 sha 集 + 定位 ----
    with open(DIR_INDEX, "rb") as f:
        dindex = pickle.load(f)
    benign_stems, malware_stems = set(), set()
    for d, s in dindex.items():
        tt = tree_of_dir(d)
        for fn in s:
            stem = os.path.splitext(fn)[0].casefold()
            if tt == "benign":
                benign_stems.add(stem)
            elif tt == "malware":
                malware_stems.add(stem)
    print(f"[trees] benign stems={len(benign_stems):,}  malware stems={len(malware_stems):,}")

    def resolve(p: str) -> str:
        if p.startswith("G:") or p.startswith("H:"):
            p = "F:" + p[2:]
        return p

    def locatable(i: int) -> tuple[str, bool]:
        p = resolve(meta[i]["raw_path"])
        if not p:
            return "", False
        s = dindex.get(os.path.dirname(p))
        if s is None:
            return p, False
        return p, (os.path.basename(p) in s)

    # ---- 候选集 ----
    rows = []
    for i in test_idx + val_idx:
        md = meta[i]
        pr = pred.get(i)
        bp = float(base_prob[i])
        p_res, ok = locatable(i)
        sha = md["sha"]
        conflict = "1" if (sha in benign_stems and sha in malware_stems) else "0"
        if i in v2_test_order:
            ver = int(v2[v2_test_order[i], j_ver] > 0)
        elif i in v2_val_order:
            ver = int(v2_val[v2_val_order[i], j_ver] > 0)
        else:
            ver = None
        rec = {
            "index": i, "split": "test" if i in test_idx else "val",
            "sha256": sha, "label": md["label"],
            "base_prob": round(bp, 4), "s2_score": round(pr["s2"], 4) if pr else None,
            "s2_pred": pr["s2_pred"] if pr else None,
            "is_dll": int(v1[i, j_dll] > 0),
            "has_version_res": ver,
            "cross_tree_conflict": conflict,
            "locatable": int(ok), "raw_path": p_res,
        }
        # 归属候选类别
        cat = None
        if md["label"] == 1 and pr and pr["s2_pred"] == 0 and pr["s2"] < FN_S2_CUT:
            cat = "fn_noise_suspected_mislabel_black"
        elif md["label"] == 0 and pr and pr["s2_pred"] == 1 and bp > FP_BASE_CUT:
            cat = "fp_noise_suspected_mislabel_white"
        if cat:
            rec["candidate"] = cat
            rows.append(rec)
        elif conflict == "1":
            rec["candidate"] = "cross_tree_conflict"
            rows.append(rec)
        elif not ok:
            rec["candidate"] = "unlocatable"
            rows.append(rec)

    # ---- 汇总 ----
    from collections import Counter
    by_cat = Counter(r["candidate"] for r in rows)
    by_split = Counter((r["candidate"], r["split"]) for r in rows)
    print(f"\ncandidates: {len(rows):,}")
    for k, v in by_cat.most_common():
        print(f"  {k:38s} {v:6,}")
    print("\nby split:")
    for (cat, sp), n in sorted(by_split.items()):
        print(f"  {cat:38s} {sp:4s} {n:6,}")

    # 排序：FP 侧 base 置信降序 / FN 侧 s2 升序（最确信的错误最靠前）
    def sort_key(r):
        if r["candidate"] == "fn_noise_suspected_mislabel_black":
            return (0, r["s2_score"])
        if r["candidate"] == "fp_noise_suspected_mislabel_white":
            return (1, -r["base_prob"])
        if r["candidate"] == "cross_tree_conflict":
            return (2, 0)
        return (3, 0)
    rows.sort(key=sort_key)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["index", "split", "sha256", "label", "base_prob", "s2_score", "s2_pred",
              "is_dll", "has_version_res", "cross_tree_conflict", "locatable",
              "candidate", "raw_path"]
    with open(OUT_DIR / "review_queue.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = {
        "fn_cut_s2_lt": FN_S2_CUT, "fp_cut_base_gt": FP_BASE_CUT,
        "n_candidates": len(rows),
        "by_candidate": dict(by_cat),
        "by_candidate_split": {f"{c}|{s}": n for (c, s), n in sorted(by_split.items())},
        "note": ("复核动作：fn_noise 疑似真良性误标黑（应改标 0）；fp_noise 疑似真恶意误标白（应改标 1）；"
                 "cross_tree_conflict 同 sha 两树并存；unlocatable 需查数据来源。"),
        "elapsed_sec": time.time() - t0,
    }
    json.dump(summary, open(OUT_DIR / "summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[saved] {OUT_DIR / 'review_queue.csv'} ({len(rows):,} rows)")
    print(f"[saved] {OUT_DIR / 'summary.json'}")
    print(f"[done] {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
