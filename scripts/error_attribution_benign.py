#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""错误归因：良性扩充重训后 Stage-2 剩余 FP/FN 的构成分析。

输入：base_prob/meta.csv（index->sha,label,prob,path）、
      test739k_benign_stage2v2_predictions.csv（index->sha,label,s2_score,s2_pred）、
      UPX 白名单语料 sha 集、旧 manifest 边界（738,983 = 扩充追加起点）。
切分维度：UPX/非UPX、旧良性/新良性、DLL/EXE、基座置信。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPX_DIR = Path(r"F:\私人\良性文件\待加入白名单_upx")
BASE_META = PROJECT_ROOT / "reports" / "full_739k_benign" / "base_prob" / "meta.csv"
CONTENT_V1_META = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_pe_v1" / "meta.csv"
PRED_CSV = PROJECT_ROOT / "reports" / "full_739k_benign" / "test739k_benign_stage2v2_predictions.csv"
OUT = PROJECT_ROOT / "reports" / "full_739k_benign" / "error_attribution.json"
OLD_MANIFEST_N = 738983  # 扩充前样本数（新良性从该索引追加）


def main() -> None:
    upx_shas = {p.stem.casefold() for p in UPX_DIR.iterdir() if p.is_file()}

    # base_prob meta
    bp = {}
    with open(BASE_META, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            i = int(r["index"])
            bp[i] = (r["source_sha256"].strip().casefold(), int(r["label"]),
                     float(r["prob"]), r["cache_path"])
    # content_v1 meta: raw_path（含原始扩展名）
    raw_path = {}
    with open(CONTENT_V1_META, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raw_path[int(r["index"])] = r.get("raw_path", "")
    # pred
    pred = {}
    with open(PRED_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            i = int(r["index"])
            pred[i] = (r["source_sha256"].strip().casefold(), int(r["label"]),
                       float(r["s2_score"]), int(r["s2_pred"]))
    print(f"base_prob={len(bp)} pred={len(pred)}")

    def is_dll(idx: int) -> bool:
        p = raw_path.get(idx, "").strip().casefold()
        if not p:
            return False
        return p.endswith(".dll") or ".dll" in p

    def is_new(idx: int) -> bool:
        return idx >= OLD_MANIFEST_N

    FP, FN = [], []
    for idx, (sha, label, s2s, s2p) in pred.items():
        if label == 0 and s2p == 1:
            FP.append(idx)
        elif label == 1 and s2p == 0:
            FN.append(idx)

    def summarize(idx_list, tag):
        n = len(idx_list)
        if not n:
            print(f"  [{tag}] 0"); return {}
        upx = sum(1 for i in idx_list if pred[i][0] in upx_shas)
        new = sum(1 for i in idx_list if is_new(i))
        dll = sum(1 for i in idx_list if is_dll(bp[i][3]))
        high_conf = sum(1 for i in idx_list if bp[i][2] >= 0.9)  # 基座>90%置信
        s2s = [pred[i][2] for i in idx_list]
        bs = [bp[i][2] for i in idx_list]
        rep = {
            "n": n,
            "upx": upx, "non_upx": n - upx,
            "new_benign_pool": new, "old_benign_pool": n - new,
            "dll": dll, "exe": n - dll,
            "base_conf>=0.9": high_conf,
            "s2_score_mean": round(float(np.mean(s2s)), 4),
            "s2_score_min": round(float(np.min(s2s)), 4),
            "base_prob_mean": round(float(np.mean(bs)), 4),
        }
        print(f"  [{tag}] n={n} upx={upx} non_upx={n-upx} new={new} old={n-new} "
              f"dll={dll} exe={n-dll} base>=0.9={high_conf} "
              f"s2mean={rep['s2_score_mean']} s2min={rep['s2_score_min']}")
        return rep

    print("=== FP (label0 判黑) ===")
    fp_rep = summarize(FP, "FP")
    print("=== FN (label1 判白) ===")
    fn_rep = summarize(FN, "FN")

    # FN 按基座置信分层（标签噪声嫌疑）
    fn_by_base = {"<0.1": 0, "0.1-0.5": 0, "0.5-0.9": 0, ">=0.9": 0}
    for i in FN:
        b = bp[i][2]
        if b < 0.1: fn_by_base["<0.1"] += 1
        elif b < 0.5: fn_by_base["0.1-0.5"] += 1
        elif b < 0.9: fn_by_base["0.5-0.9"] += 1
        else: fn_by_base[">=0.9"] += 1
    print("   FN 按基座置信:", fn_by_base)

    # FP 中高基座置信的白文件（模型"自信判黑"的结构性错误 → 纠错层的最大机会）
    fp_base_conf = sum(1 for i in FP if bp[i][2] >= 0.9)
    print(f"   FP 中基座>=0.9: {fp_base_conf}")

    rep = {"fp": fp_rep, "fn": fn_rep, "fn_by_base_conf": fn_by_base,
           "fp_base_conf_ge_09": fp_base_conf}
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
