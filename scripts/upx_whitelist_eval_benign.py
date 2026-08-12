#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UPX 白名单语料专项评估（良性扩充重训后）。

语料：F:\\私人\\良性文件\\待加入白名单_upx（文件名即 sha256，17,051 个 UPX 加壳白文件）
评估其在良性 test 池中的误报：base 模型（prob>=0.5）与 Stage-2（s2_pred=1）各报多少 FP。
对照扩充前（test 内 3482 个 UPX 白文件：base FP 236 → Stage-2 FP 52）。
输出 reports/full_739k_benign/upx_whitelist_report.json
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
PRED_CSV = PROJECT_ROOT / "reports" / "full_739k_benign" / "test739k_benign_stage2v2_predictions.csv"
OUT = PROJECT_ROOT / "reports" / "full_739k_benign" / "upx_whitelist_report.json"


def main() -> None:
    files = [p for p in UPX_DIR.iterdir() if p.is_file()]
    upx_shas = {p.stem.casefold() for p in files}
    print(f"[upx] corpus files={len(files)}, unique shas={len(upx_shas)}")

    # base_prob meta: index -> (sha, label, prob)
    bp = {}
    with open(BASE_META, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bp[int(r["index"])] = (r["source_sha256"].strip().casefold(),
                                   int(r["label"]), float(r["prob"]))
    print(f"[base_prob] rows={len(bp)}")

    # stage2 predictions: index -> (sha, label, s2_score, s2_pred)
    pred = {}
    with open(PRED_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pred[int(r["index"])] = (r["source_sha256"].strip().casefold(),
                                     int(r["label"]), float(r["s2_score"]), int(r["s2_pred"]))
    print(f"[pred] rows={len(pred)}")

    # 求交集：test 中属于 UPX 白名单语料的样本
    rows = []
    for idx, (sha, label, s2score, s2pred) in pred.items():
        if sha in upx_shas:
            bprob = bp[idx][2]
            rows.append((idx, sha, label, bprob, s2score, s2pred))
    n_upx_test = len(rows)
    upx_benign = [r for r in rows if r[2] == 0]
    n_benign = len(upx_benign)
    print(f"[upx] in test={n_upx_test} (benign={n_benign})")

    # base FP: prob>=0.5 且 label=0
    base_fp = sum(1 for r in upx_benign if r[3] >= 0.5)
    # stage2 FP: s2_pred=1 且 label=0
    s2_fp = sum(1 for r in upx_benign if r[5] == 1)
    # 追加：stage2 高阈值 0.7/0.9 下的 FP
    s2_fp_07 = sum(1 for r in upx_benign if r[4] >= 0.7)
    s2_fp_09 = sum(1 for r in upx_benign if r[4] >= 0.9)

    # 非 UPX 白文件对照
    non_upx_benign_cnt = sum(1 for _idx, (sha, label, _s, _p) in pred.items()
                             if label == 0 and sha not in upx_shas)
    non_upx_base_fp = sum(1 for idx, (sha, label, _s, _p) in pred.items()
                          if label == 0 and sha not in upx_shas and bp[idx][2] >= 0.5)
    non_upx_s2_fp = sum(1 for _idx, (sha, label, _s, s2p) in pred.items()
                        if label == 0 and sha not in upx_shas and s2p == 1)

    rep = {
        "upx_dir": str(UPX_DIR),
        "upx_files_total": len(files),
        "upx_in_test": n_upx_test,
        "test_contains_upx_benign": n_benign > 0,
        "upx_test_base_fp": base_fp,
        "upx_test_stage2_fp": s2_fp,
        "upx_test_stage2_fp@0.7": s2_fp_07,
        "upx_test_stage2_fp@0.9": s2_fp_09,
        "upx_base_fp_rate": round(base_fp / n_benign, 5) if n_benign else None,
        "upx_stage2_fp_rate": round(s2_fp / n_benign, 5) if n_benign else None,
        "non_upx_benign_in_test": non_upx_benign_cnt,
        "non_upx_base_fp": non_upx_base_fp,
        "non_upx_stage2_fp": non_upx_s2_fp,
        "prev_expansion": {"upx_in_test": 3482, "upx_test_base_fp": 236, "upx_test_stage2_fp": 52},
        "stage2_threshold": 0.52,
        "conclusion": (f"良性 test 中 {n_benign} 个 UPX 白文件，base FP {base_fp} → Stage-2 FP {s2_fp}"
                       f"（扩充前 {3482} 个：236 → 52）。"
                       f"FP 率 base {round(base_fp/n_benign,4) if n_benign else 'NA'} → Stage-2 "
                       f"{round(s2_fp/n_benign,4) if n_benign else 'NA'}"
                       + ("，目标(UPX 白零误报)未达成。" if s2_fp > 0 else "，UPX 白文件零误报达成。")),
    }
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
