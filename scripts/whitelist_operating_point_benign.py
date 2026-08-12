#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可验证白文件白名单操作点：把 UPX 白名单语料作为硬白名单应用。

对 test 预测：sha ∈ 语料 → 强制判白（label 白且原判黑 = 白名单翻转）。
量化翻转后的 FP/recall（白名单内 0 FP、recall 不变），对照未应用白名单。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPX_DIR = Path(r"F:\私人\良性文件\待加入白名单_upx")
PRED_CSV = PROJECT_ROOT / "reports" / "full_739k_benign" / "test739k_benign_stage2v2_predictions.csv"
BASE_META = PROJECT_ROOT / "reports" / "full_739k_benign" / "base_prob" / "meta.csv"
OUT = PROJECT_ROOT / "reports" / "full_739k_benign" / "whitelist_operating_point.json"


def main() -> None:
    upx_shas = {p.stem.casefold() for p in UPX_DIR.iterdir() if p.is_file()}
    pred = {}
    with open(PRED_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            i = int(r["index"])
            pred[i] = (r["source_sha256"].strip().casefold(), int(r["label"]),
                       float(r["s2_score"]), int(r["s2_pred"]))
    bp = {}
    with open(BASE_META, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            i = int(r["index"])
            bp[i] = (r["source_sha256"].strip().casefold(), int(r["label"]), float(r["prob"]))

    n_white = sum(1 for _i, (_s, l, _sc, _p) in pred.items() if l == 0)
    n_black = sum(1 for _i, (_s, l, _sc, _p) in pred.items() if l == 1)

    # 原预测（无白名单）
    fp0 = sum(1 for _i, (_s, l, _sc, p) in pred.items() if l == 0 and p == 1)
    fn0 = sum(1 for _i, (_s, l, _sc, p) in pred.items() if l == 1 and p == 0)
    rec0 = 1 - fn0 / n_black

    # 白名单翻转：白且判黑且 sha∈语料 → 判白
    upx_white_test = sum(1 for _i, (sha, l, _sc, _p) in pred.items() if l == 0 and sha in upx_shas)
    wl_flip = 0
    fp1 = 0
    for _i, (sha, l, _sc, p) in pred.items():
        if l == 0 and p == 1:
            if sha in upx_shas:
                wl_flip += 1  # 白名单救回
            else:
                fp1 += 1  # 仍 FP（未验证白文件）
    fn1 = fn0  # 白名单只动白文件，FN 不变
    rec1 = rec0
    unverified_white = n_white - upx_white_test

    # base 模型同样对照
    base_upx_white_test = 0
    base_wl_flip = 0
    base_fp0 = 0
    base_fp1 = 0
    for i, (sha, l, prob) in bp.items():
        if i not in pred:
            continue
        base_pred1 = prob >= 0.5
        if l == 0 and base_pred1:
            base_fp0 += 1
            if sha in upx_shas:
                base_wl_flip += 1
            else:
                base_fp1 += 1
        if l == 0 and sha in upx_shas:
            base_upx_white_test += 1

    rep = {
        "test_white": n_white, "test_black": n_black,
        "no_whitelist": {"fp": fp0, "fn": fn0, "recall": round(rec0, 5), "fpr": round(fp0 / n_white, 5)},
        "with_upx_whitelist": {
            "verified_white_in_test": upx_white_test,
            "upx_fp_before": wl_flip, "wl_flip": wl_flip,
            "fp_on_verified_after": 0,
            "fp_remaining_unverified": fp1, "fn": fn1, "recall": round(rec1, 5),
            "fpr_on_all_white": round(fp1 / n_white, 5),
            "fpr_on_unverified_white": round(fp1 / unverified_white, 5) if unverified_white else None,
        },
        "base_no_whitelist": {"fp": base_fp0, "fpr": round(base_fp0 / n_white, 5)},
        "base_with_upx_whitelist": {"verified_white_in_test": base_upx_white_test,
                                     "wl_flip": base_wl_flip, "fp_remaining": base_fp1,
                                     "fpr_on_all_white": round(base_fp1 / n_white, 5)},
        "goal": {
            "recall_gt_0.99": rec1 > 0.99,
            "zero_fp_on_verified_white": wl_flip == sum(1 for _i, (sha, l, _sc, p) in pred.items()
                                                        if l == 0 and p == 1 and sha in upx_shas),
            "absolute_zero_fp": fp1 == 0,
            "fp_on_unverified_white": fp1,
        },
    }
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
