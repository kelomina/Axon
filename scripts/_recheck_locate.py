#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修正盘符映射后重新统计定位率。"""
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

pred = {}
with open(PROJECT_ROOT / "reports/full_739k_benign/test739k_benign_stage2v2_predictions.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        pred[int(r["index"])] = (int(r["label"]), int(r["s2_pred"]))
m = json.load(open(PROJECT_ROOT / "reports/full_739k_benign/content_v2string/meta.json", encoding="utf-8"))
ti = [int(i) for i in m["test_indices"]]
vi = [int(i) for i in m["val_indices"]]
meta = {}
with open(PROJECT_ROOT / "reports/full_739k_benign/content_pe_v1/meta.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        meta[int(r["index"])] = r["raw_path"]


def resolve(p):
    if p.startswith("G:"):
        return "F:" + p[2:]
    if p.startswith("H:"):
        return "F:" + p[2:]
    return p


def locatable(i):
    p = meta.get(i, "")
    return bool(p) and os.path.isfile(resolve(p))


FP = [i for i in ti if pred[i][0] == 0 and pred[i][1] == 1]
loc_fp = sum(1 for i in FP if locatable(i))
print(f"FP={len(FP)}  locatable={loc_fp} ({100*loc_fp/len(FP):.1f}%)")

# overall val/test coverage
loc_v = sum(1 for i in vi if locatable(i))
loc_t = sum(1 for i in ti if locatable(i))
print(f"VAL locatable={loc_v}/{len(vi)} ({100*loc_v/len(vi):.1f}%)")
print(f"TEST locatable={loc_t}/{len(ti)} ({100*loc_t/len(ti):.1f}%)")

# benign non-DLL specifically
from kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES as N1
import numpy as np, glob, sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))
V1 = np.concatenate([np.load(f) for f in sorted(glob.glob(str(PROJECT_ROOT / "reports/full_739k_benign/content_pe_v1/chunk_*.npy")))]).astype(np.float32)
j_dll = N1.index("content_is_dll")
labs = {}
with open(PROJECT_ROOT / "reports/full_739k_benign/content_pe_v1/meta.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        labs[int(r["index"])] = int(r["label"])
ben_exe_test = [i for i in ti if labs[i] == 0 and V1[i, j_dll] <= 0]
loc_benexe = sum(1 for i in ben_exe_test if locatable(i))
print(f"TEST benign non-DLL n={len(ben_exe_test)} locatable={loc_benexe} ({100*loc_benexe/max(len(ben_exe_test),1):.1f}%)")

# prefix breakdown of FP
from collections import Counter
c = Counter()
for i in FP:
    p = meta.get(i, "")
    if p.startswith("G:"):
        c["G"] += 1
    elif p.startswith("H:"):
        c["H"] += 1
    elif p.startswith("E:"):
        c["E"] += 1
    elif p.startswith("F:"):
        c["F"] += 1
    else:
        c["other"] += 1
print("FP by raw_path prefix:", dict(c))
