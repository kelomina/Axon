#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交叉验证：v2_resource_type_version_present vs 原始版本字符串检测是否一致。

假设：v2 特征可能漏检 RT_VERSION，导致模型没学到版本信号。
对 locatable FP + 抽样 TN 同时算两个信号，输出一致率。
"""
from __future__ import annotations

import csv
import glob
import json
import os
import random
import sys
import time

import numpy as np

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pefile  # noqa: E402

V1_META = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_pe_v1" / "meta.csv"
PRED_CSV = PROJECT_ROOT / "reports" / "full_739k_benign" / "test739k_benign_stage2v2_predictions.csv"
V2 = np.load(PROJECT_ROOT / "reports" / "full_739k_benign" / "content_v2string" / "test.npy").astype(np.float32)

WANT = {"CompanyName", "FileDescription", "ProductName", "OriginalFilename",
        "FileVersion", "ProductVersion", "LegalCopyright", "InternalName"}


def raw_version_flag(path: str):
    """True if VS_VERSION_INFO has any non-empty target string; None if unparseable."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 512:
            return False
        pe = pefile.PE(data=data, fast_load=True)
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"],
        ])
        found = set()
        for flist in getattr(pe, "FileInfo", []) or []:
            for fe in flist:
                for tbl in getattr(fe, "StringTable", []) or []:
                    for k in (getattr(tbl, "entries", None) or {}):
                        key = k.decode("utf-8", "replace") if isinstance(k, bytes) else str(k)
                        if key in WANT:
                            found.add(key)
        return bool(found)
    except Exception:
        return False


def main() -> None:
    t0 = time.time()
    with open(V1_META, encoding="utf-8") as f:
        meta = {int(r["index"]): r for r in csv.DictReader(f)}
    pred = {}
    with open(PRED_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pred[int(r["index"])] = (int(r["label"]), int(r["s2_pred"]))
    test_benign = [i for i in pred if pred[i][0] == 0]
    FP = [i for i in test_benign if pred[i][1] == 1]
    random.seed(1)
    TN = random.sample([i for i in test_benign if pred[i][1] == 0], 3000)

    # v2 index: need test-array position; meta.json maps test_indices
    with open(PROJECT_ROOT / "reports" / "full_739k_benign" / "content_v2string" / "meta.json", encoding="utf-8") as f:
        m2 = json.load(f)
    order = {int(g): k for k, g in enumerate(m2["test_indices"])}
    j_ver = m2["v2_names"].index("v2_resource_type_version_present")

    def resolve(i):
        p = meta.get(i, {}).get("raw_path", "")
        # 盘符映射：G:/H: 均已换盘到 F:（纯前缀替换）
        if p.startswith("G:") or p.startswith("H:"):
            p = "F:" + p[2:]
        return p if os.path.isfile(p) else ""

    agg = {}
    for grp, idxs in (("fp", FP), ("tn", TN)):
        n = 0
        both, v2only, rawonly, none = 0, 0, 0, 0
        for i in idxs:
            p = resolve(i)
            if not p:
                continue
            v2flag = V2[order[i], j_ver] > 0
            raw = raw_version_flag(p)
            n += 1
            if raw and v2flag:
                both += 1
            elif v2flag and not raw:
                v2only += 1
            elif raw and not v2flag:
                rawonly += 1
            else:
                none += 1
        agg[grp] = {"n": n, "both": both, "v2_only": v2only, "raw_only": rawonly,
                    "neither": none, "v2_frac": round((both + v2only) / n, 3),
                    "raw_frac": round((both + rawonly) / n, 3)}
        print(f"{grp}: n={n} both={both} v2_only={v2only} raw_only={rawonly} neither={none}")
        print(f"   v2_frac={agg[grp]['v2_frac']} raw_frac={agg[grp]['raw_frac']} "
              f"disagree={(v2only + rawonly)} ({100*(v2only + rawonly)/n:.1f}%)")
    print(f"elapsed {time.time()-t0:.0f}s")
    json.dump(agg, open(PROJECT_ROOT / "reports" / "full_739k_benign" / "v2_vs_raw_version.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
