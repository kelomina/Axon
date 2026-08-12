#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""签名有效性探针：可解析证书 + 签名者 CN + 时间戳，能否强分离 FP vs TN。

仅 locatable 子集（G: 消失）。决定要不要做全量 Authenticode 验证工程。
输出 reports/full_739k_benign/probe_signature_validity.json
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

V1_META = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_pe_v1" / "meta.csv"
PRED_CSV = PROJECT_ROOT / "reports" / "full_739k_benign" / "test739k_benign_stage2v2_predictions.csv"
OUT = PROJECT_ROOT / "reports" / "full_739k_benign" / "probe_signature_validity.json"


def sig_row(path: str):
    """返回签名相关信号；None 若无法解析。"""
    try:
        import pefile
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 512:
            return None
        pe = pefile.PE(data=data)
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_SECURITY"],
        ])
        sec = getattr(pe, "DIRECTORY_ENTRY_SECURITY", None)
        if not sec or not sec.entry:
            return {"sec_present": 0.0, "blob_len": 0.0, "has_cert": 0.0, "has_cn": 0.0, "has_timestamp": 0.0, "n_cn": 0}
        blob = sec.entry.get_data()
        blob_len = float(len(blob))
        # PKCS7 结构：0x30 0x82 len... 粗搜 cert 序列（0x30 0x82）与 CN
        has_cert = 1.0 if re.search(rb"\x30\x82[\x01-\xff][\x00-\xff]\x30\x82", blob) else 0.0
        cns = re.findall(rb"CN=([^,\x00-\x1f\x7f]{2,64})", blob)
        n_cn = len(cns)
        # timestamp: OID 1.3.6.1.4.1.311.3.3.1 (Microsoft timestamp) 或 signingTime
        has_timestamp = 1.0 if re.search(rb"\x2b\x06\x01\x04\x01\x82\x37\x03\x03\x01", blob) else 0.0
        return {"sec_present": 1.0, "blob_len": blob_len, "has_cert": has_cert,
                "has_cn": float(n_cn > 0), "has_timestamp": has_timestamp, "n_cn": min(float(n_cn), 10.0)}
    except Exception:
        return None


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return 0.5
    v = np.concatenate([pos, neg])
    o = np.argsort(v)
    rk = np.empty_like(o, dtype=np.float64)
    rk[o] = np.arange(1, len(v) + 1)
    sv = v[o]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            rk[o[i:j + 1]] = avg
        i = j + 1
    ar = rk[:pos.size]
    return float((ar.sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def main() -> None:
    t0 = time.time()
    print("=== probe signature validity (locatable subset) ===")
    with open(V1_META, encoding="utf-8") as f:
        meta = {int(r["index"]): r for r in csv.DictReader(f)}
    pred = {}
    with open(PRED_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pred[int(r["index"])] = (int(r["label"]), int(r["s2_pred"]))
    benign = [i for i in pred if pred[i][0] == 0]
    FP = [i for i in benign if pred[i][1] == 1]
    random.seed(7)
    TN = random.sample([i for i in benign if pred[i][1] == 0], 3000)

    def resolve(i):
        p = meta.get(i, {}).get("raw_path", "")
        # 盘符映射：G:/H: 均已换盘到 F:（纯前缀替换）
        if p.startswith("G:") or p.startswith("H:"):
            p = "F:" + p[2:]
        return p if os.path.isfile(p) else ""

    rows = {"fp": [], "tn": []}
    for grp, idxs in (("fp", FP), ("tn", TN)):
        for i in idxs:
            p = resolve(i)
            if not p:
                continue
            r = sig_row(p)
            if r is not None:
                rows[grp].append(r)
    print(f"parsed FP={len(rows['fp'])} TN={len(rows['tn'])}")

    rep = {}
    for key in ["sec_present", "has_cert", "has_cn", "has_timestamp", "n_cn"]:
        a = np.asarray([r[key] for r in rows["fp"]], dtype=np.float32)
        b = np.asarray([r[key] for r in rows["tn"]], dtype=np.float32)
        auc = rank_auc(a, b)
        arrow = "FP>TN" if auc > 0.5 else "FP<TN"
        print(f"  {key:14s} FP={a.mean():.3f} TN={b.mean():.3f} AUC={auc:.4f} {arrow}")
        rep[key] = {"fp_mean": round(float(a.mean()), 4), "tn_mean": round(float(b.mean()), 4),
                    "auc": round(float(auc), 4)}
    # 有效签名证据分（cert+cn+timestamp）
    def score(r):
        return r["has_cert"] + r["has_cn"] + r["has_timestamp"]
    sa = np.asarray([score(r) for r in rows["fp"]], dtype=np.float32)
    sb = np.asarray([score(r) for r in rows["tn"]], dtype=np.float32)
    auc = rank_auc(sa, sb)
    print(f"  sig_score(0-3) FP={sa.mean():.3f} TN={sb.mean():.3f} AUC={auc:.4f}")
    rep["sig_score"] = {"fp_mean": round(float(sa.mean()), 4), "tn_mean": round(float(sb.mean()), 4),
                        "auc": round(float(auc), 4)}
    for th in range(4):
        print(f"    score>={th}: FP {(sa>=th).mean():.3f} TN {(sb>=th).mean():.3f}")
    rep["n_fp_parsed"] = len(rows["fp"])
    rep["n_tn_parsed"] = len(rows["tn"])
    rep["elapsed_sec"] = time.time() - t0
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
