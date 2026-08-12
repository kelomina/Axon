#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小样本探针：版本信息字符串内容 + 签名证书，能不能区分 FP vs TN 良性。

只对 test 里 FP(1047) + 抽样 TN(~3000) 提取，验证后再决定是否全量重提 + 重训 Stage-2。
输出 reports/full_739k_benign/probe_benign_exe_signals.json
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES as N1

PRED_CSV = PROJECT_ROOT / "reports" / "full_739k_benign" / "test739k_benign_stage2v2_predictions.csv"
V1_META = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_pe_v1" / "meta.csv"
OUT = PROJECT_ROOT / "reports" / "full_739k_benign" / "probe_benign_exe_signals.json"

PE = None


def load_pe():
    global PE
    if PE is None:
        import pefile
        PE = pefile
    return PE


def sig_present(pe) -> bool:
    return bool(getattr(pe, "DIRECTORY_ENTRY_SECURITY", None))


def cert_cn(pe):
    """从证书表 blob 里尽量抠出 CN 字符串（粗糙 ASN.1 搜索，够判别用）。"""
    try:
        sec = getattr(pe, "DIRECTORY_ENTRY_SECURITY", None)
        if not sec or not sec.entry:
            return []
        blob = sec.entry.get_data()
        if not blob:
            return []
        # 在 PKCS7 里找 "CN=" 后面的字符串
        out = []
        for m in re.finditer(rb"CN=([^,\x00-\x1f]{2,64})", blob):
            try:
                out.append(m.group(1).decode("latin1"))
            except Exception:
                continue
        return out
    except Exception:
        return []


def version_strings(pe):
    """从 VS_VERSION_INFO 提取关键字符串字段。返回 {field: text}。"""
    info = {}
    WANT = {"CompanyName", "FileDescription", "ProductName", "OriginalFilename",
            "FileVersion", "ProductVersion", "LegalCopyright", "InternalName"}
    try:
        for flist in getattr(pe, "FileInfo", []) or []:
            for fe in flist:
                for tbl in getattr(fe, "StringTable", []) or []:
                    entries = getattr(tbl, "entries", None) or {}
                    for k, v in entries.items():
                        key = k.decode("utf-8", "replace") if isinstance(k, bytes) else str(k)
                        if key in WANT:
                            vv = (v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)).strip()
                            if vv:
                                info.setdefault(key, vv)
    except Exception:
        pass
    return info


def parse_one(path: str) -> dict:
    r = {"ok": False}
    try:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 512:
            return r
        pefile = load_pe()
        pe = pefile.PE(data=data)
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_SECURITY"],
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"],
        ])
        r["sig_present"] = sig_present(pe)
        r["certs"] = cert_cn(pe)
        vs = version_strings(pe)
        r["has_company"] = bool(vs.get("CompanyName"))
        r["has_filedesc"] = bool(vs.get("FileDescription"))
        r["has_product"] = bool(vs.get("ProductName"))
        r["has_version_str"] = bool(vs.get("FileVersion") or vs.get("ProductVersion"))
        r["has_original_fn"] = bool(vs.get("OriginalFilename"))
        r["has_legal_cp"] = bool(vs.get("LegalCopyright"))
        r["n_string_fields"] = len(vs)
        r["ok"] = True
    except Exception:
        pass
    return r


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
    global re
    import re
    t0 = time.time()
    print("=== probe benign-exe signals (version strings + signature) ===")

    with open(V1_META, encoding="utf-8") as f:
        meta = {int(r["index"]): r for r in csv.DictReader(f)}

    pred = {}
    with open(PRED_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pred[int(r["index"])] = (int(r["label"]), int(r["s2_pred"]))
    test_idx = [int(i) for i in pred if pred[i][0] == 0]
    FP = [i for i in test_idx if pred[i][1] == 1]
    random.seed(42)
    TN = random.sample([i for i in test_idx if pred[i][1] == 0], 3000)
    print(f"FP={len(FP)} TN_sample={len(TN)}")

    def resolve(i):
        p = meta.get(i, {}).get("raw_path", "")
        # 盘符映射：G:/H: 均已换盘到 F:（纯前缀替换）
        if p.startswith("G:") or p.startswith("H:"):
            p = "F:" + p[2:]
        return p

    results = {"fp": [], "tn": []}
    missing = {"fp": 0, "tn": 0}
    for grp, idxs in (("fp", FP), ("tn", TN)):
        for i in idxs:
            p = resolve(i)
            if not p or not Path(p).is_file():
                missing[grp] += 1
                continue
            d = parse_one(p)
            if d["ok"]:
                d["index"] = i
                results[grp].append(d)
    print(f"parsed FP={len(results['fp'])} TN={len(results['tn'])}  missing FP={missing['fp']} TN={missing['tn']}  ({time.time()-t0:.0f}s)")

    def col(vals, key):
        return np.asarray([1.0 if v.get(key) else 0.0 for v in vals], dtype=np.float32)

    fp = results["fp"]
    tn = results["tn"]
    rep = {}
    print("\n--- 单个信号 rank-AUC (FP vs TN) ---")
    for key in ["sig_present", "has_company", "has_filedesc", "has_product",
                "has_version_str", "has_original_fn", "has_legal_cp", "n_string_fields"]:
        a = col(fp, key)
        b = col(tn, key)
        auc = rank_auc(a, b)
        arrow = "FP>TN" if auc > 0.5 else "FP<TN"
        print(f"  {key:18s} FP={a.mean():.3f} TN={b.mean():.3f} AUC={auc:.4f} {arrow}")
        rep[key] = {"fp_mean": round(float(a.mean()), 4), "tn_mean": round(float(b.mean()), 4),
                    "auc": round(float(auc), 4)}

    # 组合：良性证据分 = 签名 + company + filedesc + product + version_str
    def benign_score(v):
        return sum(1.0 for k in ["sig_present", "has_company", "has_filedesc", "has_product", "has_version_str"] if v.get(k))
    bsc = np.asarray([benign_score(v) for v in fp], dtype=np.float32)
    bst = np.asarray([benign_score(v) for v in tn], dtype=np.float32)
    auc = rank_auc(bsc, bst)
    print(f"\n  良性证据分(0-5) FP_mean={bsc.mean():.3f} TN_mean={bst.mean():.3f} AUC={auc:.4f}")
    rep["benign_score"] = {"fp_mean": round(float(bsc.mean()), 4), "tn_mean": round(float(bst.mean()), 4), "auc": round(float(auc), 4)}
    # 分段
    for th in range(0, 6):
        fp_ge = (bsc >= th).mean()
        tn_ge = (bst >= th).mean()
        print(f"    score>={th}: FP {fp_ge:.3f} TN {tn_ge:.3f}")

    rep["n_fp_parsed"] = len(fp)
    rep["n_tn_parsed"] = len(tn)
    rep["elapsed_sec"] = time.time() - t0
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
