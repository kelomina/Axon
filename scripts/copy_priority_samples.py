#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 priority_review.csv 的 28 个高价值样本的实际文件复制到 D:\\待复核\\samples\\。

每个样本：
  <index>_<candidate>_<split>.benign<ext>    —— 良性树版本（raw_path resolve 后）
  <index>_<candidate>_<split>.malware<ext>   —— 恶意树版本（仅 cross_tree_conflict，sha 同名反查）

定位用 content_versionstr/dir_index.pkl（不逐文件 stat）。F: 远程盘慢但只有 ~50 个文件。
"""
from __future__ import annotations

import csv
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "reports" / "full_739k_benign"
PRIORITY_CSV = BASE / "label_governance" / "priority_review.csv"
DIR_INDEX = BASE / "content_versionstr" / "dir_index.pkl"
DEST = Path(r"D:\待复核\samples")


def tree_of_dir(d: str) -> str:
    low = d.casefold()
    if "恶意" in low or "malicious" in low or "待拉黑" in low:
        return "malware"
    if "良性" in low or "benign" in low or "待加入白名单" in low:
        return "benign"
    return "unknown"


def main() -> None:
    t0 = time.time()
    with open(DIR_INDEX, "rb") as f:
        dindex = pickle.load(f)

    # 恶意树 sha-stem -> 完整路径 反查表（跨树冲突要拿恶意侧文件）
    malware_lookup: dict[str, str] = {}
    for d, s in dindex.items():
        if tree_of_dir(d) != "malware":
            continue
        for fn in s:
            stem = os.path.splitext(fn)[0].casefold()
            if stem not in malware_lookup:
                malware_lookup[stem] = os.path.join(d, fn)
    print(f"[malware lookup] {len(malware_lookup):,} stems")

    rows = list(csv.DictReader(open(PRIORITY_CSV, encoding="utf-8-sig")))
    DEST.mkdir(parents=True, exist_ok=True)

    n_ok = n_benign_missing = n_mal_missing = 0
    for r in rows:
        idx, cand, split, sha = r["index"], r["candidate"], r["split"], r["sha256"]
        stem = sha.casefold()
        conflict = r["cross_tree_conflict"] == "1"
        tag = f"{idx}_{cand}_{split}"

        # ---- 良性侧 ----
        src_benign = r["raw_path"]
        if src_benign.startswith("G:") or src_benign.startswith("H:"):
            src_benign = "F:" + src_benign[2:]
        ext = Path(src_benign).suffix or ".bin"
        dst_b = DEST / f"{tag}.benign{ext}"
        if os.path.isfile(src_benign):
            shutil.copy2(src_benign, dst_b)
            n_ok += 1
            b_note = f"{os.path.getsize(src_benign):,}B"
        else:
            n_benign_missing += 1
            b_note = "MISSING"
        line = f"[{tag}] benign={b_note}"

        # ---- 恶意侧（仅冲突）----
        if conflict:
            src_mal = malware_lookup.get(stem)
            if src_mal and os.path.isfile(src_mal):
                dst_m = DEST / f"{tag}.malware{os.path.splitext(src_mal)[1] or '.bin'}"
                shutil.copy2(src_mal, dst_m)
                n_ok += 1
                m_note = f"{os.path.getsize(src_mal):,}B  <- {os.path.dirname(src_mal)}"
            else:
                n_mal_missing += 1
                m_note = f"MISSING (malware tree, sha={stem[:16]}…)"
            line += f" | malware={m_note}"
        print(line)

    print(f"\n[done] copied={n_ok}  benign_missing={n_benign_missing}  mal_missing={n_mal_missing}  "
          f"dest={DEST}  {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
