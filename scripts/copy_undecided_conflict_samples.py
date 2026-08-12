#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 corpus_conflicts.csv 中 72 个未判定（conflict_undecided）跨树冲突样本的文件复制到
D:\\待复核\\samples_batch2\\，并生成 VT 查询链接清单。

命名：<index>.benign<ext>  良性树位置的文件（本侧或反查）
      <index>.malware<ext> 恶意树位置的文件
同 sha 必同内容，两版是同一文件的良/恶两树位置；meta 只收一侧，另一侧从 dir_index 反查。

产物：
  D:\\待复核\\samples_batch2\\         72×2 个文件（对侧缺失则只拷本侧）
  D:\\待复核\\batch2_manifest.csv      index/sha/label/本侧树/两版路径
  D:\\待复核\\batch2_vt_links.txt      每行一个 VirusTotal 链接
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
META = BASE / "content_pe_v1" / "meta.csv"
DIR_INDEX = BASE / "content_versionstr" / "dir_index.pkl"
CONFLICTS = BASE / "label_governance" / "corpus_conflicts.csv"
DEST = Path(r"D:\待复核\samples_batch2")
MANIFEST = Path(r"D:\待复核\batch2_manifest.csv")
VT_LINKS = Path(r"D:\待复核\batch2_vt_links.txt")


def tree_of_dir(d: str) -> str:
    low = d.casefold()
    if "恶意" in low or "malicious" in low or "待拉黑" in low:
        return "malware"
    if "良性" in low or "benign" in low or "待加入白名单" in low:
        return "benign"
    return "unknown"


def resolve(p: str) -> str:
    if p.startswith("G:") or p.startswith("H:"):
        p = "F:" + p[2:]
    return p


def main() -> None:
    t0 = time.time()
    # 未判定冲突行
    undecided = [r for r in csv.DictReader(open(CONFLICTS, encoding="utf-8-sig"))
                 if r["action"] == "conflict_undecided"]
    print(f"[undecided] {len(undecided)} shas")

    # meta index -> raw_path
    meta_path = {}
    for r in csv.DictReader(open(META, encoding="utf-8")):
        meta_path[int(r["index"])] = r["raw_path"]

    # dir_index -> 良/恶两侧 sha 路径反查
    with open(DIR_INDEX, "rb") as f:
        dindex = pickle.load(f)
    benign_map, malware_map = {}, {}
    for d, s in dindex.items():
        tt = tree_of_dir(d)
        for fn in s:
            stem = os.path.splitext(fn)[0].casefold()
            if tt == "benign":
                benign_map.setdefault(stem, os.path.join(d, fn))
            elif tt == "malware":
                malware_map.setdefault(stem, os.path.join(d, fn))

    DEST.mkdir(parents=True, exist_ok=True)
    man_rows, vt_lines, missing_opposite = [], [], 0
    n_copy = 0
    for r in sorted(undecided, key=lambda x: int(x["index"])):
        idx, sha = int(r["index"]), r["sha256"].casefold()
        own_tree = r["tree"]
        src_own = resolve(meta_path.get(idx, ""))
        # 对侧路径
        opp_map = malware_map if own_tree == "benign" else benign_map
        src_opp = opp_map.get(sha, "")

        own_ext = Path(src_own).suffix or ".bin"
        opp_ext = Path(src_opp).suffix or ".bin" if src_opp else ".bin"
        dst_own = DEST / f"{idx}.{own_tree}{own_ext}"
        dst_opp = DEST / f"{idx}.{'malware' if own_tree=='benign' else 'benign'}{opp_ext}"

        if os.path.isfile(src_own):
            shutil.copy2(src_own, dst_own)
            n_copy += 1
            own_note = f"{os.path.getsize(src_own):,}B"
        else:
            own_note = "MISSING"
        if src_opp and os.path.isfile(src_opp):
            shutil.copy2(src_opp, dst_opp)
            n_copy += 1
            opp_note = f"{os.path.getsize(src_opp):,}B"
        else:
            missing_opposite += 1
            opp_note = "MISSING"

        man_rows.append({
            "index": idx, "sha256": sha, "label": r["label"],
            "own_tree": own_tree, "benign_path": str(dst_own if own_tree == "benign" else dst_opp),
            "malware_path": str(dst_opp if own_tree == "benign" else dst_own),
            "own_note": own_note, "opposite_note": opp_note,
        })
        vt_lines.append(f"https://www.virustotal.com/gui/file/{sha}")
        print(f"[{idx}] {own_tree} own={own_note} opp={opp_note}  {sha[:16]}…")

    with open(MANIFEST, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["index", "sha256", "label", "own_tree",
                                          "benign_path", "malware_path",
                                          "own_note", "opposite_note"])
        w.writeheader()
        w.writerows(man_rows)
    VT_LINKS.write_text("\n".join(vt_lines) + "\n", encoding="utf-8")

    print(f"\n[done] copied={n_copy} files  missing_opposite={missing_opposite}  "
          f"dest={DEST}  manifest={MANIFEST}  vt_links={VT_LINKS}  {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
