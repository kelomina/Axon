#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探查：100 个冲突 sha 的良/恶两侧物理路径与存在性，输出 move_plan 草稿。

只读，不移动任何文件。输出：
  reports/full_739k_benign/label_governance/move_plan_preview.csv
每行一个冲突 sha + 良性侧路径 + 恶意侧路径 + 最终 label + 推荐动作。
"""
from __future__ import annotations

import csv
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(r"E:\Project\python\Axon_v2.6Exp\reports\full_739k_benign")
META = BASE / "content_pe_v1" / "meta.csv"
DINDEX = BASE / "content_versionstr" / "dir_index.pkl"
CONFLICTS = BASE / "label_governance" / "corpus_conflicts.csv"
OUT = BASE / "label_governance" / "move_plan_preview.csv"


def tree_of_dir(d: str) -> str:
    low = d.casefold()
    if "恶意" in low or "malicious" in low:
        return "malware"
    if "良性" in low or "benign" in low or "白名单" in low:
        return "benign"
    return "unknown"


def main() -> None:
    with open(DINDEX, "rb") as f:
        dindex = pickle.load(f)

    # sha -> list of (dir, fn, tree)
    sha_locs = defaultdict(list)
    for d, s in dindex.items():
        tt = tree_of_dir(d)
        for fn in s:
            stem = os.path.splitext(fn)[0].casefold()
            sha_locs[stem].append((d, fn, tt))

    # meta: sha -> raw_path (which side the corpus recorded)
    meta_side = {}
    for r in csv.DictReader(open(META, encoding="utf-8")):
        sha = r["source_sha256"].strip().casefold()
        meta_side[sha] = {"index": r["index"], "label": r["label"],
                          "raw_path": r["raw_path"]}

    conf = list(csv.DictReader(open(CONFLICTS, encoding="utf-8-sig")))

    rows = []
    benign_side = Counter()   # raw_path side vs final label consistency
    for r in conf:
        sha = r["sha256"].casefold()
        final_label = int(r["new_label"])
        meta = meta_side.get(sha, {})
        locs = sha_locs.get(sha, [])
        benign = [os.path.join(d, fn) for d, fn, t in locs if t == "benign"]
        malware = [os.path.join(d, fn) for d, fn, t in locs if t == "malware"]
        # raw_path recorded side
        rp = meta.get("raw_path", "")
        rp_side = tree_of_dir(os.path.dirname(rp)) if rp else ""
        # where does corpus think the file is? raw_path is authority for meta row.
        actual = os.path.exists(rp) if rp else False
        # if final=benign: file should live in benign tree only -> malware copies to remove
        # if final=malware: file should live in malware tree only -> benign copies to remove
        if final_label == 0:
            action = "KEEP_BENIGN_DROP_MALWARE"
            move_srcs = malware
        else:
            action = "KEEP_MALWARE_DROP_BENIGN"
            move_srcs = benign
        rows.append({
            "sha256": sha, "index": meta.get("index", ""), "orig_label": r["label"],
            "new_label": r["new_label"], "action_conf": r["action"], "action_move": action,
            "raw_path": rp, "raw_path_side": rp_side, "raw_path_exists": actual,
            "benign_locs": "; ".join(benign) or "-",
            "malware_locs": "; ".join(malware) or "-",
            "n_benign": len(benign), "n_malware": len(malware),
            "move_srcs": "; ".join(move_srcs) or "-",
        })

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary
    mv = Counter(r["action_move"] for r in rows)
    raw_side = Counter(r["raw_path_side"] for r in rows)
    raw_side_exist = Counter((r["raw_path_side"], bool(r["raw_path_exists"])) for r in rows)
    n_missing_both = sum(1 for r in rows if not r["benign_locs"].strip() or not r["malware_locs"].strip())
    print(f"[saved] {OUT} ({len(rows)})")
    print("[move action]", dict(mv))
    print("[raw_path_side]", dict(raw_side))
    print("[raw_path exists]", {str(k): v for k, v in raw_side_exist.items()})
    print("[rows missing one side]", n_missing_both)


if __name__ == "__main__":
    main()
