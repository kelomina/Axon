#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""移动后重建缓存：dir_index.pkl（受影响目录）+ meta.csv（冲突 sha 标签/路径）+ name_index.pkl（可选全量）。

用法：
  python scripts/rebuild_corpus_cache.py                # 增量 dir_index + meta.csv 更新
  python scripts/rebuild_corpus_cache.py --name-index   # 额外全量重建 name_index（os.walk 全树，较久）

说明：
  - dir_index.pkl（content_versionstr）：加载现有，仅重 listdir 受影响目录（drop src / keep tgt 的父目录）。
  - meta.csv：对 100 个跨树冲突 sha，label->new_label，raw_path->正确树 keep 路径（G:/H: 纯前缀换 F:）。
  - name_index.pkl（reports/full_739k）：用 E:/F: 现存 roots 全量 os.walk 重建，清掉失效 G:/H: 指针。
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "reports" / "full_739k_benign"
META = BASE / "content_pe_v1" / "meta.csv"
DIR_INDEX = BASE / "content_versionstr" / "dir_index.pkl"
NAME_INDEX = PROJECT_ROOT / "reports" / "full_739k" / "name_index.pkl"
PREVIEW = BASE / "label_governance" / "move_plan_preview.csv"
MOVE_LOG = BASE / "label_governance" / "move_executed.csv"

BENIGN_ROOTS = [
    r"E:\Project\python\KoloVirusDetector_ML_V2-main\benign_samples\待加入白名单",
    r"F:\私人\良性文件\待加入白名单",
    r"F:\私人\良性文件",  # 覆盖待加入白名单 + 待加入白名单_upx（同 extract 的 H:\私人\良性文件 语义）
]
MALWARE_ROOTS = [
    r"E:\Project\python\KoloVirusDetector_ML_V2-main\malicious_samples\待拉黑",
    r"F:\私人\恶意\MB\unziped",
]


def resolve(p: str) -> str:
    if p.startswith("G:") or p.startswith("H:"):
        return "F:" + p[2:]
    return p


def load_conflict_map() -> dict:
    m = {}
    for r in csv.DictReader(open(PREVIEW, encoding="utf-8-sig")):
        sha = r["sha256"].strip().casefold()
        keep = r["benign_locs"] if r["action_move"] == "KEEP_BENIGN_DROP_MALWARE" else r["malware_locs"]
        paths = [x.strip() for x in keep.split(";") if x.strip() and x != "-"]
        resolved = [resolve(p) for p in paths]
        keep_path = next((p for p in resolved if os.path.exists(p)), resolved[0] if resolved else "")
        # drop 侧（move_srcs）目录：dir_index 需刷新移除残留
        drop_dirs = {os.path.dirname(resolve(x))
                     for x in r["move_srcs"].split(";")
                     if x.strip() and x != "-"}
        m[sha] = {"new_label": r["new_label"], "keep_path": keep_path,
                  "drop_dirs": drop_dirs}
    return m


def rebuild_dir_index(conflict_map: dict) -> None:
    affected = set()
    for info in conflict_map.values():
        if info["keep_path"]:
            affected.add(os.path.dirname(info["keep_path"]))
        affected |= info["drop_dirs"]
    if MOVE_LOG.exists():
        for row in csv.DictReader(open(MOVE_LOG, encoding="utf-8-sig")):
            affected.add(os.path.dirname(resolve(row["src"])))
            affected.add(os.path.dirname(resolve(row["tgt"])))
    with open(DIR_INDEX, "rb") as f:
        index = pickle.load(f)
    n_upd = 0
    for d in sorted(affected):
        if d in index:
            try:
                index[d] = set(os.listdir(d))
            except OSError:
                index[d] = set()
            n_upd += 1
    with open(DIR_INDEX, "wb") as f:
        pickle.dump(index, f, protocol=4)
    print(f"[dir_index] refreshed {n_upd} dirs (total {len(index)} dirs) -> {DIR_INDEX}")


def rebuild_name_index() -> None:
    t0 = time.time()
    idx: dict = {}
    roots = BENIGN_ROOTS + MALWARE_ROOTS
    for root in roots:
        if not os.path.isdir(root):
            print(f"[name_index] skip missing root: {root}")
            continue
        n = 0
        for dirpath, _d, files in os.walk(root):
            for fn in files:
                idx.setdefault(fn.casefold(), os.path.join(dirpath, fn))
                n += 1
        print(f"[name_index] {root}: {n:,} files ({time.time()-t0:.0f}s)")
    NAME_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(NAME_INDEX, "wb") as f:
        pickle.dump(idx, f, protocol=4)
    print(f"[name_index] saved {len(idx):,} names -> {NAME_INDEX} ({time.time()-t0:.0f}s)")


def update_meta(conflict_map: dict) -> None:
    rows = list(csv.DictReader(open(META, encoding="utf-8")))
    n_upd = n_keep = 0
    for r in rows:
        sha = r["source_sha256"].strip().casefold()
        if sha in conflict_map:
            info = conflict_map[sha]
            if r["label"] != info["new_label"]:
                r["label"] = info["new_label"]
            if info["keep_path"]:
                r["raw_path"] = info["keep_path"]
                r["located"] = "1"
                n_keep += 1
            n_upd += 1
    with open(META, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[meta] updated {n_upd} conflict rows (raw_path set on {n_keep}) -> {META}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name-index", action="store_true", help="同时全量重建 name_index.pkl")
    args = ap.parse_args()

    conflict_map = load_conflict_map()
    print(f"[conflict] {len(conflict_map)} shas with final label")

    rebuild_dir_index(conflict_map)
    update_meta(conflict_map)
    if args.name_index:
        rebuild_name_index()
    print("[done]")


if __name__ == "__main__":
    main()
