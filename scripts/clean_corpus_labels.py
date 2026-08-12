#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语料清理：应用人工复核判定，生成干净标签 + 跨树冲突剔除清单。

数据源：
  - content_pe_v1/meta.csv                    全量语料（index/cache_path/source_sha256/label/raw_path）
  - content_versionstr/dir_index.pkl          良/恶树文件索引（判定跨树冲突 sha）
  - label_governance/review_verdicts.csv      28 个人工复核判定（sha -> 最终 label）

处理：
  1) 全量扫描 meta，标记所有跨树冲突 sha（sha 同时在良性树与恶意树的文件系统里）
  2) 对每个冲突 sha 决策：
       - 有人工判定  -> label 统一为判定值；同 sha 的 meta 多行中，与判定相反的行剔除
       - 无判定      -> 默认仅标记不删（--drop-undecided 时两行都剔除，保守防污染）
  3) 输出：
       - label_override.csv        重训消费契约：index -> new_label（改标 + 冲突判定合并）
       - corpus_conflicts.csv      冲突 sha 全量清单 + 决策 + meta 行定位
       - cleaned_meta.csv          全量 meta + cleaned_label + action
       - corpus_clean_report.json  统计

接入方式：
  - Stage-2 重训：读 label_override.csv 覆盖 meta.csv 的 label（按 index）。
  - 基座重训：需再改 manifest/cache npz 的 label（本脚本只生成契约，不物理改写）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = PROJECT_ROOT / "reports" / "full_739k_benign"
META = BASE / "content_pe_v1" / "meta.csv"
DIR_INDEX = BASE / "content_versionstr" / "dir_index.pkl"
VERDICTS = BASE / "label_governance" / "review_verdicts.csv"
OUT_DIR = BASE / "label_governance"


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
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drop-undecided", action="store_true",
                    help="无人工判定的冲突 sha：其所有 meta 行标记 dropped（保守剔除，防污染）。"
                         "默认仅标记 conflict 不删。")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    t0 = time.time()
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. 良/恶树 sha 集合（跨树冲突检测）----
    with open(DIR_INDEX, "rb") as f:
        dindex = pickle.load(f)
    benign_stems, malware_stems = set(), set()
    for d, s in dindex.items():
        tt = tree_of_dir(d)
        for fn in s:
            stem = os.path.splitext(fn)[0].casefold()
            if tt == "benign":
                benign_stems.add(stem)
            elif tt == "malware":
                malware_stems.add(stem)
    print(f"[trees] benign stems={len(benign_stems):,}  malware stems={len(malware_stems):,}")

    # ---- 2. 人工复核判定 sha -> final label ----
    verdict = {}
    for r in csv.DictReader(open(VERDICTS, encoding="utf-8-sig")):
        verdict[r["sha256"]] = int(r["review_label"])
    print(f"[verdict] {len(verdict)} reviewed shas")

    # ---- 3. 读全量 meta ----
    rows = list(csv.DictReader(open(META, encoding="utf-8")))
    print(f"[meta] {len(rows):,} rows")

    # ---- 4. 逐行判定 ----
    meta_by_sha = defaultdict(list)
    for i, r in enumerate(rows):
        sha = r["source_sha256"].strip().casefold()
        conflict = "1" if (sha in benign_stems and sha in malware_stems) else "0"
        tree = tree_of_dir(resolve(r["raw_path"]))
        rec = {
            "pos": i, "index": int(r["index"]), "sha256": sha,
            "label": int(r["label"]), "tree": tree, "conflict": conflict,
        }
        meta_by_sha[sha].append(rec)

    # ---- 5. 决策 ----
    # action: unchanged / corrected / conflict_kept / conflict_dropped
    for sha, recs in meta_by_sha.items():
        has_verdict = sha in verdict
        is_conflict = any(r["conflict"] == "1" for r in recs)
        for r in recs:
            if not is_conflict:
                if has_verdict and verdict[sha] != r["label"]:
                    r["action"] = "corrected"; r["new_label"] = verdict[sha]
                else:
                    r["action"] = "unchanged"; r["new_label"] = r["label"]
            else:
                r["conflict_sha"] = True
                if has_verdict:
                    r["new_label"] = verdict[sha]
                    r["action"] = ("conflict_kept" if verdict[sha] == r["label"]
                                   else "conflict_corrected")
                else:
                    r["new_label"] = r["label"]
                    r["action"] = "conflict_undecided"
                    if args.drop_undecided:
                        r["action"] = "conflict_dropped"

    # ---- 6. 汇总 ----
    n_conflict_shas = sum(1 for sha, recs in meta_by_sha.items()
                          if any(r.get("conflict_sha") for r in recs))
    n_undecided = sum(1 for recs in meta_by_sha.values()
                      for r in recs if r["action"] == "conflict_undecided")
    n_dropped = sum(1 for recs in meta_by_sha.values()
                    for r in recs if r["action"] == "conflict_dropped")
    n_corrected = sum(1 for recs in meta_by_sha.values()
                      for r in recs if r["action"] in ("corrected", "conflict_corrected"))
    act_counter = Counter(r["action"] for recs in meta_by_sha.values() for r in recs)

    print(f"\n[conflict shas] {n_conflict_shas}")
    print(f"[undecided rows] {n_undecided}  [dropped rows] {n_dropped}  [corrected rows] {n_corrected}")
    print(f"[actions] {dict(act_counter)}")

    # ---- 7. 写产物 ----
    info_by_pos = {rec["pos"]: rec for recs in meta_by_sha.values() for rec in recs}
    override_rows, conflict_rows, cleaned = [], [], []
    for i, r in enumerate(rows):
        info = info_by_pos[i]
        cleaned.append({**r, "cleaned_label": info["new_label"], "action": info["action"]})
        if info["action"] in ("corrected", "conflict_corrected"):
            override_rows.append({"index": r["index"], "sha256": r["source_sha256"].strip().casefold(),
                                  "orig_label": r["label"], "new_label": info["new_label"],
                                  "action": info["action"]})
    for sha, recs in meta_by_sha.items():
        if any(r.get("conflict_sha") for r in recs):
            for r in recs:
                conflict_rows.append({"sha256": sha, "index": rows[r["pos"]]["index"],
                                      "label": r["label"], "new_label": r["new_label"],
                                      "action": r["action"], "tree": r["tree"]})

    with open(OUT_DIR / "label_override.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["index", "sha256", "orig_label", "new_label", "action"])
        w.writeheader(); w.writerows(sorted(override_rows, key=lambda x: x["index"]))
    with open(OUT_DIR / "corpus_conflicts.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["sha256", "index", "label", "new_label", "action", "tree"])
        w.writeheader(); w.writerows(sorted(conflict_rows, key=lambda x: x["index"]))
    with open(OUT_DIR / "cleaned_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cleaned[0].keys()))
        w.writeheader(); w.writerows(cleaned)

    report = {
        "n_meta_rows": len(rows),
        "n_conflict_shas": n_conflict_shas,
        "n_reviewed_shas": len(verdict),
        "n_undecided_conflict_rows": n_undecided,
        "n_dropped_rows": n_dropped,
        "n_corrected_rows": n_corrected,
        "actions": dict(act_counter),
        "override_rows": len(override_rows),
        "conflict_rows": len(conflict_rows),
        "note": ("label_override.csv 供重训按 index 覆盖 label；corpus_conflicts.csv 是冲突 sha 全量清单；"
                 "cleaned_meta.csv 是全量 meta + cleaned_label/action（action=conflict_dropped 的行重训应跳过）。"
                 "基座重训还须同步 manifest/npz 的 label。"),
    }
    json.dump(report, open(OUT_DIR / "corpus_clean_report.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\n[saved] {OUT_DIR/'label_override.csv'} ({len(override_rows)})  "
          f"{OUT_DIR/'corpus_conflicts.csv'} ({len(conflict_rows)})  "
          f"{OUT_DIR/'cleaned_meta.csv'} ({len(cleaned)})")
    print(f"[done] {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
