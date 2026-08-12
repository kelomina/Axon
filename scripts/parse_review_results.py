#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析用户复核结果（D:\\待复核\\复核完成的列表.txt）并交叉 priority_review.csv。

格式：<sha256>:<Begin|Mal>    Begin=良性(label 0)  Mal=恶意(label 1)
输出：每 sha 的 index/split/candidate/原label/复核判定/是否改标 + 汇总。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

BASE = Path(r"E:\Project\python\Axon_v2.6Exp\reports\full_739k_benign")
PRIORITY = BASE / "label_governance" / "priority_review.csv"
REVIEW_TXT = Path(r"D:\待复核\复核完成的列表.txt")
OUT = BASE / "label_governance" / "review_results.json"

TAG = {"Begin": "良性", "Mal": "恶意"}


def main() -> None:
    # 复核结果（保留重复行，sha -> list of tags）
    reviews: dict[str, list[str]] = {}
    for line in REVIEW_TXT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        sha, _, tag = line.partition(":")
        sha = sha.strip().casefold()
        tag = tag.strip()
        if not sha or tag not in TAG:
            print(f"[skip-line] {line!r}")
            continue
        reviews.setdefault(sha, []).append(tag)

    # priority 样本
    rows = {r["sha256"]: r for r in csv.DictReader(open(PRIORITY, encoding="utf-8-sig"))}
    n_reviewed = 0
    out_rows = []
    for sha, md in rows.items():
        if sha in reviews:
            tags = reviews[sha]
            tag = tags[-1]  # 最后一次判定
            new_label = 0 if tag == "Begin" else 1
            changed = (new_label != int(md["label"]))
            rec = {
                "sha256": sha, "index": int(md["index"]), "split": md["split"],
                "candidate": md["candidate"], "orig_label": int(md["label"]),
                "review_label": new_label, "review_tag": tag,
                "label_changed": changed,
            }
            if len(tags) > 1:
                rec["n_duplicate_lines"] = len(tags)
            out_rows.append(rec)
            if changed:
                print(f"[改标] {md['index']:>6} {md['split']:4s} {md['candidate']:38s} "
                      f"label {md['label']} -> {new_label}  ({tag})  {sha[:16]}…")
            else:
                print(f"[确认] {md['index']:>6} {md['split']:4s} {md['candidate']:38s} "
                      f"label {md['label']} (保持)  ({tag})  {sha[:16]}…")
            n_reviewed += 1
        else:
            print(f"[未复核] {md['index']:>6} {md['split']:4s} {md['candidate']:38s}  {sha[:16]}…")

    # 复核列表里不在 priority 的 sha
    extra = [sha for sha in reviews if sha not in rows]
    if extra:
        print(f"\n[警告] 复核列表含 priority 之外的 sha {len(extra)} 个: {extra}")

    changed = [r for r in out_rows if r["label_changed"]]
    print(f"\n=== 汇总 ===")
    print(f"priority 共 {len(rows)}，已复核 {n_reviewed}，未复核 {len(rows)-n_reviewed}")
    from collections import Counter
    print(f"改标 {len(changed)}：误标黑翻正(1->0) {sum(1 for r in changed if r['orig_label']==1)}，"
          f"误标白改恶(0->1) {sum(1 for r in changed if r['orig_label']==0)}")
    print(f"复核分布: {dict(Counter(r['review_tag'] for r in out_rows))}")

    json.dump({"verdict_note": "Begin=良性(label0), Mal=恶意(label1)，取每 sha 最后一行",
               "n_reviewed": n_reviewed, "n_unreviewed": len(rows) - n_reviewed,
               "changed": len(changed), "results": out_rows},
              open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ---- 归档：完整判定表 + 改标修正表（重训管线应用契约）----
    VERDICTS_CSV = BASE / "label_governance" / "review_verdicts.csv"
    CORR_CSV = BASE / "label_governance" / "label_corrections.csv"
    with open(VERDICTS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["index", "split", "sha256", "candidate",
                                          "orig_label", "review_label", "label_changed"],
                           extrasaction="ignore")
        w.writeheader()
        for r in sorted(out_rows, key=lambda x: x["index"]):
            w.writerow(r)
    with open(CORR_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["index", "split", "sha256", "candidate",
                                          "orig_label", "new_label"])
        w.writeheader()
        for r in sorted(changed, key=lambda x: x["index"]):
            w.writerow({"index": r["index"], "split": r["split"], "sha256": r["sha256"],
                        "candidate": r["candidate"], "orig_label": r["orig_label"],
                        "new_label": r["review_label"]})
    print(f"[saved] {VERDICTS_CSV} ({len(out_rows)} 条)  {CORR_CSV} ({len(changed)} 改标)")


if __name__ == "__main__":
    main()
