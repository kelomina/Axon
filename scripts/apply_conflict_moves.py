#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨树冲突 sha 物理归位：把错误树的副本移动到正确树（os.replace，同内容覆盖安全）。

依据 reports/full_739k_benign/label_governance/move_plan_preview.csv：
  - KEEP_BENIGN_DROP_MALWARE  (final=0): 恶意树副本 -> 移入良性树 keep 目录
  - KEEP_MALWARE_DROP_BENIGN  (final=1): 良性树副本 -> 移入恶意树 keep 目录

keep 树已有同 sha（keep_ok=100/100 已验证），故 drop 文件移入后：
  - tgt 已存在且同内容 -> os.remove(src) 删冗余（等效"移动合并"，保留 keep 侧）
  - tgt 不存在         -> shutil.move（跨盘自动 copy+unlink，同盘即 rename）
注意：os.replace/os.rename 不支持跨盘（WinError 17），跨盘必须 shutil.move。
结果：错误树不再含该 sha，正确树保留一份，数据零丢失。

审计：写 move_executed.csv（src,tgt,ok,error），可回滚/复核。
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
from collections import Counter

BASE = r"E:\Project\python\Axon_v2.6Exp\reports\full_739k_benign\label_governance"
PREVIEW = os.path.join(BASE, "move_plan_preview.csv")
LOG = os.path.join(BASE, "move_executed.csv")


def locs(s: str) -> list[str]:
    return [x.strip() for x in s.split(";") if x.strip() and x != "-"]


def main() -> None:
    rows = list(csv.DictReader(open(PREVIEW, encoding="utf-8-sig")))
    print(f"[preview] {len(rows)} conflict shas")

    ops = []
    for r in rows:
        if r["action_move"] == "KEEP_BENIGN_DROP_MALWARE":
            keep = locs(r["benign_locs"])
            drop = locs(r["malware_locs"])
        else:
            keep = locs(r["malware_locs"])
            drop = locs(r["benign_locs"])
        keep_dir = os.path.dirname(keep[0])
        for p in drop:
            if not os.path.exists(p):
                print(f"[SKIP] missing src: {p}")
                continue
            ops.append({
                "sha": r["sha256"], "final": r["new_label"], "action_move": r["action_move"],
                "src": p, "tgt": os.path.join(keep_dir, os.path.basename(p)),
            })
    print(f"[ops] {len(ops)} files to relocate (F-src: {sum(1 for o in ops if o['src'].startswith('F:'))}, "
          f"E-src: {sum(1 for o in ops if o['src'].startswith('E:'))})")

    results = []
    n_ok = n_cover = n_err = 0
    for i, o in enumerate(ops, 1):
        src, tgt = o["src"], o["tgt"]
        if not os.path.exists(src):
            # 已在上一轮成功归位 -> 跳过
            results.append({**o, "ok": True, "covered": os.path.exists(tgt), "error": "already-moved"})
            n_ok += 1
            continue
        tgt_existed = os.path.exists(tgt)
        try:
            if tgt_existed:
                # keep 树已有同 sha（必同内容）：删冗余副本即可，等效"移动合并"
                os.remove(src)
            else:
                # 跨盘用 shutil.move（自动 copy+unlink）；同盘即 rename
                shutil.move(src, tgt)
            ok, err = True, ""
            if tgt_existed:
                n_cover += 1
            else:
                n_ok += 1
        except Exception as e:  # noqa: BLE001
            ok, err = False, str(e)
            n_err += 1
            print(f"[ERR {i}/{len(ops)}] {os.path.basename(src)} -> {tgt}: {err}")
        results.append({**o, "ok": ok, "covered": tgt_existed, "error": err})
        if i % 25 == 0:
            print(f"[{i}/{len(ops)}] ok={n_ok} cover={n_cover} err={n_err}")

    with open(LOG, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"[saved] {LOG} ({len(results)})")

    # ---- 验证：drop 树不再含该 sha ----
    v_ok = v_missing = 0
    for o in ops:
        if os.path.exists(o["src"]):
            v_missing += 1
            print(f"[VERIFY-FAIL] still exists: {o['src']}")
        else:
            v_ok += 1
    print(f"[verify] drop-side cleared {v_ok}/{len(ops)}  residual={v_missing}")

    # ---- 验证：每个 sha 在正确树至少一份 ----
    from collections import defaultdict
    kept = defaultdict(int)
    for o in ops:
        kept[o["sha"]] += 1
    keep_missing = 0
    for r in rows:
        if r["action_move"] == "KEEP_BENIGN_DROP_MALWARE":
            keep = locs(r["benign_locs"])
        else:
            keep = locs(r["malware_locs"])
        if not any(os.path.exists(p) for p in keep):
            keep_missing += 1
            print(f"[VERIFY-FAIL] sha lost keep-side: {r['sha256'][:12]}")
    print(f"[verify] keep-side intact {len(rows)-keep_missing}/{len(rows)}  lost={keep_missing}")

    print(f"\n=== move summary ===")
    print(f"moved={n_ok} covered_merge={n_cover} errors={n_err}")
    print(f"act: {dict(Counter(o['action_move'] for o in results))}")


if __name__ == "__main__":
    main()
