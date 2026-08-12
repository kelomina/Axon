#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键复检：VirusTotal API 批量查询未判定跨树冲突样本，输出判定建议。

用法：
  python scripts/vt_recheck.py                # 复检 72 个未判定冲突（默认）
  python scripts/vt_recheck.py --all          # 复检全部 100 个冲突 sha
  python scripts/vt_recheck.py --sha <sha>    # 只查单个
  python scripts/vt_recheck.py --delay 15 --mal-threshold 3

API key：优先环境变量 VT_API_KEY，否则读 config/vt_api_key.txt（跳过空行和 # 行）。
免费 key 限速 4 req/min → 每次请求后 sleep --delay（默认 15s）；HTTP 429 时读
Retry-After 退避后重试（最多 3 次）。

断点续跑：结果追加写 vt_recheck/vt_results.jsonl，重跑自动跳过已查 sha。

判定建议（启发式，供人工参考，非定论）：
  malicious >= mal-threshold(3)  -> Mal
  malicious == 0                -> Begin
  其余                          -> sus（存疑，人工看引擎明细）

输出：
  reports/full_739k_benign/label_governance/vt_recheck/vt_results.csv/jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFLICTS = PROJECT_ROOT / "reports" / "full_739k_benign" / "label_governance" / "corpus_conflicts.csv"
KEY_FILE = PROJECT_ROOT / "config" / "vt_api_key.txt"
OUT_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "label_governance" / "vt_recheck"
API = "https://www.virustotal.com/api/v3/files/{}"


def load_api_key() -> str:
    env = os.environ.get("VT_API_KEY", "").strip()
    if env:
        return env
    if KEY_FILE.exists():
        for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
    raise SystemExit(f"[err] 未找到 API key：请设置环境变量 VT_API_KEY 或写入 {KEY_FILE}")


def load_targets(all_flag: bool, single: str | None) -> list[dict]:
    if single:
        return [{"index": "manual", "sha256": single.strip().casefold(), "label": ""}]
    rows = list(csv.DictReader(open(CONFLICTS, encoding="utf-8-sig")))
    if all_flag:
        return [{"index": r["index"], "sha256": r["sha256"].casefold(), "label": r["label"]} for r in rows]
    return [{"index": r["index"], "sha256": r["sha256"].casefold(), "label": r["label"]}
            for r in rows if r["action"] == "conflict_undecided"]


def fetch_vt(sha: str, key: str, delay: float) -> dict:
    req = urllib.request.Request(API.format(sha), headers={"x-apikey": key})
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return {"ok": True, "data": data}
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                wait = float(ra) if ra else max(delay, 60.0)
                print(f"    [429] 限速，退避 {wait:.0f}s")
                time.sleep(wait)
                last_err = f"429 x{attempt + 1}"
                continue
            if e.code == 404:
                return {"ok": False, "status": "not_found", "detail": "VT 无此 hash 记录"}
            last_err = f"HTTP {e.code}"
            time.sleep(delay)
            continue
        except Exception as e:
            last_err = str(e)
            time.sleep(delay)
            continue
    return {"ok": False, "status": "error", "detail": last_err}


def verdict_from(stats: dict, mal_threshold: int) -> str:
    mal = int(stats.get("malicious", 0))
    if mal >= mal_threshold:
        return "Mal"
    if mal == 0:
        return "Begin"
    return "sus"


def top_engines(results: dict, category: str, n: int = 6) -> list[str]:
    names = []
    for eng, info in results.items():
        if info.get("category") == category and len(names) < n:
            names.append(f"{eng}:{info.get('result') or ''}")
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="复检全部 100 个冲突 sha（含已判定）")
    ap.add_argument("--sha", type=str, help="只复检单个 sha")
    ap.add_argument("--delay", type=float, default=15.0, help="请求间隔秒数（免费 key 建议 >=15）")
    ap.add_argument("--mal-threshold", type=int, default=3, help="判定恶意所需的 malicious 引擎数")
    args = ap.parse_args()

    key = load_api_key()
    targets = load_targets(args.all, args.sha)
    print(f"[key] loaded (len={len(key)})  [targets] {len(targets)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jl = OUT_DIR / "vt_results.jsonl"

    # 断点：已查 sha
    done = {}
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                done[rec["sha256"]] = rec
            except json.JSONDecodeError:
                continue
    print(f"[resume] already done: {len(done)}")

    t0 = time.time()
    n_fetch = n_skip = n_err = 0
    results = []
    for i, t in enumerate(targets, 1):
        sha = t["sha256"]
        if sha in done:
            rec = done[sha]
            n_skip += 1
            results.append(rec)
            print(f"[{i}/{len(targets)}] (skip) {sha[:16]}… -> {rec.get('verdict', rec.get('status'))}")
            continue
        r = fetch_vt(sha, key, args.delay)
        if r["ok"]:
            attr = r["data"].get("data", {}).get("attributes", {})
            stats = attr.get("last_analysis_stats", {})
            verdict = verdict_from(stats, args.mal_threshold)
            rec = {
                "index": t["index"], "sha256": sha, "orig_label": t["label"],
                "verdict": verdict,
                "malicious": stats.get("malicious", 0),
                "suspicious": stats.get("suspicious", 0),
                "harmless": stats.get("harmless", 0),
                "undetected": stats.get("undetected", 0),
                "top_mal_engines": top_engines(attr.get("last_analysis_results", {}), "malicious"),
            }
            n_fetch += 1
            print(f"[{i}/{len(targets)}] {sha[:16]}… mal={rec['malicious']} "
                  f"sus={rec['suspicious']} harm={rec['harmless']} -> {verdict}")
        else:
            rec = {"index": t["index"], "sha256": sha, "orig_label": t["label"],
                   "verdict": r.get("status", "error"), "status_detail": r.get("detail")}
            n_err += 1
            print(f"[{i}/{len(targets)}] {sha[:16]}… -> {rec['verdict']} ({rec.get('status_detail')})")
        with open(jl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        results.append(rec)
        time.sleep(args.delay)

    # 汇总 + CSV
    fieldnames = ["index", "sha256", "orig_label", "verdict", "malicious", "suspicious",
                  "harmless", "undetected", "top_mal_engines", "status_detail"]
    with open(OUT_DIR / "vt_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    from collections import Counter
    vc = Counter(r.get("verdict") for r in results)
    print(f"\n=== 汇总 ===")
    print(f"fetched={n_fetch} skipped={n_skip} errors={n_err}")
    print(f"verdicts: {dict(vc)}")
    print(f"[saved] {OUT_DIR/'vt_results.csv'} ({len(results)})  {(time.time()-t0)/60:.1f} min")
    print("注意：verdict 是启发式建议（Mal=malicious>=阈值, Begin=malicious==0, sus=存疑），"
          "最终判定请结合引擎明细/人工确认。")


if __name__ == "__main__":
    main()
