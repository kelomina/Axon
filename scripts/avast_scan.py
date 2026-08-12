#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地 Avast ashCmd 逐文件扫描未判定跨树冲突样本，输出感染判定。

设计：逐文件调用 ashCmd.exe，每文件独立退出码判定（不依赖解析报告格式）。
退出码映射（avast ashCmd 惯例）：
  0        -> clean
  1,3,4    -> infected
  2        -> error
  负值      -> crash（进程崩溃，如非交互会话/缺组件）

用法（在桌面终端运行，ashCmd 需要桌面会话）：
  python scripts/avast_scan.py                 # 全量扫描 72 个
  python scripts/avast_scan.py --file <path>   # pilot：先扫单个文件验证输出/退出码
  python scripts/avast_scan.py --ashcmd <exe>  # 指定 ashCmd 路径

输出：
  reports/full_739k_benign/label_governance/avast_scan/avast_results.csv
  .../avast_scan/raw/                       每文件原始 stdout/stderr + 报告
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(r"D:\待复核\batch2_manifest.csv")
OUT_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "label_governance" / "avast_scan"
DEFAULT_ASHCMD = Path(r"C:\Program Files\Avast Software\Avast\ashCmd.exe")

# avast 惯例返回码 -> 判定
EXIT_VERDICT = {0: "clean", 1: "infected", 3: "infected", 4: "infected", 2: "error"}


def probe_ashcmd() -> Path:
    candidates = [
        DEFAULT_ASHCMD,
        Path(r"C:\Program Files (x86)\Avast Software\Avast\ashCmd.exe"),
        Path(r"C:\Program Files\AVAST Software\Avast\ashCmd.exe"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit("[err] 未找到 ashCmd.exe，请用 --ashcmd 指定路径")


def scan_one(ash: Path, file_path: Path, delay: float) -> dict:
    rpt = OUT_DIR / "raw" / f"{file_path.stem}.rpt"
    cmd = [str(ash), str(file_path), "/P", "/S", "/C", f"/R={rpt}"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=180,
                              text=True, encoding="utf-8", errors="replace")
        code = proc.returncode
    except subprocess.TimeoutExpired:
        return {"exit_code": "TIMEOUT", "verdict": "error", "detail": "扫描超时 180s",
                "elapsed_s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"exit_code": "EXC", "verdict": "error", "detail": str(e),
                "elapsed_s": round(time.time() - t0, 1)}

    verdict = EXIT_VERDICT.get(code, "crash" if code < 0 else "unknown")
    # 保留原始输出
    (OUT_DIR / "raw").mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / "raw" / f"{file_path.stem}.log"
    log.write_text(f"cmd: {' '.join(cmd)}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
                   encoding="utf-8", errors="replace")
    # 病毒名：stdout/报告里常见 "Found virus" / 病毒名行
    detail = ""
    combined = proc.stdout + "\n" + proc.stderr
    for kw in ("virus", "infected", "malware", "Found", "PUP"):
        idx = combined.lower().find(kw.lower())
        if idx >= 0:
            detail = combined[max(0, idx - 40):idx + 80].replace("\n", " ").strip()
            break
    return {"exit_code": code, "verdict": verdict, "detail": detail,
            "elapsed_s": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ashcmd", type=Path, default=None)
    ap.add_argument("--file", type=Path, default=None, help="pilot：只扫单个文件")
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args()

    ash = args.ashcmd or probe_ashcmd()
    print(f"[ashCmd] {ash}")

    if args.file:
        targets = [{"index": "pilot", "sha256": args.file.stem, "file": args.file}]
    else:
        if not MANIFEST.exists():
            raise SystemExit(f"[err] 找不到 {MANIFEST}，请先跑 copy_undecided_conflict_samples.py")
        rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
        targets = [{"index": r["index"], "sha256": r["sha256"],
                    "file": Path(r["malware_path"])} for r in rows]
    print(f"[targets] {len(targets)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.time()
    for i, t in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] 扫 {t['file'].name} ...", end=" ", flush=True)
        res = scan_one(ash, t["file"], args.delay)
        res.update({"index": t["index"], "sha256": t["sha256"], "file": str(t["file"])})
        results.append(res)
        print(f"{res['verdict']} (exit={res['exit_code']}, {res['elapsed_s']}s)")
        if res["detail"]:
            print(f"        {res['detail']}")
        time.sleep(args.delay)

    with open(OUT_DIR / "avast_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["index", "sha256", "file", "verdict",
                                          "exit_code", "detail", "elapsed_s"])
        w.writeheader()
        w.writerows(results)
    from collections import Counter
    print(f"\n=== 汇总 ===")
    print(f"verdicts: {dict(Counter(r['verdict'] for r in results))}")
    print(f"[saved] {OUT_DIR/'avast_results.csv'} ({len(results)})  {(time.time()-t0)/60:.1f} min")
    print("提示：exit_code 负值(crash)或 TIMEOUT 表示该文件扫描失败；verdict 是退出码映射，"
          "若 avast 在此环境也崩溃请检查是否为桌面会话/是否已激活。")


if __name__ == "__main__":
    main()
