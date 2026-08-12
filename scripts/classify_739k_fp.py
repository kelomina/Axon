#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""739k 测试集 FP 拆解：定位原始文件 -> Authenticode -> 归类。

输入：reports/full_739k/test739k_fp_list.csv（由 decompose_739k_fp.py 产出，
      cache_path / source_sha256 / true_label / prob_malicious / pred）
步骤：
  1. 按 basename==source_sha256 在原始树（E:/G:/H:）中定位 FP 的原始 PE 文件，
     并检测跨树冲突（同一 sha 同时出现在良性树与恶意树）。
  2. 对定位到的文件批量调 Windows PowerShell Get-AuthenticodeSignature。
  3. 归类到：可信签名(12/14 名单) / 有效签名非可信 / 跨树冲突 / 高置信 / 近阈值 / 真实错误。

CPU-only（签名收集走单次 powershell 子进程）。
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "reports" / "full_739k"

# Loop151（champion，12 家）与生产管线 Loop198（14 家）的冻结可信厂商名单
TRUSTED_TERMS_12 = (
    "Microsoft Corporation", "Microsoft Windows", "Seagate Technology", "FinalWire",
    "NetEase", "Beijing Sogou", "Beijing Kingsoft", "Beijing Qihu",
    "Wondershare", "IObit", "Yozosoft", "Huya",
)
TRUSTED_TERMS_14 = (
    "microsoft corporation", "microsoft windows", "microsoft azure", "google llc",
    "google inc", "nvidia corporation", "adobe inc", "adobe systems",
    "intel corporation", "cisco systems", "oracle america", "vmware inc",
    "realtek semiconductor", "logitech inc",
)

# 原始树候选根（仅遍历存在的目录）。树名 -> (路径)
RAW_TREE_ROOTS = [
    ("benign_e", r"E:\Project\python\KoloVirusDetector_ML_V2-main\benign_samples\待加入白名单"),
    ("malware_e", r"E:\Project\python\KoloVirusDetector_ML_V2-main\malicious_samples\待拉黑"),
    ("benign_g", r"G:\私人\良性文件\待加入白名单"),
    ("malware_g", r"G:\私人\恶意\MB\unziped"),
    ("benign_h", r"H:\私人\良性文件"),
    ("malware_h", r"H:\私人\恶意\MB\unziped"),
]

PS_SCRIPT = r"""
$InputCsv = $env:AXON_AUTH_INPUT
$OutputCsv = $env:AXON_AUTH_OUTPUT
Import-Module Microsoft.PowerShell.Security -ErrorAction Stop
$rows = Import-Csv -LiteralPath $InputCsv
$out = foreach ($row in $rows) {
  try {
    $sig = Get-AuthenticodeSignature -LiteralPath $row.source_path
    [PSCustomObject]@{
      row_ordinal = $row.row_ordinal
      source_path = $row.source_path
      source_sha256 = $row.source_sha256
      auth_status = [string]$sig.Status
      auth_status_message = [string]$sig.StatusMessage
      signer_subject = if ($sig.SignerCertificate) { [string]$sig.SignerCertificate.Subject } else { "" }
      signer_issuer = if ($sig.SignerCertificate) { [string]$sig.SignerCertificate.Issuer } else { "" }
      signer_thumbprint = if ($sig.SignerCertificate) { [string]$sig.SignerCertificate.Thumbprint } else { "" }
      timestamper_subject = if ($sig.TimeStamperCertificate) { [string]$sig.TimeStamperCertificate.Subject } else { "" }
      collection_error = ""
    }
  } catch {
    [PSCustomObject]@{
      row_ordinal = $row.row_ordinal
      source_path = $row.source_path
      source_sha256 = $row.source_sha256
      auth_status = "CollectionError"
      auth_status_message = ""
      signer_subject = ""
      signer_issuer = ""
      signer_thumbprint = ""
      timestamper_subject = ""
      collection_error = [string]$_.Exception.Message
    }
  }
}
$out | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding UTF8
"""


def _decode_name(raw: str) -> str:
    """raw_source_path 是损坏的 Unicode mojibake，仅能还原出 basename（=完整 sha256）。"""
    return Path(raw.replace("\\", "/")).name


def build_tree_index(root: Path, tree: str) -> tuple[dict, dict]:
    """返回 (name_index, stem_index): basename/stem -> (tree, path)。"""
    name_idx: dict = {}
    stem_idx: dict = {}
    n = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            p = Path(fn)
            name_idx[fn.casefold()] = (tree, full)
            stem_idx[p.stem.casefold()] = (tree, full)
            n += 1
    return name_idx, stem_idx, n


def collect_signatures(rows: list[dict], powershell_exe: str) -> dict:
    """批量 Get-AuthenticodeSignature，返回 {source_sha256: sig_row}。"""
    if not rows:
        return {}
    input_csv = Path(tempfile.mkdtemp()) / "auth_input.csv"
    output_csv = OUTPUT_DIR / "test739k_fp_authenticode.csv"
    fields = ["row_ordinal", "source_path", "source_sha256"]
    with open(input_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({
                "row_ordinal": i,
                "source_path": r["raw_path"],
                "source_sha256": r["source_sha256"],
            })
    env = os.environ.copy()
    env["AXON_AUTH_INPUT"] = str(input_csv)
    env["AXON_AUTH_OUTPUT"] = str(output_csv)
    env["PSModulePath"] = ";".join([
        str(Path.home() / "Documents" / "WindowsPowerShell" / "Modules"),
        r"C:\Program Files\WindowsPowerShell\Modules",
        r"C:\WINDOWS\system32\WindowsPowerShell\v1.0\Modules",
    ])
    subprocess.run(
        [powershell_exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", PS_SCRIPT],
        check=True, text=True, capture_output=True, env=env,
    )
    result: dict = {}
    if output_csv.exists():
        with open(output_csv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                result[row.get("source_sha256", "")] = row
    return result


def matches_trusted(subject: str, terms) -> bool:
    subj = subject.casefold()
    return any(t.casefold() in subj for t in terms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-csv", default=str(OUTPUT_DIR / "test739k_fp_list.csv"))
    parser.add_argument("--powershell-exe", default="powershell.exe")
    args = parser.parse_args()

    fp_csv = Path(args.fp_csv)
    if not fp_csv.exists():
        print(f"Missing FP list: {fp_csv}  (run decompose_739k_fp.py first)")
        sys.exit(1)

    with open(fp_csv, newline="", encoding="utf-8") as f:
        fps = list(csv.DictReader(f))
    print(f"Loaded {len(fps):,} FPs from {fp_csv}")

    # ---- 1. 建立原始树索引 ----
    benign_names, benign_stems, benign_n = {}, {}, 0
    malware_names, malware_stems, malware_n = {}, {}, 0
    for tree, root in RAW_TREE_ROOTS:
        if not os.path.isdir(root):
            print(f"[skip] tree {tree}: not found {root}")
            continue
        name_idx, stem_idx, n = build_tree_index(Path(root), tree)
        print(f"[index] {tree}: {n:,} files ({root})")
        if tree.startswith("benign"):
            benign_names.update(name_idx)
            benign_stems.update(stem_idx)
            benign_n += n
        else:
            malware_names.update(name_idx)
            malware_stems.update(stem_idx)
            malware_n += n
    print(f"[index] benign total {benign_n:,}, malware total {malware_n:,}")

    # ---- 2. 定位 + 跨树冲突 ----
    for r in fps:
        sha = r["source_sha256"].strip().casefold()
        b_loc = benign_names.get(sha) or benign_stems.get(sha)
        m_loc = malware_names.get(sha) or malware_stems.get(sha)
        r["in_benign_tree"] = "1" if b_loc else "0"
        r["in_malware_tree"] = "1" if m_loc else "0"
        r["cross_tree_conflict"] = "1" if (b_loc and m_loc) else "0"
        # 优先良性树（FP 是 benign 标签），其次恶意树
        if b_loc:
            r["raw_path"] = b_loc[1]
            r["raw_tree"] = b_loc[0]
        elif m_loc:
            r["raw_path"] = m_loc[1]
            r["raw_tree"] = m_loc[0]
        else:
            r["raw_path"] = ""
            r["raw_tree"] = "not_located"

    located = [r for r in fps if r["raw_path"]]
    not_located = [r for r in fps if not r["raw_path"]]
    n_conflict = sum(1 for r in fps if r["cross_tree_conflict"] == "1")
    print(f"[locate] located {len(located):,}, not_located {len(not_located):,}, "
          f"cross_tree_conflict {n_conflict:,}")

    # ---- 3. Authenticode ----
    print(f"[auth] collecting signatures for {len(located):,} files...")
    sigs = collect_signatures(located, args.powershell_exe) if located else {}
    print(f"[auth] collected {len(sigs):,} signature rows")

    # ---- 4. 归类 ----
    out_fields = [
        "cache_path", "source_sha256", "true_label", "prob_malicious", "pred",
        "raw_tree", "in_benign_tree", "in_malware_tree", "cross_tree_conflict",
        "auth_status", "signer_subject", "signed_valid", "signed_trusted_12",
        "signed_trusted_14", "high_conf", "near_threshold", "bucket",
    ]
    buckets = {
        "signed_trusted_14": 0, "signed_trusted_12_only": 0, "signed_valid_other": 0,
        "cross_tree_conflict": 0, "high_conf_unsure": 0, "near_threshold": 0,
        "genuine_error": 0, "not_located": 0,
    }
    for r in fps:
        p = float(r["prob_malicious"])
        r["high_conf"] = "1" if p >= 0.99 else "0"
        r["near_threshold"] = "1" if 0.5 <= p < 0.6 else "0"
        sig = sigs.get(r["source_sha256"].strip().casefold(), {})
        status = str(sig.get("auth_status") or "").strip()
        subject = str(sig.get("signer_subject") or "").strip()
        r["auth_status"] = status
        r["signer_subject"] = subject
        r["signed_valid"] = "1" if status == "Valid" else "0"
        t12 = matches_trusted(subject, TRUSTED_TERMS_12)
        t14 = matches_trusted(subject, TRUSTED_TERMS_14)
        r["signed_trusted_12"] = "1" if (status == "Valid" and t12) else "0"
        r["signed_trusted_14"] = "1" if (status == "Valid" and t14) else "0"

        if not r["raw_path"]:
            r["bucket"] = "not_located"; buckets["not_located"] += 1
        elif status == "Valid" and t14:
            r["bucket"] = "signed_trusted_14"; buckets["signed_trusted_14"] += 1
        elif status == "Valid" and t12:
            r["bucket"] = "signed_trusted_12_only"; buckets["signed_trusted_12_only"] += 1
        elif status == "Valid":
            r["bucket"] = "signed_valid_other"; buckets["signed_valid_other"] += 1
        elif r["cross_tree_conflict"] == "1":
            r["bucket"] = "cross_tree_conflict"; buckets["cross_tree_conflict"] += 1
        elif p >= 0.99:
            r["bucket"] = "high_conf_unsure"; buckets["high_conf_unsure"] += 1
        elif 0.5 <= p < 0.6:
            r["bucket"] = "near_threshold"; buckets["near_threshold"] += 1
        else:
            r["bucket"] = "genuine_error"; buckets["genuine_error"] += 1

    out_csv = OUTPUT_DIR / "test739k_fp_classification.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(fps)
    print(f"[saved] {out_csv} ({len(fps):,} rows)")

    print("\n===== FP 拆解汇总 =====")
    print(f"总 FP: {len(fps):,}  (占 test 的 {len(fps)/147796*100:.2f}%)")
    total_errors = len(fps) + 1525  # 与 FN 1525 合并为总错误
    for k, v in buckets.items():
        print(f"  {k:24s} {v:6,}  ({v/len(fps)*100:5.1f}% of FP)")
    # 信息：若可信签名 14 名单的 FP 全部可翻正（外部信号），F1 提升预估
    print(f"\n注: FP 总数 {len(fps):,} + FN 1525 = 总错误 ~{total_errors:,}；"
          f"F1>0.99 需总错误 < ~2300")


if __name__ == "__main__":
    main()
