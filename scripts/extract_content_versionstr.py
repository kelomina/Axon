#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""版本字符串内容特征提取（val+test，仅可定位样本）。

从原始 PE 的 VS_VERSION_INFO 提取良性 EXE 判别信号：
  has_company / has_filedesc / has_product / has_version_str / has_original_fn /
  has_legal_cp / has_internal_name / n_string_fields_log
可定位样本（E:/F: 现存，G:/H:→F: 纯前缀替换）真实提取；不可定位/解析失败 → 全 0（unknown）。
目录定位用一次性 listdir 索引集（pickle 持久化），避免逐文件 stat（F: 盘 ~7ms/次，24 万次≈30min）。
解析循环挂起免疫（chunksize=1 + 45s 超时 + 池重启）：F: 恶意语料按访问触发 AV 扫描，个别文件 open 可能无限阻塞。

输出 reports/full_739k_benign/content_versionstr/{val,npy,test.npy,meta.json}
meta.json 记录 8 个特征名 + 覆盖统计（每类可定位率）。
CPU-only，multiprocessing 并行。
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

V2_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_v2string"
V1_META = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_pe_v1" / "meta.csv"
OUT_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "content_versionstr"

FEATURES = [
    "vs_has_company", "vs_has_filedesc", "vs_has_product", "vs_has_version_str",
    "vs_has_original_fn", "vs_has_legal_cp", "vs_has_internal_name", "vs_n_fields_log",
]

WANT = {"CompanyName", "FileDescription", "ProductName", "OriginalFilename",
        "FileVersion", "ProductVersion", "LegalCopyright", "InternalName"}


def version_row(path: str):
    """Return 8-dim float32 row for one file; None if unreadable/not a PE."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) < 512:
            return None
        import pefile
        pe = pefile.PE(data=data, fast_load=True)
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"],
        ])
        info = {}
        for flist in getattr(pe, "FileInfo", []) or []:
            for fe in flist:
                for tbl in getattr(fe, "StringTable", []) or []:
                    entries = getattr(tbl, "entries", None) or {}
                    for k, v in entries.items():
                        key = k.decode("utf-8", "replace") if isinstance(k, bytes) else str(k)
                        if key in WANT:
                            vv = (v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)).strip()
                            if vv:
                                info.setdefault(key, vv)
        return np.asarray([
            float("CompanyName" in info),
            float("FileDescription" in info),
            float("ProductName" in info),
            float(("FileVersion" in info) or ("ProductVersion" in info)),
            float("OriginalFilename" in info),
            float("LegalCopyright" in info),
            float("InternalName" in info),
            min(float(len(info)), 15.0) / 15.0,
        ], dtype=np.float32)
    except Exception:
        return None


def worker(args):
    idx, path = args
    r = version_row(path)
    return (idx, r)


def main() -> None:
    import multiprocessing as mp
    t0 = time.time()
    sys.stdout.reconfigure(line_buffering=True)  # 重定向时仍逐行刷出
    print("=== content_versionstr extraction (val+test) ===")

    with open(V2_DIR / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    val_idx = np.asarray(meta["val_indices"], dtype=np.int64)
    test_idx = np.asarray(meta["test_indices"], dtype=np.int64)

    all_idx = lambda: np.concatenate([val_idx, test_idx])  # noqa: E731

    paths = {}
    with open(V1_META, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            paths[int(r["index"])] = r["raw_path"]

    # ---- 目录索引 resolve（F: 盘逐文件 isfile 慢 ~7ms/次 → 24 万次≈30min） ----
    # 每个唯一父目录 listdir 一次建 set，持久化 pickle 复用；成员判断代替 stat。
    INDEX_PKL = OUT_DIR / "dir_index.pkl"

    def _resolve_str(p):
        # 盘符映射：G:/H: 均已换盘到 F:（纯前缀替换）
        if p.startswith("G:") or p.startswith("H:"):
            p = "F:" + p[2:]
        return p

    def load_or_build_index(parents):
        if INDEX_PKL.exists():
            with open(INDEX_PKL, "rb") as f:
                index = pickle.load(f)
            missing = parents - set(index)
        else:
            index, missing = {}, parents
        if missing:
            for d in sorted(missing):
                try:
                    index[d] = set(os.listdir(d))
                except OSError:
                    index[d] = set()
            with open(INDEX_PKL, "wb") as f:
                pickle.dump(index, f)
        return index

    parents = set()
    for i in all_idx():
        p = _resolve_str(paths.get(i, ""))
        if p:
            parents.add(os.path.dirname(p))
    index = load_or_build_index(parents)
    print(f"[index] {len(index)} dirs indexed ({time.time()-t0:.0f}s)")

    def resolve(i):
        p = _resolve_str(paths.get(i, ""))
        if not p:
            return ""
        s = index.get(os.path.dirname(p))
        if s is None:
            return ""
        return p if os.path.basename(p) in s else ""

    def run_block(idxs, out_npy, out_meta):
        tasks = []
        for i in idxs:
            p = resolve(int(i))
            if p:
                tasks.append((int(i), p))
        print(f"[{out_npy.name}] {len(tasks)}/{len(idxs)} locatable, extracting...")
        # 挂起免疫：chunksize=1 + 45s 结果超时 + 池重启。
        # 根因：F: 恶意语料按访问触发 AV 扫描，个别文件 open 可能无限阻塞，拖死整池。
        # 每轮只重试未完成文件；连续 3 轮无进展 → 放弃并记录（防整目录被锁死时无限空转）。
        ncpu = min(16, os.cpu_count() or 1)
        mat = np.zeros((len(idxs), 8), dtype=np.float32)
        covered = np.zeros(len(idxs), dtype=bool)
        pending = list(tasks)
        done_total = 0
        last_done = -1
        stalled = 0
        for restart in range(12):
            if not pending:
                break
            with mp.Pool(ncpu) as pool:
                it = pool.imap_unordered(worker, pending, chunksize=1)
                batch_done = set()
                while True:
                    try:
                        idx, row = it.next(timeout=45)
                    except mp.TimeoutError:
                        print(f"    [warn] 45s no result (left {len(pending)}, restart#{restart})")
                        pool.terminate()
                        pool.join()
                        break
                    except StopIteration:
                        pool.close()
                        pool.join()
                        break
                    batch_done.add(idx)
                    pos = int(np.where(idxs == idx)[0][0])
                    if row is not None:
                        mat[pos] = row
                        covered[pos] = True
                    done_total += 1
                    if done_total % 10000 == 0:
                        print(f"    {done_total}/{len(tasks)} ({time.time()-t0:.0f}s, restart#{restart})")
                if batch_done:
                    pending = [t for t in pending if t[0] not in batch_done]
            if done_total == last_done:
                stalled += 1
            else:
                stalled = 0
            last_done = done_total
            print(f"    [restart] done={done_total}/{len(tasks)} left={len(pending)} stalled={stalled}")
            if stalled >= 3:
                print(f"    [giveup] {len(pending)} files unresponsive across 3 rounds")
                break
        if pending:
            print(f"    [warn] gave up on {len(pending)} persistently-blocked files")
        np.save(out_npy, mat)
        out_meta.write_text(json.dumps({
            "names": FEATURES,
            "n_total": int(len(idxs)),
            "n_located": int(len(tasks)),
            "n_parsed_ok": int(covered.sum()),
            "n_gave_up": int(len(pending)),
        }, indent=2), encoding="utf-8")
        print(f"[{out_npy.name}] saved: {len(tasks)} located, {int(covered.sum())} parsed, "
              f"gave_up={len(pending)}, coverage={covered.mean():.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_block(val_idx, OUT_DIR / "val.npy", OUT_DIR / "val_meta.json")
    run_block(test_idx, OUT_DIR / "test.npy", OUT_DIR / "test_meta.json")
    print(f"[done] {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
