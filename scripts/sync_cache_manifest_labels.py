#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同步 data/.cache 缓存 manifest 的 label 为最终判定值。

背景：sync_cache_conflict_labels.py 把冲突 sha 的 cache npz label 改为 new_label，
但 manifest（manifest_<hash>.json）里每个 sample 的 label 字段仍是旧值。
FeatureCacheDataset 从 manifest 构建 label_list（作为 expected_label），训练/验证时
_load_cached_feature_npz 校验 npz label == expected_label，不一致即崩溃：
  "Cache label mismatch ... expected 1, got 0"

依据：
  - manifest 内 sample["source_sha256"]（全 64 位，与 move_plan 的 sha256 键一致）
  - move_plan_preview.csv   sha256 -> new_label（最终判定）

只改匹配 sample 的 label 字段；manifest 294MB 太大，用 dataset 流式读写，
不改写其他字段；tmp + replace 原子写。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import _iter_manifest_sample_entries, _write_cache_manifest_stream  # noqa: E402

MANIFEST = PROJECT_ROOT / "data" / ".cache" / "manifest_a807341e.json"
MOVE_PLAN = PROJECT_ROOT / "reports" / "full_739k_benign" / "label_governance" / "move_plan_preview.csv"


def _read_header(manifest_path: Path) -> dict:
    """读取 manifest header（samples 之前的部分），不加载整棵 JSON 树。

    返回含 samples 空列表的 dict，供 _write_cache_manifest_stream 复用 header 字段。
    """
    buf = ""
    with manifest_path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                raise ValueError(f"manifest has no samples array: {manifest_path}")
            buf += chunk
            idx = buf.find('"samples":')
            if idx >= 0:
                header_text = buf[:idx].rstrip()
                if header_text.endswith(","):
                    header_text += '"samples":[]}'
                else:
                    header_text += ',"samples":[]}'
                return json.loads(header_text)
            buf = buf[-32:]  # 保留尾部，防 "samples" 跨 chunk 被截断


def main() -> None:
    plan: dict[str, int] = {}
    with open(MOVE_PLAN, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            plan[row["sha256"].strip().casefold()] = int(row["new_label"])
    print(f"[plan] {len(plan)} conflict shas with final labels")

    header = _read_header(MANIFEST)
    print(f"[header] cache_config_hash={header.get('cache_config_hash')} "
          f"version={header.get('version')}")

    fixes: list[tuple[str, int, int]] = []

    def gen():
        n_total = 0
        for sample in _iter_manifest_sample_entries(MANIFEST):
            n_total += 1
            sha = str(sample.get("source_sha256") or "").strip().casefold()
            if sha in plan:
                old = int(sample["label"])
                new = plan[sha]
                if old != new:
                    sample["label"] = new
                    fixes.append((sha, old, new))
            yield sample
        print(f"[total] {n_total} manifest samples streamed")

    out = MANIFEST.with_name(MANIFEST.name + ".fixed")
    _write_cache_manifest_stream(out, header, gen())

    # 验证修复结果
    import os
    n_new = 0
    for sample in _iter_manifest_sample_entries(out):
        sha = str(sample.get("source_sha256") or "").strip().casefold()
        if sha in plan:
            if int(sample["label"]) != plan[sha]:
                raise SystemExit(f"VERIFY FAILED for {sha[:16]}: label not synced")
            n_new += 1
    print(f"[verify] {n_new} conflict shas all match new_label in {out.name}")

    os.replace(out, MANIFEST)
    print(f"[ok] replaced {MANIFEST.name}")

    print(f"\n=== summary ===")
    print(f"fixed={len(fixes)}")
    if fixes:
        n_1to0 = sum(1 for _, o, n in fixes if o == 1 and n == 0)
        n_0to1 = sum(1 for _, o, n in fixes if o == 0 and n == 1)
        print(f"  1->0 (恶->良): {n_1to0}   0->1 (良->恶): {n_0to1}")
        for sha, old, new in fixes[:30]:
            print(f"  {sha[:16]}... label {old}->{new}")


if __name__ == "__main__":
    main()
