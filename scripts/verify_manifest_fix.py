#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 manifest label 修复：对全部冲突 sha 的样本走一遍 dataset.__getitem__。

背景：之前训练在 val 阶段崩溃 —— _load_cached_feature_npz 报
"Cache label mismatch ... expected 1, got 0"（manifest label 与 npz label 不一致）。
sync_cache_manifest_labels.py 已把 manifest 的 label 同步为最终判定值。
本脚本复现崩溃路径，若全部通过则训练可安全重启。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from config import AxonExperimentConfig  # noqa: E402
from dataset import FeatureCacheDataset  # noqa: E402

MOVE_PLAN = PROJECT_ROOT / "reports" / "full_739k_benign" / "label_governance" / "move_plan_preview.csv"
CKPT = PROJECT_ROOT / "models" / "full_739k_benign" / "best_model_739k.pt"


def main() -> None:
    plan: set[str] = set()
    with open(MOVE_PLAN, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            plan.add(row["sha256"].strip().casefold())
    print(f"[plan] {len(plan)} conflict shas")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    raw_cfg = ckpt["config"]
    config = AxonExperimentConfig.from_dict(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg

    dataset = FeatureCacheDataset(
        data_dir=str(PROJECT_ROOT / "data"),
        cache_dir=str(PROJECT_ROOT / "data" / ".cache"),
        max_byte_length=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        max_samples_per_class=None,
        axon_config=config,
    )
    print(f"[dataset] {len(dataset):,} samples")

    sha_list = dataset.source_sha256_list
    matched = 0
    for i, sha in enumerate(sha_list):
        if sha in plan:
            item = dataset[i]  # 崩溃路径
            matched += 1
    print(f"[ok] all {matched} conflict-sha samples __getitem__ passed (no label mismatch)")

    assert matched == len(plan), f"only {matched}/{len(plan)} conflict shas found in dataset"
    print("[PASS] manifest fix verified — training can restart safely")


if __name__ == "__main__":
    main()
