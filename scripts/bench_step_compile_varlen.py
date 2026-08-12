#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 torch.compile 在可变长度输入（真实训练）下的速度与重编译开销。

真实训练里每个 batch 的 byte_seq 长度不同（截断到 4096，不补零），chunk 数 1~8。
torch.compile(dynamic=True) 可能：①对每个新形状重编译（慢），②graph break（无收益），
③符号化动态形状一次编译全覆盖（理想）。本脚本分别测 512/1024/2048/3072/4096 长度。

用法（GPU 独占）：
  python scripts/bench_step_compile_varlen.py [--compile] [--profile]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import AxonExperimentConfig  # noqa: E402
from dataset import FeatureCacheDataset, create_stratified_split, SubDataset  # noqa: E402
from model import AxonMalwareModel  # noqa: E402

CKPT = PROJECT_ROOT / "models" / "full_739k_benign" / "best_model_739k.pt"
BATCH = 64
LENGTHS = [512, 1024, 2048, 3072, 4096]
REPEAT = 20


def build_batch(dataset, base, length: int, device) -> tuple:
    """从真实 train split 采样 BATCH 个样本，统一截断到指定 length（与训练一致）。"""
    idx = np.random.RandomState(0).choice(len(base), size=BATCH, replace=False)
    byte_seq = torch.stack([base[i][0][:length] for i in idx]).to(device)
    pe = torch.stack([base[i][1] for i in idx]).to(device)
    stat = torch.stack([base[i][2] for i in idx]).to(device)
    label = torch.tensor([base[i][3] for i in idx]).to(device)
    return byte_seq, pe, stat, label


def step(model, opt, batch, DIV_W=0.03):
    byte_seq, pe, stat, label = batch
    outputs = model(byte_seq, pe, stat, return_state=True, compute_diversity_loss=True)
    logits = outputs["logits"].float()
    ce = torch.nn.functional.cross_entropy(logits, label, reduction="none", label_smoothing=0.03)
    loss = ce.mean()
    div = outputs.get("diversity_loss")
    if div is not None:
        loss = loss + DIV_W * div
    loss.backward()
    opt.step()
    opt.zero_grad()
    return float(loss.detach().item())


def main() -> None:
    do_compile = "--compile" in sys.argv
    do_profile = "--profile" in sys.argv
    device = torch.device("cuda")
    print(f"[varlen] batch={BATCH} lengths={LENGTHS} compile={do_compile}")

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
    train_ds, _, _ = create_stratified_split(
        dataset, val_ratio=0.10, test_ratio=0.20, seed=42, axon_config=config,
    )

    model = AxonMalwareModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).train()
    if do_compile:
        model = torch.compile(model, dynamic=True)
        print("[varlen] torch.compile enabled")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # 每种长度：第 1 次（含编译/重编译）+ 后续稳定速度
    for L in LENGTHS:
        batch = build_batch(dataset, train_ds, L, device)
        if do_compile:
            model.compile()  # no-op safety
        t0 = time.perf_counter()
        try:
            loss = step(model, opt, batch)
            torch.cuda.synchronize()
            t_first = time.perf_counter() - t0
        except Exception as e:
            print(f"  L={L:5d}  ERROR: {type(e).__name__}: {str(e)[:120]}")
            continue
        times = []
        for _ in range(REPEAT):
            t0 = time.perf_counter()
            step(model, opt, batch)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        a = np.array(times[3:])
        print(f"  L={L:5d} ({L//512} chunks)  first={t_first:.3f}s  "
              f"steady={a.mean():.4f}s  p90={np.percentile(a,90):.4f}  loss={loss:.4f}")

    if do_profile:
        from torch.profiler import ProfilerActivity, profile
        batch = build_batch(dataset, train_ds, 4096, device)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(4):
                step(model, opt, batch)
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
