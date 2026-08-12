#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU 算子级 profile：单训练步 forward+backward 的热点算子。

在剩余显存（batch 16）运行 torch.profiler，定位每步 ~0.47s 中
哪些算子在烧时间。与正在运行的训练共享 GPU，仅 6 步，短暂竞争。

用法: python scripts/profile_step_cuda.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from config import AxonExperimentConfig  # noqa: E402
from dataset import FeatureCacheDataset, SubDataset  # noqa: E402
from model import AxonMalwareModel  # noqa: E402

CKPT = PROJECT_ROOT / "models" / "full_739k_benign" / "best_model_739k.pt"
TRUNCATE_BYTE_LENGTH = 4096
N_STEPS = 6


class _TruncatedByteDataset(torch.utils.data.Dataset):
    def __init__(self, base, max_len: int = TRUNCATE_BYTE_LENGTH):
        self.base = base
        self.max_len = int(max_len)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        byte_seq = item[0]
        if byte_seq.shape[0] > self.max_len:
            byte_seq = byte_seq[: self.max_len]
        return (byte_seq,) + tuple(item[1:])


def main() -> None:
    device = torch.device("cuda")
    print(f"Device: {device}  free_mem={torch.cuda.mem_get_info()[0]/2**20:.0f} MiB")

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
    sub = SubDataset(_TruncatedByteDataset(dataset, TRUNCATE_BYTE_LENGTH), list(range(64)))
    loader = torch.utils.data.DataLoader(sub, batch_size=16, shuffle=False, num_workers=2)

    model = AxonMalwareModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).train()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    criterion = torch.nn.CrossEntropyLoss()

    batch = next(iter(loader))
    byte_seq, pe, stat, label = [t.to(device) for t in batch]

    # warmup
    for _ in range(2):
        out = model(byte_seq, pe, stat)
        loss = criterion(out["logits"], label)
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    print("[warmup] done")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(N_STEPS):
            out = model(byte_seq, pe, stat)
            loss = criterion(out["logits"], label)
            loss.backward()
            opt.step()
            opt.zero_grad()
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
    print("[done]")


if __name__ == "__main__":
    main()
