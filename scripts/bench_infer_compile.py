#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""推理吞吐基准：eager vs torch.compile（forward-only，含 softmax 打分）。

用法（GPU 独占）：
  python scripts/bench_infer_compile.py [--compile] [--batch 64]
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import AxonExperimentConfig  # noqa: E402
from dataset import FeatureCacheDataset, create_stratified_split, SubDataset  # noqa: E402
from model import AxonMalwareModel  # noqa: E402

CKPT = PROJECT_ROOT / "models" / "full_739k_benign" / "best_model_739k.pt"
N_REPEAT = 50
N_WARMUP = 5
# 真实部署形状：训练/DLL 都截断到 4096（8 chunks）。checkpoint config 的
# max_byte_length=65536 是缓存长度，不能直接喂模型（会 16x 放大计算）。
TRUNCATE_BYTE_LENGTH = 4096


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
    do_compile = "--compile" in sys.argv
    compile_mode = "dynamic"  # "dynamic" | "default" | "reduce-overhead"
    batch = 64
    for a in sys.argv[1:]:
        if a.startswith("--batch="):
            batch = int(a.split("=", 1)[1])
        if a.startswith("--mode="):
            compile_mode = a.split("=", 1)[1]
    device = torch.device("cuda")
    print(f"[bench-infer] batch={batch} compile={do_compile}")

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
    sub = SubDataset(train_ds, list(range(batch * 4)))
    sub = _TruncatedByteDataset(sub, TRUNCATE_BYTE_LENGTH)
    loader = torch.utils.data.DataLoader(sub, batch_size=batch, shuffle=False, num_workers=4)
    batches = []
    for b in loader:
        batches.append(tuple(t.to(device) for t in b))

    model = AxonMalwareModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    if do_compile:
        if compile_mode == "reduce-overhead":
            model = torch.compile(model, mode="reduce-overhead")
            print("[bench-infer] torch.compile enabled (reduce-overhead)")
        elif compile_mode == "default":
            model = torch.compile(model)
            print("[bench-infer] torch.compile enabled (default static)")
        else:
            model = torch.compile(model, dynamic=True)
            print("[bench-infer] torch.compile enabled (dynamic)")

    def infer(b):
        byte_seq, pe, stat, label = b
        with torch.no_grad(), torch.inference_mode():
            out = model(byte_seq, pe, stat)
            p = torch.softmax(out["logits"], dim=1)[:, 1]
        return float(p.mean().item())

    for i in range(N_WARMUP):
        infer(batches[i % len(batches)])
    torch.cuda.synchronize()
    print("[warmup] done")

    times = []
    for i in range(N_REPEAT):
        t0 = time.perf_counter()
        infer(batches[i % len(batches)])
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    a = np.array(times)
    per_sample_ms = a.mean() / batch * 1000
    print(f"[result] mean={a.mean():.4f}s/batch  p50={np.median(a):.4f}  "
          f"p90={np.percentile(a,90):.4f}  {per_sample_ms:.3f} ms/sample  "
          f"throughput={batch/a.mean():.0f} samples/s")


if __name__ == "__main__":
    main()
