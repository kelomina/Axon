#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""忠实复刻真实训练 step 的 A/B benchmark。

与 benchmark_step.py 的区别：完全照搬 train_739k_benign_hardneg.py + trainer._train_impl 的 step，
包括：
  - model(return_state=True, compute_diversity_loss=True)   ← diversity loss 路径
  - _training_loss：focal(gamma=1.0, alpha=0.55) + label_smoothing(0.03) + sample_weights 加权
  - loss += 0.03 * diversity_loss
  - SubDataset 带 sample_weights（batch 5 元组）
  - 数据来自真实 base_prob 权重

用法（GPU 独占）：
  python scripts/benchmark_step_faithful.py --tag=opt --dtype=fp32
  # 手工还原 DSRA 4 处优化后
  python scripts/benchmark_step_faithful.py --tag=base --dtype=fp32
  # bf16 AMP（RTX 4070 支持 tensor core bf16）
  python scripts/benchmark_step_faithful.py --tag=opt --dtype=bf16

输出 reports/benchmark_step_faithful_<tag>_<dtype>.json
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

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import train_739k_full as T  # noqa: E402

from config import AxonExperimentConfig, TrainingConfig  # noqa: E402
from dataset import FeatureCacheDataset, create_stratified_split, SubDataset  # noqa: E402
from model import AxonMalwareModel  # noqa: E402

CKPT = PROJECT_ROOT / "models" / "full_739k_benign" / "best_model_739k.pt"
TRUNCATE_BYTE_LENGTH = 4096
BATCH_SIZE = 64
N_STEPS = 30
N_WARMUP = 3
BASE_PROB_DIR = PROJECT_ROOT / "reports" / "full_739k_benign" / "base_prob"


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
    tag = "opt"
    dtype_str = "fp32"
    BATCH_SIZE = 64  # 模块默认；--batch= 覆盖
    do_compile = False
    do_profile = False
    do_tf32 = False
    compile_mode = "dynamic"  # "dynamic" | "default" | "reduce-overhead"
    for a in sys.argv[1:]:
        if a.startswith("--tag="):
            tag = a.split("=", 1)[1]
        if a.startswith("--dtype="):
            dtype_str = a.split("=", 1)[1]
        if a.startswith("--batch="):
            BATCH_SIZE = int(a.split("=", 1)[1])
        if a == "--compile":
            do_compile = True
        if a == "--tf32":
            do_tf32 = True
        if a.startswith("--mode="):
            compile_mode = a.split("=", 1)[1]
        if a == "--profile":
            do_profile = True
    use_bf16 = dtype_str == "bf16"

    device = torch.device("cuda")
    print(f"[bench-faithful] tag={tag} batch={BATCH_SIZE} n={N_STEPS} "
          f"dtype={dtype_str} compile={do_compile} profile={do_profile} device={device}")
    if do_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print("[bench-faithful] TF32 matmul enabled")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    raw_cfg = ckpt["config"]
    config = AxonExperimentConfig.from_dict(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg

    # —— 与 train_739k_benign_hardneg.py 完全一致的 dataset/split/权重 ——
    dataset = FeatureCacheDataset(
        data_dir=str(PROJECT_ROOT / "data"),
        cache_dir=str(PROJECT_ROOT / "data" / ".cache"),
        max_byte_length=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim,
        stat_feature_dim=config.stat_feature_dim,
        max_samples_per_class=None,
        axon_config=config,
    )
    total = len(dataset)
    label_arr = np.array(dataset.label_list)

    base_prob = np.concatenate(
        [np.load(f) for f in sorted((BASE_PROB_DIR).glob("chunk_*.npy"))]
    ).astype(np.float32)
    assert len(base_prob) == total
    w = np.ones(total, dtype=np.float32)
    ben = label_arr == 0
    w[ben] = 1.0 + 5.0 * base_prob[ben]

    train_ds, val_ds, test_ds = create_stratified_split(
        dataset, val_ratio=0.10, test_ratio=0.20, seed=42, axon_config=config,
    )
    train_ds = SubDataset(train_ds.base_dataset, train_ds.indices, sample_weights=w[train_ds.indices])
    train_ds = _TruncatedByteDataset(train_ds, TRUNCATE_BYTE_LENGTH)

    n_sub = BATCH_SIZE * 4
    sub = SubDataset(train_ds, list(range(n_sub)))
    loader = torch.utils.data.DataLoader(sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    batches = []
    for b in loader:
        batches.append(tuple(t.to(device) for t in b))
    print(f"[data] {len(batches)} batches staged ({len(batches)*BATCH_SIZE} samples)")

    model = AxonMalwareModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).train()
    if do_compile:
        if compile_mode == "reduce-overhead":
            # CUDA graph 捕获：消除 launch 开销（kernel-bound 负载最大杠杆）；要求静态形状
            model = torch.compile(model, mode="reduce-overhead")
            print("[bench-faithful] torch.compile enabled (reduce-overhead)")
        elif compile_mode == "default":
            model = torch.compile(model)
            print("[bench-faithful] torch.compile enabled (default static)")
        else:
            model = torch.compile(model, dynamic=True)
            print("[bench-faithful] torch.compile enabled (dynamic)")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # —— trainer._create_criterion 完全复刻：focal + label_smoothing ——
    def focal_loss(logits, targets, gamma=1.0, alpha=0.55, reduction='mean'):
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none',
                                                    label_smoothing=0.03)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** gamma
        alpha_t = alpha * (targets == 1).float() + (1 - alpha) * (targets == 0).float()
        per_sample = alpha_t * focal_weight * ce_loss
        if reduction == 'none':
            return per_sample
        return per_sample.mean()

    DIV_W = 0.03
    bf16 = use_bf16

    def step(batch):
        byte_seq, pe, stat, label, sw = batch
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16)
        with autocast_ctx:
            outputs = model(
                byte_seq, pe, stat,
                return_state=True,
                compute_diversity_loss=True,
            )
        # loss 始终在 fp32 计算（AMP 推荐模式），避免 bf16 精度损失污染 loss
        logits = outputs["logits"].float()
        loss = focal_loss(logits, label)  # sample_weights=None 时 trainer 直接 criterion
        dsra_state = outputs.get("dsra_state")
        if dsra_state is not None:
            div_loss = outputs.get("diversity_loss")
            if div_loss is None:
                s = dsra_state[0] if isinstance(dsra_state, (list, tuple)) else dsra_state
                div_loss = model.dsra.diversity_loss(s)
            loss = loss + DIV_W * div_loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        return float(loss.detach().item())

    for i in range(N_WARMUP):
        step(batches[i % len(batches)])
    torch.cuda.synchronize()
    print("[warmup] done")

    times = []
    losses = []
    if do_profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            for i in range(6):
                loss = step(batches[i % len(batches)])
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
        prof.export_chrome_trace(
            str(PROJECT_ROOT / "reports" / f"trace_step_faithful_{tag}.json")
        )
        return

    for i in range(N_STEPS):
        t0 = time.perf_counter()
        loss = step(batches[i % len(batches)])
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(loss)

    a = np.array(times)
    print(f"[result] tag={tag} mean={a.mean():.4f}s  p50={np.median(a):.4f}  "
          f"p90={np.percentile(a, 90):.4f}  min={a.min():.4f}  "
          f"loss_first={losses[0]:.4f} loss_last={losses[-1]:.4f}")
    print(f"[estimate] epoch(569170/{BATCH_SIZE} steps) = {a.mean()*569170/BATCH_SIZE/3600:.2f} h")

    out = {
        "tag": tag, "batch_size": BATCH_SIZE, "n_steps": N_STEPS, "faithful": True,
        "dtype": dtype_str,
        "mean_sec": float(a.mean()), "p50_sec": float(np.median(a)),
        "p90_sec": float(np.percentile(a, 90)), "min_sec": float(a.min()),
        "loss_first": losses[0], "loss_last": losses[-1], "losses": losses,
    }
    out_path = PROJECT_ROOT / "reports" / f"benchmark_step_faithful_{tag}_{dtype_str}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
