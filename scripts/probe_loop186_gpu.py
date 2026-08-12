"""Loop186 GPU 显存探测：验证 17.3M 参数 + SAM 是否在 6.5 GiB 预算内。

运行 2 个 effective batch（含 SAM 双 forward-backward），测量峰值显存。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from src.loop186 import (
    HGConvRegionConfig,
    HGConvRegionNet,
    SAM_RHO,
    assert_contract_invariants,
)


class SAM(torch.optim.Optimizer):
    """SAM wrapper（从 run_loop184 复制，用于探测）。"""

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        if rho <= 0:
            raise ValueError(f"Invalid rho: {rho}")
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad.detach().float() * scale.to(p.grad.device)
                e_w = e_w.to(p.dtype)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = self.state[p].get("e_w")
                if e_w is not None:
                    p.sub_(e_w)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.detach().float().norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm


def make_synthetic_batch(batch_size: int = 2, device: torch.device = torch.device("cpu")) -> dict:
    """生成合成 batch（与 Loop186 输入 ABI 一致）。"""
    cfg = HGConvRegionConfig()
    region_tokens = torch.randint(0, cfg.vocabulary_size - 1, (batch_size, cfg.expected_regions, cfg.expected_region_bytes), device=device)
    region_lengths = torch.full((batch_size, cfg.expected_regions), cfg.expected_region_bytes, dtype=torch.long, device=device)
    # 随机置空一些 region
    region_lengths[:, 8:] = 0
    region_types = torch.ones((batch_size, cfg.expected_regions), dtype=torch.long, device=device)
    region_types[:, 8:] = 0
    offset_buckets = torch.randint(0, cfg.bucket_count, (batch_size, cfg.expected_regions), dtype=torch.long, device=device)
    offset_buckets[:, 8:] = 0
    length_buckets = torch.randint(0, cfg.bucket_count, (batch_size, cfg.expected_regions), dtype=torch.long, device=device)
    length_buckets[:, 8:] = 0
    # 修复 padding
    for b in range(batch_size):
        for r in range(cfg.expected_regions):
            if region_lengths[b, r] < cfg.expected_region_bytes:
                region_tokens[b, r, region_lengths[b, r]:] = cfg.padding_token
    b0_features = torch.randn((batch_size, cfg.b0_feature_dim), device=device)
    labels = torch.randint(0, 2, (batch_size,), dtype=torch.long, device=device)
    return {
        "region_tokens": region_tokens,
        "region_lengths": region_lengths,
        "region_types": region_types,
        "offset_buckets": offset_buckets,
        "length_buckets": length_buckets,
        "b0_features": b0_features,
        "labels": labels,
    }


def main() -> int:
    assert_contract_invariants()
    print("[probe] Loop186 GPU 显存探测")

    device = torch.device("cuda")
    print(f"[probe] device: {device}, BF16: {torch.cuda.is_bf16_supported()}")

    config = HGConvRegionConfig()
    model = HGConvRegionNet(config).to(device=device, dtype=torch.float32)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[probe] 参数量: {param_count:,}")

    sam_optimizer = SAM(
        model.parameters(),
        torch.optim.AdamW,
        rho=SAM_RHO,
        lr=3e-4,
        weight_decay=3e-2,
    )

    # 重置显存统计
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    print(f"[probe] 初始显存: {torch.cuda.memory_allocated(device) / 1024**3:.3f} GiB")

    # 模拟 2 个 SAM effective batch
    for step in range(2):
        # SAM first pass
        sam_optimizer.zero_grad(set_to_none=True)
        batch = make_synthetic_batch(batch_size=2, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = model(
                batch["region_tokens"], batch["region_lengths"], batch["region_types"],
                batch["offset_buckets"], batch["length_buckets"], batch["b0_features"],
            )["fusion_logits"]
            loss = torch.nn.functional.cross_entropy(logits.float(), batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sam_optimizer.first_step(zero_grad=True)

        # SAM second pass
        batch2 = make_synthetic_batch(batch_size=2, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits2 = model(
                batch2["region_tokens"], batch2["region_lengths"], batch2["region_types"],
                batch2["offset_buckets"], batch2["length_buckets"], batch2["b0_features"],
            )["fusion_logits"]
            loss2 = torch.nn.functional.cross_entropy(logits2.float(), batch2["labels"])
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sam_optimizer.second_step(zero_grad=True)

        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"[probe] step {step+1} 完成, peak={peak:.3f} GiB")

    peak_bytes = torch.cuda.max_memory_allocated(device)
    print(f"\n[probe] 峰值显存: {peak_bytes / 1024**3:.3f} GiB ({peak_bytes:,} bytes)")
    print(f"[probe] Phase A gate: 6.500 GiB (6,500,000,000 bytes)")
    if peak_bytes < 6_500_000_000:
        print(f"[probe] ✅ 在预算内 (余量 {(6_500_000_000 - peak_bytes) / 1024**3:.3f} GiB)")
    else:
        print(f"[probe] ❌ 超出预算 (超出 {(peak_bytes - 6_500_000_000) / 1024**3:.3f} GiB)")
        print(f"[probe] 需要降低 microbatch 或关闭 SAM")

    return 0


if __name__ == "__main__":
    sys.exit(main())
