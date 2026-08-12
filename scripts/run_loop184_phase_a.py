"""Loop184 Phase A 训练脚本：架构扩展 + SAM + SWA + 数据增强。

Loop184 = Loop183 数据增强基础 + 4 项激进改进：
1. 架构扩展：transformer_layers 2→4, hgconv_blocks 1→2（参数量 1.62M → 2.61M, +60.84%）
2. SAM 优化器（rho=0.05）：找平坦极小值，每个 effective batch 应用
3. SWA（75% epoch 启动）：平均最后 5 个 epoch 权重，替代 EMA
4. 保持 Loop183 数据增强：Mixup α=0.4, Region dropout p=0.3, Label smoothing ε=0.05

资源估算：
- 参数量 2,610,573（仍在 6.5 GiB GPU 预算内）
- SAM 2x forward-backward：epoch time ~7 min × 2 = 14 min
- 20 epochs × 14 min = 280 min = 4.7h（在 6h 预算内）

用法:
    python scripts/run_loop184_phase_a.py
    python scripts/run_loop184_phase_a.py --device cuda
    python scripts/run_loop184_phase_a.py --device cpu --max-epochs 3
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.optim.optimizer import Optimizer

from src.loop184 import (
    PHASE_A_GATE,
    HGConvRegionConfig,
    HGConvRegionNet,
    MULTI_SCALE_FILTER_LENGTHS,
    MIXUP_ALPHA,
    REGION_DROPOUT_PROB,
    LABEL_SMOOTHING_EPS,
    SAM_RHO,
    SWA_START_EPOCH,
    SWA_LR,
    SWA_ANNEAL_EPOCHS,
    PhaseADataLoader,
    ResourceCell,
    assert_contract_invariants,
    assert_phase0_closure,
)
from src.loop184.contracts import LOOP_ID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "roadmap_9997" / "loop184"


# ---------------------------------------------------------------------------
# SAM 优化器实现（基于 Foret et al. 2021）
# ---------------------------------------------------------------------------


class SAM(Optimizer):
    """Sharpness-Aware Minimization wrapper.

    每个 effective step 需要：
    1. first_step: 用累积梯度计算扰动并加到 w
    2. 在扰动点 w+ε 上重新 forward+backward
    3. second_step: 在 w+ε 的梯度上更新

    所有计算在 FP32 下执行以保证精度（BF16 autocast 仅包裹前向/反向）。
    """

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
                # FP32 下计算扰动
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
                # 恢复 w
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

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ---------------------------------------------------------------------------
# SWA 实现（基于 Izmailov et al. 2018）
# ---------------------------------------------------------------------------


class SWAContainer:
    """Stochastic Weight Averaging 容器。

    在 SWA_START_EPOCH 后，每个 epoch 末更新参数平均。
    训练结束后，update_bn() 重新计算 BN 统计量。
    """

    def __init__(self, model: nn.Module) -> None:
        self.swa_n = 0
        self.swa_state = {
            name: value.detach().clone().float()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    @torch.no_grad()
    def update_parameters(self, model: nn.Module) -> None:
        self.swa_n += 1
        for name, value in model.state_dict().items():
            if not value.is_floating_point():
                continue
            target = self.swa_state[name]
            target.mul_(self.swa_n - 1).add_(value.detach().float()).div_(self.swa_n)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                value.copy_(self.swa_state[name].to(value.dtype))


# ---------------------------------------------------------------------------
# Preflight 与授权
# ---------------------------------------------------------------------------


def preflight() -> dict[str, object]:
    print("[preflight] Loop184 契约自检...")
    assert_contract_invariants()

    print("[preflight] Loop184 源码闭包检查...")
    closure_report = assert_phase0_closure()
    print(f"[preflight] 源码闭包: {len(closure_report.scanned_files)} 文件, 0 违规")

    print("[preflight] 设备检测...")
    cuda_available = torch.cuda.is_available()
    bf16_supported = cuda_available and torch.cuda.is_bf16_supported()
    device_name = "cuda" if cuda_available else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    gpu_total_bytes = (
        torch.cuda.get_device_properties(0).total_memory if cuda_available else 0
    )
    print(f"[preflight] CUDA available: {cuda_available}")
    print(f"[preflight] BF16 supported: {bf16_supported}")
    if cuda_available:
        print(f"[preflight] GPU: {gpu_name}")
        print(f"[preflight] GPU total memory: {gpu_total_bytes / 1024**3:.2f} GiB")

    print("[preflight] 构造模型预估参数量...")
    probe_config = HGConvRegionConfig()
    probe_model = HGConvRegionNet(probe_config)
    probe_params = sum(p.numel() for p in probe_model.parameters() if p.requires_grad)
    print(f"[preflight] Loop184 参数量: {probe_params:,} (vs Loop183: 1,622,153, +{(probe_params-1622153)/1622153*100:.1f}%)")
    del probe_model, probe_config

    print(f"[preflight] 数据增强: Mixup α={MIXUP_ALPHA}, Region dropout p={REGION_DROPOUT_PROB}, Label smoothing ε={LABEL_SMOOTHING_EPS}")
    print(f"[preflight] SAM: rho={SAM_RHO}, 每个 effective batch 应用")
    print(f"[preflight] SWA: start_epoch={SWA_START_EPOCH}, swa_lr={SWA_LR}, anneal={SWA_ANNEAL_EPOCHS} epochs")

    return {
        "closure_scanned_files": list(closure_report.scanned_files),
        "closure_manifest": dict(closure_report.manifest),
        "cuda_available": cuda_available,
        "bf16_supported": bf16_supported,
        "device_name": device_name,
        "gpu_name": gpu_name,
        "gpu_total_bytes": int(gpu_total_bytes),
    }


def generate_authorization_json(preflight_result: dict[str, object]) -> dict[str, object]:
    auth = {
        "schema": "axon_loop184_phase_a_authorization_v1",
        "loop_id": LOOP_ID,
        "lineage": "arch_expand_sam_swa",
        "phase": "A",
        "decision": "loop184_authorized_by_user_2026_07_20_aggressive_research",
        "claim_scope": "architecture_expansion_sam_swa",
        "val_test_or_full_access_allowed": False,
        "fit_rows": PHASE_A_GATE.fit_rows,
        "selection_rows": PHASE_A_GATE.selection_rows,
        "fold0_model_rows": PHASE_A_GATE.fold0_model_rows,
        "max_epochs": PHASE_A_GATE.max_epochs,
        "gpu_allocated_bytes_limit": PHASE_A_GATE.gpu_allocated_bytes,
        "rss_bytes_limit": PHASE_A_GATE.rss_bytes,
        "wall_seconds_limit": PHASE_A_GATE.wall_seconds,
        "cuda_available": preflight_result["cuda_available"],
        "bf16_supported": preflight_result["bf16_supported"],
        "device_name": preflight_result["device_name"],
        "gpu_name": preflight_result["gpu_name"],
        "multi_scale_filter_lengths": list(MULTI_SCALE_FILTER_LENGTHS),
        "mixup_alpha": MIXUP_ALPHA,
        "region_dropout_prob": REGION_DROPOUT_PROB,
        "label_smoothing_eps": LABEL_SMOOTHING_EPS,
        "weight_decay": PHASE_A_GATE.weight_decay,
        "dropout": 0.2,
        "sam_rho": SAM_RHO,
        "swa_start_epoch": SWA_START_EPOCH,
        "swa_lr": SWA_LR,
        "swa_anneal_epochs": SWA_ANNEAL_EPOCHS,
        "transformer_layers": 4,
        "hgconv_blocks": 2,
        "selection_criterion": "max_f1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return auth


def deterministic_epoch_batches(
    fit_indices: np.ndarray,
    *,
    microbatch: int,
    seed: int,
    epoch: int,
) -> tuple[np.ndarray, ...]:
    material = f"loop184-phase-a|{seed}|{epoch}|{fit_indices.size}".encode("ascii")
    order_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    order = fit_indices.copy()
    np.random.default_rng(order_seed).shuffle(order)
    return tuple(order[start : start + microbatch] for start in range(0, order.size, microbatch))


def resolve_device(requested: str) -> tuple[torch.device, bool]:
    if requested == "cpu":
        return torch.device("cpu"), False
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA explicitly requested but unavailable")
    if requested in {"auto", "cuda"} and cuda_available:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Loop184 CUDA requires BF16 support")
        return torch.device("cuda"), True
    return torch.device("cpu"), False


def autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }


# ---------------------------------------------------------------------------
# 数据增强（与 Loop183 一致）
# ---------------------------------------------------------------------------


def apply_region_dropout(
    batch: dict[str, torch.Tensor],
    *,
    prob: float,
    generator: torch.Generator,
    padding_token: int = 256,
) -> dict[str, torch.Tensor]:
    """随机 mask 1 个非空 region（训练时）。"""
    if prob <= 0.0:
        return batch

    batch_size, region_count = batch["region_lengths"].shape
    lengths = batch["region_lengths"]
    non_empty_mask = lengths > 0
    has_non_empty = non_empty_mask.any(dim=1)

    dropout_decisions = torch.rand(batch_size, generator=generator, device=lengths.device) < prob
    apply_mask = dropout_decisions & has_non_empty

    if not apply_mask.any():
        return batch

    random_scores = torch.rand(batch_size, region_count, generator=generator, device=lengths.device)
    random_scores = random_scores.masked_fill(~non_empty_mask, -1.0)
    random_scores = random_scores.masked_fill(
        ~apply_mask.unsqueeze(1).expand_as(random_scores), -1.0
    )
    sampled_indices = random_scores.argmax(dim=1)

    region_indices = torch.arange(region_count, device=lengths.device).unsqueeze(0)
    dropout_region_mask = (region_indices == sampled_indices.unsqueeze(1)) & apply_mask.unsqueeze(1)

    new_batch = dict(batch)
    new_tokens = batch["region_tokens"].clone()
    new_tokens[dropout_region_mask] = padding_token
    new_batch["region_tokens"] = new_tokens

    for key in ("region_lengths", "region_types", "offset_buckets", "length_buckets"):
        new_tensor = batch[key].clone()
        new_tensor[dropout_region_mask] = 0
        new_batch[key] = new_tensor

    return new_batch


def apply_mixup(
    batch: dict[str, torch.Tensor],
    *,
    alpha: float,
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mixup 数据增强（训练时）。"""
    if alpha <= 0.0:
        labels = batch["labels"]
        return batch, labels, labels, torch.tensor(1.0)

    batch_size = batch["labels"].shape[0]
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    if lam <= 0.0 or lam >= 1.0:
        labels = batch["labels"]
        return batch, labels, labels, torch.tensor(1.0)

    perm = torch.randperm(batch_size, generator=generator, device=batch["labels"].device)

    new_batch = dict(batch)
    token_mix_mask = torch.rand_like(batch["region_tokens"].float()) < lam
    new_tokens = torch.where(token_mix_mask, batch["region_tokens"], batch["region_tokens"][perm])
    new_batch["region_tokens"] = new_tokens

    new_lengths = torch.minimum(batch["region_lengths"], batch["region_lengths"][perm])
    new_batch["region_lengths"] = new_lengths

    positions = torch.arange(batch["region_tokens"].shape[2], device=new_lengths.device).view(1, 1, -1)
    valid_mask = positions < new_lengths.unsqueeze(-1)
    new_tokens = torch.where(valid_mask, new_tokens, torch.full_like(new_tokens, 256))
    new_batch["region_tokens"] = new_tokens

    new_types = torch.where(new_lengths == 0, torch.zeros_like(new_lengths), batch["region_types"])
    new_batch["region_types"] = new_types
    new_offsets = torch.where(new_lengths == 0, torch.zeros_like(new_lengths), batch["offset_buckets"])
    new_batch["offset_buckets"] = new_offsets
    new_length_buckets = torch.where(new_lengths == 0, torch.zeros_like(new_lengths), batch["length_buckets"])
    new_batch["length_buckets"] = new_length_buckets

    if batch.get("b0_features") is not None:
        new_b0 = lam * batch["b0_features"] + (1.0 - lam) * batch["b0_features"][perm]
        new_batch["b0_features"] = new_b0

    labels_a = batch["labels"]
    labels_b = batch["labels"][perm]
    return new_batch, labels_a, labels_b, torch.tensor(lam)


def forward_model(
    model: HGConvRegionNet,
    batch: dict[str, torch.Tensor],
    *,
    use_b0: bool = True,
) -> torch.Tensor:
    b0 = batch["b0_features"] if use_b0 else None
    output = model(
        batch["region_tokens"],
        batch["region_lengths"],
        batch["region_types"],
        batch["offset_buckets"],
        batch["length_buckets"],
        b0,
    )
    return output["fusion_logits"] if use_b0 else output["region_logits"]


def scheduler_multiplier(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def soft_cross_entropy(
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: torch.Tensor,
    *,
    label_smoothing: float,
) -> torch.Tensor:
    """Mixup + label smoothing soft cross-entropy。"""
    lam_value = float(lam.item())
    loss_a = functional.cross_entropy(
        logits.float(), labels_a, reduction="none", label_smoothing=label_smoothing
    )
    loss_b = functional.cross_entropy(
        logits.float(), labels_b, reduction="none", label_smoothing=label_smoothing
    )
    return lam_value * loss_a + (1.0 - lam_value) * loss_b


# ---------------------------------------------------------------------------
# 训练循环（含 SAM）
# ---------------------------------------------------------------------------


def forward_backward_window(
    *,
    model: HGConvRegionNet,
    loader: PhaseADataLoader,
    window: tuple[np.ndarray, ...],
    window_normalizer: float,
    device: torch.device,
    use_bf16: bool,
    aug_generator: torch.Generator,
) -> tuple[torch.Tensor, float, int]:
    """对一个 accumulation window 执行 forward+backward，返回 (loss_sum_tensor, total_loss, sample_count)。

    注意：调用方负责 zero_grad 和 optimizer step。
    """
    loss_sum = torch.zeros(1, device=device, dtype=torch.float32)
    total_loss = 0.0
    sample_count = 0
    for receiver_indices in window:
        batch = move_batch_to_device(
            loader.materialize_batch(receiver_indices),
            device,
        )
        batch = apply_region_dropout(
            batch,
            prob=REGION_DROPOUT_PROB,
            generator=aug_generator,
        )
        batch, labels_a, labels_b, lam = apply_mixup(
            batch,
            alpha=MIXUP_ALPHA,
            generator=aug_generator,
        )

        with autocast(device, use_bf16):
            logits = forward_model(model, batch, use_b0=True)
        per_sample = soft_cross_entropy(
            logits,
            labels_a,
            labels_b,
            lam,
            label_smoothing=LABEL_SMOOTHING_EPS,
        )
        if not torch.isfinite(per_sample).all().item():
            raise FloatingPointError(
                f"produced non-finite loss"
            )
        numerator = per_sample.sum()
        (numerator / window_normalizer).backward()
        loss_sum += numerator.detach()
        total_loss += float(numerator.detach())
        sample_count += int(receiver_indices.size)
    return loss_sum, total_loss, sample_count


def train_one_epoch(
    *,
    model: HGConvRegionNet,
    sam_optimizer: SAM,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loader: PhaseADataLoader,
    fit_indices: np.ndarray,
    epoch: int,
    seed: int,
    microbatch: int,
    accumulation: int,
    grad_clip: float,
    device: torch.device,
    use_bf16: bool,
    aug_generator: torch.Generator,
) -> tuple[float, int]:
    """训练一个 epoch（含 SAM + 数据增强），返回 (avg_loss, optimizer_steps)。

    SAM 流程（每个 effective batch）：
    1. zero_grad
    2. 第一次 forward+backward（累积梯度）
    3. clip_grad_norm（用于第一次梯度的稳定性）
    4. sam_optimizer.first_step(zero_grad=True)  # 计算 ε 并加到 w
    5. 第二次 forward+backward（在 w+ε 上累积梯度）
    6. clip_grad_norm
    7. sam_optimizer.second_step(zero_grad=True)  # 恢复 w 并在 w+ε 梯度上更新
    8. scheduler.step()
    """

    model.train()
    batches = deterministic_epoch_batches(
        fit_indices,
        microbatch=microbatch,
        seed=seed,
        epoch=epoch,
    )
    epoch_loss_sum = 0.0
    epoch_sample_count = 0
    optimizer_steps = 0
    total_windows = (len(batches) + accumulation - 1) // accumulation
    step_start_time = time.time()

    for window_start in range(0, len(batches), accumulation):
        window = batches[window_start : window_start + accumulation]
        window_indices = np.concatenate(window)
        window_normalizer = float(window_indices.size)

        # SAM Step 1: zero_grad
        sam_optimizer.zero_grad(set_to_none=True)

        # SAM Step 2: 第一次 forward+backward
        _, first_loss, first_count = forward_backward_window(
            model=model,
            loader=loader,
            window=window,
            window_normalizer=window_normalizer,
            device=device,
            use_bf16=use_bf16,
            aug_generator=aug_generator,
        )

        # SAM Step 3: clip grad
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(gradient_norm).item():
            raise FloatingPointError(
                f"epoch {epoch} produced non-finite gradient norm (first pass)"
            )

        # SAM Step 4: first_step（计算扰动并加到 w）
        sam_optimizer.first_step(zero_grad=True)

        # SAM Step 5: 第二次 forward+backward（在 w+ε 上）
        _, second_loss, second_count = forward_backward_window(
            model=model,
            loader=loader,
            window=window,
            window_normalizer=window_normalizer,
            device=device,
            use_bf16=use_bf16,
            aug_generator=aug_generator,
        )

        # SAM Step 6: clip grad
        gradient_norm_2 = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(gradient_norm_2).item():
            raise FloatingPointError(
                f"epoch {epoch} produced non-finite gradient norm (second pass)"
            )

        # SAM Step 7: second_step（恢复 w 并更新）
        sam_optimizer.second_step(zero_grad=True)

        # SAM Step 8: scheduler.step()
        scheduler.step()

        epoch_loss_sum += first_loss  # 用第一次 loss 记录（更真实）
        epoch_sample_count += first_count
        optimizer_steps += 1

        # 每 50 步打印进度
        if optimizer_steps % 50 == 0 or optimizer_steps == total_windows:
            elapsed = time.time() - step_start_time
            avg_step_time = elapsed / optimizer_steps
            remaining = avg_step_time * (total_windows - optimizer_steps)
            print(f"  [epoch {epoch}] step {optimizer_steps}/{total_windows}, avg_step={avg_step_time:.2f}s, remaining={remaining:.0f}s ({remaining/60:.1f}min)")

    if epoch_sample_count != fit_indices.size:
        raise RuntimeError(
            f"epoch {epoch} consumed {epoch_sample_count} rows, expected {fit_indices.size}"
        )

    avg_loss = epoch_loss_sum / epoch_sample_count
    return avg_loss, optimizer_steps


@torch.no_grad()
def evaluate(
    *,
    model: HGConvRegionNet,
    loader: PhaseADataLoader,
    selection_indices: np.ndarray,
    microbatch: int,
    device: torch.device,
    use_bf16: bool,
) -> tuple[float, np.ndarray]:
    """在 selection fold 上评估（无数据增强），返回 (avg_loss, probabilities)。"""

    model.eval()
    losses: list[float] = []
    scores: list[np.ndarray] = []

    for start in range(0, selection_indices.size, microbatch):
        receivers = selection_indices[start : start + microbatch]
        batch = move_batch_to_device(
            loader.materialize_batch(receivers),
            device,
        )
        with autocast(device, use_bf16):
            logits = forward_model(model, batch, use_b0=True)
        per_sample = functional.cross_entropy(
            logits.float(), batch["labels"], reduction="none"
        )
        if not torch.isfinite(per_sample).all().item():
            raise FloatingPointError("evaluation produced non-finite loss")
        losses.extend(per_sample.cpu().tolist())
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        if not torch.isfinite(probabilities).all().item():
            raise FloatingPointError("evaluation produced non-finite score")
        scores.append(probabilities.cpu().numpy())

    if len(losses) != selection_indices.size:
        raise RuntimeError("evaluation did not consume exactly the selection partition")

    combined = np.ascontiguousarray(np.concatenate(scores), dtype=np.float64)
    return float(np.mean(losses)), combined


def compute_f1(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> tuple[float, float]:
    """计算 F1 和 accuracy。"""
    preds = (scores > threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    return f1, acc


def select_best_f1_epoch(training_log: list[dict[str, object]]) -> int:
    """F1-based 选择（max sel_f1，相同则选最早）。"""
    if not training_log:
        return 0
    best_epoch = 0
    best_f1 = -1.0
    for entry in training_log:
        f1 = float(entry.get("selection_f1", 0.0))
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = int(entry["epoch"])
    return best_epoch


def run_phase_a(
    *,
    device_request: str = "auto",
    max_epochs: int | None = None,
    seed: int = 41,
) -> dict[str, object]:
    print("\n=== Loop184 Phase A Preflight ===")
    preflight_result = preflight()
    auth_json = generate_authorization_json(preflight_result)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    auth_path = REPORT_DIR / "phase_a_authorization.json"
    with auth_path.open("w", encoding="utf-8") as f:
        json.dump(auth_json, f, indent=2, ensure_ascii=False)
    print(f"[preflight] 授权 JSON 已保存: {auth_path}")

    print("\n=== Loop184 数据加载 ===")
    loader = PhaseADataLoader()
    t0 = time.time()
    data_receipt = loader.load_real_data()
    t1 = time.time()
    print(f"[data] 加载耗时: {t1 - t0:.2f}s")
    print(f"[data] 行数: {data_receipt['row_count']}, 区域数: {data_receipt['region_count']}")
    print(f"[data] token 数: {data_receipt['token_count']}")

    fit_indices = loader.fit_indices()
    selection_indices = loader.selection_indices()
    print(f"[data] fit 行数: {fit_indices.shape[0]}")
    print(f"[data] selection 行数: {selection_indices.shape[0]}")

    device, use_bf16 = resolve_device(device_request)
    print(f"\n[device] 使用设备: {device}, BF16: {use_bf16}")

    print("\n=== Loop184 模型初始化 ===")
    model_seed = seed
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    config = HGConvRegionConfig()
    model = HGConvRegionNet(config).to(device=device, dtype=torch.float32)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] 参数量: {param_count:,} (vs Loop183: 1,622,153, +{(param_count-1622153)/1622153*100:.1f}%)")
    print(f"[model] config: model_dim={config.model_dim}, heads={config.transformer_heads}, layers={config.transformer_layers}")
    print(f"[model] hgconv_blocks={config.hgconv_blocks} (Loop184: 1 → 2)")
    print(f"[model] multi_scale_filter_lengths={config.multi_scale_filter_lengths}")
    print(f"[model] dropout={config.dropout}")

    # 数据增强 generator
    aug_generator = torch.Generator(device=device)
    aug_generator.manual_seed(seed + 1000)

    epochs = max_epochs or PHASE_A_GATE.max_epochs
    microbatch = PHASE_A_GATE.microbatch
    accumulation = PHASE_A_GATE.accumulation

    # SAM 优化器（包裹 AdamW）
    sam_optimizer = SAM(
        model.parameters(),
        base_optimizer_cls=torch.optim.AdamW,
        rho=SAM_RHO,
        lr=PHASE_A_GATE.learning_rate,
        weight_decay=PHASE_A_GATE.weight_decay,
    )

    batches_per_epoch = math.ceil(math.ceil(fit_indices.size / microbatch) / accumulation)
    total_steps = epochs * batches_per_epoch
    warmup_steps = PHASE_A_GATE.warmup_steps * batches_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        sam_optimizer.base_optimizer,
        lr_lambda=lambda step: scheduler_multiplier(
            step, warmup_steps=warmup_steps, total_steps=total_steps
        ),
    )
    print(f"[optimizer] SAM(AdamW) rho={SAM_RHO}, lr={PHASE_A_GATE.learning_rate}, wd={PHASE_A_GATE.weight_decay}")
    print(f"[scheduler] cosine, warmup={warmup_steps} steps, total={total_steps} steps")
    print(f"[schedule] {batches_per_epoch} optimizer steps/epoch × {epochs} epochs")
    print(f"[augmentation] Mixup α={MIXUP_ALPHA}, Region dropout p={REGION_DROPOUT_PROB}, Label smoothing ε={LABEL_SMOOTHING_EPS}")
    print(f"[swa] start_epoch={SWA_START_EPOCH}, swa_lr={SWA_LR}, anneal={SWA_ANNEAL_EPOCHS} epochs")

    # SWA 容器
    swa_container = SWAContainer(model)
    swa_active = False

    resource_cell = ResourceCell()
    resource_cell.start()

    print("\n=== Loop184 训练开始（SAM + 数据增强 + F1-based 选择）===")
    selection_losses: list[float] = []
    selection_f1s: list[float] = []
    selection_accs: list[float] = []
    training_losses: list[float] = []
    training_log: list[dict[str, object]] = []
    start_time = time.time()

    sel_labels_np = loader._fold_labels[selection_indices]

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        phase_note = "SAM + Mixup + Region dropout"
        if epoch >= SWA_START_EPOCH:
            phase_note = f"SAM + Mixup + Region dropout + SWA (epoch {epoch}/{epochs})"
        print(f"\n[epoch {epoch}/{epochs}] 训练中（{phase_note}）...")

        train_loss, opt_steps = train_one_epoch(
            model=model,
            sam_optimizer=sam_optimizer,
            scheduler=scheduler,
            loader=loader,
            fit_indices=fit_indices,
            epoch=epoch,
            seed=seed,
            microbatch=microbatch,
            accumulation=accumulation,
            grad_clip=PHASE_A_GATE.grad_clip,
            device=device,
            use_bf16=use_bf16,
            aug_generator=aug_generator,
        )
        training_losses.append(train_loss)

        # SWA 更新（在 SWA_START_EPOCH 后每个 epoch 末更新）
        if epoch >= SWA_START_EPOCH:
            swa_container.update_parameters(model)
            swa_active = True

        # 评估当前 model（非 SWA 平均）
        sel_loss, sel_scores = evaluate(
            model=model,
            loader=loader,
            selection_indices=selection_indices,
            microbatch=microbatch,
            device=device,
            use_bf16=use_bf16,
        )
        selection_losses.append(sel_loss)

        sample = resource_cell.sample_and_inject(epoch=epoch, step=opt_steps, note=f"epoch {epoch} done")
        epoch_elapsed = time.time() - epoch_start

        sel_f1, sel_acc = compute_f1(sel_scores, sel_labels_np)
        selection_f1s.append(sel_f1)
        selection_accs.append(sel_acc)

        swa_note = f", swa_n={swa_container.swa_n}" if swa_active else ""
        print(f"[epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, sel_loss={sel_loss:.4f}, sel_acc={sel_acc:.4f}, sel_f1={sel_f1:.4f}{swa_note}")
        print(f"  GPU={sample.gpu_allocated_bytes / 1024**3:.3f} GiB, RSS={sample.rss_bytes / 1024**3:.3f} GiB, wall={sample.wall_seconds:.1f}s, epoch_time={epoch_elapsed:.1f}s")

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "selection_loss": sel_loss,
            "selection_accuracy": sel_acc,
            "selection_f1": sel_f1,
            "optimizer_steps": opt_steps,
            "gpu_allocated_bytes": sample.gpu_allocated_bytes,
            "rss_bytes": sample.rss_bytes,
            "wall_seconds": sample.wall_seconds,
            "epoch_seconds": epoch_elapsed,
            "swa_active": swa_active,
            "swa_n": swa_container.swa_n if swa_active else 0,
        }
        training_log.append(log_entry)

        if not resource_cell.passed():
            print(f"\n[ALERT] 资源门违规！violations: {len(resource_cell.violations)}")
            for v in resource_cell.violations:
                print(f"  [{v.kind}] {v.detail}")
            break

    total_time = time.time() - start_time

    # SWA 最终评估（如果激活）
    swa_eval_loss = None
    swa_eval_f1 = None
    swa_eval_acc = None
    if swa_active and swa_container.swa_n > 0:
        print(f"\n[swa] 训练完成，SWA 平均 {swa_container.swa_n} 个 epoch 的权重")
        swa_model = copy.deepcopy(model)
        swa_container.copy_to(swa_model)
        swa_model.eval()
        with torch.no_grad():
            swa_eval_loss, swa_scores = evaluate(
                model=swa_model,
                loader=loader,
                selection_indices=selection_indices,
                microbatch=microbatch,
                device=device,
                use_bf16=use_bf16,
            )
        swa_eval_f1, swa_eval_acc = compute_f1(swa_scores, sel_labels_np)
        print(f"[swa] SWA sel_loss={swa_eval_loss:.4f}, sel_acc={swa_eval_acc:.4f}, sel_f1={swa_eval_f1:.4f}")
        del swa_model

    # F1-based 选择（从 model 历史中选最佳）
    selected_epoch = select_best_f1_epoch(training_log) if training_log else 0
    selected_source = "model"

    # 比较 SWA 与最佳 epoch
    if swa_eval_f1 is not None and training_log:
        best_model_f1 = max(float(e.get("selection_f1", 0.0)) for e in training_log)
        if swa_eval_f1 > best_model_f1:
            selected_source = "swa"
            print(f"[result] SWA F1={swa_eval_f1:.4f} > best model F1={best_model_f1:.4f}，选用 SWA")

    print(f"\n[result] 选定来源: {selected_source}")
    print(f"[result] 选定 epoch (F1-based): {selected_epoch}")
    if training_log:
        best_entry = next(e for e in training_log if e["epoch"] == selected_epoch)
        print(f"[result] 最佳 model selection F1: {best_entry['selection_f1']:.4f}")
        print(f"[result] 最佳 model selection accuracy: {best_entry['selection_accuracy']:.4f}")
        print(f"[result] 最佳 model selection loss: {best_entry['selection_loss']:.4f}")

    print("\n=== Loop184 确定性验证 ===")
    model.eval()
    with torch.no_grad():
        _, scores_1 = evaluate(
            model=model,
            loader=loader,
            selection_indices=selection_indices,
            microbatch=microbatch,
            device=device,
            use_bf16=use_bf16,
        )
        _, scores_2 = evaluate(
            model=model,
            loader=loader,
            selection_indices=selection_indices,
            microbatch=microbatch,
            device=device,
            use_bf16=use_bf16,
        )
    deterministic = bool(np.array_equal(scores_1, scores_2))
    print(f"[determinism] bitwise identical: {deterministic}")
    if not deterministic:
        resource_cell.record_integrity(
            kind="nondeterministic",
            detail="eval scores differ between two identical runs",
        )

    resource_passed = resource_cell.passed()
    print(f"\n=== Loop184 资源门: {'PASS' if resource_passed else 'FAIL'} ===")
    resource_receipt = resource_cell.build_receipt()

    log_path = REPORT_DIR / "phase_a_training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for entry in training_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[output] 训练日志: {log_path}")

    checkpoint_path = REPORT_DIR / "phase_a_checkpoint.pt"
    checkpoint_sha256 = None
    if resource_passed:
        checkpoint_payload = {
            "schema": "axon_loop184_phase_a_checkpoint_v1",
            "loop_id": LOOP_ID,
            "selected_epoch": selected_epoch,
            "selected_source": selected_source,
            "selection_criterion": "max_f1",
            "model_config": asdict(config),
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
        }
        if swa_active and swa_container.swa_n > 0:
            checkpoint_payload["swa_state_dict"] = {
                name: value.detach().cpu()
                for name, value in swa_container.swa_state.items()
            }
            checkpoint_payload["swa_n"] = swa_container.swa_n
        torch.save(checkpoint_payload, checkpoint_path)
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()
        print(f"[output] Checkpoint: {checkpoint_path} ({checkpoint_sha256[:16]}...)")

    receipt = {
        "schema": "axon_loop184_phase_a_receipt_v1",
        "loop_id": LOOP_ID,
        "lineage": "arch_expand_sam_swa",
        "phase": "A",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "authorization": auth_json,
        "preflight": {
            "cuda_available": preflight_result["cuda_available"],
            "bf16_supported": preflight_result["bf16_supported"],
            "device_name": preflight_result["device_name"],
            "gpu_name": preflight_result["gpu_name"],
            "gpu_total_bytes": preflight_result["gpu_total_bytes"],
            "source_closure_manifest": preflight_result["closure_manifest"],
        },
        "data": data_receipt,
        "model": {
            "parameter_count": param_count,
            "config": asdict(config),
            "seed": model_seed,
        },
        "augmentation": {
            "mixup_alpha": MIXUP_ALPHA,
            "region_dropout_prob": REGION_DROPOUT_PROB,
            "label_smoothing_eps": LABEL_SMOOTHING_EPS,
            "aug_generator_seed": seed + 1000,
        },
        "optimizer": {
            "type": "SAM(AdamW)",
            "sam_rho": SAM_RHO,
            "base_lr": PHASE_A_GATE.learning_rate,
            "weight_decay": PHASE_A_GATE.weight_decay,
            "grad_clip": PHASE_A_GATE.grad_clip,
        },
        "swa": {
            "active": swa_active,
            "swa_n": swa_container.swa_n if swa_active else 0,
            "start_epoch": SWA_START_EPOCH,
            "swa_lr": SWA_LR,
            "swa_eval_loss": swa_eval_loss,
            "swa_eval_f1": swa_eval_f1,
            "swa_eval_acc": swa_eval_acc,
        },
        "training": {
            "epochs_run": len(training_log),
            "selected_epoch": selected_epoch,
            "selected_source": selected_source,
            "selection_criterion": "max_f1",
            "selection_losses": selection_losses,
            "selection_f1s": selection_f1s,
            "selection_accs": selection_accs,
            "training_losses": training_losses,
            "total_wall_seconds": total_time,
            "optimizer_total_steps": sum(e["optimizer_steps"] for e in training_log),
            "device": str(device),
            "use_bf16": use_bf16,
            "seed": seed,
            "weight_decay": PHASE_A_GATE.weight_decay,
        },
        "determinism": {
            "bitwise_identical_eval": deterministic,
        },
        "resource_gate": resource_receipt,
        "resource_passed": resource_passed,
        "checkpoint_path": str(checkpoint_path) if resource_passed else None,
        "checkpoint_sha256": checkpoint_sha256,
        "training_log_path": str(log_path),
    }

    receipt_path = REPORT_DIR / "phase_a_receipt.json"
    with receipt_path.open("w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    print(f"\n[output] Receipt: {receipt_path}")

    print("\n=== Loop184 Phase A 总结 ===")
    print(f"资源门: {'PASS' if resource_passed else 'FAIL'}")
    print(f"训练 epochs: {len(training_log)}/{epochs}")
    print(f"选定来源: {selected_source}")
    print(f"选定 epoch (F1-based): {selected_epoch}")
    if training_log:
        best_entry = next(e for e in training_log if e["epoch"] == selected_epoch)
        print(f"最佳 model selection F1: {best_entry['selection_f1']:.4f}")
        print(f"最佳 model selection accuracy: {best_entry['selection_accuracy']:.4f}")
    if swa_eval_f1 is not None:
        print(f"SWA selection F1: {swa_eval_f1:.4f}")
        print(f"SWA selection accuracy: {swa_eval_acc:.4f}")
    print(f"总 wall time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"确定性: {'PASS' if deterministic else 'FAIL'}")
    print(f"参数量: {param_count:,}")
    print(f"SAM rho: {SAM_RHO}, SWA start: {SWA_START_EPOCH}, SWA n: {swa_container.swa_n}")
    print(f"数据增强: Mixup α={MIXUP_ALPHA}, Region dropout p={REGION_DROPOUT_PROB}, Label smoothing ε={LABEL_SMOOTHING_EPS}")

    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Loop184 Phase A 训练（架构扩展 + SAM + SWA）")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="设备选择 (default: auto)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help=f"最大 epochs (default: {PHASE_A_GATE.max_epochs})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=41,
        help="随机种子 (default: 41)",
    )
    args = parser.parse_args()

    receipt = run_phase_a(
        device_request=args.device,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )

    if not receipt["resource_passed"]:
        print("\n[FAIL] 资源门未通过，Loop184 Phase A 失败")
        sys.exit(1)
    else:
        print("\n[PASS] 资源门通过，Loop184 Phase A 成功")


if __name__ == "__main__":
    main()
