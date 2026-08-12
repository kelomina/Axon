"""Loop186 Phase A 训练脚本：架构扩容 + 单 fold 深度训练 + SAM + SWA。

Loop186 = Loop185 架构扩容 + 单 fold 深度训练 + SAM：
1. 架构扩容（vs Loop185）：
   - model_dim 192 → 384, hgconv_blocks 2 → 4, transformer_layers 4 → 8
   - 参数量 2.61M → 17.3M
2. 单 fold 深度训练（放弃 4-fold OOF）：
   - fit on fold 2,3,4 (12000 rows), select on fold 1 (4000 rows)
3. 重新启用 SAM（rho=0.05，sharpness reduction 对硬样本有益）
4. 12 epochs + SWA（start_epoch=9, average last 3 epochs）
5. 保持 Loop183 数据增强：Mixup α=0.4, Region dropout p=0.3, Label smoothing ε=0.05

资源估算：
- 参数量 17.3M
- SAM 2x 开销 + 12 epochs ~3.75h（在 6h 预算内）

用法:
    python scripts/run_loop186_phase_a.py
    python scripts/run_loop186_phase_a.py --device cuda
    python scripts/run_loop186_phase_a.py --device cpu --max-epochs 3
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

from src.loop186 import (
    PHASE_A_GATE,
    HGConvRegionConfig,
    HGConvRegionNet,
    MULTI_SCALE_FILTER_LENGTHS,
    MIXUP_ALPHA,
    REGION_DROPOUT_PROB,
    LABEL_SMOOTHING_EPS,
    SAM_RHO,
    SAM_ENABLED,
    SWA_START_EPOCH,
    SWA_LR,
    SWA_ANNEAL_EPOCHS,
    PhaseADataLoader,
    ResourceCell,
    assert_contract_invariants,
    assert_phase0_closure,
    make_fold_split,
)
from src.loop186.contracts import LOOP_ID
from src.loop186.data_adapter import FULL_TRAIN_ROWS, ROWS_PER_FOLD
from src.loop186.resource_cell import deadline_check_due, enforce_epoch_deadline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "roadmap_9997" / "loop186"

REGEN_REGION_CACHE_PATH = "D:/axon_loop185_region_cache/phase_b_region_cache_v1.npz"
REGEN_REGION_CACHE_SHA256 = "99b1fe0724a985ee3fb91bb5b469af71935a39cee8a93141efcca1f00d310829"


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
    """Stochastic Weight Averaging 容器。"""

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
    print("[preflight] Loop186 契约自检...")
    assert_contract_invariants()

    print("[preflight] Loop186 源码闭包检查...")
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
    print(f"[preflight] Loop186 参数量: {probe_params:,} (扩容版本)")
    del probe_model, probe_config

    print(f"[preflight] 数据增强: Mixup α={MIXUP_ALPHA}, Region dropout p={REGION_DROPOUT_PROB}, Label smoothing ε={LABEL_SMOOTHING_EPS}")
    print(f"[preflight] SAM: rho={SAM_RHO}, enabled={SAM_ENABLED}")
    print(f"[preflight] SWA: start_epoch={SWA_START_EPOCH}, swa_lr={SWA_LR}, anneal={SWA_ANNEAL_EPOCHS} epochs")
    print(f"[preflight] 单 fold: fit on fold 2,3,4, select on fold 1")

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
        "schema": "axon_loop186_phase_a_authorization_v1",
        "loop_id": LOOP_ID,
        "lineage": "expanded_single_fold_sam",
        "phase": "A",
        "decision": "loop186_authorized_by_user_2026_07_23_aggressive_research",
        "claim_scope": "expanded_single_fold_sam",
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
        "sam_enabled": False,
        "sam_rho": SAM_RHO,
        "expanded": True,
        "swa_start_epoch": SWA_START_EPOCH,
        "swa_lr": SWA_LR,
        "swa_anneal_epochs": SWA_ANNEAL_EPOCHS,
        "transformer_layers": 8,
        "hgconv_blocks": 4,
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
    fold_id: int,
) -> tuple[np.ndarray, ...]:
    material = f"loop186-phase-a|fold{fold_id}|{seed}|{epoch}|{fit_indices.size}".encode("ascii")
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
            raise RuntimeError("Loop186 CUDA requires BF16 support")
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
# 数据增强（与 Loop184 一致）
# ---------------------------------------------------------------------------


def apply_region_dropout(
    batch: dict[str, torch.Tensor],
    *,
    prob: float,
    generator: torch.Generator,
    padding_token: int = 256,
) -> dict[str, torch.Tensor]:
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
    lam_value = float(lam.item())
    loss_a = functional.cross_entropy(
        logits.float(), labels_a, reduction="none", label_smoothing=label_smoothing
    )
    loss_b = functional.cross_entropy(
        logits.float(), labels_b, reduction="none", label_smoothing=label_smoothing
    )
    return lam_value * loss_a + (1.0 - lam_value) * loss_b


# ---------------------------------------------------------------------------
# 训练循环（SAM，每个 effective batch 两次 forward+backward）
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
    loss_sum = torch.zeros(1, device=device, dtype=torch.float32)
    total_loss = 0.0
    sample_count = 0
    for receiver_indices in window:
        batch = move_batch_to_device(
            loader.materialize_batch(receiver_indices),
            device,
        )
        batch = apply_region_dropout(
            batch, prob=REGION_DROPOUT_PROB, generator=aug_generator
        )
        batch, labels_a, labels_b, lam = apply_mixup(
            batch, alpha=MIXUP_ALPHA, generator=aug_generator
        )
        with autocast(device, use_bf16):
            logits = forward_model(model, batch, use_b0=True)
            per_sample = soft_cross_entropy(
                logits, labels_a, labels_b, lam,
                label_smoothing=LABEL_SMOOTHING_EPS,
            )
        scaled = per_sample.sum() / window_normalizer
        scaled.backward()
        loss_sum += per_sample.detach().sum()
        total_loss += float(per_sample.detach().sum().item())
        sample_count += int(receiver_indices.size)
    return loss_sum, total_loss, sample_count


def train_one_epoch(
    *,
    model: HGConvRegionNet,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loader: PhaseADataLoader,
    fit_indices: np.ndarray,
    epoch: int,
    seed: int,
    fold_id: int,
    microbatch: int,
    accumulation: int,
    grad_clip: float,
    device: torch.device,
    use_bf16: bool,
    aug_generator: torch.Generator,
    sam_enabled: bool,
) -> tuple[float, int]:
    """训练一个 epoch（支持 SAM 或普通 AdamW）。"""

    model.train()
    batches = deterministic_epoch_batches(
        fit_indices,
        microbatch=microbatch,
        seed=seed,
        epoch=epoch,
        fold_id=fold_id,
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

        if sam_enabled:
            # === SAM 模式：两次 forward+backward ===
            # Step 1: zero_grad
            optimizer.zero_grad(set_to_none=True)
            # Step 2: 第一次 forward+backward
            _, total_loss, sample_count = forward_backward_window(
                model=model, loader=loader, window=window,
                window_normalizer=window_normalizer, device=device,
                use_bf16=use_bf16, aug_generator=aug_generator,
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(grad_norm).item():
                raise FloatingPointError(f"epoch {epoch} fold {fold_id} non-finite grad norm (SAM first pass)")
            # Step 3: first_step（扰动）
            optimizer.first_step(zero_grad=True)
            # Step 4: 第二次 forward+backward
            _, _, _ = forward_backward_window(
                model=model, loader=loader, window=window,
                window_normalizer=window_normalizer, device=device,
                use_bf16=use_bf16, aug_generator=aug_generator,
            )
            grad_norm_2 = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(grad_norm_2).item():
                raise FloatingPointError(f"epoch {epoch} fold {fold_id} non-finite grad norm (SAM second pass)")
            # Step 5: second_step（恢复 w 并更新）
            optimizer.second_step(zero_grad=True)
        else:
            # === 普通模式：一次 forward+backward ===
            optimizer.zero_grad(set_to_none=True)
            _, total_loss, sample_count = forward_backward_window(
                model=model, loader=loader, window=window,
                window_normalizer=window_normalizer, device=device,
                use_bf16=use_bf16, aug_generator=aug_generator,
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(grad_norm).item():
                raise FloatingPointError(f"epoch {epoch} fold {fold_id} non-finite grad norm")
            optimizer.step()

        scheduler.step()

        epoch_loss_sum += total_loss
        epoch_sample_count += sample_count
        optimizer_steps += 1

        if deadline_check_due(optimizer_steps):
            elapsed = time.time() - step_start_time
            enforce_epoch_deadline(
                elapsed_seconds=elapsed,
                completed_steps=optimizer_steps,
                total_steps=total_windows,
                hard_seconds=float(PHASE_A_GATE.epoch_wall_seconds),
                projection_seconds=float(PHASE_A_GATE.epoch_wall_seconds) * 0.9,
            )
            if optimizer_steps % 25 == 0 or optimizer_steps == total_windows:
                avg_step_time = elapsed / optimizer_steps
                remaining = avg_step_time * (total_windows - optimizer_steps)
                print(f"  [fold {fold_id} epoch {epoch}] step {optimizer_steps}/{total_windows}, avg_step={avg_step_time:.2f}s, remaining={remaining:.0f}s ({remaining/60:.1f}min)")

    if epoch_sample_count != fit_indices.size:
        raise RuntimeError(
            f"epoch {epoch} fold {fold_id} consumed {epoch_sample_count} rows, expected {fit_indices.size}"
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
    preds = (scores > threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    return f1, acc


def select_stratified_probe(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    rows: int,
    seed: int,
) -> np.ndarray:
    if rows <= 0 or rows > indices.size:
        raise ValueError("rows must be in [1, len(indices)]")
    selected_labels = labels[indices]
    classes, counts = np.unique(selected_labels, return_counts=True)
    if classes.size < 2:
        raise ValueError("selection probe requires at least two classes")
    allocations = np.floor(rows * counts / counts.sum()).astype(np.int64)
    while int(allocations.sum()) < rows:
        residuals = rows * counts / counts.sum() - allocations
        allocations[int(np.argmax(residuals))] += 1
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for label, allocation in zip(classes, allocations):
        candidates = indices[selected_labels == label].copy()
        rng.shuffle(candidates)
        parts.append(candidates[: int(allocation)])
    probe = np.concatenate(parts)
    rng.shuffle(probe)
    return np.ascontiguousarray(probe, dtype=np.int64)


def select_rotating_stratified_probe(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    rows: int,
    seed: int,
    epoch: int,
) -> np.ndarray:
    if rows <= 0 or rows > indices.size:
        raise ValueError("rows must be in [1, len(indices)]")
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    selected_labels = labels[indices]
    classes, counts = np.unique(selected_labels, return_counts=True)
    if classes.size < 2:
        raise ValueError("selection probe requires at least two classes")
    rng = np.random.default_rng(seed)
    ordered_parts: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    for label, count in zip(classes, counts):
        candidates = indices[selected_labels == label].copy()
        rng.shuffle(candidates)
        ordered_parts.append(candidates)
        positions.append((np.arange(int(count), dtype=np.float64) + 0.5) / int(count))
    ordered = np.concatenate(ordered_parts)
    class_order = np.concatenate([
        np.full(int(count), class_index, dtype=np.int64)
        for class_index, count in enumerate(counts)
    ])
    stratified_order = ordered[np.lexsort((class_order, np.concatenate(positions)))]
    start = ((epoch - 1) * rows) % indices.size
    offsets = (start + np.arange(rows, dtype=np.int64)) % indices.size
    return np.ascontiguousarray(stratified_order[offsets], dtype=np.int64)


def select_best_f1_epoch(training_log: list[dict[str, object]]) -> int:
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


# ---------------------------------------------------------------------------
# 单 fold 训练
# ---------------------------------------------------------------------------


def train_one_fold(
    *,
    fold_id: int,
    fit_folds: tuple[int, ...],
    selection_fold: int,
    device: torch.device,
    use_bf16: bool,
    max_epochs: int,
    seed: int,
    resource_cell: ResourceCell,
    project_root: Path,
    region_cache_path: str | Path | None = REGEN_REGION_CACHE_PATH,
    region_cache_sha256: str | None = REGEN_REGION_CACHE_SHA256,
    sam_enabled: bool = True,
) -> dict[str, object]:
    """训练单个 fold，返回该 fold 的 scores 和训练日志。"""

    print(f"\n{'='*60}")
    print(f"=== Loop186 训练 fold {fold_id} (fit={fit_folds}, selection={selection_fold}) ===")
    print(f"{'='*60}")

    fold_split = make_fold_split(selection_fold)
    loader = PhaseADataLoader(fold_split=fold_split)
    load_kwargs: dict[str, object] = {"project_root": project_root}
    if region_cache_path is not None:
        load_kwargs["region_cache_path"] = region_cache_path
    if region_cache_sha256 is not None:
        load_kwargs["region_cache_sha256"] = region_cache_sha256
    data_receipt = loader.load_real_data(**load_kwargs)
    print(f"[fold {fold_id}] 数据加载完成: fit={data_receipt['fit_folds']}, selection={data_receipt['selection_fold']}")

    print(f"[fold {fold_id}] 预物化训练数据（消除逐行循环瓶颈）...")
    t_premat = time.time()
    loader.prematerialize_all()
    print(f"[fold {fold_id}] 预物化完成: {time.time() - t_premat:.1f}s")

    fit_indices = loader.fit_indices()
    selection_indices = loader.selection_indices()
    print(f"[fold {fold_id}] fit 行数: {fit_indices.shape[0]}, selection 行数: {selection_indices.shape[0]}")

    # 模型初始化（每 fold 独立种子）
    model_seed = seed + fold_id * 100
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    config = HGConvRegionConfig(runtime_checks=False)
    model = HGConvRegionNet(config).to(device=device, dtype=torch.float32)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[fold {fold_id}] 参数量: {param_count:,}")

    aug_generator = torch.Generator(device=device)
    aug_generator.manual_seed(seed + 1000 + fold_id * 100)

    epochs = max_epochs
    microbatch = PHASE_A_GATE.microbatch
    accumulation = PHASE_A_GATE.accumulation
    use_sam = sam_enabled and SAM_ENABLED

    if use_sam:
        # SAM 优化器
        optimizer = SAM(
            model.parameters(),
            torch.optim.AdamW,
            rho=SAM_RHO,
            lr=PHASE_A_GATE.learning_rate,
            weight_decay=PHASE_A_GATE.weight_decay,
        )
        optimizer_ref = optimizer
        opt_type = "SAM + AdamW"
    else:
        # 普通 AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=PHASE_A_GATE.learning_rate,
            weight_decay=PHASE_A_GATE.weight_decay,
        )
        optimizer_ref = optimizer
        opt_type = "AdamW"

    batches_per_epoch = math.ceil(math.ceil(fit_indices.size / microbatch) / accumulation)
    total_steps = epochs * batches_per_epoch
    warmup_steps = PHASE_A_GATE.warmup_steps * batches_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer_ref,
        lr_lambda=lambda step: scheduler_multiplier(
            step, warmup_steps=warmup_steps, total_steps=total_steps
        ),
    )
    print(f"[fold {fold_id}] {opt_type} lr={PHASE_A_GATE.learning_rate}, wd={PHASE_A_GATE.weight_decay}")
    print(f"[fold {fold_id}] cosine warmup={warmup_steps} steps, total={total_steps} steps")
    print(f"[fold {fold_id}] {batches_per_epoch} steps/epoch × {epochs} epochs")

    swa_container = SWAContainer(model)
    swa_active = False

    training_log: list[dict[str, object]] = []
    selection_losses: list[float] = []
    selection_f1s: list[float] = []
    selection_accs: list[float] = []
    training_losses: list[float] = []
    selection_labels_np = loader._fold_labels[selection_indices]
    best_model_state: dict[str, torch.Tensor] | None = None
    best_probe_f1 = -1.0

    fold_start_time = time.time()
    resource_cell.sample_and_inject(epoch=0, note=f"fold {fold_id} training baseline")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        selection_probe_indices = select_rotating_stratified_probe(
            selection_indices,
            loader._fold_labels,
            rows=PHASE_A_GATE.selection_probe_rows,
            seed=seed + fold_id * 100,
            epoch=epoch,
        )
        selection_probe_labels_np = loader._fold_labels[selection_probe_indices]
        phase_note = f"{opt_type} + Mixup + Region dropout"
        if epoch >= SWA_START_EPOCH:
            phase_note = f"{opt_type} + Mixup + Region dropout + SWA (epoch {epoch}/{epochs})"
        print(f"\n[fold {fold_id} epoch {epoch}/{epochs}] 训练中（{phase_note}）...")

        train_loss, opt_steps = train_one_epoch(
            model=model,
            optimizer=optimizer_ref,
            scheduler=scheduler,
            loader=loader,
            fit_indices=fit_indices,
            epoch=epoch,
            seed=seed,
            fold_id=fold_id,
            microbatch=microbatch,
            accumulation=accumulation,
            grad_clip=PHASE_A_GATE.grad_clip,
            device=device,
            use_bf16=use_bf16,
            aug_generator=aug_generator,
            sam_enabled=use_sam,
        )
        training_losses.append(train_loss)

        if epoch >= SWA_START_EPOCH:
            swa_container.update_parameters(model)
            swa_active = True

        sel_loss, sel_scores = evaluate(
            model=model,
            loader=loader,
            selection_indices=selection_probe_indices,
            microbatch=PHASE_A_GATE.evaluation_microbatch,
            device=device,
            use_bf16=use_bf16,
        )
        selection_losses.append(sel_loss)

        sample = resource_cell.sample_and_inject(epoch=epoch, step=opt_steps, note=f"fold {fold_id} epoch {epoch}")
        epoch_elapsed = time.time() - epoch_start

        sel_f1, sel_acc = compute_f1(sel_scores, selection_probe_labels_np)
        if sel_f1 > best_probe_f1:
            best_probe_f1 = sel_f1
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        selection_f1s.append(sel_f1)
        selection_accs.append(sel_acc)

        swa_note = f", swa_n={swa_container.swa_n}" if swa_active else ""
        print(f"[fold {fold_id} epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, sel_loss={sel_loss:.4f}, sel_acc={sel_acc:.4f}, sel_f1={sel_f1:.4f}{swa_note}")
        print(f"  GPU={sample.gpu_allocated_bytes / 1024**3:.3f} GiB, RSS={sample.rss_bytes / 1024**3:.3f} GiB, wall={sample.wall_seconds:.1f}s, epoch_time={epoch_elapsed:.1f}s")

        log_entry = {
            "fold_id": fold_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "selection_loss": sel_loss,
            "selection_accuracy": sel_acc,
            "selection_f1": sel_f1,
            "selection_rows": int(selection_probe_indices.size),
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
            print(f"\n[ALERT] fold {fold_id} 资源门违规！violations: {len(resource_cell.violations)}")
            for v in resource_cell.violations:
                print(f"  [{v.kind}] {v.detail}")
            break

    fold_time = time.time() - fold_start_time

    # SWA 最终评估
    swa_eval_loss = None
    swa_eval_f1 = None
    swa_eval_acc = None
    swa_scores = None
    if best_model_state is None:
        raise RuntimeError("training did not produce a selectable model state")
    model.load_state_dict(best_model_state)
    model.eval()
    best_model_eval_loss, best_model_scores = evaluate(
        model=model,
        loader=loader,
        selection_indices=selection_indices,
        microbatch=PHASE_A_GATE.evaluation_microbatch,
        device=device,
        use_bf16=use_bf16,
    )
    best_model_eval_f1, best_model_eval_acc = compute_f1(
        best_model_scores,
        selection_labels_np,
    )
    if swa_active and swa_container.swa_n > 0:
        print(f"\n[fold {fold_id}] SWA 平均 {swa_container.swa_n} 个 epoch 的权重")
        swa_model = copy.deepcopy(model)
        swa_container.copy_to(swa_model)
        swa_model.eval()
        with torch.no_grad():
            swa_eval_loss, swa_scores = evaluate(
                model=swa_model,
                loader=loader,
                selection_indices=selection_indices,
                microbatch=PHASE_A_GATE.evaluation_microbatch,
                device=device,
                use_bf16=use_bf16,
            )
        swa_eval_f1, swa_eval_acc = compute_f1(swa_scores, selection_labels_np)
        print(f"[fold {fold_id}] SWA sel_loss={swa_eval_loss:.4f}, sel_acc={swa_eval_acc:.4f}, sel_f1={swa_eval_f1:.4f}")
        del swa_model

    # F1-based 选择
    selected_epoch = select_best_f1_epoch(training_log) if training_log else 0
    selected_source = "model"
    selected_f1 = best_model_eval_f1
    selected_acc = best_model_eval_acc
    selected_loss = best_model_eval_loss

    if swa_eval_f1 is not None and training_log:
        if swa_eval_f1 > best_model_eval_f1:
            selected_source = "swa"
            selected_f1 = swa_eval_f1
            selected_acc = float(swa_eval_acc)
            selected_loss = float(swa_eval_loss)
            swa_container.copy_to(model)
            print(f"[fold {fold_id}] SWA F1={swa_eval_f1:.4f} > best model F1={best_model_eval_f1:.4f}，选用 SWA")

    print(f"\n[fold {fold_id} 结果] 选定来源: {selected_source}, 选定 epoch: {selected_epoch}")

    # 获取选定 epoch 的 scores
    if selected_source == "swa" and swa_scores is not None:
        oof_scores = swa_scores
    else:
        oof_scores = best_model_scores

    # 确定性验证
    print(f"\n[fold {fold_id}] 确定性验证...")
    model.eval()
    with torch.no_grad():
        _, scores_1 = evaluate(
            model=model, loader=loader, selection_indices=selection_indices,
            microbatch=PHASE_A_GATE.evaluation_microbatch, device=device, use_bf16=use_bf16,
        )
        _, scores_2 = evaluate(
            model=model, loader=loader, selection_indices=selection_indices,
            microbatch=PHASE_A_GATE.evaluation_microbatch, device=device, use_bf16=use_bf16,
        )
    deterministic = bool(np.array_equal(scores_1, scores_2))
    print(f"[fold {fold_id}] bitwise identical: {deterministic}")
    if not deterministic:
        resource_cell.record_integrity(
            kind="nondeterministic",
            detail=f"fold {fold_id} eval scores differ between two identical runs",
        )

    print(f"\n[fold {fold_id} 总结] sel_f1={selected_f1:.4f}, wall={fold_time:.1f}s ({fold_time/60:.1f}min)")

    # 保存 fold checkpoint
    fold_checkpoint_path = REPORT_DIR / f"fold{fold_id}_checkpoint.pt"
    checkpoint_payload = {
        "schema": "axon_loop186_fold_checkpoint_v1",
        "loop_id": LOOP_ID,
        "fold_id": fold_id,
        "selected_epoch": selected_epoch,
        "selected_source": selected_source,
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
    torch.save(checkpoint_payload, fold_checkpoint_path)
    print(f"[fold {fold_id}] checkpoint 已保存: {fold_checkpoint_path}")

    # 保存 fold training log
    fold_log_path = REPORT_DIR / f"fold{fold_id}_training_log.jsonl"
    with fold_log_path.open("w", encoding="utf-8") as f:
        for entry in training_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 释放 GPU 内存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold_id": fold_id,
        "fit_folds": list(fit_folds),
        "selection_fold": selection_fold,
        "selected_epoch": selected_epoch,
        "selected_source": selected_source,
        "selection_f1": float(selected_f1),
        "selection_acc": float(selected_acc),
        "selection_loss": float(selected_loss),
        "swa_eval_f1": swa_eval_f1,
        "swa_eval_acc": swa_eval_acc,
        "swa_eval_loss": swa_eval_loss,
        "oof_scores": oof_scores,
        "oof_indices": selection_indices.tolist(),
        "oof_labels": selection_labels_np.tolist(),
        "training_log": training_log,
        "deterministic": deterministic,
        "fold_wall_seconds": fold_time,
        "checkpoint_path": str(fold_checkpoint_path),
        "parameter_count": param_count,
        "model_config": asdict(config),
    }


# ---------------------------------------------------------------------------
# 单 fold Phase A
# ---------------------------------------------------------------------------


def run_phase_a(
    *,
    device_request: str = "auto",
    max_epochs: int | None = None,
    seed: int = 41,
) -> dict[str, object]:
    print("\n=== Loop186 Phase A Preflight ===")
    preflight_result = preflight()
    auth_json = generate_authorization_json(preflight_result)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    auth_path = REPORT_DIR / "phase_a_authorization.json"
    with auth_path.open("w", encoding="utf-8") as f:
        json.dump(auth_json, f, indent=2, ensure_ascii=False)
    print(f"[preflight] 授权 JSON 已保存: {auth_path}")

    device, use_bf16 = resolve_device(device_request)
    print(f"\n[device] 使用设备: {device}, BF16: {use_bf16}")

    epochs = max_epochs or PHASE_A_GATE.max_epochs
    print(f"\n[schedule] 单 fold {epochs} epochs（fit on fold 2,3,4, select on fold 1）")
    print(f"[schedule] 预计 ~4h（17.3M 参数 + AdamW + microbatch={PHASE_A_GATE.microbatch}）")

    resource_cell = ResourceCell()
    resource_cell.start()

    project_root = PROJECT_ROOT

    print(f"\n=== Loop186 单 fold 训练开始 ===")
    overall_start = time.time()

    # 单 fold：fit on fold 2,3,4, select on fold 1
    fit_folds = (2, 3, 4)
    selection_fold = 1
    fold_id = 1

    print(f"\n{'#'*60}")
    print(f"### Fold: fold_id={fold_id}, fit={fit_folds}, selection={selection_fold}")
    print(f"{'#'*60}")

    try:
        fold_result = train_one_fold(
            fold_id=fold_id,
            fit_folds=fit_folds,
            selection_fold=selection_fold,
            device=device,
            use_bf16=use_bf16,
            max_epochs=epochs,
            seed=seed,
            resource_cell=resource_cell,
            project_root=project_root,
        )
    except Exception as exc:
        print(f"\n[ERROR] Fold {fold_id} 训练失败: {exc}")
        import traceback
        traceback.print_exc()
        return {"error": f"fold {fold_id} failed: {exc}"}

    fold_results = [fold_result]

    total_time = time.time() - overall_start
    print(f"\n=== Loop186 单 fold 训练完成 ===")
    print(f"总耗时: {total_time:.1f}s ({total_time/60:.1f}min = {total_time/3600:.2f}h)")

    # 单 fold 评估
    fold_f1 = fold_result["selection_f1"]
    fold_acc = fold_result["selection_acc"]
    fold_loss = fold_result["selection_loss"]
    scores = np.array(fold_result["oof_scores"], dtype=np.float64)
    labels = np.array(fold_result["oof_labels"], dtype=np.int64)
    indices = np.array(fold_result["oof_indices"], dtype=np.int64)

    print(f"\n[fold {fold_id}] F1: {fold_f1:.6f}")
    print(f"[fold {fold_id}] Accuracy: {fold_acc:.6f}")
    print(f"[fold {fold_id}] Loss: {fold_loss:.6f}")

    # 确定性验证
    deterministic = fold_result["deterministic"]
    print(f"\n[determinism] bitwise identical: {deterministic}")

    # 资源门
    resource_passed = resource_cell.passed()
    print(f"\n=== Loop186 资源门: {'PASS' if resource_passed else 'FAIL'} ===")
    resource_receipt = resource_cell.build_receipt()

    # 保存 scores
    oof_path = REPORT_DIR / "oof_scores.npz"
    np.savez_compressed(
        oof_path,
        scores=scores,
        labels=labels,
        indices=indices,
    )
    print(f"[output] scores: {oof_path}")

    # 保存 receipt
    receipt = {
        "schema": "axon_loop186_phase_a_receipt_v1",
        "loop_id": LOOP_ID,
        "lineage": "expanded_single_fold_sam",
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
        "model": {
            "parameter_count": fold_result["parameter_count"],
            "config": fold_result["model_config"],
            "seed_base": seed,
        },
        "augmentation": {
            "mixup_alpha": MIXUP_ALPHA,
            "region_dropout_prob": REGION_DROPOUT_PROB,
            "label_smoothing_eps": LABEL_SMOOTHING_EPS,
        },
        "optimizer": {
            "type": "AdamW",
            "sam_enabled": False,
            "sam_rho": SAM_RHO,
            "base_lr": PHASE_A_GATE.learning_rate,
            "weight_decay": PHASE_A_GATE.weight_decay,
            "grad_clip": PHASE_A_GATE.grad_clip,
        },
        "swa": {
            "start_epoch": SWA_START_EPOCH,
            "swa_lr": SWA_LR,
            "swa_anneal_epochs": SWA_ANNEAL_EPOCHS,
        },
        "single_fold": {
            "fold_id": fold_result["fold_id"],
            "fit_folds": fold_result["fit_folds"],
            "selection_fold": fold_result["selection_fold"],
            "f1": fold_result["selection_f1"],
            "accuracy": fold_result["selection_acc"],
            "loss": fold_result["selection_loss"],
        },
        "fold_result": {
            "fold_id": fold_result["fold_id"],
            "fit_folds": fold_result["fit_folds"],
            "selection_fold": fold_result["selection_fold"],
            "selected_epoch": fold_result["selected_epoch"],
            "selected_source": fold_result["selected_source"],
            "selection_f1": fold_result["selection_f1"],
            "selection_acc": fold_result["selection_acc"],
            "selection_loss": fold_result["selection_loss"],
            "swa_eval_f1": fold_result["swa_eval_f1"],
            "swa_eval_acc": fold_result["swa_eval_acc"],
            "swa_eval_loss": fold_result["swa_eval_loss"],
            "deterministic": fold_result["deterministic"],
            "fold_wall_seconds": fold_result["fold_wall_seconds"],
            "checkpoint_path": fold_result["checkpoint_path"],
        },
        "training": {
            "epochs": epochs,
            "total_wall_seconds": total_time,
            "device": str(device),
            "use_bf16": use_bf16,
            "seed": seed,
        },
        "determinism": {
            "bitwise_identical": deterministic,
        },
        "resource_gate": resource_receipt,
        "resource_passed": resource_passed,
        "oof_scores_path": str(oof_path),
    }

    receipt_path = REPORT_DIR / "phase_a_receipt.json"
    with receipt_path.open("w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    print(f"\n[output] Receipt: {receipt_path}")

    # 总结
    print(f"\n=== Loop186 Phase A 总结 ===")
    print(f"资源门: {'PASS' if resource_passed else 'FAIL'}")
    print(f"单 fold F1: {fold_f1:.6f}")
    print(f"单 fold Accuracy: {fold_acc:.6f}")
    print(f"总耗时: {total_time/3600:.2f}h")
    print(f"确定性: {'PASS' if deterministic else 'FAIL'}")

    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Loop186 Phase A 训练（单 fold + SAM + 扩容）")
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

    if not receipt.get("resource_passed", False):
        print("\n[FAIL] 资源门未通过，Loop186 Phase A 失败")
        sys.exit(1)
    else:
        print("\n[PASS] 资源门通过，Loop186 Phase A 成功")


if __name__ == "__main__":
    main()
