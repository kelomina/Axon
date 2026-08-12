"""Loop180 极端扩容谱系训练脚本。

基于 Loop179 Phase A 训练流程，但激进扩容模型 + 强正则 + 数据增广：

架构改动（vs Loop179）：
- model_dim: 192 → 384 (2x)
- byte_embedding_dim: 64 → 96 (1.5x)
- hgconv_blocks: 1 → 3 (3x)
- transformer_layers: 2 → 4 (2x)
- transformer_heads: 6 → 8
- transformer_ffn_dim: 768 → 1536 (2x)
- dropout: 0.1 → 0.3 (3x)
- 预计参数量: ~8-10M (vs Loop179 1.6M)

训练改动：
- learning_rate: 3e-4 → 5e-4
- weight_decay: 1e-2 → 3e-2 (3x)
- label_smoothing: 0 → 0.1
- mixup_alpha: 0 → 0.2
- early_stopping_patience: 4
- max_epochs: 12 → 20
- warmup: 1 epoch
- EMA decay=0.999, BF16, grad_clip=1.0
- microbatch=2, accumulation=16, effective_batch=32

资源门：复用 Loop179 Phase A 资源门（6.5 GiB GPU, 11 GiB RSS, 6h wall）
数据：复用 Loop179 PhaseADataLoader（12000 fit rows + 4000 selection rows）

用法：
    python -u scripts/run_loop180_phase_a.py --device cuda
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

# 添加项目根到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from src.loop179 import (
    PHASE_A_GATE,
    HGConvRegionConfig,
    HGConvRegionNet,
    PhaseADataLoader,
    ResourceCell,
    assert_contract_invariants,
    assert_phase0_closure,
)
from src.loop179.contracts import LOOP_ID as LOOP179_ID


# ---------------------------------------------------------------------------
# Loop180 身份与路径
# ---------------------------------------------------------------------------

LOOP_ID = "Loop180"
LINEAGE = "extreme_capacity_expansion"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "roadmap_9997" / LOOP_ID.lower()


# ---------------------------------------------------------------------------
# Loop180 冻结超参（与 Loop179 不同的部分）
# ---------------------------------------------------------------------------

LOOP180_CONFIG = HGConvRegionConfig(
    vocabulary_size=257,
    padding_token=256,
    byte_embedding_dim=96,        # 64 → 96
    model_dim=384,                # 192 → 384
    patch_size=16,
    hgconv_blocks=3,              # 1 → 3
    hgconv_filter_length=32,
    region_type_count=6,
    bucket_count=64,
    transformer_layers=4,         # 2 → 4
    transformer_heads=8,          # 6 → 8 (384 % 8 == 0)
    transformer_ffn_dim=1536,     # 768 → 1536
    b0_feature_dim=571,
    expected_regions=16,
    expected_region_bytes=8192,
    dropout=0.3,                  # 0.1 → 0.3
)

LOOP180_TRAIN_HYPERS = {
    "max_epochs": 20,             # 12 → 20
    "microbatch": 2,
    "accumulation": 16,
    "effective_batch": 32,
    "learning_rate": 5.0e-4,      # 3e-4 → 5e-4
    "weight_decay": 3.0e-2,       # 1e-2 → 3e-2
    "warmup_epochs": 1,
    "grad_clip": 1.0,
    "ema_decay": 0.999,
    "label_smoothing": 0.1,       # 新增
    "mixup_alpha": 0.2,           # 新增
    "early_stopping_patience": 4, # 新增
    "seed": 41,
}


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight() -> dict[str, object]:
    """Loop180 preflight: 契约自检、源码闭包、设备检测、参数量预估。"""

    print(f"[preflight] {LOOP_ID} 契约自检（借用 Loop179 contracts）...")
    assert_contract_invariants()

    print(f"[preflight] {LOOP_ID} 源码闭包检查（借用 Loop179 closure）...")
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

    # 参数量预估（CPU 上构造一次）
    print("[preflight] 构造模型预估参数量...")
    model_probe = HGConvRegionNet(LOOP180_CONFIG)
    param_count_probe = sum(p.numel() for p in model_probe.parameters() if p.requires_grad)
    print(f"[preflight] {LOOP_ID} 参数量: {param_count_probe:,} (vs Loop179 1,604,677)")
    del model_probe

    return {
        "closure_scanned_files": list(closure_report.scanned_files),
        "closure_manifest": dict(closure_report.manifest),
        "cuda_available": cuda_available,
        "bf16_supported": bf16_supported,
        "device_name": device_name,
        "gpu_name": gpu_name,
        "gpu_total_bytes": int(gpu_total_bytes),
        "param_count_probe": int(param_count_probe),
    }


def generate_authorization_json(preflight_result: dict[str, object]) -> dict[str, object]:
    """生成 Loop180 授权 JSON。"""

    auth = {
        "schema": f"axon_{LOOP_ID.lower()}_phase_a_authorization_v1",
        "loop_id": LOOP_ID,
        "lineage": LINEAGE,
        "phase": "A",
        "decision": f"{LOOP_ID.lower()}_authorized_by_user_2026_07_20_aggressive_research",
        "claim_scope": "extreme_capacity_expansion_phase_a",
        "val_test_or_full_access_allowed": False,
        "fit_rows": PHASE_A_GATE.fit_rows,
        "selection_rows": PHASE_A_GATE.selection_rows,
        "max_epochs": LOOP180_TRAIN_HYPERS["max_epochs"],
        "gpu_allocated_bytes_limit": PHASE_A_GATE.gpu_allocated_bytes,
        "rss_bytes_limit": PHASE_A_GATE.rss_bytes,
        "wall_seconds_limit": PHASE_A_GATE.wall_seconds,
        "cuda_available": preflight_result["cuda_available"],
        "bf16_supported": preflight_result["bf16_supported"],
        "device_name": preflight_result["device_name"],
        "gpu_name": preflight_result["gpu_name"],
        "config_overrides": asdict(LOOP180_CONFIG),
        "train_hypers": LOOP180_TRAIN_HYPERS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return auth


# ---------------------------------------------------------------------------
# 训练组件
# ---------------------------------------------------------------------------

def deterministic_epoch_batches(
    fit_indices: np.ndarray,
    *,
    microbatch: int,
    seed: int,
    epoch: int,
    loop_id: str = LOOP_ID,
) -> tuple[np.ndarray, ...]:
    """确定性 epoch batch 排序。"""

    material = f"{loop_id.lower()}-phase-a|{seed}|{epoch}|{fit_indices.size}".encode("ascii")
    order_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    order = fit_indices.copy()
    np.random.default_rng(order_seed).shuffle(order)
    return tuple(order[start : start + microbatch] for start in range(0, order.size, microbatch))


def resolve_device(requested: str) -> tuple[torch.device, bool]:
    """解析设备，返回 (device, use_bf16)。"""

    if requested == "cpu":
        return torch.device("cpu"), False
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA explicitly requested but unavailable")
    if requested in {"auto", "cuda"} and cuda_available:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"{LOOP_ID} CUDA requires BF16 support")
        return torch.device("cuda"), True
    return torch.device("cpu"), False


def autocast(device: torch.device, enabled: bool):
    """BF16 autocast context manager。"""

    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True)


class ExponentialMovingAverage:
    """EMA 模型权重（decay=0.999）。"""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            target = self.state[name]
            if target.is_floating_point():
                target.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                target.copy_(value.detach())

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.state, strict=True)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """将 batch 张量移动到设备。"""

    return {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }


def forward_model(
    model: HGConvRegionNet,
    batch: dict[str, torch.Tensor],
    *,
    use_b0: bool = True,
) -> torch.Tensor:
    """模型 forward，返回 fusion_logits（或 region_logits）。"""

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
    """Cosine LR schedule with linear warmup。"""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def mixup_batch(
    batch: dict[str, torch.Tensor],
    *,
    alpha: float,
    rng: np.random.Generator,
    num_classes: int = 2,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """对 batch 应用 mixup，返回 (mixed_batch, soft_labels)。

    mixup_lambda ~ Beta(alpha, alpha)
    mixed_inputs = lambda * x_a + (1 - lambda) * x_b
    soft_labels = lambda * y_a + (1 - lambda) * y_b (one-hot)
    """

    if alpha <= 0.0:
        # 不做 mixup，返回原始 batch + hard labels
        labels = batch["labels"]
        soft = functional.one_hot(labels.long(), num_classes=num_classes).float()
        return batch, soft

    lam = float(rng.beta(alpha, alpha))
    batch_size = batch["labels"].shape[0]
    perm = torch.from_numpy(rng.permutation(batch_size)).long().to(batch["labels"].device)

    mixed_batch = dict(batch)
    # 对 region_tokens 做 mixup（int64 用 lambda 加权四舍五入会破坏 token 含义）
    # 改为对 embedding 后的 patches 做 mixup 不容易，这里改为：只对 b0_features 做 mixup
    # region_tokens 保持原样，但用 perm 索引；labels 做 soft mixup
    if batch.get("b0_features") is not None:
        mixed_batch["b0_features"] = (
            lam * batch["b0_features"] + (1.0 - lam) * batch["b0_features"][perm]
        )

    labels_a = batch["labels"]
    labels_b = batch["labels"][perm]
    soft_a = functional.one_hot(labels_a.long(), num_classes=num_classes).float()
    soft_b = functional.one_hot(labels_b.long(), num_classes=num_classes).float()
    soft_labels = lam * soft_a + (1.0 - lam) * soft_b

    # 同时返回 mixed region_tokens：选择 lam 阈值上的样本
    # 简化策略：region_tokens 不 mixup（保持 token id），但 loss 用 soft labels
    return mixed_batch, soft_labels


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------

def train_one_epoch(
    *,
    model: HGConvRegionNet,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
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
    label_smoothing: float,
    mixup_alpha: float,
    mixup_rng: np.random.Generator,
) -> tuple[float, int]:
    """训练一个 epoch，返回 (avg_loss, optimizer_steps)。"""

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

    for window_start in range(0, len(batches), accumulation):
        window = batches[window_start : window_start + accumulation]
        window_indices = np.concatenate(window)
        window_normalizer = float(window_indices.size)

        optimizer.zero_grad(set_to_none=True)
        for receiver_indices in window:
            batch = move_batch_to_device(
                loader.materialize_batch(receiver_indices),
                device,
            )
            # mixup + soft labels
            mixed_batch, soft_labels = mixup_batch(
                batch, alpha=mixup_alpha, rng=mixup_rng, num_classes=2
            )
            with autocast(device, use_bf16):
                logits = forward_model(model, mixed_batch, use_b0=True)
            # cross entropy with soft labels + label smoothing
            log_probs = functional.log_softmax(logits.float(), dim=1)
            # soft target cross entropy
            nll = -(soft_labels * log_probs).sum(dim=1)
            # label smoothing: (1 - eps) * nll + eps * uniform
            eps = label_smoothing
            uniform = -log_probs.mean(dim=1)
            per_sample = (1.0 - eps) * nll + eps * uniform
            if not torch.isfinite(per_sample).all().item():
                raise FloatingPointError(
                    f"epoch {epoch} produced non-finite loss"
                )
            numerator = per_sample.sum()
            (numerator / window_normalizer).backward()
            epoch_loss_sum += float(numerator.detach())
            epoch_sample_count += int(receiver_indices.size)

        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(gradient_norm).item():
            raise FloatingPointError(
                f"epoch {epoch} produced non-finite gradient norm"
            )
        optimizer.step()
        scheduler.step()
        ema.update(model)
        optimizer_steps += 1

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
    """在 selection fold 上评估，返回 (avg_loss, probabilities)。"""

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


def select_earliest_minimum_epoch(losses: list[float]) -> int:
    """选择最早的 minimum loss epoch（1-indexed）。"""

    values = [float(v) for v in losses]
    if not values or not np.isfinite(values).all():
        raise ValueError("selection losses must be nonempty finite")
    minimum = min(values)
    return values.index(minimum) + 1


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------

def run_loop180_phase_a(
    *,
    device_request: str = "auto",
    max_epochs: int | None = None,
    seed: int = 41,
) -> dict[str, object]:
    """执行 Loop180 Phase A 训练，返回 receipt dict。"""

    # 1. Preflight
    print(f"\n=== {LOOP_ID} Phase A Preflight ===")
    preflight_result = preflight()
    auth_json = generate_authorization_json(preflight_result)

    # 2. 准备输出目录
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    auth_path = REPORT_DIR / "phase_a_authorization.json"
    with auth_path.open("w", encoding="utf-8") as f:
        json.dump(auth_json, f, indent=2, ensure_ascii=False)
    print(f"[preflight] 授权 JSON 已保存: {auth_path}")

    # 3. 加载数据
    print(f"\n=== {LOOP_ID} 数据加载 ===")
    loader = PhaseADataLoader()
    t0 = time.time()
    data_receipt = loader.load_real_data()
    t1 = time.time()
    print(f"[data] 加载耗时: {t1 - t0:.2f}s")
    print(f"[data] 行数: {data_receipt['row_count']}, 区域数: {data_receipt['region_count']}")

    fit_indices = loader.get_fit_indices()
    selection_indices = loader.get_selection_indices()
    print(f"[data] fit 行数: {fit_indices.shape[0]}")
    print(f"[data] selection 行数: {selection_indices.shape[0]}")

    # 4. 设备解析
    device, use_bf16 = resolve_device(device_request)
    print(f"\n[device] 使用设备: {device}, BF16: {use_bf16}")

    # 5. 模型初始化
    print(f"\n=== {LOOP_ID} 模型初始化 ===")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = LOOP180_CONFIG
    model = HGConvRegionNet(config).to(device=device, dtype=torch.float32)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] 参数量: {param_count:,} (vs Loop179 1,604,677, 扩容 {param_count / 1604677:.2f}x)")
    print(f"[model] config: model_dim={config.model_dim}, heads={config.transformer_heads}, layers={config.transformer_layers}, hgconv_blocks={config.hgconv_blocks}")

    ema = ExponentialMovingAverage(model, LOOP180_TRAIN_HYPERS["ema_decay"])

    # 6. 优化器
    epochs = max_epochs or LOOP180_TRAIN_HYPERS["max_epochs"]
    microbatch = LOOP180_TRAIN_HYPERS["microbatch"]
    accumulation = LOOP180_TRAIN_HYPERS["accumulation"]
    learning_rate = LOOP180_TRAIN_HYPERS["learning_rate"]
    weight_decay = LOOP180_TRAIN_HYPERS["weight_decay"]
    label_smoothing = LOOP180_TRAIN_HYPERS["label_smoothing"]
    mixup_alpha = LOOP180_TRAIN_HYPERS["mixup_alpha"]
    patience = LOOP180_TRAIN_HYPERS["early_stopping_patience"]
    warmup_epochs = LOOP180_TRAIN_HYPERS["warmup_epochs"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    batches_per_epoch = math.ceil(math.ceil(fit_indices.size / microbatch) / accumulation)
    total_steps = epochs * batches_per_epoch
    warmup_steps = warmup_epochs * batches_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: scheduler_multiplier(
            step, warmup_steps=warmup_steps, total_steps=total_steps
        ),
    )
    print(f"[optimizer] AdamW lr={learning_rate}, wd={weight_decay}")
    print(f"[scheduler] cosine, warmup={warmup_steps} steps, total={total_steps} steps")
    print(f"[schedule] {batches_per_epoch} optimizer steps/epoch × {epochs} epochs")
    print(f"[regularization] dropout={config.dropout}, label_smoothing={label_smoothing}, mixup_alpha={mixup_alpha}, wd={weight_decay}")
    print(f"[early_stopping] patience={patience}")

    # mixup RNG（每 epoch 重置以保证确定性）
    mixup_rng = np.random.default_rng(seed)

    # 7. 资源门
    resource_cell = ResourceCell()
    resource_cell.start()

    # 8. 训练循环 + 早停
    print(f"\n=== {LOOP_ID} 训练开始 ===")
    selection_losses: list[float] = []
    training_losses: list[float] = []
    training_log: list[dict[str, object]] = []
    start_time = time.time()
    best_sel_loss = float("inf")
    best_epoch = 0
    epochs_since_best = 0
    early_stopped = False

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n[epoch {epoch}/{epochs}] 训练中...")

        # 重置 mixup RNG 每 epoch（确定性）
        mixup_rng = np.random.default_rng(seed + epoch)

        train_loss, opt_steps = train_one_epoch(
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            loader=loader,
            fit_indices=fit_indices,
            epoch=epoch,
            seed=seed,
            microbatch=microbatch,
            accumulation=accumulation,
            grad_clip=LOOP180_TRAIN_HYPERS["grad_clip"],
            device=device,
            use_bf16=use_bf16,
            label_smoothing=label_smoothing,
            mixup_alpha=mixup_alpha,
            mixup_rng=mixup_rng,
        )
        training_losses.append(train_loss)

        # EMA 评估
        ema_model = copy.deepcopy(model)
        ema.copy_to(ema_model)
        sel_loss, sel_scores = evaluate(
            model=ema_model,
            loader=loader,
            selection_indices=selection_indices,
            microbatch=microbatch,
            device=device,
            use_bf16=use_bf16,
        )
        selection_losses.append(sel_loss)
        del ema_model

        # 资源采样
        sample = resource_cell.sample_and_inject(epoch=epoch, step=opt_steps, note=f"epoch {epoch} done")
        epoch_elapsed = time.time() - epoch_start

        # 计算 selection accuracy / F1
        sel_labels = loader._fold_labels[selection_indices]
        sel_preds = (sel_scores > 0.5).astype(np.int64)
        sel_acc = float((sel_preds == sel_labels).mean())
        # 二分类 F1（pos class = 1 = malware）
        tp = int(((sel_preds == 1) & (sel_labels == 1)).sum())
        fp = int(((sel_preds == 1) & (sel_labels == 0)).sum())
        fn = int(((sel_preds == 0) & (sel_labels == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        print(f"[epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, sel_loss={sel_loss:.4f}, sel_acc={sel_acc:.4f}, sel_f1={f1:.4f}")
        print(f"  GPU={sample.gpu_allocated_bytes / 1024**3:.3f} GiB, RSS={sample.rss_bytes / 1024**3:.3f} GiB, wall={sample.wall_seconds:.1f}s, epoch_time={epoch_elapsed:.1f}s")

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "selection_loss": sel_loss,
            "selection_accuracy": sel_acc,
            "selection_f1": f1,
            "selection_precision": precision,
            "selection_recall": recall,
            "optimizer_steps": opt_steps,
            "gpu_allocated_bytes": sample.gpu_allocated_bytes,
            "rss_bytes": sample.rss_bytes,
            "wall_seconds": sample.wall_seconds,
            "epoch_seconds": epoch_elapsed,
        }
        training_log.append(log_entry)

        # 早停判断
        if sel_loss < best_sel_loss:
            best_sel_loss = sel_loss
            best_epoch = epoch
            epochs_since_best = 0
            print(f"  [early_stop] 新最佳 sel_loss={best_sel_loss:.4f} @ epoch {best_epoch}")
        else:
            epochs_since_best += 1
            print(f"  [early_stop] 未提升 ({epochs_since_best}/{patience})")

        # 检查资源门是否已超限
        if not resource_cell.passed():
            print(f"\n[ALERT] 资源门违规！violations: {len(resource_cell.violations)}")
            for v in resource_cell.violations:
                print(f"  [{v.kind}] {v.detail}")
            break

        # 早停触发
        if epochs_since_best >= patience:
            print(f"\n[early_stop] 触发早停：{patience} epochs 未提升，停止训练")
            early_stopped = True
            break

    total_time = time.time() - start_time

    # 9. 选择最佳 epoch
    selected_epoch = select_earliest_minimum_epoch(selection_losses) if selection_losses else 0
    print(f"\n[result] 选定 epoch: {selected_epoch} (best_epoch={best_epoch})")
    print(f"[result] 最小 selection loss: {min(selection_losses) if selection_losses else 'N/A'}")

    # 10. 确定性验证
    print(f"\n=== {LOOP_ID} 确定性验证 ===")
    ema_model = copy.deepcopy(model)
    ema.copy_to(ema_model)
    ema_model.eval()
    with torch.no_grad():
        _, scores_1 = evaluate(
            model=ema_model,
            loader=loader,
            selection_indices=selection_indices,
            microbatch=microbatch,
            device=device,
            use_bf16=use_bf16,
        )
        _, scores_2 = evaluate(
            model=ema_model,
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

    # 11. 最终资源门检查
    resource_passed = resource_cell.passed()
    print(f"\n=== {LOOP_ID} 资源门: {'PASS' if resource_passed else 'FAIL'} ===")
    resource_receipt = resource_cell.build_receipt()

    # 12. 保存训练日志
    log_path = REPORT_DIR / "phase_a_training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for entry in training_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[output] 训练日志: {log_path}")

    # 13. 保存 checkpoint（如果资源门通过）
    checkpoint_path = REPORT_DIR / "phase_a_checkpoint.pt"
    checkpoint_sha256 = None
    if resource_passed:
        torch.save(
            {
                "schema": f"axon_{LOOP_ID.lower()}_phase_a_checkpoint_v1",
                "loop_id": LOOP_ID,
                "selected_epoch": selected_epoch,
                "best_epoch": best_epoch,
                "model_config": asdict(config),
                "train_hypers": LOOP180_TRAIN_HYPERS,
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "ema_state_dict": {
                    name: value.detach().cpu()
                    for name, value in ema.state.items()
                },
            },
            checkpoint_path,
        )
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()
        print(f"[output] Checkpoint: {checkpoint_path} ({checkpoint_sha256[:16]}...)")

    # 14. 生成 receipt
    receipt = {
        "schema": f"axon_{LOOP_ID.lower()}_phase_a_receipt_v1",
        "loop_id": LOOP_ID,
        "lineage": LINEAGE,
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
            "parameter_count_vs_loop179": param_count / 1604677,
            "config": asdict(config),
            "config_delta_vs_loop179": {
                "byte_embedding_dim": f"64 → {config.byte_embedding_dim}",
                "model_dim": f"192 → {config.model_dim}",
                "hgconv_blocks": f"1 → {config.hgconv_blocks}",
                "transformer_layers": f"2 → {config.transformer_layers}",
                "transformer_heads": f"6 → {config.transformer_heads}",
                "transformer_ffn_dim": f"768 → {config.transformer_ffn_dim}",
                "dropout": f"0.1 → {config.dropout}",
            },
            "seed": seed,
        },
        "train_hypers": LOOP180_TRAIN_HYPERS,
        "training": {
            "epochs_run": len(training_log),
            "epochs_planned": epochs,
            "early_stopped": early_stopped,
            "selected_epoch": selected_epoch,
            "best_epoch": best_epoch,
            "selection_losses": selection_losses,
            "training_losses": training_losses,
            "total_wall_seconds": total_time,
            "optimizer_total_steps": sum(e["optimizer_steps"] for e in training_log),
            "device": str(device),
            "use_bf16": use_bf16,
            "seed": seed,
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

    # 15. 总结
    print(f"\n=== {LOOP_ID} Phase A 总结 ===")
    print(f"资源门: {'PASS' if resource_passed else 'FAIL'}")
    print(f"训练 epochs: {len(training_log)}/{epochs} (early_stopped={early_stopped})")
    print(f"选定 epoch: {selected_epoch}, best_epoch: {best_epoch}")
    best_f1 = max((e["selection_f1"] for e in training_log), default=0.0)
    best_acc = max((e["selection_accuracy"] for e in training_log), default=0.0)
    print(f"最佳 selection F1: {best_f1:.4f}")
    print(f"最佳 selection accuracy: {best_acc:.4f}")
    print(f"总 wall time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"确定性: {'PASS' if deterministic else 'FAIL'}")
    print(f"参数量: {param_count:,} (vs Loop179 1,604,677, 扩容 {param_count / 1604677:.2f}x)")

    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{LOOP_ID} Phase A 训练")
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
        help=f"最大 epochs (default: {LOOP180_TRAIN_HYPERS['max_epochs']})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=41,
        help="随机种子 (default: 41)",
    )
    args = parser.parse_args()

    receipt = run_loop180_phase_a(
        device_request=args.device,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )

    if not receipt["resource_passed"]:
        print(f"\n[FAIL] 资源门未通过，{LOOP_ID} Phase A 失败")
        sys.exit(1)
    else:
        print(f"\n[PASS] 资源门通过，{LOOP_ID} Phase A 成功")
        training_log_path = receipt["training_log_path"]
        best_f1 = 0.0
        if training_log_path:
            with open(training_log_path, "r", encoding="utf-8") as f:
                logs = [json.loads(line) for line in f if line.strip()]
            best_f1 = max((e["selection_f1"] for e in logs), default=0.0)
        print(f"最佳 selection F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
