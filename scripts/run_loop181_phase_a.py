"""Loop181 扁扩容 + 弱正则训练脚本。

基于 Loop180 扩容（9.58M 参数），但去除所有额外正则：

Loop180 失败诊断（vs Loop179）：
- Loop180 epoch 5 最佳 sel_acc=96.83% << Loop179 epoch 5 的 97.52%
- 原因：4 重正则（dropout=0.3, wd=3e-2, mixup=0.2, label_smoothing=0.1）叠加导致欠拟合
- train_loss 从 0.46→0.33 下降极慢（Loop179 epoch 5 已到 0.0164）

Loop181 修正策略：
- 保留 Loop180 扩容（model_dim=384, hgconv_blocks=3, transformer_layers=4, heads=8, ffn=1536）
- 去除 mixup（alpha=0）
- 去除 label_smoothing（eps=0）
- dropout: 0.3 → 0.15（轻微提升 vs Loop179 的 0.1）
- weight_decay: 3e-2 → 1e-2（与 Loop179 一致）
- learning_rate: 5e-4 → 3e-4（与 Loop179 一致）
- max_epochs: 20 → 15
- early_stopping_patience: 4 → 3（更激进）

预期：纯验证扩容效果，若仍 <97.52% 说明扩容本身无效，需 Loop182 改架构。
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


# ---------------------------------------------------------------------------
# Loop181 身份与路径
# ---------------------------------------------------------------------------

LOOP_ID = "Loop181"
LINEAGE = "capacity_expansion_weak_regularization"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "roadmap_9997" / LOOP_ID.lower()


# ---------------------------------------------------------------------------
# Loop181 冻结超参
# ---------------------------------------------------------------------------

LOOP181_CONFIG = HGConvRegionConfig(
    vocabulary_size=257,
    padding_token=256,
    byte_embedding_dim=96,        # 与 Loop180 一致
    model_dim=384,                # 与 Loop180 一致
    patch_size=16,
    hgconv_blocks=3,              # 与 Loop180 一致
    hgconv_filter_length=32,
    region_type_count=6,
    bucket_count=64,
    transformer_layers=4,         # 与 Loop180 一致
    transformer_heads=8,          # 与 Loop180 一致
    transformer_ffn_dim=1536,     # 与 Loop180 一致
    b0_feature_dim=571,
    expected_regions=16,
    expected_region_bytes=8192,
    dropout=0.15,                 # Loop180 0.3 → 0.15（弱化）
)

LOOP181_TRAIN_HYPERS = {
    "max_epochs": 15,             # Loop180 20 → 15
    "microbatch": 2,
    "accumulation": 16,
    "effective_batch": 32,
    "learning_rate": 3.0e-4,      # Loop180 5e-4 → 3e-4（与 Loop179 一致）
    "weight_decay": 1.0e-2,       # Loop180 3e-2 → 1e-2（与 Loop179 一致）
    "warmup_epochs": 1,
    "grad_clip": 1.0,
    "ema_decay": 0.999,
    "label_smoothing": 0.0,       # Loop180 0.1 → 0（去除）
    "mixup_alpha": 0.0,           # Loop180 0.2 → 0（去除）
    "early_stopping_patience": 3, # Loop180 4 → 3（更激进）
    "seed": 41,
}


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight() -> dict[str, object]:
    """Loop181 preflight。"""

    print(f"[preflight] {LOOP_ID} 契约自检...")
    assert_contract_invariants()

    print(f"[preflight] {LOOP_ID} 源码闭包检查...")
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
    model_probe = HGConvRegionNet(LOOP181_CONFIG)
    param_count_probe = sum(p.numel() for p in model_probe.parameters() if p.requires_grad)
    print(f"[preflight] {LOOP_ID} 参数量: {param_count_probe:,}")
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
    """生成 Loop181 授权 JSON。"""

    auth = {
        "schema": f"axon_{LOOP_ID.lower()}_phase_a_authorization_v1",
        "loop_id": LOOP_ID,
        "lineage": LINEAGE,
        "phase": "A",
        "decision": f"{LOOP_ID.lower()}_authorized_by_user_2026_07_20_aggressive_research",
        "claim_scope": "capacity_expansion_weak_regularization",
        "val_test_or_full_access_allowed": False,
        "fit_rows": PHASE_A_GATE.fit_rows,
        "selection_rows": PHASE_A_GATE.selection_rows,
        "max_epochs": LOOP181_TRAIN_HYPERS["max_epochs"],
        "gpu_allocated_bytes_limit": PHASE_A_GATE.gpu_allocated_bytes,
        "rss_bytes_limit": PHASE_A_GATE.rss_bytes,
        "wall_seconds_limit": PHASE_A_GATE.wall_seconds,
        "cuda_available": preflight_result["cuda_available"],
        "bf16_supported": preflight_result["bf16_supported"],
        "device_name": preflight_result["device_name"],
        "gpu_name": preflight_result["gpu_name"],
        "config": asdict(LOOP181_CONFIG),
        "train_hypers": LOOP181_TRAIN_HYPERS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return auth


# ---------------------------------------------------------------------------
# 训练组件（与 Loop180 相同，但移除 mixup + label_smoothing）
# ---------------------------------------------------------------------------

def deterministic_epoch_batches(
    fit_indices: np.ndarray,
    *,
    microbatch: int,
    seed: int,
    epoch: int,
) -> tuple[np.ndarray, ...]:
    """确定性 epoch batch 排序。"""

    material = f"{LOOP_ID.lower()}-phase-a|{seed}|{epoch}|{fit_indices.size}".encode("ascii")
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
            raise RuntimeError(f"{LOOP_ID} CUDA requires BF16 support")
        return torch.device("cuda"), True
    return torch.device("cpu"), False


def autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True)


class ExponentialMovingAverage:
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
) -> tuple[float, int]:
    """训练一个 epoch（无 mixup，无 label smoothing）。"""

    model.train()
    batches = deterministic_epoch_batches(
        fit_indices, microbatch=microbatch, seed=seed, epoch=epoch,
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
            with autocast(device, use_bf16):
                logits = forward_model(model, batch, use_b0=True)
            per_sample = functional.cross_entropy(
                logits.float(), batch["labels"], reduction="none"
            )
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
    values = [float(v) for v in losses]
    if not values or not np.isfinite(values).all():
        raise ValueError("selection losses must be nonempty finite")
    minimum = min(values)
    return values.index(minimum) + 1


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------

def run_loop181_phase_a(
    *,
    device_request: str = "auto",
    max_epochs: int | None = None,
    seed: int = 41,
) -> dict[str, object]:
    """执行 Loop181 Phase A 训练。"""

    print(f"\n=== {LOOP_ID} Phase A Preflight ===")
    preflight_result = preflight()
    auth_json = generate_authorization_json(preflight_result)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    auth_path = REPORT_DIR / "phase_a_authorization.json"
    with auth_path.open("w", encoding="utf-8") as f:
        json.dump(auth_json, f, indent=2, ensure_ascii=False)
    print(f"[preflight] 授权 JSON 已保存: {auth_path}")

    print(f"\n=== {LOOP_ID} 数据加载 ===")
    loader = PhaseADataLoader()
    t0 = time.time()
    data_receipt = loader.load_real_data()
    t1 = time.time()
    print(f"[data] 加载耗时: {t1 - t0:.2f}s")
    print(f"[data] 行数: {data_receipt['row_count']}")

    fit_indices = loader.get_fit_indices()
    selection_indices = loader.get_selection_indices()
    print(f"[data] fit: {fit_indices.shape[0]}, selection: {selection_indices.shape[0]}")

    device, use_bf16 = resolve_device(device_request)
    print(f"\n[device] 使用设备: {device}, BF16: {use_bf16}")

    print(f"\n=== {LOOP_ID} 模型初始化 ===")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = LOOP181_CONFIG
    model = HGConvRegionNet(config).to(device=device, dtype=torch.float32)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] 参数量: {param_count:,} (vs Loop179 1,604,677, 扩容 {param_count / 1604677:.2f}x)")
    print(f"[model] config: model_dim={config.model_dim}, heads={config.transformer_heads}, layers={config.transformer_layers}, hgconv_blocks={config.hgconv_blocks}, dropout={config.dropout}")

    ema = ExponentialMovingAverage(model, LOOP181_TRAIN_HYPERS["ema_decay"])

    epochs = max_epochs or LOOP181_TRAIN_HYPERS["max_epochs"]
    microbatch = LOOP181_TRAIN_HYPERS["microbatch"]
    accumulation = LOOP181_TRAIN_HYPERS["accumulation"]
    learning_rate = LOOP181_TRAIN_HYPERS["learning_rate"]
    weight_decay = LOOP181_TRAIN_HYPERS["weight_decay"]
    patience = LOOP181_TRAIN_HYPERS["early_stopping_patience"]
    warmup_epochs = LOOP181_TRAIN_HYPERS["warmup_epochs"]

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
    print(f"[regularization] dropout={config.dropout}, wd={weight_decay}, NO mixup, NO label_smoothing")
    print(f"[early_stopping] patience={patience}")

    resource_cell = ResourceCell()
    resource_cell.start()

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
            grad_clip=LOOP181_TRAIN_HYPERS["grad_clip"],
            device=device,
            use_bf16=use_bf16,
        )
        training_losses.append(train_loss)

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

        sample = resource_cell.sample_and_inject(epoch=epoch, step=opt_steps, note=f"epoch {epoch} done")
        epoch_elapsed = time.time() - epoch_start

        sel_labels = loader._fold_labels[selection_indices]
        sel_preds = (sel_scores > 0.5).astype(np.int64)
        sel_acc = float((sel_preds == sel_labels).mean())
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

        if sel_loss < best_sel_loss:
            best_sel_loss = sel_loss
            best_epoch = epoch
            epochs_since_best = 0
            print(f"  [early_stop] 新最佳 sel_loss={best_sel_loss:.4f} @ epoch {best_epoch}")
        else:
            epochs_since_best += 1
            print(f"  [early_stop] 未提升 ({epochs_since_best}/{patience})")

        if not resource_cell.passed():
            print(f"\n[ALERT] 资源门违规！")
            for v in resource_cell.violations:
                print(f"  [{v.kind}] {v.detail}")
            break

        if epochs_since_best >= patience:
            print(f"\n[early_stop] 触发早停：{patience} epochs 未提升")
            early_stopped = True
            break

    total_time = time.time() - start_time

    selected_epoch = select_earliest_minimum_epoch(selection_losses) if selection_losses else 0
    print(f"\n[result] 选定 epoch: {selected_epoch} (best_epoch={best_epoch})")
    print(f"[result] 最小 selection loss: {min(selection_losses) if selection_losses else 'N/A'}")

    print(f"\n=== {LOOP_ID} 确定性验证 ===")
    ema_model = copy.deepcopy(model)
    ema.copy_to(ema_model)
    ema_model.eval()
    with torch.no_grad():
        _, scores_1 = evaluate(
            model=ema_model, loader=loader, selection_indices=selection_indices,
            microbatch=microbatch, device=device, use_bf16=use_bf16,
        )
        _, scores_2 = evaluate(
            model=ema_model, loader=loader, selection_indices=selection_indices,
            microbatch=microbatch, device=device, use_bf16=use_bf16,
        )
    deterministic = bool(np.array_equal(scores_1, scores_2))
    print(f"[determinism] bitwise identical: {deterministic}")
    if not deterministic:
        resource_cell.record_integrity(
            kind="nondeterministic",
            detail="eval scores differ between two identical runs",
        )

    resource_passed = resource_cell.passed()
    print(f"\n=== {LOOP_ID} 资源门: {'PASS' if resource_passed else 'FAIL'} ===")
    resource_receipt = resource_cell.build_receipt()

    log_path = REPORT_DIR / "phase_a_training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for entry in training_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[output] 训练日志: {log_path}")

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
                "train_hypers": LOOP181_TRAIN_HYPERS,
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
            "seed": seed,
        },
        "train_hypers": LOOP181_TRAIN_HYPERS,
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
    print(f"参数量: {param_count:,} (扩容 {param_count / 1604677:.2f}x)")

    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{LOOP_ID} Phase A 训练")
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto",
        help="设备选择 (default: auto)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help=f"最大 epochs (default: {LOOP181_TRAIN_HYPERS['max_epochs']})",
    )
    parser.add_argument(
        "--seed", type=int, default=41,
        help="随机种子 (default: 41)",
    )
    args = parser.parse_args()

    receipt = run_loop181_phase_a(
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
