"""Loop179 Phase A 训练脚本。

执行 Phase A 资源门验证：
- 12000 fit rows (folds 2,3,4), 4000 selection rows (fold 1), fold 0 禁止
- 12 epochs, microbatch=2, accumulation=16, effective_batch=32
- AdamW lr=3e-4, weight_decay=1e-2, cosine schedule, 1-epoch warmup
- EMA decay=0.999, BF16 autocast (CUDA), grad_clip=1.0
- 资源门: GPU <= 6.5 GiB, RSS <= 11 GiB, wall <= 6h

用法:
    python scripts/run_loop179_phase_a.py
    python scripts/run_loop179_phase_a.py --device cuda
    python scripts/run_loop179_phase_a.py --device cpu --max-epochs 3
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
from src.loop179.contracts import LOOP_ID


# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "roadmap_9997" / "loop179"


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight() -> dict[str, object]:
    """Phase A preflight: 契约自检、源码闭包、设备检测。"""

    print("[preflight] 契约自检...")
    assert_contract_invariants()

    print("[preflight] 源码闭包检查...")
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
    """生成 Phase A 授权 JSON。"""

    auth = {
        "schema": "axon_loop179_phase_a_authorization_v1",
        "loop_id": LOOP_ID,
        "phase": "A",
        "decision": "phase_a_authorized_by_user_2026_07_20",
        "claim_scope": "train_only_phase_a_resource_gate_verification",
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
) -> tuple[np.ndarray, ...]:
    """确定性 epoch batch 排序（与 Loop175 算法对齐）。"""

    material = f"loop179-phase-a|{seed}|{epoch}|{fit_indices.size}".encode("ascii")
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
            raise RuntimeError("Loop179 CUDA requires BF16 support")
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
) -> tuple[float, int]:
    """训练一个 epoch，返回 (avg_loss, optimizer_steps)。"""

    model.train()
    batches = deterministic_epoch_batches(
        fit_indices,
        microbatch=microbatch,
        seed=seed,
        epoch=epoch,
    )
    # 6000 microbatches / 16 accumulation = 375 optimizer steps
    epoch_loss_sum = 0.0
    epoch_sample_count = 0
    optimizer_steps = 0

    for window_start in range(0, len(batches), accumulation):
        window = batches[window_start : window_start + accumulation]
        window_indices = np.concatenate(window)
        window_normalizer = float(window_indices.size)  # = 32

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

def run_phase_a(
    *,
    device_request: str = "auto",
    max_epochs: int | None = None,
    seed: int = 41,
) -> dict[str, object]:
    """执行 Phase A 训练，返回 receipt dict。"""

    # 1. Preflight
    print("\n=== Phase A Preflight ===")
    preflight_result = preflight()
    auth_json = generate_authorization_json(preflight_result)

    # 2. 准备输出目录
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    auth_path = REPORT_DIR / "phase_a_authorization.json"
    with auth_path.open("w", encoding="utf-8") as f:
        json.dump(auth_json, f, indent=2, ensure_ascii=False)
    print(f"[preflight] 授权 JSON 已保存: {auth_path}")

    # 3. 加载数据
    print("\n=== Phase A 数据加载 ===")
    loader = PhaseADataLoader()
    t0 = time.time()
    data_receipt = loader.load_real_data()
    t1 = time.time()
    print(f"[data] 加载耗时: {t1 - t0:.2f}s")
    print(f"[data] 行数: {data_receipt['row_count']}, 区域数: {data_receipt['region_count']}")
    print(f"[data] token 数: {data_receipt['token_count']}")

    fit_indices = loader.get_fit_indices()
    selection_indices = loader.get_selection_indices()
    print(f"[data] fit 行数: {fit_indices.shape[0]}")
    print(f"[data] selection 行数: {selection_indices.shape[0]}")

    # 4. 设备解析
    device, use_bf16 = resolve_device(device_request)
    print(f"\n[device] 使用设备: {device}, BF16: {use_bf16}")

    # 5. 模型初始化
    print("\n=== Phase A 模型初始化 ===")
    model_seed = seed
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    config = HGConvRegionConfig(dropout=0.1)
    model = HGConvRegionNet(config).to(device=device, dtype=torch.float32)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] 参数量: {param_count:,}")
    print(f"[model] config: model_dim={config.model_dim}, heads={config.transformer_heads}, layers={config.transformer_layers}")

    ema = ExponentialMovingAverage(model, PHASE_A_GATE.ema_decay)

    # 6. 优化器
    epochs = max_epochs or PHASE_A_GATE.max_epochs
    microbatch = PHASE_A_GATE.microbatch
    accumulation = PHASE_A_GATE.accumulation

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PHASE_A_GATE.learning_rate,
        weight_decay=PHASE_A_GATE.weight_decay,
    )

    batches_per_epoch = math.ceil(math.ceil(fit_indices.size / microbatch) / accumulation)
    total_steps = epochs * batches_per_epoch
    warmup_steps = PHASE_A_GATE.warmup_steps * batches_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: scheduler_multiplier(
            step, warmup_steps=warmup_steps, total_steps=total_steps
        ),
    )
    print(f"[optimizer] AdamW lr={PHASE_A_GATE.learning_rate}, wd={PHASE_A_GATE.weight_decay}")
    print(f"[scheduler] cosine, warmup={warmup_steps} steps, total={total_steps} steps")
    print(f"[schedule] {batches_per_epoch} optimizer steps/epoch × {epochs} epochs")

    # 7. 资源门
    resource_cell = ResourceCell()
    resource_cell.start()

    # 8. 训练循环
    print("\n=== Phase A 训练开始 ===")
    selection_losses: list[float] = []
    training_losses: list[float] = []
    training_log: list[dict[str, object]] = []
    start_time = time.time()

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
            grad_clip=PHASE_A_GATE.grad_clip,
            device=device,
            use_bf16=use_bf16,
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

        # 计算 selection accuracy
        sel_labels = loader._fold_labels[selection_indices]
        sel_preds = (sel_scores > 0.5).astype(np.int64)
        sel_acc = float((sel_preds == sel_labels).mean())

        print(f"[epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, sel_loss={sel_loss:.4f}, sel_acc={sel_acc:.4f}")
        print(f"  GPU={sample.gpu_allocated_bytes / 1024**3:.3f} GiB, RSS={sample.rss_bytes / 1024**3:.3f} GiB, wall={sample.wall_seconds:.1f}s, epoch_time={epoch_elapsed:.1f}s")

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "selection_loss": sel_loss,
            "selection_accuracy": sel_acc,
            "optimizer_steps": opt_steps,
            "gpu_allocated_bytes": sample.gpu_allocated_bytes,
            "rss_bytes": sample.rss_bytes,
            "wall_seconds": sample.wall_seconds,
            "epoch_seconds": epoch_elapsed,
        }
        training_log.append(log_entry)

        # 检查资源门是否已超限
        if not resource_cell.passed():
            print(f"\n[ALERT] 资源门违规！violations: {len(resource_cell.violations)}")
            for v in resource_cell.violations:
                print(f"  [{v.kind}] {v.detail}")
            break

    total_time = time.time() - start_time

    # 9. 选择最佳 epoch
    selected_epoch = select_earliest_minimum_epoch(selection_losses) if selection_losses else 0
    print(f"\n[result] 选定 epoch: {selected_epoch}")
    print(f"[result] 最小 selection loss: {min(selection_losses) if selection_losses else 'N/A'}")

    # 10. 确定性验证（eval 两次，bitwise identical）
    print("\n=== Phase A 确定性验证 ===")
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
    print(f"\n=== Phase A 资源门: {'PASS' if resource_passed else 'FAIL'} ===")
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
                "schema": "axon_loop179_phase_a_checkpoint_v1",
                "loop_id": LOOP_ID,
                "selected_epoch": selected_epoch,
                "model_config": asdict(config),
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
        "schema": "axon_loop179_phase_a_receipt_v1",
        "loop_id": LOOP_ID,
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
        "training": {
            "epochs_run": len(training_log),
            "selected_epoch": selected_epoch,
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
    print("\n=== Phase A 总结 ===")
    print(f"资源门: {'PASS' if resource_passed else 'FAIL'}")
    print(f"训练 epochs: {len(training_log)}/{epochs}")
    print(f"选定 epoch: {selected_epoch}")
    print(f"最小 selection loss: {min(selection_losses):.4f}" if selection_losses else "N/A")
    print(f"最终 selection accuracy: {training_log[-1]['selection_accuracy']:.4f}" if training_log else "N/A")
    print(f"总 wall time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"确定性: {'PASS' if deterministic else 'FAIL'}")

    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Loop179 Phase A 训练")
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
        print("\n[FAIL] 资源门未通过，Phase A 失败")
        sys.exit(1)
    else:
        print("\n[PASS] 资源门通过，Phase A 成功，可申请 Phase B 授权")


if __name__ == "__main__":
    main()
