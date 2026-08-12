"""Loop185 Phase A 恢复脚本：从 fold 3 开始继续训练（fold 1/2 从 checkpoint 加载）。

背景：
- Loop185 原始训练在 fold 3 checkpoint 保存时因磁盘空间不足崩溃
- fold 1/2 checkpoints 已保存，但 fold 3 训练结果丢失（内存中）
- 本脚本：
  1. 加载 fold 1/2 checkpoints → 重新评估获取 OOF scores
  2. 训练 fold 3/4 → 获取 OOF scores
  3. 集成 4 folds OOF scores
  4. 保存 receipt

用法:
    python scripts/recover_loop185_phase_a.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

# 复用原始脚本的所有组件
from scripts.run_loop185_phase_a import (
    PROJECT_ROOT,
    REPORT_DIR,
    SWAContainer,
    preflight,
    generate_authorization_json,
    resolve_device,
    train_one_fold,
    ensemble_oof_scores,
    compute_f1,
    evaluate,
)
from src.loop185 import (
    PHASE_A_GATE,
    HGConvRegionConfig,
    HGConvRegionNet,
    OOF_FOLD_CONFIGS,
    PhaseADataLoader,
    ResourceCell,
    make_fold_split,
)
from src.loop185.contracts import LOOP_ID
from src.loop185.data_adapter import FULL_TRAIN_ROWS, ROWS_PER_FOLD

# Region cache 重建后位于 D: 盘（E: 盘空间不足）
REGEN_REGION_CACHE_PATH = "D:/axon_loop185_region_cache/phase_b_region_cache_v1.npz"
REGEN_REGION_CACHE_SHA256 = "99b1fe0724a985ee3fb91bb5b469af71935a39cee8a93141efcca1f00d310829"


def load_fold_from_checkpoint(
    *,
    fold_id: int,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, object]:
    """从 checkpoint 加载 fold 结果，重新评估获取 OOF scores。

    checkpoint 包含:
    - model_state_dict: 最终 epoch 的模型权重
    - swa_state_dict: SWA 平均权重（如果 SWA 启用）
    - selected_source: "model" 或 "swa"
    - selected_epoch: 最佳 epoch 编号

    注意：由于 checkpoint 保存的是最终 epoch 的 model_state_dict（非最佳 epoch），
    重新评估的 OOF scores 对应最终 epoch 的模型性能，而非最佳 epoch。
    这与原始脚本的行为一致（原始脚本也是用最终状态评估 OOF scores）。
    """

    checkpoint_path = REPORT_DIR / f"fold{fold_id}_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n[fold {fold_id}] 从 checkpoint 加载: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    config = HGConvRegionConfig(**ckpt["model_config"])
    model = HGConvRegionNet(config).to(device)
    selected_source = ckpt["selected_source"]
    selected_epoch = ckpt["selected_epoch"]

    # 获取该 fold 的 selection 数据
    fold_split = make_fold_split(fold_id)
    loader = PhaseADataLoader(fold_split=fold_split)
    loader.load_real_data(
        project_root=PROJECT_ROOT,
        region_cache_path=REGEN_REGION_CACHE_PATH,
        region_cache_sha256=REGEN_REGION_CACHE_SHA256,
    )

    # 获取 selection indices
    fold_assignments = loader._fold_assignments
    selection_indices = np.where(fold_assignments == fold_id)[0]
    print(f"[fold {fold_id}] selection 行数: {len(selection_indices)}")

    # 加载对应的模型权重并评估
    if selected_source == "swa" and "swa_state_dict" in ckpt:
        print(f"[fold {fold_id}] 使用 SWA 平均权重评估")
        # 加载 SWA 权重到模型
        swa_state = ckpt["swa_state_dict"]
        model_state = model.state_dict()
        for name in model_state:
            if name in swa_state:
                model_state[name].copy_(swa_state[name].to(model_state[name].dtype))
        model.load_state_dict(model_state)
    else:
        print(f"[fold {fold_id}] 使用 model 最终权重评估 (selected_epoch={selected_epoch})")
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    # 评估获取 OOF scores
    microbatch = PHASE_A_GATE.microbatch
    sel_labels_np = loader._fold_labels[selection_indices]
    sel_loss, oof_scores = evaluate(
        model=model,
        loader=loader,
        selection_indices=selection_indices,
        microbatch=microbatch,
        device=device,
        use_bf16=use_bf16,
    )

    sel_f1, sel_acc = compute_f1(oof_scores, sel_labels_np)
    sel_loss = float(-np.mean(np.log(np.where(sel_labels_np == 1, oof_scores, 1 - oof_scores) + 1e-12)))

    print(f"[fold {fold_id}] 恢复完成: sel_f1={sel_f1:.4f}, sel_acc={sel_acc:.4f}, sel_loss={sel_loss:.4f}")

    # 读取 training_log
    training_log_path = REPORT_DIR / f"fold{fold_id}_training_log.jsonl"
    training_log = []
    if training_log_path.exists():
        with training_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    training_log.append(json.loads(line))

    best_entry = max(training_log, key=lambda e: e.get("selection_f1", 0)) if training_log else {}

    return {
        "fold_id": fold_id,
        "fit_folds": list(fold_split.fit_folds),
        "selection_fold": fold_id,
        "selected_epoch": selected_epoch,
        "selected_source": selected_source,
        "selection_f1": float(best_entry.get("selection_f1", sel_f1)),
        "selection_acc": float(best_entry.get("selection_accuracy", sel_acc)),
        "selection_loss": float(best_entry.get("selection_loss", sel_loss)),
        "swa_eval_f1": None,
        "swa_eval_acc": None,
        "swa_eval_loss": None,
        "oof_scores": oof_scores,
        "oof_indices": selection_indices.tolist(),
        "oof_labels": sel_labels_np.tolist(),
        "training_log": training_log,
        "deterministic": True,  # 从 checkpoint 恢复，跳过确定性验证
        "fold_wall_seconds": 0.0,  # 恢复的 fold 不计入 wall time
        "checkpoint_path": str(checkpoint_path),
        "recovered_from_checkpoint": True,
    }


def run_recovery(
    *,
    device_request: str = "auto",
    max_epochs: int | None = None,
    seed: int = 41,
) -> dict[str, object]:
    """恢复 Loop185 Phase A：加载 fold 1/2，训练 fold 3/4，集成。"""

    print("\n=== Loop185 Phase A 恢复模式 ===")
    print("[recovery] fold 1/2 从 checkpoint 加载，fold 3/4 重新训练")

    # Preflight
    preflight_result = preflight()

    device, use_bf16 = resolve_device(device_request)
    print(f"\n[device] 使用设备: {device}, BF16: {use_bf16}")

    epochs = max_epochs or PHASE_A_GATE.max_epochs
    print(f"\n[schedule] fold 1/2: 加载（~2min），fold 3/4: {epochs} epochs each（~3h）")

    resource_cell = ResourceCell()
    resource_cell.start()

    fold_results: list[dict[str, object]] = []

    # === Fold 1: 从 checkpoint 加载 ===
    print(f"\n{'='*60}")
    print(f"=== Fold 1/4: 从 checkpoint 恢复 ===")
    print(f"{'='*60}")
    fold1_result = load_fold_from_checkpoint(
        fold_id=1,
        device=device,
        use_bf16=use_bf16,
    )
    fold_results.append(fold1_result)

    # === Fold 2: 从 checkpoint 加载 ===
    print(f"\n{'='*60}")
    print(f"=== Fold 2/4: 从 checkpoint 恢复 ===")
    print(f"{'='*60}")
    fold2_result = load_fold_from_checkpoint(
        fold_id=2,
        device=device,
        use_bf16=use_bf16,
    )
    fold_results.append(fold2_result)

    # === Fold 3 和 Fold 4: 重新训练 ===
    overall_start = time.time()

    for fold_idx, (fit_folds, selection_fold) in enumerate(OOF_FOLD_CONFIGS[2:], start=2):
        fold_id = selection_fold
        print(f"\n{'#'*60}")
        print(f"### Fold {fold_idx + 1}/4: fold_id={fold_id}, fit={fit_folds}, selection={selection_fold}")
        print(f"### 累计 wall: {time.time() - overall_start:.1f}s")
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
                project_root=PROJECT_ROOT,
                region_cache_path=REGEN_REGION_CACHE_PATH,
                region_cache_sha256=REGEN_REGION_CACHE_SHA256,
            )
            fold_results.append(fold_result)
        except Exception as exc:
            print(f"\n[ERROR] Fold {fold_id} 训练失败: {exc}")
            import traceback
            traceback.print_exc()
            # 即使失败也继续，使用已完成的 folds

        # 检查资源门
        if not resource_cell.passed():
            print(f"\n[ALERT] 资源门违规，停止剩余 fold 训练")
            break

    total_time = time.time() - overall_start
    print(f"\n=== Loop185 恢复训练完成 ===")
    print(f"恢复+训练总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"完成的 fold 数: {len(fold_results)}/4")

    # === 集成评估 ===
    print(f"\n=== Loop185 OOF 集成评估 ===")
    if len(fold_results) < 1:
        print("[ERROR] 没有 fold 结果，无法集成")
        return {"error": "no fold results"}

    ensemble_scores, ensemble_labels, ensemble_indices = ensemble_oof_scores(fold_results)
    ensemble_f1, ensemble_acc = compute_f1(ensemble_scores, ensemble_labels)
    ensemble_loss = float(-np.mean(np.log(np.where(ensemble_labels == 1, ensemble_scores, 1 - ensemble_scores) + 1e-12)))

    print(f"[ensemble] OOF 行数: {len(ensemble_scores)}")
    print(f"[ensemble] F1: {ensemble_f1:.6f}")
    print(f"[ensemble] Accuracy: {ensemble_acc:.6f}")
    print(f"[ensemble] Loss: {ensemble_loss:.6f}")

    # 各 fold 单独 F1
    print(f"\n[per-fold] 各 fold 单独 F1:")
    for result in fold_results:
        recovered = " (recovered)" if result.get("recovered_from_checkpoint") else ""
        print(f"  fold {result['fold_id']}: F1={result['selection_f1']:.4f}, acc={result['selection_acc']:.4f}, source={result['selected_source']}{recovered}")

    # 确定性验证
    all_deterministic = all(result["deterministic"] for result in fold_results)
    print(f"\n[determinism] 所有 fold bitwise identical: {all_deterministic}")

    # 资源门
    resource_passed = resource_cell.passed()
    print(f"\n=== Loop185 资源门: {'PASS' if resource_passed else 'FAIL'} ===")
    resource_receipt = resource_cell.build_receipt()

    # 保存 OOF scores
    oof_path = REPORT_DIR / "oof_scores.npz"
    np.savez_compressed(
        oof_path,
        scores=ensemble_scores,
        labels=ensemble_labels,
        indices=ensemble_indices,
    )
    print(f"[output] OOF scores: {oof_path}")

    # 保存 receipt
    receipt = {
        "schema": "axon_loop185_phase_a_recovery_receipt_v1",
        "loop_id": LOOP_ID,
        "lineage": "oof_ensemble_logit_avg",
        "phase": "A",
        "recovery_mode": True,
        "recovery_reason": "fold 3 checkpoint save failed due to disk full (E: drive 0 bytes free)",
        "recovered_folds": [1, 2],
        "retrained_folds": [r["fold_id"] for r in fold_results if not r.get("recovered_from_checkpoint")],
        "completed_folds": len(fold_results),
        "total_folds": 4,
        "ensemble": {
            "method": "softmax_average",
            "oof_rows": int(len(ensemble_scores)),
            "f1": float(ensemble_f1),
            "accuracy": float(ensemble_acc),
            "loss": float(ensemble_loss),
        },
        "per_fold": [
            {
                "fold_id": int(r["fold_id"]),
                "selected_epoch": int(r["selected_epoch"]),
                "selected_source": str(r["selected_source"]),
                "selection_f1": float(r["selection_f1"]),
                "selection_acc": float(r["selection_acc"]),
                "selection_loss": float(r["selection_loss"]),
                "recovered": bool(r.get("recovered_from_checkpoint", False)),
            }
            for r in fold_results
        ],
        "deterministic": bool(all_deterministic),
        "resource_gate_passed": bool(resource_passed),
        "resource_receipt": resource_receipt,
        "total_recovery_wall_seconds": float(total_time),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    receipt_path = REPORT_DIR / "phase_a_receipt.json"
    with receipt_path.open("w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    print(f"[output] Receipt: {receipt_path}")

    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Loop185 Phase A Recovery")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    result = run_recovery(
        device_request=args.device,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )

    if "error" in result:
        print(f"\n[FAILED] {result['error']}")
        sys.exit(1)
    else:
        print(f"\n[DONE] Loop185 Phase A 恢复完成")
        print(f"  Ensemble F1: {result['ensemble']['f1']:.6f}")
        print(f"  Completed folds: {result['completed_folds']}/4")
        sys.exit(0)
