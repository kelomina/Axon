"""Loop185 OOF 集成失败原因分析。

分析内容:
1. 验证 oof_scores.npz 结构与 receipt 一致
2. 检查 fold 间预测相关性（高相关 = 集成收益低）
3. 分析错误样本分布（共同错误 vs 独立错误）
4. 对比多种集成方法：softmax avg / logit avg / majority vote / 加权
5. 检查是否某些 fold 应该被剔除（劣质 fold 拖累集成）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "roadmap_9997" / "loop185"


def compute_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """二分类 F1（scores > 0.5 为正）。"""
    preds = (scores > 0.5).astype(np.int32)
    labels = labels.astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall <= 0:
        return 0.0, float((preds == labels).mean())
    f1 = 2 * precision * recall / (precision + recall)
    acc = float((preds == labels).mean())
    return f1, acc


def main() -> int:
    oof_path = REPORT_DIR / "oof_scores.npz"
    if not oof_path.exists():
        print(f"[ERROR] OOF scores not found: {oof_path}")
        return 1

    data = np.load(oof_path, allow_pickle=True)
    print(f"[INFO] OOF npz keys: {list(data.keys())}")
    for key in data.keys():
        arr = data[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    # 兼容两种 key 命名
    ensemble_scores = data["ensemble_scores"] if "ensemble_scores" in data.keys() else data["scores"]
    ensemble_labels = data["ensemble_labels"] if "ensemble_labels" in data.keys() else data["labels"]
    ensemble_indices = data["ensemble_indices"] if "ensemble_indices" in data.keys() else data["indices"]

    n = ensemble_scores.shape[0]
    print(f"\n[INFO] OOF rows: {n}")
    print(f"[INFO] Label distribution: pos={int(ensemble_labels.sum())}, neg={int((ensemble_labels == 0).sum())}")
    print(f"[INFO] Indices range: [{ensemble_indices.min()}, {ensemble_indices.max()}]")

    # --- 1. 验证 ensemble F1 ---
    ens_f1, ens_acc = compute_f1(ensemble_scores, ensemble_labels)
    print(f"\n[1] Ensemble (softmax avg) F1={ens_f1:.6f}, Acc={ens_acc:.6f}")

    # --- 2. 检查是否有 per-fold scores ---
    has_per_fold = False
    per_fold_keys = [k for k in data.keys() if k.startswith("fold") and "scores" in k]
    if per_fold_keys:
        has_per_fold = True
        print(f"\n[2] Found per-fold scores: {per_fold_keys}")
    else:
        print(f"\n[2] No per-fold scores in npz, only ensemble available")
        print("    Cannot analyze fold correlation from this file alone.")

    # --- 3. 重新加载每个 fold checkpoint 评估 OOF ---
    # 由于 npz 只有 ensemble，需要从 checkpoints 重新评估
    # 但这里先做错误样本分析
    print(f"\n[3] Error pattern analysis on ensemble:")

    preds = (ensemble_scores > 0.5).astype(np.int32)
    errors = preds != ensemble_labels.astype(np.int32)
    n_errors = int(errors.sum())
    print(f"    Total errors: {n_errors}/{n} = {n_errors/n*100:.2f}%")

    # 分数分布
    pos_scores = ensemble_scores[ensemble_labels == 1]
    neg_scores = ensemble_scores[ensemble_labels == 0]
    print(f"    Positive samples: mean={pos_scores.mean():.4f}, std={pos_scores.std():.4f}")
    print(f"    Negative samples: mean={neg_scores.mean():.4f}, std={neg_scores.std():.4f}")

    # 边界附近的错误（0.3-0.7）
    boundary_mask = (ensemble_scores > 0.3) & (ensemble_scores < 0.7)
    boundary_errors = errors & boundary_mask
    print(f"    Boundary errors (0.3<scores<0.7): {int(boundary_errors.sum())}/{n_errors} = {boundary_errors.sum()/max(n_errors,1)*100:.1f}%")

    # 极端错误（score>0.8 但 label=0, 或 score<0.2 但 label=1）
    extreme_fp = (ensemble_scores > 0.8) & (ensemble_labels == 0)
    extreme_fn = (ensemble_scores < 0.2) & (ensemble_labels == 1)
    print(f"    Extreme FP (score>0.8, neg): {int(extreme_fp.sum())}")
    print(f"    Extreme FN (score<0.2, pos): {int(extreme_fn.sum())}")

    # 错误分数分布
    err_scores = ensemble_scores[errors]
    print(f"    Error scores: min={err_scores.min():.4f}, max={err_scores.max():.4f}, mean={err_scores.mean():.4f}")

    # --- 4. 尝试不同阈值 ---
    print(f"\n[4] Threshold sweep:")
    best_f1 = 0.0
    best_thr = 0.5
    for thr in np.arange(0.1, 0.9, 0.02):
        preds_t = (ensemble_scores > thr).astype(np.int32)
        tp = int(((preds_t == 1) & (ensemble_labels == 1)).sum())
        fp = int(((preds_t == 1) & (ensemble_labels == 0)).sum())
        fn = int(((preds_t == 0) & (ensemble_labels == 1)).sum())
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    print(f"    Best threshold: {best_thr:.2f}, F1={best_f1:.6f}")

    # --- 5. 结论 ---
    print(f"\n[5] Summary:")
    print(f"    Ensemble F1: {ens_f1:.6f}")
    print(f"    Best-threshold F1: {best_f1:.6f} (thr={best_thr:.2f})")
    print(f"    Total errors: {n_errors}")
    print(f"    Boundary errors: {int(boundary_errors.sum())} ({boundary_errors.sum()/max(n_errors,1)*100:.1f}%)")
    print(f"    Extreme errors: {int((extreme_fp | extreme_fn).sum())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
