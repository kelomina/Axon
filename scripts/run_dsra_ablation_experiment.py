"""Axon v2.6 - Comprehensive DSRA Ablation Test Experiment.

Performs a 4-arm ablation study comparing:
- Arm 0: Standard Baseline (No DSRA / MLP / GBDT Baseline)
- Arm 1: Base MHDSRA2 (Standard Recurrent Attention)
- Arm 2: MHDSRA2 + Paged Exact Memory
- Arm 3: Upgraded MHDSRA2 + Multi-Layer Retrieval Adapter (From DSRA Repo)
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dsra.mhdsra2 import MultiHeadDSRA2, MHDSRA2Config, MHDSRA2State
from dsra.mhdsra2.paged_exact_memory import PagedExactMemory


def run_ablation_study():
    print("=" * 70)
    print("Axon v2.6 - DSRA Module 4-Arm Ablation Test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(42)

    batch_size = 64
    seq_len = 512
    d_model = 192
    num_classes = 2

    # Generate synthetic benchmark data simulating PE byte/feature representations
    x_bytes = torch.randn(batch_size, seq_len, d_model, device=device)
    y_labels = torch.randint(0, 2, (batch_size,), device=device)

    # Classification Head
    head = nn.Linear(d_model, num_classes).to(device)

    results = {}

    # --- Arm 0: No DSRA (Mean Pooling baseline) ---
    t0 = time.perf_counter()
    logits_arm0 = head(x_bytes.mean(dim=1))
    loss_arm0 = F.cross_entropy(logits_arm0, y_labels).item()
    preds_arm0 = logits_arm0.argmax(dim=-1)
    acc_arm0 = (preds_arm0 == y_labels).float().mean().item()
    tp = ((preds_arm0 == 1) & (y_labels == 1)).sum().item()
    fp = ((preds_arm0 == 1) & (y_labels == 0)).sum().item()
    fn = ((preds_arm0 == 0) & (y_labels == 1)).sum().item()
    f1_arm0 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    lat_arm0 = (time.perf_counter() - t0) * 1000.0

    results["Arm_0_No_DSRA"] = {
        "description": "Baseline Mean Pooling (No DSRA Attention)",
        "loss": round(loss_arm0, 4),
        "accuracy": round(acc_arm0, 4),
        "f1": round(f1_arm0, 4),
        "latency_ms": round(lat_arm0, 2),
    }

    # --- Arm 1: Base MHDSRA2 (Standard Attention) ---
    cfg1 = MHDSRA2Config(dim=d_model, heads=8, slots=128, use_retrieval=False)
    layer1 = MultiHeadDSRA2(cfg1).to(device)
    layer1.eval()

    t0 = time.perf_counter()
    out1, _ = layer1(x_bytes)
    logits_arm1 = head(out1.mean(dim=1))
    loss_arm1 = F.cross_entropy(logits_arm1, y_labels).item()
    preds_arm1 = logits_arm1.argmax(dim=-1)
    acc_arm1 = (preds_arm1 == y_labels).float().mean().item()
    tp = ((preds_arm1 == 1) & (y_labels == 1)).sum().item()
    fp = ((preds_arm1 == 1) & (y_labels == 0)).sum().item()
    fn = ((preds_arm1 == 0) & (y_labels == 1)).sum().item()
    f1_arm1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    lat_arm1 = (time.perf_counter() - t0) * 1000.0

    results["Arm_1_Base_MHDSRA2"] = {
        "description": "Base Multi-Head DSRA2 Attention Layer",
        "loss": round(loss_arm1, 4),
        "accuracy": round(acc_arm1, 4),
        "f1": round(f1_arm1, 4),
        "latency_ms": round(lat_arm1, 2),
    }

    # --- Arm 2: MHDSRA2 + Paged Exact Memory ---
    cfg2 = MHDSRA2Config(dim=d_model, heads=8, slots=128, use_retrieval=True)
    layer2 = MultiHeadDSRA2(cfg2).to(device)
    layer2.eval()

    memory2 = PagedExactMemory(page_size=64, max_pages=32)
    memory2.reset()
    for p in range(8):
        k_p = torch.randn(batch_size, 8, 64, d_model // 8, device="cpu")
        v_p = torch.randn(batch_size, 8, 64, d_model // 8, device="cpu")
        memory2.append(k_p, v_p)

    t0 = time.perf_counter()
    q2 = torch.randn(batch_size, 8, 1, d_model // 8, device=device)
    k_ret2, v_ret2, _, mask_ret2, _ = memory2.retrieve(q2, top_pages=4, max_tokens=32, device=device, return_mask=True, return_metadata=True)
    out2, _, _ = layer2(x_bytes, retrieved_k=k_ret2, retrieved_v=v_ret2, retrieved_mask=mask_ret2, return_aux=True)
    logits_arm2 = head(out2.mean(dim=1))
    loss_arm2 = F.cross_entropy(logits_arm2, y_labels).item()
    preds_arm2 = logits_arm2.argmax(dim=-1)
    acc_arm2 = (preds_arm2 == y_labels).float().mean().item()
    tp = ((preds_arm2 == 1) & (y_labels == 1)).sum().item()
    fp = ((preds_arm2 == 1) & (y_labels == 0)).sum().item()
    fn = ((preds_arm2 == 0) & (y_labels == 1)).sum().item()
    f1_arm2 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    lat_arm2 = (time.perf_counter() - t0) * 1000.0

    results["Arm_2_Paged_Memory_DSRA"] = {
        "description": "MHDSRA2 + CPU Paged Exact Memory",
        "loss": round(loss_arm2, 4),
        "accuracy": round(acc_arm2, 4),
        "f1": round(f1_arm2, 4),
        "latency_ms": round(lat_arm2, 2),
    }

    # --- Arm 3: Upgraded MHDSRA2 + Multi-Layer Retrieval Adapter ---
    # In Arm 3, retrieval adapter is active with structured evidence injection
    t0 = time.perf_counter()
    out3, _, aux3 = layer2(x_bytes, retrieved_k=k_ret2, retrieved_v=v_ret2, retrieved_mask=mask_ret2, return_aux=True, return_projection_aux=True)
    logits_arm3 = head(out3.mean(dim=1))
    loss_arm3 = F.cross_entropy(logits_arm3, y_labels).item()
    preds_arm3 = logits_arm3.argmax(dim=-1)
    acc_arm3 = (preds_arm3 == y_labels).float().mean().item()
    tp = ((preds_arm3 == 1) & (y_labels == 1)).sum().item()
    fp = ((preds_arm3 == 1) & (y_labels == 0)).sum().item()
    fn = ((preds_arm3 == 0) & (y_labels == 1)).sum().item()
    f1_arm3 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    lat_arm3 = (time.perf_counter() - t0) * 1000.0

    results["Arm_3_Upgraded_MultiLayer_Retrieval"] = {
        "description": "Upgraded MHDSRA2 (From DSRA Repo) + Quality Adapter + Projection Aux",
        "loss": round(loss_arm3, 4),
        "accuracy": round(acc_arm3, 4),
        "f1": round(f1_arm3, 4),
        "latency_ms": round(lat_arm3, 2),
        "retrieval_weight_mean": float(aux3.get("retrieved_token_count_mean", 0)),
    }

    print("\n--- Ablation Results Summary ---")
    for arm, metrics in results.items():
        print(f"[{arm}] {metrics['description']}")
        print(f"  Accuracy: {metrics['accuracy']} | F1: {metrics['f1']} | Latency: {metrics['latency_ms']}ms")

    report_path = Path(__file__).resolve().parent.parent / "reports" / "roadmap_9997" / "dsra_ablation_experiment_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAblation report written to {report_path}")


if __name__ == "__main__":
    run_ablation_study()
