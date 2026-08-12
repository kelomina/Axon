"""Loop207: Supervised Contrastive Hard Example Projection Training Script.

Trains contrastive projection network using SupConLoss (temperature=0.07).
Generates reports/roadmap_9997/loop207_contrastive_receipt.json.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loop207_contrastive_projection import Loop207ContrastiveProjection, SupConLoss


def run_loop207():
    print("=" * 70)
    print("Axon v2.6 - Loop207 Supervised Contrastive Hard Example Projection Training")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(42)

    batch_size = 64
    in_dim = 256
    proj_dim = 128
    num_fit = 12000
    epochs = 10

    model = Loop207ContrastiveProjection(in_dim=in_dim, proj_dim=proj_dim).to(device)
    criterion = SupConLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    print(f"[Model] Initialized Loop207ContrastiveProjection | Params: {sum(p.numel() for p in model.parameters()):,}")

    fit_x = torch.randn(num_fit, in_dim, device="cpu")
    fit_y = torch.randint(0, 2, (num_fit,), device="cpu", dtype=torch.long)

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = num_fit // batch_size

        for i in range(total_batches):
            idx = i * batch_size
            bx = fit_x[idx : idx + batch_size].to(device)
            by = fit_y[idx : idx + batch_size].to(device)

            optimizer.zero_grad()
            proj = model(bx)
            loss = criterion(proj, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / total_batches
        print(f"Epoch {epoch:02d}/{epochs:02d} | SupCon Loss: {avg_loss:.4f}")

    elapsed = time.time() - t0

    ckpt_path = Path(__file__).resolve().parent.parent / "models" / "loop207_contrastive_projection.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    print(f"\n[Loop207] Completed in {elapsed:.2f}s | Final Loss: {avg_loss:.4f}")

    receipt = {
        "schema": "axon_loop207_contrastive_receipt_v1",
        "loop_id": "Loop207",
        "model_architecture": "Loop207ContrastiveProjection",
        "loss_function": "SupConLoss(temperature=0.07)",
        "training": {
            "epochs": epochs,
            "fit_rows": num_fit,
            "final_loss": avg_loss,
            "elapsed_seconds": round(elapsed, 2),
            "device": str(device),
        },
    }

    report_path = Path(__file__).resolve().parent.parent / "reports" / "roadmap_9997" / "loop207_contrastive_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"Saved receipt to {report_path}")


if __name__ == "__main__":
    run_loop207()
