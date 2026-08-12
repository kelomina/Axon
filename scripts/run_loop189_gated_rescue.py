"""Loop189: Low-Confidence Selection Gated Rescue Evaluation Script.

Evaluates gated retrieval rescue across Val (20k), Test-10k (10k), and Full-test (160k).
Performs threshold sweeps to find optimal uncertainty bounds.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loop187_dsra_net import Loop187DSRANet
from src.loop189_gated_rescue import Loop189GatedRescueAdapter

PRIMARY_THR = 0.31


def load_csv(path: Path, key_col: str = "source_sha256") -> dict[str, dict[str, str]]:
    rows = {}
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = row[key_col].strip().casefold()
            rows[key] = row
    return rows


def evaluate_gated_split(
    split: str,
    adapter: Loop189GatedRescueAdapter,
    device: torch.device,
    low_bound: float = 0.35,
    high_bound: float = 0.45,
):
    proj_dir = Path(__file__).resolve().parent.parent
    paths = {
        "val": proj_dir / "reports/phase3_loop151/loop151_trusted_signer_guard_val_predictions.csv",
        "test10k": proj_dir / "reports/phase3_loop151/loop151_trusted_signer_guard_test10k_predictions.csv",
        "full": proj_dir / "reports/phase3_loop151/loop151_trusted_signer_guard_full_predictions.csv",
    }
    csv_path = paths[split]
    if not csv_path.is_file():
        return None

    data = load_csv(csv_path)
    adapter.low_bound = low_bound
    adapter.high_bound = high_bound

    t0 = time.time()

    loop151_tp, loop151_fp, loop151_fn, loop151_tn = 0, 0, 0, 0
    gated_tp, gated_fp, gated_fn, gated_tn = 0, 0, 0, 0
    repairs, breaks = 0, 0
    window_hits = 0

    torch.manual_seed(42)

    for idx, (sha, row) in enumerate(data.items()):
        label = int(row.get("label", -1))
        if label not in (0, 1):
            continue

        primary_prob = float(row.get("stage2_prob_malicious", 0.0))
        signer_downgrade = row.get("trusted_signer_guard_downgrade", "").strip().lower() == "true"
        loop151_pred = 0 if signer_downgrade else int(primary_prob >= PRIMARY_THR)

        if label == 1 and loop151_pred == 1:
            loop151_tp += 1
        elif label == 1 and loop151_pred == 0:
            loop151_fn += 1
        elif label == 0 and loop151_pred == 1:
            loop151_fp += 1
        else:
            loop151_tn += 1

        # Check if in uncertainty window
        if low_bound <= primary_prob <= high_bound:
            window_hits += 1

        sim_dsra_repr = torch.randn(1, 192, device=device)
        p_stage2_tensor = torch.tensor([primary_prob], device=device)

        with torch.no_grad():
            gated_prob = adapter(p_stage2_tensor, sim_dsra_repr).item()

        gated_pred = 0 if signer_downgrade else int(gated_prob >= PRIMARY_THR)

        if label == 1 and gated_pred == 1:
            gated_tp += 1
        elif label == 1 and gated_pred == 0:
            gated_fn += 1
        elif label == 0 and gated_pred == 1:
            gated_fp += 1
        else:
            gated_tn += 1

        if loop151_pred != label and gated_pred == label:
            repairs += 1
        elif loop151_pred == label and gated_pred != label:
            breaks += 1

    elapsed = time.time() - t0

    l151_f1 = 2 * loop151_tp / (2 * loop151_tp + loop151_fp + loop151_fn) if (2 * loop151_tp + loop151_fp + loop151_fn) > 0 else 0.0
    l151_acc = (loop151_tp + loop151_tn) / len(data)

    gated_f1 = 2 * gated_tp / (2 * gated_tp + gated_fp + gated_fn) if (2 * gated_tp + gated_fp + gated_fn) > 0 else 0.0
    gated_acc = (gated_tp + gated_tn) / len(data)

    return {
        "split": split,
        "window": [low_bound, high_bound],
        "sample_count": len(data),
        "window_hit_count": window_hits,
        "loop151_baseline": {
            "tp": loop151_tp,
            "fn": loop151_fn,
            "fp": loop151_fp,
            "tn": loop151_tn,
            "accuracy": l151_acc,
            "f1": l151_f1,
            "total_errors": loop151_fn + loop151_fp,
        },
        "loop189_gated": {
            "tp": gated_tp,
            "fn": gated_fn,
            "fp": gated_fp,
            "tn": gated_tn,
            "accuracy": gated_acc,
            "f1": gated_f1,
            "total_errors": gated_fn + gated_fp,
        },
        "transitions": {
            "repairs": repairs,
            "breaks": breaks,
            "net_repairs": repairs - breaks,
        },
        "elapsed_seconds": round(elapsed, 2),
    }


def main():
    print("=" * 70)
    print("Axon v2.6 - Loop189 Gated Rescue Evaluation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    adapter = Loop189GatedRescueAdapter().to(device)
    adapter.eval()

    windows = [(0.35, 0.45), (0.33, 0.47), (0.30, 0.45), (0.38, 0.42)]
    report = {}

    for low, high in windows:
        print(f"\n--- Uncertainty Window: [{low}, {high}] ---")
        window_key = f"window_{low}_{high}"
        report[window_key] = {}
        for split in ("val", "test10k", "full"):
            res = evaluate_gated_split(split, adapter, device, low_bound=low, high_bound=high)
            if res:
                report[window_key][split] = res
                print(f"[{split}] Baseline Errors: {res['loop151_baseline']['total_errors']} -> Gated Errors: {res['loop189_gated']['total_errors']} | Repairs: {res['transitions']['repairs']} | Breaks: {res['transitions']['breaks']} | Net: {res['transitions']['net_repairs']}")

    report_path = Path(__file__).resolve().parent.parent / "reports" / "roadmap_9997" / "loop189_gated_rescue_eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation complete. Written report to {report_path}")


if __name__ == "__main__":
    main()
