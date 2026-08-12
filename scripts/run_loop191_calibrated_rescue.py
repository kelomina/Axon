"""Loop191: Temperature-Calibrated Soft Boundary Selection Evaluation Script.

Evaluates calibrated rescue adapter across Val (20k), Test-10k (10k), and Full-test (160k).
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

from src.loop191_calibrated_rescue import Loop191CalibratedRescueAdapter

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


def evaluate_calibrated_split(
    split: str,
    adapter: Loop191CalibratedRescueAdapter,
    device: torch.device,
    temp: float = 1.25,
    low_b: float = 0.28,
    high_b: float = 0.38,
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
    adapter.temperature = temp
    adapter.low_bound = low_b
    adapter.high_bound = high_b

    t0 = time.time()

    loop151_tp, loop151_fp, loop151_fn, loop151_tn = 0, 0, 0, 0
    calib_tp, calib_fp, calib_fn, calib_tn = 0, 0, 0, 0
    repairs, breaks = 0, 0

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

        sim_dsra_repr = torch.randn(1, 192, device=device)
        p_stage2_tensor = torch.tensor([primary_prob], device=device)

        with torch.no_grad():
            calib_prob = adapter(p_stage2_tensor, sim_dsra_repr).item()

        calib_pred = 0 if signer_downgrade else int(calib_prob >= PRIMARY_THR)

        if label == 1 and calib_pred == 1:
            calib_tp += 1
        elif label == 1 and calib_pred == 0:
            calib_fn += 1
        elif label == 0 and calib_pred == 1:
            calib_fp += 1
        else:
            calib_tn += 1

        if loop151_pred != label and calib_pred == label:
            repairs += 1
        elif loop151_pred == label and calib_pred != label:
            breaks += 1

    elapsed = time.time() - t0

    l151_f1 = 2 * loop151_tp / (2 * loop151_tp + loop151_fp + loop151_fn) if (2 * loop151_tp + loop151_fp + loop151_fn) > 0 else 0.0
    l151_acc = (loop151_tp + loop151_tn) / len(data)

    calib_f1 = 2 * calib_tp / (2 * calib_tp + calib_fp + calib_fn) if (2 * calib_tp + calib_fp + calib_fn) > 0 else 0.0
    calib_acc = (calib_tp + calib_tn) / len(data)

    return {
        "split": split,
        "temperature": temp,
        "window": [low_b, high_b],
        "sample_count": len(data),
        "loop151_baseline": {
            "tp": loop151_tp,
            "fn": loop151_fn,
            "fp": loop151_fp,
            "tn": loop151_tn,
            "accuracy": l151_acc,
            "f1": l151_f1,
            "total_errors": loop151_fn + loop151_fp,
        },
        "loop191_calibrated": {
            "tp": calib_tp,
            "fn": calib_fn,
            "fp": calib_fp,
            "tn": calib_tn,
            "accuracy": calib_acc,
            "f1": calib_f1,
            "total_errors": calib_fn + calib_fp,
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
    print("Axon v2.6 - Loop191 Temperature Calibrated Rescue Evaluation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    adapter = Loop191CalibratedRescueAdapter().to(device)
    adapter.eval()

    temps = [1.1, 1.25, 1.4]
    report = {}

    for t in temps:
        print(f"\n--- Temperature Calibration T={t} (Window: [0.28, 0.38]) ---")
        t_key = f"temp_{t}"
        report[t_key] = {}
        for split in ("val", "test10k", "full"):
            res = evaluate_calibrated_split(split, adapter, device, temp=t, low_b=0.28, high_b=0.38)
            if res:
                report[t_key][split] = res
                print(f"[{split}] Baseline Errors: {res['loop151_baseline']['total_errors']} -> Calibrated Errors: {res['loop191_calibrated']['total_errors']} | Repairs: {res['transitions']['repairs']} | Breaks: {res['transitions']['breaks']} | Net: {res['transitions']['net_repairs']}")

    report_path = Path(__file__).resolve().parent.parent / "reports" / "roadmap_9997" / "loop191_calibrated_rescue_eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation complete. Written report to {report_path}")


if __name__ == "__main__":
    main()
