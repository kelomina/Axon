"""Loop214: Second-Order Zero-Breaks Residual Stacker Evaluation Script.

Evaluates second-order residual stacker across Val (20k), Test-10k (10k), and Full-test (160k).
Generates reports/roadmap_9997/loop214_second_order_receipt.json.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loop214_second_order_stacker import Loop214SecondOrderStacker
from src.loop213_annealed_calibrator import Loop213AnnealedCalibrator

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


def evaluate_loop214_split(split: str, stacker: Loop214SecondOrderStacker, calibrator: Loop213AnnealedCalibrator):
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
    t0 = time.time()

    loop151_tp, loop151_fp, loop151_fn, loop151_tn = 0, 0, 0, 0
    stacker_tp, stacker_fp, stacker_fn, stacker_tn = 0, 0, 0, 0
    repairs, breaks = 0, 0

    torch.manual_seed(42)

    for sha, row in data.items():
        label = int(row.get("label", -1))
        if label not in (0, 1):
            continue

        primary_prob = float(row.get("stage2_prob_malicious", 0.0))
        signer_downgrade = row.get("trusted_signer_guard_downgrade", "").strip().lower() == "true"
        loop151_pred = 0 if signer_downgrade else int(primary_prob >= PRIMARY_THR)

        auth_status = row.get("auth_status", "")
        signer_subject = row.get("signer_subject", "")

        # Base metrics
        if label == 1 and loop151_pred == 1:
            loop151_tp += 1
        elif label == 1 and loop151_pred == 0:
            loop151_fn += 1
        elif label == 0 and loop151_pred == 1:
            loop151_fp += 1
        else:
            loop151_tn += 1

        calib_prob = calibrator.calibrate(primary_prob)
        adv_prob = primary_prob

        stacker_pred, _ = stacker(primary_prob, adv_prob, calib_prob, auth_status, signer_subject)

        # Stacker metrics
        if label == 1 and stacker_pred == 1:
            stacker_tp += 1
        elif label == 1 and stacker_pred == 0:
            stacker_fn += 1
        elif label == 0 and stacker_pred == 1:
            stacker_fp += 1
        else:
            stacker_tn += 1

        if loop151_pred != label and stacker_pred == label:
            repairs += 1
        elif loop151_pred == label and stacker_pred != label:
            breaks += 1

    elapsed = time.time() - t0

    l151_f1 = 2 * loop151_tp / (2 * loop151_tp + loop151_fp + loop151_fn) if (2 * loop151_tp + loop151_fp + loop151_fn) > 0 else 0.0
    stacker_f1 = 2 * stacker_tp / (2 * stacker_tp + stacker_fp + stacker_fn) if (2 * stacker_tp + stacker_fp + stacker_fn) > 0 else 0.0

    return {
        "split": split,
        "sample_count": len(data),
        "loop151_baseline": {
            "tp": loop151_tp,
            "fn": loop151_fn,
            "fp": loop151_fp,
            "tn": loop151_tn,
            "f1": l151_f1,
            "total_errors": loop151_fn + loop151_fp,
        },
        "loop214_second_order_stacker": {
            "tp": stacker_tp,
            "fn": stacker_fn,
            "fp": stacker_fp,
            "tn": stacker_tn,
            "f1": stacker_f1,
            "total_errors": stacker_fn + stacker_fp,
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
    print("Axon v2.6 - Loop214 Second-Order Zero-Breaks Residual Stacker Evaluation")
    print("=" * 70)

    stacker = Loop214SecondOrderStacker()
    calibrator = Loop213AnnealedCalibrator()
    report = {}

    for split in ("val", "test10k", "full"):
        res = evaluate_loop214_split(split, stacker, calibrator)
        if res:
            report[split] = res
            print(f"[{split}] Baseline Errors: {res['loop151_baseline']['total_errors']} -> Second-Order Errors: {res['loop214_second_order_stacker']['total_errors']} | Repairs: {res['transitions']['repairs']} | Breaks: {res['transitions']['breaks']} | Net Repairs: +{res['transitions']['net_repairs']} | Stacker F1: {res['loop214_second_order_stacker']['f1']:.6f}")

    report_path = Path(__file__).resolve().parent.parent / "reports" / "roadmap_9997" / "loop214_second_order_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation complete. Saved receipt to {report_path}")


if __name__ == "__main__":
    main()
