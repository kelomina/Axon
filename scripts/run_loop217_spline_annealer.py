"""Loop217: Dynamic Cubic Polynomial Spline Annealer Evaluation Script.

Evaluates cubic spline probability annealing near PRIMARY_THR = 0.31.
Generates reports/roadmap_9997/loop217_spline_receipt.json.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loop217_spline_annealer import Loop217SplineAnnealer

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


def evaluate_spline_split(split: str, annealer: Loop217SplineAnnealer):
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

    base_tp, base_fp, base_fn, base_tn = 0, 0, 0, 0
    spline_tp, spline_fp, spline_fn, spline_tn = 0, 0, 0, 0
    repairs, breaks = 0, 0

    for sha, row in data.items():
        label = int(row.get("label", -1))
        if label not in (0, 1):
            continue

        primary_prob = float(row.get("stage2_prob_malicious", 0.0))
        signer_downgrade = row.get("trusted_signer_guard_downgrade", "").strip().lower() == "true"
        base_pred = 0 if signer_downgrade else int(primary_prob >= PRIMARY_THR)

        spline_prob = annealer.anneal(primary_prob)
        spline_pred = 0 if signer_downgrade else int(spline_prob >= PRIMARY_THR)

        # Base metrics
        if label == 1 and base_pred == 1:
            base_tp += 1
        elif label == 1 and base_pred == 0:
            base_fn += 1
        elif label == 0 and base_pred == 1:
            base_fp += 1
        else:
            base_tn += 1

        # Spline metrics
        if label == 1 and spline_pred == 1:
            spline_tp += 1
        elif label == 1 and spline_pred == 0:
            spline_fn += 1
        elif label == 0 and spline_pred == 1:
            spline_fp += 1
        else:
            spline_tn += 1

        if base_pred != label and spline_pred == label:
            repairs += 1
        elif base_pred == label and spline_pred != label:
            breaks += 1

    elapsed = time.time() - t0

    base_f1 = 2 * base_tp / (2 * base_tp + base_fp + base_fn) if (2 * base_tp + base_fp + base_fn) > 0 else 0.0
    spline_f1 = 2 * spline_tp / (2 * spline_tp + spline_fp + spline_fn) if (2 * spline_tp + spline_fp + spline_fn) > 0 else 0.0

    return {
        "split": split,
        "sample_count": len(data),
        "base_metrics": {
            "tp": base_tp,
            "fn": base_fn,
            "fp": base_fp,
            "tn": base_tn,
            "f1": base_f1,
            "total_errors": base_fn + base_fp,
        },
        "spline_metrics": {
            "tp": spline_tp,
            "fn": spline_fn,
            "fp": spline_fp,
            "tn": spline_tn,
            "f1": spline_f1,
            "total_errors": spline_fn + spline_fp,
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
    print("Axon v2.6 - Loop217 Dynamic Cubic Polynomial Spline Annealing Evaluation")
    print("=" * 70)

    annealer = Loop217SplineAnnealer()
    report = {}

    for split in ("val", "test10k", "full"):
        res = evaluate_spline_split(split, annealer)
        if res:
            report[split] = res
            print(f"[{split}] Base Errors: {res['base_metrics']['total_errors']} -> Spline Errors: {res['spline_metrics']['total_errors']} | Repairs: {res['transitions']['repairs']} | Breaks: {res['transitions']['breaks']} | Net: {res['transitions']['net_repairs']}")

    report_path = Path(__file__).resolve().parent.parent / "reports" / "roadmap_9997" / "loop217_spline_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation complete. Saved receipt to {report_path}")


if __name__ == "__main__":
    main()
