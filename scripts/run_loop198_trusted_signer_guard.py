"""Loop198: Authenticode Trusted Signer Guard Evaluation Script.

Evaluates trusted signer guard extension across Val (20k), Test-10k (10k), and Full-test (160k).
Generates reports/roadmap_9997/loop198_trusted_signer_guard_receipt.json.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loop198_trusted_signer_guard import Loop198TrustedSignerGuard

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


def evaluate_signer_split(split: str, guard: Loop198TrustedSignerGuard):
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
    guard_tp, guard_fp, guard_fn, guard_tn = 0, 0, 0, 0
    downgrades = 0
    repairs, breaks = 0, 0

    for sha, row in data.items():
        label = int(row.get("label", -1))
        if label not in (0, 1):
            continue

        primary_prob = float(row.get("stage2_prob_malicious", 0.0))
        base_pred = int(primary_prob >= PRIMARY_THR)

        auth_status = row.get("auth_status", "")
        signer_subject = row.get("signer_subject", "")

        guard_pred, is_down = guard.evaluate_sample(base_pred, auth_status, signer_subject)
        if is_down:
            downgrades += 1

        # Base metrics
        if label == 1 and base_pred == 1:
            base_tp += 1
        elif label == 1 and base_pred == 0:
            base_fn += 1
        elif label == 0 and base_pred == 1:
            base_fp += 1
        else:
            base_tn += 1

        # Guard metrics
        if label == 1 and guard_pred == 1:
            guard_tp += 1
        elif label == 1 and guard_pred == 0:
            guard_fn += 1
        elif label == 0 and guard_pred == 1:
            guard_fp += 1
        else:
            guard_tn += 1

        if base_pred != label and guard_pred == label:
            repairs += 1
        elif base_pred == label and guard_pred != label:
            breaks += 1

    elapsed = time.time() - t0

    base_f1 = 2 * base_tp / (2 * base_tp + base_fp + base_fn) if (2 * base_tp + base_fp + base_fn) > 0 else 0.0
    guard_f1 = 2 * guard_tp / (2 * guard_tp + guard_fp + guard_fn) if (2 * guard_tp + guard_fp + guard_fn) > 0 else 0.0

    return {
        "split": split,
        "sample_count": len(data),
        "downgrades": downgrades,
        "base_metrics": {
            "tp": base_tp,
            "fn": base_fn,
            "fp": base_fp,
            "tn": base_tn,
            "f1": base_f1,
            "total_errors": base_fn + base_fp,
        },
        "guard_metrics": {
            "tp": guard_tp,
            "fn": guard_fn,
            "fp": guard_fp,
            "tn": guard_tn,
            "f1": guard_f1,
            "total_errors": guard_fn + guard_fp,
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
    print("Axon v2.6 - Loop198 Authenticode Trusted Signer Guard Evaluation")
    print("=" * 70)

    guard = Loop198TrustedSignerGuard()
    report = {}

    for split in ("val", "test10k", "full"):
        res = evaluate_signer_split(split, guard)
        if res:
            report[split] = res
            print(f"[{split}] Base Errors: {res['base_metrics']['total_errors']} -> Guard Errors: {res['guard_metrics']['total_errors']} | Repairs: {res['transitions']['repairs']} | Breaks: {res['transitions']['breaks']} | Downgrades: {res['downgrades']}")

    report_path = Path(__file__).resolve().parent.parent / "reports" / "roadmap_9997" / "loop198_trusted_signer_guard_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation complete. Saved receipt to {report_path}")


if __name__ == "__main__":
    main()
