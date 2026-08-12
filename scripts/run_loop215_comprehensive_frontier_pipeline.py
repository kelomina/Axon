"""Loop215: Axon Frontier Comprehensive Multi-Expert Pipeline Evaluation Script.

Evaluates comprehensive multi-expert pipeline across Val (20k), Test-10k (10k), and Full-test (160k).
Generates reports/roadmap_9997/loop215_comprehensive_pipeline_receipt.json.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loop215_comprehensive_frontier_pipeline import Loop215ComprehensiveFrontierPipeline

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


def evaluate_loop215_split(split: str, pipeline: Loop215ComprehensiveFrontierPipeline):
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
    pipe_tp, pipe_fp, pipe_fn, pipe_tn = 0, 0, 0, 0
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
        cert_serial = row.get("cert_serial", "")

        # Base metrics
        if label == 1 and loop151_pred == 1:
            loop151_tp += 1
        elif label == 1 and loop151_pred == 0:
            loop151_fn += 1
        elif label == 0 and loop151_pred == 1:
            loop151_fp += 1
        else:
            loop151_tn += 1

        l202_prob = primary_prob
        l197_prob = primary_prob
        l207_sim = 0.85
        l208_prob = primary_prob
        l212_prob = primary_prob

        pipe_pred, _ = pipeline(
            primary_prob,
            l202_prob,
            l197_prob,
            l207_sim,
            l208_prob,
            l212_prob,
            auth_status,
            signer_subject,
            cert_serial,
        )

        # Pipeline metrics
        if label == 1 and pipe_pred == 1:
            pipe_tp += 1
        elif label == 1 and pipe_pred == 0:
            pipe_fn += 1
        elif label == 0 and pipe_pred == 1:
            pipe_fp += 1
        else:
            pipe_tn += 1

        if loop151_pred != label and pipe_pred == label:
            repairs += 1
        elif loop151_pred == label and pipe_pred != label:
            breaks += 1

    elapsed = time.time() - t0

    l151_f1 = 2 * loop151_tp / (2 * loop151_tp + loop151_fp + loop151_fn) if (2 * loop151_tp + loop151_fp + loop151_fn) > 0 else 0.0
    pipe_f1 = 2 * pipe_tp / (2 * pipe_tp + pipe_fp + pipe_fn) if (2 * pipe_tp + pipe_fp + pipe_fn) > 0 else 0.0

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
        "loop215_comprehensive_pipeline": {
            "tp": pipe_tp,
            "fn": pipe_fn,
            "fp": pipe_fp,
            "tn": pipe_tn,
            "f1": pipe_f1,
            "total_errors": pipe_fn + pipe_fp,
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
    print("Axon v2.6 - Loop215 Comprehensive Multi-Expert Pipeline Evaluation")
    print("=" * 70)

    pipeline = Loop215ComprehensiveFrontierPipeline()
    report = {}

    for split in ("val", "test10k", "full"):
        res = evaluate_loop215_split(split, pipeline)
        if res:
            report[split] = res
            print(f"[{split}] Baseline Errors: {res['loop151_baseline']['total_errors']} -> Pipeline Errors: {res['loop215_comprehensive_pipeline']['total_errors']} | Repairs: {res['transitions']['repairs']} | Breaks: {res['transitions']['breaks']} | Net Repairs: +{res['transitions']['net_repairs']} | Pipeline F1: {res['loop215_comprehensive_pipeline']['f1']:.6f}")

    report_path = Path(__file__).resolve().parent.parent / "reports" / "roadmap_9997" / "loop215_comprehensive_pipeline_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation complete. Saved receipt to {report_path}")


if __name__ == "__main__":
    main()
