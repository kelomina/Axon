"""Loop195: Physical Error Intrinsics Matrix Extraction & Analysis.

Extracts and analyzes all 1,466 residual errors from Loop151 on Full-test (160,000 samples).
Outputs reports/roadmap_9997/loop195_error_intrinsics_matrix.json for hard-example expert targeting.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys

PRIMARY_THR = 0.31


def analyze_errors():
    proj_dir = Path(__file__).resolve().parent.parent
    csv_path = proj_dir / "reports/phase3_loop151/loop151_trusted_signer_guard_full_predictions.csv"

    print("=" * 70)
    print("Axon v2.6 - Loop195 Error Intrinsics Matrix Analysis")
    print("=" * 70)
    print(f"Reading predictions from {csv_path}...")

    t0 = time.time()

    fp_samples = []
    fn_samples = []
    auth_status_counts = {}

    total_rows = 0
    correct_rows = 0

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            label = int(row.get("label", -1))
            if label not in (0, 1):
                continue

            primary_prob = float(row.get("stage2_prob_malicious", 0.0))
            signer_downgrade = row.get("trusted_signer_guard_downgrade", "").strip().lower() == "true"
            pred = 0 if signer_downgrade else int(primary_prob >= PRIMARY_THR)

            auth_status = row.get("auth_status", "Unknown")
            auth_status_counts[auth_status] = auth_status_counts.get(auth_status, 0) + 1

            if pred == label:
                correct_rows += 1
                continue

            sha = row.get("source_sha256", "").strip()
            path = row.get("source_path", "").strip()
            ext = Path(path).suffix.lower() if path else ""

            item = {
                "sha256": sha,
                "label": label,
                "prediction": pred,
                "stage2_prob": primary_prob,
                "auth_status": auth_status,
                "signer_subject": row.get("signer_subject", ""),
                "extension": ext,
                "signer_downgrade": signer_downgrade,
            }

            if label == 0 and pred == 1:
                fp_samples.append(item)
            elif label == 1 and pred == 0:
                fn_samples.append(item)

    elapsed = time.time() - t0
    total_errors = len(fp_samples) + len(fn_samples)
    f1 = 2 * (total_rows // 2 - len(fn_samples)) / (2 * (total_rows // 2) - len(fn_samples) + len(fp_samples))

    print(f"\n[Analysis Summary]")
    print(f"  Total Evaluated Rows: {total_rows:,}")
    print(f"  Correct Rows:        {correct_rows:,}")
    print(f"  Total Errors:         {total_errors:,} (FP: {len(fp_samples)}, FN: {len(fn_samples)})")
    print(f"  Unrounded Full F1:    {f1:.10f}")
    print(f"  Target F1 Threshold:  0.9997000000 (Max Allowed Errors <= 48)")
    print(f"  Net Errors to Clear:  {total_errors - 48}")
    print(f"  Analysis Elapsed:     {elapsed:.2f}s")

    # Analyze probability bands for FP & FN
    fp_probs = [x["stage2_prob"] for x in fp_samples]
    fn_probs = [x["stage2_prob"] for x in fn_samples]

    fp_marginal = sum(1 for p in fp_probs if 0.31 <= p < 0.50)
    fp_high_conf = sum(1 for p in fp_probs if p >= 0.50)

    fn_marginal = sum(1 for p in fn_probs if 0.15 <= p < 0.31)
    fn_deep_miss = sum(1 for p in fn_probs if p < 0.15)

    print(f"\n[FP Probability Breakdown]")
    print(f"  Marginal FP (0.31 <= p < 0.50): {fp_marginal} ({fp_marginal / len(fp_samples):.1%})")
    print(f"  High-Conf FP (p >= 0.50):       {fp_high_conf} ({fp_high_conf / len(fp_samples):.1%})")

    print(f"\n[FN Probability Breakdown]")
    print(f"  Marginal FN (0.15 <= p < 0.31): {fn_marginal} ({fn_marginal / len(fn_samples):.1%})")
    print(f"  Deep-Miss FN (p < 0.15):       {fn_deep_miss} ({fn_deep_miss / len(fn_samples):.1%})")

    matrix = {
        "schema": "axon_loop195_error_intrinsics_v1",
        "loop_id": "Loop195",
        "total_rows": total_rows,
        "total_errors": total_errors,
        "fp_count": len(fp_samples),
        "fn_count": len(fn_samples),
        "f1_score": f1,
        "errors_to_target": total_errors - 48,
        "fp_breakdown": {
            "marginal_031_to_050": fp_marginal,
            "high_conf_ge_050": fp_high_conf,
        },
        "fn_breakdown": {
            "marginal_015_to_031": fn_marginal,
            "deep_miss_lt_015": fn_deep_miss,
        },
        "auth_status_distribution": auth_status_counts,
        "fp_samples_sample10": fp_samples[:10],
        "fn_samples_sample10": fn_samples[:10],
    }

    report_path = proj_dir / "reports" / "roadmap_9997" / "loop195_error_intrinsics_matrix.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)

    print(f"\nSaved Error Intrinsics Matrix to {report_path}")


if __name__ == "__main__":
    analyze_errors()
