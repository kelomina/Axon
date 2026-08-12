"""Loop210: Final Certified 0.9997 F1 Verification Script.

Audit equation according to goal.md:
  10003 * FN + 9997 * FP <= 480000
Max allowed total errors <= 48 on 160,000 balanced reference set.
Generates reports/roadmap_9997/loop210_certification_receipt.json.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MAX_ALLOWED_ERRORS = 48


def run_certification():
    proj_dir = Path(__file__).resolve().parent.parent
    integ_receipt = proj_dir / "reports/roadmap_9997/loop209_multi_stage_receipt.json"

    print("=" * 70)
    print("Axon v2.6 - Loop210 Final Certified 0.9997 F1 Verification")
    print("=" * 70)

    if not integ_receipt.is_file():
        print(f"Waiting for Loop209 receipt at {integ_receipt}...")
        return

    with open(integ_receipt, encoding="utf-8") as f:
        report = json.load(f)

    full_res = report.get("full", {})
    integ_metrics = full_res.get("loop209_multi_stage", {})

    tp = integ_metrics.get("tp", 0)
    fn = integ_metrics.get("fn", 0)
    fp = integ_metrics.get("fp", 0)
    tn = integ_metrics.get("tn", 0)
    total_errors = fn + fp
    f1 = integ_metrics.get("f1", 0.0)

    equation_val = 10003 * fn + 9997 * fp
    is_certified = (equation_val <= 480000) and (total_errors <= MAX_ALLOWED_ERRORS)

    print(f"[Certification Metrics]")
    print(f"  TP: {tp:,} | FN: {fn:,} | FP: {fp:,} | TN: {tn:,}")
    print(f"  Total Errors:        {total_errors:,}")
    print(f"  Target Error Budget: <= {MAX_ALLOWED_ERRORS}")
    print(f"  Equation Score:      {equation_val:,} (Limit: 480,000)")
    print(f"  Measured Full F1:    {f1:.10f}")
    print(f"  Certification Pass:  {is_certified}")

    cert_record = {
        "schema": "axon_loop210_certification_receipt_v1",
        "loop_id": "Loop210",
        "certification_target": "Full-test F1 >= 0.9997 (Total Errors <= 48)",
        "full_test_metrics": {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "total_errors": total_errors,
            "f1_score": f1,
            "equation_score": equation_val,
        },
        "is_certified": is_certified,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    report_path = proj_dir / "reports" / "roadmap_9997" / "loop210_certification_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(cert_record, f, indent=2)

    print(f"\nSaved Certification Record to {report_path}")


if __name__ == "__main__":
    run_certification()
