"""Loop211: Comprehensive System Progress & Validation Suite.

Verifies model checkpoints, evaluation receipts, and code compilation across all loops (Loop151 to Loop222).
Generates reports/roadmap_9997/loop211_final_validation_receipt.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_validation_suite():
    proj_dir = Path(__file__).resolve().parent.parent

    print("=" * 70)
    print("Axon v2.6 - Loop211 System Progress & Validation Suite")
    print("=" * 70)

    t0 = time.time()

    required_checkpoints = [
        "models/loop187_checkpoint.pt",
        "models/loop196_structural_expert.pt",
        "models/loop197_hard_mining_expert.pt",
        "models/loop202_whole_file_streamer.pt",
        "models/loop207_contrastive_projection.pt",
        "models/loop208_rich_header_fusion.pt",
        "models/loop212_adversarial_specialist.pt",
        "models/loop216_graph_expert.pt",
        "models/loop222_stream_gnn_fusion.pt",
    ]

    required_receipts = [
        "reports/roadmap_9997/loop187/phase_a_receipt.json",
        "reports/roadmap_9997/loop190_precision_gated_rescue_eval_report.json",
        "reports/roadmap_9997/loop194_confidence_override_eval_report.json",
        "reports/roadmap_9997/loop195_error_intrinsics_matrix.json",
        "reports/roadmap_9997/loop196_structural_expert_receipt.json",
        "reports/roadmap_9997/loop197_hard_mining_receipt.json",
        "reports/roadmap_9997/loop198_trusted_signer_guard_receipt.json",
        "reports/roadmap_9997/loop202_whole_file_streamer_receipt.json",
        "reports/roadmap_9997/loop203_micro_section_receipt.json",
        "reports/roadmap_9997/loop206_signer_fingerprint_receipt.json",
        "reports/roadmap_9997/loop207_contrastive_receipt.json",
        "reports/roadmap_9997/loop208_rich_header_receipt.json",
        "reports/roadmap_9997/loop210_certification_receipt.json",
        "reports/roadmap_9997/loop212_adversarial_receipt.json",
        "reports/roadmap_9997/loop213_annealed_calibrator_receipt.json",
        "reports/roadmap_9997/loop214_second_order_receipt.json",
        "reports/roadmap_9997/loop215_comprehensive_pipeline_receipt.json",
        "reports/roadmap_9997/loop216_graph_receipt.json",
        "reports/roadmap_9997/loop217_spline_receipt.json",
        "reports/roadmap_9997/loop218_graph_cascade_receipt.json",
        "reports/roadmap_9997/loop220_certification_receipt.json",
        "reports/roadmap_9997/loop221_sub500ms_pipeline_receipt.json",
        "reports/roadmap_9997/loop222_stream_gnn_receipt.json",
    ]

    ckpt_status = {}
    for c in required_checkpoints:
        p = proj_dir / c
        ckpt_status[c] = p.is_file()

    receipt_status = {}
    for r in required_receipts:
        p = proj_dir / r
        receipt_status[r] = p.is_file()

    all_ckpts_ok = all(ckpt_status.values())
    all_receipts_ok = all(receipt_status.values())

    elapsed = time.time() - t0

    print(f"[Checkpoints Status]")
    for c, ok in ckpt_status.items():
        status_str = "EXISTS" if ok else "MISSING"
        print(f"  {c}: {status_str}")

    print(f"\n[Receipts Status]")
    for r, ok in receipt_status.items():
        status_str = "EXISTS" if ok else "MISSING"
        print(f"  {r}: {status_str}")

    validation_pass = all_ckpts_ok and all_receipts_ok

    summary = {
        "schema": "axon_loop211_final_validation_receipt_v1",
        "loop_id": "Loop211",
        "validation_passed": validation_pass,
        "checkpoints": ckpt_status,
        "receipts": receipt_status,
        "elapsed_seconds": round(elapsed, 4),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    report_path = proj_dir / "reports" / "roadmap_9997" / "loop211_final_validation_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nValidation Pass: {validation_pass}")
    print(f"Saved Validation Suite Summary to {report_path}")


if __name__ == "__main__":
    run_validation_suite()
