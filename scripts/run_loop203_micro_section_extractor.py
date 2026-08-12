"""Loop203: PE Micro-Section Anomaly Extraction & Evaluation Script.

Evaluates micro-section entropy and relocation anomaly features.
Generates reports/roadmap_9997/loop203_micro_section_receipt.json.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loop203_micro_section_extractor import Loop203MicroSectionExtractor


def run_loop203():
    print("=" * 70)
    print("Axon v2.6 - Loop203 Micro-Section & Relocation Anomaly Extraction")
    print("=" * 70)

    extractor = Loop203MicroSectionExtractor()
    t0 = time.time()

    # Test extractor on synthetic byte samples
    sample_clean = b"\x4d\x5a" + b"\x00" * 1024
    sample_packed = b"\x4d\x5a" + bytes(np.random.randint(0, 256, 1024, dtype=np.uint8))

    feat_clean = extractor.extract_from_bytes(sample_clean)
    feat_packed = extractor.extract_from_bytes(sample_packed)

    elapsed = time.time() - t0

    print(f"[Extractor Output]")
    print(f"  Clean Sample Entropy Mean:  {feat_clean[1]:.4f} | Std: {feat_clean[2]:.4f}")
    print(f"  Packed Sample Entropy Mean: {feat_packed[1]:.4f} | Std: {feat_packed[2]:.4f}")
    print(f"  Feature Vector Dim:         {len(feat_clean)}")

    receipt = {
        "schema": "axon_loop203_micro_section_receipt_v1",
        "loop_id": "Loop203",
        "feature_dim": len(feat_clean),
        "clean_sample_mean_entropy": float(feat_clean[1]),
        "packed_sample_mean_entropy": float(feat_packed[1]),
        "elapsed_seconds": round(elapsed, 4),
    }

    report_path = Path(__file__).resolve().parent.parent / "reports" / "roadmap_9997" / "loop203_micro_section_receipt.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved receipt to {report_path}")


if __name__ == "__main__":
    run_loop203()
