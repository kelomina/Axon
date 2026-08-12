#!/usr/bin/env python3
"""Collect complete Val Authenticode evidence without touching Test or Full-test."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from probe_loop73_authenticode_val import build_signature_cache


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports/phase3_loop136/r5_oof_noise_pairwise_selector_recall_valonly/loop135_pairwise_selector_val_predictions.csv"
REPORT = ROOT / "reports/roadmap_9997/loop169_train_authenticode"
OUTPUT = REPORT / "val_authenticode.csv"
SUMMARY = REPORT / "val_collection_summary.json"


def main() -> None:
    with INPUT.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20_000 or {row.get("split") for row in rows} != {"val"}:
        raise ValueError("Loop169 requires exactly the existing 20k Val prediction manifest")
    receipt = build_signature_cache(
        predictions_csv=INPUT,
        output_csv=OUTPUT,
        only_predicted_positive=False,
        max_rows=None,
        powershell_exe="powershell.exe",
    )
    with OUTPUT.open(encoding="utf-8-sig", newline="") as handle:
        evidence = list(csv.DictReader(handle))
    if len(evidence) != 20_000:
        raise ValueError("Loop169 Val signature denominator drifted")
    result = {
        "schema": "axon_loop169_val_authenticode_collection_v1",
        "scope": {"train_access": False, "val_access": True, "test10k_access": False, "full_test_access": False},
        "rows": len(evidence),
        "status_counts": dict(sorted(Counter(row.get("auth_status") or "missing" for row in evidence).items())),
        "collection": receipt,
    }
    SUMMARY.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
