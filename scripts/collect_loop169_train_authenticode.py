#!/usr/bin/env python3
"""Collect Authenticode evidence for the isolated 20k Train-only manifest."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from probe_loop73_authenticode_val import build_signature_cache


ROOT = Path(__file__).resolve().parents[1]
FOLDS = ROOT / "reports/roadmap_9997/loop164/local_train_diagnostic_folds.jsonl"
REPORT_DIR = ROOT / "reports/roadmap_9997/loop169_train_authenticode"
INPUT_CSV = REPORT_DIR / "train_manifest_alignment.csv"
SIGNATURE_CSV = REPORT_DIR / "train_authenticode.csv"
SUMMARY_JSON = REPORT_DIR / "collection_summary.json"


def main() -> None:
    records = [json.loads(line) for line in FOLDS.read_text(encoding="utf-8").splitlines()]
    if len(records) != 20_000 or {record["split_role"] for record in records} != {"train"}:
        raise ValueError("Loop169 requires exactly the isolated 20k Train-only manifest")
    if [record["train_row_index"] for record in records] != list(range(20_000)):
        raise ValueError("Loop169 Train-only manifest ordering drifted")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with INPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "source_sha256", "sample_index", "prediction", "label"],
            lineterminator="\n",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_path": record["source_path"],
                    "source_sha256": record["source_sha256"],
                    "sample_index": record["sample_index"],
                    "prediction": 0,
                    "label": record["label"],
                }
            )
    receipt = build_signature_cache(
        predictions_csv=INPUT_CSV,
        output_csv=SIGNATURE_CSV,
        only_predicted_positive=False,
        max_rows=None,
        powershell_exe="powershell.exe",
    )
    with SIGNATURE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20_000:
        raise ValueError("Loop169 signature collection denominator drifted")
    statuses = Counter(str(row.get("auth_status") or "missing") for row in rows)
    summary = {
        "schema": "axon_loop169_train_authenticode_collection_v1",
        "scope": {"train_only": True, "raw_access": True, "heldout_access": False, "public_key_required": False},
        "rows": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "collection": receipt,
        "decision": "train_authenticode_evidence_collected_not_yet_fitted",
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
