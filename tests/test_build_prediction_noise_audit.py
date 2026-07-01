from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_prediction_noise_audit import build_audit  # noqa: E402


def test_noise_audit_accepts_blend_probability_column(tmp_path):
    predictions = tmp_path / "blend_predictions.csv"
    fieldnames = [
        "source_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "blend_prob_malicious",
    ]
    with predictions.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "source_path": "data/benign.exe",
                "source_sha256": "sha-benign",
                "label": 0,
                "split": "val",
                "sample_index": 1,
                "blend_prob_malicious": 0.99,
            }
        )
        writer.writerow(
            {
                "source_path": "data/malicious.exe",
                "source_sha256": "sha-malicious",
                "label": 1,
                "split": "val",
                "sample_index": 2,
                "blend_prob_malicious": 0.01,
            }
        )

    summary = build_audit(predictions, tmp_path / "audit", threshold=0.50)

    assert summary["errors"] == 2
    assert summary["false_positive_count"] == 1
    assert summary["false_negative_count"] == 1
    assert summary["noise_bucket_counts"]["severe_fp_conflict_prob_ge_0.99"] == 1
    assert summary["noise_bucket_counts"]["severe_fn_conflict_prob_le_0.01"] == 1
