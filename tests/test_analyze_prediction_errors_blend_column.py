from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_prediction_errors import analyze_errors  # noqa: E402


def test_analyze_errors_accepts_blend_probability_column(tmp_path):
    predictions = tmp_path / "blend_predictions.csv"
    fieldnames = [
        "source_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "blend_prob_malicious",
        "prediction",
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
                "blend_prob_malicious": 0.80,
                "prediction": 1,
            }
        )

    summary = analyze_errors(predictions, tmp_path / "blend", threshold=0.50)

    assert summary["false_positive_count"] == 1
    assert summary["false_negative_count"] == 0
