import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_prediction_errors import analyze_errors  # noqa: E402


def test_analyze_errors_recomputes_prediction_from_threshold(tmp_path):
    predictions = tmp_path / "predictions.csv"
    fieldnames = [
        "source_path",
        "cache_path",
        "sample_index",
        "group_id",
        "source_group_id",
        "group_size",
        "sample_weight",
        "hard_family_role",
        "is_rare_group",
        "group_source",
        "label",
        "split",
        "prob_malicious",
        "prediction",
    ]
    with predictions.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "source_path": "data/待拉黑/sample.exe",
            "cache_path": "cache/sample.npz",
            "sample_index": 1,
            "group_id": "g1",
            "source_group_id": "g1",
            "group_size": 1,
            "sample_weight": "",
            "hard_family_role": "",
            "is_rare_group": "True",
            "group_source": "singleton",
            "label": 1,
            "split": "test",
            "prob_malicious": 0.58,
            "prediction": 1,
        })

    low = analyze_errors(predictions, tmp_path / "low", threshold=0.55)
    high = analyze_errors(predictions, tmp_path / "high", threshold=0.60)

    assert low["false_negative_count"] == 0
    assert high["false_negative_count"] == 1
