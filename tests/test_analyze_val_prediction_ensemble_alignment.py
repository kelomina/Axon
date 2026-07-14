import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_val_prediction_ensemble import analyze


def _write_predictions(path: Path, rows: list[dict]) -> None:
    fieldnames = ["sample_index", "source_path", "source_sha256", "label", "prob_malicious"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_ensemble_analysis_rejects_misaligned_sample_index(tmp_path: Path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_predictions(
        first,
        [
            {"sample_index": "1", "source_path": "a", "source_sha256": "sha-a", "label": "0", "prob_malicious": "0.1"},
            {"sample_index": "2", "source_path": "b", "source_sha256": "sha-b", "label": "1", "prob_malicious": "0.9"},
        ],
    )
    _write_predictions(
        second,
        [
            {"sample_index": "1", "source_path": "different", "source_sha256": "sha-x", "label": "0", "prob_malicious": "0.2"},
            {"sample_index": "2", "source_path": "b", "source_sha256": "sha-b", "label": "1", "prob_malicious": "0.8"},
        ],
    )

    with pytest.raises(ValueError, match="not aligned"):
        analyze(
            [("first", first, "prob_malicious"), ("second", second, "prob_malicious")],
            [0.5],
            key_column="sample_index",
        )
