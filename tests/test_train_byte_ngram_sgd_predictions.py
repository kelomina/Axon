import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_byte_ngram_sgd import write_predictions  # noqa: E402


def test_write_predictions_preserves_audit_identity_fields(tmp_path: Path):
    output_path = tmp_path / "predictions.csv"
    records = [
        {
            "source_path": "materialized.exe",
            "original_source_path": "original.exe",
            "cache_path": "sample.npz",
            "source_sha256": "a" * 64,
            "label": 1,
            "split": "val",
            "sample_index": "7",
        }
    ]

    write_predictions(
        output_path,
        records,
        np.asarray([1], dtype=np.int64),
        np.asarray([0.75], dtype=np.float32),
        threshold=0.5,
    )

    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["source_sha256"] == "a" * 64
    assert rows[0]["original_source_path"] == "original.exe"
    assert rows[0]["prediction"] == "1"
    assert rows[0]["correct"] == "True"
