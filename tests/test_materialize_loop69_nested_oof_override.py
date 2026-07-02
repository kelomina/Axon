from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.materialize_loop69_nested_oof_override import write_nested_oof_predictions


def test_write_nested_oof_predictions_exports_required_protocol_columns(tmp_path: Path):
    output = tmp_path / "oof.csv"
    rows = [
        {
            "source_path": "ignored-for-alignment.exe",
            "cache_path": "cache-a.npz",
            "source_sha256": "sha-a",
            "split": "train",
            "sample_index": "10",
        },
        {
            "source_path": "renamed.bin",
            "cache_path": "cache-b.npz",
            "source_sha256": "sha-b",
            "split": "train",
            "sample_index": "11",
        },
    ]

    write_nested_oof_predictions(
        output,
        rows,
        np.asarray([1, 0], dtype=np.int64),
        oof_fold=np.asarray([1, 2], dtype=np.int64),
        base_scores=np.asarray([0.2, 0.1], dtype=np.float32),
        candidate_scores=np.asarray([0.8, 0.7], dtype=np.float32),
        allow_scores=np.asarray([0.9, 0.1], dtype=np.float32),
        final_scores=np.asarray([0.8, 0.1], dtype=np.float32),
        final_predictions=np.asarray([1, 0], dtype=np.int64),
        override_mask=np.asarray([True, False]),
        possible_mask=np.asarray([True, True]),
        candidate_thresholds=np.asarray([0.46, 0.47], dtype=np.float32),
        allow_thresholds=np.asarray([0.74, 0.75], dtype=np.float32),
        selected_candidate="extra_trees_300_leaf1",
        selected_override_model="override_logreg_balanced_c1",
    )

    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        exported = list(reader)

    assert reader.fieldnames is not None
    for column in [
        "base_oof_prob_malicious",
        "candidate_oof_prob_malicious",
        "allow_oof_prob",
        "final_oof_prob_malicious",
        "final_oof_prediction",
        "oof_override_flag",
        "oof_fold",
    ]:
        assert column in reader.fieldnames
    assert "correct" not in reader.fieldnames
    assert exported[0]["oof_override_flag"] == "True"
    assert exported[1]["final_oof_prediction"] == "0"
