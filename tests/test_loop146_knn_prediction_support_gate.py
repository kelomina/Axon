from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_loop146_knn_prediction_support_gate import _apply_rule, _metric  # noqa: E402


def test_apply_rule_flips_only_when_knn_supports_opposite_prediction():
    names = [
        "knn25_mal_ratio",
        "knn25_weighted_mal_ratio",
        "knn25_mean_similarity",
        "knn_top1_label",
        "knn_top1_similarity",
        "knn_top1_top2_gap",
    ]
    base_predictions = np.asarray([0, 1, 0, 1], dtype=np.int64)
    features = np.asarray(
        [
            [0.95, 0.96, 0.99, 1.0, 0.99, 0.10],
            [0.04, 0.03, 0.98, 0.0, 0.98, 0.10],
            [0.80, 0.81, 0.99, 0.0, 0.99, 0.10],
            [0.20, 0.19, 0.99, 1.0, 0.99, 0.10],
        ],
        dtype=np.float32,
    )

    predictions, flips = _apply_rule(
        base_predictions,
        features,
        names,
        ref_k=25,
        min_mal_for_0to1=0.90,
        max_mal_for_1to0=0.10,
        min_weighted_agree=0.90,
        min_top1_similarity=0.95,
        min_top1_gap=0.0,
        mode="both",
    )

    assert predictions.tolist() == [1, 0, 0, 1]
    assert flips.tolist() == [True, True, False, False]


def test_metric_counts_binary_errors():
    result = _metric(
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0, 1, 0, 1], dtype=np.int64),
    )

    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["errors"] == 2
