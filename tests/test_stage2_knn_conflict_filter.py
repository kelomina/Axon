from __future__ import annotations

import numpy as np
import pytest

from scripts.train_stage2_cache_matrix import _knn_feature_names, sample_weights, summarize_knn_conflicts


def _feature_block(rows: int = 2) -> tuple[np.ndarray, list[str], dict[str, int]]:
    names = _knn_feature_names([10, 25])
    return np.zeros((rows, len(names)), dtype=np.float32), names, {name: index for index, name in enumerate(names)}


def test_none_mode_keeps_unit_weights_without_knn_features():
    labels = np.asarray([0, 1, 0], dtype=np.int64)
    base_probs = np.asarray([0.9, 0.1, 0.2], dtype=np.float32)

    weights = sample_weights(labels, base_probs, "none")

    assert weights.tolist() == [1.0, 1.0, 1.0]


def test_knn_noise_modes_require_knn_features():
    labels = np.asarray([0], dtype=np.int64)
    base_probs = np.asarray([0.5], dtype=np.float32)

    with pytest.raises(ValueError, match="requires --knn-features"):
        sample_weights(labels, base_probs, "knn_trim_strong_conflict")


def test_knn_trim_strong_conflict_uses_oof_neighbor_evidence():
    labels = np.asarray([0, 1], dtype=np.int64)
    base_probs = np.asarray([0.1, 0.9], dtype=np.float32)
    features, names, columns = _feature_block()

    features[0, columns["knn25_mal_ratio"]] = 0.90
    features[0, columns["knn25_weighted_mal_ratio"]] = 0.90
    features[0, columns["knn10_mal_ratio"]] = 0.80
    features[0, columns["knn10_weighted_mal_ratio"]] = 0.80
    features[0, columns["knn_top1_label"]] = 1.0
    features[0, columns["knn_top1_similarity"]] = 0.96

    features[1, columns["knn25_mal_ratio"]] = 0.95
    features[1, columns["knn25_weighted_mal_ratio"]] = 0.95
    features[1, columns["knn10_mal_ratio"]] = 0.90
    features[1, columns["knn10_weighted_mal_ratio"]] = 0.90
    features[1, columns["knn_top1_label"]] = 1.0
    features[1, columns["knn_top1_similarity"]] = 0.96

    weights = sample_weights(
        labels,
        base_probs,
        "knn_trim_strong_conflict",
        knn_features=features,
        knn_feature_names=names,
    )

    assert weights.tolist() == [0.0, 1.0]


def test_knn_conflict_summary_reports_label_counts():
    labels = np.asarray([0, 1], dtype=np.int64)
    features, names, columns = _feature_block()

    features[0, columns["knn25_mal_ratio"]] = 0.95
    features[0, columns["knn25_weighted_mal_ratio"]] = 0.95
    features[0, columns["knn10_mal_ratio"]] = 0.80
    features[0, columns["knn10_weighted_mal_ratio"]] = 0.80
    features[0, columns["knn_top1_label"]] = 1.0
    features[0, columns["knn_top1_similarity"]] = 0.96

    summary = summarize_knn_conflicts(labels, features, names)

    assert summary["enabled"] is True
    assert summary["rule_version"] == "train_oof_knn_conflict_v2"
    assert summary["strong_count"] == 1
    assert summary["strong_label0"] == 1
    assert summary["strong_label1"] == 0
    assert summary["exact_opposite_count"] == 1
