import numpy as np

from scripts.train_loop128_fp_guard_selector import (
    AlignedPredictions,
    FP_GUARD_FEATURE_NAMES,
    apply_guard,
    build_guard_features,
    metrics_at_predictions,
    possible_flip_mask,
)
from scripts.identity_feature_guard import identity_feature_violations


def test_fp_guard_feature_names_are_identity_safe():
    assert identity_feature_violations(FP_GUARD_FEATURE_NAMES) == []


def test_build_guard_features_uses_only_scores():
    primary = np.asarray([0.9, 0.4], dtype=np.float32)
    conservative = np.asarray([0.2, 0.3], dtype=np.float32)

    matrix = build_guard_features(primary, conservative)

    assert matrix.shape == (2, len(FP_GUARD_FEATURE_NAMES))
    np.testing.assert_allclose(matrix[:, 0], primary)
    np.testing.assert_allclose(matrix[:, 1], conservative)
    np.testing.assert_allclose(matrix[:, 2], primary - conservative)


def test_apply_guard_only_flips_primary_positive_conservative_negative():
    aligned = AlignedPredictions(
        rows=[{}, {}, {}],
        labels=np.asarray([0, 1, 0], dtype=np.int64),
        primary_prob=np.asarray([0.8, 0.7, 0.2], dtype=np.float32),
        conservative_prob=np.asarray([0.2, 0.8, 0.1], dtype=np.float32),
        primary_pred=np.asarray([1, 1, 0], dtype=np.int64),
        conservative_pred=np.asarray([0, 1, 0], dtype=np.int64),
    )
    scores = np.asarray([0.9, 0.9, 0.9], dtype=np.float32)

    predictions, flip = apply_guard(aligned, scores, 0.5)

    assert possible_flip_mask(aligned).tolist() == [True, False, False]
    assert flip.tolist() == [True, False, False]
    assert predictions.tolist() == [0, 1, 0]


def test_metrics_at_predictions_counts_fp_fn():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    predictions = np.asarray([0, 1, 1, 0], dtype=np.int64)

    metrics = metrics_at_predictions(labels, predictions)

    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["errors"] == 2
