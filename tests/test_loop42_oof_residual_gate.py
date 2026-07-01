import numpy as np

from scripts.train_loop42_oof_residual_gate import (
    build_gate_score_features,
    gate_training_targets,
    override_predictions,
    prediction_metrics,
)


def test_build_gate_score_features_is_score_only_and_stable():
    base = np.asarray([0.2, 0.8], dtype=np.float32)
    candidate = np.asarray([0.7, 0.6], dtype=np.float32)

    matrix, names = build_gate_score_features(base, candidate)

    assert matrix.shape == (2, 9)
    assert names == [
        "gate_base_score",
        "gate_candidate_score",
        "gate_score_delta",
        "gate_abs_score_delta",
        "gate_base_confidence",
        "gate_candidate_confidence",
        "gate_base_logit",
        "gate_candidate_logit",
        "gate_logit_delta",
    ]
    np.testing.assert_allclose(matrix[:, 0], base)
    np.testing.assert_allclose(matrix[:, 1], candidate)
    np.testing.assert_allclose(matrix[:, 2], candidate - base)


def test_gate_training_targets_weight_benefit_and_harm_over_neutral():
    labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
    base_scores = np.asarray([0.2, 0.1, 0.9, 0.8], dtype=np.float32)
    candidate_scores = np.asarray([0.9, 0.9, 0.8, 0.7], dtype=np.float32)

    targets, weights, summary = gate_training_targets(
        labels,
        base_scores,
        candidate_scores,
        base_threshold=0.5,
        candidate_threshold=0.5,
        neutral_weight=0.05,
    )

    assert targets.tolist() == [1, 0, 0, 0]
    np.testing.assert_allclose(weights, [1.0, 1.0, 0.05, 0.05])
    assert summary["beneficial_overrides"] == 1
    assert summary["harmful_overrides"] == 1
    assert summary["neutral_rows"] == 2


def test_override_predictions_uses_gate_threshold():
    base_scores = np.asarray([0.2, 0.8, 0.2], dtype=np.float32)
    candidate_scores = np.asarray([0.9, 0.1, 0.8], dtype=np.float32)
    gate_scores = np.asarray([0.9, 0.4, 0.8], dtype=np.float32)

    predictions, final_scores, override = override_predictions(
        base_scores=base_scores,
        candidate_scores=candidate_scores,
        gate_scores=gate_scores,
        base_threshold=0.5,
        candidate_threshold=0.5,
        gate_threshold=0.75,
    )

    assert predictions.tolist() == [1, 1, 1]
    np.testing.assert_allclose(final_scores, [0.9, 0.8, 0.8])
    assert override.tolist() == [True, False, True]


def test_prediction_metrics_counts_errors():
    labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
    predictions = np.asarray([1, 1, 0, 0], dtype=np.int64)

    metrics = prediction_metrics(labels, predictions)

    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["errors"] == 2
