import numpy as np
import pytest

from scripts.train_loop42_oof_residual_gate import (
    RegionHashConfig,
    build_gate_score_features,
    gate_training_targets,
    oof_region_ngram_scores,
    override_predictions,
    prediction_metrics,
)
from scripts.train_stage2_cache_matrix import filter_model_candidates


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


def test_filter_model_candidates_accepts_explicit_none_sentinel():
    candidates = [("first", object()), ("second", object())]

    assert filter_model_candidates(candidates, "__none__") == []
    assert filter_model_candidates(candidates, "") == candidates


def test_oof_region_ngram_scores_aligns_fold_and_val_labels(monkeypatch: pytest.MonkeyPatch):
    train_records = [
        {"label": 0, "score": 0.1},
        {"label": 1, "score": 0.9},
        {"label": 0, "score": 0.2},
        {"label": 1, "score": 0.8},
    ]
    val_records = [
        {"label": 0, "score": 0.3},
        {"label": 1, "score": 0.7},
    ]
    config = RegionHashConfig(
        n_features=32,
        prefix_len=8,
        ngram_min=2,
        ngram_max=2,
        ngram_stride=1,
        include_prefix_features=False,
        include_full_ngram_features=False,
        include_region_ngram_features=True,
        include_region_scalar_features=True,
        include_byte_hist=False,
        include_cache_features=False,
        region_window=8,
        tail_window=8,
        max_byte_length=8,
        pe_feature_dim=2,
        stat_feature_dim=2,
        lightweight_feature_dim=2,
    )

    def fake_train_candidate(records, _config, **_kwargs):
        return {"rows": len(records)}

    def fake_predict_scores(_model, records, _config, _batch_size, *, allow_missing_source):
        assert allow_missing_source is True
        labels = np.asarray([record["label"] for record in records], dtype=np.int64)
        scores = np.asarray([record["score"] for record in records], dtype=np.float32)
        return labels, scores

    monkeypatch.setattr("scripts.train_loop42_oof_residual_gate.train_region_candidate", fake_train_candidate)
    monkeypatch.setattr("scripts.train_loop42_oof_residual_gate.predict_region_scores", fake_predict_scores)

    oof, val_scores, model, report = oof_region_ngram_scores(
        train_records=train_records,
        train_y=np.asarray([0, 1, 0, 1], dtype=np.int64),
        val_records=val_records,
        config=config,
        alpha=1e-5,
        l1_ratio=0.0,
        epochs=1,
        batch_size=2,
        folds=2,
        seed=42,
        allow_missing_source=True,
    )

    np.testing.assert_allclose(oof, [0.1, 0.9, 0.2, 0.8])
    np.testing.assert_allclose(val_scores, [0.3, 0.7])
    assert model == {"rows": 4}
    assert report["name"] == "region_byte_ngram_sgd"
    assert report["labels"].tolist() == [0, 1]
    assert len(report["folds"]) == 2
