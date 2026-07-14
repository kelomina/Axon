from __future__ import annotations

import numpy as np

from scripts import evaluate_stage2_oof_stacker as eval_oof


class ColumnProbModel:
    def __init__(self, column: int):
        self.column = column

    def predict_proba(self, matrix):
        scores = np.clip(matrix[:, self.column], 1.0e-6, 1.0 - 1.0e-6)
        return np.column_stack([1.0 - scores, scores])


class MeanProbModel:
    def predict_proba(self, matrix):
        scores = np.clip(matrix[:, :2].mean(axis=1), 1.0e-6, 1.0 - 1.0e-6)
        return np.column_stack([1.0 - scores, scores])


def test_score_oof_payload_runs_base_models_then_meta_model():
    payload = {
        "base_models": [ColumnProbModel(0), ColumnProbModel(1)],
        "meta_model": MeanProbModel(),
        "drop_base_prob_features": False,
    }
    matrix = np.asarray([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32)

    scores, info = eval_oof.score_oof_payload(payload, matrix)

    np.testing.assert_allclose(scores, [0.5, 0.5], atol=1e-6)
    assert info["base_model_count"] == 2
    assert info["stack_feature_dim"] == 9


def test_score_oof_payload_can_drop_stage2_probability_features():
    payload = {
        "base_models": [ColumnProbModel(0)],
        "meta_model": ColumnProbModel(0),
        "drop_base_prob_features": True,
    }
    matrix = np.asarray([[0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.25]], dtype=np.float32)

    scores, info = eval_oof.score_oof_payload(payload, matrix)

    assert float(scores[0]) == np.float32(0.25)
    assert info["dropped_feature_count"] == 6
    assert info["scoring_feature_dim"] == 1
