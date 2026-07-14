import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.train_stage2_oof_stacker import (
    STAGE2_PROB_FEATURE_COUNT,
    build_stack_features,
    drop_stage2_probability_features,
    fit_model,
    parse_args,
)


def test_build_stack_features_uses_only_base_scores():
    base_scores = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.9, 0.7, 0.8],
        ],
        dtype=np.float32,
    )

    matrix, names = build_stack_features(base_scores)

    assert matrix.shape == (2, 10)
    assert names[:3] == ["base_model_0_score", "base_model_1_score", "base_model_2_score"]
    assert names[-1] == "base_score_logit_mean"
    np.testing.assert_allclose(matrix[:, :3], base_scores)
    np.testing.assert_allclose(matrix[:, 3], base_scores.mean(axis=1))


def test_stage2_probability_feature_count_documents_strict_drop_boundary():
    assert STAGE2_PROB_FEATURE_COUNT == 6


def test_drop_stage2_probability_features_returns_copy_without_base_columns():
    matrix = np.arange(24, dtype=np.float32).reshape(3, 8)

    dropped = drop_stage2_probability_features(matrix)

    assert dropped.shape == (3, 2)
    np.testing.assert_array_equal(dropped, matrix[:, STAGE2_PROB_FEATURE_COUNT:])
    assert not np.shares_memory(dropped, matrix)


def test_fit_model_routes_sample_weight_to_pipeline_final_step():
    matrix = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [1.0, 1.0],
            [1.1, 0.9],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    weights = np.asarray([1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="liblinear"))

    fitted = fit_model(model, matrix, labels, weights)

    probabilities = fitted.predict_proba(matrix)
    assert probabilities.shape == (4, 2)


def test_parse_args_accepts_oof_row_limits(tmp_path):
    args = parse_args(
        [
            "--checkpoint",
            str(tmp_path / "model.pt"),
            "--train-predictions",
            str(tmp_path / "train.csv"),
            "--val-predictions",
            str(tmp_path / "val.csv"),
            "--output-dir",
            str(tmp_path / "out"),
            "--max-train-rows",
            "7",
            "--max-val-rows",
            "5",
        ]
    )

    assert args.max_train_rows == 7
    assert args.max_val_rows == 5
