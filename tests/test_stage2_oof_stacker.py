import numpy as np

from scripts.train_stage2_oof_stacker import STAGE2_PROB_FEATURE_COUNT, build_stack_features


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
