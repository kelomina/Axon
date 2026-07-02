from __future__ import annotations

import numpy as np

from scripts.train_loop70_nested_oof_meta import build_meta_score_features


def test_loop70_meta_score_features_are_score_only_and_stable():
    matrix, names = build_meta_score_features(
        base_scores=np.asarray([0.1, 0.9], dtype=np.float32),
        candidate_scores=np.asarray([0.8, 0.7], dtype=np.float32),
        allow_scores=np.asarray([0.6, 0.2], dtype=np.float32),
        final_scores=np.asarray([0.8, 0.9], dtype=np.float32),
        final_predictions=np.asarray([1, 1], dtype=np.int64),
        override_mask=np.asarray([True, False]),
        possible_mask=np.asarray([True, False]),
    )

    assert matrix.shape == (2, len(names))
    assert "meta_base_score" in names
    assert "meta_previous_override_flag" in names
    assert all("path" not in name and "sha" not in name and "split" not in name for name in names)
    assert float(matrix[0, names.index("meta_previous_override_flag")]) == 1.0
