from __future__ import annotations

import numpy as np
import pytest

from scripts.train_loop57_fn_overlay_gate import (
    align_external_scores,
    build_fn_gate_matrix,
    fn_gate_training_targets,
    fn_override_predictions,
    select_fn_gate_threshold,
)
from scripts.train_loop55_overlay_boundary import OVERLAY_BOUNDARY_FEATURE_NAMES


def test_fn_override_predictions_only_flips_benign_to_malicious():
    base_scores = np.asarray([0.2, 0.8, 0.2, 0.8], dtype=np.float32)
    candidate_scores = np.asarray([0.9, 0.1, 0.4, 0.9], dtype=np.float32)
    gate_scores = np.asarray([0.95, 0.95, 0.95, 0.95], dtype=np.float32)

    predictions, final_scores, override = fn_override_predictions(
        base_scores=base_scores,
        candidate_scores=candidate_scores,
        gate_scores=gate_scores,
        base_threshold=0.5,
        candidate_threshold=0.5,
        gate_threshold=0.9,
    )

    assert predictions.tolist() == [1, 1, 0, 1]
    assert override.tolist() == [True, False, False, False]
    np.testing.assert_allclose(final_scores, [0.9, 0.8, 0.2, 0.8])


def test_fn_gate_training_targets_marks_repairs_and_new_fp():
    labels = np.asarray([1, 0, 1, 0, 1], dtype=np.int64)
    base_scores = np.asarray([0.2, 0.2, 0.8, 0.8, 0.1], dtype=np.float32)
    candidate_scores = np.asarray([0.9, 0.7, 0.9, 0.1, 0.2], dtype=np.float32)

    targets, weights, summary = fn_gate_training_targets(
        labels,
        base_scores,
        candidate_scores,
        base_threshold=0.5,
        candidate_threshold=0.5,
        neutral_weight=0.02,
    )

    assert targets.tolist() == [1, 0, 0, 0, 0]
    np.testing.assert_allclose(weights, [1.0, 1.0, 0.02, 0.02, 0.02])
    assert summary["possible_overrides"] == 2
    assert summary["beneficial_fn_repairs"] == 1
    assert summary["harmful_new_fp"] == 1


def test_build_fn_gate_matrix_adds_safe_overlay_aliases():
    overlay = np.ones((2, len(OVERLAY_BOUNDARY_FEATURE_NAMES)), dtype=np.float32)
    base = np.asarray([0.1, 0.9], dtype=np.float32)
    candidate = np.asarray([0.8, 0.2], dtype=np.float32)

    content = np.zeros((2, 3), dtype=np.float32)
    matrix, names = build_fn_gate_matrix(content, overlay, base, candidate, include_overlay_features=True)

    assert matrix.shape == (2, 9 + len(OVERLAY_BOUNDARY_FEATURE_NAMES))
    assert names[0] == "gate_base_score"
    assert names[9] == f"gate_{OVERLAY_BOUNDARY_FEATURE_NAMES[0]}"


def test_build_fn_gate_matrix_can_add_content_aliases():
    overlay = np.ones((2, len(OVERLAY_BOUNDARY_FEATURE_NAMES)), dtype=np.float32)
    content = np.ones((2, 4), dtype=np.float32)
    base = np.asarray([0.1, 0.9], dtype=np.float32)
    candidate = np.asarray([0.8, 0.2], dtype=np.float32)

    matrix, names = build_fn_gate_matrix(
        content,
        overlay,
        base,
        candidate,
        include_overlay_features=True,
        include_content_features=True,
    )

    assert matrix.shape == (2, 9 + len(OVERLAY_BOUNDARY_FEATURE_NAMES) + 4)
    assert names[-1] == "gate_content_feature_3"


def test_select_fn_gate_threshold_reports_override_labels():
    labels = np.asarray([1, 0, 1], dtype=np.int64)
    base_scores = np.asarray([0.2, 0.2, 0.8], dtype=np.float32)
    candidate_scores = np.asarray([0.9, 0.9, 0.9], dtype=np.float32)
    gate_scores = np.asarray([0.95, 0.4, 0.2], dtype=np.float32)

    selected = select_fn_gate_threshold(
        labels=labels,
        base_scores=base_scores,
        candidate_scores=candidate_scores,
        gate_scores=gate_scores,
        base_threshold=0.5,
        candidate_threshold=0.5,
        gate_thresholds=[0.3, 0.9],
    )

    assert selected["override_count"] == 1
    assert selected["override_label1_count"] == 1
    assert selected["override_label0_count"] == 0
    assert selected["errors"] == 0


def test_align_external_scores_rejects_label_mismatch(tmp_path):
    prediction_path = tmp_path / "predictions.csv"
    prediction_path.write_text(
        "sample_index,source_sha256,label,stage2_prob_malicious\n"
        "1,abc,0,0.7\n",
        encoding="utf-8",
    )
    rows = [{"sample_index": "1", "source_sha256": "abc", "label": "1"}]

    with pytest.raises(ValueError, match="label mismatch"):
        align_external_scores(
            rows=rows,
            prediction_path=prediction_path,
            probability_column="stage2_prob_malicious",
            key_column="sample_index",
        )
