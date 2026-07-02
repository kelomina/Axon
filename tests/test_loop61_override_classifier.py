from __future__ import annotations

import numpy as np
import pytest

from scripts.train_loop61_override_classifier import (
    _fit_override_classifier,
    override_classifier_predictions,
    override_model_candidates,
    override_target_summary,
    possible_override_mask,
    select_override_allow_threshold,
)


def test_possible_override_mask_only_selects_base_benign_candidate_malicious():
    base_scores = np.asarray([0.2, 0.8, 0.2, 0.8], dtype=np.float32)
    candidate_scores = np.asarray([0.9, 0.9, 0.3, 0.1], dtype=np.float32)

    mask = possible_override_mask(
        base_scores=base_scores,
        candidate_scores=candidate_scores,
        base_threshold=0.5,
        candidate_threshold=0.5,
    )

    assert mask.tolist() == [True, False, False, False]


def test_override_classifier_predictions_only_flips_benign_to_malicious():
    base_scores = np.asarray([0.2, 0.8, 0.2, 0.8], dtype=np.float32)
    candidate_scores = np.asarray([0.9, 0.9, 0.7, 0.1], dtype=np.float32)
    allow_scores = np.asarray([0.8, 0.9, 0.2, 0.9], dtype=np.float32)

    predictions, final_scores, override = override_classifier_predictions(
        base_scores=base_scores,
        candidate_scores=candidate_scores,
        allow_scores=allow_scores,
        base_threshold=0.5,
        candidate_threshold=0.5,
        allow_threshold=0.5,
    )

    assert predictions.tolist() == [1, 1, 0, 1]
    assert override.tolist() == [True, False, False, False]
    np.testing.assert_allclose(final_scores, [0.9, 0.8, 0.2, 0.8])


def test_override_target_summary_counts_beneficial_and_harmful_rows():
    labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
    possible = np.asarray([True, True, False, False])

    summary = override_target_summary(labels, possible)

    assert summary["possible_overrides"] == 2
    assert summary["beneficial_fn_repairs"] == 1
    assert summary["harmful_new_fp"] == 1
    assert summary["label1_ratio_in_possible"] == 0.5


def test_select_override_allow_threshold_reports_blocked_possible_rows():
    labels = np.asarray([1, 0, 1], dtype=np.int64)
    base_scores = np.asarray([0.2, 0.2, 0.8], dtype=np.float32)
    candidate_scores = np.asarray([0.9, 0.9, 0.9], dtype=np.float32)
    allow_scores = np.asarray([0.95, 0.1, 0.9], dtype=np.float32)

    selected = select_override_allow_threshold(
        labels=labels,
        base_scores=base_scores,
        candidate_scores=candidate_scores,
        allow_scores=allow_scores,
        base_threshold=0.5,
        candidate_threshold=0.5,
        allow_thresholds=[0.05, 0.5],
    )

    assert selected["override_count"] == 1
    assert selected["override_label1_count"] == 1
    assert selected["override_label0_count"] == 0
    assert selected["blocked_possible_count"] == 1
    assert selected["blocked_label0_count"] == 1
    assert selected["errors"] == 0


def test_fit_override_classifier_rejects_one_class_possible_rows():
    model = override_model_candidates(seed=61)[0][1]
    matrix = np.ones((3, 2), dtype=np.float32)
    labels = np.ones(3, dtype=np.int64)

    with pytest.raises(ValueError, match="only one class"):
        _fit_override_classifier(model, matrix, labels)
