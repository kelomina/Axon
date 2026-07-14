from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np

from scripts import evaluate_loop135_pairwise_selector as eval_loop135
from scripts import train_loop135_pairwise_selector as loop135
from scripts.identity_feature_guard import identity_feature_violations


class _FixedSelectorModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = np.asarray(scores, dtype=np.float32)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[0] != self.scores.shape[0]:
            raise ValueError("unexpected matrix row count")
        return np.column_stack([1.0 - self.scores, self.scores])


def _pair() -> loop135.AlignedPair:
    return loop135.AlignedPair(
        rows=[{}, {}, {}],
        labels=np.asarray([0, 1, 1], dtype=np.int64),
        baseline_prob=np.asarray([0.8, 0.4, 0.8], dtype=np.float32),
        candidate_prob=np.asarray([0.2, 0.7, 0.2], dtype=np.float32),
        baseline_pred=np.asarray([1, 0, 1], dtype=np.int64),
        candidate_pred=np.asarray([0, 1, 0], dtype=np.int64),
    )


def test_loop135_feature_names_are_identity_safe():
    assert identity_feature_violations(loop135.SCORE_FEATURE_NAMES) == []
    assert identity_feature_violations(loop135.CONTENT_FEATURE_NAMES) == []


def test_loop135_score_features_describe_frozen_prediction_disagreement():
    pair = _pair()
    features = loop135.build_score_features(pair, np.asarray([0, 1], dtype=np.int64))

    assert features.shape == (2, len(loop135.SCORE_FEATURE_NAMES))
    np.testing.assert_allclose(features[:, 0], [0.8, 0.4])
    np.testing.assert_allclose(features[:, 1], [0.2, 0.7])
    assert features[:, -2:].tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_loop135_support_feature_names_are_identity_safe():
    names = loop135.support_feature_names([5, 25])

    assert names
    assert identity_feature_violations(names) == []
    assert all(name.startswith("support_") for name in names)


def test_loop135_selector_features_append_support_rows():
    pair = _pair()
    support = np.asarray(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        dtype=np.float32,
    )

    features, names = loop135.build_selector_features(
        pair,
        np.asarray([2, 0], dtype=np.int64),
        content_pe_cache_dir=None,
        content_pe_v2_cache_dir=None,
        content_string_cache_dir=None,
        support_matrix=support,
        support_names=["support_a", "support_b"],
    )

    assert names[-2:] == ["support_a", "support_b"]
    np.testing.assert_allclose(features[:, -2:], [[0.5, 0.6], [0.1, 0.2]])


def test_loop135_apply_selector_keeps_baseline_by_default():
    pair = _pair()
    diff_indices = np.asarray([0, 1, 2], dtype=np.int64)
    scores = np.asarray([0.9, 0.1, 0.9], dtype=np.float32)

    predictions, accept = loop135.apply_selector(pair, diff_indices, scores, 0.5)

    assert accept.tolist() == [True, False, True]
    assert predictions.tolist() == [0, 0, 0]


def test_loop135_apply_selector_directional_uses_direction_thresholds():
    pair = _pair()
    diff_indices = np.asarray([0, 1, 2], dtype=np.int64)
    scores = np.asarray([0.8, 0.6, 0.7], dtype=np.float32)

    predictions, accept = loop135.apply_selector_directional(
        pair,
        diff_indices,
        scores,
        threshold_0to1=0.65,
        threshold_1to0=0.75,
    )

    assert accept.tolist() == [True, False, False]
    assert predictions.tolist() == [0, 0, 1]


def test_loop135_write_predictions_deduplicates_selector_fields(tmp_path: Path):
    pair = loop135.AlignedPair(
        rows=[
            {
                "source_sha256": "a",
                "sample_index": "1",
                "label": "0",
                "stage2_prob_malicious": "0.8",
                "prediction": "1",
                "correct": "False",
                "baseline_prob_malicious": "old",
            }
        ],
        labels=np.asarray([0], dtype=np.int64),
        baseline_prob=np.asarray([0.8], dtype=np.float32),
        candidate_prob=np.asarray([0.2], dtype=np.float32),
        baseline_pred=np.asarray([1], dtype=np.int64),
        candidate_pred=np.asarray([0], dtype=np.int64),
    )
    output = tmp_path / "predictions.csv"

    loop135.write_predictions(
        output,
        pair,
        np.asarray([0.9], dtype=np.float32),
        np.asarray([0], dtype=np.int64),
        np.asarray([True], dtype=bool),
    )

    header = output.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert len(header) == len(set(header))
    assert header.count("baseline_prob_malicious") == 1


def test_loop135_evaluator_applies_directional_payload(tmp_path: Path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    fieldnames = ["source_sha256", "sample_index", "label", "stage2_prob_malicious", "prediction"]
    baseline_rows = [
        {"source_sha256": "a", "sample_index": "1", "label": "0", "stage2_prob_malicious": "0.8", "prediction": "1"},
        {"source_sha256": "b", "sample_index": "2", "label": "1", "stage2_prob_malicious": "0.4", "prediction": "0"},
        {"source_sha256": "c", "sample_index": "3", "label": "1", "stage2_prob_malicious": "0.8", "prediction": "1"},
    ]
    candidate_rows = [
        {"source_sha256": "a", "sample_index": "1", "label": "0", "stage2_prob_malicious": "0.2", "prediction": "0"},
        {"source_sha256": "b", "sample_index": "2", "label": "1", "stage2_prob_malicious": "0.7", "prediction": "1"},
        {"source_sha256": "c", "sample_index": "3", "label": "1", "stage2_prob_malicious": "0.2", "prediction": "0"},
    ]
    for path, rows in ((baseline, baseline_rows), (candidate, candidate_rows)):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    payload = {
        "model": _FixedSelectorModel([0.8, 0.6, 0.7]),
        "selected": {
            "threshold_mode": "directional",
            "thresholds_by_direction": {
                "baseline0_candidate1": 0.65,
                "baseline1_candidate0": 0.75,
            },
        },
        "feature_names": loop135.SCORE_FEATURE_NAMES,
        "key_columns": ("sample_index", "source_sha256"),
        "baseline_score_column": "stage2_prob_malicious",
        "candidate_score_column": "stage2_prob_malicious",
    }
    model_path = tmp_path / "selector.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle)
    output_json = tmp_path / "eval.json"
    output_csv = tmp_path / "predictions.csv"

    assert eval_loop135.main(
        [
            "--model",
            str(model_path),
            "--baseline-predictions",
            str(baseline),
            "--candidate-predictions",
            str(candidate),
            "--output-json",
            str(output_json),
            "--output-predictions-csv",
            str(output_csv),
        ]
    ) == 0

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["threshold_mode"] == "directional"
    assert report["threshold_0to1"] == 0.65
    assert report["threshold_1to0"] == 0.75
    assert report["accepted_candidate"] == 1
    assert report["metrics"]["errors"] == 1


def test_loop135_val_improvement_constraint_rejects_non_improving_candidate():
    baseline = {"errors": 10, "f1": 0.99}
    tied = {"errors": 10, "f1": 0.99}
    worse_error = {"errors": 11, "f1": 0.991}

    assert not loop135.passes_val_improvement_constraint(
        tied,
        baseline,
        require_val_improvement=True,
        min_val_error_reduction=1,
        min_val_f1_delta=0.0,
    )
    assert not loop135.passes_val_improvement_constraint(
        worse_error,
        baseline,
        require_val_improvement=True,
        min_val_error_reduction=1,
        min_val_f1_delta=0.0,
    )


def test_loop135_val_improvement_constraint_accepts_error_and_f1_gain():
    baseline = {"errors": 10, "f1": 0.99}
    improved = {"errors": 9, "f1": 0.9901}

    assert loop135.passes_val_improvement_constraint(
        improved,
        baseline,
        require_val_improvement=True,
        min_val_error_reduction=1,
        min_val_f1_delta=0.0,
    )


def test_loop135_val_improvement_constraint_can_be_disabled():
    baseline = {"errors": 10, "f1": 0.99}
    worse = {"errors": 12, "f1": 0.98}

    assert loop135.passes_val_improvement_constraint(
        worse,
        baseline,
        require_val_improvement=False,
        min_val_error_reduction=1,
        min_val_f1_delta=0.0,
    )


def test_loop135_align_pair_rejects_duplicate_candidate_keys(tmp_path: Path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    fieldnames = ["source_sha256", "sample_index", "label", "stage2_prob_malicious", "prediction"]
    with baseline.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "source_sha256": "a",
                "sample_index": "1",
                "label": "0",
                "stage2_prob_malicious": "0.8",
                "prediction": "1",
            }
        )
    with candidate.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(2):
            writer.writerow(
                {
                    "source_sha256": "a",
                    "sample_index": "1",
                    "label": "0",
                    "stage2_prob_malicious": "0.2",
                    "prediction": "0",
                }
            )

    try:
        loop135.align_pair(
            baseline,
            candidate,
            key_columns=("sample_index", "source_sha256"),
            baseline_score_column="stage2_prob_malicious",
            candidate_score_column="stage2_prob_malicious",
        )
    except ValueError as exc:
        assert "duplicate alignment keys" in str(exc)
    else:
        raise AssertionError("duplicate candidate keys should fail")
