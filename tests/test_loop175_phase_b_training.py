from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.model import RegionNet, RegionNetConfig  # noqa: E402
from src.loop175.phase_b_training import (  # noqa: E402
    B0_FEATURE_DIMENSION,
    FROZEN_B0_HGB_PARAMETERS,
    FailClosedRegionNet,
    build_e_residual_weights,
    deterministic_region_record_permutation,
    generate_b0_inner_oof_scores,
    shuffle_region_record_ownership,
    strict_hard_decisions,
)


def _region_batch(rows: int = 2) -> tuple[torch.Tensor, ...]:
    tokens = torch.full((rows, 3, 32), 256, dtype=torch.int64)
    lengths = torch.tensor([[32, 17, 0]] * rows, dtype=torch.int64)
    for row in range(rows):
        tokens[row, 0, :] = row + 1
        tokens[row, 1, :17] = row + 11
    types = torch.tensor([[1, 3, 0]] * rows, dtype=torch.int64)
    offsets = torch.tensor([[0, 17, 0]] * rows, dtype=torch.int64)
    length_buckets = torch.tensor([[63, 34, 0]] * rows, dtype=torch.int64)
    b0 = torch.zeros((rows, B0_FEATURE_DIMENSION), dtype=torch.float32)
    return tokens, lengths, types, offsets, length_buckets, b0


def test_fail_closed_wrapper_accepts_valid_batch_and_rejects_silent_repairs() -> None:
    config = RegionNetConfig(
        model_dim=24,
        byte_embedding_dim=8,
        block_count=1,
        block_expansion=2,
        dilations=(1,),
        transformer_layers=1,
        transformer_heads=3,
        transformer_ffn_dim=48,
    )
    model = FailClosedRegionNet(
        RegionNet(config),
        expected_regions=3,
        expected_region_bytes=32,
    ).eval()
    valid = _region_batch()
    with torch.no_grad():
        output = model(*valid)
    assert output["fusion_logits"].shape == (2, 2)

    invalid_token = list(valid)
    invalid_token[0] = invalid_token[0].clone()
    invalid_token[0][0, 0, 0] = 257
    with pytest.raises(ValueError, match="out-of-range token"):
        model(*invalid_token)

    invalid_padding = list(valid)
    invalid_padding[0] = invalid_padding[0].clone()
    invalid_padding[0][0, 1, 20] = 7
    with pytest.raises(ValueError, match="padding bytes"):
        model(*invalid_padding)

    invalid_type = list(valid)
    invalid_type[2] = invalid_type[2].clone()
    invalid_type[2][0, 0] = 6
    with pytest.raises(ValueError, match="region_types"):
        model(*invalid_type)

    invalid_bucket = list(valid)
    invalid_bucket[3] = invalid_bucket[3].clone()
    invalid_bucket[3][0, 0] = 64
    with pytest.raises(ValueError, match="offset_buckets"):
        model(*invalid_bucket)

    invalid_b0 = list(valid)
    invalid_b0[5] = invalid_b0[5].clone()
    invalid_b0[5][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(*invalid_b0)


def test_frozen_b0_parameters_and_strict_threshold_are_exact() -> None:
    assert dict(FROZEN_B0_HGB_PARAMETERS) == {
        "loss": "log_loss",
        "learning_rate": 0.06,
        "max_iter": 260,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_bins": 255,
        "early_stopping": False,
        "random_state": 41,
    }
    np.testing.assert_array_equal(
        strict_hard_decisions(np.array([0.0, 0.5, np.nextafter(0.5, 1.0), 1.0])),
        np.array([0, 0, 1, 1], dtype=np.uint8),
    )


class _RecordingEstimator:
    classes_ = np.array([0, 1], dtype=np.uint8)

    def __init__(self, fit_ids: np.ndarray) -> None:
        self.fit_ids = fit_ids

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        score = np.full(values.shape[0], 0.6, dtype=np.float64)
        return np.column_stack((1.0 - score, score))


def test_inner_oof_uses_only_four_outer_fit_folds(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = 50
    values = np.zeros((rows, B0_FEATURE_DIMENSION), dtype=np.float32)
    values[:, 0] = np.arange(rows)
    folds = np.arange(rows, dtype=np.int64) % 5
    labels = (np.arange(rows, dtype=np.int64) // 5) % 2
    fitted_ids: list[set[int]] = []

    def fake_fit(matrix: np.ndarray, targets: np.ndarray) -> _RecordingEstimator:
        del targets
        ids = matrix[:, 0].astype(int)
        fitted_ids.append(set(ids.tolist()))
        return _RecordingEstimator(ids)

    monkeypatch.setattr(
        "src.loop175.phase_b_training.fit_frozen_b0_hgb",
        fake_fit,
    )
    result = generate_b0_inner_oof_scores(
        values,
        labels,
        folds,
        outer_holdout_fold=2,
    )

    outer_ids = set(np.flatnonzero(folds == 2).tolist())
    assert len(fitted_ids) == 4
    assert all(ids.isdisjoint(outer_ids) for ids in fitted_ids)
    assert set(result.row_indices.tolist()) == set(np.flatnonzero(folds != 2).tolist())
    assert set(result.inner_folds.tolist()) == {0, 1, 3, 4}
    assert result.scores.shape == (40,)
    assert np.isfinite(result.scores).all()


def test_e_weights_use_frozen_error_near_and_ordinary_rules() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    scores = np.array([0.8, 0.35, 0.1, 0.2, 0.65, 0.9], dtype=np.float64)
    weights = build_e_residual_weights(labels, scores)
    expected = np.array([2.0, 0.75, 0.25, 2.0, 0.75, 0.25], dtype=np.float32)
    np.testing.assert_allclose(weights, expected, rtol=0.0, atol=0.0)
    assert float(weights.mean()) == pytest.approx(1.0)
    assert float(weights.max()) <= 8.0


def test_d_shuffle_moves_the_entire_record_with_zero_fixed_points() -> None:
    values = list(_region_batch(rows=7))
    for row in range(7):
        values[2][row, 0] = 1 + row % 5
        values[3][row, 0] = row
        values[4][row, 0] = 63 - row
    shuffled = shuffle_region_record_ownership(
        *values[:5],
        protocol_sha256="a" * 64,
        seed=41,
        outer_fold=3,
        role="fit",
        expected_regions=3,
        expected_region_bytes=32,
    )
    permutation = deterministic_region_record_permutation(
        7,
        protocol_sha256="a" * 64,
        seed=41,
        outer_fold=3,
        role="fit",
    )
    np.testing.assert_array_equal(shuffled.permutation, permutation)
    assert not np.any(permutation == np.arange(7))
    indices = torch.as_tensor(permutation.copy(), dtype=torch.long)
    for original, candidate in zip(values[:5], (
        shuffled.region_tokens,
        shuffled.region_lengths,
        shuffled.region_types,
        shuffled.offset_buckets,
        shuffled.length_buckets,
    )):
        torch.testing.assert_close(candidate, original.index_select(0, indices))

    changed_protocol = deterministic_region_record_permutation(
        7,
        protocol_sha256="b" * 64,
        seed=41,
        outer_fold=3,
        role="fit",
    )
    assert not np.array_equal(permutation, changed_protocol)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        deterministic_region_record_permutation(
            7,
            protocol_sha256="A" * 64,
            seed=41,
            outer_fold=3,
            role="fit",
        )
