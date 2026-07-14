from __future__ import annotations

import numpy as np

from src.loop167_phase_b.counterfactual import (
    assert_counterfactual_api_has_no_identity_or_label_surface,
    permute_complete_novel_blocks,
)

PROTOCOL_SHA256 = "a" * 64


def test_counterfactual_is_deterministic_role_separated_and_preserves_missing_rows() -> None:
    blocks = np.arange(6 * 292, dtype=np.float32).reshape(6, 292)
    complete = np.array([True, False, True, True, False, True])
    first, first_permutation = permute_complete_novel_blocks(
        blocks,
        complete,
        protocol_sha256=PROTOCOL_SHA256,
        seed=41,
        outer_fold=2,
        role="fit",
    )
    second, second_permutation = permute_complete_novel_blocks(
        blocks,
        complete,
        protocol_sha256=PROTOCOL_SHA256,
        seed=41,
        outer_fold=2,
        role="fit",
    )
    holdout, holdout_permutation = permute_complete_novel_blocks(
        blocks,
        complete,
        protocol_sha256=PROTOCOL_SHA256,
        seed=41,
        outer_fold=2,
        role="holdout",
    )
    assert np.array_equal(first, second)
    assert np.array_equal(first_permutation, second_permutation)
    assert np.array_equal(first[~complete], blocks[~complete])
    assert sorted(first_permutation[complete].tolist()) == sorted(np.flatnonzero(complete).tolist())
    assert not np.array_equal(first_permutation, holdout_permutation)
    assert np.array_equal(holdout[~complete], blocks[~complete])


def test_counterfactual_rejects_bad_shapes_and_identity_surface() -> None:
    assert_counterfactual_api_has_no_identity_or_label_surface()
    with np.testing.assert_raises_regex(ValueError, "shape"):
        permute_complete_novel_blocks(
            np.zeros((2, 291), dtype=np.float32),
            np.array([True, True]),
            protocol_sha256=PROTOCOL_SHA256,
            seed=1,
            outer_fold=0,
            role="fit",
        )
