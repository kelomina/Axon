from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.loop167_phase_b import arm_contract
from src.loop167_phase_b.arm_contract import (
    B0_MATRIX_DIMENSION,
    B1_MATRIX_DIMENSION,
    FULL_TRAIN_ROWS,
    MATRIX_DIMENSIONS,
    NOVEL_MATRIX_DIMENSION,
    assert_deterministic_matrix_replay,
    assert_deterministic_replay_hashes,
    assert_novel_missing_fallback,
    build_arm_matrices,
    evaluation_replay_hash,
    finalize_novel_arm_evaluation,
    hard_decisions,
    matrix_replay_hash,
    select_global_primary_control,
)

PROTOCOL_SHA256 = "a" * 64


def _inputs(rows: int = 4) -> tuple[np.ndarray, ...]:
    return (
        np.arange(rows * 571, dtype=np.float32).reshape(rows, 571),
        np.zeros((rows, 6), dtype=np.float32),
        np.arange(rows * 536, dtype=np.float32).reshape(rows, 536),
        np.zeros((rows, 4), dtype=np.float32),
        np.arange(rows * 292, dtype=np.float32).reshape(rows, 292),
        np.array([(index % 2) == 0 for index in range(rows)], dtype=bool),
    )


def _build(replay_seed: int):
    return build_arm_matrices(
        *_inputs(),
        protocol_sha256=PROTOCOL_SHA256,
        replay_seed=replay_seed,
        outer_fold=2,
        role="fit",
    )


def test_arm_matrices_freeze_all_dimensions_and_canonicalize_replay_seed() -> None:
    replays = {seed: _build(seed) for seed in (41, 42, 43)}
    first = replays[41]

    assert first.b0.shape == (4, B0_MATRIX_DIMENSION)
    assert first.b1.shape == (4, B1_MATRIX_DIMENSION)
    assert first.a.shape == (4, NOVEL_MATRIX_DIMENSION)
    assert first.m.shape == (4, MATRIX_DIMENSIONS["M"])
    assert first.cf.shape == (4, MATRIX_DIMENSIONS["CF"])
    assert first.a[:, -1].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert np.array_equal(first.m[:, :B0_MATRIX_DIMENSION], first.b0)
    assert np.array_equal(first.cf[:, :B0_MATRIX_DIMENSION], first.b0)
    assert all(not replays[seed].b0.flags.writeable for seed in replays)
    assert len({matrix_replay_hash(replays[seed]) for seed in replays}) == 1
    assert assert_deterministic_matrix_replay(replays) == matrix_replay_hash(first)
    assert np.array_equal(replays[41].cf, replays[42].cf)
    assert np.array_equal(replays[42].counterfactual_permutation, replays[43].counterfactual_permutation)


def test_arm_contract_rejects_shape_nonfinite_and_unapproved_replay_inputs() -> None:
    values = list(_inputs())
    values[0] = np.zeros((4, 570), dtype=np.float32)
    with pytest.raises(ValueError, match="b0_values.*shape"):
        build_arm_matrices(
            *values,
            protocol_sha256=PROTOCOL_SHA256,
            replay_seed=41,
            outer_fold=0,
            role="fit",
        )

    values = list(_inputs())
    values[2][0, 0] = np.nan
    with pytest.raises(ValueError, match="b1_values.*finite"):
        build_arm_matrices(
            *values,
            protocol_sha256=PROTOCOL_SHA256,
            replay_seed=41,
            outer_fold=0,
            role="fit",
        )

    values = list(_inputs())
    values[5] = np.ones(4, dtype=np.uint8)
    with pytest.raises(ValueError, match="novel_complete.*boolean"):
        build_arm_matrices(
            *values,
            protocol_sha256=PROTOCOL_SHA256,
            replay_seed=41,
            outer_fold=0,
            role="fit",
        )

    values = list(_inputs())
    values[3][0, 0] = 0.5
    with pytest.raises(ValueError, match="b1_missing_indicators.*binary"):
        build_arm_matrices(
            *values,
            protocol_sha256=PROTOCOL_SHA256,
            replay_seed=41,
            outer_fold=0,
            role="fit",
        )

    with pytest.raises(ValueError, match="replay_seed"):
        build_arm_matrices(
            *_inputs(),
            protocol_sha256=PROTOCOL_SHA256,
            replay_seed=40,
            outer_fold=0,
            role="fit",
        )


def test_m_and_cf_novel_missing_rows_copy_b0_scores_and_decisions_bitwise() -> None:
    complete = np.array([True, False, True, False], dtype=bool)
    b0_scores = np.array([0.2, 0.5, 0.9, 0.0], dtype=np.float64)
    candidate_scores = np.array([0.8, 0.9, 0.1, 0.9], dtype=np.float64)

    for arm in ("M", "CF"):
        result = finalize_novel_arm_evaluation(
            arm,
            b0_scores=b0_scores,
            arm_scores=candidate_scores,
            novel_complete=complete,
        )
        assert result.scores[~complete].tobytes() == b0_scores[~complete].tobytes()
        assert result.hard_decisions[~complete].tobytes() == hard_decisions(b0_scores)[~complete].tobytes()
        assert result.hard_decisions.tolist() == [1, 0, 0, 0]

        wrong_scores = result.scores.copy()
        wrong_scores[1] = 0.7
        with pytest.raises(ValueError, match="score fallback"):
            assert_novel_missing_fallback(
                arm,
                b0_scores=b0_scores,
                arm_scores=wrong_scores,
                b0_hard_decisions=hard_decisions(b0_scores),
                arm_hard_decisions=hard_decisions(wrong_scores),
                novel_complete=complete,
            )

        wrong_decisions = result.hard_decisions.copy()
        wrong_decisions[3] = 1
        with pytest.raises(ValueError, match="hard-decision fallback"):
            assert_novel_missing_fallback(
                arm,
                b0_scores=b0_scores,
                arm_scores=result.scores,
                b0_hard_decisions=hard_decisions(b0_scores),
                arm_hard_decisions=wrong_decisions,
                novel_complete=complete,
            )


def test_global_primary_control_requires_all_20k_rows_and_uses_b0_ties() -> None:
    b0_errors = np.zeros(FULL_TRAIN_ROWS, dtype=bool)
    b1_errors = np.zeros(FULL_TRAIN_ROWS, dtype=bool)
    tied = select_global_primary_control(b0_errors, b1_errors)
    assert tied.arm == "B0"
    assert tied.error_count == 0

    b1_errors[:3] = True
    b0_wins = select_global_primary_control(b0_errors, b1_errors)
    assert b0_wins.arm == "B0"

    b0_errors[:5] = True
    b1_wins = select_global_primary_control(b0_errors, b1_errors)
    assert b1_wins.arm == "B1"
    assert b1_wins.error_count == 3

    with pytest.raises(ValueError, match="exactly 20000"):
        select_global_primary_control(b0_errors[:-1], b1_errors[:-1])


def test_replay_hashes_require_identical_41_42_43_results() -> None:
    complete = np.array([True, False, True], dtype=bool)
    b0_scores = np.array([0.2, 0.5, 0.9], dtype=np.float64)
    hash_value = evaluation_replay_hash(
        b0_scores=b0_scores,
        b1_scores=np.array([0.1, 0.6, 0.8]),
        m_scores=np.array([0.7, 0.8, 0.2]),
        a_scores=np.array([0.4, 0.5, 0.6]),
        cf_scores=np.array([0.3, 0.9, 0.1]),
        novel_complete=complete,
    )
    assert assert_deterministic_replay_hashes({41: hash_value, 42: hash_value, 43: hash_value}) == hash_value
    with pytest.raises(ValueError, match="differ"):
        assert_deterministic_replay_hashes({41: hash_value, 42: "b" * 64, 43: hash_value})
    with pytest.raises(ValueError, match="exactly these labels"):
        assert_deterministic_replay_hashes({41: hash_value, 42: hash_value})


def test_arm_contract_has_no_storage_or_model_io_surface() -> None:
    source = inspect.getsource(arm_contract)
    for forbidden in ("Path", ".open(", "np.load", "torch", "sklearn", "pandas"):
        assert forbidden not in source
