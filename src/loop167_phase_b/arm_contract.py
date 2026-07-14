"""Pure NumPy matrix and score contract for Loop167 Phase B arms."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .counterfactual import permute_complete_novel_blocks

B0_VALUE_DIMENSION = 571
B0_MISSING_DIMENSION = 6
B1_VALUE_DIMENSION = 536
B1_MISSING_DIMENSION = 4
NOVEL_VALUE_DIMENSION = 292
NOVEL_COMPLETE_DIMENSION = 1

B0_MATRIX_DIMENSION = B0_VALUE_DIMENSION + B0_MISSING_DIMENSION
B1_MATRIX_DIMENSION = B1_VALUE_DIMENSION + B1_MISSING_DIMENSION
NOVEL_MATRIX_DIMENSION = NOVEL_VALUE_DIMENSION + NOVEL_COMPLETE_DIMENSION
MATRIX_DIMENSIONS = {
    "B0": B0_MATRIX_DIMENSION,
    "B1": B1_MATRIX_DIMENSION,
    "M": B0_MATRIX_DIMENSION + NOVEL_MATRIX_DIMENSION,
    "A": NOVEL_MATRIX_DIMENSION,
    "CF": B0_MATRIX_DIMENSION + NOVEL_MATRIX_DIMENSION,
}

ARM_NAMES = ("B0", "B1", "M", "A", "CF")
NOVEL_FALLBACK_ARMS = frozenset({"M", "CF"})
REPLAY_SEEDS = (41, 42, 43)
CANONICAL_REPLAY_SEED = REPLAY_SEEDS[0]
FULL_TRAIN_ROWS = 20_000
HARD_DECISION_THRESHOLD = 0.5
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ArmMatrices:
    """Immutable feature matrices for the five preregistered arms."""

    b0: np.ndarray
    b1: np.ndarray
    m: np.ndarray
    a: np.ndarray
    cf: np.ndarray
    novel_complete: np.ndarray
    counterfactual_permutation: np.ndarray

    def for_arm(self, arm: str) -> np.ndarray:
        """Return one frozen arm matrix by its preregistered name."""

        matrices = {
            "B0": self.b0,
            "B1": self.b1,
            "M": self.m,
            "A": self.a,
            "CF": self.cf,
        }
        try:
            return matrices[arm]
        except KeyError as error:
            raise ValueError(f"Unknown Loop167 arm: {arm}") from error


@dataclass(frozen=True)
class ArmEvaluation:
    """Frozen final scores and strict-threshold decisions for one novel arm."""

    arm: str
    scores: np.ndarray
    hard_decisions: np.ndarray


@dataclass(frozen=True)
class GlobalPrimaryControl:
    """The sole global B0/B1 comparator for one complete 20k evaluation."""

    arm: str
    error_count: int
    b0_error_count: int
    b1_error_count: int


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values).copy()
    result.setflags(write=False)
    return result


def _float_matrix(values: np.ndarray, *, name: str, columns: int) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite float32 matrix") from error
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise ValueError(f"{name} must have shape [rows, {columns}]")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix)


def _binary_indicator_matrix(values: np.ndarray, *, name: str, columns: int) -> np.ndarray:
    matrix = _float_matrix(values, name=name, columns=columns)
    if not np.isin(matrix, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only binary indicators")
    return matrix


def _boolean_vector(values: np.ndarray, *, name: str, rows: int) -> np.ndarray:
    vector = np.asarray(values)
    if vector.dtype != np.dtype(bool):
        raise ValueError(f"{name} must have boolean dtype")
    if vector.shape != (rows,):
        raise ValueError(f"{name} must have one value per row")
    return np.ascontiguousarray(vector)


def _score_vector(values: np.ndarray, *, name: str, rows: int | None = None) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite probability vector") from error
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if rows is not None and vector.shape != (rows,):
        raise ValueError(f"{name} must have one value per row")
    if not np.isfinite(vector).all() or np.any(vector < 0.0) or np.any(vector > 1.0):
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    return np.ascontiguousarray(vector)


def _decision_vector(values: np.ndarray, *, name: str, rows: int) -> np.ndarray:
    vector = np.asarray(values)
    if vector.shape != (rows,):
        raise ValueError(f"{name} must have one value per row")
    if vector.dtype != np.dtype(bool) and not np.issubdtype(vector.dtype, np.integer):
        raise ValueError(f"{name} must be binary")
    if not np.isin(vector, (0, 1)).all():
        raise ValueError(f"{name} must be binary")
    return np.ascontiguousarray(vector, dtype=np.uint8)


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def _hard_decisions_from_scores(scores: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((scores > HARD_DECISION_THRESHOLD).astype(np.uint8))


def hard_decisions(scores: np.ndarray) -> np.ndarray:
    """Use the sealed strict threshold: a score exactly 0.5 decides benign."""

    normalized = _score_vector(scores, name="scores")
    return _readonly(_hard_decisions_from_scores(normalized))


def build_arm_matrices(
    b0_values: np.ndarray,
    b0_missing_indicators: np.ndarray,
    b1_values: np.ndarray,
    b1_missing_indicators: np.ndarray,
    novel_values: np.ndarray,
    novel_complete: np.ndarray,
    *,
    protocol_sha256: str,
    replay_seed: int,
    outer_fold: int,
    role: str,
) -> ArmMatrices:
    """Build the five fixed matrices without allowing replay labels to change data."""

    if isinstance(replay_seed, bool) or replay_seed not in REPLAY_SEEDS:
        raise ValueError(f"replay_seed must be one of {REPLAY_SEEDS}")
    b0_value_matrix = _float_matrix(b0_values, name="b0_values", columns=B0_VALUE_DIMENSION)
    rows = b0_value_matrix.shape[0]
    b0_missing_matrix = _binary_indicator_matrix(
        b0_missing_indicators,
        name="b0_missing_indicators",
        columns=B0_MISSING_DIMENSION,
    )
    b1_value_matrix = _float_matrix(b1_values, name="b1_values", columns=B1_VALUE_DIMENSION)
    b1_missing_matrix = _binary_indicator_matrix(
        b1_missing_indicators,
        name="b1_missing_indicators",
        columns=B1_MISSING_DIMENSION,
    )
    novel_value_matrix = _float_matrix(
        novel_values,
        name="novel_values",
        columns=NOVEL_VALUE_DIMENSION,
    )
    for name, matrix in (
        ("b0_missing_indicators", b0_missing_matrix),
        ("b1_values", b1_value_matrix),
        ("b1_missing_indicators", b1_missing_matrix),
        ("novel_values", novel_value_matrix),
    ):
        if matrix.shape[0] != rows:
            raise ValueError(f"{name} must have the same row count as b0_values")
    complete = _boolean_vector(novel_complete, name="novel_complete", rows=rows)

    b0_matrix = np.concatenate((b0_value_matrix, b0_missing_matrix), axis=1)
    b1_matrix = np.concatenate((b1_value_matrix, b1_missing_matrix), axis=1)
    novel_matrix = np.concatenate((novel_value_matrix, complete[:, None].astype(np.float32)), axis=1)
    cf_values, permutation = permute_complete_novel_blocks(
        novel_value_matrix,
        complete,
        protocol_sha256=protocol_sha256,
        seed=CANONICAL_REPLAY_SEED,
        outer_fold=outer_fold,
        role=role,
    )
    if not np.array_equal(cf_values[~complete], novel_value_matrix[~complete]):
        raise RuntimeError("counterfactual permutation changed a novel-missing row")
    cf_novel_matrix = np.concatenate((cf_values, complete[:, None].astype(np.float32)), axis=1)
    matrices = ArmMatrices(
        b0=_readonly(b0_matrix),
        b1=_readonly(b1_matrix),
        m=_readonly(np.concatenate((b0_matrix, novel_matrix), axis=1)),
        a=_readonly(novel_matrix),
        cf=_readonly(np.concatenate((b0_matrix, cf_novel_matrix), axis=1)),
        novel_complete=_readonly(complete),
        counterfactual_permutation=_readonly(permutation),
    )
    for arm, expected_dimension in MATRIX_DIMENSIONS.items():
        matrix = matrices.for_arm(arm)
        if matrix.shape != (rows, expected_dimension):
            raise RuntimeError(f"{arm} matrix dimension drift")
    return matrices


def assert_novel_missing_fallback(
    arm: str,
    *,
    b0_scores: np.ndarray,
    arm_scores: np.ndarray,
    b0_hard_decisions: np.ndarray,
    arm_hard_decisions: np.ndarray,
    novel_complete: np.ndarray,
) -> None:
    """Reject an M or CF result that did not copy B0 on a novel-missing row."""

    if arm not in NOVEL_FALLBACK_ARMS:
        raise ValueError("only M and CF use the novel-missing fallback")
    baseline = _score_vector(b0_scores, name="b0_scores")
    rows = baseline.size
    candidate = _score_vector(arm_scores, name=f"{arm}_scores", rows=rows)
    complete = _boolean_vector(novel_complete, name="novel_complete", rows=rows)
    baseline_decisions = _decision_vector(
        b0_hard_decisions,
        name="b0_hard_decisions",
        rows=rows,
    )
    candidate_decisions = _decision_vector(
        arm_hard_decisions,
        name=f"{arm}_hard_decisions",
        rows=rows,
    )
    expected_baseline_decisions = _hard_decisions_from_scores(baseline)
    expected_candidate_decisions = _hard_decisions_from_scores(candidate)
    if not _bitwise_equal(baseline_decisions, expected_baseline_decisions):
        raise ValueError("b0_hard_decisions must use the sealed strict threshold")
    missing = ~complete
    if not _bitwise_equal(baseline[missing], candidate[missing]):
        raise ValueError(f"{arm} score fallback differs from B0 on a novel-missing row")
    if not _bitwise_equal(baseline_decisions[missing], candidate_decisions[missing]):
        raise ValueError(f"{arm} hard-decision fallback differs from B0 on a novel-missing row")
    if not _bitwise_equal(candidate_decisions, expected_candidate_decisions):
        raise ValueError(f"{arm}_hard_decisions must use the sealed strict threshold")


def finalize_novel_arm_evaluation(
    arm: str,
    *,
    b0_scores: np.ndarray,
    arm_scores: np.ndarray,
    novel_complete: np.ndarray,
) -> ArmEvaluation:
    """Apply the exact B0 fallback before recording M or CF evaluation outputs."""

    if arm not in NOVEL_FALLBACK_ARMS:
        raise ValueError("only M and CF use the novel-missing fallback")
    baseline = _score_vector(b0_scores, name="b0_scores")
    rows = baseline.size
    candidate = _score_vector(arm_scores, name=f"{arm}_scores", rows=rows)
    complete = _boolean_vector(novel_complete, name="novel_complete", rows=rows)
    final_scores = candidate.copy()
    final_scores[~complete] = baseline[~complete]
    baseline_decisions = _hard_decisions_from_scores(baseline)
    final_decisions = _hard_decisions_from_scores(final_scores)
    assert_novel_missing_fallback(
        arm,
        b0_scores=baseline,
        arm_scores=final_scores,
        b0_hard_decisions=baseline_decisions,
        arm_hard_decisions=final_decisions,
        novel_complete=complete,
    )
    return ArmEvaluation(
        arm=arm,
        scores=_readonly(final_scores),
        hard_decisions=_readonly(final_decisions),
    )


def _error_mask(values: np.ndarray, *, name: str) -> np.ndarray:
    mask = np.asarray(values)
    if mask.dtype != np.dtype(bool):
        raise ValueError(f"{name} must have boolean dtype")
    if mask.shape != (FULL_TRAIN_ROWS,):
        raise ValueError(f"{name} must contain exactly {FULL_TRAIN_ROWS} rows")
    return np.ascontiguousarray(mask)


def select_global_primary_control(
    b0_errors: np.ndarray,
    b1_errors: np.ndarray,
) -> GlobalPrimaryControl:
    """Select the lower-error B0/B1 control on all 20,000 rows, with B0 ties."""

    b0_mask = _error_mask(b0_errors, name="b0_errors")
    b1_mask = _error_mask(b1_errors, name="b1_errors")
    b0_count = int(np.count_nonzero(b0_mask))
    b1_count = int(np.count_nonzero(b1_mask))
    if b0_count <= b1_count:
        return GlobalPrimaryControl("B0", b0_count, b0_count, b1_count)
    return GlobalPrimaryControl("B1", b1_count, b0_count, b1_count)


def _hash_array(digest: Any, name: str, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values)
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes())


def matrix_replay_hash(matrices: ArmMatrices) -> str:
    """Hash all frozen matrices without including a replay-run label."""

    digest = hashlib.sha256(b"loop167-phase-b-matrix-replay-v1\0")
    for arm in ARM_NAMES:
        _hash_array(digest, arm, matrices.for_arm(arm))
    _hash_array(digest, "novel_complete", matrices.novel_complete)
    _hash_array(digest, "counterfactual_permutation", matrices.counterfactual_permutation)
    return digest.hexdigest()


def evaluation_replay_hash(
    *,
    b0_scores: np.ndarray,
    b1_scores: np.ndarray,
    m_scores: np.ndarray,
    a_scores: np.ndarray,
    cf_scores: np.ndarray,
    novel_complete: np.ndarray,
) -> str:
    """Hash final score and decision arrays after enforcing the M/CF fallback."""

    baseline = _score_vector(b0_scores, name="b0_scores")
    rows = baseline.size
    b1 = _score_vector(b1_scores, name="b1_scores", rows=rows)
    a = _score_vector(a_scores, name="a_scores", rows=rows)
    complete = _boolean_vector(novel_complete, name="novel_complete", rows=rows)
    m = finalize_novel_arm_evaluation(
        "M",
        b0_scores=baseline,
        arm_scores=m_scores,
        novel_complete=complete,
    )
    cf = finalize_novel_arm_evaluation(
        "CF",
        b0_scores=baseline,
        arm_scores=cf_scores,
        novel_complete=complete,
    )
    digest = hashlib.sha256(b"loop167-phase-b-evaluation-replay-v1\0")
    for arm, scores, decisions in (
        ("B0", baseline, _hard_decisions_from_scores(baseline)),
        ("B1", b1, _hard_decisions_from_scores(b1)),
        ("M", m.scores, m.hard_decisions),
        ("A", a, _hard_decisions_from_scores(a)),
        ("CF", cf.scores, cf.hard_decisions),
    ):
        _hash_array(digest, f"{arm}_scores", scores)
        _hash_array(digest, f"{arm}_hard_decisions", decisions)
    _hash_array(digest, "novel_complete", complete)
    return digest.hexdigest()


def assert_deterministic_replay_hashes(replay_hashes: Mapping[int, str]) -> str:
    """Require the three replay labels to report one identical SHA-256 digest."""

    if set(replay_hashes) != set(REPLAY_SEEDS):
        raise ValueError(f"replay hashes must have exactly these labels: {REPLAY_SEEDS}")
    hashes = []
    for seed in REPLAY_SEEDS:
        value = replay_hashes[seed]
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise ValueError("each replay hash must be a lowercase SHA-256 digest")
        hashes.append(value)
    if len(set(hashes)) != 1:
        raise ValueError("deterministic replay hashes differ across 41/42/43")
    return hashes[0]


def assert_deterministic_matrix_replay(replays: Mapping[int, ArmMatrices]) -> str:
    """Require every replay label to use the same complete matrix bundle."""

    return assert_deterministic_replay_hashes(
        {seed: matrix_replay_hash(matrices) for seed, matrices in replays.items()}
    )
