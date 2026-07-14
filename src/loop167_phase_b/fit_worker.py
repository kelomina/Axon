"""In-memory, fixed-grid fitting for the unexecuted Loop167 Phase-B study."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .arm_contract import (
    ARM_NAMES,
    B0_MISSING_DIMENSION,
    B0_VALUE_DIMENSION,
    B1_MISSING_DIMENSION,
    B1_VALUE_DIMENSION,
    CANONICAL_REPLAY_SEED,
    FULL_TRAIN_ROWS,
    MATRIX_DIMENSIONS,
    NOVEL_FALLBACK_ARMS,
    NOVEL_VALUE_DIMENSION,
    REPLAY_SEEDS,
    ArmEvaluation,
    ArmMatrices,
    GlobalPrimaryControl,
    assert_deterministic_replay_hashes,
    assert_novel_missing_fallback,
    build_arm_matrices,
    evaluation_replay_hash,
    finalize_novel_arm_evaluation,
    hard_decisions,
    select_global_primary_control,
)
from .progress_ledger import EXPECTED_FIT_UNIT_COUNT, FitLedger

OUTER_FOLD_COUNT = 5
PRODUCTION_ROWS_PER_FOLD = FULL_TRAIN_ROWS // OUTER_FOLD_COUNT
FROZEN_HGB_PARAMETERS = MappingProxyType(
    {
        "loss": "log_loss",
        "learning_rate": 0.06,
        "max_iter": 260,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_bins": 255,
        "early_stopping": False,
        "random_state": CANONICAL_REPLAY_SEED,
    }
)


@dataclass(frozen=True)
class PhaseBFeatureCache:
    """The only feature payload accepted by the fitting boundary."""

    b0_values: np.ndarray
    b0_missing_indicators: np.ndarray
    b1_values: np.ndarray
    b1_missing_indicators: np.ndarray
    novel_values: np.ndarray
    novel_complete: np.ndarray


@dataclass(frozen=True)
class PhaseBFitInputSummary:
    """Validated shape facts for one closed fit input."""

    row_count: int
    rows_per_fold: int
    synthetic: bool


@dataclass(frozen=True)
class PhaseBFitResult:
    """Immutable out-of-fold evaluations from the complete fixed fit grid."""

    replay_evaluations: Mapping[int, Mapping[str, ArmEvaluation]]
    matrix_replay_sha256: str
    evaluation_replay_sha256: str
    fit_ledger_final_record_sha256: str
    total_fit_units: int
    primary_controls: Mapping[int, GlobalPrimaryControl] | None


@dataclass(frozen=True)
class _ValidatedFitInput:
    cache: PhaseBFeatureCache
    labels: np.ndarray
    folds: np.ndarray
    summary: PhaseBFitInputSummary


@dataclass(frozen=True)
class _CounterfactualFoldMatrices:
    train_indices: np.ndarray
    heldout_indices: np.ndarray
    train_matrix: np.ndarray
    heldout_matrix: np.ndarray


@dataclass(frozen=True)
class _BaseArmMatrices:
    b0: np.ndarray
    b1: np.ndarray
    m: np.ndarray
    a: np.ndarray

    def for_arm(self, arm: str) -> np.ndarray:
        matrices = {"B0": self.b0, "B1": self.b1, "M": self.m, "A": self.a}
        try:
            return matrices[arm]
        except KeyError as error:
            raise ValueError(f"base matrix is unavailable for arm {arm}") from error


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
    return matrix


def _indicator_matrix(values: np.ndarray, *, name: str, columns: int, rows: int) -> np.ndarray:
    matrix = _float_matrix(values, name=name, columns=columns)
    if matrix.shape[0] != rows:
        raise ValueError(f"{name} must have the same row count as b0_values")
    if not np.isin(matrix, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only binary indicators")
    return matrix


def _boolean_vector(values: np.ndarray, *, name: str, rows: int) -> np.ndarray:
    vector = np.asarray(values)
    if vector.dtype != np.dtype(bool):
        raise ValueError(f"{name} must have boolean dtype")
    if vector.shape != (rows,):
        raise ValueError(f"{name} must have one value per row")
    return vector


def _binary_labels(values: np.ndarray, *, rows: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.shape != (rows,):
        raise ValueError("labels must have one value per cached row")
    if labels.dtype != np.dtype(bool) and not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must have boolean or integer dtype")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary 0/1 values")
    normalized = np.ascontiguousarray(labels, dtype=np.uint8).copy()
    if np.unique(normalized).size != 2:
        raise ValueError("labels must contain both binary classes")
    normalized.setflags(write=False)
    return normalized


def _fold_vector(values: np.ndarray, *, rows: int, synthetic: bool) -> np.ndarray:
    folds = np.asarray(values)
    if folds.shape != (rows,):
        raise ValueError("folds must have one outer-fold value per cached row")
    if folds.dtype == np.dtype(bool) or not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("folds must have integer dtype")
    if not np.isin(folds, tuple(range(OUTER_FOLD_COUNT))).all():
        raise ValueError("folds must use exactly the fixed 0..4 outer-fold labels")
    normalized = np.ascontiguousarray(folds, dtype=np.int8).copy()
    expected_rows_per_fold = rows // OUTER_FOLD_COUNT
    if rows % OUTER_FOLD_COUNT != 0:
        raise ValueError("row count must divide exactly into five outer folds")
    counts = np.bincount(normalized, minlength=OUTER_FOLD_COUNT)
    if counts.shape != (OUTER_FOLD_COUNT,) or not np.all(counts == expected_rows_per_fold):
        raise ValueError("each fixed outer fold must contain the same number of rows")
    if not synthetic and expected_rows_per_fold != PRODUCTION_ROWS_PER_FOLD:
        raise ValueError("production folds must contain exactly 4000 rows each")
    normalized.setflags(write=False)
    return normalized


def _validate_cache(cache: PhaseBFeatureCache) -> PhaseBFeatureCache:
    if not isinstance(cache, PhaseBFeatureCache):
        raise TypeError("cache must be a PhaseBFeatureCache")
    b0_values = _float_matrix(cache.b0_values, name="b0_values", columns=B0_VALUE_DIMENSION)
    rows = b0_values.shape[0]
    b0_missing = _indicator_matrix(
        cache.b0_missing_indicators,
        name="b0_missing_indicators",
        columns=B0_MISSING_DIMENSION,
        rows=rows,
    )
    b1_values = _float_matrix(cache.b1_values, name="b1_values", columns=B1_VALUE_DIMENSION)
    if b1_values.shape[0] != rows:
        raise ValueError("b1_values must have the same row count as b0_values")
    b1_missing = _indicator_matrix(
        cache.b1_missing_indicators,
        name="b1_missing_indicators",
        columns=B1_MISSING_DIMENSION,
        rows=rows,
    )
    novel_values = _float_matrix(cache.novel_values, name="novel_values", columns=NOVEL_VALUE_DIMENSION)
    if novel_values.shape[0] != rows:
        raise ValueError("novel_values must have the same row count as b0_values")
    novel_complete = _boolean_vector(cache.novel_complete, name="novel_complete", rows=rows)
    return PhaseBFeatureCache(
        b0_values=b0_values,
        b0_missing_indicators=b0_missing,
        b1_values=b1_values,
        b1_missing_indicators=b1_missing,
        novel_values=novel_values,
        novel_complete=novel_complete,
    )


def _validate_fit_input(
    cache: PhaseBFeatureCache,
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    synthetic: bool,
) -> _ValidatedFitInput:
    normalized_cache = _validate_cache(cache)
    rows = normalized_cache.b0_values.shape[0]
    if not synthetic and rows != FULL_TRAIN_ROWS:
        raise ValueError("production fitting requires exactly 20000 cached rows")
    if synthetic and rows < OUTER_FOLD_COUNT:
        raise ValueError("synthetic fitting requires at least one row per outer fold")
    normalized_labels = _binary_labels(labels, rows=rows)
    normalized_folds = _fold_vector(folds, rows=rows, synthetic=synthetic)
    for fold in range(OUTER_FOLD_COUNT):
        if np.unique(normalized_labels[normalized_folds != fold]).size != 2:
            raise ValueError("each outer-fold training partition must contain both classes")
    return _ValidatedFitInput(
        cache=normalized_cache,
        labels=normalized_labels,
        folds=normalized_folds,
        summary=PhaseBFitInputSummary(
            row_count=rows,
            rows_per_fold=rows // OUTER_FOLD_COUNT,
            synthetic=synthetic,
        ),
    )


def validate_phase_b_fit_input(
    cache: PhaseBFeatureCache,
    labels: np.ndarray,
    folds: np.ndarray,
) -> PhaseBFitInputSummary:
    """Validate the non-bypassable 20,000-row production input without fitting."""

    return _validate_fit_input(cache, labels, folds, synthetic=False).summary


def _subset_cache(cache: PhaseBFeatureCache, indices: np.ndarray) -> PhaseBFeatureCache:
    return PhaseBFeatureCache(
        b0_values=cache.b0_values[indices],
        b0_missing_indicators=cache.b0_missing_indicators[indices],
        b1_values=cache.b1_values[indices],
        b1_missing_indicators=cache.b1_missing_indicators[indices],
        novel_values=cache.novel_values[indices],
        novel_complete=cache.novel_complete[indices],
    )


def _build_arm_matrices(
    cache: PhaseBFeatureCache,
    *,
    protocol_sha256: str,
    outer_fold: int,
    role: str,
) -> ArmMatrices:
    return build_arm_matrices(
        cache.b0_values,
        cache.b0_missing_indicators,
        cache.b1_values,
        cache.b1_missing_indicators,
        cache.novel_values,
        cache.novel_complete,
        protocol_sha256=protocol_sha256,
        replay_seed=CANONICAL_REPLAY_SEED,
        outer_fold=outer_fold,
        role=role,
    )


def _build_base_matrices(cache: PhaseBFeatureCache) -> _BaseArmMatrices:
    b0 = _readonly(np.concatenate((cache.b0_values, cache.b0_missing_indicators), axis=1))
    b1 = _readonly(np.concatenate((cache.b1_values, cache.b1_missing_indicators), axis=1))
    novel = np.concatenate(
        (cache.novel_values, cache.novel_complete[:, None].astype(np.float32)),
        axis=1,
    )
    base = _BaseArmMatrices(
        b0=b0,
        b1=b1,
        m=_readonly(np.concatenate((b0, novel), axis=1)),
        a=_readonly(novel),
    )
    for arm in ("B0", "B1", "M", "A"):
        if base.for_arm(arm).shape != (cache.b0_values.shape[0], MATRIX_DIMENSIONS[arm]):
            raise RuntimeError(f"{arm} base matrix dimension drift")
    return base


def _build_counterfactual_folds(
    cache: PhaseBFeatureCache,
    folds: np.ndarray,
    *,
    protocol_sha256: str,
) -> Mapping[int, _CounterfactualFoldMatrices]:
    partitions: dict[int, _CounterfactualFoldMatrices] = {}
    for fold in range(OUTER_FOLD_COUNT):
        heldout_indices = np.flatnonzero(folds == fold)
        train_indices = np.flatnonzero(folds != fold)
        train_matrices = _build_arm_matrices(
            _subset_cache(cache, train_indices),
            protocol_sha256=protocol_sha256,
            outer_fold=fold,
            role="fit",
        )
        heldout_matrices = _build_arm_matrices(
            _subset_cache(cache, heldout_indices),
            protocol_sha256=protocol_sha256,
            outer_fold=fold,
            role="holdout",
        )
        partitions[fold] = _CounterfactualFoldMatrices(
            train_indices=_readonly(train_indices),
            heldout_indices=_readonly(heldout_indices),
            train_matrix=train_matrices.cf,
            heldout_matrix=heldout_matrices.cf,
        )
    return MappingProxyType(partitions)


def _hash_array(digest: Any, name: str, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values)
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
    digest.update(contiguous.tobytes())


def _matrix_grid_sha256(
    base_matrices: _BaseArmMatrices,
    folds: np.ndarray,
    counterfactual_folds: Mapping[int, _CounterfactualFoldMatrices],
) -> str:
    digest = hashlib.sha256(b"axon_loop167_phase_b_fit_matrix_grid_v1\0")
    for arm in ("B0", "B1", "M", "A"):
        _hash_array(digest, arm, base_matrices.for_arm(arm))
    _hash_array(digest, "outer_folds", folds)
    for fold in range(OUTER_FOLD_COUNT):
        partition = counterfactual_folds[fold]
        _hash_array(digest, f"cf_fit_{fold}", partition.train_matrix)
        _hash_array(digest, f"cf_holdout_{fold}", partition.heldout_matrix)
    return digest.hexdigest()


def _synthetic_positive_probability(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    heldout_matrix: np.ndarray,
) -> np.ndarray:
    positive_mean = train_matrix[train_labels == 1].mean(axis=0, dtype=np.float64)
    negative_mean = train_matrix[train_labels == 0].mean(axis=0, dtype=np.float64)
    direction = positive_mean - negative_mean
    midpoint = (positive_mean + negative_mean) * 0.5
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm == 0.0:
        linear = np.zeros(heldout_matrix.shape[0], dtype=np.float64)
    else:
        linear = (heldout_matrix.astype(np.float64) - midpoint) @ direction / direction_norm
    positive_count = int(np.count_nonzero(train_labels == 1))
    prior = (positive_count + 1.0) / (train_labels.size + 2.0)
    prior_logit = np.log(prior / (1.0 - prior))
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-np.clip(linear + prior_logit, -36.0, 36.0)))


def _production_positive_probability(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    heldout_matrix: np.ndarray,
) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier

    estimator = HistGradientBoostingClassifier(**dict(FROZEN_HGB_PARAMETERS))
    estimator.fit(train_matrix, train_labels)
    if not np.array_equal(estimator.classes_, np.array([0, 1], dtype=np.uint8)):
        raise RuntimeError("frozen HGB fit did not retain the required binary classes")
    return np.asarray(estimator.predict_proba(heldout_matrix)[:, 1], dtype=np.float64)


def _positive_probability(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    heldout_matrix: np.ndarray,
    *,
    synthetic: bool,
) -> np.ndarray:
    scores = (
        _synthetic_positive_probability(train_matrix, train_labels, heldout_matrix)
        if synthetic
        else _production_positive_probability(train_matrix, train_labels, heldout_matrix)
    )
    if scores.shape != (heldout_matrix.shape[0],):
        raise RuntimeError("estimator did not return exactly one score per heldout row")
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise RuntimeError("estimator produced invalid probability scores")
    return np.ascontiguousarray(scores, dtype=np.float64)


def _finalize_arm_evaluation(
    arm: str,
    scores: np.ndarray,
    *,
    baseline: ArmEvaluation | None,
    novel_complete: np.ndarray,
) -> ArmEvaluation:
    if arm in NOVEL_FALLBACK_ARMS:
        if baseline is None:
            raise RuntimeError(f"{arm} requires the completed same-replay B0 baseline")
        finalized = finalize_novel_arm_evaluation(
            arm,
            b0_scores=baseline.scores,
            arm_scores=scores,
            novel_complete=novel_complete,
        )
        assert_novel_missing_fallback(
            arm,
            b0_scores=baseline.scores,
            arm_scores=finalized.scores,
            b0_hard_decisions=baseline.hard_decisions,
            arm_hard_decisions=finalized.hard_decisions,
            novel_complete=novel_complete,
        )
        return finalized
    return ArmEvaluation(arm=arm, scores=_readonly(scores), hard_decisions=hard_decisions(scores))


def _assert_identical_replays(
    replay_evaluations: Mapping[int, Mapping[str, ArmEvaluation]],
    *,
    novel_complete: np.ndarray,
) -> str:
    evaluation_hashes: dict[int, str] = {}
    canonical = replay_evaluations[CANONICAL_REPLAY_SEED]
    for replay_seed in REPLAY_SEEDS:
        current = replay_evaluations[replay_seed]
        for arm in ARM_NAMES:
            if current[arm].scores.tobytes() != canonical[arm].scores.tobytes():
                raise RuntimeError("deterministic replay scores differ across run labels")
            if current[arm].hard_decisions.tobytes() != canonical[arm].hard_decisions.tobytes():
                raise RuntimeError("deterministic replay decisions differ across run labels")
        evaluation_hashes[replay_seed] = evaluation_replay_hash(
            b0_scores=current["B0"].scores,
            b1_scores=current["B1"].scores,
            m_scores=current["M"].scores,
            a_scores=current["A"].scores,
            cf_scores=current["CF"].scores,
            novel_complete=novel_complete,
        )
    return assert_deterministic_replay_hashes(evaluation_hashes)


def _run_phase_b_fit(
    cache: PhaseBFeatureCache,
    labels: np.ndarray,
    folds: np.ndarray,
    ledger: FitLedger,
    *,
    fit_protocol_commitment_sha256: str,
    feature_rows_commitment_sha256: str,
    raw_ledger_final_record_sha256: str,
    synthetic: bool,
) -> PhaseBFitResult:
    if not isinstance(ledger, FitLedger):
        raise TypeError("ledger must be a FitLedger")
    validated = _validate_fit_input(cache, labels, folds, synthetic=synthetic)
    ledger.fit_started(
        fit_protocol_commitment_sha256=fit_protocol_commitment_sha256,
        feature_rows_commitment_sha256=feature_rows_commitment_sha256,
        raw_ledger_final_record_sha256=raw_ledger_final_record_sha256,
    )
    base_matrices = _build_base_matrices(validated.cache)
    counterfactual_folds = _build_counterfactual_folds(
        validated.cache,
        validated.folds,
        protocol_sha256=fit_protocol_commitment_sha256,
    )
    matrix_grid_sha256 = _matrix_grid_sha256(base_matrices, validated.folds, counterfactual_folds)
    matrix_replay_sha256 = assert_deterministic_replay_hashes(
        {replay_seed: matrix_grid_sha256 for replay_seed in REPLAY_SEEDS}
    )

    replay_evaluations: dict[int, dict[str, ArmEvaluation]] = {
        replay_seed: {} for replay_seed in REPLAY_SEEDS
    }
    completed_units = 0
    for arm_ordinal, arm in enumerate(ARM_NAMES):
        for replay_ordinal, replay_seed in enumerate(REPLAY_SEEDS):
            scores = np.full(validated.summary.row_count, np.nan, dtype=np.float64)
            for fold_ordinal in range(OUTER_FOLD_COUNT):
                partition = counterfactual_folds[fold_ordinal]
                if arm == "CF":
                    train_matrix = partition.train_matrix
                    heldout_matrix = partition.heldout_matrix
                    train_indices = partition.train_indices
                    heldout_indices = partition.heldout_indices
                else:
                    matrix = base_matrices.for_arm(arm)
                    heldout_indices = np.flatnonzero(validated.folds == fold_ordinal)
                    train_indices = np.flatnonzero(validated.folds != fold_ordinal)
                    train_matrix = matrix[train_indices]
                    heldout_matrix = matrix[heldout_indices]
                fold_scores = _positive_probability(
                    train_matrix,
                    validated.labels[train_indices],
                    heldout_matrix,
                    synthetic=synthetic,
                )
                if arm in NOVEL_FALLBACK_ARMS:
                    baseline = replay_evaluations[replay_seed].get("B0")
                    if baseline is None:
                        raise RuntimeError(f"{arm} ran before the same-replay B0 baseline")
                    fold_scores = finalize_novel_arm_evaluation(
                        arm,
                        b0_scores=baseline.scores[heldout_indices],
                        arm_scores=fold_scores,
                        novel_complete=validated.cache.novel_complete[heldout_indices],
                    ).scores
                scores[heldout_indices] = fold_scores
                ledger.fit_unit_completed(
                    arm_ordinal=arm_ordinal,
                    replay_ordinal=replay_ordinal,
                    fold_ordinal=fold_ordinal,
                )
                completed_units += 1
            if not np.isfinite(scores).all():
                raise RuntimeError("an arm did not produce exactly one finite score per cached row")
            replay_evaluations[replay_seed][arm] = _finalize_arm_evaluation(
                arm,
                scores,
                baseline=replay_evaluations[replay_seed].get("B0"),
                novel_complete=validated.cache.novel_complete,
            )

    if completed_units != EXPECTED_FIT_UNIT_COUNT:
        raise RuntimeError("fixed Phase-B fit grid did not contain exactly 75 units")
    ledger.fit_completed(unit_count=completed_units)
    if ledger.final_record_sha256 is None:
        raise RuntimeError("completed fit ledger has no final record hash")

    frozen_replays = MappingProxyType(
        {
            replay_seed: MappingProxyType(dict(replay_evaluations[replay_seed]))
            for replay_seed in REPLAY_SEEDS
        }
    )
    evaluation_replay_sha256 = _assert_identical_replays(
        frozen_replays,
        novel_complete=validated.cache.novel_complete,
    )
    if synthetic:
        primary_controls: Mapping[int, GlobalPrimaryControl] | None = None
    else:
        primary_controls = MappingProxyType(
            {
                replay_seed: select_global_primary_control(
                    frozen_replays[replay_seed]["B0"].hard_decisions != validated.labels,
                    frozen_replays[replay_seed]["B1"].hard_decisions != validated.labels,
                )
                for replay_seed in REPLAY_SEEDS
            }
        )
    return PhaseBFitResult(
        replay_evaluations=frozen_replays,
        matrix_replay_sha256=matrix_replay_sha256,
        evaluation_replay_sha256=evaluation_replay_sha256,
        fit_ledger_final_record_sha256=ledger.final_record_sha256,
        total_fit_units=completed_units,
        primary_controls=primary_controls,
    )


def run_phase_b_fit(
    cache: PhaseBFeatureCache,
    labels: np.ndarray,
    folds: np.ndarray,
    ledger: FitLedger,
    *,
    fit_protocol_commitment_sha256: str,
    feature_rows_commitment_sha256: str,
    raw_ledger_final_record_sha256: str,
) -> PhaseBFitResult:
    """Run the only production path: 20k rows, five arms, 75 fixed HGB fits."""

    return _run_phase_b_fit(
        cache,
        labels,
        folds,
        ledger,
        fit_protocol_commitment_sha256=fit_protocol_commitment_sha256,
        feature_rows_commitment_sha256=feature_rows_commitment_sha256,
        raw_ledger_final_record_sha256=raw_ledger_final_record_sha256,
        synthetic=False,
    )


def run_phase_b_fit_for_test(
    cache: PhaseBFeatureCache,
    labels: np.ndarray,
    folds: np.ndarray,
    ledger: FitLedger,
    *,
    fit_protocol_commitment_sha256: str,
    feature_rows_commitment_sha256: str,
    raw_ledger_final_record_sha256: str,
    synthetic: bool,
) -> PhaseBFitResult:
    """Exercise the full grid on a small synthetic matrix only when explicitly marked."""

    if synthetic is not True:
        raise ValueError("test fitting requires the explicit synthetic=True marker")
    return _run_phase_b_fit(
        cache,
        labels,
        folds,
        ledger,
        fit_protocol_commitment_sha256=fit_protocol_commitment_sha256,
        feature_rows_commitment_sha256=feature_rows_commitment_sha256,
        raw_ledger_final_record_sha256=raw_ledger_final_record_sha256,
        synthetic=True,
    )
