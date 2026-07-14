"""Aggregate-only preregistered evaluation for a completed Loop167 Phase-B fit."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .arm_contract import ARM_NAMES, FULL_TRAIN_ROWS, REPLAY_SEEDS, ArmEvaluation
from .fit_worker import PhaseBFitResult

BOOTSTRAP_REPLICATES = 200_000
BOOTSTRAP_CONFIDENCE = 0.95


@dataclass(frozen=True)
class BinaryMetrics:
    """Aggregate binary-classification metrics with a fixed denominator."""

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    errors: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "errors": self.errors,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def _labels(values: np.ndarray, *, rows: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.shape != (rows,) or (labels.dtype != np.dtype(bool) and not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError("labels must be a binary vector with the fixed row count")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must contain only binary values")
    return np.ascontiguousarray(labels, dtype=np.uint8)


def _decisions(values: np.ndarray, *, rows: int) -> np.ndarray:
    decisions = np.asarray(values)
    if decisions.shape != (rows,) or (decisions.dtype != np.dtype(bool) and not np.issubdtype(decisions.dtype, np.integer)):
        raise ValueError("decisions must be a binary vector with the fixed row count")
    if not np.isin(decisions, (0, 1)).all():
        raise ValueError("decisions must contain only binary values")
    return np.ascontiguousarray(decisions, dtype=np.uint8)


def _folds(values: np.ndarray, *, rows: int) -> np.ndarray:
    folds = np.asarray(values)
    if folds.shape != (rows,) or folds.dtype == np.dtype(bool) or not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("folds must contain one integer fold per row")
    if not np.isin(folds, (0, 1, 2, 3, 4)).all():
        raise ValueError("folds must use exactly the sealed 0..4 labels")
    normalized = np.ascontiguousarray(folds, dtype=np.int8)
    if not np.all(np.bincount(normalized, minlength=5) == 4_000):
        raise ValueError("folds must contain exactly 4000 rows per outer fold")
    return normalized


def _component_keys(values: np.ndarray, *, rows: int) -> np.ndarray:
    components = np.asarray(values)
    if components.shape != (rows,):
        raise ValueError("component_ids must contain one key per row")
    normalized = np.asarray([str(value) for value in components], dtype=object)
    if any(not value for value in normalized):
        raise ValueError("component_ids must be nonempty")
    return normalized


def binary_metrics(labels: np.ndarray, decisions: np.ndarray) -> BinaryMetrics:
    """Calculate metrics without accepting scores or a tunable threshold."""

    normalized_labels = _labels(labels, rows=np.asarray(labels).size)
    normalized_decisions = _decisions(decisions, rows=normalized_labels.size)
    true_positive = int(np.count_nonzero((normalized_labels == 1) & (normalized_decisions == 1)))
    true_negative = int(np.count_nonzero((normalized_labels == 0) & (normalized_decisions == 0)))
    false_positive = int(np.count_nonzero((normalized_labels == 0) & (normalized_decisions == 1)))
    false_negative = int(np.count_nonzero((normalized_labels == 1) & (normalized_decisions == 0)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(np.float64).eps)
    return BinaryMetrics(
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        errors=false_positive + false_negative,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def _bootstrap_seed(protocol_sha256: str, replay_seed: int) -> int:
    if not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64:
        raise ValueError("protocol_sha256 must be a SHA-256 digest")
    material = f"loop167-phase-b-bootstrap-v4\0{protocol_sha256}\0{replay_seed}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _component_reductions(
    control_errors: np.ndarray,
    candidate_errors: np.ndarray,
    component_ids: np.ndarray,
) -> np.ndarray:
    grouped: dict[str, int] = defaultdict(int)
    for component, control_error, candidate_error in zip(component_ids, control_errors, candidate_errors, strict=True):
        grouped[str(component)] += int(control_error) - int(candidate_error)
    return np.asarray([grouped[key] for key in sorted(grouped)], dtype=np.int64)


def component_bootstrap_lower_bound(
    control_errors: np.ndarray,
    candidate_errors: np.ndarray,
    component_ids: np.ndarray,
    *,
    protocol_sha256: str,
    replay_seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> float:
    """Return the one-sided 95% component-bootstrap lower bound on error reduction."""

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    rows = np.asarray(control_errors).size
    control = _decisions(control_errors, rows=rows)
    candidate = _decisions(candidate_errors, rows=rows)
    components = _component_keys(component_ids, rows=rows)
    reductions = _component_reductions(control, candidate, components)
    component_count = reductions.size
    if component_count == 0:
        raise ValueError("component_ids must not be empty")
    values, counts = np.unique(reductions, return_counts=True)
    probabilities = counts.astype(np.float64) / float(component_count)
    generator = np.random.Generator(np.random.PCG64(_bootstrap_seed(protocol_sha256, replay_seed)))
    estimates = np.empty(replicates, dtype=np.float64)
    batch_size = 1_024 if values.size <= 512 else 64
    for start in range(0, replicates, batch_size):
        end = min(start + batch_size, replicates)
        if values.size <= 512:
            weights = generator.multinomial(component_count, probabilities, size=end - start)
            estimates[start:end] = weights @ values
        else:
            draw_indices = generator.integers(0, component_count, size=(end - start, component_count))
            estimates[start:end] = reductions[draw_indices].sum(axis=1, dtype=np.int64)
    return float(np.quantile(estimates, 1.0 - BOOTSTRAP_CONFIDENCE, method="lower"))


def _relative_worsening(candidate: int, control: int) -> float:
    if control == 0:
        return 0.0 if candidate == 0 else math.inf
    return (candidate - control) / control


def _fold_improvements(control_errors: np.ndarray, candidate_errors: np.ndarray, folds: np.ndarray) -> int:
    return sum(
        int(np.count_nonzero(candidate_errors[folds == fold])) < int(np.count_nonzero(control_errors[folds == fold]))
        for fold in range(5)
    )


def _gate_summary(
    *,
    control_name: str,
    control_decisions: np.ndarray,
    candidate_decisions: np.ndarray,
    cf_decisions: np.ndarray,
    a_decisions: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    components: np.ndarray,
    protocol_sha256: str,
    replay_seed: int,
) -> dict[str, object]:
    control = binary_metrics(labels, control_decisions)
    candidate = binary_metrics(labels, candidate_decisions)
    counterfactual = binary_metrics(labels, cf_decisions)
    a_metrics = binary_metrics(labels, a_decisions)
    control_errors = control_decisions != labels
    candidate_errors = candidate_decisions != labels
    cf_errors = cf_decisions != labels
    a_errors = a_decisions != labels
    changed = candidate_decisions != control_decisions
    repairs = int(np.count_nonzero(control_errors & ~candidate_errors))
    overrides = int(np.count_nonzero(changed))
    override_precision = repairs / max(overrides, 1)
    required_reduction = max(30, math.ceil(0.10 * control.errors))
    reduction = control.errors - candidate.errors
    lcb = component_bootstrap_lower_bound(
        control_errors.astype(np.uint8),
        candidate_errors.astype(np.uint8),
        components,
        protocol_sha256=protocol_sha256,
        replay_seed=replay_seed,
    )
    a_overlap = int(np.count_nonzero(a_errors & control_errors)) / max(control.errors, 1)
    fold_improvements = _fold_improvements(control_errors, candidate_errors, folds)
    fp_worsening = _relative_worsening(candidate.false_positive, control.false_positive)
    fn_worsening = _relative_worsening(candidate.false_negative, control.false_negative)
    cf_reduction = counterfactual.errors - candidate.errors
    candidate_passed = all(
        (
            reduction >= required_reduction,
            repairs >= 50,
            override_precision >= 0.80,
            fold_improvements >= 4,
            lcb > 0.0,
            fp_worsening <= 0.05,
            fn_worsening <= 0.05,
            a_overlap <= 0.80,
            cf_reduction >= 30,
        )
    )
    cf_lcb = component_bootstrap_lower_bound(
        control_errors.astype(np.uint8),
        cf_errors.astype(np.uint8),
        components,
        protocol_sha256=protocol_sha256,
        replay_seed=replay_seed,
    )
    cf_passed = (control.errors - counterfactual.errors) >= required_reduction and cf_lcb > 0.0
    return {
        "primary_control": control_name,
        "control": control.as_dict(),
        "m": candidate.as_dict(),
        "a": a_metrics.as_dict(),
        "cf": counterfactual.as_dict(),
        "net_error_reduction": reduction,
        "required_net_error_reduction": required_reduction,
        "repairs": repairs,
        "override_count": overrides,
        "override_precision": override_precision,
        "fold_improvements": fold_improvements,
        "component_bootstrap_one_sided_95_lcb": lcb,
        "fp_relative_worsening": fp_worsening,
        "fn_relative_worsening": fn_worsening,
        "a_control_error_overlap": a_overlap,
        "m_over_cf_net_error_reduction": cf_reduction,
        "cf_component_bootstrap_one_sided_95_lcb": cf_lcb,
        "m_passed_all_gates": candidate_passed,
        "cf_passed_any_effect_gate": cf_passed,
    }


def evaluate_phase_b_fit(
    fit_result: PhaseBFitResult,
    labels: np.ndarray,
    folds: np.ndarray,
    component_ids: np.ndarray,
    *,
    protocol_sha256: str,
) -> Mapping[int, Mapping[str, object]]:
    """Evaluate the sealed 20k local diagnostic without writing prediction rows."""

    if not isinstance(fit_result, PhaseBFitResult):
        raise TypeError("fit_result must be a PhaseBFitResult")
    normalized_labels = _labels(labels, rows=FULL_TRAIN_ROWS)
    normalized_folds = _folds(folds, rows=FULL_TRAIN_ROWS)
    components = _component_keys(component_ids, rows=FULL_TRAIN_ROWS)
    if fit_result.primary_controls is None or set(fit_result.primary_controls) != set(REPLAY_SEEDS):
        raise ValueError("production Phase-B result must contain every global primary control")
    result: dict[int, Mapping[str, object]] = {}
    for replay_seed in REPLAY_SEEDS:
        evaluations = fit_result.replay_evaluations.get(replay_seed)
        control = fit_result.primary_controls[replay_seed]
        if evaluations is None or set(evaluations) != set(ARM_NAMES):
            raise ValueError("fit result replay arm coverage drifted")
        control_evaluation: ArmEvaluation = evaluations[control.arm]
        result[replay_seed] = MappingProxyType(
            _gate_summary(
                control_name=control.arm,
                control_decisions=control_evaluation.hard_decisions,
                candidate_decisions=evaluations["M"].hard_decisions,
                cf_decisions=evaluations["CF"].hard_decisions,
                a_decisions=evaluations["A"].hard_decisions,
                labels=normalized_labels,
                folds=normalized_folds,
                components=components,
                protocol_sha256=protocol_sha256,
                replay_seed=replay_seed,
            )
        )
    return MappingProxyType(result)
