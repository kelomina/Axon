"""Fail-closed aggregate evaluation for the Loop175 seed-41 OOF contract.

The evaluator accepts only labels, outer-fold assignments, aggregate component
keys, and one probability vector per preregistered arm.  Component keys are
used for clustered uncertainty only; they are never feature inputs.  It does
not read files, checkpoints, or any validation/held-out split.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

ARM_NAMES = ("A", "B", "C", "D", "E")
PRIMARY_ARM = "C"
BASELINE_ARM = "A"
COUNTERFACTUAL_ARM = "D"
DEFAULT_ROWS = 20_000
DEFAULT_FOLDS = 5
DEFAULT_ROWS_PER_FOLD = 4_000
DEFAULT_THRESHOLD = 0.5
DEFAULT_BOOTSTRAP_REPLICATES = 200_000
DEFAULT_BOOTSTRAP_CONFIDENCE = 0.95
MAX_GPU_ALLOCATED_BYTES = 6_979_321_856
MAX_RSS_BYTES = 11 * 1024**3
MAX_NEW_DISK_BYTES = 30 * 1024**3
MAX_FOLD_WALL_SECONDS = 6 * 60 * 60
MAX_SEED_WALL_SECONDS = 30 * 60 * 60
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PhaseBEvaluationError(ValueError):
    """Raised when an OOF evaluation contract is malformed or incomplete."""


def _as_binary_vector(values: Sequence[Any], *, name: str, rows: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise PhaseBEvaluationError(f"{name} must be a non-empty one-dimensional vector")
    if rows is not None and array.shape != (rows,):
        raise PhaseBEvaluationError(f"{name} must contain exactly {rows} rows")
    if array.dtype != np.dtype(bool) and not np.issubdtype(array.dtype, np.integer):
        raise PhaseBEvaluationError(f"{name} must contain binary integers")
    if not np.isin(array, (0, 1)).all():
        raise PhaseBEvaluationError(f"{name} must contain only 0 and 1")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _as_fold_vector(values: Sequence[Any], *, rows: int, expected_rows_per_fold: int | None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape != (rows,):
        raise PhaseBEvaluationError("folds must contain exactly one fold per row")
    if array.dtype == np.dtype(bool) or not np.issubdtype(array.dtype, np.integer):
        raise PhaseBEvaluationError("folds must contain integer labels")
    if not np.isin(array, np.arange(DEFAULT_FOLDS)).all():
        raise PhaseBEvaluationError("folds must use exactly the outer fold labels 0..4")
    normalized = np.ascontiguousarray(array, dtype=np.int8)
    counts = np.bincount(normalized, minlength=DEFAULT_FOLDS)
    if np.any(counts == 0):
        raise PhaseBEvaluationError("every outer fold must contain at least one holdout row")
    if expected_rows_per_fold is not None and not np.all(counts == expected_rows_per_fold):
        raise PhaseBEvaluationError(
            f"outer folds must contain exactly {expected_rows_per_fold} rows each"
        )
    return normalized


def _as_component_keys(values: Sequence[Any], *, rows: int) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise PhaseBEvaluationError("component_ids must be a sequence, not one string")
    normalized = np.asarray([str(value) for value in values], dtype=object)
    if normalized.shape != (rows,) or any(not value for value in normalized):
        raise PhaseBEvaluationError("component_ids must contain one non-empty key per row")
    return normalized


def _as_probability_vector(values: Sequence[Any], *, name: str, rows: int) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise PhaseBEvaluationError(f"{name} must be a probability vector") from error
    if array.ndim != 1 or array.shape != (rows,):
        raise PhaseBEvaluationError(f"{name} must contain exactly {rows} rows")
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise PhaseBEvaluationError(f"{name} must contain finite probabilities in [0, 1]")
    return np.ascontiguousarray(array)


def strict_decisions(scores: Sequence[Any], *, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Convert probabilities to hard decisions; equality at 0.5 is benign."""

    if threshold != DEFAULT_THRESHOLD:
        raise PhaseBEvaluationError("Loop175 uses the frozen threshold 0.5")
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise PhaseBEvaluationError("scores must be a non-empty one-dimensional vector")
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise PhaseBEvaluationError("scores must contain finite probabilities in [0, 1]")
    return np.ascontiguousarray((array > DEFAULT_THRESHOLD).astype(np.uint8))


def binary_metrics(labels: Sequence[Any], decisions: Sequence[Any]) -> dict[str, float | int]:
    """Return confusion-matrix metrics for a fixed binary decision vector."""

    normalized_labels = _as_binary_vector(labels, name="labels")
    normalized_decisions = _as_binary_vector(decisions, name="decisions", rows=normalized_labels.size)
    true_positive = int(np.count_nonzero((normalized_labels == 1) & (normalized_decisions == 1)))
    true_negative = int(np.count_nonzero((normalized_labels == 0) & (normalized_decisions == 0)))
    false_positive = int(np.count_nonzero((normalized_labels == 0) & (normalized_decisions == 1)))
    false_negative = int(np.count_nonzero((normalized_labels == 1) & (normalized_decisions == 0)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(np.float64).eps)
    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "errors": false_positive + false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _relative_worsening(candidate: int, control: int) -> float:
    if control == 0:
        return 0.0 if candidate == 0 else math.inf
    return float((candidate - control) / control)


def _validate_protocol_sha(protocol_sha256: str) -> str:
    if not isinstance(protocol_sha256, str) or not SHA256_PATTERN.fullmatch(protocol_sha256):
        raise PhaseBEvaluationError("protocol_sha256 must be a lowercase SHA-256 digest")
    return protocol_sha256


def bootstrap_seed(protocol_sha256: str, seed: int, control_arm: str, candidate_arm: str) -> int:
    """Derive a reproducible uncertainty seed from protocol, replay seed, and pair."""

    protocol = _validate_protocol_sha(protocol_sha256)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PhaseBEvaluationError("seed must be a non-negative integer")
    if control_arm not in ARM_NAMES or candidate_arm not in ARM_NAMES or control_arm == candidate_arm:
        raise PhaseBEvaluationError("bootstrap arms must be two distinct preregistered arms")
    material = (
        f"loop175-phase-b-component-bootstrap-v1\0{protocol}\0{seed}\0"
        f"{control_arm}>{candidate_arm}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def _component_reductions(
    control_errors: np.ndarray,
    candidate_errors: np.ndarray,
    component_ids: np.ndarray,
) -> np.ndarray:
    grouped: dict[str, int] = defaultdict(int)
    for component, control_error, candidate_error in zip(
        component_ids, control_errors, candidate_errors, strict=True
    ):
        grouped[str(component)] += int(control_error) - int(candidate_error)
    if not grouped:
        raise PhaseBEvaluationError("component_ids must not be empty")
    return np.asarray([grouped[key] for key in sorted(grouped)], dtype=np.int64)


def component_bootstrap_lower_bound(
    control_errors: Sequence[Any],
    candidate_errors: Sequence[Any],
    component_ids: Sequence[Any],
    *,
    protocol_sha256: str,
    seed: int,
    control_arm: str,
    candidate_arm: str,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, float | int | str]:
    """Return the deterministic one-sided 95% clustered error-reduction LCB."""

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise PhaseBEvaluationError("bootstrap replicates must be a positive integer")
    control = _as_binary_vector(control_errors, name="control_errors")
    candidate = _as_binary_vector(candidate_errors, name="candidate_errors", rows=control.size)
    components = _as_component_keys(component_ids, rows=control.size)
    reduction_values = _component_reductions(control, candidate, components)
    component_count = reduction_values.size
    derived_seed = bootstrap_seed(protocol_sha256, seed, control_arm, candidate_arm)
    generator = np.random.Generator(np.random.PCG64(derived_seed))
    unique_values, value_counts = np.unique(reduction_values, return_counts=True)
    probabilities = value_counts.astype(np.float64) / float(component_count)
    estimates = np.empty(replicates, dtype=np.float64)
    batch_size = 1_024 if unique_values.size <= 512 else 64
    for start in range(0, replicates, batch_size):
        end = min(start + batch_size, replicates)
        batch_count = end - start
        if unique_values.size <= 512:
            counts = generator.multinomial(component_count, probabilities, size=batch_count)
            estimates[start:end] = counts @ unique_values
        else:
            indices = generator.integers(0, component_count, size=(batch_count, component_count))
            estimates[start:end] = reduction_values[indices].sum(axis=1, dtype=np.int64)
    lower_bound = float(
        np.quantile(estimates, 1.0 - DEFAULT_BOOTSTRAP_CONFIDENCE, method="lower")
    )
    observed = int(reduction_values.sum(dtype=np.int64))
    return {
        "control_arm": control_arm,
        "candidate_arm": candidate_arm,
        "seed": int(derived_seed),
        "replicates": int(replicates),
        "confidence": DEFAULT_BOOTSTRAP_CONFIDENCE,
        "component_count": int(component_count),
        "observed_reduction": observed,
        "one_sided_95_lcb": lower_bound,
    }


def paired_component_bootstrap_lower_bound(*args: Any, **kwargs: Any) -> dict[str, float | int | str]:
    """Compatibility name emphasizing that each component delta is paired."""

    return component_bootstrap_lower_bound(*args, **kwargs)


def _array_commitment(name: str, values: np.ndarray) -> str:
    digest = hashlib.sha256(b"loop175-phase-b-input-v1\0")
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    if values.dtype == object:
        for value in values.tolist():
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        contiguous = np.ascontiguousarray(values)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _gate(
    name: str,
    observed: float | int | bool | None,
    required: str,
    passed: bool,
    *,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "required": required,
        "passed": bool(passed),
        "reason": reason,
    }


def _runtime_gates(runtime: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if runtime is None:
        return [
            _gate(name, None, requirement, False, reason="runtime_evidence_missing")
            for name, requirement in (
                ("coverage", ">=0.995"),
                ("class_coverage_gap", "<=0.02"),
                ("silent_drops", "==0"),
                ("oom", "false"),
                ("timeout", "false"),
                ("nonfinite", "false"),
                ("gpu_allocated_bytes", f"<={MAX_GPU_ALLOCATED_BYTES}"),
                ("rss_bytes", f"<={MAX_RSS_BYTES}"),
                ("new_disk_bytes", f"<={MAX_NEW_DISK_BYTES}"),
                ("maximum_fold_wall_seconds", f"<={MAX_FOLD_WALL_SECONDS}"),
                ("seed_wall_seconds", f"<={MAX_SEED_WALL_SECONDS}"),
            )
        ]
    required_fields = {
        "coverage",
        "class_coverage_gap",
        "silent_drops",
        "oom",
        "timeout",
        "nonfinite",
        "gpu_allocated_bytes",
        "rss_bytes",
        "new_disk_bytes",
        "maximum_fold_wall_seconds",
        "seed_wall_seconds",
    }
    missing = sorted(required_fields - set(runtime))
    if missing:
        return [
            _gate("runtime_contract", None, "all runtime fields present", False, reason=f"missing:{','.join(missing)}")
        ]
    return [
        _gate("coverage", runtime["coverage"], ">=0.995", float(runtime["coverage"]) >= 0.995),
        _gate(
            "class_coverage_gap",
            runtime["class_coverage_gap"],
            "<=0.02",
            float(runtime["class_coverage_gap"]) <= 0.02,
        ),
        _gate("silent_drops", runtime["silent_drops"], "==0", int(runtime["silent_drops"]) == 0),
        _gate("oom", runtime["oom"], "false", runtime["oom"] is False),
        _gate("timeout", runtime["timeout"], "false", runtime["timeout"] is False),
        _gate("nonfinite", runtime["nonfinite"], "false", runtime["nonfinite"] is False),
        _gate(
            "gpu_allocated_bytes",
            runtime["gpu_allocated_bytes"],
            f"<={MAX_GPU_ALLOCATED_BYTES}",
            int(runtime["gpu_allocated_bytes"]) <= MAX_GPU_ALLOCATED_BYTES,
        ),
        _gate(
            "rss_bytes",
            runtime["rss_bytes"],
            f"<={MAX_RSS_BYTES}",
            int(runtime["rss_bytes"]) <= MAX_RSS_BYTES,
        ),
        _gate(
            "new_disk_bytes",
            runtime["new_disk_bytes"],
            f"<={MAX_NEW_DISK_BYTES}",
            int(runtime["new_disk_bytes"]) <= MAX_NEW_DISK_BYTES,
        ),
        _gate(
            "maximum_fold_wall_seconds",
            runtime["maximum_fold_wall_seconds"],
            f"<={MAX_FOLD_WALL_SECONDS}",
            float(runtime["maximum_fold_wall_seconds"]) <= MAX_FOLD_WALL_SECONDS,
        ),
        _gate(
            "seed_wall_seconds",
            runtime["seed_wall_seconds"],
            f"<={MAX_SEED_WALL_SECONDS}",
            float(runtime["seed_wall_seconds"]) <= MAX_SEED_WALL_SECONDS,
        ),
    ]


def _arm_metrics_by_fold(
    labels: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    decisions = strict_decisions(scores)
    return {
        str(fold): binary_metrics(labels[folds == fold], decisions[folds == fold])
        for fold in range(DEFAULT_FOLDS)
    }


def evaluate_phase_b_oof(
    labels: Sequence[Any],
    folds: Sequence[Any],
    component_ids: Sequence[Any],
    arm_scores: Mapping[str, Sequence[Any]],
    *,
    protocol_sha256: str,
    seed: int = 41,
    expected_rows: int = DEFAULT_ROWS,
    expected_rows_per_fold: int | None = DEFAULT_ROWS_PER_FOLD,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and aggregate one complete seed-41 A-E outer OOF replay."""

    _validate_protocol_sha(protocol_sha256)
    if seed != 41:
        raise PhaseBEvaluationError("this Phase-B evaluator is frozen to seed 41")
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows <= 0:
        raise PhaseBEvaluationError("expected_rows must be a positive integer")
    if expected_rows_per_fold is not None and (
        isinstance(expected_rows_per_fold, bool)
        or not isinstance(expected_rows_per_fold, int)
        or expected_rows_per_fold <= 0
    ):
        raise PhaseBEvaluationError("expected_rows_per_fold must be a positive integer or None")
    normalized_labels = _as_binary_vector(labels, name="labels", rows=expected_rows)
    normalized_folds = _as_fold_vector(
        folds,
        rows=expected_rows,
        expected_rows_per_fold=expected_rows_per_fold,
    )
    normalized_components = _as_component_keys(component_ids, rows=expected_rows)
    component_folds: dict[str, set[int]] = defaultdict(set)
    for component, fold in zip(normalized_components, normalized_folds, strict=True):
        component_folds[str(component)].add(int(fold))
    split_components = [component for component, values in component_folds.items() if len(values) != 1]
    if split_components:
        raise PhaseBEvaluationError("a content component crosses outer folds")
    if set(arm_scores) != set(ARM_NAMES):
        missing = sorted(set(ARM_NAMES) - set(arm_scores))
        extra = sorted(set(arm_scores) - set(ARM_NAMES))
        raise PhaseBEvaluationError(f"A-E arm set mismatch; missing={missing}, extra={extra}")
    normalized_scores = {
        arm: _as_probability_vector(arm_scores[arm], name=f"{arm}_scores", rows=expected_rows)
        for arm in ARM_NAMES
    }
    decisions = {arm: strict_decisions(normalized_scores[arm]) for arm in ARM_NAMES}
    metrics = {arm: binary_metrics(normalized_labels, decisions[arm]) for arm in ARM_NAMES}
    fold_metrics = {
        arm: _arm_metrics_by_fold(normalized_labels, normalized_folds, normalized_scores[arm])
        for arm in ARM_NAMES
    }

    baseline_errors = decisions[BASELINE_ARM] != normalized_labels
    candidate_errors = decisions[PRIMARY_ARM] != normalized_labels
    counterfactual_errors = decisions[COUNTERFACTUAL_ARM] != normalized_labels
    repairs = int(np.count_nonzero(baseline_errors & ~candidate_errors))
    breaks = int(np.count_nonzero(~baseline_errors & candidate_errors))
    changed = int(np.count_nonzero(decisions[BASELINE_ARM] != decisions[PRIMARY_ARM]))
    override_precision = repairs / max(changed, 1)
    c_a_reduction = metrics[BASELINE_ARM]["errors"] - metrics[PRIMARY_ARM]["errors"]
    c_d_advantage = metrics[COUNTERFACTUAL_ARM]["errors"] - metrics[PRIMARY_ARM]["errors"]
    positive_folds = sum(
        fold_metrics[PRIMARY_ARM][str(fold)]["errors"]
        < fold_metrics[BASELINE_ARM][str(fold)]["errors"]
        for fold in range(DEFAULT_FOLDS)
    )
    c_a_bootstrap = component_bootstrap_lower_bound(
        baseline_errors.astype(np.uint8),
        candidate_errors.astype(np.uint8),
        normalized_components,
        protocol_sha256=protocol_sha256,
        seed=seed,
        control_arm=BASELINE_ARM,
        candidate_arm=PRIMARY_ARM,
        replicates=bootstrap_replicates,
    )
    d_c_bootstrap = component_bootstrap_lower_bound(
        counterfactual_errors.astype(np.uint8),
        candidate_errors.astype(np.uint8),
        normalized_components,
        protocol_sha256=protocol_sha256,
        seed=seed,
        control_arm=COUNTERFACTUAL_ARM,
        candidate_arm=PRIMARY_ARM,
        replicates=bootstrap_replicates,
    )
    c_a_fp_worsening = _relative_worsening(
        int(metrics[PRIMARY_ARM]["false_positive"]),
        int(metrics[BASELINE_ARM]["false_positive"]),
    )
    c_a_fn_worsening = _relative_worsening(
        int(metrics[PRIMARY_ARM]["false_negative"]),
        int(metrics[BASELINE_ARM]["false_negative"]),
    )
    primary_gates = [
        _gate("C_net_error_reduction_vs_A", c_a_reduction, ">=30", c_a_reduction >= 30),
        _gate("C_repairs_vs_A", repairs, ">=50", repairs >= 50),
        _gate("C_override_precision_vs_A", override_precision, ">=0.80", override_precision >= 0.80),
        _gate("C_net_positive_folds_vs_A", positive_folds, ">=4", positive_folds >= 4),
        _gate(
            "C_A_component_bootstrap_one_sided_95_lcb",
            c_a_bootstrap["one_sided_95_lcb"],
            ">0",
            float(c_a_bootstrap["one_sided_95_lcb"]) > 0.0,
        ),
        _gate("C_FP_relative_worsening_vs_A", c_a_fp_worsening, "<=0.05", c_a_fp_worsening <= 0.05),
        _gate("C_FN_relative_worsening_vs_A", c_a_fn_worsening, "<=0.05", c_a_fn_worsening <= 0.05),
        _gate("C_net_advantage_over_D", c_d_advantage, ">=30", c_d_advantage >= 30),
        _gate(
            "C_D_component_bootstrap_one_sided_95_lcb",
            d_c_bootstrap["one_sided_95_lcb"],
            ">0",
            float(d_c_bootstrap["one_sided_95_lcb"]) > 0.0,
        ),
    ]
    runtime_gates = _runtime_gates(runtime)
    all_gates = primary_gates + runtime_gates
    passed = all(bool(gate["passed"]) for gate in all_gates)
    e_vs_c_reduction = metrics[PRIMARY_ARM]["errors"] - metrics["E"]["errors"]
    return {
        "schema": "axon_loop175_seed41_oof_evaluation_v1",
        "loop_id": "Loop175",
        "claim_scope": "train_only_outer_oof_not_val_test10k_or_full_test",
        "seed": seed,
        "folds": DEFAULT_FOLDS,
        "threshold": DEFAULT_THRESHOLD,
        "row_accounting": {
            "expected_rows": expected_rows,
            "rows_scored_per_arm": {arm: expected_rows for arm in ARM_NAMES},
            "outer_holdout_once": True,
            "holdout_counts_by_fold": {
                str(fold): int(np.count_nonzero(normalized_folds == fold))
                for fold in range(DEFAULT_FOLDS)
            },
            "duplicate_rows": 0,
            "missing_rows": 0,
            "component_count": len(component_folds),
            "component_cross_fold_count": len(split_components),
        },
        "input_commitments": {
            "labels": _array_commitment("labels", normalized_labels),
            "folds": _array_commitment("folds", normalized_folds),
            "components": _array_commitment("component_ids", normalized_components),
            "arms": {arm: _array_commitment(f"{arm}_scores", normalized_scores[arm]) for arm in ARM_NAMES},
        },
        "arm_metrics": metrics,
        "fold_metrics": fold_metrics,
        "paired": {
            "A_to_C": {
                "repairs": repairs,
                "breaks": breaks,
                "changed_rows": changed,
                "override_precision": float(override_precision),
                "net_error_reduction": int(c_a_reduction),
                "fp_relative_worsening": c_a_fp_worsening,
                "fn_relative_worsening": c_a_fn_worsening,
            },
            "D_to_C": {
                "net_error_advantage": int(c_d_advantage),
            },
        },
        "component_bootstrap": {
            "replicates": bootstrap_replicates,
            "confidence": DEFAULT_BOOTSTRAP_CONFIDENCE,
            "A_to_C": c_a_bootstrap,
            "D_to_C": d_c_bootstrap,
        },
        "e_ablation": {
            "comparison": "E_vs_C",
            "net_error_reduction": int(e_vs_c_reduction),
            "enters_primary_C_gate": False,
            "decision": "record_only_no_C_gate_effect",
        },
        "gates": {
            "primary_C": primary_gates,
            "runtime": runtime_gates,
            "all_passed": passed,
        },
        "decision": "seed41_pass_allow_seed42_43" if passed else "closed_seed41_gate",
    }


__all__ = [
    "ARM_NAMES",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "MAX_NEW_DISK_BYTES",
    "PhaseBEvaluationError",
    "binary_metrics",
    "bootstrap_seed",
    "component_bootstrap_lower_bound",
    "evaluate_phase_b_oof",
    "paired_component_bootstrap_lower_bound",
    "strict_decisions",
]
