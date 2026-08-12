from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.loop175.phase_b_evaluation import (
    ARM_NAMES,
    DEFAULT_BOOTSTRAP_REPLICATES,
    PhaseBEvaluationError,
    binary_metrics,
    bootstrap_seed,
    component_bootstrap_lower_bound,
    evaluate_phase_b_oof,
    strict_decisions,
)

PROTOCOL_SHA256 = "a" * 64


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    rows = 200
    folds = np.repeat(np.arange(5, dtype=np.int8), 40)
    labels = np.tile(np.array([0, 1], dtype=np.uint8), rows // 2)
    components = np.asarray(
        [f"component-{fold}" for fold in folds],
        dtype=object,
    )
    wrong = np.where(labels == 0, 1.0, 0.0)
    right = np.where(labels == 0, 0.0, 1.0)
    scores = {
        "A": wrong.copy(),
        "B": np.zeros(rows, dtype=np.float64),
        "C": right.copy(),
        "D": wrong.copy(),
        "E": wrong.copy(),
    }
    runtime = {
        "coverage": 1.0,
        "class_coverage_gap": 0.0,
        "silent_drops": 0,
        "oom": False,
        "timeout": False,
        "nonfinite": False,
        "gpu_allocated_bytes": 0,
        "rss_bytes": 0,
        "new_disk_bytes": 0,
        "maximum_fold_wall_seconds": 1.0,
        "seed_wall_seconds": 5.0,
    }
    return labels, folds, components, scores, runtime


_DEFAULT_RUNTIME = object()


def _evaluate(*, scores: dict[str, np.ndarray] | None = None, runtime: object = _DEFAULT_RUNTIME):
    labels, folds, components, default_scores, default_runtime = _inputs()
    return evaluate_phase_b_oof(
        labels,
        folds,
        components,
        default_scores if scores is None else scores,
        protocol_sha256=PROTOCOL_SHA256,
        expected_rows=200,
        expected_rows_per_fold=40,
        bootstrap_replicates=128,
        runtime=default_runtime if runtime is _DEFAULT_RUNTIME else runtime,
    )


def test_strict_threshold_and_binary_metrics() -> None:
    assert strict_decisions([0.0, 0.5, 0.5000001, 1.0]).tolist() == [0, 0, 1, 1]
    assert binary_metrics([0, 0, 1, 1], [0, 1, 0, 1]) == {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "errors": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_complete_arms_fold_pairing_and_structured_gates() -> None:
    result = _evaluate()
    assert result["decision"] == "seed41_pass_allow_seed42_43"
    assert result["row_accounting"]["outer_holdout_once"] is True
    assert result["row_accounting"]["rows_scored_per_arm"] == {arm: 200 for arm in ARM_NAMES}
    assert result["row_accounting"]["holdout_counts_by_fold"] == {str(fold): 40 for fold in range(5)}
    assert result["arm_metrics"]["A"]["errors"] == 200
    assert result["arm_metrics"]["C"]["errors"] == 0
    assert result["paired"]["A_to_C"] == {
        "repairs": 200,
        "breaks": 0,
        "changed_rows": 200,
        "override_precision": 1.0,
        "net_error_reduction": 200,
        "fp_relative_worsening": -1.0,
        "fn_relative_worsening": -1.0,
    }
    assert result["paired"]["D_to_C"]["net_error_advantage"] == 200
    assert result["component_bootstrap"]["A_to_C"]["one_sided_95_lcb"] > 0
    assert result["component_bootstrap"]["D_to_C"]["one_sided_95_lcb"] > 0
    assert all(gate["passed"] for gate in result["gates"]["primary_C"])
    gate_names = {gate["name"] for gate in result["gates"]["primary_C"]}
    assert "C_net_error_reduction_vs_A" in gate_names
    assert "C_net_advantage_over_D" in gate_names


def test_e_is_reported_but_c_gate_is_unchanged() -> None:
    first = _evaluate()
    labels, _folds, _components, scores, _runtime = _inputs()
    scores["E"] = np.where(labels == 0, 0.0, 1.0)
    second = _evaluate(scores=scores)
    assert first["gates"]["primary_C"] == second["gates"]["primary_C"]
    assert first["e_ablation"]["enters_primary_C_gate"] is False
    assert second["e_ablation"]["enters_primary_C_gate"] is False
    assert first["arm_metrics"]["E"]["errors"] != second["arm_metrics"]["E"]["errors"]


def test_missing_runtime_evidence_closes_promotion() -> None:
    result = _evaluate(runtime=None)
    assert result["decision"] == "closed_seed41_gate"
    assert result["gates"]["all_passed"] is False
    assert all(gate["reason"] == "runtime_evidence_missing" for gate in result["gates"]["runtime"])


def test_arm_completeness_and_outer_component_integrity_fail_closed() -> None:
    labels, folds, components, scores, runtime = _inputs()
    with pytest.raises(PhaseBEvaluationError, match="arm set mismatch"):
        evaluate_phase_b_oof(
            labels,
            folds,
            components,
            {key: value for key, value in scores.items() if key != "E"},
            protocol_sha256=PROTOCOL_SHA256,
            expected_rows=200,
            expected_rows_per_fold=40,
            bootstrap_replicates=8,
            runtime=runtime,
        )

    bad_components = components.copy()
    bad_components[0] = "component-crossing"
    bad_components[40] = "component-crossing"
    with pytest.raises(PhaseBEvaluationError, match="crosses outer folds"):
        evaluate_phase_b_oof(
            labels,
            folds,
            bad_components,
            scores,
            protocol_sha256=PROTOCOL_SHA256,
            expected_rows=200,
            expected_rows_per_fold=40,
            bootstrap_replicates=8,
            runtime=runtime,
        )

    bad_scores = dict(scores)
    bad_scores["C"] = bad_scores["C"].copy()
    bad_scores["C"][0] = np.nan
    with pytest.raises(PhaseBEvaluationError, match="C_scores"):
        evaluate_phase_b_oof(
            labels,
            folds,
            components,
            bad_scores,
            protocol_sha256=PROTOCOL_SHA256,
            expected_rows=200,
            expected_rows_per_fold=40,
            bootstrap_replicates=8,
            runtime=runtime,
        )


def test_bootstrap_is_deterministic_and_pair_specific() -> None:
    errors = np.ones(20, dtype=np.uint8)
    correct = np.zeros(20, dtype=np.uint8)
    components = np.asarray([f"component-{index // 4}" for index in range(20)], dtype=object)
    first = component_bootstrap_lower_bound(
        errors,
        correct,
        components,
        protocol_sha256=PROTOCOL_SHA256,
        seed=41,
        control_arm="A",
        candidate_arm="C",
        replicates=128,
    )
    repeated = component_bootstrap_lower_bound(
        errors,
        correct,
        components,
        protocol_sha256=PROTOCOL_SHA256,
        seed=41,
        control_arm="A",
        candidate_arm="C",
        replicates=128,
    )
    alternate_pair = component_bootstrap_lower_bound(
        errors,
        correct,
        components,
        protocol_sha256=PROTOCOL_SHA256,
        seed=41,
        control_arm="D",
        candidate_arm="C",
        replicates=128,
    )
    assert first == repeated
    assert first["one_sided_95_lcb"] == 20.0
    assert first["seed"] != alternate_pair["seed"]
    assert DEFAULT_BOOTSTRAP_REPLICATES == 200_000
    assert bootstrap_seed(PROTOCOL_SHA256, 41, "A", "C") == first["seed"]


def test_gate_regression_and_runtime_limits_are_structured() -> None:
    labels, folds, components, scores, runtime = _inputs()
    runtime["coverage"] = 0.994
    runtime["gpu_allocated_bytes"] = 6_979_321_857
    runtime["new_disk_bytes"] = 30 * 1024**3 + 1
    result = _evaluate(runtime=runtime)
    runtime_by_name = {gate["name"]: gate for gate in result["gates"]["runtime"]}
    assert runtime_by_name["coverage"]["passed"] is False
    assert runtime_by_name["gpu_allocated_bytes"]["passed"] is False
    assert runtime_by_name["new_disk_bytes"]["passed"] is False
    assert result["decision"] == "closed_seed41_gate"


def test_evaluator_has_no_identity_feature_parameter_surface() -> None:
    parameter_names = set(inspect.signature(evaluate_phase_b_oof).parameters)
    forbidden = {"path", "filename", "extension", "directory", "hash", "row_order", "identity"}
    assert parameter_names.isdisjoint(forbidden)
