#!/usr/bin/env python3
"""Build the aggregate-only static preflight for the Loop164 F1 mainline."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_DIR = Path("manifests/roadmap_9997/loop164_whole_file_residual_expert")
DEFAULT_PROPOSAL = LOOP_DIR / "proposal.json"
DEFAULT_AUTHORIZATION = LOOP_DIR / "authorization.json"
DEFAULT_SOURCES = LOOP_DIR / "frontier_sources.json"
ISOLATION_CONTRACT_VALIDATOR = Path("scripts/validate_loop164_isolation_contract.py")
NESTED_OOF_RECEIPT_VALIDATOR = Path("scripts/validate_loop164_nested_oof_execution_receipt.py")
FOLD_SCOPE_PLAN_VALIDATOR = Path("scripts/validate_loop164_fold_scope_plan.py")
TRAINING_AUTHORITY_VALIDATOR = Path("scripts/validate_loop164_training_authority.py")
CERTIFICATION_EVIDENCE_VALIDATOR = Path("scripts/validate_loop164_certification_evidence.py")
A2_REQUEST_VALIDATOR = Path("scripts/validate_loop164_a2_request.py")
A2_METADATA_REQUEST_TEMPLATE = Path(
    "manifests/roadmap_9997/loop164_whole_file_residual_expert/"
    "templates/a2_metadata_request.template.json"
)
A2_TRAINING_REQUEST_TEMPLATE = Path(
    "manifests/roadmap_9997/loop164_whole_file_residual_expert/"
    "templates/a2_training_request.template.json"
)
IMPLEMENTATION_MANIFEST_VALIDATOR = Path("scripts/validate_loop164_whole_file_implementation.py")
WHOLE_FILE_EXACTNESS_ORACLE = Path("tests/test_loop164_whole_file_gcg.py")
IMPLEMENTATION_CONTRACT_AUTHORIZATION = Path(
    "manifests/roadmap_9997/loop164_whole_file_residual_expert/"
    "implementation_contract_authorization.json"
)
ISOLATION_CONTRACT_SCHEMA = "axon_loop164_full_pool_isolation_contract_v2"
ISOLATION_RECEIPT_SCHEMA = "axon_loop164_full_pool_isolation_validation_v4"
NESTED_OOF_RECEIPT_SCHEMA = "axon_loop164_train_oof_execution_receipt_v1"
NESTED_OOF_VALIDATION_SCHEMA = "axon_loop164_train_oof_execution_validation_v1"
FOLD_SCOPE_PLAN_SCHEMA = "axon_loop164_fold_scope_plan_v1"
FOLD_SCOPE_PLAN_VALIDATION_SCHEMA = "axon_loop164_fold_scope_plan_validation_v1"
TRAINING_AUTHORIZATION_SCHEMA = "axon_loop164_training_authorization_v2"
TRAINING_LEASE_SCHEMA = "axon_loop164_training_lease_consumption_v2"
TRAINING_INPUT_BUNDLE_SCHEMA = "axon_loop164_train_oof_input_bundle_v1"
IMPLEMENTATION_MANIFEST_SCHEMA = "axon_loop164_whole_file_implementation_manifest_v2"
STATIC_PREFLIGHT_AUTHORIZATION_SCHEMA = "axon_loop164_static_preflight_authorization_v1"
STATIC_IMPLEMENTATION_AUTHORIZATION_SCHEMA = (
    "axon_loop164_static_implementation_contract_authorization_v1"
)
ZERO_EXECUTION_BUDGET_FIELDS = {
    "raw_files",
    "checkpoint_loads",
    "prediction_rows",
    "cache_rows",
    "training_runs",
    "model_evaluations",
    "gpu_runs",
    "dependency_installs",
}
EXECUTION_SEQUENCE_FIELDS = {
    "a1_static_prerequisites",
    "a2_metadata_request_prerequisites",
    "a2_metadata_outputs",
    "a2_training_request_prerequisites",
    "a2_training_outputs",
}
METADATA_A2_AUTHORIZATION_PREREQUISITE = (
    "externally_pinned_A2_metadata_v3_authorization_with_exact_argv_source_closure_"
    "fresh_resource_guard_custodian_root_and_stable_lease_before_inventory_open"
)
METADATA_A2_RECEIPT_OUTPUT = "aggregate_validation_receipt_with_partition_and_feature_semantics_only"
TRAINING_A2_AUTHORIZATION_PREREQUISITE = "externally_pinned_A2_training_authorization_v2"
TRAINING_A2_RECEIPT_OUTPUT = "complete_nested_oof_execution_receipt_before_train_or_fusion_claim"
CERTIFICATION_PROTOCOL_SCHEMA = "axon_loop164_certification_protocol_v1"
CERTIFICATION_FUTURE_ARTIFACTS = {
    "certification_power_analysis": Path(
        "reports/roadmap_9997/loop164/certification_power_analysis.json"
    ),
    "sealed_window_w1_manifest": Path(
        "reports/roadmap_9997/loop164/sealed_window_w1_manifest.json"
    ),
    "sealed_window_w1_receipt": Path(
        "reports/roadmap_9997/loop164/sealed_window_w1_evaluation_receipt.json"
    ),
    "sealed_window_w2_manifest": Path(
        "reports/roadmap_9997/loop164/sealed_window_w2_manifest.json"
    ),
    "sealed_window_w2_receipt": Path(
        "reports/roadmap_9997/loop164/sealed_window_w2_evaluation_receipt.json"
    ),
    "certification_replication_receipt": Path(
        "reports/roadmap_9997/loop164/certification_replication_receipt.json"
    ),
}

AGGREGATE_INPUTS = {
    "loop151_val": Path("reports/phase3_loop151/loop151_trusted_signer_guard_val_eval.json"),
    "loop151_test10k": Path(
        "reports/phase3_loop151/loop151_trusted_signer_guard_test10k_eval.json"
    ),
    "loop151_full": Path("reports/phase3_loop151/loop151_trusted_signer_guard_full_eval.json"),
    "loop161_guard": Path("reports/phase3_loop161/loop161_test10k_promotion_margin_guard.json"),
    "loop163_support": Path("reports/phase3_loop163/loop163_r11_rescue_support_audit.json"),
    "loop158_annotations": Path(
        "reports/phase3_loop158/loop158_loop157_external_annotation_import_summary.json"
    ),
    "group_distribution": Path("reports/raw_group_diagnostics/group_distribution_summary.json"),
    "content_pe_v2_cache": Path(
        "reports/random_20w_split/stage2_loop32_content_pe_v2_cache_train_val/content_pe_v2_cache_report.json"
    ),
    "content_string_cache": Path(
        "reports/random_20w_split/stage2_loop30_content_string_cache_train_val/content_string_cache_report.json"
    ),
    "content_cert_cache": Path(
        "reports/random_20w_split/stage2_loop31_content_cert_cache_train_val/content_cert_cache_report.json"
    ),
}

FUTURE_ARTIFACTS = {
    "a2_isolation_validation_authorization": Path(
        "manifests/roadmap_9997/loop164_whole_file_residual_expert/"
        "a2_isolation_validation_authorization.json"
    ),
    "a2_training_authorization": Path(
        "manifests/roadmap_9997/loop164_whole_file_residual_expert/a2_training_authorization.json"
    ),
    "whole_file_implementation_manifest": Path(
        "reports/roadmap_9997/loop164/whole_file_expert_implementation_manifest.json"
    ),
    "loop151_train_oof_manifest": Path(
        "reports/roadmap_9997/loop164/loop151_train_oof_manifest.json"
    ),
    "loop164_train_oof_execution_receipt": Path(
        "reports/roadmap_9997/loop164/loop164_train_oof_execution_receipt.json"
    ),
    "fold_scope_plan": Path("reports/roadmap_9997/loop164/fold_scope_plan.json"),
    "fold_scope_plan_validation": Path(
        "reports/roadmap_9997/loop164/fold_scope_plan_validation.json"
    ),
    "train_oof_input_bundle": Path(
        "reports/roadmap_9997/loop164/train_oof_input_bundle_manifest.json"
    ),
    "training_final_lease": Path(
        "reports/roadmap_9997/loop164/training_lease_consumption.final.json"
    ),
    "full_pool_group_manifest": Path("reports/roadmap_9997/loop164/full_pool_group_manifest.json"),
    "full_pool_isolation_validation": Path(
        "reports/roadmap_9997/loop164/full_pool_isolation_validation.json"
    ),
    "resource_guard": Path("reports/roadmap_9997/loop164/resource_guard.json"),
    "val_a_manifest": Path("manifests/roadmap_9997/p1_evaluation_reset/val_a_manifest.json"),
    "val_b_manifest": Path("manifests/roadmap_9997/p1_evaluation_reset/val_b_manifest.json"),
    "champion_registry": Path("reports/model_review/final_model_selection/champion_registry.json"),
}
RUNTIME_VERIFICATION_ARTIFACTS = {
    "a2_isolation_validation_authorization",
    "a2_training_authorization",
    "whole_file_implementation_manifest",
    "loop151_train_oof_manifest",
    "loop164_train_oof_execution_receipt",
    "fold_scope_plan",
    "fold_scope_plan_validation",
    "train_oof_input_bundle",
    "training_final_lease",
    "full_pool_group_manifest",
    "full_pool_isolation_validation",
    "resource_guard",
    "val_a_manifest",
    "val_b_manifest",
}


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _validate_static_scope(payload: object, label: str) -> dict[str, bool]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label}.authority_scope must be an object")
    expected = {
        "tier": "A1",
        "protected_input_access": False,
        "model_or_data_execution": False,
        "creates_runtime_artifact": False,
        "heldout_evaluation": False,
    }
    require_equal(payload, expected, f"{label}.authority_scope")
    return {
        "protected_input_access": False,
        "model_or_data_execution": False,
        "creates_runtime_artifact": False,
        "heldout_evaluation": False,
    }


def _validate_zero_execution_budget(payload: object, label: str) -> None:
    if not isinstance(payload, dict) or set(payload) != ZERO_EXECUTION_BUDGET_FIELDS:
        raise ValueError(f"{label}.execution_budget shape is invalid")
    if any(value != 0 for value in payload.values()):
        raise ValueError(f"{label}.execution_budget must be zero")


def validate_static_preflight_authorization(authorization: dict[str, Any]) -> dict[str, bool]:
    require_equal(
        authorization.get("schema"),
        STATIC_PREFLIGHT_AUTHORIZATION_SCHEMA,
        "authorization.schema",
    )
    require_equal(
        authorization.get("authorization_level"),
        "A1_scoped_design_and_static_preflight_only",
        "authorization.authorization_level",
    )
    require_equal(
        authorization.get("decision"),
        "allow_static_preflight_only_block_all_model_and_data_execution",
        "authorization.decision",
    )
    scope = _validate_static_scope(authorization.get("authority_scope"), "authorization")
    _validate_zero_execution_budget(authorization.get("execution_budget"), "authorization")
    return scope


def validate_static_implementation_authorization(
    authorization: dict[str, Any],
    *,
    root: Path,
    static_authorization_path: Path,
) -> dict[str, bool]:
    require_equal(
        authorization.get("schema"),
        STATIC_IMPLEMENTATION_AUTHORIZATION_SCHEMA,
        "implementation_authorization.schema",
    )
    require_equal(
        authorization.get("loop_id"),
        "loop164_whole_file_residual_expert",
        "implementation_authorization.loop_id",
    )
    require_equal(
        authorization.get("authorization_level"),
        "A1_scoped_static_contract_only",
        "implementation_authorization.authorization_level",
    )
    require_equal(
        authorization.get("decision"),
        "allow_synthetic_static_contract_validation_only",
        "implementation_authorization.decision",
    )
    basis = authorization.get("basis_authorization")
    if not isinstance(basis, dict):
        raise ValueError("implementation_authorization.basis_authorization must be an object")
    require_equal(
        basis.get("path"),
        static_authorization_path.relative_to(root).as_posix(),
        "implementation_authorization.basis_authorization.path",
    )
    require_equal(
        basis.get("sha256"),
        sha256_file(static_authorization_path),
        "implementation_authorization.basis_authorization.sha256",
    )
    scope = _validate_static_scope(
        authorization.get("authority_scope"), "implementation_authorization"
    )
    _validate_zero_execution_budget(
        authorization.get("execution_budget"), "implementation_authorization"
    )
    return scope


def validate_execution_sequence(proposal: dict[str, Any]) -> dict[str, list[str]]:
    compute_budget = proposal.get("compute_budget")
    if not isinstance(compute_budget, dict):
        raise ValueError("proposal.compute_budget is missing")
    require_equal(
        set(compute_budget) - {"state"},
        EXECUTION_SEQUENCE_FIELDS,
        "proposal.compute_budget execution sequence fields",
    )
    require_equal(
        compute_budget.get("state"),
        "not_authorized_and_not_yet_sized",
        "proposal.compute_budget.state",
    )
    sequence: dict[str, list[str]] = {}
    for field_name in sorted(EXECUTION_SEQUENCE_FIELDS):
        values = compute_budget.get(field_name)
        if not isinstance(values, list) or not values or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(f"proposal.compute_budget.{field_name} is invalid")
        sequence[field_name] = list(values)
    if METADATA_A2_AUTHORIZATION_PREREQUISITE not in sequence[
        "a2_metadata_request_prerequisites"
    ]:
        raise ValueError("proposal.compute_budget metadata A2 authorization is missing")
    if METADATA_A2_RECEIPT_OUTPUT not in sequence["a2_metadata_outputs"]:
        raise ValueError("proposal.compute_budget metadata A2 receipt output is missing")
    if TRAINING_A2_AUTHORIZATION_PREREQUISITE not in sequence[
        "a2_training_request_prerequisites"
    ]:
        raise ValueError("proposal.compute_budget training A2 authorization is missing")
    if TRAINING_A2_RECEIPT_OUTPUT not in sequence["a2_training_outputs"]:
        raise ValueError("proposal.compute_budget training A2 receipt output is missing")
    if TRAINING_A2_AUTHORIZATION_PREREQUISITE in sequence["a2_metadata_request_prerequisites"]:
        raise ValueError("proposal.compute_budget training A2 cannot precede metadata outputs")
    if METADATA_A2_RECEIPT_OUTPUT in sequence["a2_metadata_request_prerequisites"]:
        raise ValueError("proposal.compute_budget metadata receipt cannot precede metadata A2")
    return sequence


def validate_certification_protocol(proposal: dict[str, Any]) -> dict[str, Any]:
    protocol = proposal.get("certification_protocol")
    expected_fields = {
        "schema",
        "state",
        "amendment_policy",
        "claim_scope",
        "sealed_windows",
        "candidate_freeze",
        "isolation",
        "scoring",
        "inference",
        "acceptance",
        "power_analysis",
        "operational_gates",
        "future_artifacts",
    }
    if not isinstance(protocol, dict) or set(protocol) != expected_fields:
        raise ValueError("proposal.certification_protocol shape is invalid")
    require_equal(protocol.get("schema"), CERTIFICATION_PROTOCOL_SCHEMA, "certification.schema")
    require_equal(
        protocol.get("state"),
        "static_preregistration_only_no_a3_authorization",
        "certification.state",
    )
    require_equal(
        protocol.get("amendment_policy"),
        "forbidden_after_first_sealed_window_lock",
        "certification.amendment_policy",
    )
    target_fraction = proposal["target"]["f1_fraction"]
    require_equal(
        protocol.get("claim_scope"),
        {
            "legacy_point_is_non_certification": True,
            "target_f1_fraction": target_fraction,
            "two_window_replication_required": True,
        },
        "certification.claim_scope",
    )
    require_equal(
        protocol.get("sealed_windows"),
        {
            "window_ids": ["W1_certification", "W2_later_replication"],
            "time_order": "W2_starts_after_W1_ends",
            "evaluation_generation": 1,
            "independent_a3_authorization_and_lease_required": True,
            "window_pooling_or_replacement_forbidden": True,
        },
        "certification.sealed_windows",
    )
    required_hashes = [
        "bundle_manifest_sha256",
        "checkpoint_sha256",
        "model_source_closure_sha256",
        "config_sha256",
        "calibration_sha256",
        "threshold_policy_sha256",
        "abstain_policy_sha256",
        "runtime_environment_sha256",
        "sbom_sha256",
        "statistics_runner_sha256",
    ]
    require_equal(
        protocol.get("candidate_freeze"),
        {
            "frozen_before_W1_release": True,
            "required_hash_fields": required_hashes,
            "post_W1_selection_or_recalibration_forbidden": True,
        },
        "certification.candidate_freeze",
    )
    require_equal(
        protocol.get("isolation"),
        {
            "component_relation_fields": [
                "exact_cluster_id",
                "near_duplicate_cluster_id",
                "family_id",
                "campaign_id",
                "source_group_id",
            ],
            "component_time": "max_first_seen_time_utc",
            "temporal_stratification": "preregistered_calendar_blocks",
            "zero_cross_window_component_overlap": True,
            "identity_or_grouping_model_inputs_forbidden": True,
        },
        "certification.isolation",
    )
    require_equal(
        protocol.get("scoring"),
        {
            "metric": "f1_from_full_system_confusion_matrix",
            "report_unrounded_TP_TN_FP_FN": True,
            "denominator_outcomes": [
                "abstain",
                "timeout",
                "missing_feature",
                "unsupported",
                "parser_failure",
                "runtime_failure",
            ],
            "fixed_outcome_mapping_required_before_A3": True,
            "silent_drop_forbidden": True,
        },
        "certification.scoring",
    )
    require_equal(
        protocol.get("inference"),
        {
            "familywise_alpha": 0.05,
            "per_window_one_sided_alpha": 0.025,
            "per_window_lower_confidence_level": 0.975,
            "method": "stratified_relationship_component_bootstrap_with_conservative_guard",
            "resampling_unit": "unioned_relationship_component_with_temporal_block",
            "bootstrap_replicates": 200000,
            "seed_derivation": "sha256(protocol_sha256|window_manifest_sha256|bundle_sha256|statistics_runner_sha256)",
            "statistical_failure_decision": "insufficient_evidence",
        },
        "certification.inference",
    )
    require_equal(
        protocol.get("acceptance"),
        {
            "per_window_point_f1_fraction": target_fraction,
            "per_window_lower_bound_fraction": target_fraction,
            "both_window_pass_required": True,
            "planned_point_floor": "blinded_power_analysis_must_set_strictly_above_target_before_A3",
            "failed_window_burns_lineage_without_replacement": True,
        },
        "certification.acceptance",
    )
    require_equal(
        protocol.get("power_analysis"),
        {
            "required_before_A3": True,
            "method": "aggregate_only_grouped_bootstrap_simulation",
            "minimum_simulations": 50000,
            "minimum_joint_power": 0.9,
            "output_sha256_required": True,
            "insufficient_support_blocks_A3": True,
        },
        "certification.power_analysis",
    )
    require_equal(
        protocol.get("operational_gates"),
        {
            "state": "product_owner_thresholds_required_before_A3",
            "required_metrics": [
                "FPR",
                "FNR",
                "FP_per_million_benign",
                "FN_per_1000_malicious",
                "coverage",
                "P95_latency",
                "cost",
                "critical_slice_error_share",
            ],
        },
        "certification.operational_gates",
    )
    require_equal(
        protocol.get("future_artifacts"),
        [path.as_posix() for path in CERTIFICATION_FUTURE_ARTIFACTS.values()],
        "certification.future_artifacts",
    )
    return protocol


def verify_binding(
    root: Path, binding: dict[str, Any], expected_path: Path, label: str
) -> dict[str, str]:
    bound_path = Path(str(binding.get("path") or ""))
    require_equal(bound_path.as_posix(), expected_path.as_posix(), f"{label}.path")
    resolved = resolve_path(root, expected_path)
    if not resolved.is_file():
        raise ValueError(f"Missing {label}: {expected_path}")
    actual_sha256 = sha256_file(resolved)
    require_equal(str(binding.get("sha256") or ""), actual_sha256, f"{label}.sha256")
    return {"path": expected_path.as_posix(), "sha256": actual_sha256}


def bind_static_source(root: Path, relative_path: Path, label: str) -> dict[str, str]:
    resolved = resolve_path(root, relative_path)
    if not resolved.is_file():
        raise ValueError(f"Missing required static source {label}: {relative_path}")
    return {"path": relative_path.as_posix(), "sha256": sha256_file(resolved)}


def _nonnegative_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_integer(value: object, label: str) -> int:
    integer = _nonnegative_integer(value, label)
    if integer == 0:
        raise ValueError(f"{label} must be positive")
    return integer


def _exact_fraction(value: object, label: str) -> Fraction:
    try:
        fraction = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{label} must be a finite rational value") from error
    if fraction <= 0:
        raise ValueError(f"{label} must be positive")
    return fraction


def metric_summary(payload: dict[str, Any]) -> dict[str, object]:
    candidate = payload.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("Loop151 aggregate is missing candidate metrics")
    required = (
        "f1",
        "errors",
        "false_positive",
        "false_negative",
        "true_positive",
        "true_negative",
    )
    missing = [key for key in required if key not in candidate]
    if missing:
        raise ValueError(f"Loop151 candidate metrics missing keys: {missing}")
    true_positive = _nonnegative_integer(candidate["true_positive"], "candidate.true_positive")
    true_negative = _nonnegative_integer(candidate["true_negative"], "candidate.true_negative")
    false_positive = _nonnegative_integer(candidate["false_positive"], "candidate.false_positive")
    false_negative = _nonnegative_integer(candidate["false_negative"], "candidate.false_negative")
    errors = _nonnegative_integer(candidate["errors"], "candidate.errors")
    if errors != false_positive + false_negative:
        raise ValueError("Loop151 candidate errors do not match false-positive plus false-negative")
    positive_rows = true_positive + false_negative
    negative_rows = true_negative + false_positive
    rows = positive_rows + negative_rows
    if positive_rows == 0 or negative_rows == 0:
        raise ValueError("Loop151 candidate must retain both class denominators")
    exact_f1 = Fraction(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    reported_f1 = _exact_fraction(candidate["f1"], "candidate.f1")
    if abs(reported_f1 - exact_f1) > Fraction(1, 10**12):
        raise ValueError("Loop151 candidate F1 conflicts with its confusion matrix")
    return {
        "f1": float(candidate["f1"]),
        "errors": errors,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
        "tn": true_negative,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "rows": rows,
        "f1_exact": {"numerator": exact_f1.numerator, "denominator": exact_f1.denominator},
    }


def build_legacy_point_geometry(
    proposal: dict[str, Any], legacy_full_test: dict[str, object]
) -> dict[str, object]:
    target = proposal.get("target")
    if not isinstance(target, dict):
        raise ValueError("Proposal target is missing")
    declared_fraction = target.get("f1_fraction")
    if not isinstance(declared_fraction, dict) or set(declared_fraction) != {
        "numerator",
        "denominator",
    }:
        raise ValueError("proposal.target.f1_fraction is invalid")
    numerator = _positive_integer(declared_fraction["numerator"], "target.f1_fraction.numerator")
    denominator = _positive_integer(
        declared_fraction["denominator"], "target.f1_fraction.denominator"
    )
    if numerator >= denominator:
        raise ValueError("proposal.target.f1_fraction must be below one")
    target_fraction = Fraction(numerator, denominator)
    if _exact_fraction(target.get("f1"), "target.f1") != target_fraction:
        raise ValueError("proposal.target.f1 disagrees with target.f1_fraction")

    reference = target.get("legacy_balanced_reference")
    if not isinstance(reference, dict) or set(reference) != {
        "positive_rows",
        "negative_rows",
        "scope",
    }:
        raise ValueError("proposal.target.legacy_balanced_reference is invalid")
    positive_rows = _positive_integer(
        reference["positive_rows"], "target.legacy_balanced_reference.positive_rows"
    )
    negative_rows = _positive_integer(
        reference["negative_rows"], "target.legacy_balanced_reference.negative_rows"
    )
    require_equal(
        reference["scope"],
        "legacy_development_point_reference_only",
        "proposal.target.legacy_balanced_reference.scope",
    )
    require_equal(
        legacy_full_test["positive_rows"],
        positive_rows,
        "legacy full-test positive denominator",
    )
    require_equal(
        legacy_full_test["negative_rows"],
        negative_rows,
        "legacy full-test negative denominator",
    )

    false_negative_coefficient = 2 * denominator - numerator
    false_positive_coefficient = numerator
    weighted_error_budget = 2 * positive_rows * (denominator - numerator)
    all_allocations_pass_through_errors = (
        weighted_error_budget // max(false_negative_coefficient, false_positive_coefficient)
    )
    maximum_total_errors = weighted_error_budget // min(
        false_negative_coefficient, false_positive_coefficient
    )
    maximum_false_negatives_at_maximum_total_errors = (
        weighted_error_budget - maximum_total_errors * false_positive_coefficient
    ) // (false_negative_coefficient - false_positive_coefficient)
    all_allocations_fail_from_errors = maximum_total_errors + 1
    minimum_net_error_removal = int(legacy_full_test["errors"]) - maximum_total_errors
    if minimum_net_error_removal < 0:
        raise ValueError("Legacy full-test already exceeds the declared point target")

    declared_geometry = target.get("legacy_point_geometry")
    expected_declared_geometry = {
        "all_allocations_pass_through_errors": all_allocations_pass_through_errors,
        "maximum_total_errors": maximum_total_errors,
        "maximum_false_negatives_at_maximum_total_errors": (
            maximum_false_negatives_at_maximum_total_errors
        ),
        "all_allocations_fail_from_errors": all_allocations_fail_from_errors,
        "point_reference_only_not_certification_evidence": True,
    }
    require_equal(
        declared_geometry,
        expected_declared_geometry,
        "proposal.target.legacy_point_geometry",
    )
    require_equal(
        target.get("legacy_point_error_budget"),
        "47-48",
        "proposal.target.legacy_point_error_budget",
    )
    require_equal(
        target.get("minimum_net_error_removal_from_loop151"),
        minimum_net_error_removal,
        "proposal.target.minimum_net_error_removal_from_loop151",
    )
    require_equal(
        target.get("certification_requires_two_future_sealed_windows"),
        True,
        "proposal.target.certification_requires_two_future_sealed_windows",
    )

    boundary_cases = []
    for label, false_positive, false_negative, expected_pass in (
        ("all_fn_47", 0, 47, True),
        ("balanced_24_24", 24, 24, True),
        ("all_fp_48", 48, 0, True),
        ("fn_heavy_23_25", 23, 25, False),
        ("all_fn_48", 0, 48, False),
        ("all_fp_49", 49, 0, False),
    ):
        weighted_errors = (
            false_negative_coefficient * false_negative
            + false_positive_coefficient * false_positive
        )
        passes = weighted_errors <= weighted_error_budget
        if passes is not expected_pass:
            raise AssertionError(f"Incorrect Loop164 point-F1 boundary derivation: {label}")
        boundary_cases.append(
            {
                "label": label,
                "fp": false_positive,
                "fn": false_negative,
                "weighted_errors": weighted_errors,
                "meets_point_target": passes,
            }
        )
    return {
        "f1_fraction": {"numerator": numerator, "denominator": denominator},
        "reference": {
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
            "scope": reference["scope"],
        },
        "false_negative_coefficient": false_negative_coefficient,
        "false_positive_coefficient": false_positive_coefficient,
        "weighted_error_budget": weighted_error_budget,
        "all_allocations_pass_through_errors": all_allocations_pass_through_errors,
        "maximum_total_errors": maximum_total_errors,
        "maximum_false_negatives_at_maximum_total_errors": (
            maximum_false_negatives_at_maximum_total_errors
        ),
        "all_allocations_fail_from_errors": all_allocations_fail_from_errors,
        "minimum_net_error_removal": minimum_net_error_removal,
        "boundary_cases": boundary_cases,
        "development_point_reference_only_not_certification_evidence": True,
    }


def verify_champion(proposal: dict[str, Any], split_name: str, observed: dict[str, object]) -> None:
    champion = proposal.get("champion")
    if not isinstance(champion, dict) or not isinstance(champion.get(split_name), dict):
        raise ValueError(f"Proposal champion is missing {split_name}")
    expected = champion[split_name]
    for key in ("f1", "errors", "fp", "fn"):
        require_equal(observed[key], expected.get(key), f"champion.{split_name}.{key}")


def _artifact_state(root: Path) -> dict[str, dict[str, object]]:
    state = {}
    for name, path in FUTURE_ARTIFACTS.items():
        present = resolve_path(root, path).is_file()
        state[name] = {
            "path": path.as_posix(),
            "present": present,
            "a1_verification": (
                "not_present"
                if not present
                else "present_but_unverified_by_static_preflight"
                if name in RUNTIME_VERIFICATION_ARTIFACTS
                else "present_static_reference"
            ),
        }
    return state


def runtime_artifact_blocker(name: str, artifact: dict[str, object]) -> str:
    return (
        name + "_missing"
        if not bool(artifact.get("present"))
        else name + "_unverified_by_static_preflight"
    )


def certification_artifact_state(root: Path) -> dict[str, dict[str, object]]:
    state = {}
    for name, path in CERTIFICATION_FUTURE_ARTIFACTS.items():
        present = resolve_path(root, path).is_file()
        state[name] = {
            "path": path.as_posix(),
            "present": present,
            "a1_verification": (
                "not_present" if not present else "present_but_unverified_by_static_preflight"
            ),
        }
    return state


def build_preflight(
    *,
    root: Path,
    proposal_path: Path = DEFAULT_PROPOSAL,
    authorization_path: Path = DEFAULT_AUTHORIZATION,
    sources_path: Path = DEFAULT_SOURCES,
    generated_at_utc: Optional[str] = None,
) -> dict[str, Any]:
    root = root.resolve()
    proposal_resolved = resolve_path(root, proposal_path)
    authorization_resolved = resolve_path(root, authorization_path)
    sources_resolved = resolve_path(root, sources_path)
    proposal = read_json(proposal_resolved)
    authorization = read_json(authorization_resolved)
    sources = read_json(sources_resolved)

    require_equal(proposal.get("loop_id"), "loop164_whole_file_residual_expert", "proposal.loop_id")
    static_authorization_scope = validate_static_preflight_authorization(authorization)
    proposal_binding = verify_binding(
        root, authorization.get("proposal", {}), proposal_path, "proposal"
    )
    sources_binding = verify_binding(
        root, authorization.get("frontier_sources", {}), sources_path, "frontier_sources"
    )
    isolation_contract_validator = bind_static_source(
        root,
        ISOLATION_CONTRACT_VALIDATOR,
        "Loop164 full-pool isolation contract validator",
    )
    nested_oof_receipt_validator = bind_static_source(
        root,
        NESTED_OOF_RECEIPT_VALIDATOR,
        "Loop164 nested OOF execution receipt validator",
    )
    fold_scope_plan_validator = bind_static_source(
        root,
        FOLD_SCOPE_PLAN_VALIDATOR,
        "Loop164 fold scope plan validator",
    )
    training_authority_validator = bind_static_source(
        root,
        TRAINING_AUTHORITY_VALIDATOR,
        "Loop164 training authority validator",
    )
    certification_evidence_validator = bind_static_source(
        root,
        CERTIFICATION_EVIDENCE_VALIDATOR,
        "Loop164 dual-window certification evidence validator",
    )
    a2_request_validator = bind_static_source(
        root,
        A2_REQUEST_VALIDATOR,
        "Loop164 non-authorizing A2 request validator",
    )
    a2_metadata_request_template = bind_static_source(
        root,
        A2_METADATA_REQUEST_TEMPLATE,
        "Loop164 metadata A2 request template",
    )
    a2_training_request_template = bind_static_source(
        root,
        A2_TRAINING_REQUEST_TEMPLATE,
        "Loop164 training A2 request template",
    )
    implementation_manifest_validator = bind_static_source(
        root,
        IMPLEMENTATION_MANIFEST_VALIDATOR,
        "Loop164 whole-file implementation manifest validator",
    )
    whole_file_exactness_oracle = bind_static_source(
        root,
        WHOLE_FILE_EXACTNESS_ORACLE,
        "Loop164 pure-synthetic whole-file exactness oracle",
    )
    implementation_contract_authorization_path = resolve_path(
        root, IMPLEMENTATION_CONTRACT_AUTHORIZATION
    )
    implementation_contract_authorization_payload = read_json(
        implementation_contract_authorization_path
    )
    implementation_authorization_scope = validate_static_implementation_authorization(
        implementation_contract_authorization_payload,
        root=root,
        static_authorization_path=authorization_resolved,
    )
    implementation_contract_authorization = bind_static_source(
        root,
        IMPLEMENTATION_CONTRACT_AUTHORIZATION,
        "Loop164 whole-file implementation contract authorization",
    )

    # 安全边界：本构建器只允许打开授权中逐项列出的聚合 JSON，绝不解析逐行预测、split、raw、cache 或模型载荷。
    authorized_inputs = authorization.get("allowed_aggregate_inputs")
    if not isinstance(authorized_inputs, list):
        raise ValueError("authorization.allowed_aggregate_inputs must be a list")
    expected_inputs = sorted(path.as_posix() for path in AGGREGATE_INPUTS.values())
    require_equal(
        sorted(str(item).replace("\\", "/") for item in authorized_inputs),
        expected_inputs,
        "aggregate allowlist",
    )

    aggregate_payloads: dict[str, dict[str, Any]] = {}
    aggregate_evidence: dict[str, dict[str, str]] = {}
    for evidence_id, relative_path in AGGREGATE_INPUTS.items():
        resolved = resolve_path(root, relative_path)
        if not resolved.is_file():
            raise ValueError(f"Missing authorized aggregate evidence: {relative_path}")
        aggregate_payloads[evidence_id] = read_json(resolved)
        aggregate_evidence[evidence_id] = {
            "path": relative_path.as_posix(),
            "sha256": sha256_file(resolved),
        }

    champion_metrics = {
        "val": metric_summary(aggregate_payloads["loop151_val"]),
        "test10k": metric_summary(aggregate_payloads["loop151_test10k"]),
        "legacy_full_test": metric_summary(aggregate_payloads["loop151_full"]),
    }
    for split_name, observed in champion_metrics.items():
        verify_champion(proposal, split_name, observed)
    legacy_point_geometry = build_legacy_point_geometry(
        proposal, champion_metrics["legacy_full_test"]
    )
    execution_sequence = validate_execution_sequence(proposal)
    certification_protocol = validate_certification_protocol(proposal)

    loop161 = aggregate_payloads["loop161_guard"]
    require_equal(loop161.get("decision"), "guard_active", "loop161.decision")
    loop161_thresholds = loop161.get("thresholds")
    if not isinstance(loop161_thresholds, dict):
        raise ValueError("Loop161 thresholds are missing")
    require_equal(
        int(loop161_thresholds.get("min_val_error_improvement", -1)), 3, "loop161 Val floor"
    )
    require_equal(
        int(loop161_thresholds.get("min_test10k_error_improvement", -1)),
        3,
        "loop161 Test-10k floor",
    )

    loop163 = aggregate_payloads["loop163_support"]
    require_equal(
        loop163.get("decision"), "reject_low_support_no_selector_training", "loop163.decision"
    )
    split_summaries = loop163.get("split_summaries")
    if not isinstance(split_summaries, dict) or not isinstance(split_summaries.get("val"), dict):
        raise ValueError("Loop163 Val support summary is missing")
    loop163_val = split_summaries["val"]

    loop158 = aggregate_payloads["loop158_annotations"]
    annotation_audit = loop158.get("external_annotation_audit")
    if not isinstance(annotation_audit, dict):
        raise ValueError("Loop158 external annotation audit is missing")
    returned_annotations = int(annotation_audit.get("rows", -1))
    require_equal(loop158.get("decision"), "ready_noop_no_external_annotations", "loop158.decision")
    require_equal(returned_annotations, 0, "loop158 returned annotation rows")

    group_distribution_payload = aggregate_payloads["group_distribution"]
    group_distribution = group_distribution_payload.get("distribution")
    if not isinstance(group_distribution, dict):
        raise ValueError("Group distribution summary is missing distribution")

    cache_capabilities = {}
    for evidence_id in ("content_pe_v2_cache", "content_string_cache", "content_cert_cache"):
        cache_payload = aggregate_payloads[evidence_id]
        counts = cache_payload.get("counts")
        if not isinstance(counts, dict):
            raise ValueError(f"{evidence_id} is missing counts")
        cache_capabilities[evidence_id] = {
            "schema": cache_payload.get("schema"),
            "input_rows": int(cache_payload.get("input_rows", 0)),
            "unique_rows": int(cache_payload.get("unique_rows", 0)),
            "feature_dim": int(cache_payload.get("feature_dim", 0)),
            "zero_features": int(counts.get("zero_features", 0)),
        }

    artifact_state = _artifact_state(root)
    execution_blockers = [
        runtime_artifact_blocker(name, artifact_state[name])
        for name in (
            "a2_isolation_validation_authorization",
            "a2_training_authorization",
            "whole_file_implementation_manifest",
            "loop151_train_oof_manifest",
            "loop164_train_oof_execution_receipt",
            "fold_scope_plan",
            "fold_scope_plan_validation",
            "train_oof_input_bundle",
            "training_final_lease",
            "full_pool_group_manifest",
            "full_pool_isolation_validation",
            "resource_guard",
        )
    ]
    promotion_blockers = [
        runtime_artifact_blocker(name, artifact_state[name])
        for name in ("val_a_manifest", "val_b_manifest")
    ]
    if not artifact_state["champion_registry"]["present"]:
        promotion_blockers.append("champion_registry_missing")
    certification_artifacts = certification_artifact_state(root)
    certification_blockers = [
        "certification_operational_thresholds_not_preregistered",
        *[
            name + ("_missing" if not artifact["present"] else "_unverified_by_static_preflight")
            for name, artifact in certification_artifacts.items()
        ],
    ]

    verified_sources = sources.get("sources")
    if not isinstance(verified_sources, list) or not any(
        item.get("id") == "malconv2" for item in verified_sources
    ):
        raise ValueError("Frontier sources do not contain the MalConv2 primary reference")

    generated = generated_at_utc or datetime.now(timezone.utc).isoformat()
    payload = {
        "schema": "axon_loop164_mainline_preflight_v1",
        "loop_id": "loop164_whole_file_residual_expert",
        "generated_at_utc": generated,
        "authority": {
            "proposal": proposal_binding,
            "frontier_sources": sources_binding,
            "authorization": {
                "path": authorization_path.as_posix(),
                "sha256": sha256_file(authorization_resolved),
            },
            "authorization_level": authorization.get("authorization_level"),
            "static_preflight_authorized": True,
            "a1_static_contract_verified": True,
            "model_or_data_execution_authorized": static_authorization_scope[
                "model_or_data_execution"
            ],
            "creates_runtime_artifact": static_authorization_scope["creates_runtime_artifact"],
            "implementation_contract_a1_scope_verified": implementation_authorization_scope,
        },
        "aggregate_evidence": aggregate_evidence,
        "a2_request_contract": {
            "validator": a2_request_validator,
            "metadata_template": a2_metadata_request_template,
            "training_template": a2_training_request_template,
            "document_kind": "custodian_request_not_authorization",
            "authorization_granted": False,
            "metadata_state": "draft",
            "training_state": "blocked_pending_metadata_and_static_review",
            "gate": (
                "A request is not an A2 authorization: it cannot carry an allow decision, lease, "
                "issuance window, runtime binding, or protected-input grant. Only a separately "
                "custodian-issued authorization may satisfy a future execution blocker."
            ),
        },
        "research_champion": champion_metrics,
        "target_gap": {
            "target_f1": float(proposal["target"]["f1"]),
            "f1_fraction": legacy_point_geometry["f1_fraction"],
            "legacy_point_error_budget": proposal["target"]["legacy_point_error_budget"],
            "legacy_point_geometry": legacy_point_geometry,
            "minimum_net_error_removal": legacy_point_geometry["minimum_net_error_removal"],
        },
        "execution_sequence": execution_sequence,
        "closed_routes": {
            "probability_and_r11_selector": {
                "decision": loop163.get("decision"),
                "val_disagreements": int(loop163_val.get("disagreement_rows", 0)),
                "val_fixes": int(
                    loop163_val.get("outcome_counts", {}).get("candidate_fixes_base_error", 0)
                ),
                "val_breaks": int(
                    loop163_val.get("outcome_counts", {}).get("candidate_breaks_base_correct", 0)
                ),
            },
            "promotion_floor": {
                "decision": loop161.get("decision"),
                "min_val_error_improvement": 3,
                "min_test10k_error_improvement": 3,
            },
        },
        "parallel_data_governance": {
            "loop158_decision": loop158.get("decision"),
            "returned_annotations": returned_annotations,
            "private_join_performed": bool(loop158.get("private_join_performed")),
            "promotion_dependency": "parallel_only_not_a_loop164_promotion_blocker",
        },
        "current_capabilities": {
            "legacy_group_diagnostics": {
                "total_samples": int(group_distribution.get("total_samples", 0)),
                "total_groups": int(group_distribution.get("total_groups", 0)),
                "cross_split_groups": int(group_distribution.get("cross_split_groups", 0)),
                "leakage_groups": int(group_distribution.get("leakage_groups", 0)),
                "scope": "historical_40k_diagnostic_not_loop164_full_pool_contract",
            },
            "existing_sidecar_caches": cache_capabilities,
            "whole_file_expert_present": artifact_state["whole_file_implementation_manifest"][
                "present"
            ],
            "loop151_train_oof_present": artifact_state["loop151_train_oof_manifest"]["present"],
        },
        "full_pool_isolation_contract": {
            "validator": isolation_contract_validator,
            "contract_schema": ISOLATION_CONTRACT_SCHEMA,
            "receipt_schema": ISOLATION_RECEIPT_SCHEMA,
            "required_hard_relations": [
                "exact_cluster_id",
                "near_duplicate_cluster_id",
                "family_id",
                "campaign_id",
                "source_group_id",
            ],
            "required_time_boundary": "purged_forward_group_oof_with_explicit_warmup",
            "model_identity_feature_count_required": 0,
            "feature_contract": {
                "feature_fields": [
                    "loop151_oof_score",
                    "whole_file_oof_score",
                    "loop151_oof_uncertainty",
                    "whole_file_oof_uncertainty",
                    "loop151_missingness",
                    "whole_file_missingness",
                ],
                "implementation_binding_phase": "deferred_to_a2_training_authority",
                "metadata_receipt_binds_production_implementation": False,
            },
            "future_manifest": artifact_state["full_pool_group_manifest"],
            "future_validation_receipt": artifact_state["full_pool_isolation_validation"],
            "future_fold_scope_plan": artifact_state["fold_scope_plan"],
            "future_fold_scope_plan_validation": artifact_state["fold_scope_plan_validation"],
            "future_inner_oof_execution_receipt": artifact_state[
                "loop164_train_oof_execution_receipt"
            ],
            "a2_gate": (
                "An externally anchored A2 metadata v3 authorization must bind the exact argv, source closure, "
                "canonical output, stable lease, metadata-only scope, and empty grants before opening rows; its "
                "provenance-bound aggregate-only receipt freezes only partition and residual-feature semantics. The production implementation "
                "manifest is intentionally deferred to the separately authorized A2 training authority."
            ),
        },
        "fold_scope_plan_contract": {
            "validator": fold_scope_plan_validator,
            "plan_schema": FOLD_SCOPE_PLAN_SCHEMA,
            "validation_schema": FOLD_SCOPE_PLAN_VALIDATION_SCHEMA,
            "future_scope_plan": artifact_state["fold_scope_plan"],
            "future_validation_receipt": artifact_state["fold_scope_plan_validation"],
            "gate": (
                "The aggregate-only validator must produce the canonical validation receipt after the "
                "custodian freezes the scope plan. A pass freezes scope commitments only; it does not "
                "authorize training, consume a training lease, or open any heldout split."
            ),
        },
        "whole_file_implementation_contract": {
            "validator": implementation_manifest_validator,
            "synthetic_exactness_oracle": whole_file_exactness_oracle,
            "authorization": implementation_contract_authorization,
            "manifest_schema": IMPLEMENTATION_MANIFEST_SCHEMA,
            "future_manifest": artifact_state["whole_file_implementation_manifest"],
            "required_source_roles": [
                "controller",
                "model",
                "input_loader",
                "oof_protocol",
                "fusion",
                "dense_equivalence_test",
            ],
            "required_semantics": [
                "all_bytes_chunked_no_silent_truncation",
                "exact_independent_regions",
                "reserved_pad_token_or_explicit_length_mask",
                "denominator_with_explicit_missingness",
                "zero_identity_features",
            ],
            "oracle_scope": "in_memory_synthetic_only_no_protected_input_or_f1_claim",
            "gate": (
                "A1 only source-binds the validator and synthetic review contract. The production manifest "
                "remains missing until the full-pool contract passes and a future implementation is separately "
                "authorized and statically reviewed."
            ),
        },
        "training_authority_contract": {
            "validator": training_authority_validator,
            "authorization_schema": TRAINING_AUTHORIZATION_SCHEMA,
            "final_lease_schema": TRAINING_LEASE_SCHEMA,
            "input_bundle_schema": TRAINING_INPUT_BUNDLE_SCHEMA,
            "future_training_authorization": artifact_state["a2_training_authorization"],
            "future_scope_plan_validation": artifact_state["fold_scope_plan_validation"],
            "future_input_bundle": artifact_state["train_oof_input_bundle"],
            "future_final_lease": artifact_state["training_final_lease"],
            "gate": (
                "The future controller must validate the external trust anchor, fixed runtime, canonical argv, "
                "scope validation, input bundle, and fresh resource guard in-process before atomically consuming "
                "the final lease. This static source binding does not authorize execution."
            ),
        },
        "nested_oof_execution_contract": {
            "validator": nested_oof_receipt_validator,
            "receipt_schema": NESTED_OOF_RECEIPT_SCHEMA,
            "validation_schema": NESTED_OOF_VALIDATION_SCHEMA,
            "fixed_seeds": [41, 42, 43],
            "outer_fold_count": 5,
            "inner_fold_count_per_outer_fold": 5,
            "frozen_fusion_feature_fields": [
                "loop151_oof_score",
                "whole_file_oof_score",
                "loop151_oof_uncertainty",
                "whole_file_oof_uncertainty",
                "loop151_missingness",
                "whole_file_missingness",
            ],
            "required_future_artifacts": {
                "fold_scope_plan": artifact_state["fold_scope_plan"],
                "fold_scope_plan_validation": artifact_state["fold_scope_plan_validation"],
                "train_oof_input_bundle": artifact_state["train_oof_input_bundle"],
                "training_authorization": artifact_state["a2_training_authorization"],
                "training_final_lease": artifact_state["training_final_lease"],
                "execution_receipt": artifact_state["loop164_train_oof_execution_receipt"],
            },
            "gate": (
                "The aggregate receipt must bind the canonical proposal, provenance-attested isolation contract and pass "
                "receipt, scope plan, implementation review, Loop151 Train OOF manifest, resource "
                "guard, v2 training authority, input bundle, content-addressed lease marker, and final lease. "
                "A receipt pass only proves the Train-OOF boundary; it never authorizes Val, Test-10k, or "
                "full-test access."
            ),
        },
        "future_artifacts": artifact_state,
        "execution_blockers": execution_blockers,
        "promotion_blockers": promotion_blockers,
        "certification_contract": {
            "status": "static_protocol_preregistered_runtime_evidence_missing",
            "legacy_point_geometry_scope": "development_point_reference_only_not_certification_evidence",
            "protocol": certification_protocol,
            "validator": certification_evidence_validator,
            "validator_scope": "aggregate_only_dual_window_hash_statistical_and_operational_gates",
            "future_artifacts": certification_artifacts,
        },
        "certification_blockers": certification_blockers,
        "ready_for": {
            "static_design_review": True,
            "implementation": False,
            "train_oof": False,
            "legacy_val_development": False,
            "test10k": False,
            "legacy_full_test": False,
            "certification": False,
        },
        "decision": "static_preflight_ready_execution_blocked_missing_prerequisites",
        "next_action": (
            "Obtain a metadata-only A2 authorization before any future full-pool inventory open. Its pass "
            "freezes partition semantics only; then complete scope validation and static implementation review "
            "before requesting the separate A2 training authority."
        ),
    }
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Loop164 aggregate-only static preflight."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--proposal", type=Path, default=DEFAULT_PROPOSAL)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--frontier-sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--manifest-output-json", type=Path, required=True)
    parser.add_argument("--generated-at-utc", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    authorization = read_json(resolve_path(root, args.authorization))
    authorized_outputs = authorization.get("authorized_generated_paths")
    expected_outputs = {
        LOOP_DIR / "preflight.json",
        Path("reports/roadmap_9997/loop164/preflight.json"),
    }
    if (
        not isinstance(authorized_outputs, list)
        or {Path(str(path)) for path in authorized_outputs} != expected_outputs
    ):
        raise ValueError("authorization.authorized_generated_paths mismatch")
    if Path(args.output_json) != LOOP_DIR / "preflight.json":
        raise ValueError("output-json is not authorized for Loop164 static preflight")
    if Path(args.manifest_output_json) != Path("reports/roadmap_9997/loop164/preflight.json"):
        raise ValueError("manifest-output-json is not authorized for Loop164 static preflight")
    payload = build_preflight(
        root=root,
        proposal_path=args.proposal,
        authorization_path=args.authorization,
        sources_path=args.frontier_sources,
        generated_at_utc=args.generated_at_utc,
    )
    write_json(resolve_path(root, args.output_json), payload)
    write_json(resolve_path(root, args.manifest_output_json), payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "execution_blockers": payload["execution_blockers"],
                "promotion_blockers": payload["promotion_blockers"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
