from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop164_mainline_preflight import (  # noqa: E402
    A2_METADATA_REQUEST_TEMPLATE,
    A2_REQUEST_VALIDATOR,
    A2_TRAINING_REQUEST_TEMPLATE,
    AGGREGATE_INPUTS,
    CERTIFICATION_EVIDENCE_VALIDATOR,
    CERTIFICATION_FUTURE_ARTIFACTS,
    FOLD_SCOPE_PLAN_VALIDATOR,
    FUTURE_ARTIFACTS,
    IMPLEMENTATION_CONTRACT_AUTHORIZATION,
    IMPLEMENTATION_MANIFEST_VALIDATOR,
    ISOLATION_CONTRACT_VALIDATOR,
    METADATA_A2_AUTHORIZATION_PREREQUISITE,
    NESTED_OOF_RECEIPT_VALIDATOR,
    TRAINING_AUTHORITY_VALIDATOR,
    WHOLE_FILE_EXACTNESS_ORACLE,
    build_preflight,
)

LOOP_DIR = Path("manifests/roadmap_9997/loop164_whole_file_residual_expert")


def write_json(root: Path, relative_path: Path, payload: dict) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_implementation_authorization_basis(root: Path, authorization_path: Path) -> None:
    amendment_path = root / IMPLEMENTATION_CONTRACT_AUTHORIZATION
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    amendment["basis_authorization"]["sha256"] = sha256_file(authorization_path)
    amendment_path.write_text(json.dumps(amendment, indent=2) + "\n", encoding="utf-8")


def candidate_metrics(
    *, positive_rows: int, negative_rows: int, false_positive: int, false_negative: int
) -> dict:
    true_positive = positive_rows - false_negative
    true_negative = negative_rows - false_positive
    errors = false_positive + false_negative
    f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    return {
        "candidate": {
            "f1": f1,
            "errors": errors,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
            "true_negative": true_negative,
        }
    }


def certification_protocol() -> dict:
    target_fraction = {"numerator": 9997, "denominator": 10000}
    return {
        "schema": "axon_loop164_certification_protocol_v1",
        "state": "static_preregistration_only_no_a3_authorization",
        "amendment_policy": "forbidden_after_first_sealed_window_lock",
        "claim_scope": {
            "legacy_point_is_non_certification": True,
            "target_f1_fraction": target_fraction,
            "two_window_replication_required": True,
        },
        "sealed_windows": {
            "window_ids": ["W1_certification", "W2_later_replication"],
            "time_order": "W2_starts_after_W1_ends",
            "evaluation_generation": 1,
            "independent_a3_authorization_and_lease_required": True,
            "window_pooling_or_replacement_forbidden": True,
        },
        "candidate_freeze": {
            "frozen_before_W1_release": True,
            "required_hash_fields": [
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
            ],
            "post_W1_selection_or_recalibration_forbidden": True,
        },
        "isolation": {
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
        "scoring": {
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
        "inference": {
            "familywise_alpha": 0.05,
            "per_window_one_sided_alpha": 0.025,
            "per_window_lower_confidence_level": 0.975,
            "method": "stratified_relationship_component_bootstrap_with_conservative_guard",
            "resampling_unit": "unioned_relationship_component_with_temporal_block",
            "bootstrap_replicates": 200000,
            "seed_derivation": "sha256(protocol_sha256|window_manifest_sha256|bundle_sha256|statistics_runner_sha256)",
            "statistical_failure_decision": "insufficient_evidence",
        },
        "acceptance": {
            "per_window_point_f1_fraction": target_fraction,
            "per_window_lower_bound_fraction": target_fraction,
            "both_window_pass_required": True,
            "planned_point_floor": "blinded_power_analysis_must_set_strictly_above_target_before_A3",
            "failed_window_burns_lineage_without_replacement": True,
        },
        "power_analysis": {
            "required_before_A3": True,
            "method": "aggregate_only_grouped_bootstrap_simulation",
            "minimum_simulations": 50000,
            "minimum_joint_power": 0.9,
            "output_sha256_required": True,
            "insufficient_support_blocks_A3": True,
        },
        "operational_gates": {
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
        "future_artifacts": [path.as_posix() for path in CERTIFICATION_FUTURE_ARTIFACTS.values()],
    }


def create_fixture(root: Path) -> tuple[Path, Path, Path]:
    validator_path = root / ISOLATION_CONTRACT_VALIDATOR
    validator_path.parent.mkdir(parents=True, exist_ok=True)
    validator_path.write_text("# synthetic static validator\n", encoding="utf-8")
    nested_validator_path = root / NESTED_OOF_RECEIPT_VALIDATOR
    nested_validator_path.parent.mkdir(parents=True, exist_ok=True)
    nested_validator_path.write_text("# synthetic nested OOF validator\n", encoding="utf-8")
    scope_validator_path = root / FOLD_SCOPE_PLAN_VALIDATOR
    scope_validator_path.parent.mkdir(parents=True, exist_ok=True)
    scope_validator_path.write_text("# synthetic fold scope validator\n", encoding="utf-8")
    training_authority_validator_path = root / TRAINING_AUTHORITY_VALIDATOR
    training_authority_validator_path.parent.mkdir(parents=True, exist_ok=True)
    training_authority_validator_path.write_text(
        "# synthetic training authority validator\n", encoding="utf-8"
    )
    certification_validator_path = root / CERTIFICATION_EVIDENCE_VALIDATOR
    certification_validator_path.parent.mkdir(parents=True, exist_ok=True)
    certification_validator_path.write_text("# synthetic certification validator\n", encoding="utf-8")
    a2_request_validator_path = root / A2_REQUEST_VALIDATOR
    a2_request_validator_path.parent.mkdir(parents=True, exist_ok=True)
    a2_request_validator_path.write_text("# synthetic A2 request validator\n", encoding="utf-8")
    for template_path in (A2_METADATA_REQUEST_TEMPLATE, A2_TRAINING_REQUEST_TEMPLATE):
        write_json(root, template_path, {"synthetic": True})
    implementation_validator_path = root / IMPLEMENTATION_MANIFEST_VALIDATOR
    implementation_validator_path.parent.mkdir(parents=True, exist_ok=True)
    implementation_validator_path.write_text(
        "# synthetic implementation manifest validator\n", encoding="utf-8"
    )
    exactness_oracle_path = root / WHOLE_FILE_EXACTNESS_ORACLE
    exactness_oracle_path.parent.mkdir(parents=True, exist_ok=True)
    exactness_oracle_path.write_text("# synthetic exactness oracle\n", encoding="utf-8")
    val_metrics = candidate_metrics(
        positive_rows=10_000,
        negative_rows=10_000,
        false_positive=105,
        false_negative=57,
    )
    test10k_metrics = candidate_metrics(
        positive_rows=5_000,
        negative_rows=5_000,
        false_positive=49,
        false_negative=29,
    )
    legacy_full_metrics = candidate_metrics(
        positive_rows=80_000,
        negative_rows=80_000,
        false_positive=879,
        false_negative=587,
    )
    proposal_path = write_json(
        root,
        LOOP_DIR / "proposal.json",
        {
            "loop_id": "loop164_whole_file_residual_expert",
            "champion": {
                "val": {"f1": val_metrics["candidate"]["f1"], "errors": 162, "fp": 105, "fn": 57},
                "test10k": {
                    "f1": test10k_metrics["candidate"]["f1"],
                    "errors": 78,
                    "fp": 49,
                    "fn": 29,
                },
                "legacy_full_test": {
                    "f1": legacy_full_metrics["candidate"]["f1"],
                    "errors": 1466,
                    "fp": 879,
                    "fn": 587,
                },
            },
            "target": {
                "f1": 0.9997,
                "f1_fraction": {"numerator": 9997, "denominator": 10000},
                "legacy_point_error_budget": "47-48",
                "minimum_net_error_removal_from_loop151": 1418,
                "legacy_balanced_reference": {
                    "positive_rows": 80000,
                    "negative_rows": 80000,
                    "scope": "legacy_development_point_reference_only",
                },
                "legacy_point_geometry": {
                    "all_allocations_pass_through_errors": 47,
                    "maximum_total_errors": 48,
                    "maximum_false_negatives_at_maximum_total_errors": 24,
                    "all_allocations_fail_from_errors": 49,
                    "point_reference_only_not_certification_evidence": True,
                },
                "certification_requires_two_future_sealed_windows": True,
            },
            "compute_budget": {
                "state": "not_authorized_and_not_yet_sized",
                "a1_static_prerequisites": ["synthetic_static_contract"],
                "a2_metadata_request_prerequisites": [
                    METADATA_A2_AUTHORIZATION_PREREQUISITE
                ],
                "a2_metadata_outputs": [
                    "aggregate_validation_receipt_with_partition_and_feature_semantics_only"
                ],
                "a2_training_request_prerequisites": [
                    "externally_pinned_A2_training_authorization_v2"
                ],
                "a2_training_outputs": [
                    "complete_nested_oof_execution_receipt_before_train_or_fusion_claim"
                ],
            },
            "certification_protocol": certification_protocol(),
        },
    )
    sources_path = write_json(
        root, LOOP_DIR / "frontier_sources.json", {"sources": [{"id": "malconv2"}]}
    )
    authorization_path = write_json(
        root,
        LOOP_DIR / "authorization.json",
        {
            "schema": "axon_loop164_static_preflight_authorization_v1",
            "loop_id": "loop164_whole_file_residual_expert",
            "authorization_level": "A1_scoped_design_and_static_preflight_only",
            "decision": "allow_static_preflight_only_block_all_model_and_data_execution",
            "authority_scope": {
                "tier": "A1",
                "protected_input_access": False,
                "model_or_data_execution": False,
                "creates_runtime_artifact": False,
                "heldout_evaluation": False,
            },
            "proposal": {
                "path": (LOOP_DIR / "proposal.json").as_posix(),
                "sha256": sha256_file(proposal_path),
            },
            "frontier_sources": {
                "path": (LOOP_DIR / "frontier_sources.json").as_posix(),
                "sha256": sha256_file(sources_path),
            },
            "allowed_aggregate_inputs": [path.as_posix() for path in AGGREGATE_INPUTS.values()],
            "execution_budget": {
                "raw_files": 0,
                "checkpoint_loads": 0,
                "prediction_rows": 0,
                "cache_rows": 0,
                "training_runs": 0,
                "model_evaluations": 0,
                "gpu_runs": 0,
                "dependency_installs": 0,
            },
        },
    )
    write_json(
        root,
        IMPLEMENTATION_CONTRACT_AUTHORIZATION,
        {
            "schema": "axon_loop164_static_implementation_contract_authorization_v1",
            "loop_id": "loop164_whole_file_residual_expert",
            "authorization_level": "A1_scoped_static_contract_only",
            "decision": "allow_synthetic_static_contract_validation_only",
            "basis_authorization": {
                "path": (LOOP_DIR / "authorization.json").as_posix(),
                "sha256": sha256_file(authorization_path),
            },
            "authority_scope": {
                "tier": "A1",
                "protected_input_access": False,
                "model_or_data_execution": False,
                "creates_runtime_artifact": False,
                "heldout_evaluation": False,
            },
            "execution_budget": {
                "raw_files": 0,
                "checkpoint_loads": 0,
                "prediction_rows": 0,
                "cache_rows": 0,
                "training_runs": 0,
                "model_evaluations": 0,
                "gpu_runs": 0,
                "dependency_installs": 0,
            },
        },
    )

    write_json(root, AGGREGATE_INPUTS["loop151_val"], val_metrics)
    write_json(root, AGGREGATE_INPUTS["loop151_test10k"], test10k_metrics)
    write_json(root, AGGREGATE_INPUTS["loop151_full"], legacy_full_metrics)
    write_json(
        root,
        AGGREGATE_INPUTS["loop161_guard"],
        {
            "decision": "guard_active",
            "thresholds": {"min_val_error_improvement": 3, "min_test10k_error_improvement": 3},
        },
    )
    write_json(
        root,
        AGGREGATE_INPUTS["loop163_support"],
        {
            "decision": "reject_low_support_no_selector_training",
            "split_summaries": {
                "val": {
                    "disagreement_rows": 9,
                    "outcome_counts": {
                        "candidate_fixes_base_error": 8,
                        "candidate_breaks_base_correct": 1,
                    },
                }
            },
        },
    )
    write_json(
        root,
        AGGREGATE_INPUTS["loop158_annotations"],
        {
            "decision": "ready_noop_no_external_annotations",
            "private_join_performed": False,
            "external_annotation_audit": {"rows": 0},
        },
    )
    write_json(
        root,
        AGGREGATE_INPUTS["group_distribution"],
        {
            "distribution": {
                "total_samples": 40000,
                "total_groups": 35052,
                "cross_split_groups": 338,
                "leakage_groups": 273,
            }
        },
    )
    for evidence_id, schema, feature_dim, zero_features in (
        ("content_pe_v2_cache", "pe_v2", 182, 0),
        ("content_string_cache", "string", 43, 0),
        ("content_cert_cache", "cert", 55, 26249),
    ):
        write_json(
            root,
            AGGREGATE_INPUTS[evidence_id],
            {
                "schema": schema,
                "input_rows": 40000,
                "unique_rows": 40000,
                "feature_dim": feature_dim,
                "counts": {"zero_features": zero_features},
            },
        )
    return proposal_path, authorization_path, sources_path


def test_preflight_freezes_champion_and_blocks_execution(tmp_path: Path):
    create_fixture(tmp_path)

    payload = build_preflight(root=tmp_path, generated_at_utc="2026-07-12T00:00:00Z")

    assert payload["decision"] == "static_preflight_ready_execution_blocked_missing_prerequisites"
    assert payload["research_champion"]["legacy_full_test"]["errors"] == 1466
    assert payload["target_gap"]["minimum_net_error_removal"] == 1418
    geometry = payload["target_gap"]["legacy_point_geometry"]
    assert geometry["false_negative_coefficient"] == 10003
    assert geometry["false_positive_coefficient"] == 9997
    assert geometry["weighted_error_budget"] == 480000
    assert geometry["all_allocations_pass_through_errors"] == 47
    assert geometry["maximum_total_errors"] == 48
    assert geometry["maximum_false_negatives_at_maximum_total_errors"] == 24
    assert geometry["all_allocations_fail_from_errors"] == 49
    assert {
        (case["fp"], case["fn"]): case["meets_point_target"]
        for case in geometry["boundary_cases"]
    } == {
        (0, 47): True,
        (24, 24): True,
        (48, 0): True,
        (23, 25): False,
        (0, 48): False,
        (49, 0): False,
    }
    assert payload["closed_routes"]["probability_and_r11_selector"]["val_disagreements"] == 9
    assert payload["parallel_data_governance"]["returned_annotations"] == 0
    assert payload["ready_for"]["static_design_review"] is True
    assert payload["authority"]["a1_static_contract_verified"] is True
    assert payload["authority"]["model_or_data_execution_authorized"] is False
    assert payload["authority"]["creates_runtime_artifact"] is False
    assert payload["ready_for"]["train_oof"] is False
    assert "a2_training_authorization_missing" in payload["execution_blockers"]
    assert "a2_isolation_validation_authorization_missing" in payload["execution_blockers"]
    assert "loop164_train_oof_execution_receipt_missing" in payload["execution_blockers"]
    assert "full_pool_isolation_validation_missing" in payload["execution_blockers"]
    assert "fold_scope_plan_validation_missing" in payload["execution_blockers"]
    assert "train_oof_input_bundle_missing" in payload["execution_blockers"]
    assert "independent_loop157_annotations_missing" not in payload["promotion_blockers"]
    assert payload["parallel_data_governance"]["promotion_dependency"] == (
        "parallel_only_not_a_loop164_promotion_blocker"
    )
    assert set(payload["aggregate_evidence"]) == set(AGGREGATE_INPUTS)
    assert payload["full_pool_isolation_contract"]["contract_schema"] == (
        "axon_loop164_full_pool_isolation_contract_v2"
    )
    assert payload["nested_oof_execution_contract"]["receipt_schema"] == (
        "axon_loop164_train_oof_execution_receipt_v1"
    )
    assert payload["nested_oof_execution_contract"]["outer_fold_count"] == 5
    assert payload["nested_oof_execution_contract"]["required_future_artifacts"][
        "fold_scope_plan"
    ]["present"] is False
    assert payload["nested_oof_execution_contract"]["required_future_artifacts"][
        "train_oof_input_bundle"
    ]["present"] is False
    assert payload["fold_scope_plan_contract"]["validator"]["path"] == (
        FOLD_SCOPE_PLAN_VALIDATOR.as_posix()
    )
    assert payload["fold_scope_plan_contract"]["validation_schema"] == (
        "axon_loop164_fold_scope_plan_validation_v1"
    )
    assert payload["training_authority_contract"]["validator"]["path"] == (
        TRAINING_AUTHORITY_VALIDATOR.as_posix()
    )
    assert payload["a2_request_contract"]["validator"]["path"] == (
        A2_REQUEST_VALIDATOR.as_posix()
    )
    assert payload["a2_request_contract"]["authorization_granted"] is False
    assert payload["a2_request_contract"]["metadata_template"]["path"] == (
        A2_METADATA_REQUEST_TEMPLATE.as_posix()
    )
    assert payload["training_authority_contract"]["authorization_schema"] == (
        "axon_loop164_training_authorization_v2"
    )
    assert payload["whole_file_implementation_contract"]["validator"]["path"] == (
        IMPLEMENTATION_MANIFEST_VALIDATOR.as_posix()
    )
    assert payload["whole_file_implementation_contract"]["future_manifest"]["present"] is False
    assert payload["whole_file_implementation_contract"]["synthetic_exactness_oracle"]["path"] == (
        WHOLE_FILE_EXACTNESS_ORACLE.as_posix()
    )
    assert payload["full_pool_isolation_contract"]["feature_contract"][
        "implementation_binding_phase"
    ] == "deferred_to_a2_training_authority"
    assert payload["certification_contract"]["status"] == (
        "static_protocol_preregistered_runtime_evidence_missing"
    )
    assert payload["certification_contract"]["validator"]["path"] == (
        CERTIFICATION_EVIDENCE_VALIDATOR.as_posix()
    )
    assert "certification_design_missing" not in payload["certification_blockers"]
    assert "certification_power_analysis_missing" in payload["certification_blockers"]
    assert "certification_operational_thresholds_not_preregistered" in payload[
        "certification_blockers"
    ]
    assert payload["execution_sequence"]["a2_metadata_request_prerequisites"] == [
        METADATA_A2_AUTHORIZATION_PREREQUISITE
    ]
    assert "metadata-only A2 authorization before" in payload["next_action"]


def test_preflight_rejects_stale_proposal_binding(tmp_path: Path):
    proposal_path, _, _ = create_fixture(tmp_path)
    proposal_path.write_text(proposal_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="proposal.sha256 mismatch"):
        build_preflight(root=tmp_path, generated_at_utc="2026-07-12T00:00:00Z")


def test_preflight_rejects_unexpected_aggregate_allowlist(tmp_path: Path):
    _, authorization_path, _ = create_fixture(tmp_path)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["allowed_aggregate_inputs"].append("reports/private_predictions.csv")
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    refresh_implementation_authorization_basis(tmp_path, authorization_path)

    with pytest.raises(ValueError, match="aggregate allowlist mismatch"):
        build_preflight(root=tmp_path, generated_at_utc="2026-07-12T00:00:00Z")


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda authorization: authorization.update(
                {"authorization_level": "A2_heavy_compute"}
            ),
            "authorization.authorization_level mismatch",
        ),
        (
            lambda authorization: authorization["execution_budget"].update({"training_runs": 1}),
            "authorization.execution_budget must be zero",
        ),
    ],
)
def test_preflight_rejects_nonstatic_main_authorization(
    tmp_path: Path, mutate, error: str
):
    _, authorization_path, _ = create_fixture(tmp_path)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    mutate(authorization)
    authorization_path.write_text(json.dumps(authorization, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        build_preflight(root=tmp_path, generated_at_utc="2026-07-12T00:00:00Z")


def test_preflight_rejects_nonstatic_implementation_amendment(tmp_path: Path):
    create_fixture(tmp_path)
    amendment_path = tmp_path / IMPLEMENTATION_CONTRACT_AUTHORIZATION
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    amendment["authority_scope"]["creates_runtime_artifact"] = True
    amendment_path.write_text(json.dumps(amendment, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="implementation_authorization.authority_scope mismatch"
    ):
        build_preflight(root=tmp_path, generated_at_utc="2026-07-12T00:00:00Z")


def test_preflight_rejects_inconsistent_point_geometry_after_refreshing_binding(tmp_path: Path):
    proposal_path, authorization_path, _ = create_fixture(tmp_path)
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal["target"]["legacy_point_geometry"]["maximum_false_negatives_at_maximum_total_errors"] = 25
    proposal_path.write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["proposal"]["sha256"] = sha256_file(proposal_path)
    authorization_path.write_text(json.dumps(authorization, indent=2) + "\n", encoding="utf-8")
    refresh_implementation_authorization_basis(tmp_path, authorization_path)

    with pytest.raises(ValueError, match="proposal.target.legacy_point_geometry mismatch"):
        build_preflight(root=tmp_path, generated_at_utc="2026-07-12T00:00:00Z")


def test_preflight_rejects_inconsistent_confusion_matrix(tmp_path: Path):
    create_fixture(tmp_path)
    full_path = tmp_path / AGGREGATE_INPUTS["loop151_full"]
    full_payload = json.loads(full_path.read_text(encoding="utf-8"))
    full_payload["candidate"]["errors"] = 1465
    full_path.write_text(json.dumps(full_payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="errors do not match false-positive plus false-negative"):
        build_preflight(root=tmp_path, generated_at_utc="2026-07-12T00:00:00Z")


def test_preflight_rejects_training_a2_as_metadata_prerequisite(tmp_path: Path):
    proposal_path, authorization_path, _ = create_fixture(tmp_path)
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal["compute_budget"]["a2_metadata_request_prerequisites"].append(
        "externally_pinned_A2_training_authorization_v2"
    )
    proposal_path.write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["proposal"]["sha256"] = sha256_file(proposal_path)
    authorization_path.write_text(json.dumps(authorization, indent=2) + "\n", encoding="utf-8")
    refresh_implementation_authorization_basis(tmp_path, authorization_path)

    with pytest.raises(ValueError, match="training A2 cannot precede metadata outputs"):
        build_preflight(root=tmp_path, generated_at_utc="2026-07-12T00:00:00Z")


def test_preflight_rejects_weakened_certification_inference(tmp_path: Path):
    proposal_path, authorization_path, _ = create_fixture(tmp_path)
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal["certification_protocol"]["inference"]["bootstrap_replicates"] = 10000
    proposal_path.write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["proposal"]["sha256"] = sha256_file(proposal_path)
    authorization_path.write_text(json.dumps(authorization, indent=2) + "\n", encoding="utf-8")
    refresh_implementation_authorization_basis(tmp_path, authorization_path)

    with pytest.raises(ValueError, match="certification.inference mismatch"):
        build_preflight(root=tmp_path, generated_at_utc="2026-07-12T00:00:00Z")


def test_preflight_keeps_runtime_artifacts_blocked_when_placeholder_files_exist(tmp_path: Path):
    create_fixture(tmp_path)
    placeholder_path = tmp_path / FUTURE_ARTIFACTS["a2_training_authorization"]
    placeholder_path.parent.mkdir(parents=True, exist_ok=True)
    placeholder_path.write_text("{}\n", encoding="utf-8")
    scope_validation_placeholder = tmp_path / FUTURE_ARTIFACTS["fold_scope_plan_validation"]
    scope_validation_placeholder.parent.mkdir(parents=True, exist_ok=True)
    scope_validation_placeholder.write_text("{}\n", encoding="utf-8")
    input_bundle_placeholder = tmp_path / FUTURE_ARTIFACTS["train_oof_input_bundle"]
    input_bundle_placeholder.parent.mkdir(parents=True, exist_ok=True)
    input_bundle_placeholder.write_text("{}\n", encoding="utf-8")

    payload = build_preflight(root=tmp_path, generated_at_utc="2026-07-12T00:00:00Z")

    assert (
        "a2_training_authorization_unverified_by_static_preflight" in payload["execution_blockers"]
    )
    assert (
        "fold_scope_plan_validation_unverified_by_static_preflight"
        in payload["execution_blockers"]
    )
    assert "train_oof_input_bundle_unverified_by_static_preflight" in payload[
        "execution_blockers"
    ]


def test_preflight_keeps_certification_artifacts_blocked_when_placeholder_exists(tmp_path: Path):
    create_fixture(tmp_path)
    placeholder_path = tmp_path / CERTIFICATION_FUTURE_ARTIFACTS["sealed_window_w1_receipt"]
    placeholder_path.parent.mkdir(parents=True, exist_ok=True)
    placeholder_path.write_text('{"decision":"pass"}\n', encoding="utf-8")

    payload = build_preflight(root=tmp_path, generated_at_utc="2026-07-12T00:00:00Z")

    assert payload["ready_for"]["certification"] is False
    assert "sealed_window_w1_receipt_unverified_by_static_preflight" in payload[
        "certification_blockers"
    ]
