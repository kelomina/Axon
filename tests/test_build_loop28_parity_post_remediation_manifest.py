from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_loop28_parity_post_remediation_manifest as post  # noqa: E402


def _write_bytes(root: Path, path: Path, payload: bytes) -> str:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_json(root: Path, path: Path, payload: dict) -> str:
    return _write_bytes(
        root,
        path,
        (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def _artifact_record(root: Path, name: str, path: Path) -> dict:
    target = root / path
    return {
        "name": name,
        "role": "test",
        "path": path.as_posix(),
        "required": True,
        "expected_sha256": None,
        "exists": True,
        "size_bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "expected_sha256_match": None,
    }


def _valid_gate() -> dict:
    return {
        "tolerance": post.FIXED_TOLERANCE,
        "checks": {name: True for name in sorted(post.GATE_CHECKS)},
        "stage2_probability_transform_mismatch_indices": [],
        "base_probability_max_absolute_delta": 1.0e-8,
        "stage2_probability_absolute_delta": 2.0e-8,
        "final_probability_max_absolute_delta": 2.0e-8,
        "passed": True,
        "decision": "frozen_train_remediation_gate_passed",
    }


def _valid_budget_audit() -> dict:
    return {
        "generation": {"count": 1, "limit": 1, "within_budget": True},
        "verified_raw_snapshot": {"count": 1, "limit": 1, "within_budget": True},
        "python": {"count": 1, "limit": 2, "within_budget": True},
        "native": {"count": 1, "limit": 6, "within_budget": True},
        "crossfeed": {"count": 1, "limit": 4, "within_budget": True},
        "total_wall_clock": {"within_budget": True},
        "output": {"within_budget": True},
        "within_budget": True,
    }


def _build_chain(root: Path) -> dict[str, object]:
    for path, payload in (
        (post.PROPOSAL, b"proposal\n"),
        (post.AUTHORIZATION, b"authorization\n"),
        (post.PREFLIGHT, b"preflight\n"),
        (post.POST_BUILDER, b"post builder\n"),
        (post.POST_BUILDER_TEST, b"post builder tests\n"),
        (post.RECOMMENDATIONS, b"recommendations final\n"),
        (post.EXPERIMENT_JOURNAL, b"journal final\n"),
        (post.GOAL, b"goal final\n"),
    ):
        _write_bytes(root, path, payload)

    implementation_artifacts = [
        _artifact_record(root, name, path)
        for name, path in post.REQUIRED_IMPLEMENTATION_RECORDS.items()
    ]
    implementation = {
        "schema": post.IMPLEMENTATION_SCHEMA,
        "loop_id": post.LOOP_ID,
        "generated_at_utc": "2026-07-12T00:00:00Z",
        "contract": {
            "manifest_self_hashed": False,
            "output_replace_allowed": False,
        },
        "claim_scope": {
            "implementation_hash_closure_only": True,
            "raw_execution_performed": False,
            "quality_claim_allowed": False,
            "parity_claim_allowed": False,
            "certification_claim_allowed": False,
        },
        "artifacts": implementation_artifacts,
        "integrity": {
            "artifact_count": len(implementation_artifacts),
            "required_artifact_count": len(implementation_artifacts),
            "present_required_artifact_count": len(implementation_artifacts),
            "predeclared_sha256_count": 0,
            "verified_predeclared_sha256_count": 0,
            "blockers": [],
        },
        "decision": "implementation_hash_closure_verified_run_authorization_pending",
    }
    implementation_sha256 = _write_json(root, post.IMPLEMENTATION_MANIFEST, implementation)
    implementation_by_name = {row["name"]: row for row in implementation_artifacts}
    parent_evidence = {
        "post_diagnostic_manifest": "1" * 64,
        "diagnostic_receipt": "2" * 64,
        "historical_truth_manifest": "3" * 64,
    }
    run_authorization = {
        "schema": post.RUN_AUTHORIZATION_SCHEMA,
        "loop_id": post.LOOP_ID,
        "issued_at_utc": "2026-07-12T00:01:00Z",
        "prereg_authorization_sha256": implementation_by_name["authorization"]["sha256"],
        "proposal_sha256": implementation_by_name["proposal"]["sha256"],
        "parent_evidence": parent_evidence,
        "implementation_manifest": {
            "path": post.IMPLEMENTATION_MANIFEST.as_posix(),
            "sha256": implementation_sha256,
        },
        "frozen_sample": post.FIXED_SAMPLE,
        "budget": post.EXPECTED_BUDGET,
        "timeout_enforcement": post.EXPECTED_TIMEOUT_ENFORCEMENT,
        "frozen_tolerance": post.FIXED_TOLERANCE,
        "attempt_id": post.FIXED_ATTEMPT_ID,
        "attempt_lease_path": post.ATTEMPT_LEASE.as_posix(),
        "generation": "final",
        "output_path": post.FINAL_RECEIPT.as_posix(),
        "claim_scope": post.RUN_CLAIM_SCOPE,
        "decision": "allow_bounded_loop28_parity_remediation_run",
    }
    run_sha256 = _write_json(root, post.RUN_AUTHORIZATION, run_authorization)
    lease = {
        "schema": post.ATTEMPT_LEASE_SCHEMA,
        "loop_id": post.LOOP_ID,
        "attempt_id": post.FIXED_ATTEMPT_ID,
        "generation": "final",
        "run_authorization_sha256": run_sha256,
        "output_path": post.FINAL_RECEIPT.as_posix(),
        "consumed_at_utc": "2026-07-12T00:02:00Z",
        "status": "authorization_consumed_before_raw_access",
    }
    lease_sha256 = _write_json(root, post.ATTEMPT_LEASE, lease)
    receipt_run = {
        "schema": post.RUN_AUTHORIZATION_SCHEMA,
        "loop_id": post.LOOP_ID,
        "authorization_sha256": run_sha256,
        "prereg_authorization_sha256": run_authorization["prereg_authorization_sha256"],
        "proposal_sha256": run_authorization["proposal_sha256"],
        "parent_evidence_sha256": parent_evidence,
        "implementation_manifest_sha256": implementation_sha256,
        "implementation_artifact_count": len(implementation_artifacts),
        "generation": "final",
        "attempt_id": post.FIXED_ATTEMPT_ID,
        "attempt_lease_path": post.ATTEMPT_LEASE.as_posix(),
        "status": "bounded_run_authorized",
    }
    receipt = {
        "schema": post.RECEIPT_SCHEMA,
        "generated_at_utc": "2026-07-12T00:03:00Z",
        "claim_scope": post.RECEIPT_CLAIM_SCOPE,
        "generation": "final",
        "authorization": {
            "preregistration": {
                "schema": "axon_loop28_parity_remediation_authorization_v1",
                "loop_id": post.LOOP_ID,
                "authorization_sha256": run_authorization["prereg_authorization_sha256"],
                "proposal_sha256": run_authorization["proposal_sha256"],
                "parent_evidence_sha256": parent_evidence,
                "budget": post.EXPECTED_BUDGET,
                "frozen_tolerance": post.FIXED_TOLERANCE,
                "status": "authorized_contract_verified",
            },
            "run": receipt_run,
            "attempt_lease": {
                "path": post.ATTEMPT_LEASE.as_posix(),
                "sha256": lease_sha256,
                "consumed_at_utc": lease["consumed_at_utc"],
                "status": lease["status"],
            },
        },
        "sample_identity": {**post.FIXED_SAMPLE, "label": 0},
        "evidence_sha256": {
            **parent_evidence,
            "implementation_manifest": implementation_sha256,
            "split_csv": "4" * 64,
            "checkpoint": "5" * 64,
            "python_stage2": "6" * 64,
            "python_stage2_metadata": "7" * 64,
            "native_selftest": "8" * 64,
            "native_dll": "9" * 64,
            "native_onnx": "a" * 64,
            "native_stage2": "b" * 64,
        },
        "identity_audit": {
            "scope": "complete_split_metadata_identity_audit_only",
            "raw_files_opened": 1,
            "heldout_raw_files_opened": 0,
            "prediction_or_metric_rows_read": 0,
            "rows_scanned": 200_000,
            "split_metadata_counts": {"train": 20_000, "val": 20_000, "test": 160_000},
            "selected_count": 1,
            "reported_count": 1,
        },
        "budget_audit": _valid_budget_audit(),
        "authenticated_comparison": {},
        "success_gate": _valid_gate(),
        "decision": "frozen_train_remediation_gate_passed",
    }
    receipt_sha256 = _write_json(root, post.FINAL_RECEIPT, receipt)
    return {
        "implementation": implementation,
        "implementation_sha256": implementation_sha256,
        "run_authorization": run_authorization,
        "run_sha256": run_sha256,
        "lease": lease,
        "lease_sha256": lease_sha256,
        "receipt": receipt,
        "receipt_sha256": receipt_sha256,
    }


def _build_blocked_chain(root: Path) -> dict[str, object]:
    for path, payload in (
        (post.IMPLEMENTATION_BUILDER, b"implementation builder\n"),
        (post.IMPLEMENTATION_BUILDER_TEST, b"implementation builder tests\n"),
        (post.POST_BUILDER, b"post builder\n"),
        (post.POST_BUILDER_TEST, b"post builder tests\n"),
        (post.RECOMMENDATIONS, b"recommendations blocked\n"),
        (post.EXPERIMENT_JOURNAL, b"journal blocked\n"),
        (post.GOAL, b"synthetic-only ONNX fidelity localization\n"),
    ):
        _write_bytes(root, path, payload)

    verified_artifacts = {}
    for evidence_name, (_closure_name, _role, path) in post.BLOCKED_VERIFIED_ARTIFACTS.items():
        sha256 = _write_bytes(root, path, f"{evidence_name}\n".encode())
        verified_artifacts[evidence_name] = {"path": path.as_posix(), "sha256": sha256}

    fixture_results = [
        {
            "name": "pe32_numeric_resource_tls_callbacks",
            "pe_plus": False,
            "named_resource": False,
            "tls_callbacks": True,
            "base_probability_absolute_delta": post.EXPECTED_BASE_MAX_DELTA,
            "stage2_probability_absolute_delta": 0.04110092303690749,
            "stage2_mismatch_indices": [0, 1, 2, 3, 4, 5],
            "base_probability_within_tolerance": False,
            "stage2_probability_within_tolerance": False,
            "gate_passed": False,
        },
        {
            "name": "pe32_named_resource_tls_callbacks",
            "pe_plus": False,
            "named_resource": True,
            "tls_callbacks": True,
            "base_probability_absolute_delta": 0.003849865531158403,
            "stage2_probability_absolute_delta": 3.0369075032510295e-9,
            "stage2_mismatch_indices": [0, 1, 2, 3, 4, 5],
            "base_probability_within_tolerance": False,
            "stage2_probability_within_tolerance": True,
            "gate_passed": False,
        },
        {
            "name": "pe32_numeric_resource_zero_tls_callbacks",
            "pe_plus": False,
            "named_resource": False,
            "tls_callbacks": False,
            "base_probability_absolute_delta": 0.004712764657287649,
            "stage2_probability_absolute_delta": post.EXPECTED_STAGE2_MAX_DELTA,
            "stage2_mismatch_indices": [0, 1, 2, 3, 4, 5],
            "base_probability_within_tolerance": False,
            "stage2_probability_within_tolerance": False,
            "gate_passed": False,
        },
        {
            "name": "pe32_plus_named_resource_zero_tls_callbacks",
            "pe_plus": True,
            "named_resource": True,
            "tls_callbacks": False,
            "base_probability_absolute_delta": 1.289367723700252e-9,
            "stage2_probability_absolute_delta": 1.2860954523574719e-8,
            "stage2_mismatch_indices": [],
            "base_probability_within_tolerance": True,
            "stage2_probability_within_tolerance": True,
            "gate_passed": True,
        },
    ]
    gate_summary = {
        "fixture_count": 4,
        "passed_fixture_count": 1,
        "failed_fixture_count": 3,
        "pe32_passed_fixture_count": 0,
        "pe32_failed_fixture_count": 3,
        "pe32_plus_passed_fixture_count": 1,
        "base_probability_max_absolute_delta": post.EXPECTED_BASE_MAX_DELTA,
        "stage2_probability_max_absolute_delta": post.EXPECTED_STAGE2_MAX_DELTA,
        "input_dependent_runtime_drift_observed": True,
        "feature_remediation_passed": True,
        "cross_runtime_base_model_parity_passed": False,
        "one_train_sample_rerun_allowed": False,
    }
    blocked = {
        "schema": post.BLOCKED_EVIDENCE_SCHEMA,
        "loop_id": post.LOOP_ID,
        "generated_at_utc": "2026-07-12T00:00:00Z",
        "scope": {
            "synthetic_pe_only": True,
            "dataset_raw_accessed": False,
            "split_metadata_accessed": False,
            "heldout_raw_accessed": False,
            "heldout_predictions_accessed": False,
            "heldout_metrics_accessed": False,
            "training_or_fitting_performed": False,
            "f1_or_quality_metric_computed": False,
            "frozen_model_artifacts_loaded_for_inference": True,
            "implementation_manifest_generated": False,
            "run_authorization_generated": False,
            "attempt_lease_consumed": False,
            "train_remediation_run_performed": False,
        },
        "superseded_discovery": {
            "path": post.SYNTHETIC_DISCOVERY.as_posix(),
            "reason": "The repaired expanded fixture invalidated the original single-fixture pass.",
            "previous_single_fixture_parity_is_sufficient": False,
            "ort_disable_all_is_sufficient_for_cross_runtime_parity": False,
        },
        "expanded_fixture_contract": {
            "fixture_count": 4,
            "pe32_fixture_count": 3,
            "pe32_plus_fixture_count": 1,
            "named_resource_covered": True,
            "numeric_resource_covered": True,
            "tls_callbacks_present_covered": True,
            "tls_zero_callbacks_covered": True,
            "resource_tree_really_parsed": True,
            "relocation_directory_really_parsed": True,
            "invalid_rva_fail_closed_covered": True,
            "truncated_relocation_fail_closed_covered": True,
            "number_of_rva_and_sizes_covered": True,
            "numpy_pairwise_reduction_mutation_sensitive": True,
        },
        "common_authenticated_results": {
            "byte_seq_exact": True,
            "pe_features_exact": True,
            "stat_features_exact": True,
            "stage2_features_indices_6_through_1519_exact": True,
            "base_decisions_match": True,
            "final_decisions_match": True,
            "frozen_tolerance": post.FIXED_TOLERANCE,
        },
        "fixture_results": fixture_results,
        "gate_summary": gate_summary,
        "verified_artifacts": verified_artifacts,
        "verification": [
            {
                "scope": "static_and_python_synthetic",
                "command": "pytest static",
                "result": "15 passed, 9 skipped",
                "duration_seconds": 1.0,
                "model_inference_performed": False,
            },
            {
                "scope": "feature_only_native_dll",
                "command": "pytest features",
                "result": "20 passed, 4 skipped",
                "duration_seconds": 1.0,
                "model_inference_performed": False,
            },
            {
                "scope": "expanded_native_model_parity",
                "command": "pytest expanded",
                "result": "21 passed, 3 failed",
                "duration_seconds": 137.66,
                "model_inference_performed": True,
                "expected_gate_result": "failed_closed",
            },
            {
                "scope": "variant_delta_capture",
                "command": "pytest variants",
                "result": "1 passed, 2 failed, 21 deselected",
                "duration_seconds": 1.0,
                "model_inference_performed": True,
                "expected_gate_result": "failed_closed",
            },
        ],
        "failure_analysis": {
            "first_remaining_divergence_stage": "base_inference",
            "feature_extraction_is_still_the_first_divergence": False,
            "all_base_model_inputs_are_authenticated_exact": True,
            "stage2_non_probability_features_are_authenticated_exact": True,
            "remaining_stage2_feature_drift_is_limited_to_base_probability_transforms": True,
            "observed_shape_dependency": "PE32 failed and PE32+ passed.",
            "strongest_current_explanation": "The frozen runtimes are not uniformly faithful.",
            "sole_cause_claim_allowed": False,
            "next_required_evidence": ["synthetic-only intermediate activation localization"],
        },
        "claim_boundary": {
            "feature_implementation_parity_observed_on_expanded_synthetic_matrix": True,
            "base_runtime_parity_claim_allowed": False,
            "raw_remediation_claim_allowed": False,
            "population_parity_claim_allowed": False,
            "quality_claim_allowed": False,
            "native_loop28_ready_claim_allowed": False,
            "native_loop151_ready_claim_allowed": False,
            "certification_claim_allowed": False,
        },
        "decision": post.BLOCKED_DECISION,
    }
    blocked_sha256 = _write_json(root, post.BLOCKED_EVIDENCE, blocked)

    discovery = {
        "schema": post.SYNTHETIC_DISCOVERY_SCHEMA,
        "loop_id": post.LOOP_ID,
        "scope": {
            "dataset_raw_accessed": False,
            "split_metadata_accessed": False,
            "heldout_accessed": False,
            "training_or_fitting_performed": False,
            "f1_or_quality_metric_computed": False,
        },
        "supersession": {
            "status": "superseded",
            "historical_observation_retained": True,
            "resource_fixture_valid_for_parity_authorization": False,
            "single_fixture_pass_authorizes_raw_rerun": False,
            "invalidated_by": {
                "path": post.BLOCKED_EVIDENCE.as_posix(),
                "sha256": blocked_sha256,
            },
        },
        "decision": "superseded_pre_run_blocked",
    }
    discovery_sha256 = _write_json(root, post.SYNTHETIC_DISCOVERY, discovery)
    blocked_binding = {
        "path": post.BLOCKED_EVIDENCE.as_posix(),
        "sha256": blocked_sha256,
        "decision": post.BLOCKED_DECISION,
        "train_raw_rerun_allowed": False,
    }
    proposal = {
        "schema": post.PROPOSAL_SCHEMA,
        "loop_id": post.LOOP_ID,
        "synthetic_pre_run_block": blocked_binding,
        "expected_artifacts": [
            post.SYNTHETIC_DISCOVERY.as_posix(),
            post.BLOCKED_EVIDENCE.as_posix(),
            post.GOAL.as_posix(),
            post.DEFAULT_OUTPUT.as_posix(),
        ],
        "decision": post.PROPOSAL_BLOCKED_DECISION,
    }
    proposal_sha256 = _write_json(root, post.PROPOSAL, proposal)
    authorization = {
        "schema": post.AUTHORIZATION_SCHEMA,
        "loop_id": post.LOOP_ID,
        "proposal": {"path": post.PROPOSAL.as_posix(), "sha256": proposal_sha256},
        "synthetic_pre_run_block": blocked_binding,
        "allowed_splits": [],
        "authorized_generated_paths": [post.DEFAULT_OUTPUT.as_posix()],
        "execution_requires_separate_run_authorization": False,
        "decision": post.AUTHORIZATION_BLOCKED_DECISION,
    }
    authorization_sha256 = _write_json(root, post.AUTHORIZATION, authorization)
    preflight = {
        "schema": post.PREFLIGHT_SCHEMA,
        "loop_id": post.LOOP_ID,
        "governance_binding": {
            "proposal": {"path": post.PROPOSAL.as_posix(), "sha256": proposal_sha256},
            "authorization": {
                "path": post.AUTHORIZATION.as_posix(),
                "sha256": authorization_sha256,
            },
        },
        "synthetic_pre_run_block": blocked_binding,
        "pre_implementation_checks": {
            "implementation_manifest_present": False,
            "new_run_authorization_present": False,
            "new_attempt_lease_present": False,
            "remediation_receipt_present": False,
            "raw_accessed": False,
            "heldout_accessed": False,
        },
        "decision": post.PREFLIGHT_BLOCKED_DECISION,
    }
    preflight_sha256 = _write_json(root, post.PREFLIGHT, preflight)
    return {
        "blocked": blocked,
        "blocked_sha256": blocked_sha256,
        "discovery": discovery,
        "discovery_sha256": discovery_sha256,
        "proposal_sha256": proposal_sha256,
        "authorization_sha256": authorization_sha256,
        "preflight_sha256": preflight_sha256,
    }


def test_builds_structured_post_remediation_closure(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    manifest = post.build_post_remediation_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:04:00Z",
    )

    assert manifest["integrity"] == {
        "artifact_count": 6,
        "required_artifact_count": 6,
        "present_required_artifact_count": 6,
        "structured_chain_links_verified": 4,
        "blockers": [],
    }
    assert manifest["lineage"]["implementation_manifest_sha256"] == chain["implementation_sha256"]
    assert manifest["lineage"]["run_authorization_sha256"] == chain["run_sha256"]
    assert manifest["lineage"]["consumed_lease_sha256"] == chain["lease_sha256"]
    assert manifest["lineage"]["final_receipt_sha256"] == chain["receipt_sha256"]
    assert post.DEFAULT_OUTPUT.as_posix() not in {row["path"] for row in manifest["artifacts"]}


def test_build_and_verify_then_detect_document_drift(tmp_path: Path) -> None:
    _build_chain(tmp_path)
    manifest = post.build_post_remediation_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:04:00Z",
    )
    output = post.resolve_fixed_output(tmp_path, post.DEFAULT_OUTPUT)
    post._write_exclusive(output, manifest)
    assert post.verify_post_remediation_manifest(tmp_path, post.DEFAULT_OUTPUT) == manifest

    (tmp_path / post.RECOMMENDATIONS).write_text("drift\n", encoding="utf-8")
    with pytest.raises(post.PostRemediationManifestError, match="no longer matches"):
        post.verify_post_remediation_manifest(tmp_path, post.DEFAULT_OUTPUT)


def test_output_is_exclusive_fixed_and_confined(tmp_path: Path) -> None:
    _build_chain(tmp_path)
    manifest = post.build_post_remediation_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:04:00Z",
    )
    output = post.resolve_fixed_output(tmp_path, post.DEFAULT_OUTPUT)
    post._write_exclusive(output, manifest)
    with pytest.raises(post.PostRemediationManifestError, match="already exists"):
        post._write_exclusive(output, manifest)
    with pytest.raises(post.PostRemediationManifestError, match="not fixed"):
        post.resolve_fixed_output(tmp_path, Path("other.json"))
    with pytest.raises(post.PostRemediationManifestError, match="project-relative"):
        post.resolve_fixed_output(tmp_path, Path("../escape.json"))
    with pytest.raises(post.PostRemediationManifestError, match="project-relative"):
        post.resolve_fixed_output(tmp_path, tmp_path.parent / "escape.json")


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    _build_chain(tmp_path)
    path = tmp_path / post.RUN_AUTHORIZATION
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("{\n", '{\n  "schema": "duplicate",\n', 1), encoding="utf-8")
    with pytest.raises(post.PostRemediationManifestError, match="Duplicate JSON key"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


def test_schema_mismatch_is_rejected(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    run_authorization = chain["run_authorization"]
    assert isinstance(run_authorization, dict)
    run_authorization["schema"] = "wrong"
    _write_json(tmp_path, post.RUN_AUTHORIZATION, run_authorization)
    with pytest.raises(post.PostRemediationManifestError, match="schema mismatch"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


def test_run_authorization_must_bind_implementation(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    run_authorization = chain["run_authorization"]
    assert isinstance(run_authorization, dict)
    run_authorization["implementation_manifest"]["sha256"] = "f" * 64
    _write_json(tmp_path, post.RUN_AUTHORIZATION, run_authorization)
    with pytest.raises(post.PostRemediationManifestError, match="does not bind"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


def test_consumed_lease_must_bind_run_authorization(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    lease = chain["lease"]
    assert isinstance(lease, dict)
    lease["run_authorization_sha256"] = "f" * 64
    _write_json(tmp_path, post.ATTEMPT_LEASE, lease)
    with pytest.raises(post.PostRemediationManifestError, match="lease binding drifted"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


def test_receipt_must_bind_consumed_lease(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    receipt = chain["receipt"]
    assert isinstance(receipt, dict)
    receipt["authorization"]["attempt_lease"]["sha256"] = "f" * 64
    _write_json(tmp_path, post.FINAL_RECEIPT, receipt)
    with pytest.raises(post.PostRemediationManifestError, match="lease binding drifted"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


def test_failed_gate_cannot_claim_successful_closure(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    receipt = chain["receipt"]
    assert isinstance(receipt, dict)
    receipt["success_gate"]["passed"] = False
    receipt["success_gate"]["checks"]["pe_features"] = False
    _write_json(tmp_path, post.FINAL_RECEIPT, receipt)
    with pytest.raises(post.PostRemediationManifestError, match="checks did not all pass"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


def test_receipt_decision_must_match_gate(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    receipt = chain["receipt"]
    assert isinstance(receipt, dict)
    receipt["decision"] = "frozen_train_remediation_gate_failed"
    _write_json(tmp_path, post.FINAL_RECEIPT, receipt)
    with pytest.raises(post.PostRemediationManifestError, match="decision does not match"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


@pytest.mark.parametrize("forbidden_path", [post.DEFAULT_OUTPUT, post.GOAL])
def test_implementation_manifest_cannot_bind_post_run_outputs(
    tmp_path: Path,
    forbidden_path: Path,
) -> None:
    chain = _build_chain(tmp_path)
    implementation = chain["implementation"]
    assert isinstance(implementation, dict)
    _write_bytes(tmp_path, forbidden_path, b"forbidden\n")
    implementation["artifacts"].append(
        _artifact_record(tmp_path, "forbidden_post_run_artifact", forbidden_path)
    )
    count = len(implementation["artifacts"])
    implementation["integrity"].update(
        {
            "artifact_count": count,
            "required_artifact_count": count,
            "present_required_artifact_count": count,
        }
    )
    _write_json(tmp_path, post.IMPLEMENTATION_MANIFEST, implementation)
    with pytest.raises(post.PostRemediationManifestError, match="cyclic output"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


def test_implementation_manifest_requires_post_builder_and_tests(tmp_path: Path) -> None:
    chain = _build_chain(tmp_path)
    implementation = chain["implementation"]
    assert isinstance(implementation, dict)
    implementation["artifacts"] = [
        row
        for row in implementation["artifacts"]
        if row["name"] != "post_remediation_builder_tests"
    ]
    count = len(implementation["artifacts"])
    implementation["integrity"].update(
        {
            "artifact_count": count,
            "required_artifact_count": count,
            "present_required_artifact_count": count,
        }
    )
    _write_json(tmp_path, post.IMPLEMENTATION_MANIFEST, implementation)
    with pytest.raises(post.PostRemediationManifestError, match="binding is missing"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


def test_builds_synthetic_pre_run_blocked_closure(tmp_path: Path) -> None:
    chain = _build_blocked_chain(tmp_path)
    manifest = post.build_post_remediation_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:05:00Z",
    )

    assert manifest["decision"] == post.BLOCKED_CLOSURE_DECISION
    assert manifest["claim_scope"] == {
        "remediation_gate_passed": False,
        "train_raw_execution_performed": False,
        "implementation_manifest_absent_by_contract": True,
        "run_authorization_absent_by_contract": True,
        "attempt_lease_absent_by_contract": True,
        "remediation_receipt_absent_by_contract": True,
        "quality_claim_allowed": False,
        "population_parity_claim_allowed": False,
        "native_loop28_ready": False,
        "native_loop151_ready": False,
        "certification_claim_allowed": False,
    }
    assert manifest["lineage"]["synthetic_pre_run_blocked_sha256"] == chain["blocked_sha256"]
    assert manifest["lineage"]["synthetic_discovery_sha256"] == chain["discovery_sha256"]
    assert manifest["integrity"] == {
        "artifact_count": 19,
        "required_artifact_count": 19,
        "present_required_artifact_count": 19,
        "structured_chain_links_verified": 12,
        "run_chain_artifact_count": 0,
        "blockers": [],
    }
    artifact_paths = {record["path"] for record in manifest["artifacts"]}
    assert post.GOAL.as_posix() in artifact_paths
    assert not artifact_paths.intersection(path.as_posix() for path in post.RUN_CHAIN_PATHS)
    assert not (tmp_path / post.DEFAULT_OUTPUT).exists()


@pytest.mark.parametrize(
    "present_paths",
    [
        (post.IMPLEMENTATION_MANIFEST,),
        (post.RUN_AUTHORIZATION,),
        (post.ATTEMPT_LEASE,),
        (post.FINAL_RECEIPT,),
        (post.IMPLEMENTATION_MANIFEST, post.RUN_AUTHORIZATION),
    ],
)
def test_blocked_closure_rejects_any_or_mixed_run_chain_presence(
    tmp_path: Path,
    present_paths: tuple[Path, ...],
) -> None:
    _build_blocked_chain(tmp_path)
    for path in present_paths:
        _write_bytes(tmp_path, path, b"forbidden run chain\n")
    with pytest.raises(post.PostRemediationManifestError, match="cannot coexist"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:05:00Z",
        )


def test_missing_blocked_evidence_without_success_chain_is_rejected(tmp_path: Path) -> None:
    _build_blocked_chain(tmp_path)
    (tmp_path / post.BLOCKED_EVIDENCE).unlink()
    with pytest.raises(post.PostRemediationManifestError, match="Blocked evidence is missing"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:05:00Z",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("failed_fixture_count", 2),
        ("base_probability_max_absolute_delta", 0.0),
        ("stage2_probability_max_absolute_delta", 0.0),
    ],
)
def test_blocked_gate_count_and_delta_drift_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    chain = _build_blocked_chain(tmp_path)
    blocked = chain["blocked"]
    assert isinstance(blocked, dict)
    blocked["gate_summary"][field] = value
    _write_json(tmp_path, post.BLOCKED_EVIDENCE, blocked)
    with pytest.raises(post.PostRemediationManifestError, match="gate counts or deltas"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:05:00Z",
        )


def test_blocked_fixture_identity_fields_are_frozen(tmp_path: Path) -> None:
    chain = _build_blocked_chain(tmp_path)
    blocked = chain["blocked"]
    assert isinstance(blocked, dict)
    blocked["fixture_results"][0]["named_resource"] = True
    _write_json(tmp_path, post.BLOCKED_EVIDENCE, blocked)
    with pytest.raises(post.PostRemediationManifestError, match="identity fields drifted"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:05:00Z",
        )


def test_blocked_claim_boundary_drift_is_rejected(tmp_path: Path) -> None:
    chain = _build_blocked_chain(tmp_path)
    blocked = chain["blocked"]
    assert isinstance(blocked, dict)
    blocked["claim_boundary"]["quality_claim_allowed"] = True
    _write_json(tmp_path, post.BLOCKED_EVIDENCE, blocked)
    with pytest.raises(post.PostRemediationManifestError, match="claim boundary drifted"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:05:00Z",
        )


def test_blocked_verified_artifact_hash_drift_is_rejected(tmp_path: Path) -> None:
    _build_blocked_chain(tmp_path)
    _write_bytes(tmp_path, post.NATIVE_SOURCE, b"drifted native source\n")
    with pytest.raises(post.PostRemediationManifestError, match="artifact hash mismatch"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:05:00Z",
        )


def test_blocked_evidence_hash_must_be_bound_by_discovery(tmp_path: Path) -> None:
    chain = _build_blocked_chain(tmp_path)
    blocked = chain["blocked"]
    assert isinstance(blocked, dict)
    blocked["generated_at_utc"] = "2026-07-12T00:00:01Z"
    _write_json(tmp_path, post.BLOCKED_EVIDENCE, blocked)
    with pytest.raises(post.PostRemediationManifestError, match="supersession binding drifted"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:05:00Z",
        )


def test_blocked_governance_hash_drift_is_rejected(tmp_path: Path) -> None:
    _build_blocked_chain(tmp_path)
    authorization_path = tmp_path / post.AUTHORIZATION
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["proposal"]["sha256"] = "f" * 64
    _write_json(tmp_path, post.AUTHORIZATION, authorization)
    with pytest.raises(post.PostRemediationManifestError, match="authorization contract drifted"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:05:00Z",
        )


@pytest.mark.parametrize("document", [post.RECOMMENDATIONS, post.EXPERIMENT_JOURNAL, post.GOAL])
def test_blocked_closure_verification_detects_final_document_drift(
    tmp_path: Path,
    document: Path,
) -> None:
    _build_blocked_chain(tmp_path)
    manifest = post.build_post_remediation_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:05:00Z",
    )
    post._write_exclusive(post.resolve_fixed_output(tmp_path, post.DEFAULT_OUTPUT), manifest)
    assert post.verify_post_remediation_manifest(tmp_path, post.DEFAULT_OUTPUT) == manifest

    _write_bytes(tmp_path, document, b"document drift\n")
    with pytest.raises(post.PostRemediationManifestError, match="no longer matches"):
        post.verify_post_remediation_manifest(tmp_path, post.DEFAULT_OUTPUT)


def test_duplicate_keys_in_blocked_evidence_are_rejected(tmp_path: Path) -> None:
    _build_blocked_chain(tmp_path)
    path = tmp_path / post.BLOCKED_EVIDENCE
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("{\n", '{\n  "schema": "duplicate",\n', 1), encoding="utf-8")
    with pytest.raises(post.PostRemediationManifestError, match="Duplicate JSON key"):
        post.build_post_remediation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:05:00Z",
        )
