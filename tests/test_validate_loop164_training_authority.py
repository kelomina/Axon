from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
TESTS_DIR = Path(__file__).resolve().parent
for directory in (SCRIPTS_DIR, TESTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from loop164_whole_file_contract_fixture import (  # noqa: E402
    create_whole_file_implementation_fixture,
)
from validate_loop164_training_authority import (  # noqa: E402
    FEATURE_CONTRACT_SCHEMA,
    FORBIDDEN_SPLIT_ROLES,
    IMPLEMENTATION_BINDING_PHASE,
    INPUT_BUNDLE_SCHEMA,
    ISOLATION_AUTHORIZATION_PROVENANCE_SCHEMA,
    ISOLATION_CONTRACT_SCHEMA,
    ISOLATION_METADATA_AUTHORITY_SCOPE,
    ISOLATION_RECEIPT_SCHEMA,
    LOOP_ID,
    REQUIRED_FUSION_FIELDS,
    RESOURCE_GUARD_SCHEMA,
    SCOPE_PLAN_VALIDATION_SCHEMA,
    TRAINING_AUTHORIZATION_SCHEMA,
    TRUST_ANCHOR_SCHEMA,
    consume_training_final_lease,
    default_training_authority_paths,
    sha256_file,
    validate_training_authority,
    verify_consumed_training_final_lease,
)
from validate_loop164_whole_file_implementation import sha256_json  # noqa: E402

NOW_UTC = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def binding(path: Path, root: Path) -> dict[str, str]:
    return {"path": relative(path, root), "sha256": sha256_file(path)}


def refresh_binding(authorization_path: Path, name: str, path: Path, root: Path) -> None:
    authorization = read_json(authorization_path)
    authorization["bindings"][name] = binding(path, root)
    write_json(authorization_path, authorization)


def create_case(tmp_path: Path) -> tuple[dict[str, Path], list[str]]:
    root = tmp_path / "project"
    root.mkdir()
    paths = default_training_authority_paths(root)
    implementation_payload = create_whole_file_implementation_fixture(
        root, paths.implementation_manifest
    )
    fold_fingerprint = digest("fold-assignment")
    argv = ["--fixed-controller-mode", "nested-oof"]
    write_json(
        paths.proposal,
        {
            "loop_id": LOOP_ID,
            "decision": "propose_loop164_whole_file_residual_expert_no_execution",
        },
    )
    write_json(
        paths.contract,
        {
            "loop_id": LOOP_ID,
            "schema": ISOLATION_CONTRACT_SCHEMA,
            "model_input_fields": list(REQUIRED_FUSION_FIELDS),
            "feature_contract": {
                "schema": FEATURE_CONTRACT_SCHEMA,
                "feature_fields": list(REQUIRED_FUSION_FIELDS),
                "feature_matrix_receipt_required": True,
                "implementation_binding_phase": IMPLEMENTATION_BINDING_PHASE,
            },
        },
    )
    write_json(
        paths.isolation_receipt,
        {
            "loop_id": LOOP_ID,
            "schema": ISOLATION_RECEIPT_SCHEMA,
            "decision": "pass",
            "ready_for": {"loop164_train_oof_partition": True},
            "a2_authorization_provenance": {
                "schema": ISOLATION_AUTHORIZATION_PROVENANCE_SCHEMA,
                "authority_scope": ISOLATION_METADATA_AUTHORITY_SCOPE,
                "authorization_sha256": digest("metadata-authorization"),
                "trust_anchor_sha256": digest("metadata-trust-anchor"),
                "trusted_key_fingerprint": digest("metadata-trusted-key"),
                "verification_receipt_sha256": digest("metadata-verification"),
                "validator_source_closure_sha256": digest("metadata-source-closure"),
                "runtime_python_sha256": digest("metadata-runtime"),
                "resource_guard_sha256": digest("metadata-resource-guard"),
                "canonical_argv_sha256": digest("metadata-argv"),
                "lease_consumption_id": digest("metadata-lease"),
                "lease_marker_sha256": digest("metadata-marker"),
            },
        },
    )
    write_json(
        paths.scope_plan,
        {"loop_id": LOOP_ID, "fold_assignment_fingerprint": fold_fingerprint},
    )
    write_json(
        paths.scope_plan_validation,
        {
            "schema": SCOPE_PLAN_VALIDATION_SCHEMA,
            "loop_id": LOOP_ID,
            "aggregate_only_verified": True,
            "proposal_binding_verified": True,
            "contract_binding_verified": True,
            "isolation_receipt_binding_verified": True,
            "scope_plan_binding_verified": True,
            "binding_fingerprints": {
                "proposal_sha256": sha256_file(paths.proposal),
                "contract_sha256": sha256_file(paths.contract),
                "isolation_receipt_sha256": sha256_file(paths.isolation_receipt),
                "scope_plan_sha256": sha256_file(paths.scope_plan),
            },
            "plan_summary": {"fold_assignment_fingerprint": fold_fingerprint},
            "blockers": [],
            "ready_for": {
                "fold_scope_frozen": True,
                "a2_training_authorization": False,
                "train_oof": False,
            },
            "decision": "pass",
            "notes": ["synthetic aggregate-only fixture"],
        },
    )
    write_json(paths.loop151_train_oof_manifest, {"loop_id": LOOP_ID, "schema": "synthetic_loop151"})
    write_json(
        paths.input_bundle,
        {
            "schema": INPUT_BUNDLE_SCHEMA,
            "loop_id": LOOP_ID,
            "allowed_split_roles": ["train_anchor", "train_oof"],
            "forbidden_split_roles": list(FORBIDDEN_SPLIT_ROLES),
            "feature_fields": list(REQUIRED_FUSION_FIELDS),
            "fold_assignment_fingerprint": fold_fingerprint,
            "scope_plan_validation_sha256": sha256_file(paths.scope_plan_validation),
            "protected_input_open_policy": "after_final_lease_only",
            "input_artifact_commitments": {
                "train_anchor_sha256": digest("train-anchor"),
                "train_oof_sha256": digest("train-oof"),
            },
        },
    )
    runtime_python = Path(sys.executable).resolve()
    argv_sha256 = hashlib.sha256(
        json.dumps(argv, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json(
        paths.resource_guard,
        {
            "schema": RESOURCE_GUARD_SCHEMA,
            "loop_id": LOOP_ID,
            "operation": "loop164_three_seed_nested_train_oof",
            "guard_ready": True,
            "decision": "pass",
            "runtime_binding": {
                "cwd": str(root.resolve()),
                "python_sha256": sha256_file(runtime_python),
                "controller_path": relative(paths.controller, root),
                "controller_sha256": sha256_file(paths.controller),
                "canonical_argv_sha256": argv_sha256,
            },
            "implementation_binding": {
                "implementation_manifest_sha256": sha256_file(paths.implementation_manifest),
                "source_closure_sha256": implementation_payload["source_closure"]["closure_sha256"],
                "memory_contract_sha256": sha256_json(implementation_payload["memory_contract"]),
            },
            "receipt": {
                "created_at_utc": "2026-07-12T11:55:00Z",
                "controller_sha256": sha256_file(paths.controller),
                "resource_budget_sha256": digest("resource-budget"),
            },
        },
    )
    trust_anchor = tmp_path / "external_custody" / "trust_anchor.json"
    trusted_key = digest("external-custody-key")
    verification_receipt = digest("external-verification-receipt")
    write_json(
        trust_anchor,
        {
            "schema": TRUST_ANCHOR_SCHEMA,
            "loop_id": LOOP_ID,
            "trusted_key_fingerprint": trusted_key,
            "verification_receipt_sha256": verification_receipt,
            "root_state": "externally_verified",
        },
    )
    bound_paths = {
        "proposal": paths.proposal,
        "contract": paths.contract,
        "isolation_receipt": paths.isolation_receipt,
        "scope_plan": paths.scope_plan,
        "scope_plan_validation": paths.scope_plan_validation,
        "implementation_manifest": paths.implementation_manifest,
        "loop151_train_oof_manifest": paths.loop151_train_oof_manifest,
        "resource_guard": paths.resource_guard,
        "input_bundle": paths.input_bundle,
    }
    write_json(
        paths.authorization,
        {
            "schema": TRAINING_AUTHORIZATION_SCHEMA,
            "loop_id": LOOP_ID,
            "authorization_level": "A2_train_only_nested_oof",
            "decision": "allow_single_loop164_train_oof_execution",
            "execution_environment": "custodian_side_train_only",
            "operation": "loop164_three_seed_nested_train_oof",
            "issued_at_utc": "2026-07-12T11:00:00Z",
            "not_before_utc": "2026-07-12T11:30:00Z",
            "expires_at_utc": "2026-07-12T13:00:00Z",
            "authority_attestation": {
                "trusted_key_fingerprint": trusted_key,
                "trust_anchor_sha256": sha256_file(trust_anchor),
                "verification_receipt_sha256": verification_receipt,
            },
            "runtime_binding": {
                "cwd": str(root.resolve()),
                "python_executable": str(runtime_python),
                "python_sha256": sha256_file(runtime_python),
                "controller_path": relative(paths.controller, root),
                "controller_sha256": sha256_file(paths.controller),
                "entrypoint": "run_loop164_train_oof_controller.main",
            },
            "canonical_argv": argv,
            "allowed_split_roles": ["train_anchor", "train_oof"],
            "forbidden_split_roles": list(FORBIDDEN_SPLIT_ROLES),
            "feature_fields": list(REQUIRED_FUSION_FIELDS),
            "fold_assignment_fingerprint": fold_fingerprint,
            "outer_run_budget": 15,
            "bindings": {name: binding(path, root) for name, path in bound_paths.items()},
            "output_binding": {
                "execution_receipt_path": relative(paths.execution_receipt, root),
                "final_lease_path": relative(paths.final_lease, root),
                "lease_marker_directory": relative(paths.lease_marker_directory, root),
            },
            "one_shot_lease": {
                "lease_id": "synthetic-a2-training-lease",
                "purpose": "single_loop164_three_seed_nested_train_oof",
                "state": "ready",
            },
            "max_resource_guard_age_seconds": 600,
        },
    )
    return {
        "root": root,
        "authorization": paths.authorization,
        "trust_anchor": trust_anchor,
        "scope_validation": paths.scope_plan_validation,
        "input_bundle": paths.input_bundle,
        "execution_receipt": paths.execution_receipt,
        "trusted_key": trusted_key,
    }, argv


def validate_case(case: dict[str, Path], argv: list[str]):
    return validate_training_authority(
        trust_anchor_json=case["trust_anchor"],
        expected_trusted_key_fingerprint=str(case["trusted_key"]),
        paths=default_training_authority_paths(case["root"]),
        actual_argv=argv,
        now_utc=NOW_UTC,
    )


def test_training_authority_accepts_synthetic_aggregate_chain_without_input_open(tmp_path: Path):
    case, argv = create_case(tmp_path)

    result = validate_case(case, argv)

    assert result.ready is True
    assert result.context is not None
    assert result.context.paths.input_bundle == case["root"] / (
        "reports/roadmap_9997/loop164/train_oof_input_bundle_manifest.json"
    )
    assert not case["execution_receipt"].exists()


def test_training_authority_blocks_scope_validation_and_input_bundle_drift(tmp_path: Path):
    case, argv = create_case(tmp_path)
    scope_validation = read_json(case["scope_validation"])
    scope_validation["ready_for"]["fold_scope_frozen"] = False
    write_json(case["scope_validation"], scope_validation)
    input_bundle = read_json(case["input_bundle"])
    input_bundle["scope_plan_validation_sha256"] = sha256_file(case["scope_validation"])
    input_bundle["allowed_split_roles"] = ["train_oof"]
    write_json(case["input_bundle"], input_bundle)
    refresh_binding(case["authorization"], "scope_plan_validation", case["scope_validation"], case["root"])
    refresh_binding(case["authorization"], "input_bundle", case["input_bundle"], case["root"])

    result = validate_case(case, argv)

    assert result.ready is False
    assert "scope_plan_validation_ready_state_invalid" in result.blockers
    assert "training_input_bundle_split_roles_invalid" in result.blockers


def test_training_authority_blocks_runtime_argv_and_external_trust_anchor_drift(tmp_path: Path):
    case, argv = create_case(tmp_path)
    trust_anchor = read_json(case["trust_anchor"])
    trust_anchor["trusted_key_fingerprint"] = digest("untrusted-key")
    write_json(case["trust_anchor"], trust_anchor)

    result = validate_case(case, [*argv, "--drift"])

    assert result.ready is False
    assert "training_authorization_canonical_argv_mismatch" in result.blockers
    assert "training_authorization_attestation_trusted_key_fingerprint_mismatch" in result.blockers


def test_training_authority_rejects_unattested_isolation_receipt(tmp_path: Path):
    case, argv = create_case(tmp_path)
    isolation_receipt = case["root"] / "reports/roadmap_9997/loop164/full_pool_isolation_validation.json"
    payload = read_json(isolation_receipt)
    payload.pop("a2_authorization_provenance")
    write_json(isolation_receipt, payload)
    refresh_binding(case["authorization"], "isolation_receipt", isolation_receipt, case["root"])

    result = validate_case(case, argv)

    assert result.ready is False
    assert "training_authorization_isolation_receipt_provenance_not_object" in result.blockers


def test_training_authority_rejects_isolation_receipt_with_training_scope(tmp_path: Path):
    case, argv = create_case(tmp_path)
    isolation_receipt = case["root"] / "reports/roadmap_9997/loop164/full_pool_isolation_validation.json"
    payload = read_json(isolation_receipt)
    payload["a2_authorization_provenance"]["authority_scope"]["grants"] = ["train_oof"]
    write_json(isolation_receipt, payload)
    refresh_binding(case["authorization"], "isolation_receipt", isolation_receipt, case["root"])

    result = validate_case(case, argv)

    assert result.ready is False
    assert "training_authorization_isolation_receipt_provenance_invalid" in result.blockers


def test_training_authority_rejects_legacy_metadata_implementation_placeholder(tmp_path: Path):
    case, argv = create_case(tmp_path)
    contract_path = case["root"] / "reports/roadmap_9997/loop164/full_pool_group_manifest.json"
    contract = read_json(contract_path)
    contract["feature_contract"] = {
        "schema": "axon_loop164_residual_fusion_feature_contract_v1",
        "implementation_manifest_sha256": digest("legacy-placeholder"),
        "feature_matrix_receipt_required": True,
    }
    write_json(contract_path, contract)
    refresh_binding(case["authorization"], "contract", contract_path, case["root"])

    result = validate_case(case, argv)

    assert result.ready is False
    assert "training_authorization_contract_feature_contract_shape_invalid" in result.blockers
    assert "training_authorization_contract_implementation_binding_phase_invalid" in result.blockers


def test_training_authority_rejects_v1_or_unreviewed_implementation_manifest(tmp_path: Path):
    case, argv = create_case(tmp_path)
    implementation_path = case["root"] / (
        "reports/roadmap_9997/loop164/whole_file_expert_implementation_manifest.json"
    )
    write_json(implementation_path, {"loop_id": LOOP_ID, "schema": "synthetic_implementation"})
    refresh_binding(case["authorization"], "implementation_manifest", implementation_path, case["root"])

    result = validate_case(case, argv)

    assert result.ready is False
    assert "training_implementation_manifest_missing_fields" in result.blockers


def test_training_authority_rejects_resource_guard_implementation_drift(tmp_path: Path):
    case, argv = create_case(tmp_path)
    resource_guard_path = case["root"] / "reports/roadmap_9997/loop164/resource_guard.json"
    resource_guard = read_json(resource_guard_path)
    resource_guard["implementation_binding"]["memory_contract_sha256"] = digest("different-memory")
    write_json(resource_guard_path, resource_guard)
    refresh_binding(case["authorization"], "resource_guard", resource_guard_path, case["root"])

    result = validate_case(case, argv)

    assert result.ready is False
    assert "training_resource_guard_implementation_binding_mismatch" in result.blockers


def test_training_lease_is_immutable_and_replay_is_blocked(tmp_path: Path):
    case, argv = create_case(tmp_path)
    authorization_before = case["authorization"].read_bytes()
    authority = validate_case(case, argv)

    assert authority.ready is True
    assert authority.context is not None
    consumed = consume_training_final_lease(
        authority.context, consumed_at_utc=NOW_UTC, actual_argv=argv
    )
    verified = verify_consumed_training_final_lease(authority.context)
    replay = consume_training_final_lease(
        authority.context, consumed_at_utc=NOW_UTC, actual_argv=argv
    )

    assert consumed.consumed is True
    assert verified.consumed is True
    assert replay.consumed is False
    assert "training_authorization_lease_already_consumed" in replay.blockers
    assert case["authorization"].read_bytes() == authorization_before


def test_training_lease_refuses_binding_drift_before_protected_open(tmp_path: Path):
    case, argv = create_case(tmp_path)
    authority = validate_case(case, argv)

    assert authority.ready is True
    assert authority.context is not None
    scope_validation = read_json(case["scope_validation"])
    scope_validation["notes"].append("post-validation drift")
    write_json(case["scope_validation"], scope_validation)
    consumed = consume_training_final_lease(
        authority.context, consumed_at_utc=NOW_UTC, actual_argv=argv
    )

    assert consumed.consumed is False
    assert "training_authorization_binding_changed_before_lease" in consumed.blockers


def test_training_lease_refuses_whole_file_source_closure_drift_before_protected_open(tmp_path: Path):
    case, argv = create_case(tmp_path)
    authority = validate_case(case, argv)
    assert authority.ready is True
    assert authority.context is not None
    model_path = case["root"] / "src/loop164/whole_file_gcg.py"
    model_path.write_text(model_path.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")

    consumed = consume_training_final_lease(
        authority.context, consumed_at_utc=NOW_UTC, actual_argv=argv
    )

    assert consumed.consumed is False
    assert "training_authorization_implementation_closure_changed_before_lease" in consumed.blockers
    assert not authority.context.marker_path.exists()


def test_training_authority_refuses_existing_execution_receipt_before_lease(tmp_path: Path):
    case, argv = create_case(tmp_path)
    write_json(case["execution_receipt"], {"synthetic": "existing output"})

    result = validate_case(case, argv)

    assert result.ready is False
    assert "training_authorization_output_already_exists" in result.blockers
