from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
TESTS_DIR = Path(__file__).resolve().parent
for directory in (SCRIPTS_DIR, TESTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from loop164_whole_file_contract_fixture import (  # noqa: E402
    create_whole_file_implementation_fixture,
)
from validate_loop164_nested_oof_execution_receipt import (  # noqa: E402
    FEATURE_CONTRACT_SCHEMA,
    IMPLEMENTATION_BINDING_PHASE,
    ISOLATION_AUTHORIZATION_PROVENANCE_SCHEMA,
    ISOLATION_CONTRACT_SCHEMA,
    ISOLATION_METADATA_AUTHORITY_SCOPE,
    ISOLATION_RECEIPT_SCHEMA,
    LOOP151_TRAIN_OOF_MANIFEST_SCHEMA,
    LOOP_ID,
    RECEIPT_SCHEMA,
    REQUIRED_EMBARGO_SECONDS,
    REQUIRED_FUSION_FIELDS,
    REQUIRED_SEEDS,
    SCOPE_PLAN_SCHEMA,
    TRAINING_AUTHORIZATION_SCHEMA,
    TRAINING_LEASE_SCHEMA,
    validate_loop164_nested_oof_execution_receipt,
)
from validate_loop164_training_authority import (  # noqa: E402
    INPUT_BUNDLE_SCHEMA,
    RESOURCE_GUARD_SCHEMA,
    SCOPE_PLAN_VALIDATION_SCHEMA,
    TRAINING_LEASE_MARKER_SCHEMA,
    build_lease_consumption_id,
)
from validate_loop164_whole_file_implementation import sha256_json  # noqa: E402


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_argv_sha256(argv: list[str]) -> str:
    encoded = json.dumps(argv, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def label_counts(rows: int) -> dict[str, int]:
    return {"0": rows // 2, "1": rows - rows // 2}


def build_scope_plan(contract_sha256: str, isolation_sha256: str) -> tuple[dict, dict[int, dict]]:
    fingerprint = digest("frozen-fold-assignment")
    scopes: list[dict] = []
    by_fold: dict[int, dict] = {}
    outer_windows = (
        ("2025-03-01T00:00:00Z", "2025-03-02T00:00:00Z"),
        ("2025-04-02T00:00:00Z", "2025-04-03T00:00:00Z"),
        ("2025-05-04T00:00:00Z", "2025-05-05T00:00:00Z"),
        ("2025-06-05T00:00:00Z", "2025-06-06T00:00:00Z"),
        ("2025-07-07T00:00:00Z", "2025-07-08T00:00:00Z"),
    )
    for fold_id, (outer_minimum, outer_maximum) in enumerate(outer_windows):
        inner_scopes = []
        for inner_fold in range(5):
            inner_scopes.append(
                {
                    "inner_fold": inner_fold,
                    "fit_scope_commitment": digest(f"outer-{fold_id}-inner-{inner_fold}-fit"),
                    "holdout_scope_commitment": digest(f"outer-{fold_id}-inner-{inner_fold}-holdout"),
                    "parent_outer_fit_scope_commitment": digest(f"outer-{fold_id}-fit"),
                    "fit_component_set_commitment": digest(
                        f"outer-{fold_id}-inner-{inner_fold}-fit-components"
                    ),
                    "holdout_component_set_commitment": digest(
                        f"outer-{fold_id}-inner-{inner_fold}-holdout-components"
                    ),
                    "fit_rows": 800,
                    "holdout_rows": 200,
                    "fit_label_counts": label_counts(800),
                    "holdout_label_counts": label_counts(200),
                    "fit_max_component_time_utc": "2025-01-01T00:00:00Z",
                    "holdout_min_component_time_utc": "2025-02-01T00:00:00Z",
                    "holdout_max_component_time_utc": "2025-02-02T00:00:00Z",
                    "overlap_audit": {
                        "fit_holdout_row_overlap": 0,
                        "fit_holdout_component_overlap": 0,
                        "outer_holdout_component_overlap": 0,
                        "inner_fit_outside_parent_outer_fit_components": 0,
                        "inner_holdout_outside_parent_outer_fit_components": 0,
                    },
                }
            )
        scope = {
            "outer_fold": fold_id,
            "fit_scope_commitment": digest(f"outer-{fold_id}-fit"),
            "holdout_scope_commitment": digest(f"outer-{fold_id}-holdout"),
            "fit_component_set_commitment": digest(f"outer-{fold_id}-fit-components"),
            "holdout_component_set_commitment": digest(f"outer-{fold_id}-holdout-components"),
            "fit_rows": 1000,
            "holdout_rows": 200,
            "fit_label_counts": label_counts(1000),
            "holdout_label_counts": label_counts(200),
            "fit_max_component_time_utc": "2025-01-01T00:00:00Z",
            "holdout_min_component_time_utc": outer_minimum,
            "holdout_max_component_time_utc": outer_maximum,
            "inner_oof_meta_eligible_rows": 1000,
            "inner_oof_union_holdout_rows": 1000,
            "inner_oof_warmup_rows": 0,
            "inner_oof_purged_rows": 0,
            "overlap_audit": {
                "outer_fit_holdout_row_overlap": 0,
                "outer_fit_holdout_component_overlap": 0,
                "prior_outer_holdout_component_overlap": 0,
            },
            "inner_scopes": inner_scopes,
        }
        scopes.append(scope)
        by_fold[fold_id] = scope
    return (
        {
            "schema": SCOPE_PLAN_SCHEMA,
            "loop_id": LOOP_ID,
            "aggregate_only": True,
            "contract_sha256": contract_sha256,
            "isolation_receipt_sha256": isolation_sha256,
            "fold_assignment_fingerprint": fingerprint,
            "seeds": list(REQUIRED_SEEDS),
            "eligible_rows": 1000,
            "warmup_rows": 200,
            "embargo_seconds": REQUIRED_EMBARGO_SECONDS,
            "outer_scopes": scopes,
            "custodian_attestation": {
                "attestation_id_sha256": digest("custodian-attestation"),
                "key_fingerprint": digest("custodian-key"),
                "verification_receipt_sha256": digest("custodian-verification"),
            },
        },
        by_fold,
    )


def fit_artifact(
    *,
    stage: str,
    expert_id: str,
    seed: int,
    outer_fold: int,
    inner_fold: int | None,
    fit_scope: str,
    output_scope: str,
    whole_file_contract: dict | None = None,
    input_bundle_sha256: str | None = None,
) -> dict:
    suffix = f"{stage}-{expert_id}-{seed}-{outer_fold}-{inner_fold}"
    artifact = {
        "run_id_sha256": digest(f"run-{suffix}"),
        "stage": stage,
        "expert_id": expert_id,
        "seed": seed,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "fit_scope_commitment": fit_scope,
        "output_scope_commitment": output_scope,
        "model_artifact_sha256": digest(f"model-{suffix}"),
        "config_sha256": digest(f"config-{expert_id}"),
        "code_sha256": digest(f"code-{expert_id}"),
        "input_manifest_sha256": digest(f"inputs-{suffix}"),
        "depends_on": [],
    }
    if expert_id == "whole_file" and whole_file_contract is not None:
        artifact["config_sha256"] = whole_file_contract["config"]["sha256"]
        artifact["code_sha256"] = whole_file_contract["source_closure"]["closure_sha256"]
        artifact["input_manifest_sha256"] = input_bundle_sha256
    return artifact


def build_outer_runs(
    scopes: dict[int, dict], *, whole_file_contract: dict, input_bundle_sha256: str
) -> list[dict]:
    runs = []
    for seed in REQUIRED_SEEDS:
        for outer_fold, scope in scopes.items():
            inner_runs = []
            commitments = []
            for inner_scope in scope["inner_scopes"]:
                inner_fold = inner_scope["inner_fold"]
                loop151 = fit_artifact(
                    stage="inner_fit",
                    expert_id="loop151",
                    seed=seed,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    fit_scope=inner_scope["fit_scope_commitment"],
                    output_scope=inner_scope["holdout_scope_commitment"],
                )
                whole_file = fit_artifact(
                    stage="inner_fit",
                    expert_id="whole_file",
                    seed=seed,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    fit_scope=inner_scope["fit_scope_commitment"],
                    output_scope=inner_scope["holdout_scope_commitment"],
                    whole_file_contract=whole_file_contract,
                    input_bundle_sha256=input_bundle_sha256,
                )
                commitments.extend(
                    [loop151["output_scope_commitment"], whole_file["output_scope_commitment"]]
                )
                inner_runs.append(
                    {"inner_fold": inner_fold, "loop151": loop151, "whole_file": whole_file}
                )
            outer_experts = {
                expert_id: fit_artifact(
                    stage="outer_fit",
                    expert_id=expert_id,
                    seed=seed,
                    outer_fold=outer_fold,
                    inner_fold=None,
                    fit_scope=scope["fit_scope_commitment"],
                    output_scope=scope["holdout_scope_commitment"],
                    whole_file_contract=whole_file_contract,
                    input_bundle_sha256=input_bundle_sha256,
                )
                for expert_id in ("loop151", "whole_file")
            }
            runs.append(
                {
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "fit_scope_commitment": scope["fit_scope_commitment"],
                    "holdout_scope_commitment": scope["holdout_scope_commitment"],
                    "fit_component_set_commitment": scope["fit_component_set_commitment"],
                    "holdout_component_set_commitment": scope["holdout_component_set_commitment"],
                    "fit_rows": scope["fit_rows"],
                    "holdout_rows": scope["holdout_rows"],
                    "fit_label_counts": scope["fit_label_counts"],
                    "holdout_label_counts": scope["holdout_label_counts"],
                    "inner_runs": inner_runs,
                    "outer_experts": outer_experts,
                    "fusion": {
                        "fit_scope_commitment": scope["fit_scope_commitment"],
                        "inner_oof_input_commitments": sorted(commitments),
                        "inner_oof_matrix_commitment": digest(f"matrix-{seed}-{outer_fold}"),
                        "model_artifact_sha256": digest(f"fusion-model-{seed}-{outer_fold}"),
                        "config_sha256": digest("fusion-config"),
                        "threshold_policy_sha256": digest("threshold-policy"),
                        "feature_fields": list(REQUIRED_FUSION_FIELDS),
                        "label_column_in_matrix": False,
                        "target_labels_scope": "outer_fit_inner_oof_only",
                        "forbidden_feature_count": 0,
                        "frozen_before_outer_inference": True,
                    },
                    "outer_output": {
                        "row_set_commitment": scope["holdout_scope_commitment"],
                        "output_commitment": digest(f"output-{seed}-{outer_fold}"),
                        "loop151_rows": 200,
                        "whole_file_rows": 200,
                        "whole_file_success_rows": 200,
                        "whole_file_missing_rows": 0,
                        "fusion_rows": 200,
                        "denominator_rows": 200,
                        "duplicate_rows": 0,
                        "unmatched_rows": 0,
                        "dropped_rows": 0,
                        "missingness_reason_counts": {
                            "timeout": 0,
                            "unsupported": 0,
                            "read_failure": 0,
                            "parse_failure": 0,
                            "oversize": 0,
                        },
                    },
                    "access_audit": {
                        "scope_token_sha256": digest(f"scope-token-{seed}-{outer_fold}"),
                        "audit_log_sha256": digest(f"audit-{seed}-{outer_fold}"),
                        "outer_holdout_label_reads_during_fit": 0,
                        "outer_holdout_feature_reads_during_fit": 0,
                        "outer_holdout_metric_or_threshold_reads": 0,
                        "outer_inference_feature_rows": 200,
                    },
                }
            )
    return runs


def create_receipt_case(tmp_path: Path) -> tuple[dict[str, Path], dict]:
    paths = {
        "proposal": tmp_path / "proposal.json",
        "contract": tmp_path / "full_pool_group_manifest.json",
        "isolation_receipt": tmp_path / "full_pool_isolation_validation.json",
        "scope_plan": tmp_path / "fold_scope_plan.json",
        "scope_plan_validation": tmp_path / "fold_scope_plan_validation.json",
        "implementation_manifest": tmp_path / "whole_file_implementation_manifest.json",
        "loop151_train_oof_manifest": tmp_path / "loop151_train_oof_manifest.json",
        "resource_guard": tmp_path / "resource_guard.json",
        "input_bundle": tmp_path / "train_oof_input_bundle_manifest.json",
        "training_authorization": tmp_path / "a2_training_authorization.json",
        "training_final_lease": tmp_path / "training_lease_consumption.final.json",
        "training_lease_marker_directory": tmp_path / "training_lease_consumptions",
        "receipt": tmp_path / "loop164_train_oof_execution_receipt.json",
    }
    write_json(
        paths["proposal"],
        {"loop_id": LOOP_ID, "decision": "propose_loop164_whole_file_residual_expert_no_execution"},
    )
    write_json(
        paths["contract"],
        {
            "schema": ISOLATION_CONTRACT_SCHEMA,
            "loop_id": LOOP_ID,
            "model_input_fields": list(REQUIRED_FUSION_FIELDS),
            "feature_contract": {
                "schema": FEATURE_CONTRACT_SCHEMA,
                "feature_fields": list(REQUIRED_FUSION_FIELDS),
                "feature_matrix_receipt_required": True,
                "implementation_binding_phase": IMPLEMENTATION_BINDING_PHASE,
            },
        },
    )
    contract_sha256 = sha256_file(paths["contract"])
    fingerprint = digest("frozen-fold-assignment")
    write_json(
        paths["isolation_receipt"],
        {
            "schema": ISOLATION_RECEIPT_SCHEMA,
            "loop_id": LOOP_ID,
            "decision": "pass",
            "ready_for": {"loop164_train_oof_partition": True},
            "binding_fingerprints": {
                "contract_sha256": contract_sha256,
                "a2_authorization_sha256": digest("metadata-authorization"),
                "a2_trust_anchor_sha256": digest("metadata-trust-anchor"),
                "a2_validator_source_closure_sha256": digest("metadata-source-closure"),
                "a2_runtime_python_sha256": digest("metadata-runtime"),
                "a2_resource_guard_sha256": digest("metadata-resource-guard"),
                "a2_canonical_argv_sha256": digest("metadata-argv"),
                "a2_lease_marker_sha256": digest("metadata-marker"),
                "a2_lease_consumption_id": digest("metadata-lease"),
            },
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
            "oof": {
                "fold_assignment_fingerprint": fingerprint,
                "eligible_rows": 1000,
                "warmup_rows": 200,
            },
        },
    )
    isolation_sha256 = sha256_file(paths["isolation_receipt"])
    scope_plan, scopes = build_scope_plan(contract_sha256, isolation_sha256)
    write_json(paths["scope_plan"], scope_plan)
    write_json(
        paths["scope_plan_validation"],
        {
            "schema": SCOPE_PLAN_VALIDATION_SCHEMA,
            "loop_id": LOOP_ID,
            "aggregate_only_verified": True,
            "proposal_binding_verified": True,
            "contract_binding_verified": True,
            "isolation_receipt_binding_verified": True,
            "scope_plan_binding_verified": True,
            "binding_fingerprints": {
                "proposal_sha256": sha256_file(paths["proposal"]),
                "contract_sha256": contract_sha256,
                "isolation_receipt_sha256": isolation_sha256,
                "scope_plan_sha256": sha256_file(paths["scope_plan"]),
            },
            "plan_summary": {"fold_assignment_fingerprint": fingerprint},
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
    implementation_payload = create_whole_file_implementation_fixture(
        tmp_path, paths["implementation_manifest"]
    )
    write_json(
        paths["loop151_train_oof_manifest"],
        {
            "schema": LOOP151_TRAIN_OOF_MANIFEST_SCHEMA,
            "loop_id": LOOP_ID,
            "model_id": "loop151_equivalent",
            "train_only": True,
            "initialization_policy": "from_scratch",
            "recipe_sha256": digest("loop151-recipe"),
            "runtime_lock_sha256": digest("loop151-runtime-lock"),
            "fold_assignment_fingerprint": fingerprint,
            "seeds": list(REQUIRED_SEEDS),
            "feature_fields": list(REQUIRED_FUSION_FIELDS),
        },
    )
    canonical_argv = ["python", "loop164_controller.py"]
    controller_sha256 = digest("controller")
    write_json(
        paths["input_bundle"],
        {
            "schema": INPUT_BUNDLE_SCHEMA,
            "loop_id": LOOP_ID,
            "allowed_split_roles": ["train_anchor", "train_oof"],
            "forbidden_split_roles": [
                "val_a",
                "val_b",
                "test10k",
                "legacy_full_test",
                "sentinel",
                "confirmation",
                "certification",
            ],
            "feature_fields": list(REQUIRED_FUSION_FIELDS),
            "fold_assignment_fingerprint": fingerprint,
            "scope_plan_validation_sha256": sha256_file(paths["scope_plan_validation"]),
            "protected_input_open_policy": "after_final_lease_only",
            "input_artifact_commitments": {
                "train_anchor_sha256": digest("train-anchor"),
                "train_oof_sha256": digest("train-oof"),
            },
        },
    )
    guard_runtime = {
        "cwd": str(Path(__file__).resolve().parents[1]),
        "python_sha256": digest("python"),
        "controller_path": "scripts/run_loop164_train_oof_controller.py",
        "controller_sha256": controller_sha256,
        "canonical_argv_sha256": canonical_argv_sha256(canonical_argv),
    }
    write_json(
        paths["resource_guard"],
        {
            "schema": RESOURCE_GUARD_SCHEMA,
            "loop_id": LOOP_ID,
            "operation": "loop164_three_seed_nested_train_oof",
            "guard_ready": True,
            "decision": "pass",
            "runtime_binding": guard_runtime,
            "implementation_binding": {
                "implementation_manifest_sha256": sha256_file(paths["implementation_manifest"]),
                "source_closure_sha256": implementation_payload["source_closure"]["closure_sha256"],
                "memory_contract_sha256": sha256_json(implementation_payload["memory_contract"]),
            },
            "receipt": {
                "created_at_utc": "2025-01-01T01:30:00Z",
                "controller_sha256": controller_sha256,
                "resource_budget_sha256": digest("resource-budget"),
            },
        },
    )
    parent_names = (
        "proposal",
        "contract",
        "isolation_receipt",
        "scope_plan",
        "scope_plan_validation",
        "implementation_manifest",
        "loop151_train_oof_manifest",
        "resource_guard",
        "input_bundle",
    )
    parent_hashes = {name: sha256_file(paths[name]) for name in parent_names}
    write_json(
        paths["training_authorization"],
        {
            "schema": TRAINING_AUTHORIZATION_SCHEMA,
            "loop_id": LOOP_ID,
            "authorization_level": "A2_train_only_nested_oof",
            "decision": "allow_single_loop164_train_oof_execution",
            "execution_environment": "custodian_side_train_only",
            "operation": "loop164_three_seed_nested_train_oof",
            "issued_at_utc": "2025-01-01T00:00:00Z",
            "not_before_utc": "2025-01-01T00:00:00Z",
            "expires_at_utc": "2025-01-01T12:00:00Z",
            "authority_attestation": {
                "trusted_key_fingerprint": digest("custodian-key"),
                "trust_anchor_sha256": digest("external-trust-anchor"),
                "verification_receipt_sha256": digest("custodian-verification"),
            },
            "runtime_binding": {
                "cwd": str(Path(__file__).resolve().parents[1]),
                "python_executable": "synthetic-python",
                "python_sha256": digest("python"),
                "controller_path": "scripts/run_loop164_train_oof_controller.py",
                "controller_sha256": controller_sha256,
                "entrypoint": "run_loop164_train_oof_controller.main",
            },
            "canonical_argv": canonical_argv,
            "allowed_split_roles": ["train_anchor", "train_oof"],
            "forbidden_split_roles": [
                "val_a",
                "val_b",
                "test10k",
                "legacy_full_test",
                "sentinel",
                "confirmation",
                "certification",
            ],
            "feature_fields": list(REQUIRED_FUSION_FIELDS),
            "fold_assignment_fingerprint": fingerprint,
            "outer_run_budget": 15,
            "bindings": {
                name: {"path": str(paths[name]), "sha256": digest}
                for name, digest in parent_hashes.items()
            },
            "output_binding": {
                "execution_receipt_path": str(paths["receipt"]),
                "final_lease_path": str(paths["training_final_lease"]),
                "lease_marker_directory": str(paths["training_lease_marker_directory"]),
            },
            "one_shot_lease": {
                "lease_id": "loop164-synthetic-train-lease",
                "purpose": "single_loop164_three_seed_nested_train_oof",
                "state": "ready",
            },
            "max_resource_guard_age_seconds": 3600,
        },
    )
    authorization_sha256 = sha256_file(paths["training_authorization"])
    scope_validation_sha256 = sha256_file(paths["scope_plan_validation"])
    input_bundle_sha256 = sha256_file(paths["input_bundle"])
    consumption_id = build_lease_consumption_id(
        authorization_sha256=authorization_sha256,
        lease_id="loop164-synthetic-train-lease",
        controller_sha256=controller_sha256,
        input_bundle_sha256=input_bundle_sha256,
        scope_plan_validation_sha256=scope_validation_sha256,
        canonical_argv_sha256=canonical_argv_sha256(canonical_argv),
        execution_receipt_path=paths["receipt"],
    )
    marker_path = paths["training_lease_marker_directory"] / f"{consumption_id}.final.json"
    lease_payload = {
        "loop_id": LOOP_ID,
        "state": "consumed_before_protected_open",
        "lease_consumption_id": consumption_id,
        "authorization_sha256": authorization_sha256,
        "lease_id": "loop164-synthetic-train-lease",
        "scope_plan_validation_sha256": scope_validation_sha256,
        "input_bundle_sha256": input_bundle_sha256,
        "implementation_manifest_sha256": sha256_file(paths["implementation_manifest"]),
        "source_closure_sha256": implementation_payload["source_closure"]["closure_sha256"],
        "memory_contract_sha256": sha256_json(implementation_payload["memory_contract"]),
        "controller_sha256": controller_sha256,
        "canonical_argv_sha256": canonical_argv_sha256(canonical_argv),
        "canonical_argv": canonical_argv,
        "output_receipt_path": str(paths["receipt"]),
        "fold_assignment_fingerprint": fingerprint,
        "outer_run_budget": 15,
        "consumed_at_utc": "2025-01-01T01:45:00Z",
        "marker_path": str(marker_path),
    }
    write_json(marker_path, {"schema": TRAINING_LEASE_MARKER_SCHEMA, **lease_payload})
    write_json(
        paths["training_final_lease"],
        {
            "schema": TRAINING_LEASE_SCHEMA,
            **lease_payload,
        },
    )
    paths["training_lease_marker"] = marker_path
    binding_paths = {
        **{name: paths[name] for name in parent_names},
        "training_authorization": paths["training_authorization"],
        "training_final_lease": paths["training_final_lease"],
        "training_lease_marker": paths["training_lease_marker"],
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "loop_id": LOOP_ID,
        "aggregate_only": True,
        "decision": "pass",
        "completed_at_utc": "2025-01-01T02:00:00Z",
        "bindings": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in binding_paths.items()
        },
        "partition_plan": {
            "fold_assignment_fingerprint": fingerprint,
            "eligible_rows": 1000,
            "warmup_rows": 200,
            "seeds": list(REQUIRED_SEEDS),
            "outer_fold_ids": [0, 1, 2, 3, 4],
        },
        "lineage": {
            "loop151_equivalent": {
                "train_oof_manifest_sha256": sha256_file(paths["loop151_train_oof_manifest"]),
                "recipe_sha256": digest("loop151-recipe"),
                "runtime_lock_sha256": digest("loop151-runtime-lock"),
                "initialization_policy": "from_scratch",
            },
            "whole_file_expert": {
                "implementation_manifest_sha256": sha256_file(paths["implementation_manifest"]),
                "source_closure_sha256": implementation_payload["source_closure"]["closure_sha256"],
                "config_sha256": implementation_payload["config"]["sha256"],
                "runtime_lock_sha256": implementation_payload["runtime_lock"]["sha256"],
                "input_contract_sha256": sha256_json(implementation_payload["input_contract"]),
                "missingness_contract_sha256": sha256_json(
                    implementation_payload["missingness_contract"]
                ),
                "whole_file_input_policy": "all_bytes_chunked_no_silent_truncation",
            },
            "fusion": {
                "config_sha256": digest("fusion-config"),
                "threshold_policy_sha256": digest("threshold-policy"),
                "selection_policy": "nested_inner_oof_only",
                "feature_fields": list(REQUIRED_FUSION_FIELDS),
            },
        },
        "outer_runs": build_outer_runs(
            scopes,
            whole_file_contract=implementation_payload,
            input_bundle_sha256=input_bundle_sha256,
        ),
        "coverage": {
            "eligible_rows": 1000,
            "per_seed_outer_holdout": [
                {
                    "seed": seed,
                    "expected_rows": 1000,
                    "observed_rows": 1000,
                    "unique_outer_holdout_rows": 1000,
                    "duplicate_rows": 0,
                    "unmatched_rows": 0,
                    "dropped_rows": 0,
                }
                for seed in REQUIRED_SEEDS
            ],
            "global_duplicate_rows": 0,
            "global_unmatched_rows": 0,
            "global_dropped_rows": 0,
        },
        "access_audit": {
            "controller_audit_log_sha256": digest("controller-audit"),
            "identity_rows_exported": 0,
            "prediction_rows_exported": 0,
            "in_sample_score_substitution_count": 0,
            "fold_drift_count": 0,
            "missing_output_drop_count": 0,
            "forbidden_split_access_counts": {
                "val_a": 0,
                "val_b": 0,
                "test10k": 0,
                "legacy_full_test": 0,
                "sentinel": 0,
                "confirmation": 0,
                "certification": 0,
            },
        },
        "blockers": [],
    }
    write_json(paths["receipt"], receipt)
    return paths, receipt


def validate_case(paths: dict[str, Path]) -> dict:
    return validate_loop164_nested_oof_execution_receipt(
        receipt_json=paths["receipt"],
        proposal_json=paths["proposal"],
        contract_json=paths["contract"],
        isolation_receipt_json=paths["isolation_receipt"],
        scope_plan_json=paths["scope_plan"],
        scope_plan_validation_json=paths["scope_plan_validation"],
        implementation_manifest_json=paths["implementation_manifest"],
        loop151_train_oof_manifest_json=paths["loop151_train_oof_manifest"],
        resource_guard_json=paths["resource_guard"],
        input_bundle_json=paths["input_bundle"],
        training_authorization_json=paths["training_authorization"],
        training_final_lease_json=paths["training_final_lease"],
        training_lease_marker_directory=paths["training_lease_marker_directory"],
    )


def write_receipt(paths: dict[str, Path], receipt: dict) -> None:
    write_json(paths["receipt"], receipt)


def test_valid_aggregate_nested_oof_receipt_passes(tmp_path: Path):
    paths, _ = create_receipt_case(tmp_path)

    payload = validate_case(paths)

    assert payload["decision"] == "pass"
    assert payload["ready_for"]["loop164_train_oof_data_boundary"] is True
    assert payload["ready_for"]["val_a"] is False
    assert payload["ready_for"]["test10k"] is False
    assert payload["coverage"]["eligible_rows"] == 1000
    assert "source_sha256" not in json.dumps(payload)


def test_receipt_rejects_unattested_isolation_receipt(tmp_path: Path):
    paths, _ = create_receipt_case(tmp_path)
    isolation_receipt = json.loads(paths["isolation_receipt"].read_text(encoding="utf-8"))
    isolation_receipt.pop("a2_authorization_provenance")
    write_json(paths["isolation_receipt"], isolation_receipt)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    receipt["bindings"]["isolation_receipt"]["sha256"] = sha256_file(paths["isolation_receipt"])
    write_receipt(paths, receipt)

    payload = validate_case(paths)

    assert payload["decision"] == "block"
    assert "isolation_receipt_authorization_provenance_not_object" in payload["blockers"]


def test_receipt_rejects_isolation_receipt_with_training_scope(tmp_path: Path):
    paths, receipt = create_receipt_case(tmp_path)
    isolation_receipt = json.loads(paths["isolation_receipt"].read_text(encoding="utf-8"))
    isolation_receipt["a2_authorization_provenance"]["authority_scope"]["grants"] = [
        "train_oof"
    ]
    write_json(paths["isolation_receipt"], isolation_receipt)
    receipt["bindings"]["isolation_receipt"]["sha256"] = sha256_file(
        paths["isolation_receipt"]
    )
    write_receipt(paths, receipt)

    payload = validate_case(paths)

    assert payload["decision"] == "block"
    assert "isolation_receipt_authorization_provenance_invalid" in payload["blockers"]


def test_receipt_rejects_legacy_metadata_implementation_placeholder(tmp_path: Path):
    paths, receipt = create_receipt_case(tmp_path)
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    contract["feature_contract"] = {
        "schema": "axon_loop164_residual_fusion_feature_contract_v1",
        "implementation_manifest_sha256": digest("legacy-placeholder"),
        "feature_matrix_receipt_required": True,
    }
    write_json(paths["contract"], contract)
    receipt["bindings"]["contract"]["sha256"] = sha256_file(paths["contract"])
    write_receipt(paths, receipt)

    payload = validate_case(paths)

    assert payload["decision"] == "block"
    assert "contract_feature_contract_shape_invalid" in payload["blockers"]
    assert "contract_feature_contract_implementation_binding_phase_invalid" in payload["blockers"]


def test_receipt_blocks_identity_feature_alias_and_label_column(tmp_path: Path):
    paths, receipt = create_receipt_case(tmp_path)
    receipt["outer_runs"][0]["fusion"]["feature_fields"] = [
        *REQUIRED_FUSION_FIELDS[:-1],
        "family_code",
    ]
    receipt["outer_runs"][0]["fusion"]["label_column_in_matrix"] = True
    write_receipt(paths, receipt)

    payload = validate_case(paths)

    assert payload["decision"] == "block"
    assert "outer_run_fusion_feature_allowlist_invalid" in payload["blockers"]
    assert "outer_run_fusion_label_column_detected" in payload["blockers"]


def test_receipt_blocks_outer_holdout_access_during_fit_and_missing_rows(tmp_path: Path):
    paths, receipt = create_receipt_case(tmp_path)
    receipt["outer_runs"][3]["access_audit"]["outer_holdout_label_reads_during_fit"] = 1
    receipt["outer_runs"][3]["outer_output"]["dropped_rows"] = 1
    write_receipt(paths, receipt)

    payload = validate_case(paths)

    assert payload["decision"] == "block"
    assert "outer_run_outer_holdout_label_reads_during_fit_nonzero" in payload["blockers"]
    assert "outer_run_dropped_rows_nonzero" in payload["blockers"]


def test_receipt_blocks_whole_file_input_drift_and_missingness_nonconservation(tmp_path: Path):
    paths, receipt = create_receipt_case(tmp_path)
    whole_file = receipt["outer_runs"][0]["outer_experts"]["whole_file"]
    whole_file["input_manifest_sha256"] = digest("different-input-bundle")
    outer_output = receipt["outer_runs"][0]["outer_output"]
    outer_output["whole_file_success_rows"] = 199
    outer_output["whole_file_missing_rows"] = 1
    write_receipt(paths, receipt)

    payload = validate_case(paths)

    assert payload["decision"] == "block"
    assert "outer_run_outer_whole_file_whole_file_manifest_binding_mismatch" in payload["blockers"]
    assert "outer_run_missingness_reason_total_mismatch" in payload["blockers"]


def test_receipt_blocks_fold_drift_and_reused_model_artifact(tmp_path: Path):
    paths, receipt = create_receipt_case(tmp_path)
    first = receipt["outer_runs"][0]["outer_experts"]["whole_file"]
    second = receipt["outer_runs"][1]["outer_experts"]["whole_file"]
    second["model_artifact_sha256"] = first["model_artifact_sha256"]
    receipt["outer_runs"][1]["fit_scope_commitment"] = digest("drifted-fit-scope")
    write_receipt(paths, receipt)

    payload = validate_case(paths)

    assert payload["decision"] == "block"
    assert "outer_run_fit_scope_commitment_mismatch" in payload["blockers"]
    assert "model_artifact_reused_across_fit_scopes" in payload["blockers"]


def test_receipt_blocks_scope_plan_overlap_and_forbidden_holdout_access(tmp_path: Path):
    paths, receipt = create_receipt_case(tmp_path)
    scope_plan = json.loads(paths["scope_plan"].read_text(encoding="utf-8"))
    scope_plan["outer_scopes"][0]["overlap_audit"]["outer_fit_holdout_component_overlap"] = 1
    write_json(paths["scope_plan"], scope_plan)
    receipt["bindings"]["scope_plan"]["sha256"] = sha256_file(paths["scope_plan"])
    receipt["access_audit"]["forbidden_split_access_counts"]["test10k"] = 1
    write_receipt(paths, receipt)

    payload = validate_case(paths)

    assert payload["decision"] == "block"
    assert "scope_plan_outer_overlap_detected" in payload["blockers"]
    assert "receipt_access_audit_forbidden_split_access_detected" in payload["blockers"]


def test_receipt_blocks_training_authorization_fold_or_feature_drift(tmp_path: Path):
    paths, receipt = create_receipt_case(tmp_path)
    authorization = json.loads(paths["training_authorization"].read_text(encoding="utf-8"))
    authorization["fold_assignment_fingerprint"] = digest("different-fold-assignment")
    write_json(paths["training_authorization"], authorization)
    receipt["bindings"]["training_authorization"]["sha256"] = sha256_file(
        paths["training_authorization"]
    )
    write_receipt(paths, receipt)

    payload = validate_case(paths)

    assert payload["decision"] == "block"
    assert "training_authorization_fold_fingerprint_mismatch" in payload["blockers"]


def test_receipt_blocks_v2_lease_marker_drift(tmp_path: Path):
    paths, receipt = create_receipt_case(tmp_path)
    marker = json.loads(paths["training_lease_marker"].read_text(encoding="utf-8"))
    marker["input_bundle_sha256"] = digest("different-input-bundle")
    write_json(paths["training_lease_marker"], marker)
    receipt["bindings"]["training_lease_marker"]["sha256"] = sha256_file(
        paths["training_lease_marker"]
    )
    write_receipt(paths, receipt)

    payload = validate_case(paths)

    assert payload["decision"] == "block"
    assert "training_lease_marker_final_mismatch" in payload["blockers"]
