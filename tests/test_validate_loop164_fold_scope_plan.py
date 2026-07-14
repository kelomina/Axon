from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_loop164_fold_scope_plan import (  # noqa: E402
    ISOLATION_AUTHORIZATION_PROVENANCE_SCHEMA,
    ISOLATION_CONTRACT_SCHEMA,
    ISOLATION_METADATA_AUTHORITY_SCOPE,
    ISOLATION_RECEIPT_SCHEMA,
    LOOP_ID,
    REQUIRED_EMBARGO_SECONDS,
    REQUIRED_FUSION_FIELDS,
    SCOPE_PLAN_SCHEMA,
    validate_loop164_fold_scope_plan,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def labels(rows: int) -> dict[str, int]:
    return {"0": rows // 2, "1": rows - rows // 2}


def create_scope_plan_case(tmp_path: Path) -> tuple[dict[str, Path], dict]:
    paths = {
        "proposal": tmp_path / "proposal.json",
        "contract": tmp_path / "contract.json",
        "isolation": tmp_path / "isolation_receipt.json",
        "scope_plan": tmp_path / "fold_scope_plan.json",
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
                "schema": "axon_loop164_residual_fusion_feature_contract_v2",
                "feature_fields": list(REQUIRED_FUSION_FIELDS),
                "feature_matrix_receipt_required": True,
                "implementation_binding_phase": "deferred_to_a2_training_authority",
            },
        },
    )
    contract_sha256 = sha256_file(paths["contract"])
    fingerprint = digest("scope-plan-fingerprint")
    write_json(
        paths["isolation"],
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
    isolation_sha256 = sha256_file(paths["isolation"])
    windows = (
        ("2025-03-01T00:00:00Z", "2025-03-02T00:00:00Z"),
        ("2025-04-02T00:00:00Z", "2025-04-03T00:00:00Z"),
        ("2025-05-04T00:00:00Z", "2025-05-05T00:00:00Z"),
        ("2025-06-05T00:00:00Z", "2025-06-06T00:00:00Z"),
        ("2025-07-07T00:00:00Z", "2025-07-08T00:00:00Z"),
    )
    outer_scopes = []
    for outer_fold, (holdout_minimum, holdout_maximum) in enumerate(windows):
        outer_fit = digest(f"outer-{outer_fold}-fit")
        inner_scopes = []
        for inner_fold in range(5):
            inner_scopes.append(
                {
                    "inner_fold": inner_fold,
                    "fit_scope_commitment": digest(f"inner-{outer_fold}-{inner_fold}-fit"),
                    "holdout_scope_commitment": digest(f"inner-{outer_fold}-{inner_fold}-holdout"),
                    "parent_outer_fit_scope_commitment": outer_fit,
                    "fit_component_set_commitment": digest(
                        f"inner-{outer_fold}-{inner_fold}-fit-components"
                    ),
                    "holdout_component_set_commitment": digest(
                        f"inner-{outer_fold}-{inner_fold}-holdout-components"
                    ),
                    "fit_rows": 800,
                    "holdout_rows": 200,
                    "fit_label_counts": labels(800),
                    "holdout_label_counts": labels(200),
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
        outer_scopes.append(
            {
                "outer_fold": outer_fold,
                "fit_scope_commitment": outer_fit,
                "holdout_scope_commitment": digest(f"outer-{outer_fold}-holdout"),
                "fit_component_set_commitment": digest(f"outer-{outer_fold}-fit-components"),
                "holdout_component_set_commitment": digest(
                    f"outer-{outer_fold}-holdout-components"
                ),
                "fit_rows": 1000,
                "holdout_rows": 200,
                "fit_label_counts": labels(1000),
                "holdout_label_counts": labels(200),
                "fit_max_component_time_utc": "2025-01-01T00:00:00Z",
                "holdout_min_component_time_utc": holdout_minimum,
                "holdout_max_component_time_utc": holdout_maximum,
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
        )
    plan = {
        "schema": SCOPE_PLAN_SCHEMA,
        "loop_id": LOOP_ID,
        "aggregate_only": True,
        "contract_sha256": contract_sha256,
        "isolation_receipt_sha256": isolation_sha256,
        "fold_assignment_fingerprint": fingerprint,
        "seeds": [41, 42, 43],
        "eligible_rows": 1000,
        "warmup_rows": 200,
        "embargo_seconds": REQUIRED_EMBARGO_SECONDS,
        "outer_scopes": outer_scopes,
        "custodian_attestation": {
            "attestation_id_sha256": digest("attestation"),
            "key_fingerprint": digest("key"),
            "verification_receipt_sha256": digest("verification"),
        },
    }
    write_json(paths["scope_plan"], plan)
    return paths, plan


def validate_case(paths: dict[str, Path]) -> dict:
    return validate_loop164_fold_scope_plan(
        proposal_json=paths["proposal"],
        contract_json=paths["contract"],
        isolation_receipt_json=paths["isolation"],
        scope_plan_json=paths["scope_plan"],
    )


def test_valid_scope_plan_passes_without_opening_row_identities(tmp_path: Path):
    paths, _ = create_scope_plan_case(tmp_path)

    payload = validate_case(paths)

    assert payload["decision"] == "pass"
    assert payload["ready_for"]["fold_scope_frozen"] is True
    assert payload["ready_for"]["train_oof"] is False
    assert payload["plan_summary"]["outer_fold_count"] == 5
    assert "source_sha256" not in json.dumps(payload)


def test_scope_plan_rejects_legacy_implementation_placeholder_contract(tmp_path: Path):
    paths, _ = create_scope_plan_case(tmp_path)
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    contract["feature_contract"] = {
        "schema": "axon_loop164_residual_fusion_feature_contract_v1",
        "implementation_manifest_sha256": digest("legacy-placeholder"),
        "feature_matrix_receipt_required": True,
    }
    write_json(paths["contract"], contract)

    payload = validate_case(paths)

    assert payload["decision"] == "block"
    assert "contract_feature_contract_shape_invalid" in payload["blockers"]
    assert "contract_feature_contract_implementation_binding_phase_invalid" in payload["blockers"]


def test_scope_plan_blocks_parent_fit_drift_and_inner_coverage_gap(tmp_path: Path):
    paths, plan = create_scope_plan_case(tmp_path)
    plan["outer_scopes"][0]["inner_scopes"][0]["parent_outer_fit_scope_commitment"] = digest(
        "wrong-parent"
    )
    plan["outer_scopes"][0]["inner_oof_union_holdout_rows"] = 999
    write_json(paths["scope_plan"], plan)

    payload = validate_case(paths)

    assert payload["decision"] == "block"
    assert "scope_plan_inner_parent_outer_fit_mismatch" in payload["blockers"]
    assert "scope_plan_inner_oof_union_coverage_invalid" in payload["blockers"]


def test_scope_plan_blocks_outer_fit_embargo_and_component_overlap(tmp_path: Path):
    paths, plan = create_scope_plan_case(tmp_path)
    plan["outer_scopes"][1]["fit_max_component_time_utc"] = "2025-04-01T00:00:00Z"
    plan["outer_scopes"][1]["overlap_audit"]["outer_fit_holdout_component_overlap"] = 1
    write_json(paths["scope_plan"], plan)

    payload = validate_case(paths)

    assert payload["decision"] == "block"
    assert "scope_plan_outer_fit_temporal_or_embargo_invalid" in payload["blockers"]
    assert "scope_plan_outer_overlap_detected" in payload["blockers"]


def test_scope_plan_blocks_fold_reordering_and_inner_partition_gap(tmp_path: Path):
    paths, plan = create_scope_plan_case(tmp_path)
    plan["outer_scopes"][0], plan["outer_scopes"][1] = (
        plan["outer_scopes"][1],
        plan["outer_scopes"][0],
    )
    plan["outer_scopes"][2]["inner_oof_warmup_rows"] = 1
    write_json(paths["scope_plan"], plan)

    payload = validate_case(paths)

    assert payload["decision"] == "block"
    assert "scope_plan_outer_fold_order_invalid" in payload["blockers"]
    assert "scope_plan_inner_oof_partition_accounting_invalid" in payload["blockers"]
