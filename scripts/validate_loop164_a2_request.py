#!/usr/bin/env python3
"""Validate non-authorizing Loop164 A2 custodian request documents.

These documents are static requests only. They cannot authorize protected input
access, create a lease, or stand in for either future A2 authorization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "loop164_whole_file_residual_expert"
METADATA_REQUEST_SCHEMA = "axon_loop164_a2_metadata_request_v1"
TRAINING_REQUEST_SCHEMA = "axon_loop164_a2_training_request_v1"
METADATA_AUTHORIZATION_SCHEMA = "axon_loop164_isolation_validation_authorization_v3"
TRAINING_AUTHORIZATION_SCHEMA = "axon_loop164_training_authorization_v2"
METADATA_AUTHORITY_SCOPE = {
    "tier": "A2",
    "operation": "metadata_isolation_only",
    "protected_input_scope": "metadata_only",
    "grants": [],
}
METADATA_AUTHORIZATION_PATH = Path(
    "manifests/roadmap_9997/loop164_whole_file_residual_expert/"
    "a2_isolation_validation_authorization.json"
)
TRAINING_AUTHORIZATION_PATH = Path(
    "manifests/roadmap_9997/loop164_whole_file_residual_expert/"
    "a2_training_authorization.json"
)
FORBIDDEN_TRAINING_ROLES = [
    "val_a",
    "val_b",
    "test10k",
    "legacy_full_test",
    "sentinel",
    "confirmation",
    "certification",
]
FORBIDDEN_AUTHORIZATION_FIELDS = {
    "authority_attestation",
    "canonical_argv",
    "decision",
    "expires_at_utc",
    "issued_at_utc",
    "not_before_utc",
    "one_shot_lease",
    "runtime_binding",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request document must be a JSON object")
    return payload


def _exact_keys(payload: dict[str, Any], expected: set[str], label: str) -> list[str]:
    actual = set(payload)
    failures = []
    if actual - expected:
        failures.append(f"{label}_unexpected_fields")
    if expected - actual:
        failures.append(f"{label}_missing_fields")
    return failures


def _contains_forbidden_authorization_field(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            key in FORBIDDEN_AUTHORIZATION_FIELDS
            or _contains_forbidden_authorization_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_authorization_field(item) for item in value)
    return False


def _validate_common(payload: dict[str, Any], *, schema: str, state: str) -> list[str]:
    failures: list[str] = []
    if payload.get("schema") != schema:
        failures.append("a2_request_schema_invalid")
    if payload.get("document_kind") != "custodian_request_not_authorization":
        failures.append("a2_request_document_kind_invalid")
    if payload.get("loop_id") != LOOP_ID:
        failures.append("a2_request_loop_id_invalid")
    if payload.get("request_state") != state:
        failures.append("a2_request_state_invalid")
    if payload.get("authorization_granted") is not False:
        failures.append("a2_request_must_not_grant_authorization")
    if _contains_forbidden_authorization_field(payload):
        failures.append("a2_request_contains_authorization_field")
    if not isinstance(payload.get("target_paths"), dict):
        failures.append("a2_request_target_paths_invalid")
    if not isinstance(payload.get("custodian_required_bindings"), list):
        failures.append("a2_request_custodian_bindings_invalid")
    if not isinstance(payload.get("forbidden_operations"), list):
        failures.append("a2_request_forbidden_operations_invalid")
    return failures


def validate_a2_request_payload(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return ["a2_request_not_object"]
    schema = payload.get("schema")
    if schema == METADATA_REQUEST_SCHEMA:
        failures = _validate_common(payload, schema=schema, state="draft")
        failures.extend(
            _exact_keys(
                payload,
                {
                    "schema",
                    "document_kind",
                    "loop_id",
                    "request_state",
                    "authorization_granted",
                    "target_authorization_schema",
                    "target_paths",
                    "custodian_required_bindings",
                    "forbidden_operations",
                    "requested_operation",
                    "authority_scope",
                },
                "metadata_a2_request",
            )
        )
        if payload.get("target_authorization_schema") != METADATA_AUTHORIZATION_SCHEMA:
            failures.append("metadata_request_target_schema_invalid")
        if payload.get("requested_operation") != "loop164_metadata_isolation_validation":
            failures.append("metadata_request_operation_invalid")
        if payload.get("authority_scope") != METADATA_AUTHORITY_SCOPE:
            failures.append("metadata_request_scope_invalid")
        expected_paths = {
            "authorization": METADATA_AUTHORIZATION_PATH.as_posix(),
            "output": "reports/roadmap_9997/loop164/full_pool_isolation_validation.json",
            "lease_marker_directory": "reports/roadmap_9997/loop164/metadata_lease_consumptions",
        }
        if payload.get("target_paths") != expected_paths:
            failures.append("metadata_request_target_paths_invalid")
        expected_bindings = [
            "external_trust_anchor",
            "authority_attestation",
            "runtime_binding",
            "validator_binding",
            "validator_source_closure",
            "canonical_argv",
            "contract_binding",
            "rows_artifact_binding",
            "metadata_root_binding",
            "resource_guard_binding",
            "one_shot_lease",
        ]
        if payload.get("custodian_required_bindings") != expected_bindings:
            failures.append("metadata_request_custodian_bindings_invalid")
        if payload.get("forbidden_operations") != [
            "training",
            "implementation_review",
            "val",
            "test10k",
            "legacy_full_test",
            "certification",
        ]:
            failures.append("metadata_request_forbidden_operations_invalid")
    elif schema == TRAINING_REQUEST_SCHEMA:
        failures = _validate_common(
            payload,
            schema=schema,
            state="blocked_pending_metadata_and_static_review",
        )
        failures.extend(
            _exact_keys(
                payload,
                {
                    "schema",
                    "document_kind",
                    "loop_id",
                    "request_state",
                    "authorization_granted",
                    "target_authorization_schema",
                    "target_paths",
                    "custodian_required_bindings",
                    "forbidden_operations",
                    "allowed_split_roles",
                    "forbidden_split_roles",
                    "outer_run_budget",
                    "prerequisites",
                },
                "training_a2_request",
            )
        )
        if payload.get("target_authorization_schema") != TRAINING_AUTHORIZATION_SCHEMA:
            failures.append("training_request_target_schema_invalid")
        if payload.get("target_paths") != {
            "authorization": TRAINING_AUTHORIZATION_PATH.as_posix(),
            "final_lease": "reports/roadmap_9997/loop164/training_lease_consumption.final.json",
            "execution_receipt": "reports/roadmap_9997/loop164/loop164_train_oof_execution_receipt.json",
        }:
            failures.append("training_request_target_paths_invalid")
        if payload.get("allowed_split_roles") != ["train_anchor", "train_oof"]:
            failures.append("training_request_allowed_roles_invalid")
        if payload.get("forbidden_split_roles") != FORBIDDEN_TRAINING_ROLES:
            failures.append("training_request_forbidden_roles_invalid")
        if payload.get("outer_run_budget") != 15:
            failures.append("training_request_run_budget_invalid")
        required_prerequisites = [
            "metadata_isolation_receipt_pass",
            "fold_scope_plan_validation_pass",
            "independently_authorized_implementation_manifest",
            "loop151_train_oof_manifest",
            "train_only_input_bundle",
            "fresh_training_resource_guard",
            "controller_source_closure",
        ]
        if payload.get("prerequisites") != required_prerequisites:
            failures.append("training_request_prerequisites_invalid")
        if payload.get("custodian_required_bindings") != [
            "external_trust_anchor",
            "authority_attestation",
            "runtime_binding",
            "canonical_argv",
            "all_prerequisite_bindings",
            "output_binding",
            "one_shot_lease",
        ]:
            failures.append("training_request_custodian_bindings_invalid")
        if payload.get("forbidden_operations") != [
            "val",
            "test10k",
            "legacy_full_test",
            "sentinel",
            "confirmation",
            "certification",
        ]:
            failures.append("training_request_forbidden_operations_invalid")
    else:
        return ["a2_request_schema_invalid"]
    return sorted(set(failures))


def validate_a2_request_file(request_json: Path, *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved_request = request_json.resolve()
    forbidden_paths = {
        (resolved_root / METADATA_AUTHORIZATION_PATH).resolve(),
        (resolved_root / TRAINING_AUTHORIZATION_PATH).resolve(),
    }
    if resolved_request in forbidden_paths:
        return {"decision": "block", "blockers": ["a2_request_path_is_authorization_path"]}
    try:
        payload = _read_json(resolved_request)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return {"decision": "block", "blockers": ["a2_request_unreadable"]}
    blockers = validate_a2_request_payload(payload)
    return {
        "schema": "axon_loop164_a2_request_validation_v1",
        "loop_id": LOOP_ID,
        "decision": "pass" if not blockers else "block",
        "authorization_granted": False,
        "blockers": blockers,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a non-authorizing Loop164 A2 request.")
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_a2_request_file(args.request_json, root=args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
