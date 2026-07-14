#!/usr/bin/env python3
"""Fail-closed aggregate-only verifier for the future Loop164 nested OOF run.

This A1 utility validates future JSON manifests and receipts only.  It never
opens raw samples, caches, predictions, checkpoints, or row-level split data.
Passing this verifier proves a declared aggregate contract, not model quality,
and never authorizes Val, Test-10k, or full-test access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from validate_loop164_training_authority import (
    INPUT_BUNDLE_SCHEMA,
    ISOLATION_METADATA_AUTHORITY_SCOPE,
    RESOURCE_GUARD_SCHEMA,
    SCOPE_PLAN_VALIDATION_SCHEMA,
    TRAINING_AUTHORIZATION_SCHEMA,
    TRAINING_LEASE_MARKER_SCHEMA,
    TRAINING_LEASE_SCHEMA,
    build_lease_consumption_id,
)
from validate_loop164_whole_file_implementation import (
    WholeFileImplementationManifestResult,
    validate_implementation_manifest_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "loop164_whole_file_residual_expert"
RECEIPT_SCHEMA = "axon_loop164_train_oof_execution_receipt_v1"
VALIDATION_SCHEMA = "axon_loop164_train_oof_execution_validation_v1"
SCOPE_PLAN_SCHEMA = "axon_loop164_fold_scope_plan_v1"
LOOP151_TRAIN_OOF_MANIFEST_SCHEMA = "axon_loop164_loop151_equivalent_train_oof_manifest_v1"
ISOLATION_CONTRACT_SCHEMA = "axon_loop164_full_pool_isolation_contract_v2"
ISOLATION_RECEIPT_SCHEMA = "axon_loop164_full_pool_isolation_validation_v4"
ISOLATION_AUTHORIZATION_PROVENANCE_SCHEMA = "axon_loop164_isolation_authorization_provenance_v2"
FEATURE_CONTRACT_SCHEMA = "axon_loop164_residual_fusion_feature_contract_v2"
IMPLEMENTATION_BINDING_PHASE = "deferred_to_a2_training_authority"
REQUIRED_SEEDS = (41, 42, 43)
REQUIRED_FOLDS = (0, 1, 2, 3, 4)
REQUIRED_EMBARGO_SECONDS = 30 * 24 * 60 * 60
REQUIRED_FUSION_FIELDS = (
    "loop151_oof_score",
    "whole_file_oof_score",
    "loop151_oof_uncertainty",
    "whole_file_oof_uncertainty",
    "loop151_missingness",
    "whole_file_missingness",
)
MISSINGNESS_REASONS = (
    "timeout",
    "unsupported",
    "read_failure",
    "parse_failure",
    "oversize",
)
FORBIDDEN_SPLIT_ROLES = (
    "val_a",
    "val_b",
    "test10k",
    "legacy_full_test",
    "sentinel",
    "confirmation",
    "certification",
)
DEFAULT_PATHS = {
    "proposal": PROJECT_ROOT
    / "manifests/roadmap_9997/loop164_whole_file_residual_expert/proposal.json",
    "contract": PROJECT_ROOT / "reports/roadmap_9997/loop164/full_pool_group_manifest.json",
    "isolation_receipt": PROJECT_ROOT
    / "reports/roadmap_9997/loop164/full_pool_isolation_validation.json",
    "scope_plan": PROJECT_ROOT / "reports/roadmap_9997/loop164/fold_scope_plan.json",
    "scope_plan_validation": PROJECT_ROOT
    / "reports/roadmap_9997/loop164/fold_scope_plan_validation.json",
    "implementation_manifest": PROJECT_ROOT
    / "reports/roadmap_9997/loop164/whole_file_expert_implementation_manifest.json",
    "loop151_train_oof_manifest": PROJECT_ROOT
    / "reports/roadmap_9997/loop164/loop151_train_oof_manifest.json",
    "resource_guard": PROJECT_ROOT / "reports/roadmap_9997/loop164/resource_guard.json",
    "input_bundle": PROJECT_ROOT
    / "reports/roadmap_9997/loop164/train_oof_input_bundle_manifest.json",
    "training_authorization": PROJECT_ROOT
    / "manifests/roadmap_9997/loop164_whole_file_residual_expert/a2_training_authorization.json",
    "training_final_lease": PROJECT_ROOT
    / "reports/roadmap_9997/loop164/training_lease_consumption.final.json",
}
TRAINING_LEASE_MARKER_DIRECTORY = PROJECT_ROOT / "reports/roadmap_9997/loop164/training_lease_consumptions"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def parse_utc(value: object) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("Expected explicit UTC timestamp")
    return parsed.astimezone(timezone.utc)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant: {value}")


def read_json_object(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
    )
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def resolve_path(value: object, *, base: Path = PROJECT_ROOT) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Path is empty")
    candidate = Path(text)
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _record(failures: Counter[str], condition: bool, code: str) -> None:
    if not condition:
        failures[code] += 1


def _require_exact_keys(
    payload: object,
    expected: set[str],
    *,
    label: str,
    failures: Counter[str],
) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        failures[f"{label}_not_object"] += 1
        return None
    actual = set(payload)
    if actual != expected:
        if expected - actual:
            failures[f"{label}_missing_fields"] += 1
        if actual - expected:
            failures[f"{label}_unexpected_fields"] += 1
    return payload


def _valid_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_positive_count(value: object) -> bool:
    return _valid_count(value) and int(value) > 0


def _validate_label_counts(
    payload: object,
    *,
    expected_total: int,
    label: str,
    failures: Counter[str],
) -> None:
    if not isinstance(payload, dict) or set(payload) != {"0", "1"}:
        failures[f"{label}_invalid"] += 1
        return
    if not all(_valid_positive_count(payload[key]) for key in ("0", "1")):
        failures[f"{label}_support_invalid"] += 1
        return
    if int(payload["0"]) + int(payload["1"]) != expected_total:
        failures[f"{label}_total_mismatch"] += 1


def _validate_hash_fields(
    payload: dict[str, Any], fields: Sequence[str], *, label: str, failures: Counter[str]
) -> None:
    for field_name in fields:
        if not is_sha256(payload.get(field_name)):
            failures[f"{label}_{field_name}_invalid"] += 1


def _binding_payload(
    binding: object,
    *,
    expected_path: Path,
    label: str,
    failures: Counter[str],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    binding_payload = _require_exact_keys(
        binding,
        {"path", "sha256"},
        label=f"binding_{label}",
        failures=failures,
    )
    if binding_payload is None:
        return None, None
    try:
        bound_path = resolve_path(binding_payload.get("path"))
    except ValueError:
        failures[f"binding_{label}_path_invalid"] += 1
        return None, None
    if bound_path != expected_path.resolve():
        failures[f"binding_{label}_path_mismatch"] += 1
        return None, None
    if not is_sha256(binding_payload.get("sha256")):
        failures[f"binding_{label}_sha256_invalid"] += 1
        return None, None
    try:
        payload, actual_sha256 = read_json_object(expected_path)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        failures[f"binding_{label}_unreadable"] += 1
        return None, None
    if actual_sha256 != str(binding_payload["sha256"]).casefold():
        failures[f"binding_{label}_sha256_mismatch"] += 1
        return None, None
    return payload, actual_sha256


def _empty_result() -> dict[str, Any]:
    return {
        "schema": VALIDATION_SCHEMA,
        "loop_id": LOOP_ID,
        "aggregate_only_verified": False,
        "authority_chain_verified": False,
        "partition_plan_verified": False,
        "nested_oof_verified": False,
        "binding_fingerprints": {},
        "coverage": {},
        "blockers": [],
        "ready_for": {
            "loop164_train_oof_data_boundary": False,
            "a2_training_authorization": False,
            "val_a": False,
            "val_b": False,
            "test10k": False,
            "full_test": False,
        },
        "decision": "block",
        "notes": [
            "The verifier accepts aggregate commitments and counts only; it never opens identities or predictions.",
            "A pass cannot select a model, authorize a new A2 run, or open any heldout split.",
        ],
    }


def _validate_proposal(payload: Optional[dict[str, Any]], failures: Counter[str]) -> None:
    if payload is None:
        return
    _record(failures, payload.get("loop_id") == LOOP_ID, "proposal_loop_id_invalid")
    _record(
        failures,
        payload.get("decision") == "propose_loop164_whole_file_residual_expert_no_execution",
        "proposal_decision_invalid",
    )


def _validate_contract(payload: Optional[dict[str, Any]], failures: Counter[str]) -> None:
    if payload is None:
        return
    _record(failures, payload.get("schema") == ISOLATION_CONTRACT_SCHEMA, "contract_schema_invalid")
    _record(failures, payload.get("loop_id") == LOOP_ID, "contract_loop_id_invalid")
    _record(
        failures,
        tuple(payload.get("model_input_fields") or []) == REQUIRED_FUSION_FIELDS,
        "contract_fusion_feature_allowlist_invalid",
    )
    feature_contract = payload.get("feature_contract")
    if not isinstance(feature_contract, dict):
        failures["contract_feature_contract_missing"] += 1
        return
    _record(
        failures,
        set(feature_contract)
        == {
            "schema",
            "feature_fields",
            "feature_matrix_receipt_required",
            "implementation_binding_phase",
        },
        "contract_feature_contract_shape_invalid",
    )
    _record(
        failures,
        feature_contract.get("schema") == FEATURE_CONTRACT_SCHEMA,
        "contract_feature_contract_schema_invalid",
    )
    _record(
        failures,
        tuple(feature_contract.get("feature_fields") or []) == REQUIRED_FUSION_FIELDS,
        "contract_feature_contract_feature_allowlist_invalid",
    )
    _record(
        failures,
        feature_contract.get("feature_matrix_receipt_required") is True,
        "contract_feature_matrix_receipt_not_required",
    )
    _record(
        failures,
        feature_contract.get("implementation_binding_phase") == IMPLEMENTATION_BINDING_PHASE,
        "contract_feature_contract_implementation_binding_phase_invalid",
    )


def _validate_isolation_receipt(
    payload: Optional[dict[str, Any]],
    *,
    contract_sha256: Optional[str],
    failures: Counter[str],
) -> tuple[Optional[str], Optional[int], Optional[int]]:
    if payload is None:
        return None, None, None
    _record(
        failures,
        payload.get("schema") == ISOLATION_RECEIPT_SCHEMA,
        "isolation_receipt_schema_invalid",
    )
    _record(failures, payload.get("loop_id") == LOOP_ID, "isolation_receipt_loop_id_invalid")
    _record(failures, payload.get("decision") == "pass", "isolation_receipt_not_pass")
    ready_for = payload.get("ready_for")
    _record(
        failures,
        isinstance(ready_for, dict) and ready_for.get("loop164_train_oof_partition") is True,
        "isolation_receipt_partition_not_ready",
    )
    binding_fingerprints = payload.get("binding_fingerprints")
    if not isinstance(binding_fingerprints, dict):
        failures["isolation_receipt_binding_fingerprints_missing"] += 1
    elif contract_sha256 is not None and (
        binding_fingerprints.get("contract_sha256") != contract_sha256
    ):
        failures["isolation_receipt_contract_binding_mismatch"] += 1
    provenance = _require_exact_keys(
        payload.get("a2_authorization_provenance"),
        {
            "schema",
            "authority_scope",
            "authorization_sha256",
            "trust_anchor_sha256",
            "trusted_key_fingerprint",
            "verification_receipt_sha256",
            "validator_source_closure_sha256",
            "runtime_python_sha256",
            "resource_guard_sha256",
            "canonical_argv_sha256",
            "lease_consumption_id",
            "lease_marker_sha256",
        },
        label="isolation_receipt_authorization_provenance",
        failures=failures,
    )
    if provenance is not None:
        _record(
            failures,
            provenance.get("schema") == ISOLATION_AUTHORIZATION_PROVENANCE_SCHEMA
            and provenance.get("authority_scope") == ISOLATION_METADATA_AUTHORITY_SCOPE
            and all(
                is_sha256(provenance.get(field_name))
                for field_name in (
                    "authorization_sha256",
                    "trust_anchor_sha256",
                    "trusted_key_fingerprint",
                    "verification_receipt_sha256",
                    "validator_source_closure_sha256",
                    "runtime_python_sha256",
                    "resource_guard_sha256",
                    "canonical_argv_sha256",
                    "lease_consumption_id",
                    "lease_marker_sha256",
                )
            ),
            "isolation_receipt_authorization_provenance_invalid",
        )
        _record(
            failures,
            isinstance(binding_fingerprints, dict)
            and binding_fingerprints.get("a2_authorization_sha256")
            == provenance.get("authorization_sha256")
            and binding_fingerprints.get("a2_trust_anchor_sha256")
            == provenance.get("trust_anchor_sha256")
            and binding_fingerprints.get("a2_validator_source_closure_sha256")
            == provenance.get("validator_source_closure_sha256")
            and binding_fingerprints.get("a2_runtime_python_sha256")
            == provenance.get("runtime_python_sha256")
            and binding_fingerprints.get("a2_resource_guard_sha256")
            == provenance.get("resource_guard_sha256")
            and binding_fingerprints.get("a2_canonical_argv_sha256")
            == provenance.get("canonical_argv_sha256")
            and binding_fingerprints.get("a2_lease_marker_sha256")
            == provenance.get("lease_marker_sha256")
            and binding_fingerprints.get("a2_lease_consumption_id")
            == provenance.get("lease_consumption_id"),
            "isolation_receipt_authorization_provenance_binding_mismatch",
        )
    oof = payload.get("oof")
    if not isinstance(oof, dict):
        failures["isolation_receipt_oof_missing"] += 1
        return None, None, None
    fingerprint = oof.get("fold_assignment_fingerprint")
    if not is_sha256(fingerprint):
        failures["isolation_receipt_fold_assignment_fingerprint_invalid"] += 1
        fingerprint = None
    eligible_rows = oof.get("eligible_rows")
    warmup_rows = oof.get("warmup_rows")
    if not _valid_positive_count(eligible_rows):
        failures["isolation_receipt_eligible_rows_invalid"] += 1
        eligible_rows = None
    if not _valid_positive_count(warmup_rows):
        failures["isolation_receipt_warmup_rows_invalid"] += 1
        warmup_rows = None
    return fingerprint, eligible_rows, warmup_rows


def _validate_scope_plan(
    payload: Optional[dict[str, Any]],
    *,
    contract_sha256: Optional[str],
    isolation_receipt_sha256: Optional[str],
    expected_fingerprint: Optional[str],
    expected_eligible_rows: Optional[int],
    expected_warmup_rows: Optional[int],
    failures: Counter[str],
) -> dict[int, dict[str, Any]]:
    if payload is None:
        return {}
    expected_fields = {
        "schema",
        "loop_id",
        "aggregate_only",
        "contract_sha256",
        "isolation_receipt_sha256",
        "fold_assignment_fingerprint",
        "seeds",
        "eligible_rows",
        "warmup_rows",
        "embargo_seconds",
        "outer_scopes",
        "custodian_attestation",
    }
    plan = _require_exact_keys(
        payload, expected_fields, label="scope_plan", failures=failures
    )
    if plan is None:
        return {}
    _record(failures, plan.get("schema") == SCOPE_PLAN_SCHEMA, "scope_plan_schema_invalid")
    _record(failures, plan.get("loop_id") == LOOP_ID, "scope_plan_loop_id_invalid")
    _record(failures, plan.get("aggregate_only") is True, "scope_plan_not_aggregate_only")
    _record(
        failures,
        plan.get("contract_sha256") == contract_sha256,
        "scope_plan_contract_binding_mismatch",
    )
    _record(
        failures,
        plan.get("isolation_receipt_sha256") == isolation_receipt_sha256,
        "scope_plan_isolation_receipt_binding_mismatch",
    )
    _record(
        failures,
        plan.get("fold_assignment_fingerprint") == expected_fingerprint,
        "scope_plan_fold_assignment_fingerprint_mismatch",
    )
    _record(failures, tuple(plan.get("seeds") or []) == REQUIRED_SEEDS, "scope_plan_seed_set_invalid")
    _record(
        failures,
        plan.get("eligible_rows") == expected_eligible_rows,
        "scope_plan_eligible_rows_mismatch",
    )
    _record(
        failures,
        plan.get("warmup_rows") == expected_warmup_rows,
        "scope_plan_warmup_rows_mismatch",
    )
    _record(
        failures,
        plan.get("embargo_seconds") == REQUIRED_EMBARGO_SECONDS,
        "scope_plan_embargo_invalid",
    )
    attestation = _require_exact_keys(
        plan.get("custodian_attestation"),
        {"attestation_id_sha256", "key_fingerprint", "verification_receipt_sha256"},
        label="scope_plan_custodian_attestation",
        failures=failures,
    )
    if attestation is not None:
        _validate_hash_fields(
            attestation,
            ("attestation_id_sha256", "key_fingerprint", "verification_receipt_sha256"),
            label="scope_plan_custodian_attestation",
            failures=failures,
        )

    outer_scopes = plan.get("outer_scopes")
    if not isinstance(outer_scopes, list) or len(outer_scopes) != len(REQUIRED_FOLDS):
        failures["scope_plan_outer_fold_count_invalid"] += 1
        return {}
    _record(
        failures,
        tuple(
            scope.get("outer_fold") if isinstance(scope, dict) else None
            for scope in outer_scopes
        )
        == REQUIRED_FOLDS,
        "scope_plan_outer_fold_order_invalid",
    )
    scopes: dict[int, dict[str, Any]] = {}
    previous_maximum: Optional[datetime] = None
    for outer_scope in outer_scopes:
        expected_scope_fields = {
            "outer_fold",
            "fit_scope_commitment",
            "holdout_scope_commitment",
            "fit_component_set_commitment",
            "holdout_component_set_commitment",
            "fit_rows",
            "holdout_rows",
            "fit_label_counts",
            "holdout_label_counts",
            "fit_max_component_time_utc",
            "holdout_min_component_time_utc",
            "holdout_max_component_time_utc",
            "inner_oof_meta_eligible_rows",
            "inner_oof_union_holdout_rows",
            "inner_oof_warmup_rows",
            "inner_oof_purged_rows",
            "overlap_audit",
            "inner_scopes",
        }
        scope = _require_exact_keys(
            outer_scope,
            expected_scope_fields,
            label="scope_plan_outer_scope",
            failures=failures,
        )
        if scope is None:
            continue
        fold_id = scope.get("outer_fold")
        if fold_id not in REQUIRED_FOLDS or fold_id in scopes:
            failures["scope_plan_outer_fold_id_invalid"] += 1
            continue
        scopes[fold_id] = scope
        _validate_hash_fields(
            scope,
            (
                "fit_scope_commitment",
                "holdout_scope_commitment",
                "fit_component_set_commitment",
                "holdout_component_set_commitment",
            ),
            label="scope_plan_outer_scope",
            failures=failures,
        )
        if not _valid_positive_count(scope.get("fit_rows")):
            failures["scope_plan_outer_fit_rows_invalid"] += 1
        if not _valid_positive_count(scope.get("holdout_rows")):
            failures["scope_plan_outer_holdout_rows_invalid"] += 1
        if _valid_positive_count(scope.get("fit_rows")):
            _validate_label_counts(
                scope.get("fit_label_counts"),
                expected_total=int(scope["fit_rows"]),
                label="scope_plan_outer_fit_label_counts",
                failures=failures,
            )
        if _valid_positive_count(scope.get("holdout_rows")):
            _validate_label_counts(
                scope.get("holdout_label_counts"),
                expected_total=int(scope["holdout_rows"]),
                label="scope_plan_outer_holdout_label_counts",
                failures=failures,
            )
        try:
            fit_maximum = parse_utc(scope.get("fit_max_component_time_utc"))
            holdout_minimum = parse_utc(scope.get("holdout_min_component_time_utc"))
            holdout_maximum = parse_utc(scope.get("holdout_max_component_time_utc"))
        except ValueError:
            failures["scope_plan_outer_temporal_bounds_invalid"] += 1
            holdout_minimum = None
            holdout_maximum = None
        else:
            if holdout_minimum > holdout_maximum:
                failures["scope_plan_outer_temporal_bounds_order_invalid"] += 1
            if fit_maximum + timedelta(seconds=REQUIRED_EMBARGO_SECONDS) > holdout_minimum:
                failures["scope_plan_outer_fit_temporal_or_embargo_invalid"] += 1
            if previous_maximum is not None and holdout_minimum < previous_maximum + timedelta(
                seconds=REQUIRED_EMBARGO_SECONDS
            ):
                failures["scope_plan_outer_temporal_or_embargo_invalid"] += 1
            previous_maximum = holdout_maximum
        inner_meta_eligible_rows = scope.get("inner_oof_meta_eligible_rows")
        inner_union_holdout_rows = scope.get("inner_oof_union_holdout_rows")
        inner_warmup_rows = scope.get("inner_oof_warmup_rows")
        inner_purged_rows = scope.get("inner_oof_purged_rows")
        if not all(
            _valid_count(value)
            for value in (
                inner_meta_eligible_rows,
                inner_union_holdout_rows,
                inner_warmup_rows,
                inner_purged_rows,
            )
        ):
            failures["scope_plan_inner_oof_accounting_invalid"] += 1
        elif inner_union_holdout_rows != inner_meta_eligible_rows:
            failures["scope_plan_inner_oof_union_coverage_invalid"] += 1
        elif _valid_positive_count(scope.get("fit_rows")) and (
            int(inner_meta_eligible_rows)
            + int(inner_warmup_rows)
            + int(inner_purged_rows)
            != int(scope["fit_rows"])
        ):
            failures["scope_plan_inner_oof_partition_accounting_invalid"] += 1
        overlap_audit = _require_exact_keys(
            scope.get("overlap_audit"),
            {
                "outer_fit_holdout_row_overlap",
                "outer_fit_holdout_component_overlap",
                "prior_outer_holdout_component_overlap",
            },
            label="scope_plan_outer_overlap_audit",
            failures=failures,
        )
        if overlap_audit is not None and any(
            overlap_audit.get(field_name) != 0 for field_name in overlap_audit
        ):
            failures["scope_plan_outer_overlap_detected"] += 1
        inner_scopes = scope.get("inner_scopes")
        if not isinstance(inner_scopes, list) or len(inner_scopes) != len(REQUIRED_FOLDS):
            failures["scope_plan_inner_fold_count_invalid"] += 1
            continue
        seen_inner_folds: set[int] = set()
        inner_holdout_rows_total = 0
        for inner_scope in inner_scopes:
            expected_inner_fields = {
                "inner_fold",
                "fit_scope_commitment",
                "holdout_scope_commitment",
                "parent_outer_fit_scope_commitment",
                "fit_component_set_commitment",
                "holdout_component_set_commitment",
                "fit_rows",
                "holdout_rows",
                "fit_label_counts",
                "holdout_label_counts",
                "fit_max_component_time_utc",
                "holdout_min_component_time_utc",
                "holdout_max_component_time_utc",
                "overlap_audit",
            }
            inner = _require_exact_keys(
                inner_scope,
                expected_inner_fields,
                label="scope_plan_inner_scope",
                failures=failures,
            )
            if inner is None:
                continue
            inner_fold = inner.get("inner_fold")
            if inner_fold not in REQUIRED_FOLDS or inner_fold in seen_inner_folds:
                failures["scope_plan_inner_fold_id_invalid"] += 1
                continue
            seen_inner_folds.add(inner_fold)
            _validate_hash_fields(
                inner,
                (
                    "fit_scope_commitment",
                    "holdout_scope_commitment",
                    "parent_outer_fit_scope_commitment",
                    "fit_component_set_commitment",
                    "holdout_component_set_commitment",
                ),
                label="scope_plan_inner_scope",
                failures=failures,
            )
            _record(
                failures,
                inner.get("parent_outer_fit_scope_commitment")
                == scope.get("fit_scope_commitment"),
                "scope_plan_inner_parent_outer_fit_mismatch",
            )
            if not _valid_positive_count(inner.get("fit_rows")):
                failures["scope_plan_inner_fit_rows_invalid"] += 1
            if not _valid_positive_count(inner.get("holdout_rows")):
                failures["scope_plan_inner_holdout_rows_invalid"] += 1
            if _valid_positive_count(inner.get("fit_rows")):
                _validate_label_counts(
                    inner.get("fit_label_counts"),
                    expected_total=int(inner["fit_rows"]),
                    label="scope_plan_inner_fit_label_counts",
                    failures=failures,
                )
            if _valid_positive_count(inner.get("holdout_rows")):
                inner_holdout_rows_total += int(inner["holdout_rows"])
                _validate_label_counts(
                    inner.get("holdout_label_counts"),
                    expected_total=int(inner["holdout_rows"]),
                    label="scope_plan_inner_holdout_label_counts",
                    failures=failures,
                )
            try:
                inner_fit_maximum = parse_utc(inner.get("fit_max_component_time_utc"))
                inner_holdout_minimum = parse_utc(inner.get("holdout_min_component_time_utc"))
                inner_holdout_maximum = parse_utc(inner.get("holdout_max_component_time_utc"))
            except ValueError:
                failures["scope_plan_inner_temporal_bounds_invalid"] += 1
            else:
                if inner_holdout_minimum > inner_holdout_maximum:
                    failures["scope_plan_inner_temporal_bounds_order_invalid"] += 1
                if inner_fit_maximum + timedelta(
                    seconds=REQUIRED_EMBARGO_SECONDS
                ) > inner_holdout_minimum:
                    failures["scope_plan_inner_temporal_or_embargo_invalid"] += 1
                if holdout_minimum is not None and inner_holdout_maximum >= holdout_minimum:
                    failures["scope_plan_inner_scope_not_before_outer_holdout"] += 1
            inner_overlap = _require_exact_keys(
                inner.get("overlap_audit"),
                {
                    "fit_holdout_row_overlap",
                    "fit_holdout_component_overlap",
                    "outer_holdout_component_overlap",
                    "inner_fit_outside_parent_outer_fit_components",
                    "inner_holdout_outside_parent_outer_fit_components",
                },
                label="scope_plan_inner_overlap_audit",
                failures=failures,
            )
            if inner_overlap is not None and any(
                inner_overlap.get(field_name) != 0 for field_name in inner_overlap
            ):
                failures["scope_plan_inner_overlap_detected"] += 1
        if seen_inner_folds != set(REQUIRED_FOLDS):
            failures["scope_plan_inner_fold_coverage_invalid"] += 1
        if inner_union_holdout_rows is not None and inner_holdout_rows_total != inner_union_holdout_rows:
            failures["scope_plan_inner_oof_holdout_total_mismatch"] += 1
    if set(scopes) != set(REQUIRED_FOLDS):
        failures["scope_plan_outer_fold_coverage_invalid"] += 1
    if expected_eligible_rows is not None and sum(
        int(scope.get("holdout_rows", 0)) for scope in scopes.values()
    ) != expected_eligible_rows:
        failures["scope_plan_eligible_outer_holdout_coverage_invalid"] += 1
    return scopes


def _infer_implementation_project_root(manifest_path: Path) -> Path:
    for candidate in (manifest_path.parent, *manifest_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return manifest_path.parent


def _validate_implementation_manifest(
    payload: Optional[dict[str, Any]], *, root: Path, failures: Counter[str]
) -> Optional[WholeFileImplementationManifestResult]:
    if payload is None:
        return None
    result = validate_implementation_manifest_payload(payload, root=root)
    for blocker in result.blockers:
        failures[blocker] += 1
    return result


def _validate_loop151_train_oof_manifest(
    payload: Optional[dict[str, Any]],
    *,
    fold_assignment_fingerprint: Optional[str],
    failures: Counter[str],
) -> None:
    if payload is None:
        return
    expected_fields = {
        "schema",
        "loop_id",
        "model_id",
        "train_only",
        "initialization_policy",
        "recipe_sha256",
        "runtime_lock_sha256",
        "fold_assignment_fingerprint",
        "seeds",
        "feature_fields",
    }
    manifest = _require_exact_keys(
        payload, expected_fields, label="loop151_train_oof_manifest", failures=failures
    )
    if manifest is None:
        return
    _record(
        failures,
        manifest.get("schema") == LOOP151_TRAIN_OOF_MANIFEST_SCHEMA,
        "loop151_train_oof_manifest_schema_invalid",
    )
    _record(
        failures,
        manifest.get("loop_id") == LOOP_ID,
        "loop151_train_oof_manifest_loop_id_invalid",
    )
    _record(
        failures,
        manifest.get("model_id") == "loop151_equivalent",
        "loop151_train_oof_manifest_model_id_invalid",
    )
    _record(
        failures, manifest.get("train_only") is True, "loop151_train_oof_manifest_not_train_only"
    )
    _record(
        failures,
        manifest.get("initialization_policy") == "from_scratch",
        "loop151_train_oof_manifest_initialization_invalid",
    )
    _record(
        failures,
        manifest.get("fold_assignment_fingerprint") == fold_assignment_fingerprint,
        "loop151_train_oof_manifest_fold_fingerprint_mismatch",
    )
    _record(
        failures,
        tuple(manifest.get("seeds") or []) == REQUIRED_SEEDS,
        "loop151_train_oof_manifest_seed_set_invalid",
    )
    _record(
        failures,
        tuple(manifest.get("feature_fields") or []) == REQUIRED_FUSION_FIELDS,
        "loop151_train_oof_manifest_feature_allowlist_invalid",
    )
    _validate_hash_fields(
        manifest,
        ("recipe_sha256", "runtime_lock_sha256"),
        label="loop151_train_oof_manifest",
        failures=failures,
    )


def _validate_resource_guard(payload: Optional[dict[str, Any]], failures: Counter[str]) -> None:
    if payload is None:
        return
    expected_fields = {
        "schema",
        "operation",
        "guard_ready",
        "decision",
        "receipt",
    }
    guard = _require_exact_keys(payload, expected_fields, label="training_resource_guard", failures=failures)
    if guard is None:
        return
    _record(
        failures,
        guard.get("schema") == "axon_loop164_train_oof_resource_guard_v1",
        "training_resource_guard_schema_invalid",
    )
    _record(
        failures,
        guard.get("operation") == "loop164_three_seed_nested_train_oof",
        "training_resource_guard_operation_invalid",
    )
    _record(
        failures,
        guard.get("guard_ready") is True and guard.get("decision") == "pass",
        "training_resource_guard_not_ready",
    )
    receipt = _require_exact_keys(
        guard.get("receipt"),
        {"created_at_utc", "controller_sha256", "resource_budget_sha256"},
        label="training_resource_guard_receipt",
        failures=failures,
    )
    if receipt is not None:
        try:
            parse_utc(receipt.get("created_at_utc"))
        except ValueError:
            failures["training_resource_guard_timestamp_invalid"] += 1
        _validate_hash_fields(
            receipt,
            ("controller_sha256", "resource_budget_sha256"),
            label="training_resource_guard_receipt",
            failures=failures,
        )


def _validate_training_authorization(
    payload: Optional[dict[str, Any]],
    *,
    authorization_sha256: Optional[str],
    receipt_path: Path,
    final_lease_path: Path,
    completed_at_utc: Optional[datetime],
    binding_sha256: dict[str, Optional[str]],
    fold_assignment_fingerprint: Optional[str],
    scope_plan_payload: Optional[dict[str, Any]],
    failures: Counter[str],
) -> tuple[Optional[str], Optional[str], Optional[list[str]]]:
    if payload is None:
        return None, None, None
    expected_fields = {
        "schema",
        "loop_id",
        "authorization_level",
        "decision",
        "execution_environment",
        "operation",
        "issued_at_utc",
        "not_before_utc",
        "expires_at_utc",
        "authority_attestation",
        "runtime_binding",
        "canonical_argv",
        "allowed_split_roles",
        "forbidden_split_roles",
        "feature_fields",
        "fold_assignment_fingerprint",
        "bindings",
        "output_binding",
        "one_shot_lease",
    }
    authorization = _require_exact_keys(
        payload, expected_fields, label="training_authorization", failures=failures
    )
    if authorization is None:
        return None, None, None
    expected_header = {
        "schema": TRAINING_AUTHORIZATION_SCHEMA,
        "loop_id": LOOP_ID,
        "authorization_level": "A2_train_only_nested_oof",
        "decision": "allow_single_loop164_train_oof_execution",
        "execution_environment": "custodian_side_train_only",
        "operation": "loop164_three_seed_nested_train_oof",
    }
    for field_name, expected_value in expected_header.items():
        _record(
            failures,
            authorization.get(field_name) == expected_value,
            f"training_authorization_{field_name}_invalid",
        )
    try:
        issued_at = parse_utc(authorization.get("issued_at_utc"))
        not_before = parse_utc(authorization.get("not_before_utc"))
        expires_at = parse_utc(authorization.get("expires_at_utc"))
    except ValueError:
        failures["training_authorization_time_window_invalid"] += 1
    else:
        if issued_at > not_before or not_before >= expires_at:
            failures["training_authorization_time_window_order_invalid"] += 1
        if expires_at - issued_at > timedelta(hours=24):
            failures["training_authorization_ttl_exceeds_maximum"] += 1
        if completed_at_utc is None or not_before > completed_at_utc or completed_at_utc > expires_at:
            failures["training_authorization_execution_outside_window"] += 1
    attestation = _require_exact_keys(
        authorization.get("authority_attestation"),
        {"key_fingerprint", "verification_receipt_sha256", "verification_state"},
        label="training_authorization_attestation",
        failures=failures,
    )
    if attestation is not None:
        _validate_hash_fields(
            attestation,
            ("key_fingerprint", "verification_receipt_sha256"),
            label="training_authorization_attestation",
            failures=failures,
        )
        _record(
            failures,
            attestation.get("verification_state") == "externally_verified",
            "training_authorization_external_attestation_not_verified",
        )
        if scope_plan_payload is not None:
            scope_attestation = scope_plan_payload.get("custodian_attestation")
            if isinstance(scope_attestation, dict):
                _record(
                    failures,
                    attestation.get("key_fingerprint") == scope_attestation.get("key_fingerprint"),
                    "training_authorization_scope_attestation_key_mismatch",
                )
    runtime_binding = _require_exact_keys(
        authorization.get("runtime_binding"),
        {"cwd", "python_executable_sha256", "controller_sha256"},
        label="training_authorization_runtime_binding",
        failures=failures,
    )
    if runtime_binding is not None:
        _record(
            failures,
            runtime_binding.get("cwd") == str(PROJECT_ROOT.resolve()),
            "training_authorization_runtime_cwd_invalid",
        )
        _validate_hash_fields(
            runtime_binding,
            ("python_executable_sha256", "controller_sha256"),
            label="training_authorization_runtime_binding",
            failures=failures,
        )
    argv = authorization.get("canonical_argv")
    if not isinstance(argv, list) or not argv or any(
        not isinstance(value, str) or not value for value in argv
    ):
        failures["training_authorization_canonical_argv_invalid"] += 1
        argv = None
    _record(
        failures,
        authorization.get("allowed_split_roles") == ["train_anchor", "train_oof"],
        "training_authorization_allowed_split_roles_invalid",
    )
    _record(
        failures,
        tuple(authorization.get("forbidden_split_roles") or []) == FORBIDDEN_SPLIT_ROLES,
        "training_authorization_forbidden_split_roles_invalid",
    )
    _record(
        failures,
        tuple(authorization.get("feature_fields") or []) == REQUIRED_FUSION_FIELDS,
        "training_authorization_feature_allowlist_invalid",
    )
    _record(
        failures,
        authorization.get("fold_assignment_fingerprint") == fold_assignment_fingerprint,
        "training_authorization_fold_fingerprint_mismatch",
    )
    bindings = authorization.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(binding_sha256):
        failures["training_authorization_bindings_shape_invalid"] += 1
    else:
        for name, expected_sha256 in binding_sha256.items():
            if bindings.get(name) != expected_sha256:
                failures[f"training_authorization_{name}_binding_mismatch"] += 1
    output_binding = _require_exact_keys(
        authorization.get("output_binding"),
        {"path"},
        label="training_authorization_output_binding",
        failures=failures,
    )
    if output_binding is not None:
        try:
            output_path = resolve_path(output_binding.get("path"))
        except ValueError:
            failures["training_authorization_output_path_invalid"] += 1
        else:
            _record(
                failures,
                output_path == receipt_path.resolve(),
                "training_authorization_output_path_mismatch",
            )
    lease = _require_exact_keys(
        authorization.get("one_shot_lease"),
        {"lease_id", "purpose", "state", "final_lease_path"},
        label="training_authorization_lease",
        failures=failures,
    )
    if lease is None:
        return authorization_sha256, None, argv
    lease_id = str(lease.get("lease_id") or "").strip()
    if not lease_id or len(lease_id) > 128:
        failures["training_authorization_lease_id_invalid"] += 1
    _record(
        failures,
        lease.get("purpose") == "single_loop164_three_seed_nested_train_oof",
        "training_authorization_lease_purpose_invalid",
    )
    _record(
        failures,
        lease.get("state") == "ready",
        "training_authorization_lease_state_invalid",
    )
    try:
        bound_final_lease_path = resolve_path(lease.get("final_lease_path"))
    except ValueError:
        failures["training_authorization_final_lease_path_invalid"] += 1
    else:
        _record(
            failures,
            bound_final_lease_path == final_lease_path.resolve(),
            "training_authorization_final_lease_path_mismatch",
        )
    return authorization_sha256, lease_id or None, argv


def _canonical_argv_sha256(argv: list[str]) -> str:
    encoded = json.dumps(argv, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_scope_plan_validation_v2(
    payload: Optional[dict[str, Any]],
    *,
    proposal_sha256: Optional[str],
    contract_sha256: Optional[str],
    isolation_receipt_sha256: Optional[str],
    scope_plan_sha256: Optional[str],
    fold_assignment_fingerprint: Optional[str],
    failures: Counter[str],
) -> None:
    if payload is None:
        return
    validation = _require_exact_keys(
        payload,
        {
            "schema",
            "loop_id",
            "aggregate_only_verified",
            "proposal_binding_verified",
            "contract_binding_verified",
            "isolation_receipt_binding_verified",
            "scope_plan_binding_verified",
            "binding_fingerprints",
            "plan_summary",
            "blockers",
            "ready_for",
            "decision",
            "notes",
        },
        label="scope_plan_validation",
        failures=failures,
    )
    if validation is None:
        return
    _record(
        failures,
        validation.get("schema") == SCOPE_PLAN_VALIDATION_SCHEMA
        and validation.get("loop_id") == LOOP_ID,
        "scope_plan_validation_schema_invalid",
    )
    _record(
        failures,
        validation.get("decision") == "pass"
        and validation.get("aggregate_only_verified") is True
        and validation.get("proposal_binding_verified") is True
        and validation.get("contract_binding_verified") is True
        and validation.get("isolation_receipt_binding_verified") is True
        and validation.get("scope_plan_binding_verified") is True,
        "scope_plan_validation_not_passed",
    )
    ready_for = validation.get("ready_for")
    _record(
        failures,
        isinstance(ready_for, dict)
        and ready_for.get("fold_scope_frozen") is True
        and ready_for.get("a2_training_authorization") is False
        and ready_for.get("train_oof") is False,
        "scope_plan_validation_ready_state_invalid",
    )
    fingerprints = validation.get("binding_fingerprints")
    _record(
        failures,
        isinstance(fingerprints, dict)
        and fingerprints
        == {
            "proposal_sha256": proposal_sha256,
            "contract_sha256": contract_sha256,
            "isolation_receipt_sha256": isolation_receipt_sha256,
            "scope_plan_sha256": scope_plan_sha256,
        },
        "scope_plan_validation_binding_mismatch",
    )
    summary = validation.get("plan_summary")
    _record(
        failures,
        isinstance(summary, dict)
        and summary.get("fold_assignment_fingerprint") == fold_assignment_fingerprint,
        "scope_plan_validation_fold_fingerprint_mismatch",
    )


def _validate_input_bundle_v2(
    payload: Optional[dict[str, Any]],
    *,
    scope_validation_sha256: Optional[str],
    fold_assignment_fingerprint: Optional[str],
    failures: Counter[str],
) -> None:
    if payload is None:
        return
    bundle = _require_exact_keys(
        payload,
        {
            "schema",
            "loop_id",
            "allowed_split_roles",
            "forbidden_split_roles",
            "feature_fields",
            "fold_assignment_fingerprint",
            "scope_plan_validation_sha256",
            "protected_input_open_policy",
            "input_artifact_commitments",
        },
        label="training_input_bundle",
        failures=failures,
    )
    if bundle is None:
        return
    _record(
        failures,
        bundle.get("schema") == INPUT_BUNDLE_SCHEMA and bundle.get("loop_id") == LOOP_ID,
        "training_input_bundle_schema_invalid",
    )
    _record(
        failures,
        bundle.get("allowed_split_roles") == ["train_anchor", "train_oof"]
        and tuple(bundle.get("forbidden_split_roles") or []) == FORBIDDEN_SPLIT_ROLES,
        "training_input_bundle_split_roles_invalid",
    )
    _record(
        failures,
        tuple(bundle.get("feature_fields") or []) == REQUIRED_FUSION_FIELDS,
        "training_input_bundle_feature_allowlist_invalid",
    )
    _record(
        failures,
        bundle.get("fold_assignment_fingerprint") == fold_assignment_fingerprint,
        "training_input_bundle_fold_fingerprint_mismatch",
    )
    _record(
        failures,
        bundle.get("scope_plan_validation_sha256") == scope_validation_sha256,
        "training_input_bundle_scope_validation_mismatch",
    )
    _record(
        failures,
        bundle.get("protected_input_open_policy") == "after_final_lease_only",
        "training_input_bundle_open_policy_invalid",
    )
    commitments = bundle.get("input_artifact_commitments")
    if not isinstance(commitments, dict) or set(commitments) != {
        "train_anchor_sha256",
        "train_oof_sha256",
    }:
        failures["training_input_bundle_commitments_invalid"] += 1
    else:
        _validate_hash_fields(
            commitments,
            ("train_anchor_sha256", "train_oof_sha256"),
            label="training_input_bundle_commitments",
            failures=failures,
        )


def _validate_resource_guard_v2(
    payload: Optional[dict[str, Any]],
    *,
    runtime_binding: Optional[dict[str, Any]],
    canonical_argv_sha256: Optional[str],
    implementation_manifest_sha256: Optional[str],
    implementation_contract: Optional[WholeFileImplementationManifestResult],
    completed_at_utc: Optional[datetime],
    max_age_seconds: object,
    failures: Counter[str],
) -> None:
    if payload is None:
        return
    guard = _require_exact_keys(
        payload,
        {
            "schema",
            "loop_id",
            "operation",
            "guard_ready",
            "decision",
            "runtime_binding",
            "implementation_binding",
            "receipt",
        },
        label="training_resource_guard",
        failures=failures,
    )
    if guard is None:
        return
    _record(
        failures,
        guard.get("schema") == RESOURCE_GUARD_SCHEMA
        and guard.get("loop_id") == LOOP_ID
        and guard.get("operation") == "loop164_three_seed_nested_train_oof"
        and guard.get("guard_ready") is True
        and guard.get("decision") == "pass",
        "training_resource_guard_not_ready",
    )
    guard_runtime = _require_exact_keys(
        guard.get("runtime_binding"),
        {"cwd", "python_sha256", "controller_path", "controller_sha256", "canonical_argv_sha256"},
        label="training_resource_guard_runtime_binding",
        failures=failures,
    )
    if guard_runtime is not None and runtime_binding is not None:
        _record(
            failures,
            guard_runtime.get("cwd") == runtime_binding.get("cwd")
            and guard_runtime.get("python_sha256") == runtime_binding.get("python_sha256")
            and guard_runtime.get("controller_path") == runtime_binding.get("controller_path")
            and guard_runtime.get("controller_sha256") == runtime_binding.get("controller_sha256")
            and guard_runtime.get("canonical_argv_sha256") == canonical_argv_sha256,
            "training_resource_guard_runtime_binding_mismatch",
        )
    implementation_binding = _require_exact_keys(
        guard.get("implementation_binding"),
        {"implementation_manifest_sha256", "source_closure_sha256", "memory_contract_sha256"},
        label="training_resource_guard_implementation_binding",
        failures=failures,
    )
    if implementation_binding is not None:
        _record(
            failures,
            implementation_contract is not None
            and implementation_binding.get("implementation_manifest_sha256")
            == implementation_manifest_sha256
            and implementation_binding.get("source_closure_sha256")
            == implementation_contract.source_closure_sha256
            and implementation_binding.get("memory_contract_sha256")
            == implementation_contract.memory_contract_sha256,
            "training_resource_guard_implementation_binding_mismatch",
        )
    receipt = _require_exact_keys(
        guard.get("receipt"),
        {"created_at_utc", "controller_sha256", "resource_budget_sha256"},
        label="training_resource_guard_receipt",
        failures=failures,
    )
    if receipt is None:
        return
    try:
        created_at = parse_utc(receipt.get("created_at_utc"))
        age_limit = int(max_age_seconds)
    except (TypeError, ValueError):
        failures["training_resource_guard_age_invalid"] += 1
    else:
        _record(
            failures,
            completed_at_utc is not None
            and age_limit >= 1
            and timedelta(0) <= completed_at_utc - created_at <= timedelta(seconds=age_limit),
            "training_resource_guard_stale",
        )
    _validate_hash_fields(
        receipt,
        ("controller_sha256", "resource_budget_sha256"),
        label="training_resource_guard_receipt",
        failures=failures,
    )
    if runtime_binding is not None:
        _record(
            failures,
            receipt.get("controller_sha256") == runtime_binding.get("controller_sha256"),
            "training_resource_guard_controller_mismatch",
        )


def _validate_training_authorization_v2(
    payload: Optional[dict[str, Any]],
    *,
    authorization_sha256: Optional[str],
    receipt_path: Path,
    final_lease_path: Path,
    lease_marker_directory: Path,
    completed_at_utc: Optional[datetime],
    binding_sha256: dict[str, Optional[str]],
    binding_paths: dict[str, Path],
    fold_assignment_fingerprint: Optional[str],
    failures: Counter[str],
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if payload is None:
        return context
    expected_fields = {
        "schema",
        "loop_id",
        "authorization_level",
        "decision",
        "execution_environment",
        "operation",
        "issued_at_utc",
        "not_before_utc",
        "expires_at_utc",
        "authority_attestation",
        "runtime_binding",
        "canonical_argv",
        "allowed_split_roles",
        "forbidden_split_roles",
        "feature_fields",
        "fold_assignment_fingerprint",
        "outer_run_budget",
        "bindings",
        "output_binding",
        "one_shot_lease",
        "max_resource_guard_age_seconds",
    }
    authorization = _require_exact_keys(
        payload, expected_fields, label="training_authorization", failures=failures
    )
    if authorization is None:
        return context
    expected_header = {
        "schema": TRAINING_AUTHORIZATION_SCHEMA,
        "loop_id": LOOP_ID,
        "authorization_level": "A2_train_only_nested_oof",
        "decision": "allow_single_loop164_train_oof_execution",
        "execution_environment": "custodian_side_train_only",
        "operation": "loop164_three_seed_nested_train_oof",
    }
    for field_name, expected_value in expected_header.items():
        _record(
            failures,
            authorization.get(field_name) == expected_value,
            f"training_authorization_{field_name}_invalid",
        )
    try:
        issued_at = parse_utc(authorization.get("issued_at_utc"))
        not_before = parse_utc(authorization.get("not_before_utc"))
        expires_at = parse_utc(authorization.get("expires_at_utc"))
    except ValueError:
        failures["training_authorization_time_window_invalid"] += 1
    else:
        if issued_at > not_before or not_before >= expires_at:
            failures["training_authorization_time_window_order_invalid"] += 1
        if expires_at - issued_at > timedelta(hours=24):
            failures["training_authorization_ttl_exceeds_maximum"] += 1
        if completed_at_utc is None or not_before > completed_at_utc or completed_at_utc > expires_at:
            failures["training_authorization_execution_outside_window"] += 1
    attestation = _require_exact_keys(
        authorization.get("authority_attestation"),
        {"trusted_key_fingerprint", "trust_anchor_sha256", "verification_receipt_sha256"},
        label="training_authorization_attestation",
        failures=failures,
    )
    if attestation is not None:
        _validate_hash_fields(
            attestation,
            ("trusted_key_fingerprint", "trust_anchor_sha256", "verification_receipt_sha256"),
            label="training_authorization_attestation",
            failures=failures,
        )
    runtime_binding = _require_exact_keys(
        authorization.get("runtime_binding"),
        {"cwd", "python_executable", "python_sha256", "controller_path", "controller_sha256", "entrypoint"},
        label="training_authorization_runtime_binding",
        failures=failures,
    )
    if runtime_binding is not None:
        _record(
            failures,
            isinstance(runtime_binding.get("cwd"), str)
            and runtime_binding.get("cwd")
            and isinstance(runtime_binding.get("python_executable"), str)
            and runtime_binding.get("python_executable")
            and isinstance(runtime_binding.get("controller_path"), str)
            and runtime_binding.get("controller_path")
            and runtime_binding.get("entrypoint") == "run_loop164_train_oof_controller.main",
            "training_authorization_runtime_path_invalid",
        )
        _validate_hash_fields(
            runtime_binding,
            ("python_sha256", "controller_sha256"),
            label="training_authorization_runtime_binding",
            failures=failures,
        )
    argv = authorization.get("canonical_argv")
    if not isinstance(argv, list) or not argv or any(
        not isinstance(value, str) or not value for value in argv
    ):
        failures["training_authorization_canonical_argv_invalid"] += 1
        argv = None
    _record(
        failures,
        authorization.get("allowed_split_roles") == ["train_anchor", "train_oof"]
        and tuple(authorization.get("forbidden_split_roles") or []) == FORBIDDEN_SPLIT_ROLES,
        "training_authorization_split_roles_invalid",
    )
    _record(
        failures,
        tuple(authorization.get("feature_fields") or []) == REQUIRED_FUSION_FIELDS,
        "training_authorization_feature_allowlist_invalid",
    )
    _record(
        failures,
        authorization.get("fold_assignment_fingerprint") == fold_assignment_fingerprint,
        "training_authorization_fold_fingerprint_mismatch",
    )
    _record(
        failures,
        authorization.get("outer_run_budget") == len(REQUIRED_SEEDS) * len(REQUIRED_FOLDS),
        "training_authorization_outer_run_budget_invalid",
    )
    bindings = authorization.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(binding_sha256):
        failures["training_authorization_bindings_shape_invalid"] += 1
    else:
        for name, expected_sha256 in binding_sha256.items():
            binding = _require_exact_keys(
                bindings.get(name),
                {"path", "sha256"},
                label=f"training_authorization_binding_{name}",
                failures=failures,
            )
            if binding is None:
                continue
            try:
                bound_path = resolve_path(binding.get("path"))
            except ValueError:
                failures[f"training_authorization_binding_{name}_path_invalid"] += 1
                continue
            _record(
                failures,
                bound_path == binding_paths[name].resolve()
                and binding.get("sha256") == expected_sha256,
                f"training_authorization_binding_{name}_mismatch",
            )
    output_binding = _require_exact_keys(
        authorization.get("output_binding"),
        {"execution_receipt_path", "final_lease_path", "lease_marker_directory"},
        label="training_authorization_output_binding",
        failures=failures,
    )
    if output_binding is not None:
        try:
            execution_path = resolve_path(output_binding.get("execution_receipt_path"))
            final_lease_bound_path = resolve_path(output_binding.get("final_lease_path"))
            marker_directory = resolve_path(output_binding.get("lease_marker_directory"))
        except ValueError:
            failures["training_authorization_output_path_invalid"] += 1
        else:
            _record(
                failures,
                execution_path == receipt_path.resolve()
                and final_lease_bound_path == final_lease_path.resolve()
                and marker_directory == lease_marker_directory.resolve(),
                "training_authorization_output_path_mismatch",
            )
    lease = _require_exact_keys(
        authorization.get("one_shot_lease"),
        {"lease_id", "purpose", "state"},
        label="training_authorization_lease",
        failures=failures,
    )
    lease_id: Optional[str] = None
    if lease is not None:
        parsed_lease_id = str(lease.get("lease_id") or "").strip()
        if not parsed_lease_id or len(parsed_lease_id) > 128:
            failures["training_authorization_lease_id_invalid"] += 1
        else:
            lease_id = parsed_lease_id
        _record(
            failures,
            lease.get("purpose") == "single_loop164_three_seed_nested_train_oof"
            and lease.get("state") == "ready",
            "training_authorization_lease_invalid",
        )
    context.update(
        {
            "authorization_sha256": authorization_sha256,
            "lease_id": lease_id,
            "canonical_argv": argv,
            "canonical_argv_sha256": _canonical_argv_sha256(argv) if argv is not None else None,
            "controller_sha256": runtime_binding.get("controller_sha256")
            if runtime_binding is not None
            else None,
            "runtime_binding": runtime_binding,
            "max_resource_guard_age_seconds": authorization.get("max_resource_guard_age_seconds"),
        }
    )
    return context


def _validate_fit_artifact(
    payload: object,
    *,
    expected_stage: str,
    expected_expert: str,
    expected_seed: int,
    expected_outer_fold: int,
    expected_inner_fold: Optional[int],
    expected_fit_scope: str,
    expected_output_scope: str,
    label: str,
    whole_file_contract: Optional[WholeFileImplementationManifestResult],
    input_bundle_sha256: Optional[str],
    seen_model_artifacts: dict[str, tuple[str, int, int, Optional[int]]],
    failures: Counter[str],
) -> Optional[dict[str, Any]]:
    expected_fields = {
        "run_id_sha256",
        "stage",
        "expert_id",
        "seed",
        "outer_fold",
        "inner_fold",
        "fit_scope_commitment",
        "output_scope_commitment",
        "model_artifact_sha256",
        "config_sha256",
        "code_sha256",
        "input_manifest_sha256",
        "depends_on",
    }
    artifact = _require_exact_keys(payload, expected_fields, label=label, failures=failures)
    if artifact is None:
        return None
    _record(failures, artifact.get("stage") == expected_stage, f"{label}_stage_invalid")
    _record(failures, artifact.get("expert_id") == expected_expert, f"{label}_expert_invalid")
    _record(failures, artifact.get("seed") == expected_seed, f"{label}_seed_invalid")
    _record(
        failures, artifact.get("outer_fold") == expected_outer_fold, f"{label}_outer_fold_invalid"
    )
    _record(
        failures, artifact.get("inner_fold") == expected_inner_fold, f"{label}_inner_fold_invalid"
    )
    _record(
        failures,
        artifact.get("fit_scope_commitment") == expected_fit_scope,
        f"{label}_fit_scope_mismatch",
    )
    _record(
        failures,
        artifact.get("output_scope_commitment") == expected_output_scope,
        f"{label}_output_scope_mismatch",
    )
    _validate_hash_fields(
        artifact,
        (
            "run_id_sha256",
            "fit_scope_commitment",
            "output_scope_commitment",
            "model_artifact_sha256",
            "config_sha256",
            "code_sha256",
            "input_manifest_sha256",
        ),
        label=label,
        failures=failures,
    )
    depends_on = artifact.get("depends_on")
    if not isinstance(depends_on, list) or any(not is_sha256(value) for value in depends_on):
        failures[f"{label}_depends_on_invalid"] += 1
    model_sha256 = artifact.get("model_artifact_sha256")
    if is_sha256(model_sha256):
        key = (expected_expert, expected_seed, expected_outer_fold, expected_inner_fold)
        previous = seen_model_artifacts.setdefault(str(model_sha256), key)
        if previous != key:
            failures["model_artifact_reused_across_fit_scopes"] += 1
    if expected_expert == "whole_file":
        _record(
            failures,
            whole_file_contract is not None
            and artifact.get("code_sha256") == whole_file_contract.source_closure_sha256
            and artifact.get("config_sha256") == whole_file_contract.config_sha256
            and artifact.get("input_manifest_sha256") == input_bundle_sha256,
            f"{label}_whole_file_manifest_binding_mismatch",
        )
    return artifact


def _validate_outer_runs(
    payload: object,
    *,
    scopes: dict[int, dict[str, Any]],
    lineage: dict[str, Any],
    whole_file_contract: Optional[WholeFileImplementationManifestResult],
    input_bundle_sha256: Optional[str],
    failures: Counter[str],
) -> tuple[dict[int, dict[int, int]], dict[int, int]]:
    if not isinstance(payload, list) or len(payload) != len(REQUIRED_SEEDS) * len(REQUIRED_FOLDS):
        failures["outer_run_count_invalid"] += 1
        return {}, {}
    expected_run_fields = {
        "seed",
        "outer_fold",
        "fit_scope_commitment",
        "holdout_scope_commitment",
        "fit_component_set_commitment",
        "holdout_component_set_commitment",
        "fit_rows",
        "holdout_rows",
        "fit_label_counts",
        "holdout_label_counts",
        "inner_runs",
        "outer_experts",
        "fusion",
        "outer_output",
        "access_audit",
    }
    seen_cells: set[tuple[int, int]] = set()
    seen_model_artifacts: dict[str, tuple[str, int, int, Optional[int]]] = {}
    observed_rows: dict[int, dict[int, int]] = {seed: {} for seed in REQUIRED_SEEDS}
    for raw_run in payload:
        run = _require_exact_keys(
            raw_run, expected_run_fields, label="outer_run", failures=failures
        )
        if run is None:
            continue
        seed = run.get("seed")
        fold_id = run.get("outer_fold")
        if seed not in REQUIRED_SEEDS or fold_id not in REQUIRED_FOLDS:
            failures["outer_run_seed_or_fold_invalid"] += 1
            continue
        cell = (seed, fold_id)
        if cell in seen_cells:
            failures["outer_run_seed_fold_duplicate"] += 1
            continue
        seen_cells.add(cell)
        scope = scopes.get(fold_id)
        if scope is None:
            failures["outer_run_scope_plan_missing"] += 1
            continue
        for field_name in (
            "fit_scope_commitment",
            "holdout_scope_commitment",
            "fit_component_set_commitment",
            "holdout_component_set_commitment",
            "fit_rows",
            "holdout_rows",
        ):
            if run.get(field_name) != scope.get(field_name):
                failures[f"outer_run_{field_name}_mismatch"] += 1
        if _valid_positive_count(scope.get("fit_rows")):
            _validate_label_counts(
                run.get("fit_label_counts"),
                expected_total=int(scope["fit_rows"]),
                label="outer_run_fit_label_counts",
                failures=failures,
            )
        if _valid_positive_count(scope.get("holdout_rows")):
            _validate_label_counts(
                run.get("holdout_label_counts"),
                expected_total=int(scope["holdout_rows"]),
                label="outer_run_holdout_label_counts",
                failures=failures,
            )
        observed_rows[seed][fold_id] = int(scope.get("holdout_rows", 0))

        inner_runs = run.get("inner_runs")
        plan_inner_scopes = {
            int(inner_scope["inner_fold"]): inner_scope for inner_scope in scope.get("inner_scopes", [])
            if isinstance(inner_scope, dict) and inner_scope.get("inner_fold") in REQUIRED_FOLDS
        }
        if not isinstance(inner_runs, list) or len(inner_runs) != len(REQUIRED_FOLDS):
            failures["outer_run_inner_fold_count_invalid"] += 1
            inner_runs = []
        seen_inner_folds: set[int] = set()
        inner_output_commitments: list[str] = []
        for raw_inner_run in inner_runs:
            expected_inner_fields = {"inner_fold", "loop151", "whole_file"}
            inner_run = _require_exact_keys(
                raw_inner_run,
                expected_inner_fields,
                label="outer_run_inner",
                failures=failures,
            )
            if inner_run is None:
                continue
            inner_fold = inner_run.get("inner_fold")
            if inner_fold not in REQUIRED_FOLDS or inner_fold in seen_inner_folds:
                failures["outer_run_inner_fold_id_invalid"] += 1
                continue
            seen_inner_folds.add(inner_fold)
            inner_scope = plan_inner_scopes.get(inner_fold)
            if inner_scope is None:
                failures["outer_run_inner_scope_missing"] += 1
                continue
            for expert_id in ("loop151", "whole_file"):
                artifact = _validate_fit_artifact(
                    inner_run.get(expert_id),
                    expected_stage="inner_fit",
                    expected_expert=expert_id,
                    expected_seed=seed,
                    expected_outer_fold=fold_id,
                    expected_inner_fold=inner_fold,
                    expected_fit_scope=str(inner_scope["fit_scope_commitment"]),
                    expected_output_scope=str(inner_scope["holdout_scope_commitment"]),
                    label=f"outer_run_inner_{expert_id}",
                    whole_file_contract=whole_file_contract,
                    input_bundle_sha256=input_bundle_sha256,
                    seen_model_artifacts=seen_model_artifacts,
                    failures=failures,
                )
                if artifact is not None and is_sha256(artifact.get("output_scope_commitment")):
                    inner_output_commitments.append(str(artifact["output_scope_commitment"]))
        if seen_inner_folds != set(REQUIRED_FOLDS):
            failures["outer_run_inner_fold_coverage_invalid"] += 1

        outer_experts = _require_exact_keys(
            run.get("outer_experts"),
            {"loop151", "whole_file"},
            label="outer_run_outer_experts",
            failures=failures,
        )
        if outer_experts is not None:
            for expert_id in ("loop151", "whole_file"):
                artifact = _validate_fit_artifact(
                    outer_experts.get(expert_id),
                    expected_stage="outer_fit",
                    expected_expert=expert_id,
                    expected_seed=seed,
                    expected_outer_fold=fold_id,
                    expected_inner_fold=None,
                    expected_fit_scope=str(scope["fit_scope_commitment"]),
                    expected_output_scope=str(scope["holdout_scope_commitment"]),
                    label=f"outer_run_outer_{expert_id}",
                    whole_file_contract=whole_file_contract,
                    input_bundle_sha256=input_bundle_sha256,
                    seen_model_artifacts=seen_model_artifacts,
                    failures=failures,
                )

        fusion = _require_exact_keys(
            run.get("fusion"),
            {
                "fit_scope_commitment",
                "inner_oof_input_commitments",
                "inner_oof_matrix_commitment",
                "model_artifact_sha256",
                "config_sha256",
                "threshold_policy_sha256",
                "feature_fields",
                "label_column_in_matrix",
                "target_labels_scope",
                "forbidden_feature_count",
                "frozen_before_outer_inference",
            },
            label="outer_run_fusion",
            failures=failures,
        )
        if fusion is not None:
            _record(
                failures,
                fusion.get("fit_scope_commitment") == scope.get("fit_scope_commitment"),
                "outer_run_fusion_fit_scope_mismatch",
            )
            expected_inner_outputs = sorted(inner_output_commitments)
            actual_inner_outputs = fusion.get("inner_oof_input_commitments")
            if not isinstance(actual_inner_outputs, list) or sorted(actual_inner_outputs) != expected_inner_outputs:
                failures["outer_run_fusion_not_bound_to_all_inner_oof_outputs"] += 1
            _validate_hash_fields(
                fusion,
                (
                    "fit_scope_commitment",
                    "inner_oof_matrix_commitment",
                    "model_artifact_sha256",
                    "config_sha256",
                    "threshold_policy_sha256",
                ),
                label="outer_run_fusion",
                failures=failures,
            )
            _record(
                failures,
                fusion.get("config_sha256") == lineage.get("fusion_config_sha256"),
                "outer_run_fusion_config_drift",
            )
            _record(
                failures,
                fusion.get("threshold_policy_sha256") == lineage.get("threshold_policy_sha256"),
                "outer_run_fusion_threshold_policy_drift",
            )
            _record(
                failures,
                tuple(fusion.get("feature_fields") or []) == REQUIRED_FUSION_FIELDS,
                "outer_run_fusion_feature_allowlist_invalid",
            )
            _record(
                failures,
                fusion.get("label_column_in_matrix") is False,
                "outer_run_fusion_label_column_detected",
            )
            _record(
                failures,
                fusion.get("target_labels_scope") == "outer_fit_inner_oof_only",
                "outer_run_fusion_target_scope_invalid",
            )
            _record(
                failures,
                fusion.get("forbidden_feature_count") == 0,
                "outer_run_fusion_forbidden_feature_detected",
            )
            _record(
                failures,
                fusion.get("frozen_before_outer_inference") is True,
                "outer_run_fusion_not_frozen_before_inference",
            )

        outer_output = _require_exact_keys(
            run.get("outer_output"),
            {
                "row_set_commitment",
                "output_commitment",
                "loop151_rows",
                "whole_file_rows",
                "whole_file_success_rows",
                "whole_file_missing_rows",
                "fusion_rows",
                "denominator_rows",
                "duplicate_rows",
                "unmatched_rows",
                "dropped_rows",
                "missingness_reason_counts",
            },
            label="outer_run_outer_output",
            failures=failures,
        )
        if outer_output is not None:
            holdout_rows = int(scope.get("holdout_rows", 0))
            _record(
                failures,
                outer_output.get("row_set_commitment") == scope.get("holdout_scope_commitment"),
                "outer_run_outer_output_scope_mismatch",
            )
            _validate_hash_fields(
                outer_output,
                ("row_set_commitment", "output_commitment"),
                label="outer_run_outer_output",
                failures=failures,
            )
            for field_name in (
                "loop151_rows",
                "whole_file_rows",
                "fusion_rows",
                "denominator_rows",
            ):
                _record(
                    failures,
                    outer_output.get(field_name) == holdout_rows,
                    f"outer_run_{field_name}_denominator_mismatch",
                )
            whole_file_success_rows = outer_output.get("whole_file_success_rows")
            whole_file_missing_rows = outer_output.get("whole_file_missing_rows")
            _record(
                failures,
                _valid_count(whole_file_success_rows)
                and _valid_count(whole_file_missing_rows)
                and int(whole_file_success_rows) + int(whole_file_missing_rows)
                == outer_output.get("whole_file_rows"),
                "outer_run_whole_file_missingness_denominator_mismatch",
            )
            for field_name in ("duplicate_rows", "unmatched_rows", "dropped_rows"):
                _record(
                    failures,
                    outer_output.get(field_name) == 0,
                    f"outer_run_{field_name}_nonzero",
                )
            missingness = outer_output.get("missingness_reason_counts")
            if not isinstance(missingness, dict) or set(missingness) != set(MISSINGNESS_REASONS):
                failures["outer_run_missingness_reason_counts_invalid"] += 1
            elif not all(_valid_count(value) for value in missingness.values()):
                failures["outer_run_missingness_reason_count_invalid"] += 1
            else:
                _record(
                    failures,
                    sum(int(value) for value in missingness.values()) == whole_file_missing_rows,
                    "outer_run_missingness_reason_total_mismatch",
                )

        access_audit = _require_exact_keys(
            run.get("access_audit"),
            {
                "scope_token_sha256",
                "audit_log_sha256",
                "outer_holdout_label_reads_during_fit",
                "outer_holdout_feature_reads_during_fit",
                "outer_holdout_metric_or_threshold_reads",
                "outer_inference_feature_rows",
            },
            label="outer_run_access_audit",
            failures=failures,
        )
        if access_audit is not None:
            _validate_hash_fields(
                access_audit,
                ("scope_token_sha256", "audit_log_sha256"),
                label="outer_run_access_audit",
                failures=failures,
            )
            for field_name in (
                "outer_holdout_label_reads_during_fit",
                "outer_holdout_feature_reads_during_fit",
                "outer_holdout_metric_or_threshold_reads",
            ):
                _record(
                    failures,
                    access_audit.get(field_name) == 0,
                    f"outer_run_{field_name}_nonzero",
                )
            _record(
                failures,
                access_audit.get("outer_inference_feature_rows") == scope.get("holdout_rows"),
                "outer_run_outer_inference_coverage_mismatch",
            )
    if seen_cells != {(seed, fold_id) for seed in REQUIRED_SEEDS for fold_id in REQUIRED_FOLDS}:
        failures["outer_run_seed_fold_coverage_invalid"] += 1
    expected_rows = {fold_id: int(scope.get("holdout_rows", 0)) for fold_id, scope in scopes.items()}
    return observed_rows, expected_rows


def _validate_lineage(
    payload: object,
    *,
    implementation_manifest_sha256: Optional[str],
    implementation_contract: Optional[WholeFileImplementationManifestResult],
    loop151_manifest_sha256: Optional[str],
    failures: Counter[str],
) -> dict[str, Any]:
    expected_fields = {"loop151_equivalent", "whole_file_expert", "fusion"}
    lineage = _require_exact_keys(payload, expected_fields, label="receipt_lineage", failures=failures)
    if lineage is None:
        return {}
    loop151 = _require_exact_keys(
        lineage.get("loop151_equivalent"),
        {"train_oof_manifest_sha256", "recipe_sha256", "runtime_lock_sha256", "initialization_policy"},
        label="receipt_lineage_loop151",
        failures=failures,
    )
    if loop151 is not None:
        _record(
            failures,
            loop151.get("train_oof_manifest_sha256") == loop151_manifest_sha256,
            "receipt_lineage_loop151_manifest_binding_mismatch",
        )
        _record(
            failures,
            loop151.get("initialization_policy") == "from_scratch",
            "receipt_lineage_loop151_initialization_invalid",
        )
        _validate_hash_fields(
            loop151,
            ("train_oof_manifest_sha256", "recipe_sha256", "runtime_lock_sha256"),
            label="receipt_lineage_loop151",
            failures=failures,
        )
    whole_file = _require_exact_keys(
        lineage.get("whole_file_expert"),
        {
            "implementation_manifest_sha256",
            "source_closure_sha256",
            "config_sha256",
            "runtime_lock_sha256",
            "input_contract_sha256",
            "missingness_contract_sha256",
            "whole_file_input_policy",
        },
        label="receipt_lineage_whole_file",
        failures=failures,
    )
    if whole_file is not None:
        _record(
            failures,
            whole_file.get("implementation_manifest_sha256") == implementation_manifest_sha256,
            "receipt_lineage_implementation_binding_mismatch",
        )
        _record(
            failures,
            implementation_contract is not None
            and whole_file.get("source_closure_sha256")
            == implementation_contract.source_closure_sha256
            and whole_file.get("config_sha256") == implementation_contract.config_sha256
            and whole_file.get("runtime_lock_sha256")
            == implementation_contract.runtime_lock_sha256
            and whole_file.get("input_contract_sha256")
            == implementation_contract.input_contract_sha256
            and whole_file.get("missingness_contract_sha256")
            == implementation_contract.missingness_contract_sha256
            and whole_file.get("whole_file_input_policy")
            == "all_bytes_chunked_no_silent_truncation",
            "receipt_lineage_whole_file_contract_mismatch",
        )
        _validate_hash_fields(
            whole_file,
            (
                "implementation_manifest_sha256",
                "source_closure_sha256",
                "config_sha256",
                "runtime_lock_sha256",
                "input_contract_sha256",
                "missingness_contract_sha256",
            ),
            label="receipt_lineage_whole_file",
            failures=failures,
        )
    fusion = _require_exact_keys(
        lineage.get("fusion"),
        {"config_sha256", "threshold_policy_sha256", "selection_policy", "feature_fields"},
        label="receipt_lineage_fusion",
        failures=failures,
    )
    result: dict[str, Any] = {}
    if fusion is not None:
        _validate_hash_fields(
            fusion,
            ("config_sha256", "threshold_policy_sha256"),
            label="receipt_lineage_fusion",
            failures=failures,
        )
        _record(
            failures,
            fusion.get("selection_policy") == "nested_inner_oof_only",
            "receipt_lineage_fusion_selection_policy_invalid",
        )
        _record(
            failures,
            tuple(fusion.get("feature_fields") or []) == REQUIRED_FUSION_FIELDS,
            "receipt_lineage_fusion_feature_allowlist_invalid",
        )
        result = {
            "fusion_config_sha256": fusion.get("config_sha256"),
            "threshold_policy_sha256": fusion.get("threshold_policy_sha256"),
        }
    return result


def _validate_coverage(
    payload: object,
    *,
    eligible_rows: Optional[int],
    observed_rows: dict[int, dict[int, int]],
    expected_rows: dict[int, int],
    failures: Counter[str],
) -> dict[str, Any]:
    expected_fields = {"eligible_rows", "per_seed_outer_holdout", "global_duplicate_rows", "global_unmatched_rows", "global_dropped_rows"}
    coverage = _require_exact_keys(payload, expected_fields, label="receipt_coverage", failures=failures)
    if coverage is None:
        return {}
    _record(
        failures,
        coverage.get("eligible_rows") == eligible_rows,
        "receipt_coverage_eligible_rows_mismatch",
    )
    per_seed = coverage.get("per_seed_outer_holdout")
    if not isinstance(per_seed, list) or len(per_seed) != len(REQUIRED_SEEDS):
        failures["receipt_coverage_per_seed_count_invalid"] += 1
    else:
        seen_seeds: set[int] = set()
        expected_per_seed_total = sum(expected_rows.values())
        for entry in per_seed:
            seed_coverage = _require_exact_keys(
                entry,
                {
                    "seed",
                    "expected_rows",
                    "observed_rows",
                    "unique_outer_holdout_rows",
                    "duplicate_rows",
                    "unmatched_rows",
                    "dropped_rows",
                },
                label="receipt_coverage_seed",
                failures=failures,
            )
            if seed_coverage is None:
                continue
            seed = seed_coverage.get("seed")
            if seed not in REQUIRED_SEEDS or seed in seen_seeds:
                failures["receipt_coverage_seed_id_invalid"] += 1
                continue
            seen_seeds.add(seed)
            for field_name in ("expected_rows", "observed_rows", "unique_outer_holdout_rows"):
                _record(
                    failures,
                    seed_coverage.get(field_name) == expected_per_seed_total,
                    f"receipt_coverage_{field_name}_mismatch",
                )
            for field_name in ("duplicate_rows", "unmatched_rows", "dropped_rows"):
                _record(
                    failures,
                    seed_coverage.get(field_name) == 0,
                    f"receipt_coverage_{field_name}_nonzero",
                )
            if sum(observed_rows.get(seed, {}).values()) != expected_per_seed_total:
                failures["receipt_coverage_outer_run_denominator_mismatch"] += 1
        if seen_seeds != set(REQUIRED_SEEDS):
            failures["receipt_coverage_seed_coverage_invalid"] += 1
    for field_name in ("global_duplicate_rows", "global_unmatched_rows", "global_dropped_rows"):
        _record(
            failures,
            coverage.get(field_name) == 0,
            f"receipt_coverage_{field_name}_nonzero",
        )
    return coverage


def _validate_access_audit(payload: object, failures: Counter[str]) -> None:
    expected_fields = {
        "controller_audit_log_sha256",
        "identity_rows_exported",
        "prediction_rows_exported",
        "in_sample_score_substitution_count",
        "fold_drift_count",
        "missing_output_drop_count",
        "forbidden_split_access_counts",
    }
    audit = _require_exact_keys(payload, expected_fields, label="receipt_access_audit", failures=failures)
    if audit is None:
        return
    _validate_hash_fields(
        audit,
        ("controller_audit_log_sha256",),
        label="receipt_access_audit",
        failures=failures,
    )
    for field_name in (
        "identity_rows_exported",
        "prediction_rows_exported",
        "in_sample_score_substitution_count",
        "fold_drift_count",
        "missing_output_drop_count",
    ):
        _record(
            failures,
            audit.get(field_name) == 0,
            f"receipt_access_audit_{field_name}_nonzero",
        )
    forbidden = audit.get("forbidden_split_access_counts")
    if not isinstance(forbidden, dict) or set(forbidden) != set(FORBIDDEN_SPLIT_ROLES):
        failures["receipt_access_audit_forbidden_split_shape_invalid"] += 1
    elif any(forbidden.get(role) != 0 for role in FORBIDDEN_SPLIT_ROLES):
        failures["receipt_access_audit_forbidden_split_access_detected"] += 1


def _validate_training_final_lease(
    payload: Optional[dict[str, Any]],
    *,
    authorization_sha256: Optional[str],
    lease_id: Optional[str],
    canonical_argv: Optional[list[str]],
    receipt_path: Path,
    fold_assignment_fingerprint: Optional[str],
    failures: Counter[str],
) -> None:
    if payload is None:
        return
    expected_fields = {
        "schema",
        "loop_id",
        "state",
        "authorization_sha256",
        "lease_id",
        "canonical_argv",
        "output_receipt_path",
        "fold_assignment_fingerprint",
        "outer_run_budget",
        "consumed_at_utc",
    }
    lease = _require_exact_keys(payload, expected_fields, label="training_final_lease", failures=failures)
    if lease is None:
        return
    _record(failures, lease.get("schema") == TRAINING_LEASE_SCHEMA, "training_final_lease_schema_invalid")
    _record(failures, lease.get("loop_id") == LOOP_ID, "training_final_lease_loop_id_invalid")
    _record(
        failures,
        lease.get("state") == "consumed_before_execution",
        "training_final_lease_state_invalid",
    )
    _record(
        failures,
        lease.get("authorization_sha256") == authorization_sha256,
        "training_final_lease_authorization_binding_mismatch",
    )
    _record(failures, lease.get("lease_id") == lease_id, "training_final_lease_id_mismatch")
    _record(
        failures,
        lease.get("canonical_argv") == canonical_argv,
        "training_final_lease_argv_mismatch",
    )
    try:
        output_path = resolve_path(lease.get("output_receipt_path"))
    except ValueError:
        failures["training_final_lease_output_path_invalid"] += 1
    else:
        _record(
            failures,
            output_path == receipt_path.resolve(),
            "training_final_lease_output_path_mismatch",
        )
    _record(
        failures,
        lease.get("fold_assignment_fingerprint") == fold_assignment_fingerprint,
        "training_final_lease_fold_fingerprint_mismatch",
    )
    _record(
        failures,
        lease.get("outer_run_budget") == len(REQUIRED_SEEDS) * len(REQUIRED_FOLDS),
        "training_final_lease_outer_run_budget_invalid",
    )
    try:
        parse_utc(lease.get("consumed_at_utc"))
    except ValueError:
        failures["training_final_lease_timestamp_invalid"] += 1


def _validate_training_final_lease_v2(
    payload: Optional[dict[str, Any]],
    *,
    marker_payload: Optional[dict[str, Any]],
    authorization_context: dict[str, Any],
    scope_validation_sha256: Optional[str],
    input_bundle_sha256: Optional[str],
    implementation_manifest_sha256: Optional[str],
    implementation_contract: Optional[WholeFileImplementationManifestResult],
    receipt_path: Path,
    lease_marker_directory: Path,
    fold_assignment_fingerprint: Optional[str],
    failures: Counter[str],
) -> Optional[Path]:
    if payload is None:
        return None
    expected_fields = {
        "schema",
        "loop_id",
        "state",
        "lease_consumption_id",
        "authorization_sha256",
        "lease_id",
        "scope_plan_validation_sha256",
        "input_bundle_sha256",
        "implementation_manifest_sha256",
        "source_closure_sha256",
        "memory_contract_sha256",
        "controller_sha256",
        "canonical_argv_sha256",
        "canonical_argv",
        "output_receipt_path",
        "fold_assignment_fingerprint",
        "outer_run_budget",
        "consumed_at_utc",
        "marker_path",
    }
    lease = _require_exact_keys(
        payload, expected_fields, label="training_final_lease", failures=failures
    )
    if lease is None:
        return None
    _record(failures, lease.get("schema") == TRAINING_LEASE_SCHEMA, "training_final_lease_schema_invalid")
    _record(failures, lease.get("loop_id") == LOOP_ID, "training_final_lease_loop_id_invalid")
    _record(
        failures,
        lease.get("state") == "consumed_before_protected_open",
        "training_final_lease_state_invalid",
    )
    try:
        output_path = resolve_path(lease.get("output_receipt_path"))
        marker_path = resolve_path(lease.get("marker_path"))
    except ValueError:
        failures["training_final_lease_output_path_invalid"] += 1
        return None
    expected_authorization_sha256 = authorization_context.get("authorization_sha256")
    expected_lease_id = authorization_context.get("lease_id")
    expected_argv = authorization_context.get("canonical_argv")
    expected_argv_sha256 = authorization_context.get("canonical_argv_sha256")
    expected_controller_sha256 = authorization_context.get("controller_sha256")
    expected_consumption_id: Optional[str] = None
    if all(
        isinstance(value, str)
        for value in (
            expected_authorization_sha256,
            expected_lease_id,
            expected_controller_sha256,
            input_bundle_sha256,
            scope_validation_sha256,
            expected_argv_sha256,
        )
    ):
        expected_consumption_id = build_lease_consumption_id(
            authorization_sha256=str(expected_authorization_sha256),
            lease_id=str(expected_lease_id),
            controller_sha256=str(expected_controller_sha256),
            input_bundle_sha256=str(input_bundle_sha256),
            scope_plan_validation_sha256=str(scope_validation_sha256),
            canonical_argv_sha256=str(expected_argv_sha256),
            execution_receipt_path=receipt_path.resolve(),
        )
    expected_marker_path = (
        lease_marker_directory.resolve() / f"{expected_consumption_id}.final.json"
        if expected_consumption_id is not None
        else None
    )
    _record(
        failures,
        lease.get("authorization_sha256") == expected_authorization_sha256
        and lease.get("lease_id") == expected_lease_id
        and lease.get("scope_plan_validation_sha256") == scope_validation_sha256
        and lease.get("input_bundle_sha256") == input_bundle_sha256
        and lease.get("implementation_manifest_sha256") == implementation_manifest_sha256
        and implementation_contract is not None
        and lease.get("source_closure_sha256") == implementation_contract.source_closure_sha256
        and lease.get("memory_contract_sha256") == implementation_contract.memory_contract_sha256
        and lease.get("controller_sha256") == expected_controller_sha256
        and lease.get("canonical_argv") == expected_argv
        and lease.get("canonical_argv_sha256") == expected_argv_sha256
        and lease.get("lease_consumption_id") == expected_consumption_id,
        "training_final_lease_binding_mismatch",
    )
    _record(
        failures,
        output_path == receipt_path.resolve() and marker_path == expected_marker_path,
        "training_final_lease_output_path_mismatch",
    )
    _record(
        failures,
        lease.get("fold_assignment_fingerprint") == fold_assignment_fingerprint
        and lease.get("outer_run_budget") == len(REQUIRED_SEEDS) * len(REQUIRED_FOLDS),
        "training_final_lease_partition_binding_invalid",
    )
    try:
        parse_utc(lease.get("consumed_at_utc"))
    except ValueError:
        failures["training_final_lease_timestamp_invalid"] += 1
    marker = _require_exact_keys(
        marker_payload, expected_fields, label="training_lease_marker", failures=failures
    )
    if marker is not None:
        _record(
            failures,
            marker.get("schema") == TRAINING_LEASE_MARKER_SCHEMA
            and marker.get("state") == "consumed_before_protected_open"
            and marker.get("loop_id") == LOOP_ID,
            "training_lease_marker_schema_invalid",
        )
        _record(
            failures,
            marker.get("consumed_at_utc") == lease.get("consumed_at_utc")
            and {
                name: value
                for name, value in marker.items()
                if name not in {"schema", "consumed_at_utc"}
            }
            == {
                name: value
                for name, value in lease.items()
                if name not in {"schema", "consumed_at_utc"}
            },
            "training_lease_marker_final_mismatch",
        )
        try:
            parse_utc(marker.get("consumed_at_utc"))
        except ValueError:
            failures["training_lease_marker_timestamp_invalid"] += 1
    return expected_marker_path


def validate_loop164_nested_oof_execution_receipt(
    *,
    receipt_json: Path,
    proposal_json: Path = DEFAULT_PATHS["proposal"],
    contract_json: Path = DEFAULT_PATHS["contract"],
    isolation_receipt_json: Path = DEFAULT_PATHS["isolation_receipt"],
    scope_plan_json: Path = DEFAULT_PATHS["scope_plan"],
    scope_plan_validation_json: Path = DEFAULT_PATHS["scope_plan_validation"],
    implementation_manifest_json: Path = DEFAULT_PATHS["implementation_manifest"],
    loop151_train_oof_manifest_json: Path = DEFAULT_PATHS["loop151_train_oof_manifest"],
    resource_guard_json: Path = DEFAULT_PATHS["resource_guard"],
    input_bundle_json: Path = DEFAULT_PATHS["input_bundle"],
    training_authorization_json: Path = DEFAULT_PATHS["training_authorization"],
    training_final_lease_json: Path = DEFAULT_PATHS["training_final_lease"],
    training_lease_marker_directory: Path = TRAINING_LEASE_MARKER_DIRECTORY,
) -> dict[str, Any]:
    """Validate a future aggregate nested-OOF receipt without reading prediction rows."""

    result = _empty_result()
    failures: Counter[str] = Counter()
    try:
        receipt_path = resolve_path(receipt_json)
        expected_paths = {
            "proposal": resolve_path(proposal_json),
            "contract": resolve_path(contract_json),
            "isolation_receipt": resolve_path(isolation_receipt_json),
            "scope_plan": resolve_path(scope_plan_json),
            "scope_plan_validation": resolve_path(scope_plan_validation_json),
            "implementation_manifest": resolve_path(implementation_manifest_json),
            "loop151_train_oof_manifest": resolve_path(loop151_train_oof_manifest_json),
            "resource_guard": resolve_path(resource_guard_json),
            "input_bundle": resolve_path(input_bundle_json),
            "training_authorization": resolve_path(training_authorization_json),
            "training_final_lease": resolve_path(training_final_lease_json),
        }
        marker_directory = resolve_path(training_lease_marker_directory)
        receipt, receipt_sha256 = read_json_object(receipt_path)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        result["blockers"] = ["nested_oof_receipt_unreadable"]
        return result

    expected_receipt_fields = {
        "schema",
        "loop_id",
        "aggregate_only",
        "decision",
        "completed_at_utc",
        "bindings",
        "partition_plan",
        "lineage",
        "outer_runs",
        "coverage",
        "access_audit",
        "blockers",
    }
    receipt_payload = _require_exact_keys(
        receipt, expected_receipt_fields, label="nested_oof_receipt", failures=failures
    )
    if receipt_payload is None:
        result["blockers"] = sorted(failures)
        return result
    _record(failures, receipt.get("schema") == RECEIPT_SCHEMA, "nested_oof_receipt_schema_invalid")
    _record(failures, receipt.get("loop_id") == LOOP_ID, "nested_oof_receipt_loop_id_invalid")
    _record(failures, receipt.get("aggregate_only") is True, "nested_oof_receipt_not_aggregate_only")
    _record(failures, receipt.get("decision") == "pass", "nested_oof_receipt_not_pass")
    _record(failures, receipt.get("blockers") == [], "nested_oof_receipt_declared_blockers_nonempty")
    try:
        completed_at_utc = parse_utc(receipt.get("completed_at_utc"))
    except ValueError:
        completed_at_utc = None
        failures["nested_oof_receipt_completed_at_invalid"] += 1

    bindings = receipt.get("bindings")
    binding_payloads: dict[str, Optional[dict[str, Any]]] = {}
    binding_sha256: dict[str, Optional[str]] = {}
    expected_binding_names = {*expected_paths, "training_lease_marker"}
    if not isinstance(bindings, dict) or set(bindings) != expected_binding_names:
        failures["nested_oof_receipt_bindings_shape_invalid"] += 1
    else:
        for name, expected_path in expected_paths.items():
            payload, fingerprint = _binding_payload(
                bindings.get(name),
                expected_path=expected_path,
                label=name,
                failures=failures,
            )
            binding_payloads[name] = payload
            binding_sha256[name] = fingerprint
    result["binding_fingerprints"] = {
        name: fingerprint for name, fingerprint in sorted(binding_sha256.items()) if fingerprint is not None
    }

    _validate_proposal(binding_payloads.get("proposal"), failures)
    _validate_contract(binding_payloads.get("contract"), failures)
    fold_assignment_fingerprint, eligible_rows, warmup_rows = _validate_isolation_receipt(
        binding_payloads.get("isolation_receipt"),
        contract_sha256=binding_sha256.get("contract"),
        failures=failures,
    )
    scopes = _validate_scope_plan(
        binding_payloads.get("scope_plan"),
        contract_sha256=binding_sha256.get("contract"),
        isolation_receipt_sha256=binding_sha256.get("isolation_receipt"),
        expected_fingerprint=fold_assignment_fingerprint,
        expected_eligible_rows=eligible_rows,
        expected_warmup_rows=warmup_rows,
        failures=failures,
    )
    _validate_scope_plan_validation_v2(
        binding_payloads.get("scope_plan_validation"),
        proposal_sha256=binding_sha256.get("proposal"),
        contract_sha256=binding_sha256.get("contract"),
        isolation_receipt_sha256=binding_sha256.get("isolation_receipt"),
        scope_plan_sha256=binding_sha256.get("scope_plan"),
        fold_assignment_fingerprint=fold_assignment_fingerprint,
        failures=failures,
    )
    _validate_input_bundle_v2(
        binding_payloads.get("input_bundle"),
        scope_validation_sha256=binding_sha256.get("scope_plan_validation"),
        fold_assignment_fingerprint=fold_assignment_fingerprint,
        failures=failures,
    )
    implementation_manifest_path = expected_paths["implementation_manifest"]
    implementation_contract = _validate_implementation_manifest(
        binding_payloads.get("implementation_manifest"),
        root=_infer_implementation_project_root(implementation_manifest_path),
        failures=failures,
    )
    _validate_loop151_train_oof_manifest(
        binding_payloads.get("loop151_train_oof_manifest"),
        fold_assignment_fingerprint=fold_assignment_fingerprint,
        failures=failures,
    )
    lineage = _validate_lineage(
        receipt.get("lineage"),
        implementation_manifest_sha256=binding_sha256.get("implementation_manifest"),
        implementation_contract=implementation_contract,
        loop151_manifest_sha256=binding_sha256.get("loop151_train_oof_manifest"),
        failures=failures,
    )
    authorization_context = _validate_training_authorization_v2(
        binding_payloads.get("training_authorization"),
        authorization_sha256=binding_sha256.get("training_authorization"),
        receipt_path=receipt_path,
        final_lease_path=expected_paths["training_final_lease"],
        lease_marker_directory=marker_directory,
        completed_at_utc=completed_at_utc,
        binding_sha256={
            name: binding_sha256.get(name)
            for name in (
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
        },
        binding_paths={
            name: expected_paths[name]
            for name in (
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
        },
        fold_assignment_fingerprint=fold_assignment_fingerprint,
        failures=failures,
    )
    _validate_resource_guard_v2(
        binding_payloads.get("resource_guard"),
        runtime_binding=authorization_context.get("runtime_binding"),
        canonical_argv_sha256=authorization_context.get("canonical_argv_sha256"),
        implementation_manifest_sha256=binding_sha256.get("implementation_manifest"),
        implementation_contract=implementation_contract,
        completed_at_utc=completed_at_utc,
        max_age_seconds=authorization_context.get("max_resource_guard_age_seconds"),
        failures=failures,
    )
    final_lease_payload = binding_payloads.get("training_final_lease")
    marker_path: Optional[Path] = None
    if isinstance(final_lease_payload, dict):
        consumption_id = final_lease_payload.get("lease_consumption_id")
        if is_sha256(consumption_id):
            marker_path = marker_directory / f"{consumption_id}.final.json"
        else:
            failures["training_final_lease_consumption_id_invalid"] += 1
    marker_payload: Optional[dict[str, Any]] = None
    if marker_path is not None and isinstance(bindings, dict):
        marker_payload, marker_sha256 = _binding_payload(
            bindings.get("training_lease_marker"),
            expected_path=marker_path,
            label="training_lease_marker",
            failures=failures,
        )
        binding_payloads["training_lease_marker"] = marker_payload
        binding_sha256["training_lease_marker"] = marker_sha256
    _validate_training_final_lease_v2(
        binding_payloads.get("training_final_lease"),
        marker_payload=marker_payload,
        authorization_context=authorization_context,
        scope_validation_sha256=binding_sha256.get("scope_plan_validation"),
        input_bundle_sha256=binding_sha256.get("input_bundle"),
        implementation_manifest_sha256=binding_sha256.get("implementation_manifest"),
        implementation_contract=implementation_contract,
        receipt_path=receipt_path,
        lease_marker_directory=marker_directory,
        fold_assignment_fingerprint=fold_assignment_fingerprint,
        failures=failures,
    )
    result["binding_fingerprints"] = {
        name: fingerprint
        for name, fingerprint in sorted(binding_sha256.items())
        if fingerprint is not None
    }

    partition_plan = _require_exact_keys(
        receipt.get("partition_plan"),
        {"fold_assignment_fingerprint", "eligible_rows", "warmup_rows", "seeds", "outer_fold_ids"},
        label="receipt_partition_plan",
        failures=failures,
    )
    if partition_plan is not None:
        _record(
            failures,
            partition_plan.get("fold_assignment_fingerprint") == fold_assignment_fingerprint,
            "receipt_partition_plan_fold_fingerprint_mismatch",
        )
        _record(
            failures,
            partition_plan.get("eligible_rows") == eligible_rows,
            "receipt_partition_plan_eligible_rows_mismatch",
        )
        _record(
            failures,
            partition_plan.get("warmup_rows") == warmup_rows,
            "receipt_partition_plan_warmup_rows_mismatch",
        )
        _record(
            failures,
            tuple(partition_plan.get("seeds") or []) == REQUIRED_SEEDS,
            "receipt_partition_plan_seed_set_invalid",
        )
        _record(
            failures,
            tuple(partition_plan.get("outer_fold_ids") or []) == REQUIRED_FOLDS,
            "receipt_partition_plan_fold_set_invalid",
        )
    observed_rows, expected_rows = _validate_outer_runs(
        receipt.get("outer_runs"),
        scopes=scopes,
        lineage=lineage,
        whole_file_contract=implementation_contract,
        input_bundle_sha256=binding_sha256.get("input_bundle"),
        failures=failures,
    )
    result["coverage"] = _validate_coverage(
        receipt.get("coverage"),
        eligible_rows=eligible_rows,
        observed_rows=observed_rows,
        expected_rows=expected_rows,
        failures=failures,
    )
    _validate_access_audit(receipt.get("access_audit"), failures)

    result["aggregate_only_verified"] = not any(
        code.startswith("nested_oof_receipt_") or "unexpected_fields" in code
        for code in failures
    )
    result["authority_chain_verified"] = not any(
        code.startswith("training_") or code.startswith("binding_") for code in failures
    )
    result["partition_plan_verified"] = not any(
        code.startswith("scope_plan_")
        or code.startswith("isolation_receipt_")
        or code.startswith("receipt_partition_plan_")
        for code in failures
    )
    result["nested_oof_verified"] = not any(
        code.startswith("outer_run_")
        or code.startswith("model_artifact_")
        or code.startswith("receipt_coverage_")
        or code.startswith("receipt_access_audit_")
        for code in failures
    )
    result["blockers"] = sorted(failures)
    result["decision"] = "pass" if not failures else "block"
    result["ready_for"]["loop164_train_oof_data_boundary"] = not failures
    result["ready_for"]["a2_training_authorization"] = False
    return result


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("Refusing to overwrite an existing validation result") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an aggregate-only future Loop164 nested OOF execution receipt."
    )
    parser.add_argument("--receipt-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = validate_loop164_nested_oof_execution_receipt(receipt_json=args.receipt_json)
    try:
        _write_json_exclusive(resolve_path(args.output_json), payload)
    except (OSError, ValueError) as exc:
        print(json.dumps({"decision": "block", "blockers": [str(exc)]}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "blockers": payload["blockers"],
                "output_json": str(resolve_path(args.output_json)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
