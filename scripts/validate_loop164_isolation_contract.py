#!/usr/bin/env python3
"""Fail-closed metadata-only isolation gate for the future Loop164 A2 run.

The validator intentionally accepts only a small public contract JSON and a
custodian-produced JSONL metadata inventory.  It never opens raw binaries,
feature caches, prediction rows, checkpoints, or model payloads.  Under the
current A1 scope this source and its synthetic tests may be reviewed, but the
validator must not be run against project inventory rows until a separate A2
authorization exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SCHEMA = "axon_loop164_full_pool_isolation_contract_v2"
VALIDATION_SCHEMA = "axon_loop164_full_pool_isolation_validation_v4"
METADATA_AUTHORIZATION_SCHEMA = "axon_loop164_isolation_validation_authorization_v3"
METADATA_TRUST_ANCHOR_SCHEMA = "axon_loop164_external_trust_anchor_v1"
METADATA_VALIDATOR_CLOSURE_SCHEMA = "axon_loop164_metadata_validator_source_closure_v1"
METADATA_AUTHORIZATION_PROVENANCE_SCHEMA = "axon_loop164_isolation_authorization_provenance_v2"
METADATA_AUTHORITY_SCOPE = {
    "tier": "A2",
    "operation": "metadata_isolation_only",
    "protected_input_scope": "metadata_only",
    "grants": [],
}
FEATURE_CONTRACT_SCHEMA = "axon_loop164_residual_fusion_feature_contract_v2"
IMPLEMENTATION_BINDING_PHASE = "deferred_to_a2_training_authority"
LOOP_ID = "loop164_whole_file_residual_expert"
CANONICAL_PROPOSAL_PATH = (
    PROJECT_ROOT / "manifests/roadmap_9997/loop164_whole_file_residual_expert/proposal.json"
)
CANONICAL_A2_METADATA_AUTHORIZATION_PATH = (
    PROJECT_ROOT
    / "manifests/roadmap_9997/loop164_whole_file_residual_expert/"
    "a2_isolation_validation_authorization.json"
)
CANONICAL_ISOLATION_RECEIPT_PATH = (
    PROJECT_ROOT / "reports/roadmap_9997/loop164/full_pool_isolation_validation.json"
)
METADATA_LEASE_DIRECTORY = PROJECT_ROOT / "reports/roadmap_9997/loop164/metadata_lease_consumptions"
MAX_A2_METADATA_AUTHORIZATION_TTL_SECONDS = 24 * 60 * 60
ROLE_ORDER = (
    "train_anchor",
    "train_oof",
    "val_a",
    "val_b",
    "sentinel",
    "confirmation",
    "certification",
)
REQUIRED_SEEDS = (41, 42, 43)
REQUIRED_FOLD_COUNT = 5
MINIMUM_FULL_POOL_ROWS = 200000
REQUIRED_EMBARGO_SECONDS = 30 * 24 * 60 * 60
REQUIRED_MINIMUM_FIT_ROWS_PER_LABEL = 1000
REQUIRED_MINIMUM_HOLDOUT_ROWS_PER_LABEL = 100
TRUSTED_TIMESTAMP_PROVENANCE = frozenset(
    {"custodian_verified", "provider_signed_as_of", "source_ledger_verified"}
)
RESIDUAL_FUSION_INPUT_FIELDS = (
    "loop151_oof_score",
    "whole_file_oof_score",
    "loop151_oof_uncertainty",
    "whole_file_oof_uncertainty",
    "loop151_missingness",
    "whole_file_missingness",
)
GROUP_FIELDS = (
    "exact_cluster_id",
    "near_duplicate_cluster_id",
    "family_id",
    "campaign_id",
    "source_group_id",
)
REQUIRED_RECORD_FIELDS = (
    "sample_uid",
    "source_sha256",
    "locked_label",
    "label_provenance",
    "label_evidence_version",
    "label_frozen_at_utc",
    "schema_version",
    "acquisition_time_utc",
    "first_seen_time_utc",
    "timestamp_provenance",
    "source_id",
    "source_group_id",
    "exact_cluster_id",
    "near_duplicate_cluster_id",
    "family_id",
    "family_evidence_version",
    "campaign_id",
    "campaign_evidence_version",
    "isolation_component_id",
    "parser_status",
    "grouping_status",
    "feature_schema_version",
    "split_role",
    "oof_role",
    "outer_fold_id",
    "inner_fold_id",
    "calibration_role",
    "evaluation_generation",
    "denominator_status",
)
UNRESOLVED_VALUES = {"", "-", "n/a", "na", "none", "null", "unknown", "unresolved"}
FORBIDDEN_MODEL_INPUT_FIELDS = {
    "sample_uid",
    "source_sha256",
    "locked_label",
    "label_provenance",
    "label_evidence_version",
    "label_frozen_at_utc",
    "acquisition_time_utc",
    "first_seen_time_utc",
    "timestamp_provenance",
    "source_id",
    "source_group_id",
    "exact_cluster_id",
    "near_duplicate_cluster_id",
    "family_id",
    "campaign_id",
    "isolation_component_id",
    "split_role",
    "oof_role",
    "outer_fold_id",
    "inner_fold_id",
    "calibration_role",
    "evaluation_generation",
    "label",
    "path",
    "source_path",
    "filename",
    "directory",
    "extension",
    "sample_index",
    "row_order",
}
FORBIDDEN_MODEL_INPUT_TOKENS = (
    "source_",
    "sha256",
    "hash",
    "family",
    "campaign",
    "component",
    "group_id",
    "split",
    "fold",
    "timestamp",
    "time_utc",
    "provenance",
    "sample_uid",
    "sample_index",
    "filename",
    "directory",
    "extension",
    "path",
)
VALIDATOR_SOURCE_PATHS = (
    Path("scripts/pre_run_resource_leak_guard.py"),
    Path("scripts/validate_loop164_isolation_contract.py"),
)


class UnionFind:
    """Small deterministic union-find for metadata relation closure."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self._parent.setdefault(item, item)

    def find(self, item: str) -> str:
        parent = self._parent[item]
        if parent != item:
            parent = self.find(parent)
            self._parent[item] = parent
        return parent

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root


@dataclass(frozen=True)
class MetadataRecord:
    sample_uid: str
    source_sha256: str
    locked_label: int
    acquisition_time: datetime
    first_seen_time: datetime
    split_role: str
    oof_role: str
    outer_fold_id: Optional[int]
    inner_fold_id: Optional[int]
    isolation_component_id: str
    group_values: tuple[str, ...]


def resolve_path(path: Path, *, base: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else base / path


def _require_no_symlink(path: Path) -> None:
    candidate = path.absolute()
    for ancestor in (candidate, *candidate.parents):
        if ancestor.is_symlink():
            raise ValueError("Symbolic-link path bindings are forbidden")


def _resolve_external_path(value: object, *, root: Path = PROJECT_ROOT) -> Path:
    text = normalized_text(value)
    if not text:
        raise ValueError("External path binding is empty")
    candidate = Path(text).absolute()
    _require_no_symlink(candidate)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return resolved
    raise ValueError("External path binding must be outside the project root")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validator_source_closure() -> tuple[dict[str, str], str]:
    files: dict[str, str] = {}
    for relative_path in VALIDATOR_SOURCE_PATHS:
        path = PROJECT_ROOT / relative_path
        _require_no_symlink(path)
        files[relative_path.as_posix()] = sha256_file(path)
    encoded = json.dumps(files, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return files, hashlib.sha256(encoded).hexdigest()


def _sha256_json_value(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def sha256_open_file(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Duplicate JSON object key: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant: {value}")


def strict_json_loads(payload: str) -> object:
    return json.loads(
        payload,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_nonfinite_json_constant,
    )


def _read_json_object_with_sha256(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = strict_json_loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def is_valid_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def normalized_text(value: object) -> str:
    return str(value or "").strip()


def is_resolved_identifier(value: object) -> bool:
    normalized = normalized_text(value).casefold()
    return normalized not in UNRESOLVED_VALUES and not normalized.startswith(
        ("unknown", "unresolved", "missing", "placeholder")
    )


def parse_utc_timestamp(value: object) -> datetime:
    text = normalized_text(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be explicit UTC")
    return parsed.astimezone(timezone.utc)


def stable_component_id(source_hashes: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(source_hashes))).encode("ascii")
    return "component_sha256:" + hashlib.sha256(payload).hexdigest()


def _empty_result(*, contract_bound: bool, rows_bound: bool) -> dict[str, Any]:
    return {
        "schema": VALIDATION_SCHEMA,
        "loop_id": LOOP_ID,
        "contract_binding_verified": contract_bound,
        "proposal_binding_verified": False,
        "rows_artifact_binding_verified": rows_bound,
        "binding_fingerprints": {
            "contract_sha256": None,
            "proposal_sha256": None,
            "inventory_sha256": None,
            "rows_artifact_sha256": None,
            "grouping_parameters_sha256": None,
            "a2_authorization_sha256": None,
            "a2_trust_anchor_sha256": None,
            "a2_validator_source_closure_sha256": None,
            "a2_runtime_python_sha256": None,
            "a2_resource_guard_sha256": None,
            "a2_canonical_argv_sha256": None,
            "a2_lease_marker_sha256": None,
            "a2_lease_consumption_id": None,
        },
        "a2_authorization_provenance": None,
        "feature_contract": {
            "schema": None,
            "feature_fields": [],
            "feature_matrix_receipt_required": False,
            "implementation_binding_phase": None,
        },
        "rows_read": 0,
        "expected_rows": None,
        "counts": {
            "valid_rows": 0,
            "invalid_rows": 0,
            "components": 0,
            "roles": {},
            "labels_by_role": {},
            "sealed_rows_by_role": {},
        },
        "temporal_bounds": {},
        "oof": {
            "fold_assignment_fingerprint": None,
            "eligible_rows": 0,
            "warmup_rows": 0,
            "inner_fold_execution_receipt_required": True,
        },
        "identity_feature_violations": [],
        "violation_counts": {},
        "blockers": [],
        "ready_for": {
            "full_pool_isolation_contract": False,
            "loop164_train_oof_partition": False,
            "loop164_train_oof_data_boundary": False,
            "a2_training_authorization": False,
            "val_a": False,
            "val_b": False,
            "test10k": False,
            "full_test": False,
        },
        "decision": "block",
        "notes": [
            "This receipt contains aggregate counts only; it intentionally omits paths, row identities, and source hashes.",
            "Passing the metadata gate never authorizes model, data, Val, Test-10k, or full-test execution.",
        ],
    }


def _record_violation(violations: Counter[str], code: str) -> None:
    violations[code] += 1


def _validate_contract_header(
    contract: dict[str, Any],
    *,
    minimum_full_pool_rows: int,
    required_embargo_seconds: int,
    required_minimum_fit_rows_per_label: int,
    required_minimum_holdout_rows_per_label: int,
    required_roles: tuple[str, ...],
) -> tuple[Counter[str], list[str], dict[str, Any]]:
    violations: Counter[str] = Counter()
    if contract.get("schema") != CONTRACT_SCHEMA:
        _record_violation(violations, "contract_schema_mismatch")
    if contract.get("loop_id") != LOOP_ID:
        _record_violation(violations, "contract_loop_id_mismatch")
    if not is_resolved_identifier(contract.get("manifest_version")):
        _record_violation(violations, "contract_manifest_version_missing")
    if contract.get("pool_scope") != "full_pool":
        _record_violation(violations, "contract_pool_scope_invalid")
    if tuple(contract.get("required_roles") or []) != required_roles:
        _record_violation(violations, "contract_required_roles_mismatch")

    proposal_binding = contract.get("proposal_binding")
    if not isinstance(proposal_binding, dict) or not is_valid_sha256(
        proposal_binding.get("sha256")
    ):
        _record_violation(violations, "contract_proposal_binding_invalid")

    inventory = contract.get("inventory")
    expected_rows: Optional[int] = None
    if not isinstance(inventory, dict):
        _record_violation(violations, "contract_inventory_missing")
    else:
        try:
            expected_rows = int(inventory.get("expected_active_rows"))
        except (TypeError, ValueError):
            expected_rows = None
        if expected_rows is None or expected_rows <= 0:
            _record_violation(violations, "contract_expected_active_rows_invalid")
        elif expected_rows < minimum_full_pool_rows:
            _record_violation(violations, "contract_full_pool_rows_below_minimum")
        if not is_valid_sha256(inventory.get("inventory_sha256")):
            _record_violation(violations, "contract_inventory_sha256_invalid")

    rows_artifact = contract.get("rows_artifact")
    if not isinstance(rows_artifact, dict):
        _record_violation(violations, "contract_rows_artifact_missing")
    else:
        if not is_resolved_identifier(rows_artifact.get("path")):
            _record_violation(violations, "contract_rows_artifact_path_missing")
        if not is_valid_sha256(rows_artifact.get("sha256")):
            _record_violation(violations, "contract_rows_artifact_sha256_invalid")
        try:
            artifact_rows = int(rows_artifact.get("rows"))
        except (TypeError, ValueError):
            artifact_rows = None
        if artifact_rows is None or artifact_rows <= 0:
            _record_violation(violations, "contract_rows_artifact_count_invalid")
        elif expected_rows is not None and artifact_rows != expected_rows:
            _record_violation(violations, "contract_inventory_rows_artifact_count_mismatch")

    grouping = contract.get("grouping")
    if not isinstance(grouping, dict):
        _record_violation(violations, "contract_grouping_missing")
    else:
        algorithm = normalized_text(grouping.get("algorithm")).casefold()
        if algorithm != "multi_relation_union_find":
            _record_violation(violations, "contract_grouping_algorithm_invalid")
        if "path" in algorithm or "date" in algorithm:
            _record_violation(violations, "contract_path_derived_grouping_forbidden")
        if not is_resolved_identifier(grouping.get("version")):
            _record_violation(violations, "contract_grouping_version_missing")
        if not is_valid_sha256(grouping.get("parameters_sha256")):
            _record_violation(violations, "contract_grouping_parameters_sha256_invalid")
        if grouping.get("path_derived_groups_forbidden") is not True:
            _record_violation(violations, "contract_path_derived_grouping_not_forbidden")
        if grouping.get("unresolved_grouping_policy") != "block":
            _record_violation(violations, "contract_unresolved_grouping_not_blocked")
        if grouping.get("candidate_coverage_complete") is not True:
            _record_violation(violations, "contract_group_candidate_coverage_incomplete")
        if grouping.get("oversized_bucket_policy") != "block":
            _record_violation(violations, "contract_oversized_group_bucket_not_blocked")
        if tuple(grouping.get("required_relation_fields") or []) != GROUP_FIELDS:
            _record_violation(violations, "contract_relation_fields_mismatch")
        source_group_provenance = normalized_text(
            grouping.get("source_group_provenance")
        ).casefold()
        if source_group_provenance != "custodian_provided":
            _record_violation(violations, "contract_source_group_provenance_invalid")

    temporal_policy = contract.get("temporal_policy")
    embargo_seconds = 0
    allowed_timestamp_provenance: frozenset[str] = frozenset()
    if not isinstance(temporal_policy, dict):
        _record_violation(violations, "contract_temporal_policy_missing")
    else:
        if temporal_policy.get("event_time_field") != "first_seen_time_utc":
            _record_violation(violations, "contract_temporal_event_time_invalid")
        if temporal_policy.get("component_time") != "max_first_seen_time_utc":
            _record_violation(violations, "contract_component_time_policy_invalid")
        if tuple(temporal_policy.get("roles_in_order") or []) != ROLE_ORDER:
            _record_violation(violations, "contract_role_order_mismatch")
        raw_allowed_provenance = temporal_policy.get("allowed_timestamp_provenance")
        if not isinstance(raw_allowed_provenance, list) or not raw_allowed_provenance:
            _record_violation(violations, "contract_timestamp_provenance_policy_missing")
        else:
            normalized_provenance = {
                normalized_text(item).casefold() for item in raw_allowed_provenance
            }
            if (
                "" in normalized_provenance
                or not normalized_provenance.issubset(TRUSTED_TIMESTAMP_PROVENANCE)
                or any(_has_forbidden_timestamp_provenance(item) for item in normalized_provenance)
            ):
                _record_violation(violations, "contract_timestamp_provenance_policy_invalid")
            else:
                allowed_timestamp_provenance = frozenset(normalized_provenance)
        try:
            embargo_seconds = int(temporal_policy.get("embargo_seconds"))
        except (TypeError, ValueError):
            embargo_seconds = -1
        if embargo_seconds < required_embargo_seconds:
            _record_violation(violations, "contract_embargo_below_preregistered_floor")

    oof_policy = contract.get("oof_policy")
    fold_count = 0
    minimum_fit_rows_per_label = 0
    minimum_holdout_rows_per_label = 0
    if not isinstance(oof_policy, dict):
        _record_violation(violations, "contract_oof_policy_missing")
    else:
        if oof_policy.get("mode") != "purged_forward_group":
            _record_violation(violations, "contract_oof_mode_invalid")
        try:
            fold_count = int(oof_policy.get("fold_count"))
        except (TypeError, ValueError):
            fold_count = 0
        if fold_count != REQUIRED_FOLD_COUNT:
            _record_violation(violations, "contract_oof_fold_count_invalid")
        if oof_policy.get("warmup_required") is not True:
            _record_violation(violations, "contract_oof_warmup_not_required")
        if oof_policy.get("eligible_once") is not True:
            _record_violation(violations, "contract_oof_eligible_once_not_required")
        if oof_policy.get("fold_manifest_shared_across_seeds") is not True:
            _record_violation(violations, "contract_oof_seed_partition_not_frozen")
        if tuple(oof_policy.get("seeds") or []) != REQUIRED_SEEDS:
            _record_violation(violations, "contract_oof_seed_set_mismatch")
        try:
            minimum_fit_rows_per_label = int(oof_policy.get("minimum_fit_rows_per_label"))
            minimum_holdout_rows_per_label = int(oof_policy.get("minimum_holdout_rows_per_label"))
        except (TypeError, ValueError):
            minimum_fit_rows_per_label = 0
            minimum_holdout_rows_per_label = 0
        if minimum_fit_rows_per_label < required_minimum_fit_rows_per_label:
            _record_violation(violations, "contract_oof_minimum_fit_label_support_invalid")
        if minimum_holdout_rows_per_label < required_minimum_holdout_rows_per_label:
            _record_violation(violations, "contract_oof_minimum_holdout_label_support_invalid")

    model_input_fields = contract.get("model_input_fields")
    identity_feature_violations: list[str] = []
    if not isinstance(model_input_fields, list) or not model_input_fields:
        _record_violation(violations, "contract_model_input_fields_missing")
    else:
        normalized_fields = tuple(normalized_text(item).casefold() for item in model_input_fields)
        if normalized_fields != RESIDUAL_FUSION_INPUT_FIELDS:
            identity_feature_violations = sorted(
                set(normalized_fields).symmetric_difference(RESIDUAL_FUSION_INPUT_FIELDS)
            )
            if not identity_feature_violations:
                identity_feature_violations = ["residual_fusion_input_field_order"]
        if any(
            field_name in FORBIDDEN_MODEL_INPUT_FIELDS
            or any(token in field_name for token in FORBIDDEN_MODEL_INPUT_TOKENS)
            for field_name in normalized_fields
        ):
            identity_feature_violations.extend(
                field_name
                for field_name in normalized_fields
                if field_name in FORBIDDEN_MODEL_INPUT_FIELDS
                or any(token in field_name for token in FORBIDDEN_MODEL_INPUT_TOKENS)
            )
        if identity_feature_violations:
            _record_violation(violations, "identity_feature_input_detected")
    feature_contract_summary = {
        "schema": None,
        "feature_fields": [],
        "feature_matrix_receipt_required": False,
        "implementation_binding_phase": None,
    }
    feature_contract = contract.get("feature_contract")
    if not isinstance(feature_contract, dict):
        _record_violation(violations, "contract_feature_contract_missing")
    else:
        expected_feature_contract_fields = {
            "schema",
            "feature_fields",
            "feature_matrix_receipt_required",
            "implementation_binding_phase",
        }
        if set(feature_contract) != expected_feature_contract_fields:
            _record_violation(violations, "contract_feature_contract_shape_invalid")
        if feature_contract.get("schema") != FEATURE_CONTRACT_SCHEMA:
            _record_violation(violations, "contract_feature_contract_schema_invalid")
        feature_fields = tuple(feature_contract.get("feature_fields") or [])
        if feature_fields != RESIDUAL_FUSION_INPUT_FIELDS:
            _record_violation(
                violations, "contract_feature_contract_feature_allowlist_invalid"
            )
        if feature_contract.get("feature_matrix_receipt_required") is not True:
            _record_violation(violations, "contract_feature_matrix_receipt_not_required")
        if feature_contract.get("implementation_binding_phase") != IMPLEMENTATION_BINDING_PHASE:
            _record_violation(
                violations, "contract_feature_contract_implementation_binding_phase_invalid"
            )
        feature_contract_summary = {
            "schema": normalized_text(feature_contract.get("schema")),
            "feature_fields": list(feature_fields),
            "feature_matrix_receipt_required": feature_contract.get(
                "feature_matrix_receipt_required"
            )
            is True,
            "implementation_binding_phase": normalized_text(
                feature_contract.get("implementation_binding_phase")
            ),
        }
    if not is_resolved_identifier(contract.get("identity_feature_policy")):
        _record_violation(violations, "contract_identity_feature_policy_missing")

    normalized = {
        "expected_rows": expected_rows,
        "minimum_full_pool_rows": minimum_full_pool_rows,
        "required_embargo_seconds": required_embargo_seconds,
        "required_roles": required_roles,
        "embargo_seconds": max(0, embargo_seconds),
        "fold_count": max(0, fold_count),
        "minimum_fit_rows_per_label": max(0, minimum_fit_rows_per_label),
        "minimum_holdout_rows_per_label": max(0, minimum_holdout_rows_per_label),
        "allowed_timestamp_provenance": allowed_timestamp_provenance,
        "feature_contract": feature_contract_summary,
    }
    return violations, sorted(set(identity_feature_violations)), normalized


def _verify_proposal_binding(
    contract: dict[str, Any],
    *,
    expected_proposal_json: Optional[Path],
) -> tuple[Counter[str], Optional[str]]:
    violations: Counter[str] = Counter()
    binding = contract.get("proposal_binding")
    if not isinstance(binding, dict):
        _record_violation(violations, "proposal_binding_missing")
        return violations, None
    bound_path_text = normalized_text(binding.get("path"))
    if not bound_path_text:
        _record_violation(violations, "proposal_binding_path_missing")
        return violations, None
    bound_path = resolve_path(Path(bound_path_text)).resolve()
    expected_path = (
        resolve_path(expected_proposal_json).resolve()
        if expected_proposal_json is not None
        else CANONICAL_PROPOSAL_PATH.resolve()
    )
    if bound_path != expected_path:
        _record_violation(violations, "proposal_binding_path_mismatch")
        return violations, None
    if not expected_path.is_file():
        _record_violation(violations, "proposal_binding_file_missing")
        return violations, None
    try:
        proposal, actual_sha256 = _read_json_object_with_sha256(expected_path)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        _record_violation(violations, "proposal_binding_json_unreadable")
        return violations, None
    if actual_sha256 != normalized_text(binding.get("sha256")).casefold():
        _record_violation(violations, "proposal_binding_sha256_mismatch")
        return violations, None
    if not isinstance(proposal, dict) or proposal.get("loop_id") != LOOP_ID:
        _record_violation(violations, "proposal_binding_loop_id_invalid")
        return violations, None
    if proposal.get("decision") != "propose_loop164_whole_file_residual_expert_no_execution":
        _record_violation(violations, "proposal_binding_decision_invalid")
        return violations, None
    return violations, actual_sha256


def _parse_fold_id(value: object, *, allow_empty: bool) -> Optional[int]:
    if value in (None, ""):
        if allow_empty:
            return None
        raise ValueError("fold id is required")
    if isinstance(value, bool):
        raise ValueError("fold id must be an integer")
    parsed = int(value)
    if str(parsed) != str(value).strip():
        raise ValueError("fold id must be a canonical integer")
    return parsed


def _has_forbidden_timestamp_provenance(value: object) -> bool:
    provenance = normalized_text(value).casefold()
    return not provenance or any(token in provenance for token in ("path", "mtime", "filesystem"))


def _validate_record(
    raw_record: object,
    *,
    fold_count: int,
    allowed_timestamp_provenance: frozenset[str],
) -> tuple[Optional[MetadataRecord], list[str]]:
    if not isinstance(raw_record, dict):
        return None, ["record_not_object"]
    issues: list[str] = []
    if raw_record.get("record_type") != "sample":
        issues.append("record_type_invalid")
    missing_fields = [field for field in REQUIRED_RECORD_FIELDS if field not in raw_record]
    if missing_fields:
        issues.append("record_required_fields_missing")
        return None, issues

    sample_uid = normalized_text(raw_record.get("sample_uid"))
    if not is_resolved_identifier(sample_uid):
        issues.append("sample_uid_invalid")
    source_sha256 = normalized_text(raw_record.get("source_sha256")).casefold()
    if not is_valid_sha256(source_sha256):
        issues.append("source_sha256_invalid")
    try:
        locked_label = int(raw_record.get("locked_label"))
    except (TypeError, ValueError):
        locked_label = -1
    if locked_label not in {0, 1} or isinstance(raw_record.get("locked_label"), bool):
        issues.append("locked_label_invalid")
    for field_name in (
        "label_provenance",
        "label_evidence_version",
        "family_evidence_version",
        "campaign_evidence_version",
    ):
        if not is_resolved_identifier(raw_record.get(field_name)):
            issues.append(f"{field_name}_invalid")
    try:
        parse_utc_timestamp(raw_record.get("label_frozen_at_utc"))
    except (TypeError, ValueError):
        issues.append("label_frozen_at_invalid")

    try:
        acquisition_time = parse_utc_timestamp(raw_record.get("acquisition_time_utc"))
    except (TypeError, ValueError):
        acquisition_time = datetime.min.replace(tzinfo=timezone.utc)
        issues.append("acquisition_time_invalid")
    try:
        first_seen_time = parse_utc_timestamp(raw_record.get("first_seen_time_utc"))
    except (TypeError, ValueError):
        first_seen_time = datetime.min.replace(tzinfo=timezone.utc)
        issues.append("first_seen_time_invalid")
    timestamp_provenance = normalized_text(raw_record.get("timestamp_provenance")).casefold()
    if _has_forbidden_timestamp_provenance(timestamp_provenance):
        issues.append("path_or_mtime_timestamp_provenance")
    elif timestamp_provenance not in allowed_timestamp_provenance:
        issues.append("timestamp_provenance_untrusted")
    elif first_seen_time > acquisition_time:
        issues.append("first_seen_after_acquisition")

    for field_name in (
        "source_id",
        *GROUP_FIELDS,
        "isolation_component_id",
        "parser_status",
        "schema_version",
        "feature_schema_version",
        "calibration_role",
        "evaluation_generation",
    ):
        if not is_resolved_identifier(raw_record.get(field_name)):
            issues.append(
                "unresolved_group_id" if field_name in GROUP_FIELDS else f"{field_name}_invalid"
            )
    if raw_record.get("grouping_status") != "resolved":
        issues.append("grouping_status_not_resolved")
    if raw_record.get("denominator_status") != "included":
        issues.append("denominator_exclusion_detected")

    split_role = normalized_text(raw_record.get("split_role"))
    if split_role not in ROLE_ORDER:
        issues.append("split_role_invalid")
    oof_role = normalized_text(raw_record.get("oof_role"))
    try:
        outer_fold_id = _parse_fold_id(
            raw_record.get("outer_fold_id"), allow_empty=split_role != "train_oof"
        )
        inner_fold_id = _parse_fold_id(
            raw_record.get("inner_fold_id"), allow_empty=split_role != "train_oof"
        )
    except (TypeError, ValueError):
        outer_fold_id = None
        inner_fold_id = None
        issues.append("oof_fold_id_invalid")

    if split_role == "train_anchor":
        if oof_role != "warmup_not_meta_eligible":
            issues.append("train_anchor_oof_role_invalid")
        if outer_fold_id is not None or inner_fold_id is not None:
            issues.append("train_anchor_fold_assignment_forbidden")
    elif split_role == "train_oof":
        if oof_role != "eligible":
            issues.append("train_oof_role_invalid")
        if outer_fold_id is None or not 0 <= outer_fold_id < fold_count:
            issues.append("outer_fold_id_invalid")
        if inner_fold_id is None or not 0 <= inner_fold_id < fold_count:
            issues.append("inner_fold_id_invalid")
    else:
        if oof_role != "not_applicable":
            issues.append("holdout_oof_role_invalid")
        if outer_fold_id is not None or inner_fold_id is not None:
            issues.append("holdout_fold_assignment_forbidden")

    if issues:
        return None, issues
    return (
        MetadataRecord(
            sample_uid=sample_uid,
            source_sha256=source_sha256,
            locked_label=locked_label,
            acquisition_time=acquisition_time,
            first_seen_time=first_seen_time,
            split_role=split_role,
            oof_role=oof_role,
            outer_fold_id=outer_fold_id,
            inner_fold_id=inner_fold_id,
            isolation_component_id=normalized_text(raw_record.get("isolation_component_id")),
            group_values=(
                normalized_text(raw_record.get("exact_cluster_id")),
                normalized_text(raw_record.get("near_duplicate_cluster_id")),
                (
                    f"{normalized_text(raw_record.get('family_evidence_version'))}:"
                    f"{normalized_text(raw_record.get('family_id'))}"
                ),
                (
                    f"{normalized_text(raw_record.get('campaign_evidence_version'))}:"
                    f"{normalized_text(raw_record.get('campaign_id'))}"
                ),
                (
                    f"{normalized_text(raw_record.get('source_id'))}:"
                    f"{normalized_text(raw_record.get('source_group_id'))}"
                ),
            ),
        ),
        [],
    )


def _iter_jsonl(handle: BinaryIO) -> Iterable[object]:
    for line_number, raw_line in enumerate(handle, start=1):
        try:
            stripped = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"Invalid UTF-8 JSONL at line {line_number}") from exc
        if not stripped:
            continue
        try:
            yield strict_json_loads(stripped)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc


def _open_regular_binary(path: Path) -> BinaryIO:
    _require_no_symlink(path)
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Rows artifact must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    handle = os.fdopen(descriptor, "rb")
    try:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError("Rows artifact changed while opening")
    except Exception:
        handle.close()
        raise
    return handle


def _file_fingerprint(handle: BinaryIO) -> tuple[int, int, int, int, int]:
    status = os.fstat(handle.fileno())
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _build_components(records: Sequence[MetadataRecord]) -> dict[str, list[MetadataRecord]]:
    union_find = UnionFind()
    members_by_relation: dict[tuple[str, str], list[str]] = defaultdict(list)
    records_by_uid: dict[str, MetadataRecord] = {}
    for record in records:
        union_find.add(record.sample_uid)
        records_by_uid[record.sample_uid] = record
        for field_name, field_value in zip(GROUP_FIELDS, record.group_values):
            members_by_relation[(field_name, field_value)].append(record.sample_uid)

    # 所有硬关系取传递闭包，避免只检查单列而漏掉 family->near-duplicate 的跨角色链路。
    for member_uids in members_by_relation.values():
        first_uid = member_uids[0]
        for member_uid in member_uids[1:]:
            union_find.union(first_uid, member_uid)

    components: dict[str, list[MetadataRecord]] = defaultdict(list)
    for sample_uid, record in records_by_uid.items():
        components[union_find.find(sample_uid)].append(record)
    return dict(components)


def _component_temporal_bounds(records: Sequence[MetadataRecord]) -> tuple[datetime, datetime]:
    timestamps = [record.first_seen_time for record in records]
    return min(timestamps), max(timestamps)


def _role_time_summary(
    components: Iterable[Sequence[MetadataRecord]],
) -> dict[str, dict[str, object]]:
    times_by_role: dict[str, list[datetime]] = defaultdict(list)
    counts_by_role: Counter[str] = Counter()
    for component_records in components:
        component_role = component_records[0].split_role
        _component_minimum, component_maximum = _component_temporal_bounds(component_records)
        times_by_role[component_role].append(component_maximum)
        counts_by_role[component_role] += 1
    return {
        role: {
            "components": counts_by_role[role],
            "min_component_time_utc": min(times).isoformat().replace("+00:00", "Z"),
            "max_component_time_utc": max(times).isoformat().replace("+00:00", "Z"),
        }
        for role, times in sorted(times_by_role.items())
    }


def _validate_temporal_order(
    components: Iterable[Sequence[MetadataRecord]],
    *,
    embargo_seconds: int,
    violations: Counter[str],
) -> None:
    component_times_by_role: dict[str, list[datetime]] = defaultdict(list)
    for component_records in components:
        component_role = component_records[0].split_role
        _component_minimum, component_maximum = _component_temporal_bounds(component_records)
        component_times_by_role[component_role].append(component_maximum)
    previous_maximum: Optional[datetime] = None
    embargo = timedelta(seconds=embargo_seconds)
    for role in ROLE_ORDER:
        component_times = component_times_by_role.get(role, [])
        if not component_times:
            continue
        current_minimum = min(component_times)
        current_maximum = max(component_times)
        if previous_maximum is not None and current_minimum <= previous_maximum + embargo:
            _record_violation(violations, "temporal_role_order_or_embargo_violation")
        previous_maximum = (
            current_maximum if previous_maximum is None else max(previous_maximum, current_maximum)
        )


def _validate_oof_assignments(
    components: Iterable[Sequence[MetadataRecord]],
    *,
    fold_count: int,
    embargo_seconds: int,
    minimum_fit_rows_per_label: int,
    minimum_holdout_rows_per_label: int,
    violations: Counter[str],
) -> dict[str, Any]:
    anchors: list[Sequence[MetadataRecord]] = []
    folds: dict[int, list[Sequence[MetadataRecord]]] = defaultdict(list)
    eligible_records: list[MetadataRecord] = []
    for component_records in components:
        role = component_records[0].split_role
        if role == "train_anchor":
            anchors.append(component_records)
        elif role == "train_oof":
            outer_folds = {record.outer_fold_id for record in component_records}
            inner_folds = {record.inner_fold_id for record in component_records}
            if len(outer_folds) != 1:
                _record_violation(violations, "isolation_component_cross_outer_fold")
                continue
            if len(inner_folds) != 1:
                _record_violation(violations, "isolation_component_cross_inner_fold")
                continue
            outer_fold = next(iter(outer_folds))
            if outer_fold is None:
                _record_violation(violations, "outer_fold_id_invalid")
                continue
            folds[outer_fold].append(component_records)
            eligible_records.extend(component_records)

    if not anchors:
        _record_violation(violations, "oof_warmup_rows_missing")
    observed_folds = set(folds)
    expected_folds = set(range(fold_count))
    if observed_folds != expected_folds:
        _record_violation(violations, "oof_fold_coverage_incomplete")

    cumulative_fit_records = [record for component in anchors for record in component]
    fold_label_support: dict[str, dict[str, dict[str, int]]] = {}
    for fold_index in range(fold_count):
        fold_records = [record for component in folds.get(fold_index, []) for record in component]
        fit_labels = Counter(str(record.locked_label) for record in cumulative_fit_records)
        holdout_labels = Counter(str(record.locked_label) for record in fold_records)
        fold_label_support[str(fold_index)] = {
            "fit": dict(sorted(fit_labels.items())),
            "holdout": dict(sorted(holdout_labels.items())),
        }
        if any(fit_labels[str(label)] < minimum_fit_rows_per_label for label in (0, 1)):
            _record_violation(violations, "oof_fit_label_support_inadequate")
        if any(holdout_labels[str(label)] < minimum_holdout_rows_per_label for label in (0, 1)):
            _record_violation(violations, "oof_holdout_label_support_inadequate")
        cumulative_fit_records.extend(fold_records)

    embargo = timedelta(seconds=embargo_seconds)
    previous_maximum: Optional[datetime] = None
    if anchors:
        previous_maximum = max(_component_temporal_bounds(records)[1] for records in anchors)
    for fold_index in range(fold_count):
        fold_components = folds.get(fold_index, [])
        if not fold_components:
            continue
        fold_minimum = min(_component_temporal_bounds(records)[1] for records in fold_components)
        fold_maximum = max(_component_temporal_bounds(records)[1] for records in fold_components)
        if previous_maximum is not None and fold_minimum <= previous_maximum + embargo:
            _record_violation(violations, "outer_fold_temporal_or_embargo_violation")
        previous_maximum = (
            fold_maximum if previous_maximum is None else max(previous_maximum, fold_maximum)
        )

    assignment_payload = "\n".join(
        f"{record.source_sha256}:{record.outer_fold_id}:{record.inner_fold_id}"
        for record in sorted(eligible_records, key=lambda item: item.source_sha256)
    ).encode("ascii")
    return {
        "fold_assignment_fingerprint": hashlib.sha256(assignment_payload).hexdigest()
        if eligible_records
        else None,
        "eligible_rows": len(eligible_records),
        "warmup_rows": sum(len(records) for records in anchors),
        "observed_outer_folds": sorted(observed_folds),
        "fold_label_support": fold_label_support,
        "inner_fold_execution_receipt_required": True,
    }


def _validate_loop164_isolation_contract_core(
    *,
    contract_json: Path,
    rows_jsonl: Optional[Path] = None,
    expected_proposal_json: Optional[Path] = None,
    minimum_full_pool_rows: int = MINIMUM_FULL_POOL_ROWS,
    required_embargo_seconds: int = REQUIRED_EMBARGO_SECONDS,
    required_minimum_fit_rows_per_label: int = REQUIRED_MINIMUM_FIT_ROWS_PER_LABEL,
    required_minimum_holdout_rows_per_label: int = REQUIRED_MINIMUM_HOLDOUT_ROWS_PER_LABEL,
    required_roles: tuple[str, ...] = ROLE_ORDER,
    expected_contract_sha256: Optional[str] = None,
    expected_rows_sha256: Optional[str] = None,
) -> dict[str, Any]:
    contract_path = resolve_path(contract_json).resolve()
    result = _empty_result(contract_bound=False, rows_bound=False)
    violations: Counter[str] = Counter()
    try:
        contract, contract_sha256 = _read_json_object_with_sha256(contract_path)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        _record_violation(violations, "contract_json_unreadable")
        result["violation_counts"] = dict(sorted(violations.items()))
        result["blockers"] = sorted(violations)
        return result
    if expected_contract_sha256 is not None and (
        contract_sha256 != normalized_text(expected_contract_sha256).casefold()
    ):
        _record_violation(violations, "contract_sha256_changed_after_authorization")
        result["violation_counts"] = dict(sorted(violations.items()))
        result["blockers"] = sorted(violations)
        return result

    result["binding_fingerprints"]["contract_sha256"] = contract_sha256

    header_violations, identity_feature_violations, header = _validate_contract_header(
        contract,
        minimum_full_pool_rows=minimum_full_pool_rows,
        required_embargo_seconds=required_embargo_seconds,
        required_minimum_fit_rows_per_label=required_minimum_fit_rows_per_label,
        required_minimum_holdout_rows_per_label=required_minimum_holdout_rows_per_label,
        required_roles=required_roles,
    )
    violations.update(header_violations)
    result["contract_binding_verified"] = not bool(header_violations)
    result["expected_rows"] = header["expected_rows"]
    inventory = contract.get("inventory")
    if isinstance(inventory, dict):
        result["binding_fingerprints"]["inventory_sha256"] = normalized_text(
            inventory.get("inventory_sha256")
        )
    grouping = contract.get("grouping")
    if isinstance(grouping, dict):
        result["binding_fingerprints"]["grouping_parameters_sha256"] = normalized_text(
            grouping.get("parameters_sha256")
        )
    result["feature_contract"] = header["feature_contract"]
    result["identity_feature_violations"] = identity_feature_violations
    if header_violations:
        result["violation_counts"] = dict(sorted(violations.items()))
        result["blockers"] = sorted(violations)
        return result

    proposal_violations, proposal_sha256 = _verify_proposal_binding(
        contract,
        expected_proposal_json=expected_proposal_json,
    )
    violations.update(proposal_violations)
    result["proposal_binding_verified"] = not bool(proposal_violations)
    result["binding_fingerprints"]["proposal_sha256"] = proposal_sha256
    if proposal_violations:
        result["contract_binding_verified"] = False
        result["violation_counts"] = dict(sorted(violations.items()))
        result["blockers"] = sorted(violations)
        return result

    rows_artifact = contract["rows_artifact"]
    expected_rows_path = resolve_path(
        Path(str(rows_artifact["path"])), base=contract_path.parent
    ).resolve()
    selected_rows_path = (
        resolve_path(rows_jsonl).resolve() if rows_jsonl is not None else expected_rows_path
    )
    if selected_rows_path != expected_rows_path:
        _record_violation(violations, "rows_artifact_path_mismatch")
    elif not selected_rows_path.is_file():
        _record_violation(violations, "rows_artifact_missing")
    else:
        try:
            with _open_regular_binary(selected_rows_path) as rows_handle:
                rows_sha256 = sha256_open_file(rows_handle)
        except OSError:
            _record_violation(violations, "rows_artifact_missing")
        else:
            if rows_sha256 != str(rows_artifact["sha256"]).casefold():
                _record_violation(violations, "rows_artifact_sha256_mismatch")
            if expected_rows_sha256 is not None and (
                rows_sha256 != normalized_text(expected_rows_sha256).casefold()
            ):
                _record_violation(violations, "rows_sha256_changed_after_authorization")
    if violations:
        result["violation_counts"] = dict(sorted(violations.items()))
        result["blockers"] = sorted(violations)
        return result
    result["rows_artifact_binding_verified"] = True
    result["binding_fingerprints"]["rows_artifact_sha256"] = rows_sha256

    valid_records: list[MetadataRecord] = []
    source_labels: dict[str, set[int]] = defaultdict(set)
    source_counts: Counter[str] = Counter()
    uid_counts: Counter[str] = Counter()
    exact_cluster_labels: dict[str, set[int]] = defaultdict(set)
    rows_read = 0
    try:
        with _open_regular_binary(selected_rows_path) as rows_handle:
            before_parse = _file_fingerprint(rows_handle)
            if sha256_open_file(rows_handle) != rows_sha256:
                raise ValueError("Rows artifact changed before parsing")
            if _file_fingerprint(rows_handle) != before_parse:
                raise ValueError("Rows artifact changed before parsing")
            for raw_record in _iter_jsonl(rows_handle):
                rows_read += 1
                parsed_record, record_issues = _validate_record(
                    raw_record,
                    fold_count=header["fold_count"],
                    allowed_timestamp_provenance=header["allowed_timestamp_provenance"],
                )
                for code in record_issues:
                    _record_violation(violations, code)
                if parsed_record is None:
                    continue
                valid_records.append(parsed_record)
                source_counts[parsed_record.source_sha256] += 1
                source_labels[parsed_record.source_sha256].add(parsed_record.locked_label)
                uid_counts[parsed_record.sample_uid] += 1
                exact_cluster_labels[parsed_record.group_values[0]].add(parsed_record.locked_label)
            if sha256_open_file(rows_handle) != rows_sha256 or _file_fingerprint(
                rows_handle
            ) != before_parse:
                raise ValueError("Rows artifact changed during parsing")
    except (OSError, ValueError):
        _record_violation(violations, "rows_artifact_changed_during_parse")

    result["rows_read"] = rows_read
    if rows_read != header["expected_rows"]:
        _record_violation(violations, "rows_artifact_count_mismatch")
    for source_sha256, count in source_counts.items():
        if count > 1:
            _record_violation(violations, "duplicate_source_sha256")
        if len(source_labels[source_sha256]) > 1:
            _record_violation(violations, "conflicting_locked_label_for_source_sha256")
    for count in uid_counts.values():
        if count > 1:
            _record_violation(violations, "duplicate_sample_uid")
    for labels in exact_cluster_labels.values():
        if len(labels) > 1:
            _record_violation(violations, "conflicting_locked_label_for_exact_cluster")

    components = _build_components(valid_records)
    declared_component_roots: dict[str, set[str]] = defaultdict(set)
    for component_root, component_records in components.items():
        expected_component_id = stable_component_id(
            record.source_sha256 for record in component_records
        )
        declared_ids = {record.isolation_component_id for record in component_records}
        if declared_ids != {expected_component_id}:
            _record_violation(violations, "isolation_component_id_mismatch")
        for declared_id in declared_ids:
            declared_component_roots[declared_id].add(component_root)
        roles = {record.split_role for record in component_records}
        if len(roles) > 1:
            _record_violation(violations, "isolation_component_cross_split_role")
    for roots in declared_component_roots.values():
        if len(roots) > 1:
            _record_violation(violations, "isolation_component_id_not_transitive_closure")

    _validate_temporal_order(
        components.values(), embargo_seconds=header["embargo_seconds"], violations=violations
    )
    oof_summary = _validate_oof_assignments(
        components.values(),
        fold_count=header["fold_count"],
        embargo_seconds=header["embargo_seconds"],
        minimum_fit_rows_per_label=header["minimum_fit_rows_per_label"],
        minimum_holdout_rows_per_label=header["minimum_holdout_rows_per_label"],
        violations=violations,
    )
    role_counts = Counter(record.split_role for record in valid_records)
    for role in header["required_roles"]:
        if role_counts[role] == 0:
            _record_violation(violations, "required_split_role_missing")
    label_counts_by_role: dict[str, Counter[str]] = defaultdict(Counter)
    sealed_rows_by_role: Counter[str] = Counter()
    for record in valid_records:
        if record.split_role in {"train_anchor", "train_oof"}:
            label_counts_by_role[record.split_role][str(record.locked_label)] += 1
        else:
            sealed_rows_by_role[record.split_role] += 1

    result["counts"] = {
        "valid_rows": len(valid_records),
        "invalid_rows": rows_read - len(valid_records),
        "components": len(components),
        "roles": dict(sorted(role_counts.items())),
        "labels_by_role": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(label_counts_by_role.items())
        },
        "sealed_rows_by_role": dict(sorted(sealed_rows_by_role.items())),
    }
    result["temporal_bounds"] = _role_time_summary(components.values()) if components else {}
    result["oof"] = oof_summary
    result["violation_counts"] = dict(sorted(violations.items()))
    result["blockers"] = sorted(violations)
    contract_ready = not violations and bool(valid_records)
    result["ready_for"]["full_pool_isolation_contract"] = contract_ready
    result["ready_for"]["loop164_train_oof_partition"] = contract_ready
    result["decision"] = "pass" if contract_ready else "block"
    return result


def validate_loop164_isolation_contract(
    *,
    contract_json: Path,
    rows_jsonl: Optional[Path] = None,
    expected_proposal_json: Optional[Path] = None,
    minimum_full_pool_rows: int = MINIMUM_FULL_POOL_ROWS,
    required_embargo_seconds: int = REQUIRED_EMBARGO_SECONDS,
    required_minimum_fit_rows_per_label: int = REQUIRED_MINIMUM_FIT_ROWS_PER_LABEL,
    required_minimum_holdout_rows_per_label: int = REQUIRED_MINIMUM_HOLDOUT_ROWS_PER_LABEL,
    required_roles: tuple[str, ...] = ROLE_ORDER,
    expected_contract_sha256: Optional[str] = None,
    expected_rows_sha256: Optional[str] = None,
    allow_synthetic_test_inputs: bool = False,
) -> dict[str, Any]:
    """Run the core only for explicitly declared temporary synthetic fixtures."""

    if not allow_synthetic_test_inputs:
        raise RuntimeError(
            "Direct metadata validation is disabled; use the authorized CLI after an A2 lease."
        )
    return _validate_loop164_isolation_contract_core(
        contract_json=contract_json,
        rows_jsonl=rows_jsonl,
        expected_proposal_json=expected_proposal_json,
        minimum_full_pool_rows=minimum_full_pool_rows,
        required_embargo_seconds=required_embargo_seconds,
        required_minimum_fit_rows_per_label=required_minimum_fit_rows_per_label,
        required_minimum_holdout_rows_per_label=required_minimum_holdout_rows_per_label,
        required_roles=required_roles,
        expected_contract_sha256=expected_contract_sha256,
        expected_rows_sha256=expected_rows_sha256,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    payload, _ = _read_json_object_with_sha256(path)
    return payload


def _resolve_required_path(value: object, *, base: Path = PROJECT_ROOT) -> Path:
    text = normalized_text(value)
    if not text:
        raise ValueError("Path binding is empty")
    candidate = resolve_path(Path(text), base=base).absolute()
    _require_no_symlink(candidate)
    return candidate.resolve(strict=False)


def _authorization_lease_marker_path(
    authorization_sha256: str,
    lease_id: str,
    *,
    lease_directory: Path = METADATA_LEASE_DIRECTORY,
) -> Path:
    material = f"{authorization_sha256}:{lease_id}".encode("utf-8")
    marker_id = hashlib.sha256(material).hexdigest()
    return lease_directory.resolve() / f"{marker_id}.final.json"


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        file_descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("Refusing to overwrite an existing receipt") from exc
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _validate_a2_metadata_authorization_v1_disabled(
    *,
    authorization_json: Path,
    contract_json: Path,
    output_json: Path,
    now_utc: Optional[datetime] = None,
    expected_authorization_json: Optional[Path] = None,
    expected_output_json: Optional[Path] = None,
    expected_resource_guard_json: Optional[Path] = None,
    expected_validator_script: Optional[Path] = None,
    lease_directory: Optional[Path] = None,
) -> dict[str, Any]:
    """Validate a one-shot A2 metadata authority before any JSONL is opened."""

    raise RuntimeError("Metadata authorization v1 is disabled")

    try:
        authorization_path = _resolve_required_path(authorization_json)
        contract_path = _resolve_required_path(contract_json)
        output_path = _resolve_required_path(output_json)
        canonical_authorization_path = _resolve_required_path(
            expected_authorization_json or CANONICAL_A2_METADATA_AUTHORIZATION_PATH
        )
        canonical_output_path = _resolve_required_path(
            expected_output_json or CANONICAL_ISOLATION_RECEIPT_PATH
        )
        canonical_guard_path = _resolve_required_path(
            expected_resource_guard_json
            or (PROJECT_ROOT / "reports/roadmap_9997/loop164/resource_guard.json")
        )
        validator_path = _resolve_required_path(expected_validator_script or Path(__file__))
        lease_root = _resolve_required_path(lease_directory or METADATA_LEASE_DIRECTORY)
    except ValueError:
        return {"ready": False, "failures": ["a2_authorization_path_binding_invalid"]}
    if authorization_path != canonical_authorization_path:
        return {"ready": False, "failures": ["a2_authorization_path_not_canonical"]}
    if output_path != canonical_output_path:
        return {"ready": False, "failures": ["a2_authorization_output_path_not_canonical"]}

    current_time = now_utc or datetime.now(timezone.utc)
    failures: list[str] = []
    try:
        authorization, authorization_sha256 = _read_json_object_with_sha256(authorization_path)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return {"ready": False, "failures": ["a2_authorization_unreadable"]}

    expected_header = {
        "schema": METADATA_AUTHORIZATION_SCHEMA,
        "loop_id": LOOP_ID,
        "authorization_level": "A2_metadata_isolation_validation_only",
        "decision": "allow_single_metadata_isolation_validation",
        "execution_environment": "custodian_side_metadata_only",
        "operation": "loop164_metadata_isolation_validation",
    }
    for field_name, expected_value in expected_header.items():
        if authorization.get(field_name) != expected_value:
            failures.append(f"a2_authorization_{field_name}_invalid")

    runtime_binding = authorization.get("runtime_binding")
    if not isinstance(runtime_binding, dict):
        failures.append("a2_authorization_runtime_binding_missing")
    else:
        try:
            bound_cwd = _resolve_required_path(runtime_binding.get("cwd"))
        except ValueError:
            bound_cwd = None
            failures.append("a2_authorization_cwd_missing")
        if bound_cwd is not None and bound_cwd != PROJECT_ROOT.resolve():
            failures.append("a2_authorization_cwd_mismatch")
        try:
            bound_python = _resolve_required_path(runtime_binding.get("python_executable"))
        except ValueError:
            bound_python = None
            failures.append("a2_authorization_python_executable_missing")
        if bound_python is not None and bound_python != Path(sys.executable).resolve():
            failures.append("a2_authorization_python_executable_mismatch")
        if not is_valid_sha256(runtime_binding.get("python_sha256")):
            failures.append("a2_authorization_python_sha256_invalid")
        elif bound_python is not None:
            try:
                if sha256_file(bound_python) != normalized_text(
                    runtime_binding.get("python_sha256")
                ).casefold():
                    failures.append("a2_authorization_python_sha256_mismatch")
            except OSError:
                failures.append("a2_authorization_python_executable_unreadable")

    validator_binding = authorization.get("validator_binding")
    if not isinstance(validator_binding, dict):
        failures.append("a2_authorization_validator_binding_missing")
    else:
        try:
            bound_validator_path = _resolve_required_path(validator_binding.get("path"))
        except ValueError:
            bound_validator_path = None
            failures.append("a2_authorization_validator_path_invalid")
        if bound_validator_path is not None and bound_validator_path != validator_path:
            failures.append("a2_authorization_validator_path_mismatch")
        if not is_valid_sha256(validator_binding.get("sha256")):
            failures.append("a2_authorization_validator_sha256_invalid")
        else:
            try:
                if sha256_file(validator_path) != normalized_text(
                    validator_binding.get("sha256")
                ).casefold():
                    failures.append("a2_authorization_validator_sha256_mismatch")
            except OSError:
                failures.append("a2_authorization_validator_unreadable")

    canonical_argv = authorization.get("canonical_argv")
    if (
        not isinstance(canonical_argv, list)
        or not canonical_argv
        or any(not isinstance(item, str) or not item for item in canonical_argv)
    ):
        failures.append("a2_authorization_canonical_argv_invalid")

    lease = authorization.get("one_shot_lease")
    if not isinstance(lease, dict):
        lease_id = ""
        failures.append("a2_authorization_lease_missing")
    else:
        lease_id = normalized_text(lease.get("lease_id"))
        if not lease_id or len(lease_id) > 128:
            failures.append("a2_authorization_lease_id_invalid")
        if lease.get("state") != "ready":
            failures.append("a2_authorization_lease_not_ready")
        if lease.get("purpose") != "single_metadata_isolation_validation":
            failures.append("a2_authorization_lease_purpose_invalid")

    try:
        issued_at_utc = parse_utc_timestamp(authorization.get("issued_at_utc"))
        not_before_utc = parse_utc_timestamp(authorization.get("not_before_utc"))
        expires_at_utc = parse_utc_timestamp(authorization.get("expires_at_utc"))
    except (TypeError, ValueError):
        failures.append("a2_authorization_time_window_invalid")
    else:
        if issued_at_utc > not_before_utc or not_before_utc >= expires_at_utc:
            failures.append("a2_authorization_time_window_order_invalid")
        if expires_at_utc - issued_at_utc > timedelta(
            seconds=MAX_A2_METADATA_AUTHORIZATION_TTL_SECONDS
        ):
            failures.append("a2_authorization_ttl_exceeds_maximum")
        if current_time < not_before_utc:
            failures.append("a2_authorization_not_yet_valid")
        if expires_at_utc <= current_time:
            failures.append("a2_authorization_expired")

    contract_binding = authorization.get("contract_binding")
    contract: Optional[dict[str, Any]] = None
    contract_sha256: Optional[str] = None
    if not isinstance(contract_binding, dict):
        failures.append("a2_authorization_contract_binding_missing")
    else:
        try:
            bound_contract_path = _resolve_required_path(contract_binding.get("path"))
        except ValueError:
            bound_contract_path = None
            failures.append("a2_authorization_contract_path_invalid")
        if bound_contract_path is not None and bound_contract_path != contract_path:
            failures.append("a2_authorization_contract_path_mismatch")
        try:
            contract, contract_sha256 = _read_json_object_with_sha256(contract_path)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            failures.append("a2_authorization_contract_unreadable")
        if not is_valid_sha256(contract_binding.get("sha256")):
            failures.append("a2_authorization_contract_sha256_invalid")
        elif contract_sha256 is not None and contract_sha256 != normalized_text(
            contract_binding.get("sha256")
        ).casefold():
            failures.append("a2_authorization_contract_sha256_mismatch")

    output_binding = authorization.get("output_binding")
    if not isinstance(output_binding, dict):
        failures.append("a2_authorization_output_binding_missing")
    else:
        try:
            bound_output_path = _resolve_required_path(output_binding.get("path"))
        except ValueError:
            bound_output_path = None
            failures.append("a2_authorization_output_path_invalid")
        if bound_output_path is not None and bound_output_path != output_path:
            failures.append("a2_authorization_output_path_mismatch")

    resource_guard_binding = authorization.get("resource_guard_binding")
    try:
        max_guard_age_seconds = int(authorization.get("max_resource_guard_age_seconds"))
    except (TypeError, ValueError):
        max_guard_age_seconds = 0
    if not 1 <= max_guard_age_seconds <= 3600:
        failures.append("a2_authorization_resource_guard_age_limit_invalid")
    if not isinstance(resource_guard_binding, dict):
        failures.append("a2_authorization_resource_guard_binding_missing")
    else:
        try:
            guard_path = _resolve_required_path(resource_guard_binding.get("path"))
        except ValueError:
            guard_path = None
            failures.append("a2_authorization_resource_guard_path_invalid")
        if guard_path is not None and guard_path != canonical_guard_path:
            failures.append("a2_authorization_resource_guard_path_mismatch")
        if guard_path is not None and not guard_path.is_file():
            failures.append("a2_authorization_resource_guard_missing")
        elif guard_path is not None and not is_valid_sha256(resource_guard_binding.get("sha256")):
            failures.append("a2_authorization_resource_guard_sha256_invalid")
        elif guard_path is not None:
            try:
                guard_payload, guard_sha256 = _read_json_object_with_sha256(guard_path)
            except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
                failures.append("a2_authorization_resource_guard_unreadable")
            else:
                if guard_sha256 != normalized_text(resource_guard_binding.get("sha256")).casefold():
                    failures.append("a2_authorization_resource_guard_sha256_mismatch")
                else:
                    from pre_run_resource_leak_guard import validate_guard_receipt

                    guard_check = validate_guard_receipt(
                        guard_payload,
                        expected_target_scripts=[validator_path],
                        expected_command=canonical_argv
                        if isinstance(canonical_argv, list)
                        else None,
                        expected_cwd=PROJECT_ROOT,
                        max_age_seconds=max_guard_age_seconds,
                        now=current_time.timestamp(),
                    )
                    if guard_payload.get("decision") != "pass" or not guard_check["valid"]:
                        failures.append("a2_authorization_resource_guard_not_ready")

    if failures:
        return {"ready": False, "failures": sorted(set(failures))}
    if contract is None or contract_sha256 is None:
        return {"ready": False, "failures": ["a2_authorization_contract_unreadable"]}
    rows_artifact = contract.get("rows_artifact")
    rows_binding = authorization.get("rows_artifact_binding")
    if not isinstance(rows_artifact, dict) or not isinstance(rows_binding, dict):
        return {"ready": False, "failures": ["a2_authorization_rows_binding_missing"]}
    try:
        expected_rows_path = _resolve_required_path(
            rows_artifact.get("path"), base=contract_path.parent
        )
        bound_rows_path = _resolve_required_path(rows_binding.get("path"))
    except ValueError:
        return {"ready": False, "failures": ["a2_authorization_rows_path_invalid"]}
    if bound_rows_path != expected_rows_path:
        return {"ready": False, "failures": ["a2_authorization_rows_path_mismatch"]}
    contract_rows_sha256 = normalized_text(rows_artifact.get("sha256")).casefold()
    bound_rows_sha256 = normalized_text(rows_binding.get("sha256")).casefold()
    if not is_valid_sha256(contract_rows_sha256) or bound_rows_sha256 != contract_rows_sha256:
        return {"ready": False, "failures": ["a2_authorization_rows_sha256_mismatch"]}
    marker_path = _authorization_lease_marker_path(
        authorization_sha256,
        lease_id,
        lease_directory=lease_root,
    )
    if marker_path.exists():
        return {"ready": False, "failures": ["a2_authorization_lease_already_consumed"]}
    return {
        "ready": True,
        "authorization_path": authorization_path,
        "authorization_sha256": authorization_sha256,
        "contract_sha256": contract_sha256,
        "lease_id": lease_id,
        "lease_marker_path": marker_path,
        "rows_path": expected_rows_path,
        "rows_sha256": contract_rows_sha256,
        "output_path": output_path,
    }


def _legacy_validate_a2_metadata_authorization(
    *,
    authorization_json: Path,
    contract_json: Path,
    output_json: Path,
    now_utc: Optional[datetime] = None,
) -> dict[str, Any]:
    """Disabled compatibility stub; callers must use the hardened verifier."""

    raise RuntimeError("Legacy A2 metadata authorization validation is disabled")

    authorization_path = resolve_path(authorization_json).resolve()
    contract_path = resolve_path(contract_json).resolve()
    output_path = resolve_path(output_json).resolve()
    current_time = now_utc or datetime.now(timezone.utc)
    failures: list[str] = []
    try:
        authorization = _read_json_object(authorization_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {"ready": False, "failures": ["a2_authorization_unreadable"]}
    if authorization.get("schema") != METADATA_AUTHORIZATION_SCHEMA:
        failures.append("a2_authorization_schema_mismatch")
    if authorization.get("loop_id") != LOOP_ID:
        failures.append("a2_authorization_loop_id_mismatch")
    if authorization.get("authorization_level") != "A2_metadata_isolation_validation_only":
        failures.append("a2_authorization_level_invalid")
    if authorization.get("decision") != "allow_single_metadata_isolation_validation":
        failures.append("a2_authorization_decision_invalid")
    if authorization.get("execution_environment") != "custodian_side_metadata_only":
        failures.append("a2_authorization_execution_environment_invalid")
    runtime_binding = authorization.get("runtime_binding")
    if not isinstance(runtime_binding, dict):
        failures.append("a2_authorization_runtime_binding_missing")
    else:
        bound_cwd = resolve_path(Path(normalized_text(runtime_binding.get("cwd")))).resolve()
        if bound_cwd != PROJECT_ROOT.resolve():
            failures.append("a2_authorization_cwd_mismatch")
        bound_python = resolve_path(
            Path(normalized_text(runtime_binding.get("python_executable")))
        ).resolve()
        if bound_python != Path(sys.executable).resolve():
            failures.append("a2_authorization_python_executable_mismatch")
    lease = authorization.get("one_shot_lease")
    if not isinstance(lease, dict):
        failures.append("a2_authorization_lease_missing")
        lease_id = ""
    else:
        lease_id = normalized_text(lease.get("lease_id"))
        if not lease_id:
            failures.append("a2_authorization_lease_id_invalid")
        if lease.get("state") != "ready":
            failures.append("a2_authorization_lease_not_ready")
    expires_at = authorization.get("expires_at_utc")
    try:
        expires_at_utc = parse_utc_timestamp(expires_at)
    except (TypeError, ValueError):
        failures.append("a2_authorization_expiry_invalid")
    else:
        if expires_at_utc <= current_time:
            failures.append("a2_authorization_expired")

    contract_binding = authorization.get("contract_binding")
    if not isinstance(contract_binding, dict):
        failures.append("a2_authorization_contract_binding_missing")
    else:
        bound_contract_path = resolve_path(
            Path(normalized_text(contract_binding.get("path")))
        ).resolve()
        if bound_contract_path != contract_path:
            failures.append("a2_authorization_contract_path_mismatch")
        elif not contract_path.is_file():
            failures.append("a2_authorization_contract_file_missing")
        elif (
            sha256_file(contract_path) != normalized_text(contract_binding.get("sha256")).casefold()
        ):
            failures.append("a2_authorization_contract_sha256_mismatch")

    output_binding = authorization.get("output_binding")
    if not isinstance(output_binding, dict):
        failures.append("a2_authorization_output_binding_missing")
    else:
        bound_output_path = resolve_path(
            Path(normalized_text(output_binding.get("path")))
        ).resolve()
        if bound_output_path != output_path:
            failures.append("a2_authorization_output_path_mismatch")

    resource_guard_binding = authorization.get("resource_guard_binding")
    try:
        max_resource_guard_age_seconds = int(authorization.get("max_resource_guard_age_seconds"))
    except (TypeError, ValueError):
        max_resource_guard_age_seconds = 0
    if max_resource_guard_age_seconds < 1:
        failures.append("a2_authorization_resource_guard_age_limit_invalid")
    if not isinstance(resource_guard_binding, dict):
        failures.append("a2_authorization_resource_guard_binding_missing")
    else:
        guard_path = resolve_path(
            Path(normalized_text(resource_guard_binding.get("path")))
        ).resolve()
        if not guard_path.is_file():
            failures.append("a2_authorization_resource_guard_missing")
        elif (
            sha256_file(guard_path)
            != normalized_text(resource_guard_binding.get("sha256")).casefold()
        ):
            failures.append("a2_authorization_resource_guard_sha256_mismatch")
        else:
            try:
                guard_payload = _read_json_object(guard_path)
            except (OSError, ValueError, json.JSONDecodeError):
                failures.append("a2_authorization_resource_guard_unreadable")
            else:
                if (
                    guard_payload.get("guard_ready") is not True
                    or guard_payload.get("decision") != "pass"
                ):
                    failures.append("a2_authorization_resource_guard_not_ready")
                receipt = guard_payload.get("receipt")
                try:
                    created_at_unix = float(receipt.get("created_at_unix"))
                except (AttributeError, TypeError, ValueError):
                    failures.append("a2_authorization_resource_guard_receipt_invalid")
                else:
                    age_seconds = current_time.timestamp() - created_at_unix
                    if age_seconds < -300 or age_seconds > max_resource_guard_age_seconds:
                        failures.append("a2_authorization_resource_guard_stale")

    if failures:
        return {"ready": False, "failures": sorted(set(failures))}
    try:
        contract = _read_json_object(contract_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {"ready": False, "failures": ["a2_authorization_contract_unreadable"]}
    rows_artifact = contract.get("rows_artifact")
    rows_binding = authorization.get("rows_artifact_binding")
    if not isinstance(rows_artifact, dict) or not isinstance(rows_binding, dict):
        return {"ready": False, "failures": ["a2_authorization_rows_binding_missing"]}
    expected_rows_path = resolve_path(
        Path(normalized_text(rows_artifact.get("path"))), base=contract_path.parent
    ).resolve()
    bound_rows_path = resolve_path(Path(normalized_text(rows_binding.get("path")))).resolve()
    if bound_rows_path != expected_rows_path:
        return {"ready": False, "failures": ["a2_authorization_rows_path_mismatch"]}
    if (
        normalized_text(rows_binding.get("sha256")).casefold()
        != normalized_text(rows_artifact.get("sha256")).casefold()
    ):
        return {"ready": False, "failures": ["a2_authorization_rows_sha256_mismatch"]}
    marker_path = _authorization_lease_marker_path(authorization_path, lease_id)
    if marker_path.exists():
        return {"ready": False, "failures": ["a2_authorization_lease_already_consumed"]}
    return {
        "ready": True,
        "authorization_path": authorization_path,
        "authorization_sha256": sha256_file(authorization_path),
        "lease_id": lease_id,
        "lease_marker_path": marker_path,
        "rows_path": expected_rows_path,
    }


def _require_exact_object(
    payload: object,
    expected_fields: set[str],
    *,
    label: str,
    failures: list[str],
) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        failures.append(f"{label}_not_object")
        return None
    actual_fields = set(payload)
    if expected_fields - actual_fields:
        failures.append(f"{label}_missing_fields")
    if actual_fields - expected_fields:
        failures.append(f"{label}_unexpected_fields")
    return payload


def _metadata_lease_consumption_id(
    *, trusted_key_fingerprint: str, verification_receipt_sha256: str, lease_id: str
) -> str:
    material = ":".join(
        (LOOP_ID, trusted_key_fingerprint, verification_receipt_sha256, lease_id)
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _metadata_lease_marker_path(*, consumption_id: str, lease_directory: Path) -> Path:
    return lease_directory / f"{consumption_id}.final.json"


def _validate_metadata_trust_anchor(
    *,
    trust_anchor_json: Optional[Path],
    expected_trusted_key_fingerprint: Optional[str],
    authority_attestation: object,
    failures: list[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if trust_anchor_json is None or not is_valid_sha256(expected_trusted_key_fingerprint):
        failures.append("a2_authorization_external_trust_anchor_not_configured")
        return None, None, None
    try:
        anchor_path = _resolve_external_path(trust_anchor_json)
        anchor, anchor_sha256 = _read_json_object_with_sha256(anchor_path)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        failures.append("a2_authorization_trust_anchor_unreadable")
        return None, None, None
    anchor = _require_exact_object(
        anchor,
        {
            "schema",
            "loop_id",
            "trusted_key_fingerprint",
            "verification_receipt_sha256",
            "root_state",
        },
        label="a2_authorization_trust_anchor",
        failures=failures,
    )
    attestation = _require_exact_object(
        authority_attestation,
        {"trusted_key_fingerprint", "trust_anchor_sha256", "verification_receipt_sha256"},
        label="a2_authorization_attestation",
        failures=failures,
    )
    if anchor is None or attestation is None:
        return None, None, None
    trusted_key = normalized_text(anchor.get("trusted_key_fingerprint")).casefold()
    verification_receipt = normalized_text(anchor.get("verification_receipt_sha256")).casefold()
    if anchor.get("schema") != METADATA_TRUST_ANCHOR_SCHEMA or anchor.get("loop_id") != LOOP_ID:
        failures.append("a2_authorization_trust_anchor_schema_invalid")
    if anchor.get("root_state") != "externally_verified":
        failures.append("a2_authorization_trust_anchor_not_externally_verified")
    if not is_valid_sha256(trusted_key) or not is_valid_sha256(verification_receipt):
        failures.append("a2_authorization_trust_anchor_binding_invalid")
    if trusted_key != normalized_text(expected_trusted_key_fingerprint).casefold():
        failures.append("a2_authorization_trusted_key_not_pinned")
    if (
        attestation.get("trusted_key_fingerprint") != trusted_key
        or attestation.get("verification_receipt_sha256") != verification_receipt
        or normalized_text(attestation.get("trust_anchor_sha256")).casefold() != anchor_sha256
    ):
        failures.append("a2_authorization_attestation_mismatch")
    return anchor_sha256, trusted_key, verification_receipt


def _validate_metadata_resource_guard(
    *,
    guard_path: Path,
    expected_sha256: str,
    validator_path: Path,
    canonical_argv: Sequence[str],
    max_guard_age_seconds: int,
    current_time: datetime,
    failures: list[str],
) -> Optional[str]:
    try:
        guard_payload, guard_sha256 = _read_json_object_with_sha256(guard_path)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        failures.append("a2_authorization_resource_guard_unreadable")
        return None
    if guard_sha256 != expected_sha256:
        failures.append("a2_authorization_resource_guard_sha256_mismatch")
        return None
    from pre_run_resource_leak_guard import validate_guard_receipt

    guard_check = validate_guard_receipt(
        guard_payload,
        expected_target_scripts=[validator_path],
        expected_command=canonical_argv,
        expected_cwd=PROJECT_ROOT,
        max_age_seconds=max_guard_age_seconds,
        now=current_time.timestamp(),
    )
    if guard_payload.get("decision") != "pass" or not guard_check["valid"]:
        failures.append("a2_authorization_resource_guard_not_ready")
        return None
    return guard_sha256


def validate_a2_metadata_authorization(
    *,
    authorization_json: Path,
    contract_json: Path,
    output_json: Path,
    now_utc: Optional[datetime] = None,
    expected_authorization_json: Optional[Path] = None,
    expected_output_json: Optional[Path] = None,
    expected_resource_guard_json: Optional[Path] = None,
    expected_validator_script: Optional[Path] = None,
    lease_directory: Optional[Path] = None,
    trust_anchor_json: Optional[Path] = None,
    expected_trusted_key_fingerprint: Optional[str] = None,
    actual_argv: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Validate a v2 metadata authority without opening protected JSONL rows."""

    try:
        authorization_path = _resolve_required_path(authorization_json)
        contract_path = _resolve_required_path(contract_json)
        output_path = _resolve_required_path(output_json)
        canonical_authorization_path = _resolve_required_path(
            expected_authorization_json or CANONICAL_A2_METADATA_AUTHORIZATION_PATH
        )
        canonical_output_path = _resolve_required_path(
            expected_output_json or CANONICAL_ISOLATION_RECEIPT_PATH
        )
        canonical_guard_path = _resolve_required_path(
            expected_resource_guard_json
            or (PROJECT_ROOT / "reports/roadmap_9997/loop164/resource_guard.json")
        )
        validator_path = _resolve_required_path(expected_validator_script or Path(__file__))
        lease_root = _resolve_required_path(lease_directory or METADATA_LEASE_DIRECTORY)
    except ValueError:
        return {"ready": False, "failures": ["a2_authorization_path_binding_invalid"]}
    if authorization_path != canonical_authorization_path:
        return {"ready": False, "failures": ["a2_authorization_path_not_canonical"]}
    if output_path != canonical_output_path:
        return {"ready": False, "failures": ["a2_authorization_output_path_not_canonical"]}
    if output_path.exists():
        return {"ready": False, "failures": ["a2_authorization_output_already_exists"]}

    current_time = now_utc or datetime.now(timezone.utc)
    failures: list[str] = []
    try:
        authorization, authorization_sha256 = _read_json_object_with_sha256(authorization_path)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return {"ready": False, "failures": ["a2_authorization_unreadable"]}
    authorization = _require_exact_object(
        authorization,
        {
            "schema",
            "loop_id",
            "authorization_level",
            "decision",
            "execution_environment",
            "operation",
            "authority_scope",
            "issued_at_utc",
            "not_before_utc",
            "expires_at_utc",
            "authority_attestation",
            "runtime_binding",
            "validator_binding",
            "validator_source_closure",
            "canonical_argv",
            "contract_binding",
            "rows_artifact_binding",
            "metadata_root_binding",
            "output_binding",
            "resource_guard_binding",
            "max_resource_guard_age_seconds",
            "one_shot_lease",
        },
        label="a2_authorization",
        failures=failures,
    )
    if authorization is None:
        return {"ready": False, "failures": sorted(set(failures))}
    expected_header = {
        "schema": METADATA_AUTHORIZATION_SCHEMA,
        "loop_id": LOOP_ID,
        "authorization_level": "A2_metadata_isolation_validation_only",
        "decision": "allow_single_metadata_isolation_validation",
        "execution_environment": "custodian_side_metadata_only",
        "operation": "loop164_metadata_isolation_validation",
    }
    for field_name, expected_value in expected_header.items():
        if authorization.get(field_name) != expected_value:
            failures.append(f"a2_authorization_{field_name}_invalid")
    if authorization.get("authority_scope") != METADATA_AUTHORITY_SCOPE:
        failures.append("a2_authorization_authority_scope_invalid")

    canonical_argv = authorization.get("canonical_argv")
    if (
        not isinstance(canonical_argv, list)
        or not canonical_argv
        or any(not isinstance(item, str) or not item for item in canonical_argv)
    ):
        failures.append("a2_authorization_canonical_argv_invalid")
        canonical_argv = []
    observed_argv = list(actual_argv if actual_argv is not None else sys.argv[1:])
    if observed_argv != canonical_argv:
        failures.append("a2_authorization_canonical_argv_mismatch")

    try:
        issued_at_utc = parse_utc_timestamp(authorization.get("issued_at_utc"))
        not_before_utc = parse_utc_timestamp(authorization.get("not_before_utc"))
        expires_at_utc = parse_utc_timestamp(authorization.get("expires_at_utc"))
    except (TypeError, ValueError):
        failures.append("a2_authorization_time_window_invalid")
    else:
        if issued_at_utc > not_before_utc or not_before_utc >= expires_at_utc:
            failures.append("a2_authorization_time_window_order_invalid")
        if expires_at_utc - issued_at_utc > timedelta(
            seconds=MAX_A2_METADATA_AUTHORIZATION_TTL_SECONDS
        ):
            failures.append("a2_authorization_ttl_exceeds_maximum")
        if not_before_utc > current_time or expires_at_utc <= current_time:
            failures.append("a2_authorization_not_currently_valid")

    trust_anchor_sha256, trusted_key_fingerprint, verification_receipt_sha256 = (
        _validate_metadata_trust_anchor(
            trust_anchor_json=trust_anchor_json,
            expected_trusted_key_fingerprint=expected_trusted_key_fingerprint,
            authority_attestation=authorization.get("authority_attestation"),
            failures=failures,
        )
    )

    runtime_binding = _require_exact_object(
        authorization.get("runtime_binding"),
        {"cwd", "python_executable", "python_sha256"},
        label="a2_authorization_runtime_binding",
        failures=failures,
    )
    runtime_python: Optional[Path] = None
    runtime_python_sha256: Optional[str] = None
    if runtime_binding is not None:
        try:
            bound_cwd = _resolve_required_path(runtime_binding.get("cwd"))
            runtime_python = _resolve_required_path(runtime_binding.get("python_executable"))
        except ValueError:
            failures.append("a2_authorization_runtime_path_invalid")
        else:
            if bound_cwd != PROJECT_ROOT.resolve():
                failures.append("a2_authorization_cwd_mismatch")
            if runtime_python != Path(sys.executable).resolve():
                failures.append("a2_authorization_python_executable_mismatch")
            if not is_valid_sha256(runtime_binding.get("python_sha256")):
                failures.append("a2_authorization_python_sha256_invalid")
            else:
                try:
                    runtime_python_sha256 = sha256_file(runtime_python)
                except OSError:
                    failures.append("a2_authorization_python_executable_unreadable")
                else:
                    if runtime_python_sha256 != normalized_text(
                        runtime_binding.get("python_sha256")
                    ).casefold():
                        failures.append("a2_authorization_python_sha256_mismatch")

    validator_binding = _require_exact_object(
        authorization.get("validator_binding"),
        {"path", "sha256"},
        label="a2_authorization_validator_binding",
        failures=failures,
    )
    if validator_binding is not None:
        try:
            bound_validator_path = _resolve_required_path(validator_binding.get("path"))
        except ValueError:
            failures.append("a2_authorization_validator_path_invalid")
        else:
            if bound_validator_path != validator_path:
                failures.append("a2_authorization_validator_path_mismatch")
            if not is_valid_sha256(validator_binding.get("sha256")) or sha256_file(
                validator_path
            ) != normalized_text(validator_binding.get("sha256")).casefold():
                failures.append("a2_authorization_validator_sha256_mismatch")

    source_closure = _require_exact_object(
        authorization.get("validator_source_closure"),
        {"schema", "files", "closure_sha256"},
        label="a2_authorization_validator_source_closure",
        failures=failures,
    )
    validator_source_closure_sha256: Optional[str] = None
    if source_closure is not None:
        try:
            expected_files, validator_source_closure_sha256 = _validator_source_closure()
        except (OSError, ValueError):
            failures.append("a2_authorization_validator_source_closure_unreadable")
        else:
            if (
                source_closure.get("schema") != METADATA_VALIDATOR_CLOSURE_SCHEMA
                or source_closure.get("files") != expected_files
                or source_closure.get("closure_sha256") != validator_source_closure_sha256
            ):
                failures.append("a2_authorization_validator_source_closure_mismatch")

    lease = _require_exact_object(
        authorization.get("one_shot_lease"),
        {"lease_id", "purpose", "state"},
        label="a2_authorization_lease",
        failures=failures,
    )
    lease_id = ""
    if lease is not None:
        lease_id = normalized_text(lease.get("lease_id"))
        if not lease_id or len(lease_id) > 128:
            failures.append("a2_authorization_lease_id_invalid")
        if lease.get("state") != "ready" or lease.get("purpose") != "single_metadata_isolation_validation":
            failures.append("a2_authorization_lease_invalid")

    contract_binding = _require_exact_object(
        authorization.get("contract_binding"),
        {"path", "sha256"},
        label="a2_authorization_contract_binding",
        failures=failures,
    )
    contract: Optional[dict[str, Any]] = None
    contract_sha256: Optional[str] = None
    if contract_binding is not None:
        try:
            bound_contract_path = _resolve_required_path(contract_binding.get("path"))
            contract, contract_sha256 = _read_json_object_with_sha256(contract_path)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            failures.append("a2_authorization_contract_unreadable")
        else:
            if bound_contract_path != contract_path:
                failures.append("a2_authorization_contract_path_mismatch")
            if not is_valid_sha256(contract_binding.get("sha256")) or contract_sha256 != normalized_text(
                contract_binding.get("sha256")
            ).casefold():
                failures.append("a2_authorization_contract_sha256_mismatch")

    output_binding = _require_exact_object(
        authorization.get("output_binding"),
        {"path"},
        label="a2_authorization_output_binding",
        failures=failures,
    )
    if output_binding is not None:
        try:
            bound_output_path = _resolve_required_path(output_binding.get("path"))
        except ValueError:
            failures.append("a2_authorization_output_path_invalid")
        else:
            if bound_output_path != output_path:
                failures.append("a2_authorization_output_path_mismatch")

    resource_guard_binding = _require_exact_object(
        authorization.get("resource_guard_binding"),
        {"path", "sha256"},
        label="a2_authorization_resource_guard_binding",
        failures=failures,
    )
    try:
        max_guard_age_seconds = int(authorization.get("max_resource_guard_age_seconds"))
    except (TypeError, ValueError):
        max_guard_age_seconds = 0
    if not 1 <= max_guard_age_seconds <= 3600:
        failures.append("a2_authorization_resource_guard_age_limit_invalid")
    guard_path: Optional[Path] = None
    guard_sha256: Optional[str] = None
    if resource_guard_binding is not None:
        try:
            guard_path = _resolve_required_path(resource_guard_binding.get("path"))
        except ValueError:
            failures.append("a2_authorization_resource_guard_path_invalid")
        else:
            if guard_path != canonical_guard_path:
                failures.append("a2_authorization_resource_guard_path_mismatch")
            if not guard_path.is_file():
                failures.append("a2_authorization_resource_guard_missing")
            elif not is_valid_sha256(resource_guard_binding.get("sha256")):
                failures.append("a2_authorization_resource_guard_sha256_invalid")
            else:
                guard_sha256 = _validate_metadata_resource_guard(
                    guard_path=guard_path,
                    expected_sha256=normalized_text(resource_guard_binding.get("sha256")).casefold(),
                    validator_path=validator_path,
                    canonical_argv=canonical_argv,
                    max_guard_age_seconds=max_guard_age_seconds,
                    current_time=current_time,
                    failures=failures,
                )

    metadata_root_binding = _require_exact_object(
        authorization.get("metadata_root_binding"),
        {"path"},
        label="a2_authorization_metadata_root_binding",
        failures=failures,
    )
    metadata_root: Optional[Path] = None
    if metadata_root_binding is not None:
        try:
            metadata_root = _resolve_external_path(metadata_root_binding.get("path"))
        except ValueError:
            failures.append("a2_authorization_metadata_root_invalid")

    rows_path: Optional[Path] = None
    rows_sha256: Optional[str] = None
    rows_artifact = contract.get("rows_artifact") if isinstance(contract, dict) else None
    rows_binding = authorization.get("rows_artifact_binding")
    if not isinstance(rows_artifact, dict) or not isinstance(rows_binding, dict):
        failures.append("a2_authorization_rows_binding_missing")
    else:
        try:
            expected_rows_path = _resolve_required_path(
                rows_artifact.get("path"), base=contract_path.parent
            )
            bound_rows_path = _resolve_required_path(rows_binding.get("path"))
        except ValueError:
            failures.append("a2_authorization_rows_path_invalid")
        else:
            if bound_rows_path != expected_rows_path:
                failures.append("a2_authorization_rows_path_mismatch")
            elif metadata_root is not None:
                try:
                    expected_rows_path.relative_to(metadata_root)
                except ValueError:
                    failures.append("a2_authorization_rows_outside_custodian_root")
            rows_path = expected_rows_path
        contract_rows_sha256 = normalized_text(rows_artifact.get("sha256")).casefold()
        bound_rows_sha256 = normalized_text(rows_binding.get("sha256")).casefold()
        if not is_valid_sha256(contract_rows_sha256) or bound_rows_sha256 != contract_rows_sha256:
            failures.append("a2_authorization_rows_sha256_mismatch")
        else:
            rows_sha256 = contract_rows_sha256

    if failures:
        return {"ready": False, "failures": sorted(set(failures))}
    if (
        contract_sha256 is None
        or rows_path is None
        or rows_sha256 is None
        or runtime_python is None
        or runtime_python_sha256 is None
        or guard_path is None
        or guard_sha256 is None
        or trust_anchor_sha256 is None
        or trusted_key_fingerprint is None
        or verification_receipt_sha256 is None
        or validator_source_closure_sha256 is None
    ):
        return {"ready": False, "failures": ["a2_authorization_incomplete_context"]}
    consumption_id = _metadata_lease_consumption_id(
        trusted_key_fingerprint=trusted_key_fingerprint,
        verification_receipt_sha256=verification_receipt_sha256,
        lease_id=lease_id,
    )
    marker_path = _metadata_lease_marker_path(
        consumption_id=consumption_id, lease_directory=lease_root
    )
    if marker_path.exists():
        return {"ready": False, "failures": ["a2_authorization_lease_already_consumed"]}
    return {
        "ready": True,
        "authorization_path": authorization_path,
        "authorization_sha256": authorization_sha256,
        "authority_scope": dict(METADATA_AUTHORITY_SCOPE),
        "contract_path": contract_path,
        "contract_sha256": contract_sha256,
        "lease_id": lease_id,
        "lease_consumption_id": consumption_id,
        "lease_marker_path": marker_path,
        "lease_directory": lease_root,
        "rows_path": rows_path,
        "rows_sha256": rows_sha256,
        "output_path": output_path,
        "runtime_python": runtime_python,
        "runtime_python_sha256": runtime_python_sha256,
        "validator_path": validator_path,
        "validator_source_closure_sha256": validator_source_closure_sha256,
        "resource_guard_path": guard_path,
        "resource_guard_sha256": guard_sha256,
        "resource_guard_max_age_seconds": max_guard_age_seconds,
        "trust_anchor_path": _resolve_external_path(trust_anchor_json),
        "trust_anchor_sha256": trust_anchor_sha256,
        "trusted_key_fingerprint": trusted_key_fingerprint,
        "verification_receipt_sha256": verification_receipt_sha256,
        "canonical_argv": tuple(canonical_argv),
        "canonical_argv_sha256": _sha256_json_value(canonical_argv),
    }


def _metadata_authority_unchanged(
    authorization: dict[str, Any], *, actual_argv: Sequence[str], now_utc: datetime
) -> bool:
    required_paths = (
        "authorization_path",
        "contract_path",
        "output_path",
        "lease_marker_path",
        "runtime_python",
        "validator_path",
        "resource_guard_path",
        "trust_anchor_path",
    )
    if any(not isinstance(authorization.get(name), Path) for name in required_paths):
        return False
    if tuple(actual_argv) != authorization.get("canonical_argv"):
        return False
    authorization_path = authorization["authorization_path"]
    contract_path = authorization["contract_path"]
    output_path = authorization["output_path"]
    marker_path = authorization["lease_marker_path"]
    runtime_python = authorization["runtime_python"]
    validator_path = authorization["validator_path"]
    guard_path = authorization["resource_guard_path"]
    trust_anchor_path = authorization["trust_anchor_path"]
    try:
        for path in (
            authorization_path,
            contract_path,
            output_path,
            marker_path,
            runtime_python,
            validator_path,
            guard_path,
            trust_anchor_path,
        ):
            _require_no_symlink(path)
        if output_path.exists() or marker_path.exists():
            return False
        if (
            sha256_file(authorization_path) != authorization.get("authorization_sha256")
            or sha256_file(contract_path) != authorization.get("contract_sha256")
            or sha256_file(runtime_python) != authorization.get("runtime_python_sha256")
            or sha256_file(trust_anchor_path) != authorization.get("trust_anchor_sha256")
        ):
            return False
        _, closure_sha256 = _validator_source_closure()
        if closure_sha256 != authorization.get("validator_source_closure_sha256"):
            return False
    except (OSError, ValueError):
        return False
    failures: list[str] = []
    guard_sha256 = _validate_metadata_resource_guard(
        guard_path=guard_path,
        expected_sha256=str(authorization.get("resource_guard_sha256") or ""),
        validator_path=validator_path,
        canonical_argv=list(authorization.get("canonical_argv") or ()),
        max_guard_age_seconds=int(authorization.get("resource_guard_max_age_seconds") or 0),
        current_time=now_utc,
        failures=failures,
    )
    return not failures and guard_sha256 == authorization.get("resource_guard_sha256")


def consume_a2_metadata_lease(
    authorization: dict[str, Any],
    *,
    actual_argv: Optional[Sequence[str]] = None,
    now_utc: Optional[datetime] = None,
) -> bool:
    """Atomically leave a durable marker before the controller opens metadata rows."""

    marker_path = authorization.get("lease_marker_path")
    if not isinstance(marker_path, Path):
        return False
    current_time = now_utc or datetime.now(timezone.utc)
    observed_argv = list(actual_argv if actual_argv is not None else sys.argv[1:])
    if not _metadata_authority_unchanged(
        authorization, actual_argv=observed_argv, now_utc=current_time
    ):
        return False
    marker_payload = {
        "schema": "axon_loop164_metadata_lease_consumption_v2",
        "loop_id": LOOP_ID,
        "state": "consumed_before_metadata_open",
        "authorization_sha256": authorization.get("authorization_sha256"),
        "contract_sha256": authorization.get("contract_sha256"),
        "rows_sha256": authorization.get("rows_sha256"),
        "lease_id": authorization.get("lease_id"),
        "lease_consumption_id": authorization.get("lease_consumption_id"),
        "trust_anchor_sha256": authorization.get("trust_anchor_sha256"),
        "validator_source_closure_sha256": authorization.get("validator_source_closure_sha256"),
        "runtime_python_sha256": authorization.get("runtime_python_sha256"),
        "resource_guard_sha256": authorization.get("resource_guard_sha256"),
        "canonical_argv_sha256": authorization.get("canonical_argv_sha256"),
        "consumed_at_utc": current_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        _require_no_symlink(marker_path)
    except (OSError, ValueError):
        return False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(marker_path, flags, 0o600)
    except FileExistsError:
        return False
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(marker_payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # marker 一旦被创建就不可回滚，避免不确定写入后复用同一授权。
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the future Loop164 full-pool isolation contract without opening model data."
    )
    parser.add_argument("--contract-json", type=Path, required=True)
    parser.add_argument("--a2-authorization-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    observed_argv = list(argv if argv is not None else sys.argv[1:])
    args = build_parser().parse_args(observed_argv)
    trust_anchor = os.environ.get("AXON_LOOP164_METADATA_TRUST_ANCHOR", "")
    trusted_key = os.environ.get("AXON_LOOP164_METADATA_TRUSTED_KEY_FINGERPRINT", "")
    authorization = validate_a2_metadata_authorization(
        authorization_json=args.a2_authorization_json,
        contract_json=args.contract_json,
        output_json=args.output_json,
        trust_anchor_json=Path(trust_anchor) if trust_anchor else None,
        expected_trusted_key_fingerprint=trusted_key or None,
        actual_argv=observed_argv,
    )
    if not authorization["ready"]:
        print(json.dumps({"decision": "block", "blockers": authorization["failures"]}, indent=2))
        return 2
    if not consume_a2_metadata_lease(authorization, actual_argv=observed_argv):
        print(
            json.dumps(
                {"decision": "block", "blockers": ["a2_authorization_lease_already_consumed"]},
                indent=2,
            )
        )
        return 2
    payload = _validate_loop164_isolation_contract_core(
        contract_json=args.contract_json,
        expected_contract_sha256=authorization["contract_sha256"],
        expected_rows_sha256=authorization["rows_sha256"],
    )
    marker_sha256 = sha256_file(authorization["lease_marker_path"])
    provenance = {
        "schema": METADATA_AUTHORIZATION_PROVENANCE_SCHEMA,
        "authority_scope": authorization["authority_scope"],
        "authorization_sha256": authorization["authorization_sha256"],
        "trust_anchor_sha256": authorization["trust_anchor_sha256"],
        "trusted_key_fingerprint": authorization["trusted_key_fingerprint"],
        "verification_receipt_sha256": authorization["verification_receipt_sha256"],
        "validator_source_closure_sha256": authorization["validator_source_closure_sha256"],
        "runtime_python_sha256": authorization["runtime_python_sha256"],
        "resource_guard_sha256": authorization["resource_guard_sha256"],
        "canonical_argv_sha256": authorization["canonical_argv_sha256"],
        "lease_consumption_id": authorization["lease_consumption_id"],
        "lease_marker_sha256": marker_sha256,
    }
    payload["a2_authorization_provenance"] = provenance
    payload["binding_fingerprints"].update(
        {
            "a2_authorization_sha256": provenance["authorization_sha256"],
            "a2_trust_anchor_sha256": provenance["trust_anchor_sha256"],
            "a2_validator_source_closure_sha256": provenance[
                "validator_source_closure_sha256"
            ],
            "a2_runtime_python_sha256": provenance["runtime_python_sha256"],
            "a2_resource_guard_sha256": provenance["resource_guard_sha256"],
            "a2_canonical_argv_sha256": provenance["canonical_argv_sha256"],
            "a2_lease_marker_sha256": provenance["lease_marker_sha256"],
            "a2_lease_consumption_id": provenance["lease_consumption_id"],
        }
    )
    output_path = authorization["output_path"]
    try:
        _write_json_exclusive(output_path, payload)
    except (OSError, ValueError) as exc:
        print(json.dumps({"decision": "block", "blockers": [str(exc)]}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "rows_read": payload["rows_read"],
                "blockers": payload["blockers"],
                "output_json": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if payload["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
