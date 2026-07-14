#!/usr/bin/env python3
"""Fail closed before a future Loop164 controller opens protected training input.

This module validates aggregate authorization artifacts and consumes a separate,
content-addressed final lease. It never imports ML code or opens raw files,
caches, checkpoints, predictions, or row-level split payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from validate_loop164_whole_file_implementation import (
    validate_implementation_manifest_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "loop164_whole_file_residual_expert"
TRAINING_AUTHORIZATION_SCHEMA = "axon_loop164_training_authorization_v2"
TRAINING_LEASE_SCHEMA = "axon_loop164_training_lease_consumption_v2"
TRAINING_LEASE_MARKER_SCHEMA = "axon_loop164_training_lease_marker_v1"
SCOPE_PLAN_VALIDATION_SCHEMA = "axon_loop164_fold_scope_plan_validation_v1"
INPUT_BUNDLE_SCHEMA = "axon_loop164_train_oof_input_bundle_v1"
TRUST_ANCHOR_SCHEMA = "axon_loop164_external_trust_anchor_v1"
RESOURCE_GUARD_SCHEMA = "axon_loop164_train_oof_resource_guard_v3"
ISOLATION_CONTRACT_SCHEMA = "axon_loop164_full_pool_isolation_contract_v2"
ISOLATION_RECEIPT_SCHEMA = "axon_loop164_full_pool_isolation_validation_v4"
ISOLATION_AUTHORIZATION_PROVENANCE_SCHEMA = "axon_loop164_isolation_authorization_provenance_v2"
ISOLATION_METADATA_AUTHORITY_SCOPE = {
    "tier": "A2",
    "operation": "metadata_isolation_only",
    "protected_input_scope": "metadata_only",
    "grants": [],
}
FEATURE_CONTRACT_SCHEMA = "axon_loop164_residual_fusion_feature_contract_v2"
IMPLEMENTATION_BINDING_PHASE = "deferred_to_a2_training_authority"
REQUIRED_SEEDS = (41, 42, 43)
REQUIRED_FOLDS = (0, 1, 2, 3, 4)
REQUIRED_FUSION_FIELDS = (
    "loop151_oof_score",
    "whole_file_oof_score",
    "loop151_oof_uncertainty",
    "whole_file_oof_uncertainty",
    "loop151_missingness",
    "whole_file_missingness",
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
MAX_AUTHORIZATION_TTL = timedelta(hours=24)


@dataclass(frozen=True)
class TrainingAuthorityPaths:
    root: Path
    authorization: Path
    proposal: Path
    contract: Path
    isolation_receipt: Path
    scope_plan: Path
    scope_plan_validation: Path
    implementation_manifest: Path
    loop151_train_oof_manifest: Path
    resource_guard: Path
    input_bundle: Path
    controller: Path
    final_lease: Path
    execution_receipt: Path
    lease_marker_directory: Path


@dataclass(frozen=True)
class TrainingAuthorityContext:
    paths: TrainingAuthorityPaths
    authorization: dict[str, Any]
    authorization_sha256: str
    binding_sha256: dict[str, str]
    lease_id: str
    scope_plan_validation_sha256: str
    input_bundle_sha256: str
    implementation_source_closure_sha256: str
    implementation_config_sha256: str
    implementation_runtime_lock_sha256: str
    implementation_input_contract_sha256: str
    implementation_missingness_contract_sha256: str
    implementation_memory_contract_sha256: str
    fold_assignment_fingerprint: str
    controller_sha256: str
    runtime_python: Path
    runtime_python_sha256: str
    canonical_argv: tuple[str, ...]
    canonical_argv_sha256: str
    lease_consumption_id: str
    marker_path: Path


@dataclass(frozen=True)
class TrainingAuthorityResult:
    ready: bool
    blockers: tuple[str, ...]
    context: Optional[TrainingAuthorityContext] = None


@dataclass(frozen=True)
class TrainingLeaseResult:
    consumed: bool
    blockers: tuple[str, ...]
    final_lease_path: Optional[Path] = None
    marker_path: Optional[Path] = None


def default_training_authority_paths(root: Path = PROJECT_ROOT) -> TrainingAuthorityPaths:
    resolved_root = root.resolve()
    loop_manifest = resolved_root / "manifests/roadmap_9997/loop164_whole_file_residual_expert"
    loop_report = resolved_root / "reports/roadmap_9997/loop164"
    return TrainingAuthorityPaths(
        root=resolved_root,
        authorization=loop_manifest / "a2_training_authorization.json",
        proposal=loop_manifest / "proposal.json",
        contract=loop_report / "full_pool_group_manifest.json",
        isolation_receipt=loop_report / "full_pool_isolation_validation.json",
        scope_plan=loop_report / "fold_scope_plan.json",
        scope_plan_validation=loop_report / "fold_scope_plan_validation.json",
        implementation_manifest=loop_report / "whole_file_expert_implementation_manifest.json",
        loop151_train_oof_manifest=loop_report / "loop151_train_oof_manifest.json",
        resource_guard=loop_report / "resource_guard.json",
        input_bundle=loop_report / "train_oof_input_bundle_manifest.json",
        controller=resolved_root / "scripts/run_loop164_train_oof_controller.py",
        final_lease=loop_report / "training_lease_consumption.final.json",
        execution_receipt=loop_report / "loop164_train_oof_execution_receipt.json",
        lease_marker_directory=loop_report / "training_lease_consumptions",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json_value(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Non-finite JSON value: {value}")


def _read_json_object(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite,
    )
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _parse_utc(value: object) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("Expected an explicit UTC timestamp")
    return parsed.astimezone(timezone.utc)


def _require_exact_keys(
    payload: object,
    expected: set[str],
    *,
    label: str,
    blockers: list[str],
) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        blockers.append(f"{label}_not_object")
        return None
    actual = set(payload)
    if actual != expected:
        if expected - actual:
            blockers.append(f"{label}_missing_fields")
        if actual - expected:
            blockers.append(f"{label}_unexpected_fields")
    return payload


def _require_no_symlink(path: Path) -> None:
    candidate = path.absolute()
    for ancestor in (candidate, *candidate.parents):
        if ancestor.is_symlink():
            raise ValueError("Symbolic-link path bindings are forbidden")


def _resolve_project_path(value: object, *, root: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Path binding is empty")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.absolute()
    _require_no_symlink(candidate)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Path binding escapes the project root") from exc
    return resolved


def _resolve_external_path(value: object, *, root: Path) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("External trust anchor is missing")
    candidate = Path(text).absolute()
    _require_no_symlink(candidate)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return resolved
    raise ValueError("Trust anchor must be outside the project root")


def _relative_project_path(path: Path, *, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _append_if_false(blockers: list[str], condition: bool, code: str) -> None:
    if not condition:
        blockers.append(code)


def _read_bound_json(
    *,
    name: str,
    path: Path,
    root: Path,
    authorization_bindings: object,
    blockers: list[str],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if not isinstance(authorization_bindings, dict):
        blockers.append("training_authorization_bindings_not_object")
        return None, None
    binding = _require_exact_keys(
        authorization_bindings.get(name),
        {"path", "sha256"},
        label=f"training_authorization_binding_{name}",
        blockers=blockers,
    )
    if binding is None:
        return None, None
    try:
        bound_path = _resolve_project_path(binding.get("path"), root=root)
    except ValueError:
        blockers.append(f"training_authorization_binding_{name}_path_invalid")
        return None, None
    _append_if_false(
        blockers,
        bound_path == path.resolve()
        and str(binding.get("path")) == _relative_project_path(path, root=root),
        f"training_authorization_binding_{name}_path_mismatch",
    )
    if not path.is_file():
        blockers.append(f"training_authorization_binding_{name}_missing")
        return None, None
    try:
        payload, digest = _read_json_object(path)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        blockers.append(f"training_authorization_binding_{name}_unreadable")
        return None, None
    _append_if_false(
        blockers,
        _is_sha256(binding.get("sha256")) and str(binding.get("sha256")).casefold() == digest,
        f"training_authorization_binding_{name}_sha256_mismatch",
    )
    return payload, digest


def _validate_trust_anchor(
    *,
    trust_anchor_json: Path,
    root: Path,
    authority_attestation: object,
    expected_trusted_key_fingerprint: object,
    blockers: list[str],
) -> None:
    try:
        anchor_path = _resolve_external_path(trust_anchor_json, root=root)
        anchor, anchor_sha256 = _read_json_object(anchor_path)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        blockers.append("training_authorization_trust_anchor_unreadable")
        return
    anchor = _require_exact_keys(
        anchor,
        {
            "schema",
            "loop_id",
            "trusted_key_fingerprint",
            "verification_receipt_sha256",
            "root_state",
        },
        label="training_authorization_trust_anchor",
        blockers=blockers,
    )
    attestation = _require_exact_keys(
        authority_attestation,
        {"trusted_key_fingerprint", "trust_anchor_sha256", "verification_receipt_sha256"},
        label="training_authorization_attestation",
        blockers=blockers,
    )
    if anchor is None or attestation is None:
        return
    if not _is_sha256(expected_trusted_key_fingerprint):
        blockers.append("training_authorization_trusted_key_not_configured")
        return
    _append_if_false(
        blockers,
        anchor.get("schema") == TRUST_ANCHOR_SCHEMA and anchor.get("loop_id") == LOOP_ID,
        "training_authorization_trust_anchor_schema_invalid",
    )
    _append_if_false(
        blockers,
        anchor.get("root_state") == "externally_verified",
        "training_authorization_trust_anchor_not_externally_verified",
    )
    for field_name in ("trusted_key_fingerprint", "verification_receipt_sha256"):
        _append_if_false(
            blockers,
            _is_sha256(anchor.get(field_name)),
            f"training_authorization_trust_anchor_{field_name}_invalid",
        )
        _append_if_false(
            blockers,
            attestation.get(field_name) == anchor.get(field_name),
            f"training_authorization_attestation_{field_name}_mismatch",
        )
    _append_if_false(
        blockers,
        anchor.get("trusted_key_fingerprint")
        == str(expected_trusted_key_fingerprint).strip().casefold(),
        "training_authorization_trusted_key_not_pinned",
    )
    _append_if_false(
        blockers,
        _is_sha256(attestation.get("trust_anchor_sha256"))
        and str(attestation.get("trust_anchor_sha256")).casefold() == anchor_sha256,
        "training_authorization_attestation_trust_anchor_mismatch",
    )


def _validate_isolation_receipt_provenance(
    payload: object, *, blockers: list[str]
) -> None:
    provenance = _require_exact_keys(
        payload,
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
        label="training_authorization_isolation_receipt_provenance",
        blockers=blockers,
    )
    if provenance is None:
        return
    _append_if_false(
        blockers,
        provenance.get("schema") == ISOLATION_AUTHORIZATION_PROVENANCE_SCHEMA
        and provenance.get("authority_scope") == ISOLATION_METADATA_AUTHORITY_SCOPE
        and all(
            _is_sha256(provenance.get(field_name))
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
        "training_authorization_isolation_receipt_provenance_invalid",
    )


def _validate_isolation_contract(
    payload: Optional[dict[str, Any]], *, blockers: list[str]
) -> None:
    if not isinstance(payload, dict):
        blockers.append("training_authorization_contract_invalid")
        return
    _append_if_false(
        blockers,
        payload.get("schema") == ISOLATION_CONTRACT_SCHEMA and payload.get("loop_id") == LOOP_ID,
        "training_authorization_contract_invalid",
    )
    _append_if_false(
        blockers,
        tuple(payload.get("model_input_fields") or []) == REQUIRED_FUSION_FIELDS,
        "training_authorization_contract_feature_allowlist_invalid",
    )
    feature_contract = payload.get("feature_contract")
    if not isinstance(feature_contract, dict):
        blockers.append("training_authorization_contract_feature_contract_missing")
        return
    _append_if_false(
        blockers,
        set(feature_contract)
        == {
            "schema",
            "feature_fields",
            "feature_matrix_receipt_required",
            "implementation_binding_phase",
        },
        "training_authorization_contract_feature_contract_shape_invalid",
    )
    _append_if_false(
        blockers,
        feature_contract.get("schema") == FEATURE_CONTRACT_SCHEMA,
        "training_authorization_contract_feature_contract_schema_invalid",
    )
    _append_if_false(
        blockers,
        tuple(feature_contract.get("feature_fields") or []) == REQUIRED_FUSION_FIELDS,
        "training_authorization_contract_feature_contract_allowlist_invalid",
    )
    _append_if_false(
        blockers,
        feature_contract.get("feature_matrix_receipt_required") is True,
        "training_authorization_contract_feature_matrix_receipt_not_required",
    )
    _append_if_false(
        blockers,
        feature_contract.get("implementation_binding_phase") == IMPLEMENTATION_BINDING_PHASE,
        "training_authorization_contract_implementation_binding_phase_invalid",
    )


def _validate_scope_validation(
    *,
    payload: Optional[dict[str, Any]],
    proposal_sha256: Optional[str],
    contract_sha256: Optional[str],
    isolation_receipt_sha256: Optional[str],
    scope_plan_sha256: Optional[str],
    blockers: list[str],
) -> Optional[str]:
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
        blockers=blockers,
    )
    if validation is None:
        return None
    _append_if_false(
        blockers,
        validation.get("schema") == SCOPE_PLAN_VALIDATION_SCHEMA
        and validation.get("loop_id") == LOOP_ID,
        "scope_plan_validation_schema_invalid",
    )
    _append_if_false(
        blockers,
        validation.get("decision") == "pass"
        and validation.get("aggregate_only_verified") is True
        and validation.get("proposal_binding_verified") is True
        and validation.get("contract_binding_verified") is True
        and validation.get("isolation_receipt_binding_verified") is True
        and validation.get("scope_plan_binding_verified") is True,
        "scope_plan_validation_not_passed",
    )
    ready_for = validation.get("ready_for")
    _append_if_false(
        blockers,
        isinstance(ready_for, dict)
        and ready_for.get("fold_scope_frozen") is True
        and ready_for.get("a2_training_authorization") is False
        and ready_for.get("train_oof") is False,
        "scope_plan_validation_ready_state_invalid",
    )
    fingerprints = validation.get("binding_fingerprints")
    expected_fingerprints = {
        "proposal_sha256": proposal_sha256,
        "contract_sha256": contract_sha256,
        "isolation_receipt_sha256": isolation_receipt_sha256,
        "scope_plan_sha256": scope_plan_sha256,
    }
    _append_if_false(
        blockers,
        isinstance(fingerprints, dict)
        and all(fingerprints.get(name) == digest for name, digest in expected_fingerprints.items()),
        "scope_plan_validation_binding_mismatch",
    )
    plan_summary = validation.get("plan_summary")
    if not isinstance(plan_summary, dict) or not _is_sha256(
        plan_summary.get("fold_assignment_fingerprint")
    ):
        blockers.append("scope_plan_validation_fold_fingerprint_invalid")
        return None
    return str(plan_summary["fold_assignment_fingerprint"]).casefold()


def _validate_input_bundle(
    *,
    payload: Optional[dict[str, Any]],
    scope_plan_validation_sha256: Optional[str],
    fold_assignment_fingerprint: Optional[str],
    blockers: list[str],
) -> None:
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
        blockers=blockers,
    )
    if bundle is None:
        return
    _append_if_false(
        blockers,
        bundle.get("schema") == INPUT_BUNDLE_SCHEMA and bundle.get("loop_id") == LOOP_ID,
        "training_input_bundle_schema_invalid",
    )
    _append_if_false(
        blockers,
        bundle.get("allowed_split_roles") == ["train_anchor", "train_oof"]
        and tuple(bundle.get("forbidden_split_roles") or []) == FORBIDDEN_SPLIT_ROLES,
        "training_input_bundle_split_roles_invalid",
    )
    _append_if_false(
        blockers,
        tuple(bundle.get("feature_fields") or []) == REQUIRED_FUSION_FIELDS,
        "training_input_bundle_feature_allowlist_invalid",
    )
    _append_if_false(
        blockers,
        bundle.get("fold_assignment_fingerprint") == fold_assignment_fingerprint,
        "training_input_bundle_fold_fingerprint_mismatch",
    )
    _append_if_false(
        blockers,
        bundle.get("scope_plan_validation_sha256") == scope_plan_validation_sha256,
        "training_input_bundle_scope_validation_mismatch",
    )
    _append_if_false(
        blockers,
        bundle.get("protected_input_open_policy") == "after_final_lease_only",
        "training_input_bundle_open_policy_invalid",
    )
    commitments = bundle.get("input_artifact_commitments")
    if not isinstance(commitments, dict) or set(commitments) != {
        "train_anchor_sha256",
        "train_oof_sha256",
    }:
        blockers.append("training_input_bundle_commitments_invalid")
    elif not all(_is_sha256(commitments[name]) for name in commitments):
        blockers.append("training_input_bundle_commitment_sha256_invalid")


def _validate_resource_guard(
    *,
    payload: Optional[dict[str, Any]],
    root: Path,
    runtime_python: Path,
    controller: Path,
    canonical_argv_sha256: str,
    implementation_manifest_sha256: Optional[str],
    implementation_source_closure_sha256: Optional[str],
    implementation_memory_contract_sha256: Optional[str],
    now_utc: datetime,
    max_age_seconds: object,
    blockers: list[str],
) -> None:
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
        blockers=blockers,
    )
    if guard is None:
        return
    _append_if_false(
        blockers,
        guard.get("schema") == RESOURCE_GUARD_SCHEMA
        and guard.get("loop_id") == LOOP_ID
        and guard.get("operation") == "loop164_three_seed_nested_train_oof"
        and guard.get("guard_ready") is True
        and guard.get("decision") == "pass",
        "training_resource_guard_not_ready",
    )
    runtime_binding = _require_exact_keys(
        guard.get("runtime_binding"),
        {"cwd", "python_sha256", "controller_path", "controller_sha256", "canonical_argv_sha256"},
        label="training_resource_guard_runtime_binding",
        blockers=blockers,
    )
    if runtime_binding is not None:
        _append_if_false(
            blockers,
            runtime_binding.get("cwd") == str(root)
            and runtime_binding.get("python_sha256") == sha256_file(runtime_python)
            and runtime_binding.get("controller_path") == _relative_project_path(controller, root=root)
            and runtime_binding.get("controller_sha256") == sha256_file(controller)
            and runtime_binding.get("canonical_argv_sha256") == canonical_argv_sha256,
            "training_resource_guard_runtime_binding_mismatch",
        )
    implementation_binding = _require_exact_keys(
        guard.get("implementation_binding"),
        {"implementation_manifest_sha256", "source_closure_sha256", "memory_contract_sha256"},
        label="training_resource_guard_implementation_binding",
        blockers=blockers,
    )
    if implementation_binding is not None:
        _append_if_false(
            blockers,
            implementation_binding.get("implementation_manifest_sha256")
            == implementation_manifest_sha256
            and implementation_binding.get("source_closure_sha256")
            == implementation_source_closure_sha256
            and implementation_binding.get("memory_contract_sha256")
            == implementation_memory_contract_sha256
            and all(_is_sha256(value) for value in implementation_binding.values()),
            "training_resource_guard_implementation_binding_mismatch",
        )
    receipt = _require_exact_keys(
        guard.get("receipt"),
        {"created_at_utc", "controller_sha256", "resource_budget_sha256"},
        label="training_resource_guard_receipt",
        blockers=blockers,
    )
    if receipt is None:
        return
    try:
        created_at = _parse_utc(receipt.get("created_at_utc"))
        age_seconds = int(max_age_seconds)
    except (TypeError, ValueError):
        blockers.append("training_resource_guard_age_invalid")
        return
    _append_if_false(
        blockers,
        1 <= age_seconds and now_utc - created_at <= timedelta(seconds=age_seconds),
        "training_resource_guard_stale",
    )
    _append_if_false(
        blockers,
        receipt.get("controller_sha256") == sha256_file(controller)
        and _is_sha256(receipt.get("resource_budget_sha256")),
        "training_resource_guard_receipt_binding_invalid",
    )


def build_lease_consumption_id(
    *,
    authorization_sha256: str,
    lease_id: str,
    controller_sha256: str,
    input_bundle_sha256: str,
    scope_plan_validation_sha256: str,
    canonical_argv_sha256: str,
    execution_receipt_path: Path,
) -> str:
    material = ":".join(
        (
            authorization_sha256,
            lease_id,
            controller_sha256,
            input_bundle_sha256,
            scope_plan_validation_sha256,
            canonical_argv_sha256,
            str(execution_receipt_path),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _lease_marker_path(paths: TrainingAuthorityPaths, consumption_id: str) -> Path:
    return paths.lease_marker_directory / f"{consumption_id}.final.json"


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("Refusing to overwrite an existing lease artifact") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_training_authority(
    *,
    trust_anchor_json: Path,
    expected_trusted_key_fingerprint: str,
    paths: Optional[TrainingAuthorityPaths] = None,
    authorization_json: Optional[Path] = None,
    actual_argv: Optional[Sequence[str]] = None,
    now_utc: Optional[datetime] = None,
    runtime_python_executable: Optional[Path] = None,
) -> TrainingAuthorityResult:
    """Validate aggregate authority before any protected input may be opened."""

    expected_paths = paths or default_training_authority_paths()
    root = expected_paths.root.resolve()
    current_time = now_utc or datetime.now(timezone.utc)
    runtime_python = (runtime_python_executable or Path(sys.executable)).resolve()
    requested_authorization = authorization_json or expected_paths.authorization
    blockers: list[str] = []
    try:
        resolved_authorization = _resolve_project_path(requested_authorization, root=root)
        for project_path in (
            expected_paths.authorization,
            expected_paths.proposal,
            expected_paths.contract,
            expected_paths.isolation_receipt,
            expected_paths.scope_plan,
            expected_paths.scope_plan_validation,
            expected_paths.implementation_manifest,
            expected_paths.loop151_train_oof_manifest,
            expected_paths.resource_guard,
            expected_paths.input_bundle,
            expected_paths.final_lease,
            expected_paths.execution_receipt,
            expected_paths.lease_marker_directory,
        ):
            _resolve_project_path(project_path, root=root)
        _require_no_symlink(expected_paths.controller)
        _require_no_symlink(runtime_python)
    except ValueError:
        return TrainingAuthorityResult(False, ("training_authorization_path_binding_invalid",))
    if resolved_authorization != expected_paths.authorization.resolve():
        return TrainingAuthorityResult(False, ("training_authorization_path_not_canonical",))
    if not expected_paths.authorization.is_file():
        return TrainingAuthorityResult(False, ("training_authorization_missing",))
    if not expected_paths.controller.is_file() or not runtime_python.is_file():
        return TrainingAuthorityResult(False, ("training_authorization_runtime_file_missing",))
    try:
        authorization, authorization_sha256 = _read_json_object(expected_paths.authorization)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return TrainingAuthorityResult(False, ("training_authorization_unreadable",))
    expected_authorization_fields = {
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
        authorization,
        expected_authorization_fields,
        label="training_authorization",
        blockers=blockers,
    )
    if authorization is None:
        return TrainingAuthorityResult(False, tuple(sorted(set(blockers))))
    expected_header = {
        "schema": TRAINING_AUTHORIZATION_SCHEMA,
        "loop_id": LOOP_ID,
        "authorization_level": "A2_train_only_nested_oof",
        "decision": "allow_single_loop164_train_oof_execution",
        "execution_environment": "custodian_side_train_only",
        "operation": "loop164_three_seed_nested_train_oof",
    }
    for name, value in expected_header.items():
        _append_if_false(blockers, authorization.get(name) == value, f"training_authorization_{name}_invalid")
    try:
        issued_at = _parse_utc(authorization.get("issued_at_utc"))
        not_before = _parse_utc(authorization.get("not_before_utc"))
        expires_at = _parse_utc(authorization.get("expires_at_utc"))
    except ValueError:
        blockers.append("training_authorization_time_window_invalid")
    else:
        _append_if_false(
            blockers,
            issued_at <= not_before < expires_at and expires_at - issued_at <= MAX_AUTHORIZATION_TTL,
            "training_authorization_time_window_invalid",
        )
        _append_if_false(
            blockers,
            not_before <= current_time < expires_at,
            "training_authorization_not_currently_valid",
        )
    _validate_trust_anchor(
        trust_anchor_json=trust_anchor_json,
        root=root,
        authority_attestation=authorization.get("authority_attestation"),
        expected_trusted_key_fingerprint=expected_trusted_key_fingerprint,
        blockers=blockers,
    )
    canonical_argv = authorization.get("canonical_argv")
    if not isinstance(canonical_argv, list) or not canonical_argv or any(
        not isinstance(argument, str) or not argument for argument in canonical_argv
    ):
        blockers.append("training_authorization_canonical_argv_invalid")
        canonical_argv = []
    observed_argv = list(actual_argv if actual_argv is not None else sys.argv[1:])
    _append_if_false(
        blockers,
        observed_argv == canonical_argv,
        "training_authorization_canonical_argv_mismatch",
    )
    canonical_argv_sha256 = _sha256_json_value(canonical_argv)
    runtime_binding = _require_exact_keys(
        authorization.get("runtime_binding"),
        {"cwd", "python_executable", "python_sha256", "controller_path", "controller_sha256", "entrypoint"},
        label="training_authorization_runtime_binding",
        blockers=blockers,
    )
    if runtime_binding is not None:
        _append_if_false(
            blockers,
            runtime_binding.get("cwd") == str(root)
            and runtime_binding.get("python_executable") == str(runtime_python)
            and runtime_binding.get("python_sha256") == sha256_file(runtime_python)
            and runtime_binding.get("controller_path")
            == _relative_project_path(expected_paths.controller, root=root)
            and runtime_binding.get("controller_sha256") == sha256_file(expected_paths.controller)
            and runtime_binding.get("entrypoint") == "run_loop164_train_oof_controller.main",
            "training_authorization_runtime_binding_mismatch",
        )
    _append_if_false(
        blockers,
        authorization.get("allowed_split_roles") == ["train_anchor", "train_oof"]
        and tuple(authorization.get("forbidden_split_roles") or []) == FORBIDDEN_SPLIT_ROLES,
        "training_authorization_split_roles_invalid",
    )
    _append_if_false(
        blockers,
        tuple(authorization.get("feature_fields") or []) == REQUIRED_FUSION_FIELDS,
        "training_authorization_feature_allowlist_invalid",
    )
    _append_if_false(
        blockers,
        authorization.get("outer_run_budget") == len(REQUIRED_SEEDS) * len(REQUIRED_FOLDS),
        "training_authorization_outer_run_budget_invalid",
    )
    expected_binding_paths = {
        "proposal": expected_paths.proposal,
        "contract": expected_paths.contract,
        "isolation_receipt": expected_paths.isolation_receipt,
        "scope_plan": expected_paths.scope_plan,
        "scope_plan_validation": expected_paths.scope_plan_validation,
        "implementation_manifest": expected_paths.implementation_manifest,
        "loop151_train_oof_manifest": expected_paths.loop151_train_oof_manifest,
        "resource_guard": expected_paths.resource_guard,
        "input_bundle": expected_paths.input_bundle,
    }
    bindings = authorization.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(expected_binding_paths):
        blockers.append("training_authorization_bindings_shape_invalid")
        bindings = {}
    bound_payloads: dict[str, Optional[dict[str, Any]]] = {}
    bound_digests: dict[str, Optional[str]] = {}
    # 训练前只接受固定清单的聚合绑定，并对每个文件按原始字节重新验 hash。
    for name, path in expected_binding_paths.items():
        payload, digest = _read_bound_json(
            name=name,
            path=path,
            root=root,
            authorization_bindings=bindings,
            blockers=blockers,
        )
        bound_payloads[name] = payload
        bound_digests[name] = digest
    proposal = bound_payloads["proposal"]
    _append_if_false(
        blockers,
        isinstance(proposal, dict)
        and proposal.get("loop_id") == LOOP_ID
        and proposal.get("decision") == "propose_loop164_whole_file_residual_expert_no_execution",
        "training_authorization_proposal_invalid",
    )
    contract = bound_payloads["contract"]
    _validate_isolation_contract(contract, blockers=blockers)
    isolation_receipt = bound_payloads["isolation_receipt"]
    _append_if_false(
        blockers,
        isinstance(isolation_receipt, dict)
        and isolation_receipt.get("schema") == ISOLATION_RECEIPT_SCHEMA
        and isolation_receipt.get("loop_id") == LOOP_ID
        and isolation_receipt.get("decision") == "pass"
        and isinstance(isolation_receipt.get("ready_for"), dict)
        and isolation_receipt["ready_for"].get("loop164_train_oof_partition") is True,
        "training_authorization_isolation_receipt_invalid",
    )
    if isinstance(isolation_receipt, dict):
        _validate_isolation_receipt_provenance(
            isolation_receipt.get("a2_authorization_provenance"), blockers=blockers
        )
    scope_plan = bound_payloads["scope_plan"]
    plan_fingerprint = (
        scope_plan.get("fold_assignment_fingerprint") if isinstance(scope_plan, dict) else None
    )
    _append_if_false(
        blockers,
        isinstance(scope_plan, dict) and scope_plan.get("loop_id") == LOOP_ID,
        "training_authorization_scope_plan_invalid",
    )
    scope_validation_fingerprint = _validate_scope_validation(
        payload=bound_payloads["scope_plan_validation"],
        proposal_sha256=bound_digests["proposal"],
        contract_sha256=bound_digests["contract"],
        isolation_receipt_sha256=bound_digests["isolation_receipt"],
        scope_plan_sha256=bound_digests["scope_plan"],
        blockers=blockers,
    )
    _append_if_false(
        blockers,
        _is_sha256(plan_fingerprint)
        and plan_fingerprint == scope_validation_fingerprint
        and authorization.get("fold_assignment_fingerprint") == plan_fingerprint,
        "training_authorization_fold_fingerprint_mismatch",
    )
    _validate_input_bundle(
        payload=bound_payloads["input_bundle"],
        scope_plan_validation_sha256=bound_digests["scope_plan_validation"],
        fold_assignment_fingerprint=plan_fingerprint if isinstance(plan_fingerprint, str) else None,
        blockers=blockers,
    )
    implementation_result = validate_implementation_manifest_payload(
        bound_payloads["implementation_manifest"], root=root
    )
    blockers.extend(f"training_{code}" for code in implementation_result.blockers)
    _validate_resource_guard(
        payload=bound_payloads["resource_guard"],
        root=root,
        runtime_python=runtime_python,
        controller=expected_paths.controller,
        canonical_argv_sha256=canonical_argv_sha256,
        implementation_manifest_sha256=bound_digests["implementation_manifest"],
        implementation_source_closure_sha256=implementation_result.source_closure_sha256,
        implementation_memory_contract_sha256=implementation_result.memory_contract_sha256,
        now_utc=current_time,
        max_age_seconds=authorization.get("max_resource_guard_age_seconds"),
        blockers=blockers,
    )
    output_binding = _require_exact_keys(
        authorization.get("output_binding"),
        {"execution_receipt_path", "final_lease_path", "lease_marker_directory"},
        label="training_authorization_output_binding",
        blockers=blockers,
    )
    if output_binding is not None:
        _append_if_false(
            blockers,
            output_binding.get("execution_receipt_path")
            == _relative_project_path(expected_paths.execution_receipt, root=root)
            and output_binding.get("final_lease_path")
            == _relative_project_path(expected_paths.final_lease, root=root)
            and output_binding.get("lease_marker_directory")
            == _relative_project_path(expected_paths.lease_marker_directory, root=root),
            "training_authorization_output_binding_mismatch",
        )
    lease = _require_exact_keys(
        authorization.get("one_shot_lease"),
        {"lease_id", "purpose", "state"},
        label="training_authorization_lease",
        blockers=blockers,
    )
    lease_id = ""
    if lease is not None:
        lease_id = str(lease.get("lease_id") or "")
        _append_if_false(
            blockers,
            bool(re.fullmatch(r"[A-Za-z0-9._-]{1,128}", lease_id))
            and lease.get("purpose") == "single_loop164_three_seed_nested_train_oof"
            and lease.get("state") == "ready",
            "training_authorization_lease_invalid",
        )
    if expected_paths.execution_receipt.exists() or expected_paths.final_lease.exists():
        blockers.append("training_authorization_output_already_exists")
    if not all(
        isinstance(value, str) and _is_sha256(value)
        for value in (
            bound_digests["scope_plan_validation"],
            bound_digests["input_bundle"],
            plan_fingerprint,
            implementation_result.source_closure_sha256,
            implementation_result.config_sha256,
            implementation_result.runtime_lock_sha256,
            implementation_result.input_contract_sha256,
            implementation_result.missingness_contract_sha256,
            implementation_result.memory_contract_sha256,
        )
    ):
        blockers.append("training_authorization_required_binding_missing")
    if blockers:
        return TrainingAuthorityResult(False, tuple(sorted(set(blockers))))
    scope_validation_sha256 = str(bound_digests["scope_plan_validation"])
    input_bundle_sha256 = str(bound_digests["input_bundle"])
    controller_sha256 = sha256_file(expected_paths.controller)
    consumption_id = build_lease_consumption_id(
        authorization_sha256=authorization_sha256,
        lease_id=lease_id,
        controller_sha256=controller_sha256,
        input_bundle_sha256=input_bundle_sha256,
        scope_plan_validation_sha256=scope_validation_sha256,
        canonical_argv_sha256=canonical_argv_sha256,
        execution_receipt_path=expected_paths.execution_receipt,
    )
    marker_path = _lease_marker_path(expected_paths, consumption_id)
    if marker_path.exists():
        return TrainingAuthorityResult(False, ("training_authorization_lease_already_consumed",))
    context = TrainingAuthorityContext(
        paths=expected_paths,
        authorization=authorization,
        authorization_sha256=authorization_sha256,
        binding_sha256={name: str(digest) for name, digest in bound_digests.items()},
        lease_id=lease_id,
        scope_plan_validation_sha256=scope_validation_sha256,
        input_bundle_sha256=input_bundle_sha256,
        implementation_source_closure_sha256=str(implementation_result.source_closure_sha256),
        implementation_config_sha256=str(implementation_result.config_sha256),
        implementation_runtime_lock_sha256=str(implementation_result.runtime_lock_sha256),
        implementation_input_contract_sha256=str(implementation_result.input_contract_sha256),
        implementation_missingness_contract_sha256=str(
            implementation_result.missingness_contract_sha256
        ),
        implementation_memory_contract_sha256=str(implementation_result.memory_contract_sha256),
        fold_assignment_fingerprint=str(plan_fingerprint),
        controller_sha256=controller_sha256,
        runtime_python=runtime_python,
        runtime_python_sha256=sha256_file(runtime_python),
        canonical_argv=tuple(canonical_argv),
        canonical_argv_sha256=canonical_argv_sha256,
        lease_consumption_id=consumption_id,
        marker_path=marker_path,
    )
    return TrainingAuthorityResult(True, (), context)


def _lease_payload(
    context: TrainingAuthorityContext,
    *,
    schema: str,
    state: str,
    consumed_at_utc: datetime,
) -> dict[str, Any]:
    return {
        "schema": schema,
        "loop_id": LOOP_ID,
        "state": state,
        "lease_consumption_id": context.lease_consumption_id,
        "authorization_sha256": context.authorization_sha256,
        "lease_id": context.lease_id,
        "scope_plan_validation_sha256": context.scope_plan_validation_sha256,
        "input_bundle_sha256": context.input_bundle_sha256,
        "implementation_manifest_sha256": context.binding_sha256["implementation_manifest"],
        "source_closure_sha256": context.implementation_source_closure_sha256,
        "memory_contract_sha256": context.implementation_memory_contract_sha256,
        "controller_sha256": context.controller_sha256,
        "canonical_argv_sha256": context.canonical_argv_sha256,
        "canonical_argv": list(context.canonical_argv),
        "output_receipt_path": _relative_project_path(
            context.paths.execution_receipt, root=context.paths.root
        ),
        "fold_assignment_fingerprint": context.fold_assignment_fingerprint,
        "outer_run_budget": len(REQUIRED_SEEDS) * len(REQUIRED_FOLDS),
        "consumed_at_utc": consumed_at_utc.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "marker_path": _relative_project_path(context.marker_path, root=context.paths.root),
    }


def _implementation_contract_unchanged(context: TrainingAuthorityContext) -> bool:
    try:
        manifest, manifest_sha256 = _read_json_object(context.paths.implementation_manifest)
        result = validate_implementation_manifest_payload(manifest, root=context.paths.root)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return False
    return (
        result.ready
        and manifest_sha256 == context.binding_sha256["implementation_manifest"]
        and result.source_closure_sha256 == context.implementation_source_closure_sha256
        and result.config_sha256 == context.implementation_config_sha256
        and result.runtime_lock_sha256 == context.implementation_runtime_lock_sha256
        and result.input_contract_sha256 == context.implementation_input_contract_sha256
        and result.missingness_contract_sha256 == context.implementation_missingness_contract_sha256
        and result.memory_contract_sha256 == context.implementation_memory_contract_sha256
    )


def consume_training_final_lease(
    context: TrainingAuthorityContext,
    *,
    consumed_at_utc: Optional[datetime] = None,
    actual_argv: Optional[Sequence[str]] = None,
) -> TrainingLeaseResult:
    """Atomically burn the lease before a future controller can open protected input."""

    observed_argv = tuple(actual_argv if actual_argv is not None else sys.argv[1:])
    if observed_argv != context.canonical_argv:
        return TrainingLeaseResult(False, ("training_authorization_canonical_argv_mismatch",))
    try:
        _require_no_symlink(context.paths.final_lease)
        _require_no_symlink(context.paths.execution_receipt)
        _require_no_symlink(context.marker_path)
        _require_no_symlink(context.runtime_python)
        _require_no_symlink(context.paths.controller)
    except ValueError:
        return TrainingLeaseResult(False, ("training_authorization_path_binding_invalid",))
    try:
        runtime_bindings_unchanged = (
            sha256_file(context.paths.authorization) == context.authorization_sha256
            and sha256_file(context.runtime_python) == context.runtime_python_sha256
            and sha256_file(context.paths.controller) == context.controller_sha256
        )
    except OSError:
        runtime_bindings_unchanged = False
    if not runtime_bindings_unchanged:
        return TrainingLeaseResult(False, ("training_authorization_runtime_binding_changed_before_lease",))
    bound_paths = {
        "proposal": context.paths.proposal,
        "contract": context.paths.contract,
        "isolation_receipt": context.paths.isolation_receipt,
        "scope_plan": context.paths.scope_plan,
        "scope_plan_validation": context.paths.scope_plan_validation,
        "implementation_manifest": context.paths.implementation_manifest,
        "loop151_train_oof_manifest": context.paths.loop151_train_oof_manifest,
        "resource_guard": context.paths.resource_guard,
        "input_bundle": context.paths.input_bundle,
    }
    try:
        bindings_unchanged = all(
            sha256_file(path) == context.binding_sha256[name]
            for name, path in bound_paths.items()
        )
    except OSError:
        bindings_unchanged = False
    if not bindings_unchanged:
        return TrainingLeaseResult(False, ("training_authorization_binding_changed_before_lease",))
    if not _implementation_contract_unchanged(context):
        return TrainingLeaseResult(
            False, ("training_authorization_implementation_closure_changed_before_lease",)
        )
    if context.marker_path.exists():
        return TrainingLeaseResult(False, ("training_authorization_lease_already_consumed",))
    if context.paths.execution_receipt.exists() or context.paths.final_lease.exists():
        return TrainingLeaseResult(False, ("training_authorization_output_already_exists",))
    timestamp = consumed_at_utc or datetime.now(timezone.utc)
    try:
        # marker 先落盘且永不回滚，避免 final lease 写入异常后复用同一授权。
        _write_json_exclusive(
            context.marker_path,
            _lease_payload(
                context,
                schema=TRAINING_LEASE_MARKER_SCHEMA,
                state="consumed_before_protected_open",
                consumed_at_utc=timestamp,
            ),
        )
    except (OSError, ValueError):
        return TrainingLeaseResult(False, ("training_authorization_lease_already_consumed",))
    try:
        _write_json_exclusive(
            context.paths.final_lease,
            _lease_payload(
                context,
                schema=TRAINING_LEASE_SCHEMA,
                state="consumed_before_protected_open",
                consumed_at_utc=timestamp,
            ),
        )
    except (OSError, ValueError):
        return TrainingLeaseResult(
            False,
            ("training_authorization_final_lease_write_failed_marker_retained",),
            marker_path=context.marker_path,
        )
    return TrainingLeaseResult(
        True,
        (),
        final_lease_path=context.paths.final_lease,
        marker_path=context.marker_path,
    )


def verify_consumed_training_final_lease(context: TrainingAuthorityContext) -> TrainingLeaseResult:
    """Verify the marker and canonical final lease without opening protected input."""

    expected_keys = {
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
    try:
        marker, _ = _read_json_object(context.marker_path)
        final_lease, _ = _read_json_object(context.paths.final_lease)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return TrainingLeaseResult(False, ("training_authorization_consumed_lease_unreadable",))
    for payload, schema in (
        (marker, TRAINING_LEASE_MARKER_SCHEMA),
        (final_lease, TRAINING_LEASE_SCHEMA),
    ):
        if set(payload) != expected_keys:
            return TrainingLeaseResult(False, ("training_authorization_consumed_lease_shape_invalid",))
        if payload.get("schema") != schema or payload.get("loop_id") != LOOP_ID:
            return TrainingLeaseResult(False, ("training_authorization_consumed_lease_schema_invalid",))
        if payload.get("state") != "consumed_before_protected_open":
            return TrainingLeaseResult(False, ("training_authorization_consumed_lease_state_invalid",))
        try:
            _parse_utc(payload.get("consumed_at_utc"))
        except ValueError:
            return TrainingLeaseResult(False, ("training_authorization_consumed_lease_timestamp_invalid",))
    marker_compare = {key: value for key, value in marker.items() if key not in {"schema", "consumed_at_utc"}}
    final_compare = {key: value for key, value in final_lease.items() if key not in {"schema", "consumed_at_utc"}}
    if marker_compare != final_compare:
        return TrainingLeaseResult(False, ("training_authorization_marker_final_lease_mismatch",))
    expected = _lease_payload(
        context,
        schema=TRAINING_LEASE_SCHEMA,
        state="consumed_before_protected_open",
        consumed_at_utc=_parse_utc(final_lease["consumed_at_utc"]),
    )
    if final_lease != expected:
        return TrainingLeaseResult(False, ("training_authorization_final_lease_binding_mismatch",))
    return TrainingLeaseResult(
        True,
        (),
        final_lease_path=context.paths.final_lease,
        marker_path=context.marker_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check the fixed Loop164 training authority without consuming its lease."
    )
    parser.add_argument("--check", action="store_true", help="Validate only; never consume a lease.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.check:
        raise SystemExit("Only --check is available; lease consumption belongs to the future controller.")
    trust_anchor = os.environ.get("AXON_LOOP164_TRAINING_TRUST_ANCHOR", "")
    trusted_key = os.environ.get("AXON_LOOP164_TRAINING_TRUSTED_KEY_FINGERPRINT", "")
    if not trust_anchor or not trusted_key:
        print(
            json.dumps(
                {"ready": False, "blockers": ["external_trust_anchor_or_trusted_key_not_configured"]}
            )
        )
        return 2
    result = validate_training_authority(
        trust_anchor_json=Path(trust_anchor),
        expected_trusted_key_fingerprint=trusted_key,
    )
    print(json.dumps({"ready": result.ready, "blockers": list(result.blockers)}, indent=2))
    return 0 if result.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
