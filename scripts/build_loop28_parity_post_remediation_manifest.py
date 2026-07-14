#!/usr/bin/env python3
"""Freeze a successful or synthetic-blocked Loop28 remediation evidence chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "axon_loop28_parity_post_remediation_manifest_v1"
LOOP_ID = "p0_loop28_parity_remediation_001"
IMPLEMENTATION_SCHEMA = "axon_loop28_parity_remediation_implementation_manifest_v1"
RUN_AUTHORIZATION_SCHEMA = "axon_loop28_parity_remediation_run_authorization_v1"
ATTEMPT_LEASE_SCHEMA = "axon_loop28_parity_remediation_attempt_lease_v1"
RECEIPT_SCHEMA = "axon_loop28_parity_remediation_receipt_v1"
BLOCKED_EVIDENCE_SCHEMA = "axon_loop28_parity_remediation_synthetic_pre_run_blocked_v1"
SYNTHETIC_DISCOVERY_SCHEMA = "axon_loop28_parity_remediation_synthetic_discovery_v1"
PROPOSAL_SCHEMA = "axon_loop28_parity_remediation_proposal_v1"
AUTHORIZATION_SCHEMA = "axon_loop28_parity_remediation_authorization_v1"
PREFLIGHT_SCHEMA = "axon_loop28_parity_remediation_preflight_v1"

IMPLEMENTATION_MANIFEST = Path(
    "manifests/roadmap_9997/p0_loop28_parity_remediation/implementation_manifest.json"
)
RUN_AUTHORIZATION = Path(
    "manifests/roadmap_9997/p0_loop28_parity_remediation/run_authorization.json"
)
ATTEMPT_LEASE = Path("manifests/roadmap_9997/p0_loop28_parity_remediation/run_attempt.final.json")
FINAL_RECEIPT = Path(
    "reports/roadmap_9997/p0_loop28_parity_remediation/remediation_receipt.final.json"
)
RECOMMENDATIONS = Path("docs/ml_improvement_recommendations.md")
EXPERIMENT_JOURNAL = Path("reports/hard_family_finetune/experiment_journal.md")
GOAL = Path("goal.md")
DEFAULT_OUTPUT = Path(
    "manifests/roadmap_9997/p0_loop28_parity_remediation/post_remediation_manifest.json"
)
PROPOSAL = Path("manifests/roadmap_9997/p0_loop28_parity_remediation/proposal.json")
AUTHORIZATION = Path("manifests/roadmap_9997/p0_loop28_parity_remediation/authorization.json")
PREFLIGHT = Path("manifests/roadmap_9997/p0_loop28_parity_remediation/preflight.json")
POST_BUILDER = Path("scripts/build_loop28_parity_post_remediation_manifest.py")
POST_BUILDER_TEST = Path("tests/test_build_loop28_parity_post_remediation_manifest.py")
IMPLEMENTATION_BUILDER = Path("scripts/build_loop28_parity_remediation_manifest.py")
IMPLEMENTATION_BUILDER_TEST = Path("tests/test_build_loop28_parity_remediation_manifest.py")
SYNTHETIC_DISCOVERY = Path(
    "manifests/roadmap_9997/p0_loop28_parity_remediation/synthetic_discovery.json"
)
BLOCKED_EVIDENCE = Path(
    "manifests/roadmap_9997/p0_loop28_parity_remediation/synthetic_pre_run_blocked.json"
)
NATIVE_SOURCE = Path("tools/axon_onnx_dll/src/axon_onnx_predict.cpp")
NATIVE_DLL = Path("tools/axon_onnx_dll/build/bin/Release/axon_onnx_predict.dll")
NATIVE_SELFTEST = Path("tools/axon_onnx_dll/build/bin/Release/axon_onnx_selftest.exe")
NATIVE_TEST_SOURCE = Path("tests/test_native_loop28_parity_source.py")
NATIVE_ONNXRUNTIME = Path("tools/axon_onnx_dll/build/bin/Release/onnxruntime.dll")
NATIVE_ONNX = Path("models/random_20w_8192/axon_loop28_base.onnx")
NATIVE_ONNX_DATA = Path("models/random_20w_8192/axon_loop28_base.onnx.data")

FIXED_ATTEMPT_ID = "p0_loop28_parity_remediation_001_final_attempt_001"
FIXED_TOLERANCE = 1.0e-6
FIXED_SAMPLE = {
    "split": "train",
    "sample_index": 0,
    "source_sha256": "09b6b8c80bc31846312bd6958e1d4bf1bcd72d25450d7f7dec2bce6ba81798cc",
    "size_bytes": 4_218_880,
}
EXPECTED_BUDGET = {
    "max_remediation_generations": 1,
    "max_verified_raw_snapshots": 1,
    "max_python_executions": 2,
    "max_native_executions": 6,
    "max_crossfeed_executions": 4,
    "per_execution_timeout_seconds": 120,
    "total_wall_clock_seconds": 1200,
    "cpu_only": True,
    "gpu_executions": 0,
    "network_requests": 0,
    "max_output_bytes": 67_108_864,
}
EXPECTED_TIMEOUT_ENFORCEMENT = {
    "native_subprocess": "hard_timeout",
    "verified_snapshot": "post_return_elapsed_rejection",
    "python_inference": "post_return_elapsed_rejection",
    "total_flow": "checkpointed_elapsed_rejection_outer_watchdog_required",
}
RUN_CLAIM_SCOPE = {
    "train_only": True,
    "raw_identity_count": 1,
    "heldout_access_allowed": False,
    "training_or_fitting_allowed": False,
    "quality_claim_allowed": False,
    "population_parity_claim_allowed": False,
    "certification_claim_allowed": False,
}
RECEIPT_CLAIM_SCOPE = {
    "train_only_remediation_check": True,
    "quality_claim_allowed": False,
    "population_parity_claim_allowed": False,
    "heldout_raw_accessed": False,
    "heldout_predictions_accessed": False,
    "heldout_metrics_accessed": False,
    "split_metadata_use": "identity_audit_only",
    "training_or_fitting_performed": False,
    "artifact_binding": "pre_and_post_path_hash_verification",
    "same_handle_artifact_snapshot": False,
    "concurrent_adversarial_mutation_resistant": False,
    "certification_claim_allowed": False,
}
GATE_CHECKS = {
    "byte_seq",
    "pe_features",
    "stat_features",
    "stage2_features_6_through_1519_exact",
    "base_decision_match",
    "final_decision_match",
    "base_probability_within_tolerance",
    "stage2_probability_within_tolerance",
    "final_probability_within_tolerance",
}
REQUIRED_IMPLEMENTATION_RECORDS = {
    "proposal": PROPOSAL,
    "authorization": AUTHORIZATION,
    "preflight": PREFLIGHT,
    "post_remediation_builder": POST_BUILDER,
    "post_remediation_builder_tests": POST_BUILDER_TEST,
}
FORBIDDEN_IMPLEMENTATION_PATHS = {
    DEFAULT_OUTPUT.as_posix(),
    RUN_AUTHORIZATION.as_posix(),
    ATTEMPT_LEASE.as_posix(),
    FINAL_RECEIPT.as_posix(),
    RECOMMENDATIONS.as_posix(),
    EXPERIMENT_JOURNAL.as_posix(),
    GOAL.as_posix(),
}
RUN_CHAIN_PATHS = (
    IMPLEMENTATION_MANIFEST,
    RUN_AUTHORIZATION,
    ATTEMPT_LEASE,
    FINAL_RECEIPT,
)
BLOCKED_VERIFIED_ARTIFACTS = {
    "native_source": ("native_source", "remediated_native_implementation", NATIVE_SOURCE),
    "synthetic_tests": ("native_test_source", "expanded_synthetic_test_source", NATIVE_TEST_SOURCE),
    "native_dll": ("native_dll", "verified_release_binary", NATIVE_DLL),
    "native_selftest": ("native_selftest", "verified_release_selftest", NATIVE_SELFTEST),
    "onnxruntime": ("onnxruntime", "frozen_native_runtime", NATIVE_ONNXRUNTIME),
    "native_onnx": ("native_onnx", "frozen_native_base_model", NATIVE_ONNX),
    "native_onnx_data": (
        "native_onnx_data",
        "frozen_native_base_weights",
        NATIVE_ONNX_DATA,
    ),
}
BLOCKED_DECISION = "block_train_rerun_input_dependent_onnx_base_drift"
BLOCKED_CLOSURE_DECISION = "synthetic_pre_run_blocked_closure_frozen_no_raw_execution"
PROPOSAL_BLOCKED_DECISION = "propose_synthetic_only_onnx_fidelity_localization_raw_rerun_blocked"
AUTHORIZATION_BLOCKED_DECISION = "allow_synthetic_only_onnx_fidelity_localization_raw_rerun_blocked"
PREFLIGHT_BLOCKED_DECISION = "synthetic_pre_run_blocked_closure_ready"
EXPECTED_BASE_MAX_DELTA = 0.0074248304705810675
EXPECTED_STAGE2_MAX_DELTA = 0.041100940333939406
EXPECTED_BLOCKED_FIXTURES = {
    "pe32_numeric_resource_tls_callbacks": (False, False, True),
    "pe32_named_resource_tls_callbacks": (False, True, True),
    "pe32_numeric_resource_zero_tls_callbacks": (False, False, False),
    "pe32_plus_named_resource_zero_tls_callbacks": (True, True, False),
}


class PostRemediationManifestError(RuntimeError):
    pass


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise PostRemediationManifestError(f"Duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def _validate_generated_at(value: str) -> str:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise PostRemediationManifestError("generated_at_utc must be RFC3339") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PostRemediationManifestError("generated_at_utc must include a timezone")
    return value


def _resolve_project_path(
    project_root: Path,
    relative_path: Path,
    *,
    purpose: str,
    must_exist: bool,
) -> Path:
    root = project_root.resolve(strict=True)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise PostRemediationManifestError(f"{purpose} must be a canonical project-relative path")
    resolved = (root / relative_path).resolve(strict=must_exist)
    if root != resolved and root not in resolved.parents:
        raise PostRemediationManifestError(f"{purpose} escapes project root")
    return resolved


def resolve_fixed_output(project_root: Path, requested_path: Path) -> Path:
    requested = _resolve_project_path(
        project_root,
        requested_path,
        purpose="Post-remediation manifest output",
        must_exist=False,
    )
    frozen = _resolve_project_path(
        project_root,
        DEFAULT_OUTPUT,
        purpose="Frozen post-remediation manifest output",
        must_exist=False,
    )
    if requested != frozen:
        raise PostRemediationManifestError("Post-remediation manifest output path is not fixed")
    return requested


def _read_stable_bytes(project_root: Path, relative_path: Path) -> tuple[bytes, int, str]:
    path = _resolve_project_path(
        project_root,
        relative_path,
        purpose=f"Closure artifact {relative_path}",
        must_exist=True,
    )
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise PostRemediationManifestError(
            f"Closure artifact is not a regular file: {relative_path}"
        )
    payload = path.read_bytes()
    after = path.lstat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise PostRemediationManifestError(
            f"Closure artifact changed while reading: {relative_path}"
        )
    return payload, before.st_size, hashlib.sha256(payload).hexdigest()


def _read_json(
    project_root: Path,
    relative_path: Path,
    *,
    expected_schema: str,
) -> tuple[dict, int, str]:
    payload_bytes, size_bytes, sha256 = _read_stable_bytes(project_root, relative_path)
    try:
        payload = json.loads(
            payload_bytes.decode("utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostRemediationManifestError(f"Invalid JSON artifact: {relative_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise PostRemediationManifestError(f"JSON schema mismatch: {relative_path}")
    return payload, size_bytes, sha256


def _file_record(project_root: Path, name: str, role: str, path: Path) -> dict:
    _payload, size_bytes, sha256 = _read_stable_bytes(project_root, path)
    return {
        "name": name,
        "role": role,
        "path": path.as_posix(),
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _validate_implementation_manifest(
    project_root: Path,
    payload: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    expected_fields = {
        "schema",
        "loop_id",
        "generated_at_utc",
        "contract",
        "claim_scope",
        "artifacts",
        "integrity",
        "decision",
    }
    if set(payload) != expected_fields or payload.get("loop_id") != LOOP_ID:
        raise PostRemediationManifestError("Implementation manifest contract drifted")
    if payload.get("decision") != "implementation_hash_closure_verified_run_authorization_pending":
        raise PostRemediationManifestError(
            "Implementation manifest is not a verified run-ready closure"
        )
    contract = payload.get("contract")
    if not isinstance(contract, Mapping) or (
        contract.get("manifest_self_hashed") is not False
        or contract.get("output_replace_allowed") is not False
    ):
        raise PostRemediationManifestError("Implementation manifest cycle contract drifted")
    claim_scope = payload.get("claim_scope")
    if not isinstance(claim_scope, Mapping) or (
        claim_scope.get("implementation_hash_closure_only") is not True
        or claim_scope.get("raw_execution_performed") is not False
        or claim_scope.get("quality_claim_allowed") is not False
        or claim_scope.get("parity_claim_allowed") is not False
        or claim_scope.get("certification_claim_allowed") is not False
    ):
        raise PostRemediationManifestError("Implementation manifest claim scope drifted")

    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise PostRemediationManifestError("Implementation artifact inventory is missing")
    records: dict[str, Mapping[str, object]] = {}
    paths: set[str] = set()
    expected_count = 0
    verified_expected = 0
    expected_record_fields = {
        "name",
        "role",
        "path",
        "required",
        "expected_sha256",
        "exists",
        "size_bytes",
        "sha256",
        "expected_sha256_match",
    }
    for record in raw_artifacts:
        if not isinstance(record, Mapping) or set(record) != expected_record_fields:
            raise PostRemediationManifestError("Implementation artifact record drifted")
        name = record.get("name")
        path = record.get("path")
        size_bytes = record.get("size_bytes")
        sha256 = record.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or name in records
            or not isinstance(path, str)
            or not path
            or path.casefold() in paths
            or record.get("required") is not True
            or record.get("exists") is not True
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not _is_sha256(sha256)
        ):
            raise PostRemediationManifestError("Implementation artifact inventory is invalid")
        if path in FORBIDDEN_IMPLEMENTATION_PATHS:
            raise PostRemediationManifestError(
                f"Implementation manifest contains a cyclic output: {path}"
            )
        expected_sha256 = record.get("expected_sha256")
        expected_match = record.get("expected_sha256_match")
        if expected_sha256 is None:
            if expected_match is not None:
                raise PostRemediationManifestError("Dynamic implementation hash match flag drifted")
        else:
            expected_count += 1
            if not _is_sha256(expected_sha256) or expected_match is not True:
                raise PostRemediationManifestError("Predeclared implementation hash did not verify")
            verified_expected += 1
        records[name] = record
        paths.add(path.casefold())

    integrity = payload.get("integrity")
    expected_integrity = {
        "artifact_count": len(records),
        "required_artifact_count": len(records),
        "present_required_artifact_count": len(records),
        "predeclared_sha256_count": expected_count,
        "verified_predeclared_sha256_count": verified_expected,
        "blockers": [],
    }
    if integrity != expected_integrity:
        raise PostRemediationManifestError("Implementation manifest integrity summary drifted")

    for name, expected_path in REQUIRED_IMPLEMENTATION_RECORDS.items():
        record = records.get(name)
        if record is None or record.get("path") != expected_path.as_posix():
            raise PostRemediationManifestError(
                f"Required implementation binding is missing: {name}"
            )
        current = _file_record(project_root, name, "current_binding", expected_path)
        if current["size_bytes"] != record.get("size_bytes") or current["sha256"] != record.get(
            "sha256"
        ):
            raise PostRemediationManifestError(f"Required implementation binding drifted: {name}")
    return records


def _validate_run_authorization(
    payload: Mapping[str, object],
    *,
    implementation_sha256: str,
    implementation_records: Mapping[str, Mapping[str, object]],
) -> None:
    expected_fields = {
        "schema",
        "loop_id",
        "issued_at_utc",
        "prereg_authorization_sha256",
        "proposal_sha256",
        "parent_evidence",
        "implementation_manifest",
        "frozen_sample",
        "budget",
        "timeout_enforcement",
        "frozen_tolerance",
        "attempt_id",
        "attempt_lease_path",
        "generation",
        "output_path",
        "claim_scope",
        "decision",
    }
    if set(payload) != expected_fields or payload.get("loop_id") != LOOP_ID:
        raise PostRemediationManifestError("Run authorization contract drifted")
    _validate_generated_at(str(payload.get("issued_at_utc") or ""))
    expected_values = {
        "prereg_authorization_sha256": implementation_records["authorization"]["sha256"],
        "proposal_sha256": implementation_records["proposal"]["sha256"],
        "frozen_sample": FIXED_SAMPLE,
        "budget": EXPECTED_BUDGET,
        "timeout_enforcement": EXPECTED_TIMEOUT_ENFORCEMENT,
        "frozen_tolerance": FIXED_TOLERANCE,
        "attempt_id": FIXED_ATTEMPT_ID,
        "attempt_lease_path": ATTEMPT_LEASE.as_posix(),
        "generation": "final",
        "output_path": FINAL_RECEIPT.as_posix(),
        "claim_scope": RUN_CLAIM_SCOPE,
        "decision": "allow_bounded_loop28_parity_remediation_run",
    }
    for field, expected in expected_values.items():
        if payload.get(field) != expected:
            raise PostRemediationManifestError(f"Run authorization binding drifted: {field}")
    implementation_record = payload.get("implementation_manifest")
    if implementation_record != {
        "path": IMPLEMENTATION_MANIFEST.as_posix(),
        "sha256": implementation_sha256,
    }:
        raise PostRemediationManifestError(
            "Run authorization does not bind implementation manifest"
        )
    parent_evidence = payload.get("parent_evidence")
    if (
        not isinstance(parent_evidence, Mapping)
        or not parent_evidence
        or any(not _is_sha256(value) for value in parent_evidence.values())
    ):
        raise PostRemediationManifestError("Run authorization parent evidence is invalid")


def _validate_lease(payload: Mapping[str, object], *, run_authorization_sha256: str) -> None:
    if set(payload) != {
        "schema",
        "loop_id",
        "attempt_id",
        "generation",
        "run_authorization_sha256",
        "output_path",
        "consumed_at_utc",
        "status",
    }:
        raise PostRemediationManifestError("Attempt lease contract drifted")
    expected_values = {
        "loop_id": LOOP_ID,
        "attempt_id": FIXED_ATTEMPT_ID,
        "generation": "final",
        "run_authorization_sha256": run_authorization_sha256,
        "output_path": FINAL_RECEIPT.as_posix(),
        "status": "authorization_consumed_before_raw_access",
    }
    for field, expected in expected_values.items():
        if payload.get(field) != expected:
            raise PostRemediationManifestError(f"Attempt lease binding drifted: {field}")
    _validate_generated_at(str(payload.get("consumed_at_utc") or ""))


def _validate_budget_audit(payload: object) -> None:
    if not isinstance(payload, Mapping) or payload.get("within_budget") is not True:
        raise PostRemediationManifestError("Receipt budget audit did not pass")
    limits = {
        "generation": EXPECTED_BUDGET["max_remediation_generations"],
        "verified_raw_snapshot": EXPECTED_BUDGET["max_verified_raw_snapshots"],
        "python": EXPECTED_BUDGET["max_python_executions"],
        "native": EXPECTED_BUDGET["max_native_executions"],
        "crossfeed": EXPECTED_BUDGET["max_crossfeed_executions"],
    }
    for name, limit in limits.items():
        record = payload.get(name)
        if (
            not isinstance(record, Mapping)
            or record.get("within_budget") is not True
            or record.get("limit") != limit
            or isinstance(record.get("count"), bool)
            or not isinstance(record.get("count"), int)
            or record["count"] < 0
            or record["count"] > limit
        ):
            raise PostRemediationManifestError(f"Receipt budget row is invalid: {name}")
    for name in ("total_wall_clock", "output"):
        record = payload.get(name)
        if not isinstance(record, Mapping) or record.get("within_budget") is not True:
            raise PostRemediationManifestError(f"Receipt budget row is invalid: {name}")


def _validate_success_gate(payload: object) -> dict:
    if not isinstance(payload, Mapping) or set(payload) != {
        "tolerance",
        "checks",
        "stage2_probability_transform_mismatch_indices",
        "base_probability_max_absolute_delta",
        "stage2_probability_absolute_delta",
        "final_probability_max_absolute_delta",
        "passed",
        "decision",
    }:
        raise PostRemediationManifestError("Receipt success gate contract drifted")
    checks = payload.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != GATE_CHECKS
        or not all(value is True for value in checks.values())
    ):
        raise PostRemediationManifestError("Receipt success gate checks did not all pass")
    if (
        payload.get("tolerance") != FIXED_TOLERANCE
        or payload.get("passed") is not True
        or payload.get("decision") != "frozen_train_remediation_gate_passed"
    ):
        raise PostRemediationManifestError("Receipt cannot claim a successful remediation closure")
    mismatch_indices = payload.get("stage2_probability_transform_mismatch_indices")
    if (
        not isinstance(mismatch_indices, list)
        or mismatch_indices != sorted(set(mismatch_indices))
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= 6
            for index in mismatch_indices
        )
    ):
        raise PostRemediationManifestError("Receipt Stage-2 mismatch scope drifted")
    for field in (
        "base_probability_max_absolute_delta",
        "stage2_probability_absolute_delta",
        "final_probability_max_absolute_delta",
    ):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PostRemediationManifestError(f"Receipt success delta is invalid: {field}")
        value = float(value)
        if not math.isfinite(value) or value < 0.0 or value > FIXED_TOLERANCE:
            raise PostRemediationManifestError(f"Receipt success delta exceeds tolerance: {field}")
    return dict(payload)


def _validate_receipt(
    payload: Mapping[str, object],
    *,
    implementation_sha256: str,
    run_authorization_sha256: str,
    lease_sha256: str,
    run_authorization: Mapping[str, object],
    lease: Mapping[str, object],
    implementation_records: Mapping[str, Mapping[str, object]],
) -> dict:
    if set(payload) != {
        "schema",
        "generated_at_utc",
        "claim_scope",
        "generation",
        "authorization",
        "sample_identity",
        "evidence_sha256",
        "identity_audit",
        "budget_audit",
        "authenticated_comparison",
        "success_gate",
        "decision",
    }:
        raise PostRemediationManifestError("Final receipt contract drifted")
    _validate_generated_at(str(payload.get("generated_at_utc") or ""))
    if payload.get("claim_scope") != RECEIPT_CLAIM_SCOPE or payload.get("generation") != "final":
        raise PostRemediationManifestError("Final receipt claim scope drifted")
    if payload.get("sample_identity") != {**FIXED_SAMPLE, "label": 0} and payload.get(
        "sample_identity"
    ) != {**FIXED_SAMPLE, "label": 1}:
        raise PostRemediationManifestError("Final receipt sample identity drifted")

    authorization = payload.get("authorization")
    if not isinstance(authorization, Mapping) or set(authorization) != {
        "preregistration",
        "run",
        "attempt_lease",
    }:
        raise PostRemediationManifestError("Final receipt authorization chain drifted")
    preregistration = authorization["preregistration"]
    if not isinstance(preregistration, Mapping) or (
        preregistration.get("schema") != "axon_loop28_parity_remediation_authorization_v1"
        or preregistration.get("loop_id") != LOOP_ID
        or preregistration.get("authorization_sha256")
        != implementation_records["authorization"]["sha256"]
        or preregistration.get("proposal_sha256") != implementation_records["proposal"]["sha256"]
        or preregistration.get("budget") != EXPECTED_BUDGET
        or preregistration.get("frozen_tolerance") != FIXED_TOLERANCE
        or preregistration.get("status") != "authorized_contract_verified"
    ):
        raise PostRemediationManifestError("Final receipt preregistration binding drifted")
    receipt_run = authorization["run"]
    expected_run = {
        "schema": RUN_AUTHORIZATION_SCHEMA,
        "loop_id": LOOP_ID,
        "authorization_sha256": run_authorization_sha256,
        "prereg_authorization_sha256": run_authorization["prereg_authorization_sha256"],
        "proposal_sha256": run_authorization["proposal_sha256"],
        "parent_evidence_sha256": run_authorization["parent_evidence"],
        "implementation_manifest_sha256": implementation_sha256,
        "implementation_artifact_count": len(implementation_records),
        "generation": "final",
        "attempt_id": FIXED_ATTEMPT_ID,
        "attempt_lease_path": ATTEMPT_LEASE.as_posix(),
        "status": "bounded_run_authorized",
    }
    if receipt_run != expected_run:
        raise PostRemediationManifestError("Final receipt run authorization binding drifted")
    attempt_lease = authorization["attempt_lease"]
    if attempt_lease != {
        "path": ATTEMPT_LEASE.as_posix(),
        "sha256": lease_sha256,
        "consumed_at_utc": lease["consumed_at_utc"],
        "status": "authorization_consumed_before_raw_access",
    }:
        raise PostRemediationManifestError("Final receipt lease binding drifted")

    evidence = payload.get("evidence_sha256")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("implementation_manifest") != implementation_sha256
        or any(not _is_sha256(value) for value in evidence.values())
    ):
        raise PostRemediationManifestError("Final receipt evidence binding drifted")
    identity_audit = payload.get("identity_audit")
    if not isinstance(identity_audit, Mapping) or (
        identity_audit.get("scope") != "complete_split_metadata_identity_audit_only"
        or identity_audit.get("raw_files_opened") != 1
        or identity_audit.get("heldout_raw_files_opened") != 0
        or identity_audit.get("prediction_or_metric_rows_read") != 0
        or identity_audit.get("selected_count") != 1
        or identity_audit.get("reported_count") != 1
    ):
        raise PostRemediationManifestError("Final receipt identity audit drifted")
    _validate_budget_audit(payload.get("budget_audit"))
    success_gate = _validate_success_gate(payload.get("success_gate"))
    if payload.get("decision") != success_gate["decision"]:
        raise PostRemediationManifestError("Final receipt decision does not match its success gate")
    if not isinstance(payload.get("authenticated_comparison"), Mapping):
        raise PostRemediationManifestError("Final receipt authenticated comparison is missing")
    return success_gate


def _chain_record(name: str, role: str, path: Path, size_bytes: int, sha256: str) -> dict:
    return {
        "name": name,
        "role": role,
        "path": path.as_posix(),
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _path_present(project_root: Path, path: Path) -> bool:
    resolved = _resolve_project_path(
        project_root,
        path,
        purpose=f"Closure mode artifact {path}",
        must_exist=False,
    )
    return os.path.lexists(resolved)


def _require_run_chain_absent(project_root: Path) -> None:
    present = [path.as_posix() for path in RUN_CHAIN_PATHS if _path_present(project_root, path)]
    if present:
        raise PostRemediationManifestError(
            "Synthetic pre-run blocked closure forbids run-chain artifacts: " + ", ".join(present)
        )


def _validate_blocked_evidence(
    project_root: Path,
    payload: Mapping[str, object],
) -> tuple[dict, list[dict]]:
    expected_fields = {
        "schema",
        "loop_id",
        "generated_at_utc",
        "scope",
        "superseded_discovery",
        "expanded_fixture_contract",
        "common_authenticated_results",
        "fixture_results",
        "gate_summary",
        "verified_artifacts",
        "verification",
        "failure_analysis",
        "claim_boundary",
        "decision",
    }
    if set(payload) != expected_fields or payload.get("loop_id") != LOOP_ID:
        raise PostRemediationManifestError("Synthetic pre-run blocked evidence contract drifted")
    _validate_generated_at(str(payload.get("generated_at_utc") or ""))
    if payload.get("decision") != BLOCKED_DECISION:
        raise PostRemediationManifestError("Synthetic pre-run blocked decision drifted")

    expected_scope = {
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
    }
    if payload.get("scope") != expected_scope:
        raise PostRemediationManifestError("Synthetic pre-run blocked scope drifted")

    superseded = payload.get("superseded_discovery")
    if not isinstance(superseded, Mapping) or (
        superseded.get("path") != SYNTHETIC_DISCOVERY.as_posix()
        or superseded.get("previous_single_fixture_parity_is_sufficient") is not False
        or superseded.get("ort_disable_all_is_sufficient_for_cross_runtime_parity") is not False
        or not isinstance(superseded.get("reason"), str)
        or not superseded["reason"].strip()
    ):
        raise PostRemediationManifestError("Superseded synthetic discovery binding drifted")

    expected_fixture_contract = {
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
    }
    if payload.get("expanded_fixture_contract") != expected_fixture_contract:
        raise PostRemediationManifestError("Expanded synthetic fixture contract drifted")

    expected_common_results = {
        "byte_seq_exact": True,
        "pe_features_exact": True,
        "stat_features_exact": True,
        "stage2_features_indices_6_through_1519_exact": True,
        "base_decisions_match": True,
        "final_decisions_match": True,
        "frozen_tolerance": FIXED_TOLERANCE,
    }
    if payload.get("common_authenticated_results") != expected_common_results:
        raise PostRemediationManifestError("Authenticated synthetic feature results drifted")

    raw_fixtures = payload.get("fixture_results")
    if not isinstance(raw_fixtures, list) or len(raw_fixtures) != 4:
        raise PostRemediationManifestError("Expanded synthetic fixture results are incomplete")
    fixture_fields = {
        "name",
        "pe_plus",
        "named_resource",
        "tls_callbacks",
        "base_probability_absolute_delta",
        "stage2_probability_absolute_delta",
        "stage2_mismatch_indices",
        "base_probability_within_tolerance",
        "stage2_probability_within_tolerance",
        "gate_passed",
    }
    fixtures: dict[str, Mapping[str, object]] = {}
    for fixture in raw_fixtures:
        if not isinstance(fixture, Mapping) or set(fixture) != fixture_fields:
            raise PostRemediationManifestError("Expanded synthetic fixture row drifted")
        name = fixture.get("name")
        if not isinstance(name, str) or name in fixtures or name not in EXPECTED_BLOCKED_FIXTURES:
            raise PostRemediationManifestError("Expanded synthetic fixture identity drifted")
        expected_identity = EXPECTED_BLOCKED_FIXTURES[name]
        actual_identity = (
            fixture.get("pe_plus"),
            fixture.get("named_resource"),
            fixture.get("tls_callbacks"),
        )
        if actual_identity != expected_identity:
            raise PostRemediationManifestError("Expanded synthetic fixture identity fields drifted")
        deltas: dict[str, float] = {}
        for field in (
            "base_probability_absolute_delta",
            "stage2_probability_absolute_delta",
        ):
            value = fixture.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PostRemediationManifestError(
                    f"Synthetic probability delta is invalid: {name}"
                )
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise PostRemediationManifestError(
                    f"Synthetic probability delta is invalid: {name}"
                )
            deltas[field] = numeric
        base_within = deltas["base_probability_absolute_delta"] <= FIXED_TOLERANCE
        stage2_within = deltas["stage2_probability_absolute_delta"] <= FIXED_TOLERANCE
        if (
            fixture.get("base_probability_within_tolerance") is not base_within
            or fixture.get("stage2_probability_within_tolerance") is not stage2_within
            or fixture.get("gate_passed") is not (base_within and stage2_within)
        ):
            raise PostRemediationManifestError("Synthetic fixture gate is internally inconsistent")
        expected_mismatches = [] if fixture["pe_plus"] else [0, 1, 2, 3, 4, 5]
        if fixture.get("stage2_mismatch_indices") != expected_mismatches:
            raise PostRemediationManifestError("Synthetic Stage-2 mismatch scope drifted")
        fixtures[name] = fixture
    if set(fixtures) != set(EXPECTED_BLOCKED_FIXTURES):
        raise PostRemediationManifestError("Expanded synthetic fixture matrix drifted")

    base_max = max(float(row["base_probability_absolute_delta"]) for row in fixtures.values())
    stage2_max = max(float(row["stage2_probability_absolute_delta"]) for row in fixtures.values())
    expected_gate_summary = {
        "fixture_count": 4,
        "passed_fixture_count": 1,
        "failed_fixture_count": 3,
        "pe32_passed_fixture_count": 0,
        "pe32_failed_fixture_count": 3,
        "pe32_plus_passed_fixture_count": 1,
        "base_probability_max_absolute_delta": EXPECTED_BASE_MAX_DELTA,
        "stage2_probability_max_absolute_delta": EXPECTED_STAGE2_MAX_DELTA,
        "input_dependent_runtime_drift_observed": True,
        "feature_remediation_passed": True,
        "cross_runtime_base_model_parity_passed": False,
        "one_train_sample_rerun_allowed": False,
    }
    gate_summary = payload.get("gate_summary")
    if (
        gate_summary != expected_gate_summary
        or base_max != EXPECTED_BASE_MAX_DELTA
        or stage2_max != EXPECTED_STAGE2_MAX_DELTA
    ):
        raise PostRemediationManifestError("Synthetic pre-run gate counts or deltas drifted")

    verification = payload.get("verification")
    if not isinstance(verification, list) or len(verification) != 4:
        raise PostRemediationManifestError("Synthetic verification inventory drifted")
    by_scope: dict[str, Mapping[str, object]] = {}
    for record in verification:
        if not isinstance(record, Mapping):
            raise PostRemediationManifestError("Synthetic verification record drifted")
        scope = record.get("scope")
        duration = record.get("duration_seconds")
        if (
            not isinstance(scope, str)
            or scope in by_scope
            or not isinstance(record.get("command"), str)
            or not isinstance(record.get("result"), str)
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0.0
            or not isinstance(record.get("model_inference_performed"), bool)
        ):
            raise PostRemediationManifestError("Synthetic verification record drifted")
        by_scope[scope] = record
    if set(by_scope) != {
        "static_and_python_synthetic",
        "feature_only_native_dll",
        "expanded_native_model_parity",
        "variant_delta_capture",
    }:
        raise PostRemediationManifestError("Synthetic verification scope drifted")
    expanded_verification = by_scope["expanded_native_model_parity"]
    if (
        expanded_verification.get("result") != "21 passed, 3 failed"
        or expanded_verification.get("model_inference_performed") is not True
        or expanded_verification.get("expected_gate_result") != "failed_closed"
    ):
        raise PostRemediationManifestError("Expanded 24-integration result drifted")

    failure_analysis = payload.get("failure_analysis")
    if not isinstance(failure_analysis, Mapping) or (
        failure_analysis.get("first_remaining_divergence_stage") != "base_inference"
        or failure_analysis.get("feature_extraction_is_still_the_first_divergence") is not False
        or failure_analysis.get("all_base_model_inputs_are_authenticated_exact") is not True
        or failure_analysis.get("stage2_non_probability_features_are_authenticated_exact")
        is not True
        or failure_analysis.get("sole_cause_claim_allowed") is not False
    ):
        raise PostRemediationManifestError("Synthetic failure analysis drifted")
    expected_claim_boundary = {
        "feature_implementation_parity_observed_on_expanded_synthetic_matrix": True,
        "base_runtime_parity_claim_allowed": False,
        "raw_remediation_claim_allowed": False,
        "population_parity_claim_allowed": False,
        "quality_claim_allowed": False,
        "native_loop28_ready_claim_allowed": False,
        "native_loop151_ready_claim_allowed": False,
        "certification_claim_allowed": False,
    }
    if payload.get("claim_boundary") != expected_claim_boundary:
        raise PostRemediationManifestError("Synthetic blocked claim boundary drifted")

    verified_artifacts = payload.get("verified_artifacts")
    if not isinstance(verified_artifacts, Mapping) or set(verified_artifacts) != set(
        BLOCKED_VERIFIED_ARTIFACTS
    ):
        raise PostRemediationManifestError("Verified synthetic artifact inventory drifted")
    artifact_rows = []
    for evidence_name, (closure_name, role, path) in BLOCKED_VERIFIED_ARTIFACTS.items():
        expected_record = verified_artifacts.get(evidence_name)
        if not isinstance(expected_record, Mapping) or set(expected_record) != {"path", "sha256"}:
            raise PostRemediationManifestError(
                f"Verified synthetic artifact record drifted: {evidence_name}"
            )
        if expected_record.get("path") != path.as_posix() or not _is_sha256(
            expected_record.get("sha256")
        ):
            raise PostRemediationManifestError(
                f"Verified synthetic artifact binding drifted: {evidence_name}"
            )
        current = _file_record(project_root, closure_name, role, path)
        if current["sha256"] != expected_record["sha256"]:
            raise PostRemediationManifestError(
                f"Verified synthetic artifact hash mismatch: {evidence_name}"
            )
        artifact_rows.append(current)
    return dict(gate_summary), artifact_rows


def _validate_superseded_discovery(
    payload: Mapping[str, object],
    *,
    blocked_sha256: str,
) -> None:
    if payload.get("loop_id") != LOOP_ID or payload.get("decision") != "superseded_pre_run_blocked":
        raise PostRemediationManifestError("Synthetic discovery was not superseded")
    scope = payload.get("scope")
    if not isinstance(scope, Mapping) or (
        scope.get("dataset_raw_accessed") is not False
        or scope.get("split_metadata_accessed") is not False
        or scope.get("heldout_accessed") is not False
        or scope.get("training_or_fitting_performed") is not False
        or scope.get("f1_or_quality_metric_computed") is not False
    ):
        raise PostRemediationManifestError("Superseded synthetic discovery scope drifted")
    supersession = payload.get("supersession")
    if not isinstance(supersession, Mapping) or (
        supersession.get("status") != "superseded"
        or supersession.get("historical_observation_retained") is not True
        or supersession.get("resource_fixture_valid_for_parity_authorization") is not False
        or supersession.get("single_fixture_pass_authorizes_raw_rerun") is not False
        or supersession.get("invalidated_by")
        != {"path": BLOCKED_EVIDENCE.as_posix(), "sha256": blocked_sha256}
    ):
        raise PostRemediationManifestError("Synthetic discovery supersession binding drifted")


def _blocked_binding(blocked_sha256: str) -> dict:
    return {
        "path": BLOCKED_EVIDENCE.as_posix(),
        "sha256": blocked_sha256,
        "decision": BLOCKED_DECISION,
        "train_raw_rerun_allowed": False,
    }


def _validate_blocked_governance(
    project_root: Path,
    *,
    blocked_sha256: str,
) -> tuple[list[dict], dict[str, str]]:
    proposal, proposal_size, proposal_sha256 = _read_json(
        project_root,
        PROPOSAL,
        expected_schema=PROPOSAL_SCHEMA,
    )
    authorization, authorization_size, authorization_sha256 = _read_json(
        project_root,
        AUTHORIZATION,
        expected_schema=AUTHORIZATION_SCHEMA,
    )
    preflight, preflight_size, preflight_sha256 = _read_json(
        project_root,
        PREFLIGHT,
        expected_schema=PREFLIGHT_SCHEMA,
    )
    expected_binding = _blocked_binding(blocked_sha256)
    if (
        proposal.get("loop_id") != LOOP_ID
        or proposal.get("decision") != PROPOSAL_BLOCKED_DECISION
        or proposal.get("synthetic_pre_run_block") != expected_binding
    ):
        raise PostRemediationManifestError("Blocked proposal contract drifted")
    expected_artifacts = proposal.get("expected_artifacts")
    if not isinstance(expected_artifacts, list) or (
        BLOCKED_EVIDENCE.as_posix() not in expected_artifacts
        or SYNTHETIC_DISCOVERY.as_posix() not in expected_artifacts
        or DEFAULT_OUTPUT.as_posix() not in expected_artifacts
        or GOAL.as_posix() not in expected_artifacts
        or any(path.as_posix() in expected_artifacts for path in RUN_CHAIN_PATHS)
    ):
        raise PostRemediationManifestError("Blocked proposal artifact contract drifted")

    if (
        authorization.get("loop_id") != LOOP_ID
        or authorization.get("decision") != AUTHORIZATION_BLOCKED_DECISION
        or authorization.get("proposal") != {"path": PROPOSAL.as_posix(), "sha256": proposal_sha256}
        or authorization.get("synthetic_pre_run_block") != expected_binding
        or authorization.get("allowed_splits") != []
        or authorization.get("execution_requires_separate_run_authorization") is not False
    ):
        raise PostRemediationManifestError("Blocked authorization contract drifted")
    generated_paths = authorization.get("authorized_generated_paths")
    if not isinstance(generated_paths, list) or any(
        path.as_posix() in generated_paths for path in RUN_CHAIN_PATHS
    ):
        raise PostRemediationManifestError("Blocked authorization output contract drifted")

    if (
        preflight.get("loop_id") != LOOP_ID
        or preflight.get("decision") != PREFLIGHT_BLOCKED_DECISION
        or preflight.get("synthetic_pre_run_block") != expected_binding
        or preflight.get("governance_binding")
        != {
            "proposal": {"path": PROPOSAL.as_posix(), "sha256": proposal_sha256},
            "authorization": {
                "path": AUTHORIZATION.as_posix(),
                "sha256": authorization_sha256,
            },
        }
    ):
        raise PostRemediationManifestError("Blocked preflight governance binding drifted")
    checks = preflight.get("pre_implementation_checks")
    if not isinstance(checks, Mapping) or (
        checks.get("implementation_manifest_present") is not False
        or checks.get("new_run_authorization_present") is not False
        or checks.get("new_attempt_lease_present") is not False
        or checks.get("remediation_receipt_present") is not False
        or checks.get("raw_accessed") is not False
        or checks.get("heldout_accessed") is not False
    ):
        raise PostRemediationManifestError("Blocked preflight absence contract drifted")

    rows = [
        _chain_record(
            "proposal", "final_blocked_proposal", PROPOSAL, proposal_size, proposal_sha256
        ),
        _chain_record(
            "authorization",
            "final_blocked_authorization",
            AUTHORIZATION,
            authorization_size,
            authorization_sha256,
        ),
        _chain_record(
            "preflight",
            "final_blocked_preflight",
            PREFLIGHT,
            preflight_size,
            preflight_sha256,
        ),
    ]
    hashes = {
        "proposal_sha256": proposal_sha256,
        "authorization_sha256": authorization_sha256,
        "preflight_sha256": preflight_sha256,
    }
    return rows, hashes


def _build_blocked_post_remediation_manifest(
    project_root: Path,
    *,
    generated_at_utc: str,
) -> dict:
    _require_run_chain_absent(project_root)
    blocked, blocked_size, blocked_sha256 = _read_json(
        project_root,
        BLOCKED_EVIDENCE,
        expected_schema=BLOCKED_EVIDENCE_SCHEMA,
    )
    gate_summary, verified_artifact_rows = _validate_blocked_evidence(project_root, blocked)
    discovery, discovery_size, discovery_sha256 = _read_json(
        project_root,
        SYNTHETIC_DISCOVERY,
        expected_schema=SYNTHETIC_DISCOVERY_SCHEMA,
    )
    _validate_superseded_discovery(discovery, blocked_sha256=blocked_sha256)
    governance_rows, governance_hashes = _validate_blocked_governance(
        project_root,
        blocked_sha256=blocked_sha256,
    )
    artifacts = [
        *governance_rows,
        _chain_record(
            "synthetic_discovery",
            "superseded_historical_synthetic_observation",
            SYNTHETIC_DISCOVERY,
            discovery_size,
            discovery_sha256,
        ),
        _chain_record(
            "synthetic_pre_run_blocked",
            "expanded_synthetic_fail_closed_evidence",
            BLOCKED_EVIDENCE,
            blocked_size,
            blocked_sha256,
        ),
        _file_record(
            project_root,
            "implementation_builder",
            "blocked_implementation_manifest_builder",
            IMPLEMENTATION_BUILDER,
        ),
        _file_record(
            project_root,
            "implementation_builder_tests",
            "blocked_implementation_builder_tests",
            IMPLEMENTATION_BUILDER_TEST,
        ),
        _file_record(
            project_root,
            "post_remediation_builder",
            "blocked_closure_builder",
            POST_BUILDER,
        ),
        _file_record(
            project_root,
            "post_remediation_builder_tests",
            "blocked_closure_builder_tests",
            POST_BUILDER_TEST,
        ),
        *verified_artifact_rows,
        _file_record(
            project_root,
            "recommendations",
            "owner_facing_final_blocked_status",
            RECOMMENDATIONS,
        ),
        _file_record(
            project_root,
            "experiment_journal",
            "durable_blocked_experiment_record",
            EXPERIMENT_JOURNAL,
        ),
        _file_record(
            project_root,
            "goal",
            "current_synthetic_only_onnx_fidelity_plan",
            GOAL,
        ),
    ]
    return {
        "schema": SCHEMA,
        "loop_id": LOOP_ID,
        "generated_at_utc": generated_at_utc,
        "contract": {
            "closure_mode": "synthetic_pre_run_blocked",
            "structured_evidence_validation": True,
            "duplicate_json_keys_rejected": True,
            "blocked_artifact_hashes_reverified": True,
            "run_chain_absence_verified": True,
            "raw_split_heldout_payloads_opened": False,
            "model_payloads_parsed": False,
            "manifest_self_hashed": False,
            "output_replace_allowed": False,
        },
        "claim_scope": {
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
        },
        "lineage": {
            **governance_hashes,
            "synthetic_discovery_sha256": discovery_sha256,
            "synthetic_pre_run_blocked_sha256": blocked_sha256,
        },
        "synthetic_gate": gate_summary,
        "artifacts": artifacts,
        "integrity": {
            "artifact_count": len(artifacts),
            "required_artifact_count": len(artifacts),
            "present_required_artifact_count": len(artifacts),
            "structured_chain_links_verified": 12,
            "run_chain_artifact_count": 0,
            "blockers": [],
        },
        "decision": BLOCKED_CLOSURE_DECISION,
    }


def _build_successful_post_remediation_manifest(
    project_root: Path, *, generated_at_utc: str
) -> dict:
    generated_at_utc = _validate_generated_at(generated_at_utc)
    implementation, implementation_size, implementation_sha256 = _read_json(
        project_root,
        IMPLEMENTATION_MANIFEST,
        expected_schema=IMPLEMENTATION_SCHEMA,
    )
    implementation_records = _validate_implementation_manifest(project_root, implementation)
    run_authorization, run_size, run_sha256 = _read_json(
        project_root,
        RUN_AUTHORIZATION,
        expected_schema=RUN_AUTHORIZATION_SCHEMA,
    )
    _validate_run_authorization(
        run_authorization,
        implementation_sha256=implementation_sha256,
        implementation_records=implementation_records,
    )
    lease, lease_size, lease_sha256 = _read_json(
        project_root,
        ATTEMPT_LEASE,
        expected_schema=ATTEMPT_LEASE_SCHEMA,
    )
    _validate_lease(lease, run_authorization_sha256=run_sha256)
    receipt, receipt_size, receipt_sha256 = _read_json(
        project_root,
        FINAL_RECEIPT,
        expected_schema=RECEIPT_SCHEMA,
    )
    success_gate = _validate_receipt(
        receipt,
        implementation_sha256=implementation_sha256,
        run_authorization_sha256=run_sha256,
        lease_sha256=lease_sha256,
        run_authorization=run_authorization,
        lease=lease,
        implementation_records=implementation_records,
    )

    artifacts = [
        _chain_record(
            "implementation_manifest",
            "pre_run_implementation_closure",
            IMPLEMENTATION_MANIFEST,
            implementation_size,
            implementation_sha256,
        ),
        _chain_record(
            "run_authorization",
            "manifest_bound_run_authorization",
            RUN_AUTHORIZATION,
            run_size,
            run_sha256,
        ),
        _chain_record(
            "consumed_lease",
            "one_shot_attempt_lease",
            ATTEMPT_LEASE,
            lease_size,
            lease_sha256,
        ),
        _chain_record(
            "final_receipt",
            "successful_train_only_remediation_receipt",
            FINAL_RECEIPT,
            receipt_size,
            receipt_sha256,
        ),
        _file_record(
            project_root,
            "recommendations",
            "owner_facing_final_status",
            RECOMMENDATIONS,
        ),
        _file_record(
            project_root,
            "experiment_journal",
            "durable_final_experiment_record",
            EXPERIMENT_JOURNAL,
        ),
    ]
    return {
        "schema": SCHEMA,
        "loop_id": LOOP_ID,
        "generated_at_utc": generated_at_utc,
        "contract": {
            "structured_chain_validation": True,
            "duplicate_json_keys_rejected": True,
            "dynamic_hashes_verified_from_chain": True,
            "raw_split_model_payloads_opened": False,
            "manifest_self_hashed": False,
            "output_replace_allowed": False,
        },
        "claim_scope": {
            "train_only_remediation_gate_passed": True,
            "quality_claim_allowed": False,
            "population_parity_claim_allowed": False,
            "native_loop151_ready": False,
            "certification_claim_allowed": False,
            "productization_performance_gate_pending": True,
        },
        "lineage": {
            "implementation_manifest_sha256": implementation_sha256,
            "run_authorization_sha256": run_sha256,
            "consumed_lease_sha256": lease_sha256,
            "final_receipt_sha256": receipt_sha256,
            "prereg_authorization_sha256": run_authorization["prereg_authorization_sha256"],
            "proposal_sha256": run_authorization["proposal_sha256"],
            "attempt_id": FIXED_ATTEMPT_ID,
        },
        "success_gate": success_gate,
        "artifacts": artifacts,
        "integrity": {
            "artifact_count": len(artifacts),
            "required_artifact_count": len(artifacts),
            "present_required_artifact_count": len(artifacts),
            "structured_chain_links_verified": 4,
            "blockers": [],
        },
        "decision": "post_remediation_closure_frozen_train_gate_passed_productization_pending",
    }


def build_post_remediation_manifest(project_root: Path, *, generated_at_utc: str) -> dict:
    generated_at_utc = _validate_generated_at(generated_at_utc)
    blocked_present = _path_present(project_root, BLOCKED_EVIDENCE)
    run_chain_present = [path for path in RUN_CHAIN_PATHS if _path_present(project_root, path)]
    if blocked_present:
        if run_chain_present:
            raise PostRemediationManifestError(
                "Synthetic pre-run blocked evidence cannot coexist with run-chain artifacts"
            )
        return _build_blocked_post_remediation_manifest(
            project_root,
            generated_at_utc=generated_at_utc,
        )
    missing_run_chain = [path for path in RUN_CHAIN_PATHS if path not in run_chain_present]
    if missing_run_chain:
        missing = ", ".join(path.as_posix() for path in missing_run_chain)
        raise PostRemediationManifestError(
            "Blocked evidence is missing and successful run chain is incomplete: " + missing
        )
    return _build_successful_post_remediation_manifest(
        project_root,
        generated_at_utc=generated_at_utc,
    )


def _write_exclusive(output_path: Path, payload: Mapping[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with output_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PostRemediationManifestError(f"Output already exists: {output_path}") from exc


def verify_post_remediation_manifest(project_root: Path, manifest_path: Path) -> dict:
    resolved = resolve_fixed_output(project_root, manifest_path)
    payload_bytes, _size_bytes, _sha256 = _read_stable_bytes(project_root, manifest_path)
    if resolved != _resolve_project_path(
        project_root,
        manifest_path,
        purpose="Requested post-remediation manifest",
        must_exist=True,
    ):
        raise PostRemediationManifestError("Post-remediation manifest path drifted")
    try:
        payload = json.loads(
            payload_bytes.decode("utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostRemediationManifestError("Post-remediation manifest is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise PostRemediationManifestError("Post-remediation manifest schema mismatch")
    rebuilt = build_post_remediation_manifest(
        project_root,
        generated_at_utc=str(payload.get("generated_at_utc") or ""),
    )
    if payload != rebuilt:
        raise PostRemediationManifestError("Post-remediation manifest no longer matches its chain")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-at-utc")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve(strict=True)
    output_path = resolve_fixed_output(project_root, args.output)
    if args.verify:
        manifest = verify_post_remediation_manifest(project_root, args.output)
    else:
        if not args.generated_at_utc:
            raise PostRemediationManifestError("--generated-at-utc is required when building")
        manifest = build_post_remediation_manifest(
            project_root,
            generated_at_utc=args.generated_at_utc,
        )
        _write_exclusive(output_path, manifest)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "artifact_count": manifest["integrity"]["artifact_count"],
                "decision": manifest["decision"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
