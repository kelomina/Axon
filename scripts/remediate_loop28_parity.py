#!/usr/bin/env python3
"""Execute one authenticated train-only Loop28 parity remediation check."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import build_loop28_parity_remediation_manifest as implementation_manifest
import diagnose_loop28_parity as diagnostic
import replay_loop151_raw as replay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "p0_loop28_parity_remediation_001"
AUTHORIZATION_SCHEMA = "axon_loop28_parity_remediation_authorization_v1"
RUN_AUTHORIZATION_SCHEMA = "axon_loop28_parity_remediation_run_authorization_v1"
ATTEMPT_LEASE_SCHEMA = "axon_loop28_parity_remediation_attempt_lease_v1"
RECEIPT_SCHEMA = "axon_loop28_parity_remediation_receipt_v1"
DEFAULT_PROPOSAL = Path("manifests/roadmap_9997/p0_loop28_parity_remediation/proposal.json")
DEFAULT_AUTHORIZATION = Path(
    "manifests/roadmap_9997/p0_loop28_parity_remediation/authorization.json"
)
DEFAULT_IMPLEMENTATION_MANIFEST = implementation_manifest.DEFAULT_OUTPUT
DEFAULT_RUN_AUTHORIZATION = Path(
    "manifests/roadmap_9997/p0_loop28_parity_remediation/run_authorization.json"
)
DEFAULT_ATTEMPT_LEASE = Path(
    "manifests/roadmap_9997/p0_loop28_parity_remediation/run_attempt.final.json"
)
DEFAULT_OUTPUT = Path(
    "reports/roadmap_9997/p0_loop28_parity_remediation/remediation_receipt.final.json"
)
FIXED_ATTEMPT_ID = "p0_loop28_parity_remediation_001_final_attempt_001"
FIXED_SAMPLE_SHA256 = diagnostic.FIXED_SAMPLE_SHA256
FIXED_SAMPLE_INDEX = diagnostic.FIXED_SAMPLE_INDEX
FIXED_SAMPLE_SIZE_BYTES = diagnostic.FIXED_SAMPLE_SIZE_BYTES
FIXED_SPLIT = diagnostic.FIXED_SPLIT
FIXED_LOGICAL_RAW_ROOT = diagnostic.FIXED_LOGICAL_RAW_ROOT
FIXED_RESOLVED_RAW_ROOTS = diagnostic.FIXED_RESOLVED_RAW_ROOTS
FIXED_TOLERANCE = replay.DEFAULT_TOLERANCE

PARENT_EVIDENCE_PATHS = {
    "post_diagnostic_manifest": Path(
        "manifests/roadmap_9997/p0_loop28_parity_diagnostic/post_diagnostic_manifest.json"
    ),
    "diagnostic_receipt": Path(
        "reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.final.json"
    ),
    "historical_truth_manifest": replay.DEFAULT_TRUTH_MANIFEST,
}
PARENT_EVIDENCE_SHA256 = {
    "post_diagnostic_manifest": "9f57dee431d61a1a1ebc99f64ed9bcb9f65804fca8426c748d51b468e81e6d31",
    "diagnostic_receipt": "de8f0c5885df08646298f67f59b5427696252f7cb921d84eeb6527bef6878bc7",
    "historical_truth_manifest": "174861be850a681025a7040798c59b7157cc67ab5503437088359692dad5659d",
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
EXPECTED_TIMEOUT_ENFORCEMENT = diagnostic.EXPECTED_TIMEOUT_ENFORCEMENT
EXPECTED_SAMPLE = {
    "split": FIXED_SPLIT,
    "sample_index": FIXED_SAMPLE_INDEX,
    "source_sha256": FIXED_SAMPLE_SHA256,
    "size_bytes": FIXED_SAMPLE_SIZE_BYTES,
}


class RemediationContractError(ValueError):
    """Raised when the frozen remediation contract is violated."""


Clock = Callable[[], float]


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise RemediationContractError(f"Duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def _read_json_object(path: Path, schema: str) -> dict:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except FileNotFoundError as exc:
        raise RemediationContractError(f"Required JSON is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RemediationContractError(f"Invalid JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise RemediationContractError(f"Unsupported JSON contract: {path}")
    return payload


def _resolve_frozen(project_root: Path, path: Path, *, purpose: str) -> Path:
    return replay.resolve_within(project_root, path, purpose=purpose)


def _verify_fixed_sha(project_root: Path, relative_path: Path, expected_sha256: str) -> str:
    path = _resolve_frozen(project_root, relative_path, purpose=f"Frozen artifact {relative_path}")
    if not path.is_file():
        raise RemediationContractError(f"Frozen artifact is missing: {relative_path}")
    actual_sha256 = replay.file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise RemediationContractError(f"Frozen artifact SHA-256 mismatch: {relative_path}")
    return actual_sha256


def _verify_path_sha_record(
    project_root: Path,
    record: object,
    *,
    expected_path: Path,
    field: str,
) -> str:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise RemediationContractError(f"{field} record is invalid")
    declared_path = _resolve_frozen(
        project_root,
        Path(str(record.get("path") or "")),
        purpose=f"Declared {field}",
    )
    frozen_path = _resolve_frozen(project_root, expected_path, purpose=f"Frozen {field}")
    declared_sha256 = str(record.get("sha256") or "").strip().casefold()
    if declared_path != frozen_path or not _is_sha256(declared_sha256):
        raise RemediationContractError(f"{field} binding drifted")
    if not frozen_path.is_file() or replay.file_sha256(frozen_path) != declared_sha256:
        raise RemediationContractError(f"{field} SHA-256 mismatch")
    return declared_sha256


def verify_remediation_authorization(project_root: Path) -> dict:
    """Verify the scoped change authorization without opening model or raw payloads."""

    authorization_path = _resolve_frozen(
        project_root,
        DEFAULT_AUTHORIZATION,
        purpose="Loop28 remediation authorization",
    )
    payload = _read_json_object(authorization_path, AUTHORIZATION_SCHEMA)
    expected_scalars = {
        "loop_id": LOOP_ID,
        "authorization_level": "A1_scoped_change",
        "allowed_splits": [FIXED_SPLIT],
        "allowed_logical_raw_root": FIXED_LOGICAL_RAW_ROOT,
        "allowed_resolved_raw_roots": FIXED_RESOLVED_RAW_ROOTS,
        "frozen_tolerance": FIXED_TOLERANCE,
        "budget": EXPECTED_BUDGET,
        "timeout_enforcement": EXPECTED_TIMEOUT_ENFORCEMENT,
        "execution_requires_separate_run_authorization": True,
        "success_gate": "feature_components_exact_and_probability_deltas_at_most_1e-6",
        "decision": (
            "allow_scoped_implementation_and_enumerated_synthetic_validation_"
            "train_run_blocked_pending_hash_manifest"
        ),
    }
    for field, expected in expected_scalars.items():
        if payload.get(field) != expected:
            raise RemediationContractError(f"Remediation authorization {field} drifted")
    if payload.get("frozen_sample") != EXPECTED_SAMPLE:
        raise RemediationContractError("Remediation authorization frozen sample drifted")

    proposal_sha256 = _verify_path_sha_record(
        project_root,
        payload.get("proposal"),
        expected_path=DEFAULT_PROPOSAL,
        field="proposal",
    )
    parent_evidence = payload.get("parent_evidence")
    if not isinstance(parent_evidence, dict) or set(parent_evidence) != set(PARENT_EVIDENCE_PATHS):
        raise RemediationContractError("Remediation authorization parent evidence drifted")
    verified_parent_sha256 = {}
    for name, relative_path in PARENT_EVIDENCE_PATHS.items():
        declared_sha256 = str(parent_evidence.get(name) or "").strip().casefold()
        expected_sha256 = PARENT_EVIDENCE_SHA256[name]
        if declared_sha256 != expected_sha256:
            raise RemediationContractError(f"Parent evidence declaration drifted: {name}")
        verified_parent_sha256[name] = _verify_fixed_sha(
            project_root,
            relative_path,
            expected_sha256,
        )
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "loop_id": LOOP_ID,
        "authorization_sha256": replay.file_sha256(authorization_path),
        "proposal_sha256": proposal_sha256,
        "parent_evidence_sha256": verified_parent_sha256,
        "budget": dict(EXPECTED_BUDGET),
        "frozen_tolerance": FIXED_TOLERANCE,
        "status": "authorized_contract_verified",
    }


def verify_run_authorization(
    project_root: Path,
    run_authorization_path: Path,
    output_path: Path,
    preregistration: Mapping[str, object],
) -> dict:
    """Verify the final manifest-bound authorization before consuming its lease."""

    frozen_run_path = _resolve_frozen(
        project_root,
        DEFAULT_RUN_AUTHORIZATION,
        purpose="Frozen Loop28 remediation run authorization",
    )
    requested_run_path = _resolve_frozen(
        project_root,
        run_authorization_path,
        purpose="Requested Loop28 remediation run authorization",
    )
    if requested_run_path != frozen_run_path:
        raise RemediationContractError("Run authorization path is not server-owned")
    payload = _read_json_object(requested_run_path, RUN_AUTHORIZATION_SCHEMA)
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
    if set(payload) != expected_fields:
        raise RemediationContractError("Run authorization fields drifted")
    expected_values = {
        "loop_id": LOOP_ID,
        "prereg_authorization_sha256": preregistration.get("authorization_sha256"),
        "proposal_sha256": preregistration.get("proposal_sha256"),
        "parent_evidence": preregistration.get("parent_evidence_sha256"),
        "frozen_sample": EXPECTED_SAMPLE,
        "budget": EXPECTED_BUDGET,
        "timeout_enforcement": EXPECTED_TIMEOUT_ENFORCEMENT,
        "frozen_tolerance": FIXED_TOLERANCE,
        "attempt_id": FIXED_ATTEMPT_ID,
        "attempt_lease_path": DEFAULT_ATTEMPT_LEASE.as_posix(),
        "generation": "final",
        "output_path": DEFAULT_OUTPUT.as_posix(),
        "decision": "allow_bounded_loop28_parity_remediation_run",
    }
    for field, expected in expected_values.items():
        if payload.get(field) != expected:
            raise RemediationContractError(f"Run authorization {field} drifted")
    if payload.get("claim_scope") != {
        "train_only": True,
        "raw_identity_count": 1,
        "heldout_access_allowed": False,
        "training_or_fitting_allowed": False,
        "quality_claim_allowed": False,
        "population_parity_claim_allowed": False,
        "certification_claim_allowed": False,
    }:
        raise RemediationContractError("Run authorization claim scope drifted")
    frozen_output = _resolve_frozen(project_root, DEFAULT_OUTPUT, purpose="Frozen receipt output")
    requested_output = _resolve_frozen(
        project_root, output_path, purpose="Requested receipt output"
    )
    if requested_output != frozen_output:
        raise RemediationContractError("Requested receipt output is not authorized")

    manifest_sha256 = _verify_path_sha_record(
        project_root,
        payload.get("implementation_manifest"),
        expected_path=DEFAULT_IMPLEMENTATION_MANIFEST,
        field="implementation_manifest",
    )
    try:
        manifest = implementation_manifest.verify_manifest(
            project_root,
            DEFAULT_IMPLEMENTATION_MANIFEST,
        )
    except (implementation_manifest.ManifestError, OSError, ValueError) as exc:
        raise RemediationContractError(
            f"Remediation implementation manifest verification failed: {exc}"
        ) from exc
    integrity = manifest.get("integrity")
    if (
        not isinstance(integrity, dict)
        or integrity.get("blockers") != []
        or integrity.get("artifact_count") != integrity.get("required_artifact_count")
        or integrity.get("artifact_count") != integrity.get("present_required_artifact_count")
    ):
        raise RemediationContractError("Remediation implementation closure is incomplete")
    return {
        "schema": RUN_AUTHORIZATION_SCHEMA,
        "loop_id": LOOP_ID,
        "authorization_sha256": replay.file_sha256(requested_run_path),
        "prereg_authorization_sha256": preregistration["authorization_sha256"],
        "proposal_sha256": preregistration["proposal_sha256"],
        "parent_evidence_sha256": dict(preregistration["parent_evidence_sha256"]),
        "implementation_manifest_sha256": manifest_sha256,
        "implementation_artifact_count": integrity["artifact_count"],
        "generation": "final",
        "attempt_id": FIXED_ATTEMPT_ID,
        "attempt_lease_path": DEFAULT_ATTEMPT_LEASE.as_posix(),
        "status": "bounded_run_authorized",
    }


def _consume_attempt_lease(project_root: Path, run_authorization: Mapping[str, object]) -> dict:
    lease_path = _resolve_frozen(
        project_root,
        DEFAULT_ATTEMPT_LEASE,
        purpose="Remediation one-shot attempt lease",
    )
    payload = {
        "schema": ATTEMPT_LEASE_SCHEMA,
        "loop_id": LOOP_ID,
        "attempt_id": FIXED_ATTEMPT_ID,
        "generation": "final",
        "run_authorization_sha256": run_authorization["authorization_sha256"],
        "output_path": DEFAULT_OUTPUT.as_posix(),
        "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "authorization_consumed_before_raw_access",
    }
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lease_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RemediationContractError(
            "Remediation run authorization was already consumed"
        ) from exc
    return {
        "path": DEFAULT_ATTEMPT_LEASE.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "payload": payload,
    }


def _verify_attempt_lease(project_root: Path, expected: Mapping[str, object]) -> dict:
    lease_path = _resolve_frozen(
        project_root,
        DEFAULT_ATTEMPT_LEASE,
        purpose="Remediation one-shot attempt lease",
    )
    try:
        payload_bytes = lease_path.read_bytes()
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemediationContractError(
            "Remediation attempt lease is unavailable or invalid"
        ) from exc
    if payload != expected.get("payload"):
        raise RemediationContractError("Remediation attempt lease changed after consumption")
    sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if sha256 != expected.get("sha256"):
        raise RemediationContractError("Remediation attempt lease SHA-256 drifted")
    return {
        "path": DEFAULT_ATTEMPT_LEASE.as_posix(),
        "sha256": sha256,
        "consumed_at_utc": payload["consumed_at_utc"],
        "status": payload["status"],
    }


def evaluate_success_gate(result: Mapping[str, object]) -> dict:
    """Apply the preregistered parity gate to an authenticated comparison result."""

    component_rows = result.get("component_results")
    if not isinstance(component_rows, list):
        raise RemediationContractError("Diagnostic component results are missing")
    by_name = {
        str(row.get("name")): row
        for row in component_rows
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }
    if set(by_name) != set(diagnostic.EXPECTED_COMPONENT_DTYPES):
        raise RemediationContractError("Diagnostic component set drifted")
    expected_counts = {
        "byte_seq": 8192,
        "pe_features": 256,
        "stat_features": 49,
        "base_logits": 2,
        "base_probabilities": 2,
        "stage2_features": 1520,
    }
    for name, expected_count in expected_counts.items():
        if by_name[name].get("element_count") != expected_count:
            raise RemediationContractError(f"Diagnostic component shape drifted: {name}")

    stage2_row = by_name["stage2_features"]
    mismatch_indices = stage2_row.get("mismatch_indices")
    if not isinstance(mismatch_indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= 1520
        for index in mismatch_indices
    ):
        raise RemediationContractError("Stage-2 mismatch index contract drifted")
    if stage2_row.get("whole_match") is False and stage2_row.get("mismatch_count") != len(
        mismatch_indices
    ):
        raise RemediationContractError("Stage-2 mismatch accounting drifted")

    predictions = result.get("predictions")
    if not isinstance(predictions, Mapping):
        raise RemediationContractError("Diagnostic predictions are missing")
    deltas = predictions.get("absolute_probability_deltas")
    if not isinstance(deltas, Mapping):
        raise RemediationContractError("Diagnostic probability deltas are missing")
    required_delta_fields = {
        "prob_benign",
        "prob_malicious",
        "base_prob_benign",
        "base_prob_malicious",
        "stage2_prob_malicious",
    }
    if set(deltas) != required_delta_fields:
        raise RemediationContractError("Diagnostic probability delta fields drifted")
    numeric_deltas = {}
    for field in required_delta_fields:
        value = float(deltas[field])
        if not math.isfinite(value) or value < 0.0:
            raise RemediationContractError(f"Diagnostic probability delta is invalid: {field}")
        numeric_deltas[field] = value

    feature_exact = {
        name: by_name[name].get("whole_match") is True
        for name in ("byte_seq", "pe_features", "stat_features")
    }
    suffix_exact = all(index < 6 for index in mismatch_indices)
    base_probability_delta = max(
        numeric_deltas["base_prob_benign"],
        numeric_deltas["base_prob_malicious"],
    )
    stage2_probability_delta = numeric_deltas["stage2_prob_malicious"]
    final_probability_delta = max(
        numeric_deltas["prob_benign"],
        numeric_deltas["prob_malicious"],
    )
    checks = {
        **feature_exact,
        "stage2_features_6_through_1519_exact": suffix_exact,
        "base_decision_match": predictions.get("base_decision_match") is True,
        "final_decision_match": predictions.get("decision_match") is True,
        "base_probability_within_tolerance": base_probability_delta <= FIXED_TOLERANCE,
        "stage2_probability_within_tolerance": stage2_probability_delta <= FIXED_TOLERANCE,
        "final_probability_within_tolerance": final_probability_delta <= FIXED_TOLERANCE,
    }
    passed = all(checks.values())
    return {
        "tolerance": FIXED_TOLERANCE,
        "checks": checks,
        "stage2_probability_transform_mismatch_indices": mismatch_indices,
        "base_probability_max_absolute_delta": base_probability_delta,
        "stage2_probability_absolute_delta": stage2_probability_delta,
        "final_probability_max_absolute_delta": final_probability_delta,
        "passed": passed,
        "decision": (
            "frozen_train_remediation_gate_passed"
            if passed
            else "frozen_train_remediation_gate_failed"
        ),
    }


def _build_budget_audit(
    *,
    snapshot_duration: float,
    python_duration: float,
    diagnostic_result: Mapping[str, object],
    total_duration: float,
    native_output_sizes: Sequence[int],
    require_native_output_accounting: bool,
) -> dict:
    counts = diagnostic_result["execution_counts"]
    durations = diagnostic_result["execution_durations_seconds"]
    native_durations = list(durations["native"])
    crossfeed_durations = list(durations["crossfeed"])
    output_sizes = list(native_output_sizes)
    timeout = float(EXPECTED_BUDGET["per_execution_timeout_seconds"])
    total_limit = float(EXPECTED_BUDGET["total_wall_clock_seconds"])
    output_limit = int(EXPECTED_BUDGET["max_output_bytes"])
    rows = {
        "generation": {
            "count": 1,
            "limit": EXPECTED_BUDGET["max_remediation_generations"],
            "duration_seconds": total_duration,
            "within_budget": total_duration <= total_limit,
        },
        "verified_raw_snapshot": {
            "count": 1,
            "limit": EXPECTED_BUDGET["max_verified_raw_snapshots"],
            "durations_seconds": [snapshot_duration],
            "per_execution_limit_seconds": timeout,
            "enforcement": EXPECTED_TIMEOUT_ENFORCEMENT["verified_snapshot"],
            "within_budget": snapshot_duration <= timeout,
        },
        "python": {
            "count": counts["python"],
            "limit": EXPECTED_BUDGET["max_python_executions"],
            "durations_seconds": [python_duration],
            "per_execution_limit_seconds": timeout,
            "enforcement": EXPECTED_TIMEOUT_ENFORCEMENT["python_inference"],
            "within_budget": counts["python"] <= EXPECTED_BUDGET["max_python_executions"]
            and python_duration <= timeout,
        },
        "native": {
            "count": counts["native"],
            "limit": EXPECTED_BUDGET["max_native_executions"],
            "durations_seconds": native_durations,
            "per_execution_limit_seconds": timeout,
            "enforcement": EXPECTED_TIMEOUT_ENFORCEMENT["native_subprocess"],
            "within_budget": counts["native"] <= EXPECTED_BUDGET["max_native_executions"]
            and len(native_durations) == counts["native"]
            and all(duration <= timeout for duration in native_durations),
        },
        "crossfeed": {
            "count": counts["crossfeed"],
            "limit": EXPECTED_BUDGET["max_crossfeed_executions"],
            "durations_seconds": crossfeed_durations,
            "per_execution_limit_seconds": timeout,
            "within_budget": counts["crossfeed"] <= EXPECTED_BUDGET["max_crossfeed_executions"]
            and len(crossfeed_durations) == counts["crossfeed"]
            and all(duration <= timeout for duration in crossfeed_durations),
        },
        "total_wall_clock": {
            "count": 1,
            "limit": 1,
            "duration_seconds": total_duration,
            "limit_seconds": total_limit,
            "enforcement": EXPECTED_TIMEOUT_ENFORCEMENT["total_flow"],
            "within_budget": total_duration <= total_limit,
        },
        "output": {
            "native_execution_bytes": output_sizes,
            "native_total_bytes": sum(output_sizes),
            "receipt_limit_bytes": output_limit,
            "native_total_limit_bytes": output_limit,
            "within_budget": (
                (not require_native_output_accounting or len(output_sizes) == counts["native"])
                and all(size >= 0 for size in output_sizes)
                and sum(output_sizes) <= output_limit
            ),
        },
    }
    rows["within_budget"] = all(row["within_budget"] for row in rows.values())
    return rows


def _enforce_budget(audit: Mapping[str, object]) -> None:
    if audit.get("within_budget") is not True:
        raise RemediationContractError("Remediation execution exceeded its authorization budget")


def _assert_fixed_sample(sample: replay.SampleIdentity) -> None:
    if (
        sample.source_sha256 != FIXED_SAMPLE_SHA256
        or sample.sample_index != FIXED_SAMPLE_INDEX
        or sample.split != FIXED_SPLIT
    ):
        raise RemediationContractError("Split row does not match the frozen remediation sample")


def run_remediation(
    project_root: Path,
    *,
    run_authorization_path: Path,
    output_path: Path,
    python_trace_builder: Optional[Callable[..., diagnostic.PythonDiagnosticTrace]] = None,
    native_runner_factory: Optional[Callable[..., diagnostic.NativeRunner]] = None,
    key_factory: Callable[[int], bytes] = secrets.token_bytes,
    clock: Optional[Clock] = None,
    flow_started: Optional[float] = None,
) -> dict:
    project_root = project_root.resolve()
    clock = clock or time.monotonic
    flow_started = clock() if flow_started is None else flow_started
    total_limit = float(EXPECTED_BUDGET["total_wall_clock_seconds"])

    def enforce_total(scope: str) -> float:
        duration = diagnostic._duration_since(flow_started, clock)
        diagnostic._enforce_duration(duration, total_limit, scope=f"total flow after {scope}")
        return duration

    # 新 lease 必须在任何 raw、checkpoint 或 pickle 读取前完成原子消费。
    preregistration = verify_remediation_authorization(project_root)
    run_authorization = verify_run_authorization(
        project_root,
        run_authorization_path,
        output_path,
        preregistration,
    )
    authorized_output = _resolve_frozen(project_root, output_path, purpose="Authorized receipt")
    if authorized_output.exists():
        raise RemediationContractError("Authorized remediation receipt already exists")
    attempt_lease = _consume_attempt_lease(project_root, run_authorization)
    enforce_total("authorization")

    split_csv = _resolve_frozen(project_root, replay.DEFAULT_SPLIT_CSV, purpose="Frozen split CSV")
    samples, identity_audit = replay.read_split_samples(
        split_csv,
        requested_split=FIXED_SPLIT,
        max_samples=1,
    )
    sample = samples[0]
    _assert_fixed_sample(sample)
    enforce_total("split identity audit")

    checkpoint_path = _resolve_frozen(
        project_root,
        replay.DEFAULT_CHECKPOINT,
        purpose="Frozen Python checkpoint",
    )
    python_stage2_path = _resolve_frozen(
        project_root,
        replay.DEFAULT_PYTHON_STAGE2,
        purpose="Frozen Python Stage-2",
    )
    pickle_guard = replay.guard_pickle_before_load(
        project_root,
        python_stage2_path,
        replay.DEFAULT_PICKLE_ALLOWLIST,
    )
    native_paths = {
        "selftest": _resolve_frozen(
            project_root,
            replay.DEFAULT_NATIVE_SELFTEST,
            purpose="Frozen native selftest",
        ),
        "dll": _resolve_frozen(
            project_root,
            replay.DEFAULT_NATIVE_DLL,
            purpose="Frozen native DLL",
        ),
        "onnx": _resolve_frozen(
            project_root,
            replay.DEFAULT_NATIVE_ONNX,
            purpose="Frozen native ONNX",
        ),
        "stage2": _resolve_frozen(
            project_root,
            replay.DEFAULT_NATIVE_STAGE2,
            purpose="Frozen native Stage-2",
        ),
    }
    for name, path in {"checkpoint": checkpoint_path, **native_paths}.items():
        if not path.is_file():
            raise RemediationContractError(f"Frozen remediation artifact is missing: {name}")
    enforce_total("artifact guards")

    logical_root = _resolve_frozen(project_root, Path("data"), purpose="Frozen logical raw root")
    raw_authorization = replay.verify_a1_authorization(
        project_root,
        mode="native-parity",
        max_samples=1,
        allowed_raw_root=logical_root,
    )
    allowed_resolved_roots = [
        Path(path) for path in raw_authorization["allowed_resolved_raw_roots"]
    ]
    enforce_total("raw authorization")

    python_trace_builder = python_trace_builder or diagnostic.build_python_trace
    require_native_output_accounting = native_runner_factory is None
    native_output_sizes: list[int] = []
    timeout_seconds = int(EXPECTED_BUDGET["per_execution_timeout_seconds"])
    max_output_bytes = int(EXPECTED_BUDGET["max_output_bytes"])
    with tempfile.TemporaryDirectory(prefix="axon-loop28-parity-remediation-") as temp_dir:
        snapshot_root = Path(temp_dir)
        snapshot_started = clock()
        sample_path, snapshot_record = replay.snapshot_verified_sample(
            sample,
            allowed_raw_root=logical_root,
            allowed_resolved_roots=allowed_resolved_roots,
            snapshot_root=snapshot_root,
        )
        if snapshot_record["size_bytes"] != FIXED_SAMPLE_SIZE_BYTES:
            raise RemediationContractError("Frozen remediation sample size mismatch")
        snapshot_duration = diagnostic._duration_since(snapshot_started, clock)
        diagnostic._enforce_duration(
            snapshot_duration,
            timeout_seconds,
            scope="verified raw snapshot",
        )
        enforce_total("verified raw snapshot")

        python_started = clock()
        trace = python_trace_builder(
            project_root=project_root,
            sample_path=sample_path,
            checkpoint_path=checkpoint_path,
            stage2_path=python_stage2_path,
        )
        python_duration = diagnostic._duration_since(python_started, clock)
        diagnostic._enforce_duration(
            python_duration,
            timeout_seconds,
            scope="Python execution",
        )
        enforce_total("Python execution")

        if native_runner_factory is None:

            def native_runner(
                key: bytes,
                component: Optional[str],
                block_elements: Optional[int],
            ) -> Mapping[str, object]:
                return diagnostic.run_native_diagnostics(
                    sample_path=sample_path,
                    allowed_raw_root=snapshot_root,
                    selftest_path=native_paths["selftest"],
                    dll_path=native_paths["dll"],
                    onnx_path=native_paths["onnx"],
                    stage2_path=native_paths["stage2"],
                    key=key,
                    timeout_seconds=timeout_seconds,
                    max_output_bytes=max_output_bytes,
                    output_sizes=native_output_sizes,
                    component=component,
                    block_elements=block_elements,
                )

        else:
            native_runner = native_runner_factory(
                sample_path=sample_path,
                allowed_raw_root=snapshot_root,
            )
        diagnostic_result = diagnostic.diagnose_trace(
            trace,
            native_runner=native_runner,
            key_factory=key_factory,
            clock=clock,
            per_execution_limit_seconds=timeout_seconds,
        )
        total_duration = enforce_total("native diagnostics")

    budget_audit = _build_budget_audit(
        snapshot_duration=snapshot_duration,
        python_duration=python_duration,
        diagnostic_result=diagnostic_result,
        total_duration=total_duration,
        native_output_sizes=native_output_sizes,
        require_native_output_accounting=require_native_output_accounting,
    )
    _enforce_budget(budget_audit)
    success_gate = evaluate_success_gate(diagnostic_result)
    return {
        "schema": RECEIPT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": {
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
        },
        "generation": "final",
        "authorization": {
            "preregistration": preregistration,
            "run": run_authorization,
            "attempt_lease": _verify_attempt_lease(project_root, attempt_lease),
        },
        "sample_identity": {
            **EXPECTED_SAMPLE,
            "label": sample.label,
        },
        "evidence_sha256": {
            **preregistration["parent_evidence_sha256"],
            "implementation_manifest": run_authorization["implementation_manifest_sha256"],
            "split_csv": replay.file_sha256(split_csv),
            "checkpoint": replay.file_sha256(checkpoint_path),
            "python_stage2": pickle_guard["model_sha256"],
            "python_stage2_metadata": pickle_guard["metadata_sha256"],
            "native_selftest": replay.file_sha256(native_paths["selftest"]),
            "native_dll": replay.file_sha256(native_paths["dll"]),
            "native_onnx": replay.file_sha256(native_paths["onnx"]),
            "native_stage2": replay.file_sha256(native_paths["stage2"]),
        },
        "identity_audit": {
            "scope": "complete_split_metadata_identity_audit_only",
            "raw_files_opened": 1,
            "heldout_raw_files_opened": 0,
            "prediction_or_metric_rows_read": 0,
            "rows_scanned": identity_audit["rows_scanned"],
            "split_metadata_counts": identity_audit["split_counts"],
            "selected_count": identity_audit["selected_count"],
            "reported_count": 1,
        },
        "budget_audit": budget_audit,
        "authenticated_comparison": diagnostic_result,
        "success_gate": success_gate,
        "decision": success_gate["decision"],
    }


def _write_receipt_exclusive(path: Path, receipt: Mapping[str, object]) -> None:
    encoded = (json.dumps(receipt, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if len(encoded) > int(EXPECTED_BUDGET["max_output_bytes"]):
        raise RemediationContractError("Remediation receipt exceeded its output budget")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RemediationContractError("Authorized remediation receipt already exists") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--run-authorization", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    clock = time.monotonic
    flow_started = clock()
    try:
        output_path = _resolve_frozen(
            project_root,
            args.output_json,
            purpose="Remediation output receipt",
        )
        receipt = run_remediation(
            project_root,
            run_authorization_path=args.run_authorization,
            output_path=output_path,
            clock=clock,
            flow_started=flow_started,
        )
        # 写盘前再次核验授权、manifest 与 lease，拒绝执行期间的闭包漂移。
        final_preregistration = verify_remediation_authorization(project_root)
        final_run_authorization = verify_run_authorization(
            project_root,
            args.run_authorization,
            output_path,
            final_preregistration,
        )
        if (
            final_run_authorization["authorization_sha256"]
            != receipt["authorization"]["run"]["authorization_sha256"]
            or final_run_authorization["implementation_manifest_sha256"]
            != receipt["evidence_sha256"]["implementation_manifest"]
        ):
            raise RemediationContractError("Run authorization changed before receipt write")
        _verify_attempt_lease(
            project_root,
            {
                "payload": {
                    "schema": ATTEMPT_LEASE_SCHEMA,
                    "loop_id": LOOP_ID,
                    "attempt_id": FIXED_ATTEMPT_ID,
                    "generation": "final",
                    "run_authorization_sha256": final_run_authorization["authorization_sha256"],
                    "output_path": DEFAULT_OUTPUT.as_posix(),
                    "consumed_at_utc": receipt["authorization"]["attempt_lease"]["consumed_at_utc"],
                    "status": "authorization_consumed_before_raw_access",
                },
                "sha256": receipt["authorization"]["attempt_lease"]["sha256"],
            },
        )
        final_duration = diagnostic._duration_since(flow_started, clock)
        diagnostic._enforce_duration(
            final_duration,
            float(EXPECTED_BUDGET["total_wall_clock_seconds"]),
            scope="total flow before receipt write",
        )
        receipt["budget_audit"]["generation"]["duration_seconds"] = final_duration
        receipt["budget_audit"]["total_wall_clock"]["duration_seconds"] = final_duration
        receipt["budget_audit"]["generation"]["within_budget"] = True
        receipt["budget_audit"]["total_wall_clock"]["within_budget"] = True
        receipt["budget_audit"]["within_budget"] = True
        _enforce_budget(receipt["budget_audit"])
        _write_receipt_exclusive(output_path, receipt)
    except (
        RemediationContractError,
        diagnostic.DiagnosticContractError,
        replay.ReplayContractError,
    ) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": DEFAULT_OUTPUT.as_posix(),
                "decision": receipt["decision"],
                "gate_passed": receipt["success_gate"]["passed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if receipt["success_gate"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
