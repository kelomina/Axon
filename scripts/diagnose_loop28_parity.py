#!/usr/bin/env python3
"""Bounded, authenticated localization of Loop28 Python/native drift.

The diagnostic exchanges only ephemeral HMACs with the native self-test.  It
never writes tensor values, HMAC keys, or HMAC digests to the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import secrets
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import build_loop28_parity_diagnostic_manifest as implementation_manifest
import numpy as np
import replay_loop151_raw as replay

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION_SCHEMA = "axon_loop28_parity_diagnostic_authorization_v1"
RUN_AUTHORIZATION_SCHEMA = "axon_loop28_parity_diagnostic_run_authorization_v1"
ATTEMPT_LEASE_SCHEMA = "axon_loop28_parity_diagnostic_attempt_lease_v1"
RECEIPT_SCHEMA = "axon_loop28_parity_diagnostic_receipt_v1"
NATIVE_DIAGNOSTIC_SCHEMA = "axon_parity_diagnostics_v1"
TENSOR_ENCODING = "axon_tensor_le_v1"
TENSOR_MESSAGE_PREFIX = b"axon_tensor_le_v1\0"
FIXED_SAMPLE_SHA256 = "09b6b8c80bc31846312bd6958e1d4bf1bcd72d25450d7f7dec2bce6ba81798cc"
FIXED_SAMPLE_INDEX = 0
FIXED_SAMPLE_SIZE_BYTES = 4_218_880
FIXED_SPLIT = "train"
FIXED_LOGICAL_RAW_ROOT = "data"
FIXED_RESOLVED_RAW_ROOTS = [
    r"E:\Project\python\KoloVirusDetector_ML_V2-main\benign_samples\待加入白名单",
    r"E:\Project\python\KoloVirusDetector_ML_V2-main\malicious_samples\待拉黑",
]

DEFAULT_AUTHORIZATION = Path(
    "manifests/roadmap_9997/p0_loop28_parity_diagnostic/authorization.json"
)
DEFAULT_RUN_AUTHORIZATION = Path(
    "manifests/roadmap_9997/p0_loop28_parity_diagnostic/run_authorization.json"
)
DEFAULT_ATTEMPT_LEASE = Path(
    "manifests/roadmap_9997/p0_loop28_parity_diagnostic/run_attempt.final.json"
)
FIXED_ATTEMPT_ID = "p0_loop28_parity_diagnostic_001_final_attempt_001"
DEFAULT_IMPLEMENTATION_MANIFEST = implementation_manifest.DEFAULT_OUTPUT
OUTPUT_PATHS_BY_GENERATION = {
    "final": Path("reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.final.json"),
}
DEFAULT_TRUTH_MANIFEST = replay.DEFAULT_TRUTH_MANIFEST
DEFAULT_SPLIT_CSV = replay.DEFAULT_SPLIT_CSV
DEFAULT_CHECKPOINT = replay.DEFAULT_CHECKPOINT
DEFAULT_PYTHON_STAGE2 = replay.DEFAULT_PYTHON_STAGE2
DEFAULT_PICKLE_ALLOWLIST = replay.DEFAULT_PICKLE_ALLOWLIST
DEFAULT_NATIVE_SELFTEST = replay.DEFAULT_NATIVE_SELFTEST
DEFAULT_NATIVE_DLL = replay.DEFAULT_NATIVE_DLL
DEFAULT_NATIVE_ONNX = replay.DEFAULT_NATIVE_ONNX
DEFAULT_NATIVE_STAGE2 = replay.DEFAULT_NATIVE_STAGE2

PARENT_EVIDENCE_PATHS = {
    "truth_manifest": DEFAULT_TRUTH_MANIFEST,
    "train_smoke_receipt": Path("manifests/roadmap_9997/p0_raw_replay/train_smoke_receipt.json"),
    "native_parity_receipt": Path(
        "manifests/roadmap_9997/p0_raw_replay/native_parity_receipt.json"
    ),
    "complete_replay_verify_receipt": Path(
        "manifests/roadmap_9997/p0_raw_replay/verify_receipt.json"
    ),
}
IMPLEMENTATION_ARTIFACT_PATHS = {
    "python_diagnostic": Path("scripts/diagnose_loop28_parity.py"),
    "native_runtime_source": Path("tools/axon_onnx_dll/src/axon_onnx_predict.cpp"),
    "native_public_header": Path("tools/axon_onnx_dll/include/axon_onnx_predict.h"),
    "native_selftest_source": Path("tools/axon_onnx_dll/examples/axon_onnx_selftest.cpp"),
    "native_cmake": Path("tools/axon_onnx_dll/CMakeLists.txt"),
    "native_dll": DEFAULT_NATIVE_DLL,
    "native_selftest": DEFAULT_NATIVE_SELFTEST,
}
RUNTIME_ARTIFACT_PATHS = {
    "python_checkpoint": DEFAULT_CHECKPOINT,
    "python_stage2": DEFAULT_PYTHON_STAGE2,
    "python_stage2_metadata": replay.DEFAULT_PYTHON_STAGE2_METADATA,
    "pickle_allowlist": DEFAULT_PICKLE_ALLOWLIST,
    "native_onnx": DEFAULT_NATIVE_ONNX,
    "native_onnx_data": Path("models/random_20w_8192/axon_loop28_base.onnx.data"),
    "native_stage2": DEFAULT_NATIVE_STAGE2,
}
EXPECTED_BUDGET = {
    "max_diagnostic_generations": 1,
    "max_verified_raw_snapshots": 1,
    "max_python_executions": 6,
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
EXPECTED_COMPONENT_DTYPES = {
    "byte_seq": "i64le",
    "pe_features": "f32le",
    "stat_features": "f32le",
    "base_logits": "f32le",
    "base_probabilities": "f32le",
    "stage2_features": "f32le",
}
COMPONENT_ORDER = tuple(EXPECTED_COMPONENT_DTYPES)
DRILLDOWN_COMPONENTS = frozenset({"pe_features", "stat_features", "stage2_features"})


class DiagnosticContractError(ValueError):
    """Raised when the bounded diagnostic contract is violated."""


@dataclass(frozen=True)
class FeatureBoundary:
    name: str
    start: int
    count: int


@dataclass(frozen=True)
class PythonDiagnosticTrace:
    components: Mapping[str, np.ndarray]
    prediction: Mapping[str, object]
    stage2_boundaries: tuple[FeatureBoundary, ...] = ()


Clock = Callable[[], float]


def _duration_since(start: float, clock: Clock) -> float:
    duration = float(clock()) - float(start)
    if not math.isfinite(duration) or duration < 0.0:
        raise DiagnosticContractError("Diagnostic monotonic clock moved backwards")
    return duration


def _enforce_duration(duration: float, limit: float, *, scope: str) -> None:
    if not math.isfinite(duration) or duration < 0.0 or duration > limit:
        raise DiagnosticContractError(
            f"Diagnostic {scope} exceeded its {limit:.0f}s authorization limit"
        )


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _read_json_object(path: Path, schema: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise DiagnosticContractError(f"Required JSON is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DiagnosticContractError(f"Invalid JSON: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise DiagnosticContractError(f"Unsupported JSON contract: {path}")
    return payload


def _exact_int(value: object, expected: int, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise DiagnosticContractError(f"Authorization {field} drifted from {expected!r}")


def _resolve_frozen(project_root: Path, path: Path, *, purpose: str) -> Path:
    return replay.resolve_within(project_root, path, purpose=purpose)


def verify_diagnostic_authorization(project_root: Path) -> dict:
    """Validate the server-owned authorization and all immutable parent hashes."""

    authorization_path = _resolve_frozen(
        project_root,
        DEFAULT_AUTHORIZATION,
        purpose="Loop28 parity diagnostic authorization",
    )
    payload = _read_json_object(authorization_path, AUTHORIZATION_SCHEMA)
    if payload.get("loop_id") != "p0_loop28_parity_diagnostic_001":
        raise DiagnosticContractError("Diagnostic authorization loop_id mismatch")
    if payload.get("authorization_level") != "A1_scoped_diagnostic":
        raise DiagnosticContractError("Diagnostic authorization level drifted")
    if payload.get("allowed_splits") != [FIXED_SPLIT]:
        raise DiagnosticContractError("Diagnostic authorization split scope drifted")
    if payload.get("allowed_logical_raw_root") != FIXED_LOGICAL_RAW_ROOT:
        raise DiagnosticContractError("Diagnostic authorization logical root drifted")
    if payload.get("allowed_resolved_raw_roots") != FIXED_RESOLVED_RAW_ROOTS:
        raise DiagnosticContractError("Diagnostic authorization resolved roots drifted")
    if payload.get("execution_requires_separate_run_authorization") is not True:
        raise DiagnosticContractError("Diagnostic authorization execution gate drifted")
    if payload.get("success_gate") != "first_divergence_localized":
        raise DiagnosticContractError("Diagnostic authorization success gate drifted")
    if payload.get("decision") != (
        "allow_implementation_and_bounded_train_only_diagnostic_after_run_authorization"
    ):
        raise DiagnosticContractError("Diagnostic authorization decision drifted")
    tolerance = payload.get("frozen_tolerance")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise DiagnosticContractError("Diagnostic authorization tolerance is invalid")
    if float(tolerance) != replay.DEFAULT_TOLERANCE:
        raise DiagnosticContractError("Diagnostic authorization tolerance drifted")

    budget = payload.get("budget")
    if not isinstance(budget, dict) or set(budget) != set(EXPECTED_BUDGET):
        raise DiagnosticContractError("Diagnostic authorization budget fields drifted")
    for field, expected in EXPECTED_BUDGET.items():
        value = budget.get(field)
        if isinstance(expected, bool):
            if value is not expected:
                raise DiagnosticContractError(f"Authorization budget.{field} drifted")
        else:
            _exact_int(value, expected, field=f"budget.{field}")
    if payload.get("timeout_enforcement") != EXPECTED_TIMEOUT_ENFORCEMENT:
        raise DiagnosticContractError("Diagnostic authorization timeout semantics drifted")

    sample = payload.get("frozen_sample")
    expected_sample = {
        "split": FIXED_SPLIT,
        "sample_index": FIXED_SAMPLE_INDEX,
        "source_sha256": FIXED_SAMPLE_SHA256,
        "size_bytes": FIXED_SAMPLE_SIZE_BYTES,
    }
    if not isinstance(sample, dict) or set(sample) != set(expected_sample):
        raise DiagnosticContractError("Diagnostic authorization frozen_sample fields drifted")
    for field, expected in expected_sample.items():
        value = sample.get(field)
        if isinstance(expected, int):
            _exact_int(value, expected, field=f"frozen_sample.{field}")
        elif value != expected:
            raise DiagnosticContractError(f"Authorization frozen_sample.{field} drifted")

    parent_evidence = payload.get("parent_evidence")
    if not isinstance(parent_evidence, dict) or set(parent_evidence) != set(PARENT_EVIDENCE_PATHS):
        raise DiagnosticContractError("Diagnostic authorization parent evidence drifted")
    verified_parent_sha256: dict[str, str] = {}
    for name, frozen_relative_path in PARENT_EVIDENCE_PATHS.items():
        record = parent_evidence.get(name)
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise DiagnosticContractError(f"Parent evidence record is invalid: {name}")
        frozen_path = _resolve_frozen(
            project_root,
            frozen_relative_path,
            purpose=f"Frozen parent evidence {name}",
        )
        declared_path = _resolve_frozen(
            project_root,
            Path(str(record.get("path") or "")),
            purpose=f"Declared parent evidence {name}",
        )
        if declared_path != frozen_path:
            raise DiagnosticContractError(f"Parent evidence path drifted: {name}")
        expected_sha = str(record.get("sha256") or "").strip().casefold()
        if not _is_sha256(expected_sha) or not frozen_path.is_file():
            raise DiagnosticContractError(f"Parent evidence is unavailable: {name}")
        actual_sha = replay.file_sha256(frozen_path)
        if actual_sha != expected_sha:
            raise DiagnosticContractError(f"Parent evidence SHA-256 mismatch: {name}")
        verified_parent_sha256[name] = actual_sha

    return {
        "schema": AUTHORIZATION_SCHEMA,
        "loop_id": payload["loop_id"],
        "authorization_sha256": replay.file_sha256(authorization_path),
        "parent_evidence_sha256": verified_parent_sha256,
        "budget": dict(EXPECTED_BUDGET),
        "frozen_tolerance": replay.DEFAULT_TOLERANCE,
        "success_gate": payload["success_gate"],
        "status": "authorized_contract_verified",
    }


def _verify_bound_artifact_records(
    project_root: Path,
    value: object,
    frozen_paths: Mapping[str, Path],
    *,
    field: str,
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(frozen_paths):
        raise DiagnosticContractError(f"Run authorization {field} fields drifted")
    verified = {}
    for name, frozen_relative_path in frozen_paths.items():
        record = value.get(name)
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise DiagnosticContractError(f"Run authorization {field}.{name} is invalid")
        frozen_path = _resolve_frozen(
            project_root,
            frozen_relative_path,
            purpose=f"Frozen {field} {name}",
        )
        declared_path = _resolve_frozen(
            project_root,
            Path(str(record.get("path") or "")),
            purpose=f"Declared {field} {name}",
        )
        expected_sha = str(record.get("sha256") or "").strip().casefold()
        if declared_path != frozen_path or not _is_sha256(expected_sha):
            raise DiagnosticContractError(f"Run authorization {field}.{name} drifted")
        if not frozen_path.is_file() or replay.file_sha256(frozen_path) != expected_sha:
            raise DiagnosticContractError(f"Run authorization {field}.{name} SHA-256 mismatch")
        verified[name] = expected_sha
    return verified


def verify_run_authorization(
    project_root: Path,
    run_authorization_path: Path,
    requested_output_path: Path,
    preregistration: Mapping[str, object],
) -> dict:
    """Verify a separate, final SHA-bound run authorization before execution."""

    frozen_path = _resolve_frozen(
        project_root,
        DEFAULT_RUN_AUTHORIZATION,
        purpose="Frozen Loop28 parity run authorization",
    )
    requested_path = replay.resolve_within(
        project_root,
        run_authorization_path,
        purpose="Requested Loop28 parity run authorization",
    )
    if requested_path != frozen_path:
        raise DiagnosticContractError("Run authorization path is not server-owned")
    payload = _read_json_object(requested_path, RUN_AUTHORIZATION_SCHEMA)
    if payload.get("loop_id") != "p0_loop28_parity_diagnostic_001":
        raise DiagnosticContractError("Run authorization loop_id mismatch")
    if payload.get("decision") != "allow_bounded_loop28_parity_diagnostic_run":
        raise DiagnosticContractError("Run authorization does not allow execution")
    if payload.get("prereg_authorization_sha256") != preregistration.get("authorization_sha256"):
        raise DiagnosticContractError("Run authorization does not bind preregistration")
    if payload.get("parent_evidence") != preregistration.get("parent_evidence_sha256"):
        raise DiagnosticContractError("Run authorization parent evidence drifted")
    if payload.get("frozen_sample") != {
        "split": FIXED_SPLIT,
        "sample_index": FIXED_SAMPLE_INDEX,
        "source_sha256": FIXED_SAMPLE_SHA256,
        "size_bytes": FIXED_SAMPLE_SIZE_BYTES,
    }:
        raise DiagnosticContractError("Run authorization frozen sample drifted")
    if payload.get("budget") != EXPECTED_BUDGET:
        raise DiagnosticContractError("Run authorization budget drifted")
    if payload.get("timeout_enforcement") != EXPECTED_TIMEOUT_ENFORCEMENT:
        raise DiagnosticContractError("Run authorization timeout semantics drifted")
    if payload.get("frozen_tolerance") != replay.DEFAULT_TOLERANCE:
        raise DiagnosticContractError("Run authorization tolerance drifted")
    if payload.get("attempt_id") != FIXED_ATTEMPT_ID:
        raise DiagnosticContractError("Run authorization attempt_id drifted")
    if payload.get("attempt_lease_path") != DEFAULT_ATTEMPT_LEASE.as_posix():
        raise DiagnosticContractError("Run authorization attempt lease path drifted")
    generation = payload.get("generation")
    if generation not in OUTPUT_PATHS_BY_GENERATION:
        raise DiagnosticContractError("Run authorization generation is invalid")
    frozen_output_relative = OUTPUT_PATHS_BY_GENERATION[str(generation)]
    if payload.get("output_path") != frozen_output_relative.as_posix():
        raise DiagnosticContractError("Run authorization output path drifted")
    frozen_output = _resolve_frozen(
        project_root,
        frozen_output_relative,
        purpose="Frozen generation output receipt",
    )
    requested_output = replay.resolve_within(
        project_root,
        requested_output_path,
        purpose="Requested generation output receipt",
    )
    if requested_output != frozen_output:
        raise DiagnosticContractError("Requested output is not bound by run authorization")

    implementation_manifest_record = payload.get("implementation_manifest")
    if not isinstance(implementation_manifest_record, dict) or set(
        implementation_manifest_record
    ) != {"path", "sha256"}:
        raise DiagnosticContractError("Run authorization implementation manifest is invalid")
    declared_manifest_path = _resolve_frozen(
        project_root,
        Path(str(implementation_manifest_record.get("path") or "")),
        purpose="Declared implementation manifest",
    )
    frozen_manifest_path = _resolve_frozen(
        project_root,
        DEFAULT_IMPLEMENTATION_MANIFEST,
        purpose="Frozen implementation manifest",
    )
    declared_manifest_sha256 = (
        str(implementation_manifest_record.get("sha256") or "").strip().casefold()
    )
    if declared_manifest_path != frozen_manifest_path or not _is_sha256(declared_manifest_sha256):
        raise DiagnosticContractError("Run authorization implementation manifest drifted")
    try:
        manifest_verification = implementation_manifest.verify_implementation_manifest(
            project_root,
            DEFAULT_IMPLEMENTATION_MANIFEST,
        )
    except implementation_manifest.ManifestContractError as exc:
        raise DiagnosticContractError(
            f"Implementation manifest verification failed: {exc}"
        ) from exc
    if manifest_verification["implementation_manifest_sha256"] != declared_manifest_sha256:
        raise DiagnosticContractError("Run authorization implementation manifest SHA-256 mismatch")
    if manifest_verification["parent_evidence_sha256"] != preregistration.get(
        "parent_evidence_sha256"
    ):
        raise DiagnosticContractError("Implementation manifest parent evidence drifted")

    implementation_sha256 = _verify_bound_artifact_records(
        project_root,
        payload.get("implementation_artifacts"),
        IMPLEMENTATION_ARTIFACT_PATHS,
        field="implementation_artifacts",
    )
    runtime_sha256 = _verify_bound_artifact_records(
        project_root,
        payload.get("runtime_artifacts"),
        RUNTIME_ARTIFACT_PATHS,
        field="runtime_artifacts",
    )
    return {
        "schema": RUN_AUTHORIZATION_SCHEMA,
        "loop_id": payload["loop_id"],
        "authorization_sha256": replay.file_sha256(requested_path),
        "prereg_authorization_sha256": preregistration["authorization_sha256"],
        "parent_evidence_sha256": dict(preregistration["parent_evidence_sha256"]),
        "implementation_artifact_sha256": implementation_sha256,
        "runtime_artifact_sha256": runtime_sha256,
        "implementation_manifest": manifest_verification,
        "generation": generation,
        "attempt_id": FIXED_ATTEMPT_ID,
        "attempt_lease_path": DEFAULT_ATTEMPT_LEASE.as_posix(),
        "status": "bounded_run_authorized",
    }


def _consume_attempt_lease(project_root: Path, run_authorization: Mapping[str, object]) -> dict:
    lease_path = _resolve_frozen(
        project_root,
        DEFAULT_ATTEMPT_LEASE,
        purpose="Diagnostic one-shot attempt lease",
    )
    payload = {
        "schema": ATTEMPT_LEASE_SCHEMA,
        "loop_id": "p0_loop28_parity_diagnostic_001",
        "attempt_id": FIXED_ATTEMPT_ID,
        "generation": run_authorization["generation"],
        "run_authorization_sha256": run_authorization["authorization_sha256"],
        "output_path": OUTPUT_PATHS_BY_GENERATION[str(run_authorization["generation"])].as_posix(),
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
        raise DiagnosticContractError("Diagnostic run authorization was already consumed") from exc
    return {
        "path": DEFAULT_ATTEMPT_LEASE.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "payload": payload,
    }


def _verify_attempt_lease(project_root: Path, expected: Mapping[str, object]) -> dict:
    lease_path = _resolve_frozen(
        project_root,
        DEFAULT_ATTEMPT_LEASE,
        purpose="Diagnostic one-shot attempt lease",
    )
    try:
        payload_bytes = lease_path.read_bytes()
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticContractError("Diagnostic attempt lease is unavailable or invalid") from exc
    if payload != expected.get("payload"):
        raise DiagnosticContractError(
            "Diagnostic attempt lease changed after authorization consumption"
        )
    sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if sha256 != expected.get("sha256"):
        raise DiagnosticContractError("Diagnostic attempt lease SHA-256 drifted")
    return {
        "path": DEFAULT_ATTEMPT_LEASE.as_posix(),
        "sha256": sha256,
        "consumed_at_utc": payload["consumed_at_utc"],
        "status": payload["status"],
    }


def _canonical_array(array: np.ndarray, dtype: str) -> np.ndarray:
    numpy_dtype = {"i64le": np.dtype("<i8"), "f32le": np.dtype("<f4")}.get(dtype)
    if numpy_dtype is None:
        raise DiagnosticContractError(f"Unsupported diagnostic dtype: {dtype}")
    return np.ascontiguousarray(np.asarray(array, dtype=numpy_dtype)).reshape(-1)


def tensor_hmac(
    key: bytes,
    *,
    name: str,
    dtype: str,
    array: np.ndarray,
    start: int = 0,
    count: Optional[int] = None,
) -> str:
    """Return the frozen HMAC over one whole tensor or one contiguous block."""

    if not isinstance(key, bytes) or len(key) != 32:
        raise DiagnosticContractError("Diagnostic HMAC key must contain exactly 32 bytes")
    flat = _canonical_array(array, dtype)
    total = int(flat.size)
    count = total - start if count is None else count
    if start < 0 or count < 0 or start + count > total:
        raise DiagnosticContractError("Diagnostic tensor block is outside the tensor")
    message = b"".join(
        (
            TENSOR_MESSAGE_PREFIX,
            name.encode("utf-8"),
            b"\0",
            dtype.encode("ascii"),
            b"\0",
            struct.pack("<QQQ", total, start, count),
            flat[start : start + count].tobytes(order="C"),
        )
    )
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _valid_hmac(value: object) -> bool:
    return _is_sha256(value)


def _validate_diagnostic_envelope(
    prediction: Mapping[str, object],
    *,
    required_components: set[str],
) -> Mapping[str, object]:
    diagnostics = prediction.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise DiagnosticContractError("Native prediction has no diagnostics object")
    if diagnostics.get("schema") != NATIVE_DIAGNOSTIC_SCHEMA:
        raise DiagnosticContractError("Native diagnostics schema mismatch")
    if diagnostics.get("encoding") != TENSOR_ENCODING:
        raise DiagnosticContractError("Native diagnostics encoding mismatch")
    if diagnostics.get("digest") != "hmac-sha256":
        raise DiagnosticContractError("Native diagnostics digest algorithm mismatch")
    if set(diagnostics) != {"schema", "encoding", "digest", "components"}:
        raise DiagnosticContractError("Native diagnostics envelope has unexpected fields")
    components = diagnostics.get("components")
    if not isinstance(components, Mapping) or set(components) != required_components:
        raise DiagnosticContractError("Native diagnostics component set is not exact")
    unknown = set(components) - set(EXPECTED_COMPONENT_DTYPES)
    if unknown:
        raise DiagnosticContractError(f"Native diagnostics has unknown components: {unknown}")
    return components


def _validate_component_record(
    name: str,
    record: object,
    python_array: np.ndarray,
    *,
    require_blocks: bool,
) -> tuple[str, int]:
    if not isinstance(record, Mapping):
        raise DiagnosticContractError(f"Native component is not an object: {name}")
    expected_fields = {"dtype", "shape", "digest"}
    if require_blocks:
        expected_fields.add("blocks")
    if set(record) != expected_fields:
        raise DiagnosticContractError(f"Native component has unexpected fields: {name}")
    dtype = EXPECTED_COMPONENT_DTYPES[name]
    if record.get("dtype") != dtype or not _valid_hmac(record.get("digest")):
        raise DiagnosticContractError(f"Native component contract mismatch: {name}")
    shape = record.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape
        )
    ):
        raise DiagnosticContractError(f"Native component shape is invalid: {name}")
    expected_shape = list(np.asarray(python_array).shape)
    if shape != expected_shape:
        raise DiagnosticContractError(f"Native component shape mismatch: {name}")
    total = math.prod(shape)
    if total != int(np.asarray(python_array).size):
        raise DiagnosticContractError(f"Native component element count mismatch: {name}")
    return dtype, total


def compare_whole_trace(
    trace: PythonDiagnosticTrace,
    native_prediction: Mapping[str, object],
    key: bytes,
) -> list[dict]:
    if set(trace.components) != set(EXPECTED_COMPONENT_DTYPES):
        raise DiagnosticContractError("Python trace component set drifted")
    native_components = _validate_diagnostic_envelope(
        native_prediction,
        required_components=set(EXPECTED_COMPONENT_DTYPES),
    )
    results = []
    for name in COMPONENT_ORDER:
        array = trace.components[name]
        record = native_components[name]
        dtype, total = _validate_component_record(
            name,
            record,
            array,
            require_blocks=False,
        )
        expected = tensor_hmac(key, name=name, dtype=dtype, array=array)
        matched = hmac.compare_digest(expected, str(record["digest"]).casefold())
        results.append(
            {
                "name": name,
                "dtype": dtype,
                "element_count": total,
                "whole_match": matched,
                "drilldown_executions": 0,
                "match_count": total if matched else None,
                "mismatch_count": 0 if matched else None,
                "mismatch_indices": [],
                "max_matching_index": total - 1 if matched and total else None,
                "first_mismatch_index": None,
                "last_mismatch_index": None,
            }
        )
    return results


def localize_single_element_blocks(
    *,
    name: str,
    array: np.ndarray,
    native_prediction: Mapping[str, object],
    key: bytes,
) -> dict:
    if name not in DRILLDOWN_COMPONENTS:
        raise DiagnosticContractError(f"Component is not authorized for drilldown: {name}")
    components = _validate_diagnostic_envelope(
        native_prediction,
        required_components={name},
    )
    record = components[name]
    dtype, total = _validate_component_record(
        name,
        record,
        array,
        require_blocks=True,
    )
    if not isinstance(record, Mapping):  # narrowed by _validate_component_record
        raise DiagnosticContractError(f"Native component is not an object: {name}")
    blocks = record.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != total:
        raise DiagnosticContractError(f"Native block=1 coverage is incomplete: {name}")

    mismatches = []
    seen = set()
    for block in blocks:
        if not isinstance(block, Mapping):
            raise DiagnosticContractError(f"Native diagnostic block is invalid: {name}")
        start, count = block.get("start"), block.get("count")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
            or isinstance(count, bool)
            or count != 1
            or start >= total
            or start in seen
            or not _valid_hmac(block.get("digest"))
        ):
            raise DiagnosticContractError(f"Native diagnostic block contract mismatch: {name}")
        seen.add(start)
        expected = tensor_hmac(
            key,
            name=name,
            dtype=dtype,
            array=array,
            start=start,
            count=1,
        )
        if not hmac.compare_digest(expected, str(block["digest"]).casefold()):
            mismatches.append(start)
    if seen != set(range(total)):
        raise DiagnosticContractError(f"Native diagnostic blocks do not cover tensor: {name}")
    if not mismatches:
        raise DiagnosticContractError(
            f"Whole HMAC mismatch was not reproducible with block=1: {name}"
        )

    mismatch_set = set(mismatches)
    max_matching_index = next(
        (index for index in range(total - 1, -1, -1) if index not in mismatch_set),
        None,
    )
    contiguous_prefix = next((index for index in range(total) if index in mismatch_set), total)
    return {
        "drilldown_executions": 1,
        "match_count": total - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatch_indices": mismatches,
        "max_matching_index": max_matching_index,
        "max_contiguous_matching_prefix_count": contiguous_prefix,
        "first_mismatch_index": mismatches[0],
        "last_mismatch_index": mismatches[-1],
    }


def _parse_native_prediction(stdout: str) -> dict:
    candidates = []
    for line in stdout.splitlines():
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("diagnostics"), dict):
            candidates.append(value)
    if len(candidates) != 1:
        raise DiagnosticContractError("Native selftest emitted an ambiguous diagnostic result")
    payload = candidates[0]
    if set(payload) != {
        "ok",
        "prediction",
        "prob_benign",
        "prob_malicious",
        "base_model",
        "stage2",
        "diagnostics",
    }:
        raise DiagnosticContractError("Native diagnostic result has unexpected fields")
    base_model = payload.get("base_model")
    stage2 = payload.get("stage2")
    if not isinstance(base_model, dict) or set(base_model) != {
        "prediction",
        "prob_benign",
        "prob_malicious",
    }:
        raise DiagnosticContractError("Native base-model summary has unexpected fields")
    if not isinstance(stage2, dict) or set(stage2) != {"enabled", "prob_malicious"}:
        raise DiagnosticContractError("Native Stage-2 summary has unexpected fields")
    if payload.get("ok") is not True:
        raise DiagnosticContractError("Native diagnostic prediction failed")
    return payload


def run_native_diagnostics(
    *,
    sample_path: Path,
    allowed_raw_root: Path,
    selftest_path: Path,
    dll_path: Path,
    onnx_path: Path,
    stage2_path: Path,
    key: bytes,
    timeout_seconds: int,
    max_output_bytes: int,
    output_sizes: Optional[list[int]] = None,
    component: Optional[str] = None,
    block_elements: Optional[int] = None,
) -> dict:
    if not isinstance(key, bytes) or len(key) != 32:
        raise DiagnosticContractError("Diagnostic HMAC key must contain exactly 32 bytes")
    if (component is None) != (block_elements is None):
        raise DiagnosticContractError("Diagnostic drilldown flags must be supplied together")
    if component is not None and (component not in DRILLDOWN_COMPONENTS or block_elements != 1):
        raise DiagnosticContractError("Only one block=1 drilldown per feature component is allowed")
    remaining_output_bytes = max_output_bytes
    if output_sizes is not None:
        remaining_output_bytes -= sum(output_sizes)
    if remaining_output_bytes <= 0:
        raise DiagnosticContractError("Native diagnostics exhausted the aggregate output budget")
    command = [
        replay._windows_cli_path(selftest_path),
        "--dll",
        replay._windows_cli_path(dll_path),
        "--onnx",
        replay._windows_cli_path(onnx_path),
        "--target",
        replay._windows_cli_path(sample_path),
        "--allowed_root",
        replay._windows_cli_path(allowed_raw_root),
        "--stage2",
        replay._windows_cli_path(stage2_path),
        "--parity_diagnostics",
    ]
    if component is not None:
        command.extend(
            ["--diagnostic_component", component, "--block_elements", str(block_elements)]
        )
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        completed = subprocess.run(
            command,
            input=f"{key.hex()}\n".encode("ascii"),
            check=False,
            stdout=stdout_file,
            stderr=stderr_file,
            timeout=timeout_seconds,
        )
        stdout_file.seek(0, 2)
        stderr_file.seek(0, 2)
        output_size = stdout_file.tell() + stderr_file.tell()
        if output_size > remaining_output_bytes:
            raise DiagnosticContractError("Native diagnostics exceeded the aggregate output budget")
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read().decode("utf-8", errors="replace")
        stderr_file.read()
    if completed.returncode != 0:
        raise DiagnosticContractError(
            f"Native diagnostic failed with exit code {completed.returncode}"
        )
    if output_sizes is not None:
        output_sizes.append(output_size)
    return _parse_native_prediction(stdout)


def _prediction_summary(payload: Mapping[str, object]) -> dict:
    base = payload.get("base_model")
    stage2 = payload.get("stage2")
    if not isinstance(base, Mapping) or not isinstance(stage2, Mapping):
        raise DiagnosticContractError("Prediction summary is incomplete")
    try:
        summary = {
            "prediction": int(payload["prediction"]),
            "prob_benign": float(payload["prob_benign"]),
            "prob_malicious": float(payload["prob_malicious"]),
            "base_prediction": int(base["prediction"]),
            "base_prob_benign": float(base["prob_benign"]),
            "base_prob_malicious": float(base["prob_malicious"]),
            "stage2_prob_malicious": float(stage2["prob_malicious"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise DiagnosticContractError("Prediction summary has invalid values") from exc
    if summary["prediction"] not in (0, 1) or summary["base_prediction"] not in (0, 1):
        raise DiagnosticContractError("Prediction summary has a non-binary decision")
    for field, value in summary.items():
        if "prob_" in field and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
            raise DiagnosticContractError(f"Prediction probability is invalid: {field}")
    return summary


def _stage_boundary_summary(
    component_results: Sequence[Mapping[str, object]],
    *,
    stage2_inference_match: bool,
) -> dict:
    by_name = {str(result["name"]): result for result in component_results}
    stages = (
        ("feature_extraction", ("byte_seq", "pe_features", "stat_features")),
        ("base_inference", ("base_logits", "base_probabilities")),
        ("stage2_assembly", ("stage2_features",)),
    )
    stage_rows = []
    for name, components in stages:
        mismatches = [
            component for component in components if not by_name[component]["whole_match"]
        ]
        stage_rows.append(
            {
                "name": name,
                "component_count": len(components),
                "matched_component_count": len(components) - len(mismatches),
                "mismatched_components": mismatches,
                "whole_match": not mismatches,
            }
        )
    stage_rows.append(
        {
            "name": "stage2_inference",
            "component_count": 0,
            "matched_component_count": 0,
            "mismatched_components": [] if stage2_inference_match else ["stage2_probability"],
            "whole_match": stage2_inference_match,
        }
    )
    first_mismatch = next(
        (row["name"] for row in stage_rows if not row["whole_match"]),
        None,
    )
    maximum_confirmed = None
    for row in stage_rows:
        if not row["whole_match"]:
            break
        maximum_confirmed = row["name"]
    return {
        "stages": stage_rows,
        "first_mismatch_stage": first_mismatch,
        "maximum_contiguous_confirmed_stage": maximum_confirmed,
    }


def _stage2_mismatch_boundaries(
    boundaries: Sequence[FeatureBoundary],
    mismatch_indices: Sequence[int],
) -> list[dict]:
    rows = []
    mismatch_set = set(mismatch_indices)
    for boundary in boundaries:
        stop = boundary.start + boundary.count
        indices = sorted(index for index in mismatch_set if boundary.start <= index < stop)
        rows.append(
            {
                "name": boundary.name,
                "start": boundary.start,
                "count": boundary.count,
                "mismatch_count": len(indices),
                "first_mismatch_index": indices[0] if indices else None,
                "last_mismatch_index": indices[-1] if indices else None,
            }
        )
    return rows


NativeRunner = Callable[[bytes, Optional[str], Optional[int]], Mapping[str, object]]


def diagnose_trace(
    trace: PythonDiagnosticTrace,
    *,
    native_runner: NativeRunner,
    key_factory: Callable[[int], bytes] = secrets.token_bytes,
    clock: Optional[Clock] = None,
    per_execution_limit_seconds: float = 120.0,
) -> dict:
    clock = clock or time.monotonic
    native_durations = []

    def invoke_native(
        key: bytes,
        component: Optional[str],
        block_elements: Optional[int],
    ) -> Mapping[str, object]:
        started = clock()
        prediction = native_runner(key, component, block_elements)
        duration = _duration_since(started, clock)
        _enforce_duration(
            duration,
            per_execution_limit_seconds,
            scope="native execution",
        )
        native_durations.append(duration)
        return prediction

    whole_key = key_factory(32)
    if not isinstance(whole_key, bytes) or len(whole_key) != 32:
        raise DiagnosticContractError("Diagnostic key factory returned an invalid key")
    whole_prediction = invoke_native(whole_key, None, None)
    results = compare_whole_trace(trace, whole_prediction, whole_key)

    native_execution_count = 1
    crossfeed_execution_count = 1
    for result in results:
        name = str(result["name"])
        if result["whole_match"] or name not in DRILLDOWN_COMPONENTS:
            continue
        drill_key = key_factory(32)
        if not isinstance(drill_key, bytes) or len(drill_key) != 32:
            raise DiagnosticContractError("Diagnostic key factory returned an invalid key")
        drill_prediction = invoke_native(drill_key, name, 1)
        result.update(
            localize_single_element_blocks(
                name=name,
                array=trace.components[name],
                native_prediction=drill_prediction,
                key=drill_key,
            )
        )
        native_execution_count += 1
        crossfeed_execution_count += 1

    mismatched_components = [str(result["name"]) for result in results if not result["whole_match"]]
    stage2_result = next(result for result in results if result["name"] == "stage2_features")
    stage2_boundary_rows = _stage2_mismatch_boundaries(
        trace.stage2_boundaries,
        stage2_result["mismatch_indices"],
    )
    python_prediction = _prediction_summary(trace.prediction)
    native_prediction = _prediction_summary(whole_prediction)
    probability_fields = (
        "prob_benign",
        "prob_malicious",
        "base_prob_benign",
        "base_prob_malicious",
        "stage2_prob_malicious",
    )
    probability_deltas = {
        field: abs(python_prediction[field] - native_prediction[field])
        for field in probability_fields
    }
    decision_match = python_prediction["prediction"] == native_prediction["prediction"]
    base_decision_match = (
        python_prediction["base_prediction"] == native_prediction["base_prediction"]
    )
    stage2_probability_delta = probability_deltas["stage2_prob_malicious"]
    stage2_probability_within_tolerance = stage2_probability_delta <= replay.DEFAULT_TOLERANCE
    stage2_inference_match = stage2_probability_within_tolerance and decision_match
    stage_boundaries = _stage_boundary_summary(
        results,
        stage2_inference_match=stage2_inference_match,
    )
    divergence_detected = bool(mismatched_components) or not stage2_inference_match
    decision = (
        "no_divergence_observed_diagnostic_not_parity_evidence"
        if not divergence_detected
        else "first_divergence_localized"
    )
    return {
        "component_results": results,
        "matched_component_count": len(results) - len(mismatched_components),
        "mismatched_component_count": len(mismatched_components),
        "mismatched_components": mismatched_components,
        "stage_boundaries": stage_boundaries,
        "stage2_feature_boundaries": stage2_boundary_rows,
        "stage2_inference": {
            "tolerance": replay.DEFAULT_TOLERANCE,
            "absolute_probability_delta": stage2_probability_delta,
            "probability_within_tolerance": stage2_probability_within_tolerance,
            "decision_match": decision_match,
            "divergence": not stage2_inference_match,
        },
        "predictions": {
            "python": python_prediction,
            "native": native_prediction,
            "decision_match": decision_match,
            "base_decision_match": base_decision_match,
            "absolute_probability_deltas": probability_deltas,
        },
        "execution_counts": {
            "python": 1,
            "native": native_execution_count,
            "crossfeed": crossfeed_execution_count,
        },
        "execution_durations_seconds": {
            "native": native_durations,
            "crossfeed": list(native_durations),
        },
        "decision": decision,
    }


def _stage2_boundaries(features, feature_config, vector: np.ndarray) -> tuple[FeatureBoundary, ...]:
    boundaries = []
    cursor = 0

    def append(name: str, count: int) -> None:
        nonlocal cursor
        boundaries.append(FeatureBoundary(name, cursor, int(count)))
        cursor += int(count)

    append("base_probability_transforms", 6)
    if feature_config.include_stat:
        append("stat_features", np.asarray(features.stat_features).size)
    if feature_config.include_pe:
        append("pe_features", np.asarray(features.pe_features).size)
    if feature_config.include_lightweight:
        append("lightweight_features", np.asarray(features.lightweight_features).size)
    if feature_config.include_byte_summary:
        byte_summary_count = (
            512 + int(feature_config.prefix_len) + 5 * max(1, int(feature_config.chunk_count)) + 5
        )
        append("byte_summary", byte_summary_count)
    remaining = int(vector.size) - cursor
    if feature_config.include_content_pe:
        append("content_pe", remaining)
    elif remaining:
        raise DiagnosticContractError("Stage2 feature boundary accounting drifted")
    if cursor != int(vector.size):
        raise DiagnosticContractError("Stage2 feature boundaries do not cover the vector")
    return tuple(boundaries)


def build_python_trace(
    *,
    project_root: Path,
    sample_path: Path,
    checkpoint_path: Path,
    stage2_path: Path,
) -> PythonDiagnosticTrace:
    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    import torch  # noqa: PLC0415

    from predict_api import (  # noqa: PLC0415
        PredictRequest,
        _extract_features,
        _load_prediction_context,
        _stage2_feature_vector,
    )

    request = PredictRequest(
        file=str(sample_path),
        checkpoint=str(checkpoint_path),
        device="cpu",
        stage2_model=str(stage2_path),
        family_classifier="",
    )
    context = _load_prediction_context(request, checkpoint_path, "cpu")
    if context.stage2 is None:
        raise DiagnosticContractError("Frozen Python Stage2 model was not loaded")
    features = _extract_features(context.config, sample_path)
    if features is None:
        raise DiagnosticContractError("Python feature extraction failed")

    byte_tensor = torch.from_numpy(features.byte_seq).long().unsqueeze(0)
    pe_tensor = torch.from_numpy(features.pe_features).float().unsqueeze(0)
    stat_tensor = torch.from_numpy(features.stat_features).float().unsqueeze(0)
    with torch.no_grad():
        logits_tensor = context.model(byte_tensor, pe_tensor, stat_features=stat_tensor)["logits"]
        probabilities_tensor = torch.softmax(logits_tensor, dim=1)
    logits = logits_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
    probabilities = probabilities_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
    base_prediction = int(np.argmax(probabilities))
    stage2_vector = _stage2_feature_vector(
        sample_path,
        features,
        float(probabilities[1]),
        context.stage2.feature_config,
    ).astype(np.float32, copy=False)
    stage2_probability = context.stage2.predict_probability(stage2_vector)
    final_prediction = int(stage2_probability >= context.stage2.threshold)
    components = {
        "byte_seq": np.asarray(features.byte_seq, dtype=np.int64),
        "pe_features": np.asarray(features.pe_features, dtype=np.float32),
        "stat_features": np.asarray(features.stat_features, dtype=np.float32),
        "base_logits": logits,
        "base_probabilities": probabilities,
        "stage2_features": stage2_vector,
    }
    prediction = {
        "prediction": final_prediction,
        "prob_benign": 1.0 - stage2_probability,
        "prob_malicious": stage2_probability,
        "base_model": {
            "prediction": base_prediction,
            "prob_benign": float(probabilities[0]),
            "prob_malicious": float(probabilities[1]),
        },
        "stage2": {"prob_malicious": stage2_probability},
    }
    return PythonDiagnosticTrace(
        components=components,
        prediction=prediction,
        stage2_boundaries=_stage2_boundaries(
            features,
            context.stage2.feature_config,
            stage2_vector,
        ),
    )


def _assert_fixed_sample(sample: replay.SampleIdentity) -> None:
    if (
        sample.source_sha256 != FIXED_SAMPLE_SHA256
        or sample.sample_index != FIXED_SAMPLE_INDEX
        or sample.split != FIXED_SPLIT
    ):
        raise DiagnosticContractError("Split row does not match the frozen diagnostic sample")


def _build_budget_audit(
    *,
    snapshot_duration: float,
    python_duration: float,
    diagnostic: Mapping[str, object],
    total_duration: float,
    native_output_sizes: Sequence[int] = (),
    require_native_output_accounting: bool = False,
) -> dict:
    counts = diagnostic["execution_counts"]
    durations = diagnostic["execution_durations_seconds"]
    per_execution_limit = float(EXPECTED_BUDGET["per_execution_timeout_seconds"])
    total_limit = float(EXPECTED_BUDGET["total_wall_clock_seconds"])
    native_durations = list(durations["native"])
    crossfeed_durations = list(durations["crossfeed"])
    output_sizes = list(native_output_sizes)
    output_limit = int(EXPECTED_BUDGET["max_output_bytes"])
    rows = {
        "generation": {
            "count": 1,
            "limit": EXPECTED_BUDGET["max_diagnostic_generations"],
            "duration_seconds": total_duration,
            "within_budget": total_duration <= total_limit,
        },
        "verified_raw_snapshot": {
            "count": 1,
            "limit": EXPECTED_BUDGET["max_verified_raw_snapshots"],
            "durations_seconds": [snapshot_duration],
            "per_execution_limit_seconds": per_execution_limit,
            "enforcement": EXPECTED_TIMEOUT_ENFORCEMENT["verified_snapshot"],
            "within_budget": snapshot_duration <= per_execution_limit,
        },
        "python": {
            "count": counts["python"],
            "limit": EXPECTED_BUDGET["max_python_executions"],
            "durations_seconds": [python_duration],
            "per_execution_limit_seconds": per_execution_limit,
            "enforcement": EXPECTED_TIMEOUT_ENFORCEMENT["python_inference"],
            "within_budget": counts["python"] <= EXPECTED_BUDGET["max_python_executions"]
            and python_duration <= per_execution_limit,
        },
        "native": {
            "count": counts["native"],
            "limit": EXPECTED_BUDGET["max_native_executions"],
            "durations_seconds": native_durations,
            "per_execution_limit_seconds": per_execution_limit,
            "enforcement": EXPECTED_TIMEOUT_ENFORCEMENT["native_subprocess"],
            "within_budget": counts["native"] <= EXPECTED_BUDGET["max_native_executions"]
            and len(native_durations) == counts["native"]
            and all(duration <= per_execution_limit for duration in native_durations),
        },
        "crossfeed": {
            "count": counts["crossfeed"],
            "limit": EXPECTED_BUDGET["max_crossfeed_executions"],
            "durations_seconds": crossfeed_durations,
            "per_execution_limit_seconds": per_execution_limit,
            "within_budget": counts["crossfeed"] <= EXPECTED_BUDGET["max_crossfeed_executions"]
            and len(crossfeed_durations) == counts["crossfeed"]
            and all(duration <= per_execution_limit for duration in crossfeed_durations),
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


def _enforce_budget_audit(audit: Mapping[str, object]) -> None:
    if audit.get("within_budget") is not True:
        raise DiagnosticContractError("Diagnostic execution exceeded authorization budget")


def run_diagnostic(
    project_root: Path,
    *,
    run_authorization_path: Path,
    output_path: Path,
    python_trace_builder: Optional[Callable[..., PythonDiagnosticTrace]] = None,
    native_runner_factory: Optional[Callable[..., NativeRunner]] = None,
    key_factory: Callable[[int], bytes] = secrets.token_bytes,
    clock: Optional[Clock] = None,
    flow_started: Optional[float] = None,
) -> dict:
    project_root = project_root.resolve()
    clock = clock or time.monotonic
    flow_started = clock() if flow_started is None else flow_started
    total_limit = float(EXPECTED_BUDGET["total_wall_clock_seconds"])

    def enforce_total(scope: str) -> float:
        duration = _duration_since(flow_started, clock)
        _enforce_duration(duration, total_limit, scope=f"total flow after {scope}")
        return duration

    # 授权与父证据必须先通过；此行之前不会读取 raw、checkpoint 或 pickle。
    preregistration = verify_diagnostic_authorization(project_root)
    run_authorization = verify_run_authorization(
        project_root,
        run_authorization_path,
        output_path,
        preregistration,
    )
    authorized_output = replay.resolve_within(
        project_root,
        output_path,
        purpose="Authorized generation output receipt",
    )
    if authorized_output.exists():
        raise DiagnosticContractError("Authorized generation receipt already exists")
    attempt_lease = _consume_attempt_lease(project_root, run_authorization)
    enforce_total("authorization")
    # Parent truth is immutable historical evidence. Current implementation hashes are
    # replayed through the separate hash-only manifest to avoid a path-level hash cycle.
    historical_truth_path = _resolve_frozen(
        project_root,
        DEFAULT_TRUTH_MANIFEST,
        purpose="Historical parent truth manifest",
    )
    historical_truth_sha256 = replay.file_sha256(historical_truth_path)
    if historical_truth_sha256 != preregistration["parent_evidence_sha256"]["truth_manifest"]:
        raise DiagnosticContractError("Historical parent truth manifest SHA-256 drifted")
    enforce_total("historical truth hash verification")

    split_csv = _resolve_frozen(project_root, DEFAULT_SPLIT_CSV, purpose="Frozen split CSV")
    samples, identity_audit = replay.read_split_samples(
        split_csv,
        requested_split=FIXED_SPLIT,
        max_samples=1,
    )
    sample = samples[0]
    _assert_fixed_sample(sample)
    enforce_total("split identity audit")

    checkpoint_path = _resolve_frozen(
        project_root, DEFAULT_CHECKPOINT, purpose="Frozen Python checkpoint"
    )
    stage2_path = _resolve_frozen(
        project_root, DEFAULT_PYTHON_STAGE2, purpose="Frozen Python Stage2"
    )
    pickle_guard = replay.guard_pickle_before_load(
        project_root,
        stage2_path,
        DEFAULT_PICKLE_ALLOWLIST,
    )
    native_paths = {
        "selftest": _resolve_frozen(
            project_root, DEFAULT_NATIVE_SELFTEST, purpose="Frozen native selftest"
        ),
        "dll": _resolve_frozen(project_root, DEFAULT_NATIVE_DLL, purpose="Frozen native DLL"),
        "onnx": _resolve_frozen(project_root, DEFAULT_NATIVE_ONNX, purpose="Frozen native ONNX"),
        "stage2": _resolve_frozen(
            project_root, DEFAULT_NATIVE_STAGE2, purpose="Frozen native Stage2"
        ),
    }
    for name, path in {"checkpoint": checkpoint_path, **native_paths}.items():
        if not path.is_file():
            raise DiagnosticContractError(f"Frozen diagnostic artifact is missing: {name}")
    enforce_total("artifact guards")

    logical_root = _resolve_frozen(project_root, Path("data"), purpose="Frozen logical raw root")
    raw_parent_authorization = replay.verify_a1_authorization(
        project_root,
        mode="native-parity",
        max_samples=1,
        allowed_raw_root=logical_root,
    )
    allowed_resolved_roots = [
        Path(path) for path in raw_parent_authorization["allowed_resolved_raw_roots"]
    ]
    enforce_total("raw authorization")
    python_trace_builder = python_trace_builder or build_python_trace
    require_native_output_accounting = native_runner_factory is None
    native_output_sizes: list[int] = []
    timeout_seconds = int(preregistration["budget"]["per_execution_timeout_seconds"])
    max_output_bytes = int(preregistration["budget"]["max_output_bytes"])

    with tempfile.TemporaryDirectory(prefix="axon-loop28-parity-diagnostic-") as temp_dir:
        snapshot_root = Path(temp_dir)
        snapshot_started = clock()
        sample_path, snapshot_record = replay.snapshot_verified_sample(
            sample,
            allowed_raw_root=logical_root,
            allowed_resolved_roots=allowed_resolved_roots,
            snapshot_root=snapshot_root,
        )
        if snapshot_record["size_bytes"] != FIXED_SAMPLE_SIZE_BYTES:
            raise DiagnosticContractError("Frozen diagnostic sample size mismatch")
        snapshot_duration = _duration_since(snapshot_started, clock)
        _enforce_duration(
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
            stage2_path=stage2_path,
        )
        python_duration = _duration_since(python_started, clock)
        _enforce_duration(
            python_duration,
            timeout_seconds,
            scope="Python execution",
        )
        enforce_total("Python execution")
        if native_runner_factory is None:

            def native_runner(key: bytes, component: Optional[str], block_elements: Optional[int]):
                return run_native_diagnostics(
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
        diagnostic = diagnose_trace(
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
        diagnostic=diagnostic,
        total_duration=total_duration,
        native_output_sizes=native_output_sizes,
        require_native_output_accounting=require_native_output_accounting,
    )
    _enforce_budget_audit(budget_audit)

    return {
        "schema": RECEIPT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": {
            "train_only_diagnostic": True,
            "quality_claim_allowed": False,
            "parity_claim_allowed": False,
            "heldout_raw_accessed": False,
            "heldout_predictions_accessed": False,
            "heldout_metrics_accessed": False,
            "complete_split_metadata_use": "identity_audit_only",
            "split_metadata_used_to_select_frozen_train_identity": True,
            "split_metadata_used_for_model_selection_or_metrics": False,
            "training_or_fitting_performed": False,
            "artifact_binding": "pre_and_post_path_hash_verification",
            "same_handle_artifact_snapshot": False,
            "concurrent_adversarial_mutation_resistant": False,
            "certification_claim_allowed": False,
        },
        "generation": run_authorization["generation"],
        "authorization": {
            "preregistration": preregistration,
            "run": run_authorization,
            "attempt_lease": _verify_attempt_lease(project_root, attempt_lease),
        },
        "sample_identity": {
            "source_sha256": FIXED_SAMPLE_SHA256,
            "sample_index": FIXED_SAMPLE_INDEX,
            "split": FIXED_SPLIT,
            "label": sample.label,
            "size_bytes": FIXED_SAMPLE_SIZE_BYTES,
        },
        "evidence_sha256": {
            "historical_truth_manifest": historical_truth_sha256,
            "implementation_manifest": run_authorization["implementation_manifest"][
                "implementation_manifest_sha256"
            ],
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
            "heldout_metadata_rows_scanned": sum(
                count
                for split, count in identity_audit["split_counts"].items()
                if split != FIXED_SPLIT
            ),
            "metadata_fields_parsed": [
                "source_path",
                "source_sha256",
                "sample_index",
                "label",
                "split",
            ],
            "selected_count": identity_audit["selected_count"],
            "reported_count": 1,
            "dropped_row_count": 0,
        },
        "budget_audit": budget_audit,
        "diagnostic": diagnostic,
        "decision": diagnostic["decision"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Localize the frozen Loop28 Python/native drift without exposing tensors."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--run-authorization",
        type=Path,
        required=True,
        help="Explicit server-owned, final SHA-bound authorization for this one diagnostic run.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Must equal the single final path bound by the run authorization.",
    )
    return parser


def _write_receipt_exclusive(path: Path, receipt: Mapping[str, object]) -> None:
    encoded = json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    if len(encoded.encode("utf-8")) > int(EXPECTED_BUDGET["max_output_bytes"]):
        raise DiagnosticContractError("Diagnostic receipt exceeded its output budget")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise DiagnosticContractError("Authorized generation receipt already exists") from exc


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    clock = time.monotonic
    flow_started = clock()
    try:
        output_path = replay.resolve_within(
            project_root,
            args.output_json,
            purpose="Diagnostic output receipt",
        )
        receipt = run_diagnostic(
            project_root,
            run_authorization_path=args.run_authorization,
            output_path=output_path,
            clock=clock,
            flow_started=flow_started,
        )
        # 写盘前重新绑定同一授权 SHA、generation 和 output，拒绝运行期间的策略漂移。
        final_preregistration = verify_diagnostic_authorization(project_root)
        final_run_authorization = verify_run_authorization(
            project_root,
            args.run_authorization,
            output_path,
            final_preregistration,
        )
        if (
            final_run_authorization["authorization_sha256"]
            != receipt["authorization"]["run"]["authorization_sha256"]
            or final_run_authorization["generation"] != receipt["generation"]
        ):
            raise DiagnosticContractError("Run authorization changed before receipt write")
        _verify_attempt_lease(
            project_root,
            {
                "payload": {
                    "schema": ATTEMPT_LEASE_SCHEMA,
                    "loop_id": "p0_loop28_parity_diagnostic_001",
                    "attempt_id": final_run_authorization["attempt_id"],
                    "generation": final_run_authorization["generation"],
                    "run_authorization_sha256": final_run_authorization["authorization_sha256"],
                    "output_path": OUTPUT_PATHS_BY_GENERATION[
                        str(final_run_authorization["generation"])
                    ].as_posix(),
                    "consumed_at_utc": receipt["authorization"]["attempt_lease"].get(
                        "consumed_at_utc"
                    ),
                    "status": "authorization_consumed_before_raw_access",
                },
                "sha256": receipt["authorization"]["attempt_lease"]["sha256"],
            },
        )
        final_duration = _duration_since(flow_started, clock)
        _enforce_duration(
            final_duration,
            float(EXPECTED_BUDGET["total_wall_clock_seconds"]),
            scope="total flow before receipt write",
        )
        receipt["budget_audit"]["generation"]["duration_seconds"] = final_duration
        receipt["budget_audit"]["total_wall_clock"]["duration_seconds"] = final_duration
        receipt["budget_audit"]["generation"]["within_budget"] = True
        receipt["budget_audit"]["total_wall_clock"]["within_budget"] = True
        receipt["budget_audit"]["within_budget"] = True
        _enforce_budget_audit(receipt["budget_audit"])
        _write_receipt_exclusive(output_path, receipt)
    except (DiagnosticContractError, replay.ReplayContractError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": receipt["decision"] == "first_divergence_localized",
                "decision": receipt["decision"],
                "output_sha256": replay.file_sha256(output_path),
            }
        )
    )
    return 0 if receipt["decision"] == "first_divergence_localized" else 2


if __name__ == "__main__":
    raise SystemExit(main())
