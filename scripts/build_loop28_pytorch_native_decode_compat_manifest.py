#!/usr/bin/env python3
"""Build immutable manifests for the PyTorch v2.13 decoder-compat successor."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "p0_loop28_pytorch_native_decode_compat_001"
MANIFEST_DIR = Path("manifests/roadmap_9997/p0_loop28_pytorch_native_decode_compat")
REPORT_DIR = Path("reports/roadmap_9997/p0_loop28_pytorch_native_decode_compat")
ARTIFACT_ROOT = Path(
    "artifacts/roadmap_9997/p0_loop28_pytorch_native_decode_compat/tiny_v2/package_attempt_001"
)

PROPOSAL = MANIFEST_DIR / "proposal.json"
AUTHORIZATION = MANIFEST_DIR / "authorization.json"
PREFLIGHT_AUTHORIZATION = MANIFEST_DIR / "preflight_authorization.json"
PREFLIGHT_FINAL_LEASE = MANIFEST_DIR / "preflight_lease.final.json"
PREFLIGHT_EVIDENCE = REPORT_DIR / "decode_probe_evidence.final.json"
PREFLIGHT_FAILURE = REPORT_DIR / "decode_probe_failure.final.json"
PREFLIGHT_MANIFEST = MANIFEST_DIR / "preflight.json"
IMPLEMENTATION_MANIFEST = MANIFEST_DIR / "implementation_manifest.json"
OFFICIAL_RESEARCH_MANIFEST = MANIFEST_DIR / "official_pytorch_v213_source_manifest.json"
REUSED_BINARIES_MANIFEST = MANIFEST_DIR / "reused_cpp_binaries_manifest.json"
PACKAGE_AUTHORIZATION = MANIFEST_DIR / "package_authorization.json"
PACKAGE_FINAL_LEASE = MANIFEST_DIR / "package_lease.final.json"
PACKAGE_RECEIPT = REPORT_DIR / "package_receipt.final.json"
PACKAGE_FAILURE = REPORT_DIR / "package_failure.final.json"
PACKAGE_MANIFEST = MANIFEST_DIR / "package_manifest.json"
POST_MANIFEST = MANIFEST_DIR / "post_manifest.json"
PREFLIGHT_WORK_ROOT = REPORT_DIR / "work/decode_probe_attempt_001"
PACKAGE_WORK_ROOT = REPORT_DIR / "work/package_attempt_001"

RUNNER = Path("scripts/run_loop28_pytorch_native_decode_compat.py")
RUNNER_TEST = Path("tests/test_run_loop28_pytorch_native_decode_compat.py")
BUILDER = Path("scripts/build_loop28_pytorch_native_decode_compat_manifest.py")
BUILDER_TEST = Path("tests/test_build_loop28_pytorch_native_decode_compat_manifest.py")
BASE_MODEL_SOURCE = Path("scripts/run_loop28_pytorch_native_feasibility.py")
PARENT_SAFETY_SOURCE = Path("scripts/run_loop28_pytorch_native_package_controller.py")
ATEN_HOST = Path("tools/axon_tiny_pytorch_native/build/bin/Release/axon_tiny_aten_probe.exe")
AOTI_HOST = Path("tools/axon_tiny_pytorch_native/build/bin/Release/axon_tiny_aoti_probe.exe")
PARENT_POST = Path("manifests/roadmap_9997/p0_loop28_pytorch_native_feasibility/post_manifest.json")
PARENT_FAILURE = Path(
    "manifests/roadmap_9997/p0_loop28_pytorch_native_feasibility/package_failure_manifest.json"
)
UPSTREAM_RESEARCH = Path(
    "reports/roadmap_9997/p0_loop28_pytorch_native_feasibility/"
    "upstream_decode_compatibility_research.final.json"
)
PARENT_BUILD_RECEIPT = Path(
    "reports/roadmap_9997/p0_loop28_pytorch_native_feasibility/cpp_build_receipt.final.json"
)
INPUT_PATH = ARTIFACT_ROOT / "input_v2.f32.bin"
AOTI_PACKAGE = ARTIFACT_ROOT / "tiny_cpu_model_v2.pt2"
PARENT_PARTIAL_ROOT = Path(
    "artifacts/roadmap_9997/p0_loop28_pytorch_native_feasibility/tiny_v1"
)

FINAL_DOCS = (
    ("goal_delta", REPORT_DIR / "goal_delta.final.md", "immutable_goal_delta"),
    ("journal_entry", REPORT_DIR / "journal_entry.final.md", "immutable_journal_entry"),
    ("status", REPORT_DIR / "status.final.md", "immutable_owner_status"),
)

EXPECTED_PARENT_POST_SHA256 = "71110ee310a340fb231051df3c4fe3be865e4fa28a0281d96cdf3ddf2ef115cc"
EXPECTED_PARENT_FAILURE_SHA256 = "a445fa8956b444c500fe370cbf9493871a0efbc5c924c21a4d988acfb1db4792"
EXPECTED_UPSTREAM_RESEARCH_SHA256 = (
    "280c8fc1792ae4affaf6252a0c0dfbd3f053a7d544100d1832b267a6678fa608"
)
EXPECTED_PARENT_BUILD_RECEIPT_SHA256 = (
    "2ef9fc018cf72c0552902d4e9cca2190b4c5e983bdd15d4917dd5d926043dc77"
)
EXPECTED_BASE_MODEL_SHA256 = "b2c61788e0e7ae1348090fcc41e7eeb2efecb3be3efa912fd6560ad030935166"
EXPECTED_PARENT_SAFETY_SHA256 = (
    "9baea0517a0b5cf7ecd80edb6f4e84911f1148d3a9d73d929d813f075e693f48"
)
EXPECTED_ATEN_HOST_SHA256 = "595705bdc0716b7323a0da71b424470c0d474a6556ac7fa3c6507e5d4edfd524"
EXPECTED_AOTI_HOST_SHA256 = "097a41e61bfceeeea4592ff2f1dd079e3b447275b677980b7eceb3f02a5dcca0"
EXPECTED_ATEN_HOST_SIZE = 159232
EXPECTED_AOTI_HOST_SIZE = 160768
EXPECTED_PYTHON_SHA256 = "4b8c3912806b3c1591ba3cb403bff77ad309c3fe5756f87c20b7a6f8f0174262"
EXPECTED_CPP_BUILDER_SHA256 = "9952fcc6ae0b660c3fb9b4f279b30caacb31167ae9bc2959872c2460998ae014"
EXPECTED_CPU_VEC_ISA_SHA256 = "371882a23012d93fa51ca5f3a66d827dcfca507d3bc2b1606af65cf660c18fb3"
EXPECTED_TORCH_VERSION = "2.12.0+cu132"
PARENT_PARTIAL_INPUT_SHA256 = "caa371218bdbb95cb73bfe7ab65ec2f8f69222a747fca8f889b2bdc3e693d28b"
EXPECTED_INPUT_SHA256 = "19428a55d01cbd3d1b64a546a5cae7f93091cc4fa3acd1339d78d9b5b87264eb"

OFFICIAL_V213_TAG = "v2.13.0"
OFFICIAL_V213_TAG_COMMIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
OFFICIAL_V213_SOURCE_URL = (
    "https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/torch/_inductor/cpp_builder.py"
)
OFFICIAL_V213_SOURCE_SHA256 = "cb369b351ac1021ecd6127e536ef35c1a6d57de6ae3689e95d097c9ab3ebad02"
OFFICIAL_V213_EXCERPT = (
    'SUBPROCESS_DECODE_ARGS = ((locale.getpreferredencoding(), "replace") '
    "if _IS_WINDOWS else ())"
)
OFFICIAL_V213_EXCERPT_SHA256 = "e53c891435fab7d062d7bf64b95631cb2869ddc3e60477b07c58e1d663c297fd"
OFFICIAL_ISSUE_URL = "https://github.com/pytorch/pytorch/issues/157673"

PREFLIGHT_ATTEMPT_ID = "p0_loop28_pytorch_native_decode_compat_001_decode_probe_attempt_001"
PACKAGE_ATTEMPT_ID = "p0_loop28_pytorch_native_decode_compat_001_package_attempt_001"
PREFLIGHT_AUTHORIZATION_SCHEMA = "axon_loop28_pytorch_native_decode_preflight_authorization_v1"
PREFLIGHT_LEASE_SCHEMA = "axon_loop28_pytorch_native_decode_preflight_lease_v1"
PACKAGE_AUTHORIZATION_SCHEMA = "axon_loop28_pytorch_native_decode_package_authorization_v1"
PACKAGE_LEASE_SCHEMA = "axon_loop28_pytorch_native_decode_package_lease_v1"
PREFLIGHT_AUTHORIZATION_DECISION = "authorize_single_upstream_v213_decode_preflight"
PACKAGE_AUTHORIZATION_DECISION = "authorize_single_decode_compat_tiny_v2_package_no_load"

ZERO_SCIENTIFIC_COUNTERS = (
    "model_constructions",
    "torch_export_calls",
    "aoti_compile_and_package_calls",
    "package_load_calls",
    "native_probe_execution_count",
    "gpu_execution_count",
    "network_request_count",
    "quality_metric_count",
)
WORKER_COUNTER_NAMES = (
    "torch_imports",
    "model_constructions",
    "torch_export_calls",
    "torch_export_completed",
    "aoti_compile_and_package_calls",
    "aoti_compile_and_package_completed",
    "torchscript_export_calls",
    "package_load_calls",
    "native_probe_execution_count",
    "gpu_execution_count",
    "network_request_count",
    "quality_metric_count",
    "total_subprocesses",
    "compiler_processes",
    "compiler_help_processes",
    "dumpbin_processes",
)
PREFLIGHT_BUDGET = {
    "worker_processes": 1,
    "vcvars_activations": 1,
    "torch_imports": 1,
    "compiler_help_processes_max": 4,
    "wall_clock_seconds_max": 1800,
    "model_constructions": 0,
    "torch_export_calls": 0,
    "aoti_compile_and_package_calls": 0,
    "package_load_calls": 0,
    "native_probe_execution_count": 0,
    "gpu_execution_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
}
PACKAGE_COUNTER_LIMITS = {
    "torch_imports": 1,
    "model_constructions": 1,
    "torch_export_calls": 1,
    "torch_export_completed": 1,
    "aoti_compile_and_package_calls": 1,
    "aoti_compile_and_package_completed": 1,
    "torchscript_export_calls": 0,
    "package_load_calls": 0,
    "native_probe_execution_count": 0,
    "gpu_execution_count": 0,
    "network_request_count": 0,
    "quality_metric_count": 0,
}
PACKAGE_BUDGET = {
    "worker_processes": 1,
    "vcvars_activations": 1,
    "compiler_help_processes_max": 4,
    "dumpbin_processes_max": 64,
    "wall_clock_seconds_max": 1800,
    "max_retained_output_bytes": 536870912,
    **PACKAGE_COUNTER_LIMITS,
}
TERMINAL_EVIDENCE_CONTRACT = {
    "source_record_fields": ["path", "size_bytes", "sha256"],
    "source_before_and_after_exact_authorization_match": True,
    "worker_temp_and_work_roots_removed": True,
    "console_code_pages_utf8_around_started_child_and_exactly_restored": True,
    "process_tree_termination_must_be_confirmed": True,
    "windows_job_object_containment_required": True,
    "job_assignment_gate_before_worker_start_required": True,
    "taskkill_fallback_pre_gate_zero_journal_only": True,
    "structured_worker_receipt_required_for_every_terminal": True,
    "administrative_failure_zero_counter_receipt_required": True,
    "frozen_python_torch_cuda_shim_hashes_required_when_torch_imported": True,
    "worker_telemetry_counter_reconciliation_required": True,
    "durable_worker_journal_hash_chain_required": True,
    "journal_reconstruction_must_preserve_last_counters": True,
    "journal_corruption_zero_fallback_forbidden": True,
    "partial_inventory_error_forbidden_in_terminal_manifest": True,
    "failure_reason_exact_classification_required": True,
    "finite_budget_actual_reconciliation_required": True,
}


class DecodeCompatManifestError(RuntimeError):
    """Raised when a successor manifest cannot be proven."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise DecodeCompatManifestError(f"Duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite_constant(value: str) -> object:
    raise DecodeCompatManifestError(f"Non-finite JSON number is forbidden: {value}")


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DecodeCompatManifestError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise DecodeCompatManifestError(f"JSON artifact must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise DecodeCompatManifestError(f"{field} must be a lowercase SHA-256")
    return text


def _resolve_within(project_root: Path, relative: Path, *, must_exist: bool = True) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise DecodeCompatManifestError(f"Path must remain project-relative: {relative}")
    root = project_root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=must_exist)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DecodeCompatManifestError(f"Path escapes project root: {relative}") from exc
    if must_exist and not candidate.is_file():
        raise DecodeCompatManifestError(f"Required artifact is not a file: {relative}")
    return candidate


def _validate_timestamp(value: str) -> str:
    if not value or not value.endswith("Z"):
        raise DecodeCompatManifestError("generated_at_utc must end in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise DecodeCompatManifestError("generated_at_utc is invalid") from exc
    if parsed.tzinfo != dt.timezone.utc:
        raise DecodeCompatManifestError("generated_at_utc must use UTC")
    return value


def _artifact_record(project_root: Path, name: str, path: Path, role: str) -> dict[str, Any]:
    resolved = _resolve_within(project_root, path)
    return {
        "name": name,
        "role": role,
        "path": path.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _artifact_record_by_path(project_root: Path, path: Path) -> dict[str, Any]:
    resolved = _resolve_within(project_root, path)
    return {
        "path": path.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _index_source_records(records: object) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise DecodeCompatManifestError("Stage authorization must bind source_artifacts")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise DecodeCompatManifestError("Stage source artifact record must be an object")
        if set(record) != {"path", "size_bytes", "sha256"}:
            raise DecodeCompatManifestError(
                "Stage source artifact records must contain only path, size_bytes, and sha256"
            )
        raw_path = str(record.get("path") or "")
        pure = PurePosixPath(raw_path)
        if not raw_path or pure.is_absolute() or ".." in pure.parts:
            raise DecodeCompatManifestError("Stage source artifact path is unsafe")
        normalized = Path(*pure.parts).as_posix()
        if normalized in indexed:
            raise DecodeCompatManifestError(f"Duplicate stage source path: {normalized}")
        size = record.get("size_bytes")
        if not isinstance(size, int) or size < 0:
            raise DecodeCompatManifestError(f"Invalid stage source size: {normalized}")
        _require_sha256(record.get("sha256"), field=f"source_artifacts[{normalized}].sha256")
        indexed[normalized] = record
    return indexed


def _verify_source_binding(
    project_root: Path,
    authorization: Mapping[str, Any],
    *,
    embedded_authorization: object | None = None,
) -> list[dict[str, Any]]:
    if embedded_authorization is not None and embedded_authorization != authorization:
        raise DecodeCompatManifestError(
            "Evidence authorization differs from the stage authorization bytes"
        )
    indexed = _index_source_records(authorization.get("source_artifacts"))
    required = (
        RUNNER,
        RUNNER_TEST,
        BUILDER,
        BUILDER_TEST,
        BASE_MODEL_SOURCE,
        PARENT_SAFETY_SOURCE,
        ATEN_HOST,
        AOTI_HOST,
        OFFICIAL_RESEARCH_MANIFEST,
        REUSED_BINARIES_MANIFEST,
    )
    for path in required:
        expected = _artifact_record_by_path(project_root, path)
        record = indexed.get(path.as_posix())
        if record is None:
            raise DecodeCompatManifestError(
                f"Stage authorization omitted required source: {path.as_posix()}"
            )
        for field in ("path", "size_bytes", "sha256"):
            if record.get(field) != expected[field]:
                raise DecodeCompatManifestError(
                    f"Stage source binding drifted: {path.as_posix()}:{field}"
                )
    for path_text, record in indexed.items():
        current = _artifact_record_by_path(project_root, Path(path_text))
        for field in ("path", "size_bytes", "sha256"):
            if record.get(field) != current[field]:
                raise DecodeCompatManifestError(
                    f"Additional stage source binding drifted: {path_text}:{field}"
                )
    return [indexed[key] for key in sorted(indexed)]


def _validate_command(command: object, *, mode: str) -> list[str]:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise DecodeCompatManifestError("Stage canonical_command must be an argv list")
    if len(command) != 3:
        raise DecodeCompatManifestError("Stage canonical_command must have three argv entries")
    normalized_python = command[0].replace("\\", "/").casefold()
    if not normalized_python.endswith("/vnev/scripts/python.exe"):
        raise DecodeCompatManifestError("Stage command does not use the frozen venv Python")
    if command[1] != RUNNER.as_posix() or command[2] != mode:
        raise DecodeCompatManifestError("Stage command does not match the governed runner mode")
    return command


def _validate_stage_chain(
    project_root: Path,
    *,
    authorization_path: Path,
    final_lease_path: Path,
    authorization_schema: str,
    authorization_decision: str,
    lease_schema: str,
    attempt_id: str,
    mode: str,
    work_root: Path,
    output_paths: Sequence[Path],
    implementation_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = project_root.resolve(strict=True)
    authorization = load_json_strict(_resolve_within(root, authorization_path))
    final_lease = load_json_strict(_resolve_within(root, final_lease_path))
    if authorization.get("schema") != authorization_schema:
        raise DecodeCompatManifestError("Stage authorization schema mismatch")
    if authorization.get("loop_id") != LOOP_ID or authorization.get("attempt_id") != attempt_id:
        raise DecodeCompatManifestError("Stage authorization identity mismatch")
    if authorization.get("decision") != authorization_decision:
        raise DecodeCompatManifestError("Stage authorization decision mismatch")
    command = _validate_command(authorization.get("canonical_command"), mode=mode)
    if authorization.get("artifact_root") != ARTIFACT_ROOT.as_posix():
        raise DecodeCompatManifestError("Stage authorization artifact root mismatch")
    if authorization.get("work_root") != work_root.as_posix():
        raise DecodeCompatManifestError("Stage authorization work root mismatch")
    expected_outputs = [path.as_posix() for path in output_paths]
    if authorization.get("terminal_outputs") != expected_outputs:
        raise DecodeCompatManifestError("Stage authorization terminal outputs mismatch")
    budget = authorization.get("budget")
    if not isinstance(budget, dict) or not budget:
        raise DecodeCompatManifestError("Stage authorization budget is missing")
    if implementation_sha256 is not None and authorization.get(
        "implementation_manifest_sha256"
    ) != implementation_sha256:
        raise DecodeCompatManifestError("Package authorization implementation binding mismatch")

    if final_lease.get("schema") != lease_schema or final_lease.get("loop_id") != LOOP_ID:
        raise DecodeCompatManifestError("Consumed stage lease schema or loop mismatch")
    if final_lease.get("lease_id") != attempt_id or final_lease.get("attempt_id") != attempt_id:
        raise DecodeCompatManifestError("Consumed stage lease identity mismatch")
    if final_lease.get("status") != "consumed_before_execution":
        raise DecodeCompatManifestError("Stage lease was not consumed before execution")
    if final_lease.get("single_use") is not True:
        raise DecodeCompatManifestError("Consumed stage lease is not single-use")
    authorization_sha = sha256_file(_resolve_within(root, authorization_path))
    if final_lease.get("authorization_path") != authorization_path.as_posix():
        raise DecodeCompatManifestError("Consumed stage lease authorization path mismatch")
    if final_lease.get("authorization_sha256") != authorization_sha:
        raise DecodeCompatManifestError("Consumed stage lease authorization hash mismatch")
    if final_lease.get("canonical_command") != command:
        raise DecodeCompatManifestError("Consumed stage lease command mismatch")
    if final_lease.get("artifact_root") != ARTIFACT_ROOT.as_posix():
        raise DecodeCompatManifestError("Consumed stage lease artifact root mismatch")
    if final_lease.get("work_root") != work_root.as_posix():
        raise DecodeCompatManifestError("Consumed stage lease work root mismatch")
    if final_lease.get("terminal_outputs") != expected_outputs:
        raise DecodeCompatManifestError("Consumed stage lease terminal outputs mismatch")
    if final_lease.get("budget_sha256") != _canonical_sha256(budget):
        raise DecodeCompatManifestError("Consumed stage lease budget digest mismatch")
    if final_lease.get("consumed_path") != final_lease_path.as_posix():
        raise DecodeCompatManifestError("Consumed stage lease final path mismatch")
    _require_sha256(final_lease.get("original_lease_sha256"), field="original_lease_sha256")
    _validate_timestamp(str(final_lease.get("consumed_at_utc") or ""))
    return authorization, final_lease


def _validate_evidence_lease(
    evidence: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    authorization_path: Path,
    final_lease: Mapping[str, Any],
    final_lease_path: Path,
    project_root: Path,
) -> None:
    embedded = evidence.get("lease")
    if not isinstance(embedded, dict):
        raise DecodeCompatManifestError("Stage evidence omitted its consumed lease receipt")
    if embedded.get("authorization") != authorization:
        raise DecodeCompatManifestError("Stage evidence authorization object drifted")
    authorization_sha = sha256_file(_resolve_within(project_root, authorization_path))
    final_sha = sha256_file(_resolve_within(project_root, final_lease_path))
    if embedded.get("authorization_sha256") != authorization_sha:
        raise DecodeCompatManifestError("Stage evidence authorization hash drifted")
    if embedded.get("path") != final_lease_path.as_posix() or embedded.get("sha256") != final_sha:
        raise DecodeCompatManifestError("Stage evidence consumed lease binding drifted")
    if embedded.get("ready_lease_sha256") != final_lease.get("original_lease_sha256"):
        raise DecodeCompatManifestError("Stage evidence ready lease hash drifted")
    if embedded.get("status") != "consumed_before_execution":
        raise DecodeCompatManifestError("Stage evidence lease status drifted")


def _load_runner():
    path = PROJECT_ROOT / RUNNER
    spec = importlib.util.spec_from_file_location("decode_compat_runner_for_manifest", path)
    if spec is None or spec.loader is None:
        raise DecodeCompatManifestError("Unable to import successor runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_base_chain(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    expected = (
        (PARENT_POST, EXPECTED_PARENT_POST_SHA256, "parent post"),
        (PARENT_FAILURE, EXPECTED_PARENT_FAILURE_SHA256, "parent failure"),
        (UPSTREAM_RESEARCH, EXPECTED_UPSTREAM_RESEARCH_SHA256, "upstream research"),
        (
            PARENT_BUILD_RECEIPT,
            EXPECTED_PARENT_BUILD_RECEIPT_SHA256,
            "parent build receipt",
        ),
        (BASE_MODEL_SOURCE, EXPECTED_BASE_MODEL_SHA256, "base model source"),
        (PARENT_SAFETY_SOURCE, EXPECTED_PARENT_SAFETY_SHA256, "parent safety source"),
        (ATEN_HOST, EXPECTED_ATEN_HOST_SHA256, "ATen host"),
        (AOTI_HOST, EXPECTED_AOTI_HOST_SHA256, "AOTI host"),
    )
    for relative, digest, purpose in expected:
        path = _resolve_within(root, relative)
        if sha256_file(path) != digest:
            raise DecodeCompatManifestError(f"Successor parent hash drifted: {purpose}")
    proposal = load_json_strict(_resolve_within(root, PROPOSAL))
    authorization = load_json_strict(_resolve_within(root, AUTHORIZATION))
    if proposal.get("schema") != "axon_loop28_pytorch_native_decode_compat_proposal_v1":
        raise DecodeCompatManifestError("Successor proposal schema mismatch")
    if proposal.get("loop_id") != LOOP_ID:
        raise DecodeCompatManifestError("Successor proposal loop mismatch")
    if proposal.get("decision") != "propose_upstream_v213_decode_compat_package_only_successor":
        raise DecodeCompatManifestError("Successor proposal decision mismatch")
    if proposal.get("parent_closure") != {
        "path": PARENT_POST.as_posix(),
        "sha256": EXPECTED_PARENT_POST_SHA256,
        "decision": "post_tiny_aoti_package_closed_utf8_cp936_compiler_probe_failure_no_load",
    }:
        raise DecodeCompatManifestError("Successor proposal parent closure mismatch")
    if proposal.get("parent_failure_manifest") != {
        "path": PARENT_FAILURE.as_posix(),
        "sha256": EXPECTED_PARENT_FAILURE_SHA256,
    }:
        raise DecodeCompatManifestError("Successor proposal parent failure mismatch")
    upstream_basis = proposal.get("upstream_basis", {})
    if upstream_basis.get("path") != UPSTREAM_RESEARCH.as_posix() or upstream_basis.get(
        "sha256"
    ) != EXPECTED_UPSTREAM_RESEARCH_SHA256:
        raise DecodeCompatManifestError("Successor proposal upstream basis mismatch")
    frozen_reuse = proposal.get("frozen_reuse", {})
    expected_reuse = {
        "tiny_model_source": (BASE_MODEL_SOURCE, EXPECTED_BASE_MODEL_SHA256),
        "archive_safety_source": (PARENT_SAFETY_SOURCE, EXPECTED_PARENT_SAFETY_SHA256),
        "aoti_host": (AOTI_HOST, EXPECTED_AOTI_HOST_SHA256),
        "direct_aten_host": (ATEN_HOST, EXPECTED_ATEN_HOST_SHA256),
    }
    for name, (path, digest) in expected_reuse.items():
        record = frozen_reuse.get(name, {})
        if record.get("path") != path.as_posix() or record.get("sha256") != digest:
            raise DecodeCompatManifestError(f"Successor frozen reuse drifted: {name}")
    expected_stage_budgets = {
        "preflight": PREFLIGHT_BUDGET,
        "package": PACKAGE_BUDGET,
    }
    if proposal.get("stage_budgets") != expected_stage_budgets:
        raise DecodeCompatManifestError("Successor proposal stage budgets drifted")
    if proposal.get("terminal_evidence_contract") != TERMINAL_EVIDENCE_CONTRACT:
        raise DecodeCompatManifestError("Successor proposal terminal evidence contract drifted")
    if authorization.get("schema") != "axon_loop28_pytorch_native_decode_compat_authorization_v1":
        raise DecodeCompatManifestError("Successor authorization schema mismatch")
    if authorization.get("loop_id") != LOOP_ID:
        raise DecodeCompatManifestError("Successor authorization loop mismatch")
    if authorization.get("proposal", {}).get("path") != PROPOSAL.as_posix():
        raise DecodeCompatManifestError("Successor authorization proposal path mismatch")
    if authorization.get("proposal", {}).get("sha256") != sha256_file(
        _resolve_within(root, PROPOSAL)
    ):
        raise DecodeCompatManifestError("Successor authorization proposal binding mismatch")
    if authorization.get("decision") != (
        "authorize_single_upstream_v213_decode_compat_package_only_loop"
    ):
        raise DecodeCompatManifestError("Successor authorization decision mismatch")
    if authorization.get("authorized_stage_budgets") != expected_stage_budgets:
        raise DecodeCompatManifestError("Successor authorization stage budgets drifted")
    if authorization.get("terminal_evidence_contract") != TERMINAL_EVIDENCE_CONTRACT:
        raise DecodeCompatManifestError("Successor authorization terminal evidence contract drifted")
    return {"proposal": proposal, "authorization": authorization}


def build_official_research_manifest(
    project_root: Path, *, generated_at_utc: str
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    timestamp = _validate_timestamp(generated_at_utc)
    _verify_base_chain(root)
    research = load_json_strict(_resolve_within(root, UPSTREAM_RESEARCH))
    if research.get("schema") != "axon_loop28_pytorch_native_upstream_decode_research_v1":
        raise DecodeCompatManifestError("Frozen upstream research schema mismatch")
    if research.get("decision") != (
        "upstream_v213_decoder_fix_identified_successor_still_requires_new_loop"
    ):
        raise DecodeCompatManifestError("Frozen upstream research decision mismatch")
    matching_sources = [
        row
        for row in research.get("sources", [])
        if isinstance(row, dict) and row.get("url") == OFFICIAL_V213_SOURCE_URL
    ]
    if len(matching_sources) != 1 or "errors='replace'" not in str(
        matching_sources[0].get("finding")
    ):
        raise DecodeCompatManifestError("Frozen research lacks the official v2.13 decoder fact")
    facts = research.get("verified_facts", {})
    if facts.get("v2_13_0_contains_decoder_fix") is not True:
        raise DecodeCompatManifestError("Frozen research did not verify the v2.13 decoder fix")
    if _sha256_bytes(OFFICIAL_V213_EXCERPT.encode("utf-8")) != OFFICIAL_V213_EXCERPT_SHA256:
        raise DecodeCompatManifestError("Official v2.13 normalized excerpt constant drifted")
    artifacts = [
        _artifact_record(root, "parent_post", PARENT_POST, "negative_parent_closure"),
        _artifact_record(root, "upstream_research", UPSTREAM_RESEARCH, "frozen_official_research"),
    ]
    return {
        "schema": "axon_loop28_pytorch_native_official_pytorch_v213_source_manifest_v1",
        "loop_id": LOOP_ID,
        "generated_at_utc": timestamp,
        "lineage": {
            "parent_post_sha256": EXPECTED_PARENT_POST_SHA256,
            "upstream_research_sha256": EXPECTED_UPSTREAM_RESEARCH_SHA256,
            "proposal_sha256": sha256_file(_resolve_within(root, PROPOSAL)),
            "authorization_sha256": sha256_file(_resolve_within(root, AUTHORIZATION)),
        },
        "official_source": {
            "repository": "pytorch/pytorch",
            "tag": OFFICIAL_V213_TAG,
            "tag_commit": OFFICIAL_V213_TAG_COMMIT,
            "path": "torch/_inductor/cpp_builder.py",
            "url": OFFICIAL_V213_SOURCE_URL,
            "source_payload_sha256": OFFICIAL_V213_SOURCE_SHA256,
            "normalized_excerpt": OFFICIAL_V213_EXCERPT,
            "normalized_excerpt_sha256": OFFICIAL_V213_EXCERPT_SHA256,
            "symbol": "SUBPROCESS_DECODE_ARGS",
            "semantic_delta": "windows_decode_adds_errors_replace",
            "issue_url": OFFICIAL_ISSUE_URL,
        },
        "contract": {
            "protected_stage_network_requests": 0,
            "installed_torch_mutation_allowed": False,
            "process_local_semantic_mirror_only": True,
            "actual_v213_runtime_qualified": False,
            "production_upgrade_qualified": False,
        },
        "artifacts": artifacts,
        "integrity": {"artifact_count": len(artifacts), "all_required_present": True},
        "decision": "official_pytorch_v213_decoder_source_frozen_for_local_compatibility",
    }


def verify_official_research_manifest(
    project_root: Path, output: Path = OFFICIAL_RESEARCH_MANIFEST
) -> dict[str, Any]:
    payload = load_json_strict(_resolve_within(project_root, output))
    if payload.get("schema") != (
        "axon_loop28_pytorch_native_official_pytorch_v213_source_manifest_v1"
    ):
        raise DecodeCompatManifestError("Official v2.13 source manifest schema mismatch")
    rebuilt = build_official_research_manifest(
        project_root, generated_at_utc=str(payload.get("generated_at_utc") or "")
    )
    if payload != rebuilt:
        raise DecodeCompatManifestError("Official v2.13 source manifest drifted")
    return payload


def build_reused_binaries_manifest(
    project_root: Path, *, generated_at_utc: str
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    timestamp = _validate_timestamp(generated_at_utc)
    _verify_base_chain(root)
    receipt = load_json_strict(_resolve_within(root, PARENT_BUILD_RECEIPT))
    if receipt.get("schema") != "axon_loop28_pytorch_native_cpp_build_receipt_v1":
        raise DecodeCompatManifestError("Parent C++ build receipt schema mismatch")
    if receipt.get("decision") != "lean_aten_and_aoti_hosts_built_package_requires_authorization":
        raise DecodeCompatManifestError("Parent C++ build decision mismatch")
    if receipt.get("execution_count") != 0 or receipt.get("quality_metric_count") != 0:
        raise DecodeCompatManifestError("Parent C++ build receipt exceeds the reuse boundary")
    binary_specs = (
        ("direct_aten_host", "aten", ATEN_HOST, EXPECTED_ATEN_HOST_SHA256, EXPECTED_ATEN_HOST_SIZE),
        ("aoti_host", "aoti", AOTI_HOST, EXPECTED_AOTI_HOST_SHA256, EXPECTED_AOTI_HOST_SIZE),
    )
    binaries: list[dict[str, Any]] = []
    for name, receipt_name, path, digest, size in binary_specs:
        source = receipt.get("binaries", {}).get(receipt_name, {})
        resolved = _resolve_within(root, path)
        if (
            source.get("path") != path.as_posix()
            or source.get("sha256") != digest
            or source.get("size_bytes") != size
            or sha256_file(resolved) != digest
            or resolved.stat().st_size != size
        ):
            raise DecodeCompatManifestError(f"Frozen binary reuse drifted: {name}")
        dependencies = source.get("dependencies")
        if not isinstance(dependencies, list) or any(
            token in str(dependency).casefold()
            for dependency in dependencies
            for token in ("python", "torch_python", "torch.dll", "torch_cuda", "cuda")
        ):
            raise DecodeCompatManifestError(f"Frozen binary dependency boundary drifted: {name}")
        binaries.append(
            {
                "name": name,
                "path": path.as_posix(),
                "sha256": digest,
                "size_bytes": size,
                "dependencies": dependencies,
            }
        )
    artifacts = [
        _artifact_record(root, "parent_build_receipt", PARENT_BUILD_RECEIPT, "build_authority"),
        _artifact_record(root, "direct_aten_host", ATEN_HOST, "reused_frozen_binary"),
        _artifact_record(root, "aoti_host", AOTI_HOST, "reused_frozen_binary"),
    ]
    return {
        "schema": "axon_loop28_pytorch_native_reused_cpp_binaries_manifest_v1",
        "loop_id": LOOP_ID,
        "generated_at_utc": timestamp,
        "lineage": {
            "parent_post_sha256": EXPECTED_PARENT_POST_SHA256,
            "parent_failure_sha256": EXPECTED_PARENT_FAILURE_SHA256,
            "parent_build_receipt_sha256": EXPECTED_PARENT_BUILD_RECEIPT_SHA256,
            "proposal_sha256": sha256_file(_resolve_within(root, PROPOSAL)),
            "authorization_sha256": sha256_file(_resolve_within(root, AUTHORIZATION)),
        },
        "binaries": binaries,
        "authority_boundary": {
            "artifact_hash_reuse_allowed": True,
            "parent_authorization_reuse_allowed": False,
            "parent_ready_or_consumed_lease_reuse_allowed": False,
            "build_or_binary_execution_allowed": False,
            "hash_and_dependency_audit_only": True,
        },
        "artifacts": artifacts,
        "integrity": {"artifact_count": len(artifacts), "all_required_present": True},
        "decision": "frozen_cpp_binaries_verified_artifact_only_reuse",
    }


def verify_reused_binaries_manifest(
    project_root: Path, output: Path = REUSED_BINARIES_MANIFEST
) -> dict[str, Any]:
    payload = load_json_strict(_resolve_within(project_root, output))
    if payload.get("schema") != "axon_loop28_pytorch_native_reused_cpp_binaries_manifest_v1":
        raise DecodeCompatManifestError("Reused C++ binaries manifest schema mismatch")
    rebuilt = build_reused_binaries_manifest(
        project_root, generated_at_utc=str(payload.get("generated_at_utc") or "")
    )
    if payload != rebuilt:
        raise DecodeCompatManifestError("Reused C++ binaries manifest drifted")
    return payload


def _validate_stage_lineage(
    project_root: Path,
    authorization: Mapping[str, Any],
    *,
    expected_budget: Mapping[str, int],
) -> None:
    root = project_root.resolve(strict=True)
    expected = {
        "proposal_sha256": sha256_file(_resolve_within(root, PROPOSAL)),
        "successor_authorization_sha256": sha256_file(_resolve_within(root, AUTHORIZATION)),
        "official_pytorch_v213_source_manifest_sha256": sha256_file(
            _resolve_within(root, OFFICIAL_RESEARCH_MANIFEST)
        ),
        "reused_cpp_binaries_manifest_sha256": sha256_file(
            _resolve_within(root, REUSED_BINARIES_MANIFEST)
        ),
    }
    for field, value in expected.items():
        if authorization.get(field) != value:
            raise DecodeCompatManifestError(f"Stage authorization lineage drifted: {field}")
    if authorization.get("budget") != dict(expected_budget):
        raise DecodeCompatManifestError("Stage authorization budget differs from the frozen contract")


def _counter_value(counters: Mapping[str, Any], name: str) -> int:
    value = counters.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DecodeCompatManifestError(f"Invalid or missing protected counter: {name}")
    return value


def _validate_counter_limits(
    counters: Mapping[str, Any],
    limits: Mapping[str, int],
    *,
    exact: bool,
) -> None:
    for name, limit in limits.items():
        value = _counter_value(counters, name)
        if (exact and value != limit) or (not exact and value > limit):
            relation = "equal" if exact else "not exceed"
            raise DecodeCompatManifestError(
                f"Protected counter {name} must {relation} its frozen limit {limit}"
            )


def _validate_budget_actual(
    terminal: Mapping[str, Any], authorization_budget: Mapping[str, Any], *, success: bool
) -> dict[str, int | float]:
    actual = terminal.get("budget_actual")
    if not isinstance(actual, dict):
        raise DecodeCompatManifestError("Package terminal receipt omitted budget_actual")
    mappings = {
        "worker_processes": "worker_processes",
        "vcvars_activations": "vcvars_activations",
        "compiler_help_processes": "compiler_help_processes_max",
        "dumpbin_processes": "dumpbin_processes_max",
        "wall_clock_seconds": "wall_clock_seconds_max",
        "retained_output_bytes": "max_retained_output_bytes",
    }
    for actual_name, limit_name in mappings.items():
        value = actual.get(actual_name)
        limit = authorization_budget.get(limit_name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            or not isinstance(limit, int)
            or value > limit
        ):
            raise DecodeCompatManifestError(
                f"Package budget reconciliation failed: {actual_name}/{limit_name}"
            )
    launch = terminal.get("launch")
    worker = terminal.get("worker")
    if not isinstance(launch, dict) or not isinstance(worker, dict):
        raise DecodeCompatManifestError("Package budget lacks launch or worker evidence")
    process_started = launch.get("process_started")
    if not isinstance(process_started, bool):
        raise DecodeCompatManifestError("Package launch process_started must be boolean")
    expected_processes = 1 if process_started else 0
    if (
        actual.get("worker_processes") != expected_processes
        or actual.get("vcvars_activations") != expected_processes
    ):
        raise DecodeCompatManifestError("Package process budget does not match launch evidence")
    counters = worker.get("counters")
    if not isinstance(counters, dict):
        raise DecodeCompatManifestError("Package budget lacks worker counters")
    if actual.get("compiler_help_processes") != _counter_value(
        counters, "compiler_help_processes"
    ):
        raise DecodeCompatManifestError("Package compiler-help budget drifted from worker counters")
    if actual.get("dumpbin_processes", 0) < _counter_value(counters, "dumpbin_processes"):
        raise DecodeCompatManifestError("Package dumpbin budget understates worker telemetry")
    if success and (
        actual.get("worker_processes") != 1 or actual.get("vcvars_activations") != 1
    ):
        raise DecodeCompatManifestError("Package terminal must use one worker and one vcvars activation")
    return actual


def _validate_preflight_budget_actual(
    terminal: Mapping[str, Any], authorization_budget: Mapping[str, Any], *, success: bool
) -> dict[str, int | float]:
    actual = terminal.get("budget_actual")
    if not isinstance(actual, dict):
        raise DecodeCompatManifestError("Preflight terminal receipt omitted budget_actual")
    checks = {
        "worker_processes": authorization_budget.get("worker_processes"),
        "vcvars_activations": authorization_budget.get("vcvars_activations"),
        "compiler_help_processes": authorization_budget.get("compiler_help_processes_max"),
        "wall_clock_seconds": authorization_budget.get("wall_clock_seconds_max"),
    }
    for name, limit in checks.items():
        value = actual.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            or not isinstance(limit, int)
            or value > limit
        ):
            raise DecodeCompatManifestError(f"Preflight budget reconciliation failed: {name}")
    if actual.get("dumpbin_processes") != 0 or actual.get("retained_output_bytes") != 0:
        raise DecodeCompatManifestError("Preflight retained or dumpbin budget must remain zero")
    launch = terminal.get("launch")
    worker = terminal.get("worker")
    if not isinstance(launch, dict) or not isinstance(worker, dict):
        raise DecodeCompatManifestError("Preflight budget lacks launch or worker evidence")
    process_started = launch.get("process_started")
    if not isinstance(process_started, bool):
        raise DecodeCompatManifestError("Preflight launch process_started must be boolean")
    expected_processes = 1 if process_started else 0
    if (
        actual.get("worker_processes") != expected_processes
        or actual.get("vcvars_activations") != expected_processes
    ):
        raise DecodeCompatManifestError("Preflight process budget does not match launch evidence")
    counters = worker.get("counters")
    if not isinstance(counters, dict):
        raise DecodeCompatManifestError("Preflight budget lacks worker counters")
    if actual.get("compiler_help_processes") != _counter_value(
        counters, "compiler_help_processes"
    ):
        raise DecodeCompatManifestError("Preflight compiler-help budget drifted from worker counters")
    if success and (
        actual.get("worker_processes") != 1 or actual.get("vcvars_activations") != 1
    ):
        raise DecodeCompatManifestError("Passing preflight must use one worker and one vcvars activation")
    return actual


def _validate_launch_integrity(
    terminal: Mapping[str, Any], *, success: bool
) -> dict[str, Any]:
    launch = terminal.get("launch")
    if not isinstance(launch, dict):
        raise DecodeCompatManifestError("Stage terminal receipt omitted launch evidence")
    cleanup = launch.get("temp_cleanup")
    code_pages = launch.get("console_code_pages")
    if not isinstance(cleanup, dict) or cleanup.get("removed") is not True:
        raise DecodeCompatManifestError("Worker TEMP cleanup was not proven")
    if not isinstance(code_pages, dict):
        raise DecodeCompatManifestError("Worker console-code-page evidence is missing")
    process_started = launch.get("process_started")
    if not isinstance(process_started, bool):
        raise DecodeCompatManifestError("Worker launch process_started must be boolean")
    configured = code_pages.get("configured")
    if success or process_started:
        if configured != {"input": 65001, "output": 65001}:
            raise DecodeCompatManifestError("Worker console code pages were not configured to UTF-8")
    elif configured not in (None, {"input": 65001, "output": 65001}):
        raise DecodeCompatManifestError("Pre-launch console code-page evidence drifted")
    if (
        code_pages.get("restore_verified") is not True
        or code_pages.get("restored") != code_pages.get("before")
    ):
        raise DecodeCompatManifestError("Worker console code pages were not restored")
    work_cleanup = terminal.get("work_root_cleanup")
    if not isinstance(work_cleanup, dict) or work_cleanup.get("removed") is not True:
        raise DecodeCompatManifestError("Worker work-root cleanup was not proven")
    if terminal.get("work_root_cleanup_error") not in {None, ""}:
        raise DecodeCompatManifestError("Worker work-root cleanup reported an error")
    job = launch.get("windows_job_object")
    if job is not None:
        if not isinstance(job, dict) or set(job) != {
            "created",
            "assigned",
            "gate_released",
            "active_processes_after",
            "closed",
        }:
            raise DecodeCompatManifestError("Worker Job Object evidence schema drifted")
        for name in ("created", "assigned", "gate_released", "closed"):
            if not isinstance(job.get(name), bool):
                raise DecodeCompatManifestError(f"Worker Job Object flag is invalid: {name}")
        active_after = job.get("active_processes_after")
        if active_after is not None and (
            not isinstance(active_after, int)
            or isinstance(active_after, bool)
            or active_after < 0
        ):
            raise DecodeCompatManifestError("Worker Job Object active count is invalid")
        if job["assigned"] and not job["created"]:
            raise DecodeCompatManifestError("Worker Job Object was assigned before creation")
        if job["gate_released"] and not job["assigned"]:
            raise DecodeCompatManifestError("Worker gate was released before Job Object assignment")
    if process_started:
        if not isinstance(job, dict) or job.get("created") is not True:
            raise DecodeCompatManifestError("Started worker lacks a created Job Object")
        if job.get("closed") is not True:
            raise DecodeCompatManifestError("Started worker Job Object was not closed")
    else:
        if (
            launch.get("returncode") is not None
            or launch.get("timed_out") is not False
            or launch.get("process_tree_termination") is not None
            or launch.get("launch_error") in {None, ""}
        ):
            raise DecodeCompatManifestError("Pre-launch failure evidence drifted")
        if isinstance(job, dict) and (
            job.get("assigned") is not False
            or job.get("gate_released") is not False
            or job.get("active_processes_after") not in {None, 0}
            or job.get("closed") is not job.get("created")
        ):
            raise DecodeCompatManifestError("Pre-launch Job Object evidence drifted")
        journal = launch.get("durable_worker_journal")
        worker = terminal.get("worker")
        if not isinstance(journal, dict) or not isinstance(worker, dict):
            raise DecodeCompatManifestError("Pre-launch failure omitted zero-state evidence")
        journal_counters = journal.get("last_counters")
        worker_counters = worker.get("counters")
        if (
            journal.get("record_count") != 0
            or journal.get("records") != []
            or journal.get("integrity_error") not in {None, ""}
            or not isinstance(journal_counters, dict)
            or not isinstance(worker_counters, dict)
            or set(journal_counters) != set(WORKER_COUNTER_NAMES)
            or set(worker_counters) != set(WORKER_COUNTER_NAMES)
            or journal_counters != worker_counters
            or any(_counter_value(worker_counters, name) for name in WORKER_COUNTER_NAMES)
        ):
            raise DecodeCompatManifestError("Pre-launch failure is not a proven zero state")
    if success:
        if (
            process_started is not True
            or launch.get("returncode") != 0
            or launch.get("timed_out") is not False
            or launch.get("launch_error") is not None
            or launch.get("worker_parse_error") is not None
            or launch.get("process_tree_termination") is not None
        ):
            raise DecodeCompatManifestError("Passing worker launch integrity failed")
        if (
            not isinstance(job, dict)
            or job.get("assigned") is not True
            or job.get("gate_released") is not True
            or job.get("active_processes_after") != 0
            or job.get("closed") is not True
        ):
            raise DecodeCompatManifestError("Passing worker Job Object containment failed")
    termination = launch.get("process_tree_termination")
    if termination is not None and (
        not isinstance(termination, dict)
        or termination.get("tree_termination_requested") is not True
        or termination.get("tree_termination_confirmed") is not True
    ):
        raise DecodeCompatManifestError("Worker process-tree termination proof is invalid")
    if process_started and not success:
        if termination is None:
            if (
                not isinstance(job, dict)
                or job.get("assigned") is not True
                or job.get("gate_released") is not True
                or job.get("active_processes_after") != 0
            ):
                raise DecodeCompatManifestError("Failed worker escaped Job Object containment")
        elif termination.get("method") == "windows_job_object_terminate":
            if (
                termination.get("job_assigned") is not True
                or termination.get("active_processes_after") != 0
                or not isinstance(job, dict)
                or job.get("assigned") is not True
                or job.get("active_processes_after") != 0
            ):
                raise DecodeCompatManifestError("Job Object termination evidence drifted")
        elif termination.get("method") == "taskkill_before_job_gate_release":
            journal = launch.get("durable_worker_journal")
            worker = terminal.get("worker")
            journal_counters = journal.get("last_counters") if isinstance(journal, dict) else None
            worker_counters = worker.get("counters") if isinstance(worker, dict) else None
            if (
                not isinstance(job, dict)
                or job.get("assigned") is not False
                or job.get("gate_released") is not False
                or termination.get("job_assigned") is not False
                or not isinstance(journal, dict)
                or journal.get("record_count") != 0
                or journal.get("records") != []
                or not isinstance(journal_counters, dict)
                or not isinstance(worker_counters, dict)
                or journal_counters != worker_counters
                or any(
                    _counter_value(worker_counters, name) for name in WORKER_COUNTER_NAMES
                )
            ):
                raise DecodeCompatManifestError("Unsafe taskkill fallback evidence")
        else:
            raise DecodeCompatManifestError("Unknown process-tree termination method")
    if not success and launch.get("timed_out") is True:
        termination = launch.get("process_tree_termination")
        if (
            not isinstance(termination, dict)
            or termination.get("tree_termination_requested") is not True
            or termination.get("tree_termination_confirmed") is not True
            or termination.get("method") != "windows_job_object_terminate"
            or termination.get("job_assigned") is not True
            or termination.get("active_processes_after") != 0
            or not isinstance(job, dict)
            or job.get("assigned") is not True
            or job.get("gate_released") is not True
            or job.get("active_processes_after") != 0
        ):
            raise DecodeCompatManifestError("Timed-out worker lacks process-tree termination proof")
    return launch


def _validate_source_snapshots(
    terminal: Mapping[str, Any], source_records: Sequence[Mapping[str, Any]], *, success: bool
) -> None:
    del success
    expected = list(source_records)
    if terminal.get("source_artifacts_before") != expected:
        raise DecodeCompatManifestError("Stage source-before snapshot drifted")
    if terminal.get("source_artifacts_after") != expected:
        raise DecodeCompatManifestError("Stage source-after snapshot drifted")


def _validate_durable_worker_journal(
    terminal: Mapping[str, Any],
    worker: Mapping[str, Any],
    *,
    stage: str,
    success: bool,
) -> bool:
    launch = terminal.get("launch")
    if not isinstance(launch, dict):
        raise DecodeCompatManifestError("Worker journal lacks launch evidence")
    journal = launch.get("durable_worker_journal")
    if not isinstance(journal, dict):
        raise DecodeCompatManifestError("Worker durable journal evidence is missing")
    if journal.get("integrity_error") not in {None, ""}:
        raise DecodeCompatManifestError("Worker durable journal integrity failed")
    records = journal.get("records")
    record_count = journal.get("record_count")
    last_counters = journal.get("last_counters")
    if (
        not isinstance(records, list)
        or not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count != len(records)
        or not isinstance(last_counters, dict)
    ):
        raise DecodeCompatManifestError("Worker durable journal summary is invalid")
    if set(last_counters) != set(WORKER_COUNTER_NAMES):
        raise DecodeCompatManifestError("Worker durable journal counter schema drifted")
    normalized_last = {
        name: _counter_value(last_counters, name) for name in WORKER_COUNTER_NAMES
    }
    previous_sha256: str | None = None
    previous_counters = {name: 0 for name in WORKER_COUNTER_NAMES}
    for sequence, record in enumerate(records, start=1):
        if not isinstance(record, dict) or set(record) != {
            "schema",
            "loop_id",
            "stage",
            "sequence",
            "event",
            "previous_record_sha256",
            "counters",
            "record_sha256",
        }:
            raise DecodeCompatManifestError("Worker durable journal record schema drifted")
        counters = record.get("counters")
        if not isinstance(counters, dict) or set(counters) != set(WORKER_COUNTER_NAMES):
            raise DecodeCompatManifestError("Worker durable journal record counters drifted")
        normalized = {name: _counter_value(counters, name) for name in WORKER_COUNTER_NAMES}
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        if (
            record.get("schema") != "axon_loop28_pytorch_native_decode_worker_event_v1"
            or record.get("loop_id") != LOOP_ID
            or record.get("stage") != stage
            or record.get("sequence") != sequence
            or not isinstance(record.get("event"), str)
            or not record.get("event")
            or record.get("previous_record_sha256") != previous_sha256
            or record.get("record_sha256") != _canonical_sha256(body)
        ):
            raise DecodeCompatManifestError("Worker durable journal hash chain drifted")
        if any(normalized[name] < previous_counters[name] for name in WORKER_COUNTER_NAMES):
            raise DecodeCompatManifestError("Worker durable journal counters regressed")
        previous_sha256 = str(record["record_sha256"])
        previous_counters = normalized
    if records:
        if records[0].get("event") != "worker_started" or any(
            records[0]["counters"].values()
        ):
            raise DecodeCompatManifestError("Worker durable journal start record drifted")
        if normalized_last != previous_counters:
            raise DecodeCompatManifestError("Worker durable journal last counters drifted")
        if journal.get("last_event") != records[-1].get("event"):
            raise DecodeCompatManifestError("Worker durable journal last event drifted")
        if journal.get("last_record_sha256") != previous_sha256:
            raise DecodeCompatManifestError("Worker durable journal terminal hash drifted")
    elif (
        any(normalized_last.values())
        or journal.get("last_event") not in {None, ""}
        or journal.get("last_record_sha256") not in {None, ""}
    ):
        raise DecodeCompatManifestError("Empty worker journal contains terminal state")
    if worker.get("counters") != last_counters:
        raise DecodeCompatManifestError("Worker counters differ from durable journal")
    if worker.get("durable_journal_last_record_sha256") != previous_sha256:
        raise DecodeCompatManifestError("Worker receipt journal hash drifted")
    reconstructed = worker.get("receipt_reconstructed_from_journal")
    if not isinstance(reconstructed, bool):
        raise DecodeCompatManifestError("Worker journal reconstruction flag is invalid")
    if reconstructed:
        if success or not records or worker.get("status") != "failed":
            raise DecodeCompatManifestError("Worker journal reconstruction boundary drifted")
        if launch.get("process_started") is not True:
            raise DecodeCompatManifestError("Reconstructed worker was never launched")
        if not (
            launch.get("timed_out") is True
            or launch.get("returncode") not in {None, 0}
            or launch.get("launch_error") not in {None, ""}
            or launch.get("worker_parse_error") not in {None, ""}
        ):
            raise DecodeCompatManifestError("Reconstructed worker lacks termination evidence")
    elif not records and any(_counter_value(worker["counters"], name) for name in WORKER_COUNTER_NAMES):
        raise DecodeCompatManifestError("Journal-free worker receipt contains nonzero counters")
    return reconstructed


def _validate_worker_runtime(
    worker: Mapping[str, Any], *, success: bool, reconstructed: bool = False
) -> None:
    counters = worker.get("counters")
    if not isinstance(counters, dict):
        raise DecodeCompatManifestError("Worker runtime evidence omitted counters")
    torch_imported = _counter_value(counters, "torch_imports") > 0
    environment = worker.get("environment")
    if not reconstructed and (success or torch_imported):
        if not isinstance(environment, dict):
            raise DecodeCompatManifestError("Worker runtime evidence omitted environment")
        if environment.get("python_executable_sha256") != EXPECTED_PYTHON_SHA256:
            raise DecodeCompatManifestError("Worker Python hash drifted")
        if environment.get("utf8_mode") != 0:
            raise DecodeCompatManifestError("Worker Python UTF-8 mode drifted")
        if str(environment.get("preferred_encoding", "")).casefold() not in {"cp936", "gbk"}:
            raise DecodeCompatManifestError("Worker preferred encoding drifted")
        if environment.get("console_code_pages") != {"input": 65001, "output": 65001}:
            raise DecodeCompatManifestError("Worker console code-page contract drifted")
        if environment.get("torchinductor_compile_threads_env") != 1:
            raise DecodeCompatManifestError("Worker Inductor compile-thread contract drifted")
        if environment.get("torchinductor_autotune_in_subproc_env") != "0":
            raise DecodeCompatManifestError("Worker Inductor autotune subprocess contract drifted")
    torch_record = worker.get("torch")
    if not reconstructed and (success or torch_imported):
        if not isinstance(torch_record, dict):
            raise DecodeCompatManifestError("Worker runtime evidence omitted Torch metadata")
        if torch_record.get("version") != EXPECTED_TORCH_VERSION:
            raise DecodeCompatManifestError("Worker Torch version drifted")
        for name in ("cuda_initialized_before", "cuda_initialized_after"):
            if torch_record.get(name) is not False:
                raise DecodeCompatManifestError(f"Worker CUDA boundary drifted: {name}")
        if torch_record.get("cpp_builder_sha256_after") != EXPECTED_CPP_BUILDER_SHA256:
            raise DecodeCompatManifestError("Worker cpp_builder hash drifted")
        if torch_record.get("cpu_vec_isa_sha256_after") != EXPECTED_CPU_VEC_ISA_SHA256:
            raise DecodeCompatManifestError("Worker cpu_vec_isa hash drifted")
    shim = worker.get("shim")
    if not reconstructed and (success or torch_imported):
        if not isinstance(shim, dict):
            raise DecodeCompatManifestError("Worker runtime evidence omitted decoder shim")
        if (
            shim.get("process_local") is not True
            or shim.get("installed_file_modified") is not False
            or shim.get("installed_file_sha256_before") != EXPECTED_CPP_BUILDER_SHA256
            or shim.get("after_args", [])[-1:] != ["replace"]
        ):
            raise DecodeCompatManifestError("Worker decoder shim boundary drifted")
        watched = shim.get("watched_modules_before")
        compiler_after = shim.get("compiler_modules_after_torch_import")
        caches = shim.get("cache_sizes_before")
        if (
            not isinstance(watched, dict)
            or any(watched.values())
            or not isinstance(compiler_after, dict)
            or any(compiler_after.values())
            or not isinstance(caches, dict)
            or any(caches.values())
        ):
            raise DecodeCompatManifestError("Worker import/cache precondition drifted")
    telemetry = worker.get("process_telemetry")
    if not isinstance(telemetry, dict):
        raise DecodeCompatManifestError("Worker runtime evidence omitted process telemetry")
    telemetry_names = (
        "total_subprocesses",
        "compiler_processes",
        "compiler_help_processes",
        "dumpbin_processes",
    )
    for name in telemetry_names:
        value = telemetry.get(name)
        if not isinstance(value, int) or value < 0:
            raise DecodeCompatManifestError(f"Worker telemetry counter is invalid: {name}")
    for name in ("compiler_processes", "compiler_help_processes", "dumpbin_processes"):
        if telemetry.get(name) != _counter_value(counters, name):
            raise DecodeCompatManifestError(f"Worker process telemetry drifted: {name}")
    if telemetry["total_subprocesses"] < max(
        telemetry["compiler_processes"],
        telemetry["compiler_help_processes"],
        telemetry["dumpbin_processes"],
    ):
        raise DecodeCompatManifestError("Worker total subprocess telemetry is inconsistent")


def _classify_package_failure(failure: Mapping[str, Any]) -> str:
    worker = failure.get("worker")
    if not isinstance(worker, dict):
        raise DecodeCompatManifestError("Package failure omitted the structured worker receipt")
    counters = worker.get("counters")
    if not isinstance(counters, dict):
        raise DecodeCompatManifestError("Package failure omitted worker counters")
    _validate_counter_limits(counters, PACKAGE_COUNTER_LIMITS, exact=False)
    launch = failure.get("launch")
    error = str(failure.get("error") or "").casefold()
    if (isinstance(launch, dict) and launch.get("timed_out") is True) or any(
        token in error
        for token in ("budget", "deadline", "timed out", "timeout")
    ):
        return "budget"
    status = worker.get("status")
    if status == "passed" or _counter_value(counters, "aoti_compile_and_package_completed") == 1:
        dependency_tokens = ("dependency", "forbidden", "unresolved", "ambiguous")
        return "dependency" if any(token in error for token in dependency_tokens) else "static_audit"
    if _counter_value(counters, "aoti_compile_and_package_calls") > 0:
        return "protected_call"
    if any(
        _counter_value(counters, name) > 0
        for name in ("torch_imports", "model_constructions", "torch_export_calls")
    ):
        return "pre_export"
    return "administrative"


def _expected_failure_decision(failure_class: str) -> str:
    decisions = {
        "administrative": "administrative_failure_no_protected_package_call",
        "pre_export": "decode_compat_pre_export_failure_no_package",
        "protected_call": "decode_compat_applied_aoti_compile_or_package_still_unsupported",
        "dependency": "decode_compat_package_dependency_leakage_no_load",
        "static_audit": "decode_compat_package_static_audit_failed_no_load",
        "budget": "budget_exhausted_no_claim",
    }
    try:
        return decisions[failure_class]
    except KeyError as exc:
        raise DecodeCompatManifestError(f"Unknown package failure class: {failure_class}") from exc


def _validate_partial_artifacts(
    project_root: Path, records: object
) -> list[tuple[str, Path]]:
    if not isinstance(records, list):
        raise DecodeCompatManifestError("Package partial_artifacts must be a list")
    root = project_root.resolve(strict=True)
    seen: set[str] = set()
    validated: list[tuple[str, Path]] = []
    artifact_parts = ARTIFACT_ROOT.parts
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise DecodeCompatManifestError("Package partial artifact must be an object")
        raw_path = str(record.get("path") or "")
        pure = PurePosixPath(raw_path)
        if pure.is_absolute() or ".." in pure.parts or tuple(pure.parts[: len(artifact_parts)]) != (
            artifact_parts
        ):
            raise DecodeCompatManifestError("Package partial artifact escapes the successor root")
        normalized = Path(*pure.parts)
        if normalized.as_posix() in seen:
            raise DecodeCompatManifestError("Duplicate package partial artifact path")
        seen.add(normalized.as_posix())
        current = root
        for part in normalized.parts:
            current /= part
            if current.is_symlink():
                raise DecodeCompatManifestError("Package partial artifact uses a symlinked path")
        resolved = _resolve_within(root, normalized)
        expected_root = (root / ARTIFACT_ROOT).resolve(strict=True)
        if not resolved.is_relative_to(expected_root):
            raise DecodeCompatManifestError("Package partial artifact resolves outside its root")
        digest = sha256_file(resolved)
        size = resolved.stat().st_size
        if record.get("sha256") != digest or record.get("size_bytes") != size:
            raise DecodeCompatManifestError("Package partial artifact record drifted")
        if digest == PARENT_PARTIAL_INPUT_SHA256:
            raise DecodeCompatManifestError("Package failure attempted to bind the parent partial input")
        validated.append((f"partial_{index}", normalized))
    return validated


def _validate_failure_partials(
    project_root: Path, failure: Mapping[str, Any]
) -> list[tuple[str, Path]]:
    if failure.get("partial_inventory_error") not in {None, ""}:
        raise DecodeCompatManifestError(
            "Decode package partial inventory is incomplete or unsafe"
        )
    return _validate_partial_artifacts(project_root, failure.get("partial_artifacts"))


def build_preflight_manifest(project_root: Path, *, generated_at_utc: str) -> dict[str, Any]:
    # preflight 成功或失败都必须形成终态；失败不能悬空，也不能继续生成 implementation。
    root = project_root.resolve(strict=True)
    timestamp = _validate_timestamp(generated_at_utc)
    chain = _verify_base_chain(root)
    verify_official_research_manifest(root)
    verify_reused_binaries_manifest(root)
    success_exists = (root / PREFLIGHT_EVIDENCE).is_file()
    failure_exists = (root / PREFLIGHT_FAILURE).is_file()
    if success_exists == failure_exists:
        raise DecodeCompatManifestError("Exactly one decode preflight terminal receipt must exist")
    terminal_path = PREFLIGHT_EVIDENCE if success_exists else PREFLIGHT_FAILURE
    terminal = load_json_strict(_resolve_within(root, terminal_path))
    preflight_authorization, final_lease = _validate_stage_chain(
        root,
        authorization_path=PREFLIGHT_AUTHORIZATION,
        final_lease_path=PREFLIGHT_FINAL_LEASE,
        authorization_schema=PREFLIGHT_AUTHORIZATION_SCHEMA,
        authorization_decision=PREFLIGHT_AUTHORIZATION_DECISION,
        lease_schema=PREFLIGHT_LEASE_SCHEMA,
        attempt_id=PREFLIGHT_ATTEMPT_ID,
        mode="preflight",
        work_root=PREFLIGHT_WORK_ROOT,
        output_paths=(PREFLIGHT_EVIDENCE, PREFLIGHT_FAILURE),
    )
    _validate_stage_lineage(root, preflight_authorization, expected_budget=PREFLIGHT_BUDGET)
    source_records = _verify_source_binding(
        root,
        preflight_authorization,
        embedded_authorization=terminal.get("lease", {}).get("authorization"),
    )
    _validate_evidence_lease(
        terminal,
        authorization=preflight_authorization,
        authorization_path=PREFLIGHT_AUTHORIZATION,
        final_lease=final_lease,
        final_lease_path=PREFLIGHT_FINAL_LEASE,
        project_root=root,
    )
    worker = terminal.get("worker")
    if not isinstance(worker, dict):
        raise DecodeCompatManifestError("Decode preflight terminal receipt omitted worker evidence")
    _validate_launch_integrity(terminal, success=success_exists)
    _validate_source_snapshots(terminal, source_records, success=success_exists)
    reconstructed = _validate_durable_worker_journal(
        terminal,
        worker,
        stage="preflight",
        success=success_exists,
    )
    _validate_worker_runtime(
        worker,
        success=success_exists,
        reconstructed=reconstructed,
    )
    budget_actual = _validate_preflight_budget_actual(
        terminal,
        preflight_authorization["budget"],
        success=success_exists,
    )
    if (root / ARTIFACT_ROOT).exists():
        raise DecodeCompatManifestError("Decode preflight created the package artifact root")
    counters = worker.get("counters")
    if not isinstance(counters, dict):
        raise DecodeCompatManifestError("Decode preflight worker omitted counters")
    if _counter_value(counters, "torch_imports") > PREFLIGHT_BUDGET["torch_imports"]:
        raise DecodeCompatManifestError("Decode preflight exceeded the Torch import budget")
    if _counter_value(counters, "compiler_help_processes") > PREFLIGHT_BUDGET[
        "compiler_help_processes_max"
    ]:
        raise DecodeCompatManifestError("Decode preflight exceeded its compiler probe budget")
    for counter in ZERO_SCIENTIFIC_COUNTERS:
        if _counter_value(counters, counter) != 0:
            raise DecodeCompatManifestError(f"Decode preflight counter drifted: {counter}")

    if success_exists:
        if terminal.get("schema") != "axon_loop28_pytorch_native_decode_probe_evidence_v1":
            raise DecodeCompatManifestError("Decode preflight evidence schema mismatch")
        if terminal.get("decision") != (
            "upstream_v213_process_local_decode_preflight_passed_"
            "package_implementation_may_freeze"
        ):
            raise DecodeCompatManifestError("Decode preflight evidence did not pass")
        if worker.get("status") != "passed" or worker.get("decision") != (
            "upstream_v213_process_local_decode_preflight_passed"
        ):
            raise DecodeCompatManifestError("Decode preflight worker decision mismatch")
        if _counter_value(counters, "torch_imports") != 1:
            raise DecodeCompatManifestError("Decode preflight did not use exactly one Torch import")
        if _counter_value(counters, "compiler_help_processes") != 3:
            raise DecodeCompatManifestError(
                "Decode preflight did not use exactly three compiler-help processes"
            )
        environment = worker.get("environment", {})
        shim = worker.get("shim", {})
        compiler = worker.get("compiler", {})
        if environment.get("utf8_mode") != 0:
            raise DecodeCompatManifestError("Decode preflight worker UTF-8 mode drifted")
        if str(environment.get("preferred_encoding", "")).casefold() not in {"cp936", "gbk"}:
            raise DecodeCompatManifestError("Decode preflight preferred encoding drifted")
        if environment.get("console_code_pages") != {"input": 65001, "output": 65001}:
            raise DecodeCompatManifestError("Decode preflight console code pages drifted")
        if shim.get("after_args")[-1:] != ["replace"] or shim.get("process_local") is not True:
            raise DecodeCompatManifestError("Decode preflight did not mirror v2.13")
        if (
            compiler.get("decode_matrix", {}).get("v212_strict_preferred", {}).get("passed")
            is not False
        ):
            raise DecodeCompatManifestError("Decode preflight did not reproduce v2.12 failure")
        if (
            compiler.get("decode_matrix", {}).get("v213_preferred_replace", {}).get("passed")
            is not True
        ):
            raise DecodeCompatManifestError("Decode preflight v2.13 control failed")
        schema = "axon_loop28_pytorch_native_decode_compat_preflight_manifest_v1"
        decision = "decode_compat_preflight_frozen_implementation_may_freeze"
        preflight = {
            "passed": True,
            "selected_strategy": "default_locale_cp936_console65001_v213_replace",
            "v212_failure_reproduced": True,
            "v213_process_local_behavior_passed": True,
            "installed_torch_files_modified": False,
            "artifact_root_created": False,
            "package_authorization_allowed": True,
            "package_call_count": 0,
            "package_load_count": 0,
        }
    else:
        if terminal.get("schema") != "axon_loop28_pytorch_native_decode_probe_failure_v1":
            raise DecodeCompatManifestError("Decode preflight failure schema mismatch")
        if terminal.get("decision") != "decode_compat_preflight_failed_no_package_authorization":
            raise DecodeCompatManifestError("Decode preflight failure decision mismatch")
        forbidden_outputs = (
            IMPLEMENTATION_MANIFEST,
            PACKAGE_AUTHORIZATION,
            PACKAGE_FINAL_LEASE,
            PACKAGE_RECEIPT,
            PACKAGE_FAILURE,
            PACKAGE_MANIFEST,
        )
        if any((root / path).exists() for path in forbidden_outputs):
            raise DecodeCompatManifestError("Failed preflight was followed by a forbidden package stage")
        schema = "axon_loop28_pytorch_native_decode_compat_preflight_failure_manifest_v1"
        decision = "decode_compat_preflight_failed_no_package_authorization"
        preflight = {
            "passed": False,
            "failure_class": "decode_or_environment_preflight",
            "installed_torch_files_modified": False,
            "artifact_root_created": False,
            "package_authorization_allowed": False,
            "package_call_count": 0,
            "package_load_count": 0,
            "error_type": terminal.get("error_type"),
            "error": terminal.get("error"),
        }
    artifact_specs = (
        ("proposal", PROPOSAL, "successor_proposal"),
        ("authorization", AUTHORIZATION, "successor_authorization"),
        (
            "official_pytorch_v213_source_manifest",
            OFFICIAL_RESEARCH_MANIFEST,
            "official_source_contract",
        ),
        ("reused_cpp_binaries_manifest", REUSED_BINARIES_MANIFEST, "artifact_reuse_contract"),
        ("preflight_authorization", PREFLIGHT_AUTHORIZATION, "decode_probe_authority"),
        ("preflight_final_lease", PREFLIGHT_FINAL_LEASE, "consumed_decode_probe_lease"),
        ("preflight_terminal", terminal_path, "decode_probe_terminal_result"),
        ("runner", RUNNER, "decode_compat_runner"),
        ("runner_test", RUNNER_TEST, "runner_contract_tests"),
        ("manifest_builder", BUILDER, "successor_manifest_builder"),
        ("manifest_builder_test", BUILDER_TEST, "manifest_builder_tests"),
        ("base_model_source", BASE_MODEL_SOURCE, "frozen_tiny_model_constructor"),
        ("parent_safety_source", PARENT_SAFETY_SOURCE, "frozen_archive_safety_parent"),
        ("aten_host", ATEN_HOST, "reused_direct_aten_host"),
        ("aoti_host", AOTI_HOST, "reused_aoti_host"),
    )
    artifacts = [_artifact_record(root, name, path, role) for name, path, role in artifact_specs]
    claim_boundary = dict(chain["authorization"]["claim_boundary"])
    claim_boundary["decoder_compatibility_claim_allowed"] = success_exists
    claim_boundary["tiny_package_generation_claim_allowed"] = False
    claim_boundary["package_load_allowed"] = False
    return {
        "schema": schema,
        "loop_id": LOOP_ID,
        "generated_at_utc": timestamp,
        "lineage": {
            "parent_post_sha256": EXPECTED_PARENT_POST_SHA256,
            "parent_failure_sha256": EXPECTED_PARENT_FAILURE_SHA256,
            "upstream_research_sha256": EXPECTED_UPSTREAM_RESEARCH_SHA256,
            "proposal_sha256": sha256_file(_resolve_within(root, PROPOSAL)),
            "authorization_sha256": sha256_file(_resolve_within(root, AUTHORIZATION)),
            "official_pytorch_v213_source_manifest_sha256": sha256_file(
                _resolve_within(root, OFFICIAL_RESEARCH_MANIFEST)
            ),
            "reused_cpp_binaries_manifest_sha256": sha256_file(
                _resolve_within(root, REUSED_BINARIES_MANIFEST)
            ),
            "preflight_authorization_sha256": sha256_file(
                _resolve_within(root, PREFLIGHT_AUTHORIZATION)
            ),
            "preflight_final_lease_sha256": sha256_file(
                _resolve_within(root, PREFLIGHT_FINAL_LEASE)
            ),
            "preflight_terminal_sha256": sha256_file(_resolve_within(root, terminal_path)),
            "attempt_id": PREFLIGHT_ATTEMPT_ID,
        },
        "source_binding": {
            "artifact_count": len(source_records),
            "records_sha256": _canonical_sha256({"records": source_records}),
            "authorization_evidence_current_exact_match": True,
        },
        "budget": {
            "authorized": preflight_authorization["budget"],
            "actual": budget_actual,
            "within_budget": True,
        },
        "preflight": preflight,
        "artifacts": artifacts,
        "integrity": {"artifact_count": len(artifacts), "all_required_present": True},
        "claim_boundary": claim_boundary,
        "decision": decision,
    }


def verify_preflight_manifest(
    project_root: Path, output: Path = PREFLIGHT_MANIFEST
) -> dict[str, Any]:
    payload = load_json_strict(_resolve_within(project_root, output))
    if payload.get("schema") not in {
        "axon_loop28_pytorch_native_decode_compat_preflight_manifest_v1",
        "axon_loop28_pytorch_native_decode_compat_preflight_failure_manifest_v1",
    }:
        raise DecodeCompatManifestError("Decode preflight manifest schema mismatch")
    rebuilt = build_preflight_manifest(
        project_root, generated_at_utc=str(payload.get("generated_at_utc") or "")
    )
    if payload != rebuilt:
        raise DecodeCompatManifestError("Decode preflight manifest no longer matches evidence")
    return payload


def build_implementation_manifest(project_root: Path, *, generated_at_utc: str) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    timestamp = _validate_timestamp(generated_at_utc)
    chain = _verify_base_chain(root)
    preflight = verify_preflight_manifest(root)
    if preflight.get("schema") != "axon_loop28_pytorch_native_decode_compat_preflight_manifest_v1":
        raise DecodeCompatManifestError("Failed decode preflight cannot freeze implementation")
    if preflight.get("decision") != "decode_compat_preflight_frozen_implementation_may_freeze":
        raise DecodeCompatManifestError("Decode preflight did not authorize implementation freeze")
    artifact_specs = (
        ("proposal", PROPOSAL, "successor_proposal"),
        ("authorization", AUTHORIZATION, "successor_authorization"),
        (
            "official_pytorch_v213_source_manifest",
            OFFICIAL_RESEARCH_MANIFEST,
            "official_source_contract",
        ),
        ("reused_cpp_binaries_manifest", REUSED_BINARIES_MANIFEST, "artifact_reuse_contract"),
        ("preflight_manifest", PREFLIGHT_MANIFEST, "frozen_decode_preflight"),
        ("runner", RUNNER, "decode_compat_runner"),
        ("runner_test", RUNNER_TEST, "runner_contract_tests"),
        ("manifest_builder", BUILDER, "successor_manifest_builder"),
        ("manifest_builder_test", BUILDER_TEST, "manifest_builder_tests"),
        ("base_model_source", BASE_MODEL_SOURCE, "frozen_tiny_model_constructor"),
        ("parent_safety_source", PARENT_SAFETY_SOURCE, "frozen_archive_safety_parent"),
        ("aten_host", ATEN_HOST, "reused_direct_aten_host"),
        ("aoti_host", AOTI_HOST, "reused_aoti_host"),
    )
    artifacts = [_artifact_record(root, name, path, role) for name, path, role in artifact_specs]
    return {
        "schema": "axon_loop28_pytorch_native_decode_compat_implementation_manifest_v1",
        "loop_id": LOOP_ID,
        "generated_at_utc": timestamp,
        "lineage": {
            "parent_post_sha256": EXPECTED_PARENT_POST_SHA256,
            "proposal_sha256": sha256_file(_resolve_within(root, PROPOSAL)),
            "authorization_sha256": sha256_file(_resolve_within(root, AUTHORIZATION)),
            "official_pytorch_v213_source_manifest_sha256": sha256_file(
                _resolve_within(root, OFFICIAL_RESEARCH_MANIFEST)
            ),
            "reused_cpp_binaries_manifest_sha256": sha256_file(
                _resolve_within(root, REUSED_BINARIES_MANIFEST)
            ),
            "preflight_manifest_sha256": sha256_file(_resolve_within(root, PREFLIGHT_MANIFEST)),
        },
        "contract": {
            "single_process_local_v213_decoder_shim": True,
            "installed_torch_modification_allowed": False,
            "fresh_input_and_artifact_root_required": True,
            "parent_partial_artifact_read_allowed": False,
            "torch_export_calls": 1,
            "aoti_compile_and_package_calls": 1,
            "torchscript_export_calls": 0,
            "package_load_calls": 0,
            "native_runtime_execution_count": 0,
            "retained_output_cap_bytes": 536870912,
            "output_replacement_allowed": False,
        },
        "validation_contract": {
            "focused_tests_required": True,
            "ruff_check_required": True,
            "ruff_format_required": True,
            "py_compile_required": True,
        },
        "artifacts": artifacts,
        "integrity": {"artifact_count": len(artifacts), "all_required_present": True},
        "source_binding": preflight["source_binding"],
        "claim_boundary": preflight["claim_boundary"],
        "decision": "decode_compat_source_frozen_package_requires_authorization",
        "preflight_decision": preflight["decision"],
    }


def verify_implementation_manifest(
    project_root: Path, output: Path = IMPLEMENTATION_MANIFEST
) -> dict[str, Any]:
    payload = load_json_strict(_resolve_within(project_root, output))
    if payload.get("schema") != (
        "axon_loop28_pytorch_native_decode_compat_implementation_manifest_v1"
    ):
        raise DecodeCompatManifestError("Decode implementation manifest schema mismatch")
    rebuilt = build_implementation_manifest(
        project_root, generated_at_utc=str(payload.get("generated_at_utc") or "")
    )
    if payload != rebuilt:
        raise DecodeCompatManifestError("Decode implementation manifest no longer matches source")
    return payload


def build_package_manifest(project_root: Path, *, generated_at_utc: str) -> dict[str, Any]:
    # success/failure 共用终态入口；authorization、lease、budget 和 partials 均逐项闭合。
    root = project_root.resolve(strict=True)
    timestamp = _validate_timestamp(generated_at_utc)
    implementation = verify_implementation_manifest(root)
    receipt_path = root / PACKAGE_RECEIPT
    failure_path = root / PACKAGE_FAILURE
    if receipt_path.is_file() == failure_path.is_file():
        raise DecodeCompatManifestError("Exactly one package terminal receipt must exist")
    terminal_path = PACKAGE_RECEIPT if receipt_path.is_file() else PACKAGE_FAILURE
    terminal = load_json_strict(_resolve_within(root, terminal_path))
    implementation_sha = sha256_file(_resolve_within(root, IMPLEMENTATION_MANIFEST))
    package_authorization, final_lease = _validate_stage_chain(
        root,
        authorization_path=PACKAGE_AUTHORIZATION,
        final_lease_path=PACKAGE_FINAL_LEASE,
        authorization_schema=PACKAGE_AUTHORIZATION_SCHEMA,
        authorization_decision=PACKAGE_AUTHORIZATION_DECISION,
        lease_schema=PACKAGE_LEASE_SCHEMA,
        attempt_id=PACKAGE_ATTEMPT_ID,
        mode="package",
        work_root=PACKAGE_WORK_ROOT,
        output_paths=(PACKAGE_RECEIPT, PACKAGE_FAILURE),
        implementation_sha256=implementation_sha,
    )
    _validate_stage_lineage(root, package_authorization, expected_budget=PACKAGE_BUDGET)
    if package_authorization.get("preflight_manifest_sha256") != sha256_file(
        _resolve_within(root, PREFLIGHT_MANIFEST)
    ):
        raise DecodeCompatManifestError("Package authorization preflight binding mismatch")
    source_records = _verify_source_binding(
        root,
        package_authorization,
        embedded_authorization=terminal.get("lease", {}).get("authorization"),
    )
    _validate_evidence_lease(
        terminal,
        authorization=package_authorization,
        authorization_path=PACKAGE_AUTHORIZATION,
        final_lease=final_lease,
        final_lease_path=PACKAGE_FINAL_LEASE,
        project_root=root,
    )
    if terminal.get("loop_id") != LOOP_ID:
        raise DecodeCompatManifestError("Package terminal receipt loop mismatch")
    worker = terminal.get("worker")
    if not isinstance(worker, dict):
        raise DecodeCompatManifestError("Package terminal receipt omitted worker evidence")
    _validate_launch_integrity(terminal, success=receipt_path.is_file())
    _validate_source_snapshots(terminal, source_records, success=receipt_path.is_file())
    reconstructed = _validate_durable_worker_journal(
        terminal,
        worker,
        stage="package",
        success=receipt_path.is_file(),
    )
    _validate_worker_runtime(
        worker,
        success=receipt_path.is_file(),
        reconstructed=reconstructed,
    )
    budget_actual = _validate_budget_actual(
        terminal,
        package_authorization["budget"],
        success=receipt_path.is_file(),
    )
    for counter in (
        "package_load_count",
        "native_probe_execution_count",
        "gpu_execution_count",
        "network_request_count",
        "quality_metric_count",
    ):
        if terminal.get(counter) != 0:
            raise DecodeCompatManifestError(f"Package terminal counter drifted: {counter}")
    if not isinstance(worker.get("counters"), dict):
        raise DecodeCompatManifestError("Package terminal receipt omitted worker counters")
    counters = worker["counters"]
    if _counter_value(counters, "compiler_help_processes") > PACKAGE_BUDGET[
        "compiler_help_processes_max"
    ]:
        raise DecodeCompatManifestError("Package worker exceeded compiler probe budget")

    common_artifacts = [
        _artifact_record(root, "implementation_manifest", IMPLEMENTATION_MANIFEST, "source"),
        _artifact_record(root, "preflight_manifest", PREFLIGHT_MANIFEST, "decode_preflight"),
        _artifact_record(
            root,
            "official_pytorch_v213_source_manifest",
            OFFICIAL_RESEARCH_MANIFEST,
            "official_source_contract",
        ),
        _artifact_record(
            root,
            "reused_cpp_binaries_manifest",
            REUSED_BINARIES_MANIFEST,
            "artifact_reuse_contract",
        ),
        _artifact_record(
            root, "package_authorization", PACKAGE_AUTHORIZATION, "package_authority"
        ),
        _artifact_record(
            root, "package_final_lease", PACKAGE_FINAL_LEASE, "consumed_package_lease"
        ),
        _artifact_record(root, "package_terminal", terminal_path, "package_terminal_result"),
    ]
    if receipt_path.is_file():
        receipt = terminal
        if receipt.get("schema") != ("axon_loop28_pytorch_native_decode_compat_package_receipt_v1"):
            raise DecodeCompatManifestError("Decode package receipt schema mismatch")
        expected_decision = (
            "upstream_v213_decode_compat_package_generated_dependency_closure_"
            "ready_for_new_runtime_loop"
        )
        if receipt.get("decision") != expected_decision:
            raise DecodeCompatManifestError("Decode package receipt decision mismatch")
        if receipt.get("implementation_manifest_sha256") != implementation_sha:
            raise DecodeCompatManifestError("Decode package implementation binding mismatch")
        if worker.get("status") != "passed" or worker.get("decision") != (
            "decode_compat_package_worker_generated_static_audit_required"
        ):
            raise DecodeCompatManifestError("Decode package worker success decision mismatch")
        _validate_counter_limits(counters, PACKAGE_COUNTER_LIMITS, exact=True)
        if _counter_value(counters, "compiler_help_processes") != 4:
            raise DecodeCompatManifestError(
                "Decode package did not use exactly four compiler-help processes"
            )
        audit = receipt.get("audit", {})
        if audit.get("input", {}).get("sha256") != EXPECTED_INPUT_SHA256:
            raise DecodeCompatManifestError("Decode package input hash mismatch")
        if audit.get("input", {}).get("differs_from_parent_partial") is not True:
            raise DecodeCompatManifestError("Decode package reused the parent partial input")
        closure = audit.get("dependency_closure", {})
        if closure.get("forbidden_hits") or closure.get("unresolved") or closure.get("ambiguous"):
            raise DecodeCompatManifestError("Decode package dependency closure is unsafe")
        if budget_actual["retained_output_bytes"] != audit.get("retained_output_bytes"):
            raise DecodeCompatManifestError("Decode package retained-output budget drifted")
        expected_dumpbin_calls = 2 * len(closure.get("nodes", []))
        if closure.get("dumpbin_invocation_count") != expected_dumpbin_calls:
            raise DecodeCompatManifestError("Decode package dependency-audit call count drifted")
        if budget_actual["dumpbin_processes"] != expected_dumpbin_calls:
            raise DecodeCompatManifestError("Decode package dumpbin budget drifted")
        runner = _load_runner()
        package_path = _resolve_within(root, AOTI_PACKAGE)
        rebuilt_archive = runner.audit_package_archive(package_path)
        if rebuilt_archive != audit.get("archive"):
            raise DecodeCompatManifestError("Decode package archive audit drifted")
        artifacts = [
            *common_artifacts,
            _artifact_record(root, "input_v2", INPUT_PATH, "fresh_synthetic_input"),
            _artifact_record(root, "aoti_package_v2", AOTI_PACKAGE, "precompiled_package"),
        ]
        outcome = {
            "package_generated": True,
            "static_dependency_closure_passed": True,
            "package_load_count": 0,
            "runtime_executed": False,
            "quality_metric_count": 0,
            "failure_class": None,
            "protected_package_budget_consumed": True,
        }
        schema = "axon_loop28_pytorch_native_decode_compat_package_manifest_v1"
        decision = expected_decision
    else:
        failure = terminal
        if failure.get("schema") != ("axon_loop28_pytorch_native_decode_compat_package_failure_v1"):
            raise DecodeCompatManifestError("Decode package failure schema mismatch")
        _validate_counter_limits(counters, PACKAGE_COUNTER_LIMITS, exact=False)
        failure_class = _classify_package_failure(failure)
        expected_decision = _expected_failure_decision(failure_class)
        if failure.get("failure_class") != failure_class:
            raise DecodeCompatManifestError("Decode package failure class mismatch")
        if failure.get("failure_reason") != failure_class:
            raise DecodeCompatManifestError("Decode package failure reason mismatch")
        if failure.get("decision") != expected_decision:
            raise DecodeCompatManifestError("Decode package failure decision mismatch")
        partials = _validate_failure_partials(root, failure)
        if budget_actual["retained_output_bytes"] != sum(
            int(record.get("size_bytes", 0))
            for record in failure.get("partial_artifacts", [])
        ):
            raise DecodeCompatManifestError("Decode package partial-output budget drifted")
        artifacts = list(common_artifacts)
        for name, path in partials:
            artifacts.append(_artifact_record(root, name, path, "quarantined_partial"))
        generated_but_rejected = failure_class in {"dependency", "static_audit"}
        outcome = {
            "package_generated": generated_but_rejected,
            "package_usable": False,
            "static_dependency_closure_passed": False,
            "package_load_count": 0,
            "runtime_executed": False,
            "quality_metric_count": 0,
            "failure_class": failure_class,
            "protected_package_budget_consumed": failure_class != "administrative",
        }
        schema = "axon_loop28_pytorch_native_decode_compat_package_failure_manifest_v1"
        decision = expected_decision
    claim_boundary = dict(package_authorization["claim_boundary"])
    claim_boundary["decoder_compatibility_claim_allowed"] = True
    claim_boundary["tiny_package_generation_claim_allowed"] = outcome["package_generated"]
    claim_boundary["package_dependency_closure_claim_allowed"] = (
        schema == "axon_loop28_pytorch_native_decode_compat_package_manifest_v1"
    )
    claim_boundary["package_load_allowed"] = False
    claim_boundary["tiny_runtime_feasibility_claim_allowed"] = False
    return {
        "schema": schema,
        "loop_id": LOOP_ID,
        "generated_at_utc": timestamp,
        "lineage": {
            "parent_post_sha256": EXPECTED_PARENT_POST_SHA256,
            "implementation_manifest_sha256": sha256_file(
                _resolve_within(root, IMPLEMENTATION_MANIFEST)
            ),
            "preflight_manifest_sha256": implementation["lineage"]["preflight_manifest_sha256"],
            "official_pytorch_v213_source_manifest_sha256": sha256_file(
                _resolve_within(root, OFFICIAL_RESEARCH_MANIFEST)
            ),
            "reused_cpp_binaries_manifest_sha256": sha256_file(
                _resolve_within(root, REUSED_BINARIES_MANIFEST)
            ),
            "package_authorization_sha256": sha256_file(
                _resolve_within(root, PACKAGE_AUTHORIZATION)
            ),
            "package_final_lease_sha256": sha256_file(
                _resolve_within(root, PACKAGE_FINAL_LEASE)
            ),
            "package_terminal_sha256": sha256_file(_resolve_within(root, terminal_path)),
            "attempt_id": PACKAGE_ATTEMPT_ID,
        },
        "outcome": outcome,
        "budget": {
            "authorized": package_authorization["budget"],
            "actual": budget_actual,
            "within_budget": True,
        },
        "source_binding": {
            "artifact_count": len(source_records),
            "records_sha256": _canonical_sha256({"records": source_records}),
            "authorization_evidence_current_implementation_exact_match": True,
        },
        "artifacts": artifacts,
        "integrity": {"artifact_count": len(artifacts), "all_required_present": True},
        "claim_boundary": claim_boundary,
        "decision": decision,
    }


def verify_package_manifest(project_root: Path, output: Path = PACKAGE_MANIFEST) -> dict[str, Any]:
    payload = load_json_strict(_resolve_within(project_root, output))
    if payload.get("schema") not in {
        "axon_loop28_pytorch_native_decode_compat_package_manifest_v1",
        "axon_loop28_pytorch_native_decode_compat_package_failure_manifest_v1",
    }:
        raise DecodeCompatManifestError("Decode package terminal manifest schema mismatch")
    rebuilt = build_package_manifest(
        project_root, generated_at_utc=str(payload.get("generated_at_utc") or "")
    )
    if payload != rebuilt:
        raise DecodeCompatManifestError("Decode package manifest no longer matches evidence")
    return payload


def build_post_manifest(project_root: Path, *, generated_at_utc: str) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    timestamp = _validate_timestamp(generated_at_utc)
    preflight = verify_preflight_manifest(root)
    if preflight.get("schema") == (
        "axon_loop28_pytorch_native_decode_compat_preflight_failure_manifest_v1"
    ):
        if (root / PACKAGE_MANIFEST).exists():
            raise DecodeCompatManifestError("Failed preflight cannot have a package manifest")
        terminal = preflight
        terminal_name = "preflight_manifest"
        terminal_path = PREFLIGHT_MANIFEST
        terminal_role = "terminal_preflight_failure"
        success = False
        decision = "post_decode_compat_preflight_failed_no_package_authorization"
        outcome = {
            "preflight_passed": False,
            "package_authorized": False,
            "package_generated": False,
            "package_usable": False,
            "package_load_count": 0,
            "runtime_executed": False,
            "quality_metric_count": 0,
            "failure_class": "decode_or_environment_preflight",
        }
        terminal_sha_field = "preflight_manifest_sha256"
    else:
        terminal = verify_package_manifest(root)
        terminal_name = "package_manifest"
        terminal_path = PACKAGE_MANIFEST
        terminal_role = "terminal_package_result"
        success = terminal.get("schema") == (
            "axon_loop28_pytorch_native_decode_compat_package_manifest_v1"
        )
        decision = (
            "post_v212_process_local_v213_decode_compat_package_generated_no_load"
            if success
            else "post_v213_decode_compat_package_failed_no_load"
        )
        outcome = {"preflight_passed": True, "package_authorized": True, **terminal["outcome"]}
        terminal_sha_field = "package_manifest_sha256"
    artifacts = [
        _artifact_record(root, terminal_name, terminal_path, terminal_role),
        *[_artifact_record(root, name, path, role) for name, path, role in FINAL_DOCS],
    ]
    return {
        "schema": "axon_loop28_pytorch_native_decode_compat_post_manifest_v1",
        "loop_id": LOOP_ID,
        "generated_at_utc": timestamp,
        "lineage": {
            "parent_post_sha256": EXPECTED_PARENT_POST_SHA256,
            terminal_sha_field: sha256_file(_resolve_within(root, terminal_path)),
            "official_pytorch_v213_source_manifest_sha256": sha256_file(
                _resolve_within(root, OFFICIAL_RESEARCH_MANIFEST)
            ),
            "reused_cpp_binaries_manifest_sha256": sha256_file(
                _resolve_within(root, REUSED_BINARIES_MANIFEST)
            ),
        },
        "outcome": {
            **outcome,
            "actual_v213_runtime_qualified": False,
            "production_upgrade_qualified": False,
            "successor_runtime_loop_required": success,
        },
        "artifacts": artifacts,
        "integrity": {"artifact_count": len(artifacts), "all_required_present": True},
        "claim_boundary": terminal["claim_boundary"],
        "decision": decision,
    }


def verify_post_manifest(project_root: Path, output: Path = POST_MANIFEST) -> dict[str, Any]:
    payload = load_json_strict(_resolve_within(project_root, output))
    if payload.get("schema") != "axon_loop28_pytorch_native_decode_compat_post_manifest_v1":
        raise DecodeCompatManifestError("Decode post manifest schema mismatch")
    rebuilt = build_post_manifest(
        project_root, generated_at_utc=str(payload.get("generated_at_utc") or "")
    )
    if payload != rebuilt:
        raise DecodeCompatManifestError("Decode post manifest no longer matches evidence")
    return payload


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise DecodeCompatManifestError(f"Output already exists: {path}") from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "official-research",
            "reused-binaries",
            "preflight",
            "implementation",
            "package",
            "post",
        ),
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--generated-at-utc")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve(strict=True)
    defaults = {
        "official-research": OFFICIAL_RESEARCH_MANIFEST,
        "reused-binaries": REUSED_BINARIES_MANIFEST,
        "preflight": PREFLIGHT_MANIFEST,
        "implementation": IMPLEMENTATION_MANIFEST,
        "package": PACKAGE_MANIFEST,
        "post": POST_MANIFEST,
    }
    builders = {
        "official-research": build_official_research_manifest,
        "reused-binaries": build_reused_binaries_manifest,
        "preflight": build_preflight_manifest,
        "implementation": build_implementation_manifest,
        "package": build_package_manifest,
        "post": build_post_manifest,
    }
    verifiers = {
        "official-research": verify_official_research_manifest,
        "reused-binaries": verify_reused_binaries_manifest,
        "preflight": verify_preflight_manifest,
        "implementation": verify_implementation_manifest,
        "package": verify_package_manifest,
        "post": verify_post_manifest,
    }
    output = args.output or defaults[args.mode]
    if args.verify:
        payload = verifiers[args.mode](root, output)
    else:
        if not args.generated_at_utc:
            raise DecodeCompatManifestError("--generated-at-utc is required when building")
        payload = builders[args.mode](root, generated_at_utc=args.generated_at_utc)
        _write_exclusive(_resolve_within(root, output, must_exist=False), payload)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "output": output.as_posix(),
                "artifact_count": payload["integrity"]["artifact_count"],
                "decision": payload["decision"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
