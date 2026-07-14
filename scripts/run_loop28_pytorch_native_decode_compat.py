#!/usr/bin/env python3
"""Run the lease-gated PyTorch v2.13 decoder-compat package successor."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import locale
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import unicodedata
import zipfile
from collections import deque
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path, PurePosixPath
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "p0_loop28_pytorch_native_decode_compat_001"
MANIFEST_DIR = Path("manifests/roadmap_9997/p0_loop28_pytorch_native_decode_compat")
REPORT_DIR = Path("reports/roadmap_9997/p0_loop28_pytorch_native_decode_compat")
ARTIFACT_ROOT = Path(
    "artifacts/roadmap_9997/p0_loop28_pytorch_native_decode_compat/tiny_v2/package_attempt_001"
)
PREFLIGHT_WORK_ROOT = REPORT_DIR / "work/decode_probe_attempt_001"
PACKAGE_WORK_ROOT = REPORT_DIR / "work/package_attempt_001"

PROPOSAL = MANIFEST_DIR / "proposal.json"
AUTHORIZATION = MANIFEST_DIR / "authorization.json"
PREFLIGHT_AUTHORIZATION = MANIFEST_DIR / "preflight_authorization.json"
PREFLIGHT_READY_LEASE = MANIFEST_DIR / "preflight_lease.json"
PREFLIGHT_FINAL_LEASE = MANIFEST_DIR / "preflight_lease.final.json"
PREFLIGHT_EVIDENCE = REPORT_DIR / "decode_probe_evidence.final.json"
PREFLIGHT_FAILURE = REPORT_DIR / "decode_probe_failure.final.json"
IMPLEMENTATION_MANIFEST = MANIFEST_DIR / "implementation_manifest.json"
PACKAGE_AUTHORIZATION = MANIFEST_DIR / "package_authorization.json"
PACKAGE_READY_LEASE = MANIFEST_DIR / "package_lease.json"
PACKAGE_FINAL_LEASE = MANIFEST_DIR / "package_lease.final.json"
PACKAGE_RECEIPT = REPORT_DIR / "package_receipt.final.json"
PACKAGE_FAILURE = REPORT_DIR / "package_failure.final.json"
DEPENDENCY_AUDIT_ROOT = REPORT_DIR / "work/dependency_audit_attempt_001"
OFFICIAL_V213_SOURCE_MANIFEST = MANIFEST_DIR / "official_pytorch_v213_source_manifest.json"
REUSED_CPP_BINARIES_MANIFEST = MANIFEST_DIR / "reused_cpp_binaries_manifest.json"

RUNNER = Path("scripts/run_loop28_pytorch_native_decode_compat.py")
RUNNER_TEST = Path("tests/test_run_loop28_pytorch_native_decode_compat.py")
MANIFEST_BUILDER = Path("scripts/build_loop28_pytorch_native_decode_compat_manifest.py")
MANIFEST_BUILDER_TEST = Path("tests/test_build_loop28_pytorch_native_decode_compat_manifest.py")
BASE_MODEL_SOURCE = Path("scripts/run_loop28_pytorch_native_feasibility.py")
PARENT_SAFETY_SOURCE = Path("scripts/run_loop28_pytorch_native_package_controller.py")
BASE_MANIFEST_BUILDER = Path("scripts/build_loop28_pytorch_native_feasibility_manifest.py")

INPUT_PATH = ARTIFACT_ROOT / "input_v2.f32.bin"
AOTI_PACKAGE = ARTIFACT_ROOT / "tiny_cpu_model_v2.pt2"
ATEN_HOST = Path("tools/axon_tiny_pytorch_native/build/bin/Release/axon_tiny_aten_probe.exe")
AOTI_HOST = Path("tools/axon_tiny_pytorch_native/build/bin/Release/axon_tiny_aoti_probe.exe")
PYTHON_EXE = PROJECT_ROOT / "vnev/Scripts/python.exe"
TORCH_ROOT = PROJECT_ROOT / "vnev/Lib/site-packages/torch"
TORCH_LIB = TORCH_ROOT / "lib"
CPP_BUILDER = PROJECT_ROOT / "vnev/Lib/site-packages/torch/_inductor/cpp_builder.py"
CPU_VEC_ISA = PROJECT_ROOT / "vnev/Lib/site-packages/torch/_inductor/cpu_vec_isa.py"
VCVARS64 = Path(
    "C:/Program Files/Microsoft Visual Studio/18/Insiders/VC/Auxiliary/Build/vcvars64.bat"
)
CL_EXE = Path(
    "C:/Program Files/Microsoft Visual Studio/18/Insiders/VC/Tools/MSVC/"
    "14.51.36231/bin/Hostx64/x64/cl.exe"
)
DUMPBIN_EXE = Path(
    "C:/Program Files/Microsoft Visual Studio/18/Insiders/VC/Tools/MSVC/"
    "14.51.36231/bin/Hostx64/x64/dumpbin.exe"
)

EXPECTED_PARENT_POST_SHA256 = "71110ee310a340fb231051df3c4fe3be865e4fa28a0281d96cdf3ddf2ef115cc"
EXPECTED_PARENT_FAILURE_SHA256 = "a445fa8956b444c500fe370cbf9493871a0efbc5c924c21a4d988acfb1db4792"
EXPECTED_UPSTREAM_RESEARCH_SHA256 = (
    "280c8fc1792ae4affaf6252a0c0dfbd3f053a7d544100d1832b267a6678fa608"
)
EXPECTED_PYTHON_SHA256 = "4b8c3912806b3c1591ba3cb403bff77ad309c3fe5756f87c20b7a6f8f0174262"
EXPECTED_CPP_BUILDER_SHA256 = "9952fcc6ae0b660c3fb9b4f279b30caacb31167ae9bc2959872c2460998ae014"
EXPECTED_CPU_VEC_ISA_SHA256 = "371882a23012d93fa51ca5f3a66d827dcfca507d3bc2b1606af65cf660c18fb3"
EXPECTED_VCVARS_SHA256 = "6b516d8fcf543c14b2d861e1f45661e0029230fe0dc48e86ce78522801822209"
EXPECTED_CL_SHA256 = "68844528d5917d57057a2196f610b71a30273b83af57fd519a8c3fcc14c8de4c"
EXPECTED_DUMPBIN_SHA256 = "da26e534fecb88a695dba9d5ff3269872b93f2939a25ad7c96e215955baa0318"
EXPECTED_BASE_MODEL_SHA256 = "b2c61788e0e7ae1348090fcc41e7eeb2efecb3be3efa912fd6560ad030935166"
EXPECTED_ATEN_HOST_SHA256 = "595705bdc0716b7323a0da71b424470c0d474a6556ac7fa3c6507e5d4edfd524"
EXPECTED_AOTI_HOST_SHA256 = "097a41e61bfceeeea4592ff2f1dd079e3b447275b677980b7eceb3f02a5dcca0"
PARENT_PARTIAL_INPUT_SHA256 = "caa371218bdbb95cb73bfe7ab65ec2f8f69222a747fca8f889b2bdc3e693d28b"
EXPECTED_INPUT_SHA256 = "19428a55d01cbd3d1b64a546a5cae7f93091cc4fa3acd1339d78d9b5b87264eb"
EXPECTED_TORCH_VERSION = "2.12.0+cu132"
MAX_RETAINED_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 4096
MAX_COMPRESSION_RATIO = 1000.0
WORKER_TIMEOUT_SECONDS = 1650
TERMINAL_RESERVE_SECONDS = 10
FILE_ATTRIBUTE_REPARSE_POINT = 0x400
WORK_OWNER_NAME = ".axon_worker_owner.json"
ARTIFACT_OWNER_NAME = ".axon_artifact_owner.json"
WORKER_EVENTS_NAME = "worker_events"
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
PACKAGE_BUDGET = {
    "worker_processes": 1,
    "vcvars_activations": 1,
    "compiler_help_processes_max": 4,
    "dumpbin_processes_max": 64,
    "wall_clock_seconds_max": 1800,
    "max_retained_output_bytes": MAX_RETAINED_BYTES,
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

PARENT_POST = Path("manifests/roadmap_9997/p0_loop28_pytorch_native_feasibility/post_manifest.json")
PARENT_FAILURE_MANIFEST = Path(
    "manifests/roadmap_9997/p0_loop28_pytorch_native_feasibility/package_failure_manifest.json"
)
UPSTREAM_RESEARCH = Path(
    "reports/roadmap_9997/p0_loop28_pytorch_native_feasibility/"
    "upstream_decode_compatibility_research.final.json"
)

NEW_INPUT_VALUES = (
    0.0,
    0.375,
    -0.625,
    0.875,
    -1.125,
    1.375,
    -1.625,
    2.125,
    2.125,
    -1.625,
    1.375,
    -1.125,
    0.875,
    -0.625,
    0.375,
    0.0,
)

WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
SYSTEM_DLL_NAMES = {
    "advapi32.dll",
    "bcrypt.dll",
    "bcryptprimitives.dll",
    "cabinet.dll",
    "cfgmgr32.dll",
    "combase.dll",
    "crypt32.dll",
    "dbghelp.dll",
    "gdi32.dll",
    "iphlpapi.dll",
    "kernel32.dll",
    "msvcp140.dll",
    "ntdll.dll",
    "ole32.dll",
    "oleaut32.dll",
    "rpcrt4.dll",
    "secur32.dll",
    "shell32.dll",
    "shlwapi.dll",
    "user32.dll",
    "userenv.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "version.dll",
    "winmm.dll",
    "ws2_32.dll",
}


class DecodeCompatError(RuntimeError):
    """Raised when the successor contract fails closed."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise DecodeCompatError(f"Duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite_json(value: str) -> None:
    raise DecodeCompatError(f"Non-finite JSON number is forbidden: {value}")


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DecodeCompatError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise DecodeCompatError(f"JSON artifact must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return _sha256_bytes(encoded)


def _is_reparse_point(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise DecodeCompatError(f"Unable to inspect path metadata: {path}") from exc
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT
    )


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _assert_no_reparse_chain(path: Path, boundary: Path) -> None:
    boundary = Path(boundary).absolute()
    candidate = Path(path).absolute()
    try:
        relative = candidate.relative_to(boundary)
    except ValueError as exc:
        raise DecodeCompatError(f"Path escapes its governed boundary: {path}") from exc
    current = boundary
    if _lexists(current) and _is_reparse_point(current):
        raise DecodeCompatError(f"Governed boundary is a reparse point: {boundary}")
    for component in relative.parts:
        current /= component
        if not _lexists(current):
            break
        if _is_reparse_point(current):
            raise DecodeCompatError(f"Reparse path is forbidden: {current}")


def _assert_regular_single_link(path: Path, purpose: str) -> None:
    if _is_reparse_point(path):
        raise DecodeCompatError(f"{purpose} must not be a reparse point")
    metadata = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise DecodeCompatError(f"{purpose} must be a regular file")
    if metadata.st_nlink != 1:
        raise DecodeCompatError(f"{purpose} must not be a hardlink")


def _project_output_path(project_root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise DecodeCompatError(f"Output path must remain project-relative: {relative}")
    root = project_root.resolve(strict=True)
    candidate = root.joinpath(relative)
    _assert_no_reparse_chain(candidate.parent, root)
    try:
        candidate.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise DecodeCompatError(f"Output path escapes project root: {relative}") from exc
    return candidate


def _resolve_within(project_root: Path, relative: Path, *, must_exist: bool = True) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise DecodeCompatError(f"Path must remain project-relative: {relative}")
    root = project_root.resolve(strict=True)
    lexical = root / relative
    _assert_no_reparse_chain(lexical, root)
    try:
        candidate = lexical.resolve(strict=must_exist)
    except OSError as exc:
        raise DecodeCompatError(f"Required artifact is missing: {relative}") from exc
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise DecodeCompatError(f"Path escapes project root: {relative}") from exc
    if must_exist and not candidate.is_file():
        raise DecodeCompatError(f"Required artifact is not a file: {relative}")
    if must_exist and _is_reparse_point(lexical):
        raise DecodeCompatError(f"Required artifact is a reparse point: {relative}")
    return candidate


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise DecodeCompatError(f"Output already exists: {path}") from exc


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_exclusive(path, encoded)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_module(relative: Path, name: str):
    path = PROJECT_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DecodeCompatError(f"Unable to import governed module: {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _consume_lease(
    project_root: Path,
    *,
    authorization_path: Path,
    ready_path: Path,
    final_path: Path,
    authorization_schema: str,
    authorization_decision: str,
    lease_schema: str,
    expected_authorization_sha256: str | None = None,
    expected_ready_sha256: str | None = None,
) -> dict[str, Any]:
    # 在任何 Torch 导入或 package 调用前先独占写入 final lease，避免 ready lease 被重复消费。
    root = project_root.resolve(strict=True)
    authorization_file = _resolve_within(root, authorization_path)
    ready_file = _resolve_within(root, ready_path)
    final_file = _resolve_within(root, final_path, must_exist=False)
    authorization_bytes = authorization_file.read_bytes()
    ready_bytes = ready_file.read_bytes()
    authorization = json.loads(
        authorization_bytes.decode("utf-8-sig"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_json,
    )
    ready = json.loads(
        ready_bytes.decode("utf-8-sig"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_json,
    )
    if not isinstance(authorization, dict) or not isinstance(ready, dict):
        raise DecodeCompatError("Authorization and lease must be JSON objects")
    if authorization.get("schema") != authorization_schema:
        raise DecodeCompatError("Stage authorization schema mismatch")
    if authorization.get("decision") != authorization_decision:
        raise DecodeCompatError("Stage authorization decision mismatch")
    if ready.get("schema") != lease_schema or ready.get("status") != "ready":
        raise DecodeCompatError("Ready lease schema or status mismatch")
    if ready.get("single_use") is not True:
        raise DecodeCompatError("Ready lease is not single-use")
    authorization_sha256 = _sha256_bytes(authorization_bytes)
    ready_sha256 = _sha256_bytes(ready_bytes)
    if expected_authorization_sha256 and authorization_sha256 != expected_authorization_sha256:
        raise DecodeCompatError("Stage authorization changed after the ready gate")
    if expected_ready_sha256 and ready_sha256 != expected_ready_sha256:
        raise DecodeCompatError("Ready lease changed after the ready gate")
    if ready.get("authorization_sha256") != authorization_sha256:
        raise DecodeCompatError("Ready lease authorization binding mismatch")
    source_artifacts = _verify_source_records(root, authorization)
    consumed = dict(ready)
    consumed.update(
        {
            "status": "consumed_before_execution",
            "authorization": {
                "path": authorization_path.as_posix(),
                "sha256": authorization_sha256,
                "schema": authorization_schema,
                "decision": authorization_decision,
            },
            "source_artifacts": source_artifacts,
            "original_lease_sha256": ready_sha256,
            "consumed_at_utc": _utc_now(),
        }
    )
    _write_json_exclusive(final_file, consumed)
    ready_file.unlink()
    return {
        "authorization": authorization,
        "authorization_sha256": authorization_sha256,
        "ready_lease_sha256": ready_sha256,
        "path": final_path.as_posix(),
        "sha256": sha256_file(final_file),
        "status": "consumed_before_execution",
        "source_artifacts": source_artifacts,
        "final_payload": consumed,
    }


def _verify_static_parent_chain(project_root: Path) -> None:
    expected = (
        (PARENT_POST, EXPECTED_PARENT_POST_SHA256, "parent post"),
        (PARENT_FAILURE_MANIFEST, EXPECTED_PARENT_FAILURE_SHA256, "parent failure"),
        (UPSTREAM_RESEARCH, EXPECTED_UPSTREAM_RESEARCH_SHA256, "upstream research"),
        (BASE_MODEL_SOURCE, EXPECTED_BASE_MODEL_SHA256, "base model source"),
        (ATEN_HOST, EXPECTED_ATEN_HOST_SHA256, "ATen host"),
        (AOTI_HOST, EXPECTED_AOTI_HOST_SHA256, "AOTI host"),
    )
    for relative, digest, purpose in expected:
        path = _resolve_within(project_root, relative)
        if sha256_file(path) != digest:
            raise DecodeCompatError(f"Static parent hash drifted: {purpose}")
    external = (
        (PYTHON_EXE, EXPECTED_PYTHON_SHA256, "Python"),
        (CPP_BUILDER, EXPECTED_CPP_BUILDER_SHA256, "cpp_builder"),
        (CPU_VEC_ISA, EXPECTED_CPU_VEC_ISA_SHA256, "cpu_vec_isa"),
        (VCVARS64, EXPECTED_VCVARS_SHA256, "vcvars64"),
        (CL_EXE, EXPECTED_CL_SHA256, "cl.exe"),
        (DUMPBIN_EXE, EXPECTED_DUMPBIN_SHA256, "dumpbin.exe"),
    )
    for path, digest, purpose in external:
        if not path.is_file() or _is_reparse_point(path) or sha256_file(path) != digest:
            raise DecodeCompatError(f"External tool hash drifted: {purpose}")


def _verify_source_records(
    project_root: Path, authorization: Mapping[str, Any]
) -> list[dict[str, Any]]:
    records = authorization.get("source_artifacts")
    if not isinstance(records, list) or not records:
        raise DecodeCompatError("Stage authorization lacks source artifacts")
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise DecodeCompatError("Stage source artifact record is invalid")
        raw_name = record.get("name")
        relative_text = record.get("path")
        digest = record.get("sha256")
        size_bytes = record.get("size_bytes")
        if (
            not isinstance(relative_text, str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(digest))
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise DecodeCompatError("Stage source artifact fields are invalid")
        name = raw_name if isinstance(raw_name, str) and raw_name else relative_text
        relative = Path(relative_text)
        canonical_relative = relative.as_posix()
        if name in names or canonical_relative.casefold() in paths:
            raise DecodeCompatError("Stage source artifact names and paths must be unique")
        path = _resolve_within(project_root, relative)
        actual_digest = sha256_file(path)
        actual_size = path.stat().st_size
        if actual_digest != digest:
            raise DecodeCompatError(f"Stage source hash drifted: {name}")
        if actual_size != size_bytes:
            raise DecodeCompatError(f"Stage source size drifted: {name}")
        names.add(name)
        paths.add(canonical_relative.casefold())
        normalized_record = {
            "path": canonical_relative,
            "sha256": actual_digest,
            "size_bytes": actual_size,
        }
        if isinstance(raw_name, str) and raw_name:
            normalized_record["name"] = raw_name
        normalized.append(normalized_record)
    return sorted(normalized, key=lambda item: item["path"])


def _stage_contract(stage: str) -> dict[str, Any]:
    if stage == "preflight":
        attempt_id = f"{LOOP_ID}_decode_probe_attempt_001"
        return {
            "authorization_path": PREFLIGHT_AUTHORIZATION,
            "authorization_schema": "axon_loop28_pytorch_native_decode_preflight_authorization_v1",
            "authorization_decision": "authorize_single_upstream_v213_decode_preflight",
            "ready_path": PREFLIGHT_READY_LEASE,
            "final_path": PREFLIGHT_FINAL_LEASE,
            "lease_schema": "axon_loop28_pytorch_native_decode_preflight_lease_v1",
            "attempt_id": attempt_id,
            "lease_id": attempt_id,
            "work_root": PREFLIGHT_WORK_ROOT,
            "terminal_paths": (PREFLIGHT_EVIDENCE, PREFLIGHT_FAILURE),
            "budget": PREFLIGHT_BUDGET,
        }
    if stage == "package":
        attempt_id = f"{LOOP_ID}_package_attempt_001"
        return {
            "authorization_path": PACKAGE_AUTHORIZATION,
            "authorization_schema": "axon_loop28_pytorch_native_decode_package_authorization_v1",
            "authorization_decision": "authorize_single_decode_compat_tiny_v2_package_no_load",
            "ready_path": PACKAGE_READY_LEASE,
            "final_path": PACKAGE_FINAL_LEASE,
            "lease_schema": "axon_loop28_pytorch_native_decode_package_lease_v1",
            "attempt_id": attempt_id,
            "lease_id": attempt_id,
            "work_root": PACKAGE_WORK_ROOT,
            "terminal_paths": (PACKAGE_RECEIPT, PACKAGE_FAILURE),
            "budget": PACKAGE_BUDGET,
        }
    raise DecodeCompatError(f"Unsupported stage contract: {stage}")


def _canonical_stage_argv(stage: str) -> list[str]:
    return [
        str(PYTHON_EXE.resolve(strict=True)),
        RUNNER.as_posix(),
        stage,
    ]


def _verify_loop_authority(project_root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    proposal_path = _resolve_within(project_root, PROPOSAL)
    authorization_path = _resolve_within(project_root, AUTHORIZATION)
    proposal = load_json_strict(proposal_path)
    authorization = load_json_strict(authorization_path)
    if (
        proposal.get("schema") != "axon_loop28_pytorch_native_decode_compat_proposal_v1"
        or proposal.get("loop_id") != LOOP_ID
        or proposal.get("decision") != "propose_upstream_v213_decode_compat_package_only_successor"
    ):
        raise DecodeCompatError("Successor proposal authority is invalid")
    if (
        authorization.get("schema") != "axon_loop28_pytorch_native_decode_compat_authorization_v1"
        or authorization.get("loop_id") != LOOP_ID
        or authorization.get("decision")
        != "authorize_single_upstream_v213_decode_compat_package_only_loop"
    ):
        raise DecodeCompatError("Successor loop authorization is invalid")
    proposal_binding = authorization.get("proposal")
    if not isinstance(proposal_binding, dict) or proposal_binding.get("sha256") != sha256_file(
        proposal_path
    ):
        raise DecodeCompatError("Loop authorization proposal binding mismatch")
    budget = proposal.get("budget")
    if not isinstance(budget, dict):
        raise DecodeCompatError("Successor proposal budget is invalid")
    return proposal, authorization, _canonical_json_sha256(budget)


def _verify_current_interpreter() -> dict[str, Any]:
    current = Path(sys.executable).resolve(strict=True)
    frozen = PYTHON_EXE.resolve(strict=True)
    if current != frozen or sha256_file(current) != EXPECTED_PYTHON_SHA256:
        raise DecodeCompatError("Controller must use the frozen Windows venv interpreter")
    _assert_regular_single_link(current, "Frozen Python interpreter")
    return {
        "path": str(current),
        "sha256": EXPECTED_PYTHON_SHA256,
        "matches_frozen_interpreter": True,
    }


def _verify_stage_ready_gate(project_root: Path, stage: str) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    contract = _stage_contract(stage)
    _verify_static_parent_chain(root)
    proposal, loop_authorization, _proposal_budget_digest = _verify_loop_authority(root)
    interpreter = _verify_current_interpreter()
    authorization_file = _resolve_within(root, contract["authorization_path"])
    ready_file = _resolve_within(root, contract["ready_path"])
    authorization = load_json_strict(authorization_file)
    ready = load_json_strict(ready_file)
    authorization_sha256 = sha256_file(authorization_file)
    ready_sha256 = sha256_file(ready_file)
    if (
        authorization.get("schema") != contract["authorization_schema"]
        or authorization.get("decision") != contract["authorization_decision"]
        or authorization.get("loop_id") != LOOP_ID
        or authorization.get("attempt_id") != contract["attempt_id"]
        or authorization.get("canonical_command") != _canonical_stage_argv(stage)
        or authorization.get("artifact_root") != ARTIFACT_ROOT.as_posix()
        or authorization.get("work_root") != contract["work_root"].as_posix()
        or authorization.get("terminal_outputs")
        != [path.as_posix() for path in contract["terminal_paths"]]
        or authorization.get("budget") != contract["budget"]
    ):
        raise DecodeCompatError(f"{stage} authorization contract mismatch")
    lineage = {
        "proposal_sha256": sha256_file(_resolve_within(root, PROPOSAL)),
        "successor_authorization_sha256": sha256_file(_resolve_within(root, AUTHORIZATION)),
        "official_pytorch_v213_source_manifest_sha256": sha256_file(
            _resolve_within(root, OFFICIAL_V213_SOURCE_MANIFEST)
        ),
        "reused_cpp_binaries_manifest_sha256": sha256_file(
            _resolve_within(root, REUSED_CPP_BINARIES_MANIFEST)
        ),
    }
    if any(authorization.get(field) != value for field, value in lineage.items()):
        raise DecodeCompatError(f"{stage} authorization lineage mismatch")
    claim_boundary = authorization.get("claim_boundary")
    if (
        not isinstance(claim_boundary, dict)
        or claim_boundary.get("package_load_allowed") is not False
        or claim_boundary.get("quality_claim_allowed") is not False
    ):
        raise DecodeCompatError(f"{stage} authorization claim boundary is unsafe")
    source_artifacts = _verify_source_records(root, authorization)
    builder = _load_module(
        MANIFEST_BUILDER, f"loop28_native_decode_compat_manifest_builder_for_{stage}_gate"
    )
    builder_source_artifacts = builder._verify_source_binding(root, authorization)
    if builder_source_artifacts != source_artifacts:
        raise DecodeCompatError(f"{stage} source contract differs from manifest governance")
    builder.verify_official_research_manifest(root, OFFICIAL_V213_SOURCE_MANIFEST)
    builder.verify_reused_binaries_manifest(root, REUSED_CPP_BINARIES_MANIFEST)
    if (
        ready.get("schema") != contract["lease_schema"]
        or ready.get("loop_id") != LOOP_ID
        or ready.get("attempt_id") != contract["attempt_id"]
        or ready.get("lease_id") != contract["lease_id"]
        or ready.get("status") != "ready"
        or ready.get("single_use") is not True
        or ready.get("authorization_sha256") != authorization_sha256
        or ready.get("authorization_path") != contract["authorization_path"].as_posix()
        or ready.get("canonical_command") != _canonical_stage_argv(stage)
        or ready.get("artifact_root") != ARTIFACT_ROOT.as_posix()
        or ready.get("work_root") != contract["work_root"].as_posix()
        or ready.get("terminal_outputs") != [path.as_posix() for path in contract["terminal_paths"]]
        or ready.get("budget_sha256") != _canonical_json_sha256(contract["budget"])
        or ready.get("consumed_path") != contract["final_path"].as_posix()
    ):
        raise DecodeCompatError(f"{stage} ready lease contract mismatch")
    if stage == "package":
        builder.verify_implementation_manifest(root, IMPLEMENTATION_MANIFEST)
        implementation_sha256 = sha256_file(_resolve_within(root, IMPLEMENTATION_MANIFEST))
        if authorization.get("implementation_manifest_sha256") != implementation_sha256:
            raise DecodeCompatError("Package authorization implementation binding mismatch")
        preflight_manifest = getattr(builder, "PREFLIGHT_MANIFEST")
        if authorization.get("preflight_manifest_sha256") != sha256_file(
            _resolve_within(root, preflight_manifest)
        ):
            raise DecodeCompatError("Package authorization preflight binding mismatch")
    outputs = (
        contract["final_path"],
        contract["work_root"],
        ARTIFACT_ROOT,
        DEPENDENCY_AUDIT_ROOT,
        *contract["terminal_paths"],
    )
    for relative in outputs:
        path = _project_output_path(root, relative)
        if _lexists(path):
            raise DecodeCompatError(f"{stage} output already exists: {relative}")
    return {
        "contract": contract,
        "proposal": proposal,
        "loop_authorization": loop_authorization,
        "authorization": authorization,
        "authorization_sha256": authorization_sha256,
        "ready_sha256": ready_sha256,
        "budget_digest": _canonical_json_sha256(contract["budget"]),
        "source_artifacts": source_artifacts,
        "interpreter": interpreter,
    }


def _stage_runtime_paths(project_root: Path, stage: str) -> dict[str, Path]:
    contract = _stage_contract(stage)
    work_root = _project_output_path(project_root, contract["work_root"])
    return {
        "work_root": work_root,
        "temp_root": work_root / "disposable_temp",
        "worker_output": work_root / "worker_result.json",
        "script_path": work_root / "launch_worker.cmd",
        "owner_path": work_root / WORK_OWNER_NAME,
        "events_root": work_root / WORKER_EVENTS_NAME,
        "job_gate": work_root / "job_assigned.flag",
        "artifact_root": _project_output_path(project_root, ARTIFACT_ROOT),
    }


def _worker_ownership_token(stage: str, lease: Mapping[str, Any]) -> str:
    final_sha256 = lease.get("sha256")
    if not isinstance(final_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", final_sha256):
        raise DecodeCompatError("Consumed lease final hash is invalid")
    return _sha256_bytes(f"{LOOP_ID}:{stage}:{final_sha256}".encode("ascii"))


def _mkdir_fresh(path: Path, boundary: Path) -> None:
    boundary = boundary.resolve(strict=True)
    _assert_no_reparse_chain(path.parent, boundary)
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_chain(path.parent, boundary)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise DecodeCompatError(f"Owned directory already exists: {path}") from exc
    if _is_reparse_point(path):
        raise DecodeCompatError(f"Fresh owned directory became a reparse point: {path}")


def _prepare_worker_root(
    project_root: Path, stage: str, token: str, lease: Mapping[str, Any]
) -> dict[str, Path]:
    paths = _stage_runtime_paths(project_root, stage)
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise DecodeCompatError("Worker ownership token is invalid")
    _mkdir_fresh(paths["work_root"], project_root)
    paths["temp_root"].mkdir()
    owner = {
        "schema": "axon_loop28_pytorch_native_decode_worker_owner_v1",
        "loop_id": LOOP_ID,
        "stage": stage,
        "attempt_id": _stage_contract(stage)["attempt_id"],
        "ownership_token": token,
        "consumed_lease_sha256": lease["sha256"],
    }
    _write_json_exclusive(paths["owner_path"], owner)
    return paths


def _verify_worker_owned_paths(project_root: Path, stage: str, token: str) -> dict[str, Path]:
    paths = _stage_runtime_paths(project_root, stage)
    expected_work = _project_output_path(project_root, _stage_contract(stage)["work_root"])
    if paths["work_root"] != expected_work:
        raise DecodeCompatError("Worker root is not the exact stage-owned root")
    _assert_no_reparse_chain(paths["work_root"], project_root)
    owner = load_json_strict(paths["owner_path"])
    if (
        owner.get("schema") != "axon_loop28_pytorch_native_decode_worker_owner_v1"
        or owner.get("loop_id") != LOOP_ID
        or owner.get("stage") != stage
        or owner.get("attempt_id") != _stage_contract(stage)["attempt_id"]
        or owner.get("ownership_token") != token
    ):
        raise DecodeCompatError("Worker ownership marker mismatch")
    if not paths["temp_root"].is_dir() or _is_reparse_point(paths["temp_root"]):
        raise DecodeCompatError("Worker TEMP root is not an owned regular directory")
    if _lexists(paths["worker_output"]):
        raise DecodeCompatError("Worker output already exists")
    if _lexists(paths["events_root"]):
        raise DecodeCompatError("Worker event journal already exists")
    _assert_regular_single_link(paths["job_gate"], "Worker Job Object release gate")
    if paths["job_gate"].read_bytes() != b"job-assigned\n":
        raise DecodeCompatError("Worker Job Object release gate is invalid")
    if _lexists(paths["artifact_root"]):
        raise DecodeCompatError("Successor artifact root already exists before worker execution")
    return paths


def _audit_tree_no_reparse(root: Path) -> None:
    if not root.is_dir() or _is_reparse_point(root):
        raise DecodeCompatError(f"Owned cleanup root is unsafe: {root}")
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink() or _is_reparse_point(path):
                    raise DecodeCompatError(f"Owned tree contains a reparse entry: {path}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif not entry.is_file(follow_symlinks=False):
                    raise DecodeCompatError(f"Owned tree contains a special entry: {path}")


def _remove_owned_worker_root(project_root: Path, stage: str, token: str) -> dict[str, Any]:
    paths = _stage_runtime_paths(project_root, stage)
    if not _lexists(paths["work_root"]):
        return {"attempted": False, "removed": True}
    _assert_no_reparse_chain(paths["work_root"], project_root)
    owner = load_json_strict(paths["owner_path"])
    if owner.get("ownership_token") != token or owner.get("stage") != stage:
        raise DecodeCompatError("Refusing to remove an unowned worker root")
    _audit_tree_no_reparse(paths["work_root"])
    shutil.rmtree(paths["work_root"])
    if _lexists(paths["work_root"]):
        raise DecodeCompatError("Owned worker root removal did not complete")
    return {"attempted": True, "removed": True}


def _worker_script_bytes(*, stage: str, ownership_token: str) -> bytes:
    # 解释器、单进程 Inductor 和临时目录均在 Torch 导入前冻结。
    paths = _stage_runtime_paths(PROJECT_ROOT, stage)
    python = PYTHON_EXE.resolve(strict=True)
    runner = (PROJECT_ROOT / RUNNER).resolve(strict=True)
    command = (
        f'"{python}" -X utf8=0 "{runner}" worker --stage {stage} '
        f"--ownership-token {ownership_token}"
    )
    lines = [
        "@echo off",
        "setlocal EnableExtensions DisableDelayedExpansion",
        ":AXON_WAIT_FOR_JOB",
        'if not exist "job_assigned.flag" goto AXON_WAIT_FOR_JOB',
        f'call "{VCVARS64}" >nul',
        "if errorlevel 1 exit /b %errorlevel%",
        'set "PYTHONDONTWRITEBYTECODE=1"',
        'set "PYTHONNOUSERSITE=1"',
        'set "PYTHONHASHSEED=0"',
        f'set "TEMP={paths["temp_root"]}"',
        f'set "TMP={paths["temp_root"]}"',
        f'set "TORCHINDUCTOR_CACHE_DIR={paths["temp_root"] / "inductor_cache"}"',
        'set "TORCHINDUCTOR_COMPILE_THREADS=1"',
        'set "TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=0"',
        'set "CUDA_VISIBLE_DEVICES="',
        'set "OMP_NUM_THREADS=1"',
        'set "MKL_NUM_THREADS=1"',
        'set "CXX=cl"',
        command,
        "exit /b %ERRORLEVEL%",
    ]
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


def _set_console_code_pages(code_pages: Mapping[str, int]) -> None:
    kernel32 = ctypes.windll.kernel32
    if not kernel32.SetConsoleCP(int(code_pages["input"])):
        raise DecodeCompatError("SetConsoleCP failed")
    if not kernel32.SetConsoleOutputCP(int(code_pages["output"])):
        raise DecodeCompatError("SetConsoleOutputCP failed")
    if _console_code_pages() != {
        "input": int(code_pages["input"]),
        "output": int(code_pages["output"]),
    }:
        raise DecodeCompatError("Console code-page update did not persist")


class _IOCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JobExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobBasicLimitInformation),
        ("IoInfo", _IOCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JobBasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class _WindowsKillOnCloseJob:
    _EXTENDED_LIMIT_CLASS = 9
    _BASIC_ACCOUNTING_CLASS = 1
    _KILL_ON_JOB_CLOSE = 0x00002000

    def __init__(self) -> None:
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise DecodeCompatError("CreateJobObjectW failed")
        self.kernel32 = kernel32
        self.handle = handle
        self.assigned = False
        self.closed = False
        limits = _JobExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = self._KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle,
            self._EXTENDED_LIMIT_CLASS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            kernel32.CloseHandle(handle)
            self.closed = True
            raise DecodeCompatError("SetInformationJobObject failed")

    def assign(self, process: subprocess.Popen[bytes]) -> None:
        process_handle = wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
        if not self.kernel32.AssignProcessToJobObject(self.handle, process_handle):
            raise DecodeCompatError("AssignProcessToJobObject failed")
        self.assigned = True

    def active_processes(self) -> int:
        accounting = _JobBasicAccountingInformation()
        if not self.kernel32.QueryInformationJobObject(
            self.handle,
            self._BASIC_ACCOUNTING_CLASS,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            None,
        ):
            raise DecodeCompatError("QueryInformationJobObject failed")
        return int(accounting.ActiveProcesses)

    def wait_empty(self, deadline_monotonic: float) -> int:
        while True:
            active = self.active_processes()
            if active == 0:
                return active
            _remaining_timeout(deadline_monotonic, 0.05)
            time.sleep(0.05)

    def terminate(self, deadline_monotonic: float) -> dict[str, Any]:
        requested = bool(self.kernel32.TerminateJobObject(self.handle, 1))
        active_after = self.wait_empty(deadline_monotonic) if requested else self.active_processes()
        return {
            "method": "windows_job_object_terminate",
            "job_assigned": self.assigned,
            "tree_termination_requested": requested,
            "active_processes_after": active_after,
            "tree_termination_confirmed": requested and active_after == 0,
        }

    def close(self) -> None:
        if not self.closed:
            self.kernel32.CloseHandle(self.handle)
            self.closed = True


def _remaining_timeout(deadline_monotonic: float, maximum: float) -> float:
    remaining = deadline_monotonic - time.perf_counter()
    if remaining <= 0:
        raise DecodeCompatError("Stage wall-clock deadline expired")
    return max(0.1, min(maximum, remaining))


def _terminate_windows_process_tree(
    process: subprocess.Popen[bytes],
    *,
    deadline_monotonic: float,
    job: _WindowsKillOnCloseJob | None,
) -> dict[str, Any]:
    if job is not None and job.assigned:
        result = job.terminate(deadline_monotonic)
        if process.poll() is None:
            process.wait(timeout=_remaining_timeout(deadline_monotonic, 30))
        result["process_returncode"] = process.returncode
        return result
    command = ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"]
    stdout = b""
    stderr = b""
    returncode = None
    taskkill_error = None
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            timeout=_remaining_timeout(deadline_monotonic, 30),
            check=False,
        )
        stdout = completed.stdout or b""
        stderr = completed.stderr or b""
        returncode = completed.returncode
    except Exception as exc:
        taskkill_error = f"{type(exc).__name__}: {str(exc)[:500]}"
    if process.poll() is None:
        try:
            process.wait(timeout=_remaining_timeout(deadline_monotonic, 30))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_remaining_timeout(deadline_monotonic, 30))
    tree_termination_confirmed = returncode == 0 and process.poll() is not None
    return {
        "method": "taskkill_before_job_gate_release",
        "job_assigned": False,
        "command": command,
        "returncode": returncode,
        "taskkill_error": taskkill_error,
        "stdout_sha256": _sha256_bytes(stdout),
        "stderr_sha256": _sha256_bytes(stderr),
        "process_returncode": process.returncode,
        "tree_termination_requested": True,
        "tree_termination_confirmed": tree_termination_confirmed,
    }


def _launch_worker(
    *,
    project_root: Path,
    stage: str,
    lease: Mapping[str, Any],
    deadline_monotonic: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    token = _worker_ownership_token(stage, lease)
    paths = _prepare_worker_root(project_root, stage, token, lease)
    _write_exclusive(
        paths["script_path"],
        _worker_script_bytes(stage=stage, ownership_token=token),
    )
    started = dt.datetime.now(dt.timezone.utc)
    command = ["cmd.exe", "/d", "/c", "launch_worker.cmd"]
    before = _console_code_pages()
    configured: dict[str, int] | None = None
    after_worker: dict[str, int] | None = None
    restored: dict[str, int] | None = None
    stdout = b""
    stderr = b""
    returncode: int | None = None
    timed_out = False
    termination: dict[str, Any] | None = None
    launch_error: str | None = None
    process: subprocess.Popen[bytes] | None = None
    job: _WindowsKillOnCloseJob | None = None
    job_evidence: dict[str, Any] = {
        "created": False,
        "assigned": False,
        "gate_released": False,
        "active_processes_after": None,
        "closed": False,
    }
    try:
        _set_console_code_pages({"input": 65001, "output": 65001})
        configured = _console_code_pages()
        job = _WindowsKillOnCloseJob()
        job_evidence["created"] = True
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen(
            command,
            cwd=paths["work_root"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        job.assign(process)
        job_evidence["assigned"] = True
        _write_exclusive(paths["job_gate"], b"job-assigned\n")
        job_evidence["gate_released"] = True
        try:
            remaining = deadline_monotonic - time.perf_counter() - TERMINAL_RESERVE_SECONDS
            if remaining <= 0:
                raise DecodeCompatError("Stage wall-clock budget expired before worker wait")
            stdout, stderr = process.communicate(timeout=min(WORKER_TIMEOUT_SECONDS, remaining))
        except subprocess.TimeoutExpired:
            timed_out = True
            termination = _terminate_windows_process_tree(
                process, deadline_monotonic=deadline_monotonic, job=job
            )
            stdout, stderr = process.communicate(timeout=_remaining_timeout(deadline_monotonic, 30))
        returncode = process.returncode
        active_processes = job.active_processes()
        if active_processes:
            termination = job.terminate(deadline_monotonic)
            raise DecodeCompatError("Worker root exited while Job Object descendants remained")
        job_evidence["active_processes_after"] = active_processes
    except Exception as exc:
        launch_error = f"{type(exc).__name__}: {str(exc)[:1000]}"
        job_has_active_processes = False
        if job is not None and job.assigned:
            try:
                job_has_active_processes = job.active_processes() > 0
            except Exception:
                job_has_active_processes = True
        if process is not None and (process.poll() is None or job_has_active_processes):
            try:
                termination = _terminate_windows_process_tree(
                    process, deadline_monotonic=deadline_monotonic, job=job
                )
                stdout, stderr = process.communicate(
                    timeout=_remaining_timeout(deadline_monotonic, 30)
                )
                returncode = process.returncode
            except Exception as termination_exc:
                launch_error += (
                    f"; termination:{type(termination_exc).__name__}:{str(termination_exc)[:500]}"
                )
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
    finally:
        after_worker = _console_code_pages()
        try:
            _set_console_code_pages(before)
            restored = _console_code_pages()
        except Exception as exc:
            launch_error = (
                f"{launch_error}; " if launch_error else ""
            ) + f"console_restore:{type(exc).__name__}:{str(exc)[:500]}"
        if job is not None:
            try:
                if job_evidence["active_processes_after"] is None and job.assigned:
                    job_evidence["active_processes_after"] = job.active_processes()
            except Exception as exc:
                launch_error = (
                    f"{launch_error}; " if launch_error else ""
                ) + f"job_query:{type(exc).__name__}:{str(exc)[:500]}"
            finally:
                job.close()
                job_evidence["closed"] = True
    elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    worker: dict[str, Any] = {}
    worker_parse_error = None
    if _lexists(paths["worker_output"]) and paths["worker_output"].is_file():
        try:
            worker = load_json_strict(paths["worker_output"])
        except Exception as exc:
            worker_parse_error = f"{type(exc).__name__}: {str(exc)[:1000]}"
    try:
        journal_evidence = _read_worker_events(paths["events_root"], stage)
    except Exception as exc:
        conservative_counters = _base_counters()
        conservative_counters.update(
            {
                "torch_imports": 1,
                "model_constructions": 1,
                "torch_export_calls": 1,
                "aoti_compile_and_package_calls": 1,
            }
        )
        journal_evidence = {
            "record_count": 0,
            "records": [],
            "last_counters": conservative_counters,
            "integrity_error": f"{type(exc).__name__}: {str(exc)[:1000]}",
        }
        worker_parse_error = (
            f"{worker_parse_error}; " if worker_parse_error else ""
        ) + "worker event journal integrity failed"
    if not worker:
        worker = _administrative_worker_failure(
            stage,
            journal_evidence["last_counters"],
            "Worker exited without a structured receipt",
        )
        worker["receipt_reconstructed_from_journal"] = bool(
            journal_evidence.get("record_count") or journal_evidence.get("integrity_error")
        )
    elif worker.get("counters") != journal_evidence.get("last_counters"):
        worker_parse_error = (
            f"{worker_parse_error}; " if worker_parse_error else ""
        ) + "worker counters differ from durable journal"
    else:
        counters = worker.get("counters", {})
        runtime_incomplete = bool(
            worker.get("status") == "failed"
            and int(counters.get("torch_imports", 0) or 0) > 0
            and (not worker.get("torch") or not worker.get("shim"))
        )
        worker["receipt_reconstructed_from_journal"] = runtime_incomplete
    worker["durable_journal_last_record_sha256"] = journal_evidence.get("last_record_sha256")
    temp_cleanup = {"attempted": False, "removed": not _lexists(paths["temp_root"])}
    if _lexists(paths["temp_root"]):
        try:
            _audit_tree_no_reparse(paths["temp_root"])
            shutil.rmtree(paths["temp_root"])
            temp_cleanup = {"attempted": True, "removed": not _lexists(paths["temp_root"])}
        except Exception as exc:
            temp_cleanup = {
                "attempted": True,
                "removed": False,
                "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
            }
    launch = {
        "command": command,
        "worker_argv": [
            str(PYTHON_EXE.resolve(strict=True)),
            "-X",
            "utf8=0",
            str((PROJECT_ROOT / RUNNER).resolve(strict=True)),
            "worker",
            "--stage",
            stage,
            "--ownership-token",
            token,
        ],
        "ownership_token_sha256": _sha256_bytes(token.encode("ascii")),
        "script_sha256": sha256_file(paths["script_path"]),
        "returncode": returncode,
        "process_started": process is not None,
        "elapsed_seconds": elapsed,
        "timed_out": timed_out,
        "process_tree_termination": termination,
        "windows_job_object": job_evidence,
        "stdout_sha256": _sha256_bytes(stdout or b""),
        "stderr_sha256": _sha256_bytes(stderr or b""),
        "stdout_bytes": len(stdout or b""),
        "stderr_bytes": len(stderr or b""),
        "worker_parse_error": worker_parse_error,
        "durable_worker_journal": journal_evidence,
        "launch_error": launch_error,
        "temp_cleanup": temp_cleanup,
        "console_code_pages": {
            "before": before,
            "configured": configured,
            "after_worker": after_worker,
            "restored": restored,
            "restore_verified": restored == before,
        },
    }
    return worker, launch


def _console_code_pages() -> dict[str, int]:
    kernel32 = ctypes.windll.kernel32
    return {
        "input": int(kernel32.GetConsoleCP()),
        "output": int(kernel32.GetConsoleOutputCP()),
    }


class _WorkerEventJournal:
    def __init__(self, root: Path, stage: str) -> None:
        if stage not in {"preflight", "package"}:
            raise DecodeCompatError("Worker event journal stage is invalid")
        try:
            root.mkdir()
        except FileExistsError as exc:
            raise DecodeCompatError("Worker event journal already exists") from exc
        if _is_reparse_point(root):
            raise DecodeCompatError("Worker event journal is a reparse point")
        self.root = root
        self.stage = stage
        self.sequence = 0
        self.previous_sha256: str | None = None

    def append(self, event: str, counters: Mapping[str, Any]) -> None:
        self.sequence += 1
        body = {
            "schema": "axon_loop28_pytorch_native_decode_worker_event_v1",
            "loop_id": LOOP_ID,
            "stage": self.stage,
            "sequence": self.sequence,
            "event": event,
            "previous_record_sha256": self.previous_sha256,
            "counters": dict(counters),
        }
        record_sha256 = _canonical_json_sha256(body)
        record = {**body, "record_sha256": record_sha256}
        _write_json_exclusive(self.root / f"{self.sequence:06d}.json", record)
        self.previous_sha256 = record_sha256


def _read_worker_events(events_root: Path, stage: str) -> dict[str, Any]:
    if not _lexists(events_root):
        return {"record_count": 0, "records": [], "last_counters": _base_counters()}
    if not events_root.is_dir() or _is_reparse_point(events_root):
        raise DecodeCompatError("Worker event journal root is unsafe")
    records: list[dict[str, Any]] = []
    previous_sha256: str | None = None
    for expected_sequence, path in enumerate(sorted(events_root.glob("*.json")), start=1):
        if path.name != f"{expected_sequence:06d}.json":
            raise DecodeCompatError("Worker event journal sequence is not contiguous")
        record = load_json_strict(path)
        record_sha256 = record.pop("record_sha256", None)
        if (
            record.get("schema") != "axon_loop28_pytorch_native_decode_worker_event_v1"
            or record.get("loop_id") != LOOP_ID
            or record.get("stage") != stage
            or record.get("sequence") != expected_sequence
            or record.get("previous_record_sha256") != previous_sha256
            or record_sha256 != _canonical_json_sha256(record)
            or not isinstance(record.get("counters"), dict)
        ):
            raise DecodeCompatError("Worker event journal integrity failed")
        record["record_sha256"] = record_sha256
        records.append(record)
        previous_sha256 = record_sha256
    last_counters = dict(records[-1]["counters"]) if records else _base_counters()
    return {
        "record_count": len(records),
        "records": records,
        "last_event": records[-1]["event"] if records else None,
        "last_record_sha256": previous_sha256,
        "last_counters": last_counters,
    }


@contextmanager
def _capture_process_telemetry(
    telemetry: dict[str, Any],
    *,
    journal: _WorkerEventJournal | None = None,
    counters: dict[str, int] | None = None,
):
    original_popen = subprocess.Popen

    def counted_popen(*args: Any, **kwargs: Any):
        command = kwargs.get("args", args[0] if args else [])
        if isinstance(command, (str, bytes)):
            argv = [os.fsdecode(command)]
        else:
            argv = [os.fsdecode(value) for value in command]
        executable = argv[0] if argv else ""
        basename = executable.replace("\\", "/").rsplit("/", 1)[-1].casefold()
        telemetry["total_subprocesses"] += 1
        if basename in {"cl", "cl.exe", "clang-cl", "clang-cl.exe"}:
            telemetry["compiler_processes"] += 1
            if any(value.casefold() == "/help" for value in argv[1:]):
                telemetry["compiler_help_processes"] += 1
        if basename in {"dumpbin", "dumpbin.exe"}:
            telemetry["dumpbin_processes"] += 1
        if len(telemetry["commands"]) < 256:
            telemetry["commands"].append(
                {
                    "ordinal": telemetry["total_subprocesses"],
                    "executable": executable,
                    "argument_count": len(argv),
                    "argv_sha256": _sha256_bytes("\0".join(argv).encode("utf-8")),
                }
            )
        if counters is not None:
            _sync_process_counters(counters, telemetry)
            if journal is not None:
                journal.append("subprocess_about_to_start", counters)
        return original_popen(*args, **kwargs)

    subprocess.Popen = counted_popen  # type: ignore[assignment]
    try:
        yield telemetry
    finally:
        subprocess.Popen = original_popen


def _new_process_telemetry() -> dict[str, Any]:
    return {
        "total_subprocesses": 0,
        "compiler_processes": 0,
        "compiler_help_processes": 0,
        "dumpbin_processes": 0,
        "commands": [],
    }


def _base_counters() -> dict[str, int]:
    return {
        "torch_imports": 0,
        "model_constructions": 0,
        "torch_export_calls": 0,
        "torch_export_completed": 0,
        "aoti_compile_and_package_calls": 0,
        "aoti_compile_and_package_completed": 0,
        "torchscript_export_calls": 0,
        "package_load_calls": 0,
        "native_probe_execution_count": 0,
        "gpu_execution_count": 0,
        "network_request_count": 0,
        "quality_metric_count": 0,
        "total_subprocesses": 0,
        "compiler_processes": 0,
        "compiler_help_processes": 0,
        "dumpbin_processes": 0,
    }


def _sync_process_counters(counters: dict[str, int], telemetry: Mapping[str, Any]) -> None:
    counters["total_subprocesses"] = int(telemetry.get("total_subprocesses", 0) or 0)
    counters["compiler_processes"] = int(telemetry.get("compiler_processes", 0) or 0)
    counters["compiler_help_processes"] = int(telemetry.get("compiler_help_processes", 0) or 0)
    counters["dumpbin_processes"] = int(telemetry.get("dumpbin_processes", 0) or 0)


def _administrative_worker_failure(
    stage: str, counters: Mapping[str, Any] | None, error: str
) -> dict[str, Any]:
    normalized = _base_counters()
    if isinstance(counters, Mapping):
        for name in normalized:
            value = counters.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                normalized[name] = value
    return {
        "schema": "axon_loop28_pytorch_native_decode_compat_worker_failure_v1",
        "loop_id": LOOP_ID,
        "stage": stage,
        "status": "failed",
        "error_type": "WorkerReceiptUnavailable",
        "error": error[:4000],
        "environment": {},
        "torch": {},
        "shim": {},
        "compiler": {},
        "process_telemetry": {
            "total_subprocesses": normalized["total_subprocesses"],
            "compiler_processes": normalized["compiler_processes"],
            "compiler_help_processes": normalized["compiler_help_processes"],
            "dumpbin_processes": normalized["dumpbin_processes"],
            "commands": [],
        },
        "counters": normalized,
        "receipt_reconstructed_from_journal": False,
        "durable_journal_last_record_sha256": None,
        "decision": "decode_compat_worker_failure_no_runtime_claim",
    }


def _apply_v213_decode_shim(
    counters: dict[str, int], journal: _WorkerEventJournal
) -> tuple[Any, Any, Any, dict[str, Any]]:
    watched_modules = (
        "torch",
        "torch._inductor.cpp_builder",
        "torch._inductor.cpu_vec_isa",
    )
    modules_before = {name: name in sys.modules for name in watched_modules}
    if any(modules_before.values()):
        raise DecodeCompatError("Torch or compiler modules were imported before the decoder shim")
    counters["torch_imports"] += 1
    journal.append("torch_import_about_to_start", counters)
    import torch  # noqa: PLC0415
    journal.append("torch_import_completed", counters)

    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise DecodeCompatError("Installed Torch version drifted")
    if torch.cuda.is_initialized():
        raise DecodeCompatError("CUDA initialized before decoder shim")
    compiler_modules_after_torch = {name: name in sys.modules for name in watched_modules[1:]}
    if any(compiler_modules_after_torch.values()):
        raise DecodeCompatError("Torch import preloaded compiler modules before the shim")
    cpp_builder = importlib.import_module("torch._inductor.cpp_builder")
    if sha256_file(Path(cpp_builder.__file__)) != EXPECTED_CPP_BUILDER_SHA256:
        raise DecodeCompatError("Installed cpp_builder source drifted")
    preferred = locale.getpreferredencoding()
    if preferred.casefold() not in {"cp936", "gbk"}:
        raise DecodeCompatError("Successor requires the frozen CP936/GBK preferred encoding")
    before = tuple(cpp_builder.SUBPROCESS_DECODE_ARGS)
    if before != (preferred,):
        raise DecodeCompatError("Installed v2.12 decoder tuple drifted")
    cpu_vec_isa = importlib.import_module("torch._inductor.cpu_vec_isa")
    if sha256_file(Path(cpu_vec_isa.__file__)) != EXPECTED_CPU_VEC_ISA_SHA256:
        raise DecodeCompatError("Installed cpu_vec_isa source drifted")
    cache_before = {
        "is_msvc_cl": cpp_builder._is_msvc_cl.cache_info().currsize,
        "check_compiler_exist_windows": (
            cpp_builder.check_compiler_exist_windows.cache_info().currsize
        ),
        "check_msvc_cl_language_id": (cpp_builder.check_msvc_cl_language_id.cache_info().currsize),
        "valid_vec_isa_list": cpu_vec_isa.valid_vec_isa_list.cache_info().currsize,
    }
    if any(cache_before.values()):
        raise DecodeCompatError("Compiler caches were populated before the decoder shim")
    # 只改当前进程内的 module global，精确复现 v2.13 的 errors="replace"，不写 wheel。
    cpp_builder.SUBPROCESS_DECODE_ARGS = (preferred, "replace")
    after = tuple(cpp_builder.SUBPROCESS_DECODE_ARGS)
    if after != (preferred, "replace"):
        raise DecodeCompatError("Process-local v2.13 decoder shim did not apply")
    return (
        torch,
        cpp_builder,
        cpu_vec_isa,
        {
            "preferred_encoding": preferred,
            "before_args": list(before),
            "after_args": list(after),
            "watched_modules_before": modules_before,
            "compiler_modules_after_torch_import": compiler_modules_after_torch,
            "cache_sizes_before": cache_before,
            "installed_file_sha256_before": EXPECTED_CPP_BUILDER_SHA256,
            "process_local": True,
            "installed_file_modified": False,
        },
    )


def _assert_decode_shim_intact(cpp_builder: Any) -> None:
    preferred = locale.getpreferredencoding()
    if tuple(cpp_builder.SUBPROCESS_DECODE_ARGS) != (preferred, "replace"):
        raise DecodeCompatError("Process-local decoder shim drifted before the protected call")
    if sha256_file(CPP_BUILDER) != EXPECTED_CPP_BUILDER_SHA256:
        raise DecodeCompatError("Installed cpp_builder changed during the worker")


def _resolve_compiler_on_path(value: str) -> Path:
    resolved = shutil.which(value)
    if not resolved:
        raise DecodeCompatError(f"Compiler is not resolvable on worker PATH: {value}")
    path = Path(resolved).resolve(strict=True)
    if sha256_file(path) != EXPECTED_CL_SHA256:
        raise DecodeCompatError("Resolved compiler hash drifted")
    return path


def _probe_compiler(cpp_builder: Any, telemetry: dict[str, Any]) -> dict[str, Any]:
    compiler_count_before = telemetry["compiler_processes"]
    compiler_path = _resolve_compiler_on_path("cl.exe")
    completed = subprocess.run(
        [str(compiler_path), "/help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        timeout=60,
        check=False,
    )
    raw = completed.stdout or b""
    if completed.returncode != 0 or not raw:
        raise DecodeCompatError("Raw MSVC help probe failed")
    try:
        strict_utf8 = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DecodeCompatError(
            "MSVC help output is not strict UTF-8 under code page 65001"
        ) from exc
    if "Microsoft" not in strict_utf8.splitlines()[0]:
        raise DecodeCompatError("MSVC help output lacks its ASCII identity token")
    preferred = locale.getpreferredencoding()
    strict_preferred_failure = None
    try:
        raw.decode(preferred, errors="strict")
    except UnicodeDecodeError as exc:
        strict_preferred_failure = {
            "start": exc.start,
            "end": exc.end,
            "byte_hex": raw[exc.start : exc.end].hex(),
        }
    if strict_preferred_failure is None:
        raise DecodeCompatError("Prior strict preferred-encoding failure was not reproduced")
    replacement = raw.decode(preferred, errors="replace")
    replacement_count = replacement.count("\ufffd")
    if "Microsoft" not in replacement.splitlines()[0] or replacement_count == 0:
        raise DecodeCompatError("Upstream replacement-decode control did not reproduce")
    if cpp_builder._is_msvc_cl("cl") is not True:
        raise DecodeCompatError("Shimmed PyTorch MSVC identity probe failed")
    compiler = cpp_builder.get_cpp_compiler()
    resolved = _resolve_compiler_on_path(compiler)
    cache_after = cpp_builder._is_msvc_cl.cache_info()
    if cache_after.currsize == 0:
        raise DecodeCompatError("MSVC identity cache was not populated by exact probes")
    stage_compiler_processes = telemetry["compiler_processes"] - compiler_count_before
    if stage_compiler_processes > 4:
        raise DecodeCompatError("Decode preflight exceeded its compiler-process budget")
    return {
        "raw_help": {
            "command": [str(compiler_path), "/help"],
            "returncode": completed.returncode,
            "size_bytes": len(raw),
            "sha256": _sha256_bytes(raw),
        },
        "decode_matrix": {
            "strict_utf8": {"passed": True, "microsoft_identity_present": True},
            "v212_strict_preferred": {
                "passed": False,
                "failure": strict_preferred_failure,
            },
            "v213_preferred_replace": {
                "passed": True,
                "replacement_character_count": replacement_count,
                "microsoft_identity_present": True,
            },
        },
        "exact_failure_path": {
            "function": "torch._inductor.cpp_builder._is_msvc_cl",
            "installed_source_line": 528,
            "resolved_compiler": str(resolved),
            "resolved_compiler_sha256": sha256_file(resolved),
            "is_msvc_cl": True,
            "is_msvc_cl_cache_size_after": cache_after.currsize,
            "compiler_processes": stage_compiler_processes,
            "compiler_help_processes": telemetry["compiler_help_processes"],
        },
    }


def _worker_environment_contract(temp_root: Path) -> dict[str, Any]:
    if sys.flags.utf8_mode != 0:
        raise DecodeCompatError("Selected worker must run with Python UTF-8 mode disabled")
    preferred = locale.getpreferredencoding()
    if preferred.casefold() not in {"cp936", "gbk"}:
        raise DecodeCompatError("Worker preferred encoding drifted")
    declared_temp = Path(os.environ.get("TEMP", "")).resolve(strict=True)
    declared_tmp = Path(os.environ.get("TMP", "")).resolve(strict=True)
    if declared_temp != temp_root.resolve(strict=True) or declared_tmp != declared_temp:
        raise DecodeCompatError("Worker TEMP/TMP is not the declared disposable root")
    cache = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")).resolve(strict=False)
    if not cache.is_relative_to(declared_temp):
        raise DecodeCompatError("Inductor cache escapes disposable TEMP")
    code_pages = _console_code_pages()
    if code_pages != {"input": 65001, "output": 65001}:
        raise DecodeCompatError("Worker console code pages are not both UTF-8")
    current_python = Path(sys.executable).resolve(strict=True)
    if current_python != PYTHON_EXE.resolve(strict=True):
        raise DecodeCompatError("Worker interpreter is not the frozen Python executable")
    if sha256_file(current_python) != EXPECTED_PYTHON_SHA256:
        raise DecodeCompatError("Worker interpreter hash drifted")
    if os.environ.get("TORCHINDUCTOR_COMPILE_THREADS") != "1":
        raise DecodeCompatError("Worker did not freeze single-process Inductor compilation")
    return {
        "python_executable": str(current_python),
        "python_executable_sha256": sha256_file(current_python),
        "utf8_mode": sys.flags.utf8_mode,
        "preferred_encoding": preferred,
        "console_code_pages": code_pages,
        "temp_root": str(declared_temp),
        "inductor_cache": str(cache),
        "torchinductor_compile_threads_env": 1,
        "torchinductor_autotune_in_subproc_env": os.environ.get(
            "TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC"
        ),
    }


def _new_input_array():
    import numpy as np  # noqa: PLC0415

    array = np.asarray(NEW_INPUT_VALUES, dtype=np.float32).reshape(2, 8)
    payload = array.tobytes(order="C")
    digest = _sha256_bytes(payload)
    if len(payload) != 64 or digest != EXPECTED_INPUT_SHA256:
        raise DecodeCompatError("Successor input constant drifted")
    if digest == PARENT_PARTIAL_INPUT_SHA256:
        raise DecodeCompatError("Successor input unexpectedly matches the parent partial")
    return array, payload


def _worker_preflight(
    temp_root: Path, counters: dict[str, int], journal: _WorkerEventJournal
) -> dict[str, Any]:
    telemetry = _new_process_telemetry()
    environment: dict[str, Any] = {}
    shim: dict[str, Any] = {}
    compiler: dict[str, Any] = {}
    torch_record: dict[str, Any] = {}
    try:
        environment = _worker_environment_contract(temp_root)
        with _capture_process_telemetry(telemetry, journal=journal, counters=counters):
            torch, cpp_builder, _cpu_vec_isa, shim = _apply_v213_decode_shim(counters, journal)
            cuda_before = torch.cuda.is_initialized()
            torch_record = {
                "version": torch.__version__,
                "cuda_initialized_before": cuda_before,
                "cuda_initialized_after": torch.cuda.is_initialized(),
                "cpp_builder_sha256_after": sha256_file(CPP_BUILDER),
                "cpu_vec_isa_sha256_after": sha256_file(CPU_VEC_ISA),
            }
            compiler = _probe_compiler(cpp_builder, telemetry)
            _assert_decode_shim_intact(cpp_builder)
            cuda_after = torch.cuda.is_initialized()
        if cuda_before or cuda_after:
            raise DecodeCompatError("CUDA initialized during decode preflight")
        torch_record["cuda_initialized_after"] = cuda_after
        _sync_process_counters(counters, telemetry)
        return {
            "schema": "axon_loop28_pytorch_native_decode_probe_worker_v1",
            "loop_id": LOOP_ID,
            "status": "passed",
            "environment": environment,
            "torch": torch_record,
            "shim": shim,
            "compiler": compiler,
            "process_telemetry": telemetry,
            "counters": counters,
            "decision": "upstream_v213_process_local_decode_preflight_passed",
        }
    except Exception as exc:
        _sync_process_counters(counters, telemetry)
        if counters["torch_imports"] and "torch" in locals():
            torch_record = {
                "version": torch.__version__,
                "cuda_initialized_before": bool(
                    torch_record.get("cuda_initialized_before", torch.cuda.is_initialized())
                ),
                "cuda_initialized_after": torch.cuda.is_initialized(),
                "cpp_builder_sha256_after": sha256_file(CPP_BUILDER),
                "cpu_vec_isa_sha256_after": sha256_file(CPU_VEC_ISA),
            }
        return {
            "schema": "axon_loop28_pytorch_native_decode_probe_worker_failure_v1",
            "loop_id": LOOP_ID,
            "status": "failed",
            "environment": environment,
            "torch": torch_record,
            "shim": shim,
            "compiler": compiler,
            "process_telemetry": telemetry,
            "counters": counters,
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "decision": "decode_compat_preflight_worker_failed_no_package_call",
        }


def _inventory_artifact_root(artifact_root: Path, ownership_token: str) -> list[dict[str, Any]]:
    owner_path = artifact_root / ARTIFACT_OWNER_NAME
    owner = load_json_strict(owner_path)
    if owner.get("ownership_token") != ownership_token or owner.get("loop_id") != LOOP_ID:
        raise DecodeCompatError("Artifact ownership marker mismatch")
    _audit_tree_no_reparse(artifact_root)
    records: list[dict[str, Any]] = []
    for path in sorted(value for value in artifact_root.rglob("*") if value.is_file()):
        _assert_regular_single_link(path, "Successor artifact")
        relative = path.relative_to(PROJECT_ROOT.resolve(strict=True))
        if not path.resolve(strict=True).is_relative_to(artifact_root.resolve(strict=True)):
            raise DecodeCompatError("Successor partial escapes its fresh artifact root")
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _worker_package(
    temp_root: Path,
    artifact_root: Path,
    ownership_token: str,
    counters: dict[str, int],
    journal: _WorkerEventJournal,
) -> dict[str, Any]:
    telemetry = _new_process_telemetry()
    environment: dict[str, Any] = {}
    shim: dict[str, Any] = {}
    compiler: dict[str, Any] = {}
    torch_record: dict[str, Any] = {}
    try:
        environment = _worker_environment_contract(temp_root)
        with _capture_process_telemetry(telemetry, journal=journal, counters=counters):
            torch, cpp_builder, _cpu_vec_isa, shim = _apply_v213_decode_shim(counters, journal)
            cuda_before = torch.cuda.is_initialized()
            torch_record = {
                "version": torch.__version__,
                "cuda_initialized_before": cuda_before,
                "cuda_initialized_after": torch.cuda.is_initialized(),
                "cpp_builder_sha256_after": sha256_file(CPP_BUILDER),
                "cpu_vec_isa_sha256_after": sha256_file(CPU_VEC_ISA),
            }
            compiler = _probe_compiler(cpp_builder, telemetry)
            import torch._inductor as inductor  # noqa: PLC0415
            from torch._inductor import config as inductor_config  # noqa: PLC0415

            if cuda_before:
                raise DecodeCompatError("CUDA initialized before package generation")
            if int(inductor_config.compile_threads) != 1:
                raise DecodeCompatError("Inductor compile_threads is not frozen to one")
            if bool(inductor_config.autotune_in_subproc):
                raise DecodeCompatError("Inductor autotune subprocesses are not disabled")
            _mkdir_fresh(artifact_root, PROJECT_ROOT)
            _write_json_exclusive(
                artifact_root / ARTIFACT_OWNER_NAME,
                {
                    "schema": "axon_loop28_pytorch_native_decode_artifact_owner_v1",
                    "loop_id": LOOP_ID,
                    "attempt_id": _stage_contract("package")["attempt_id"],
                    "ownership_token": ownership_token,
                },
            )
            input_array, input_payload = _new_input_array()
            input_path = artifact_root / INPUT_PATH.name
            package_path = artifact_root / AOTI_PACKAGE.name
            _write_exclusive(input_path, input_payload)
            _assert_regular_single_link(input_path, "Successor synthetic input")
            torch.set_num_threads(1)
            input_tensor = torch.from_numpy(input_array.copy())
            base = _load_module(BASE_MODEL_SOURCE, "loop28_frozen_tiny_model_source_v2")
            if sha256_file(PROJECT_ROOT / BASE_MODEL_SOURCE) != EXPECTED_BASE_MODEL_SHA256:
                raise DecodeCompatError("Frozen tiny model source drifted before construction")
            counters["model_constructions"] += 1
            journal.append("model_construction_about_to_start", counters)
            model = base.build_tiny_model()
            counters["torch_export_calls"] += 1
            journal.append("torch_export_about_to_start", counters)
            exported = torch.export.export(model, (input_tensor,), strict=True)
            counters["torch_export_completed"] += 1
            journal.append("torch_export_completed", counters)
            _assert_decode_shim_intact(cpp_builder)
            cpp_builder._is_msvc_cl.cache_clear()
            cpp_builder.check_msvc_cl_language_id.cache_clear()
            if (
                cpp_builder._is_msvc_cl.cache_info().currsize != 0
                or cpp_builder.check_msvc_cl_language_id.cache_info().currsize != 0
            ):
                raise DecodeCompatError("Exact MSVC decode path caches did not clear")
            counters["aoti_compile_and_package_calls"] += 1
            journal.append("aoti_compile_and_package_about_to_start", counters)
            inductor.aoti_compile_and_package(exported, package_path=package_path)
            counters["aoti_compile_and_package_completed"] += 1
            journal.append("aoti_compile_and_package_completed", counters)
        if not package_path.is_file():
            raise DecodeCompatError("AOTInductor omitted the successor package")
        _assert_regular_single_link(package_path, "Successor AOTInductor package")
        cuda_after = torch.cuda.is_initialized()
        if cuda_after:
            raise DecodeCompatError("CUDA initialized during package generation")
        torch_record.update(
            {
                "cuda_initialized_after": cuda_after,
                "cpu_threads": torch.get_num_threads(),
                "inductor_compile_threads": int(inductor_config.compile_threads),
                "inductor_autotune_in_subproc": bool(inductor_config.autotune_in_subproc),
            }
        )
        _sync_process_counters(counters, telemetry)
        if counters["compiler_help_processes"] != 4:
            raise DecodeCompatError("AOTI did not re-enter the exact shimmed MSVC decode path")
        if (
            cpp_builder._is_msvc_cl.cache_info().currsize == 0
            or cpp_builder.check_msvc_cl_language_id.cache_info().currsize == 0
        ):
            raise DecodeCompatError("AOTI did not repopulate the exact MSVC decode caches")
        return {
            "schema": "axon_loop28_pytorch_native_decode_compat_package_worker_v1",
            "loop_id": LOOP_ID,
            "status": "passed",
            "environment": environment,
            "torch": torch_record,
            "shim": shim,
            "compiler": compiler,
            "input": {
                "path": INPUT_PATH.as_posix(),
                "sha256": sha256_file(input_path),
                "size_bytes": input_path.stat().st_size,
                "parent_partial_sha256": PARENT_PARTIAL_INPUT_SHA256,
                "differs_from_parent_partial": True,
            },
            "package": {
                "path": AOTI_PACKAGE.as_posix(),
                "sha256": sha256_file(package_path),
                "size_bytes": package_path.stat().st_size,
            },
            "process_telemetry": telemetry,
            "counters": counters,
            "decision": "decode_compat_package_worker_generated_static_audit_required",
        }
    except Exception as exc:
        _sync_process_counters(counters, telemetry)
        if counters["torch_imports"] and "torch" in locals():
            torch_record.update(
                {
                    "version": torch.__version__,
                    "cuda_initialized_before": bool(
                        torch_record.get("cuda_initialized_before", torch.cuda.is_initialized())
                    ),
                    "cuda_initialized_after": torch.cuda.is_initialized(),
                    "cpp_builder_sha256_after": sha256_file(CPP_BUILDER),
                    "cpu_vec_isa_sha256_after": sha256_file(CPU_VEC_ISA),
                }
            )
        partials: list[dict[str, Any]] = []
        partial_inventory_error = None
        if _lexists(artifact_root):
            try:
                partials = _inventory_artifact_root(artifact_root, ownership_token)
            except Exception as inventory_exc:
                partial_inventory_error = (
                    f"{type(inventory_exc).__name__}: {str(inventory_exc)[:1000]}"
                )
        return {
            "schema": "axon_loop28_pytorch_native_decode_compat_package_worker_failure_v1",
            "loop_id": LOOP_ID,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "environment": environment,
            "shim": shim,
            "compiler": compiler,
            "torch": torch_record,
            "process_telemetry": telemetry,
            "counters": counters,
            "partial_artifacts": partials,
            "partial_inventory_error": partial_inventory_error,
            "decision": "decode_compat_package_worker_failed_no_load",
        }


def run_worker(stage: str, ownership_token: str) -> int:
    paths = _verify_worker_owned_paths(PROJECT_ROOT, stage, ownership_token)
    counters = _base_counters()
    journal = _WorkerEventJournal(paths["events_root"], stage)
    journal.append("worker_started", counters)
    if stage == "preflight":
        result = _worker_preflight(paths["temp_root"], counters, journal)
    elif stage == "package":
        result = _worker_package(
            paths["temp_root"],
            paths["artifact_root"],
            ownership_token,
            counters,
            journal,
        )
    else:
        raise DecodeCompatError(f"Unsupported worker stage: {stage}")
    journal.append("worker_result_ready", result["counters"])
    _write_json_exclusive(paths["worker_output"], result)
    return 0 if result.get("status") == "passed" else 1


def _unsafe_windows_component(component: str) -> bool:
    if not component or component in {".", ".."}:
        return True
    if any(ord(character) < 32 for character in component):
        return True
    if ":" in component or component.endswith((" ", ".")):
        return True
    trimmed = component.rstrip(" .")
    stem = trimmed.split(".", 1)[0].casefold()
    return stem in WINDOWS_RESERVED_NAMES


def audit_package_archive(path: Path) -> dict[str, Any]:
    # 按 Windows 解压语义同时审计原始路径、Unicode/casefold 碰撞和特殊文件类型。
    if not path.is_file() or path.stat().st_size > MAX_RETAINED_BYTES:
        raise DecodeCompatError("Successor package is missing or exceeds the size cap")
    canonical_names: set[str] = set()
    exact_names: set[str] = set()
    members: list[dict[str, Any]] = []
    pyd_members: list[dict[str, Any]] = []
    total_uncompressed = 0
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_ARCHIVE_MEMBERS:
            raise DecodeCompatError("Successor archive member count is outside the contract")
        for info in infos:
            name = getattr(info, "orig_filename", info.filename)
            if name in exact_names:
                raise DecodeCompatError("Successor archive contains exact duplicate names")
            exact_names.add(name)
            if not name or "\x00" in name or "\\" in name or name.startswith(("/", "//")):
                raise DecodeCompatError("Successor archive contains an unsafe raw path")
            pure = PurePosixPath(name)
            if pure.is_absolute() or any(_unsafe_windows_component(part) for part in pure.parts):
                raise DecodeCompatError("Successor archive path is unsafe on Windows")
            canonical_parts = [
                unicodedata.normalize("NFKC", part.rstrip(" .")).casefold() for part in pure.parts
            ]
            canonical = "/".join(canonical_parts)
            if canonical in canonical_names:
                raise DecodeCompatError("Successor archive has a Windows/Unicode collision")
            canonical_names.add(canonical)
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(unix_mode) if unix_mode else 0
            if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise DecodeCompatError("Successor archive contains a non-file special member")
            if info.flag_bits & 0x1:
                raise DecodeCompatError("Successor archive contains an encrypted member")
            if info.file_size > MAX_RETAINED_BYTES:
                raise DecodeCompatError("Successor archive member exceeds the size cap")
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_RETAINED_BYTES:
                raise DecodeCompatError("Successor archive total exceeds the size cap")
            if info.file_size > 1024 * 1024:
                ratio = info.file_size / max(info.compress_size, 1)
                if ratio > MAX_COMPRESSION_RATIO:
                    raise DecodeCompatError("Successor archive compression ratio is unsafe")
            digest = hashlib.sha256()
            decoded_size = 0
            with archive.open(info, "r") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    decoded_size += len(block)
                    digest.update(block)
            if decoded_size != info.file_size:
                raise DecodeCompatError("Successor archive member size drifted during read")
            record = {
                "name": name,
                "size_bytes": info.file_size,
                "compressed_size_bytes": info.compress_size,
                "crc32": f"{info.CRC:08x}",
                "sha256": digest.hexdigest(),
            }
            members.append(record)
            if name.casefold().endswith(".pyd"):
                pyd_members.append(record)
    if len(pyd_members) != 1:
        raise DecodeCompatError("Successor archive must contain exactly one precompiled .pyd")
    return {
        "member_count": len(members),
        "members": members,
        "pyd_member": pyd_members[0],
        "precompiled_pyd_count": 1,
        "total_uncompressed_bytes": total_uncompressed,
        "unsafe_paths": 0,
        "windows_or_unicode_collisions": 0,
        "special_members": 0,
        "encrypted_members": 0,
    }


def _dumpbin_dependency_inventory(
    path: Path, *, deadline_monotonic: float | None = None
) -> dict[str, Any]:
    dependencies: set[str] = set()
    invocations: list[dict[str, Any]] = []
    dll_pattern = re.compile(r"(?i)(?<![A-Za-z0-9_.+-])([A-Za-z0-9_.+-]+\.dll)\b")
    for mode in ("/DEPENDENTS", "/IMPORTS"):
        timeout = 60.0
        if deadline_monotonic is not None:
            remaining = deadline_monotonic - time.perf_counter()
            if remaining <= 0:
                raise DecodeCompatError("Dependency audit exhausted its wall-clock budget")
            timeout = min(timeout, remaining)
        command = [str(DUMPBIN_EXE), mode, str(path)]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=False,
            timeout=timeout,
            check=False,
        )
        output = completed.stdout or b""
        invocations.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout_sha256": _sha256_bytes(output),
                "stdout_bytes": len(output),
                "stderr_sha256": _sha256_bytes(completed.stderr or b""),
                "stderr_bytes": len(completed.stderr or b""),
            }
        )
        if completed.returncode != 0:
            raise DecodeCompatError(f"dumpbin {mode} audit failed: {path.name}")
        decoded = output.decode("utf-8", errors="replace")
        dependencies.update(match.casefold() for match in dll_pattern.findall(decoded))
    dependencies.discard(path.name.casefold())
    return {
        "dependencies": sorted(dependencies),
        "invocations": invocations,
        "regular_and_delay_imports_audited": True,
    }


def _forbidden_dependency(name: str) -> bool:
    lowered = name.casefold()
    tokens = (
        "python",
        "torch_python",
        "torch_cuda",
        "cudart",
        "cublas",
        "cudnn",
        "cuda",
        "nvrtc",
        "nvjitlink",
        "cupti",
        "nvcuda",
    )
    return lowered == "torch.dll" or any(token in lowered for token in tokens)


def _directory_index(path: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    if not path.is_dir():
        return index
    for child in path.iterdir():
        if child.is_file():
            index.setdefault(child.name.casefold(), []).append(child.resolve(strict=True))
    return index


def build_dependency_closure(
    starts: Mapping[str, Path], *, deadline_monotonic: float | None = None
) -> dict[str, Any]:
    # 同时解析普通/延迟导入；同名 DLL 的多个路径即视为歧义，不能按内容哈希折叠。
    system_root = Path(os.environ.get("SystemRoot", "C:/Windows"))
    search_roots = [TORCH_LIB.resolve(strict=True), (system_root / "System32").resolve(strict=True)]
    indexes = [_directory_index(root) for root in search_roots]
    queue: deque[tuple[str, Path]] = deque(
        (name, path.resolve(strict=True)) for name, path in starts.items()
    )
    visited_paths: set[str] = set()
    nodes: list[dict[str, Any]] = []
    resolved_leaves: dict[str, dict[str, Any]] = {}
    virtual_system_dependencies: set[str] = set()
    dumpbin_invocations: list[dict[str, Any]] = []
    unresolved: set[str] = set()
    ambiguous: dict[str, list[str]] = {}
    forbidden_hits: set[str] = set()
    while queue:
        role, path = queue.popleft()
        key = str(path).casefold()
        if key in visited_paths:
            continue
        if len(visited_paths) >= 512:
            raise DecodeCompatError("Dependency closure exceeded its node cap")
        visited_paths.add(key)
        if len(dumpbin_invocations) + 2 > PACKAGE_BUDGET["dumpbin_processes_max"]:
            raise DecodeCompatError("Dependency audit exhausted its dumpbin process budget")
        if deadline_monotonic is not None and time.perf_counter() >= deadline_monotonic:
            raise DecodeCompatError("Dependency audit exhausted its wall-clock budget")
        inventory = _dumpbin_dependency_inventory(path, deadline_monotonic=deadline_monotonic)
        dependencies = inventory["dependencies"]
        dumpbin_invocations.extend(inventory["invocations"])
        nodes.append(
            {
                "role": role,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "dependencies": dependencies,
                "regular_and_delay_imports_audited": True,
            }
        )
        for dependency in dependencies:
            if _forbidden_dependency(dependency):
                forbidden_hits.add(dependency)
                continue
            if dependency.startswith(("api-ms-win-", "ext-ms-win-")):
                virtual_system_dependencies.add(dependency)
                continue
            matches: list[Path] = []
            for index in indexes:
                matches.extend(index.get(dependency, []))
            unique = {str(match).casefold(): match for match in matches}
            if not unique:
                unresolved.add(dependency)
                continue
            if len(unique) > 1:
                ambiguous[dependency] = sorted(str(match) for match in unique.values())
                continue
            resolved = next(iter(unique.values()))
            resolved_leaves[str(resolved).casefold()] = {
                "dependency": dependency,
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
                "system_leaf": not resolved.is_relative_to(TORCH_LIB.resolve(strict=True)),
            }
            if resolved.is_relative_to(TORCH_LIB.resolve(strict=True)):
                queue.append((f"dependency:{dependency}", resolved))
    if forbidden_hits or unresolved or ambiguous:
        raise DecodeCompatError(
            "Dependency closure is unsafe: "
            f"forbidden={sorted(forbidden_hits)}, unresolved={sorted(unresolved)}, "
            f"ambiguous={sorted(ambiguous)}"
        )
    return {
        "nodes": nodes,
        "resolved_modules": sorted(
            resolved_leaves.values(), key=lambda item: item["path"].casefold()
        ),
        "virtual_system_dependencies": sorted(virtual_system_dependencies),
        "dumpbin_invocations": dumpbin_invocations,
        "dumpbin_invocation_count": len(dumpbin_invocations),
        "regular_and_delay_imports_audited": True,
        "forbidden_hits": [],
        "unresolved": [],
        "ambiguous": {},
        "search_roots": [str(root) for root in search_roots],
    }


def _audit_generated_package(
    project_root: Path,
    worker: Mapping[str, Any],
    ownership_token: str,
    *,
    deadline_monotonic: float,
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    artifact_root = _project_output_path(root, ARTIFACT_ROOT)
    artifact_inventory = _inventory_artifact_root(artifact_root, ownership_token)
    input_path = _resolve_within(root, INPUT_PATH)
    package_path = _resolve_within(root, AOTI_PACKAGE)
    _assert_regular_single_link(input_path, "Successor synthetic input")
    _assert_regular_single_link(package_path, "Successor package")
    if (
        input_path.stat().st_size != 64
        or sha256_file(input_path) != EXPECTED_INPUT_SHA256
        or sha256_file(input_path) == PARENT_PARTIAL_INPUT_SHA256
    ):
        raise DecodeCompatError("Successor input postcondition failed")
    if worker.get("input", {}).get("sha256") != EXPECTED_INPUT_SHA256:
        raise DecodeCompatError("Worker input receipt drifted")
    if worker.get("package", {}).get("sha256") != sha256_file(package_path):
        raise DecodeCompatError("Worker package receipt drifted")
    archive = audit_package_archive(package_path)
    pyd_name = archive["pyd_member"]["name"]
    work_parent = _project_output_path(root, DEPENDENCY_AUDIT_ROOT)
    _mkdir_fresh(work_parent, root)
    _write_json_exclusive(
        work_parent / WORK_OWNER_NAME,
        {
            "schema": "axon_loop28_pytorch_native_decode_dependency_audit_owner_v1",
            "loop_id": LOOP_ID,
            "ownership_token": ownership_token,
        },
    )
    extracted = work_parent / "model.pyd"
    try:
        with zipfile.ZipFile(package_path, "r") as package_archive:
            with package_archive.open(pyd_name, "r") as source, extracted.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
        if sha256_file(extracted) != archive["pyd_member"]["sha256"]:
            raise DecodeCompatError("Streamed .pyd hash differs from archive inventory")
        _assert_regular_single_link(extracted, "Extracted dependency-audit module")
        dependencies = build_dependency_closure(
            {
                "generated_pyd": extracted,
                "aoti_host": root / AOTI_HOST,
                "aten_host": root / ATEN_HOST,
            },
            deadline_monotonic=deadline_monotonic,
        )
        for node in dependencies["nodes"]:
            if node.get("role") == "generated_pyd":
                node["archive_member_name"] = pyd_name
                node["archive_member_sha256"] = archive["pyd_member"]["sha256"]
    finally:
        _audit_tree_no_reparse(work_parent)
        shutil.rmtree(work_parent)
    owner_path = artifact_root / ARTIFACT_OWNER_NAME
    retained = input_path.stat().st_size + package_path.stat().st_size + owner_path.stat().st_size
    if retained > MAX_RETAINED_BYTES:
        raise DecodeCompatError("Successor retained outputs exceed the cap")
    return {
        "input": {
            "path": INPUT_PATH.as_posix(),
            "sha256": sha256_file(input_path),
            "size_bytes": input_path.stat().st_size,
            "differs_from_parent_partial": True,
        },
        "package": {
            "path": AOTI_PACKAGE.as_posix(),
            "sha256": sha256_file(package_path),
            "size_bytes": package_path.stat().st_size,
        },
        "archive": archive,
        "dependency_closure": dependencies,
        "artifact_inventory": artifact_inventory,
        "artifact_owner": {
            "path": owner_path.relative_to(root).as_posix(),
            "sha256": sha256_file(owner_path),
            "size_bytes": owner_path.stat().st_size,
        },
        "toolchain": {
            "dumpbin_path": str(DUMPBIN_EXE.resolve(strict=True)),
            "dumpbin_sha256": sha256_file(DUMPBIN_EXE),
        },
        "retained_output_bytes": retained,
    }


def _launch_passed(launch: Mapping[str, Any]) -> bool:
    code_pages = launch.get("console_code_pages")
    return bool(
        launch.get("returncode") == 0
        and launch.get("timed_out") is False
        and launch.get("launch_error") is None
        and launch.get("worker_parse_error") is None
        and isinstance(code_pages, dict)
        and code_pages.get("restore_verified") is True
        and launch.get("temp_cleanup", {}).get("removed") is True
    )


def _source_snapshot_after(
    project_root: Path, authorization: Mapping[str, Any]
) -> list[dict[str, Any]]:
    snapshot = _verify_source_records(project_root, authorization)
    _verify_static_parent_chain(project_root)
    return snapshot


def _cleanup_worker_after_attempt(
    project_root: Path, stage: str, lease: Mapping[str, Any]
) -> dict[str, Any]:
    return _remove_owned_worker_root(project_root, stage, _worker_ownership_token(stage, lease))


def _prelaunch_failure_evidence(project_root: Path, stage: str, error: str) -> dict[str, Any]:
    paths = _stage_runtime_paths(project_root, stage)
    try:
        code_pages = _console_code_pages()
        code_page_error = None
    except Exception as exc:
        code_pages = None
        code_page_error = f"{type(exc).__name__}: {str(exc)[:500]}"
    return {
        "command": ["cmd.exe", "/d", "/c", "launch_worker.cmd"],
        "process_started": False,
        "returncode": None,
        "elapsed_seconds": 0.0,
        "timed_out": False,
        "process_tree_termination": None,
        "worker_parse_error": None,
        "launch_error": error[:1000],
        "temp_cleanup": {
            "attempted": False,
            "removed": not _lexists(paths["temp_root"]),
        },
        "durable_worker_journal": {
            "record_count": 0,
            "records": [],
            "last_counters": _base_counters(),
        },
        "console_code_pages": {
            "before": code_pages,
            "configured": None,
            "after_worker": code_pages,
            "restored": code_pages,
            "restore_verified": code_pages is not None,
            "inspection_error": code_page_error,
        },
    }


def _package_failure_class(
    worker: Mapping[str, Any], error: str, launch: Mapping[str, Any] | None = None
) -> tuple[str, str]:
    counters = worker.get("counters")
    if not isinstance(counters, dict):
        counters = {}
    aoti_calls = int(counters.get("aoti_compile_and_package_calls", 0) or 0)
    aoti_completed = int(counters.get("aoti_compile_and_package_completed", 0) or 0)
    lowered = error.casefold()
    if (isinstance(launch, Mapping) and launch.get("timed_out") is True) or any(
        token in lowered for token in ("budget", "deadline", "timed out", "timeout")
    ):
        return "budget", "budget_exhausted_no_claim"
    if worker.get("status") == "passed" or aoti_completed == 1:
        if any(
            token in lowered for token in ("dependency", "forbidden", "unresolved", "ambiguous")
        ):
            return "dependency", "decode_compat_package_dependency_leakage_no_load"
        return "static_audit", "decode_compat_package_static_audit_failed_no_load"
    if aoti_calls > 0:
        return (
            "protected_call",
            "decode_compat_applied_aoti_compile_or_package_still_unsupported",
        )
    if any(
        int(counters.get(name, 0) or 0) > 0
        for name in ("torch_imports", "model_constructions", "torch_export_calls")
    ):
        return "pre_export", "decode_compat_pre_export_failure_no_package"
    return "administrative", "administrative_failure_no_protected_package_call"


def _budget_actual(
    worker: Mapping[str, Any],
    launch: Mapping[str, Any],
    *,
    dumpbin_processes: int,
    wall_clock_seconds: float,
    retained_output_bytes: int,
) -> dict[str, Any]:
    counters = worker.get("counters")
    if not isinstance(counters, dict):
        counters = {}
    return {
        "worker_processes": 1 if launch.get("process_started") is True else 0,
        "vcvars_activations": 1 if launch.get("process_started") is True else 0,
        "compiler_help_processes": int(counters.get("compiler_help_processes", 0) or 0),
        "dumpbin_processes": int(dumpbin_processes),
        "wall_clock_seconds": max(0.0, float(wall_clock_seconds)),
        "retained_output_bytes": int(retained_output_bytes),
    }


def run_preflight(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    if root != PROJECT_ROOT.resolve(strict=True):
        raise DecodeCompatError("Preflight project root must be the canonical workspace")
    gate = _verify_stage_ready_gate(root, "preflight")
    contract = gate["contract"]
    lease = _consume_lease(
        root,
        authorization_path=contract["authorization_path"],
        ready_path=contract["ready_path"],
        final_path=contract["final_path"],
        authorization_schema=contract["authorization_schema"],
        authorization_decision=contract["authorization_decision"],
        lease_schema=contract["lease_schema"],
        expected_authorization_sha256=gate["authorization_sha256"],
        expected_ready_sha256=gate["ready_sha256"],
    )
    attempt_started = time.perf_counter()
    attempt_deadline = attempt_started + float(contract["budget"]["wall_clock_seconds_max"])
    worker: dict[str, Any] = {}
    launch: dict[str, Any] = {}
    cleanup: dict[str, Any] = {"attempted": False, "removed": True}
    try:
        worker, launch = _launch_worker(
            project_root=root,
            stage="preflight",
            lease=lease,
            deadline_monotonic=attempt_deadline,
        )
        cleanup = _cleanup_worker_after_attempt(root, "preflight", lease)
        if not _launch_passed(launch) or worker.get("status") != "passed":
            raise DecodeCompatError(
                f"preflight worker failed: rc={launch.get('returncode')}, "
                f"error={worker.get('error', 'unknown')}"
            )
        if _lexists(_project_output_path(root, ARTIFACT_ROOT)):
            raise DecodeCompatError("Preflight created the forbidden artifact root")
        source_after = _source_snapshot_after(root, gate["authorization"])
        if source_after != gate["source_artifacts"]:
            raise DecodeCompatError("Preflight source inventory changed during execution")
        payload = {
            "schema": "axon_loop28_pytorch_native_decode_probe_evidence_v1",
            "loop_id": LOOP_ID,
            "lease": lease,
            "worker": worker,
            "launch": launch,
            "attempt": {
                "attempt_id": contract["attempt_id"],
                "lease_id": contract["lease_id"],
                "canonical_command": _canonical_stage_argv("preflight"),
                "artifact_root": ARTIFACT_ROOT.as_posix(),
                "work_root": contract["work_root"].as_posix(),
                "terminal_outputs": [path.as_posix() for path in contract["terminal_paths"]],
                "budget_sha256": gate["budget_digest"],
            },
            "source_artifacts_before": gate["source_artifacts"],
            "source_artifacts_after": source_after,
            "controller_interpreter": gate["interpreter"],
            "work_root_cleanup": cleanup,
            "budget_actual": _budget_actual(
                worker,
                launch,
                dumpbin_processes=0,
                wall_clock_seconds=time.perf_counter() - attempt_started,
                retained_output_bytes=0,
            ),
            "installed_files_modified": False,
            "artifact_root_created": False,
            "decision": (
                "upstream_v213_process_local_decode_preflight_passed_"
                "package_implementation_may_freeze"
            ),
        }
        _write_json_exclusive(_project_output_path(root, PREFLIGHT_EVIDENCE), payload)
        return payload
    except Exception as exc:
        if not isinstance(worker.get("counters"), dict):
            worker = _administrative_worker_failure(
                "preflight", None, "Worker launch failed before a durable receipt"
            )
        cleanup_error = None
        try:
            cleanup = _cleanup_worker_after_attempt(root, "preflight", lease)
        except Exception as cleanup_exc:
            cleanup_error = f"{type(cleanup_exc).__name__}: {str(cleanup_exc)[:1000]}"
        if not launch:
            launch = _prelaunch_failure_evidence(root, "preflight", str(exc))
        source_after: list[dict[str, Any]] = []
        source_after_error = None
        try:
            source_after = _source_snapshot_after(root, gate["authorization"])
        except Exception as source_exc:
            source_after_error = f"{type(source_exc).__name__}: {str(source_exc)[:1000]}"
        failure = {
            "schema": "axon_loop28_pytorch_native_decode_probe_failure_v1",
            "loop_id": LOOP_ID,
            "lease": lease,
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "worker": worker,
            "launch": launch,
            "attempt": {
                "attempt_id": contract["attempt_id"],
                "lease_id": contract["lease_id"],
                "canonical_command": _canonical_stage_argv("preflight"),
                "artifact_root": ARTIFACT_ROOT.as_posix(),
                "work_root": contract["work_root"].as_posix(),
                "terminal_outputs": [path.as_posix() for path in contract["terminal_paths"]],
                "budget_sha256": gate["budget_digest"],
            },
            "source_artifacts_before": gate["source_artifacts"],
            "source_artifacts_after": source_after,
            "source_artifacts_after_error": source_after_error,
            "work_root_cleanup": cleanup,
            "work_root_cleanup_error": cleanup_error,
            "budget_actual": _budget_actual(
                worker,
                launch,
                dumpbin_processes=0,
                wall_clock_seconds=time.perf_counter() - attempt_started,
                retained_output_bytes=0,
            ),
            "package_call_count": 0,
            "package_load_count": 0,
            "native_probe_execution_count": 0,
            "gpu_execution_count": 0,
            "network_request_count": 0,
            "quality_metric_count": 0,
            "decision": "decode_compat_preflight_failed_no_package_authorization",
        }
        _write_json_exclusive(_project_output_path(root, PREFLIGHT_FAILURE), failure)
        raise


def run_package(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    if root != PROJECT_ROOT.resolve(strict=True):
        raise DecodeCompatError("Package project root must be the canonical workspace")
    gate = _verify_stage_ready_gate(root, "package")
    contract = gate["contract"]
    lease = _consume_lease(
        root,
        authorization_path=contract["authorization_path"],
        ready_path=contract["ready_path"],
        final_path=contract["final_path"],
        authorization_schema=contract["authorization_schema"],
        authorization_decision=contract["authorization_decision"],
        lease_schema=contract["lease_schema"],
        expected_authorization_sha256=gate["authorization_sha256"],
        expected_ready_sha256=gate["ready_sha256"],
    )
    attempt_started = time.perf_counter()
    attempt_deadline = attempt_started + float(contract["budget"]["wall_clock_seconds_max"])
    worker: dict[str, Any] = {}
    launch: dict[str, Any] = {}
    cleanup: dict[str, Any] = {"attempted": False, "removed": True}
    token = _worker_ownership_token("package", lease)
    audit_telemetry = _new_process_telemetry()
    try:
        worker, launch = _launch_worker(
            project_root=root,
            stage="package",
            lease=lease,
            deadline_monotonic=attempt_deadline,
        )
        cleanup = _cleanup_worker_after_attempt(root, "package", lease)
        if not _launch_passed(launch) or worker.get("status") != "passed":
            raise DecodeCompatError(
                f"package worker failed: rc={launch.get('returncode')}, "
                f"error={worker.get('error', 'unknown')}"
            )
        with _capture_process_telemetry(audit_telemetry):
            audit = _audit_generated_package(
                root,
                worker,
                token,
                deadline_monotonic=attempt_deadline - TERMINAL_RESERVE_SECONDS,
            )
        source_after = _source_snapshot_after(root, gate["authorization"])
        if source_after != gate["source_artifacts"]:
            raise DecodeCompatError("Package source inventory changed during execution")
        payload = {
            "schema": "axon_loop28_pytorch_native_decode_compat_package_receipt_v1",
            "loop_id": LOOP_ID,
            "lease": lease,
            "package_authorization": {
                "path": PACKAGE_AUTHORIZATION.as_posix(),
                "sha256": gate["authorization_sha256"],
            },
            "package_final_lease": {
                "path": PACKAGE_FINAL_LEASE.as_posix(),
                "sha256": lease["sha256"],
            },
            "attempt": {
                "attempt_id": contract["attempt_id"],
                "lease_id": contract["lease_id"],
                "canonical_command": _canonical_stage_argv("package"),
                "artifact_root": ARTIFACT_ROOT.as_posix(),
                "work_root": contract["work_root"].as_posix(),
                "terminal_outputs": [path.as_posix() for path in contract["terminal_paths"]],
                "budget_sha256": gate["budget_digest"],
            },
            "implementation_manifest_sha256": sha256_file(
                _resolve_within(root, IMPLEMENTATION_MANIFEST)
            ),
            "worker": worker,
            "launch": launch,
            "audit": audit,
            "source_artifacts_before": gate["source_artifacts"],
            "source_artifacts_after": source_after,
            "work_root_cleanup": cleanup,
            "budget_actual": _budget_actual(
                worker,
                launch,
                dumpbin_processes=audit_telemetry["dumpbin_processes"],
                wall_clock_seconds=time.perf_counter() - attempt_started,
                retained_output_bytes=audit["retained_output_bytes"],
            ),
            "package_load_count": 0,
            "native_probe_execution_count": 0,
            "checkpoint_or_onnx_load_count": 0,
            "raw_split_cache_heldout_access_count": 0,
            "gpu_execution_count": 0,
            "network_request_count": 0,
            "quality_metric_count": 0,
            "claim_boundary": gate["authorization"]["claim_boundary"],
            "decision": (
                "upstream_v213_decode_compat_package_generated_dependency_closure_"
                "ready_for_new_runtime_loop"
            ),
        }
        _write_json_exclusive(_project_output_path(root, PACKAGE_RECEIPT), payload)
        return payload
    except Exception as exc:
        if not isinstance(worker.get("counters"), dict):
            worker = _administrative_worker_failure(
                "package", None, "Worker launch failed before a durable receipt"
            )
        cleanup_error = None
        try:
            cleanup = _cleanup_worker_after_attempt(root, "package", lease)
        except Exception as cleanup_exc:
            cleanup_error = f"{type(cleanup_exc).__name__}: {str(cleanup_exc)[:1000]}"
        if not launch:
            launch = _prelaunch_failure_evidence(root, "package", str(exc))
        partials: list[dict[str, Any]] = []
        partial_inventory_error = None
        artifact_root = _project_output_path(root, ARTIFACT_ROOT)
        if _lexists(artifact_root):
            try:
                partials = _inventory_artifact_root(artifact_root, token)
            except Exception as inventory_exc:
                partial_inventory_error = (
                    f"{type(inventory_exc).__name__}: {str(inventory_exc)[:1000]}"
                )
        source_after: list[dict[str, Any]] = []
        source_after_error = None
        try:
            source_after = _source_snapshot_after(root, gate["authorization"])
        except Exception as source_exc:
            source_after_error = f"{type(source_exc).__name__}: {str(source_exc)[:1000]}"
        failure_class, decision = _package_failure_class(worker, str(exc), launch)
        failure = {
            "schema": "axon_loop28_pytorch_native_decode_compat_package_failure_v1",
            "loop_id": LOOP_ID,
            "lease": lease,
            "package_authorization": {
                "path": PACKAGE_AUTHORIZATION.as_posix(),
                "sha256": gate["authorization_sha256"],
            },
            "package_final_lease": {
                "path": PACKAGE_FINAL_LEASE.as_posix(),
                "sha256": lease["sha256"],
            },
            "attempt": {
                "attempt_id": contract["attempt_id"],
                "lease_id": contract["lease_id"],
                "canonical_command": _canonical_stage_argv("package"),
                "artifact_root": ARTIFACT_ROOT.as_posix(),
                "work_root": contract["work_root"].as_posix(),
                "terminal_outputs": [path.as_posix() for path in contract["terminal_paths"]],
                "budget_sha256": gate["budget_digest"],
            },
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "failure_class": failure_class,
            "failure_reason": failure_class,
            "worker": worker,
            "launch": launch,
            "partial_artifacts": partials,
            "partial_inventory_error": partial_inventory_error,
            "source_artifacts_before": gate["source_artifacts"],
            "source_artifacts_after": source_after,
            "source_artifacts_after_error": source_after_error,
            "work_root_cleanup": cleanup,
            "work_root_cleanup_error": cleanup_error,
            "budget_actual": _budget_actual(
                worker,
                launch,
                dumpbin_processes=audit_telemetry["dumpbin_processes"],
                wall_clock_seconds=time.perf_counter() - attempt_started,
                retained_output_bytes=sum(
                    int(record.get("size_bytes", 0) or 0) for record in partials
                ),
            ),
            "package_load_count": 0,
            "native_probe_execution_count": 0,
            "checkpoint_or_onnx_load_count": 0,
            "raw_split_cache_heldout_access_count": 0,
            "gpu_execution_count": 0,
            "network_request_count": 0,
            "quality_metric_count": 0,
            "decision": decision,
        }
        _write_json_exclusive(_project_output_path(root, PACKAGE_FAILURE), failure)
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "package", "worker"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--stage", choices=("preflight", "package"))
    parser.add_argument("--ownership-token")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "worker":
        if not args.stage or not args.ownership_token:
            raise DecodeCompatError("Worker mode requires a stage and ownership token")
        return run_worker(args.stage, args.ownership_token)
    root = args.project_root.resolve(strict=True)
    payload = run_preflight(root) if args.mode == "preflight" else run_package(root)
    print(json.dumps({"mode": args.mode, "decision": payload["decision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
