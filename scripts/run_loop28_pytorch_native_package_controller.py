#!/usr/bin/env python3
"""Govern the single reissued Loop28 tiny AOTI package generation attempt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import tempfile
import unicodedata
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "p0_loop28_pytorch_native_feasibility_001"
LOOP_MANIFEST_DIR = Path("manifests/roadmap_9997/p0_loop28_pytorch_native_feasibility")
LOOP_REPORT_DIR = Path("reports/roadmap_9997/p0_loop28_pytorch_native_feasibility")
ARTIFACT_ROOT = Path("artifacts/roadmap_9997/p0_loop28_pytorch_native_feasibility/tiny_v1")
WORK_ROOT = LOOP_REPORT_DIR / "work"

BASE_RUNNER = Path("scripts/run_loop28_pytorch_native_feasibility.py")
BASE_BUILDER = Path("scripts/build_loop28_pytorch_native_feasibility_manifest.py")
IMPLEMENTATION_MANIFEST = LOOP_MANIFEST_DIR / "implementation_manifest.json"
BUILD_RECEIPT = LOOP_REPORT_DIR / "cpp_build_receipt.final.json"
REISSUE_AUTHORIZATION = LOOP_MANIFEST_DIR / "package_reissue_001_authorization.json"
CONTROLLER_MANIFEST = LOOP_MANIFEST_DIR / "package_controller_manifest.json"
PACKAGE_AUTHORIZATION = LOOP_MANIFEST_DIR / "package_authorization.json"
PACKAGE_READY_LEASE = LOOP_MANIFEST_DIR / "package_lease.json"
PACKAGE_FINAL_LEASE = LOOP_MANIFEST_DIR / "package_lease.final.json"
PACKAGE_RECEIPT = LOOP_REPORT_DIR / "package_receipt.final.json"
PACKAGE_FAILURE = LOOP_REPORT_DIR / "package_failure.final.json"
CONTROLLER_RECEIPT = LOOP_REPORT_DIR / "package_controller_receipt.final.json"
CONTROLLER_FAILURE = LOOP_REPORT_DIR / "package_controller_failure.final.json"

INPUT_PATH = ARTIFACT_ROOT / "input.f32.bin"
AOTI_PATH = ARTIFACT_ROOT / "tiny_cpu_model.pt2"
TORCHSCRIPT_PATH = ARTIFACT_ROOT / "tiny_cpu_control.pt"
ATEN_BINARY = Path("tools/axon_tiny_pytorch_native/build/bin/Release/axon_tiny_aten_probe.exe")
AOTI_BINARY = Path("tools/axon_tiny_pytorch_native/build/bin/Release/axon_tiny_aoti_probe.exe")
VCVARS64 = Path(
    "C:/Program Files/Microsoft Visual Studio/18/Insiders/VC/Auxiliary/Build/vcvars64.bat"
)

EXPECTED_IMPLEMENTATION_SHA256 = "45d38299623ab32e701d6bf1408a509caa19513287c239b3866532cd405bd3f4"
EXPECTED_BUILD_RECEIPT_SHA256 = "2ef9fc018cf72c0552902d4e9cca2190b4c5e983bdd15d4917dd5d926043dc77"
EXPECTED_ATEN_SHA256 = "595705bdc0716b7323a0da71b424470c0d474a6556ac7fa3c6507e5d4edfd524"
EXPECTED_AOTI_SHA256 = "097a41e61bfceeeea4592ff2f1dd079e3b447275b677980b7eceb3f02a5dcca0"
EXPECTED_INPUT_SHA256 = "caa371218bdbb95cb73bfe7ab65ec2f8f69222a747fca8f889b2bdc3e693d28b"
EXPECTED_TORCH_VERSION = "2.12.0+cu132"
MAX_RETAINED_OUTPUT_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 4096
MAX_COMPRESSION_RATIO = 1000.0


class PackageControllerError(RuntimeError):
    """Raised when the reissued package contract cannot be proven."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise PackageControllerError(f"Duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageControllerError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise PackageControllerError(f"JSON artifact must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_within(project_root: Path, relative: Path, *, must_exist: bool = True) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise PackageControllerError(f"Path must remain project-relative: {relative}")
    root = project_root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=must_exist)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PackageControllerError(f"Path escapes project root: {relative}") from exc
    if must_exist and not candidate.is_file():
        raise PackageControllerError(f"Required artifact is not a file: {relative}")
    return candidate


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PackageControllerError(f"Output already exists: {path}") from exc


def _load_module(relative: Path, name: str):
    path = PROJECT_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PackageControllerError(f"Unable to import governed module: {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_record(project_root: Path, record: Mapping[str, Any], purpose: str) -> Path:
    path = _resolve_within(project_root, Path(str(record.get("path", ""))))
    if sha256_file(path) != record.get("sha256"):
        raise PackageControllerError(f"Artifact hash drifted: {purpose}")
    if "size_bytes" in record and path.stat().st_size != record.get("size_bytes"):
        raise PackageControllerError(f"Artifact size drifted: {purpose}")
    return path


def verify_controller_manifest(project_root: Path) -> dict[str, Any]:
    manifest_path = _resolve_within(project_root, CONTROLLER_MANIFEST)
    manifest = load_json_strict(manifest_path)
    if manifest.get("schema") != "axon_loop28_pytorch_native_package_controller_manifest_v1":
        raise PackageControllerError("Package controller manifest schema mismatch")
    if manifest.get("loop_id") != LOOP_ID:
        raise PackageControllerError("Package controller manifest loop mismatch")
    if manifest.get("decision") != "package_controller_frozen_reissue_authorization_required":
        raise PackageControllerError("Package controller manifest decision mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 4:
        raise PackageControllerError("Package controller manifest artifact inventory mismatch")
    expected_names = {
        "reissue_authorization",
        "controller",
        "controller_test",
        "controller_validation",
    }
    actual_names = {str(record.get("name")) for record in artifacts if isinstance(record, dict)}
    if actual_names != expected_names:
        raise PackageControllerError("Package controller manifest names mismatch")
    for record in artifacts:
        if not isinstance(record, dict):
            raise PackageControllerError("Package controller artifact record is invalid")
        _verify_record(project_root, record, str(record.get("name")))
    return manifest


def _verify_build_closure(project_root: Path) -> dict[str, Any]:
    builder = _load_module(BASE_BUILDER, "loop28_native_manifest_builder_for_package_controller")
    builder.verify_implementation_manifest(project_root, IMPLEMENTATION_MANIFEST)
    implementation_path = _resolve_within(project_root, IMPLEMENTATION_MANIFEST)
    if sha256_file(implementation_path) != EXPECTED_IMPLEMENTATION_SHA256:
        raise PackageControllerError("Implementation manifest hash drifted")

    receipt_path = _resolve_within(project_root, BUILD_RECEIPT)
    if sha256_file(receipt_path) != EXPECTED_BUILD_RECEIPT_SHA256:
        raise PackageControllerError("C++ build receipt hash drifted")
    receipt = load_json_strict(receipt_path)
    if receipt.get("schema") != "axon_loop28_pytorch_native_cpp_build_receipt_v1":
        raise PackageControllerError("C++ build receipt schema mismatch")
    if receipt.get("decision") != "lean_aten_and_aoti_hosts_built_package_requires_authorization":
        raise PackageControllerError("C++ build receipt decision mismatch")
    if receipt.get("implementation_manifest_sha256") != EXPECTED_IMPLEMENTATION_SHA256:
        raise PackageControllerError("C++ build receipt implementation binding mismatch")
    if receipt.get("forbidden_aoti_dependencies") != []:
        raise PackageControllerError("C++ build receipt contains forbidden dependencies")
    for counter in ("execution_count", "checkpoint_or_onnx_load_count", "quality_metric_count"):
        if receipt.get(counter) != 0:
            raise PackageControllerError(f"C++ build receipt counter drifted: {counter}")
    expected_binaries = {
        "aten": (ATEN_BINARY, EXPECTED_ATEN_SHA256, 159232),
        "aoti": (AOTI_BINARY, EXPECTED_AOTI_SHA256, 160768),
    }
    for name, (relative, expected_sha, expected_size) in expected_binaries.items():
        path = _resolve_within(project_root, relative)
        record = receipt.get("binaries", {}).get(name, {})
        if (
            sha256_file(path) != expected_sha
            or path.stat().st_size != expected_size
            or record.get("sha256") != expected_sha
            or record.get("size_bytes") != expected_size
        ):
            raise PackageControllerError(f"Frozen native binary drifted: {name}")
    return receipt


def verify_pre_package_gate(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    controller_manifest = verify_controller_manifest(root)
    build_receipt = _verify_build_closure(root)
    reissue_path = _resolve_within(root, REISSUE_AUTHORIZATION)
    reissue = load_json_strict(reissue_path)
    if reissue.get("decision") != (
        "authorize_versioned_package_controller_after_pre_model_administrative_failure"
    ):
        raise PackageControllerError("Package reissue authorization decision mismatch")

    authorization_path = _resolve_within(root, PACKAGE_AUTHORIZATION)
    authorization = load_json_strict(authorization_path)
    if authorization.get("schema") != "axon_loop28_pytorch_native_package_authorization_v1":
        raise PackageControllerError("Package authorization schema mismatch")
    if authorization.get("attempt_id") != (
        "p0_loop28_pytorch_native_feasibility_001_package_attempt_002"
    ):
        raise PackageControllerError("Package authorization attempt mismatch")
    if authorization.get("decision") != (
        "authorize_single_governed_tiny_aoti_package_generation_reissue_001"
    ):
        raise PackageControllerError("Package authorization decision mismatch")
    expected_bindings = {
        "implementation_manifest_sha256": EXPECTED_IMPLEMENTATION_SHA256,
        "cpp_build_receipt_sha256": EXPECTED_BUILD_RECEIPT_SHA256,
        "package_reissue_authorization_sha256": sha256_file(reissue_path),
        "package_controller_manifest_sha256": sha256_file(
            _resolve_within(root, CONTROLLER_MANIFEST)
        ),
    }
    for key, expected in expected_bindings.items():
        if authorization.get(key) != expected:
            raise PackageControllerError(f"Package authorization binding mismatch: {key}")
    if authorization.get("max_retained_output_bytes") != MAX_RETAINED_OUTPUT_BYTES:
        raise PackageControllerError("Package retained-output cap mismatch")
    boundary = authorization.get("claim_boundary", {})
    if boundary.get("package_load_allowed") is not False:
        raise PackageControllerError("Package authorization accidentally permits a load")
    if boundary.get("runtime_lane_execution_allowed") is not False:
        raise PackageControllerError("Package authorization accidentally permits execution")

    ready_path = _resolve_within(root, PACKAGE_READY_LEASE)
    ready = load_json_strict(ready_path)
    if ready.get("schema") != "axon_loop28_pytorch_native_package_lease_v1":
        raise PackageControllerError("Package ready lease schema mismatch")
    if ready.get("status") != "ready" or ready.get("single_use") is not True:
        raise PackageControllerError("Package ready lease is not single-use ready")
    if ready.get("authorization_sha256") != sha256_file(authorization_path):
        raise PackageControllerError("Package ready lease authorization binding mismatch")
    for key, expected in expected_bindings.items():
        if key in ready and ready.get(key) != expected:
            raise PackageControllerError(f"Package ready lease binding mismatch: {key}")

    # 保护动作前必须没有任何会导致“租约已消费后才失败”的既存输出。
    absence_paths = (
        PACKAGE_FINAL_LEASE,
        PACKAGE_RECEIPT,
        PACKAGE_FAILURE,
        CONTROLLER_RECEIPT,
        CONTROLLER_FAILURE,
        ARTIFACT_ROOT,
    )
    for relative in absence_paths:
        if (root / relative).exists():
            raise PackageControllerError(f"Protected package output already exists: {relative}")
    return {
        "controller_manifest": controller_manifest,
        "controller_manifest_sha256": sha256_file(_resolve_within(root, CONTROLLER_MANIFEST)),
        "authorization": authorization,
        "authorization_sha256": sha256_file(authorization_path),
        "ready_lease": ready,
        "ready_lease_sha256": sha256_file(ready_path),
        "build_receipt": build_receipt,
    }


def _activation_script_bytes(vcvars64: Path = VCVARS64) -> bytes:
    return (
        "@echo off\r\n"
        f'call "{vcvars64}" >nul\r\n'
        "if errorlevel 1 exit /b %errorlevel%\r\n"
        "chcp 65001 >nul\r\n"
        "set\r\n"
    ).encode("ascii")


def activate_vcvars_environment() -> dict[str, str]:
    # 避免 cmd.exe /s /c 的嵌套引号重写：由无空格文件名的临时批处理完成激活。
    work_root = (PROJECT_ROOT / WORK_ROOT).resolve(strict=False)
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vcvars_controller_", dir=work_root) as temporary:
        temporary_root = Path(temporary)
        script_path = temporary_root / "activate.cmd"
        with script_path.open("xb") as handle:
            handle.write(_activation_script_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "activate.cmd"],
            cwd=temporary_root,
            capture_output=True,
            text=False,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            stderr_sha = hashlib.sha256(completed.stderr or b"").hexdigest()
            raise PackageControllerError(
                f"Corrected vcvars64 activation failed: rc={completed.returncode}, "
                f"stderr_sha256={stderr_sha}"
            )
        try:
            output = (completed.stdout or b"").decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise PackageControllerError("Corrected vcvars64 output is not UTF-8") from exc

    activated = dict(os.environ)
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            activated[key] = value
    for required in ("PATH", "INCLUDE", "LIB"):
        if not activated.get(required):
            raise PackageControllerError(f"Corrected vcvars64 omitted: {required}")
    return activated


def audit_aoti_archive(path: Path, *, max_bytes: int = MAX_RETAINED_OUTPUT_BYTES) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise PackageControllerError("AOTI package is missing or exceeds the retained-output cap")
    members: list[dict[str, Any]] = []
    canonical_names: set[str] = set()
    exact_names: set[str] = set()
    total_uncompressed = 0
    pyd_count = 0
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_ARCHIVE_MEMBERS:
            raise PackageControllerError("AOTI archive member count is outside the contract")
        for info in infos:
            name = getattr(info, "orig_filename", info.filename)
            if name in exact_names:
                raise PackageControllerError("AOTI archive contains duplicate member names")
            exact_names.add(name)
            if not name or "\x00" in name or "\\" in name or name.startswith(("/", "//")):
                raise PackageControllerError("AOTI archive member has an unsafe path")
            pure = PurePosixPath(name)
            if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
                raise PackageControllerError("AOTI archive member escapes its archive root")
            if pure.parts and ":" in pure.parts[0]:
                raise PackageControllerError("AOTI archive member contains a Windows drive path")
            canonical = unicodedata.normalize("NFKC", name).casefold()
            if canonical in canonical_names:
                raise PackageControllerError("AOTI archive has a Windows/Unicode name collision")
            canonical_names.add(canonical)
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if unix_mode and stat.S_ISLNK(unix_mode):
                raise PackageControllerError("AOTI archive contains a symbolic-link member")
            if info.flag_bits & 0x1:
                raise PackageControllerError("AOTI archive contains an encrypted member")
            if info.file_size > max_bytes:
                raise PackageControllerError("AOTI archive member exceeds the size cap")
            total_uncompressed += info.file_size
            if total_uncompressed > max_bytes:
                raise PackageControllerError("AOTI archive uncompressed total exceeds the size cap")
            if info.file_size > 1024 * 1024:
                ratio = info.file_size / max(info.compress_size, 1)
                if ratio > MAX_COMPRESSION_RATIO:
                    raise PackageControllerError("AOTI archive compression ratio is unsafe")
            digest = hashlib.sha256()
            decoded_size = 0
            with archive.open(info, "r") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    decoded_size += len(block)
                    digest.update(block)
            if decoded_size != info.file_size:
                raise PackageControllerError("AOTI archive member size changed while reading")
            if name.casefold().endswith(".pyd"):
                pyd_count += 1
            members.append(
                {
                    "name": name,
                    "size_bytes": info.file_size,
                    "compressed_size_bytes": info.compress_size,
                    "sha256": digest.hexdigest(),
                }
            )
    if pyd_count != 1:
        raise PackageControllerError("AOTI archive must contain exactly one precompiled .pyd")
    return {
        "member_count": len(members),
        "members": members,
        "precompiled_pyd_count": pyd_count,
        "total_uncompressed_bytes": total_uncompressed,
        "casefold_or_unicode_collisions": 0,
        "symlink_members": 0,
        "unsafe_paths": 0,
    }


def verify_package_receipt(project_root: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    receipt_path = _resolve_within(root, PACKAGE_RECEIPT)
    receipt = load_json_strict(receipt_path)
    if receipt.get("schema") != "axon_loop28_pytorch_native_package_receipt_v1":
        raise PackageControllerError("Package receipt schema mismatch")
    if (
        receipt.get("decision")
        != "tiny_aoti_package_generated_execution_requires_new_authorization"
    ):
        raise PackageControllerError("Package receipt decision mismatch")
    if receipt.get("implementation_manifest_sha256") != EXPECTED_IMPLEMENTATION_SHA256:
        raise PackageControllerError("Package receipt implementation binding mismatch")
    lease = receipt.get("lease", {})
    if lease.get("authorization") != context["authorization"]:
        raise PackageControllerError("Package receipt authorization snapshot mismatch")
    final_lease_path = _resolve_within(root, PACKAGE_FINAL_LEASE)
    final_lease = load_json_strict(final_lease_path)
    if final_lease.get("status") != "consumed_before_execution":
        raise PackageControllerError("Package final lease status mismatch")
    if final_lease.get("original_lease_sha256") != context["ready_lease_sha256"]:
        raise PackageControllerError("Package final lease ready binding mismatch")
    if lease.get("sha256") != sha256_file(final_lease_path):
        raise PackageControllerError("Package receipt final lease hash mismatch")

    torch_record = receipt.get("torch", {})
    if torch_record.get("version") != EXPECTED_TORCH_VERSION:
        raise PackageControllerError("Package Torch version drifted")
    if torch_record.get("cuda_initialized_before") is not False:
        raise PackageControllerError("CUDA was initialized before package generation")
    if torch_record.get("cuda_initialized_after") is not False:
        raise PackageControllerError("CUDA was initialized during package generation")
    if torch_record.get("cpu_threads") != 1:
        raise PackageControllerError("Package generation CPU thread count drifted")
    budget = receipt.get("budget", {})
    expected_budget = {
        "torch_export_calls": 1,
        "aoti_compile_and_package_calls": 1,
        "torchscript_export_calls": 1,
    }
    for key, expected in expected_budget.items():
        if budget.get(key) != expected:
            raise PackageControllerError(f"Package call budget drifted: {key}")
    if receipt.get("forbidden_project_artifact_access_count") != 0:
        raise PackageControllerError("Package receipt reports forbidden project access")

    artifacts = receipt.get("artifacts", {})
    input_path = _resolve_within(root, INPUT_PATH)
    input_record = artifacts.get("input", {})
    if (
        input_path.stat().st_size != 64
        or sha256_file(input_path) != EXPECTED_INPUT_SHA256
        or input_record.get("sha256") != EXPECTED_INPUT_SHA256
        or input_record.get("size_bytes") != 64
    ):
        raise PackageControllerError("Fixed tiny input drifted")
    aoti_path = _resolve_within(root, AOTI_PATH)
    aoti_record = artifacts.get("aoti_package", {})
    if (
        aoti_record.get("sha256") != sha256_file(aoti_path)
        or aoti_record.get("size_bytes") != aoti_path.stat().st_size
    ):
        raise PackageControllerError("AOTI package receipt record drifted")
    archive_audit = audit_aoti_archive(aoti_path)

    retained_bytes = input_path.stat().st_size + aoti_path.stat().st_size
    torchscript = artifacts.get("torchscript_control", {})
    if torchscript.get("status") == "generated":
        torchscript_path = _resolve_within(root, TORCHSCRIPT_PATH)
        if (
            torchscript.get("sha256") != sha256_file(torchscript_path)
            or torchscript.get("size_bytes") != torchscript_path.stat().st_size
        ):
            raise PackageControllerError("TorchScript control receipt record drifted")
        retained_bytes += torchscript_path.stat().st_size
    elif torchscript.get("status") != "unsupported_nonblocking":
        raise PackageControllerError("TorchScript control status is invalid")
    if retained_bytes > MAX_RETAINED_OUTPUT_BYTES:
        raise PackageControllerError("Package outputs exceed the retained-output cap")

    work_root = (root / WORK_ROOT).resolve(strict=False)
    leftovers = sorted(path.name for path in work_root.iterdir()) if work_root.exists() else []
    if leftovers:
        raise PackageControllerError("Package controller left temporary work artifacts")
    return {
        "receipt": receipt,
        "receipt_sha256": sha256_file(receipt_path),
        "final_lease_sha256": sha256_file(final_lease_path),
        "archive_audit": archive_audit,
        "retained_output_bytes": retained_bytes,
        "work_root_leftovers": leftovers,
    }


def _failure_payload(error: Exception, *, protected_stage_started: bool) -> dict[str, Any]:
    return {
        "schema": "axon_loop28_pytorch_native_package_controller_failure_v1",
        "loop_id": LOOP_ID,
        "error_type": type(error).__name__,
        "error": str(error)[:4000],
        "protected_stage_started": protected_stage_started,
        "package_load_count": 0,
        "native_probe_execution_count": 0,
        "checkpoint_or_onnx_load_count": 0,
        "quality_metric_count": 0,
        "decision": "package_controller_failure_no_runtime_claim",
    }


def run_package(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    protected_stage_started = False
    try:
        context = verify_pre_package_gate(root)
        runner = _load_module(BASE_RUNNER, "loop28_native_runner_for_package_controller")
        runner._activate_vcvars_environment = activate_vcvars_environment
        protected_stage_started = True
        runner.run_package(root)
        verified = verify_package_receipt(root, context)
        payload = {
            "schema": "axon_loop28_pytorch_native_package_controller_receipt_v1",
            "loop_id": LOOP_ID,
            "controller_manifest_sha256": context["controller_manifest_sha256"],
            "authorization_sha256": context["authorization_sha256"],
            "ready_lease_sha256": context["ready_lease_sha256"],
            "consumed_lease_sha256": verified["final_lease_sha256"],
            "base_package_receipt_sha256": verified["receipt_sha256"],
            "archive_audit": verified["archive_audit"],
            "retained_output_bytes": verified["retained_output_bytes"],
            "safety": {
                "corrected_vcvars_activation_used": True,
                "work_root_leftovers": verified["work_root_leftovers"],
                "cuda_initialized_before": False,
                "cuda_initialized_after": False,
                "package_load_count": 0,
                "native_probe_execution_count": 0,
                "checkpoint_or_onnx_load_count": 0,
                "raw_split_cache_heldout_access_count": 0,
                "gpu_execution_count": 0,
                "network_request_count": 0,
                "quality_metric_count": 0,
            },
            "claim_boundary": context["authorization"]["claim_boundary"],
            "decision": "tiny_aoti_package_generated_controller_contract_passed_manifest_required",
        }
        _write_json_exclusive(_resolve_within(root, CONTROLLER_RECEIPT, must_exist=False), payload)
        return payload
    except Exception as exc:
        controller_failure = (root / CONTROLLER_FAILURE).resolve(strict=False)
        if not controller_failure.exists():
            _write_json_exclusive(
                controller_failure,
                _failure_payload(exc, protected_stage_started=protected_stage_started),
            )
        if protected_stage_started and not (root / PACKAGE_FAILURE).exists():
            runner = _load_module(BASE_RUNNER, "loop28_native_runner_failure_writer")
            runner._write_json_exclusive(
                (root / PACKAGE_FAILURE).resolve(strict=False),
                runner._failure_payload("package", exc),
            )
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("package", "verify-controller"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve(strict=True)
    if args.mode == "package":
        payload = run_package(root)
    elif args.mode == "verify-controller":
        payload = verify_pre_package_gate(root)
        payload = {
            "decision": "package_controller_pre_package_gate_verified",
            "controller_manifest_sha256": payload["controller_manifest_sha256"],
        }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
