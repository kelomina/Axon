#!/usr/bin/env python3
"""Build and verify the governed tiny PyTorch-native CPU feasibility artifacts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "p0_loop28_pytorch_native_feasibility_001"
REPEATS = 3
TOLERANCE = 1.0e-6

LOOP_MANIFEST_DIR = Path("manifests/roadmap_9997/p0_loop28_pytorch_native_feasibility")
LOOP_REPORT_DIR = Path("reports/roadmap_9997/p0_loop28_pytorch_native_feasibility")
ARTIFACT_ROOT = Path("artifacts/roadmap_9997/p0_loop28_pytorch_native_feasibility/tiny_v1")

DEFAULT_IMPLEMENTATION = LOOP_MANIFEST_DIR / "implementation_manifest.json"
DEFAULT_BUILD_AUTHORIZATION = LOOP_MANIFEST_DIR / "build_authorization.json"
DEFAULT_BUILD_LEASE = LOOP_MANIFEST_DIR / "build_lease.json"
DEFAULT_BUILD_FINAL_LEASE = LOOP_MANIFEST_DIR / "build_lease.final.json"
DEFAULT_PACKAGE_AUTHORIZATION = LOOP_MANIFEST_DIR / "package_authorization.json"
DEFAULT_PACKAGE_LEASE = LOOP_MANIFEST_DIR / "package_lease.json"
DEFAULT_PACKAGE_FINAL_LEASE = LOOP_MANIFEST_DIR / "package_lease.final.json"
DEFAULT_PACKAGE_MANIFEST = LOOP_MANIFEST_DIR / "package_manifest.json"
DEFAULT_EXECUTION_AUTHORIZATION = LOOP_MANIFEST_DIR / "execution_authorization.json"
DEFAULT_EXECUTION_LEASE = LOOP_MANIFEST_DIR / "execution_lease.json"
DEFAULT_EXECUTION_FINAL_LEASE = LOOP_MANIFEST_DIR / "execution_lease.final.json"

DEFAULT_PACKAGE_RECEIPT = LOOP_REPORT_DIR / "package_receipt.final.json"
DEFAULT_BUILD_RECEIPT = LOOP_REPORT_DIR / "cpp_build_receipt.final.json"
DEFAULT_BUILD_FAILURE = LOOP_REPORT_DIR / "cpp_build_failure.final.json"
DEFAULT_PACKAGE_FAILURE = LOOP_REPORT_DIR / "package_failure.final.json"
DEFAULT_EXECUTION_EVIDENCE = LOOP_REPORT_DIR / "execution_evidence.final.json"
DEFAULT_EXECUTION_FAILURE = LOOP_REPORT_DIR / "execution_failure.final.json"
DEFAULT_WORK_ROOT = LOOP_REPORT_DIR / "work"

CPP_SOURCE_ROOT = Path("tools/axon_tiny_pytorch_native")
CPP_BUILD_ROOT = CPP_SOURCE_ROOT / "build"

DEFAULT_INPUT = ARTIFACT_ROOT / "input.f32.bin"
DEFAULT_AOTI_PACKAGE = ARTIFACT_ROOT / "tiny_cpu_model.pt2"
DEFAULT_TORCHSCRIPT = ARTIFACT_ROOT / "tiny_cpu_control.pt"
DEFAULT_CPP_AOTI = Path("tools/axon_tiny_pytorch_native/build/bin/Release/axon_tiny_aoti_probe.exe")
DEFAULT_CPP_ATEN = Path("tools/axon_tiny_pytorch_native/build/bin/Release/axon_tiny_aten_probe.exe")
DEFAULT_CPP_LIBTORCH = Path(
    "tools/axon_tiny_pytorch_native/build/bin/Release/axon_tiny_libtorch_probe.exe"
)

VCVARS64 = Path(
    "C:/Program Files/Microsoft Visual Studio/18/Insiders/VC/Auxiliary/Build/vcvars64.bat"
)
CMAKE_EXE = Path(
    "C:/Program Files/Microsoft Visual Studio/18/Insiders/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
)
DUMPBIN_EXE = Path(
    "C:/Program Files/Microsoft Visual Studio/18/Insiders/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/dumpbin.exe"
)
TORCH_ROOT = PROJECT_ROOT / "vnev/Lib/site-packages/torch"
PYTHON_ROOT = Path("C:/Users/Saika/AppData/Local/Python/pythoncore-3.14-64")

OUTPUT_CONTRACT = (
    ("linear", np.dtype("<f4"), (2, 8)),
    ("gelu", np.dtype("<f4"), (2, 8)),
    ("topk_values", np.dtype("<f4"), (2, 2)),
    ("topk_indices", np.dtype("<i8"), (2, 2)),
)


class NativeFeasibilityError(RuntimeError):
    """Raised when the tiny runtime contract fails closed."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise NativeFeasibilityError(f"Duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeFeasibilityError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise NativeFeasibilityError(f"JSON artifact must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve_within(project_root: Path, relative: Path, *, must_exist: bool = True) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise NativeFeasibilityError(f"Path must remain project-relative: {relative}")
    root = project_root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=must_exist)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise NativeFeasibilityError(f"Path escapes project root: {relative}") from exc
    if must_exist and not candidate.is_file():
        raise NativeFeasibilityError(f"Required artifact is not a file: {relative}")
    return candidate


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise NativeFeasibilityError(f"Output already exists: {path}") from exc


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _write_exclusive(path, encoded)


def tiny_input() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 2.0],
            [2.0, -1.5, 1.25, -1.0, 0.75, -0.5, 0.25, 0.0],
        ],
        dtype=np.float32,
    )


def build_tiny_model():
    import torch  # noqa: PLC0415
    import torch.nn.functional as functional  # noqa: PLC0415

    class TinyLinearGeluTopKCPU(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base_rows = torch.tensor(
                [
                    [0.5, -0.25, 0.125, 0.0, 0.75, -0.5, 0.25, -0.125],
                    [-0.25, 0.5, 0.0, 0.125, -0.5, 0.75, -0.125, 0.25],
                    [0.125, 0.0, 0.5, -0.25, 0.25, -0.125, 0.75, -0.5],
                    [0.0, 0.125, -0.25, 0.5, -0.125, 0.25, -0.5, 0.75],
                ],
                dtype=torch.float32,
            )
            base_bias = torch.tensor([0.125, -0.25, 0.375, -0.5], dtype=torch.float32)
            self.register_buffer("weight", base_rows.repeat_interleave(2, dim=0))
            self.register_buffer("bias", base_bias.repeat_interleave(2))

        def forward(self, input_tensor):
            linear = functional.linear(input_tensor, self.weight, self.bias)
            gelu = functional.gelu(linear, approximate="none")
            topk_values, topk_indices = torch.topk(gelu, 2, dim=-1)
            return linear, gelu, topk_values, topk_indices

    return TinyLinearGeluTopKCPU().eval()


def _array_record(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "nbytes": int(contiguous.nbytes),
        "sha256": _sha256_bytes(contiguous.tobytes(order="C")),
    }


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.ascontiguousarray(reference)
    candidate = np.ascontiguousarray(candidate)
    result: dict[str, Any] = {
        "reference": _array_record(reference),
        "candidate": _array_record(candidate),
        "shape_match": reference.shape == candidate.shape,
        "dtype_match": reference.dtype == candidate.dtype,
    }
    if not result["shape_match"] or not result["dtype_match"]:
        result.update({"passed": False, "reason": "shape_or_dtype_mismatch"})
        return result
    if np.issubdtype(reference.dtype, np.floating):
        delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
        max_absolute_delta = float(delta.max(initial=0.0))
        result.update(
            {
                "max_absolute_delta": max_absolute_delta,
                "above_tolerance_count": int(np.count_nonzero(delta > TOLERANCE)),
                "exact_mismatch_count": int(np.count_nonzero(reference != candidate)),
                "passed": bool(np.all(np.isfinite(candidate))) and max_absolute_delta <= TOLERANCE,
            }
        )
    else:
        mismatch = reference != candidate
        result.update(
            {
                "mismatch_count": int(np.count_nonzero(mismatch)),
                "passed": not bool(np.any(mismatch)),
            }
        )
    return result


def _normalize_outputs(value: object) -> dict[str, np.ndarray]:
    import torch  # noqa: PLC0415

    if not isinstance(value, (tuple, list)) or len(value) != len(OUTPUT_CONTRACT):
        raise NativeFeasibilityError("Tiny model must return four tensors")
    outputs: dict[str, np.ndarray] = {}
    for tensor, (name, dtype, shape) in zip(value, OUTPUT_CONTRACT, strict=True):
        if not isinstance(tensor, torch.Tensor):
            raise NativeFeasibilityError(f"Tiny output is not a tensor: {name}")
        array = tensor.detach().cpu().contiguous().numpy()
        if array.shape != shape or array.dtype != dtype:
            raise NativeFeasibilityError(f"Tiny output contract drifted: {name}")
        outputs[name] = array.copy()
    return outputs


def _determinism(records: Sequence[Mapping[str, np.ndarray]]) -> dict[str, Any]:
    if len(records) != REPEATS:
        raise NativeFeasibilityError("Each lane must contain exactly three repeats")
    baseline = records[0]
    keys_match = all(tuple(record) == tuple(baseline) for record in records[1:])
    mismatched = []
    if keys_match:
        for key, reference in baseline.items():
            if any(not np.array_equal(reference, record[key]) for record in records[1:]):
                mismatched.append(key)
    return {
        "repeat_count": len(records),
        "key_sets_match": keys_match,
        "bit_exact": keys_match and not mismatched,
        "mismatched_outputs": mismatched,
    }


def _archive_inventory(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise NativeFeasibilityError("AOTI package contains duplicate archive members")
        inventory: list[dict[str, Any]] = []
        for info in archive.infolist():
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts or "\\" in info.filename:
                raise NativeFeasibilityError("AOTI package member path is unsafe")
            payload = archive.read(info)
            inventory.append(
                {
                    "name": info.filename,
                    "size_bytes": len(payload),
                    "sha256": _sha256_bytes(payload),
                }
            )
    if not inventory:
        raise NativeFeasibilityError("AOTI package archive is empty")
    return inventory


def _load_builder():
    path = PROJECT_ROOT / "scripts/build_loop28_pytorch_native_feasibility_manifest.py"
    spec = importlib.util.spec_from_file_location("native_feasibility_manifest_builder", path)
    if spec is None or spec.loader is None:
        raise NativeFeasibilityError("Unable to import native feasibility manifest builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _consume_lease(
    project_root: Path,
    *,
    authorization_path: Path,
    ready_path: Path,
    final_path: Path,
    expected_schema: str,
) -> dict[str, Any]:
    authorization = load_json_strict(_resolve_within(project_root, authorization_path))
    ready = _resolve_within(project_root, ready_path)
    final = _resolve_within(project_root, final_path, must_exist=False)
    lease = load_json_strict(ready)
    if lease.get("schema") != expected_schema or lease.get("status") != "ready":
        raise NativeFeasibilityError("Ready lease schema or status mismatch")
    authorization_sha256 = sha256_file(_resolve_within(project_root, authorization_path))
    if lease.get("authorization_sha256") != authorization_sha256:
        raise NativeFeasibilityError("Ready lease authorization binding mismatch")
    consumed = dict(lease)
    consumed.update(
        {
            "status": "consumed_before_execution",
            "original_lease_sha256": sha256_file(ready),
        }
    )
    _write_json_exclusive(final, consumed)
    ready.unlink()
    return {
        "authorization": authorization,
        "path": final_path.as_posix(),
        "sha256": sha256_file(final),
        "status": "consumed_before_execution",
    }


def _activate_vcvars_environment() -> dict[str, str]:
    command = ["cmd.exe", "/d", "/s", "/c", f'call "{VCVARS64}" >nul && set']
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if completed.returncode != 0:
        raise NativeFeasibilityError("vcvars64 environment activation failed")
    activated = dict(os.environ)
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key:
                activated[key] = value
    for required in ("PATH", "INCLUDE", "LIB"):
        if not activated.get(required):
            raise NativeFeasibilityError(f"vcvars64 omitted required variable: {required}")
    return activated


def _run_build_command(command: list[str], *, timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=False,
        timeout=timeout,
        check=False,
    )
    stdout = completed.stdout or b""
    stderr = completed.stderr or b""
    return {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "stdout_sha256": _sha256_bytes(stdout),
        "stderr_sha256": _sha256_bytes(stderr),
        "stderr_tail": stderr[-2000:].decode("utf-8", errors="replace"),
    }


def _dumpbin_dependencies(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(DUMPBIN_EXE), "/DEPENDENTS", str(path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise NativeFeasibilityError(f"dumpbin dependency audit failed: {path.name}")
    dependencies = sorted(
        {
            line.strip().casefold()
            for line in completed.stdout.splitlines()
            if line.strip().casefold().endswith(".dll")
        }
    )
    return {
        "dependencies": dependencies,
        "stdout_sha256": _sha256_bytes(completed.stdout.encode()),
        "stderr_sha256": _sha256_bytes(completed.stderr.encode()),
    }


def run_cpp_build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    builder = _load_builder()
    builder.verify_implementation_manifest(root, DEFAULT_IMPLEMENTATION)
    lease = _consume_lease(
        root,
        authorization_path=DEFAULT_BUILD_AUTHORIZATION,
        ready_path=DEFAULT_BUILD_LEASE,
        final_path=DEFAULT_BUILD_FINAL_LEASE,
        expected_schema="axon_loop28_pytorch_native_build_lease_v1",
    )
    if lease["authorization"].get("implementation_manifest_sha256") != sha256_file(
        _resolve_within(root, DEFAULT_IMPLEMENTATION)
    ):
        raise NativeFeasibilityError("Build authorization implementation binding mismatch")

    source_root = (root / CPP_SOURCE_ROOT).resolve(strict=True)
    build_root = (root / CPP_BUILD_ROOT).resolve(strict=False)
    if build_root.exists():
        raise NativeFeasibilityError("C++ feasibility build root already exists")
    configure = _run_build_command(
        [
            str(CMAKE_EXE),
            "-S",
            str(source_root),
            "-B",
            str(build_root),
            "-G",
            "Visual Studio 18 2026",
            "-A",
            "x64",
            f"-DTORCH_ROOT={TORCH_ROOT}",
        ],
        timeout=180,
    )
    commands = {"configure": configure}
    if configure["returncode"] != 0:
        decision = "lean_aten_control_link_failed_toolchain_invalid"
        aten_build = None
        aoti_build = None
    else:
        aten_build = _run_build_command(
            [
                str(CMAKE_EXE),
                "--build",
                str(build_root),
                "--config",
                "Release",
                "--target",
                "axon_tiny_aten_probe",
                "--parallel",
                "1",
            ],
            timeout=1200,
        )
        commands["aten_build"] = aten_build
        if aten_build["returncode"] != 0:
            decision = "lean_aten_control_link_failed_toolchain_invalid"
            aoti_build = None
        else:
            aoti_build = _run_build_command(
                [
                    str(CMAKE_EXE),
                    "--build",
                    str(build_root),
                    "--config",
                    "Release",
                    "--target",
                    "axon_tiny_aoti_probe",
                    "--parallel",
                    "1",
                ],
                timeout=1200,
            )
            commands["aoti_build"] = aoti_build
            decision = (
                "lean_aten_and_aoti_hosts_built_package_requires_authorization"
                if aoti_build["returncode"] == 0
                else "lean_aoti_loader_link_surface_missing_no_package"
            )

    binaries: dict[str, Any] = {}
    for name, relative in (("aten", DEFAULT_CPP_ATEN), ("aoti", DEFAULT_CPP_AOTI)):
        path = (root / relative).resolve(strict=False)
        if path.is_file():
            dependencies = _dumpbin_dependencies(path)
            binaries[name] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                **dependencies,
            }

    forbidden = {"torch_python.dll", "python314.dll", "torch.dll", "torch_cuda.dll"}
    aoti_dependencies = set(binaries.get("aoti", {}).get("dependencies", []))
    forbidden_dependencies = sorted(forbidden.intersection(aoti_dependencies))
    if decision == "lean_aten_and_aoti_hosts_built_package_requires_authorization" and (
        forbidden_dependencies
    ):
        decision = "aoti_dependency_leakage_no_load"

    receipt = {
        "schema": "axon_loop28_pytorch_native_cpp_build_receipt_v1",
        "loop_id": LOOP_ID,
        "lease": lease,
        "implementation_manifest_sha256": sha256_file(
            _resolve_within(root, DEFAULT_IMPLEMENTATION)
        ),
        "commands": commands,
        "binaries": binaries,
        "forbidden_aoti_dependencies": forbidden_dependencies,
        "execution_count": 0,
        "checkpoint_or_onnx_load_count": 0,
        "quality_metric_count": 0,
        "decision": decision,
    }
    _write_json_exclusive(_resolve_within(root, DEFAULT_BUILD_RECEIPT, must_exist=False), receipt)
    return receipt


def run_package(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    builder = _load_builder()
    builder.verify_implementation_manifest(root, DEFAULT_IMPLEMENTATION)
    lease = _consume_lease(
        root,
        authorization_path=DEFAULT_PACKAGE_AUTHORIZATION,
        ready_path=DEFAULT_PACKAGE_LEASE,
        final_path=DEFAULT_PACKAGE_FINAL_LEASE,
        expected_schema="axon_loop28_pytorch_native_package_lease_v1",
    )
    if lease["authorization"].get("implementation_manifest_sha256") != sha256_file(
        _resolve_within(root, DEFAULT_IMPLEMENTATION)
    ):
        raise NativeFeasibilityError("Package authorization implementation binding mismatch")

    artifact_root = (root / ARTIFACT_ROOT).resolve(strict=False)
    if artifact_root.exists():
        raise NativeFeasibilityError("Tiny artifact root already exists")
    artifact_root.mkdir(parents=True)
    work_root = (root / DEFAULT_WORK_ROOT).resolve(strict=False)
    work_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    environment = _activate_vcvars_environment()

    import torch  # noqa: PLC0415
    import torch._inductor as inductor  # noqa: PLC0415

    torch.set_num_threads(1)
    with np.errstate(all="raise"):
        input_array = tiny_input()
    input_path = artifact_root / DEFAULT_INPUT.name
    _write_exclusive(input_path, input_array.tobytes(order="C"))
    input_tensor = torch.from_numpy(input_array.copy())
    model = build_tiny_model()
    if torch.cuda.is_initialized():
        raise NativeFeasibilityError("CUDA was initialized before tiny package generation")

    aoti_path = artifact_root / DEFAULT_AOTI_PACKAGE.name
    torchscript_path = artifact_root / DEFAULT_TORCHSCRIPT.name
    compile_temp = Path(tempfile.mkdtemp(prefix="package_", dir=work_root))
    environment["TEMP"] = str(compile_temp)
    environment["TMP"] = str(compile_temp)
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(compile_temp / "inductor_cache")
    old_environment = dict(os.environ)
    os.environ.clear()
    os.environ.update(environment)
    try:
        exported = torch.export.export(model, (input_tensor,), strict=True)
        inductor.aoti_compile_and_package(exported, package_path=aoti_path)
        torchscript_error = None
        try:
            traced = torch.jit.trace(model, (input_tensor,), check_trace=True, strict=True)
            traced.save(str(torchscript_path))
        except Exception as exc:  # noqa: BLE001 - Python 3.14 TorchScript is optional.
            torchscript_error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    finally:
        os.environ.clear()
        os.environ.update(old_environment)
        shutil.rmtree(compile_temp, ignore_errors=True)

    if not aoti_path.is_file():
        raise NativeFeasibilityError("Tiny package generation omitted the AOTI artifact")
    archive_members = _archive_inventory(aoti_path)
    pyd_members = [row for row in archive_members if row["name"].casefold().endswith(".pyd")]
    if len(pyd_members) != 1:
        raise NativeFeasibilityError(
            "AOTI package must contain exactly one precompiled .pyd before any load"
        )
    elapsed = time.monotonic() - started
    receipt = {
        "schema": "axon_loop28_pytorch_native_package_receipt_v1",
        "loop_id": LOOP_ID,
        "lease": lease,
        "implementation_manifest_sha256": sha256_file(
            _resolve_within(root, DEFAULT_IMPLEMENTATION)
        ),
        "torch": {
            "version": torch.__version__,
            "cuda_initialized_before": False,
            "cuda_initialized_after": torch.cuda.is_initialized(),
            "cpu_threads": torch.get_num_threads(),
        },
        "artifacts": {
            "input": {
                "path": DEFAULT_INPUT.as_posix(),
                "sha256": sha256_file(input_path),
                "size_bytes": input_path.stat().st_size,
            },
            "aoti_package": {
                "path": DEFAULT_AOTI_PACKAGE.as_posix(),
                "sha256": sha256_file(aoti_path),
                "size_bytes": aoti_path.stat().st_size,
                "archive_members": archive_members,
            },
            "torchscript_control": (
                {
                    "status": "generated",
                    "path": DEFAULT_TORCHSCRIPT.as_posix(),
                    "sha256": sha256_file(torchscript_path),
                    "size_bytes": torchscript_path.stat().st_size,
                }
                if torchscript_path.is_file()
                else {"status": "unsupported_nonblocking", "error": torchscript_error}
            ),
        },
        "budget": {
            "torch_export_calls": 1,
            "aoti_compile_and_package_calls": 1,
            "torchscript_export_calls": 1,
            "elapsed_seconds": elapsed,
        },
        "forbidden_project_artifact_access_count": 0,
        "decision": "tiny_aoti_package_generated_execution_requires_new_authorization",
    }
    _write_json_exclusive(_resolve_within(root, DEFAULT_PACKAGE_RECEIPT, must_exist=False), receipt)
    return receipt


def _write_lane_outputs(
    output_root: Path,
    *,
    backend: str,
    records: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    if output_root.exists():
        raise NativeFeasibilityError("Lane output root already exists")
    output_root.mkdir(parents=True)
    runs = []
    for run_index, record in enumerate(records):
        rows = []
        for name, dtype, shape in OUTPUT_CONTRACT:
            array = np.ascontiguousarray(record[name])
            if array.dtype != dtype or array.shape != shape:
                raise NativeFeasibilityError(f"Lane output contract drifted: {name}")
            filename = f"{backend}.run{run_index}.{name}.bin"
            path = output_root / filename
            _write_exclusive(path, array.tobytes(order="C"))
            rows.append(
                {
                    "name": name,
                    "dtype": "float32" if dtype == np.dtype("<f4") else "int64",
                    "shape": list(shape),
                    "file": filename,
                    "nbytes": array.nbytes,
                }
            )
        runs.append({"index": run_index, "outputs": rows})
    manifest = {
        "schema": "axon_tiny_pytorch_native_probe_v1",
        "backend": backend,
        "repeat_count": len(records),
        "runs": runs,
    }
    _write_json_exclusive(output_root / "probe_manifest.json", manifest)
    return manifest


def run_python_lane(backend: str, model_path: Path, input_path: Path, output_root: Path) -> int:
    import torch  # noqa: PLC0415
    import torch._inductor as inductor  # noqa: PLC0415

    torch.set_num_threads(1)
    input_array = np.fromfile(input_path, dtype="<f4")
    if input_array.size != 16:
        raise NativeFeasibilityError("Tiny lane input size mismatch")
    input_tensor = torch.from_numpy(input_array.reshape(2, 8).copy())
    if torch.cuda.is_initialized():
        raise NativeFeasibilityError("CUDA initialized before Python lane")

    if backend == "eager":
        runner = build_tiny_model()
    elif backend == "aoti_python":
        declared = os.environ.get("AXON_AOTI_DISPOSABLE_TEMP")
        if not declared:
            raise NativeFeasibilityError("AOTI lane lacks disposable TEMP declaration")
        disposable = Path(declared).resolve(strict=True)
        if Path(os.environ.get("TEMP", "")).resolve(strict=True) != disposable:
            raise NativeFeasibilityError("AOTI TEMP is not the declared disposable directory")
        if Path(os.environ.get("TMP", "")).resolve(strict=True) != disposable:
            raise NativeFeasibilityError("AOTI TMP is not the declared disposable directory")
        if any(disposable.iterdir()):
            raise NativeFeasibilityError("AOTI disposable TEMP must be empty before load")
        if output_root.resolve(strict=False).is_relative_to(disposable):
            raise NativeFeasibilityError("AOTI outputs must remain outside disposable TEMP")
        runner = inductor.aoti_load_package(model_path, run_single_threaded=True)
    elif backend == "torchscript_python":
        runner = torch.jit.load(str(model_path), map_location="cpu").eval()
    else:
        raise NativeFeasibilityError(f"Unsupported Python lane: {backend}")

    records = []
    with torch.inference_mode():
        for _ in range(REPEATS):
            records.append(_normalize_outputs(runner(input_tensor)))
    del runner
    gc.collect()
    if torch.cuda.is_initialized():
        raise NativeFeasibilityError("CUDA initialized during Python lane")
    _write_lane_outputs(output_root, backend=backend, records=records)
    return 0


def _read_probe_outputs(output_root: Path) -> list[dict[str, np.ndarray]]:
    manifest = load_json_strict(output_root / "probe_manifest.json")
    if manifest.get("schema") != "axon_tiny_pytorch_native_probe_v1":
        raise NativeFeasibilityError("Probe manifest schema mismatch")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or len(runs) != REPEATS:
        raise NativeFeasibilityError("Probe repeat count mismatch")
    decoded = []
    for run_index, run in enumerate(runs):
        if not isinstance(run, dict) or run.get("index") != run_index:
            raise NativeFeasibilityError("Probe run ordering mismatch")
        rows = run.get("outputs")
        if not isinstance(rows, list) or len(rows) != len(OUTPUT_CONTRACT):
            raise NativeFeasibilityError("Probe output inventory mismatch")
        record: dict[str, np.ndarray] = {}
        for row, (name, dtype, shape) in zip(rows, OUTPUT_CONTRACT, strict=True):
            if not isinstance(row, dict) or row.get("name") != name:
                raise NativeFeasibilityError("Probe output name mismatch")
            relative = Path(str(row.get("file", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise NativeFeasibilityError("Probe output path is unsafe")
            path = (output_root / relative).resolve(strict=True)
            if not path.is_relative_to(output_root.resolve(strict=True)):
                raise NativeFeasibilityError("Probe output escapes lane root")
            array = np.fromfile(path, dtype=dtype)
            if array.size != int(np.prod(shape)) or path.stat().st_size != row.get("nbytes"):
                raise NativeFeasibilityError("Probe output size mismatch")
            record[name] = array.reshape(shape)
        decoded.append(record)
    return decoded


def _run_lane_subprocess(
    command: list[str],
    *,
    output_root: Path,
    environment: Optional[dict[str, str]] = None,
    disposable_temp: Optional[Path] = None,
    sentinel: Optional[tuple[Path, str]] = None,
    expected_stdout: Optional[str] = None,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise NativeFeasibilityError(
            f"Lane subprocess failed ({completed.returncode}): {completed.stderr[-2000:]}"
        )
    if expected_stdout is not None and expected_stdout not in completed.stdout:
        raise NativeFeasibilityError("Lane subprocess omitted its required runtime metadata")
    if disposable_temp is not None and disposable_temp.exists():
        raise NativeFeasibilityError("AOTI loader did not remove only its disposable TEMP root")
    if sentinel is not None:
        sentinel_path, sentinel_sha256 = sentinel
        if not sentinel_path.is_file() or sha256_file(sentinel_path) != sentinel_sha256:
            raise NativeFeasibilityError("AOTI loader changed the external safety sentinel")
    records = _read_probe_outputs(output_root)
    execution = {
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout_sha256": _sha256_bytes(completed.stdout.encode()),
        "stderr_sha256": _sha256_bytes(completed.stderr.encode()),
        "determinism": _determinism(records),
    }
    shutil.rmtree(output_root)
    return records, execution


def _run_three_fresh_processes(
    *,
    command_factory,
    output_parent: Path,
    environment_factory,
    sentinel: tuple[Path, str],
    expected_stdout: Optional[str] = None,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    process_records: list[list[dict[str, np.ndarray]]] = []
    process_evidence: list[dict[str, Any]] = []
    for process_index in range(3):
        output_root = output_parent / f"process_{process_index}"
        environment, disposable_temp = environment_factory(process_index)
        records, execution = _run_lane_subprocess(
            command_factory(output_root),
            output_root=output_root,
            environment=environment,
            disposable_temp=disposable_temp,
            sentinel=sentinel,
            expected_stdout=expected_stdout,
        )
        process_records.append(records)
        process_evidence.append(execution)
    cross_process = _determinism([records[0] for records in process_records])
    return process_records[0], {
        "fresh_process_count": len(process_records),
        "repeat_count_per_process": REPEATS,
        "within_process_bit_exact": all(
            row["determinism"]["bit_exact"] for row in process_evidence
        ),
        "cross_process_determinism": cross_process,
        "processes": process_evidence,
    }


def _python_lane_command(backend: str, model: Path, input_path: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "python-lane",
        "--backend",
        backend,
        "--model",
        str(model),
        "--input",
        str(input_path),
        "--output-dir",
        str(output),
    ]


def _cpp_lane_command(executable: Path, model: Path, input_path: Path, output: Path) -> list[str]:
    return [
        str(executable),
        "--model",
        str(model),
        "--input",
        str(input_path),
        "--output-dir",
        str(output),
        "--manifest",
        str(output / "probe_manifest.json"),
        "--repeat",
        str(REPEATS),
    ]


def _isolated_aoti_environment(temp_root: Path) -> dict[str, str]:
    if temp_root.exists():
        raise NativeFeasibilityError("Disposable AOTI TEMP already exists")
    temp_root.mkdir(parents=True)
    if any(temp_root.iterdir()):
        raise NativeFeasibilityError("Disposable AOTI TEMP is not empty")
    work_root = (PROJECT_ROOT / DEFAULT_WORK_ROOT).resolve(strict=False)
    if not temp_root.resolve(strict=True).is_relative_to(work_root):
        raise NativeFeasibilityError("Disposable AOTI TEMP escapes the governed work root")
    environment = dict(os.environ)
    environment.update(
        {
            "TEMP": str(temp_root),
            "TMP": str(temp_root),
            "AXON_AOTI_DISPOSABLE_TEMP": str(temp_root),
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return environment


def run_execution(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    builder = _load_builder()
    package_manifest = builder.verify_package_manifest(root, DEFAULT_PACKAGE_MANIFEST)
    lease = _consume_lease(
        root,
        authorization_path=DEFAULT_EXECUTION_AUTHORIZATION,
        ready_path=DEFAULT_EXECUTION_LEASE,
        final_path=DEFAULT_EXECUTION_FINAL_LEASE,
        expected_schema="axon_loop28_pytorch_native_execution_lease_v1",
    )
    if lease["authorization"].get("package_manifest_sha256") != sha256_file(
        _resolve_within(root, DEFAULT_PACKAGE_MANIFEST)
    ):
        raise NativeFeasibilityError("Execution authorization package binding mismatch")

    input_path = _resolve_within(root, DEFAULT_INPUT)
    aoti_path = _resolve_within(root, DEFAULT_AOTI_PACKAGE)
    cpp_aoti = _resolve_within(root, DEFAULT_CPP_AOTI)
    cpp_aten = _resolve_within(root, DEFAULT_CPP_ATEN)
    torchscript_record = package_manifest["artifacts"].get("torchscript_control", {})
    torchscript_path = None
    cpp_libtorch = None
    if torchscript_record.get("status") == "generated":
        torchscript_path = _resolve_within(root, DEFAULT_TORCHSCRIPT)
        optional_cpp = (root / DEFAULT_CPP_LIBTORCH).resolve(strict=False)
        if optional_cpp.is_file():
            cpp_libtorch = optional_cpp
    work_root = (root / DEFAULT_WORK_ROOT).resolve(strict=False)
    work_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    lanes: dict[str, list[dict[str, np.ndarray]]] = {}
    executions: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="execution_", dir=work_root) as temporary:
        run_root = Path(temporary)
        sentinel_path = run_root / "external_safety_sentinel.bin"
        _write_exclusive(sentinel_path, b"axon-aoti-temp-containment-v1\n")
        sentinel = (sentinel_path, sha256_file(sentinel_path))

        eager_output = run_root / "eager"
        lanes["eager"], executions["eager"] = _run_lane_subprocess(
            _python_lane_command("eager", aoti_path, input_path, eager_output),
            output_root=eager_output,
            sentinel=sentinel,
        )

        if torchscript_path is not None:
            torchscript_output = run_root / "torchscript_python"
            lanes["torchscript_python"], executions["torchscript_python"] = _run_lane_subprocess(
                _python_lane_command(
                    "torchscript_python", torchscript_path, input_path, torchscript_output
                ),
                output_root=torchscript_output,
                sentinel=sentinel,
            )

        system_root = Path(os.environ.get("SystemRoot", "C:/Windows"))
        lean_runtime_path = os.pathsep.join(
            [str(TORCH_ROOT / "lib"), str(system_root / "System32"), str(system_root)]
        )
        lean_environment = dict(os.environ)
        lean_environment.update(
            {
                "PATH": lean_runtime_path,
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "AOTI_RUNTIME_CHECK_INPUTS": "1",
            }
        )

        def aten_environment(_process_index: int):
            return dict(lean_environment), None

        lanes["aten_cpp"], executions["aten_cpp"] = _run_three_fresh_processes(
            command_factory=lambda output: _cpp_lane_command(
                cpp_aten, Path("builtin"), input_path, output
            ),
            output_parent=run_root / "aten_cpp",
            environment_factory=aten_environment,
            sentinel=sentinel,
        )

        def python_aoti_environment(process_index: int):
            disposable = run_root / f"aoti_python_temp_{process_index}"
            environment = _isolated_aoti_environment(disposable)
            environment["PATH"] = os.pathsep.join(
                [
                    str(TORCH_ROOT / "lib"),
                    str(PYTHON_ROOT),
                    str(system_root / "System32"),
                    str(system_root),
                ]
            )
            environment["AOTI_RUNTIME_CHECK_INPUTS"] = "1"
            return environment, disposable

        lanes["aoti_python"], executions["aoti_python"] = _run_three_fresh_processes(
            command_factory=lambda output: _python_lane_command(
                "aoti_python", aoti_path, input_path, output
            ),
            output_parent=run_root / "aoti_python",
            environment_factory=python_aoti_environment,
            sentinel=sentinel,
        )

        def cpp_aoti_environment(process_index: int):
            disposable = run_root / f"aoti_cpp_temp_{process_index}"
            environment = _isolated_aoti_environment(disposable)
            environment["PATH"] = lean_runtime_path
            environment["AOTI_RUNTIME_CHECK_INPUTS"] = "1"
            return environment, disposable

        lanes["aoti_cpp"], executions["aoti_cpp"] = _run_three_fresh_processes(
            command_factory=lambda output: _cpp_lane_command(
                cpp_aoti, aoti_path, input_path, output
            ),
            output_parent=run_root / "aoti_cpp",
            environment_factory=cpp_aoti_environment,
            sentinel=sentinel,
            expected_stdout="AOTI_DEVICE_KEY=cpu",
        )

        if cpp_libtorch is not None and torchscript_path is not None:
            libtorch_output = run_root / "libtorch_cpp"
            lanes["libtorch_cpp"], executions["libtorch_cpp"] = _run_lane_subprocess(
                _cpp_lane_command(cpp_libtorch, torchscript_path, input_path, libtorch_output),
                output_root=libtorch_output,
                environment=lean_environment,
                sentinel=sentinel,
            )

        if not sentinel_path.is_file() or sha256_file(sentinel_path) != sentinel[1]:
            raise NativeFeasibilityError("Final AOTI safety sentinel verification failed")

    reference = lanes["eager"][0]
    comparisons: dict[str, Any] = {}
    all_passed = True
    for lane, records in lanes.items():
        rows = {name: compare_arrays(reference[name], records[0][name]) for name in reference}
        passed = executions[lane]["determinism"]["bit_exact"] and all(
            row["passed"] for row in rows.values()
        )
        comparisons[lane] = {"passed": passed, "outputs": rows}
        all_passed = all_passed and passed

    elapsed = time.monotonic() - started
    evidence = {
        "schema": "axon_loop28_pytorch_native_execution_evidence_v1",
        "loop_id": LOOP_ID,
        "lease": lease,
        "package_manifest_sha256": sha256_file(_resolve_within(root, DEFAULT_PACKAGE_MANIFEST)),
        "package_artifacts": package_manifest["artifacts"],
        "input": {
            "path": DEFAULT_INPUT.as_posix(),
            "sha256": sha256_file(input_path),
            "shape": [2, 8],
            "dtype": "float32",
        },
        "lanes": executions,
        "comparisons": comparisons,
        "all_lanes_passed": all_passed,
        "budget": {
            "python_lane_count": 3,
            "cpp_lane_count": 2,
            "repeat_count_per_lane": REPEATS,
            "elapsed_seconds": elapsed,
        },
        "safety": {
            "aoti_child_process_count": 6,
            "native_fresh_process_count": 6,
            "normal_user_temp_used_for_aoti": False,
            "disposable_temp_roots_removed_by_loader": True,
            "external_sentinel_unchanged": True,
            "compiler_on_cpp_runtime_path": False,
            "python_on_cpp_runtime_path": False,
            "gpu_execution_count": 0,
            "network_request_count": 0,
            "forbidden_project_artifact_access_count": 0,
            "quality_metric_count": 0,
        },
        "claim_boundary": lease["authorization"]["claim_boundary"],
        "decision": (
            "aoti_cpu_packaging_native_load_feasible_current_host"
            if all_passed
            else "aoti_numeric_or_route_mismatch"
        ),
    }
    _write_json_exclusive(
        _resolve_within(root, DEFAULT_EXECUTION_EVIDENCE, must_exist=False), evidence
    )
    return evidence


def _failure_payload(mode: str, error: Exception) -> dict[str, Any]:
    return {
        "schema": "axon_loop28_pytorch_native_failure_v1",
        "loop_id": LOOP_ID,
        "mode": mode,
        "error_type": type(error).__name__,
        "error": str(error)[:4000],
        "quality_metric_count": 0,
        "decision": "administrative_failure_no_execution",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "package", "execute", "python-lane"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--backend", choices=("eager", "aoti_python", "torchscript_python"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "python-lane":
        if not args.backend or not args.model or not args.input or not args.output_dir:
            raise NativeFeasibilityError("python-lane requires backend, model, input, and output")
        return run_python_lane(args.backend, args.model, args.input, args.output_dir)

    project_root = args.project_root.resolve(strict=True)
    try:
        if args.mode == "build":
            payload = run_cpp_build(project_root)
        elif args.mode == "package":
            payload = run_package(project_root)
        else:
            payload = run_execution(project_root)
    except Exception as exc:  # noqa: BLE001 - a consumed lease requires durable failure evidence.
        failure_paths = {
            "build": DEFAULT_BUILD_FAILURE,
            "package": DEFAULT_PACKAGE_FAILURE,
            "execute": DEFAULT_EXECUTION_FAILURE,
        }
        relative = failure_paths[args.mode]
        failure_path = _resolve_within(project_root, relative, must_exist=False)
        if not failure_path.exists():
            _write_json_exclusive(failure_path, _failure_payload(args.mode, exc))
        raise
    print(json.dumps({"mode": args.mode, "decision": payload["decision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
