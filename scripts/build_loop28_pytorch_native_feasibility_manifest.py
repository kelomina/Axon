#!/usr/bin/env python3
"""Build immutable manifests for the tiny PyTorch-native feasibility loop."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "p0_loop28_pytorch_native_feasibility_001"
LOOP_MANIFEST_DIR = Path("manifests/roadmap_9997/p0_loop28_pytorch_native_feasibility")
LOOP_REPORT_DIR = Path("reports/roadmap_9997/p0_loop28_pytorch_native_feasibility")

PROPOSAL = LOOP_MANIFEST_DIR / "proposal.json"
AUTHORIZATION = LOOP_MANIFEST_DIR / "authorization.json"
PREFLIGHT = LOOP_MANIFEST_DIR / "preflight.json"
AMENDMENT = LOOP_MANIFEST_DIR / "design_amendment.json"
PARENT_CLOSURE = Path(
    "manifests/roadmap_9997/p0_loop28_onnx_operator_remediation/post_manifest.json"
)
DEFAULT_IMPLEMENTATION = LOOP_MANIFEST_DIR / "implementation_manifest.json"
DEFAULT_PACKAGE_MANIFEST = LOOP_MANIFEST_DIR / "package_manifest.json"
DEFAULT_POST = LOOP_MANIFEST_DIR / "post_manifest.json"

BUILD_RECEIPT = LOOP_REPORT_DIR / "cpp_build_receipt.final.json"
PACKAGE_RECEIPT = LOOP_REPORT_DIR / "package_receipt.final.json"
EXECUTION_EVIDENCE = LOOP_REPORT_DIR / "execution_evidence.final.json"
FINAL_DOCS = (
    ("goal_delta", LOOP_REPORT_DIR / "goal_delta.final.md", "immutable_goal_delta"),
    ("journal_entry", LOOP_REPORT_DIR / "journal_entry.final.md", "immutable_journal"),
    ("final_status", LOOP_REPORT_DIR / "status.final.md", "immutable_owner_status"),
)

RUNNER = Path("scripts/run_loop28_pytorch_native_feasibility.py")
RUNNER_TEST = Path("tests/test_run_loop28_pytorch_native_feasibility.py")
BUILDER = Path("scripts/build_loop28_pytorch_native_feasibility_manifest.py")
BUILDER_TEST = Path("tests/test_build_loop28_pytorch_native_feasibility_manifest.py")
CMAKE = Path("tools/axon_tiny_pytorch_native/CMakeLists.txt")
COMMON_HEADER = Path("tools/axon_tiny_pytorch_native/src/probe_common.h")
ATEN_SOURCE = Path("tools/axon_tiny_pytorch_native/src/aten_probe.cpp")
AOTI_SOURCE = Path("tools/axon_tiny_pytorch_native/src/aoti_probe.cpp")
LIBTORCH_SOURCE = Path("tools/axon_tiny_pytorch_native/src/libtorch_probe.cpp")

IMPLEMENTATION_ARTIFACTS = (
    ("proposal", PROPOSAL, "frozen_tiny_proposal"),
    ("authorization", AUTHORIZATION, "a1_scope_authorization"),
    ("preflight", PREFLIGHT, "toolchain_and_safety_preflight"),
    ("design_amendment", AMENDMENT, "adversarial_contract_correction"),
    ("parent_closure", PARENT_CLOSURE, "immutable_onnx_stop_parent"),
    ("runner", RUNNER, "governed_build_package_execution_runner"),
    ("runner_test", RUNNER_TEST, "focused_runner_tests"),
    ("manifest_builder", BUILDER, "manifest_builder"),
    ("manifest_builder_test", BUILDER_TEST, "focused_builder_tests"),
    ("cmake", CMAKE, "lean_windows_build_contract"),
    ("probe_common", COMMON_HEADER, "native_io_and_repeat_contract"),
    ("aten_source", ATEN_SOURCE, "direct_aten_toolchain_control"),
    ("aoti_source", AOTI_SOURCE, "aoti_package_loader_probe"),
    ("libtorch_source", LIBTORCH_SOURCE, "optional_torchscript_diagnostic"),
)

DEFAULT_INPUT = Path(
    "artifacts/roadmap_9997/p0_loop28_pytorch_native_feasibility/tiny_v1/input.f32.bin"
)
DEFAULT_AOTI_PACKAGE = Path(
    "artifacts/roadmap_9997/p0_loop28_pytorch_native_feasibility/tiny_v1/tiny_cpu_model.pt2"
)
DEFAULT_TORCHSCRIPT = Path(
    "artifacts/roadmap_9997/p0_loop28_pytorch_native_feasibility/tiny_v1/tiny_cpu_control.pt"
)
DUMPBIN_EXE = Path(
    "C:/Program Files/Microsoft Visual Studio/18/Insiders/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64/dumpbin.exe"
)


class NativeManifestError(RuntimeError):
    """Raised when a feasibility manifest cannot be proven from current artifacts."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise NativeManifestError(f"Duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeManifestError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise NativeManifestError(f"JSON artifact must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_within(project_root: Path, relative: Path, *, must_exist: bool = True) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise NativeManifestError(f"Path must remain project-relative: {relative}")
    root = project_root.resolve(strict=True)
    try:
        candidate = (root / relative).resolve(strict=must_exist)
    except OSError as exc:
        raise NativeManifestError(f"Required artifact is missing: {relative}") from exc
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise NativeManifestError(f"Path escapes project root: {relative}") from exc
    if must_exist and not candidate.is_file():
        raise NativeManifestError(f"Required artifact is not a file: {relative}")
    return candidate


def _validate_timestamp(value: str) -> str:
    if not value or not value.endswith("Z"):
        raise NativeManifestError("generated_at_utc must end in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise NativeManifestError("generated_at_utc is invalid") from exc
    if parsed.tzinfo != dt.timezone.utc:
        raise NativeManifestError("generated_at_utc must use UTC")
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


def _verify_external_file(record: Mapping[str, Any], purpose: str) -> None:
    path = Path(str(record.get("path", "")))
    if not path.is_absolute() or not path.is_file():
        raise NativeManifestError(f"External preflight file is missing: {purpose}")
    if sha256_file(path) != record.get("sha256"):
        raise NativeManifestError(f"External preflight file hash drifted: {purpose}")


def _verify_chain(project_root: Path) -> dict[str, dict[str, Any]]:
    root = project_root.resolve(strict=True)
    payloads = {
        "proposal": load_json_strict(_resolve_within(root, PROPOSAL)),
        "authorization": load_json_strict(_resolve_within(root, AUTHORIZATION)),
        "preflight": load_json_strict(_resolve_within(root, PREFLIGHT)),
        "amendment": load_json_strict(_resolve_within(root, AMENDMENT)),
        "parent": load_json_strict(_resolve_within(root, PARENT_CLOSURE)),
    }
    proposal = payloads["proposal"]
    authorization = payloads["authorization"]
    preflight = payloads["preflight"]
    amendment = payloads["amendment"]
    parent = payloads["parent"]
    if proposal.get("schema") != "axon_loop28_pytorch_native_feasibility_proposal_v1":
        raise NativeManifestError("Proposal schema mismatch")
    if proposal.get("loop_id") != LOOP_ID:
        raise NativeManifestError("Proposal loop mismatch")
    if authorization.get("proposal", {}).get("sha256") != sha256_file(
        _resolve_within(root, PROPOSAL)
    ):
        raise NativeManifestError("Authorization proposal binding mismatch")
    if preflight.get("decision") != "tiny_native_toolchain_surface_ready_for_single_implementation":
        raise NativeManifestError("Preflight decision mismatch")
    bindings = amendment.get("bindings")
    expected_bindings = {
        "proposal_sha256": sha256_file(_resolve_within(root, PROPOSAL)),
        "authorization_sha256": sha256_file(_resolve_within(root, AUTHORIZATION)),
        "preflight_sha256": sha256_file(_resolve_within(root, PREFLIGHT)),
    }
    if bindings != expected_bindings:
        raise NativeManifestError("Design amendment binding mismatch")
    if amendment.get("decision") != (
        "authorize_lean_aten_control_and_temp_contained_aoti_feasibility"
    ):
        raise NativeManifestError("Design amendment decision mismatch")
    if parent.get("decision") != (
        "post_operator_preflight_exact_tie_fallback_pytorch_native_no_execution"
    ):
        raise NativeManifestError("Parent closure decision mismatch")
    if proposal.get("parent_closure", {}).get("sha256") != sha256_file(
        _resolve_within(root, PARENT_CLOSURE)
    ):
        raise NativeManifestError("Parent closure hash drifted")

    for name, record in preflight["compiler_surface"]["files"].items():
        _verify_external_file(record, f"compiler:{name}")
    for name in ("python_import_library", "python_runtime", "python_header"):
        _verify_external_file(preflight["python_surface"][name], f"python:{name}")
    for name, record in preflight["torch_surface"]["files"].items():
        path = _resolve_within(root, Path(record["path"]))
        if sha256_file(path) != record["sha256"]:
            raise NativeManifestError(f"Torch preflight hash drifted: {name}")
    safety = preflight["risk_register"]["required_temp_containment"]
    if safety.get("normal_user_TEMP_use_forbidden") is not True:
        raise NativeManifestError("AOTI TEMP containment is not fail-closed")
    if amendment["package_safety_contract"].get("precompiled_pyd_count") != 1:
        raise NativeManifestError("Precompiled package requirement drifted")
    return payloads


def build_implementation_manifest(project_root: Path, *, generated_at_utc: str) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    timestamp = _validate_timestamp(generated_at_utc)
    chain = _verify_chain(root)
    artifacts = [
        _artifact_record(root, name, path, role) for name, path, role in IMPLEMENTATION_ARTIFACTS
    ]
    return {
        "schema": "axon_loop28_pytorch_native_feasibility_implementation_manifest_v1",
        "loop_id": LOOP_ID,
        "generated_at_utc": timestamp,
        "contract": {
            "structured_chain_validation": True,
            "duplicate_json_keys_rejected": True,
            "single_implementation_generation": True,
            "direct_aten_is_toolchain_control": True,
            "torchscript_is_optional_diagnostic": True,
            "normal_user_temp_for_aoti_forbidden": True,
            "manifest_self_hashed": False,
            "output_replace_allowed": False,
        },
        "lineage": {
            "proposal_sha256": sha256_file(_resolve_within(root, PROPOSAL)),
            "authorization_sha256": sha256_file(_resolve_within(root, AUTHORIZATION)),
            "preflight_sha256": sha256_file(_resolve_within(root, PREFLIGHT)),
            "design_amendment_sha256": sha256_file(_resolve_within(root, AMENDMENT)),
            "parent_closure_sha256": sha256_file(_resolve_within(root, PARENT_CLOSURE)),
        },
        "build_contract": {
            "generator": "Visual Studio 18 2026",
            "configuration": "Release",
            "cpp_standard": 20,
            "required_targets": ["axon_tiny_aten_probe", "axon_tiny_aoti_probe"],
            "optional_targets": ["axon_tiny_libtorch_probe"],
            "aoti_link_libraries": ["torch_cpu.lib", "c10.lib"],
            "forbidden_aoti_link_libraries": [
                "torch.lib",
                "torch_python.lib",
                "python314.lib",
                "torch_cuda.lib",
            ],
        },
        "validation_contract": {
            "focused_pytest_command": "vnev/Scripts/python.exe -m pytest -q tests/test_run_loop28_pytorch_native_feasibility.py tests/test_build_loop28_pytorch_native_feasibility_manifest.py",
            "required_test_count": 8,
            "ruff_check_required": True,
            "ruff_format_required": True,
            "py_compile_required": True,
        },
        "artifacts": artifacts,
        "integrity": {
            "artifact_count": len(artifacts),
            "required_artifact_count": len(IMPLEMENTATION_ARTIFACTS),
            "all_required_present": len(artifacts) == len(IMPLEMENTATION_ARTIFACTS),
        },
        "claim_boundary": chain["amendment"]["claim_boundary"],
        "decision": "tiny_native_feasibility_source_frozen_build_requires_authorization",
    }


def verify_implementation_manifest(project_root: Path, output: Path) -> dict[str, Any]:
    path = _resolve_within(project_root, output)
    payload = load_json_strict(path)
    if payload.get("schema") != (
        "axon_loop28_pytorch_native_feasibility_implementation_manifest_v1"
    ):
        raise NativeManifestError("Implementation manifest schema mismatch")
    rebuilt = build_implementation_manifest(
        project_root,
        generated_at_utc=str(payload.get("generated_at_utc") or ""),
    )
    if payload != rebuilt:
        raise NativeManifestError("Implementation manifest no longer matches its chain")
    return payload


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "native_feasibility_runner", PROJECT_ROOT / RUNNER
    )
    if spec is None or spec.loader is None:
        raise NativeManifestError("Unable to import native feasibility runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dumpbin_dependencies(path: Path) -> list[str]:
    completed = subprocess.run(
        [str(DUMPBIN_EXE), "/DEPENDENTS", str(path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise NativeManifestError("Generated AOTI .pyd dependency audit failed")
    return sorted(
        {
            line.strip().casefold()
            for line in completed.stdout.splitlines()
            if line.strip().casefold().endswith(".dll")
        }
    )


def build_package_manifest(project_root: Path, *, generated_at_utc: str) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    timestamp = _validate_timestamp(generated_at_utc)
    implementation = verify_implementation_manifest(root, DEFAULT_IMPLEMENTATION)
    receipt = load_json_strict(_resolve_within(root, PACKAGE_RECEIPT))
    if receipt.get("schema") != "axon_loop28_pytorch_native_package_receipt_v1":
        raise NativeManifestError("Package receipt schema mismatch")
    if receipt.get("implementation_manifest_sha256") != sha256_file(
        _resolve_within(root, DEFAULT_IMPLEMENTATION)
    ):
        raise NativeManifestError("Package receipt implementation binding mismatch")
    aoti_path = _resolve_within(root, DEFAULT_AOTI_PACKAGE)
    input_path = _resolve_within(root, DEFAULT_INPUT)
    if receipt["artifacts"]["aoti_package"]["sha256"] != sha256_file(aoti_path):
        raise NativeManifestError("AOTI package receipt hash drifted")
    if receipt["artifacts"]["input"]["sha256"] != sha256_file(input_path):
        raise NativeManifestError("Tiny input receipt hash drifted")

    runner = _load_runner()
    inventory = runner._archive_inventory(aoti_path)
    pyd_members = [row for row in inventory if row["name"].casefold().endswith(".pyd")]
    if len(pyd_members) != 1:
        raise NativeManifestError("AOTI package does not contain exactly one .pyd")
    with (
        zipfile.ZipFile(aoti_path, "r") as archive,
        tempfile.TemporaryDirectory(
            prefix="aoti_dependency_audit_", dir=root / LOOP_REPORT_DIR
        ) as temporary,
    ):
        extracted = Path(temporary) / "model.pyd"
        payload = archive.read(pyd_members[0]["name"])
        with extracted.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        dependencies = _dumpbin_dependencies(extracted)

    forbidden = {
        value.casefold()
        for value in _verify_chain(root)["amendment"]["package_safety_contract"][
            "forbidden_cpp_dependencies"
        ]
    }
    leakage = sorted(forbidden.intersection(dependencies))
    torchscript = receipt["artifacts"]["torchscript_control"]
    artifacts = [
        _artifact_record(root, "implementation_manifest", DEFAULT_IMPLEMENTATION, "source_closure"),
        _artifact_record(root, "build_receipt", BUILD_RECEIPT, "lean_cpp_build_result"),
        _artifact_record(root, "package_receipt", PACKAGE_RECEIPT, "package_generation_result"),
        _artifact_record(root, "tiny_input", DEFAULT_INPUT, "fixed_synthetic_input"),
        _artifact_record(root, "aoti_package", DEFAULT_AOTI_PACKAGE, "precompiled_pt2_package"),
    ]
    if torchscript.get("status") == "generated":
        artifacts.append(
            _artifact_record(
                root, "torchscript_control", DEFAULT_TORCHSCRIPT, "optional_diagnostic_control"
            )
        )
    load_allowed = not leakage
    return {
        "schema": "axon_loop28_pytorch_native_package_manifest_v1",
        "loop_id": LOOP_ID,
        "generated_at_utc": timestamp,
        "lineage": {
            "implementation_manifest_sha256": sha256_file(
                _resolve_within(root, DEFAULT_IMPLEMENTATION)
            ),
            "package_receipt_sha256": sha256_file(_resolve_within(root, PACKAGE_RECEIPT)),
            "parent_closure_sha256": implementation["lineage"]["parent_closure_sha256"],
        },
        "package": {
            "path": DEFAULT_AOTI_PACKAGE.as_posix(),
            "sha256": sha256_file(aoti_path),
            "archive_members": inventory,
            "precompiled_pyd_count": len(pyd_members),
            "precompiled_pyd_dependencies": dependencies,
            "forbidden_dependency_hits": leakage,
            "runtime_compile_fallback_allowed": False,
            "load_allowed": load_allowed,
        },
        "artifacts": artifacts,
        "integrity": {
            "artifact_count": len(artifacts),
            "all_hashes_stable": True,
            "archive_paths_safe": True,
            "duplicate_archive_members": 0,
        },
        "claim_boundary": implementation["claim_boundary"],
        "decision": (
            "tiny_package_dependency_closure_ready_for_execution_authorization"
            if load_allowed
            else "aoti_dependency_leakage_no_load"
        ),
    }


def verify_package_manifest(project_root: Path, output: Path) -> dict[str, Any]:
    path = _resolve_within(project_root, output)
    payload = load_json_strict(path)
    if payload.get("schema") != "axon_loop28_pytorch_native_package_manifest_v1":
        raise NativeManifestError("Package manifest schema mismatch")
    rebuilt = build_package_manifest(
        project_root,
        generated_at_utc=str(payload.get("generated_at_utc") or ""),
    )
    if payload != rebuilt:
        raise NativeManifestError("Package manifest no longer matches its chain")
    return payload


def build_post_manifest(project_root: Path, *, generated_at_utc: str) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    timestamp = _validate_timestamp(generated_at_utc)
    implementation = verify_implementation_manifest(root, DEFAULT_IMPLEMENTATION)
    build_receipt = load_json_strict(_resolve_within(root, BUILD_RECEIPT))
    artifacts = [
        _artifact_record(root, "implementation_manifest", DEFAULT_IMPLEMENTATION, "source_closure"),
        _artifact_record(root, "build_receipt", BUILD_RECEIPT, "cpp_build_result"),
        *[_artifact_record(root, name, path, role) for name, path, role in FINAL_DOCS],
    ]
    outcome: dict[str, Any] = {
        "build_decision": build_receipt.get("decision"),
        "package_generated": False,
        "runtime_executed": False,
        "checkpoint_or_onnx_load_count": 0,
        "raw_split_cache_heldout_access_count": 0,
        "quality_metric_count": 0,
    }
    if (
        build_receipt.get("decision")
        != "lean_aten_and_aoti_hosts_built_package_requires_authorization"
    ):
        decision = f"post_{build_receipt.get('decision')}"
    elif (root / DEFAULT_PACKAGE_MANIFEST).is_file():
        package = verify_package_manifest(root, DEFAULT_PACKAGE_MANIFEST)
        artifacts.append(
            _artifact_record(root, "package_manifest", DEFAULT_PACKAGE_MANIFEST, "package_closure")
        )
        outcome["package_generated"] = True
        outcome["package_decision"] = package["decision"]
        if (
            package["decision"]
            != "tiny_package_dependency_closure_ready_for_execution_authorization"
        ):
            decision = f"post_{package['decision']}"
        elif (root / EXECUTION_EVIDENCE).is_file():
            evidence = load_json_strict(_resolve_within(root, EXECUTION_EVIDENCE))
            artifacts.append(
                _artifact_record(
                    root, "execution_evidence", EXECUTION_EVIDENCE, "tiny_runtime_result"
                )
            )
            outcome["runtime_executed"] = True
            outcome["execution_decision"] = evidence.get("decision")
            decision = f"post_{evidence.get('decision')}"
        else:
            raise NativeManifestError(
                "Successful package closure still requires execution evidence"
            )
    else:
        raise NativeManifestError("Successful C++ build still requires a package manifest")
    return {
        "schema": "axon_loop28_pytorch_native_feasibility_post_manifest_v1",
        "loop_id": LOOP_ID,
        "generated_at_utc": timestamp,
        "lineage": {
            "parent_closure_sha256": implementation["lineage"]["parent_closure_sha256"],
            "implementation_manifest_sha256": sha256_file(
                _resolve_within(root, DEFAULT_IMPLEMENTATION)
            ),
            "build_receipt_sha256": sha256_file(_resolve_within(root, BUILD_RECEIPT)),
        },
        "outcome": outcome,
        "artifacts": artifacts,
        "integrity": {
            "artifact_count": len(artifacts),
            "all_required_present": True,
        },
        "claim_boundary": implementation["claim_boundary"],
        "decision": decision,
    }


def verify_post_manifest(project_root: Path, output: Path) -> dict[str, Any]:
    path = _resolve_within(project_root, output)
    payload = load_json_strict(path)
    if payload.get("schema") != "axon_loop28_pytorch_native_feasibility_post_manifest_v1":
        raise NativeManifestError("Post manifest schema mismatch")
    rebuilt = build_post_manifest(
        project_root,
        generated_at_utc=str(payload.get("generated_at_utc") or ""),
    )
    if payload != rebuilt:
        raise NativeManifestError("Post manifest no longer matches its chain")
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
        raise NativeManifestError(f"Output already exists: {path}") from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("implementation", "package", "post"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--generated-at-utc")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve(strict=True)
    defaults = {
        "implementation": DEFAULT_IMPLEMENTATION,
        "package": DEFAULT_PACKAGE_MANIFEST,
        "post": DEFAULT_POST,
    }
    output = args.output or defaults[args.mode]
    build_functions = {
        "implementation": build_implementation_manifest,
        "package": build_package_manifest,
        "post": build_post_manifest,
    }
    verify_functions = {
        "implementation": verify_implementation_manifest,
        "package": verify_package_manifest,
        "post": verify_post_manifest,
    }
    if args.verify:
        payload = verify_functions[args.mode](root, output)
    else:
        if not args.generated_at_utc:
            raise NativeManifestError("--generated-at-utc is required when building")
        payload = build_functions[args.mode](root, generated_at_utc=args.generated_at_utc)
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
