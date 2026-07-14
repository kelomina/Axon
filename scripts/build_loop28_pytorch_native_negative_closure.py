#!/usr/bin/env python3
"""Build the immutable negative closure for the Loop28 native package attempt."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "p0_loop28_pytorch_native_feasibility_001"
MANIFEST_DIR = Path("manifests/roadmap_9997/p0_loop28_pytorch_native_feasibility")
REPORT_DIR = Path("reports/roadmap_9997/p0_loop28_pytorch_native_feasibility")
ARTIFACT_DIR = Path("artifacts/roadmap_9997/p0_loop28_pytorch_native_feasibility/tiny_v1")

IMPLEMENTATION = MANIFEST_DIR / "implementation_manifest.json"
BUILD_RECEIPT = REPORT_DIR / "cpp_build_receipt.final.json"
CONTROLLER_MANIFEST = MANIFEST_DIR / "package_controller_manifest.json"
NEGATIVE_AUTHORIZATION = MANIFEST_DIR / "negative_closure_authorization.json"
PACKAGE_AUTHORIZATION = MANIFEST_DIR / "package_authorization.json"
PACKAGE_FINAL_LEASE = MANIFEST_DIR / "package_lease.final.json"
PACKAGE_FAILURE = REPORT_DIR / "package_failure.final.json"
CONTROLLER_FAILURE = REPORT_DIR / "package_controller_failure.final.json"
ATTEMPT_001_DIAGNOSTIC = REPORT_DIR / "package_attempt_001_diagnostic.final.json"
ATTEMPT_002_DIAGNOSTIC = REPORT_DIR / "package_attempt_002_diagnostic.final.json"
DECODE_DIAGNOSTIC = REPORT_DIR / "compiler_decode_diagnostic.final.json"
UPSTREAM_RESEARCH = REPORT_DIR / "upstream_decode_compatibility_research.final.json"
PARTIAL_INPUT = ARTIFACT_DIR / "input.f32.bin"

BASE_BUILDER = Path("scripts/build_loop28_pytorch_native_feasibility_manifest.py")
CONTROLLER = Path("scripts/run_loop28_pytorch_native_package_controller.py")
BUILDER = Path("scripts/build_loop28_pytorch_native_negative_closure.py")
BUILDER_TEST = Path("tests/test_build_loop28_pytorch_native_negative_closure.py")

FAILURE_MANIFEST = MANIFEST_DIR / "package_failure_manifest.json"
POST_MANIFEST = MANIFEST_DIR / "post_manifest.json"
FINAL_DOCS = (
    ("goal_delta", REPORT_DIR / "goal_delta.final.md", "immutable_goal_delta"),
    ("journal_entry", REPORT_DIR / "journal_entry.final.md", "immutable_journal_entry"),
    ("status", REPORT_DIR / "status.final.md", "immutable_owner_status"),
)

EXPECTED_IMPLEMENTATION_SHA256 = "45d38299623ab32e701d6bf1408a509caa19513287c239b3866532cd405bd3f4"
EXPECTED_BUILD_RECEIPT_SHA256 = "2ef9fc018cf72c0552902d4e9cca2190b4c5e983bdd15d4917dd5d926043dc77"
EXPECTED_CONTROLLER_MANIFEST_SHA256 = (
    "d5164f3d785419fa9eec977580e55f4e8fa5e52ea798765a9a6691d8c5510dd7"
)
EXPECTED_INPUT_SHA256 = "caa371218bdbb95cb73bfe7ab65ec2f8f69222a747fca8f889b2bdc3e693d28b"


class NegativeClosureError(RuntimeError):
    """Raised when the negative closure cannot be proven from immutable evidence."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise NegativeClosureError(f"Duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NegativeClosureError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise NegativeClosureError(f"JSON artifact must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_within(project_root: Path, relative: Path, *, must_exist: bool = True) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise NegativeClosureError(f"Path must remain project-relative: {relative}")
    root = project_root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=must_exist)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise NegativeClosureError(f"Path escapes project root: {relative}") from exc
    if must_exist and not candidate.is_file():
        raise NegativeClosureError(f"Required artifact is not a file: {relative}")
    return candidate


def _validate_timestamp(value: str) -> str:
    if not value or not value.endswith("Z"):
        raise NegativeClosureError("generated_at_utc must end in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise NegativeClosureError("generated_at_utc is invalid") from exc
    if parsed.tzinfo != dt.timezone.utc:
        raise NegativeClosureError("generated_at_utc must use UTC")
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


def _load_module(relative: Path, name: str):
    path = PROJECT_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise NegativeClosureError(f"Unable to import governed module: {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_hash(project_root: Path, relative: Path, expected: str, purpose: str) -> Path:
    path = _resolve_within(project_root, relative)
    if sha256_file(path) != expected:
        raise NegativeClosureError(f"Immutable hash drifted: {purpose}")
    return path


def verify_negative_chain(project_root: Path) -> dict[str, dict[str, Any]]:
    root = project_root.resolve(strict=True)
    base_builder = _load_module(BASE_BUILDER, "loop28_base_builder_for_negative_closure")
    base_builder.verify_implementation_manifest(root, IMPLEMENTATION)
    _require_hash(root, IMPLEMENTATION, EXPECTED_IMPLEMENTATION_SHA256, "implementation")
    build_path = _require_hash(root, BUILD_RECEIPT, EXPECTED_BUILD_RECEIPT_SHA256, "build receipt")
    build_receipt = load_json_strict(build_path)
    if build_receipt.get("decision") != (
        "lean_aten_and_aoti_hosts_built_package_requires_authorization"
    ):
        raise NegativeClosureError("Build receipt decision mismatch")

    controller_path = _require_hash(
        root, CONTROLLER_MANIFEST, EXPECTED_CONTROLLER_MANIFEST_SHA256, "controller manifest"
    )
    controller_manifest = load_json_strict(controller_path)
    if controller_manifest.get("decision") != (
        "package_controller_frozen_reissue_authorization_required"
    ):
        raise NegativeClosureError("Controller manifest decision mismatch")
    controller = _load_module(CONTROLLER, "loop28_package_controller_for_negative_closure")
    controller.verify_controller_manifest(root)

    negative_authorization = load_json_strict(_resolve_within(root, NEGATIVE_AUTHORIZATION))
    if negative_authorization.get("decision") != (
        "authorize_immutable_negative_closure_after_package_budget_exhaustion"
    ):
        raise NegativeClosureError("Negative closure authorization mismatch")
    lineage = negative_authorization.get("lineage", {})
    expected_lineage = {
        "implementation_manifest_sha256": EXPECTED_IMPLEMENTATION_SHA256,
        "cpp_build_receipt_sha256": EXPECTED_BUILD_RECEIPT_SHA256,
        "package_controller_manifest_sha256": EXPECTED_CONTROLLER_MANIFEST_SHA256,
    }
    for key, expected in expected_lineage.items():
        if lineage.get(key) != expected:
            raise NegativeClosureError(f"Negative closure lineage mismatch: {key}")

    package_authorization_path = _resolve_within(root, PACKAGE_AUTHORIZATION)
    package_authorization = load_json_strict(package_authorization_path)
    if package_authorization.get("decision") != (
        "authorize_single_governed_tiny_aoti_package_generation_reissue_001"
    ):
        raise NegativeClosureError("Final package authorization mismatch")
    package_lease = load_json_strict(_resolve_within(root, PACKAGE_FINAL_LEASE))
    if package_lease.get("status") != "consumed_before_execution":
        raise NegativeClosureError("Final package lease was not consumed")
    if package_lease.get("authorization_sha256") != sha256_file(package_authorization_path):
        raise NegativeClosureError("Final package lease authorization drifted")

    package_failure = load_json_strict(_resolve_within(root, PACKAGE_FAILURE))
    controller_failure = load_json_strict(_resolve_within(root, CONTROLLER_FAILURE))
    if package_failure.get("error_type") != "InductorError":
        raise NegativeClosureError("Base package failure type mismatch")
    if controller_failure.get("protected_stage_started") is not True:
        raise NegativeClosureError("Controller failure did not record protected execution")
    if controller_failure.get("package_load_count") != 0:
        raise NegativeClosureError("Controller failure reports a package load")

    attempt_001 = load_json_strict(_resolve_within(root, ATTEMPT_001_DIAGNOSTIC))
    attempt_002 = load_json_strict(_resolve_within(root, ATTEMPT_002_DIAGNOSTIC))
    decode = load_json_strict(_resolve_within(root, DECODE_DIAGNOSTIC))
    research = load_json_strict(_resolve_within(root, UPSTREAM_RESEARCH))
    if attempt_001.get("decision") != (
        "administrative_pre_model_failure_eligible_for_single_package_reissue"
    ):
        raise NegativeClosureError("Attempt 001 diagnostic decision mismatch")
    if attempt_002.get("decision") != (
        "package_budget_exhausted_utf8_cp936_compiler_probe_failure_no_load"
    ):
        raise NegativeClosureError("Attempt 002 diagnostic decision mismatch")
    if decode.get("decision") != (
        "confirmed_pytorch_compiler_probe_decoder_console_codepage_mismatch"
    ):
        raise NegativeClosureError("Compiler decode diagnosis mismatch")
    if research.get("decision") != (
        "upstream_v213_decoder_fix_identified_successor_still_requires_new_loop"
    ):
        raise NegativeClosureError("Upstream research decision mismatch")
    if attempt_002.get("lineage", {}).get("compiler_decode_diagnostic", {}).get(
        "sha256"
    ) != sha256_file(_resolve_within(root, DECODE_DIAGNOSTIC)):
        raise NegativeClosureError("Attempt 002 decode-diagnostic binding mismatch")

    input_path = _require_hash(root, PARTIAL_INPUT, EXPECTED_INPUT_SHA256, "partial input")
    if input_path.stat().st_size != 64:
        raise NegativeClosureError("Partial tiny input size mismatch")

    # 负结论只有在 package/runtime 产物确实不存在时才成立。
    forbidden_outputs = (
        ARTIFACT_DIR / "tiny_cpu_model.pt2",
        ARTIFACT_DIR / "tiny_cpu_control.pt",
        REPORT_DIR / "package_receipt.final.json",
        MANIFEST_DIR / "package_manifest.json",
        MANIFEST_DIR / "execution_authorization.json",
        MANIFEST_DIR / "execution_lease.json",
        MANIFEST_DIR / "execution_lease.final.json",
        REPORT_DIR / "execution_evidence.final.json",
        REPORT_DIR / "package_controller_receipt.final.json",
    )
    for relative in forbidden_outputs:
        if (root / relative).exists():
            raise NegativeClosureError(f"Unexpected positive-path artifact exists: {relative}")
    return {
        "build_receipt": build_receipt,
        "controller_manifest": controller_manifest,
        "negative_authorization": negative_authorization,
        "package_authorization": package_authorization,
        "package_lease": package_lease,
        "package_failure": package_failure,
        "controller_failure": controller_failure,
        "attempt_001": attempt_001,
        "attempt_002": attempt_002,
        "decode": decode,
        "research": research,
    }


def build_failure_manifest(project_root: Path, *, generated_at_utc: str) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    timestamp = _validate_timestamp(generated_at_utc)
    chain = verify_negative_chain(root)
    artifact_specs = (
        ("implementation_manifest", IMPLEMENTATION, "frozen_source_closure"),
        ("cpp_build_receipt", BUILD_RECEIPT, "successful_lean_cpp_build"),
        ("controller_manifest", CONTROLLER_MANIFEST, "package_controller_closure"),
        ("negative_authorization", NEGATIVE_AUTHORIZATION, "negative_closure_authority"),
        ("package_authorization", PACKAGE_AUTHORIZATION, "protected_attempt_authority"),
        ("package_final_lease", PACKAGE_FINAL_LEASE, "consumed_single_use_lease"),
        ("package_failure", PACKAGE_FAILURE, "base_inductor_failure"),
        ("controller_failure", CONTROLLER_FAILURE, "controller_failure_record"),
        ("attempt_001_diagnostic", ATTEMPT_001_DIAGNOSTIC, "administrative_attempt"),
        ("attempt_002_diagnostic", ATTEMPT_002_DIAGNOSTIC, "protected_attempt_diagnosis"),
        ("decode_diagnostic", DECODE_DIAGNOSTIC, "compiler_encoding_root_cause"),
        ("upstream_research", UPSTREAM_RESEARCH, "official_upstream_comparison"),
        ("partial_input", PARTIAL_INPUT, "only_retained_tiny_artifact"),
        ("negative_builder", BUILDER, "negative_manifest_builder"),
        ("negative_builder_test", BUILDER_TEST, "focused_negative_closure_tests"),
    )
    artifacts = [_artifact_record(root, name, path, role) for name, path, role in artifact_specs]
    return {
        "schema": "axon_loop28_pytorch_native_package_failure_manifest_v1",
        "loop_id": LOOP_ID,
        "generated_at_utc": timestamp,
        "lineage": {
            "implementation_manifest_sha256": EXPECTED_IMPLEMENTATION_SHA256,
            "cpp_build_receipt_sha256": EXPECTED_BUILD_RECEIPT_SHA256,
            "package_controller_manifest_sha256": EXPECTED_CONTROLLER_MANIFEST_SHA256,
            "package_authorization_sha256": sha256_file(
                _resolve_within(root, PACKAGE_AUTHORIZATION)
            ),
            "package_final_lease_sha256": sha256_file(_resolve_within(root, PACKAGE_FINAL_LEASE)),
        },
        "outcome": {
            "direct_aten_and_aoti_hosts_built": True,
            "torch_export_completed": True,
            "aoti_compile_and_package_call_count": 1,
            "aoti_package_created": False,
            "torchscript_export_call_count": 0,
            "package_load_count": 0,
            "native_runtime_execution_count": 0,
            "checkpoint_or_onnx_load_count": 0,
            "raw_split_cache_heldout_access_count": 0,
            "protected_stage_network_request_count": 0,
            "post_failure_research_request_count": chain["research"]["research_boundary"][
                "post_failure_external_http_requests"
            ],
            "quality_metric_count": 0,
            "failure_class": chain["attempt_002"]["failure_class"],
            "failure_root_cause": chain["decode"]["decision"],
            "upstream_successor_basis": chain["research"]["decision"],
        },
        "integrity": {
            "artifact_count": len(artifacts),
            "all_required_present": True,
            "positive_package_artifacts_absent": True,
            "runtime_artifacts_absent": True,
            "output_replacement_allowed": False,
        },
        "artifacts": artifacts,
        "claim_boundary": chain["negative_authorization"]["claim_boundary"],
        "decision": "tiny_aoti_package_generation_closed_utf8_cp936_compiler_probe_failure_no_load",
    }


def verify_failure_manifest(project_root: Path, output: Path = FAILURE_MANIFEST) -> dict[str, Any]:
    path = _resolve_within(project_root, output)
    payload = load_json_strict(path)
    if payload.get("schema") != "axon_loop28_pytorch_native_package_failure_manifest_v1":
        raise NegativeClosureError("Package failure manifest schema mismatch")
    rebuilt = build_failure_manifest(
        project_root, generated_at_utc=str(payload.get("generated_at_utc") or "")
    )
    if payload != rebuilt:
        raise NegativeClosureError("Package failure manifest no longer matches evidence")
    return payload


def build_post_manifest(project_root: Path, *, generated_at_utc: str) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    timestamp = _validate_timestamp(generated_at_utc)
    failure = verify_failure_manifest(root)
    artifacts = [
        _artifact_record(root, "package_failure_manifest", FAILURE_MANIFEST, "negative_result"),
        *[_artifact_record(root, name, path, role) for name, path, role in FINAL_DOCS],
    ]
    return {
        "schema": "axon_loop28_pytorch_native_feasibility_negative_post_manifest_v1",
        "loop_id": LOOP_ID,
        "generated_at_utc": timestamp,
        "lineage": {
            "parent_onnx_operator_closure_sha256": (
                "4eae028bdffa0683b273bfadf4ac46df6cb7388c3742dd6222b28fafb9056e6e"
            ),
            "implementation_manifest_sha256": EXPECTED_IMPLEMENTATION_SHA256,
            "package_failure_manifest_sha256": sha256_file(_resolve_within(root, FAILURE_MANIFEST)),
        },
        "outcome": {
            "cpp_build_passed": True,
            "package_generated": False,
            "runtime_executed": False,
            "failure_decision": failure["decision"],
            "successor_loop_required": True,
            "quality_metric_count": 0,
            "legacy_full_test_f1_changed": False,
        },
        "artifacts": artifacts,
        "integrity": {
            "artifact_count": len(artifacts),
            "all_required_present": True,
            "closure_is_negative_not_incomplete": True,
        },
        "claim_boundary": failure["claim_boundary"],
        "decision": "post_tiny_aoti_package_closed_utf8_cp936_compiler_probe_failure_no_load",
    }


def verify_post_manifest(project_root: Path, output: Path = POST_MANIFEST) -> dict[str, Any]:
    path = _resolve_within(project_root, output)
    payload = load_json_strict(path)
    if payload.get("schema") != (
        "axon_loop28_pytorch_native_feasibility_negative_post_manifest_v1"
    ):
        raise NegativeClosureError("Negative post manifest schema mismatch")
    rebuilt = build_post_manifest(
        project_root, generated_at_utc=str(payload.get("generated_at_utc") or "")
    )
    if payload != rebuilt:
        raise NegativeClosureError("Negative post manifest no longer matches evidence")
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
        raise NegativeClosureError(f"Output already exists: {path}") from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("failure", "post"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--generated-at-utc")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve(strict=True)
    defaults = {"failure": FAILURE_MANIFEST, "post": POST_MANIFEST}
    output = args.output or defaults[args.mode]
    builders = {"failure": build_failure_manifest, "post": build_post_manifest}
    verifiers = {"failure": verify_failure_manifest, "post": verify_post_manifest}
    if args.verify:
        payload = verifiers[args.mode](root, output)
    else:
        if not args.generated_at_utc:
            raise NegativeClosureError("--generated-at-utc is required when building")
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
