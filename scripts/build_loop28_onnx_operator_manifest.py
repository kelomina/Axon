#!/usr/bin/env python3
"""Build immutable manifests for the fail-closed Loop28 ONNX operator preflight."""

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
LOOP_ID = "p0_loop28_onnx_operator_remediation_001"

LOOP_MANIFEST_DIR = Path("manifests/roadmap_9997/p0_loop28_onnx_operator_remediation")
LOOP_REPORT_DIR = Path("reports/roadmap_9997/p0_loop28_onnx_operator_remediation")
DEFAULT_IMPLEMENTATION = LOOP_MANIFEST_DIR / "implementation_manifest.json"
DEFAULT_POST = LOOP_MANIFEST_DIR / "post_manifest.json"

PROPOSAL = LOOP_MANIFEST_DIR / "proposal.json"
AUTHORIZATION = LOOP_MANIFEST_DIR / "authorization.json"
AMENDMENT = LOOP_MANIFEST_DIR / "preflight_amendment.json"
PREFLIGHT = LOOP_MANIFEST_DIR / "preflight.json"
PARENT_CLOSURE = Path("manifests/roadmap_9997/p0_loop28_onnx_fidelity/post_manifest.json")
PARENT_EVIDENCE = Path(
    "reports/roadmap_9997/p0_loop28_onnx_fidelity/localization_evidence.final.json"
)
AUDITOR = Path("scripts/remediate_loop28_onnx_operator.py")
AUDITOR_TEST = Path("tests/test_remediate_loop28_onnx_operator.py")
BUILDER = Path("scripts/build_loop28_onnx_operator_manifest.py")
BUILDER_TEST = Path("tests/test_build_loop28_onnx_operator_manifest.py")

CANDIDATE_ROOT = Path("models/roadmap_9997/p0_loop28_onnx_operator_remediation")
FORBIDDEN_EXECUTION_ARTIFACTS = (
    LOOP_MANIFEST_DIR / "generation_authorization.json",
    LOOP_MANIFEST_DIR / "generation_lease.json",
    LOOP_MANIFEST_DIR / "generation_lease.final.json",
    LOOP_MANIFEST_DIR / "candidate_manifest.json",
    LOOP_MANIFEST_DIR / "verification_authorization.json",
    LOOP_MANIFEST_DIR / "verification_lease.json",
    LOOP_MANIFEST_DIR / "verification_lease.final.json",
)

IMPLEMENTATION_ARTIFACTS = (
    ("proposal", PROPOSAL, "frozen_single_candidate_proposal"),
    ("authorization", AUTHORIZATION, "a1_scope_authorization"),
    ("preflight_amendment", AMENDMENT, "fail_closed_scope_reduction"),
    ("preflight", PREFLIGHT, "formal_exact_tie_blocker"),
    ("parent_closure", PARENT_CLOSURE, "immutable_fidelity_parent"),
    ("parent_evidence", PARENT_EVIDENCE, "immutable_route_evidence"),
    ("static_auditor", AUDITOR, "no_model_load_preflight_tool"),
    ("static_auditor_test", AUDITOR_TEST, "focused_preflight_tests"),
    ("manifest_builder", BUILDER, "closure_builder"),
    ("manifest_builder_test", BUILDER_TEST, "focused_builder_tests"),
)

FINAL_DOCS = (
    ("goal_delta", LOOP_REPORT_DIR / "goal_delta.final.md", "immutable_goal_delta"),
    (
        "journal_entry",
        LOOP_REPORT_DIR / "journal_entry.final.md",
        "immutable_experiment_record",
    ),
    ("final_status", LOOP_REPORT_DIR / "status.final.md", "immutable_owner_status"),
)


class OperatorManifestError(RuntimeError):
    """Raised when an operator-preflight manifest cannot be proven."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise OperatorManifestError(f"Duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorManifestError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise OperatorManifestError(f"JSON artifact must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_within(project_root: Path, relative: Path, *, must_exist: bool = True) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise OperatorManifestError(f"Path must remain project-relative: {relative}")
    root = project_root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=must_exist)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise OperatorManifestError(f"Path escapes project root: {relative}") from exc
    if must_exist and not candidate.is_file():
        raise OperatorManifestError(f"Required artifact is not a file: {relative}")
    return candidate


def _validate_timestamp(value: str) -> str:
    if not value or not value.endswith("Z"):
        raise OperatorManifestError("generated_at_utc must end in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise OperatorManifestError("generated_at_utc is invalid") from exc
    if parsed.tzinfo != dt.timezone.utc:
        raise OperatorManifestError("generated_at_utc must use UTC")
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


def _load_auditor(project_root: Path):
    path = _resolve_within(project_root, AUDITOR)
    spec = importlib.util.spec_from_file_location("loop28_onnx_operator_auditor", path)
    if spec is None or spec.loader is None:
        raise OperatorManifestError("Unable to import static preflight auditor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_chain(project_root: Path) -> dict[str, dict[str, Any]]:
    proposal = load_json_strict(_resolve_within(project_root, PROPOSAL))
    authorization = load_json_strict(_resolve_within(project_root, AUTHORIZATION))
    amendment = load_json_strict(_resolve_within(project_root, AMENDMENT))
    parent = load_json_strict(_resolve_within(project_root, PARENT_CLOSURE))
    if proposal.get("schema") != "axon_loop28_onnx_operator_remediation_proposal_v1":
        raise OperatorManifestError("Proposal schema mismatch")
    if proposal.get("loop_id") != LOOP_ID:
        raise OperatorManifestError("Proposal loop mismatch")
    if proposal.get("decision") != "propose_single_bounded_input_projection_gelu_remediation":
        raise OperatorManifestError("Proposal decision mismatch")
    if authorization.get("schema") != "axon_loop28_onnx_operator_remediation_authorization_v1":
        raise OperatorManifestError("Authorization schema mismatch")
    if authorization.get("proposal", {}).get("sha256") != sha256_file(
        _resolve_within(project_root, PROPOSAL)
    ):
        raise OperatorManifestError("Authorization proposal binding mismatch")
    if amendment.get("schema") != ("axon_loop28_onnx_operator_remediation_preflight_amendment_v1"):
        raise OperatorManifestError("Preflight amendment schema mismatch")
    if amendment.get("authorization", {}).get("sha256") != sha256_file(
        _resolve_within(project_root, AUTHORIZATION)
    ):
        raise OperatorManifestError("Amendment authorization binding mismatch")
    if amendment.get("added_exit_decision") != (
        "operator_preflight_exact_tie_fallback_pytorch_native_no_execution"
    ):
        raise OperatorManifestError("Amended exit decision mismatch")
    if parent.get("decision") != "post_fidelity_closure_frozen_localized_negative_no_raw":
        raise OperatorManifestError("Parent closure decision mismatch")

    auditor = _load_auditor(project_root)
    try:
        preflight = auditor.verify_preflight(project_root, PREFLIGHT)
    except Exception as exc:  # noqa: BLE001 - normalize a separately versioned verifier.
        raise OperatorManifestError("Static preflight verification failed") from exc
    if preflight.get("decision") != amendment["added_exit_decision"]:
        raise OperatorManifestError("Preflight decision does not match amendment")
    if preflight.get("execution_audit") != {
        "checkpoint_load_count": 0,
        "onnx_graph_load_count": 0,
        "native_probe_subprocess_count": 0,
        "candidate_graph_count": 0,
        "lease_count": 0,
        "raw_split_cache_heldout_access_count": 0,
        "quality_metric_count": 0,
        "f1_computation_count": 0,
    }:
        raise OperatorManifestError("Preflight execution audit is not a zero-execution closure")
    return {
        "proposal": proposal,
        "authorization": authorization,
        "amendment": amendment,
        "parent": parent,
        "preflight": preflight,
    }


def assert_execution_artifacts_absent(project_root: Path) -> None:
    root = project_root.resolve(strict=True)
    candidate = (root / CANDIDATE_ROOT).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise OperatorManifestError("Candidate root escapes project root") from exc
    if candidate.exists():
        raise OperatorManifestError(f"Forbidden candidate root exists: {CANDIDATE_ROOT}")
    present = [
        path.as_posix()
        for path in FORBIDDEN_EXECUTION_ARTIFACTS
        if (root / path).resolve(strict=False).exists()
    ]
    if present:
        raise OperatorManifestError(f"Forbidden execution artifacts exist: {present}")


def build_implementation_manifest(project_root: Path, *, generated_at_utc: str) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    timestamp = _validate_timestamp(generated_at_utc)
    chain = _verify_chain(root)
    assert_execution_artifacts_absent(root)
    artifacts = [
        _artifact_record(root, name, path, role) for name, path, role in IMPLEMENTATION_ARTIFACTS
    ]
    return {
        "schema": "axon_loop28_onnx_operator_remediation_implementation_manifest_v1",
        "loop_id": LOOP_ID,
        "generated_at_utc": timestamp,
        "contract": {
            "structured_chain_validation": True,
            "duplicate_json_keys_rejected": True,
            "static_only": True,
            "candidate_generation_forbidden": True,
            "output_replace_allowed": False,
            "manifest_self_hashed": False,
        },
        "lineage": {
            "proposal_sha256": sha256_file(_resolve_within(root, PROPOSAL)),
            "authorization_sha256": sha256_file(_resolve_within(root, AUTHORIZATION)),
            "amendment_sha256": sha256_file(_resolve_within(root, AMENDMENT)),
            "preflight_sha256": sha256_file(_resolve_within(root, PREFLIGHT)),
            "parent_closure_sha256": sha256_file(_resolve_within(root, PARENT_CLOSURE)),
            "parent_evidence_sha256": sha256_file(_resolve_within(root, PARENT_EVIDENCE)),
        },
        "proof_summary": {
            "fixture_count": chain["preflight"]["proof"]["fixture_count"],
            "shared_tied_occurrence": chain["preflight"]["proof"]["shared_tied_occurrence"],
            "shared_tied_node_index": chain["preflight"]["proof"]["shared_tied_node_index"],
            "success_branch_reachable": chain["preflight"]["proof"]["formal_proof"][
                "success_branch_reachable"
            ],
        },
        "validation_contract": {
            "focused_pytest": {
                "command": "vnev/Scripts/python.exe -m pytest -q tests/test_remediate_loop28_onnx_operator.py tests/test_build_loop28_onnx_operator_manifest.py",
                "required_pass_count": 8,
            },
            "ruff_check": {
                "command": "vnev/Scripts/python.exe -m ruff check scripts/remediate_loop28_onnx_operator.py scripts/build_loop28_onnx_operator_manifest.py tests/test_remediate_loop28_onnx_operator.py tests/test_build_loop28_onnx_operator_manifest.py",
                "required": True,
            },
            "ruff_format": {
                "command": "vnev/Scripts/python.exe -m ruff format --check scripts/remediate_loop28_onnx_operator.py scripts/build_loop28_onnx_operator_manifest.py tests/test_remediate_loop28_onnx_operator.py tests/test_build_loop28_onnx_operator_manifest.py",
                "required": True,
            },
            "py_compile": {
                "command": "vnev/Scripts/python.exe -m py_compile scripts/remediate_loop28_onnx_operator.py scripts/build_loop28_onnx_operator_manifest.py tests/test_remediate_loop28_onnx_operator.py tests/test_build_loop28_onnx_operator_manifest.py",
                "required": True,
            },
        },
        "artifacts": artifacts,
        "integrity": {
            "artifact_count": len(artifacts),
            "required_artifact_count": len(IMPLEMENTATION_ARTIFACTS),
            "all_required_present": len(artifacts) == len(IMPLEMENTATION_ARTIFACTS),
            "candidate_root_absent": True,
            "forbidden_execution_artifact_count": 0,
        },
        "claim_boundary": chain["preflight"]["claim_boundary"],
        "decision": "static_preflight_implementation_complete_candidate_execution_forbidden",
    }


def verify_implementation_manifest(project_root: Path, output: Path) -> dict[str, Any]:
    path = _resolve_within(project_root, output)
    payload = load_json_strict(path)
    if payload.get("schema") != (
        "axon_loop28_onnx_operator_remediation_implementation_manifest_v1"
    ):
        raise OperatorManifestError("Implementation manifest schema mismatch")
    rebuilt = build_implementation_manifest(
        project_root,
        generated_at_utc=str(payload.get("generated_at_utc") or ""),
    )
    if payload != rebuilt:
        raise OperatorManifestError("Implementation manifest no longer matches its chain")
    return payload


def build_post_manifest(project_root: Path, *, generated_at_utc: str) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    timestamp = _validate_timestamp(generated_at_utc)
    implementation = verify_implementation_manifest(root, DEFAULT_IMPLEMENTATION)
    chain = _verify_chain(root)
    assert_execution_artifacts_absent(root)
    post_inputs = (
        ("implementation_manifest", DEFAULT_IMPLEMENTATION, "verified_static_tooling"),
        ("preflight", PREFLIGHT, "formal_exact_tie_blocker"),
        ("proposal", PROPOSAL, "frozen_proposal"),
        ("authorization", AUTHORIZATION, "a1_authorization"),
        ("preflight_amendment", AMENDMENT, "scope_reduction"),
        ("parent_closure", PARENT_CLOSURE, "immutable_parent_closure"),
        ("parent_evidence", PARENT_EVIDENCE, "immutable_parent_evidence"),
        *FINAL_DOCS,
    )
    artifacts = [_artifact_record(root, name, path, role) for name, path, role in post_inputs]
    return {
        "schema": "axon_loop28_onnx_operator_remediation_post_manifest_v1",
        "loop_id": LOOP_ID,
        "generated_at_utc": timestamp,
        "contract": {
            "structured_chain_validation": True,
            "duplicate_json_keys_rejected": True,
            "candidate_generation_forbidden": True,
            "manifest_self_hashed": False,
            "output_replace_allowed": False,
        },
        "lineage": {
            "parent_closure_sha256": implementation["lineage"]["parent_closure_sha256"],
            "parent_evidence_sha256": implementation["lineage"]["parent_evidence_sha256"],
            "implementation_manifest_sha256": sha256_file(
                _resolve_within(root, DEFAULT_IMPLEMENTATION)
            ),
            "preflight_sha256": implementation["lineage"]["preflight_sha256"],
        },
        "outcome": {
            "fixture_count": chain["preflight"]["proof"]["fixture_count"],
            "shared_exact_tie_proven": True,
            "strict_margin_success_branch_reachable": False,
            "candidate_graph_count": 0,
            "lease_count": 0,
            "checkpoint_or_onnx_load_count": 0,
            "native_probe_subprocess_count": 0,
            "raw_split_cache_heldout_access_count": 0,
            "quality_metric_count": 0,
            "f1_computation_count": 0,
            "next_route": "pytorch_compatible_native_runtime",
        },
        "artifacts": artifacts,
        "integrity": {
            "artifact_count": len(artifacts),
            "required_artifact_count": len(post_inputs),
            "all_required_present": len(artifacts) == len(post_inputs),
            "candidate_root_absent": True,
            "forbidden_execution_artifact_count": 0,
        },
        "claim_boundary": chain["preflight"]["claim_boundary"],
        "decision": "post_operator_preflight_exact_tie_fallback_pytorch_native_no_execution",
    }


def verify_post_manifest(project_root: Path, output: Path) -> dict[str, Any]:
    path = _resolve_within(project_root, output)
    payload = load_json_strict(path)
    if payload.get("schema") != "axon_loop28_onnx_operator_remediation_post_manifest_v1":
        raise OperatorManifestError("Post manifest schema mismatch")
    rebuilt = build_post_manifest(
        project_root,
        generated_at_utc=str(payload.get("generated_at_utc") or ""),
    )
    if payload != rebuilt:
        raise OperatorManifestError("Post manifest no longer matches its chain")
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
        raise OperatorManifestError(f"Output already exists: {path}") from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("implementation", "post"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--generated-at-utc")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = args.project_root.resolve(strict=True)
    default_output = DEFAULT_IMPLEMENTATION if args.mode == "implementation" else DEFAULT_POST
    output = args.output or default_output
    if args.verify:
        payload = (
            verify_implementation_manifest(root, output)
            if args.mode == "implementation"
            else verify_post_manifest(root, output)
        )
    else:
        if not args.generated_at_utc:
            raise OperatorManifestError("--generated-at-utc is required when building")
        payload = (
            build_implementation_manifest(root, generated_at_utc=args.generated_at_utc)
            if args.mode == "implementation"
            else build_post_manifest(root, generated_at_utc=args.generated_at_utc)
        )
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
