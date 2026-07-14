#!/usr/bin/env python3
"""Build the hash-only Loop28 parity remediation implementation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import build_loop28_parity_diagnostic_manifest as diagnostic_closure
from build_loop28_parity_post_diagnostic_manifest import (
    ArtifactSpec,
    ManifestError,
    _sha256_stable_file,
    _validate_generated_at,
    write_manifest_exclusive,
)

SCHEMA = "axon_loop28_parity_remediation_implementation_manifest_v1"
LOOP_ID = "p0_loop28_parity_remediation_001"
DEFAULT_OUTPUT = Path(
    "manifests/roadmap_9997/p0_loop28_parity_remediation/implementation_manifest.json"
)
BLOCKED_EVIDENCE = Path(
    "manifests/roadmap_9997/p0_loop28_parity_remediation/synthetic_pre_run_blocked.json"
)
BLOCKED_EVIDENCE_SCHEMA = "axon_loop28_parity_remediation_synthetic_pre_run_blocked_v1"
BLOCKED_DECISION = "block_train_rerun_input_dependent_onnx_base_drift"


def _spec(
    name: str,
    role: str,
    path: str,
    expected_sha256: Optional[str] = None,
) -> ArtifactSpec:
    return ArtifactSpec(name, role, Path(path), expected_sha256)


REUSED_DIAGNOSTIC_PYTHON_ROLES = frozenset(
    {
        "diagnostic_manifest_source",
        "diagnostic_manifest_test",
        "diagnostic_source",
        "diagnostic_test",
        "diagnostic_python_dependency",
        "diagnostic_python_dependency_test",
        "python_runtime_source",
        "python_runtime_test",
        "python_dependency_contract",
    }
)


def _reused_diagnostic_python_closure() -> tuple[ArtifactSpec, ...]:
    return tuple(
        _spec(spec.name, spec.role, spec.path.as_posix())
        for spec in diagnostic_closure.DEFAULT_ARTIFACTS
        if spec.role in REUSED_DIAGNOSTIC_PYTHON_ROLES
    )


def _validate_artifact_inventory(artifacts: Sequence[ArtifactSpec]) -> None:
    names: set[str] = set()
    paths: set[str] = set()
    for spec in artifacts:
        path_text = spec.path.as_posix()
        if not spec.name or not spec.role or spec.path.is_absolute() or ".." in spec.path.parts:
            raise ManifestError(f"Invalid remediation artifact specification: {spec.name}")
        if spec.name in names or path_text.casefold() in paths:
            raise ManifestError(f"Duplicate remediation artifact specification: {spec.name}")
        if spec.expected_sha256 is not None and (
            len(spec.expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in spec.expected_sha256)
        ):
            raise ManifestError(f"Invalid expected SHA-256: {spec.name}")
        names.add(spec.name)
        paths.add(path_text.casefold())


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ManifestError(f"Duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def _reject_blocked_remediation(project_root: Path) -> None:
    blocked_path = project_root.resolve(strict=True) / BLOCKED_EVIDENCE
    if not blocked_path.exists():
        return
    try:
        payload = json.loads(
            blocked_path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError("Synthetic pre-run blocked evidence is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema") != BLOCKED_EVIDENCE_SCHEMA:
        raise ManifestError("Synthetic pre-run blocked evidence schema mismatch")
    if payload.get("loop_id") != LOOP_ID or payload.get("decision") != BLOCKED_DECISION:
        raise ManifestError("Synthetic pre-run blocked evidence decision mismatch")
    raise ManifestError(
        "Synthetic pre-run blocked evidence forbids an implementation manifest and raw rerun"
    )


ARTIFACTS = (
    _spec(
        "post_diagnostic_manifest",
        "immutable_parent_closure",
        "manifests/roadmap_9997/p0_loop28_parity_diagnostic/post_diagnostic_manifest.json",
        "9f57dee431d61a1a1ebc99f64ed9bcb9f65804fca8426c748d51b468e81e6d31",
    ),
    _spec(
        "diagnostic_receipt",
        "immutable_localization_evidence",
        "reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.final.json",
        "de8f0c5885df08646298f67f59b5427696252f7cb921d84eeb6527bef6878bc7",
    ),
    _spec(
        "proposal",
        "remediation_preregistration",
        "manifests/roadmap_9997/p0_loop28_parity_remediation/proposal.json",
        "d6c1451e73b5de5bf6aa737cf188d71cda338945a9434d64d74b265cd6ecfeeb",
    ),
    _spec(
        "authorization",
        "scoped_change_authorization",
        "manifests/roadmap_9997/p0_loop28_parity_remediation/authorization.json",
        "8304abe401bb5714a2b0edba40ea1866659f7cce05c9182ba5f359857fbe54c5",
    ),
    _spec(
        "preflight",
        "pre_implementation_audit",
        "manifests/roadmap_9997/p0_loop28_parity_remediation/preflight.json",
        "e6ed40e2aa3af35cb86acfb432b7aa46213a6249e35f4ae80bc7d1c3b21cd087",
    ),
    _spec(
        "synthetic_discovery",
        "pre_raw_synthetic_parity_evidence",
        "manifests/roadmap_9997/p0_loop28_parity_remediation/synthetic_discovery.json",
        "36925500a2a3c866effea40474083d3c0b813c72cba1227f11836ebc685a5c56",
    ),
    _spec(
        "native_runtime_source",
        "remediated_native_implementation",
        "tools/axon_onnx_dll/src/axon_onnx_predict.cpp",
    ),
    _spec(
        "native_public_header",
        "native_abi",
        "tools/axon_onnx_dll/include/axon_onnx_predict.h",
    ),
    _spec(
        "native_selftest_source",
        "native_test_driver",
        "tools/axon_onnx_dll/examples/axon_onnx_selftest.cpp",
    ),
    _spec("native_cmake", "native_build_contract", "tools/axon_onnx_dll/CMakeLists.txt"),
    *_reused_diagnostic_python_closure(),
    _spec(
        "python_schema_names",
        "feature_index_contract",
        "src/kvd_features/schema_names.py",
    ),
    _spec(
        "remediation_runner",
        "remediation_authorization_and_execution_boundary",
        "scripts/remediate_loop28_parity.py",
    ),
    _spec(
        "remediation_runner_tests",
        "remediation_runner_contract_tests",
        "tests/test_remediate_loop28_parity.py",
    ),
    _spec(
        "native_parity_source_tests",
        "synthetic_and_source_tests",
        "tests/test_native_loop28_parity_source.py",
    ),
    _spec(
        "manifest_builder",
        "hash_only_closure_builder",
        "scripts/build_loop28_parity_remediation_manifest.py",
    ),
    _spec(
        "manifest_builder_tests",
        "manifest_contract_tests",
        "tests/test_build_loop28_parity_remediation_manifest.py",
    ),
    _spec(
        "post_remediation_builder",
        "post_run_structured_closure_builder",
        "scripts/build_loop28_parity_post_remediation_manifest.py",
    ),
    _spec(
        "post_remediation_builder_tests",
        "post_run_structured_closure_tests",
        "tests/test_build_loop28_parity_post_remediation_manifest.py",
    ),
    _spec(
        "post_diagnostic_builder",
        "parent_closure_builder",
        "scripts/build_loop28_parity_post_diagnostic_manifest.py",
    ),
    _spec(
        "native_dll",
        "clean_release_binary",
        "tools/axon_onnx_dll/build/bin/Release/axon_onnx_predict.dll",
    ),
    _spec(
        "native_selftest",
        "clean_release_selftest",
        "tools/axon_onnx_dll/build/bin/Release/axon_onnx_selftest.exe",
    ),
    _spec(
        "native_onnxruntime",
        "frozen_native_runtime",
        "tools/axon_onnx_dll/build/bin/Release/onnxruntime.dll",
        "b95efb2113b603bbbf3f191061c5516a871ed546893c820e4f3b7b6c358dbf2a",
    ),
    _spec(
        "python_checkpoint",
        "frozen_python_base_model",
        "models/random_20w_8192/best_model.pt",
        "96a1b1ece41dd7dd9142a0f7f4330da3a7938a26cca8b01e0e7c7a1074e5e3a4",
    ),
    _spec(
        "python_stage2",
        "frozen_python_stage2",
        "reports/random_20w_split/stage2_loop28_content_pe_valonly/stage2_selected_model.pkl",
        "34d76eaf015e6750ca080f80a8ae528e8e10f2261ff9f1c001dc2f243672a5c2",
    ),
    _spec(
        "python_stage2_metadata",
        "frozen_stage2_provenance",
        "manifests/roadmap_9997/p0_raw_replay/loop28_stage2.metadata.json",
        "0a24dcf1ec5bab43d6afd004df006e4c8008bace854265cdd5f8cc4c79ba80f4",
    ),
    _spec(
        "pickle_allowlist",
        "frozen_pickle_trust_policy",
        "manifests/roadmap_9997/p0_raw_replay/pickle_sha256_allowlist.json",
        "744ec3792d88721e861a412acaaaa768d9d0ae99f05c41b2c3b49d78dd3ad8ee",
    ),
    _spec(
        "native_onnx",
        "frozen_native_base_model",
        "models/random_20w_8192/axon_loop28_base.onnx",
        "3199b158fc8f7e3a53a516b2681aef8b5d5aa4a210baf66152fded72a3ff07f4",
    ),
    _spec(
        "native_onnx_data",
        "frozen_native_base_weights",
        "models/random_20w_8192/axon_loop28_base.onnx.data",
        "4865d52d861d780627ca9aea4b16f83d8c2df62dd5b2136217d1e42547b8c7fa",
    ),
    _spec(
        "native_stage2",
        "frozen_native_stage2",
        "models/random_20w_8192/loop28_stage2_hgb.json",
        "c2c0cb0f39d12892891f9949e6f765e03fb1188f8fa3f3e574c0c4f73c63c648",
    ),
    _spec(
        "stage2_exporter",
        "hgb_json_semantics_reference",
        "scripts/export_stage2_hgb_json.py",
    ),
    _spec(
        "onnx_exporter",
        "checkpoint_to_onnx_lineage",
        "scripts/export_onnx_model.py",
    ),
)


def build_manifest(
    project_root: Path,
    *,
    generated_at_utc: str,
    artifacts: Sequence[ArtifactSpec] = ARTIFACTS,
) -> dict:
    generated_at_utc = _validate_generated_at(generated_at_utc)
    _reject_blocked_remediation(project_root)
    _validate_artifact_inventory(artifacts)
    rows = []
    blockers = []
    expected_count = sum(spec.expected_sha256 is not None for spec in artifacts)
    verified_expected = 0
    for spec in artifacts:
        try:
            size_bytes, sha256 = _sha256_stable_file(project_root, spec.path)
        except (FileNotFoundError, ManifestError, OSError) as exc:
            blockers.append({"artifact": spec.name, "reason": str(exc)})
            continue
        expected_match = None
        if spec.expected_sha256 is not None:
            expected_match = sha256 == spec.expected_sha256
            if expected_match:
                verified_expected += 1
            else:
                blockers.append(
                    {
                        "artifact": spec.name,
                        "reason": "predeclared_sha256_mismatch",
                        "expected_sha256": spec.expected_sha256,
                        "actual_sha256": sha256,
                    }
                )
        rows.append(
            {
                "name": spec.name,
                "role": spec.role,
                "path": spec.path.as_posix(),
                "required": True,
                "expected_sha256": spec.expected_sha256,
                "exists": True,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "expected_sha256_match": expected_match,
            }
        )

    return {
        "schema": SCHEMA,
        "loop_id": LOOP_ID,
        "generated_at_utc": generated_at_utc,
        "contract": {
            "operation": "opaque_file_stat_and_streaming_sha256_only",
            "bound_artifact_payloads_parsed": False,
            "manifest_self_hashed": False,
            "output_replace_allowed": False,
            "declared_runtime": {
                "verification_status": "declared_not_verified_by_manifest_builder",
                "python": "3.14.4",
                "numpy": "2.4.4",
                "pefile": "2024.8.26",
                "onnxruntime": "1.24.4",
                "graph_optimization": "ORT_DISABLE_ALL",
                "unverified_components": [
                    "python_executable",
                    "torch",
                    "scikit_learn",
                    "msvc_toolset",
                    "cmake_generator",
                ],
            },
        },
        "claim_scope": {
            "implementation_hash_closure_only": True,
            "synthetic_discovery_evidence_bound": True,
            "synthetic_parity_reverified_by_manifest_builder": False,
            "raw_execution_performed": False,
            "quality_claim_allowed": False,
            "parity_claim_allowed": False,
            "certification_claim_allowed": False,
            "productization_performance_gate_pending": True,
        },
        "artifacts": rows,
        "integrity": {
            "artifact_count": len(rows),
            "required_artifact_count": len(artifacts),
            "present_required_artifact_count": len(rows),
            "predeclared_sha256_count": expected_count,
            "verified_predeclared_sha256_count": verified_expected,
            "blockers": blockers,
        },
        "decision": (
            "implementation_hash_closure_verified_run_authorization_pending"
            if not blockers and len(rows) == len(artifacts)
            else "implementation_manifest_blocked"
        ),
    }


def _resolve_project_path(project_root: Path, requested_path: Path, *, purpose: str) -> Path:
    root = project_root.resolve(strict=True)
    if requested_path.is_absolute() or ".." in requested_path.parts:
        raise ManifestError(f"{purpose} must be a canonical project-relative path")
    resolved = (root / requested_path).resolve()
    if root != resolved and root not in resolved.parents:
        raise ManifestError(f"{purpose} escapes project root")
    return resolved


def resolve_fixed_output(project_root: Path, requested_path: Path) -> Path:
    resolved = _resolve_project_path(
        project_root,
        requested_path,
        purpose="Remediation implementation manifest output",
    )
    frozen = _resolve_project_path(
        project_root,
        DEFAULT_OUTPUT,
        purpose="Frozen remediation implementation manifest output",
    )
    if resolved != frozen:
        raise ManifestError("Remediation implementation manifest output path is not fixed")
    return resolved


def verify_manifest(
    project_root: Path,
    manifest_path: Path,
    *,
    artifacts: Sequence[ArtifactSpec] = ARTIFACTS,
) -> dict:
    resolved_manifest = _resolve_project_path(
        project_root,
        manifest_path,
        purpose="Requested remediation implementation manifest",
    )
    payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    rebuilt = build_manifest(
        project_root,
        generated_at_utc=str(payload.get("generated_at_utc", "")),
        artifacts=artifacts,
    )
    if payload != rebuilt:
        raise ManifestError("Manifest does not match the current remediation closure")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-at-utc")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve(strict=True)
    output_path = resolve_fixed_output(project_root, args.output)
    if args.verify:
        manifest = verify_manifest(project_root, args.output)
    else:
        if not args.generated_at_utc:
            raise ManifestError("--generated-at-utc is required when building")
        manifest = build_manifest(project_root, generated_at_utc=args.generated_at_utc)
        if manifest["integrity"]["blockers"]:
            raise ManifestError("Remediation implementation closure contains blockers")
        write_manifest_exclusive(output_path, manifest)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "artifact_count": manifest["integrity"]["artifact_count"],
                "blocker_count": len(manifest["integrity"]["blockers"]),
                "decision": manifest["decision"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
