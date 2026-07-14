#!/usr/bin/env python3
"""Build and verify fail-closed Loop28 ONNX fidelity manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_ID = "p0_loop28_onnx_fidelity_001"

PROPOSAL_SCHEMA = "axon_loop28_onnx_fidelity_proposal_v1"
AUTHORIZATION_SCHEMA = "axon_loop28_onnx_fidelity_authorization_v1"
PREFLIGHT_SCHEMA = "axon_loop28_onnx_fidelity_preflight_v1"
IMPLEMENTATION_SCHEMA = "axon_loop28_onnx_fidelity_implementation_manifest_v1"
LOCALIZATION_AUTHORIZATION_SCHEMA = "axon_loop28_onnx_fidelity_localization_authorization_v1"
LOCALIZATION_LEASE_SCHEMA = "axon_loop28_onnx_fidelity_localization_lease_v1"
LOCALIZATION_EVIDENCE_SCHEMA = "axon_loop28_onnx_fidelity_localization_evidence_v1"
POST_SCHEMA = "axon_loop28_onnx_fidelity_post_manifest_v1"
PARENT_SCHEMA = "axon_loop28_parity_post_remediation_manifest_v1"

PROPOSAL = Path("manifests/roadmap_9997/p0_loop28_onnx_fidelity/proposal.json")
AUTHORIZATION = Path("manifests/roadmap_9997/p0_loop28_onnx_fidelity/authorization.json")
PREFLIGHT = Path("manifests/roadmap_9997/p0_loop28_onnx_fidelity/preflight.json")
PARENT_CLOSURE = Path(
    "manifests/roadmap_9997/p0_loop28_parity_remediation/post_remediation_manifest.json"
)
IMPLEMENTATION_OUTPUT = Path(
    "manifests/roadmap_9997/p0_loop28_onnx_fidelity/implementation_manifest.json"
)
LOCALIZATION_AUTHORIZATION = Path(
    "manifests/roadmap_9997/p0_loop28_onnx_fidelity/localization_authorization.json"
)
LOCALIZATION_LEASE_PENDING = Path(
    "manifests/roadmap_9997/p0_loop28_onnx_fidelity/localization_lease.json"
)
LOCALIZATION_LEASE = Path(
    "manifests/roadmap_9997/p0_loop28_onnx_fidelity/localization_lease.final.json"
)
LOCALIZATION_EVIDENCE = Path(
    "reports/roadmap_9997/p0_loop28_onnx_fidelity/localization_evidence.final.json"
)
GOAL_DELTA = Path("reports/roadmap_9997/p0_loop28_onnx_fidelity/goal_delta.final.md")
JOURNAL_ENTRY = Path("reports/roadmap_9997/p0_loop28_onnx_fidelity/journal_entry.final.md")
FINAL_STATUS = Path("reports/roadmap_9997/p0_loop28_onnx_fidelity/status.final.md")
POST_OUTPUT = Path("manifests/roadmap_9997/p0_loop28_onnx_fidelity/post_manifest.json")

DIAGNOSTIC_SOURCE = Path("scripts/diagnose_loop28_onnx_fidelity.py")
DIAGNOSTIC_TEST = Path("tests/test_diagnose_loop28_onnx_fidelity.py")
PROBE_SOURCE = Path("tools/axon_onnx_fidelity/src/onnx_fidelity_probe.cpp")
PROBE_CMAKE = Path("tools/axon_onnx_fidelity/CMakeLists.txt")
PROBE_BINARY = Path("tools/axon_onnx_fidelity/build/bin/Release/axon_onnx_fidelity_probe.exe")
PROBE_RUNTIME = Path("tools/axon_onnx_fidelity/build/bin/Release/onnxruntime.dll")
BUILDER_SOURCE = Path("scripts/build_loop28_onnx_fidelity_manifest.py")
BUILDER_TEST = Path("tests/test_build_loop28_onnx_fidelity_manifest.py")

BASELINE_PATHS = {
    "checkpoint": Path("models/random_20w_8192/best_model.pt"),
    "onnx_graph": Path("models/random_20w_8192/axon_loop28_base.onnx"),
    "onnx_data": Path("models/random_20w_8192/axon_loop28_base.onnx.data"),
    "onnx_metadata": Path("models/random_20w_8192/axon_loop28_base.onnx.json"),
    "onnxruntime": Path("tools/axon_onnx_dll/build/bin/Release/onnxruntime.dll"),
    "exporter": Path("scripts/export_onnx_model.py"),
    "fixture_contract": Path("tests/test_native_loop28_parity_source.py"),
    "model_source": Path("src/model.py"),
    "mhdsra2_source": Path("src/dsra/mhdsra2/improved_dsra_mha.py"),
}

PARENT_DECISION = "synthetic_pre_run_blocked_closure_frozen_no_raw_execution"
PROPOSAL_DECISION = "propose_synthetic_only_intermediate_activation_localization"
AUTHORIZATION_DECISION = "authorize_synthetic_fidelity_tooling_only"
PREFLIGHT_DECISION = "synthetic_fidelity_tooling_implementation_ready"
IMPLEMENTATION_DECISION = "implementation_manifest_complete"
LOCALIZATION_AUTHORIZATION_DECISION = "authorize_synthetic_localization_run"
LEASE_CONSUMED_STATUS = "consumed_before_execution"

EXIT_DECISIONS = (
    "localized_negative_no_raw",
    "synthetic_fidelity_verified_raw_still_requires_new_authorization",
    "invalid_positive_control_or_lineage_drift",
    "budget_exhausted_no_claim",
)
POST_DECISIONS = {
    decision: f"post_fidelity_closure_frozen_{decision}" for decision in EXIT_DECISIONS
}

IMPLEMENTATION_FIXED_ARTIFACTS = (
    ("proposal", "fidelity_proposal", PROPOSAL),
    ("authorization", "fidelity_authorization", AUTHORIZATION),
    ("preflight", "fidelity_preflight", PREFLIGHT),
    ("parent_closure", "immutable_parent_closure", PARENT_CLOSURE),
    ("diagnostic_source", "synthetic_localization_runner", DIAGNOSTIC_SOURCE),
    ("diagnostic_test", "synthetic_localization_tests", DIAGNOSTIC_TEST),
    ("probe_source", "standalone_cpp_probe", PROBE_SOURCE),
    ("probe_cmake", "standalone_cpp_probe_build", PROBE_CMAKE),
    ("probe_binary", "standalone_cpp_probe_release_binary", PROBE_BINARY),
    ("probe_runtime", "standalone_cpp_probe_frozen_runtime", PROBE_RUNTIME),
    ("manifest_builder", "fidelity_manifest_builder", BUILDER_SOURCE),
    ("manifest_builder_test", "fidelity_manifest_builder_tests", BUILDER_TEST),
)

IMPLEMENTATION_FORBIDDEN_PATHS = {
    IMPLEMENTATION_OUTPUT.as_posix(),
    LOCALIZATION_AUTHORIZATION.as_posix(),
    LOCALIZATION_LEASE_PENDING.as_posix(),
    LOCALIZATION_LEASE.as_posix(),
    LOCALIZATION_EVIDENCE.as_posix(),
    GOAL_DELTA.as_posix(),
    JOURNAL_ENTRY.as_posix(),
    FINAL_STATUS.as_posix(),
    POST_OUTPUT.as_posix(),
}

EXPECTED_SCOPE_AUDIT = {
    "synthetic_only": True,
    "dataset_raw_accessed": False,
    "split_metadata_accessed": False,
    "cache_rows_accessed": False,
    "heldout_accessed": False,
    "prediction_or_metric_payload_accessed": False,
    "training_or_fitting_performed": False,
    "quality_metric_computed": False,
}
EXPECTED_CLAIM_BOUNDARY = {
    "synthetic_cross_runtime_localization_only": True,
    "raw_rerun_allowed": False,
    "population_parity_claim_allowed": False,
    "quality_claim_allowed": False,
    "native_loop28_ready_claim_allowed": False,
    "native_loop151_ready_claim_allowed": False,
    "certification_claim_allowed": False,
}
EXPECTED_RUN_CLAIM_SCOPE = {
    "synthetic_only": True,
    "raw_split_heldout_access_allowed": False,
    "training_or_fitting_allowed": False,
    "quality_metric_allowed": False,
    "quality_claim_allowed": False,
    "certification_claim_allowed": False,
}


class FidelityManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    role: str
    path: Path
    expected_sha256: Optional[str] = None


def _is_sha256(value: object) -> bool:
    normalized = str(value or "").casefold()
    return len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise FidelityManifestError(f"Duplicate JSON key is forbidden: {key}")
        payload[key] = value
    return payload


def _validate_generated_at(value: str) -> str:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise FidelityManifestError("Timestamp must be RFC3339") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FidelityManifestError("Timestamp must include a timezone")
    return value


def _validate_relative_path(path: Path, *, purpose: str) -> None:
    if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
        raise FidelityManifestError(f"{purpose} must be a canonical project-relative path")


def _resolve_project_path(
    project_root: Path,
    relative_path: Path,
    *,
    purpose: str,
    must_exist: bool,
) -> Path:
    root = project_root.resolve(strict=True)
    _validate_relative_path(relative_path, purpose=purpose)
    resolved = (root / relative_path).resolve(strict=must_exist)
    if root != resolved and root not in resolved.parents:
        raise FidelityManifestError(f"{purpose} escapes project root")
    return resolved


def _stable_file_record(project_root: Path, spec: ArtifactSpec) -> dict:
    root = project_root.resolve(strict=True)
    _validate_relative_path(spec.path, purpose=f"Artifact {spec.name}")
    path = root / spec.path
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FidelityManifestError(f"Artifact must be a regular non-symlink file: {spec.path}")
    resolved = path.resolve(strict=True)
    if root != resolved and root not in resolved.parents:
        raise FidelityManifestError(f"Artifact {spec.name} escapes project root")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.lstat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise FidelityManifestError(f"Artifact changed while hashing: {spec.path}")
    sha256 = digest.hexdigest()
    if spec.expected_sha256 is not None and sha256 != spec.expected_sha256:
        raise FidelityManifestError(
            f"Artifact SHA-256 mismatch: {spec.name}; expected {spec.expected_sha256}, got {sha256}"
        )
    return {
        "name": spec.name,
        "role": spec.role,
        "path": spec.path.as_posix(),
        "size_bytes": before.st_size,
        "sha256": sha256,
        "expected_sha256": spec.expected_sha256,
        "expected_sha256_match": True if spec.expected_sha256 is not None else None,
    }


def _read_json(
    project_root: Path,
    relative_path: Path,
    *,
    expected_schema: str,
) -> tuple[dict, dict]:
    record = _stable_file_record(
        project_root,
        ArtifactSpec(relative_path.stem, "structured_json", relative_path),
    )
    root = project_root.resolve(strict=True)
    path = root / relative_path
    before = path.lstat()
    try:
        encoded = path.read_bytes()
        after = path.lstat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise FidelityManifestError(f"JSON artifact changed while reading: {relative_path}")
        if (
            len(encoded) != record["size_bytes"]
            or hashlib.sha256(encoded).hexdigest() != record["sha256"]
        ):
            raise FidelityManifestError(f"JSON artifact changed after hashing: {relative_path}")
        payload = json.loads(
            encoded.decode("utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FidelityManifestError(f"Invalid JSON artifact: {relative_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise FidelityManifestError(f"JSON schema mismatch: {relative_path}")
    return payload, record


def _validate_inventory(artifacts: Sequence[ArtifactSpec], *, forbidden_paths: set[str]) -> None:
    names: set[str] = set()
    paths: set[str] = set()
    for spec in artifacts:
        _validate_relative_path(spec.path, purpose=f"Artifact {spec.name}")
        path_text = spec.path.as_posix()
        if not spec.name or not spec.role:
            raise FidelityManifestError("Artifact name and role are required")
        if spec.name in names or path_text.casefold() in paths:
            raise FidelityManifestError(f"Duplicate artifact specification: {spec.name}")
        if path_text in forbidden_paths:
            raise FidelityManifestError(
                f"Artifact inventory contains a future or cyclic output: {path_text}"
            )
        if spec.expected_sha256 is not None and not _is_sha256(spec.expected_sha256):
            raise FidelityManifestError(f"Invalid expected SHA-256: {spec.name}")
        names.add(spec.name)
        paths.add(path_text.casefold())


def _require_text_fragments(value: object, fragments: Sequence[str], *, purpose: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise FidelityManifestError(f"{purpose} must be a string list")
    normalized = "\n".join(value).casefold()
    missing = [fragment for fragment in fragments if fragment.casefold() not in normalized]
    if missing:
        raise FidelityManifestError(f"{purpose} is missing fail-closed terms: {', '.join(missing)}")


def _parent_artifact_rows(project_root: Path, parent: Mapping[str, object]) -> None:
    if parent.get("decision") != PARENT_DECISION:
        raise FidelityManifestError("Parent closure decision drifted")
    artifacts = parent.get("artifacts")
    integrity = parent.get("integrity")
    if not isinstance(artifacts, list) or not artifacts or not isinstance(integrity, Mapping):
        raise FidelityManifestError("Parent closure artifact inventory is invalid")
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise FidelityManifestError("Parent closure artifact row is invalid")
        name = row.get("name")
        path_text = row.get("path")
        size_bytes = row.get("size_bytes")
        sha256 = row.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or name in seen_names
            or not isinstance(path_text, str)
            or not path_text
            or path_text.casefold() in seen_paths
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not _is_sha256(sha256)
        ):
            raise FidelityManifestError("Parent closure artifact row is invalid")
        current = _stable_file_record(
            project_root,
            ArtifactSpec(f"parent_{name}", "parent_artifact", Path(path_text), str(sha256)),
        )
        if current["size_bytes"] != size_bytes:
            raise FidelityManifestError(f"Parent closure artifact size drifted: {name}")
        seen_names.add(name)
        seen_paths.add(path_text.casefold())
    expected_integrity = {
        "artifact_count": len(artifacts),
        "required_artifact_count": len(artifacts),
        "present_required_artifact_count": len(artifacts),
    }
    if (
        any(integrity.get(key) != value for key, value in expected_integrity.items())
        or integrity.get("blockers") != []
    ):
        raise FidelityManifestError("Parent closure integrity summary drifted")


def _path_matches_allowlist(path: str, allowlist: Sequence[str]) -> bool:
    for pattern in allowlist:
        if pattern.endswith("/**") and path.startswith(pattern[:-2]):
            return True
        if "*" not in pattern and path == pattern:
            return True
    return False


def _baseline_specs(
    authorization: Mapping[str, object],
    proposal: Mapping[str, object],
) -> tuple[ArtifactSpec, ...]:
    baseline = authorization.get("baseline_artifacts")
    if not isinstance(baseline, Mapping) or not set(BASELINE_PATHS) <= set(baseline):
        raise FidelityManifestError("Authorization baseline artifact allowlist drifted")
    read_allowlist = proposal.get("read_allowlist")
    if not isinstance(read_allowlist, list) or any(
        not isinstance(pattern, str) for pattern in read_allowlist
    ):
        raise FidelityManifestError("Proposal read allowlist is invalid")
    specs = []
    for name, raw_record in baseline.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(raw_record, Mapping)
            or set(raw_record) != {"path", "sha256"}
            or not isinstance(raw_record.get("path"), str)
            or not _is_sha256(raw_record.get("sha256"))
        ):
            raise FidelityManifestError(f"Authorization baseline record is invalid: {name}")
        path_text = str(raw_record["path"])
        if name in BASELINE_PATHS and path_text != BASELINE_PATHS[name].as_posix():
            raise FidelityManifestError(f"Authorization baseline path drifted: {name}")
        if not _path_matches_allowlist(path_text, read_allowlist):
            raise FidelityManifestError(f"Authorization baseline path is not read-allowed: {name}")
        specs.append(
            ArtifactSpec(
                f"baseline_{name}",
                "frozen_baseline_artifact",
                Path(path_text),
                str(raw_record["sha256"]),
            )
        )
    return tuple(specs)


def _validate_governance(project_root: Path) -> dict:
    proposal, proposal_record = _read_json(
        project_root,
        PROPOSAL,
        expected_schema=PROPOSAL_SCHEMA,
    )
    authorization, authorization_record = _read_json(
        project_root,
        AUTHORIZATION,
        expected_schema=AUTHORIZATION_SCHEMA,
    )
    preflight, preflight_record = _read_json(
        project_root,
        PREFLIGHT,
        expected_schema=PREFLIGHT_SCHEMA,
    )
    parent, parent_record = _read_json(
        project_root,
        PARENT_CLOSURE,
        expected_schema=PARENT_SCHEMA,
    )
    if any(payload.get("loop_id") != LOOP_ID for payload in (proposal, authorization, preflight)):
        raise FidelityManifestError("Governance loop identity drifted")
    if proposal.get("decision") != PROPOSAL_DECISION or set(
        proposal.get("exit_decisions", [])
    ) != set(EXIT_DECISIONS):
        raise FidelityManifestError("Proposal decision contract drifted")
    proposal_parent = proposal.get("parent_closure")
    expected_proposal_parent = {
        "path": PARENT_CLOSURE.as_posix(),
        "sha256": parent_record["sha256"],
        "decision": PARENT_DECISION,
    }
    if proposal_parent != expected_proposal_parent:
        raise FidelityManifestError("Proposal parent closure binding drifted")
    _require_text_fragments(
        proposal.get("forbidden"),
        ("raw", "split", "heldout", "training", "goal.md", "1e-6"),
        purpose="Proposal forbidden policy",
    )
    proposal_claim = proposal.get("claim_boundary")
    if not isinstance(proposal_claim, Mapping) or any(
        proposal_claim.get(field) is not False
        for field in (
            "population_parity_claim_allowed",
            "quality_claim_allowed",
            "native_loop28_ready_claim_allowed",
            "native_loop151_ready_claim_allowed",
            "raw_rerun_allowed",
            "certification_claim_allowed",
        )
    ):
        raise FidelityManifestError("Proposal claim boundary drifted")

    if authorization.get("decision") != AUTHORIZATION_DECISION:
        raise FidelityManifestError("Authorization decision drifted")
    if authorization.get("proposal") != {
        "path": PROPOSAL.as_posix(),
        "sha256": proposal_record["sha256"],
    } or authorization.get("parent_closure") != {
        "path": PARENT_CLOSURE.as_posix(),
        "sha256": parent_record["sha256"],
    }:
        raise FidelityManifestError("Authorization governance binding drifted")
    if authorization.get("execution_requires_separate_localization_authorization") is not True:
        raise FidelityManifestError("Authorization execution boundary drifted")
    _require_text_fragments(
        authorization.get("never_authorized"),
        ("raw", "split", "heldout", "training", "goal.md", "gpu", "f1"),
        purpose="Authorization permanent prohibitions",
    )
    _require_text_fragments(
        authorization.get("not_authorized_before_localization_run_authorization"),
        ("checkpoint", "onnx graph", "lease", "localization evidence"),
        purpose="Authorization pre-run prohibitions",
    )
    required_edit_paths = {
        PROPOSAL.as_posix(),
        AUTHORIZATION.as_posix(),
        PREFLIGHT.as_posix(),
        IMPLEMENTATION_OUTPUT.as_posix(),
        LOCALIZATION_AUTHORIZATION.as_posix(),
        LOCALIZATION_LEASE_PENDING.as_posix(),
        LOCALIZATION_LEASE.as_posix(),
        POST_OUTPUT.as_posix(),
        DIAGNOSTIC_SOURCE.as_posix(),
        DIAGNOSTIC_TEST.as_posix(),
        BUILDER_SOURCE.as_posix(),
        BUILDER_TEST.as_posix(),
        "tools/axon_onnx_fidelity/**",
        "reports/roadmap_9997/p0_loop28_onnx_fidelity/**",
    }
    authorized_edit_paths = authorization.get("authorized_edit_paths")
    if not isinstance(authorized_edit_paths, list) or not required_edit_paths <= set(
        authorized_edit_paths
    ):
        raise FidelityManifestError("Authorization edit allowlist drifted")
    forbidden_mutations = {
        "goal.md",
        "docs/ml_improvement_recommendations.md",
        "reports/hard_family_finetune/experiment_journal.md",
        PARENT_CLOSURE.as_posix(),
    }
    if forbidden_mutations.intersection(authorized_edit_paths):
        raise FidelityManifestError("Authorization edit allowlist includes a parent-bound artifact")
    output_policy = authorization.get("output_policy")
    if not isinstance(output_policy, Mapping) or any(
        output_policy.get(field) is not True
        for field in (
            "governance_outputs_exclusive_create",
            "temporary_probe_outputs_confined",
            "temporary_probe_outputs_deleted_after_evidence_freeze",
            "baseline_artifacts_rehashed_before_and_after",
        )
    ):
        raise FidelityManifestError("Authorization output policy drifted")

    expected_governance = {
        "proposal": {"path": PROPOSAL.as_posix(), "sha256": proposal_record["sha256"]},
        "authorization": {
            "path": AUTHORIZATION.as_posix(),
            "sha256": authorization_record["sha256"],
        },
        "parent_closure": {
            "path": PARENT_CLOSURE.as_posix(),
            "sha256": parent_record["sha256"],
            "verification_result": preflight.get("governance_binding", {})
            .get("parent_closure", {})
            .get("verification_result"),
        },
    }
    verification_result = expected_governance["parent_closure"]["verification_result"]
    if not isinstance(verification_result, str) or PARENT_DECISION not in verification_result:
        raise FidelityManifestError("Preflight parent closure verification result drifted")
    if (
        preflight.get("decision") != PREFLIGHT_DECISION
        or preflight.get("governance_binding") != expected_governance
    ):
        raise FidelityManifestError("Preflight governance binding drifted")
    output_preconditions = preflight.get("output_preconditions")
    if not isinstance(output_preconditions, Mapping) or (
        any(
            output_preconditions.get(field) is not False
            for field in (
                "implementation_manifest_present",
                "localization_authorization_present",
                "localization_lease_present",
                "localization_evidence_present",
                "post_manifest_present",
            )
        )
        or output_preconditions.get("exclusive_create_required") is not True
    ):
        raise FidelityManifestError("Preflight output preconditions drifted")
    access_boundary = preflight.get("access_boundary")
    if not isinstance(access_boundary, Mapping) or any(
        value is not False for value in access_boundary.values()
    ):
        raise FidelityManifestError("Preflight access boundary drifted")
    implementation_gate = preflight.get("implementation_gate")
    if not isinstance(implementation_gate, Mapping) or (
        implementation_gate.get("parent_closure_verified") is not True
        or implementation_gate.get("baseline_hashes_match_authorization") is not True
        or implementation_gate.get("implementation_allowed") is not True
        or implementation_gate.get("model_inference_allowed") is not False
        or implementation_gate.get("requires_separate_localization_authorization") is not True
    ):
        raise FidelityManifestError("Preflight implementation gate drifted")

    _parent_artifact_rows(project_root, parent)
    baseline_specs = _baseline_specs(authorization, proposal)
    baseline_integrity = preflight.get("baseline_integrity")
    if not isinstance(baseline_integrity, Mapping):
        raise FidelityManifestError("Preflight baseline integrity is missing")
    for spec in baseline_specs:
        baseline_name = spec.name.removeprefix("baseline_")
        field = f"{baseline_name}_sha256"
        if field in baseline_integrity and baseline_integrity[field] != spec.expected_sha256:
            raise FidelityManifestError(f"Preflight baseline hash drifted: {baseline_name}")

    return {
        "proposal": proposal,
        "proposal_record": proposal_record,
        "authorization": authorization,
        "authorization_record": authorization_record,
        "preflight": preflight,
        "preflight_record": preflight_record,
        "parent": parent,
        "parent_record": parent_record,
        "baseline_specs": baseline_specs,
    }


def _implementation_specs(governance: Mapping[str, object]) -> tuple[ArtifactSpec, ...]:
    records = {
        PROPOSAL: governance["proposal_record"],
        AUTHORIZATION: governance["authorization_record"],
        PREFLIGHT: governance["preflight_record"],
        PARENT_CLOSURE: governance["parent_record"],
    }
    fixed_specs = []
    for name, role, path in IMPLEMENTATION_FIXED_ARTIFACTS:
        record = records.get(path)
        expected_sha256 = str(record["sha256"]) if isinstance(record, Mapping) else None
        if path == PROBE_RUNTIME:
            expected_sha256 = str(
                governance["authorization"]["baseline_artifacts"]["onnxruntime"]["sha256"]
            )
        fixed_specs.append(
            ArtifactSpec(
                name,
                role,
                path,
                expected_sha256,
            )
        )
    return (*fixed_specs, *governance["baseline_specs"])


def build_implementation_manifest(project_root: Path, *, generated_at_utc: str) -> dict:
    generated_at_utc = _validate_generated_at(generated_at_utc)
    governance = _validate_governance(project_root)
    artifacts = _implementation_specs(governance)
    _validate_inventory(artifacts, forbidden_paths=IMPLEMENTATION_FORBIDDEN_PATHS)
    rows = [_stable_file_record(project_root, spec) for spec in artifacts]
    predeclared_count = sum(spec.expected_sha256 is not None for spec in artifacts)
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "loop_id": LOOP_ID,
        "generated_at_utc": generated_at_utc,
        "contract": {
            "operation": "structured_governance_validation_and_streaming_sha256",
            "duplicate_json_keys_rejected": True,
            "parent_closure_artifacts_reverified": True,
            "baseline_model_payloads_parsed": False,
            "model_inference_performed": False,
            "manifest_self_hashed": False,
            "output_replace_allowed": False,
            "future_outputs_bound": False,
        },
        "claim_scope": {
            "synthetic_fidelity_tooling_hash_closure_only": True,
            "localization_execution_authorized": False,
            "raw_split_heldout_accessed": False,
            "training_or_fitting_performed": False,
            "quality_metric_computed": False,
            "quality_claim_allowed": False,
            "parity_claim_allowed": False,
            "certification_claim_allowed": False,
        },
        "lineage": {
            "parent_closure_sha256": governance["parent_record"]["sha256"],
            "proposal_sha256": governance["proposal_record"]["sha256"],
            "authorization_sha256": governance["authorization_record"]["sha256"],
            "preflight_sha256": governance["preflight_record"]["sha256"],
        },
        "artifacts": rows,
        "integrity": {
            "artifact_count": len(rows),
            "required_artifact_count": len(rows),
            "present_required_artifact_count": len(rows),
            "predeclared_sha256_count": predeclared_count,
            "verified_predeclared_sha256_count": predeclared_count,
            "blockers": [],
        },
        "decision": IMPLEMENTATION_DECISION,
    }


def resolve_fixed_output(project_root: Path, requested_path: Path, *, mode: str) -> Path:
    expected = IMPLEMENTATION_OUTPUT if mode == "implementation" else POST_OUTPUT
    _validate_relative_path(requested_path, purpose=f"{mode} manifest output")
    if requested_path.as_posix() != expected.as_posix():
        raise FidelityManifestError(f"{mode} manifest output path is not fixed")
    root = project_root.resolve(strict=True)
    current = root
    for part in expected.parts:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise FidelityManifestError(f"{mode} manifest output path contains a symlink")
    requested = _resolve_project_path(
        project_root,
        requested_path,
        purpose=f"{mode} manifest output",
        must_exist=False,
    )
    fixed = _resolve_project_path(
        project_root,
        expected,
        purpose=f"fixed {mode} manifest output",
        must_exist=False,
    )
    if requested != fixed:
        raise FidelityManifestError(f"{mode} manifest output path is not fixed")
    return requested


def _write_exclusive(output_path: Path, payload: Mapping[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with output_path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FidelityManifestError(f"Output already exists: {output_path}") from exc


def verify_implementation_manifest(
    project_root: Path,
    manifest_path: Path = IMPLEMENTATION_OUTPUT,
) -> dict:
    resolve_fixed_output(project_root, manifest_path, mode="implementation")
    payload, _record = _read_json(
        project_root,
        manifest_path,
        expected_schema=IMPLEMENTATION_SCHEMA,
    )
    rebuilt = build_implementation_manifest(
        project_root,
        generated_at_utc=str(payload.get("generated_at_utc") or ""),
    )
    if payload != rebuilt:
        raise FidelityManifestError("Implementation manifest no longer matches its closure")
    return payload


def _validate_localization_authorization(
    payload: Mapping[str, object],
    *,
    governance: Mapping[str, object],
    implementation_sha256: str,
) -> None:
    required_fields = {
        "schema",
        "loop_id",
        "issued_at_utc",
        "proposal_sha256",
        "authorization_sha256",
        "preflight_sha256",
        "parent_closure_sha256",
        "implementation_manifest",
        "attempt_id",
        "ready_lease_path",
        "consumed_lease_path",
        "evidence_path",
        "fixture_names",
        "baseline_artifacts",
        "budget",
        "claim_scope",
        "decision",
    }
    if not required_fields <= set(payload) or payload.get("loop_id") != LOOP_ID:
        raise FidelityManifestError("Localization authorization contract drifted")
    _validate_generated_at(str(payload.get("issued_at_utc") or ""))
    expected_values = {
        "proposal_sha256": governance["proposal_record"]["sha256"],
        "authorization_sha256": governance["authorization_record"]["sha256"],
        "preflight_sha256": governance["preflight_record"]["sha256"],
        "parent_closure_sha256": governance["parent_record"]["sha256"],
        "implementation_manifest": {
            "path": IMPLEMENTATION_OUTPUT.as_posix(),
            "sha256": implementation_sha256,
        },
        "ready_lease_path": LOCALIZATION_LEASE_PENDING.as_posix(),
        "consumed_lease_path": LOCALIZATION_LEASE.as_posix(),
        "evidence_path": LOCALIZATION_EVIDENCE.as_posix(),
        "baseline_artifacts": governance["authorization"]["baseline_artifacts"],
        "budget": governance["proposal"]["budget"],
        "claim_scope": EXPECTED_RUN_CLAIM_SCOPE,
        "decision": LOCALIZATION_AUTHORIZATION_DECISION,
    }
    for field, expected in expected_values.items():
        if payload.get(field) != expected:
            raise FidelityManifestError(f"Localization authorization binding drifted: {field}")
    attempt_id = payload.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id.startswith(f"{LOOP_ID}_attempt_"):
        raise FidelityManifestError("Localization attempt identity drifted")
    expected_fixtures = [row["name"] for row in governance["proposal"]["fixture_matrix"]]
    if payload.get("fixture_names") != expected_fixtures:
        raise FidelityManifestError("Localization fixture matrix drifted")


def _validate_consumed_lease(
    payload: Mapping[str, object],
    *,
    localization_authorization_sha256: str,
) -> None:
    required_fields = {
        "schema",
        "loop_id",
        "localization_authorization",
        "consumed_at_utc",
        "original_lease_sha256",
        "status",
    }
    if not required_fields <= set(payload):
        raise FidelityManifestError("Localization lease contract drifted")
    expected_values = {
        "loop_id": LOOP_ID,
        "localization_authorization": {"sha256": localization_authorization_sha256},
        "status": LEASE_CONSUMED_STATUS,
    }
    for field, expected in expected_values.items():
        if payload.get(field) != expected:
            raise FidelityManifestError(f"Localization lease is not consumed or drifted: {field}")
    _validate_generated_at(str(payload.get("consumed_at_utc") or ""))
    if not _is_sha256(payload.get("original_lease_sha256")):
        raise FidelityManifestError("Consumed lease original ready-lease hash is invalid")


def _validate_evidence_decision(
    payload: Mapping[str, object],
    *,
    controls_reproduced: bool,
    localized: bool,
    deterministic: bool,
    input_hashes_stable: bool,
    baseline_stable: bool,
    within_budget: bool,
) -> dict:
    decision = payload.get("decision")
    if decision not in EXIT_DECISIONS:
        raise FidelityManifestError("Localization evidence decision is not an allowed exit branch")
    if not input_hashes_stable or not baseline_stable:
        raise FidelityManifestError("Localization evidence baseline or input hashes drifted")
    synthetic_fidelity_gate_passed = (
        controls_reproduced and deterministic and not localized and within_budget
    )
    if decision == "localized_negative_no_raw":
        valid = controls_reproduced and localized and within_budget
    elif decision == "synthetic_fidelity_verified_raw_still_requires_new_authorization":
        valid = synthetic_fidelity_gate_passed and within_budget
    elif decision == "invalid_positive_control_or_lineage_drift":
        valid = not controls_reproduced and within_budget
    else:
        valid = not within_budget and not synthetic_fidelity_gate_passed
    if not valid:
        raise FidelityManifestError(
            f"Localization evidence is inconsistent with decision: {decision}"
        )
    return {
        "known_controls_reproduced": controls_reproduced,
        "first_divergence_boundary_localized": localized,
        "all_repeated_outputs_bit_exact": deterministic,
        "input_hashes_stable": input_hashes_stable,
        "baseline_artifact_hashes_stable": baseline_stable,
        "within_budget": within_budget,
        "synthetic_fidelity_gate_passed": synthetic_fidelity_gate_passed,
        "raw_split_heldout_access_count": 0,
        "quality_metric_count": 0,
    }


def _validate_evidence(
    project_root: Path,
    payload: Mapping[str, object],
    *,
    governance: Mapping[str, object],
    implementation_payload: Mapping[str, object],
    implementation_sha256: str,
    localization_authorization_sha256: str,
    lease_sha256: str,
    attempt_id: str,
) -> tuple[dict, list[dict]]:
    required_fields = {
        "schema",
        "loop_id",
        "generated_at_utc",
        "governance",
        "scope",
        "runtime_contract",
        "fixtures",
        "baseline_integrity",
        "budget",
        "claim_boundary",
        "decision",
    }
    if not required_fields <= set(payload) or payload.get("loop_id") != LOOP_ID:
        raise FidelityManifestError("Localization evidence contract drifted")
    _validate_generated_at(str(payload.get("generated_at_utc") or ""))

    evidence_governance = payload.get("governance")
    if not isinstance(evidence_governance, Mapping):
        raise FidelityManifestError("Localization evidence governance chain is missing")
    expected_governance = {
        "proposal_sha256": governance["proposal_record"]["sha256"],
        "authorization_sha256": governance["authorization_record"]["sha256"],
        "preflight_sha256": governance["preflight_record"]["sha256"],
        "implementation_manifest_sha256": implementation_sha256,
        "localization_authorization_sha256": localization_authorization_sha256,
    }
    if any(evidence_governance.get(key) != value for key, value in expected_governance.items()):
        raise FidelityManifestError("Localization evidence authorization chain drifted")
    expected_consumed_lease = {
        "path": LOCALIZATION_LEASE.as_posix(),
        "sha256": lease_sha256,
        "original_lease_sha256": evidence_governance.get("consumed_lease", {}).get(
            "original_lease_sha256"
        )
        if isinstance(evidence_governance.get("consumed_lease"), Mapping)
        else None,
        "status": LEASE_CONSUMED_STATUS,
    }
    consumed_lease = evidence_governance.get("consumed_lease")
    if consumed_lease != expected_consumed_lease or not _is_sha256(
        expected_consumed_lease["original_lease_sha256"]
    ):
        raise FidelityManifestError("Localization evidence consumed lease binding drifted")

    scope = payload.get("scope")
    if not isinstance(scope, Mapping) or (
        scope.get("synthetic_only") is not True
        or any(
            scope.get(field) is not False
            for field in (
                "dataset_raw_accessed",
                "split_metadata_accessed",
                "cache_rows_accessed",
                "heldout_accessed",
                "training_or_fitting_performed",
                "quality_metric_computed",
                "f1_computed",
            )
        )
    ):
        raise FidelityManifestError("Localization evidence scope audit drifted")

    claim_boundary = payload.get("claim_boundary")
    if not isinstance(claim_boundary, Mapping) or any(
        claim_boundary.get(field) is not False
        for field in (
            "population_parity_claim_allowed",
            "quality_claim_allowed",
            "native_loop28_ready_claim_allowed",
            "native_loop151_ready_claim_allowed",
            "raw_rerun_allowed",
            "certification_claim_allowed",
        )
    ):
        raise FidelityManifestError("Localization evidence claim boundary drifted")

    runtime = payload.get("runtime_contract")
    probe_plan = governance["proposal"].get("probe_plan")
    if not isinstance(runtime, Mapping) or not isinstance(probe_plan, Mapping):
        raise FidelityManifestError("Localization runtime contract is missing")
    expected_runtime = {
        "cpu_only": True,
        "graph_optimization": "ORT_DISABLE_ALL",
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "execution_mode": "ORT_SEQUENTIAL",
        "repeats": probe_plan.get("runtime_determinism_repeats"),
    }
    implementation_artifacts = implementation_payload.get("artifacts")
    if not isinstance(implementation_artifacts, list):
        raise FidelityManifestError("Implementation artifact inventory is missing")
    implementation_by_name = {
        row.get("name"): row for row in implementation_artifacts if isinstance(row, Mapping)
    }
    probe_record = implementation_by_name.get("probe_binary")
    if (
        any(runtime.get(key) != value for key, value in expected_runtime.items())
        or not isinstance(probe_record, Mapping)
        or runtime.get("probe_sha256") != probe_record.get("sha256")
    ):
        raise FidelityManifestError("Localization runtime contract drifted")

    fixture_results = payload.get("fixtures")
    expected_fixture_rows = governance["proposal"]["fixture_matrix"]
    expected_fixtures = [row["name"] for row in expected_fixture_rows]
    if (
        not isinstance(fixture_results, list)
        or [
            row.get("fixture", {}).get("name")
            for row in fixture_results
            if isinstance(row, Mapping)
        ]
        != expected_fixtures
        or any(not isinstance(row, Mapping) for row in fixture_results)
    ):
        raise FidelityManifestError("Localization fixture evidence drifted")
    controls = []
    deterministic_rows = []
    input_hashes_stable = True
    for expected_fixture, row in zip(expected_fixture_rows, fixture_results, strict=True):
        fixture = row.get("fixture")
        expected_control = (
            "fail"
            if expected_fixture.get("positive_control") == "must_fail_base_probability_gate"
            else "pass"
        )
        expected_identity = {
            "name": expected_fixture["name"],
            "pe_plus": expected_fixture["pe_plus"],
            "named_resource": expected_fixture["named_resource"],
            "tls_callbacks": expected_fixture["tls_callbacks"],
            "expected_control": expected_control,
        }
        if fixture != expected_identity:
            raise FidelityManifestError(
                f"Localization fixture identity drifted: {expected_fixture['name']}"
            )
        base_probability = row.get("base_probability")
        inputs = row.get("inputs")
        profiles = row.get("profiles")
        pytorch_determinism = row.get("pytorch_determinism")
        if (
            not isinstance(base_probability, Mapping)
            or not isinstance(inputs, Mapping)
            or set(inputs) != {"byte_seq", "pe_features", "stat_features"}
            or not isinstance(profiles, Mapping)
            or set(profiles) != {"macro", "routing"}
            or not isinstance(pytorch_determinism, Mapping)
        ):
            raise FidelityManifestError(
                f"Localization fixture payload drifted: {expected_fixture['name']}"
            )
        controls.append(base_probability.get("control_reproduced") is True)
        row_deterministic = pytorch_determinism.get("bit_exact") is True
        for profile in profiles.values():
            if not isinstance(profile, Mapping) or not isinstance(
                profile.get("ort_determinism"), Mapping
            ):
                raise FidelityManifestError("Localization determinism evidence drifted")
            row_deterministic = (
                row_deterministic and profile["ort_determinism"].get("bit_exact") is True
            )
        deterministic_rows.append(row_deterministic)
        for input_record in inputs.values():
            if (
                not isinstance(input_record, Mapping)
                or not isinstance(input_record.get("dtype"), str)
                or not isinstance(input_record.get("shape"), list)
                or isinstance(input_record.get("nbytes"), bool)
                or not isinstance(input_record.get("nbytes"), int)
                or input_record["nbytes"] < 0
                or not _is_sha256(input_record.get("sha256"))
            ):
                input_hashes_stable = False

    controls_reproduced = all(controls)
    deterministic = all(deterministic_rows)
    if (
        payload.get("positive_controls_reproduced") is not controls_reproduced
        or payload.get("determinism_all_passed") is not deterministic
    ):
        raise FidelityManifestError("Localization control or determinism summary drifted")
    first_divergences = payload.get("first_divergences")
    if (
        not isinstance(first_divergences, list)
        or [row.get("fixture") for row in first_divergences if isinstance(row, Mapping)]
        != expected_fixtures
        or any(not isinstance(row, Mapping) for row in first_divergences)
    ):
        raise FidelityManifestError("Localization first-divergence inventory drifted")
    localized = any(
        row.get("macro") is not None or row.get("routing") is not None for row in first_divergences
    )

    graph = payload.get("graph")
    if not isinstance(graph, Mapping) or (
        graph.get("onnx_graph_sha256")
        != governance["authorization"]["baseline_artifacts"]["onnx_graph"]["sha256"]
        or graph.get("onnx_data_sha256")
        != governance["authorization"]["baseline_artifacts"]["onnx_data"]["sha256"]
    ):
        raise FidelityManifestError("Localization graph lineage drifted")

    expected_baseline = governance["authorization"]["baseline_artifacts"]
    baseline_integrity = payload.get("baseline_integrity")
    execution_baseline_names = {"checkpoint", "onnx_graph", "onnx_data", "fixture_contract"}
    before_hashes = (
        baseline_integrity.get("before") if isinstance(baseline_integrity, Mapping) else None
    )
    after_hashes = (
        baseline_integrity.get("after") if isinstance(baseline_integrity, Mapping) else None
    )
    expected_execution_hashes = {
        name: expected_baseline[name]["sha256"] for name in execution_baseline_names
    }
    baseline_stable = (
        isinstance(baseline_integrity, Mapping)
        and before_hashes == expected_execution_hashes
        and after_hashes == expected_execution_hashes
        and baseline_integrity.get("stable") is True
    )
    if not baseline_stable:
        raise FidelityManifestError("Localization baseline evidence inventory drifted")

    budget = payload.get("budget")
    proposal_budget = governance["proposal"]["budget"]
    budget_fields = {
        "fixture_count": proposal_budget["max_fixture_count"],
        "profile_count": proposal_budget["max_probe_profiles"],
        "native_subprocess_count": proposal_budget["max_native_subprocesses"],
        "wall_clock_seconds": proposal_budget["total_wall_clock_seconds"],
        "retained_probe_output_bytes": proposal_budget["max_retained_output_bytes"],
    }
    if not isinstance(budget, Mapping):
        raise FidelityManifestError("Localization budget evidence is missing")
    within_budget = True
    for field, limit in budget_fields.items():
        value = budget.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise FidelityManifestError(f"Localization budget value is invalid: {field}")
        within_budget = within_budget and float(value) <= float(limit)

    outcome = _validate_evidence_decision(
        payload,
        controls_reproduced=controls_reproduced,
        localized=localized,
        deterministic=deterministic,
        input_hashes_stable=input_hashes_stable,
        baseline_stable=baseline_stable,
        within_budget=within_budget,
    )
    baseline_rows = []
    for name, authorized_record in expected_baseline.items():
        baseline_rows.append(
            _stable_file_record(
                project_root,
                ArtifactSpec(
                    f"baseline_{name}",
                    "baseline_verified_stable_before_and_after",
                    Path(str(authorized_record["path"])),
                    str(authorized_record["sha256"]),
                ),
            )
        )
    return outcome, baseline_rows


def _pending_lease_present(project_root: Path) -> bool:
    _resolve_project_path(
        project_root,
        LOCALIZATION_LEASE_PENDING,
        purpose="Pending localization lease",
        must_exist=False,
    )
    return os.path.lexists(project_root.resolve(strict=True) / LOCALIZATION_LEASE_PENDING)


def build_post_manifest(project_root: Path, *, generated_at_utc: str) -> dict:
    generated_at_utc = _validate_generated_at(generated_at_utc)
    if _pending_lease_present(project_root):
        raise FidelityManifestError(
            "Pending localization lease still exists; consumed final lease required"
        )

    verify_implementation_manifest(project_root, IMPLEMENTATION_OUTPUT)
    _implementation_payload, implementation_record = _read_json(
        project_root,
        IMPLEMENTATION_OUTPUT,
        expected_schema=IMPLEMENTATION_SCHEMA,
    )
    governance = _validate_governance(project_root)
    localization_authorization, localization_authorization_record = _read_json(
        project_root,
        LOCALIZATION_AUTHORIZATION,
        expected_schema=LOCALIZATION_AUTHORIZATION_SCHEMA,
    )
    _validate_localization_authorization(
        localization_authorization,
        governance=governance,
        implementation_sha256=implementation_record["sha256"],
    )
    lease, lease_record = _read_json(
        project_root,
        LOCALIZATION_LEASE,
        expected_schema=LOCALIZATION_LEASE_SCHEMA,
    )
    attempt_id = str(localization_authorization["attempt_id"])
    _validate_consumed_lease(
        lease,
        localization_authorization_sha256=localization_authorization_record["sha256"],
    )
    evidence, evidence_record = _read_json(
        project_root,
        LOCALIZATION_EVIDENCE,
        expected_schema=LOCALIZATION_EVIDENCE_SCHEMA,
    )
    outcome, baseline_rows = _validate_evidence(
        project_root,
        evidence,
        governance=governance,
        implementation_payload=_implementation_payload,
        implementation_sha256=implementation_record["sha256"],
        localization_authorization_sha256=localization_authorization_record["sha256"],
        lease_sha256=lease_record["sha256"],
        attempt_id=attempt_id,
    )

    # 最终闭包只绑定不可变 loop 文档；旧 living docs 与可变 current 指针均不进入清单。
    artifacts = [
        _stable_file_record(
            project_root,
            ArtifactSpec(
                "parent_closure",
                "immutable_parent_closure",
                PARENT_CLOSURE,
                str(governance["parent_record"]["sha256"]),
            ),
        ),
        _stable_file_record(
            project_root,
            ArtifactSpec(
                "implementation_manifest",
                "verified_tooling_implementation_closure",
                IMPLEMENTATION_OUTPUT,
                str(implementation_record["sha256"]),
            ),
        ),
        _stable_file_record(
            project_root,
            ArtifactSpec(
                "localization_authorization",
                "manifest_bound_localization_authorization",
                LOCALIZATION_AUTHORIZATION,
                str(localization_authorization_record["sha256"]),
            ),
        ),
        _stable_file_record(
            project_root,
            ArtifactSpec(
                "consumed_lease",
                "consumed_before_model_execution",
                LOCALIZATION_LEASE,
                str(lease_record["sha256"]),
            ),
        ),
        _stable_file_record(
            project_root,
            ArtifactSpec(
                "localization_evidence",
                "synthetic_fidelity_result",
                LOCALIZATION_EVIDENCE,
                str(evidence_record["sha256"]),
            ),
        ),
        _stable_file_record(
            project_root, ArtifactSpec("goal_delta", "immutable_goal_delta", GOAL_DELTA)
        ),
        _stable_file_record(
            project_root,
            ArtifactSpec("journal_entry", "immutable_loop_journal_entry", JOURNAL_ENTRY),
        ),
        _stable_file_record(
            project_root,
            ArtifactSpec("final_status", "immutable_owner_facing_status", FINAL_STATUS),
        ),
        *baseline_rows,
    ]
    paths = [row["path"].casefold() for row in artifacts]
    if len(paths) != len(set(paths)) or POST_OUTPUT.as_posix().casefold() in paths:
        raise FidelityManifestError("Post artifact inventory contains a duplicate or cycle")
    decision = str(evidence["decision"])
    return {
        "schema": POST_SCHEMA,
        "loop_id": LOOP_ID,
        "generated_at_utc": generated_at_utc,
        "contract": {
            "structured_chain_validation": True,
            "duplicate_json_keys_rejected": True,
            "consumed_lease_required": True,
            "baseline_hashes_stable_before_after_and_at_closure": True,
            "legacy_living_docs_mutated": False,
            "mutable_living_docs_pointer_bound": False,
            "model_inference_performed_by_builder": False,
            "manifest_self_hashed": False,
            "output_replace_allowed": False,
        },
        "claim_scope": {
            "synthetic_fidelity_exit_branch": decision,
            "raw_rerun_authorized": False,
            "population_parity_claim_allowed": False,
            "quality_claim_allowed": False,
            "native_loop28_ready_claim_allowed": False,
            "native_loop151_ready_claim_allowed": False,
            "certification_claim_allowed": False,
        },
        "lineage": {
            "parent_closure_sha256": governance["parent_record"]["sha256"],
            "implementation_manifest_sha256": implementation_record["sha256"],
            "localization_authorization_sha256": localization_authorization_record["sha256"],
            "consumed_lease_sha256": lease_record["sha256"],
            "localization_evidence_sha256": evidence_record["sha256"],
            "attempt_id": attempt_id,
        },
        "outcome": outcome,
        "artifacts": artifacts,
        "integrity": {
            "artifact_count": len(artifacts),
            "required_artifact_count": len(artifacts),
            "present_required_artifact_count": len(artifacts),
            "baseline_artifact_count": len(baseline_rows),
            "structured_chain_links_verified": 5,
            "blockers": [],
        },
        "decision": POST_DECISIONS[decision],
    }


def verify_post_manifest(project_root: Path, manifest_path: Path = POST_OUTPUT) -> dict:
    resolve_fixed_output(project_root, manifest_path, mode="post")
    payload, _record = _read_json(
        project_root,
        manifest_path,
        expected_schema=POST_SCHEMA,
    )
    rebuilt = build_post_manifest(
        project_root,
        generated_at_utc=str(payload.get("generated_at_utc") or ""),
    )
    if payload != rebuilt:
        raise FidelityManifestError("Post manifest no longer matches its evidence chain")
    return payload


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
    project_root = args.project_root.resolve(strict=True)
    default_output = IMPLEMENTATION_OUTPUT if args.mode == "implementation" else POST_OUTPUT
    requested_output = args.output or default_output
    output_path = resolve_fixed_output(project_root, requested_output, mode=args.mode)
    if args.verify:
        manifest = (
            verify_implementation_manifest(project_root, requested_output)
            if args.mode == "implementation"
            else verify_post_manifest(project_root, requested_output)
        )
    else:
        if not args.generated_at_utc:
            raise FidelityManifestError("--generated-at-utc is required when building")
        manifest = (
            build_implementation_manifest(project_root, generated_at_utc=args.generated_at_utc)
            if args.mode == "implementation"
            else build_post_manifest(project_root, generated_at_utc=args.generated_at_utc)
        )
        _write_exclusive(output_path, manifest)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "output": requested_output.as_posix(),
                "artifact_count": manifest["integrity"]["artifact_count"],
                "decision": manifest["decision"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
