#!/usr/bin/env python3
"""Build and verify the hash-only Loop28 parity implementation manifest.

The builder deliberately treats every governed artifact as opaque bytes. It
uses only ``stat`` and streaming SHA-256, so split rows, predictions, metrics,
pickles, checkpoints, and model JSON are never parsed by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA = "axon_loop28_parity_diagnostic_implementation_manifest_v1"
LOOP_ID = "p0_loop28_parity_diagnostic_001"
DEFAULT_OUTPUT = Path(
    "manifests/roadmap_9997/p0_loop28_parity_diagnostic/implementation_manifest.json"
)
READY_DECISION = "implementation_hash_freeze_complete_run_authorization_pending"
BLOCKED_DECISION = "implementation_hash_freeze_blocked"
VERIFIED_DECISION = "implementation_manifest_verified_current_hashes"


class ManifestContractError(ValueError):
    """Raised when the fixed implementation-manifest contract is violated."""


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    role: str
    path: Path


PARENT_EVIDENCE_PATHS: Mapping[str, Path] = {
    "truth_manifest": Path("manifests/roadmap_9997/p0_truth_freeze/loop151_truth_manifest.json"),
    "train_smoke_receipt": Path("manifests/roadmap_9997/p0_raw_replay/train_smoke_receipt.json"),
    "native_parity_receipt": Path(
        "manifests/roadmap_9997/p0_raw_replay/native_parity_receipt.json"
    ),
    "complete_replay_verify_receipt": Path(
        "manifests/roadmap_9997/p0_raw_replay/verify_receipt.json"
    ),
}
PARENT_EVIDENCE_SHA256: Mapping[str, str] = {
    "truth_manifest": "174861be850a681025a7040798c59b7157cc67ab5503437088359692dad5659d",
    "train_smoke_receipt": "3fbbd597dc14cf67a082203e680e83c1f521972e1bf1f36282ef847d9d4d2309",
    "native_parity_receipt": "abeaeb0dd28060a09b7b23fbf0300b2ba6616ef040c4c4958f870aa5d13436b8",
    "complete_replay_verify_receipt": (
        "fc39b4a0289d4fc21955030157a8fb4beb437eba78d28dec6f5a665bcf0a5c94"
    ),
}

PREREGISTRATION_PATHS = (
    Path("manifests/roadmap_9997/p0_loop28_parity_diagnostic/proposal.json"),
    Path("manifests/roadmap_9997/p0_loop28_parity_diagnostic/authorization.json"),
    Path("manifests/roadmap_9997/p0_loop28_parity_diagnostic/preflight.json"),
)

DENIED_ARTIFACT_PATHS = frozenset(
    {
        DEFAULT_OUTPUT.as_posix(),
        "manifests/roadmap_9997/p0_loop28_parity_diagnostic/run_authorization.json",
        "manifests/roadmap_9997/p0_loop28_parity_diagnostic/run_attempt.final.json",
        "manifests/roadmap_9997/p0_loop28_parity_diagnostic/post_diagnostic_manifest.json",
        "manifests/roadmap_9997/p0_loop28_parity_diagnostic/post_manifest.json",
        "reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.json",
        "reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.final.json",
        "reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.provisional.json",
        "reports/hard_family_finetune/experiment_journal.md",
    }
)


def _group(role: str, **artifacts: str | Path) -> tuple[ArtifactSpec, ...]:
    return tuple(ArtifactSpec(name, role, Path(path)) for name, path in artifacts.items())


DEFAULT_ARTIFACTS = (
    *_group("immutable_parent_evidence", **PARENT_EVIDENCE_PATHS),
    *_group(
        "preregistration",
        diagnostic_proposal=PREREGISTRATION_PATHS[0],
        diagnostic_authorization=PREREGISTRATION_PATHS[1],
        diagnostic_preflight=PREREGISTRATION_PATHS[2],
    ),
    *_group(
        "diagnostic_manifest_source",
        implementation_manifest_builder="scripts/build_loop28_parity_diagnostic_manifest.py",
    ),
    *_group(
        "diagnostic_manifest_test",
        implementation_manifest_builder_test=(
            "tests/test_build_loop28_parity_diagnostic_manifest.py"
        ),
    ),
    *_group("diagnostic_source", python_diagnostic="scripts/diagnose_loop28_parity.py"),
    *_group("diagnostic_test", python_diagnostic_test="tests/test_diagnose_loop28_parity.py"),
    *_group("diagnostic_python_dependency", raw_replay_guard="scripts/replay_loop151_raw.py"),
    *_group(
        "diagnostic_python_dependency_test",
        raw_replay_guard_test="tests/test_replay_loop151_raw.py",
    ),
    *_group(
        "python_runtime_source",
        predict_api="src/predict_api.py",
        runtime_archive_scanner="src/archive_scanner.py",
        runtime_config="src/config.py",
        runtime_model="src/model.py",
        runtime_security="src/security.py",
        runtime_feature_exports="src/kvd_features/__init__.py",
        runtime_feature_extractor="src/kvd_features/extractor.py",
        runtime_content_pe_v1="src/kvd_features/content_pe_v1.py",
        runtime_dsra_exports="src/dsra/__init__.py",
        runtime_dsra_domain_exports="src/dsra/domain/__init__.py",
        runtime_dsra_attention_spec="src/dsra/domain/attention_spec.py",
        runtime_dsra_arithmetic_domain="src/dsra/domain/arithmetic_emergence.py",
        runtime_dsra_model_spec="src/dsra/domain/model_spec.py",
        runtime_mhdsra2_exports="src/dsra/mhdsra2/__init__.py",
        runtime_mhdsra2="src/dsra/mhdsra2/improved_dsra_mha.py",
        runtime_paged_memory="src/dsra/mhdsra2/paged_exact_memory.py",
    ),
    *_group("python_runtime_test", predict_api_test="tests/test_predict_api_loop28.py"),
    *_group(
        "python_dependency_contract",
        python_project_metadata="pyproject.toml",
        python_requirements="requirements.txt",
    ),
    *_group(
        "train_identity_source_hash_only",
        frozen_split="reports/random_20w_split/loop127_full_duplicate_corrected_split.csv",
    ),
    *_group(
        "parent_a1_authorization",
        raw_replay_a1_authorization="manifests/roadmap_9997/p0_raw_replay/authorization.json",
    ),
    *_group(
        "deserialization_policy",
        pickle_allowlist=("manifests/roadmap_9997/p0_raw_replay/pickle_sha256_allowlist.json"),
    ),
    *_group(
        "runtime_metadata",
        python_stage2_metadata=("manifests/roadmap_9997/p0_raw_replay/loop28_stage2.metadata.json"),
    ),
    *_group(
        "python_runtime_model",
        python_checkpoint="models/random_20w_8192/best_model.pt",
        python_stage2=(
            "reports/random_20w_split/stage2_loop28_content_pe_valonly/stage2_selected_model.pkl"
        ),
    ),
    *_group(
        "native_runtime_model",
        native_onnx="models/random_20w_8192/axon_loop28_base.onnx",
        native_onnx_data="models/random_20w_8192/axon_loop28_base.onnx.data",
        native_stage2="models/random_20w_8192/loop28_stage2_hgb.json",
    ),
    *_group(
        "native_diagnostic_source",
        native_runtime_source="tools/axon_onnx_dll/src/axon_onnx_predict.cpp",
        native_public_header="tools/axon_onnx_dll/include/axon_onnx_predict.h",
    ),
    *_group(
        "native_diagnostic_test_source",
        native_selftest_source="tools/axon_onnx_dll/examples/axon_onnx_selftest.cpp",
    ),
    *_group("native_build_definition", native_cmake="tools/axon_onnx_dll/CMakeLists.txt"),
    *_group(
        "native_build_output",
        native_dll="tools/axon_onnx_dll/build/bin/Release/axon_onnx_predict.dll",
        native_selftest="tools/axon_onnx_dll/build/bin/Release/axon_onnx_selftest.exe",
    ),
    *_group(
        "native_runtime_dependency",
        native_onnxruntime="tools/axon_onnx_dll/build/bin/Release/onnxruntime.dll",
    ),
)


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _normalized_relative_path(path: Path) -> str:
    if path.is_absolute():
        raise ManifestContractError(f"Artifact path must be project-relative: {path}")
    normalized = path.as_posix()
    if not normalized or normalized == "." or ".." in Path(normalized).parts:
        raise ManifestContractError(f"Artifact path is not canonical: {path}")
    return normalized


def _validate_artifact_specs(specs: Sequence[ArtifactSpec]) -> None:
    names: set[str] = set()
    paths: set[str] = set()
    for spec in specs:
        if not spec.name or not spec.role:
            raise ManifestContractError("Artifact names and roles must be non-empty")
        if spec.name in names:
            raise ManifestContractError(f"Duplicate artifact name: {spec.name}")
        names.add(spec.name)
        normalized = _normalized_relative_path(spec.path)
        path_key = normalized.casefold()
        if path_key in paths:
            raise ManifestContractError(f"Duplicate artifact path: {normalized}")
        paths.add(path_key)
        if normalized in DENIED_ARTIFACT_PATHS:
            raise ManifestContractError(f"Denied artifact path: {normalized}")

    if set(PARENT_EVIDENCE_PATHS) != set(PARENT_EVIDENCE_SHA256):
        raise ManifestContractError("Parent evidence path/hash keys drifted")
    by_name = {spec.name: spec for spec in specs}
    for name, path in PARENT_EVIDENCE_PATHS.items():
        spec = by_name.get(name)
        if spec is None or spec.path != path or spec.role != "immutable_parent_evidence":
            raise ManifestContractError(f"Parent evidence artifact drifted: {name}")
        if not _is_sha256(PARENT_EVIDENCE_SHA256[name]):
            raise ManifestContractError(f"Parent evidence SHA-256 is invalid: {name}")


def _resolve_within(project_root: Path, path: Path, *, purpose: str) -> Path:
    root = project_root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ManifestContractError(f"{purpose} escapes project root: {resolved}") from exc
    return resolved


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream one opaque file into SHA-256 without decoding its contents."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_and_sha256(path: Path) -> tuple[int, str]:
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ManifestContractError(f"Artifact is not a regular file: {path}")
    digest = file_sha256(path)
    after = path.stat()
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
        raise ManifestContractError(f"Artifact changed while being hashed: {path}")
    return before.st_size, digest


def _artifact_record(project_root: Path, spec: ArtifactSpec) -> tuple[dict, Optional[str]]:
    path_text = _normalized_relative_path(spec.path)
    resolved = _resolve_within(project_root, spec.path, purpose=f"Artifact {spec.name}")
    record = {
        "name": spec.name,
        "role": spec.role,
        "path": path_text,
        "required": True,
        "exists": False,
        "size_bytes": None,
        "sha256": None,
    }
    try:
        size_bytes, digest = _stat_and_sha256(resolved)
    except FileNotFoundError:
        return record, f"missing_required_artifact:{spec.name}:{path_text}"
    except OSError as exc:
        return record, f"artifact_hash_failed:{spec.name}:{type(exc).__name__}"
    record.update({"exists": True, "size_bytes": size_bytes, "sha256": digest})

    expected_parent_sha = PARENT_EVIDENCE_SHA256.get(spec.name)
    if expected_parent_sha is not None and digest != expected_parent_sha:
        return record, f"parent_evidence_sha256_mismatch:{spec.name}"
    return record, None


def _fixed_output_path(project_root: Path) -> Path:
    return _resolve_within(project_root, DEFAULT_OUTPUT, purpose="Implementation manifest output")


def build_implementation_manifest(*, project_root: Path, replace: bool = False) -> dict:
    """Hash the fixed inventory and write the fixed implementation manifest."""

    root = project_root.resolve()
    output_path = _fixed_output_path(root)
    if output_path.exists() and not replace:
        raise FileExistsError(f"Implementation manifest already exists: {output_path}")
    _validate_artifact_specs(DEFAULT_ARTIFACTS)

    artifacts = []
    blockers = []
    for spec in DEFAULT_ARTIFACTS:
        record, blocker = _artifact_record(root, spec)
        artifacts.append(record)
        if blocker is not None:
            blockers.append(blocker)

    records_by_name = {record["name"]: record for record in artifacts}
    parent_evidence = {}
    for name, frozen_path in PARENT_EVIDENCE_PATHS.items():
        record = records_by_name[name]
        expected_sha = PARENT_EVIDENCE_SHA256[name]
        parent_evidence[name] = {
            "path": frozen_path.as_posix(),
            "sha256": expected_sha,
            "verified": bool(record["exists"] and record["sha256"] == expected_sha),
        }

    unique_blockers = sorted(set(blockers))
    present_count = sum(bool(record["exists"]) for record in artifacts)
    parent_verified_count = sum(bool(record["verified"]) for record in parent_evidence.values())
    decision = READY_DECISION if not unique_blockers else BLOCKED_DECISION
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "loop_id": LOOP_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "operation": "opaque_file_stat_and_streaming_sha256_only",
            "artifact_payloads_parsed": False,
            "split_rows_read": False,
            "prediction_or_metric_payloads_read": False,
            "model_or_pickle_payloads_loaded": False,
            "manifest_self_hashed": False,
            "supplemental_artifacts_allowed": False,
        },
        "parent_evidence": parent_evidence,
        "artifacts": artifacts,
        "integrity": {
            "artifact_count": len(artifacts),
            "required_artifact_count": len(artifacts),
            "present_required_artifact_count": present_count,
            "parent_evidence_count": len(parent_evidence),
            "verified_parent_evidence_count": parent_verified_count,
            "blockers": unique_blockers,
        },
        "excluded_outputs": sorted(DENIED_ARTIFACT_PATHS),
        "decision": decision,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if replace else "x"
    with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def _read_manifest(path: Path) -> tuple[dict, str]:
    try:
        before = path.stat()
        payload_bytes = path.read_bytes()
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ManifestContractError("Implementation manifest changed while being read")
        payload = json.loads(payload_bytes.decode("utf-8-sig"))
    except FileNotFoundError as exc:
        raise ManifestContractError(f"Implementation manifest is missing: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestContractError(f"Implementation manifest is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ManifestContractError("Implementation manifest root must be an object")
    return payload, hashlib.sha256(payload_bytes).hexdigest()


def _manifest_records(payload: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ManifestContractError("Implementation manifest artifacts must be a list")

    records: dict[str, Mapping[str, object]] = {}
    paths: set[str] = set()
    for raw_record in raw_artifacts:
        if not isinstance(raw_record, dict):
            raise ManifestContractError("Implementation artifact record must be an object")
        name = raw_record.get("name")
        path_value = raw_record.get("path")
        if not isinstance(name, str) or not name:
            raise ManifestContractError("Implementation artifact name is invalid")
        if name in records:
            raise ManifestContractError(f"Duplicate artifact name in manifest: {name}")
        if not isinstance(path_value, str):
            raise ManifestContractError(f"Implementation artifact path is invalid: {name}")
        normalized = _normalized_relative_path(Path(path_value))
        if normalized != path_value:
            raise ManifestContractError(f"Implementation artifact path is not canonical: {name}")
        path_key = normalized.casefold()
        if path_key in paths:
            raise ManifestContractError(f"Duplicate artifact path in manifest: {normalized}")
        if normalized in DENIED_ARTIFACT_PATHS:
            raise ManifestContractError(f"Denied artifact path in manifest: {normalized}")
        paths.add(path_key)
        records[name] = raw_record
    return records


def verify_implementation_manifest(project_root: Path, path: Path) -> dict:
    """Recompute every required digest and return a content-safe verification summary."""

    root = project_root.resolve()
    requested_path = _resolve_within(root, path, purpose="Requested implementation manifest")
    frozen_path = _fixed_output_path(root)
    if requested_path != frozen_path:
        raise ManifestContractError("Implementation manifest path is not the frozen output path")

    _validate_artifact_specs(DEFAULT_ARTIFACTS)
    payload, manifest_sha256 = _read_manifest(requested_path)
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ManifestContractError("Implementation manifest schema mismatch")
    if payload.get("loop_id") != LOOP_ID:
        raise ManifestContractError("Implementation manifest loop_id mismatch")
    if payload.get("decision") != READY_DECISION:
        raise ManifestContractError("Implementation manifest is not ready")

    contract = payload.get("contract")
    expected_contract = {
        "operation": "opaque_file_stat_and_streaming_sha256_only",
        "artifact_payloads_parsed": False,
        "split_rows_read": False,
        "prediction_or_metric_payloads_read": False,
        "model_or_pickle_payloads_loaded": False,
        "manifest_self_hashed": False,
        "supplemental_artifacts_allowed": False,
    }
    if contract != expected_contract:
        raise ManifestContractError("Implementation manifest hash-only contract drifted")
    if payload.get("excluded_outputs") != sorted(DENIED_ARTIFACT_PATHS):
        raise ManifestContractError("Implementation manifest exclusion contract drifted")

    records = _manifest_records(payload)
    expected_by_name = {spec.name: spec for spec in DEFAULT_ARTIFACTS}
    if set(records) != set(expected_by_name):
        raise ManifestContractError("Implementation manifest artifact inventory drifted")

    verified_sha256: dict[str, str] = {}
    for name, spec in expected_by_name.items():
        record = records[name]
        path_text = spec.path.as_posix()
        expected_keys = {
            "name",
            "role",
            "path",
            "required",
            "exists",
            "size_bytes",
            "sha256",
        }
        if set(record) != expected_keys:
            raise ManifestContractError(f"Implementation artifact fields drifted: {name}")
        if (
            record.get("role") != spec.role
            or record.get("path") != path_text
            or record.get("required") is not True
            or record.get("exists") is not True
        ):
            raise ManifestContractError(f"Implementation artifact contract drifted: {name}")
        declared_sha = record.get("sha256")
        declared_size = record.get("size_bytes")
        if (
            not _is_sha256(declared_sha)
            or isinstance(declared_size, bool)
            or not isinstance(declared_size, int)
        ):
            raise ManifestContractError(f"Implementation artifact hash/stat is invalid: {name}")

        resolved = _resolve_within(root, spec.path, purpose=f"Implementation artifact {name}")
        try:
            actual_size, actual_sha = _stat_and_sha256(resolved)
        except (FileNotFoundError, OSError) as exc:
            raise ManifestContractError(f"Implementation artifact is unavailable: {name}") from exc
        if actual_size != declared_size:
            raise ManifestContractError(f"Implementation artifact size mismatch: {name}")
        if actual_sha != str(declared_sha).casefold():
            raise ManifestContractError(f"Implementation artifact SHA-256 mismatch: {name}")
        frozen_parent_sha = PARENT_EVIDENCE_SHA256.get(name)
        if frozen_parent_sha is not None and actual_sha != frozen_parent_sha:
            raise ManifestContractError(f"Parent evidence SHA-256 mismatch: {name}")
        verified_sha256[name] = actual_sha

    parent_evidence = payload.get("parent_evidence")
    if not isinstance(parent_evidence, dict) or set(parent_evidence) != set(PARENT_EVIDENCE_PATHS):
        raise ManifestContractError("Implementation manifest parent evidence drifted")
    for name, frozen_parent_path in PARENT_EVIDENCE_PATHS.items():
        expected_record = {
            "path": frozen_parent_path.as_posix(),
            "sha256": PARENT_EVIDENCE_SHA256[name],
            "verified": True,
        }
        if parent_evidence.get(name) != expected_record:
            raise ManifestContractError(f"Parent evidence record drifted: {name}")

    integrity = payload.get("integrity")
    artifact_count = len(DEFAULT_ARTIFACTS)
    expected_integrity = {
        "artifact_count": artifact_count,
        "required_artifact_count": artifact_count,
        "present_required_artifact_count": artifact_count,
        "parent_evidence_count": len(PARENT_EVIDENCE_PATHS),
        "verified_parent_evidence_count": len(PARENT_EVIDENCE_PATHS),
        "blockers": [],
    }
    if integrity != expected_integrity:
        raise ManifestContractError("Implementation manifest integrity summary drifted")

    return {
        "schema": MANIFEST_SCHEMA,
        "loop_id": LOOP_ID,
        "implementation_manifest_sha256": manifest_sha256,
        "required_artifacts_verified": len(verified_sha256),
        "parent_evidence_verified": len(PARENT_EVIDENCE_PATHS),
        "parent_evidence_sha256": {name: verified_sha256[name] for name in PARENT_EVIDENCE_PATHS},
        "decision": VERIFIED_DECISION,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the fixed hash-only Loop28 parity implementation manifest."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly replace the existing fixed implementation manifest.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_implementation_manifest(
            project_root=args.project_root,
            replace=args.replace,
        )
    except (FileExistsError, ManifestContractError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2

    summary = {
        "schema": manifest["schema"],
        "loop_id": manifest["loop_id"],
        "artifact_count": manifest["integrity"]["artifact_count"],
        "blocker_count": len(manifest["integrity"]["blockers"]),
        "decision": manifest["decision"],
        "output": DEFAULT_OUTPUT.as_posix(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if manifest["decision"] == READY_DECISION else 2


if __name__ == "__main__":
    raise SystemExit(main())
