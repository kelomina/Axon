#!/usr/bin/env python3
"""Fail-closed raw-file replay contract for the Loop151 research chain.

This tool deliberately separates three claims:

* a bounded train-only raw-file smoke check;
* Python/native parity for the productized Loop28 sub-stage;
* a future, explicitly authorized, complete Loop151 replay.

The current repository cannot make the third claim because Loop127, Loop130,
Loop134, Loop136, and Loop151 are not connected as one raw-file runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA = "axon_loop151_raw_replay_receipt_v1"
TRUTH_MANIFEST_SCHEMA = "axon_roadmap_9997_truth_manifest_v1"
PICKLE_ALLOWLIST_SCHEMA = "axon_pickle_sha256_allowlist_v1"
VERIFY_AUTHORIZATION_SCHEMA = "axon_loop151_raw_replay_verify_authorization_v1"
MAX_SMOKE_SAMPLES = 8
DEFAULT_TOLERANCE = 1.0e-6

DEFAULT_TRUTH_MANIFEST = Path("manifests/roadmap_9997/p0_truth_freeze/loop151_truth_manifest.json")
DEFAULT_SPLIT_CSV = Path("reports/random_20w_split/loop127_full_duplicate_corrected_split.csv")
DEFAULT_PICKLE_ALLOWLIST = Path("manifests/roadmap_9997/p0_raw_replay/pickle_sha256_allowlist.json")
DEFAULT_RAW_REPLAY_AUTHORIZATION = Path("manifests/roadmap_9997/p0_raw_replay/authorization.json")
DEFAULT_CHECKPOINT = Path("models/random_20w_8192/best_model.pt")
DEFAULT_PYTHON_STAGE2 = Path(
    "reports/random_20w_split/stage2_loop28_content_pe_valonly/stage2_selected_model.pkl"
)
DEFAULT_PYTHON_STAGE2_METADATA = Path(
    "manifests/roadmap_9997/p0_raw_replay/loop28_stage2.metadata.json"
)
DEFAULT_NATIVE_SELFTEST = Path("tools/axon_onnx_dll/build/bin/Release/axon_onnx_selftest.exe")
DEFAULT_NATIVE_DLL = Path("tools/axon_onnx_dll/build/bin/Release/axon_onnx_predict.dll")
DEFAULT_NATIVE_ONNX = Path("models/random_20w_8192/axon_loop28_base.onnx")
DEFAULT_NATIVE_STAGE2 = Path("models/random_20w_8192/loop28_stage2_hgb.json")


class ReplayContractError(ValueError):
    """Raised when a fail-closed replay contract is violated."""


@dataclass(frozen=True)
class SampleIdentity:
    source_path: str
    source_sha256: str
    sample_index: int
    label: int
    split: str


@dataclass(frozen=True)
class DagStageSpec:
    stage_id: str
    role: str
    artifacts: tuple[Path, ...]
    raw_runtime_state: str
    blocker: Optional[str] = None
    frozen_parameters: tuple[tuple[str, object], ...] = ()


DAG_STAGE_SPECS = (
    DagStageSpec(
        "loop28_python",
        "Python base checkpoint plus Stage-2 HGB",
        (
            DEFAULT_CHECKPOINT,
            DEFAULT_PYTHON_STAGE2,
            DEFAULT_PYTHON_STAGE2_METADATA,
            Path("src/predict_api.py"),
            DEFAULT_PICKLE_ALLOWLIST,
        ),
        "raw_runtime_available_when_sidecar_and_allowlist_pass",
    ),
    DagStageSpec(
        "loop28_native",
        "Native ONNX base plus JSON Stage-2 HGB",
        (
            DEFAULT_NATIVE_ONNX,
            Path("models/random_20w_8192/axon_loop28_base.onnx.data"),
            DEFAULT_NATIVE_STAGE2,
            Path("tools/axon_onnx_dll/src/axon_onnx_predict.cpp"),
        ),
        "raw_runtime_available",
    ),
    DagStageSpec(
        "loop127_primary_and_content",
        "OOF fixed-v2 primary/conservative experts and content-cross expert",
        (
            Path(
                "reports/phase3_loop127/oof_fixed_v2_all_valonly_with_logreg/"
                "stage2_oof_stacker_selected_model.pkl"
            ),
            Path(
                "reports/phase3_loop127/oof_fixed_v2_all_valonly_no_logreg/"
                "stage2_oof_stacker_selected_model.pkl"
            ),
            Path(
                "reports/phase3_loop127/phase1_content_cross_hgb_local_valonly/"
                "loop43_content_cross_selected_model.pkl"
            ),
            Path("scripts/evaluate_stage2_oof_stacker.py"),
            Path("scripts/evaluate_stage2_cache_model.py"),
        ),
        "historical_artifacts_only",
        "No per-file raw adapter rebuilds all Loop127 expert features without historical caches.",
        (
            ("primary_threshold", 0.31),
            ("conservative_threshold", 0.415),
            ("content_cross_threshold", 0.4),
        ),
    ),
    DagStageSpec(
        "loop130_r5",
        "Frozen R5_r4_plus_vendor_strings guard",
        (
            Path("scripts/evaluate_loop130_content_string_guard_rules.py"),
            Path("reports/phase3_loop128/loop130_content_string_guard_val_eval.json"),
        ),
        "csv_evaluator_only",
        "The evaluator consumes upstream prediction CSVs and cache sidecars, not one raw file.",
        (("selected_rule", "R5_r4_plus_vendor_strings"),),
    ),
    DagStageSpec(
        "loop134_oof_noise",
        "Noise-aware OOF stacker",
        (
            Path(
                "reports/phase3_loop134/oof_fixed_v2_string_noise_valonly/"
                "stage2_oof_stacker_selected_model.pkl"
            ),
            Path("scripts/evaluate_stage2_oof_stacker.py"),
        ),
        "historical_artifact_and_csv_evaluator",
        "The frozen stacker has no guarded per-file raw inference adapter.",
        (("threshold", 0.39),),
    ),
    DagStageSpec(
        "loop136_selector",
        "Recall-aware pairwise selector at threshold 0.79",
        (
            Path(
                "reports/phase3_loop136/r5_oof_noise_pairwise_selector_recall_valonly/"
                "loop135_pairwise_selector.pkl"
            ),
            Path("scripts/evaluate_loop135_pairwise_selector.py"),
        ),
        "historical_artifact_and_csv_evaluator",
        "The selector depends on Loop130/Loop134 prediction rows and feature sidecars.",
        (("threshold", 0.79),),
    ),
    DagStageSpec(
        "loop151_trusted_signer",
        "Frozen Authenticode trusted-signer guard",
        (
            Path("scripts/evaluate_authenticode_trusted_signer_guard.py"),
            Path("reports/phase3_loop151/loop151_trusted_signer_guard_val_eval.json"),
        ),
        "csv_evaluator_only",
        "No connected runtime invokes Authenticode after a raw Loop136 decision with frozen provenance.",
        (("score_threshold", 1.0),),
    ),
)


def resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def resolve_within(project_root: Path, path: Path, *, purpose: str) -> Path:
    resolved_root = project_root.resolve()
    resolved = resolve_path(resolved_root, path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ReplayContractError(f"{purpose} escapes project root: {resolved}") from exc
    return resolved


def resolve_frozen_a1_artifact(
    project_root: Path,
    requested_path: Path,
    frozen_path: Path,
    *,
    purpose: str,
) -> Path:
    requested = resolve_within(project_root, requested_path, purpose=purpose)
    frozen = resolve_within(project_root, frozen_path, purpose=f"Frozen {purpose}")
    if requested != frozen:
        raise ReplayContractError(
            f"A1 {purpose} override is not authorized: {requested} != {frozen}"
        )
    return requested


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, expected_schema: Optional[str] = None) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ReplayContractError(f"Required JSON is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReplayContractError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReplayContractError(f"JSON root must be an object: {path}")
    if expected_schema is not None and payload.get("schema") != expected_schema:
        raise ReplayContractError(
            f"Unsupported schema for {path}: {payload.get('schema')!r}; expected {expected_schema!r}"
        )
    return payload


def _relative_display(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _artifact_record(project_root: Path, path: Path) -> dict:
    resolved = resolve_within(project_root, path, purpose="DAG artifact")
    exists = resolved.is_file()
    return {
        "path": _relative_display(project_root, resolved),
        "exists": exists,
        "size_bytes": resolved.stat().st_size if exists else None,
        "sha256": file_sha256(resolved) if exists else None,
    }


def verify_truth_manifest(project_root: Path, manifest_path: Path) -> tuple[dict, dict]:
    resolved = resolve_within(project_root, manifest_path, purpose="Truth manifest")
    payload = _read_json_object(resolved, TRUTH_MANIFEST_SCHEMA)
    integrity = payload.get("integrity")
    if not isinstance(integrity, dict) or not integrity.get("artifact_freeze_complete"):
        raise ReplayContractError("Truth manifest artifact freeze is not complete")
    if integrity.get("blockers"):
        raise ReplayContractError(f"Truth manifest has blockers: {integrity['blockers']}")

    # 重放必须重新核对冻结 artifact，不能只相信 manifest 自报的旧状态。
    mismatches: list[str] = []
    verified_count = 0
    for artifact in payload.get("artifacts", []):
        if not isinstance(artifact, dict) or not artifact.get("required"):
            continue
        artifact_path = resolve_within(
            project_root,
            Path(str(artifact.get("path") or "")),
            purpose=f"Truth artifact {artifact.get('name')}",
        )
        expected_sha = str(artifact.get("sha256") or "").casefold()
        if not artifact_path.is_file():
            mismatches.append(f"missing:{artifact.get('name')}:{artifact_path}")
            continue
        actual_sha = file_sha256(artifact_path)
        if not _is_sha256(expected_sha) or actual_sha != expected_sha:
            mismatches.append(f"sha256_mismatch:{artifact.get('name')}:{artifact_path}")
            continue
        verified_count += 1
    if mismatches:
        raise ReplayContractError(f"Truth manifest artifact verification failed: {mismatches[:8]}")
    return payload, {
        "path": _relative_display(project_root, resolved),
        "sha256": file_sha256(resolved),
        "required_artifacts_verified": verified_count,
    }


def load_pickle_allowlist(project_root: Path, allowlist_path: Path) -> tuple[dict, dict[str, dict]]:
    resolved = resolve_within(project_root, allowlist_path, purpose="Pickle allowlist")
    payload = _read_json_object(resolved, PICKLE_ALLOWLIST_SCHEMA)
    entries: dict[str, dict] = {}
    for entry in payload.get("entries", []):
        if not isinstance(entry, dict):
            raise ReplayContractError("Pickle allowlist entries must be objects")
        raw_path = Path(str(entry.get("path") or ""))
        path = resolve_within(project_root, raw_path, purpose="Allowlisted pickle")
        sha256 = str(entry.get("sha256") or "").strip().casefold()
        if not _is_sha256(sha256):
            raise ReplayContractError(f"Invalid allowlisted pickle SHA-256: {raw_path}")
        key = os.path.normcase(str(path))
        if key in entries:
            raise ReplayContractError(f"Duplicate pickle allowlist path: {raw_path}")
        entries[key] = {**entry, "resolved_path": path, "sha256": sha256}
    return payload, entries


def verify_a1_authorization(
    project_root: Path,
    *,
    mode: str,
    max_samples: int,
    allowed_raw_root: Path,
) -> dict:
    path = resolve_within(
        project_root,
        DEFAULT_RAW_REPLAY_AUTHORIZATION,
        purpose="Raw replay authorization",
    )
    payload = _read_json_object(path, "axon_roadmap_9997_authorization_v1")
    if payload.get("loop_id") != "p0_raw_replay_001":
        raise ReplayContractError("Raw replay authorization loop_id mismatch")
    if payload.get("authorization_level") != "A1_scoped_change":
        raise ReplayContractError("Raw replay smoke requires A1_scoped_change authorization")
    if payload.get("allowed_splits") != ["train"]:
        raise ReplayContractError("Raw replay A1 authorization must be train-only")
    configured_logical_root_value = str(payload.get("allowed_logical_raw_root") or "").strip()
    if not configured_logical_root_value:
        raise ReplayContractError("Raw replay A1 authorization has no logical raw root")
    configured_logical_root = Path(configured_logical_root_value)
    authorized_logical_root = resolve_within(
        project_root,
        configured_logical_root,
        purpose="Authorized logical raw root",
    )
    if not authorized_logical_root.is_dir():
        raise ReplayContractError(
            f"Authorized logical raw root is missing: {authorized_logical_root}"
        )
    if allowed_raw_root.resolve() != authorized_logical_root:
        raise ReplayContractError(
            "Requested logical raw root does not match A1 authorization: "
            f"{allowed_raw_root.resolve()} != {authorized_logical_root}"
        )
    if max_samples < 1 or max_samples > int(payload.get("max_raw_files") or 0):
        raise ReplayContractError("Requested sample count exceeds A1 raw replay authorization")
    mode_flag = {
        "identity-smoke": "allow_train_identity_smoke",
        "native-smoke": "allow_native_loop28_smoke",
        "native-parity": "allow_python_native_loop28_parity",
    }.get(mode)
    if mode_flag is None or payload.get(mode_flag) is not True:
        raise ReplayContractError(f"Raw replay mode is not authorized: {mode}")
    if mode in {"native-smoke", "native-parity"} and max_samples > int(
        payload.get("max_native_predictions") or 0
    ):
        raise ReplayContractError("Requested native predictions exceed A1 authorization")
    allowed_resolved_roots = []
    for value in payload.get("allowed_resolved_raw_roots", []):
        root = resolve_source_path(str(value)).resolve()
        if not root.is_dir():
            raise ReplayContractError(f"Authorized resolved raw root is missing: {root}")
        allowed_resolved_roots.append(str(root))
    return {
        "path": _relative_display(project_root, path),
        "sha256": file_sha256(path),
        "loop_id": payload.get("loop_id"),
        "authorization_level": payload.get("authorization_level"),
        "mode": mode,
        "allowed_split": "train",
        "max_samples": max_samples,
        "allowed_logical_raw_root": str(authorized_logical_root),
        "allowed_resolved_raw_roots": allowed_resolved_roots,
        "status": "authorized",
    }


def guard_pickle_before_load(
    project_root: Path,
    model_path: Path,
    allowlist_path: Path,
) -> dict:
    """Validate path, digest, and metadata before any pickle loader is called."""

    resolved_model = resolve_within(project_root, model_path, purpose="Pickle model")
    allowlist_payload, entries = load_pickle_allowlist(project_root, allowlist_path)
    entry = entries.get(os.path.normcase(str(resolved_model)))
    if entry is None:
        raise ReplayContractError(f"Pickle is not allowlisted: {resolved_model}")
    if entry.get("load_authorized") is not True:
        raise ReplayContractError(f"Pickle allowlist entry is inventory-only: {resolved_model}")
    if not resolved_model.is_file():
        raise ReplayContractError(f"Allowlisted pickle is missing: {resolved_model}")
    actual_sha = file_sha256(resolved_model)
    if actual_sha != entry["sha256"]:
        raise ReplayContractError(
            f"Pickle SHA-256 mismatch: {resolved_model}; expected {entry['sha256']}, got {actual_sha}"
        )

    raw_metadata_path = entry.get("metadata_path")
    if not raw_metadata_path:
        raise ReplayContractError(f"Allowlisted pickle has no metadata_path: {resolved_model}")
    metadata_path = resolve_within(
        project_root,
        Path(str(raw_metadata_path)),
        purpose="Pickle metadata",
    )
    expected_metadata_sha = str(entry.get("metadata_sha256") or "").strip().casefold()
    if not _is_sha256(expected_metadata_sha):
        raise ReplayContractError(
            f"Allowlisted pickle has no valid metadata SHA-256: {resolved_model}"
        )
    metadata = _read_json_object(metadata_path, "axon_stage2_model_metadata_v1")
    actual_metadata_sha = file_sha256(metadata_path)
    if actual_metadata_sha != expected_metadata_sha:
        raise ReplayContractError(
            f"Pickle metadata SHA-256 mismatch: {metadata_path}; "
            f"expected {expected_metadata_sha}, got {actual_metadata_sha}"
        )
    if str(metadata.get("model_sha256") or "").strip().casefold() != actual_sha:
        raise ReplayContractError(
            f"Pickle metadata does not bind the allowlisted model SHA-256: {metadata_path}"
        )
    if bool((metadata.get("knn") or {}).get("enabled")):
        raise ReplayContractError(
            "Loop28 Python parity does not support Stage-2 kNN pickle payloads"
        )
    return {
        "allowlist_path": _relative_display(
            project_root,
            resolve_within(project_root, allowlist_path, purpose="Pickle allowlist"),
        ),
        "allowlist_sha256": file_sha256(
            resolve_within(project_root, allowlist_path, purpose="Pickle allowlist")
        ),
        "allowlist_contract": allowlist_payload.get("contract"),
        "model_path": _relative_display(project_root, resolved_model),
        "model_sha256": actual_sha,
        "metadata_path": _relative_display(project_root, metadata_path),
        "metadata_sha256": actual_metadata_sha,
        "status": "verified_before_unpickle",
    }


def read_split_samples(
    split_csv: Path,
    *,
    requested_split: str,
    max_samples: int,
) -> tuple[list[SampleIdentity], dict]:
    if requested_split != "train":
        raise ReplayContractError(
            f"A1 raw replay smoke is train-only; requested split was {requested_split!r}"
        )
    if max_samples < 1 or max_samples > MAX_SMOKE_SAMPLES:
        raise ReplayContractError(
            f"max_samples must be in [1, {MAX_SMOKE_SAMPLES}] for train-only smoke"
        )

    required = {"source_path", "source_sha256", "label", "sample_index", "split"}
    selected: list[SampleIdentity] = []
    identities: dict[tuple[str, int], tuple[int, str]] = {}
    labels_by_sha: dict[str, int] = {}
    rows_scanned = 0
    split_counts: dict[str, int] = {}
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ReplayContractError(f"Split CSV is missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            rows_scanned += 1
            source_sha = str(row.get("source_sha256") or "").strip().casefold()
            if not _is_sha256(source_sha):
                raise ReplayContractError(f"Invalid source_sha256 at split row {row_number}")
            try:
                sample_index = int(str(row.get("sample_index") or "").strip())
                label = int(str(row.get("label") or "").strip())
            except ValueError as exc:
                raise ReplayContractError(
                    f"Invalid sample_index or label at split row {row_number}"
                ) from exc
            if sample_index < 0 or label not in (0, 1):
                raise ReplayContractError(f"Invalid identity values at split row {row_number}")
            split = str(row.get("split") or "").strip().casefold()
            if not split:
                raise ReplayContractError(f"Missing split at row {row_number}")
            split_counts[split] = split_counts.get(split, 0) + 1
            identity_key = (source_sha, sample_index)
            if identity_key in identities:
                raise ReplayContractError(
                    f"Duplicate (source_sha256, sample_index) at split row {row_number}: {identity_key}"
                )
            identities[identity_key] = (label, split)
            previous_label = labels_by_sha.setdefault(source_sha, label)
            if previous_label != label:
                raise ReplayContractError(
                    f"Conflicting labels for source_sha256 at split row {row_number}: {source_sha}"
                )
            source_path = str(row.get("source_path") or "").strip()
            if not source_path:
                raise ReplayContractError(f"Missing source_path at split row {row_number}")
            if split == requested_split and len(selected) < max_samples:
                selected.append(
                    SampleIdentity(
                        source_path=source_path,
                        source_sha256=source_sha,
                        sample_index=sample_index,
                        label=label,
                        split=split,
                    )
                )
    if len(selected) != max_samples:
        raise ReplayContractError(
            f"Requested {max_samples} {requested_split} samples, found {len(selected)}"
        )
    identity_digest = hashlib.sha256()
    for (source_sha, sample_index), (label, split) in sorted(identities.items()):
        identity_digest.update(f"{source_sha},{sample_index},{label},{split}\n".encode())
    selected_digest = hashlib.sha256()
    for sample in sorted(selected, key=lambda item: (item.source_sha256, item.sample_index)):
        selected_digest.update(
            f"{sample.source_sha256},{sample.sample_index},{sample.label},{sample.split}\n".encode()
        )
    return selected, {
        "rows_scanned": rows_scanned,
        "unique_identity_count": len(identities),
        "unique_source_sha256_count": len(labels_by_sha),
        "identity_multiset_sha256": identity_digest.hexdigest(),
        "selected_identity_multiset_sha256": selected_digest.hexdigest(),
        "split_counts": dict(sorted(split_counts.items())),
        "selected_count": len(selected),
    }


_WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def resolve_source_path(source_path: str) -> Path:
    path = Path(source_path)
    if path.is_file():
        return path
    match = _WINDOWS_DRIVE_RE.match(source_path)
    if match and os.name != "nt":
        drive, tail = match.groups()
        components = [part for part in re.split(r"[\\/]", tail) if part]
        return Path("/mnt") / drive.casefold() / Path(*components)
    return path


def _path_is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        path_text = str(path).replace("\\", "/").casefold()
        root_text = str(root).replace("\\", "/").rstrip("/").casefold()
        return path_text == root_text or path_text.startswith(f"{root_text}/")


def _resolve_authorized_sample_path(
    sample: SampleIdentity,
    allowed_raw_root: Path,
    allowed_resolved_roots: Sequence[Path] = (),
) -> tuple[Path, Path, Path]:
    source = resolve_source_path(sample.source_path)
    logical_path = Path(os.path.abspath(str(source)))
    logical_root = Path(os.path.abspath(str(allowed_raw_root)))
    if not _path_is_inside(logical_path, logical_root):
        raise ReplayContractError(
            f"Raw logical path is outside allowed root: {logical_path} not under {logical_root}"
        )
    resolved = logical_path.resolve()
    resolved_roots = [
        allowed_raw_root.resolve(),
        *(Path(path).resolve() for path in allowed_resolved_roots),
    ]
    matching_root = next(
        (root for root in resolved_roots if _path_is_inside(resolved, root)),
        None,
    )
    if matching_root is None:
        raise ReplayContractError(f"Raw resolved path is outside authorized roots: {resolved}")
    if not resolved.is_file():
        raise ReplayContractError(f"Raw file is missing: {resolved}")
    return logical_path, resolved, matching_root


def _sample_record(
    sample: SampleIdentity,
    logical_path: Path,
    resolved_path: Path,
    matching_root: Path,
) -> dict:
    return {
        "source_sha256": sample.source_sha256,
        "sample_index": sample.sample_index,
        "label": sample.label,
        "split": sample.split,
        "logical_path": str(logical_path),
        "resolved_path": str(resolved_path),
        "matched_resolved_root": str(matching_root),
        "size_bytes": resolved_path.stat().st_size,
    }


def verify_sample_file(
    sample: SampleIdentity,
    allowed_raw_root: Path,
    allowed_resolved_roots: Sequence[Path] = (),
) -> tuple[Path, dict]:
    logical_path, resolved, matching_root = _resolve_authorized_sample_path(
        sample,
        allowed_raw_root,
        allowed_resolved_roots,
    )
    actual_sha = file_sha256(resolved)
    if actual_sha != sample.source_sha256:
        raise ReplayContractError(
            f"Raw source SHA-256 mismatch for sample_index={sample.sample_index}: "
            f"expected {sample.source_sha256}, got {actual_sha}"
        )
    record = _sample_record(sample, logical_path, resolved, matching_root)
    record.update({"raw_sha256_verified": True, "status": "identity_verified"})
    return resolved, record


def snapshot_verified_sample(
    sample: SampleIdentity,
    *,
    allowed_raw_root: Path,
    allowed_resolved_roots: Sequence[Path],
    snapshot_root: Path,
) -> tuple[Path, dict]:
    logical_path, resolved, matching_root = _resolve_authorized_sample_path(
        sample,
        allowed_raw_root,
        allowed_resolved_roots,
    )
    snapshot_path = snapshot_root / f"{sample.sample_index}-{sample.source_sha256}.bin"
    digest = hashlib.sha256()
    size_bytes = 0
    # 同一字节流一边哈希一边写入私有快照；后续 runtime 只读取这个快照。
    with resolved.open("rb") as source, snapshot_path.open("xb") as destination:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            destination.write(chunk)
            size_bytes += len(chunk)
    actual_sha = digest.hexdigest()
    if actual_sha != sample.source_sha256:
        snapshot_path.unlink(missing_ok=True)
        raise ReplayContractError(
            f"Raw source SHA-256 mismatch for sample_index={sample.sample_index}: "
            f"expected {sample.source_sha256}, got {actual_sha}"
        )
    record = _sample_record(sample, logical_path, resolved, matching_root)
    record.update(
        {
            "size_bytes": size_bytes,
            "raw_sha256_verified": True,
            "snapshot_path": str(snapshot_path),
            "snapshot_sha256": actual_sha,
            "status": "identity_snapshot_verified",
        }
    )
    return snapshot_path, record


def build_stage_contract(project_root: Path) -> tuple[list[dict], list[str]]:
    stages: list[dict] = []
    blockers: list[str] = []
    try:
        _allowlist_payload, pickle_entries = load_pickle_allowlist(
            project_root,
            DEFAULT_PICKLE_ALLOWLIST,
        )
        allowlist_error = None
    except ReplayContractError as exc:
        pickle_entries = {}
        allowlist_error = str(exc)
    for spec in DAG_STAGE_SPECS:
        artifacts = [_artifact_record(project_root, path) for path in spec.artifacts]
        missing = [row["path"] for row in artifacts if not row["exists"]]
        stage_blockers = []
        pickle_policy = []
        if missing:
            stage_blockers.append(f"missing_artifacts:{missing}")
        for artifact_path, artifact in zip(spec.artifacts, artifacts):
            if artifact_path.suffix.casefold() != ".pkl":
                continue
            resolved_pickle = resolve_within(
                project_root,
                artifact_path,
                purpose="DAG pickle artifact",
            )
            allowlist_entry = pickle_entries.get(os.path.normcase(str(resolved_pickle)))
            policy_status = "verified_inventory"
            if allowlist_error:
                policy_status = "allowlist_invalid"
                stage_blockers.append(f"pickle_allowlist_invalid:{allowlist_error}")
            elif allowlist_entry is None:
                policy_status = "not_allowlisted"
                stage_blockers.append(f"pickle_not_allowlisted:{artifact['path']}")
            elif artifact["sha256"] != allowlist_entry["sha256"]:
                policy_status = "sha256_mismatch"
                stage_blockers.append(f"pickle_allowlist_sha256_mismatch:{artifact['path']}")
            elif spec.raw_runtime_state.startswith("raw_runtime_available") and (
                allowlist_entry.get("load_authorized") is not True
            ):
                policy_status = "inventory_only"
                stage_blockers.append(f"pickle_load_not_authorized:{artifact['path']}")
            pickle_policy.append(
                {
                    "path": artifact["path"],
                    "status": policy_status,
                    "load_authorized": (
                        allowlist_entry.get("load_authorized")
                        if allowlist_entry is not None
                        else False
                    ),
                }
            )
        if spec.blocker:
            stage_blockers.append(spec.blocker)
        executable = (
            spec.raw_runtime_state.startswith("raw_runtime_available") and not stage_blockers
        )
        stages.append(
            {
                "stage_id": spec.stage_id,
                "schema_version": RECEIPT_SCHEMA,
                "role": spec.role,
                "frozen_parameters": dict(spec.frozen_parameters),
                "raw_runtime_state": spec.raw_runtime_state,
                "raw_executable": executable,
                "status": "implementation_available_not_verified" if executable else "blocked",
                "artifacts": artifacts,
                "pickle_policy": pickle_policy,
                "run_id": None,
                "source_bundle_sha256": None,
                "input_identity_fingerprint": None,
                "output_identity_fingerprint": None,
                "input_rows": None,
                "output_rows": None,
                "unresolved_rows": None,
                "duplicate_rows": None,
                "missing_rows": None,
                "blockers": stage_blockers,
            }
        )
        blockers.extend(f"{spec.stage_id}:{item}" for item in stage_blockers)
    return stages, blockers


def _windows_cli_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name == "nt":
        return str(resolved)
    parts = resolved.parts
    if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper()
        return f"{drive}:\\" + "\\".join(parts[3:])
    return str(resolved)


def parse_native_selftest_output(stdout: str) -> tuple[dict, dict]:
    objects: list[dict] = []
    for line in stdout.splitlines():
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    if len(objects) < 2:
        raise ReplayContractError("Native selftest did not emit validation and prediction JSON")
    validation, prediction = objects[0], objects[-1]
    if validation.get("validate_rc") != 0:
        raise ReplayContractError(f"Native model validation failed: {validation}")
    if prediction.get("ok") is not True:
        raise ReplayContractError(f"Native prediction failed: {prediction}")
    return validation, prediction


def run_native_loop28(
    *,
    project_root: Path,
    sample_path: Path,
    allowed_raw_root: Path,
    selftest_path: Path,
    dll_path: Path,
    onnx_path: Path,
    stage2_path: Path,
    timeout_seconds: int,
) -> dict:
    selftest_path = resolve_within(project_root, selftest_path, purpose="Native selftest")
    dll_path = resolve_within(project_root, dll_path, purpose="Native DLL")
    onnx_path = resolve_within(project_root, onnx_path, purpose="Native ONNX model")
    stage2_path = resolve_within(project_root, stage2_path, purpose="Native Stage-2 model")
    artifacts = [selftest_path, dll_path, onnx_path, stage2_path]
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise ReplayContractError(f"Native Loop28 artifacts are missing: {missing}")
    command = [
        _windows_cli_path(selftest_path),
        "--dll",
        _windows_cli_path(dll_path),
        "--onnx",
        _windows_cli_path(onnx_path),
        "--target",
        _windows_cli_path(sample_path),
        "--allowed_root",
        _windows_cli_path(allowed_raw_root),
        "--stage2",
        _windows_cli_path(stage2_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise ReplayContractError(
            f"Native selftest failed with code {completed.returncode}: "
            f"stdout={completed.stdout[-2000:]!r}, stderr={completed.stderr[-2000:]!r}"
        )
    validation, prediction = parse_native_selftest_output(completed.stdout)
    return {
        "runtime": "native_loop28",
        "command": command,
        "returncode": completed.returncode,
        "validation": validation,
        "prediction": prediction,
    }


def run_python_loop28(
    *,
    project_root: Path,
    sample_path: Path,
    checkpoint_path: Path,
    stage2_path: Path,
    pickle_allowlist_path: Path,
) -> dict:
    pickle_guard = guard_pickle_before_load(project_root, stage2_path, pickle_allowlist_path)
    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from predict_api import PredictRequest, predict_file  # noqa: PLC0415

    response = predict_file(
        PredictRequest(
            file=str(sample_path),
            checkpoint=str(checkpoint_path),
            device="cpu",
            stage2_model=str(stage2_path),
            family_classifier="",
        )
    )
    if response.get("ok") is not True:
        raise ReplayContractError(f"Python Loop28 prediction failed: {response}")
    result = response.get("result")
    if not isinstance(result, dict):
        raise ReplayContractError("Python Loop28 response is missing result object")
    return {
        "runtime": "python_loop28",
        "pickle_guard": pickle_guard,
        "prediction": result,
    }


def normalize_loop28_prediction(payload: Mapping[str, object]) -> dict:
    base = payload.get("base_model")
    stage2 = payload.get("stage2")
    if not isinstance(base, Mapping) or not isinstance(stage2, Mapping):
        raise ReplayContractError("Loop28 prediction must include base_model and stage2 objects")
    try:
        normalized = {
            "prediction": int(payload["prediction"]),
            "prob_malicious": float(payload["prob_malicious"]),
            "base_prediction": int(base["prediction"]),
            "base_prob_malicious": float(base["prob_malicious"]),
            "stage2_prob_malicious": float(stage2["prob_malicious"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayContractError("Loop28 prediction has invalid probability fields") from exc
    if normalized["prediction"] not in (0, 1) or normalized["base_prediction"] not in (0, 1):
        raise ReplayContractError("Loop28 prediction contains a non-binary decision")
    for key in ("prob_malicious", "base_prob_malicious", "stage2_prob_malicious"):
        if not 0.0 <= normalized[key] <= 1.0:
            raise ReplayContractError(f"Loop28 probability is outside [0, 1]: {key}")
    return normalized


def compare_loop28_predictions(
    python_prediction: Mapping[str, object],
    native_prediction: Mapping[str, object],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict:
    if not math.isfinite(tolerance) or tolerance <= 0 or tolerance > DEFAULT_TOLERANCE:
        raise ReplayContractError(
            f"Parity tolerance must be finite and <= frozen {DEFAULT_TOLERANCE}"
        )
    python = normalize_loop28_prediction(python_prediction)
    native = normalize_loop28_prediction(native_prediction)
    probability_deltas = {
        key: abs(python[key] - native[key])
        for key in ("prob_malicious", "base_prob_malicious", "stage2_prob_malicious")
    }
    decision_match = python["prediction"] == native["prediction"]
    base_decision_match = python["base_prediction"] == native["base_prediction"]
    probability_match = all(delta <= tolerance for delta in probability_deltas.values())
    return {
        "tolerance": tolerance,
        "python": python,
        "native": native,
        "decision_match": decision_match,
        "base_decision_match": base_decision_match,
        "probability_deltas": probability_deltas,
        "probability_match": probability_match,
        "passed": decision_match and base_decision_match and probability_match,
    }


def _base_receipt(
    *,
    mode: str,
    project_root: Path,
    truth_manifest: dict,
    truth_record: dict,
    split_csv: Optional[Path],
    stages: list[dict],
    stage_blockers: list[str],
) -> dict:
    return {
        "schema": RECEIPT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "claim_scope": {
            "quality_claim_allowed": False,
            "legacy_test10k_or_full_test_used": False,
            "legacy_metrics_role": "development_only",
            "raw_file_to_legacy_report_replay_ready": False,
            "raw_file_to_loop151_replay_ready": False,
            "native_loop151_ready": False,
        },
        "truth_manifest": {
            **truth_record,
            "decision": truth_manifest.get("decision"),
            "champion_scope": (truth_manifest.get("contract") or {}).get("champion_scope"),
        },
        "inputs": {
            "split_csv": _relative_display(project_root, split_csv) if split_csv else None,
            "split_csv_sha256": file_sha256(split_csv)
            if split_csv and split_csv.is_file()
            else None,
        },
        "dag": {
            "ordered_stages": [stage["stage_id"] for stage in stages],
            "stages": stages,
            "full_dag_blockers": stage_blockers,
        },
    }


def build_smoke_receipt(
    *,
    project_root: Path,
    truth_manifest_path: Path,
    split_csv: Path,
    allowed_raw_root: Path,
    max_samples: int,
    runtime: str,
    native_runner: Optional[Callable[[Path, Path], dict]] = None,
) -> dict:
    authorization_record = verify_a1_authorization(
        project_root,
        mode="native-smoke" if runtime == "native" else "identity-smoke",
        max_samples=max_samples,
        allowed_raw_root=allowed_raw_root,
    )
    truth_manifest, truth_record = verify_truth_manifest(project_root, truth_manifest_path)
    samples, identity_audit = read_split_samples(
        split_csv,
        requested_split="train",
        max_samples=max_samples,
    )
    stages, stage_blockers = build_stage_contract(project_root)
    receipt = _base_receipt(
        mode="smoke",
        project_root=project_root,
        truth_manifest=truth_manifest,
        truth_record=truth_record,
        split_csv=split_csv,
        stages=stages,
        stage_blockers=stage_blockers,
    )
    receipt["inputs"].update(
        {
            "requested_split": "train",
            "max_samples": max_samples,
            "allowed_raw_root": str(allowed_raw_root.resolve()),
            "runtime": runtime,
        }
    )
    receipt["authorization"] = authorization_record
    receipt["identity_audit"] = identity_audit
    receipt["samples"] = []
    execution_blockers: list[str] = []
    allowed_resolved_roots = [
        Path(path) for path in authorization_record["allowed_resolved_raw_roots"]
    ]
    with tempfile.TemporaryDirectory(prefix="axon-loop151-replay-") as temp_dir:
        snapshot_root = Path(temp_dir)
        for sample in samples:
            sample_record = {
                "source_sha256": sample.source_sha256,
                "sample_index": sample.sample_index,
                "label": sample.label,
                "split": sample.split,
            }
            try:
                path, identity_record = snapshot_verified_sample(
                    sample,
                    allowed_raw_root=allowed_raw_root,
                    allowed_resolved_roots=allowed_resolved_roots,
                    snapshot_root=snapshot_root,
                )
                sample_record.update(identity_record)
                if runtime == "native":
                    if native_runner is None:
                        raise ReplayContractError(
                            "Native runtime was requested without a native runner"
                        )
                    sample_record["native_loop28"] = native_runner(path, snapshot_root)
                    sample_record["status"] = "native_loop28_smoke_passed"
                elif runtime != "identity":
                    raise ReplayContractError(f"Unsupported smoke runtime: {runtime}")
                receipt["samples"].append(sample_record)
            except Exception as exc:  # noqa: BLE001 - receipt must preserve every requested row.
                execution_blockers.append(
                    f"sample_index={sample.sample_index}:{type(exc).__name__}:{exc}"
                )
                sample_record["status"] = "blocked"
                sample_record["error"] = str(exc)
                receipt["samples"].append(sample_record)
    receipt["execution"] = {
        "requested_count": len(samples),
        "reported_count": len(receipt["samples"]),
        "dropped_row_count": len(samples) - len(receipt["samples"]),
        "blockers": execution_blockers,
    }
    if execution_blockers:
        receipt["decision"] = "train_smoke_blocked"
    elif runtime == "native":
        receipt["decision"] = "native_loop28_train_smoke_passed_full_loop151_blocked"
    else:
        receipt["decision"] = "train_identity_smoke_passed_full_loop151_blocked"
    return receipt


def build_native_parity_receipt(
    *,
    project_root: Path,
    truth_manifest_path: Path,
    split_csv: Path,
    allowed_raw_root: Path,
    max_samples: int,
    tolerance: float,
    python_runner: Callable[[Path], dict],
    native_runner: Callable[[Path, Path], dict],
) -> dict:
    if tolerance != DEFAULT_TOLERANCE:
        raise ReplayContractError(
            f"native-parity requires the frozen tolerance {DEFAULT_TOLERANCE}"
        )
    authorization_record = verify_a1_authorization(
        project_root,
        mode="native-parity",
        max_samples=max_samples,
        allowed_raw_root=allowed_raw_root,
    )
    truth_manifest, truth_record = verify_truth_manifest(project_root, truth_manifest_path)
    samples, identity_audit = read_split_samples(
        split_csv,
        requested_split="train",
        max_samples=max_samples,
    )
    stages, stage_blockers = build_stage_contract(project_root)
    receipt = _base_receipt(
        mode="native-parity",
        project_root=project_root,
        truth_manifest=truth_manifest,
        truth_record=truth_record,
        split_csv=split_csv,
        stages=stages,
        stage_blockers=stage_blockers,
    )
    receipt["inputs"].update(
        {
            "requested_split": "train",
            "max_samples": max_samples,
            "allowed_raw_root": str(allowed_raw_root.resolve()),
            "probability_tolerance": tolerance,
        }
    )
    receipt["authorization"] = authorization_record
    receipt["identity_audit"] = identity_audit
    receipt["samples"] = []
    blockers: list[str] = []
    allowed_resolved_roots = [
        Path(path) for path in authorization_record["allowed_resolved_raw_roots"]
    ]
    with tempfile.TemporaryDirectory(prefix="axon-loop151-parity-") as temp_dir:
        snapshot_root = Path(temp_dir)
        for sample in samples:
            row = {
                "source_sha256": sample.source_sha256,
                "sample_index": sample.sample_index,
                "label": sample.label,
                "split": sample.split,
            }
            try:
                path, identity_record = snapshot_verified_sample(
                    sample,
                    allowed_raw_root=allowed_raw_root,
                    allowed_resolved_roots=allowed_resolved_roots,
                    snapshot_root=snapshot_root,
                )
                row.update(identity_record)
                python_result = python_runner(path)
                native_result = native_runner(path, snapshot_root)
                parity = compare_loop28_predictions(
                    python_result["prediction"],
                    native_result["prediction"],
                    tolerance=tolerance,
                )
                row["python_runtime"] = python_result
                row["native_runtime"] = native_result
                row["parity"] = parity
                row["status"] = "parity_passed" if parity["passed"] else "parity_failed"
                if not parity["passed"]:
                    blockers.append(
                        f"sample_index={sample.sample_index}:parity_tolerance_or_decision_failure"
                    )
            except Exception as exc:  # noqa: BLE001 - fail closed and preserve the identity row.
                row["status"] = "blocked"
                row["error"] = str(exc)
                blockers.append(f"sample_index={sample.sample_index}:{type(exc).__name__}:{exc}")
            receipt["samples"].append(row)
    receipt["parity"] = {
        "requested_count": len(samples),
        "reported_count": len(receipt["samples"]),
        "dropped_row_count": len(samples) - len(receipt["samples"]),
        "passed_count": sum(row.get("status") == "parity_passed" for row in receipt["samples"]),
        "blockers": blockers,
    }
    receipt["decision"] = (
        "native_loop28_parity_passed_full_loop151_blocked"
        if not blockers
        else "native_loop28_parity_blocked"
    )
    return receipt


def build_verify_receipt(
    *,
    project_root: Path,
    truth_manifest_path: Path,
    authorization_path: Path,
) -> dict:
    # 完整重放先验授权必须在模型、pickle 或原始样本被打开前通过。
    resolved_truth = resolve_within(project_root, truth_manifest_path, purpose="Truth manifest")
    truth_manifest = _read_json_object(resolved_truth, TRUTH_MANIFEST_SCHEMA)
    truth_record = {
        "path": _relative_display(project_root, resolved_truth),
        "sha256": file_sha256(resolved_truth),
        "required_artifacts_verified": 0,
    }
    authorization_record: dict
    authorization_blockers: list[str] = []
    try:
        resolved_authorization = resolve_within(
            project_root,
            authorization_path,
            purpose="Verify authorization",
        )
        authorization = _read_json_object(
            resolved_authorization,
            VERIFY_AUTHORIZATION_SCHEMA,
        )
        authorization_record = {
            "path": _relative_display(project_root, resolved_authorization),
            "sha256": file_sha256(resolved_authorization),
            "authorization_level": authorization.get("authorization_level"),
            "allow_complete_loop151_raw_replay": authorization.get(
                "allow_complete_loop151_raw_replay"
            ),
            "truth_manifest_sha256": authorization.get("truth_manifest_sha256"),
        }
        if authorization.get("allow_complete_loop151_raw_replay") is not True:
            authorization_blockers.append(
                "authorization_does_not_allow_complete_loop151_raw_replay"
            )
        if authorization.get("authorization_level") not in {
            "A2_heavy_compute",
            "A3_heldout",
        }:
            authorization_blockers.append("authorization_level_is_not_A2_or_A3")
        if authorization.get("truth_manifest_sha256") != truth_record["sha256"]:
            authorization_blockers.append("authorization_truth_manifest_sha256_mismatch")
    except ReplayContractError as exc:
        authorization_record = {
            "path": str(authorization_path),
            "status": "invalid",
            "error": str(exc),
        }
        authorization_blockers.append(f"invalid_or_missing_verify_authorization:{exc}")

    if authorization_blockers:
        stages = [
            {
                "stage_id": spec.stage_id,
                "schema_version": RECEIPT_SCHEMA,
                "role": spec.role,
                "frozen_parameters": dict(spec.frozen_parameters),
                "raw_runtime_state": "not_inspected_before_authorization",
                "raw_executable": False,
                "status": "blocked",
                "artifacts": [],
                "pickle_policy": [],
                "run_id": None,
                "source_bundle_sha256": None,
                "input_identity_fingerprint": None,
                "output_identity_fingerprint": None,
                "input_rows": None,
                "output_rows": None,
                "unresolved_rows": None,
                "duplicate_rows": None,
                "missing_rows": None,
                "blockers": ["authorization_gate_not_passed"],
            }
            for spec in DAG_STAGE_SPECS
        ]
        stage_blockers = ["authorization_gate_not_passed"]
    else:
        truth_manifest, truth_record = verify_truth_manifest(project_root, truth_manifest_path)
        stages, stage_blockers = build_stage_contract(project_root)

    receipt = _base_receipt(
        mode="verify",
        project_root=project_root,
        truth_manifest=truth_manifest,
        truth_record=truth_record,
        split_csv=None,
        stages=stages,
        stage_blockers=stage_blockers,
    )
    blockers = [*authorization_blockers, *stage_blockers]
    receipt["authorization"] = authorization_record
    receipt["verification"] = {
        "empty_workdir_replay_attempted": False,
        "historical_prediction_csv_used": False,
        "reported_row_count": 0,
        "blockers": sorted(set(blockers)),
    }
    receipt["decision"] = "complete_loop151_raw_replay_blocked"
    return receipt


def write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _add_common_sample_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--truth-manifest", type=Path, default=DEFAULT_TRUTH_MANIFEST)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--allowed-raw-root", type=Path, default=Path("data"))
    parser.add_argument("--max-samples", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed raw-file replay and parity contract for the Loop151 chain."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run a bounded train-only identity/native smoke.")
    _add_common_sample_args(smoke)
    smoke.add_argument("--runtime", choices=("identity", "native"), default="identity")
    smoke.add_argument("--native-selftest", type=Path, default=DEFAULT_NATIVE_SELFTEST)
    smoke.add_argument("--native-dll", type=Path, default=DEFAULT_NATIVE_DLL)
    smoke.add_argument("--native-onnx", type=Path, default=DEFAULT_NATIVE_ONNX)
    smoke.add_argument("--native-stage2", type=Path, default=DEFAULT_NATIVE_STAGE2)
    smoke.add_argument("--timeout-seconds", type=int, default=120)
    smoke.add_argument(
        "--output-json",
        type=Path,
        default=Path("manifests/roadmap_9997/p0_raw_replay/train_smoke_receipt.json"),
    )

    parity = subparsers.add_parser(
        "native-parity",
        help="Compare guarded Python Loop28 with the native Loop28 runtime on train rows.",
    )
    _add_common_sample_args(parity)
    parity.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parity.add_argument("--python-stage2", type=Path, default=DEFAULT_PYTHON_STAGE2)
    parity.add_argument("--pickle-allowlist", type=Path, default=DEFAULT_PICKLE_ALLOWLIST)
    parity.add_argument("--native-selftest", type=Path, default=DEFAULT_NATIVE_SELFTEST)
    parity.add_argument("--native-dll", type=Path, default=DEFAULT_NATIVE_DLL)
    parity.add_argument("--native-onnx", type=Path, default=DEFAULT_NATIVE_ONNX)
    parity.add_argument("--native-stage2", type=Path, default=DEFAULT_NATIVE_STAGE2)
    parity.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parity.add_argument("--timeout-seconds", type=int, default=120)
    parity.add_argument(
        "--output-json",
        type=Path,
        default=Path("manifests/roadmap_9997/p0_raw_replay/native_parity_receipt.json"),
    )

    verify = subparsers.add_parser(
        "verify",
        help="Audit the future complete DAG gate; never runs without bound authorization.",
    )
    verify.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    verify.add_argument("--truth-manifest", type=Path, default=DEFAULT_TRUTH_MANIFEST)
    verify.add_argument("--authorization", type=Path, required=True)
    verify.add_argument(
        "--output-json",
        type=Path,
        default=Path("manifests/roadmap_9997/p0_raw_replay/verify_receipt.json"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_path = resolve_within(project_root, args.output_json, purpose="Output receipt")
    try:
        if args.command == "smoke":
            truth_manifest = resolve_frozen_a1_artifact(
                project_root,
                args.truth_manifest,
                DEFAULT_TRUTH_MANIFEST,
                purpose="truth manifest",
            )
            split_csv = resolve_frozen_a1_artifact(
                project_root,
                args.split_csv,
                DEFAULT_SPLIT_CSV,
                purpose="split CSV",
            )
            allowed_raw_root = resolve_within(
                project_root,
                args.allowed_raw_root,
                purpose="Allowed raw root",
            )
            native_runner = None
            if args.runtime == "native":
                selftest_path = resolve_frozen_a1_artifact(
                    project_root,
                    args.native_selftest,
                    DEFAULT_NATIVE_SELFTEST,
                    purpose="native selftest",
                )
                dll_path = resolve_frozen_a1_artifact(
                    project_root,
                    args.native_dll,
                    DEFAULT_NATIVE_DLL,
                    purpose="native DLL",
                )
                onnx_path = resolve_frozen_a1_artifact(
                    project_root,
                    args.native_onnx,
                    DEFAULT_NATIVE_ONNX,
                    purpose="native ONNX model",
                )
                native_stage2_path = resolve_frozen_a1_artifact(
                    project_root,
                    args.native_stage2,
                    DEFAULT_NATIVE_STAGE2,
                    purpose="native Stage-2 model",
                )

                def native_runner(sample_path: Path, snapshot_root: Path) -> dict:
                    return run_native_loop28(
                        project_root=project_root,
                        sample_path=sample_path,
                        allowed_raw_root=snapshot_root,
                        selftest_path=selftest_path,
                        dll_path=dll_path,
                        onnx_path=onnx_path,
                        stage2_path=native_stage2_path,
                        timeout_seconds=args.timeout_seconds,
                    )

            receipt = build_smoke_receipt(
                project_root=project_root,
                truth_manifest_path=truth_manifest,
                split_csv=split_csv,
                allowed_raw_root=allowed_raw_root,
                max_samples=args.max_samples,
                runtime=args.runtime,
                native_runner=native_runner,
            )
        elif args.command == "native-parity":
            truth_manifest = resolve_frozen_a1_artifact(
                project_root,
                args.truth_manifest,
                DEFAULT_TRUTH_MANIFEST,
                purpose="truth manifest",
            )
            split_csv = resolve_frozen_a1_artifact(
                project_root,
                args.split_csv,
                DEFAULT_SPLIT_CSV,
                purpose="split CSV",
            )
            allowed_raw_root = resolve_within(
                project_root,
                args.allowed_raw_root,
                purpose="Allowed raw root",
            )
            checkpoint_path = resolve_frozen_a1_artifact(
                project_root,
                args.checkpoint,
                DEFAULT_CHECKPOINT,
                purpose="Python checkpoint",
            )
            python_stage2 = resolve_frozen_a1_artifact(
                project_root,
                args.python_stage2,
                DEFAULT_PYTHON_STAGE2,
                purpose="Python Stage-2 model",
            )
            pickle_allowlist = resolve_frozen_a1_artifact(
                project_root,
                args.pickle_allowlist,
                DEFAULT_PICKLE_ALLOWLIST,
                purpose="Pickle allowlist",
            )
            selftest_path = resolve_frozen_a1_artifact(
                project_root,
                args.native_selftest,
                DEFAULT_NATIVE_SELFTEST,
                purpose="native selftest",
            )
            dll_path = resolve_frozen_a1_artifact(
                project_root,
                args.native_dll,
                DEFAULT_NATIVE_DLL,
                purpose="native DLL",
            )
            onnx_path = resolve_frozen_a1_artifact(
                project_root,
                args.native_onnx,
                DEFAULT_NATIVE_ONNX,
                purpose="native ONNX model",
            )
            native_stage2_path = resolve_frozen_a1_artifact(
                project_root,
                args.native_stage2,
                DEFAULT_NATIVE_STAGE2,
                purpose="native Stage-2 model",
            )
            receipt = build_native_parity_receipt(
                project_root=project_root,
                truth_manifest_path=truth_manifest,
                split_csv=split_csv,
                allowed_raw_root=allowed_raw_root,
                max_samples=args.max_samples,
                tolerance=args.tolerance,
                python_runner=lambda sample_path: run_python_loop28(
                    project_root=project_root,
                    sample_path=sample_path,
                    checkpoint_path=checkpoint_path,
                    stage2_path=python_stage2,
                    pickle_allowlist_path=pickle_allowlist,
                ),
                native_runner=lambda sample_path, snapshot_root: run_native_loop28(
                    project_root=project_root,
                    sample_path=sample_path,
                    allowed_raw_root=snapshot_root,
                    selftest_path=selftest_path,
                    dll_path=dll_path,
                    onnx_path=onnx_path,
                    stage2_path=native_stage2_path,
                    timeout_seconds=args.timeout_seconds,
                ),
            )
        else:
            receipt = build_verify_receipt(
                project_root=project_root,
                truth_manifest_path=args.truth_manifest,
                authorization_path=args.authorization,
            )
    except Exception as exc:  # noqa: BLE001 - CLI must leave a machine-readable failure receipt.
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.command,
            "decision": "replay_contract_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "claim_scope": {
                "quality_claim_allowed": False,
                "raw_file_to_legacy_report_replay_ready": False,
                "raw_file_to_loop151_replay_ready": False,
                "native_loop151_ready": False,
            },
        }
    receipt["invocation"] = {
        "argv": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "os_name": os.name,
        "resource_receipt_required": args.command == "verify",
        "authorization_scope": (
            "explicit_A2_or_A3_required"
            if args.command == "verify"
            else "A1_train_only_smoke_or_parity"
        ),
    }
    write_receipt(output_path, receipt)
    summary = {
        "decision": receipt.get("decision"),
        "mode": receipt.get("mode"),
        "output_json": str(output_path),
        "full_loop151_ready": bool(
            (receipt.get("claim_scope") or {}).get("raw_file_to_loop151_replay_ready")
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    passing_decisions = {
        "train_identity_smoke_passed_full_loop151_blocked",
        "native_loop28_train_smoke_passed_full_loop151_blocked",
        "native_loop28_parity_passed_full_loop151_blocked",
    }
    return 0 if receipt.get("decision") in passing_decisions else 2


if __name__ == "__main__":
    raise SystemExit(main())
