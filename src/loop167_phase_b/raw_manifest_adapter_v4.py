"""Fail-closed Train-only fold-manifest adapter for Loop167 Phase B v4.

The adapter reads only the SHA-bound JSONL authority artifact.  It never opens
raw sample bytes, checkpoints, or prediction artifacts.  Raw locators are kept
inside ``RawScanPlan``; labels, folds, and content-component identifiers remain
in controller memory until the fixed fitting boundary is assembled.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .contracts import (
    PhaseBContractError,
    canonical_json_bytes,
    require_sha256,
)
from .path_safety_v4 import canonical_project_relative_path, safe_project_path, safe_project_root
from .raw_worker import RawPlanEntry, RawScanPlan

if TYPE_CHECKING:
    from .feature_cache_v4 import PhaseBFitPayload
    from .fit_worker import PhaseBFeatureCache


LOOP164_FOLD_LOOP_ID = "loop164_whole_file_residual_expert"
LOOP164_FOLD_CLAIM_SCOPE = "local_train_content_similarity_diagnostic_not_family_or_time_isolation"
PHASE_B_PROTOCOL_SCHEMA = "axon_loop167_phase_b_protocol_v1"
FOLD_RECORD_SCHEMA = "axon_loop164_local_train_diagnostic_fold_record_v1"
FULL_TRAIN_ROWS = 20_000
OUTER_FOLD_COUNT = 5
ROWS_PER_OUTER_FOLD = 4_000
MAX_FOLD_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_PHASE_B_PROTOCOL_BYTES = 4 * 1024 * 1024
COMPONENT_ID_PATTERN = re.compile(r"^[0-9a-f]{24}$")
SOURCE_AUDIT_DOMAIN = b"axon_loop167_phase_b_source_audit_v4\0"

FOLD_RECORD_FIELDS = frozenset(
    {
        "schema",
        "loop_id",
        "claim_scope",
        "split_role",
        "train_row_index",
        "sample_index",
        "source_path",
        "source_sha256",
        "source_size_bytes",
        "label",
        "availability",
        "missing_reason",
        "content_component_id",
        "content_component_size",
        "diagnostic_fold",
        "identity_metadata_not_model_features",
    }
)
IDENTITY_METADATA_FIELDS = [
    "train_row_index",
    "sample_index",
    "source_path",
    "source_sha256",
    "content_component_id",
    "diagnostic_fold",
]
PHASE_B_PROTOCOL_FIELDS = frozenset(
    {
        "schema",
        "loop_id",
        "status",
        "claim_scope",
        "phase_a_bindings",
        "input_contract",
        "feature_contract",
        "fit_contract",
        "evaluation_contract",
        "resource_contract",
        "runtime_contract",
        "one_shot_lease",
        "forbidden",
        "ready_for",
    }
)
INPUT_CONTRACT_FIELDS = frozenset(
    {"folds", "scope_drift_is_fatal", "source_sha256_verified_in_same_stream"}
)
FOLD_CONTRACT_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "record_schema",
        "split_role",
        "rows",
        "folds",
        "rows_per_fold",
        "val_test_or_full_access",
    }
)


class RawManifestAdapterV4Error(PhaseBContractError):
    """Raised when a bound Train-only source manifest cannot be used safely."""


@dataclass(frozen=True, slots=True)
class TrainOnlyFitTargets:
    """In-memory labels and folds that may be combined with a feature cache later."""

    labels: np.ndarray
    folds: np.ndarray

    def to_phase_b_fit_payload(self, cache: "PhaseBFeatureCache") -> "PhaseBFitPayload":
        """Create the only fitting payload, whose fields are cache, labels, and folds."""

        from .feature_cache_v4 import make_phase_b_fit_payload

        return make_phase_b_fit_payload(cache, self.labels, self.folds)


@dataclass(frozen=True, slots=True)
class TrainOnlyManifestAdapterResult:
    """Controller-only handoff that separates raw scope from fit-only metadata."""

    raw_scan_plan: RawScanPlan
    fit_targets: TrainOnlyFitTargets
    component_ids: tuple[str, ...]
    phase_b_protocol_sha256: str
    fold_manifest_sha256: str

    def to_phase_b_fit_payload(self, cache: "PhaseBFeatureCache") -> "PhaseBFitPayload":
        return self.fit_targets.to_phase_b_fit_payload(cache)


@dataclass(frozen=True, slots=True)
class _FoldManifestContract:
    manifest_path: Path
    manifest_sha256: str
    record_schema: str
    rows: int
    folds: int
    rows_per_fold: int
    phase_b_protocol_sha256: str


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise RawManifestAdapterV4Error(f"JSON object repeats key: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite(value: str) -> object:
    raise RawManifestAdapterV4Error(f"JSON uses non-finite constant: {value}")


def _parse_canonical_json_line(raw_line: bytes, *, line_number: int) -> dict[str, Any]:
    if not raw_line:
        raise RawManifestAdapterV4Error("Fold manifest contains an empty line")
    try:
        payload = json.loads(
            raw_line.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawManifestAdapterV4Error(
            f"Fold manifest line {line_number} is not valid canonical JSON"
        ) from exc
    if not isinstance(payload, dict) or canonical_json_bytes(payload)[:-1] != raw_line:
        raise RawManifestAdapterV4Error(f"Fold manifest line {line_number} is not canonical JSON")
    return payload


def _parse_canonical_json_document(raw: bytes, *, label: str) -> dict[str, Any]:
    if not raw:
        raise RawManifestAdapterV4Error(f"{label} is empty")
    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawManifestAdapterV4Error(f"{label} is not valid canonical JSON") from exc
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise RawManifestAdapterV4Error(f"{label} is not canonical JSON")
    return payload


def _is_symlink_or_reparse(stat_result: os.stat_result) -> bool:
    if stat.S_ISLNK(stat_result.st_mode):
        return True
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400))
    return bool(attributes & reparse_flag)


def _absolute_lexical(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise RawManifestAdapterV4Error("Path must be absolute")
    return Path(os.path.abspath(os.fspath(candidate)))


def _assert_safe_directory_ancestry(path: Path, *, label: str) -> Path:
    absolute_path = _absolute_lexical(path)
    anchor = Path(absolute_path.anchor)
    if not absolute_path.is_absolute() or not anchor:
        raise RawManifestAdapterV4Error(f"{label} must be an absolute path")
    cursor = anchor
    try:
        anchor_stat = os.lstat(cursor)
    except OSError as exc:
        raise RawManifestAdapterV4Error(f"{label} anchor is inaccessible") from exc
    if _is_symlink_or_reparse(anchor_stat) or not stat.S_ISDIR(anchor_stat.st_mode):
        raise RawManifestAdapterV4Error(f"{label} anchor is unsafe")
    for component in absolute_path.parts[1:]:
        cursor = cursor / component
        try:
            current_stat = os.lstat(cursor)
        except OSError as exc:
            raise RawManifestAdapterV4Error(f"{label} ancestry is inaccessible") from exc
        if _is_symlink_or_reparse(current_stat):
            raise RawManifestAdapterV4Error(f"{label} ancestry contains a symlink or reparse point")
        if not stat.S_ISDIR(current_stat.st_mode):
            raise RawManifestAdapterV4Error(f"{label} ancestry is not a directory")
    return absolute_path


def _lexical_relative_to(path: Path, root: Path) -> Path:
    path_text = os.path.normpath(os.fspath(path))
    root_text = os.path.normpath(os.fspath(root))
    try:
        common_path = os.path.commonpath((path_text, root_text))
    except ValueError as exc:
        raise ValueError("Paths do not share a filesystem root") from exc
    if os.path.normcase(common_path) != os.path.normcase(root_text):
        raise ValueError("Path is outside the root")
    relative_text = os.path.relpath(path_text, root_text)
    return Path() if relative_text in {"", os.curdir} else Path(relative_text)


def _assert_safe_regular_or_missing_source(
    source_path: Path,
    *,
    data_root: Path,
    allow_missing_final: bool,
    declared_size: int | None,
) -> Path:
    try:
        relative_path = _lexical_relative_to(source_path, data_root)
    except ValueError as exc:
        raise RawManifestAdapterV4Error("Fold source path escapes the raw data root") from exc
    if not relative_path.parts or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise RawManifestAdapterV4Error("Fold source path has invalid relative components")

    cursor = data_root
    for index, component in enumerate(relative_path.parts):
        cursor = cursor / component
        final_component = index == len(relative_path.parts) - 1
        try:
            current_stat = os.lstat(cursor)
        except FileNotFoundError as exc:
            if final_component and allow_missing_final:
                return cursor
            raise RawManifestAdapterV4Error("Fold source ancestry is missing") from exc
        except OSError as exc:
            raise RawManifestAdapterV4Error("Fold source ancestry is inaccessible") from exc
        if _is_symlink_or_reparse(current_stat):
            raise RawManifestAdapterV4Error(
                "Fold source ancestry contains a symlink or reparse point"
            )
        if final_component:
            if not stat.S_ISREG(current_stat.st_mode):
                raise RawManifestAdapterV4Error("Fold source is not a regular file")
            if allow_missing_final:
                raise RawManifestAdapterV4Error("read_failure source unexpectedly exists")
            if declared_size is None or int(current_stat.st_size) != declared_size:
                raise RawManifestAdapterV4Error("Fold source declared size drifted")
        elif not stat.S_ISDIR(current_stat.st_mode):
            raise RawManifestAdapterV4Error("Fold source ancestry is not a directory")
    return cursor


def _read_bounded_regular_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(os.fspath(path), flags)
    except OSError as exc:
        raise RawManifestAdapterV4Error(f"{label} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise RawManifestAdapterV4Error(f"{label} is not a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            content = handle.read(max_bytes + 1)
            after = os.fstat(handle.fileno())
        if len(content) > max_bytes:
            raise RawManifestAdapterV4Error(f"{label} exceeds its byte cap")
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or len(content) != before.st_size
        ):
            raise RawManifestAdapterV4Error(f"{label} changed while it was read")
        return content
    except RawManifestAdapterV4Error:
        raise
    except OSError as exc:
        raise RawManifestAdapterV4Error(f"{label} cannot be read safely") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _require_nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RawManifestAdapterV4Error(f"{field} must be a non-negative integer")
    return value


def _require_binary_label(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
        raise RawManifestAdapterV4Error(f"{field} must be an integer binary label")
    return value


def _require_source_path(value: object, *, data_root: Path) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RawManifestAdapterV4Error("Fold source_path must be a trimmed nonempty string")
    supplied_path = Path(value)
    if not supplied_path.is_absolute() or any(
        component in {"", ".", ".."} for component in supplied_path.parts
    ):
        raise RawManifestAdapterV4Error("Fold source_path must be a clean absolute path")
    source_path = _absolute_lexical(supplied_path)
    try:
        _lexical_relative_to(source_path, data_root)
    except ValueError as exc:
        raise RawManifestAdapterV4Error("Fold source_path is outside the raw data root") from exc
    return source_path


def _source_audit_sha256(*, expected_sha256: str, declared_size: int) -> str:
    audit_payload = canonical_json_bytes(
        {"declared_size": declared_size, "expected_sha256": expected_sha256}
    )
    return hashlib.sha256(SOURCE_AUDIT_DOMAIN + audit_payload).hexdigest()


def _normalize_fit_targets(labels: list[int], folds: list[int]) -> TrainOnlyFitTargets:
    label_array = np.asarray(labels, dtype=np.uint8)
    fold_array = np.asarray(folds, dtype=np.int8)
    label_array.setflags(write=False)
    fold_array.setflags(write=False)
    return TrainOnlyFitTargets(labels=label_array, folds=fold_array)


def _validate_protocol_contract(
    project_root: Path,
    *,
    phase_b_protocol_binding: Mapping[str, str],
) -> _FoldManifestContract:
    project_root_safe = safe_project_root(project_root)
    if (
        not isinstance(phase_b_protocol_binding, Mapping)
        or set(phase_b_protocol_binding) != {"path", "sha256"}
    ):
        raise RawManifestAdapterV4Error(
            "phase_b_protocol binding must contain exactly path and sha256"
        )
    try:
        protocol_relative = canonical_project_relative_path(phase_b_protocol_binding["path"])
        protocol_path = safe_project_path(
            project_root_safe,
            protocol_relative,
            require_exists=True,
            require_regular_file=True,
        )
    except PhaseBContractError as exc:
        raise RawManifestAdapterV4Error("Phase-B protocol path is unsafe") from exc
    expected_protocol_sha256 = require_sha256(
        phase_b_protocol_binding["sha256"],
        field="phase_b_protocol.sha256",
    )
    protocol_raw = _read_bounded_regular_file(
        protocol_path,
        max_bytes=MAX_PHASE_B_PROTOCOL_BYTES,
        label="phase_b_protocol",
    )
    protocol_sha256 = hashlib.sha256(protocol_raw).hexdigest()
    if protocol_sha256 != expected_protocol_sha256:
        raise RawManifestAdapterV4Error("Phase-B protocol SHA-256 binding drifted")
    protocol = _parse_canonical_json_document(protocol_raw, label="Phase-B protocol")
    if set(protocol) != PHASE_B_PROTOCOL_FIELDS:
        raise RawManifestAdapterV4Error("Phase-B protocol fields drifted")
    if protocol.get("schema") != PHASE_B_PROTOCOL_SCHEMA:
        raise RawManifestAdapterV4Error("Phase-B protocol schema drifted")
    if protocol.get("loop_id") != "loop167_ember_v3_novel_delta":
        raise RawManifestAdapterV4Error("Phase-B protocol loop identity drifted")
    input_contract = protocol.get("input_contract")
    if not isinstance(input_contract, dict) or set(input_contract) != INPUT_CONTRACT_FIELDS:
        raise RawManifestAdapterV4Error("Phase-B input contract fields drifted")
    if (
        input_contract.get("scope_drift_is_fatal") is not True
        or input_contract.get("source_sha256_verified_in_same_stream") is not True
    ):
        raise RawManifestAdapterV4Error("Phase-B input scope safeguards drifted")
    folds = input_contract.get("folds")
    if not isinstance(folds, dict) or set(folds) != FOLD_CONTRACT_FIELDS:
        raise RawManifestAdapterV4Error("Phase-B fold contract fields drifted")
    if (
        folds.get("record_schema") != FOLD_RECORD_SCHEMA
        or folds.get("split_role") != "train"
        or folds.get("rows") != FULL_TRAIN_ROWS
        or folds.get("folds") != OUTER_FOLD_COUNT
        or folds.get("rows_per_fold") != ROWS_PER_OUTER_FOLD
        or folds.get("val_test_or_full_access") is not False
    ):
        raise RawManifestAdapterV4Error("Phase-B fold contract scope drifted")
    manifest_path_text = folds.get("path")
    if not isinstance(manifest_path_text, str) or not manifest_path_text:
        raise RawManifestAdapterV4Error("Phase-B fold manifest path is invalid")
    try:
        manifest_relative = canonical_project_relative_path(manifest_path_text)
        manifest_path = safe_project_path(
            project_root_safe,
            manifest_relative,
            require_exists=True,
            require_regular_file=True,
        )
    except PhaseBContractError as exc:
        raise RawManifestAdapterV4Error("Phase-B fold manifest path is invalid or unsafe") from exc
    return _FoldManifestContract(
        manifest_path=manifest_path,
        manifest_sha256=require_sha256(folds.get("sha256"), field="folds.sha256"),
        record_schema=FOLD_RECORD_SCHEMA,
        rows=FULL_TRAIN_ROWS,
        folds=OUTER_FOLD_COUNT,
        rows_per_fold=ROWS_PER_OUTER_FOLD,
        phase_b_protocol_sha256=protocol_sha256,
    )


def _parse_record(
    payload: Mapping[str, Any],
    *,
    ordinal: int,
    contract: _FoldManifestContract,
    data_root: Path,
) -> tuple[RawPlanEntry, int, int, str]:
    if set(payload) != FOLD_RECORD_FIELDS:
        raise RawManifestAdapterV4Error("Fold record fields do not match the sealed schema")
    if (
        payload.get("schema") != contract.record_schema
        or payload.get("loop_id") != LOOP164_FOLD_LOOP_ID
        or payload.get("claim_scope") != LOOP164_FOLD_CLAIM_SCOPE
        or payload.get("split_role") != "train"
    ):
        raise RawManifestAdapterV4Error("Fold record scope is not exact Train-only authority")
    if _require_nonnegative_integer(payload.get("train_row_index"), field="train_row_index") != ordinal:
        raise RawManifestAdapterV4Error("Fold train_row_index is not contiguous and source ordered")
    if _require_nonnegative_integer(payload.get("sample_index"), field="sample_index") != ordinal:
        raise RawManifestAdapterV4Error("Fold sample_index is not the canonical Train-only ordinal")
    if payload.get("identity_metadata_not_model_features") != IDENTITY_METADATA_FIELDS:
        raise RawManifestAdapterV4Error("Fold identity-metadata declaration drifted")

    label = _require_binary_label(payload.get("label"), field="label")
    fold = _require_nonnegative_integer(payload.get("diagnostic_fold"), field="diagnostic_fold")
    if fold >= contract.folds:
        raise RawManifestAdapterV4Error("Fold diagnostic_fold is outside the sealed outer-fold set")
    component_id = payload.get("content_component_id")
    if not isinstance(component_id, str) or not COMPONENT_ID_PATTERN.fullmatch(component_id):
        raise RawManifestAdapterV4Error("Fold content_component_id is invalid")
    _require_nonnegative_integer(payload.get("content_component_size"), field="content_component_size")
    if payload["content_component_size"] == 0:
        raise RawManifestAdapterV4Error("Fold content_component_size must be positive")

    availability = payload.get("availability")
    missing_reason = payload.get("missing_reason")
    declared_size_value = payload.get("source_size_bytes")
    if availability == "supported":
        declared_size = _require_nonnegative_integer(declared_size_value, field="source_size_bytes")
        if declared_size < 1 or missing_reason is not None:
            raise RawManifestAdapterV4Error("Supported fold source metadata drifted")
    elif availability in {"parse_failure", "oversize"}:
        declared_size = _require_nonnegative_integer(declared_size_value, field="source_size_bytes")
        if missing_reason != availability:
            raise RawManifestAdapterV4Error("Unavailable fold source metadata drifted")
    elif availability == "read_failure":
        if declared_size_value is not None or missing_reason != "read_failure":
            raise RawManifestAdapterV4Error("read_failure fold source metadata drifted")
        declared_size = 0
    else:
        raise RawManifestAdapterV4Error("Fold availability is outside the sealed vocabulary")

    expected_sha256 = require_sha256(payload.get("source_sha256"), field="source_sha256")
    source_path = _require_source_path(payload.get("source_path"), data_root=data_root)
    safe_source_path = _assert_safe_regular_or_missing_source(
        source_path,
        data_root=data_root,
        allow_missing_final=availability == "read_failure",
        declared_size=None if availability == "read_failure" else declared_size,
    )
    raw_entry = RawPlanEntry(
        ordinal=ordinal,
        source_file=safe_source_path,
        source_audit_sha256=_source_audit_sha256(
            expected_sha256=expected_sha256,
            declared_size=declared_size,
        ),
        declared_size=declared_size,
        expected_sha256=expected_sha256,
    )
    return raw_entry, label, fold, component_id


def _load_for_contract(
    contract: _FoldManifestContract,
    *,
    data_root: Path,
) -> TrainOnlyManifestAdapterResult:
    data_root_safe = _assert_safe_directory_ancestry(data_root, label="raw data root")
    manifest_parent = _assert_safe_directory_ancestry(
        contract.manifest_path.parent,
        label="fold manifest parent",
    )
    try:
        _lexical_relative_to(contract.manifest_path, manifest_parent)
    except ValueError as exc:
        raise RawManifestAdapterV4Error("Fold manifest path is unsafe") from exc
    manifest_raw = _read_bounded_regular_file(
        contract.manifest_path,
        max_bytes=MAX_FOLD_MANIFEST_BYTES,
        label="fold manifest",
    )
    observed_manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if observed_manifest_sha256 != contract.manifest_sha256:
        raise RawManifestAdapterV4Error("Fold manifest SHA-256 binding drifted")
    if not manifest_raw.endswith(b"\n") or not manifest_raw:
        raise RawManifestAdapterV4Error("Fold manifest must be a nonempty newline-terminated JSONL file")
    lines = manifest_raw[:-1].split(b"\n")
    if len(lines) != contract.rows:
        raise RawManifestAdapterV4Error("Fold manifest does not contain exactly 20,000 Train-only rows")

    raw_entries: list[RawPlanEntry] = []
    labels: list[int] = []
    folds: list[int] = []
    component_ids: list[str] = []
    seen_source_sha256: set[str] = set()
    seen_source_paths: set[str] = set()
    declared_component_sizes: dict[str, int] = {}
    component_folds: dict[str, int] = {}
    for ordinal, raw_line in enumerate(lines):
        payload = _parse_canonical_json_line(raw_line, line_number=ordinal + 1)
        entry, label, fold, component_id = _parse_record(
            payload,
            ordinal=ordinal,
            contract=contract,
            data_root=data_root_safe,
        )
        if entry.expected_sha256 in seen_source_sha256:
            raise RawManifestAdapterV4Error("Fold manifest repeats a source_sha256")
        source_key = os.path.normcase(os.fspath(entry.source_file))
        if source_key in seen_source_paths:
            raise RawManifestAdapterV4Error("Fold manifest repeats a source path")
        seen_source_sha256.add(entry.expected_sha256)
        seen_source_paths.add(source_key)
        declared_size = int(payload["content_component_size"])
        previous_size = declared_component_sizes.setdefault(component_id, declared_size)
        if previous_size != declared_size:
            raise RawManifestAdapterV4Error("Fold component size declaration drifted")
        previous_fold = component_folds.setdefault(component_id, fold)
        if previous_fold != fold:
            raise RawManifestAdapterV4Error("Fold content component crosses outer folds")
        raw_entries.append(entry)
        labels.append(label)
        folds.append(fold)
        component_ids.append(component_id)

    fold_counts = np.bincount(np.asarray(folds, dtype=np.int8), minlength=contract.folds)
    if fold_counts.shape != (contract.folds,) or not np.all(fold_counts == contract.rows_per_fold):
        raise RawManifestAdapterV4Error("Fold manifest is not exactly five 4,000-row outer folds")
    label_counts = np.bincount(np.asarray(labels, dtype=np.uint8), minlength=2)
    expected_labels_per_class = contract.rows // 2
    if label_counts.shape != (2,) or not np.all(label_counts == expected_labels_per_class):
        raise RawManifestAdapterV4Error("Fold manifest labels are not the exact balanced Train-only set")
    observed_component_sizes: dict[str, int] = {}
    for component_id in component_ids:
        observed_component_sizes[component_id] = observed_component_sizes.get(component_id, 0) + 1
    if observed_component_sizes != declared_component_sizes:
        raise RawManifestAdapterV4Error("Fold component membership counts drifted")
    normalized_targets = _normalize_fit_targets(labels, folds)
    for fold in range(contract.folds):
        if np.unique(normalized_targets.labels[normalized_targets.folds != fold]).size != 2:
            raise RawManifestAdapterV4Error("An outer-fold fit partition lacks a binary class")
    return TrainOnlyManifestAdapterResult(
        raw_scan_plan=RawScanPlan.from_entries(tuple(raw_entries)),
        fit_targets=normalized_targets,
        component_ids=tuple(component_ids),
        phase_b_protocol_sha256=contract.phase_b_protocol_sha256,
        fold_manifest_sha256=observed_manifest_sha256,
    )


def load_train_only_manifest_v4(
    project_root: Path,
    *,
    phase_b_protocol_binding: Mapping[str, str],
    data_root: Path,
) -> TrainOnlyManifestAdapterResult:
    """Parse the exact SHA-bound 20k Train-only fold authority before any raw open."""

    project_root_safe = safe_project_root(project_root)
    contract = _validate_protocol_contract(
        project_root_safe,
        phase_b_protocol_binding=phase_b_protocol_binding,
    )
    return _load_for_contract(contract, data_root=data_root)
