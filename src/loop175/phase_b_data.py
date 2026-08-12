"""Strict identity-control and numeric-cache boundary for Loop175 Phase B."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import stat
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from src.loop167_phase_b.contracts import canonical_json_bytes, require_sha256
from src.loop167_phase_b.fit_cache_loader_v13 import load_verified_v12_cache_for_v13

FULL_TRAIN_ROWS = 20_000
FOLD_COUNT = 5
ROWS_PER_FOLD = 4_000
ROWS_PER_CLASS = 10_000
B0_FEATURE_DIMENSION = 571

CANONICAL_FOLD_RELATIVE_PATH = (
    "reports/roadmap_9997/loop164/local_train_diagnostic_folds.jsonl"
)
CANONICAL_FOLD_SHA256 = "00a31a1bd86d7b887447f3e86e5e753ebcaaee45be74311199332e073a3880a5"
CANONICAL_SPLIT_RELATIVE_PATH = "reports/random_20w_split/loop127_full_duplicate_corrected_split.csv"
CANONICAL_TRAIN_PREFIX_BYTES = 4_120_895
CANONICAL_TRAIN_PREFIX_SHA256 = (
    "dfbad6994605aa0fd9b7fa049b19cd87f15e50e37490a60efc43696c540dd54a"
)

FOLD_RECORD_SCHEMA = "axon_loop164_local_train_diagnostic_fold_record_v1"
FOLD_LOOP_ID = "loop164_whole_file_residual_expert"
FOLD_CLAIM_SCOPE = "local_train_content_similarity_diagnostic_not_family_or_time_isolation"
IDENTITY_DECLARATION = [
    "train_row_index",
    "sample_index",
    "source_path",
    "source_sha256",
    "content_component_id",
    "diagnostic_fold",
]
FOLD_RECORD_FIELDS = frozenset(
    {
        "availability",
        "claim_scope",
        "content_component_id",
        "content_component_size",
        "diagnostic_fold",
        "identity_metadata_not_model_features",
        "label",
        "loop_id",
        "missing_reason",
        "sample_index",
        "schema",
        "source_path",
        "source_sha256",
        "source_size_bytes",
        "split_role",
        "train_row_index",
    }
)
SPLIT_FIELDS = ["source_path", "source_sha256", "label", "sample_index", "split"]

RAGGED_REGION_CACHE_SCHEMA = "axon_loop175_identity_free_ragged_region_cache_v1"
RAGGED_REGION_CACHE_DOMAIN = b"axon_loop175_identity_free_ragged_region_cache_v1\0"
MAXIMUM_REGIONS = 16
MAXIMUM_REGION_BYTES = 8_192
MAXIMUM_TOTAL_REGION_BYTES = 131_072
BUCKET_COUNT = 64
PADDING_TOKEN = 256
MAXIMUM_REGION_CACHE_BYTES = 30 * 1024**3
REGION_ARRAY_DTYPES: Mapping[str, np.dtype[Any]] = MappingProxyType(
    {
        "row_region_offsets": np.dtype("<i8"),
        "file_sizes": np.dtype("<i8"),
        "region_token_offsets": np.dtype("<i8"),
        "token_values": np.dtype("u1"),
        "region_types": np.dtype("u1"),
        "region_starts": np.dtype("<i8"),
        "offset_buckets": np.dtype("u1"),
        "length_buckets": np.dtype("u1"),
    }
)
REGION_ARRAY_NAMES = tuple(REGION_ARRAY_DTYPES)
REGION_ARCHIVE_NAMES = frozenset(
    {*(f"{name}.npy" for name in REGION_ARRAY_NAMES), "metadata.json"}
)
REGION_METADATA_FIELDS = frozenset(
    {
        "schema",
        "row_count",
        "region_count",
        "token_count",
        "maximum_regions",
        "maximum_region_bytes",
        "maximum_total_region_bytes",
        "bucket_count",
        "padding_token",
        "offset_bucket_formula",
        "length_bucket_formula",
        "array_dtypes",
        "array_shapes",
        "numeric_payload_sha256",
    }
)


class Loop175PhaseBDataError(ValueError):
    """Raised when a Phase-B identity or numeric artifact fails closed."""


@dataclass(frozen=True, slots=True)
class CanonicalFoldAlignment:
    labels: np.ndarray
    folds: np.ndarray
    component_ids: tuple[str, ...]
    source_sha256: tuple[str, ...]
    manifest_sha256: str

    @property
    def sha_to_ordinal(self) -> Mapping[str, int]:
        return MappingProxyType({sha256: index for index, sha256 in enumerate(self.source_sha256)})


@dataclass(frozen=True, slots=True)
class CanonicalTrainAlignment:
    labels: np.ndarray
    sample_indices: np.ndarray
    source_sha256: tuple[str, ...]
    prefix_sha256: str
    prefix_bytes: int


@dataclass(frozen=True, slots=True)
class RaggedRegionCache:
    row_region_offsets: np.ndarray
    file_sizes: np.ndarray
    region_token_offsets: np.ndarray
    token_values: np.ndarray
    region_types: np.ndarray
    region_starts: np.ndarray
    offset_buckets: np.ndarray
    length_buckets: np.ndarray


@dataclass(frozen=True, slots=True)
class RaggedRegionCacheReceipt:
    path: Path
    sha256: str
    size_bytes: int
    row_count: int
    region_count: int
    token_count: int


@dataclass(frozen=True, slots=True)
class IdentityFreePhaseBFitPayload:
    b0_values: np.ndarray
    labels: np.ndarray
    folds: np.ndarray
    regions: RaggedRegionCache


@dataclass(frozen=True, slots=True)
class AlignedPhaseBData:
    labels: np.ndarray
    folds: np.ndarray
    component_ids: tuple[str, ...]
    source_sha256: tuple[str, ...]
    b0_values: np.ndarray
    b0_missing_counts: tuple[int, ...]
    fold_manifest_sha256: str
    train_prefix_sha256: str
    b0_cache_sha256: str

    def make_fit_payload(self, regions: RaggedRegionCache) -> IdentityFreePhaseBFitPayload:
        return make_identity_free_fit_payload(self, regions)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise Loop175PhaseBDataError(f"JSON repeats key: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite(value: str) -> object:
    raise Loop175PhaseBDataError(f"JSON uses non-finite value: {value}")


def _parse_canonical_json_line(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Loop175PhaseBDataError(f"{context} is not valid canonical JSON") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise Loop175PhaseBDataError(f"{context} is not canonical JSON")
    return payload


def _require_regular_file(path: Path, *, label: str) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        result = candidate.lstat()
    except OSError as error:
        raise Loop175PhaseBDataError(f"{label} is missing or inaccessible") from error
    attributes = int(getattr(result, "st_file_attributes", 0))
    if stat.S_ISLNK(result.st_mode) or attributes & 0x0400 or not stat.S_ISREG(result.st_mode):
        raise Loop175PhaseBDataError(f"{label} must be a safe regular file")
    return candidate


def _read_bounded(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    safe_path = _require_regular_file(path, label=label)
    with safe_path.open("rb") as handle:
        content = handle.read(maximum_bytes + 1)
    if len(content) > maximum_bytes:
        raise Loop175PhaseBDataError(f"{label} exceeds its byte limit")
    return content


def _sha256_file(path: Path, *, maximum_bytes: int) -> tuple[str, int]:
    safe_path = _require_regular_file(path, label="cache")
    digest = hashlib.sha256()
    total = 0
    with safe_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            total += len(block)
            if total > maximum_bytes:
                raise Loop175PhaseBDataError("cache exceeds its byte limit")
            digest.update(block)
    return digest.hexdigest(), total


def _readonly(values: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _require_integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Loop175PhaseBDataError(f"{field} must be an integer >= {minimum}")
    return value


def _require_binary_label(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
        raise Loop175PhaseBDataError(f"{field} must be an integer binary label")
    return value


def load_canonical_fold_manifest(
    path: Path | str,
    *,
    expected_sha256: str,
) -> CanonicalFoldAlignment:
    """Load the exact 20k fold authority while returning no source paths."""

    expected_sha = require_sha256(expected_sha256, field="fold_manifest.sha256")
    raw = _read_bounded(Path(path), maximum_bytes=64 * 1024 * 1024, label="fold manifest")
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise Loop175PhaseBDataError("fold manifest SHA-256 binding drifted")
    if not raw or not raw.endswith(b"\n"):
        raise Loop175PhaseBDataError("fold manifest must be nonempty and newline terminated")
    lines = raw.splitlines(keepends=True)
    if len(lines) != FULL_TRAIN_ROWS:
        raise Loop175PhaseBDataError("fold manifest must contain exactly 20,000 rows")

    labels = np.empty(FULL_TRAIN_ROWS, dtype=np.uint8)
    folds = np.empty(FULL_TRAIN_ROWS, dtype=np.int8)
    source_sha256: list[str] = []
    component_ids: list[str] = []
    seen_sha: set[str] = set()
    component_fold: dict[str, int] = {}
    declared_component_sizes: dict[str, int] = {}
    for ordinal, raw_line in enumerate(lines):
        record = _parse_canonical_json_line(raw_line, context=f"fold row {ordinal}")
        if set(record) != FOLD_RECORD_FIELDS:
            raise Loop175PhaseBDataError("fold record fields drifted")
        if (
            record.get("schema") != FOLD_RECORD_SCHEMA
            or record.get("loop_id") != FOLD_LOOP_ID
            or record.get("claim_scope") != FOLD_CLAIM_SCOPE
            or record.get("split_role") != "train"
        ):
            raise Loop175PhaseBDataError("fold record scope drifted")
        if (
            _require_integer(record.get("train_row_index"), field="train_row_index") != ordinal
            or _require_integer(record.get("sample_index"), field="sample_index") != ordinal
        ):
            raise Loop175PhaseBDataError("fold row index is not canonical and contiguous")
        if record.get("identity_metadata_not_model_features") != IDENTITY_DECLARATION:
            raise Loop175PhaseBDataError("fold identity declaration drifted")
        source_path = record.get("source_path")
        if not isinstance(source_path, str) or not source_path or source_path != source_path.strip():
            raise Loop175PhaseBDataError("fold source_path is invalid")
        source_sha = require_sha256(record.get("source_sha256"), field="source_sha256")
        if source_sha in seen_sha:
            raise Loop175PhaseBDataError("fold manifest repeats a source_sha256")
        seen_sha.add(source_sha)

        label = _require_binary_label(record.get("label"), field="label")
        fold = _require_integer(record.get("diagnostic_fold"), field="diagnostic_fold")
        if fold >= FOLD_COUNT:
            raise Loop175PhaseBDataError("fold id is outside 0..4")
        component = record.get("content_component_id")
        if not isinstance(component, str) or len(component) != 24 or any(
            character not in "0123456789abcdef" for character in component
        ):
            raise Loop175PhaseBDataError("component id is not canonical lowercase hex")
        component_size = _require_integer(
            record.get("content_component_size"), field="content_component_size", minimum=1
        )
        if component_fold.setdefault(component, fold) != fold:
            raise Loop175PhaseBDataError("content component crosses folds")
        if declared_component_sizes.setdefault(component, component_size) != component_size:
            raise Loop175PhaseBDataError("content component size declaration drifted")

        availability = record.get("availability")
        missing_reason = record.get("missing_reason")
        source_size = record.get("source_size_bytes")
        if availability == "supported":
            _require_integer(source_size, field="source_size_bytes", minimum=1)
            if missing_reason is not None:
                raise Loop175PhaseBDataError("supported row has a missing reason")
        elif availability in {"parse_failure", "oversize"}:
            _require_integer(source_size, field="source_size_bytes")
            if missing_reason != availability:
                raise Loop175PhaseBDataError("unavailable row reason drifted")
        elif availability == "read_failure":
            if source_size is not None or missing_reason != "read_failure":
                raise Loop175PhaseBDataError("read_failure metadata drifted")
        else:
            raise Loop175PhaseBDataError("fold availability is outside the sealed vocabulary")

        labels[ordinal] = label
        folds[ordinal] = fold
        component_ids.append(component)
        source_sha256.append(source_sha)

    if np.bincount(labels, minlength=2).tolist() != [ROWS_PER_CLASS, ROWS_PER_CLASS]:
        raise Loop175PhaseBDataError("fold labels are not exactly balanced 10k/10k")
    if np.bincount(folds, minlength=FOLD_COUNT).tolist() != [ROWS_PER_FOLD] * FOLD_COUNT:
        raise Loop175PhaseBDataError("folds are not exactly five 4,000-row partitions")
    observed_component_sizes: dict[str, int] = {}
    for component in component_ids:
        observed_component_sizes[component] = observed_component_sizes.get(component, 0) + 1
    if observed_component_sizes != declared_component_sizes:
        raise Loop175PhaseBDataError("component membership counts drifted")
    return CanonicalFoldAlignment(
        labels=_readonly(labels, dtype=np.dtype("u1")),
        folds=_readonly(folds, dtype=np.dtype("i1")),
        component_ids=tuple(component_ids),
        source_sha256=tuple(source_sha256),
        manifest_sha256=expected_sha,
    )


def load_canonical_train_prefix(
    path: Path | str,
    *,
    expected_prefix_sha256: str,
    expected_prefix_bytes: int,
) -> CanonicalTrainAlignment:
    """Read only the SHA-bound Train prefix and never parse later heldout rows."""

    expected_sha = require_sha256(expected_prefix_sha256, field="train_prefix.sha256")
    prefix_bytes = _require_integer(expected_prefix_bytes, field="train_prefix.bytes", minimum=1)
    safe_path = _require_regular_file(Path(path), label="canonical split")
    with safe_path.open("rb") as handle:
        raw = handle.read(prefix_bytes)
    if len(raw) != prefix_bytes or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise Loop175PhaseBDataError("canonical Train prefix binding drifted")
    if not raw.endswith(b"\n"):
        raise Loop175PhaseBDataError("canonical Train prefix ends inside a CSV record")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise Loop175PhaseBDataError("canonical Train prefix is not UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != SPLIT_FIELDS:
        raise Loop175PhaseBDataError("canonical split columns drifted")
    rows = list(reader)
    if len(rows) != FULL_TRAIN_ROWS:
        raise Loop175PhaseBDataError("canonical Train prefix must contain exactly 20,000 rows")

    labels = np.empty(FULL_TRAIN_ROWS, dtype=np.uint8)
    sample_indices = np.empty(FULL_TRAIN_ROWS, dtype=np.int64)
    source_sha256: list[str] = []
    seen_sha: set[str] = set()
    for ordinal, row in enumerate(rows):
        if set(row) != set(SPLIT_FIELDS) or row.get("split") != "train":
            raise Loop175PhaseBDataError("canonical Train row scope drifted")
        try:
            sample_index = int(row["sample_index"])
            label = int(row["label"])
        except (TypeError, ValueError) as error:
            raise Loop175PhaseBDataError("canonical Train target is not an integer") from error
        if sample_index != ordinal or label not in {0, 1}:
            raise Loop175PhaseBDataError("canonical Train index or label drifted")
        source_sha = require_sha256(row.get("source_sha256"), field="source_sha256")
        if source_sha in seen_sha:
            raise Loop175PhaseBDataError("canonical Train prefix repeats a source_sha256")
        seen_sha.add(source_sha)
        source_path = row.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            raise Loop175PhaseBDataError("canonical Train source_path is missing")
        labels[ordinal] = label
        sample_indices[ordinal] = sample_index
        source_sha256.append(source_sha)
    if np.bincount(labels, minlength=2).tolist() != [ROWS_PER_CLASS, ROWS_PER_CLASS]:
        raise Loop175PhaseBDataError("canonical Train labels are not exactly balanced")
    return CanonicalTrainAlignment(
        labels=_readonly(labels, dtype=np.dtype("u1")),
        sample_indices=_readonly(sample_indices, dtype=np.dtype("<i8")),
        source_sha256=tuple(source_sha256),
        prefix_sha256=expected_sha,
        prefix_bytes=prefix_bytes,
    )


def load_aligned_phase_b_data(
    project_root: Path | str,
    *,
    fold_manifest_path: Path | str | None = None,
    fold_manifest_sha256: str = CANONICAL_FOLD_SHA256,
    canonical_split_path: Path | str | None = None,
    train_prefix_sha256: str = CANONICAL_TRAIN_PREFIX_SHA256,
    train_prefix_bytes: int = CANONICAL_TRAIN_PREFIX_BYTES,
) -> AlignedPhaseBData:
    """Verify the official B0 cache and align it to Train identities by SHA."""

    if fold_manifest_sha256 != CANONICAL_FOLD_SHA256:
        raise Loop175PhaseBDataError("B0 cache requires the canonical fold-manifest SHA-256")
    if (
        train_prefix_sha256 != CANONICAL_TRAIN_PREFIX_SHA256
        or train_prefix_bytes != CANONICAL_TRAIN_PREFIX_BYTES
    ):
        raise Loop175PhaseBDataError("B0 cache requires the canonical Train-prefix binding")
    root = Path(project_root).resolve(strict=True)
    fold_path = Path(fold_manifest_path) if fold_manifest_path else root / CANONICAL_FOLD_RELATIVE_PATH
    split_path = Path(canonical_split_path) if canonical_split_path else root / CANONICAL_SPLIT_RELATIVE_PATH
    folds = load_canonical_fold_manifest(fold_path, expected_sha256=fold_manifest_sha256)
    train = load_canonical_train_prefix(
        split_path,
        expected_prefix_sha256=train_prefix_sha256,
        expected_prefix_bytes=train_prefix_bytes,
    )

    # SHA 只存在于控制面对齐；B0 数组仍沿 sealed fold ordinal 进入模型面。
    train_by_sha = {sha256: index for index, sha256 in enumerate(train.source_sha256)}
    if set(train_by_sha) != set(folds.source_sha256):
        raise Loop175PhaseBDataError("fold manifest and canonical Train SHA sets differ")
    for fold_ordinal, source_sha in enumerate(folds.source_sha256):
        train_ordinal = train_by_sha[source_sha]
        if (
            int(train.sample_indices[train_ordinal]) != fold_ordinal
            or int(train.labels[train_ordinal]) != int(folds.labels[fold_ordinal])
        ):
            raise Loop175PhaseBDataError("SHA-aligned Train index or label drifted")

    verified = load_verified_v12_cache_for_v13(root)
    cache = verified.loaded_cache.cache
    b0_values = np.asarray(cache.b0_values)
    b0_missing = np.asarray(cache.b0_missing_indicators)
    if b0_values.shape != (FULL_TRAIN_ROWS, B0_FEATURE_DIMENSION):
        raise Loop175PhaseBDataError("official B0 cache is not exactly 20,000 x 571")
    if not np.isfinite(b0_values).all():
        raise Loop175PhaseBDataError("official B0 cache contains non-finite values")
    if b0_missing.shape != (FULL_TRAIN_ROWS, 6) or not np.isin(b0_missing, (0, 1)).all():
        raise Loop175PhaseBDataError("official B0 missing indicators drifted")
    return AlignedPhaseBData(
        labels=_readonly(folds.labels, dtype=np.dtype("u1")),
        folds=_readonly(folds.folds, dtype=np.dtype("i1")),
        component_ids=folds.component_ids,
        source_sha256=folds.source_sha256,
        b0_values=_readonly(b0_values, dtype=np.dtype("<f4")),
        b0_missing_counts=tuple(int(value) for value in b0_missing.sum(axis=0, dtype=np.int64)),
        fold_manifest_sha256=folds.manifest_sha256,
        train_prefix_sha256=train.prefix_sha256,
        b0_cache_sha256=verified.cache_sha256,
    )


def offset_bucket(start: int, file_size: int) -> int:
    """Return floor(63 * start / max(file_size - 1, 1)) with strict bounds."""

    start_value = _require_integer(start, field="region.start")
    file_size_value = _require_integer(file_size, field="file_size")
    if file_size_value == 0:
        if start_value != 0:
            raise Loop175PhaseBDataError("empty files only permit region start zero")
        return 0
    if start_value >= file_size_value:
        raise Loop175PhaseBDataError("region start is outside the file")
    bucket = (63 * start_value) // max(file_size_value - 1, 1)
    if not 0 <= bucket < BUCKET_COUNT:
        raise Loop175PhaseBDataError("offset bucket is outside 0..63")
    return bucket


def length_bucket(length: int) -> int:
    """Return ceil(63 * length / 8192) with strict frozen-region bounds."""

    length_value = _require_integer(length, field="region.length")
    if length_value > MAXIMUM_REGION_BYTES:
        raise Loop175PhaseBDataError("region length exceeds 8192 bytes")
    bucket = (63 * length_value + MAXIMUM_REGION_BYTES - 1) // MAXIMUM_REGION_BYTES
    if not 0 <= bucket < BUCKET_COUNT:
        raise Loop175PhaseBDataError("length bucket is outside 0..63")
    return bucket


def _require_exact_array(value: object, *, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.ndim != 1:
        raise Loop175PhaseBDataError(f"{name} must be a one-dimensional {dtype.str} array")
    return np.ascontiguousarray(value)


def validate_ragged_region_cache(
    cache: RaggedRegionCache,
    *,
    expected_rows: int | None = None,
) -> RaggedRegionCache:
    """Validate a ragged numeric cache and return immutable normalized arrays."""

    if not isinstance(cache, RaggedRegionCache):
        raise TypeError("cache must be a RaggedRegionCache")
    arrays = {
        field.name: _require_exact_array(
            getattr(cache, field.name), name=field.name, dtype=REGION_ARRAY_DTYPES[field.name]
        )
        for field in fields(RaggedRegionCache)
    }
    row_offsets = arrays["row_region_offsets"]
    if row_offsets.size < 2 or row_offsets[0] != 0 or np.any(np.diff(row_offsets) < 0):
        raise Loop175PhaseBDataError("row_region_offsets are not canonical monotonic offsets")
    row_count = int(row_offsets.size - 1)
    if expected_rows is not None and row_count != expected_rows:
        raise Loop175PhaseBDataError("ragged region cache row count drifted")
    regions_per_row = np.diff(row_offsets)
    if np.any(regions_per_row < 1) or np.any(regions_per_row > MAXIMUM_REGIONS):
        raise Loop175PhaseBDataError("each row must retain between 1 and 16 explicit regions")
    region_count = int(row_offsets[-1])
    for name in ("region_types", "region_starts", "offset_buckets", "length_buckets"):
        if arrays[name].shape != (region_count,):
            raise Loop175PhaseBDataError(f"{name} does not contain one value per region")
    if arrays["file_sizes"].shape != (row_count,) or np.any(arrays["file_sizes"] < 0):
        raise Loop175PhaseBDataError("file_sizes must contain one non-negative value per row")
    token_offsets = arrays["region_token_offsets"]
    if (
        token_offsets.shape != (region_count + 1,)
        or token_offsets[0] != 0
        or np.any(np.diff(token_offsets) < 0)
        or int(token_offsets[-1]) != arrays["token_values"].size
    ):
        raise Loop175PhaseBDataError("region_token_offsets do not span the token payload")
    region_lengths = np.diff(token_offsets)
    if np.any(region_lengths > MAXIMUM_REGION_BYTES):
        raise Loop175PhaseBDataError("a cached region exceeds 8192 bytes")
    if np.any(arrays["region_types"] >= 6):
        raise Loop175PhaseBDataError("region type is outside the frozen 0..5 vocabulary")
    if np.any(arrays["offset_buckets"] >= BUCKET_COUNT) or np.any(
        arrays["length_buckets"] >= BUCKET_COUNT
    ):
        raise Loop175PhaseBDataError("region bucket is outside 0..63")

    row_for_region = np.repeat(np.arange(row_count, dtype=np.int64), regions_per_row)
    total_by_row = np.zeros(row_count, dtype=np.int64)
    np.add.at(total_by_row, row_for_region, region_lengths)
    if np.any(total_by_row > MAXIMUM_TOTAL_REGION_BYTES):
        raise Loop175PhaseBDataError("a row exceeds the 131072-byte region budget")
    for region_index in range(region_count):
        row_index = int(row_for_region[region_index])
        file_size = int(arrays["file_sizes"][row_index])
        start = int(arrays["region_starts"][region_index])
        region_length = int(region_lengths[region_index])
        region_type = int(arrays["region_types"][region_index])
        observed_offset_bucket = int(arrays["offset_buckets"][region_index])
        observed_length_bucket = int(arrays["length_buckets"][region_index])
        if region_type == 0:
            if (start, region_length, observed_offset_bucket, observed_length_bucket) != (0, 0, 0, 0):
                raise Loop175PhaseBDataError("missing regions must be zero-length zero-bucket slots")
            continue
        if file_size == 0 or region_length == 0 or start + region_length > file_size:
            raise Loop175PhaseBDataError("non-missing region range lies outside its file")
        if observed_offset_bucket != offset_bucket(start, file_size):
            raise Loop175PhaseBDataError("offset bucket does not match the frozen formula")
        if observed_length_bucket != length_bucket(region_length):
            raise Loop175PhaseBDataError("length bucket does not match the frozen formula")
    return RaggedRegionCache(
        **{
            name: _readonly(values, dtype=REGION_ARRAY_DTYPES[name])
            for name, values in arrays.items()
        }
    )


def _region_arrays(cache: RaggedRegionCache) -> dict[str, np.ndarray]:
    return {field.name: np.ascontiguousarray(getattr(cache, field.name)) for field in fields(cache)}


def _region_numeric_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(RAGGED_REGION_CACHE_DOMAIN)
    for name in REGION_ARRAY_NAMES:
        values = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(values.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _region_metadata(cache: RaggedRegionCache) -> dict[str, object]:
    arrays = _region_arrays(cache)
    return {
        "schema": RAGGED_REGION_CACHE_SCHEMA,
        "row_count": int(arrays["file_sizes"].size),
        "region_count": int(arrays["region_types"].size),
        "token_count": int(arrays["token_values"].size),
        "maximum_regions": MAXIMUM_REGIONS,
        "maximum_region_bytes": MAXIMUM_REGION_BYTES,
        "maximum_total_region_bytes": MAXIMUM_TOTAL_REGION_BYTES,
        "bucket_count": BUCKET_COUNT,
        "padding_token": PADDING_TOKEN,
        "offset_bucket_formula": "floor(63*start/max(file_size-1,1))",
        "length_bucket_formula": "ceil(63*length/8192)",
        "array_dtypes": {name: arrays[name].dtype.str for name in REGION_ARRAY_NAMES},
        "array_shapes": {name: list(arrays[name].shape) for name in REGION_ARRAY_NAMES},
        "numeric_payload_sha256": _region_numeric_digest(arrays),
    }


def save_ragged_region_cache(path: Path | str, cache: RaggedRegionCache) -> RaggedRegionCacheReceipt:
    """Persist one identity-free region cache through an exclusive ZIP_STORED archive."""

    normalized = validate_ragged_region_cache(cache)
    arrays = _region_arrays(normalized)
    metadata = canonical_json_bytes(_region_metadata(normalized))
    output_path = Path(os.path.abspath(os.fspath(path)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        raise Loop175PhaseBDataError("ragged region cache output already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = -1
    try:
        descriptor = os.open(os.fspath(output_path), flags, 0o600)
        with os.fdopen(descriptor, "w+b", closefd=True) as raw_handle:
            descriptor = -1
            with zipfile.ZipFile(
                raw_handle, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as archive:
                for name in REGION_ARRAY_NAMES:
                    with archive.open(f"{name}.npy", mode="w", force_zip64=True) as member:
                        np.lib.format.write_array(member, arrays[name], allow_pickle=False)
                archive.writestr("metadata.json", metadata, compress_type=zipfile.ZIP_STORED)
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise Loop175PhaseBDataError("ragged region cache exclusive write failed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    sha256, size_bytes = _sha256_file(output_path, maximum_bytes=MAXIMUM_REGION_CACHE_BYTES)
    return RaggedRegionCacheReceipt(
        path=output_path,
        sha256=sha256,
        size_bytes=size_bytes,
        row_count=int(normalized.file_sizes.size),
        region_count=int(normalized.region_types.size),
        token_count=int(normalized.token_values.size),
    )


def load_ragged_region_cache(
    path: Path | str,
    *,
    expected_sha256: str,
    expected_rows: int | None = None,
) -> RaggedRegionCache:
    """Load and fully revalidate one SHA-bound identity-free region cache."""

    expected_sha = require_sha256(expected_sha256, field="region_cache.sha256")
    cache_path = _require_regular_file(Path(path), label="ragged region cache")
    observed_sha, _size_bytes = _sha256_file(cache_path, maximum_bytes=MAXIMUM_REGION_CACHE_BYTES)
    if observed_sha != expected_sha:
        raise Loop175PhaseBDataError("ragged region cache SHA-256 binding drifted")
    try:
        with zipfile.ZipFile(cache_path, mode="r") as archive:
            infos = archive.infolist()
            if len(infos) != len(REGION_ARCHIVE_NAMES) or {
                info.filename for info in infos
            } != REGION_ARCHIVE_NAMES:
                raise Loop175PhaseBDataError("ragged region cache members drifted")
            total = 0
            for info in infos:
                if info.compress_type != zipfile.ZIP_STORED or info.file_size < 0:
                    raise Loop175PhaseBDataError("ragged region cache compression drifted")
                total += int(info.file_size)
                if total > MAXIMUM_REGION_CACHE_BYTES:
                    raise Loop175PhaseBDataError("ragged region cache expands beyond its limit")
            arrays = {}
            for name in REGION_ARRAY_NAMES:
                with archive.open(f"{name}.npy", mode="r") as member:
                    arrays[name] = np.lib.format.read_array(member, allow_pickle=False)
            metadata_raw = archive.read("metadata.json")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise Loop175PhaseBDataError("ragged region cache cannot be read") from error
    metadata = _parse_canonical_json_line(metadata_raw, context="ragged region metadata")
    if set(metadata) != REGION_METADATA_FIELDS or metadata.get("schema") != RAGGED_REGION_CACHE_SCHEMA:
        raise Loop175PhaseBDataError("ragged region metadata schema drifted")
    normalized = validate_ragged_region_cache(RaggedRegionCache(**arrays), expected_rows=expected_rows)
    expected_metadata = _region_metadata(normalized)
    if metadata != expected_metadata:
        raise Loop175PhaseBDataError("ragged region metadata does not bind its numeric payload")
    return normalized


def make_identity_free_fit_payload(
    aligned: AlignedPhaseBData,
    regions: RaggedRegionCache,
) -> IdentityFreePhaseBFitPayload:
    """Cross the fit boundary after stripping all identity and path metadata."""

    if not isinstance(aligned, AlignedPhaseBData):
        raise TypeError("aligned must be AlignedPhaseBData")
    normalized_regions = validate_ragged_region_cache(regions, expected_rows=FULL_TRAIN_ROWS)
    if not np.all(np.diff(normalized_regions.row_region_offsets) == MAXIMUM_REGIONS):
        raise Loop175PhaseBDataError("production fit requires exactly 16 regions per row")
    if aligned.b0_values.shape != (FULL_TRAIN_ROWS, B0_FEATURE_DIMENSION):
        raise Loop175PhaseBDataError("aligned B0 values drifted before fit")
    # 该 dataclass 的字段集合是模型面白名单，SHA/path/component 永不越过此处。
    return IdentityFreePhaseBFitPayload(
        b0_values=_readonly(aligned.b0_values, dtype=np.dtype("<f4")),
        labels=_readonly(aligned.labels, dtype=np.dtype("u1")),
        folds=_readonly(aligned.folds, dtype=np.dtype("i1")),
        regions=normalized_regions,
    )


if tuple(field.name for field in fields(IdentityFreePhaseBFitPayload)) != (
    "b0_values",
    "labels",
    "folds",
    "regions",
):
    raise RuntimeError("Loop175 fit payload identity boundary drifted")
