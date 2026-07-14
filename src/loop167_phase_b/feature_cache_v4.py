"""Bounded v4 Phase-B feature-cache persistence without source identities or targets."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import PhaseBContractError, canonical_json_bytes, require_sha256
from .fit_worker import (
    B0_MISSING_DIMENSION,
    B0_VALUE_DIMENSION,
    B1_MISSING_DIMENSION,
    B1_VALUE_DIMENSION,
    FULL_TRAIN_ROWS,
    NOVEL_VALUE_DIMENSION,
    PhaseBFeatureCache,
    validate_phase_b_fit_input,
)
from .progress_ledger import FEATURE_ROW_GENESIS_SHA256, RAW_RESULTS, _next_feature_rows_commitment
from .raw_worker import RawFeatureRow, RawScanOutcome, _feature_row_commitment

CACHE_SCHEMA = "axon_loop167_phase_b_feature_cache_v4"
MAX_FEATURE_CACHE_BYTES = 1_073_741_824
NUMERIC_CACHE_DOMAIN = b"axon_loop167_phase_b_feature_cache_numeric_v4\0"
SAMPLING_AUDIT_DOMAIN = b"axon_loop167_phase_b_b1_sampling_audit_v4\0"
ARRAY_NAMES = (
    "b0_values",
    "b0_missing_indicators",
    "b1_values",
    "b1_missing_indicators",
    "novel_values",
    "novel_complete",
)
ARCHIVE_NAMES = frozenset({*(f"{name}.npy" for name in ARRAY_NAMES), "metadata.json"})
METADATA_FIELDS = frozenset(
    {
        "schema",
        "row_count",
        "raw_scope_commitment_sha256",
        "feature_rows_commitment_sha256",
        "raw_ledger_final_record_sha256",
        "numeric_payload_sha256",
        "b1_sampling_audit",
    }
)


class FeatureCacheV4Error(PhaseBContractError):
    """Raised when a Phase-B cache cannot be written or loaded safely."""


@dataclass(frozen=True, slots=True)
class B1SamplingAudit:
    """Aggregate-only accounting for the three B1 sampling indicators."""

    row_count: int
    indicator_counts: tuple[int, int, int]
    sha256: str

    def to_metadata(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "indicator_counts": list(self.indicator_counts),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class PhaseBFitPayload:
    """The fitting handoff contains exactly numeric cache data, labels, and folds."""

    cache: PhaseBFeatureCache
    labels: np.ndarray
    folds: np.ndarray


@dataclass(frozen=True, slots=True)
class FeatureCacheWriteReceipt:
    cache_path: Path
    cache_sha256: str
    cache_bytes: int
    raw_scope_commitment_sha256: str
    feature_rows_commitment_sha256: str
    raw_ledger_final_record_sha256: str
    sampling_audit: B1SamplingAudit


@dataclass(frozen=True, slots=True)
class LoadedFeatureCacheV4:
    cache: PhaseBFeatureCache
    raw_scope_commitment_sha256: str
    feature_rows_commitment_sha256: str
    raw_ledger_final_record_sha256: str
    sampling_audit: B1SamplingAudit


class _CappedWriter:
    """File wrapper that prevents ZIP output from ever crossing the hard byte cap."""

    def __init__(self, handle: Any, *, maximum_bytes: int) -> None:
        self._handle = handle
        self._maximum_bytes = maximum_bytes

    def write(self, content: bytes) -> int:
        if self._handle.tell() + len(content) > self._maximum_bytes:
            raise FeatureCacheV4Error("Feature cache would exceed its fixed one-GiB cap")
        return self._handle.write(content)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise FeatureCacheV4Error(f"Metadata repeats key: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite(value: str) -> object:
    raise FeatureCacheV4Error(f"Metadata uses non-finite constant: {value}")


def _parse_canonical_metadata(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeatureCacheV4Error("Feature-cache metadata is not canonical JSON") from exc
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise FeatureCacheV4Error("Feature-cache metadata is not canonical JSON")
    return payload


def _readonly(values: np.ndarray, *, dtype: np.dtype[Any] | type[np.generic]) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _float_matrix(values: object, *, name: str, rows: int, columns: int) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FeatureCacheV4Error(f"{name} is not a finite float32 matrix") from exc
    if matrix.shape != (rows, columns) or not np.isfinite(matrix).all():
        raise FeatureCacheV4Error(f"{name} shape or finiteness drifted")
    return matrix


def _float_vector(values: object, *, name: str, columns: int) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FeatureCacheV4Error(f"{name} is not a finite float32 vector") from exc
    if vector.shape != (columns,) or not np.isfinite(vector).all():
        raise FeatureCacheV4Error(f"{name} shape or finiteness drifted")
    return vector


def _binary_matrix(values: object, *, name: str, rows: int, columns: int) -> np.ndarray:
    matrix = _float_matrix(values, name=name, rows=rows, columns=columns)
    if not np.isin(matrix, (0.0, 1.0)).all():
        raise FeatureCacheV4Error(f"{name} must contain binary indicators")
    return matrix


def _binary_vector(values: object, *, name: str, columns: int) -> np.ndarray:
    vector = _float_vector(values, name=name, columns=columns)
    if not np.isin(vector, (0.0, 1.0)).all():
        raise FeatureCacheV4Error(f"{name} must contain binary indicators")
    return vector


def _boolean_vector(values: object, *, name: str, rows: int) -> np.ndarray:
    vector = np.asarray(values)
    if vector.dtype != np.dtype(bool) or vector.shape != (rows,):
        raise FeatureCacheV4Error(f"{name} must be a boolean vector with one value per row")
    return np.ascontiguousarray(vector)


def _array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(NUMERIC_CACHE_DOMAIN)
    for name in ARRAY_NAMES:
        values = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _sampling_audit(indicators: np.ndarray) -> B1SamplingAudit:
    counts = tuple(int(value) for value in np.sum(indicators, axis=0, dtype=np.int64))
    payload = canonical_json_bytes(
        {"indicator_counts": list(counts), "row_count": int(indicators.shape[0])}
    )
    return B1SamplingAudit(
        row_count=int(indicators.shape[0]),
        indicator_counts=(counts[0], counts[1], counts[2]),
        sha256=hashlib.sha256(SAMPLING_AUDIT_DOMAIN + payload).hexdigest(),
    )


def _validate_sampling_metadata(value: object, *, rows: int) -> B1SamplingAudit:
    if not isinstance(value, dict) or set(value) != {"row_count", "indicator_counts", "sha256"}:
        raise FeatureCacheV4Error("Feature-cache sampling audit fields drifted")
    if value["row_count"] != rows:
        raise FeatureCacheV4Error("Feature-cache sampling audit row count drifted")
    counts = value["indicator_counts"]
    if (
        not isinstance(counts, list)
        or len(counts) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 or item > rows for item in counts)
    ):
        raise FeatureCacheV4Error("Feature-cache sampling audit counts drifted")
    candidate = B1SamplingAudit(
        row_count=rows,
        indicator_counts=(int(counts[0]), int(counts[1]), int(counts[2])),
        sha256=require_sha256(value["sha256"], field="b1_sampling_audit.sha256"),
    )
    expected_payload = canonical_json_bytes(
        {"indicator_counts": list(candidate.indicator_counts), "row_count": rows}
    )
    expected_sha256 = hashlib.sha256(SAMPLING_AUDIT_DOMAIN + expected_payload).hexdigest()
    if candidate.sha256 != expected_sha256:
        raise FeatureCacheV4Error("Feature-cache sampling audit hash drifted")
    return candidate


def _validate_outcome(
    outcome: RawScanOutcome,
    *,
    expected_raw_scope_commitment_sha256: str,
    expected_rows: int,
) -> tuple[dict[str, np.ndarray], B1SamplingAudit]:
    if not isinstance(outcome, RawScanOutcome):
        raise TypeError("outcome must be a RawScanOutcome")
    expected_scope = require_sha256(
        expected_raw_scope_commitment_sha256,
        field="expected_raw_scope_commitment_sha256",
    )
    if outcome.raw_scope_commitment_sha256 != expected_scope:
        raise FeatureCacheV4Error("Raw outcome scope commitment drifted")
    require_sha256(outcome.feature_rows_commitment_sha256, field="feature_rows_commitment_sha256")
    require_sha256(outcome.raw_ledger_final_record_sha256, field="raw_ledger_final_record_sha256")
    if len(outcome.rows) != expected_rows:
        raise FeatureCacheV4Error("Raw outcome row count does not match the sealed cache denominator")

    arrays = {
        "b0_values": np.empty((expected_rows, B0_VALUE_DIMENSION), dtype=np.float32),
        "b0_missing_indicators": np.empty((expected_rows, B0_MISSING_DIMENSION), dtype=np.float32),
        "b1_values": np.empty((expected_rows, B1_VALUE_DIMENSION), dtype=np.float32),
        "b1_missing_indicators": np.empty((expected_rows, B1_MISSING_DIMENSION), dtype=np.float32),
        "novel_values": np.empty((expected_rows, NOVEL_VALUE_DIMENSION), dtype=np.float32),
        "novel_complete": np.empty(expected_rows, dtype=bool),
    }
    sampling = np.empty((expected_rows, 3), dtype=np.float32)
    seen_audits: set[str] = set()
    feature_rows_commitment = FEATURE_ROW_GENESIS_SHA256
    for ordinal, row in enumerate(outcome.rows):
        if not isinstance(row, RawFeatureRow) or row.ordinal != ordinal:
            raise FeatureCacheV4Error("Raw outcome rows are not contiguous")
        if not isinstance(row.result, str) or row.result not in RAW_RESULTS:
            raise FeatureCacheV4Error("Raw outcome result is outside the sealed vocabulary")
        require_sha256(row.source_audit_sha256, field="source_audit_sha256")
        require_sha256(row.feature_row_commitment_sha256, field="feature_row_commitment_sha256")
        if row.source_audit_sha256 in seen_audits:
            raise FeatureCacheV4Error("Raw outcome repeats a source audit identity")
        seen_audits.add(row.source_audit_sha256)
        b0_values = _float_vector(
            row.b0_values,
            name="b0_values",
            columns=B0_VALUE_DIMENSION,
        )
        b0_missing = _binary_vector(
            row.b0_missing_indicators,
            name="b0_missing_indicators",
            columns=B0_MISSING_DIMENSION,
        )
        b1_values = _float_vector(
            row.b1_values,
            name="b1_values",
            columns=B1_VALUE_DIMENSION,
        )
        b1_missing = _binary_vector(
            row.b1_missing_indicators,
            name="b1_missing_indicators",
            columns=B1_MISSING_DIMENSION,
        )
        b1_sampling = _binary_vector(
            row.b1_sampling_indicators,
            name="b1_sampling_indicators",
            columns=3,
        )
        novel_values = _float_vector(
            row.novel_values,
            name="novel_values",
            columns=NOVEL_VALUE_DIMENSION,
        )
        novel_missing = _binary_vector(
            row.novel_missing_indicators,
            name="novel_missing_indicators",
            columns=1,
        )
        if not isinstance(row.novel_complete, bool):
            raise FeatureCacheV4Error("novel_complete must remain boolean")
        expected_row_commitment = _feature_row_commitment(
            ordinal=ordinal,
            source_audit_sha256=row.source_audit_sha256,
            result=row.result,
            b0_values=b0_values,
            b0_missing_indicators=b0_missing,
            b1_values=b1_values,
            b1_missing_indicators=b1_missing,
            b1_sampling_indicators=b1_sampling,
            novel_values=novel_values,
            novel_missing_indicators=novel_missing,
            novel_complete=row.novel_complete,
        )
        if row.feature_row_commitment_sha256 != expected_row_commitment:
            raise FeatureCacheV4Error("Raw outcome feature-row commitment drifted")
        feature_rows_commitment = _next_feature_rows_commitment(
            feature_rows_commitment,
            ordinal=ordinal,
            source_audit_sha256=row.source_audit_sha256,
            feature_row_commitment_sha256=row.feature_row_commitment_sha256,
        )
        arrays["b0_values"][ordinal] = b0_values
        arrays["b0_missing_indicators"][ordinal] = b0_missing
        arrays["b1_values"][ordinal] = b1_values
        arrays["b1_missing_indicators"][ordinal] = b1_missing
        arrays["novel_values"][ordinal] = novel_values
        sampling[ordinal] = b1_sampling
        arrays["novel_complete"][ordinal] = row.novel_complete
    if outcome.feature_rows_commitment_sha256 != feature_rows_commitment:
        raise FeatureCacheV4Error("Raw outcome aggregate feature commitment drifted")
    frozen_arrays = {
        "b0_values": _readonly(arrays["b0_values"], dtype=np.float32),
        "b0_missing_indicators": _readonly(arrays["b0_missing_indicators"], dtype=np.float32),
        "b1_values": _readonly(arrays["b1_values"], dtype=np.float32),
        "b1_missing_indicators": _readonly(arrays["b1_missing_indicators"], dtype=np.float32),
        "novel_values": _readonly(arrays["novel_values"], dtype=np.float32),
        "novel_complete": _readonly(arrays["novel_complete"], dtype=bool),
    }
    return frozen_arrays, _sampling_audit(sampling)


def _is_symlink_or_reparse(stat_result: os.stat_result) -> bool:
    if stat.S_ISLNK(stat_result.st_mode):
        return True
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400))
    return bool(attributes & reparse_flag)


def _absolute_lexical(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise FeatureCacheV4Error("Feature cache path must be absolute")
    return Path(os.path.abspath(os.fspath(candidate)))


def _assert_safe_directory_ancestry(path: Path, *, label: str) -> Path:
    absolute_path = _absolute_lexical(path)
    anchor = Path(absolute_path.anchor)
    if not absolute_path.is_absolute() or not anchor:
        raise FeatureCacheV4Error(f"{label} must be absolute")
    cursor = anchor
    try:
        anchor_stat = os.lstat(cursor)
    except OSError as exc:
        raise FeatureCacheV4Error(f"{label} anchor is inaccessible") from exc
    if _is_symlink_or_reparse(anchor_stat) or not stat.S_ISDIR(anchor_stat.st_mode):
        raise FeatureCacheV4Error(f"{label} anchor is unsafe")
    for component in absolute_path.parts[1:]:
        cursor = cursor / component
        try:
            current_stat = os.lstat(cursor)
        except OSError as exc:
            raise FeatureCacheV4Error(f"{label} ancestry is inaccessible") from exc
        if _is_symlink_or_reparse(current_stat):
            raise FeatureCacheV4Error(f"{label} ancestry contains a symlink or reparse point")
        if not stat.S_ISDIR(current_stat.st_mode):
            raise FeatureCacheV4Error(f"{label} ancestry is not a directory")
    return absolute_path


def _prepare_new_cache_path(path: Path) -> Path:
    absolute_path = _absolute_lexical(path)
    if not absolute_path.is_absolute() or not absolute_path.name:
        raise FeatureCacheV4Error("Feature cache path must be a named absolute path")
    _assert_safe_directory_ancestry(absolute_path.parent, label="feature cache parent")
    try:
        existing = os.lstat(absolute_path)
    except FileNotFoundError:
        return absolute_path
    except OSError as exc:
        raise FeatureCacheV4Error("Feature cache path is inaccessible") from exc
    if _is_symlink_or_reparse(existing):
        raise FeatureCacheV4Error("Feature cache output path is unsafe")
    raise FeatureCacheV4Error("Feature cache output already exists")


def _estimate_archive_upper_bound(arrays: Mapping[str, np.ndarray], metadata: bytes) -> int:
    payload_bytes = sum(int(np.ascontiguousarray(arrays[name]).nbytes) for name in ARRAY_NAMES)
    return payload_bytes + len(metadata) + 64 * 1024


def _write_array(archive: zipfile.ZipFile, *, name: str, values: np.ndarray) -> None:
    with archive.open(f"{name}.npy", mode="w", force_zip64=False) as handle:
        np.lib.format.write_array(handle, values, version=(1, 0), allow_pickle=False)


def _write_cache(
    output_path: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
    maximum_bytes: int,
) -> tuple[Path, int]:
    if maximum_bytes < 1 or maximum_bytes > MAX_FEATURE_CACHE_BYTES:
        raise FeatureCacheV4Error("Feature-cache byte cap cannot exceed one GiB")
    metadata_raw = canonical_json_bytes(dict(metadata))
    if _estimate_archive_upper_bound(arrays, metadata_raw) > maximum_bytes:
        raise FeatureCacheV4Error("Feature cache exceeds its fixed one-GiB cap before writing")
    cache_path = _prepare_new_cache_path(output_path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(os.fspath(cache_path), flags, 0o600)
    except FileExistsError as exc:
        raise FeatureCacheV4Error("Feature cache output already exists") from exc
    except OSError as exc:
        raise FeatureCacheV4Error("Feature cache cannot be created with O_EXCL") from exc
    try:
        with os.fdopen(descriptor, "w+b", closefd=True) as raw_handle:
            descriptor = -1
            capped_handle = _CappedWriter(raw_handle, maximum_bytes=maximum_bytes)
            with zipfile.ZipFile(
                capped_handle,
                mode="w",
                compression=zipfile.ZIP_STORED,
                allowZip64=False,
            ) as archive:
                for name in ARRAY_NAMES:
                    _write_array(archive, name=name, values=arrays[name])
                archive.writestr("metadata.json", metadata_raw, compress_type=zipfile.ZIP_STORED)
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
            size_bytes = int(os.fstat(raw_handle.fileno()).st_size)
        if size_bytes > maximum_bytes:
            raise FeatureCacheV4Error("Feature cache exceeded its fixed one-GiB cap")
    except FeatureCacheV4Error:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise FeatureCacheV4Error("Feature-cache exclusive write failed; partial output remains sealed") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    return cache_path, size_bytes


def _sha256_regular_file(path: Path, *, maximum_bytes: int) -> str:
    absolute_path = _absolute_lexical(path)
    _assert_safe_directory_ancestry(absolute_path.parent, label="feature cache parent")
    try:
        cache_stat = os.lstat(absolute_path)
    except OSError as exc:
        raise FeatureCacheV4Error("Feature cache is missing or inaccessible") from exc
    if _is_symlink_or_reparse(cache_stat) or not stat.S_ISREG(cache_stat.st_mode):
        raise FeatureCacheV4Error("Feature cache is not a safe regular file")
    if cache_stat.st_size > maximum_bytes:
        raise FeatureCacheV4Error("Feature cache exceeds its fixed one-GiB cap")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(os.fspath(absolute_path), flags)
    except OSError as exc:
        raise FeatureCacheV4Error("Feature cache cannot be hashed safely") from exc
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
                raise FeatureCacheV4Error("Feature cache is not a bounded regular file")
            digest = hashlib.sha256()
            total = 0
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                total += len(block)
                if total > maximum_bytes:
                    raise FeatureCacheV4Error("Feature cache exceeds its fixed one-GiB cap")
                digest.update(block)
            after = os.fstat(handle.fileno())
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or total != before.st_size
        ):
            raise FeatureCacheV4Error("Feature cache changed while it was hashed")
        return digest.hexdigest()
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def write_phase_b_feature_cache_v4(
    output_path: Path,
    outcome: RawScanOutcome,
    *,
    expected_raw_scope_commitment_sha256: str,
) -> FeatureCacheWriteReceipt:
    """Persist exactly the approved 20k numeric cache through a new O_EXCL file."""

    arrays, sampling_audit = _validate_outcome(
        outcome,
        expected_raw_scope_commitment_sha256=expected_raw_scope_commitment_sha256,
        expected_rows=FULL_TRAIN_ROWS,
    )
    metadata = {
        "schema": CACHE_SCHEMA,
        "row_count": FULL_TRAIN_ROWS,
        "raw_scope_commitment_sha256": outcome.raw_scope_commitment_sha256,
        "feature_rows_commitment_sha256": outcome.feature_rows_commitment_sha256,
        "raw_ledger_final_record_sha256": outcome.raw_ledger_final_record_sha256,
        "numeric_payload_sha256": _array_digest(arrays),
        "b1_sampling_audit": sampling_audit.to_metadata(),
    }
    cache_path, size_bytes = _write_cache(
        output_path,
        arrays=arrays,
        metadata=metadata,
        maximum_bytes=MAX_FEATURE_CACHE_BYTES,
    )
    return FeatureCacheWriteReceipt(
        cache_path=cache_path,
        cache_sha256=_sha256_regular_file(cache_path, maximum_bytes=MAX_FEATURE_CACHE_BYTES),
        cache_bytes=size_bytes,
        raw_scope_commitment_sha256=outcome.raw_scope_commitment_sha256,
        feature_rows_commitment_sha256=outcome.feature_rows_commitment_sha256,
        raw_ledger_final_record_sha256=outcome.raw_ledger_final_record_sha256,
        sampling_audit=sampling_audit,
    )


def _load_arrays(
    cache_path: Path,
    *,
    expected_rows: int,
    maximum_bytes: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    absolute_path = _absolute_lexical(cache_path)
    _assert_safe_directory_ancestry(absolute_path.parent, label="feature cache parent")
    try:
        cache_stat = os.lstat(absolute_path)
    except OSError as exc:
        raise FeatureCacheV4Error("Feature cache is missing or inaccessible") from exc
    if _is_symlink_or_reparse(cache_stat) or not stat.S_ISREG(cache_stat.st_mode):
        raise FeatureCacheV4Error("Feature cache is not a safe regular file")
    if cache_stat.st_size > maximum_bytes:
        raise FeatureCacheV4Error("Feature cache exceeds its fixed one-GiB cap")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(os.fspath(absolute_path), flags)
    except OSError as exc:
        raise FeatureCacheV4Error("Feature cache cannot be opened safely") from exc
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as raw_handle:
            descriptor = -1
            before = os.fstat(raw_handle.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
                raise FeatureCacheV4Error("Feature cache is not a bounded regular file")
            with zipfile.ZipFile(raw_handle, mode="r") as archive:
                infos = archive.infolist()
                if len(infos) != len(ARCHIVE_NAMES) or {info.filename for info in infos} != ARCHIVE_NAMES:
                    raise FeatureCacheV4Error("Feature-cache archive members drifted")
                total_uncompressed = 0
                for info in infos:
                    if info.compress_type != zipfile.ZIP_STORED or info.file_size < 0:
                        raise FeatureCacheV4Error("Feature-cache archive compression drifted")
                    total_uncompressed += int(info.file_size)
                    if total_uncompressed > maximum_bytes:
                        raise FeatureCacheV4Error("Feature-cache archive expands beyond its byte cap")
                arrays: dict[str, np.ndarray] = {}
                for name in ARRAY_NAMES:
                    with archive.open(f"{name}.npy", mode="r") as handle:
                        arrays[name] = np.lib.format.read_array(handle, allow_pickle=False)
                metadata = _parse_canonical_metadata(archive.read("metadata.json"))
            after = os.fstat(raw_handle.fileno())
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise FeatureCacheV4Error("Feature cache changed while it was read")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise FeatureCacheV4Error("Feature-cache archive cannot be read safely") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if set(metadata) != METADATA_FIELDS or metadata.get("schema") != CACHE_SCHEMA:
        raise FeatureCacheV4Error("Feature-cache metadata schema drifted")
    if metadata.get("row_count") != expected_rows:
        raise FeatureCacheV4Error("Feature-cache metadata row count drifted")
    return arrays, metadata


def load_phase_b_feature_cache_v4(
    cache_path: Path,
    *,
    expected_cache_sha256: str,
    expected_raw_scope_commitment_sha256: str,
    expected_feature_rows_commitment_sha256: str,
    expected_raw_ledger_final_record_sha256: str,
) -> LoadedFeatureCacheV4:
    """Load a sealed 20k numeric cache without reconstructing source identities or targets."""

    expected_cache_sha = require_sha256(
        expected_cache_sha256,
        field="expected_cache_sha256",
    )
    observed_cache_sha = _sha256_regular_file(
        cache_path,
        maximum_bytes=MAX_FEATURE_CACHE_BYTES,
    )
    if observed_cache_sha != expected_cache_sha:
        raise FeatureCacheV4Error("Feature-cache file SHA-256 binding drifted")
    expected_scope = require_sha256(
        expected_raw_scope_commitment_sha256,
        field="expected_raw_scope_commitment_sha256",
    )
    expected_feature_rows = require_sha256(
        expected_feature_rows_commitment_sha256,
        field="expected_feature_rows_commitment_sha256",
    )
    expected_ledger = require_sha256(
        expected_raw_ledger_final_record_sha256,
        field="expected_raw_ledger_final_record_sha256",
    )
    arrays, metadata = _load_arrays(
        cache_path,
        expected_rows=FULL_TRAIN_ROWS,
        maximum_bytes=MAX_FEATURE_CACHE_BYTES,
    )
    if (
        metadata["raw_scope_commitment_sha256"] != expected_scope
        or metadata["feature_rows_commitment_sha256"] != expected_feature_rows
        or metadata["raw_ledger_final_record_sha256"] != expected_ledger
    ):
        raise FeatureCacheV4Error("Feature-cache commitment binding drifted")
    for field_name in (
        "raw_scope_commitment_sha256",
        "feature_rows_commitment_sha256",
        "raw_ledger_final_record_sha256",
        "numeric_payload_sha256",
    ):
        require_sha256(metadata[field_name], field=field_name)
    normalized = {
        "b0_values": _readonly(
            _float_matrix(arrays["b0_values"], name="b0_values", rows=FULL_TRAIN_ROWS, columns=B0_VALUE_DIMENSION),
            dtype=np.float32,
        ),
        "b0_missing_indicators": _readonly(
            _binary_matrix(
                arrays["b0_missing_indicators"],
                name="b0_missing_indicators",
                rows=FULL_TRAIN_ROWS,
                columns=B0_MISSING_DIMENSION,
            ),
            dtype=np.float32,
        ),
        "b1_values": _readonly(
            _float_matrix(arrays["b1_values"], name="b1_values", rows=FULL_TRAIN_ROWS, columns=B1_VALUE_DIMENSION),
            dtype=np.float32,
        ),
        "b1_missing_indicators": _readonly(
            _binary_matrix(
                arrays["b1_missing_indicators"],
                name="b1_missing_indicators",
                rows=FULL_TRAIN_ROWS,
                columns=B1_MISSING_DIMENSION,
            ),
            dtype=np.float32,
        ),
        "novel_values": _readonly(
            _float_matrix(
                arrays["novel_values"],
                name="novel_values",
                rows=FULL_TRAIN_ROWS,
                columns=NOVEL_VALUE_DIMENSION,
            ),
            dtype=np.float32,
        ),
        "novel_complete": _readonly(
            _boolean_vector(arrays["novel_complete"], name="novel_complete", rows=FULL_TRAIN_ROWS),
            dtype=bool,
        ),
    }
    if _array_digest(normalized) != metadata["numeric_payload_sha256"]:
        raise FeatureCacheV4Error("Feature-cache numeric payload hash drifted")
    sampling_audit = _validate_sampling_metadata(metadata["b1_sampling_audit"], rows=FULL_TRAIN_ROWS)
    return LoadedFeatureCacheV4(
        cache=PhaseBFeatureCache(**normalized),
        raw_scope_commitment_sha256=expected_scope,
        feature_rows_commitment_sha256=expected_feature_rows,
        raw_ledger_final_record_sha256=expected_ledger,
        sampling_audit=sampling_audit,
    )


def make_phase_b_fit_payload(
    cache: PhaseBFeatureCache,
    labels: np.ndarray,
    folds: np.ndarray,
) -> PhaseBFitPayload:
    """Validate the fixed fit boundary without adding controller-only metadata."""

    validate_phase_b_fit_input(cache, labels, folds)
    normalized_labels = _readonly(np.asarray(labels, dtype=np.uint8), dtype=np.uint8)
    normalized_folds = _readonly(np.asarray(folds, dtype=np.int8), dtype=np.int8)
    return PhaseBFitPayload(cache=cache, labels=normalized_labels, folds=normalized_folds)


if tuple(field.name for field in fields(PhaseBFitPayload)) != ("cache", "labels", "folds"):
    raise RuntimeError("PhaseBFitPayload field boundary drifted")
