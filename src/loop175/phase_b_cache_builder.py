"""Resumable Train-only ragged region-cache construction for Loop175 Phase B."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.loop167_phase_b.contracts import canonical_json_bytes

from .phase_b_contract import sha256_file
from .phase_b_data import (
    CANONICAL_FOLD_SHA256,
    FULL_TRAIN_ROWS,
    MAXIMUM_REGION_CACHE_BYTES,
    RaggedRegionCache,
    RaggedRegionCacheReceipt,
    length_bucket,
    load_canonical_fold_manifest,
    offset_bucket,
    save_ragged_region_cache,
    validate_ragged_region_cache,
)
from .region_extractor import (
    Region,
    RegionExtractionConfig,
    RegionExtractionResult,
    RegionKind,
    extract_regions_from_bytes,
)
from .resource_guard import ResourceGuard

LEDGER_SCHEMA = "axon_loop175_region_cache_progress_v1"
LEDGER_DOMAIN = b"axon_loop175_region_cache_progress_v1\0"
ROW_NUMERIC_DOMAIN = b"axon_loop175_region_cache_row_numeric_v1\0"
GENESIS_SHA256 = hashlib.sha256(b"axon_loop175_region_cache_progress_genesis_v1").hexdigest()
RECORD_FIELDS = frozenset(
    {
        "schema",
        "ordinal",
        "file_size",
        "status",
        "supported",
        "token_start",
        "token_end",
        "region_types",
        "region_starts",
        "region_lengths",
        "offset_buckets",
        "length_buckets",
        "row_numeric_sha256",
        "previous_record_sha256",
        "record_sha256",
    }
)


class RegionCacheBuildError(RuntimeError):
    """Raised when source integrity, progress, or cache construction drifts."""


@dataclass(frozen=True, slots=True)
class RegionSource:
    ordinal: int
    path: Path
    source_sha256: str
    declared_size: int | None
    availability: str
    label: int


@dataclass(frozen=True, slots=True)
class RegionCacheProgress:
    records: tuple[Mapping[str, Any], ...]
    committed_token_bytes: int
    final_record_sha256: str


@dataclass(frozen=True, slots=True)
class RegionCacheBuildResult:
    cache: RaggedRegionCacheReceipt
    ledger_path: Path
    ledger_sha256: str
    final_record_sha256: str
    attempted: int
    supported: int
    class_coverage: Mapping[str, float]
    class_coverage_gap: float
    silent_drops: int
    status_counts: Mapping[str, int]
    source_bytes_verified: int
    maximum_rss_bytes: int
    maximum_new_disk_bytes: int
    blockers: tuple[str, ...]
    decision: str


def _sha256_pattern(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RegionCacheBuildError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _missing_result(status: str, *, file_size: int, config: RegionExtractionConfig) -> RegionExtractionResult:
    regions = tuple(
        Region(RegionKind.MISSING, 0, b"", status)
        for _ in range(config.maximum_regions)
    )
    return RegionExtractionResult(status, file_size, 0, False, regions)


def _assert_safe_path_ancestry(path: Path, *, allow_missing_final: bool) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    anchor = Path(absolute.anchor)
    if not absolute.is_absolute() or not anchor:
        raise RegionCacheBuildError("region source path must be absolute")
    cursor = anchor
    parts = absolute.parts[1:]
    for index, component in enumerate(parts):
        cursor = cursor / component
        final = index == len(parts) - 1
        try:
            current = os.lstat(cursor)
        except FileNotFoundError as error:
            if final and allow_missing_final:
                return
            raise RegionCacheBuildError("region source ancestry is missing") from error
        attributes = int(getattr(current, "st_file_attributes", 0))
        if stat.S_ISLNK(current.st_mode) or attributes & 0x0400:
            raise RegionCacheBuildError("region source ancestry contains a symlink or reparse point")
        if final:
            if not stat.S_ISREG(current.st_mode):
                raise RegionCacheBuildError("region source is not a regular file")
        elif not stat.S_ISDIR(current.st_mode):
            raise RegionCacheBuildError("region source ancestry is not a directory")


def source_plan_commitment(sources: Sequence[RegionSource]) -> str:
    digest = hashlib.sha256(b"axon_loop175_region_source_plan_v1\0")
    for expected_ordinal, source in enumerate(sources):
        if source.ordinal != expected_ordinal:
            raise RegionCacheBuildError("region source plan ordinals are not contiguous")
        payload = {
            "ordinal": source.ordinal,
            "source_sha256": _sha256_pattern(source.source_sha256, name="source_sha256"),
            "declared_size": source.declared_size,
            "availability": source.availability,
            "label": source.label,
        }
        digest.update(canonical_json_bytes(payload))
    return digest.hexdigest()


def _bind_staging_scope(
    path: Path,
    *,
    sources: Sequence[RegionSource],
    expected_rows: int,
) -> str:
    commitment = source_plan_commitment(sources)
    expected = {
        "schema": "axon_loop175_region_cache_staging_scope_v1",
        "expected_rows": expected_rows,
        "source_plan_commitment": commitment,
        "maximum_regions": 16,
        "maximum_region_bytes": 8192,
    }
    if path.exists():
        try:
            observed = json.loads(path.read_text(encoding="ascii"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RegionCacheBuildError("region staging scope cannot be read") from error
        if observed != expected or canonical_json_bytes(observed) != path.read_bytes():
            raise RegionCacheBuildError("region staging scope commitment drifted")
        return commitment
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_json_bytes(expected))
        handle.flush()
        os.fsync(handle.fileno())
    return commitment


def load_region_sources(
    fold_manifest: Path | str,
    *,
    expected_sha256: str = CANONICAL_FOLD_SHA256,
    expected_rows: int = FULL_TRAIN_ROWS,
) -> tuple[RegionSource, ...]:
    """Return private raw locators only after the public fold authority validates."""

    path = Path(fold_manifest)
    alignment = load_canonical_fold_manifest(path, expected_sha256=expected_sha256)
    if len(alignment.source_sha256) != expected_rows:
        raise RegionCacheBuildError("region source denominator drifted")
    raw_lines = path.read_bytes().splitlines()
    if len(raw_lines) != expected_rows:
        raise RegionCacheBuildError("region source manifest line count drifted")
    sources: list[RegionSource] = []
    for ordinal, raw_line in enumerate(raw_lines):
        try:
            row = json.loads(raw_line.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RegionCacheBuildError("region source manifest is not valid JSONL") from error
        if not isinstance(row, dict):
            raise RegionCacheBuildError("region source row is not an object")
        source_sha256 = _sha256_pattern(row.get("source_sha256"), name="source_sha256")
        if source_sha256 != alignment.source_sha256[ordinal]:
            raise RegionCacheBuildError("region source identity order drifted")
        availability = row.get("availability")
        declared_size_value = row.get("source_size_bytes")
        declared_size = None if declared_size_value is None else int(declared_size_value)
        if availability == "read_failure":
            if declared_size is not None:
                raise RegionCacheBuildError("read_failure source unexpectedly declares a size")
        elif availability not in {"supported", "parse_failure", "oversize"}:
            raise RegionCacheBuildError("region source availability drifted")
        sources.append(
            RegionSource(
                ordinal=ordinal,
                path=Path(str(row["source_path"])),
                source_sha256=source_sha256,
                declared_size=declared_size,
                availability=str(availability),
                label=int(alignment.labels[ordinal]),
            )
        )
    return tuple(sources)


def extract_verified_source(
    source: RegionSource,
    config: RegionExtractionConfig,
) -> tuple[RegionExtractionResult, int]:
    """Verify one source SHA in the same raw open used to materialize extraction bytes."""

    _assert_safe_path_ancestry(
        source.path,
        allow_missing_final=source.availability == "read_failure",
    )
    if source.availability == "read_failure":
        if source.path.exists():
            raise RegionCacheBuildError("a sealed read_failure source now exists")
        return _missing_result("read_failure", file_size=0, config=config), 0
    try:
        stat_result = source.path.stat()
    except OSError as error:
        raise RegionCacheBuildError("a non-read_failure source is inaccessible") from error
    if source.declared_size is None or stat_result.st_size != source.declared_size:
        raise RegionCacheBuildError("region source size drifted")

    digest = hashlib.sha256()
    total = 0
    keep_payload = stat_result.st_size <= config.maximum_file_bytes
    chunks: list[bytes] = []
    try:
        with source.path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                total += len(block)
                digest.update(block)
                if keep_payload:
                    chunks.append(block)
    except OSError as error:
        raise RegionCacheBuildError("region source changed or failed during its verified read") from error
    if total != stat_result.st_size or digest.hexdigest() != source.source_sha256:
        raise RegionCacheBuildError("region source SHA-256 or byte count drifted")
    if not keep_payload:
        return _missing_result("oversize", file_size=total, config=config), total
    result = extract_regions_from_bytes(b"".join(chunks), config)
    return result, total


def _row_numeric_sha256(record: Mapping[str, Any], token_values: bytes) -> str:
    numeric = {
        key: record[key]
        for key in (
            "ordinal",
            "file_size",
            "status",
            "supported",
            "token_start",
            "token_end",
            "region_types",
            "region_starts",
            "region_lengths",
            "offset_buckets",
            "length_buckets",
        )
    }
    return hashlib.sha256(ROW_NUMERIC_DOMAIN + canonical_json_bytes(numeric) + token_values).hexdigest()


def _record_sha256(record_without_hash: Mapping[str, Any]) -> str:
    return hashlib.sha256(LEDGER_DOMAIN + canonical_json_bytes(record_without_hash)).hexdigest()


def _record_for_result(
    source: RegionSource,
    result: RegionExtractionResult,
    *,
    token_start: int,
) -> tuple[dict[str, Any], bytes]:
    if len(result.regions) != 16:
        raise RegionCacheBuildError("region extractor did not retain exactly 16 slots")
    token_values = b"".join(region.data for region in result.regions)
    token_end = token_start + len(token_values)
    region_types = [int(region.kind) for region in result.regions]
    region_starts = [int(region.start) for region in result.regions]
    region_lengths = [int(region.length) for region in result.regions]
    offset_buckets = [
        0 if region.kind is RegionKind.MISSING else offset_bucket(region.start, result.file_size)
        for region in result.regions
    ]
    length_buckets = [length_bucket(region.length) for region in result.regions]
    core: dict[str, Any] = {
        "schema": LEDGER_SCHEMA,
        "ordinal": source.ordinal,
        "file_size": int(result.file_size),
        "status": result.status,
        "supported": bool(result.supported),
        "token_start": token_start,
        "token_end": token_end,
        "region_types": region_types,
        "region_starts": region_starts,
        "region_lengths": region_lengths,
        "offset_buckets": offset_buckets,
        "length_buckets": length_buckets,
    }
    core["row_numeric_sha256"] = _row_numeric_sha256(core, token_values)
    return core, token_values


def _parse_ledger_line(raw_line: bytes) -> dict[str, Any]:
    try:
        pairs = json.loads(raw_line.decode("ascii"), object_pairs_hook=list)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RegionCacheBuildError("region progress ledger contains invalid JSON") from error
    if not isinstance(pairs, list):
        raise RegionCacheBuildError("region progress ledger record is not an object")
    record: dict[str, Any] = {}
    for pair in pairs:
        if not isinstance(pair, tuple | list) or len(pair) != 2:
            raise RegionCacheBuildError("region progress ledger object pairs drifted")
        key, value = pair
        if key in record:
            raise RegionCacheBuildError("region progress ledger repeats a key")
        record[str(key)] = value
    if set(record) != RECORD_FIELDS or canonical_json_bytes(record) != raw_line:
        raise RegionCacheBuildError("region progress ledger is not canonical")
    return record


def load_region_cache_progress(ledger_path: Path, token_path: Path) -> RegionCacheProgress:
    """Validate committed rows and trim only uncommitted crash tails."""

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if not ledger_path.exists():
        ledger_path.touch()
    if not token_path.exists():
        token_path.touch()
    ledger_raw = ledger_path.read_bytes()
    if ledger_raw and not ledger_raw.endswith(b"\n"):
        last_newline = ledger_raw.rfind(b"\n")
        ledger_raw = b"" if last_newline < 0 else ledger_raw[: last_newline + 1]
        with ledger_path.open("r+b") as handle:
            handle.truncate(len(ledger_raw))
            handle.flush()
            os.fsync(handle.fileno())
    records: list[Mapping[str, Any]] = []
    previous = GENESIS_SHA256
    committed_end = 0
    with token_path.open("rb") as tokens:
        for ordinal, raw_line in enumerate(ledger_raw.splitlines(keepends=True)):
            record = _parse_ledger_line(raw_line)
            if record["schema"] != LEDGER_SCHEMA or record["ordinal"] != ordinal:
                raise RegionCacheBuildError("region progress ordinal or schema drifted")
            if record["previous_record_sha256"] != previous:
                raise RegionCacheBuildError("region progress hash chain drifted")
            supplied_record_sha = _sha256_pattern(record["record_sha256"], name="record_sha256")
            without_hash = {key: value for key, value in record.items() if key != "record_sha256"}
            if _record_sha256(without_hash) != supplied_record_sha:
                raise RegionCacheBuildError("region progress record commitment drifted")
            if record["token_start"] != committed_end or record["token_end"] < committed_end:
                raise RegionCacheBuildError("region progress token offsets drifted")
            token_length = int(record["token_end"]) - int(record["token_start"])
            tokens.seek(committed_end)
            token_values = tokens.read(token_length)
            if len(token_values) != token_length:
                raise RegionCacheBuildError("region token payload is shorter than its committed ledger")
            if _row_numeric_sha256(record, token_values) != record["row_numeric_sha256"]:
                raise RegionCacheBuildError("region progress numeric commitment drifted")
            records.append(record)
            committed_end = int(record["token_end"])
            previous = supplied_record_sha
    token_size = token_path.stat().st_size
    if token_size < committed_end:
        raise RegionCacheBuildError("region token payload lost committed bytes")
    if token_size > committed_end:
        with token_path.open("r+b") as handle:
            handle.truncate(committed_end)
            handle.flush()
            os.fsync(handle.fileno())
    return RegionCacheProgress(tuple(records), committed_end, previous)


def _append_committed_block(
    ledger_path: Path,
    token_path: Path,
    records_and_tokens: Sequence[tuple[dict[str, Any], bytes]],
    *,
    previous_record_sha256: str,
) -> str:
    previous = previous_record_sha256
    token_payload = b"".join(tokens for _record, tokens in records_and_tokens)
    with token_path.open("ab") as token_handle:
        token_handle.write(token_payload)
        token_handle.flush()
        os.fsync(token_handle.fileno())
    ledger_payload = bytearray()
    for core, _tokens in records_and_tokens:
        record = dict(core)
        record["previous_record_sha256"] = previous
        record["record_sha256"] = _record_sha256(record)
        previous = record["record_sha256"]
        ledger_payload.extend(canonical_json_bytes(record))
    with ledger_path.open("ab") as ledger_handle:
        ledger_handle.write(ledger_payload)
        ledger_handle.flush()
        os.fsync(ledger_handle.fileno())
    return previous


def _cache_from_progress(progress: RegionCacheProgress, token_path: Path) -> RaggedRegionCache:
    records = progress.records
    row_offsets = np.arange(0, (len(records) + 1) * 16, 16, dtype="<i8")
    file_sizes = np.asarray([record["file_size"] for record in records], dtype="<i8")
    lengths = np.asarray(
        [length for record in records for length in record["region_lengths"]], dtype="<i8"
    )
    region_token_offsets = np.empty(lengths.size + 1, dtype="<i8")
    region_token_offsets[0] = 0
    np.cumsum(lengths, out=region_token_offsets[1:])
    token_values = np.memmap(token_path, dtype="u1", mode="r", shape=(progress.committed_token_bytes,))
    cache = RaggedRegionCache(
        row_region_offsets=row_offsets,
        file_sizes=file_sizes,
        region_token_offsets=region_token_offsets,
        token_values=token_values,
        region_types=np.asarray(
            [value for record in records for value in record["region_types"]], dtype="u1"
        ),
        region_starts=np.asarray(
            [value for record in records for value in record["region_starts"]], dtype="<i8"
        ),
        offset_buckets=np.asarray(
            [value for record in records for value in record["offset_buckets"]], dtype="u1"
        ),
        length_buckets=np.asarray(
            [value for record in records for value in record["length_buckets"]], dtype="u1"
        ),
    )
    return validate_ragged_region_cache(cache, expected_rows=len(records))


def build_region_cache(
    sources: Sequence[RegionSource],
    *,
    output_cache: Path,
    staging_directory: Path,
    block_rows: int = 64,
    expected_rows: int = FULL_TRAIN_ROWS,
    resource_guard: ResourceGuard | None = None,
) -> RegionCacheBuildResult:
    """Resume extraction, seal the numeric cache, and return aggregate-only evidence."""

    if len(sources) != expected_rows or any(source.ordinal != index for index, source in enumerate(sources)):
        raise RegionCacheBuildError("region source plan is not the exact contiguous denominator")
    if isinstance(block_rows, bool) or block_rows < 1:
        raise ValueError("block_rows must be positive")
    if output_cache.exists():
        raise RegionCacheBuildError("region cache output already exists")
    staging_directory.mkdir(parents=True, exist_ok=True)
    _bind_staging_scope(
        staging_directory / "scope.json",
        sources=sources,
        expected_rows=expected_rows,
    )
    ledger_path = staging_directory / "region_progress.jsonl"
    token_path = staging_directory / "region_tokens.bin"
    progress = load_region_cache_progress(ledger_path, token_path)
    if len(progress.records) > expected_rows:
        raise RegionCacheBuildError("region progress exceeds the source denominator")
    config = RegionExtractionConfig()
    guard = resource_guard or ResourceGuard()
    if guard.started_at == 0.0:
        guard.start()
    maximum_rss = 0
    maximum_new_disk = 0
    previous = progress.final_record_sha256
    for block_start in range(len(progress.records), expected_rows, block_rows):
        block_end = min(block_start + block_rows, expected_rows)
        token_start = token_path.stat().st_size
        pending: list[tuple[dict[str, Any], bytes]] = []
        for source in sources[block_start:block_end]:
            result, verified_bytes = extract_verified_source(source, config)
            if verified_bytes != result.file_size:
                raise RegionCacheBuildError("verified source byte count and extraction file size differ")
            core, tokens = _record_for_result(source, result, token_start=token_start)
            pending.append((core, tokens))
            token_start += len(tokens)
        previous = _append_committed_block(
            ledger_path,
            token_path,
            pending,
            previous_record_sha256=previous,
        )
        snapshot = guard.snapshot(
            new_disk_bytes=(
                token_path.stat().st_size
                + ledger_path.stat().st_size
                + (staging_directory / "scope.json").stat().st_size
            )
        )
        maximum_rss = max(maximum_rss, int(snapshot["rss_bytes"]))
        maximum_new_disk = max(maximum_new_disk, int(snapshot["new_disk_bytes"]))
    progress = load_region_cache_progress(ledger_path, token_path)
    if len(progress.records) != expected_rows:
        raise RegionCacheBuildError("region progress did not reach the complete denominator")
    cache = _cache_from_progress(progress, token_path)
    receipt = save_ragged_region_cache(output_cache, cache)
    if receipt.size_bytes > MAXIMUM_REGION_CACHE_BYTES:
        raise RegionCacheBuildError("sealed region cache exceeds 30 GiB")

    status_counts: dict[str, int] = {}
    class_attempted = {0: 0, 1: 0}
    class_supported = {0: 0, 1: 0}
    silent_drops = 0
    for source, record in zip(sources, progress.records, strict=True):
        status = str(record["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        class_attempted[source.label] += 1
        class_supported[source.label] += int(record["supported"])
        silent_drops += int(len(record["region_types"]) != 16)
    class_coverage = {
        str(label): class_supported[label] / max(class_attempted[label], 1) for label in (0, 1)
    }
    final_disk_bytes = (
        receipt.size_bytes
        + token_path.stat().st_size
        + ledger_path.stat().st_size
        + (staging_directory / "scope.json").stat().st_size
    )
    final_snapshot = guard.snapshot(new_disk_bytes=final_disk_bytes)
    maximum_rss = max(maximum_rss, int(final_snapshot["rss_bytes"]))
    maximum_new_disk = max(maximum_new_disk, int(final_snapshot["new_disk_bytes"]))
    coverage = sum(class_supported.values()) / expected_rows
    class_gap = max(class_coverage.values()) - min(class_coverage.values())
    blockers: list[str] = []
    if coverage < 0.995:
        blockers.append("coverage_below_0.995")
    if class_gap > 0.02:
        blockers.append("class_coverage_gap_above_0.02")
    if silent_drops != 0:
        blockers.append("silent_drop_nonzero")
    return RegionCacheBuildResult(
        cache=receipt,
        ledger_path=ledger_path,
        ledger_sha256=sha256_file(ledger_path),
        final_record_sha256=progress.final_record_sha256,
        attempted=expected_rows,
        supported=sum(class_supported.values()),
        class_coverage=class_coverage,
        class_coverage_gap=class_gap,
        silent_drops=silent_drops,
        status_counts=dict(sorted(status_counts.items())),
        source_bytes_verified=sum(int(record["file_size"]) for record in progress.records),
        maximum_rss_bytes=maximum_rss,
        maximum_new_disk_bytes=maximum_new_disk,
        blockers=tuple(blockers),
        decision=(
            "full_train_region_cache_gate_pass_seed41_pilot_may_begin"
            if not blockers
            else "close_loop175_current_region_cache_recipe"
        ),
    )


__all__ = [
    "RegionCacheBuildError",
    "RegionCacheBuildResult",
    "RegionCacheProgress",
    "RegionSource",
    "build_region_cache",
    "extract_verified_source",
    "load_region_cache_progress",
    "load_region_sources",
    "source_plan_commitment",
]
