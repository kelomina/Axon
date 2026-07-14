"""Fail-closed local bundle parsing and whole-file streaming for Loop164."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import torch

from .whole_file_gcg import OutputChunk, output_partitions, padded_length_for_valid_length

LOOP_ID = "loop164_whole_file_residual_expert"
LOCAL_BUNDLE_RECORD_SCHEMA = "axon_loop164_local_probe_record_v1"
LOCAL_BUNDLE_SUMMARY_SCHEMA = "axon_loop164_local_probe_bundle_summary_v1"
LOCAL_BUNDLE_ROLE = "local_train_only_runtime_probe"
LOCAL_SPLIT_ROLE = "train"
MAX_BUNDLE_BYTES = 4 * 1024 * 1024
EXPECTED_RECORD_KEYS = {
    "schema",
    "loop_id",
    "bundle_role",
    "split_role",
    "label",
    "source_path",
    "source_sha256",
    "source_size_bytes",
    "metadata_not_model_features",
    "source_path_usage",
    "source_sha256_usage",
}


class InputContractError(ValueError):
    """The local probe bundle or source contract is invalid."""


class SourceIntegrityError(RuntimeError):
    """A source changed or failed its authority-bound size/SHA contract."""


class SourceTimeoutError(TimeoutError):
    """A whole-file scan exceeded its absolute deadline."""


@dataclass(frozen=True)
class LocalProbeRecord:
    source_path: Path
    source_sha256: str
    source_size_bytes: int
    label: int


@dataclass(frozen=True)
class FileFingerprint:
    size_bytes: int
    modified_ns: int
    device: int
    inode: int


@dataclass(frozen=True)
class FileScanReceipt:
    pass_index: int
    bytes_read: int
    sha256: str
    output_count: int
    chunk_count: int


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InputContractError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise InputContractError(f"Non-finite JSON value: {value}")


def _parse_json_object(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputContractError(f"Invalid JSON for {context}") from exc
    if not isinstance(payload, dict):
        raise InputContractError(f"Expected JSON object for {context}")
    return payload


def _read_bounded(path: Path, *, max_bytes: int) -> bytes:
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise InputContractError(f"Bounded input is too large: {path}")
    return raw


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _lexical_relative_to(path: Path, root: Path) -> Path:
    absolute_path = path.absolute()
    absolute_root = root.absolute()
    try:
        return absolute_path.relative_to(absolute_root)
    except ValueError:
        path_parts = absolute_path.parts
        root_parts = absolute_root.parts
        if len(path_parts) < len(root_parts) or tuple(
            part.casefold() for part in path_parts[: len(root_parts)]
        ) != tuple(part.casefold() for part in root_parts):
            raise
        return Path(*path_parts[len(root_parts) :])


def _validate_record(payload: dict[str, Any], *, data_root: Path) -> LocalProbeRecord:
    if set(payload) != EXPECTED_RECORD_KEYS:
        raise InputContractError("Local probe record fields do not match the frozen schema")
    if (
        payload.get("schema") != LOCAL_BUNDLE_RECORD_SCHEMA
        or payload.get("loop_id") != LOOP_ID
        or payload.get("bundle_role") != LOCAL_BUNDLE_ROLE
        or payload.get("split_role") != LOCAL_SPLIT_ROLE
    ):
        raise InputContractError("Local probe record role or identity is invalid")
    if payload.get("label") not in {0, 1}:
        raise InputContractError("Local probe labels must be binary integers")
    source_sha256 = str(payload.get("source_sha256") or "").strip().casefold()
    if not _is_sha256(source_sha256):
        raise InputContractError("Local probe record has invalid source_sha256")
    source_size_bytes = payload.get("source_size_bytes")
    if not isinstance(source_size_bytes, int) or isinstance(source_size_bytes, bool):
        raise InputContractError("Local probe record has invalid source_size_bytes")
    source_path_text = payload.get("source_path")
    if not isinstance(source_path_text, str) or not source_path_text.strip():
        raise InputContractError("Local probe record has invalid source_path")
    source_path = Path(source_path_text)
    try:
        relative_path = _lexical_relative_to(source_path, data_root)
    except ValueError as exc:
        raise InputContractError("Local probe source_path escapes the materialized data root") from exc
    if not relative_path.parts or any(part in {"", ".", ".."} for part in relative_path.parts):
        raise InputContractError("Local probe source_path has invalid relative components")
    if payload.get("metadata_not_model_features") != [
        "source_path",
        "source_sha256",
        "source_size_bytes",
    ]:
        raise InputContractError("Local probe identity metadata declaration drifted")
    if (
        payload.get("source_path_usage") != "loader_identity_only_not_model_feature"
        or payload.get("source_sha256_usage") != "integrity_binding_only_not_model_feature"
    ):
        raise InputContractError("Local probe identity metadata usage drifted")
    return LocalProbeRecord(
        source_path=source_path,
        source_sha256=source_sha256,
        source_size_bytes=source_size_bytes,
        label=int(payload["label"]),
    )


def load_local_probe_bundle(
    *,
    bundle_path: Path,
    summary_path: Path,
    data_root: Path,
    expected_records_per_class: int,
) -> tuple[list[LocalProbeRecord], dict[str, Any]]:
    """Validate the aggregate summary and all train-only records before raw access."""

    if expected_records_per_class < 1:
        raise InputContractError("expected_records_per_class must be positive")
    bundle_path = bundle_path.resolve(strict=True)
    summary_path = summary_path.resolve(strict=True)
    data_root = data_root.resolve(strict=True)
    summary = _parse_json_object(
        _read_bounded(summary_path, max_bytes=MAX_BUNDLE_BYTES),
        context="local probe bundle summary",
    )
    if (
        summary.get("schema") != LOCAL_BUNDLE_SUMMARY_SCHEMA
        or summary.get("loop_id") != LOOP_ID
        or summary.get("bundle_role") != LOCAL_BUNDLE_ROLE
        or summary.get("decision") != "local_train_only_probe_bundle_ready"
    ):
        raise InputContractError("Local probe bundle summary scope is invalid")
    ready_for = summary.get("ready_for")
    if not isinstance(ready_for, dict) or ready_for != {
        "local_runtime_probe_bundle": True,
        "loop164_whole_file_training": False,
        "val_or_test_access": False,
        "f1_claim": False,
    }:
        raise InputContractError("Local probe bundle summary readiness claims drifted")
    bundle_binding = summary.get("bundle")
    if not isinstance(bundle_binding, dict):
        raise InputContractError("Local probe bundle binding is missing")
    try:
        bound_bundle_path = Path(str(bundle_binding["path"])).resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise InputContractError("Local probe bundle path binding is invalid") from exc
    if bound_bundle_path != bundle_path:
        raise InputContractError("Local probe bundle path does not match its summary")

    bundle_raw = _read_bounded(bundle_path, max_bytes=MAX_BUNDLE_BYTES)
    bundle_sha256 = hashlib.sha256(bundle_raw).hexdigest()
    expected_count = expected_records_per_class * 2
    if (
        bundle_binding.get("sha256") != bundle_sha256
        or bundle_binding.get("record_count") != expected_count
        or bundle_binding.get("record_schema") != LOCAL_BUNDLE_RECORD_SCHEMA
    ):
        raise InputContractError("Local probe bundle hash/count/schema binding is invalid")
    selection = summary.get("selection")
    if not isinstance(selection, dict) or (
        selection.get("canonical_split_role") != LOCAL_SPLIT_ROLE
        or selection.get("records_per_class") != expected_records_per_class
        or selection.get("labels") != [0, 1]
    ):
        raise InputContractError("Local probe selection contract is invalid")

    lines = bundle_raw.splitlines()
    if len(lines) != expected_count or any(not line.strip() for line in lines):
        raise InputContractError("Local probe bundle line count is invalid")
    records = [
        _validate_record(_parse_json_object(line, context="local probe record"), data_root=data_root)
        for line in lines
    ]
    source_sha256 = [record.source_sha256 for record in records]
    source_paths = [str(record.source_path).casefold() for record in records]
    if len(set(source_sha256)) != len(source_sha256) or len(set(source_paths)) != len(source_paths):
        raise InputContractError("Local probe bundle repeats a source identity")
    label_counts = {label: sum(record.label == label for record in records) for label in (0, 1)}
    if label_counts != {0: expected_records_per_class, 1: expected_records_per_class}:
        raise InputContractError("Local probe bundle class balance is invalid")
    return records, summary


def _fingerprint(path: Path) -> FileFingerprint:
    stat_result = os.stat(path, follow_symlinks=False)
    return FileFingerprint(
        size_bytes=int(stat_result.st_size),
        modified_ns=int(stat_result.st_mtime_ns),
        device=int(stat_result.st_dev),
        inode=int(stat_result.st_ino),
    )


def _resolve_regular_source(path: Path, *, data_root: Path) -> Path:
    absolute_root = data_root.absolute()
    absolute_path = path.absolute()
    try:
        relative_path = _lexical_relative_to(absolute_path, absolute_root)
    except ValueError as exc:
        raise SourceIntegrityError("Source path escapes the materialized data root") from exc
    cursor = absolute_root
    if cursor.is_symlink():
        raise SourceIntegrityError("Materialized data root cannot be a symbolic link")
    for component in relative_path.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise SourceIntegrityError("Source path cannot contain symbolic links")
    resolved_root = absolute_root.resolve(strict=True)
    resolved_path = absolute_path.resolve(strict=True)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise SourceIntegrityError("Resolved source path escapes the materialized data root") from exc
    if not resolved_path.is_file():
        raise SourceIntegrityError("Source path is not a regular file")
    return resolved_path


class StreamingWholeFileByteSource:
    """Yield exact output-coordinate chunks while hashing every raw byte once per pass."""

    def __init__(
        self,
        record: LocalProbeRecord,
        *,
        data_root: Path,
        receptive_field_bytes: int,
        output_stride_bytes: int,
        max_outputs_per_chunk: int,
        bounded_read_bytes: int,
        max_supported_file_bytes: int,
        timeout_seconds: float,
        absolute_deadline: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if receptive_field_bytes < 1 or output_stride_bytes < 1:
            raise InputContractError("Convolution geometry must be positive")
        if output_stride_bytes > receptive_field_bytes:
            raise InputContractError("Output stride cannot exceed the receptive field")
        if max_outputs_per_chunk < 1 or bounded_read_bytes < 1:
            raise InputContractError("Chunk bounds must be positive")
        if timeout_seconds <= 0:
            raise InputContractError("timeout_seconds must be positive")
        if record.source_size_bytes < 1:
            raise SourceIntegrityError("Empty files are unsupported by the whole-file model")
        if record.source_size_bytes > max_supported_file_bytes:
            raise SourceIntegrityError("Source exceeds max_supported_file_bytes")
        self.record = record
        self.data_root = data_root
        self.receptive_field_bytes = int(receptive_field_bytes)
        self.output_stride_bytes = int(output_stride_bytes)
        self.max_outputs_per_chunk = int(max_outputs_per_chunk)
        self.bounded_read_bytes = int(bounded_read_bytes)
        self._clock = clock
        local_deadline = clock() + float(timeout_seconds)
        self._deadline = min(local_deadline, absolute_deadline) if absolute_deadline else local_deadline
        self._path = _resolve_regular_source(record.source_path, data_root=data_root)
        self._initial_fingerprint = _fingerprint(self._path)
        if self._initial_fingerprint.size_bytes != record.source_size_bytes:
            raise SourceIntegrityError("Source size does not match the probe bundle")
        self._scan_receipts: list[FileScanReceipt] = []
        self._pass_active = False
        self._failed = False

    @property
    def valid_length(self) -> int:
        return self.record.source_size_bytes

    @property
    def scan_receipts(self) -> tuple[FileScanReceipt, ...]:
        return tuple(self._scan_receipts)

    def expected_output_count(
        self,
        *,
        receptive_field_bytes: int,
        output_stride_bytes: int,
    ) -> int:
        self._validate_geometry(receptive_field_bytes, output_stride_bytes)
        padded_length = padded_length_for_valid_length(
            self.valid_length,
            self.receptive_field_bytes,
            self.output_stride_bytes,
        )
        return (padded_length - self.receptive_field_bytes) // self.output_stride_bytes + 1

    def _validate_geometry(self, receptive_field_bytes: int, output_stride_bytes: int) -> None:
        if (
            receptive_field_bytes != self.receptive_field_bytes
            or output_stride_bytes != self.output_stride_bytes
        ):
            raise SourceIntegrityError("Model/source convolution geometry drifted")

    def _check_deadline(self) -> None:
        if self._clock() > self._deadline:
            raise SourceTimeoutError("Whole-file source exceeded its deadline")

    def iter_output_chunks(
        self,
        *,
        receptive_field_bytes: int,
        output_stride_bytes: int,
        max_outputs_per_chunk: int,
    ) -> Iterator[OutputChunk]:
        self._validate_geometry(receptive_field_bytes, output_stride_bytes)
        if max_outputs_per_chunk != self.max_outputs_per_chunk:
            raise SourceIntegrityError("Model/source chunk bound drifted")
        if self._pass_active or self._failed or len(self._scan_receipts) >= 2:
            raise SourceIntegrityError("Whole-file source pass lifecycle is invalid")
        self._pass_active = True
        pass_index = len(self._scan_receipts) + 1
        try:
            self._check_deadline()
            if _fingerprint(self._path) != self._initial_fingerprint:
                raise SourceIntegrityError("Source fingerprint changed before a scan")
            output_count = self.expected_output_count(
                receptive_field_bytes=self.receptive_field_bytes,
                output_stride_bytes=self.output_stride_bytes,
            )
            digest = hashlib.sha256()
            bytes_read = 0
            chunk_count = 0
            buffer = bytearray()
            buffer_start = 0

            # 每遍只顺序读取新字节；相邻输出块的重叠区仅留在有界内存中，不重复计入 SHA。
            with self._path.open("rb", buffering=0) as handle:
                for partition in output_partitions(output_count, self.max_outputs_per_chunk):
                    self._check_deadline()
                    byte_start = partition.start * self.output_stride_bytes
                    byte_end = (
                        (partition.end - 1) * self.output_stride_bytes
                        + self.receptive_field_bytes
                    )
                    drop_bytes = byte_start - buffer_start
                    if drop_bytes < 0 or drop_bytes > len(buffer):
                        raise SourceIntegrityError("Streaming overlap geometry is invalid")
                    if drop_bytes:
                        del buffer[:drop_bytes]
                    buffer_start = byte_start
                    raw_required_end = min(byte_end, self.valid_length)
                    while buffer_start + len(buffer) < raw_required_end:
                        read_size = min(
                            self.bounded_read_bytes,
                            raw_required_end - (buffer_start + len(buffer)),
                        )
                        raw_chunk = handle.read(read_size)
                        if not raw_chunk:
                            raise SourceIntegrityError("Source ended before its declared length")
                        digest.update(raw_chunk)
                        bytes_read += len(raw_chunk)
                        buffer.extend(raw_chunk)
                        self._check_deadline()

                    required_length = (
                        (partition.output_count - 1) * self.output_stride_bytes
                        + self.receptive_field_bytes
                    )
                    raw_length = max(0, raw_required_end - byte_start)
                    tokens = torch.zeros(required_length, dtype=torch.long)
                    if raw_length:
                        raw_tensor = torch.frombuffer(
                            bytearray(buffer[:raw_length]), dtype=torch.uint8
                        ).to(dtype=torch.long)
                        tokens[:raw_length] = raw_tensor + 1
                    chunk_count += 1
                    yield OutputChunk(
                        output_start=partition.start,
                        output_count=partition.output_count,
                        tokens=tokens,
                    )
                if handle.read(1):
                    raise SourceIntegrityError("Source grew beyond its declared length")

            self._check_deadline()
            actual_sha256 = digest.hexdigest()
            if bytes_read != self.valid_length or actual_sha256 != self.record.source_sha256:
                raise SourceIntegrityError("Source bytes do not match the probe bundle")
            if _fingerprint(self._path) != self._initial_fingerprint:
                raise SourceIntegrityError("Source fingerprint changed during a scan")
            self._scan_receipts.append(
                FileScanReceipt(
                    pass_index=pass_index,
                    bytes_read=bytes_read,
                    sha256=actual_sha256,
                    output_count=output_count,
                    chunk_count=chunk_count,
                )
            )
        except BaseException:
            self._failed = True
            raise
        finally:
            self._pass_active = False

    def assert_complete(self, *, expected_passes: int = 2) -> None:
        if self._failed or self._pass_active or len(self._scan_receipts) != expected_passes:
            raise SourceIntegrityError("Whole-file source did not complete every verified pass")
        if any(
            receipt.bytes_read != self.valid_length
            or receipt.sha256 != self.record.source_sha256
            for receipt in self._scan_receipts
        ):
            raise SourceIntegrityError("Whole-file source pass receipts are inconsistent")
