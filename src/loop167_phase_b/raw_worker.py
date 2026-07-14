"""Fail-closed one-pass raw feature worker for the future Loop167 controller."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Sequence

import numpy as np

from .b0_projector import extract_b0_projection
from .ember_controls import extract_context_features
from .one_pass_reader import SHA256_PATTERN, StreamReadResult, read_verified_bytes
from .progress_ledger import RawScanLedger
from .raw_context import RawFeatureContext

B0_VALUE_DIMENSION = 571
B0_MISSING_DIMENSION = 6
B1_VALUE_DIMENSION = 536
B1_MISSING_DIMENSION = 4
B1_SAMPLING_DIMENSION = 3
NOVEL_VALUE_DIMENSION = 292
NOVEL_MISSING_DIMENSION = 1

RAW_SCOPE_COMMITMENT_DOMAIN = b"axon_loop167_phase_b_raw_scope_v1\0"
FEATURE_ROW_COMMITMENT_DOMAIN = b"axon_loop167_phase_b_raw_feature_row_v1\0"
FATAL_SOURCE_DRIFT_RESULTS = frozenset(
    {"read_truncated", "declared_size_mismatch", "sha256_mismatch"}
)


class RawWorkerError(RuntimeError):
    """Raised when the isolated raw scan contract cannot be honored."""


class RawScopeDriftError(RawWorkerError):
    """Raised before raw access when the sealed safe scope no longer matches."""


class RawScanFatalError(RawWorkerError):
    """Raised only after a source-drift terminal record has been persisted."""

    def __init__(
        self,
        *,
        ordinal: int,
        result: str,
        terminal_record_sha256: str,
        feature_row_commitment_sha256: str,
    ) -> None:
        super().__init__("Raw source SHA or declared-scope drifted after its terminal ledger record")
        self.ordinal = ordinal
        self.result = result
        self.terminal_record_sha256 = terminal_record_sha256
        self.feature_row_commitment_sha256 = feature_row_commitment_sha256


class RawBudgetExhaustedError(RawWorkerError):
    """Raised only after an unopened ordinal is durably rejected by the byte budget."""

    def __init__(
        self,
        *,
        ordinal: int,
        terminal_record_sha256: str,
        feature_row_commitment_sha256: str,
    ) -> None:
        super().__init__("Raw byte budget cannot reserve the next bounded one-pass read")
        self.ordinal = ordinal
        self.terminal_record_sha256 = terminal_record_sha256
        self.feature_row_commitment_sha256 = feature_row_commitment_sha256


def _require_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_integer(value: object, *, name: str) -> int:
    value = _require_nonnegative_integer(value, name=name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        raise ValueError("Raw worker commitment payload is not canonical JSON") from error


@dataclass(frozen=True, slots=True)
class RawPlanEntry:
    """Private locator plus the safe target-free identity needed for one ordinal."""

    ordinal: int
    source_file: Path
    source_audit_sha256: str
    declared_size: int
    expected_sha256: str

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.ordinal, name="ordinal")
        if not isinstance(self.source_file, Path) or not self.source_file.is_absolute():
            raise ValueError("source_file must be an absolute pathlib.Path")
        _require_sha256(self.source_audit_sha256, name="source_audit_sha256")
        _require_nonnegative_integer(self.declared_size, name="declared_size")
        _require_sha256(self.expected_sha256, name="expected_sha256")

    def audit_record(self) -> dict[str, int | str]:
        """Return the only entry representation permitted in commitments and ledgers."""

        return {
            "declared_size": self.declared_size,
            "expected_sha256": self.expected_sha256,
            "ordinal": self.ordinal,
            "source_audit_sha256": self.source_audit_sha256,
        }


def raw_scope_commitment(entries: Sequence[RawPlanEntry]) -> str:
    """Commit the exact order and safe source identity without serializing locators."""

    normalized = tuple(entries)
    if not normalized:
        raise ValueError("Raw scan scope must contain at least one entry")
    audit_records: list[dict[str, int | str]] = []
    seen_audits: set[str] = set()
    for expected_ordinal, entry in enumerate(normalized):
        if not isinstance(entry, RawPlanEntry):
            raise TypeError("Raw scan scope contains a non-RawPlanEntry value")
        if entry.ordinal != expected_ordinal:
            raise ValueError("Raw scan ordinals must be contiguous and source ordered")
        if entry.source_audit_sha256 in seen_audits:
            raise ValueError("Raw scan scope repeats a source audit commitment")
        seen_audits.add(entry.source_audit_sha256)
        audit_records.append(entry.audit_record())
    return hashlib.sha256(RAW_SCOPE_COMMITMENT_DOMAIN + _canonical_json_bytes(audit_records)).hexdigest()


@dataclass(frozen=True, slots=True)
class RawScanPlan:
    """Immutable raw input scope whose commitment deliberately excludes local locators."""

    entries: tuple[RawPlanEntry, ...]
    raw_scope_commitment_sha256: str

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        object.__setattr__(self, "entries", entries)
        expected = raw_scope_commitment(entries)
        if _require_sha256(self.raw_scope_commitment_sha256, name="raw_scope_commitment_sha256") != expected:
            raise RawScopeDriftError("Raw scan scope commitment does not match its safe entries")

    @classmethod
    def from_entries(cls, entries: Sequence[RawPlanEntry]) -> "RawScanPlan":
        normalized = tuple(entries)
        return cls(normalized, raw_scope_commitment(normalized))


@dataclass(frozen=True, slots=True)
class RawWorkerConfig:
    """Frozen one-pass limits supplied by the eventual execution authorization."""

    maximum_source_file_bytes: int
    maximum_raw_open_attempts: int
    maximum_raw_bytes_read: int
    reader_chunk_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        _require_positive_integer(self.maximum_source_file_bytes, name="maximum_source_file_bytes")
        _require_nonnegative_integer(self.maximum_raw_open_attempts, name="maximum_raw_open_attempts")
        _require_nonnegative_integer(self.maximum_raw_bytes_read, name="maximum_raw_bytes_read")
        _require_positive_integer(self.reader_chunk_bytes, name="reader_chunk_bytes")


@dataclass(frozen=True, slots=True)
class RawFeatureRow:
    """In-memory numeric feature blocks with no locator, target, partition, or model output."""

    ordinal: int
    source_audit_sha256: str
    result: str
    b0_values: np.ndarray
    b0_missing_indicators: np.ndarray
    b1_values: np.ndarray
    b1_missing_indicators: np.ndarray
    b1_sampling_indicators: np.ndarray
    novel_values: np.ndarray
    novel_missing_indicators: np.ndarray
    novel_complete: bool
    feature_row_commitment_sha256: str


@dataclass(frozen=True, slots=True)
class RawScanOutcome:
    """Complete, in-memory row set whose length is the fixed denominator."""

    rows: tuple[RawFeatureRow, ...]
    raw_scope_commitment_sha256: str
    feature_rows_commitment_sha256: str
    raw_ledger_final_record_sha256: str


def _frozen_float_vector(values: object, *, name: str, dimension: int) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite float32 vector") from error
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} shape or finiteness drifted")
    result = np.ascontiguousarray(vector, dtype=np.float32).copy()
    result.setflags(write=False)
    return result


def _frozen_binary_vector(values: object, *, name: str, dimension: int) -> np.ndarray:
    result = _frozen_float_vector(values, name=name, dimension=dimension)
    if not np.isin(result, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain binary indicators")
    return result


def _vector_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.dtype("<f4"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _feature_row_commitment(
    *,
    ordinal: int,
    source_audit_sha256: str,
    result: str,
    b0_values: np.ndarray,
    b0_missing_indicators: np.ndarray,
    b1_values: np.ndarray,
    b1_missing_indicators: np.ndarray,
    b1_sampling_indicators: np.ndarray,
    novel_values: np.ndarray,
    novel_missing_indicators: np.ndarray,
    novel_complete: bool,
) -> str:
    payload = {
        "b0_missing_indicators_sha256": _vector_sha256(b0_missing_indicators),
        "b0_values_sha256": _vector_sha256(b0_values),
        "b1_missing_indicators_sha256": _vector_sha256(b1_missing_indicators),
        "b1_sampling_indicators_sha256": _vector_sha256(b1_sampling_indicators),
        "b1_values_sha256": _vector_sha256(b1_values),
        "novel_complete": novel_complete,
        "novel_missing_indicators_sha256": _vector_sha256(novel_missing_indicators),
        "novel_values_sha256": _vector_sha256(novel_values),
        "ordinal": ordinal,
        "result": result,
        "source_audit_sha256": source_audit_sha256,
    }
    return hashlib.sha256(FEATURE_ROW_COMMITMENT_DOMAIN + _canonical_json_bytes(payload)).hexdigest()


def _feature_row(
    entry: RawPlanEntry,
    *,
    result: str,
    b0_values: object,
    b0_missing_indicators: object,
    b1_values: object,
    b1_missing_indicators: object,
    b1_sampling_indicators: object,
    novel_values: object,
    novel_missing_indicators: object,
    novel_complete: object,
) -> RawFeatureRow:
    if not isinstance(result, str):
        raise ValueError("Raw feature result must be a string")
    if not isinstance(novel_complete, bool):
        raise ValueError("novel_complete must be boolean")
    frozen_b0_values = _frozen_float_vector(b0_values, name="b0_values", dimension=B0_VALUE_DIMENSION)
    frozen_b0_missing = _frozen_binary_vector(
        b0_missing_indicators,
        name="b0_missing_indicators",
        dimension=B0_MISSING_DIMENSION,
    )
    frozen_b1_values = _frozen_float_vector(b1_values, name="b1_values", dimension=B1_VALUE_DIMENSION)
    frozen_b1_missing = _frozen_binary_vector(
        b1_missing_indicators,
        name="b1_missing_indicators",
        dimension=B1_MISSING_DIMENSION,
    )
    frozen_b1_sampling = _frozen_binary_vector(
        b1_sampling_indicators,
        name="b1_sampling_indicators",
        dimension=B1_SAMPLING_DIMENSION,
    )
    frozen_novel_values = _frozen_float_vector(
        novel_values,
        name="novel_values",
        dimension=NOVEL_VALUE_DIMENSION,
    )
    frozen_novel_missing = _frozen_binary_vector(
        novel_missing_indicators,
        name="novel_missing_indicators",
        dimension=NOVEL_MISSING_DIMENSION,
    )
    commitment = _feature_row_commitment(
        ordinal=entry.ordinal,
        source_audit_sha256=entry.source_audit_sha256,
        result=result,
        b0_values=frozen_b0_values,
        b0_missing_indicators=frozen_b0_missing,
        b1_values=frozen_b1_values,
        b1_missing_indicators=frozen_b1_missing,
        b1_sampling_indicators=frozen_b1_sampling,
        novel_values=frozen_novel_values,
        novel_missing_indicators=frozen_novel_missing,
        novel_complete=novel_complete,
    )
    return RawFeatureRow(
        ordinal=entry.ordinal,
        source_audit_sha256=entry.source_audit_sha256,
        result=result,
        b0_values=frozen_b0_values,
        b0_missing_indicators=frozen_b0_missing,
        b1_values=frozen_b1_values,
        b1_missing_indicators=frozen_b1_missing,
        b1_sampling_indicators=frozen_b1_sampling,
        novel_values=frozen_novel_values,
        novel_missing_indicators=frozen_novel_missing,
        novel_complete=novel_complete,
        feature_row_commitment_sha256=commitment,
    )


def _zero_feature_row(entry: RawPlanEntry, *, result: str) -> RawFeatureRow:
    """Keep every unavailable ordinal in the denominator with explicit missing flags."""

    return _feature_row(
        entry,
        result=result,
        b0_values=np.zeros(B0_VALUE_DIMENSION, dtype=np.float32),
        b0_missing_indicators=np.ones(B0_MISSING_DIMENSION, dtype=np.float32),
        b1_values=np.zeros(B1_VALUE_DIMENSION, dtype=np.float32),
        b1_missing_indicators=np.ones(B1_MISSING_DIMENSION, dtype=np.float32),
        b1_sampling_indicators=np.zeros(B1_SAMPLING_DIMENSION, dtype=np.float32),
        novel_values=np.zeros(NOVEL_VALUE_DIMENSION, dtype=np.float32),
        novel_missing_indicators=np.ones(NOVEL_MISSING_DIMENSION, dtype=np.float32),
        novel_complete=False,
    )


def _open_source_stream(source_file: Path) -> BinaryIO:
    """Open one regular local file descriptor without following a final symlink."""

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(source_file), flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("Raw source is not a regular file")
        return os.fdopen(descriptor, "rb", buffering=0)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _open_and_read(entry: RawPlanEntry, config: RawWorkerConfig) -> StreamReadResult:
    try:
        stream = _open_source_stream(entry.source_file)
    except (OSError, ValueError):
        return StreamReadResult("source_open_failure", None, entry.declared_size, 0, None)

    result: StreamReadResult | None = None
    try:
        result = read_verified_bytes(
            stream,
            declared_size=entry.declared_size,
            expected_sha256=entry.expected_sha256,
            maximum_source_file_bytes=config.maximum_source_file_bytes,
            chunk_bytes=config.reader_chunk_bytes,
        )
    except Exception:
        result = StreamReadResult("read_failure", None, entry.declared_size, 0, None)
    finally:
        try:
            stream.close()
        except OSError:
            if result is None:
                result = StreamReadResult("read_failure", None, entry.declared_size, 0, None)
            else:
                result = StreamReadResult(
                    "read_failure",
                    None,
                    entry.declared_size,
                    result.bytes_read,
                    None,
                )
    assert result is not None
    return result


def _available_feature_row(
    entry: RawPlanEntry,
    bytez: bytes,
    *,
    config: RawWorkerConfig,
    pe_factory: Callable[..., Any] | None,
) -> RawFeatureRow:
    """Project all blocks through one context and release raw bytes before returning."""

    context: RawFeatureContext | None = None
    completed_row: RawFeatureRow | None = None
    result = "feature_failure"
    try:
        context_kwargs: dict[str, object] = {
            "maximum_input_bytes": config.maximum_source_file_bytes,
        }
        if pe_factory is not None:
            context_kwargs["pe_factory"] = pe_factory
        context = RawFeatureContext.from_bytes(bytez, **context_kwargs)
        if not context.pe_parse_succeeded:
            result = "pe_parse_failure"
        else:
            b0 = extract_b0_projection(context)
            projected = extract_context_features(context)
            completed_row = _feature_row(
                entry,
                result="available",
                b0_values=b0.values,
                b0_missing_indicators=b0.missing_indicators,
                b1_values=projected.controls.values,
                b1_missing_indicators=projected.controls.missing_indicators,
                b1_sampling_indicators=projected.controls.sampling_indicators,
                novel_values=projected.novel.values,
                novel_missing_indicators=projected.novel.missing_indicators,
                novel_complete=projected.novel.complete,
            )
    except Exception:
        result = "feature_failure"
        completed_row = None
    finally:
        close_failed = False
        if context is not None:
            try:
                context.close()
            except Exception:
                close_failed = True
            if context.bytez or context.pe is not None:
                close_failed = True
        # RawFeatureContext.close owns the byte lifetime; no row retains bytez or parsed PE objects.
        bytez = b""
    if close_failed:
        return _zero_feature_row(entry, result="feature_failure")
    if completed_row is not None:
        return completed_row
    return _zero_feature_row(entry, result=result)


class RawFeatureWorker:
    """Compose the sealed reader, one PE context, projections, and durable raw ledger."""

    def __init__(
        self,
        config: RawWorkerConfig,
        *,
        pe_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(config, RawWorkerConfig):
            raise TypeError("config must be RawWorkerConfig")
        self._config = config
        self._pe_factory = pe_factory

    def scan(
        self,
        plan: RawScanPlan,
        *,
        expected_raw_scope_commitment_sha256: str,
        ledger: RawScanLedger,
    ) -> RawScanOutcome:
        """Execute the exact plan once; SHA or declared-scope drift halts after a terminal row."""

        if not isinstance(plan, RawScanPlan):
            raise TypeError("plan must be RawScanPlan")
        if not isinstance(ledger, RawScanLedger):
            raise TypeError("ledger must be RawScanLedger")
        expected_scope = _require_sha256(
            expected_raw_scope_commitment_sha256,
            name="expected_raw_scope_commitment_sha256",
        )
        actual_scope = raw_scope_commitment(plan.entries)
        if plan.raw_scope_commitment_sha256 != actual_scope or expected_scope != actual_scope:
            raise RawScopeDriftError("Raw scan scope commitment drifted before any source open")
        if self._config.maximum_raw_open_attempts != len(plan.entries):
            raise RawWorkerError("Raw open budget must equal the immutable one-pass entry count")
        if ledger.final_record_sha256 is not None:
            raise RawWorkerError("Raw scan ledger must be fresh before the first source intent")

        cumulative_open_attempts = 0
        cumulative_bytes_read = 0
        rows: list[RawFeatureRow] = []
        ledger.scan_started(
            expected_record_count=len(plan.entries),
            maximum_raw_open_attempts=self._config.maximum_raw_open_attempts,
            maximum_raw_bytes_read=self._config.maximum_raw_bytes_read,
            raw_scope_commitment_sha256=actual_scope,
        )
        for entry in plan.entries:
            ledger.raw_open_intent(
                ordinal=entry.ordinal,
                source_audit_sha256=entry.source_audit_sha256,
            )
            if entry.declared_size > self._config.maximum_source_file_bytes:
                read_result = StreamReadResult(
                    "oversize_declared",
                    None,
                    entry.declared_size,
                    0,
                    None,
                )
            else:
                # reader 会额外读取一个字节验证大小，因此打开前必须为最坏情况预留预算。
                worst_case_read_bytes = entry.declared_size + 1
                if (
                    cumulative_bytes_read + worst_case_read_bytes
                    > self._config.maximum_raw_bytes_read
                ):
                    row = _zero_feature_row(entry, result="raw_byte_budget_exhausted")
                    terminal_record_sha256 = ledger.record_terminal(
                        ordinal=entry.ordinal,
                        source_audit_sha256=entry.source_audit_sha256,
                        result=row.result,
                        cumulative_raw_open_attempts=cumulative_open_attempts,
                        cumulative_raw_bytes_read=cumulative_bytes_read,
                        feature_row_commitment_sha256=row.feature_row_commitment_sha256,
                    )
                    rows.append(row)
                    raise RawBudgetExhaustedError(
                        ordinal=entry.ordinal,
                        terminal_record_sha256=terminal_record_sha256,
                        feature_row_commitment_sha256=row.feature_row_commitment_sha256,
                    )
                cumulative_open_attempts += 1
                read_result = _open_and_read(entry, self._config)
            cumulative_bytes_read += read_result.bytes_read

            if read_result.available:
                raw_bytes = read_result.bytez
                assert raw_bytes is not None
                # 读取结果不应在 context 关闭后继续持有原始字节的第二个引用。
                read_result = StreamReadResult(
                    read_result.result,
                    None,
                    read_result.declared_size,
                    read_result.bytes_read,
                    read_result.observed_sha256,
                )
                try:
                    row = _available_feature_row(
                        entry,
                        raw_bytes,
                        config=self._config,
                        pe_factory=self._pe_factory,
                    )
                finally:
                    raw_bytes = b""
            else:
                row = _zero_feature_row(entry, result=read_result.result)

            terminal_record_sha256 = ledger.record_terminal(
                ordinal=entry.ordinal,
                source_audit_sha256=entry.source_audit_sha256,
                result=row.result,
                cumulative_raw_open_attempts=cumulative_open_attempts,
                cumulative_raw_bytes_read=cumulative_bytes_read,
                feature_row_commitment_sha256=row.feature_row_commitment_sha256,
            )
            rows.append(row)
            if row.result in FATAL_SOURCE_DRIFT_RESULTS:
                raise RawScanFatalError(
                    ordinal=entry.ordinal,
                    result=row.result,
                    terminal_record_sha256=terminal_record_sha256,
                    feature_row_commitment_sha256=row.feature_row_commitment_sha256,
                )

        ledger.scan_completed(
            record_count=len(rows),
            cumulative_raw_open_attempts=cumulative_open_attempts,
            cumulative_raw_bytes_read=cumulative_bytes_read,
            feature_rows_commitment_sha256=ledger.feature_rows_commitment_sha256,
        )
        final_record_sha256 = ledger.final_record_sha256
        if final_record_sha256 is None:
            raise RawWorkerError("Raw scan ledger has no terminal completion record")
        return RawScanOutcome(
            rows=tuple(rows),
            raw_scope_commitment_sha256=actual_scope,
            feature_rows_commitment_sha256=ledger.feature_rows_commitment_sha256,
            raw_ledger_final_record_sha256=final_record_sha256,
        )


__all__ = [
    "B0_MISSING_DIMENSION",
    "B0_VALUE_DIMENSION",
    "B1_MISSING_DIMENSION",
    "B1_SAMPLING_DIMENSION",
    "B1_VALUE_DIMENSION",
    "FATAL_SOURCE_DRIFT_RESULTS",
    "NOVEL_MISSING_DIMENSION",
    "NOVEL_VALUE_DIMENSION",
    "RawBudgetExhaustedError",
    "RawFeatureRow",
    "RawFeatureWorker",
    "RawPlanEntry",
    "RawScanFatalError",
    "RawScanOutcome",
    "RawScanPlan",
    "RawScopeDriftError",
    "RawWorkerConfig",
    "RawWorkerError",
    "raw_scope_commitment",
]
