"""Fail-closed, append-only audit ledgers for the future Loop167 Phase-B run."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional

RAW_RECORD_SCHEMA = "axon_loop167_phase_b_raw_progress_ledger_record_v1"
FIT_RECORD_SCHEMA = "axon_loop167_phase_b_fit_progress_ledger_record_v1"
GENESIS_SHA256 = "0" * 64
FEATURE_ROW_GENESIS_SHA256 = hashlib.sha256(
    b"axon_loop167_phase_b_feature_row_chain_v1"
).hexdigest()
MAX_RECORD_BYTES = 2048
FIT_ARM_COUNT = 5
FIT_REPLAY_COUNT = 3
FIT_FOLD_COUNT = 5
EXPECTED_FIT_UNIT_COUNT = FIT_ARM_COUNT * FIT_REPLAY_COUNT * FIT_FOLD_COUNT
LOWER_HEX = frozenset("0123456789abcdef")

RAW_RESULTS = frozenset(
    {
        "available",
        "oversize_declared",
        "raw_byte_budget_exhausted",
        "source_open_failure",
        "read_failure",
        "read_truncated",
        "declared_size_mismatch",
        "sha256_mismatch",
        "pe_parse_failure",
        "feature_failure",
    }
)
_RAW_RESULTS_WITHOUT_OPEN = frozenset({"oversize_declared", "raw_byte_budget_exhausted"})
_CONTROL_FIELDS = frozenset(
    {
        "schema",
        "event",
        "sequence",
        "previous_record_sha256",
        "record_sha256",
    }
)
_RAW_EVENT_FIELDS = {
    "scan_started": frozenset(
        {
            "expected_record_count",
            "maximum_raw_open_attempts",
            "maximum_raw_bytes_read",
            "raw_scope_commitment_sha256",
        }
    ),
    "raw_open_intent": frozenset({"ordinal", "source_audit_sha256"}),
    "record_terminal": frozenset(
        {
            "ordinal",
            "source_audit_sha256",
            "result",
            "cumulative_raw_open_attempts",
            "cumulative_raw_bytes_read",
            "feature_row_commitment_sha256",
        }
    ),
    "scan_completed": frozenset(
        {
            "record_count",
            "cumulative_raw_open_attempts",
            "cumulative_raw_bytes_read",
            "feature_rows_commitment_sha256",
        }
    ),
}
_FIT_EVENT_FIELDS = {
    "fit_started": frozenset(
        {
            "fit_protocol_commitment_sha256",
            "feature_rows_commitment_sha256",
            "raw_ledger_final_record_sha256",
        }
    ),
    "fit_unit_completed": frozenset({"arm_ordinal", "replay_ordinal", "fold_ordinal"}),
    "fit_completed": frozenset({"unit_count"}),
}
_AUDIT_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
_EXPECTED_FIT_UNITS = frozenset(
    (arm_ordinal, replay_ordinal, fold_ordinal)
    for arm_ordinal in range(FIT_ARM_COUNT)
    for replay_ordinal in range(FIT_REPLAY_COUNT)
    for fold_ordinal in range(FIT_FOLD_COUNT)
)


class ProgressLedgerError(RuntimeError):
    """Raised when an append or validation violates a frozen ledger contract."""

    def __init__(self, code: str, message: str, *, category: str = "tampered") -> None:
        super().__init__(message)
        self.code = code
        self.category = category


RawScanLedgerError = ProgressLedgerError
RawProgressLedgerError = ProgressLedgerError
FitLedgerError = ProgressLedgerError
FitProgressLedgerError = ProgressLedgerError


@dataclass(frozen=True)
class RawScanLedgerValidation:
    status: str
    complete: bool
    issues: tuple[str, ...]
    line_count: int
    terminal_record_count: int
    expected_record_count: Optional[int]
    final_record_sha256: Optional[str]
    raw_scope_commitment_sha256: Optional[str]
    maximum_raw_open_attempts: Optional[int]
    maximum_raw_bytes_read: Optional[int]
    cumulative_raw_open_attempts: int
    cumulative_raw_bytes_read: int
    feature_rows_commitment_sha256: str


@dataclass(frozen=True)
class FitLedgerValidation:
    status: str
    complete: bool
    issues: tuple[str, ...]
    line_count: int
    completed_unit_count: int
    expected_unit_count: int
    missing_units: tuple[tuple[int, int, int], ...]
    final_record_sha256: Optional[str]
    fit_protocol_commitment_sha256: Optional[str]
    feature_rows_commitment_sha256: Optional[str]
    raw_ledger_final_record_sha256: Optional[str]


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise ProgressLedgerError(
            "record_not_canonical_json",
            "Ledger record is not canonical JSON",
        ) from exc


def _record_sha256(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= LOWER_HEX


def _sha256_field(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise ProgressLedgerError(
            "audit_sha256_invalid",
            f"{name} must be a lowercase SHA-256 digest",
        )
    return str(value)


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProgressLedgerError(
            "audit_integer_invalid",
            f"{name} must be a non-negative integer",
        )
    return value


def _bounded_ordinal(value: object, *, name: str, upper_bound: int) -> int:
    ordinal = _nonnegative_integer(value, name=name)
    if ordinal >= upper_bound:
        raise ProgressLedgerError(
            "fit_ordinal_invalid",
            f"{name} is outside the fixed Phase-B fit grid",
        )
    return ordinal


def _validate_event_fields(
    event: str,
    fields: Mapping[str, object],
    *,
    event_fields: Mapping[str, frozenset[str]],
) -> None:
    expected = event_fields.get(event)
    if expected is None:
        raise ProgressLedgerError("event_invalid", "Ledger event is outside the frozen schema")
    if any(not isinstance(name, str) or not _AUDIT_KEY.fullmatch(name) for name in fields):
        raise ProgressLedgerError(
            "audit_field_name_invalid",
            "Ledger fields must use canonical audit names",
        )
    if set(fields) != expected:
        raise ProgressLedgerError(
            "audit_fields_invalid",
            f"{event} contains missing, extra, or disallowed fields",
        )


def _next_feature_rows_commitment(
    previous_commitment_sha256: str,
    *,
    ordinal: int,
    source_audit_sha256: str,
    feature_row_commitment_sha256: str,
) -> str:
    payload = {
        "feature_row_commitment_sha256": feature_row_commitment_sha256,
        "ordinal": ordinal,
        "previous_commitment_sha256": previous_commitment_sha256,
        "source_audit_sha256": source_audit_sha256,
    }
    material = b"axon_loop167_phase_b_feature_row_chain_v1\0" + _canonical_json_bytes(payload)
    return hashlib.sha256(material).hexdigest()


class _RawScanState:
    def __init__(self) -> None:
        self.started = False
        self.completed = False
        self.expected_record_count: Optional[int] = None
        self.raw_scope_commitment_sha256: Optional[str] = None
        self.maximum_raw_open_attempts: Optional[int] = None
        self.maximum_raw_bytes_read: Optional[int] = None
        self.pending_record: Optional[tuple[int, str]] = None
        self.seen_source_audit_sha256: set[str] = set()
        self.terminal_record_count = 0
        self.cumulative_raw_open_attempts = 0
        self.cumulative_raw_bytes_read = 0
        self.feature_rows_commitment_sha256 = FEATURE_ROW_GENESIS_SHA256

    @staticmethod
    def _order_error(code: str, message: str) -> ProgressLedgerError:
        return ProgressLedgerError(code, message, category="reordered_or_deleted")

    def apply(self, event: str, fields: Mapping[str, object]) -> None:
        _validate_event_fields(event, fields, event_fields=_RAW_EVENT_FIELDS)
        if self.completed:
            raise self._order_error(
                "record_after_scan_completed",
                "No raw scan record may follow scan_completed",
            )

        if event == "scan_started":
            if self.started:
                raise self._order_error(
                    "scan_started_repeated",
                    "scan_started must be the first and only start event",
                )
            expected_record_count = _nonnegative_integer(
                fields["expected_record_count"], name="expected_record_count"
            )
            maximum_raw_open_attempts = _nonnegative_integer(
                fields["maximum_raw_open_attempts"], name="maximum_raw_open_attempts"
            )
            maximum_raw_bytes_read = _nonnegative_integer(
                fields["maximum_raw_bytes_read"], name="maximum_raw_bytes_read"
            )
            raw_scope_commitment_sha256 = _sha256_field(
                fields["raw_scope_commitment_sha256"], name="raw_scope_commitment_sha256"
            )
            if maximum_raw_open_attempts > expected_record_count:
                raise ProgressLedgerError(
                    "raw_open_budget_invalid",
                    "Raw open budget cannot exceed the one-pass record count",
                )
            self.expected_record_count = expected_record_count
            self.maximum_raw_open_attempts = maximum_raw_open_attempts
            self.maximum_raw_bytes_read = maximum_raw_bytes_read
            self.raw_scope_commitment_sha256 = raw_scope_commitment_sha256
            self.started = True
            return

        if not self.started:
            raise self._order_error(
                "scan_started_missing",
                "scan_started must precede every other raw scan event",
            )

        if event == "raw_open_intent":
            if self.pending_record is not None:
                raise self._order_error(
                    "record_terminal_missing",
                    "The preceding raw_open_intent has no record_terminal",
                )
            ordinal = _nonnegative_integer(fields["ordinal"], name="ordinal")
            source_audit_sha256 = _sha256_field(
                fields["source_audit_sha256"], name="source_audit_sha256"
            )
            if ordinal != self.terminal_record_count:
                raise self._order_error(
                    "ordinal_not_contiguous",
                    "Raw scan ordinals must be contiguous",
                )
            if self.expected_record_count is None or ordinal >= self.expected_record_count:
                raise self._order_error(
                    "record_count_exceeded",
                    "raw_open_intent exceeds the declared raw scan scope",
                )
            if source_audit_sha256 in self.seen_source_audit_sha256:
                raise self._order_error(
                    "source_audit_repeated",
                    "Each committed source may have one terminal feature row",
                )
            self.pending_record = (ordinal, source_audit_sha256)
            return

        if event == "record_terminal":
            if self.pending_record is None:
                raise self._order_error(
                    "raw_open_intent_missing",
                    "record_terminal must follow one raw_open_intent",
                )
            ordinal = _nonnegative_integer(fields["ordinal"], name="ordinal")
            source_audit_sha256 = _sha256_field(
                fields["source_audit_sha256"], name="source_audit_sha256"
            )
            if (ordinal, source_audit_sha256) != self.pending_record:
                raise self._order_error(
                    "record_identity_mismatch",
                    "record_terminal does not close its raw_open_intent",
                )
            result = fields["result"]
            if not isinstance(result, str) or result not in RAW_RESULTS:
                raise ProgressLedgerError(
                    "record_result_invalid",
                    "record_terminal result is outside the frozen allowlist",
                )
            attempts = _nonnegative_integer(
                fields["cumulative_raw_open_attempts"], name="cumulative_raw_open_attempts"
            )
            bytes_read = _nonnegative_integer(
                fields["cumulative_raw_bytes_read"], name="cumulative_raw_bytes_read"
            )
            feature_row_commitment_sha256 = _sha256_field(
                fields["feature_row_commitment_sha256"], name="feature_row_commitment_sha256"
            )
            attempt_delta = attempts - self.cumulative_raw_open_attempts
            byte_delta = bytes_read - self.cumulative_raw_bytes_read
            expected_attempt_delta = 0 if result in _RAW_RESULTS_WITHOUT_OPEN else 1
            if attempt_delta != expected_attempt_delta or byte_delta < 0:
                raise ProgressLedgerError(
                    "record_counts_invalid",
                    "record_terminal counters are inconsistent with its result",
                )
            if attempt_delta == 0 and byte_delta != 0:
                raise ProgressLedgerError(
                    "record_counts_invalid",
                    "A no-open raw result cannot increase the byte counter",
                )
            if (
                self.maximum_raw_open_attempts is None
                or self.maximum_raw_bytes_read is None
                or attempts > self.maximum_raw_open_attempts
                or bytes_read > self.maximum_raw_bytes_read
            ):
                raise ProgressLedgerError(
                    "raw_budget_exceeded",
                    "record_terminal exceeds the frozen raw access budget",
                )

            # 每个终态仅提交安全摘要；聚合摘要绑定完整的有序特征行集合。
            self.feature_rows_commitment_sha256 = _next_feature_rows_commitment(
                self.feature_rows_commitment_sha256,
                ordinal=ordinal,
                source_audit_sha256=source_audit_sha256,
                feature_row_commitment_sha256=feature_row_commitment_sha256,
            )
            self.cumulative_raw_open_attempts = attempts
            self.cumulative_raw_bytes_read = bytes_read
            self.seen_source_audit_sha256.add(source_audit_sha256)
            self.terminal_record_count += 1
            self.pending_record = None
            return

        if self.pending_record is not None:
            raise self._order_error(
                "record_terminal_missing",
                "scan_completed cannot close a pending raw_open_intent",
            )
        record_count = _nonnegative_integer(fields["record_count"], name="record_count")
        attempts = _nonnegative_integer(
            fields["cumulative_raw_open_attempts"], name="cumulative_raw_open_attempts"
        )
        bytes_read = _nonnegative_integer(
            fields["cumulative_raw_bytes_read"], name="cumulative_raw_bytes_read"
        )
        feature_rows_commitment_sha256 = _sha256_field(
            fields["feature_rows_commitment_sha256"], name="feature_rows_commitment_sha256"
        )
        if record_count != self.terminal_record_count or record_count != self.expected_record_count:
            raise self._order_error(
                "scan_record_count_mismatch",
                "scan_completed does not close the declared raw scan scope",
            )
        if (
            attempts != self.cumulative_raw_open_attempts
            or bytes_read != self.cumulative_raw_bytes_read
        ):
            raise ProgressLedgerError(
                "scan_counts_mismatch",
                "scan_completed counters do not match the terminal record",
            )
        if feature_rows_commitment_sha256 != self.feature_rows_commitment_sha256:
            raise ProgressLedgerError(
                "feature_rows_commitment_mismatch",
                "scan_completed does not bind the terminal feature row commitments",
            )
        self.completed = True

    def validation(
        self,
        *,
        status: str,
        issues: tuple[str, ...],
        line_count: int,
        final_record_sha256: Optional[str],
    ) -> RawScanLedgerValidation:
        return RawScanLedgerValidation(
            status=status,
            complete=status == "complete",
            issues=issues,
            line_count=line_count,
            terminal_record_count=self.terminal_record_count,
            expected_record_count=self.expected_record_count,
            final_record_sha256=final_record_sha256,
            raw_scope_commitment_sha256=self.raw_scope_commitment_sha256,
            maximum_raw_open_attempts=self.maximum_raw_open_attempts,
            maximum_raw_bytes_read=self.maximum_raw_bytes_read,
            cumulative_raw_open_attempts=self.cumulative_raw_open_attempts,
            cumulative_raw_bytes_read=self.cumulative_raw_bytes_read,
            feature_rows_commitment_sha256=self.feature_rows_commitment_sha256,
        )

    def incomplete_issues(self) -> tuple[str, ...]:
        issues = ["scan_not_completed"]
        if not self.started:
            issues.append("scan_started_missing")
        if self.pending_record is not None:
            issues.append("record_terminal_missing")
        return tuple(issues)


class _FitState:
    def __init__(self) -> None:
        self.started = False
        self.completed = False
        self.fit_protocol_commitment_sha256: Optional[str] = None
        self.feature_rows_commitment_sha256: Optional[str] = None
        self.raw_ledger_final_record_sha256: Optional[str] = None
        self.completed_units: set[tuple[int, int, int]] = set()

    @staticmethod
    def _order_error(code: str, message: str) -> ProgressLedgerError:
        return ProgressLedgerError(code, message, category="reordered_or_deleted")

    def apply(self, event: str, fields: Mapping[str, object]) -> None:
        _validate_event_fields(event, fields, event_fields=_FIT_EVENT_FIELDS)
        if self.completed:
            raise self._order_error(
                "record_after_fit_completed",
                "No fit ledger record may follow fit_completed",
            )

        if event == "fit_started":
            if self.started:
                raise self._order_error(
                    "fit_started_repeated",
                    "fit_started must be the first and only start event",
                )
            self.fit_protocol_commitment_sha256 = _sha256_field(
                fields["fit_protocol_commitment_sha256"], name="fit_protocol_commitment_sha256"
            )
            self.feature_rows_commitment_sha256 = _sha256_field(
                fields["feature_rows_commitment_sha256"], name="feature_rows_commitment_sha256"
            )
            self.raw_ledger_final_record_sha256 = _sha256_field(
                fields["raw_ledger_final_record_sha256"], name="raw_ledger_final_record_sha256"
            )
            self.started = True
            return

        if not self.started:
            raise self._order_error(
                "fit_started_missing",
                "fit_started must precede every fit unit",
            )

        if event == "fit_unit_completed":
            unit = (
                _bounded_ordinal(fields["arm_ordinal"], name="arm_ordinal", upper_bound=FIT_ARM_COUNT),
                _bounded_ordinal(
                    fields["replay_ordinal"], name="replay_ordinal", upper_bound=FIT_REPLAY_COUNT
                ),
                _bounded_ordinal(fields["fold_ordinal"], name="fold_ordinal", upper_bound=FIT_FOLD_COUNT),
            )
            if unit not in _EXPECTED_FIT_UNITS:
                raise ProgressLedgerError("fit_ordinal_invalid", "Fit unit is outside the fixed grid")
            if unit in self.completed_units:
                raise self._order_error(
                    "fit_unit_repeated",
                    "Each fixed fit unit may be completed exactly once",
                )
            self.completed_units.add(unit)
            return

        unit_count = _nonnegative_integer(fields["unit_count"], name="unit_count")
        if unit_count != EXPECTED_FIT_UNIT_COUNT or self.completed_units != _EXPECTED_FIT_UNITS:
            raise self._order_error(
                "fit_coverage_incomplete",
                "fit_completed requires all 5 arms x 3 replays x 5 folds exactly once",
            )
        self.completed = True

    def validation(
        self,
        *,
        status: str,
        issues: tuple[str, ...],
        line_count: int,
        final_record_sha256: Optional[str],
    ) -> FitLedgerValidation:
        missing_units = tuple(sorted(_EXPECTED_FIT_UNITS - self.completed_units))
        return FitLedgerValidation(
            status=status,
            complete=status == "complete",
            issues=issues,
            line_count=line_count,
            completed_unit_count=len(self.completed_units),
            expected_unit_count=EXPECTED_FIT_UNIT_COUNT,
            missing_units=missing_units,
            final_record_sha256=final_record_sha256,
            fit_protocol_commitment_sha256=self.fit_protocol_commitment_sha256,
            feature_rows_commitment_sha256=self.feature_rows_commitment_sha256,
            raw_ledger_final_record_sha256=self.raw_ledger_final_record_sha256,
        )

    def incomplete_issues(self) -> tuple[str, ...]:
        issues = ["fit_not_completed"]
        if not self.started:
            issues.append("fit_started_missing")
        if self.completed_units != _EXPECTED_FIT_UNITS:
            issues.append("fit_units_missing")
        return tuple(issues)


class _AppendOnlyLedger:
    """Shared durable JSONL writer; subclasses provide only frozen state machines."""

    RECORD_SCHEMA: str

    def __init__(self, path: str | Path, state: _RawScanState | _FitState) -> None:
        self.path = Path(path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        try:
            self._handle: BinaryIO = os.fdopen(descriptor, "wb")
        except Exception:
            os.close(descriptor)
            raise
        self._state = state
        self._sequence = 0
        self._previous_record_sha256 = GENESIS_SHA256
        self._closed = False
        self._broken = False

    @property
    def final_record_sha256(self) -> Optional[str]:
        if self._sequence == 0:
            return None
        return self._previous_record_sha256

    def __enter__(self) -> "_AppendOnlyLedger":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._handle.close()

    def append_event(self, event: str, **fields: object) -> str:
        if self._closed:
            raise ProgressLedgerError("ledger_closed", "Ledger is closed")
        if self._broken:
            raise ProgressLedgerError(
                "ledger_broken",
                "Ledger cannot continue after a failed durable append",
            )
        if not isinstance(event, str):
            raise ProgressLedgerError("event_invalid", "Ledger event must be a string")

        # 状态机先拒绝越序、重复和敏感字段，再把唯一规范行写入 SHA 链。
        self._state.apply(event, fields)
        body: dict[str, object] = {
            "schema": self.RECORD_SCHEMA,
            "event": event,
            "sequence": self._sequence,
            "previous_record_sha256": self._previous_record_sha256,
            **fields,
        }
        record_sha256 = _record_sha256(body)
        line = _canonical_json_bytes({**body, "record_sha256": record_sha256}) + b"\n"
        if len(line) > MAX_RECORD_BYTES:
            self._broken = True
            raise ProgressLedgerError(
                "record_too_large",
                "Ledger record exceeds its fixed size cap",
            )
        try:
            written = self._handle.write(line)
            if written != len(line):
                raise OSError("short durable ledger write")
            self._handle.flush()
            # 每条完整 JSONL 行单独 fsync，事故只能留下可审计的不可续写前缀。
            os.fsync(self._handle.fileno())
        except (OSError, ValueError) as exc:
            self._broken = True
            raise ProgressLedgerError(
                "durable_append_failed",
                "Ledger record could not be durably appended",
            ) from exc
        self._sequence += 1
        self._previous_record_sha256 = record_sha256
        return record_sha256


class RawScanLedger(_AppendOnlyLedger):
    """Exclusive one-pass raw-scan audit ledger with no path or sample payload fields."""

    RECORD_SCHEMA = RAW_RECORD_SCHEMA

    def __init__(self, path: str | Path) -> None:
        super().__init__(path, _RawScanState())

    @classmethod
    def create(cls, path: str | Path) -> "RawScanLedger":
        return cls(path)

    @property
    def feature_rows_commitment_sha256(self) -> str:
        return self._state.feature_rows_commitment_sha256

    def scan_started(
        self,
        *,
        expected_record_count: int,
        maximum_raw_open_attempts: int,
        maximum_raw_bytes_read: int,
        raw_scope_commitment_sha256: str,
    ) -> str:
        return self.append_event(
            "scan_started",
            expected_record_count=expected_record_count,
            maximum_raw_open_attempts=maximum_raw_open_attempts,
            maximum_raw_bytes_read=maximum_raw_bytes_read,
            raw_scope_commitment_sha256=raw_scope_commitment_sha256,
        )

    def raw_open_intent(self, *, ordinal: int, source_audit_sha256: str) -> str:
        return self.append_event(
            "raw_open_intent",
            ordinal=ordinal,
            source_audit_sha256=source_audit_sha256,
        )

    def record_terminal(
        self,
        *,
        ordinal: int,
        source_audit_sha256: str,
        result: str,
        cumulative_raw_open_attempts: int,
        cumulative_raw_bytes_read: int,
        feature_row_commitment_sha256: str,
    ) -> str:
        return self.append_event(
            "record_terminal",
            ordinal=ordinal,
            source_audit_sha256=source_audit_sha256,
            result=result,
            cumulative_raw_open_attempts=cumulative_raw_open_attempts,
            cumulative_raw_bytes_read=cumulative_raw_bytes_read,
            feature_row_commitment_sha256=feature_row_commitment_sha256,
        )

    def scan_completed(
        self,
        *,
        record_count: int,
        cumulative_raw_open_attempts: int,
        cumulative_raw_bytes_read: int,
        feature_rows_commitment_sha256: str,
    ) -> str:
        return self.append_event(
            "scan_completed",
            record_count=record_count,
            cumulative_raw_open_attempts=cumulative_raw_open_attempts,
            cumulative_raw_bytes_read=cumulative_raw_bytes_read,
            feature_rows_commitment_sha256=feature_rows_commitment_sha256,
        )


class FitLedger(_AppendOnlyLedger):
    """Exclusive ledger requiring all fixed 5 x 3 x 5 fit units exactly once."""

    RECORD_SCHEMA = FIT_RECORD_SCHEMA

    def __init__(self, path: str | Path) -> None:
        super().__init__(path, _FitState())

    @classmethod
    def create(cls, path: str | Path) -> "FitLedger":
        return cls(path)

    def fit_started(
        self,
        *,
        fit_protocol_commitment_sha256: str,
        feature_rows_commitment_sha256: str,
        raw_ledger_final_record_sha256: str,
    ) -> str:
        return self.append_event(
            "fit_started",
            fit_protocol_commitment_sha256=fit_protocol_commitment_sha256,
            feature_rows_commitment_sha256=feature_rows_commitment_sha256,
            raw_ledger_final_record_sha256=raw_ledger_final_record_sha256,
        )

    def fit_unit_completed(
        self,
        *,
        arm_ordinal: int,
        replay_ordinal: int,
        fold_ordinal: int,
    ) -> str:
        return self.append_event(
            "fit_unit_completed",
            arm_ordinal=arm_ordinal,
            replay_ordinal=replay_ordinal,
            fold_ordinal=fold_ordinal,
        )

    def fit_completed(self, *, unit_count: int = EXPECTED_FIT_UNIT_COUNT) -> str:
        return self.append_event("fit_completed", unit_count=unit_count)


RawProgressLedger = RawScanLedger
FitProgressLedger = FitLedger


def _open_ledger_for_read(path: str | Path) -> BinaryIO:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(Path(path), flags)
    try:
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def _validate_ledger(
    path: str | Path,
    *,
    record_schema: str,
    event_fields: Mapping[str, frozenset[str]],
    state: _RawScanState | _FitState,
) -> RawScanLedgerValidation | FitLedgerValidation:
    expected_sequence = 0
    expected_previous_record_sha256 = GENESIS_SHA256
    final_record_sha256: Optional[str] = None

    # 只读验证，绝不修复、截断或重新打开已有 ledger。
    with _open_ledger_for_read(path) as handle:
        while True:
            raw_line = handle.readline(MAX_RECORD_BYTES + 1)
            if not raw_line:
                break
            if len(raw_line) > MAX_RECORD_BYTES:
                return state.validation(
                    status="tampered",
                    issues=("record_too_large",),
                    line_count=expected_sequence,
                    final_record_sha256=final_record_sha256,
                )
            if not raw_line.endswith(b"\n"):
                return state.validation(
                    status="torn_tail",
                    issues=("torn_tail",),
                    line_count=expected_sequence,
                    final_record_sha256=final_record_sha256,
                )
            try:
                record = json.loads(raw_line[:-1].decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return state.validation(
                    status="tampered",
                    issues=("record_json_invalid",),
                    line_count=expected_sequence,
                    final_record_sha256=final_record_sha256,
                )
            if not isinstance(record, dict):
                return state.validation(
                    status="tampered",
                    issues=("record_not_object",),
                    line_count=expected_sequence,
                    final_record_sha256=final_record_sha256,
                )
            try:
                if _canonical_json_bytes(record) + b"\n" != raw_line:
                    raise ProgressLedgerError(
                        "record_not_canonical_json",
                        "Ledger line is not canonical JSON",
                    )
                event = record.get("event")
                if not isinstance(event, str) or event not in event_fields:
                    raise ProgressLedgerError(
                        "event_invalid",
                        "Ledger event is outside the frozen schema",
                    )
                if set(record) != _CONTROL_FIELDS | event_fields[event]:
                    raise ProgressLedgerError(
                        "audit_fields_invalid",
                        "Ledger record contains missing, extra, or disallowed fields",
                    )
                if record.get("schema") != record_schema:
                    raise ProgressLedgerError(
                        "record_schema_invalid",
                        "Ledger record schema drifted",
                    )
                sequence = _nonnegative_integer(record.get("sequence"), name="sequence")
                if sequence != expected_sequence:
                    raise ProgressLedgerError(
                        "record_sequence_mismatch",
                        "Ledger records were reordered or deleted",
                        category="reordered_or_deleted",
                    )
                previous_record_sha256 = _sha256_field(
                    record.get("previous_record_sha256"), name="previous_record_sha256"
                )
                if previous_record_sha256 != expected_previous_record_sha256:
                    raise ProgressLedgerError(
                        "chain_link_mismatch",
                        "Ledger hash chain was reordered or deleted",
                        category="reordered_or_deleted",
                    )
                observed_record_sha256 = _sha256_field(
                    record.get("record_sha256"), name="record_sha256"
                )
                body = {name: value for name, value in record.items() if name != "record_sha256"}
                if observed_record_sha256 != _record_sha256(body):
                    raise ProgressLedgerError(
                        "record_sha256_mismatch",
                        "Ledger record content was tampered",
                    )
                state.apply(event, {name: record[name] for name in event_fields[event]})
            except ProgressLedgerError as exc:
                return state.validation(
                    status=exc.category,
                    issues=(exc.code,),
                    line_count=expected_sequence,
                    final_record_sha256=final_record_sha256,
                )
            expected_sequence += 1
            expected_previous_record_sha256 = observed_record_sha256
            final_record_sha256 = observed_record_sha256

    if state.completed:
        return state.validation(
            status="complete",
            issues=(),
            line_count=expected_sequence,
            final_record_sha256=final_record_sha256,
        )
    return state.validation(
        status="incomplete",
        issues=state.incomplete_issues(),
        line_count=expected_sequence,
        final_record_sha256=final_record_sha256,
    )


def validate_raw_scan_ledger(path: str | Path) -> RawScanLedgerValidation:
    """Validate a raw ledger read-only and return its fail-closed audit status."""

    result = _validate_ledger(
        path,
        record_schema=RAW_RECORD_SCHEMA,
        event_fields=_RAW_EVENT_FIELDS,
        state=_RawScanState(),
    )
    assert isinstance(result, RawScanLedgerValidation)
    return result


def validate_fit_ledger(path: str | Path) -> FitLedgerValidation:
    """Validate the exact 75-unit fit ledger without reopening it for append."""

    result = _validate_ledger(
        path,
        record_schema=FIT_RECORD_SCHEMA,
        event_fields=_FIT_EVENT_FIELDS,
        state=_FitState(),
    )
    assert isinstance(result, FitLedgerValidation)
    return result


validate_raw_progress_ledger = validate_raw_scan_ledger
