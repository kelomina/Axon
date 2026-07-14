"""Fail-closed append-only progress ledger for the Loop166 raw scan."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional

RECORD_SCHEMA = "axon_loop166_raw_progress_ledger_record_v1"
GENESIS_SHA256 = "0" * 64
MAX_RECORD_BYTES = 4096
LOWER_HEX = frozenset("0123456789abcdef")
ALLOWED_RESULTS = frozenset(
    {
        "available",
        "source_unavailable",
        "parse_failure",
        "no_executable_section",
        "zero_raw_executable_section",
        "invalid_executable_section_span",
    }
)

_CONTROL_FIELDS = frozenset(
    {
        "schema",
        "event",
        "sequence",
        "previous_record_sha256",
        "record_sha256",
    }
)
_EVENT_FIELDS = {
    "scan_started": frozenset({"expected_record_count"}),
    "raw_open_intent": frozenset({"ordinal", "row_index", "source_sha256"}),
    "record_terminal": frozenset(
        {
            "ordinal",
            "row_index",
            "source_sha256",
            "result",
            "cumulative_raw_open_attempts",
            "cumulative_raw_open_successes",
            "cumulative_raw_bytes_read",
        }
    ),
    "scan_completed": frozenset(
        {
            "record_count",
            "cumulative_raw_open_attempts",
            "cumulative_raw_open_successes",
            "cumulative_raw_bytes_read",
            "corpus_commitment_sha256",
        }
    ),
}
_AUDIT_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


class RawProgressLedgerError(RuntimeError):
    """Raised when a ledger write would violate its frozen audit schema."""

    def __init__(self, code: str, message: str, *, category: str = "tampered") -> None:
        super().__init__(message)
        self.code = code
        self.category = category


@dataclass(frozen=True)
class RawProgressLedgerValidation:
    status: str
    complete: bool
    issues: tuple[str, ...]
    line_count: int
    terminal_record_count: int
    expected_record_count: Optional[int]
    final_record_sha256: Optional[str]
    cumulative_raw_open_attempts: int
    cumulative_raw_open_successes: int
    cumulative_raw_bytes_read: int
    corpus_commitment_sha256: Optional[str]


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
        raise RawProgressLedgerError(
            "record_not_canonical_json",
            "Raw progress record is not canonical JSON",
        ) from exc


def _record_sha256(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= LOWER_HEX


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RawProgressLedgerError(
            "audit_integer_invalid",
            f"{name} must be a non-negative integer",
        )
    return value


def _sha256_field(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise RawProgressLedgerError(
            "audit_sha256_invalid",
            f"{name} must be a lowercase SHA-256 digest",
        )
    return str(value)


class _LedgerState:
    def __init__(self) -> None:
        self.started = False
        self.completed = False
        self.expected_record_count: Optional[int] = None
        self.terminal_record_count = 0
        self.pending_record: Optional[tuple[int, int, str]] = None
        self.seen_row_indices: set[int] = set()
        self.cumulative_raw_open_attempts = 0
        self.cumulative_raw_open_successes = 0
        self.cumulative_raw_bytes_read = 0
        self.corpus_commitment_sha256: Optional[str] = None

    @staticmethod
    def _order_error(code: str, message: str) -> RawProgressLedgerError:
        return RawProgressLedgerError(code, message, category="reordered_or_deleted")

    @staticmethod
    def _validate_fields(event: str, fields: Mapping[str, object]) -> None:
        expected = _EVENT_FIELDS.get(event)
        if expected is None:
            raise RawProgressLedgerError(
                "event_invalid",
                f"Unknown raw progress event: {event}",
            )
        if any(not isinstance(key, str) or not _AUDIT_KEY.fullmatch(key) for key in fields):
            raise RawProgressLedgerError(
                "audit_field_name_invalid",
                "Raw progress fields must use canonical audit names",
            )
        if set(fields) != expected:
            raise RawProgressLedgerError(
                "audit_fields_invalid",
                f"{event} contains missing, extra, or sensitive fields",
            )

    def apply(self, event: str, fields: Mapping[str, object]) -> None:
        self._validate_fields(event, fields)
        if self.completed:
            raise self._order_error(
                "record_after_scan_completed",
                "No record may follow scan_completed",
            )

        if event == "scan_started":
            if self.started:
                raise self._order_error(
                    "scan_started_repeated",
                    "scan_started must be the first and only start event",
                )
            self.expected_record_count = _nonnegative_integer(
                fields["expected_record_count"],
                name="expected_record_count",
            )
            self.started = True
            return

        if not self.started:
            raise self._order_error(
                "scan_started_missing",
                "scan_started must precede every other event",
            )

        if event == "raw_open_intent":
            if self.pending_record is not None:
                raise self._order_error(
                    "record_terminal_missing",
                    "The preceding raw_open_intent has no record_terminal",
                )
            ordinal = _nonnegative_integer(fields["ordinal"], name="ordinal")
            row_index = _nonnegative_integer(fields["row_index"], name="row_index")
            source_sha256 = _sha256_field(fields["source_sha256"], name="source_sha256")
            if ordinal != self.terminal_record_count:
                raise self._order_error(
                    "ordinal_not_contiguous",
                    "Raw progress ordinals must be contiguous",
                )
            if row_index in self.seen_row_indices:
                raise self._order_error(
                    "row_index_repeated",
                    "Each source row may appear only once",
                )
            self.pending_record = (ordinal, row_index, source_sha256)
            return

        if event == "record_terminal":
            if self.pending_record is None:
                raise self._order_error(
                    "raw_open_intent_missing",
                    "record_terminal must follow one raw_open_intent",
                )
            identity = (
                _nonnegative_integer(fields["ordinal"], name="ordinal"),
                _nonnegative_integer(fields["row_index"], name="row_index"),
                _sha256_field(fields["source_sha256"], name="source_sha256"),
            )
            if identity != self.pending_record:
                raise self._order_error(
                    "record_identity_mismatch",
                    "record_terminal does not close its raw_open_intent",
                )
            result = fields["result"]
            if not isinstance(result, str) or result not in ALLOWED_RESULTS:
                raise RawProgressLedgerError(
                    "record_result_invalid",
                    "record_terminal result is outside the audit allowlist",
                )
            attempts = _nonnegative_integer(
                fields["cumulative_raw_open_attempts"],
                name="cumulative_raw_open_attempts",
            )
            successes = _nonnegative_integer(
                fields["cumulative_raw_open_successes"],
                name="cumulative_raw_open_successes",
            )
            bytes_read = _nonnegative_integer(
                fields["cumulative_raw_bytes_read"],
                name="cumulative_raw_bytes_read",
            )
            attempt_delta = attempts - self.cumulative_raw_open_attempts
            success_delta = successes - self.cumulative_raw_open_successes
            byte_delta = bytes_read - self.cumulative_raw_bytes_read
            if (
                attempt_delta not in {0, 1}
                or success_delta not in {0, 1}
                or success_delta > attempt_delta
                or byte_delta < 0
                or (byte_delta > 0 and success_delta != 1)
            ):
                raise RawProgressLedgerError(
                    "record_counts_invalid",
                    "record_terminal cumulative counters are inconsistent",
                )
            self.cumulative_raw_open_attempts = attempts
            self.cumulative_raw_open_successes = successes
            self.cumulative_raw_bytes_read = bytes_read
            self.seen_row_indices.add(identity[1])
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
            fields["cumulative_raw_open_attempts"],
            name="cumulative_raw_open_attempts",
        )
        successes = _nonnegative_integer(
            fields["cumulative_raw_open_successes"],
            name="cumulative_raw_open_successes",
        )
        bytes_read = _nonnegative_integer(
            fields["cumulative_raw_bytes_read"],
            name="cumulative_raw_bytes_read",
        )
        commitment = _sha256_field(
            fields["corpus_commitment_sha256"],
            name="corpus_commitment_sha256",
        )
        if record_count != self.terminal_record_count or record_count != self.expected_record_count:
            raise self._order_error(
                "scan_record_count_mismatch",
                "scan_completed record count does not close the declared scan",
            )
        if (
            attempts != self.cumulative_raw_open_attempts
            or successes != self.cumulative_raw_open_successes
            or bytes_read != self.cumulative_raw_bytes_read
        ):
            raise RawProgressLedgerError(
                "scan_counts_mismatch",
                "scan_completed counters do not match the final terminal record",
            )
        self.corpus_commitment_sha256 = commitment
        self.completed = True


class RawProgressLedger:
    """Create and append one new durable ledger; existing ledgers are never reopened."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        # O_EXCL 是租约边界：事故留下的完整、半行或空文件都不得被覆盖或续写。
        descriptor = os.open(self.path, flags, 0o600)
        try:
            self._handle: BinaryIO = os.fdopen(descriptor, "wb")
        except Exception:
            os.close(descriptor)
            raise
        self._state = _LedgerState()
        self._sequence = 0
        self._previous_record_sha256 = GENESIS_SHA256
        self._closed = False
        self._broken = False

    @classmethod
    def create(cls, path: str | Path) -> "RawProgressLedger":
        return cls(path)

    @property
    def final_record_sha256(self) -> Optional[str]:
        if self._sequence == 0:
            return None
        return self._previous_record_sha256

    def __enter__(self) -> "RawProgressLedger":
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
            raise RawProgressLedgerError("ledger_closed", "Raw progress ledger is closed")
        if self._broken:
            raise RawProgressLedgerError(
                "ledger_broken",
                "Raw progress ledger cannot continue after a failed durable append",
            )
        if not isinstance(event, str):
            raise RawProgressLedgerError("event_invalid", "Raw progress event must be a string")

        # 状态机先拒绝越序、重复和敏感字段，再把唯一规范行写入 hash chain。
        self._state.apply(event, fields)
        body: dict[str, object] = {
            "schema": RECORD_SCHEMA,
            "event": event,
            "sequence": self._sequence,
            "previous_record_sha256": self._previous_record_sha256,
            **fields,
        }
        record_sha256 = _record_sha256(body)
        line = _canonical_json_bytes({**body, "record_sha256": record_sha256}) + b"\n"
        if len(line) > MAX_RECORD_BYTES:
            self._broken = True
            raise RawProgressLedgerError(
                "record_too_large",
                "Raw progress record exceeds its fixed size cap",
            )
        try:
            self._handle.write(line)
            self._handle.flush()
            # 每条完整 JSONL 行都独立落盘，断电最多产生可识别的 torn tail。
            os.fsync(self._handle.fileno())
        except (OSError, ValueError) as exc:
            self._broken = True
            raise RawProgressLedgerError(
                "durable_append_failed",
                "Raw progress record could not be durably appended",
            ) from exc
        self._sequence += 1
        self._previous_record_sha256 = record_sha256
        return record_sha256

    def scan_started(self, *, expected_record_count: int) -> str:
        return self.append_event(
            "scan_started",
            expected_record_count=expected_record_count,
        )

    def raw_open_intent(self, *, ordinal: int, row_index: int, source_sha256: str) -> str:
        return self.append_event(
            "raw_open_intent",
            ordinal=ordinal,
            row_index=row_index,
            source_sha256=source_sha256,
        )

    def record_terminal(
        self,
        *,
        ordinal: int,
        row_index: int,
        source_sha256: str,
        result: str,
        cumulative_raw_open_attempts: int,
        cumulative_raw_open_successes: int,
        cumulative_raw_bytes_read: int,
    ) -> str:
        return self.append_event(
            "record_terminal",
            ordinal=ordinal,
            row_index=row_index,
            source_sha256=source_sha256,
            result=result,
            cumulative_raw_open_attempts=cumulative_raw_open_attempts,
            cumulative_raw_open_successes=cumulative_raw_open_successes,
            cumulative_raw_bytes_read=cumulative_raw_bytes_read,
        )

    def scan_completed(
        self,
        *,
        record_count: int,
        cumulative_raw_open_attempts: int,
        cumulative_raw_open_successes: int,
        cumulative_raw_bytes_read: int,
        corpus_commitment_sha256: str,
    ) -> str:
        return self.append_event(
            "scan_completed",
            record_count=record_count,
            cumulative_raw_open_attempts=cumulative_raw_open_attempts,
            cumulative_raw_open_successes=cumulative_raw_open_successes,
            cumulative_raw_bytes_read=cumulative_raw_bytes_read,
            corpus_commitment_sha256=corpus_commitment_sha256,
        )


def _validation_result(
    *,
    status: str,
    issues: tuple[str, ...],
    line_count: int,
    state: _LedgerState,
    final_record_sha256: Optional[str],
) -> RawProgressLedgerValidation:
    return RawProgressLedgerValidation(
        status=status,
        complete=status == "complete",
        issues=issues,
        line_count=line_count,
        terminal_record_count=state.terminal_record_count,
        expected_record_count=state.expected_record_count,
        final_record_sha256=final_record_sha256,
        cumulative_raw_open_attempts=state.cumulative_raw_open_attempts,
        cumulative_raw_open_successes=state.cumulative_raw_open_successes,
        cumulative_raw_bytes_read=state.cumulative_raw_bytes_read,
        corpus_commitment_sha256=state.corpus_commitment_sha256,
    )


def validate_raw_progress_ledger(path: str | Path) -> RawProgressLedgerValidation:
    """Validate a ledger without opening it for write or repairing any prefix."""

    state = _LedgerState()
    expected_sequence = 0
    expected_previous = GENESIS_SHA256
    final_record_sha256: Optional[str] = None
    ledger_path = Path(path)

    # 只读逐行验证；任何半行都原样保留给事故审计，绝不 truncate 或恢复写入。
    with ledger_path.open("rb") as handle:
        while True:
            raw_line = handle.readline(MAX_RECORD_BYTES + 1)
            if not raw_line:
                break
            if not raw_line.endswith(b"\n"):
                return _validation_result(
                    status="torn_tail",
                    issues=("torn_tail",),
                    line_count=expected_sequence,
                    state=state,
                    final_record_sha256=final_record_sha256,
                )
            if len(raw_line) > MAX_RECORD_BYTES:
                return _validation_result(
                    status="tampered",
                    issues=("record_too_large",),
                    line_count=expected_sequence,
                    state=state,
                    final_record_sha256=final_record_sha256,
                )
            try:
                record = json.loads(raw_line[:-1].decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return _validation_result(
                    status="tampered",
                    issues=("record_json_invalid",),
                    line_count=expected_sequence,
                    state=state,
                    final_record_sha256=final_record_sha256,
                )
            if not isinstance(record, dict):
                return _validation_result(
                    status="tampered",
                    issues=("record_not_object",),
                    line_count=expected_sequence,
                    state=state,
                    final_record_sha256=final_record_sha256,
                )
            try:
                if _canonical_json_bytes(record) + b"\n" != raw_line:
                    raise RawProgressLedgerError(
                        "record_not_canonical_json",
                        "Ledger line is not canonical JSON",
                    )
                event = record.get("event")
                if not isinstance(event, str) or event not in _EVENT_FIELDS:
                    raise RawProgressLedgerError(
                        "event_invalid",
                        "Ledger event is outside the frozen schema",
                    )
                expected_fields = _CONTROL_FIELDS | _EVENT_FIELDS[event]
                if set(record) != expected_fields:
                    raise RawProgressLedgerError(
                        "audit_fields_invalid",
                        "Ledger record contains missing, extra, or sensitive fields",
                    )
                if record.get("schema") != RECORD_SCHEMA:
                    raise RawProgressLedgerError(
                        "record_schema_invalid",
                        "Ledger record schema drifted",
                    )
                sequence = _nonnegative_integer(record.get("sequence"), name="sequence")
                if sequence != expected_sequence:
                    raise RawProgressLedgerError(
                        "record_sequence_mismatch",
                        "Ledger records were reordered or deleted",
                        category="reordered_or_deleted",
                    )
                previous = _sha256_field(
                    record.get("previous_record_sha256"),
                    name="previous_record_sha256",
                )
                if previous != expected_previous:
                    raise RawProgressLedgerError(
                        "chain_link_mismatch",
                        "Ledger hash chain was reordered or deleted",
                        category="reordered_or_deleted",
                    )
                observed_sha256 = _sha256_field(
                    record.get("record_sha256"),
                    name="record_sha256",
                )
                body = {key: value for key, value in record.items() if key != "record_sha256"}
                if observed_sha256 != _record_sha256(body):
                    raise RawProgressLedgerError(
                        "record_sha256_mismatch",
                        "Ledger record content was tampered",
                    )
                fields = {key: record[key] for key in _EVENT_FIELDS[event]}
                state.apply(event, fields)
            except RawProgressLedgerError as exc:
                return _validation_result(
                    status=exc.category,
                    issues=(exc.code,),
                    line_count=expected_sequence,
                    state=state,
                    final_record_sha256=final_record_sha256,
                )
            expected_sequence += 1
            expected_previous = observed_sha256
            final_record_sha256 = observed_sha256

    if state.completed:
        return _validation_result(
            status="complete",
            issues=(),
            line_count=expected_sequence,
            state=state,
            final_record_sha256=final_record_sha256,
        )
    incomplete_issues: list[str] = ["scan_not_completed"]
    if not state.started:
        incomplete_issues.append("scan_started_missing")
    if state.pending_record is not None:
        incomplete_issues.append("record_terminal_missing")
    return _validation_result(
        status="incomplete",
        issues=tuple(incomplete_issues),
        line_count=expected_sequence,
        state=state,
        final_record_sha256=final_record_sha256,
    )
