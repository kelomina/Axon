from __future__ import annotations

import hashlib
import inspect
import json
import struct
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

import src.loop167_phase_b.raw_worker as raw_worker
from src.loop167_phase_b.arm_contract import build_arm_matrices
from src.loop167_phase_b.progress_ledger import RawScanLedger, validate_raw_scan_ledger
from src.loop167_phase_b.raw_worker import (
    RawBudgetExhaustedError,
    RawFeatureRow,
    RawFeatureWorker,
    RawPlanEntry,
    RawScanFatalError,
    RawScanPlan,
    RawScopeDriftError,
    RawWorkerConfig,
)


def _sha256(value: bytes | str) -> str:
    material = value.encode("ascii") if isinstance(value, str) else value
    return hashlib.sha256(material).hexdigest()


def _minimal_pe32() -> bytes:
    dos = bytearray(0x80)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x80)
    coff = struct.pack("<HHIIIHH", 0x14C, 1, 0, 0, 0, 0xE0, 0x0102)
    optional = bytearray(0xE0)
    struct.pack_into("<H", optional, 0, 0x10B)
    struct.pack_into("<I", optional, 4, 0x200)
    struct.pack_into("<I", optional, 8, 0x200)
    struct.pack_into("<I", optional, 16, 0x1000)
    struct.pack_into("<I", optional, 20, 0x1000)
    struct.pack_into("<I", optional, 24, 0x2000)
    struct.pack_into("<I", optional, 28, 0x400000)
    struct.pack_into("<I", optional, 32, 0x1000)
    struct.pack_into("<I", optional, 36, 0x200)
    struct.pack_into("<H", optional, 40, 6)
    struct.pack_into("<H", optional, 48, 6)
    struct.pack_into("<I", optional, 56, 0x2000)
    struct.pack_into("<I", optional, 60, 0x200)
    struct.pack_into("<H", optional, 68, 3)
    struct.pack_into("<H", optional, 70, 0x140)
    struct.pack_into("<I", optional, 72, 0x100000)
    struct.pack_into("<I", optional, 76, 0x1000)
    struct.pack_into("<I", optional, 80, 0x100000)
    struct.pack_into("<I", optional, 84, 0x1000)
    struct.pack_into("<I", optional, 92, 16)
    section = struct.pack(
        "<8sIIIIIIHHI",
        b".text\x00\x00\x00",
        0x10,
        0x1000,
        0x200,
        0x200,
        0,
        0,
        0,
        0,
        0x60000020,
    )
    headers = b"PE\x00\x00" + coff + bytes(optional) + section
    return bytes(dos) + headers + bytes(0x200 - len(headers) - len(dos)) + b"\x90" * 0x200


def _entry(source_file: Path, payload: bytes, *, ordinal: int = 0) -> RawPlanEntry:
    return RawPlanEntry(
        ordinal=ordinal,
        source_file=source_file,
        source_audit_sha256=_sha256(f"source-audit-{ordinal}"),
        declared_size=len(payload),
        expected_sha256=_sha256(payload),
    )


def _worker(
    *,
    records: int,
    maximum_bytes: int = 1024 * 1024,
    maximum_raw_bytes: int | None = None,
) -> RawFeatureWorker:
    return RawFeatureWorker(
        RawWorkerConfig(
            maximum_source_file_bytes=maximum_bytes,
            maximum_raw_open_attempts=records,
            maximum_raw_bytes_read=(
                maximum_bytes * records if maximum_raw_bytes is None else maximum_raw_bytes
            ),
            reader_chunk_bytes=31,
        )
    )


def _assert_zero_missing_row(row: RawFeatureRow) -> None:
    assert np.all(row.b0_values == 0.0)
    assert row.b0_missing_indicators.tolist() == [1.0] * 6
    assert np.all(row.b1_values == 0.0)
    assert row.b1_missing_indicators.tolist() == [1.0] * 4
    assert row.b1_sampling_indicators.tolist() == [0.0] * 3
    assert np.all(row.novel_values == 0.0)
    assert row.novel_missing_indicators.tolist() == [1.0]
    assert row.novel_complete is False


def test_raw_worker_uses_one_open_one_context_and_keeps_blocks_column_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "synthetic-minimal.exe"
    payload = _minimal_pe32()
    source_file.write_bytes(payload)
    plan = RawScanPlan.from_entries((_entry(source_file, payload),))
    ledger_file = tmp_path / "raw-progress.jsonl"
    opened = 0
    contexts = []
    original_open = raw_worker._open_source_stream
    original_from_bytes = raw_worker.RawFeatureContext.from_bytes

    def observed_open(candidate: Path):
        nonlocal opened
        opened += 1
        return original_open(candidate)

    def observed_context(*args: object, **kwargs: object):
        context = original_from_bytes(*args, **kwargs)
        contexts.append(context)
        return context

    monkeypatch.setattr(raw_worker, "_open_source_stream", observed_open)
    monkeypatch.setattr(raw_worker.RawFeatureContext, "from_bytes", observed_context)
    with RawScanLedger.create(ledger_file) as ledger:
        outcome = _worker(records=1).scan(
            plan,
            expected_raw_scope_commitment_sha256=plan.raw_scope_commitment_sha256,
            ledger=ledger,
        )

    assert opened == 1
    assert len(contexts) == 1
    assert contexts[0].pe_parse_attempts == 1
    assert contexts[0].bytez == b""
    assert contexts[0].pe is None
    assert len(outcome.rows) == 1
    row = outcome.rows[0]
    assert row.result == "available"
    assert row.b0_values.shape == (571,)
    assert row.b1_values.shape == (536,)
    assert row.novel_values.shape == (292,)
    assert row.b0_values.flags.writeable is False
    assert row.b1_values.flags.writeable is False
    assert row.novel_values.flags.writeable is False
    assert "cf" not in {field.name for field in fields(RawFeatureRow)}

    matrices = build_arm_matrices(
        row.b0_values[None, :],
        row.b0_missing_indicators[None, :],
        row.b1_values[None, :],
        row.b1_missing_indicators[None, :],
        row.novel_values[None, :],
        np.asarray([row.novel_complete], dtype=bool),
        protocol_sha256=_sha256("synthetic-protocol"),
        replay_seed=41,
        outer_fold=0,
        role="fit",
    )
    assert np.array_equal(matrices.cf[:, :577], matrices.b0)
    assert matrices.cf.shape == (1, 870)

    validation = validate_raw_scan_ledger(ledger_file)
    assert validation.complete is True
    assert validation.terminal_record_count == len(outcome.rows)
    assert validation.cumulative_raw_open_attempts == 1
    assert validation.cumulative_raw_bytes_read == len(payload)
    assert validation.feature_rows_commitment_sha256 == outcome.feature_rows_commitment_sha256
    ledger_text = ledger_file.read_text(encoding="ascii")
    assert str(source_file) not in ledger_text
    assert "source_file" not in ledger_text
    assert "label" not in ledger_text
    assert "score" not in ledger_text


def test_declared_oversize_never_opens_and_remains_a_zero_missing_denominator_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "oversize.bin"
    payload = b"synthetic-oversize"
    source_file.write_bytes(payload)
    plan = RawScanPlan.from_entries((_entry(source_file, payload),))
    ledger_file = tmp_path / "oversize-progress.jsonl"

    def forbidden_open(_: Path):
        raise AssertionError("Declared oversize source must not be opened")

    monkeypatch.setattr(raw_worker, "_open_source_stream", forbidden_open)
    with RawScanLedger.create(ledger_file) as ledger:
        outcome = _worker(records=1, maximum_bytes=len(payload) - 1).scan(
            plan,
            expected_raw_scope_commitment_sha256=plan.raw_scope_commitment_sha256,
            ledger=ledger,
        )

    assert len(outcome.rows) == 1
    assert outcome.rows[0].result == "oversize_declared"
    _assert_zero_missing_row(outcome.rows[0])
    validation = validate_raw_scan_ledger(ledger_file)
    assert validation.complete is True
    assert validation.cumulative_raw_open_attempts == 0
    assert validation.cumulative_raw_bytes_read == 0


def test_missing_and_bad_sources_are_zero_filled_without_silent_row_drops(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.bin"
    malformed_file = tmp_path / "malformed.bin"
    malformed_payload = b"not-a-pe-but-synthetic"
    malformed_file.write_bytes(malformed_payload)
    missing_payload = b"declared-but-absent"
    plan = RawScanPlan.from_entries(
        (
            _entry(missing_file, missing_payload, ordinal=0),
            _entry(malformed_file, malformed_payload, ordinal=1),
        )
    )
    ledger_file = tmp_path / "missing-and-bad.jsonl"
    with RawScanLedger.create(ledger_file) as ledger:
        outcome = _worker(records=2).scan(
            plan,
            expected_raw_scope_commitment_sha256=plan.raw_scope_commitment_sha256,
            ledger=ledger,
        )

    assert [row.result for row in outcome.rows] == ["source_open_failure", "pe_parse_failure"]
    for row in outcome.rows:
        _assert_zero_missing_row(row)
    validation = validate_raw_scan_ledger(ledger_file)
    assert validation.complete is True
    assert validation.terminal_record_count == 2
    assert validation.cumulative_raw_open_attempts == 2
    assert validation.cumulative_raw_bytes_read == len(malformed_payload)


def test_raw_byte_budget_is_reserved_before_the_second_source_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_file = tmp_path / "first.bin"
    second_file = tmp_path / "second.bin"
    first_payload = b"first"
    second_payload = b"second!"
    first_file.write_bytes(first_payload)
    second_file.write_bytes(second_payload)
    plan = RawScanPlan.from_entries(
        (
            _entry(first_file, first_payload, ordinal=0),
            _entry(second_file, second_payload, ordinal=1),
        )
    )
    ledger_file = tmp_path / "pre-reserved-budget.jsonl"
    opened: list[Path] = []
    original_open = raw_worker._open_source_stream

    def observed_open(candidate: Path):
        opened.append(candidate)
        return original_open(candidate)

    monkeypatch.setattr(raw_worker, "_open_source_stream", observed_open)
    with RawScanLedger.create(ledger_file) as ledger:
        with pytest.raises(RawBudgetExhaustedError) as error:
            _worker(records=2, maximum_raw_bytes=len(first_payload) + len(second_payload)).scan(
                plan,
                expected_raw_scope_commitment_sha256=plan.raw_scope_commitment_sha256,
                ledger=ledger,
            )

    assert error.value.ordinal == 1
    assert opened == [first_file]
    records = [json.loads(line) for line in ledger_file.read_bytes().splitlines()]
    assert [record["event"] for record in records] == [
        "scan_started",
        "raw_open_intent",
        "record_terminal",
        "raw_open_intent",
        "record_terminal",
    ]
    assert records[-1]["result"] == "raw_byte_budget_exhausted"
    assert records[-1]["cumulative_raw_open_attempts"] == 1
    assert records[-1]["cumulative_raw_bytes_read"] == len(first_payload)
    validation = validate_raw_scan_ledger(ledger_file)
    assert validation.complete is False
    assert validation.terminal_record_count == 2
    assert validation.cumulative_raw_open_attempts == 1
    assert validation.cumulative_raw_bytes_read == len(first_payload)


@pytest.mark.parametrize(
    ("payload", "declared_size", "expected_sha256", "expected_result"),
    (
        (b"content-drift", len(b"content-drift"), _sha256("other-content"), "sha256_mismatch"),
        (b"size-drift", len(b"size-drift") - 1, _sha256(b"size-drift"[:-1]), "declared_size_mismatch"),
    ),
)
def test_sha_or_declared_scope_drift_writes_terminal_then_stops(
    tmp_path: Path,
    payload: bytes,
    declared_size: int,
    expected_sha256: str,
    expected_result: str,
) -> None:
    source_file = tmp_path / f"{expected_result}.bin"
    source_file.write_bytes(payload)
    entry = RawPlanEntry(
        ordinal=0,
        source_file=source_file,
        source_audit_sha256=_sha256(f"audit-{expected_result}"),
        declared_size=declared_size,
        expected_sha256=expected_sha256,
    )
    plan = RawScanPlan.from_entries((entry,))
    ledger_file = tmp_path / f"{expected_result}.jsonl"
    with RawScanLedger.create(ledger_file) as ledger:
        with pytest.raises(RawScanFatalError) as error:
            _worker(records=1).scan(
                plan,
                expected_raw_scope_commitment_sha256=plan.raw_scope_commitment_sha256,
                ledger=ledger,
            )

    assert error.value.result == expected_result
    records = [json.loads(line) for line in ledger_file.read_bytes().splitlines()]
    assert [record["event"] for record in records] == ["scan_started", "raw_open_intent", "record_terminal"]
    assert records[-1]["result"] == expected_result
    assert str(source_file) not in ledger_file.read_text(encoding="ascii")
    validation = validate_raw_scan_ledger(ledger_file)
    assert validation.complete is False
    assert validation.terminal_record_count == 1
    assert validation.issues == ("scan_not_completed",)


def test_scope_preflight_rejects_drift_before_any_open_and_plan_has_no_forbidden_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "scope.bin"
    payload = b"scope-synthetic"
    source_file.write_bytes(payload)
    plan = RawScanPlan.from_entries((_entry(source_file, payload),))
    ledger_file = tmp_path / "scope-progress.jsonl"

    def forbidden_open(_: Path):
        raise AssertionError("Scope drift must fail before a source open")

    monkeypatch.setattr(raw_worker, "_open_source_stream", forbidden_open)
    with RawScanLedger.create(ledger_file) as ledger:
        with pytest.raises(RawScopeDriftError):
            _worker(records=1).scan(
                plan,
                expected_raw_scope_commitment_sha256="0" * 64,
                ledger=ledger,
            )

    assert ledger_file.read_bytes() == b""
    assert tuple(field.name for field in fields(RawPlanEntry)) == (
        "ordinal",
        "source_file",
        "source_audit_sha256",
        "declared_size",
        "expected_sha256",
    )
    entry_source = inspect.getsource(RawPlanEntry)
    for forbidden in ("label", "fold", "score"):
        assert forbidden not in entry_source
