from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import src.loop167_phase_b.progress_ledger as ledger_module
from src.loop167_phase_b.progress_ledger import (
    EXPECTED_FIT_UNIT_COUNT,
    FitLedger,
    FitLedgerError,
    ProgressLedgerError,
    RawScanLedger,
    validate_fit_ledger,
    validate_raw_scan_ledger,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_complete_raw_ledger(path: Path) -> tuple[str, str]:
    with RawScanLedger.create(path) as ledger:
        ledger.scan_started(
            expected_record_count=2,
            maximum_raw_open_attempts=2,
            maximum_raw_bytes_read=64,
            raw_scope_commitment_sha256=_sha("scope"),
        )
        ledger.raw_open_intent(ordinal=0, source_audit_sha256=_sha("source-0"))
        ledger.record_terminal(
            ordinal=0,
            source_audit_sha256=_sha("source-0"),
            result="available",
            cumulative_raw_open_attempts=1,
            cumulative_raw_bytes_read=19,
            feature_row_commitment_sha256=_sha("feature-row-0"),
        )
        ledger.raw_open_intent(ordinal=1, source_audit_sha256=_sha("source-1"))
        ledger.record_terminal(
            ordinal=1,
            source_audit_sha256=_sha("source-1"),
            result="oversize_declared",
            cumulative_raw_open_attempts=1,
            cumulative_raw_bytes_read=19,
            feature_row_commitment_sha256=_sha("feature-row-1"),
        )
        feature_rows_commitment_sha256 = ledger.feature_rows_commitment_sha256
        ledger.scan_completed(
            record_count=2,
            cumulative_raw_open_attempts=1,
            cumulative_raw_bytes_read=19,
            feature_rows_commitment_sha256=feature_rows_commitment_sha256,
        )
        assert ledger.final_record_sha256 is not None
        return ledger.final_record_sha256, feature_rows_commitment_sha256


def _write_complete_fit_ledger(path: Path) -> str:
    with FitLedger.create(path) as ledger:
        ledger.fit_started(
            fit_protocol_commitment_sha256=_sha("fit-protocol"),
            feature_rows_commitment_sha256=_sha("feature-rows"),
            raw_ledger_final_record_sha256=_sha("raw-ledger"),
        )
        for arm_ordinal in range(5):
            for replay_ordinal in range(3):
                for fold_ordinal in range(5):
                    ledger.fit_unit_completed(
                        arm_ordinal=arm_ordinal,
                        replay_ordinal=replay_ordinal,
                        fold_ordinal=fold_ordinal,
                    )
        ledger.fit_completed()
        assert ledger.final_record_sha256 is not None
        return ledger.final_record_sha256


def test_raw_ledger_uses_exclusive_canonical_fsynced_sha_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "raw-progress.jsonl"
    fsync_calls: list[int] = []
    real_fsync = ledger_module.os.fsync

    def observed_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(ledger_module.os, "fsync", observed_fsync)
    final_record_sha256, feature_rows_commitment_sha256 = _write_complete_raw_ledger(path)

    raw_lines = path.read_bytes().splitlines(keepends=True)
    records = [json.loads(raw_line) for raw_line in raw_lines]
    assert len(fsync_calls) == len(records) == 6
    assert records[0]["previous_record_sha256"] == "0" * 64
    for sequence, record in enumerate(records):
        canonical = json.dumps(
            record,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        assert raw_lines[sequence] == canonical + b"\n"
        assert record["sequence"] == sequence
        if sequence:
            assert record["previous_record_sha256"] == records[sequence - 1]["record_sha256"]

    validation = validate_raw_scan_ledger(path)
    assert validation.complete is True
    assert validation.status == "complete"
    assert validation.line_count == 6
    assert validation.terminal_record_count == 2
    assert validation.final_record_sha256 == final_record_sha256
    assert validation.feature_rows_commitment_sha256 == feature_rows_commitment_sha256
    with pytest.raises(FileExistsError):
        RawScanLedger.create(path)


def test_raw_ledger_rejects_sensitive_or_unbounded_fields_and_budget_overruns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw-fields.jsonl"
    with RawScanLedger.create(path) as ledger:
        ledger.scan_started(
            expected_record_count=1,
            maximum_raw_open_attempts=1,
            maximum_raw_bytes_read=10,
            raw_scope_commitment_sha256=_sha("scope"),
        )
        with pytest.raises(ProgressLedgerError) as extra_field:
            ledger.append_event(
                "raw_open_intent",
                ordinal=0,
                source_audit_sha256=_sha("source"),
                path="C:/private/source.exe",
            )
        assert extra_field.value.code == "audit_fields_invalid"
        ledger.raw_open_intent(ordinal=0, source_audit_sha256=_sha("source"))
        with pytest.raises(ProgressLedgerError) as budget_error:
            ledger.record_terminal(
                ordinal=0,
                source_audit_sha256=_sha("source"),
                result="available",
                cumulative_raw_open_attempts=1,
                cumulative_raw_bytes_read=11,
                feature_row_commitment_sha256=_sha("feature-row"),
            )
        assert budget_error.value.code == "raw_budget_exceeded"
        ledger.record_terminal(
            ordinal=0,
            source_audit_sha256=_sha("source"),
            result="available",
            cumulative_raw_open_attempts=1,
            cumulative_raw_bytes_read=10,
            feature_row_commitment_sha256=_sha("feature-row"),
        )
        ledger.scan_completed(
            record_count=1,
            cumulative_raw_open_attempts=1,
            cumulative_raw_bytes_read=10,
            feature_rows_commitment_sha256=ledger.feature_rows_commitment_sha256,
        )

    ledger_text = path.read_text(encoding="ascii")
    assert "private" not in ledger_text
    assert "source.exe" not in ledger_text
    assert validate_raw_scan_ledger(path).complete is True


def test_raw_ledger_records_explicit_byte_budget_exhaustion_without_a_source_open(tmp_path: Path) -> None:
    path = tmp_path / "raw-byte-budget-exhausted.jsonl"
    with RawScanLedger.create(path) as ledger:
        ledger.scan_started(
            expected_record_count=1,
            maximum_raw_open_attempts=1,
            maximum_raw_bytes_read=0,
            raw_scope_commitment_sha256=_sha("scope"),
        )
        ledger.raw_open_intent(ordinal=0, source_audit_sha256=_sha("source"))
        ledger.record_terminal(
            ordinal=0,
            source_audit_sha256=_sha("source"),
            result="raw_byte_budget_exhausted",
            cumulative_raw_open_attempts=0,
            cumulative_raw_bytes_read=0,
            feature_row_commitment_sha256=_sha("feature-row"),
        )
        ledger.scan_completed(
            record_count=1,
            cumulative_raw_open_attempts=0,
            cumulative_raw_bytes_read=0,
            feature_rows_commitment_sha256=ledger.feature_rows_commitment_sha256,
        )

    validation = validate_raw_scan_ledger(path)
    assert validation.complete is True
    assert validation.cumulative_raw_open_attempts == 0
    assert validation.cumulative_raw_bytes_read == 0


def test_raw_validator_detects_tampering_and_torn_tail_without_repair(tmp_path: Path) -> None:
    tampered = tmp_path / "raw-tampered.jsonl"
    _write_complete_raw_ledger(tampered)
    lines = tampered.read_bytes().splitlines(keepends=True)
    record = json.loads(lines[2])
    record["cumulative_raw_bytes_read"] = 20
    lines[2] = json.dumps(record, separators=(",", ":"), sort_keys=True).encode("ascii") + b"\n"
    tampered.write_bytes(b"".join(lines))

    tampered_result = validate_raw_scan_ledger(tampered)
    assert tampered_result.status == "tampered"
    assert tampered_result.issues == ("record_sha256_mismatch",)

    torn = tmp_path / "raw-torn.jsonl"
    _write_complete_raw_ledger(torn)
    torn.write_bytes(torn.read_bytes()[:-17])
    before = torn.read_bytes()
    torn_result = validate_raw_scan_ledger(torn)
    assert torn_result.status == "torn_tail"
    assert torn_result.issues == ("torn_tail",)
    assert torn.read_bytes() == before


def test_fit_ledger_requires_every_fixed_arm_replay_fold_unit_once(tmp_path: Path) -> None:
    path = tmp_path / "fit-progress.jsonl"
    final_record_sha256 = _write_complete_fit_ledger(path)

    validation = validate_fit_ledger(path)
    assert validation.complete is True
    assert validation.status == "complete"
    assert validation.completed_unit_count == EXPECTED_FIT_UNIT_COUNT == 75
    assert validation.missing_units == ()
    assert validation.line_count == EXPECTED_FIT_UNIT_COUNT + 2
    assert validation.final_record_sha256 == final_record_sha256


def test_fit_ledger_rejects_duplicates_missing_coverage_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "fit-incomplete.jsonl"
    with FitLedger.create(path) as ledger:
        ledger.fit_started(
            fit_protocol_commitment_sha256=_sha("fit-protocol"),
            feature_rows_commitment_sha256=_sha("feature-rows"),
            raw_ledger_final_record_sha256=_sha("raw-ledger"),
        )
        ledger.fit_unit_completed(arm_ordinal=0, replay_ordinal=0, fold_ordinal=0)
        with pytest.raises(FitLedgerError) as duplicate_error:
            ledger.fit_unit_completed(arm_ordinal=0, replay_ordinal=0, fold_ordinal=0)
        assert duplicate_error.value.code == "fit_unit_repeated"
        with pytest.raises(FitLedgerError) as coverage_error:
            ledger.fit_completed()
        assert coverage_error.value.code == "fit_coverage_incomplete"

    validation = validate_fit_ledger(path)
    assert validation.complete is False
    assert validation.status == "incomplete"
    assert validation.issues == ("fit_not_completed", "fit_units_missing")
    assert validation.completed_unit_count == 1
    assert len(validation.missing_units) == 74
    with pytest.raises(FileExistsError):
        FitLedger.create(path)


def test_fit_validator_detects_tampering_and_torn_tail(tmp_path: Path) -> None:
    tampered = tmp_path / "fit-tampered.jsonl"
    _write_complete_fit_ledger(tampered)
    lines = tampered.read_bytes().splitlines(keepends=True)
    record = json.loads(lines[1])
    record["fold_ordinal"] = 4
    lines[1] = json.dumps(record, separators=(",", ":"), sort_keys=True).encode("ascii") + b"\n"
    tampered.write_bytes(b"".join(lines))

    tampered_result = validate_fit_ledger(tampered)
    assert tampered_result.status == "tampered"
    assert tampered_result.issues == ("record_sha256_mismatch",)

    torn = tmp_path / "fit-torn.jsonl"
    _write_complete_fit_ledger(torn)
    torn.write_bytes(torn.read_bytes()[:-13])
    assert validate_fit_ledger(torn).status == "torn_tail"
