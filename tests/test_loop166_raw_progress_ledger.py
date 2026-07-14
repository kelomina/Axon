from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import loop166.raw_progress_ledger as ledger_module  # noqa: E402
from loop166.raw_progress_ledger import (  # noqa: E402
    RawProgressLedger,
    RawProgressLedgerError,
    validate_raw_progress_ledger,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_complete_ledger(path: Path) -> None:
    with RawProgressLedger.create(path) as ledger:
        ledger.scan_started(expected_record_count=2)
        ledger.raw_open_intent(ordinal=0, row_index=7, source_sha256=_sha("source-0"))
        ledger.record_terminal(
            ordinal=0,
            row_index=7,
            source_sha256=_sha("source-0"),
            result="available",
            cumulative_raw_open_attempts=1,
            cumulative_raw_open_successes=1,
            cumulative_raw_bytes_read=19,
        )
        ledger.raw_open_intent(ordinal=1, row_index=11, source_sha256=_sha("source-1"))
        ledger.record_terminal(
            ordinal=1,
            row_index=11,
            source_sha256=_sha("source-1"),
            result="source_unavailable",
            cumulative_raw_open_attempts=1,
            cumulative_raw_open_successes=1,
            cumulative_raw_bytes_read=19,
        )
        ledger.scan_completed(
            record_count=2,
            cumulative_raw_open_attempts=1,
            cumulative_raw_open_successes=1,
            cumulative_raw_bytes_read=19,
            corpus_commitment_sha256=_sha("corpus"),
        )


def test_exclusive_canonical_hash_chain_and_fsync_per_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "progress.jsonl"
    fsync_calls: list[int] = []
    real_fsync = ledger_module.os.fsync

    def observed_fsync(descriptor: int) -> None:
        fsync_calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(ledger_module.os, "fsync", observed_fsync)
    _write_complete_ledger(path)

    raw_lines = path.read_bytes().splitlines(keepends=True)
    records = [json.loads(line) for line in raw_lines]
    assert len(fsync_calls) == len(records) == 6
    assert all(line.endswith(b"\n") for line in raw_lines)
    assert records[0]["previous_record_sha256"] == "0" * 64
    for sequence, record in enumerate(records):
        assert record["sequence"] == sequence
        canonical = json.dumps(
            record,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        assert raw_lines[sequence] == canonical + b"\n"
        if sequence:
            assert record["previous_record_sha256"] == records[sequence - 1][
                "record_sha256"
            ]

    result = validate_raw_progress_ledger(path)
    assert result.complete is True
    assert result.status == "complete"
    assert result.line_count == 6
    assert result.terminal_record_count == 2
    assert result.final_record_sha256 == records[-1]["record_sha256"]
    assert result.corpus_commitment_sha256 == _sha("corpus")
    with pytest.raises(FileExistsError):
        RawProgressLedger.create(path)


@pytest.mark.parametrize(
    "sensitive_field,value",
    [
        ("path", "C:/private/source.exe"),
        ("raw_bytes", "deadbeef"),
        ("window", [1, 2]),
        ("token_ids", [3, 4]),
    ],
)
def test_sensitive_or_non_audit_fields_are_rejected(
    tmp_path: Path,
    sensitive_field: str,
    value: object,
) -> None:
    path = tmp_path / f"{sensitive_field}.jsonl"
    with RawProgressLedger.create(path) as ledger:
        ledger.scan_started(expected_record_count=1)
        with pytest.raises(RawProgressLedgerError) as exc_info:
            ledger.append_event(
                "raw_open_intent",
                ordinal=0,
                row_index=3,
                source_sha256=_sha("source"),
                **{sensitive_field: value},
            )
        assert exc_info.value.code == "audit_fields_invalid"

    raw = path.read_text(encoding="ascii")
    assert "private" not in raw
    assert "deadbeef" not in raw
    assert len(raw.splitlines()) == 1


def test_event_state_machine_rejects_missing_terminal_and_identity_drift(tmp_path: Path) -> None:
    path = tmp_path / "state.jsonl"
    with RawProgressLedger.create(path) as ledger:
        ledger.scan_started(expected_record_count=1)
        ledger.raw_open_intent(ordinal=0, row_index=5, source_sha256=_sha("source"))
        with pytest.raises(RawProgressLedgerError) as exc_info:
            ledger.scan_completed(
                record_count=0,
                cumulative_raw_open_attempts=0,
                cumulative_raw_open_successes=0,
                cumulative_raw_bytes_read=0,
                corpus_commitment_sha256=_sha("corpus"),
            )
        assert exc_info.value.code == "record_terminal_missing"
        with pytest.raises(RawProgressLedgerError) as exc_info:
            ledger.record_terminal(
                ordinal=0,
                row_index=6,
                source_sha256=_sha("source"),
                result="available",
                cumulative_raw_open_attempts=1,
                cumulative_raw_open_successes=1,
                cumulative_raw_bytes_read=1,
            )
        assert exc_info.value.code == "record_identity_mismatch"


def test_validator_detects_content_tampering(tmp_path: Path) -> None:
    path = tmp_path / "tampered.jsonl"
    _write_complete_ledger(path)
    lines = path.read_bytes().splitlines(keepends=True)
    record = json.loads(lines[2])
    record["cumulative_raw_bytes_read"] = 20
    lines[2] = json.dumps(record, separators=(",", ":"), sort_keys=True).encode("ascii") + b"\n"
    path.write_bytes(b"".join(lines))

    result = validate_raw_progress_ledger(path)
    assert result.complete is False
    assert result.status == "tampered"
    assert result.issues == ("record_sha256_mismatch",)


@pytest.mark.parametrize("mutation", ["reorder", "delete"])
def test_validator_detects_reorder_and_delete(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / f"{mutation}.jsonl"
    _write_complete_ledger(path)
    lines = path.read_bytes().splitlines(keepends=True)
    if mutation == "reorder":
        lines[1], lines[2] = lines[2], lines[1]
    else:
        del lines[1]
    path.write_bytes(b"".join(lines))

    result = validate_raw_progress_ledger(path)
    assert result.complete is False
    assert result.status == "reordered_or_deleted"
    assert result.issues == ("record_sequence_mismatch",)


def test_validator_detects_torn_tail_without_modifying_it(tmp_path: Path) -> None:
    path = tmp_path / "torn.jsonl"
    _write_complete_ledger(path)
    path.write_bytes(path.read_bytes()[:-17])
    before = path.read_bytes()

    result = validate_raw_progress_ledger(path)

    assert result.complete is False
    assert result.status == "torn_tail"
    assert result.issues == ("torn_tail",)
    assert path.read_bytes() == before


def test_incomplete_prefix_is_read_only_and_cannot_be_resumed(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.jsonl"
    with RawProgressLedger.create(path) as ledger:
        ledger.scan_started(expected_record_count=1)
        ledger.raw_open_intent(ordinal=0, row_index=9, source_sha256=_sha("source"))
    before = path.read_bytes()

    result = validate_raw_progress_ledger(path)

    assert result.complete is False
    assert result.status == "incomplete"
    assert result.issues == ("scan_not_completed", "record_terminal_missing")
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        RawProgressLedger.create(path)
    assert path.read_bytes() == before
