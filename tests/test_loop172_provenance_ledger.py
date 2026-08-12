from __future__ import annotations

from datetime import datetime, timezone
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop172.provenance_ledger import ProvenanceLedgerError, audit_as_of_records, parse_record


def _record(**overrides: object):
    payload: dict[str, object] = {
        "evidence_id": "evidence-1",
        "subject_commitment": "a" * 64,
        "source_name": "independent-archive",
        "source_independence_id": "archive-a",
        "published_at_utc": "2024-01-01T00:00:00Z",
        "available_at_utc": "2024-01-01T00:00:00Z",
        "snapshot_captured_at_utc": "2024-01-02T00:00:00Z",
        "verdict": "malicious",
    }
    payload.update(overrides)
    return parse_record(payload)


def test_unknown_is_retained_without_creating_an_accepted_label() -> None:
    audit = audit_as_of_records([_record(verdict="unknown")], score_time_utc=datetime(2024, 2, 1, tzinfo=timezone.utc))

    assert audit.accepted_records == 0
    assert audit.unknown_records == 1
    assert audit.rejected_records == 0


def test_future_availability_is_rejected_not_backfilled() -> None:
    audit = audit_as_of_records([_record(available_at_utc="2024-03-01T00:00:00Z", snapshot_captured_at_utc="2024-03-02T00:00:00Z")], score_time_utc=datetime(2024, 2, 1, tzinfo=timezone.utc))

    assert audit.rejected_records == 1
    assert audit.rejection_reasons == ("not_available_as_of_score_time",)


def test_one_independent_source_cannot_vote_twice_for_one_subject() -> None:
    audit = audit_as_of_records([_record(), _record(evidence_id="evidence-2")], score_time_utc=datetime(2024, 2, 1, tzinfo=timezone.utc))

    assert audit.accepted_records == 1
    assert audit.rejected_records == 1
    assert audit.rejection_reasons == ("duplicate_independent_source_subject",)


def test_schema_forbids_raw_identity_or_current_query_fields() -> None:
    with pytest.raises(ProvenanceLedgerError, match="schema drifted"):
        _record(raw_path="forbidden")
