from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop174.dual_blind_review import (  # noqa: E402
    Adjudication,
    EvidenceRecord,
    IssuedAssignment,
    ReviewSubmission,
    validate_case,
)


def _submission(key: str, group: str, nonce: str) -> ReviewSubmission:
    return ReviewSubmission("case-1", "packet-1", key, group, nonce, "actionable_label_conflict", "malicious")


def _evidence(source: str, group: str, verdict: str = "malicious") -> EvidenceRecord:
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return EvidenceRecord("case-1", source, group, timestamp, timestamp, verdict)


def _assignment(key: str, group: str, nonce: str) -> IssuedAssignment:
    return IssuedAssignment("case-1", "packet-1", key, group, nonce, datetime(2024, 2, 2, tzinfo=timezone.utc))


def _validate(*, submissions: list[ReviewSubmission], evidence: list[EvidenceRecord], adjudication: Adjudication, issued_assignments: list[IssuedAssignment] | None = None, consumed: frozenset[str] = frozenset()) -> bool:
    assignments = issued_assignments or [_assignment(row.reviewer_key_id, row.reviewer_independence_group, row.assignment_nonce) for row in submissions]
    return validate_case(packet_id="packet-1", case_id="case-1", submissions=submissions, issued_assignments=assignments, consumed_assignment_nonces=consumed, evidence=evidence, adjudication=adjudication, score_cutoff_utc=datetime(2024, 2, 1, tzinfo=timezone.utc))


def test_confirmed_case_requires_two_independent_reviewers_and_asof_evidence() -> None:
    adjudication = Adjudication("case-1", "adjudicator", "group-c", "confirmed_actionable", "benign", "malicious", ("source-a", "source-b"))

    assert _validate(submissions=[_submission("reviewer-a", "group-a", "nonce-a"), _submission("reviewer-b", "group-b", "nonce-b")], evidence=[_evidence("source-a", "source-a"), _evidence("source-b", "source-b")], adjudication=adjudication)


def test_single_group_or_unknown_evidence_fails_closed() -> None:
    adjudication = Adjudication("case-1", "adjudicator", "group-c", "confirmed_actionable", "benign", "malicious", ("source-a",))

    assert not _validate(submissions=[_submission("reviewer-a", "group-a", "nonce-a"), _submission("reviewer-b", "group-a", "nonce-b")], evidence=[_evidence("source-a", "source-a")], adjudication=adjudication)
    assert not _validate(submissions=[_submission("reviewer-a", "group-a", "nonce-a"), _submission("reviewer-b", "group-b", "nonce-b")], evidence=[_evidence("source-a", "source-a", "unknown")], adjudication=adjudication)


def test_non_actionable_reviewer_decision_cannot_be_adjudicated_as_actionable() -> None:
    adjudication = Adjudication("case-1", "adjudicator", "group-c", "confirmed_actionable", "benign", "malicious", ("source-a",))
    submissions = [
        ReviewSubmission("case-1", "packet-1", "reviewer-a", "group-a", "nonce-a", "label_supported", None),
        _submission("reviewer-b", "group-b", "nonce-b"),
    ]

    assert not _validate(submissions=submissions, evidence=[_evidence("source-a", "source-a")], adjudication=adjudication)


def test_evidence_assignment_and_label_binding_fail_closed() -> None:
    adjudication = Adjudication("case-1", "adjudicator", "group-c", "confirmed_actionable", "benign", "malicious", ("source-a",))
    submissions = [_submission("reviewer-a", "group-a", "nonce-a"), _submission("reviewer-b", "group-b", "nonce-b")]

    assert not _validate(submissions=submissions, evidence=[_evidence("source-a", "source-a", "benign")], adjudication=adjudication)
    assert not _validate(submissions=submissions, evidence=[_evidence("source-a", "source-a")], adjudication=adjudication, issued_assignments=[_assignment("reviewer-a", "group-a", "nonce-a")])
    assert not _validate(submissions=submissions, evidence=[_evidence("source-a", "source-a")], adjudication=adjudication, consumed=frozenset({"nonce-a"}))


def test_duplicate_or_impossible_evidence_fails_closed() -> None:
    adjudication = Adjudication("case-1", "adjudicator", "group-c", "confirmed_actionable", "benign", "malicious", ("source-a",))
    submissions = [_submission("reviewer-a", "group-a", "nonce-a"), _submission("reviewer-b", "group-b", "nonce-b")]
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    impossible = EvidenceRecord("case-1", "source-a", "independent-a", timestamp, datetime(2023, 12, 1, tzinfo=timezone.utc), "malicious")

    assert not _validate(submissions=submissions, evidence=[_evidence("source-a", "independent-a"), _evidence("source-a", "independent-b")], adjudication=adjudication)
    assert not _validate(submissions=submissions, evidence=[impossible], adjudication=adjudication)


def test_directly_constructed_invalid_verdict_fails_closed() -> None:
    adjudication = Adjudication("case-1", "adjudicator", "group-c", "confirmed_actionable", "benign", "fabricated", ("source-a",))
    submissions = [
        ReviewSubmission("case-1", "packet-1", "reviewer-a", "group-a", "nonce-a", "actionable_label_conflict", "fabricated"),
        ReviewSubmission("case-1", "packet-1", "reviewer-b", "group-b", "nonce-b", "actionable_label_conflict", "fabricated"),
    ]
    evidence = [_evidence("source-a", "source-a", "fabricated")]

    assert not _validate(submissions=submissions, evidence=evidence, adjudication=adjudication)
