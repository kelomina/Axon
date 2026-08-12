"""Fail-closed validation for two independent reviews and adjudication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence


class DualBlindReviewError(ValueError):
    """Raised when a review packet cannot support a governance action."""


DECISIONS = frozenset({"label_supported", "actionable_label_conflict", "artifact_invalid", "insufficient"})
FINAL_DECISIONS = frozenset({"confirmed_actionable", "not_actionable", "unresolved"})
VERDICTS = frozenset({"malicious", "benign", "unknown"})


@dataclass(frozen=True)
class ReviewSubmission:
    case_id: str
    packet_id: str
    reviewer_key_id: str
    reviewer_independence_group: str
    assignment_nonce: str
    decision: str
    proposed_label: str | None


@dataclass(frozen=True)
class IssuedAssignment:
    case_id: str
    packet_id: str
    reviewer_key_id: str
    reviewer_independence_group: str
    assignment_nonce: str
    expires_at_utc: datetime


@dataclass(frozen=True)
class EvidenceRecord:
    case_id: str
    source_registry_id: str
    source_independence_id: str
    available_at_utc: datetime
    snapshot_captured_at_utc: datetime
    verdict: str


@dataclass(frozen=True)
class Adjudication:
    case_id: str
    adjudicator_key_id: str
    adjudicator_independence_group: str
    final_decision: str
    original_label: str
    proposed_label: str
    evidence_source_ids: tuple[str, ...]


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DualBlindReviewError(f"{field} is invalid")
    return value.strip()


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _utc(value: object, field: str) -> datetime:
    text = _text(value, field)
    if not text.endswith("Z"):
        raise DualBlindReviewError(f"{field} must be a UTC Z timestamp")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError as error:
        raise DualBlindReviewError(f"{field} is invalid") from error


def _optional_verdict(value: object, field: str) -> str | None:
    if value is None or value == "":
        return None
    verdict = _text(value, field)
    if verdict not in VERDICTS - {"unknown"}:
        raise DualBlindReviewError(f"{field} is invalid")
    return verdict


def parse_submission(payload: Mapping[str, object]) -> ReviewSubmission:
    required = {"case_id", "packet_id", "reviewer_key_id", "reviewer_independence_group", "assignment_nonce", "decision", "proposed_label"}
    if set(payload) != required:
        raise DualBlindReviewError("review submission schema drifted")
    decision = _text(payload["decision"], "decision")
    if decision not in DECISIONS:
        raise DualBlindReviewError("review decision is invalid")
    proposed_label = _optional_verdict(payload["proposed_label"], "proposed_label")
    if (decision == "actionable_label_conflict") != (proposed_label is not None):
        raise DualBlindReviewError("review decision and proposed label disagree")
    return ReviewSubmission(*(_text(payload[field], field) for field in ("case_id", "packet_id", "reviewer_key_id", "reviewer_independence_group", "assignment_nonce")), decision, proposed_label)


def parse_evidence(payload: Mapping[str, object]) -> EvidenceRecord:
    required = {"case_id", "source_registry_id", "source_independence_id", "available_at_utc", "snapshot_captured_at_utc", "verdict"}
    if set(payload) != required:
        raise DualBlindReviewError("evidence record schema drifted")
    verdict = _text(payload["verdict"], "verdict")
    if verdict not in VERDICTS:
        raise DualBlindReviewError("evidence verdict is invalid")
    return EvidenceRecord(_text(payload["case_id"], "case_id"), _text(payload["source_registry_id"], "source_registry_id"), _text(payload["source_independence_id"], "source_independence_id"), _utc(payload["available_at_utc"], "available_at_utc"), _utc(payload["snapshot_captured_at_utc"], "snapshot_captured_at_utc"), verdict)


def parse_adjudication(payload: Mapping[str, object]) -> Adjudication:
    required = {"case_id", "adjudicator_key_id", "adjudicator_independence_group", "final_decision", "original_label", "proposed_label", "evidence_source_ids"}
    if set(payload) != required or not isinstance(payload.get("evidence_source_ids"), list):
        raise DualBlindReviewError("adjudication schema drifted")
    final_decision = _text(payload["final_decision"], "final_decision")
    if final_decision not in FINAL_DECISIONS:
        raise DualBlindReviewError("adjudication decision is invalid")
    source_ids = tuple(_text(value, "evidence_source_id") for value in payload["evidence_source_ids"])
    if len(set(source_ids)) != len(source_ids):
        raise DualBlindReviewError("adjudication evidence source IDs are duplicated")
    original_label = _optional_verdict(payload["original_label"], "original_label")
    proposed_label = _optional_verdict(payload["proposed_label"], "proposed_label")
    if final_decision == "confirmed_actionable" and (original_label is None or proposed_label is None or original_label == proposed_label):
        raise DualBlindReviewError("confirmed adjudication must replace an original label")
    return Adjudication(_text(payload["case_id"], "case_id"), _text(payload["adjudicator_key_id"], "adjudicator_key_id"), _text(payload["adjudicator_independence_group"], "adjudicator_independence_group"), final_decision, original_label or "", proposed_label or "", source_ids)


def validate_case(
    *,
    packet_id: str,
    case_id: str,
    submissions: Sequence[ReviewSubmission],
    issued_assignments: Sequence[IssuedAssignment],
    consumed_assignment_nonces: frozenset[str],
    evidence: Sequence[EvidenceRecord],
    adjudication: Adjudication,
    score_cutoff_utc: datetime,
) -> bool:
    """Return true only for independently reviewed, as-of-confirmed actionable cases."""
    if (
        adjudication.final_decision not in FINAL_DECISIONS
        or adjudication.original_label not in VERDICTS - {"unknown"}
        or adjudication.proposed_label not in VERDICTS - {"unknown"}
        or not all(_has_text(value) for value in (adjudication.case_id, adjudication.adjudicator_key_id, adjudication.adjudicator_independence_group))
    ):
        return False
    case_submissions = [row for row in submissions if row.case_id == case_id]
    if len(case_submissions) != 2 or adjudication.case_id != case_id:
        return False
    if any(row.packet_id != packet_id for row in case_submissions):
        return False
    keys = {row.reviewer_key_id for row in case_submissions}
    groups = {row.reviewer_independence_group for row in case_submissions}
    nonces = {row.assignment_nonce for row in case_submissions}
    if len(keys) != 2 or len(groups) != 2 or len(nonces) != 2:
        return False
    if any(
        row.decision != "actionable_label_conflict"
        or row.proposed_label not in VERDICTS - {"unknown"}
        or not all(_has_text(value) for value in (row.case_id, row.packet_id, row.reviewer_key_id, row.reviewer_independence_group, row.assignment_nonce))
        for row in case_submissions
    ):
        return False
    proposed_labels = {row.proposed_label for row in case_submissions}
    if len(proposed_labels) != 1 or adjudication.proposed_label not in proposed_labels or adjudication.original_label == adjudication.proposed_label:
        return False
    if adjudication.adjudicator_key_id in keys or adjudication.adjudicator_independence_group in groups:
        return False
    if adjudication.final_decision != "confirmed_actionable":
        return False
    if nonces & consumed_assignment_nonces:
        return False
    issued_by_nonce = {assignment.assignment_nonce: assignment for assignment in issued_assignments}
    if len(issued_by_nonce) != len(issued_assignments) or set(issued_by_nonce) != nonces:
        return False
    for submission in case_submissions:
        assignment = issued_by_nonce[submission.assignment_nonce]
        if (
            assignment.case_id != case_id
            or assignment.packet_id != packet_id
            or assignment.reviewer_key_id != submission.reviewer_key_id
            or assignment.reviewer_independence_group != submission.reviewer_independence_group
            or assignment.expires_at_utc.tzinfo is None
            or assignment.expires_at_utc < score_cutoff_utc
        ):
            return False
    selected = [row for row in evidence if row.case_id == case_id and row.source_registry_id in adjudication.evidence_source_ids]
    # 所有可行动证据必须在评分时点前已可用，未知或冲突来源一律不能确认行动。
    selected_ids = [row.source_registry_id for row in selected]
    if (
        not selected
        or len(selected) != len(adjudication.evidence_source_ids)
        or set(selected_ids) != set(adjudication.evidence_source_ids)
        or len(set(selected_ids)) != len(selected_ids)
        or any(
            row.verdict != adjudication.proposed_label
            or row.verdict not in VERDICTS
            or not all(_has_text(value) for value in (row.case_id, row.source_registry_id, row.source_independence_id))
            or row.available_at_utc.tzinfo is None
            or row.snapshot_captured_at_utc.tzinfo is None
            or row.available_at_utc > row.snapshot_captured_at_utc
            or row.available_at_utc > score_cutoff_utc
            or row.snapshot_captured_at_utc > score_cutoff_utc
            for row in selected
        )
    ):
        return False
    if len({row.source_independence_id for row in selected}) != len(selected):
        return False
    if len({row.verdict for row in selected}) != 1:
        return False
    return True
