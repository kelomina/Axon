"""Validate as-of external-label provenance without loading malware samples."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence


class ProvenanceLedgerError(ValueError):
    """Raised when a metadata record cannot support an as-of claim."""


VERDICTS = frozenset({"malicious", "benign", "unknown"})


@dataclass(frozen=True)
class ProvenanceRecord:
    """One opaque-subject, independently timestamped external observation."""

    evidence_id: str
    subject_commitment: str
    source_name: str
    source_independence_id: str
    published_at_utc: datetime
    available_at_utc: datetime
    snapshot_captured_at_utc: datetime
    verdict: str


@dataclass(frozen=True)
class ProvenanceAudit:
    """Aggregate-only audit result; it never exposes subjects or labels."""

    accepted_records: int
    unknown_records: int
    rejected_records: int
    rejection_reasons: tuple[str, ...]


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ProvenanceLedgerError(f"{field} must be a UTC Z timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ProvenanceLedgerError(f"{field} is invalid") from error
    if parsed.tzinfo is None:
        raise ProvenanceLedgerError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _opaque_commitment(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ProvenanceLedgerError("subject_commitment must be a lowercase SHA-256 digest")
    return value


def parse_record(payload: Mapping[str, object]) -> ProvenanceRecord:
    """Parse the frozen, metadata-only record shape and reject extra fields."""
    expected = {
        "evidence_id",
        "subject_commitment",
        "source_name",
        "source_independence_id",
        "published_at_utc",
        "available_at_utc",
        "snapshot_captured_at_utc",
        "verdict",
    }
    if set(payload) != expected:
        raise ProvenanceLedgerError("provenance record schema drifted")
    text_fields = ("evidence_id", "source_name", "source_independence_id", "verdict")
    if any(not isinstance(payload[field], str) or not payload[field].strip() for field in text_fields):
        raise ProvenanceLedgerError("provenance record contains an invalid text field")
    verdict = str(payload["verdict"])
    if verdict not in VERDICTS:
        raise ProvenanceLedgerError("provenance verdict is invalid")
    return ProvenanceRecord(
        evidence_id=str(payload["evidence_id"]),
        subject_commitment=_opaque_commitment(payload["subject_commitment"]),
        source_name=str(payload["source_name"]),
        source_independence_id=str(payload["source_independence_id"]),
        published_at_utc=_parse_utc(payload["published_at_utc"], field="published_at_utc"),
        available_at_utc=_parse_utc(payload["available_at_utc"], field="available_at_utc"),
        snapshot_captured_at_utc=_parse_utc(payload["snapshot_captured_at_utc"], field="snapshot_captured_at_utc"),
        verdict=verdict,
    )


def audit_as_of_records(records: Sequence[ProvenanceRecord], *, score_time_utc: datetime) -> ProvenanceAudit:
    """Reject retroactive evidence and retain unknown rather than inventing a label."""
    if score_time_utc.tzinfo is None:
        raise ProvenanceLedgerError("score_time_utc must be timezone-aware")
    score_time = score_time_utc.astimezone(timezone.utc)
    accepted = unknown = rejected = 0
    reasons: list[str] = []
    seen_evidence: set[str] = set()
    seen_source_subject: set[tuple[str, str]] = set()
    # 同一来源不能针对同一承诺重复计票，且任何事后可用的证据一律拒绝。
    for record in records:
        reason: str | None = None
        source_subject = (record.source_independence_id, record.subject_commitment)
        if record.evidence_id in seen_evidence:
            reason = "duplicate_evidence_id"
        elif source_subject in seen_source_subject:
            reason = "duplicate_independent_source_subject"
        elif record.published_at_utc > record.available_at_utc:
            reason = "publication_after_availability"
        elif record.available_at_utc > score_time:
            reason = "not_available_as_of_score_time"
        elif record.snapshot_captured_at_utc < record.available_at_utc:
            reason = "snapshot_predates_availability"
        if reason is not None:
            rejected += 1
            reasons.append(reason)
            continue
        seen_evidence.add(record.evidence_id)
        seen_source_subject.add(source_subject)
        if record.verdict == "unknown":
            unknown += 1
        else:
            accepted += 1
    return ProvenanceAudit(accepted, unknown, rejected, tuple(sorted(reasons)))
