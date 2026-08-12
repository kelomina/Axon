"""Prepare a direction-blind Loop174 issuance without admitting any review."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .blind_packets import (
    OPAQUE_ID,
    BlindCase,
    BlindPacketError,
    Reviewer,
    _validate_context,
    _validate_reviewers,
)

CASE_MANIFEST_SCHEMA = "axon.loop174.sanitized_case_manifest.v1"
ISSUER_PREFLIGHT_SCHEMA = "axon.loop174.issuer_preflight.v1"


class IssuerPreflightError(ValueError):
    """Raised when a case manifest cannot safely enter the blind-review route."""


@dataclass(frozen=True)
class SanitizedCase:
    """A direction-blind case and its issuer-only provenance commitment."""

    case_id: str
    subject_commitment: str
    review_lane: str
    context: Mapping[str, object]


@dataclass(frozen=True)
class IssuerPreflight:
    """The immutable boundary a later packet issuer must bind before export."""

    packet_id: str
    score_cutoff_utc: datetime
    source_registry_snapshot_commitment: str
    cases: tuple[SanitizedCase, ...]
    reviewers: tuple[Reviewer, ...]
    manifest_sha256: str


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IssuerPreflightError(f"{field} is invalid")
    return value.strip()


def _opaque(value: object, field: str) -> str:
    text = _text(value, field)
    if not OPAQUE_ID.fullmatch(text):
        raise IssuerPreflightError(f"{field} must be opaque")
    return text


def _commitment(value: object, field: str) -> str:
    text = _text(value, field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise IssuerPreflightError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _utc(value: object, field: str) -> datetime:
    text = _text(value, field)
    if not text.endswith("Z"):
        raise IssuerPreflightError(f"{field} must be a UTC Z timestamp")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError as error:
        raise IssuerPreflightError(f"{field} is invalid") from error
    return parsed


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def parse_sanitized_case_manifest(payload: Mapping[str, object]) -> tuple[str, datetime, str, tuple[SanitizedCase, ...]]:
    """Reject every field outside the issuer-only, direction-blind manifest."""
    expected = {"schema_version", "packet_id", "score_cutoff_utc", "source_registry_snapshot_commitment", "cases"}
    if set(payload) != expected or payload.get("schema_version") != CASE_MANIFEST_SCHEMA:
        raise IssuerPreflightError("sanitized case manifest schema drifted")
    if not isinstance(payload["cases"], list) or not payload["cases"]:
        raise IssuerPreflightError("sanitized case manifest requires cases")
    packet_id = _opaque(payload["packet_id"], "packet_id")
    cutoff = _utc(payload["score_cutoff_utc"], "score_cutoff_utc")
    source_snapshot = _commitment(payload["source_registry_snapshot_commitment"], "source_registry_snapshot_commitment")
    cases: list[SanitizedCase] = []
    seen_case_ids: set[str] = set()
    seen_subjects: set[str] = set()
    for row in payload["cases"]:
        if not isinstance(row, Mapping) or set(row) != {"case_id", "subject_commitment", "review_lane", "context"}:
            raise IssuerPreflightError("sanitized case schema drifted")
        case = SanitizedCase(
            _opaque(row["case_id"], "case_id"),
            _commitment(row["subject_commitment"], "subject_commitment"),
            _text(row["review_lane"], "review_lane"),
            row["context"],
        )
        try:
            _validate_context(BlindCase(case.case_id, case.review_lane, case.context))
        except BlindPacketError as error:
            raise IssuerPreflightError(str(error)) from error
        if case.case_id in seen_case_ids or case.subject_commitment in seen_subjects:
            raise IssuerPreflightError("case and subject commitments must be one-to-one")
        seen_case_ids.add(case.case_id)
        seen_subjects.add(case.subject_commitment)
        cases.append(case)
    return packet_id, cutoff, source_snapshot, tuple(cases)


def build_issuer_preflight(payload: Mapping[str, object], reviewers: Sequence[Reviewer]) -> IssuerPreflight:
    """Bind future issuance to a new sanitized manifest and independent roster."""
    packet_id, cutoff, source_snapshot, cases = parse_sanitized_case_manifest(payload)
    try:
        normalized_reviewers = _validate_reviewers(reviewers)
    except BlindPacketError as error:
        raise IssuerPreflightError(str(error)) from error
    manifest_sha256 = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return IssuerPreflight(packet_id, cutoff, source_snapshot, cases, normalized_reviewers, manifest_sha256)


def _receipt_payload(preflight: IssuerPreflight, *, issued_at_utc: datetime) -> dict[str, object]:
    if issued_at_utc.tzinfo is None or issued_at_utc.utcoffset() != timedelta(0):
        raise IssuerPreflightError("issued_at_utc must be UTC-aware")
    return {
        "schema_version": ISSUER_PREFLIGHT_SCHEMA,
        "packet_id": preflight.packet_id,
        "issued_at_utc": issued_at_utc.isoformat().replace("+00:00", "Z"),
        "score_cutoff_utc": preflight.score_cutoff_utc.isoformat().replace("+00:00", "Z"),
        "manifest_sha256": preflight.manifest_sha256,
        "source_registry_snapshot_commitment": preflight.source_registry_snapshot_commitment,
        "case_bindings": [
            {"case_id": case.case_id, "subject_commitment": case.subject_commitment}
            for case in preflight.cases
        ],
        "reviewer_roster": [
            {"reviewer_key_id": reviewer.key_id, "reviewer_independence_group": reviewer.independence_group}
            for reviewer in preflight.reviewers
        ],
        "hard_boundaries": {
            "accesses_samples": False,
            "accesses_labels": False,
            "accesses_heldout": False,
            "creates_packet_delivery": False,
            "accepts_review_submission": False,
            "changes_dataset": False,
            "training_allowed": False,
            "f1_claim_allowed": False,
        },
        "decision": "issuer_preflight_ready_no_packet_submission_dataset_training_or_f1_action_authorized",
    }


def write_issuer_preflight_receipt(path: Path | str, preflight: IssuerPreflight, *, issued_at_utc: datetime) -> Path:
    """Publish a no-overwrite issuer-only receipt without reviewer packet content."""
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise IssuerPreflightError("issuer preflight receipt overwrite is forbidden")
    if not destination.parent.is_dir():
        raise IssuerPreflightError("issuer preflight receipt parent must already exist")
    encoded = _canonical_json_bytes(_receipt_payload(preflight, issued_at_utc=issued_at_utc))
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    except FileExistsError as error:
        raise IssuerPreflightError("issuer preflight receipt overwrite is forbidden") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination
