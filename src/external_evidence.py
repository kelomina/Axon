"""Fail-closed contract for independently collected malware evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping


class ExternalEvidenceError(ValueError):
    """Raised when evidence is not independent or cannot be audited."""


@dataclass(frozen=True)
class ExternalEvidenceRow:
    source_sha256: str
    provider: str
    observed_at_utc: datetime
    verdict: str
    confidence: float


def parse_external_evidence_row(payload: Mapping[str, object]) -> ExternalEvidenceRow:
    required = {"source_sha256", "provider", "observed_at_utc", "verdict", "confidence"}
    forbidden = {"label", "prediction", "probability", "path", "filename", "sample_index", "split"}
    if set(payload) != required:
        unexpected = set(payload) - required
        missing = required - set(payload)
        raise ExternalEvidenceError(f"external evidence fields drifted: missing={missing}, unexpected={unexpected}")
    if forbidden & set(payload):
        raise ExternalEvidenceError("external evidence contains prohibited identity or model fields")
    source_sha256 = str(payload["source_sha256"]).lower()
    provider = str(payload["provider"]).strip()
    verdict = str(payload["verdict"]).strip().lower()
    if len(source_sha256) != 64 or any(char not in "0123456789abcdef" for char in source_sha256):
        raise ExternalEvidenceError("source_sha256 must be a lowercase SHA-256 digest")
    if not provider or verdict not in {"malicious", "benign", "unknown"}:
        raise ExternalEvidenceError("provider or verdict is invalid")
    try:
        observed_at = datetime.fromisoformat(str(payload["observed_at_utc"]).replace("Z", "+00:00"))
        confidence = float(payload["confidence"])
    except (TypeError, ValueError) as error:
        raise ExternalEvidenceError("external evidence timestamp or confidence is invalid") from error
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ExternalEvidenceError("observed_at_utc must include a timezone")
    if not 0.0 <= confidence <= 1.0:
        raise ExternalEvidenceError("confidence must be in [0, 1]")
    return ExternalEvidenceRow(source_sha256, provider, observed_at.astimezone(timezone.utc), verdict, confidence)
