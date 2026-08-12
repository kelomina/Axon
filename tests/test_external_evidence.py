import pytest

from src.external_evidence import ExternalEvidenceError, parse_external_evidence_row


def valid_row():
    return {"source_sha256": "a" * 64, "provider": "independent_engine", "observed_at_utc": "2026-07-15T00:00:00Z", "verdict": "malicious", "confidence": 0.95}


def test_parses_auditable_independent_evidence() -> None:
    assert parse_external_evidence_row(valid_row()).provider == "independent_engine"


def test_rejects_model_or_identity_leakage() -> None:
    row = valid_row()
    row["prediction"] = 1
    with pytest.raises(ExternalEvidenceError):
        parse_external_evidence_row(row)


def test_rejects_unqualified_timestamp() -> None:
    row = valid_row()
    row["observed_at_utc"] = "2026-07-15T00:00:00"
    with pytest.raises(ExternalEvidenceError):
        parse_external_evidence_row(row)
