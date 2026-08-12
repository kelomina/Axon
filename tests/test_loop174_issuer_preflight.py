from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop174.blind_packets import Reviewer  # noqa: E402
from src.loop174.issuer_preflight import (  # noqa: E402
    CASE_MANIFEST_SCHEMA,
    ISSUER_PREFLIGHT_SCHEMA,
    IssuerPreflightError,
    build_issuer_preflight,
    write_issuer_preflight_receipt,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _manifest() -> dict[str, object]:
    return {
        "schema_version": CASE_MANIFEST_SCHEMA,
        "packet_id": "packet_001",
        "score_cutoff_utc": "2024-02-01T00:00:00Z",
        "source_registry_snapshot_commitment": _digest("source-snapshot"),
        "cases": [
            {
                "case_id": "case_0001",
                "subject_commitment": _digest("subject-1"),
                "review_lane": "content",
                "context": {"import_density": 1.2},
            },
            {
                "case_id": "case_0002",
                "subject_commitment": _digest("subject-2"),
                "review_lane": "trust",
                "context": {"vendor_indicator": True},
            },
        ],
    }


def _reviewers() -> list[Reviewer]:
    return [Reviewer("reviewer-a", "group-a"), Reviewer("reviewer-b", "group-b"), Reviewer("reviewer-c", "group-c")]


def test_preflight_binds_cases_to_provenance_and_independent_roster(tmp_path: Path) -> None:
    preflight = build_issuer_preflight(_manifest(), _reviewers())
    receipt = write_issuer_preflight_receipt(
        tmp_path / "issuer-preflight.json",
        preflight,
        issued_at_utc=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["schema_version"] == ISSUER_PREFLIGHT_SCHEMA
    assert payload["case_bindings"] == [
        {"case_id": "case_0001", "subject_commitment": _digest("subject-1")},
        {"case_id": "case_0002", "subject_commitment": _digest("subject-2")},
    ]
    assert len(payload["reviewer_roster"]) == 3
    assert payload["hard_boundaries"]["accesses_samples"] is False
    assert payload["hard_boundaries"]["accepts_review_submission"] is False


def test_preflight_rejects_legacy_direction_fields_and_nonunique_subjects() -> None:
    legacy = _manifest()
    legacy["cases"][0]["context"] = {"current_label": "benign"}  # type: ignore[index]
    with pytest.raises(IssuerPreflightError, match="discloses"):
        build_issuer_preflight(legacy, _reviewers())

    duplicate = _manifest()
    duplicate["cases"][1]["subject_commitment"] = duplicate["cases"][0]["subject_commitment"]  # type: ignore[index]
    with pytest.raises(IssuerPreflightError, match="one-to-one"):
        build_issuer_preflight(duplicate, _reviewers())


def test_receipt_is_atomic_and_non_overwritable(tmp_path: Path) -> None:
    preflight = build_issuer_preflight(_manifest(), _reviewers())
    destination = tmp_path / "issuer-preflight.json"
    write_issuer_preflight_receipt(destination, preflight, issued_at_utc=datetime(2026, 7, 15, tzinfo=timezone.utc))

    with pytest.raises(IssuerPreflightError, match="overwrite"):
        write_issuer_preflight_receipt(destination, preflight, issued_at_utc=datetime(2026, 7, 15, tzinfo=timezone.utc))
