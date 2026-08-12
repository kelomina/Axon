from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop174.blind_packets import (  # noqa: E402
    BlindCase,
    Reviewer,
    ReviewerPacket,
    build_blind_packets,
)
from src.loop174.dual_blind_review import IssuedAssignment  # noqa: E402
from src.loop174.json_export import (  # noqa: E402
    ASSIGNMENT_SCHEMA_VERSION,
    PACKET_SCHEMA_VERSION,
    BlindPacketExportError,
    export_blind_packet_delivery,
)


def _packets_and_assignments() -> tuple[tuple[ReviewerPacket, ...], tuple[IssuedAssignment, ...]]:
    nonces = iter(f"nonce-{index}" for index in range(6))
    return build_blind_packets(
        packet_id="packet_001",
        cases=[
            BlindCase("case_0001", "content", {"import_density": 1.2}),
            BlindCase("case_0002", "trust", {"vendor_indicator": True}),
            BlindCase("case_0003", "blindspot", {"entropy_bucket": "high"}),
        ],
        reviewers=[Reviewer("reviewer-a", "group-a"), Reviewer("reviewer-b", "group-b"), Reviewer("reviewer-c", "group-c")],
        expires_at_utc=datetime(2026, 8, 1, tzinfo=timezone.utc),
        nonce_factory=lambda: next(nonces),
    )


def test_export_separates_reviewer_packets_from_issuer_assignment_ledger(tmp_path: Path) -> None:
    reviewer_directory = tmp_path / "reviewer-delivery"
    issuer_directory = tmp_path / "issuer-ledger"
    reviewer_directory.mkdir()
    issuer_directory.mkdir()
    packets, assignments = _packets_and_assignments()

    result = export_blind_packet_delivery(
        reviewer_delivery_dir=reviewer_directory,
        issuer_ledger_dir=issuer_directory,
        packets=packets,
        assignments=assignments,
    )

    assert len(result.reviewer_packet_paths) == 3
    assert result.issuer_assignments_path.parent == issuer_directory.resolve()
    packet_payload = json.loads(result.reviewer_packet_paths[0].read_text(encoding="utf-8"))
    ledger_payload = json.loads(result.issuer_assignments_path.read_text(encoding="utf-8"))
    assert packet_payload["schema_version"] == PACKET_SCHEMA_VERSION
    assert ledger_payload["schema_version"] == ASSIGNMENT_SCHEMA_VERSION
    assert "reviewer_independence_group" not in json.dumps(packet_payload)
    assert len(ledger_payload["assignments"]) == 6
    assert {case["assignment_nonce"] for packet in [json.loads(path.read_text(encoding="utf-8")) for path in result.reviewer_packet_paths] for case in packet["cases"]} == {
        row["assignment_nonce"] for row in ledger_payload["assignments"]
    }


def test_export_rejects_directly_constructed_direction_leak(tmp_path: Path) -> None:
    reviewer_directory = tmp_path / "reviewer-delivery"
    issuer_directory = tmp_path / "issuer-ledger"
    reviewer_directory.mkdir()
    issuer_directory.mkdir()
    packet = ReviewerPacket(
        "packet_001",
        "reviewer-a",
        ({"case_id": "case_0001", "assignment_nonce": "nonce-a", "review_lane": "content", "context": {"content_summary": "benign"}},),
    )
    assignment = IssuedAssignment("case_0001", "packet_001", "reviewer-a", "group-a", "nonce-a", datetime(2026, 8, 1, tzinfo=timezone.utc))

    with pytest.raises(BlindPacketExportError, match="discloses"):
        export_blind_packet_delivery(
            reviewer_delivery_dir=reviewer_directory,
            issuer_ledger_dir=issuer_directory,
            packets=[packet],
            assignments=[assignment],
        )
    assert not list(reviewer_directory.iterdir())
    assert not list(issuer_directory.iterdir())


def test_export_rejects_overlapping_roots_and_preexisting_target_without_partial_delivery(tmp_path: Path) -> None:
    reviewer_directory = tmp_path / "reviewer-delivery"
    issuer_directory = tmp_path / "issuer-ledger"
    reviewer_directory.mkdir()
    issuer_directory.mkdir()
    packets, assignments = _packets_and_assignments()

    with pytest.raises(BlindPacketExportError, match="must not overlap"):
        export_blind_packet_delivery(
            reviewer_delivery_dir=reviewer_directory,
            issuer_ledger_dir=reviewer_directory,
            packets=packets,
            assignments=assignments,
        )

    (issuer_directory / "packet_001.assignments.json").write_text("reserved", encoding="utf-8")
    with pytest.raises(BlindPacketExportError, match="already exists"):
        export_blind_packet_delivery(
            reviewer_delivery_dir=reviewer_directory,
            issuer_ledger_dir=issuer_directory,
            packets=packets,
            assignments=assignments,
        )
    assert not list(reviewer_directory.iterdir())
