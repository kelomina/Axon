from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop174.blind_packets import (  # noqa: E402
    BlindCase,
    BlindPacketError,
    Reviewer,
    build_blind_packets,
)


def _reviewers() -> list[Reviewer]:
    return [Reviewer("reviewer-a", "group-a"), Reviewer("reviewer-b", "group-b"), Reviewer("reviewer-c", "group-c")]


def _cases() -> list[BlindCase]:
    return [
        BlindCase("case_0001", "content", {"import_density": 1.2, "resource_count": 4}),
        BlindCase("case_0002", "trust", {"vendor_indicator": True}),
        BlindCase("case_0003", "blindspot", {"entropy_bucket": "high"}),
    ]


def test_each_case_gets_two_independent_reviewer_packets_and_issued_assignments() -> None:
    nonces = iter(f"nonce-{index}" for index in range(6))
    packets, assignments = build_blind_packets(
        packet_id="packet_001",
        cases=_cases(),
        reviewers=_reviewers(),
        expires_at_utc=datetime(2026, 8, 1, tzinfo=timezone.utc),
        nonce_factory=lambda: next(nonces),
    )

    assert len(assignments) == 6
    assert len({assignment.assignment_nonce for assignment in assignments}) == 6
    for case in _cases():
        case_assignments = [assignment for assignment in assignments if assignment.case_id == case.case_id]
        assert len(case_assignments) == 2
        assert len({assignment.reviewer_independence_group for assignment in case_assignments}) == 2
    assert {packet.reviewer_key_id for packet in packets} == {"reviewer-a", "reviewer-b", "reviewer-c"}
    assert all("independence_group" not in row for packet in packets for row in packet.cases)


def test_direction_or_identity_context_is_rejected() -> None:
    with pytest.raises(BlindPacketError, match="discloses"):
        build_blind_packets(
            packet_id="packet_001",
            cases=[BlindCase("case_0001", "content", {"current_label": "benign"})],
            reviewers=_reviewers(),
            expires_at_utc=datetime(2026, 8, 1, tzinfo=timezone.utc),
        )


def test_duplicate_nonce_or_reviewer_group_fails_closed() -> None:
    with pytest.raises(BlindPacketError, match="nonces"):
        build_blind_packets(
            packet_id="packet_001",
            cases=[BlindCase("case_0001", "content", {})],
            reviewers=_reviewers(),
            expires_at_utc=datetime(2026, 8, 1, tzinfo=timezone.utc),
            nonce_factory=lambda: "reused-nonce",
        )
    with pytest.raises(BlindPacketError, match="independence"):
        build_blind_packets(
            packet_id="packet_001",
            cases=[BlindCase("case_0001", "content", {})],
            reviewers=[Reviewer("reviewer-a", "group-a"), Reviewer("reviewer-b", "group-a"), Reviewer("reviewer-c", "group-c")],
            expires_at_utc=datetime(2026, 8, 1, tzinfo=timezone.utc),
        )


def test_non_utc_assignment_expiry_fails_closed() -> None:
    with pytest.raises(BlindPacketError, match="UTC"):
        build_blind_packets(
            packet_id="packet_001",
            cases=[BlindCase("case_0001", "content", {})],
            reviewers=_reviewers(),
            expires_at_utc=datetime(2026, 8, 1, tzinfo=timezone(timedelta(hours=8))),
        )
