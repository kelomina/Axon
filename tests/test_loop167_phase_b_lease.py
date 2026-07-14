from __future__ import annotations

import hashlib

import pytest

from src.loop167_phase_b.lease import LeaseError, consume_lease


def test_lease_consumes_marker_exactly_once(tmp_path) -> None:
    marker = tmp_path / "phase_b_execution_consumed.json"
    payload = {"schema": "synthetic", "lease_id": "loop167-phase-b-train-oof-v1"}
    consumed = consume_lease(marker, payload)
    assert marker.is_file()
    assert consumed.marker_sha256 == hashlib.sha256(marker.read_bytes()).hexdigest()
    with pytest.raises(LeaseError, match="already exists"):
        consume_lease(marker, payload)
    assert marker.read_bytes()


def test_lease_rejects_noncanonical_payload_without_creating_marker(tmp_path) -> None:
    marker = tmp_path / "invalid.json"
    with pytest.raises(LeaseError, match="canonical JSON"):
        consume_lease(marker, {"invalid": {1, 2, 3}})
    assert not marker.exists()
