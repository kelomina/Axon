"""Fail-closed JSON delivery for Loop174 blind-review packets."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

from .blind_packets import OPAQUE_ID, BlindCase, BlindPacketError, ReviewerPacket, _validate_context
from .dual_blind_review import IssuedAssignment


class BlindPacketExportError(ValueError):
    """Raised when a blind-review delivery cannot be safely exported."""


PACKET_SCHEMA_VERSION = "axon.loop174.reviewer_packet.v1"
ASSIGNMENT_SCHEMA_VERSION = "axon.loop174.issuer_assignments.v1"


@dataclass(frozen=True)
class BlindPacketExport:
    """Paths written by a one-time Loop174 delivery export."""

    reviewer_packet_paths: tuple[Path, ...]
    issuer_assignments_path: Path


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BlindPacketExportError(f"{field} is invalid")
    return value.strip()


def _utc_timestamp(value: datetime, field: str) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise BlindPacketExportError(f"{field} must be UTC-aware")
    return value.isoformat().replace("+00:00", "Z")


def _opaque(value: object, field: str) -> str:
    text = _text(value, field)
    if not OPAQUE_ID.fullmatch(text):
        raise BlindPacketExportError(f"{field} must be opaque")
    return text


def _resolve_directory(value: Path | str, field: str) -> Path:
    path = Path(value)
    if not path.exists() or not path.is_dir():
        raise BlindPacketExportError(f"{field} must be an existing directory")
    return path.resolve(strict=True)


def _directories_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _json_safe_context(case_row: Mapping[str, object]) -> dict[str, object]:
    expected = {"case_id", "assignment_nonce", "review_lane", "context"}
    if set(case_row) != expected:
        raise BlindPacketExportError("reviewer case schema drifted")
    try:
        case = BlindCase(
            _opaque(case_row["case_id"], "case_id"),
            _text(case_row["review_lane"], "review_lane"),
            case_row["context"],
        )
        _validate_context(case)
    except BlindPacketError as error:
        raise BlindPacketExportError(str(error)) from error
    return {
        "case_id": case.case_id,
        "assignment_nonce": _text(case_row["assignment_nonce"], "assignment_nonce"),
        "review_lane": case.review_lane,
        "context": dict(case.context),
    }


def _packet_payload(packet: ReviewerPacket) -> dict[str, object]:
    packet_id = _opaque(packet.packet_id, "packet_id")
    reviewer_key_id = _opaque(packet.reviewer_key_id, "reviewer_key_id")
    if not isinstance(packet.cases, tuple) or not packet.cases:
        raise BlindPacketExportError("reviewer packet cases are invalid")
    cases = [_json_safe_context(row) for row in packet.cases]
    case_ids = [row["case_id"] for row in cases]
    nonces = [row["assignment_nonce"] for row in cases]
    if len(case_ids) != len(set(case_ids)) or len(nonces) != len(set(nonces)):
        raise BlindPacketExportError("reviewer packet case or nonce identity is duplicated")
    return {
        "schema_version": PACKET_SCHEMA_VERSION,
        "packet_id": packet_id,
        "reviewer_key_id": reviewer_key_id,
        "cases": cases,
    }


def _assignment_payload(assignments: Sequence[IssuedAssignment], packet_id: str) -> dict[str, object]:
    if not assignments:
        raise BlindPacketExportError("at least one issued assignment is required")
    rows: list[dict[str, str]] = []
    seen_nonces: set[str] = set()
    for assignment in assignments:
        if assignment.packet_id != packet_id:
            raise BlindPacketExportError("issued assignment packet identity drifted")
        nonce = _text(assignment.assignment_nonce, "assignment_nonce")
        if nonce in seen_nonces:
            raise BlindPacketExportError("issued assignment nonce is duplicated")
        seen_nonces.add(nonce)
        rows.append(
            {
                "case_id": _opaque(assignment.case_id, "case_id"),
                "packet_id": packet_id,
                "reviewer_key_id": _opaque(assignment.reviewer_key_id, "reviewer_key_id"),
                "reviewer_independence_group": _text(
                    assignment.reviewer_independence_group,
                    "reviewer_independence_group",
                ),
                "assignment_nonce": nonce,
                "expires_at_utc": _utc_timestamp(assignment.expires_at_utc, "assignment expiry"),
            }
        )
    return {"schema_version": ASSIGNMENT_SCHEMA_VERSION, "packet_id": packet_id, "assignments": rows}


def _write_once_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise BlindPacketExportError(f"delivery target already exists: {path.name}")
    encoded = (json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # hard-link publication preserves no-overwrite semantics for a completed JSON artifact.
        os.link(temporary, path)
    except FileExistsError as error:
        raise BlindPacketExportError(f"delivery target already exists: {path.name}") from error
    except OSError as error:
        raise BlindPacketExportError(f"could not publish delivery target: {path.name}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def export_blind_packet_delivery(
    *,
    reviewer_delivery_dir: Path | str,
    issuer_ledger_dir: Path | str,
    packets: Sequence[ReviewerPacket],
    assignments: Sequence[IssuedAssignment],
) -> BlindPacketExport:
    """Write reviewer-only packets and a separate issuer-only assignment ledger once."""
    reviewer_directory = _resolve_directory(reviewer_delivery_dir, "reviewer_delivery_dir")
    issuer_directory = _resolve_directory(issuer_ledger_dir, "issuer_ledger_dir")
    if _directories_overlap(reviewer_directory, issuer_directory):
        raise BlindPacketExportError("reviewer and issuer delivery directories must not overlap")
    if not packets:
        raise BlindPacketExportError("at least one reviewer packet is required")

    payloads = [_packet_payload(packet) for packet in packets]
    packet_ids = {payload["packet_id"] for payload in payloads}
    reviewer_ids = [payload["reviewer_key_id"] for payload in payloads]
    if len(packet_ids) != 1 or len(reviewer_ids) != len(set(reviewer_ids)):
        raise BlindPacketExportError("reviewer packet identities are inconsistent")
    packet_id = next(iter(packet_ids))
    ledger = _assignment_payload(assignments, packet_id)

    packet_nonces = {case["assignment_nonce"] for payload in payloads for case in payload["cases"]}
    assignment_nonces = {row["assignment_nonce"] for row in ledger["assignments"]}
    if packet_nonces != assignment_nonces:
        raise BlindPacketExportError("reviewer packets and issued assignments are not one-to-one")
    expected_reviewer_cases = {
        (row["reviewer_key_id"], row["case_id"], row["assignment_nonce"])
        for row in ledger["assignments"]
    }
    actual_reviewer_cases = {
        (payload["reviewer_key_id"], case["case_id"], case["assignment_nonce"])
        for payload in payloads
        for case in payload["cases"]
    }
    if actual_reviewer_cases != expected_reviewer_cases:
        raise BlindPacketExportError("reviewer packets do not match issued assignments")

    target_paths = [
        reviewer_directory / f"{packet_id}.{payload['reviewer_key_id']}.json"
        for payload in payloads
    ]
    issuer_path = issuer_directory / f"{packet_id}.assignments.json"
    if any(path.exists() for path in (*target_paths, issuer_path)):
        raise BlindPacketExportError("a delivery target already exists")

    packet_paths: list[Path] = []
    for payload in sorted(payloads, key=lambda row: str(row["reviewer_key_id"])):
        path = reviewer_directory / f"{packet_id}.{payload['reviewer_key_id']}.json"
        _write_once_json(path, payload)
        packet_paths.append(path)
    _write_once_json(issuer_path, ledger)
    return BlindPacketExport(tuple(packet_paths), issuer_path)
