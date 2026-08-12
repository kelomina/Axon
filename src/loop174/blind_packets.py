"""Build direction-blind review packets and issued assignment records."""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from typing import Callable, Mapping, Sequence

from .dual_blind_review import IssuedAssignment


class BlindPacketError(ValueError):
    """Raised when a packet would disclose identity or prediction direction."""


FORBIDDEN_CONTEXT_TOKENS = frozenset(
    {
        "cache",
        "current",
        "directory",
        "error",
        "extension",
        "file",
        "filename",
        "hash",
        "index",
        "label",
        "malicious",
        "benign",
        "unknown",
        "positive",
        "negative",
        "path",
        "prediction",
        "prob",
        "rank",
        "sample",
        "score",
        "sha",
        "source",
        "split",
        "threshold",
    }
)
OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,127}$")


@dataclass(frozen=True)
class BlindCase:
    case_id: str
    review_lane: str
    context: Mapping[str, object]


@dataclass(frozen=True)
class Reviewer:
    key_id: str
    independence_group: str


@dataclass(frozen=True)
class ReviewerPacket:
    packet_id: str
    reviewer_key_id: str
    cases: tuple[dict[str, object], ...]


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BlindPacketError(f"{field} is invalid")
    return value.strip()


def _tokenized(value: str) -> set[str]:
    return {part for part in re.split(r"[^a-z0-9]+", value.casefold()) if part}


def _validate_context(case: BlindCase) -> None:
    if not OPAQUE_ID.fullmatch(_text(case.case_id, "case_id")):
        raise BlindPacketError("case_id must be opaque")
    review_lane = _text(case.review_lane, "review_lane")
    if _tokenized(review_lane) & FORBIDDEN_CONTEXT_TOKENS:
        raise BlindPacketError("review lane discloses identity or prediction direction")
    if not isinstance(case.context, Mapping):
        raise BlindPacketError("case context is invalid")
    # 审阅包只能携带脱敏的内容摘要，任何身份、标签或模型方向线索都必须在导出前拒绝。
    for key, value in case.context.items():
        context_key = _text(key, "context key")
        if _tokenized(context_key) & FORBIDDEN_CONTEXT_TOKENS:
            raise BlindPacketError("context key discloses identity or prediction direction")
        if isinstance(value, str) and _tokenized(value) & FORBIDDEN_CONTEXT_TOKENS:
            raise BlindPacketError("context value discloses identity or prediction direction")
        if not isinstance(value, (str, int, float, bool)) or isinstance(value, float) and not value == value:
            raise BlindPacketError("context value is invalid")


def _validate_reviewers(reviewers: Sequence[Reviewer]) -> tuple[Reviewer, ...]:
    if len(reviewers) < 3:
        raise BlindPacketError("at least three reviewers are required")
    seen_keys: set[str] = set()
    seen_groups: set[str] = set()
    normalized: list[Reviewer] = []
    for reviewer in reviewers:
        key_id = _text(reviewer.key_id, "reviewer key")
        group = _text(reviewer.independence_group, "reviewer independence group")
        if key_id in seen_keys or group in seen_groups:
            raise BlindPacketError("reviewer identities and independence groups must be unique")
        seen_keys.add(key_id)
        seen_groups.add(group)
        normalized.append(Reviewer(key_id, group))
    return tuple(sorted(normalized, key=lambda item: item.key_id))


def build_blind_packets(
    *,
    packet_id: str,
    cases: Sequence[BlindCase],
    reviewers: Sequence[Reviewer],
    expires_at_utc: datetime,
    nonce_factory: Callable[[], str] = lambda: secrets.token_urlsafe(24),
) -> tuple[tuple[ReviewerPacket, ...], tuple[IssuedAssignment, ...]]:
    """Assign each opaque case to exactly two independent reviewers."""
    if not OPAQUE_ID.fullmatch(_text(packet_id, "packet_id")):
        raise BlindPacketError("packet_id must be opaque")
    if not cases:
        raise BlindPacketError("at least one blind case is required")
    if expires_at_utc.tzinfo is None or expires_at_utc.utcoffset() != timedelta(0):
        raise BlindPacketError("assignment expiry must be UTC-aware")

    normalized_reviewers = _validate_reviewers(reviewers)
    seen_cases: set[str] = set()
    reviewer_cases: dict[str, list[dict[str, object]]] = {reviewer.key_id: [] for reviewer in normalized_reviewers}
    assignments: list[IssuedAssignment] = []
    reviewer_pairs = tuple(combinations(normalized_reviewers, 2))
    used_nonces: set[str] = set()
    for index, case in enumerate(cases):
        _validate_context(case)
        if case.case_id in seen_cases:
            raise BlindPacketError("blind case IDs must be unique")
        seen_cases.add(case.case_id)
        for reviewer in reviewer_pairs[index % len(reviewer_pairs)]:
            nonce = _text(nonce_factory(), "assignment nonce")
            if nonce in used_nonces:
                raise BlindPacketError("assignment nonces must be unique")
            used_nonces.add(nonce)
            assignment = IssuedAssignment(
                case.case_id,
                packet_id,
                reviewer.key_id,
                reviewer.independence_group,
                nonce,
                expires_at_utc,
            )
            assignments.append(assignment)
            reviewer_cases[reviewer.key_id].append(
                {
                    "case_id": case.case_id,
                    "assignment_nonce": nonce,
                    "review_lane": case.review_lane,
                    "context": dict(case.context),
                }
            )
    packets = tuple(
        ReviewerPacket(packet_id, reviewer.key_id, tuple(reviewer_cases[reviewer.key_id]))
        for reviewer in normalized_reviewers
        if reviewer_cases[reviewer.key_id]
    )
    return packets, tuple(assignments)
