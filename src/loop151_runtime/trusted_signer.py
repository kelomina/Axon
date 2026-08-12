"""Frozen Loop151 Authenticode trusted-signer decision policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

TRUSTED_SIGNER_TERMS = (
    "Microsoft Corporation",
    "Microsoft Windows",
    "Seagate Technology",
    "FinalWire",
    "NetEase",
    "Beijing Sogou",
    "Beijing Kingsoft",
    "Beijing Qihu",
    "Wondershare",
    "IObit",
    "Yozosoft",
    "Huya",
)


@dataclass(frozen=True)
class TrustedSignerDecision:
    """The final Loop151 policy output and its auditable frozen inputs."""

    prediction: int
    downgraded: bool
    matched_terms: tuple[str, ...]


def apply_trusted_signer_guard(
    *,
    loop136_prediction: int,
    authenticode_status: str | None,
    signer_subject: str | None,
    trusted_terms: Iterable[str] = TRUSTED_SIGNER_TERMS,
) -> TrustedSignerDecision:
    """Apply Loop151's fixed precision-side downgrade without score retuning."""
    if loop136_prediction not in (0, 1):
        raise ValueError("loop136_prediction must be binary")
    status = str(authenticode_status or "").strip().casefold()
    subject = str(signer_subject or "").strip().casefold()
    terms = tuple(str(term).strip() for term in trusted_terms if str(term).strip())
    if len(set(term.casefold() for term in terms)) != len(terms):
        raise ValueError("trusted signer terms must be unique")
    matched = tuple(term for term in terms if term.casefold() in subject)
    downgraded = loop136_prediction == 1 and status == "valid" and bool(matched)
    return TrustedSignerDecision(0 if downgraded else loop136_prediction, downgraded, matched)
