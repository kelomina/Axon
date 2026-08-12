"""Fail-closed feasibility gates for independent label-quality review."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


class ReviewFeasibilityError(ValueError):
    """Raised when aggregate review evidence violates the preregistered contract."""


@dataclass(frozen=True)
class ReviewStratum:
    """Aggregate review counts for one component/time/source stratum."""

    stratum_id: str
    eligible_rows: int
    reviewed_rows: int
    agreeing_rows: int
    actionable_rows: int
    asof_verified_rows: int


@dataclass(frozen=True)
class ReviewFeasibilityAudit:
    """A non-promoting review-readiness result."""

    reviewed_rows: int
    actionable_rows: int
    agreement_lcb: float
    actionable_lcb: float
    strata_complete: bool
    decision: str


ONE_SIDED_975_Z = 1.959963984540054


def wilson_lower_bound(successes: int, total: int) -> float:
    """Return the one-sided 97.5% Wilson lower bound without row-level evidence."""
    if not isinstance(successes, int) or not isinstance(total, int) or successes < 0 or total <= 0 or successes > total:
        raise ReviewFeasibilityError("Wilson count inputs are invalid")
    proportion = successes / total
    z_squared = ONE_SIDED_975_Z**2
    denominator = 1.0 + z_squared / total
    center = proportion + z_squared / (2.0 * total)
    radius = ONE_SIDED_975_Z * math.sqrt(
        proportion * (1.0 - proportion) / total + z_squared / (4.0 * total * total)
    )
    return max(0.0, (center - radius) / denominator)


def audit_review_feasibility(
    strata: Sequence[ReviewStratum],
    *,
    required_strata: int,
    minimum_reviewed_rows: int,
    minimum_agreement_lcb: float,
    minimum_actionable_lcb: float,
) -> ReviewFeasibilityAudit:
    """Gate future data-governance work without inferring a label or model metric."""
    if required_strata <= 0 or minimum_reviewed_rows <= 0:
        raise ReviewFeasibilityError("review coverage requirements are invalid")
    if not 0.0 <= minimum_agreement_lcb <= 1.0 or not 0.0 <= minimum_actionable_lcb <= 1.0:
        raise ReviewFeasibilityError("review lower-bound requirements are invalid")
    unique_ids: set[str] = set()
    reviewed = agreeing = actionable = asof_verified = 0
    # 每个分层都必须有独立审阅和历史可用证据，不能用总体平均掩盖空白分层。
    strata_complete = len(strata) >= required_strata
    for stratum in strata:
        if not stratum.stratum_id or stratum.stratum_id in unique_ids:
            raise ReviewFeasibilityError("review stratum identity is invalid")
        unique_ids.add(stratum.stratum_id)
        counts = (
            stratum.eligible_rows,
            stratum.reviewed_rows,
            stratum.agreeing_rows,
            stratum.actionable_rows,
            stratum.asof_verified_rows,
        )
        if any(not isinstance(value, int) or value < 0 for value in counts):
            raise ReviewFeasibilityError("review stratum counts are invalid")
        if not (stratum.reviewed_rows <= stratum.eligible_rows and stratum.agreeing_rows <= stratum.reviewed_rows and stratum.actionable_rows <= stratum.agreeing_rows and stratum.asof_verified_rows >= stratum.actionable_rows):
            raise ReviewFeasibilityError("review stratum count ordering is invalid")
        strata_complete = strata_complete and stratum.reviewed_rows > 0 and stratum.asof_verified_rows >= stratum.actionable_rows
        reviewed += stratum.reviewed_rows
        agreeing += stratum.agreeing_rows
        actionable += stratum.actionable_rows
        asof_verified += stratum.asof_verified_rows
    if reviewed == 0:
        return ReviewFeasibilityAudit(0, 0, 0.0, 0.0, False, "await_independent_review")
    agreement_lcb = wilson_lower_bound(agreeing, reviewed)
    actionable_lcb = wilson_lower_bound(actionable, reviewed)
    gates_pass = (
        strata_complete
        and reviewed >= minimum_reviewed_rows
        and asof_verified >= actionable
        and agreement_lcb >= minimum_agreement_lcb
        and actionable_lcb >= minimum_actionable_lcb
    )
    return ReviewFeasibilityAudit(
        reviewed,
        actionable,
        agreement_lcb,
        actionable_lcb,
        strata_complete,
        "independent_review_feasibility_passed_governance_only" if gates_pass else "await_independent_review",
    )
