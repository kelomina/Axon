from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop173.review_feasibility import (
    ReviewFeasibilityError,
    ReviewStratum,
    audit_review_feasibility,
    wilson_lower_bound,
)


def _stratum(identifier: str, *, reviewed: int = 54, agreeing: int = 52, actionable: int = 12) -> ReviewStratum:
    return ReviewStratum(identifier, 54, reviewed, agreeing, actionable, actionable)


def test_empty_review_is_not_promoted() -> None:
    result = audit_review_feasibility([], required_strata=3, minimum_reviewed_rows=162, minimum_agreement_lcb=0.8, minimum_actionable_lcb=0.05)

    assert result.decision == "await_independent_review"
    assert result.reviewed_rows == 0


def test_complete_independent_review_can_only_pass_governance_gate() -> None:
    result = audit_review_feasibility([_stratum("component-a"), _stratum("component-b"), _stratum("component-c")], required_strata=3, minimum_reviewed_rows=162, minimum_agreement_lcb=0.8, minimum_actionable_lcb=0.05)

    assert result.decision == "independent_review_feasibility_passed_governance_only"
    assert result.actionable_lcb > 0.05


def test_empty_stratum_and_duplicate_identity_fail_closed() -> None:
    with pytest.raises(ReviewFeasibilityError, match="count ordering"):
        audit_review_feasibility([ReviewStratum("component-a", 10, 0, 0, 1, 1)], required_strata=1, minimum_reviewed_rows=1, minimum_agreement_lcb=0.8, minimum_actionable_lcb=0.05)
    with pytest.raises(ReviewFeasibilityError, match="identity"):
        audit_review_feasibility([_stratum("component-a"), _stratum("component-a")], required_strata=2, minimum_reviewed_rows=1, minimum_agreement_lcb=0.8, minimum_actionable_lcb=0.05)


def test_wilson_bound_rejects_invalid_counts() -> None:
    with pytest.raises(ReviewFeasibilityError):
        wilson_lower_bound(3, 2)


def test_preregistered_agreement_threshold_requires_at_least_140_of_162() -> None:
    assert wilson_lower_bound(139, 162) < 0.8
    assert wilson_lower_bound(140, 162) >= 0.8
