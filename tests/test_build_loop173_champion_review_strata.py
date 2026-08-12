from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_loop173_champion_review_strata import build_strata


def test_strata_reports_missing_source_provenance_without_identity_output() -> None:
    public_rows = [
        {"review_focus_id": "one", "review_lane": "benign"},
        {"review_focus_id": "two", "review_lane": "content"},
        {"review_focus_id": "three", "review_lane": "malware"},
    ]
    private_rows = [
        {"review_focus_id": "one", "source_path": "data/2024-01/a.exe"},
        {"review_focus_id": "two", "source_path": "data/2025-01/b.exe"},
        {"review_focus_id": "three", "source_path": "data/2026-01/c.exe"},
    ]

    result = build_strata(public_rows, private_rows)

    assert result["axes"]["component"]["ready"] is True
    assert result["axes"]["time"]["ready"] is True
    assert result["axes"]["source"]["ready"] is False
    assert result["decision"] == "blocked_missing_independent_source_provenance"
    assert "data/2024-01/a.exe" not in str(result)


def test_strata_rejects_alignment_drift() -> None:
    with pytest.raises(ValueError, match="do not align"):
        build_strata(
            [{"review_focus_id": "one", "review_lane": "content"}],
            [{"review_focus_id": "two", "source_path": "data/2024-01/a.exe"}],
        )
