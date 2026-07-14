from __future__ import annotations

import json
from pathlib import Path

from scripts.build_loop89_wave1_evidence_summary import build_summary


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixtures(tmp_path: Path, *, evidence_rows: int = 2):
    loop72 = _write_json(
        tmp_path / "loop72.json",
        {"wave_summaries": [{"review_wave_id": 1, "rows": 2}]},
    )
    loop88 = _write_json(
        tmp_path / "loop88.json",
        {
            "queue_coverage": {"queue_rows": 10},
            "target_gap": {"minimum_fixed_errors_best_case": 8},
        },
    )
    evidence = _write_json(
        tmp_path / "evidence.json",
        {
            "rows": evidence_rows,
            "error_type_counts": {"FN": 1, "FP": 1},
            "category_counts": {"c": evidence_rows},
            "source_exists_count": evidence_rows,
            "cache_exists_count": evidence_rows,
            "source_sha256_mismatch_count": 0,
            "pe_parse_status_counts": {"ok": evidence_rows},
            "review_tag_counts": {"overlay_present": 1},
        },
    )
    verdict = _write_json(
        tmp_path / "verdict.json",
        {
            "import_ready": True,
            "decision": "ready_noop_no_actionable_verdicts",
            "manual_quality": {"blank_verdict_rows": evidence_rows},
            "actionable_rows": 0,
            "replacement_required_rows": 0,
            "training_policy_rows": 0,
        },
    )
    return loop72, loop88, evidence, verdict, tmp_path / "out.json"


def test_loop89_summarizes_wave1_coverage(tmp_path: Path):
    files = _fixtures(tmp_path)
    summary = build_summary(
        loop72_summary_json=files[0],
        loop88_coverage_json=files[1],
        wave1_evidence_json=files[2],
        wave1_verdict_json=files[3],
        output_json=files[4],
    )

    assert summary["blockers"] == []
    assert summary["wave1"]["rows"] == 2
    assert summary["coverage_after_wave1"]["coverage_of_queue_ratio"] == 0.2
    assert summary["coverage_after_wave1"]["coverage_of_target_gap_ratio"] == 0.25
    assert summary["decisions"]["training_allowed"] is False


def test_loop89_blocks_when_evidence_rows_do_not_match_wave1(tmp_path: Path):
    files = _fixtures(tmp_path, evidence_rows=1)
    summary = build_summary(
        loop72_summary_json=files[0],
        loop88_coverage_json=files[1],
        wave1_evidence_json=files[2],
        wave1_verdict_json=files[3],
        output_json=files[4],
    )

    assert "wave1_evidence_rows_do_not_match_loop72_wave1" in summary["blockers"]
