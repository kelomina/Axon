from __future__ import annotations

import json
from pathlib import Path

from scripts.build_loop90_multiwave_evidence_summary import build_summary


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixtures(tmp_path: Path, *, wave2_rows: int = 2):
    loop72 = _write_json(
        tmp_path / "loop72.json",
        {
            "wave_summaries": [
                {"review_wave_id": 1, "rows": 2},
                {"review_wave_id": 2, "rows": 2},
            ]
        },
    )
    loop88 = _write_json(
        tmp_path / "loop88.json",
        {
            "queue_coverage": {"queue_rows": 10},
            "target_gap": {"minimum_fixed_errors_best_case": 8},
        },
    )
    ev1 = _write_json(
        tmp_path / "ev1.json",
        {
            "rows": 2,
            "error_type_counts": {"FN": 1, "FP": 1},
            "category_counts": {"a": 2},
            "review_tag_counts": {"overlay_present": 1},
            "source_exists_count": 2,
            "cache_exists_count": 2,
            "source_sha256_mismatch_count": 0,
            "pe_parse_status_counts": {"ok": 2},
        },
    )
    vd1 = _write_json(
        tmp_path / "vd1.json",
        {
            "rows": 2,
            "import_ready": True,
            "decision": "ready_noop_no_actionable_verdicts",
            "manual_quality": {"blank_verdict_rows": 2},
            "actionable_rows": 0,
            "replacement_required_rows": 0,
            "training_policy_rows": 0,
        },
    )
    ev2 = _write_json(
        tmp_path / "ev2.json",
        {
            "rows": wave2_rows,
            "error_type_counts": {"FP": wave2_rows},
            "category_counts": {"b": wave2_rows},
            "review_tag_counts": {"has_resource_directory": wave2_rows},
            "source_exists_count": wave2_rows,
            "cache_exists_count": wave2_rows,
            "source_sha256_mismatch_count": 0,
            "pe_parse_status_counts": {"ok": wave2_rows},
        },
    )
    vd2 = _write_json(
        tmp_path / "vd2.json",
        {
            "rows": wave2_rows,
            "import_ready": True,
            "decision": "ready_noop_no_actionable_verdicts",
            "manual_quality": {"blank_verdict_rows": wave2_rows},
            "actionable_rows": 0,
            "replacement_required_rows": 0,
            "training_policy_rows": 0,
        },
    )
    return loop72, loop88, ev1, vd1, ev2, vd2, tmp_path / "out.json"


def test_loop90_combines_multiple_wave_summaries(tmp_path: Path):
    files = _fixtures(tmp_path)
    summary = build_summary(
        loop72_summary_json=files[0],
        loop88_coverage_json=files[1],
        waves=[(1, files[2], files[3]), (2, files[4], files[5])],
        output_json=files[6],
    )

    assert summary["blockers"] == []
    assert summary["covered_waves"] == [1, 2]
    assert summary["combined"]["rows"] == 4
    assert summary["combined"]["coverage_of_queue_ratio"] == 0.4
    assert summary["combined"]["coverage_of_target_gap_ratio"] == 0.5
    assert summary["combined"]["error_type_counts"] == {"FN": 1, "FP": 3}
    assert summary["decisions"]["training_allowed"] is False


def test_loop90_blocks_when_wave_rows_do_not_match_loop72(tmp_path: Path):
    files = _fixtures(tmp_path, wave2_rows=1)
    summary = build_summary(
        loop72_summary_json=files[0],
        loop88_coverage_json=files[1],
        waves=[(1, files[2], files[3]), (2, files[4], files[5])],
        output_json=files[6],
    )

    assert "wave2_evidence_rows_do_not_match_loop72" in summary["blockers"]
