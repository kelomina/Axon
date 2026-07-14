from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_loop88_full_error_evidence_coverage import build_coverage_report


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixtures(tmp_path: Path, *, wave_missing: bool = False):
    queue_csv = tmp_path / "queue.csv"
    wave_csv = tmp_path / "wave.csv"
    target_json = tmp_path / "target.json"
    loop86_json = tmp_path / "loop86.json"
    loop87_json = tmp_path / "loop87.json"
    health_json = tmp_path / "health.json"
    duplicate_json = tmp_path / "duplicate.json"
    output_json = tmp_path / "out.json"

    queue_rows = [
        {
            "sample_index": "1",
            "source_sha256": "a",
            "source_path": "data/a.exe",
            "loop57_error_type": "FN",
            "review_lane": "A_persistent_error_in_high_conflict_queue",
            "priority_reason": "severe_fn_prob_le_0.01",
        },
        {
            "sample_index": "2",
            "source_sha256": "b",
            "source_path": "data/b.exe",
            "loop57_error_type": "FP",
            "review_lane": "B_persistent_error",
            "priority_reason": "high_fp_prob_ge_0.95",
        },
        {
            "sample_index": "3",
            "source_sha256": "c",
            "source_path": "data/c.exe",
            "loop57_error_type": "FP",
            "review_lane": "D_loop57_new_error",
            "priority_reason": "lower_confidence_fp",
        },
    ]
    _write_csv(queue_csv, queue_rows)
    wave_rows = queue_rows[:2] if wave_missing else queue_rows
    _write_csv(
        wave_csv,
        [
            {**row, "review_wave_id": "1" if idx < 2 else "2"}
            for idx, row in enumerate(wave_rows)
        ],
    )
    _write_json(
        target_json,
        {
            "target_f1": 0.999,
            "current_best": {"f1": 0.9, "errors": 3, "fp": 2, "fn": 1},
            "target_gap_best_case": {"minimum_fixed_errors_best_case": 2},
            "error_reduction_needed_ratio_of_current_errors": 2 / 3,
        },
    )
    _write_json(
        loop86_json,
        {
            "rows": 1,
            "source_exists_count": 1,
            "cache_exists_count": 1,
            "source_sha256_mismatch_count": 0,
            "pe_parse_status_counts": {"ok": 1},
        },
    )
    _write_json(
        loop87_json,
        {
            "rows": 1,
            "import_ready": True,
            "decision": "ready_noop_no_actionable_verdicts",
            "manual_quality": {"blank_verdict_rows": 1},
            "actionable_rows": 0,
            "replacement_required_rows": 0,
            "training_policy_rows": 0,
        },
    )
    _write_json(health_json, {"rows": 2, "objective_issue_row_count": 0, "issue_counts": {}})
    _write_json(
        duplicate_json,
        {
            "duplicate_groups": 1,
            "cross_label_groups": 0,
            "cross_split_groups": 0,
            "focus_duplicate_detail_rows": 0,
        },
    )
    return queue_csv, target_json, wave_csv, loop86_json, loop87_json, health_json, duplicate_json, output_json


def test_loop88_reports_first_package_coverage_and_next_step(tmp_path: Path):
    files = _fixtures(tmp_path)
    report = build_coverage_report(
        queue_csv=files[0],
        target_gap_json=files[1],
        loop72_wave_csv=files[2],
        loop86_summary_json=files[3],
        loop87_import_json=files[4],
        loop63_health_summary_json=files[5],
        loop64_duplicate_summary_json=files[6],
        output_json=files[7],
    )

    assert report["blockers"] == []
    assert report["queue_coverage"]["queue_rows"] == 3
    assert report["queue_coverage"]["loop72_covers_queue_keys"] is True
    assert report["evidence_package_coverage"]["loop86_rows"] == 1
    assert report["evidence_package_coverage"]["remaining_target_gap_after_loop86_package"] == 1
    assert report["verdict_gate_status"]["loop87_decision"] == "ready_noop_no_actionable_verdicts"
    assert report["decisions"]["training_allowed"] is False
    assert report["recommendation"]["priority"] == "expand_evidence_package_coverage"


def test_loop88_blocks_when_wave_plan_does_not_cover_queue(tmp_path: Path):
    files = _fixtures(tmp_path, wave_missing=True)
    report = build_coverage_report(
        queue_csv=files[0],
        target_gap_json=files[1],
        loop72_wave_csv=files[2],
        loop86_summary_json=files[3],
        loop87_import_json=files[4],
        loop63_health_summary_json=files[5],
        loop64_duplicate_summary_json=files[6],
        output_json=files[7],
    )

    assert "loop72_wave_plan_does_not_cover_same_queue_rows" in report["blockers"]
    assert report["queue_coverage"]["loop72_covers_queue_keys"] is False
