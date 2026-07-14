from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_loop72_review_wave_plan import build_wave_plan


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_loop72_builds_review_waves_without_filling_manual_fields(tmp_path: Path):
    queue_csv = tmp_path / "queue.csv"
    target_json = tmp_path / "target.json"
    health_csv = tmp_path / "health.csv"
    dup_csv = tmp_path / "dups.csv"
    out_csv = tmp_path / "waves.csv"
    out_json = tmp_path / "waves.json"

    queue_rows = [
        {
            "review_priority_rank": "1",
            "review_lane": "A_persistent_error_in_high_conflict_queue",
            "priority_reason": "severe_fn_prob_le_0.01",
            "exchange_group": "loop28_loop57_both_error",
            "loop57_error_type": "FN",
            "label": "1",
            "loop57_final_prob": "0.001",
            "source_sha256": "sha-a",
            "source_path": "data/a.exe",
            "sample_index": "1",
            "split": "test",
        },
        {
            "review_priority_rank": "2",
            "review_lane": "A_persistent_error_in_high_conflict_queue",
            "priority_reason": "high_fn_prob_le_0.05",
            "exchange_group": "loop28_loop57_both_error",
            "loop57_error_type": "FN",
            "label": "1",
            "loop57_final_prob": "0.02",
            "source_sha256": "sha-dup-a",
            "source_path": "data/dup-a.exe",
            "sample_index": "2",
            "split": "test",
        },
        {
            "review_priority_rank": "3",
            "review_lane": "B_persistent_error",
            "priority_reason": "high_fn_prob_le_0.05",
            "exchange_group": "loop28_loop57_both_error",
            "loop57_error_type": "FN",
            "label": "1",
            "loop57_final_prob": "0.03",
            "source_sha256": "sha-dup-b",
            "source_path": "data/dup-b.exe",
            "sample_index": "3",
            "split": "test",
        },
        {
            "review_priority_rank": "4",
            "review_lane": "D_loop57_new_error",
            "priority_reason": "severe_fp_prob_ge_0.99",
            "exchange_group": "loop57_new_error",
            "loop57_error_type": "FP",
            "label": "0",
            "loop57_final_prob": "0.999",
            "source_sha256": "sha-fp",
            "source_path": "data/fp.exe",
            "sample_index": "4",
            "split": "test",
        },
    ]
    _write_csv(queue_csv, queue_rows)
    _write_json(
        target_json,
        {
            "target_f1": 0.99,
            "error_reduction_needed_best_case": 3,
            "current_best": {
                "f1": 0.96,
                "errors": 4,
                "fp": 1,
                "fn": 3,
                "tp": 47,
                "tn": 49,
            },
        },
    )
    _write_csv(
        health_csv,
        [
            {"source_sha256": "sha-a", "objective_issue_count": "1", "objective_issue_flags": "bad_cache"},
            {"source_sha256": "sha-fp", "objective_issue_count": "0"},
        ],
    )
    _write_csv(
        dup_csv,
        [
            {
                "duplicate_group_id": "g1",
                "manifest_source_sha256": "content-sha",
                "split_source_sha256": "sha-dup-a",
                "group_size": "2",
                "focus_queue_rows": "2",
                "source_path": "data/dup-a.exe",
                "sample_index": "2",
            },
            {
                "duplicate_group_id": "g1",
                "manifest_source_sha256": "content-sha",
                "split_source_sha256": "sha-dup-b",
                "group_size": "2",
                "focus_queue_rows": "2",
                "source_path": "data/dup-b.exe",
                "sample_index": "3",
            },
        ],
    )

    summary = build_wave_plan(
        queue_csv=queue_csv,
        target_gap_json=target_json,
        health_audit_csv=health_csv,
        duplicate_details_csv=dup_csv,
        output_csv=out_csv,
        output_json=out_json,
        wave_size=2,
    )
    rows = list(csv.DictReader(out_csv.open("r", encoding="utf-8-sig", newline="")))

    assert summary["rows"] == 4
    assert summary["wave_count"] == 3
    assert summary["first_wave_reaching_target_if_all_actionable"] == 2
    assert summary["manual_fields_blank_output"] is True
    assert summary["review_category_counts"]["a_objective_data_issue"] == 1
    assert summary["review_category_counts"]["b_duplicate_content_group"] == 2
    assert summary["duplicate_manifest_group_count"] == 1
    assert rows[1]["review_wave_id"] == rows[2]["review_wave_id"]
    assert rows[1]["manifest_duplicate_group_id"] == "g1"
    assert rows[0]["manual_label_verdict"] == ""
    assert rows[0]["recommended_action"] == ""
    assert rows[0]["corrected_label"] == ""
    assert rows[-1]["target_reached_if_all_confirmed_by_this_row"] == "true"
    identity_policy = summary["identity_feature_policy"]
    for forbidden_field in (
        "filename",
        "path",
        "extension",
        "directory",
        "source hash",
        "cache_path",
        "sample_index",
        "split",
        "row order",
    ):
        assert forbidden_field in identity_policy
    assert "not model evidence" in identity_policy
    protocol = summary["protocol"]
    for forbidden_action in (
        "no model fitting",
        "no threshold selection",
        "no automatic relabeling",
        "no split mutation",
        "no Test-derived feature engineering",
    ):
        assert forbidden_action in protocol


def test_loop72_keeps_all_duplicate_group_rows_in_same_wave_even_when_wave_overflows(tmp_path: Path):
    queue_csv = tmp_path / "queue.csv"
    target_json = tmp_path / "target.json"
    dup_csv = tmp_path / "dups.csv"
    out_csv = tmp_path / "waves.csv"
    out_json = tmp_path / "waves.json"

    rows = []
    dup_rows = []
    for index in range(3):
        sha = f"sha-dup-{index}"
        rows.append(
            {
                "review_priority_rank": str(index + 1),
                "review_lane": "A_persistent_error_in_high_conflict_queue",
                "priority_reason": "high_fn_prob_le_0.05",
                "exchange_group": "loop28_loop57_both_error",
                "loop57_error_type": "FN",
                "label": "1",
                "loop57_final_prob": "0.02",
                "source_sha256": sha,
                "source_path": f"data/{sha}.exe",
                "sample_index": str(index + 1),
                "split": "test",
            }
        )
        dup_rows.append(
            {
                "duplicate_group_id": "big",
                "manifest_source_sha256": "content-sha",
                "split_source_sha256": sha,
                "group_size": "3",
                "focus_queue_rows": "3",
                "source_path": f"data/{sha}.exe",
                "sample_index": str(index + 1),
            }
        )
    _write_csv(queue_csv, rows)
    _write_csv(dup_csv, dup_rows)
    _write_json(
        target_json,
        {
            "target_f1": 0.99,
            "error_reduction_needed_best_case": 3,
            "current_best": {"f1": 0.96, "errors": 3, "fp": 0, "fn": 3, "tp": 47, "tn": 50},
        },
    )

    summary = build_wave_plan(
        queue_csv=queue_csv,
        target_gap_json=target_json,
        duplicate_details_csv=dup_csv,
        output_csv=out_csv,
        output_json=out_json,
        wave_size=2,
    )
    output = list(csv.DictReader(out_csv.open("r", encoding="utf-8-sig", newline="")))

    assert summary["wave_count"] == 1
    assert {row["review_wave_id"] for row in output} == {"1"}
    assert [row["review_wave_rank"] for row in output] == ["1", "2", "3"]
