from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_loop65_review_batch import build_review_batch


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_loop65_review_batch_selects_categories_and_keeps_manual_fields_blank(tmp_path: Path):
    queue_csv = tmp_path / "queue.csv"
    health_csv = tmp_path / "health.csv"
    dup_csv = tmp_path / "dups.csv"
    output_csv = tmp_path / "batch.csv"
    output_json = tmp_path / "batch.json"

    queue_rows = [
        {
            "review_priority_rank": "1",
            "review_lane": "A_persistent_error_in_high_conflict_queue",
            "priority_reason": "severe_fn_prob_le_0.01",
            "loop57_error_type": "FN",
            "label": "1",
            "loop57_final_prob": "0.001",
            "source_sha256": "sha-fn",
            "source_path": "data/fn.exe",
            "sample_index": "1",
            "loop39_corrected_by_any_compared_model": "False",
        },
        {
            "review_priority_rank": "2",
            "review_lane": "A_persistent_error_in_high_conflict_queue",
            "priority_reason": "severe_fp_prob_ge_0.99",
            "loop57_error_type": "FP",
            "label": "0",
            "loop57_final_prob": "0.999",
            "source_sha256": "sha-fp",
            "source_path": "data/fp.exe",
            "sample_index": "2",
            "loop39_corrected_by_any_compared_model": "False",
        },
        {
            "review_priority_rank": "3",
            "review_lane": "A_persistent_error_in_high_conflict_queue",
            "priority_reason": "high_fn_prob_le_0.05",
            "loop57_error_type": "FN",
            "label": "1",
            "loop57_final_prob": "0.02",
            "source_sha256": "sha-dup-a",
            "source_path": "data/dup-a.exe",
            "sample_index": "3",
            "loop39_corrected_by_any_compared_model": "False",
        },
        {
            "review_priority_rank": "4",
            "review_lane": "A_persistent_error_in_high_conflict_queue",
            "priority_reason": "high_fn_prob_le_0.05",
            "loop57_error_type": "FN",
            "label": "1",
            "loop57_final_prob": "0.03",
            "source_sha256": "sha-dup-b",
            "source_path": "data/dup-b.exe",
            "sample_index": "4",
            "loop39_corrected_by_any_compared_model": "False",
        },
        {
            "review_priority_rank": "5",
            "review_lane": "A_persistent_error_in_high_conflict_queue",
            "priority_reason": "medium_fn_prob_le_0.15",
            "loop57_error_type": "FN",
            "label": "1",
            "loop57_final_prob": "0.10",
            "source_sha256": "sha-other",
            "source_path": "data/other.exe",
            "sample_index": "5",
            "loop39_corrected_by_any_compared_model": "True",
        },
    ]
    _write_csv(queue_csv, queue_rows)
    _write_csv(
        health_csv,
        [
            {"source_sha256": "sha-fn", "objective_issue_count": "0", "pe_has_imports": "True"},
            {"source_sha256": "sha-fp", "objective_issue_count": "0", "pe_has_imports": "True"},
            {"source_sha256": "sha-dup-a", "objective_issue_count": "0", "pe_has_imports": "True"},
            {"source_sha256": "sha-dup-b", "objective_issue_count": "0", "pe_has_imports": "True"},
            {"source_sha256": "sha-other", "objective_issue_count": "0", "pe_has_imports": "True"},
        ],
    )
    _write_csv(
        dup_csv,
        [
            {
                "duplicate_group_id": "1",
                "manifest_source_sha256": "content-sha",
                "group_size": "2",
                "focus_queue_rows": "2",
                "split_source_sha256": "sha-dup-a",
                "source_path": "data/dup-a.exe",
                "sample_index": "3",
            },
            {
                "duplicate_group_id": "1",
                "manifest_source_sha256": "content-sha",
                "group_size": "2",
                "focus_queue_rows": "2",
                "split_source_sha256": "sha-dup-b",
                "source_path": "data/dup-b.exe",
                "sample_index": "4",
            },
        ],
    )

    summary = build_review_batch(
        queue_csv=queue_csv,
        health_audit_csv=health_csv,
        duplicate_details_csv=dup_csv,
        output_csv=output_csv,
        output_json=output_json,
        severe_fn_count=1,
        severe_fp_count=1,
        duplicate_group_count=1,
        corrected_by_other_count=1,
    )
    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8-sig", newline="")))

    assert summary["selected_rows"] == 5
    assert summary["manual_fields_blank_output"] is True
    assert summary["requested_duplicate_group_rows_in_queue"] == 2
    assert summary["selected_duplicate_group_rows"] == 2
    assert summary["selected_duplicate_group_category_counts"] == {"c_duplicate_content_group": 2}
    assert rows[0]["review_category"] == "a_severe_persistent_fn"
    assert rows[1]["review_category"] == "b_severe_persistent_fp"
    assert rows[2]["review_category"] == "c_duplicate_content_group"
    assert rows[2]["duplicate_manifest_sha_group"] == "true"
    assert rows[-1]["review_category"] == "d_corrected_by_other_model"
    assert all(row["manual_label_verdict"] == "" for row in rows)


def test_loop65_review_batch_reports_duplicate_rows_selected_by_higher_priority_category(tmp_path: Path):
    queue_csv = tmp_path / "queue.csv"
    health_csv = tmp_path / "health.csv"
    dup_csv = tmp_path / "dups.csv"
    output_csv = tmp_path / "batch.csv"
    output_json = tmp_path / "batch.json"

    _write_csv(
        queue_csv,
        [
            {
                "review_priority_rank": "1",
                "review_lane": "A_persistent_error_in_high_conflict_queue",
                "priority_reason": "severe_fn_prob_le_0.01",
                "loop57_error_type": "FN",
                "label": "1",
                "loop57_final_prob": "0.001",
                "source_sha256": "sha-overlap",
                "source_path": "data/overlap.exe",
                "sample_index": "1",
                "loop39_corrected_by_any_compared_model": "False",
            }
        ],
    )
    _write_csv(health_csv, [{"source_sha256": "sha-overlap", "objective_issue_count": "0"}])
    _write_csv(
        dup_csv,
        [
            {
                "duplicate_group_id": "1",
                "manifest_source_sha256": "content-sha",
                "group_size": "1",
                "focus_queue_rows": "1",
                "split_source_sha256": "sha-overlap",
                "source_path": "data/overlap.exe",
                "sample_index": "1",
            }
        ],
    )

    summary = build_review_batch(
        queue_csv=queue_csv,
        health_audit_csv=health_csv,
        duplicate_details_csv=dup_csv,
        output_csv=output_csv,
        output_json=output_json,
        severe_fn_count=1,
        severe_fp_count=0,
        duplicate_group_count=1,
        corrected_by_other_count=0,
    )
    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8-sig", newline="")))

    assert summary["selected_rows"] == 1
    assert summary["category_counts"] == {"a_severe_persistent_fn": 1}
    assert summary["requested_duplicate_group_rows_in_queue"] == 1
    assert summary["selected_duplicate_group_rows"] == 1
    assert summary["selected_duplicate_group_category_counts"] == {"a_severe_persistent_fn": 1}
    assert rows[0]["duplicate_manifest_sha_group"] == "true"
