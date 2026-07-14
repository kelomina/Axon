from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.audit_loop71_target_gap_noise_roi import build_audit, f1_from_counts


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_loop71_quantifies_target_gap_and_review_roi(tmp_path: Path):
    loop57 = tmp_path / "loop57.json"
    loop63_summary = tmp_path / "loop63.json"
    loop63_csv = tmp_path / "loop63.csv"
    loop65_summary = tmp_path / "loop65.json"
    loop65_csv = tmp_path / "loop65.csv"
    loop50 = tmp_path / "loop50.json"
    loop64 = tmp_path / "loop64.json"
    out = tmp_path / "out.json"

    _write_json(
        loop57,
        {
            "metrics": {
                "samples": 1000,
                "f1": f1_from_counts(490, 10, 20),
                "true_positive": 490,
                "true_negative": 480,
                "false_positive": 10,
                "false_negative": 20,
            }
        },
    )
    _write_json(
        loop63_summary,
        {
            "loop57_error_rows": 30,
            "review_lane_counts": {"A_persistent_error_in_high_conflict_queue": 12, "B_persistent_error": 18},
            "error_type_counts": {"FP": 10, "FN": 20},
            "loop39_intersection_rows": 12,
            "manual_label_verdict_blank_count": 30,
        },
    )
    _write_json(
        loop65_summary,
        {
            "selected_rows": 5,
            "category_counts": {"a": 3, "b": 2},
            "error_type_counts": {"FN": 4, "FP": 1},
            "manual_fields_blank_output": True,
        },
    )
    _write_json(loop50, {"rows": 12, "objective_issue_row_count": 1, "issue_counts": {"duplicate": 1}})
    _write_json(
        loop64,
        {
            "duplicate_groups": 1,
            "focus_duplicate_detail_rows": 2,
            "cross_label_groups": 0,
            "cross_split_groups": 0,
        },
    )
    _write_csv(
        loop63_csv,
        [
            {"review_lane": "A_persistent_error_in_high_conflict_queue", "priority_reason": "severe_fn"},
            {"review_lane": "B_persistent_error", "priority_reason": "high_fp"},
        ],
    )
    _write_csv(loop65_csv, [{"review_category": "a"}, {"review_category": "b"}])

    report = build_audit(
        loop57_eval_json=loop57,
        loop63_summary_json=loop63_summary,
        loop63_queue_csv=loop63_csv,
        loop65_summary_json=loop65_summary,
        loop65_batch_csv=loop65_csv,
        loop50_summary_json=loop50,
        loop64_summary_json=loop64,
        output_json=out,
        target_f1=0.99,
    )

    assert report["current_best"]["errors"] == 30
    assert report["error_reduction_needed_best_case"] > 0
    assert report["review_roi"]["loop65_selected_batch"]["review_rows"] == 5
    assert report["decision"]["replacement_rule"].startswith("Confirmed label_wrong")
    identity_policy = report["identity_feature_policy"]
    for forbidden_field in ("filename", "path", "extension", "directory", "source hash", "sample_index", "split", "row order"):
        assert forbidden_field in identity_policy
    assert "not model evidence" in identity_policy
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == "axon_loop71_target_gap_noise_roi_v1"
