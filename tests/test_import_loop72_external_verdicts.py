from __future__ import annotations

import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from import_loop72_external_verdicts import validate_loop72_external_verdicts  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


REVIEW_FIELDS = [
    "source_path",
    "source_sha256",
    "sample_index",
    "split",
    "label",
    "loop57_error_type",
    "loop57_prediction",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
    "corrected_label",
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_split(path: Path, rows: list[dict]) -> None:
    _write_csv(path, ["source_path", "source_sha256", "sample_index", "split", "label"], rows)


def _write_target_gap(path: Path, *, errors: int = 1) -> None:
    path.write_text(
        json.dumps(
            {
                "target_f1": 0.999,
                "current_best": {"tp": 10, "tn": 8, "fp": 1, "fn": 1, "errors": errors},
                "review_sources": {"loop63": {"error_rows": errors}},
            }
        ),
        encoding="utf-8",
    )


def _run_import(tmp_path: Path, review_rows: list[dict], split_rows: list[dict], *, expected_rows: int | None = None):
    review_csv = tmp_path / "review.csv"
    split_csv = tmp_path / "split.csv"
    target_gap = tmp_path / "target_gap.json"
    _write_csv(review_csv, REVIEW_FIELDS, review_rows)
    _write_split(split_csv, split_rows)
    _write_target_gap(target_gap, errors=expected_rows if expected_rows is not None else len(review_rows))
    return validate_loop72_external_verdicts(
        review_csv=review_csv,
        split_csv=split_csv,
        target_gap_json=target_gap,
        output_csv=tmp_path / "validated.csv",
        output_json=tmp_path / "validated.json",
        plan_csv=tmp_path / "plan.csv",
        plan_json=tmp_path / "plan.json",
        allow_partial=False,
        expected_rows=expected_rows,
        enforce_20w_split=False,
        allow_test_actions=False,
    )


def test_blank_loop72_verdicts_are_import_ready_noop():
    with _case_dir("loop74_blank") as tmp_path:
        summary = _run_import(
            tmp_path,
            [
                {
                    "source_path": "data/a.exe",
                    "source_sha256": "a" * 64,
                    "sample_index": "1",
                    "split": "test",
                    "label": "1",
                    "loop57_error_type": "FN",
                    "loop57_prediction": "0",
                    "manual_label_verdict": "",
                    "manual_verdict_note": "",
                    "recommended_action": "",
                    "corrected_label": "",
                }
            ],
            [{"source_path": "data/a.exe", "source_sha256": "a" * 64, "sample_index": "1", "split": "test", "label": "1"}],
        )

    assert summary["import_ready"] is True
    assert summary["import_status_counts"] == {"no_decision": 1}
    assert summary["adjustment_plan_summary"]["planned_rows"] == 0
    assert summary["adjustment_plan_summary"]["training_policy_rows"] == 0


def test_label_wrong_replacement_is_held_out_for_test_policy_but_counts_metric_feasibility():
    with _case_dir("loop74_label_wrong_test") as tmp_path:
        summary = _run_import(
            tmp_path,
            [
                {
                    "source_path": "data/fp.exe",
                    "source_sha256": "b" * 64,
                    "sample_index": "2",
                    "split": "test",
                    "label": "0",
                    "loop57_error_type": "FP",
                    "loop57_prediction": "1",
                    "manual_label_verdict": "label_wrong",
                    "manual_verdict_note": "external sandbox and vendor evidence confirm malicious content",
                    "recommended_action": "replace_sample",
                    "corrected_label": "1",
                }
            ],
            [{"source_path": "data/fp.exe", "source_sha256": "b" * 64, "sample_index": "2", "split": "test", "label": "0"}],
        )

    assert summary["import_ready"] is True
    assert summary["metric_effect_counts"] == {"label_wrong_fixes_current_fp": 1}
    assert summary["target_gap_metrics"]["strict_label_correction_only"]["fixed_current_errors"] == 1
    assert summary["target_gap_metrics"]["replacement_required_rows"] == 1
    assert summary["adjustment_plan_summary"]["action_counts"] == {"held_out_test_verdict_only": 1}
    assert summary["adjustment_plan_summary"]["training_policy_rows"] == 0


def test_feature_broken_row_requires_replacement_plan_not_self_fill():
    with _case_dir("loop74_replace") as tmp_path:
        summary = _run_import(
            tmp_path,
            [
                {
                    "source_path": "data/bad.exe",
                    "source_sha256": "c" * 64,
                    "sample_index": "3",
                    "split": "val",
                    "label": "1",
                    "loop57_error_type": "FN",
                    "loop57_prediction": "0",
                    "manual_label_verdict": "feature_broken",
                    "manual_verdict_note": "strict PE feature extraction failed for this file",
                    "recommended_action": "replace_sample",
                    "corrected_label": "",
                }
            ],
            [{"source_path": "data/bad.exe", "source_sha256": "c" * 64, "sample_index": "3", "split": "val", "label": "1"}],
        )

    assert summary["import_ready"] is True
    assert summary["metric_effect_counts"] == {"replacement_required_same_original_label_1": 1}
    assert summary["target_gap_metrics"]["replacement_required_rows"] == 1
    assert summary["adjustment_plan_summary"]["replacement_required"] == 1
    assert summary["adjustment_plan_summary"]["replacement_counts_by_original_label"] == {"1": 1}


def test_conflicting_verdict_action_blocks_import():
    with _case_dir("loop74_conflict") as tmp_path:
        summary = _run_import(
            tmp_path,
            [
                {
                    "source_path": "data/conflict.exe",
                    "source_sha256": "d" * 64,
                    "sample_index": "4",
                    "split": "train",
                    "label": "0",
                    "loop57_error_type": "FP",
                    "loop57_prediction": "1",
                    "manual_label_verdict": "label_wrong",
                    "manual_verdict_note": "external evidence confirms the current label is wrong",
                    "recommended_action": "relabel_train_only",
                    "corrected_label": "1",
                }
            ],
            [{"source_path": "data/conflict.exe", "source_sha256": "d" * 64, "sample_index": "4", "split": "train", "label": "0"}],
        )

    assert summary["import_ready"] is False
    assert summary["invalid_rows"] == 1
    assert summary["row_issue_counts"]["label_wrong_requires_replace_or_quarantine_action"] == 1
    assert summary["adjustment_plan_summary"] is None


def test_duplicate_sample_index_blocks_import_even_with_distinct_paths():
    with _case_dir("loop74_duplicate_sample_index") as tmp_path:
        summary = _run_import(
            tmp_path,
            [
                {
                    "source_path": "data/a.exe",
                    "source_sha256": "e" * 64,
                    "sample_index": "5",
                    "split": "test",
                    "label": "1",
                    "loop57_error_type": "FN",
                    "loop57_prediction": "0",
                    "manual_label_verdict": "",
                    "manual_verdict_note": "",
                    "recommended_action": "",
                    "corrected_label": "",
                },
                {
                    "source_path": "data/b.exe",
                    "source_sha256": "f" * 64,
                    "sample_index": "5",
                    "split": "test",
                    "label": "1",
                    "loop57_error_type": "FN",
                    "loop57_prediction": "0",
                    "manual_label_verdict": "",
                    "manual_verdict_note": "",
                    "recommended_action": "",
                    "corrected_label": "",
                },
            ],
            [{"source_path": "data/a.exe", "source_sha256": "e" * 64, "sample_index": "5", "split": "test", "label": "1"}],
            expected_rows=2,
        )

    assert summary["import_ready"] is False
    assert "duplicate_review_sample_index" in summary["blocking_issues"]
    assert summary["duplicate_review_sample_index_rows"] == 1


def test_corrected_label_without_relabel_verdict_blocks_import():
    with _case_dir("loop74_stray_corrected") as tmp_path:
        summary = _run_import(
            tmp_path,
            [
                {
                    "source_path": "data/stray.exe",
                    "source_sha256": "1" * 64,
                    "sample_index": "6",
                    "split": "val",
                    "label": "1",
                    "loop57_error_type": "FN",
                    "loop57_prediction": "0",
                    "manual_label_verdict": "label_correct",
                    "manual_verdict_note": "trusted external evidence confirms original malicious label",
                    "recommended_action": "keep_label",
                    "corrected_label": "0",
                }
            ],
            [{"source_path": "data/stray.exe", "source_sha256": "1" * 64, "sample_index": "6", "split": "val", "label": "1"}],
        )

    assert summary["import_ready"] is False
    assert summary["row_issue_counts"]["label_correct_must_not_have_corrected_label"] == 1


def test_actionable_verdict_without_note_blocks_import():
    with _case_dir("loop74_missing_note") as tmp_path:
        summary = _run_import(
            tmp_path,
            [
                {
                    "source_path": "data/no-note.exe",
                    "source_sha256": "3" * 64,
                    "sample_index": "16",
                    "split": "val",
                    "label": "0",
                    "loop57_error_type": "FP",
                    "loop57_prediction": "1",
                    "manual_label_verdict": "feature_broken",
                    "manual_verdict_note": "",
                    "recommended_action": "replace_sample",
                    "corrected_label": "",
                }
            ],
            [{"source_path": "data/no-note.exe", "source_sha256": "3" * 64, "sample_index": "16", "split": "val", "label": "0"}],
        )

    assert summary["import_ready"] is False
    assert summary["row_issue_counts"]["actionable_verdict_requires_manual_verdict_note"] == 1
    assert summary["manual_quality"]["actionable_verdict_missing_note_rows"] == 1


def test_identity_only_note_blocks_import():
    with _case_dir("loop75_identity_only_note") as tmp_path:
        summary = _run_import(
            tmp_path,
            [
                {
                    "source_path": "data/name-only.exe",
                    "source_sha256": "4" * 64,
                    "sample_index": "17",
                    "split": "test",
                    "label": "0",
                    "loop57_error_type": "FP",
                    "loop57_prediction": "1",
                    "manual_label_verdict": "label_correct",
                    "manual_verdict_note": "filename and directory indicate the sample should be benign",
                    "recommended_action": "keep_label",
                    "corrected_label": "",
                }
            ],
            [{"source_path": "data/name-only.exe", "source_sha256": "4" * 64, "sample_index": "17", "split": "test", "label": "0"}],
        )

    assert summary["import_ready"] is False
    assert summary["row_issue_counts"]["manual_verdict_note_missing_content_or_external_evidence"] == 1
    assert summary["row_issue_counts"]["manual_verdict_note_identity_or_score_only"] == 1
    assert summary["manual_quality"]["evidence_note_identity_or_score_only_rows"] == 1


def test_model_score_only_note_blocks_import():
    with _case_dir("loop75_score_only_note") as tmp_path:
        summary = _run_import(
            tmp_path,
            [
                {
                    "source_path": "data/score-only.exe",
                    "source_sha256": "5" * 64,
                    "sample_index": "18",
                    "split": "test",
                    "label": "1",
                    "loop57_error_type": "FN",
                    "loop57_prediction": "0",
                    "manual_label_verdict": "label_correct",
                    "manual_verdict_note": "loop57 final_prob is below threshold so the dataset label is correct",
                    "recommended_action": "model_blindspot",
                    "corrected_label": "",
                }
            ],
            [{"source_path": "data/score-only.exe", "source_sha256": "5" * 64, "sample_index": "18", "split": "test", "label": "1"}],
        )

    assert summary["import_ready"] is False
    assert summary["row_issue_counts"]["manual_verdict_note_missing_content_or_external_evidence"] == 1
    assert summary["row_issue_counts"]["manual_verdict_note_identity_or_score_only"] == 1
    assert summary["manual_quality"]["evidence_note_missing_content_or_external_rows"] == 1


def test_20w_split_gate_blocks_small_splits_when_enabled():
    with _case_dir("loop74_20w_gate") as tmp_path:
        review_csv = tmp_path / "review.csv"
        split_csv = tmp_path / "split.csv"
        target_gap = tmp_path / "target_gap.json"
        _write_csv(
            review_csv,
            REVIEW_FIELDS,
            [
                {
                    "source_path": "data/a.exe",
                    "source_sha256": "2" * 64,
                    "sample_index": "7",
                    "split": "test",
                    "label": "0",
                    "loop57_error_type": "FP",
                    "loop57_prediction": "1",
                    "manual_label_verdict": "",
                    "manual_verdict_note": "",
                    "recommended_action": "",
                    "corrected_label": "",
                }
            ],
        )
        _write_split(
            split_csv,
            [{"source_path": "data/a.exe", "source_sha256": "2" * 64, "sample_index": "7", "split": "test", "label": "0"}],
        )
        _write_target_gap(target_gap, errors=1)

        summary = validate_loop72_external_verdicts(
            review_csv=review_csv,
            split_csv=split_csv,
            target_gap_json=target_gap,
            output_csv=tmp_path / "validated.csv",
            output_json=tmp_path / "validated.json",
            plan_csv=tmp_path / "plan.csv",
            plan_json=tmp_path / "plan.json",
            enforce_20w_split=True,
        )

    assert summary["import_ready"] is False
    assert "split_row_count_not_200000" in summary["blocking_issues"]
    assert "split_counts_not_20000_20000_160000" in summary["blocking_issues"]
