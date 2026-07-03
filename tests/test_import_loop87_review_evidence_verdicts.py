from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.import_loop87_review_evidence_verdicts import validate_loop86_verdicts


FIELDS = [
    "review_batch_rank",
    "review_category",
    "source_path",
    "cache_path",
    "source_sha256",
    "sample_index",
    "split",
    "label",
    "loop57_error_type",
    "loop57_final_prob",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = FIELDS
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _base_row(**overrides) -> dict:
    row = {
        "review_batch_rank": "1",
        "review_category": "a_severe_persistent_fn",
        "source_path": "data/a.exe",
        "cache_path": "data/.cache/a.npz",
        "source_sha256": "a" * 64,
        "sample_index": "101",
        "split": "test",
        "label": "1",
        "loop57_error_type": "FN",
        "loop57_final_prob": "0.001",
        "manual_label_verdict": "",
        "manual_verdict_note": "",
        "recommended_action": "",
    }
    row.update(overrides)
    return row


def _run(tmp_path: Path, rows: list[dict], *, expected_rows: int | None = None):
    evidence = tmp_path / "evidence.csv"
    output_csv = tmp_path / "validated.csv"
    output_json = tmp_path / "summary.json"
    _write_csv(evidence, rows)
    return validate_loop86_verdicts(
        evidence_csv=evidence,
        output_csv=output_csv,
        output_json=output_json,
        expected_rows=expected_rows if expected_rows is not None else len(rows),
    )


def test_loop87_blank_verdicts_are_ready_noop(tmp_path: Path):
    summary = _run(tmp_path, [_base_row()])
    validated = list(csv.DictReader((tmp_path / "validated.csv").open("r", encoding="utf-8-sig", newline="")))

    assert summary["import_ready"] is True
    assert summary["decision"] == "ready_noop_no_actionable_verdicts"
    assert summary["status_counts"] == {"no_decision": 1}
    assert summary["replacement_required_rows"] == 0
    assert summary["training_policy_rows"] == 0
    assert validated[0]["loop87_plan_action"] == "no_action"


def test_loop87_identity_only_note_blocks_actionable_verdict(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            _base_row(
                manual_label_verdict="label_correct",
                manual_verdict_note="filename and source_path show this row should keep the label",
                recommended_action="model_blindspot",
            )
        ],
    )

    assert summary["import_ready"] is False
    assert summary["decision"] == "blocked_invalid_verdicts"
    assert summary["row_issue_counts"]["manual_verdict_note_missing_content_or_external_evidence"] == 1
    assert summary["row_issue_counts"]["manual_verdict_note_identity_or_score_only"] == 1
    assert summary["manual_quality"]["evidence_note_identity_or_score_only_rows"] == 1


def test_loop87_model_score_only_note_blocks_actionable_verdict(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            _base_row(
                manual_label_verdict="label_correct",
                manual_verdict_note="loop57 final_prob is below threshold",
                recommended_action="model_blindspot",
            )
        ],
    )

    assert summary["import_ready"] is False
    assert summary["row_issue_counts"]["manual_verdict_note_identity_or_score_only"] == 1
    assert summary["manual_quality"]["evidence_note_missing_content_or_external_rows"] == 1


def test_loop87_feature_broken_creates_redraw_request_only(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            _base_row(
                split="val",
                label="0",
                loop57_error_type="FP",
                manual_label_verdict="feature_broken",
                manual_verdict_note="PE parse evidence and npz feature mismatch confirm broken feature extraction",
                recommended_action="replace_with_fresh_same_label_candidate",
            )
        ],
    )
    validated = list(csv.DictReader((tmp_path / "validated.csv").open("r", encoding="utf-8-sig", newline="")))

    assert summary["import_ready"] is True
    assert summary["decision"] == "ready_for_redraw_plan_review_only"
    assert summary["status_counts"] == {"exclude_replace": 1}
    assert summary["replacement_required_rows"] == 1
    assert summary["replacement_counts_by_original_label"] == {"0": 1}
    assert summary["decisions"]["automatic_replacement_allowed"] is False
    assert validated[0]["loop87_plan_action"] == "quarantine_and_fresh_redraw"
    assert validated[0]["loop87_replacement_required"] == "true"


def test_loop87_duplicate_sample_index_blocks_even_when_rows_are_blank(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            _base_row(review_batch_rank="1", sample_index="7", source_sha256="a" * 64),
            _base_row(review_batch_rank="2", sample_index="7", source_sha256="b" * 64),
        ],
    )

    assert summary["import_ready"] is False
    assert "duplicate_sample_index" in summary["blocking_issues"]
    assert summary["duplicate_sample_index_rows"] == 1


def test_loop87_missing_required_column_blocks_import(tmp_path: Path):
    evidence = tmp_path / "evidence.csv"
    output_csv = tmp_path / "validated.csv"
    output_json = tmp_path / "summary.json"
    fieldnames = [field for field in FIELDS if field != "source_sha256"]
    _write_csv(evidence, [_base_row()], fieldnames=fieldnames)

    summary = validate_loop86_verdicts(
        evidence_csv=evidence,
        output_csv=output_csv,
        output_json=output_json,
        expected_rows=1,
    )
    saved = json.loads(output_json.read_text(encoding="utf-8"))

    assert summary == saved
    assert summary["import_ready"] is False
    assert "missing_required_columns" in summary["blocking_issues"]
    assert summary["missing_required_columns"] == ["source_sha256"]
