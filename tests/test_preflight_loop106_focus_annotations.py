from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from preflight_loop106_focus_annotations import preflight_focus_annotations  # noqa: E402


FIELDS = [
    "blind_review_id",
    "current_label",
    "loop106_focus_rank",
    "loop106_focus_score",
    "loop106_focus_bucket",
    "loop106_focus_reasons",
    "file_entropy",
    "pe_parse_status",
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
        "blind_review_id": "blind-00001",
        "current_label": "0",
        "loop106_focus_rank": "1",
        "loop106_focus_score": "10.0",
        "loop106_focus_bucket": "benign_label_content_review",
        "loop106_focus_reasons": "overlay_present",
        "file_entropy": "7.5",
        "pe_parse_status": "ok",
        "manual_label_verdict": "",
        "manual_verdict_note": "",
        "recommended_action": "",
    }
    row.update(overrides)
    return row


def _run(tmp_path: Path, rows: list[dict], *, fieldnames: list[str] | None = None, expected_rows: int | None = None):
    focus_csv = tmp_path / "focus.csv"
    output_csv = tmp_path / "validated.csv"
    output_json = tmp_path / "summary.json"
    _write_csv(focus_csv, rows, fieldnames=fieldnames)
    return preflight_focus_annotations(
        focus_annotations_csv=focus_csv,
        output_csv=output_csv,
        output_json=output_json,
        expected_rows=len(rows) if expected_rows is None else expected_rows,
    )


def test_focus_annotation_preflight_allows_blank_noop(tmp_path: Path):
    summary = _run(tmp_path, [_base_row()])
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    validated = list(csv.DictReader((tmp_path / "validated.csv").open("r", encoding="utf-8-sig", newline="")))

    assert summary == saved
    assert summary["decision"] == "ready_noop_no_focus_annotations"
    assert summary["ready_for_focus_merge"] is True
    assert summary["annotated_rows"] == 0
    assert summary["actionable_rows"] == 0
    assert validated[0]["loop109_status"] == "no_decision"


def test_focus_annotation_preflight_allows_content_evidence_actionable_row(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            _base_row(
                manual_label_verdict="feature_broken",
                manual_verdict_note="PE parse evidence and npz feature mismatch confirm broken extraction",
                recommended_action="replace_with_fresh_same_label_candidate",
            )
        ],
    )

    assert summary["decision"] == "ready_for_focus_merge"
    assert summary["ready_for_focus_merge"] is True
    assert summary["annotated_rows"] == 1
    assert summary["actionable_rows"] == 1
    assert summary["row_issue_counts"] == {}


def test_focus_annotation_preflight_blocks_identity_only_note(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            _base_row(
                manual_label_verdict="label_correct",
                manual_verdict_note="filename and source_path prove this should keep the label",
                recommended_action="model_blindspot",
            )
        ],
    )

    assert summary["decision"] == "blocked_invalid_focus_annotations"
    assert summary["ready_for_focus_merge"] is False
    assert "invalid_focus_annotation_rows" in summary["blockers"]
    assert summary["row_issue_counts"]["manual_verdict_note_missing_content_or_external_evidence"] == 1
    assert summary["row_issue_counts"]["manual_verdict_note_identity_or_score_only"] == 1


def test_focus_annotation_preflight_blocks_model_score_only_note(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            _base_row(
                manual_label_verdict="label_correct",
                manual_verdict_note="loop57 final probability is below threshold",
                recommended_action="model_blindspot",
            )
        ],
    )

    assert summary["ready_for_focus_merge"] is False
    assert summary["manual_quality"]["evidence_note_identity_or_score_only_rows"] == 1


def test_focus_annotation_preflight_blocks_forbidden_columns(tmp_path: Path):
    row = _base_row()
    row["source_sha256"] = "a" * 64
    row["loop57_final_prob"] = "0.8"
    fieldnames = [*FIELDS, "source_sha256", "loop57_final_prob"]

    summary = _run(tmp_path, [row], fieldnames=fieldnames)

    assert "focus_contains_identity_or_model_columns" in summary["blockers"]
    assert "source_sha256" in summary["forbidden_focus_columns"]
    assert "loop57_final_prob" in summary["forbidden_focus_columns"]


def test_focus_annotation_preflight_blocks_duplicate_blind_ids(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            _base_row(blind_review_id="blind-00001"),
            _base_row(blind_review_id="blind-00001"),
        ],
    )

    assert "duplicate_blind_review_id" in summary["blockers"]
    assert summary["duplicate_blind_review_id_rows"] == 1
