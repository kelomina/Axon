from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from import_loop111_focus_external_annotations import import_focus_external_annotations  # noqa: E402


FOCUS_FIELDS = [
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
EXTERNAL_FIELDS = [
    "blind_review_id",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _focus_row(blind_id: str, **manual: str) -> dict:
    row = {
        "blind_review_id": blind_id,
        "current_label": "1",
        "loop106_focus_rank": "1",
        "loop106_focus_score": "12.0",
        "loop106_focus_bucket": "malicious_label_content_review",
        "loop106_focus_reasons": "overlay_present",
        "file_entropy": "7.5",
        "pe_parse_status": "ok",
        "manual_label_verdict": "",
        "manual_verdict_note": "",
        "recommended_action": "",
    }
    row.update(manual)
    return row


def _external_row(blind_id: str, **manual: str) -> dict:
    row = {
        "blind_review_id": blind_id,
        "manual_label_verdict": "feature_broken",
        "manual_verdict_note": "PE parse evidence and npz feature mismatch confirm broken extraction",
        "recommended_action": "replace_with_fresh_same_label_candidate",
    }
    row.update(manual)
    return row


def _run(tmp_path: Path, *, focus_rows: list[dict], external_rows: list[dict], external_fields: list[str] | None = None):
    focus_csv = _write_csv(tmp_path / "focus.csv", focus_rows, FOCUS_FIELDS)
    external_csv = _write_csv(tmp_path / "external.csv", external_rows, external_fields or EXTERNAL_FIELDS)
    return import_focus_external_annotations(
        focus_csv=focus_csv,
        external_annotations=external_csv,
        output_csv=tmp_path / "annotated_focus.csv",
        output_json=tmp_path / "summary.json",
        preflight_output_csv=tmp_path / "preflight.csv",
        preflight_output_json=tmp_path / "preflight.json",
        expected_focus_rows=len(focus_rows),
    )


def test_loop111_noop_external_file_keeps_focus_ready(tmp_path: Path):
    summary = _run(
        tmp_path,
        focus_rows=[_focus_row("blind-00001"), _focus_row("blind-00002")],
        external_rows=[],
    )
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    rows = _read_csv(tmp_path / "annotated_focus.csv")

    assert summary == saved
    assert summary["decision"] == "ready_noop_no_external_annotations"
    assert summary["ready_for_loop110_focus_pipeline"] is True
    assert summary["blockers"] == []
    assert summary["counts"]["imported_rows"] == 0
    assert rows[0]["loop111_import_status"] == "not_targeted"
    assert summary["post_import_preflight"]["decision"] == "ready_noop_no_focus_annotations"


def test_loop111_blocks_empty_external_file_without_required_header(tmp_path: Path):
    focus_csv = _write_csv(tmp_path / "focus.csv", [_focus_row("blind-00001")], FOCUS_FIELDS)
    external_csv = tmp_path / "external.csv"
    external_csv.write_text("", encoding="utf-8")

    summary = import_focus_external_annotations(
        focus_csv=focus_csv,
        external_annotations=external_csv,
        output_csv=tmp_path / "annotated_focus.csv",
        output_json=tmp_path / "summary.json",
        preflight_output_csv=tmp_path / "preflight.csv",
        preflight_output_json=tmp_path / "preflight.json",
        expected_focus_rows=1,
    )

    assert summary["decision"] == "blocked_invalid_external_annotations"
    assert "external_missing_required_fields" in summary["blockers"]
    assert summary["external"]["missing_required_fields"] == EXTERNAL_FIELDS


def test_loop111_imports_valid_content_annotation_and_runs_preflight(tmp_path: Path):
    summary = _run(
        tmp_path,
        focus_rows=[_focus_row("blind-00001"), _focus_row("blind-00002")],
        external_rows=[_external_row("blind-00002")],
    )
    rows = _read_csv(tmp_path / "annotated_focus.csv")

    assert summary["decision"] == "ready_for_focus_verdict_pipeline"
    assert summary["blockers"] == []
    assert summary["counts"]["imported_rows"] == 1
    assert summary["post_import_preflight"]["decision"] == "ready_for_focus_merge"
    assert rows[0]["manual_label_verdict"] == ""
    assert rows[1]["manual_label_verdict"] == "feature_broken"
    assert rows[1]["loop111_import_status"] == "imported"


def test_loop111_imports_jsonl_annotations(tmp_path: Path):
    focus_csv = _write_csv(tmp_path / "focus.csv", [_focus_row("blind-00001")], FOCUS_FIELDS)
    external_jsonl = _write_jsonl(tmp_path / "external.jsonl", [_external_row("blind-00001")])

    summary = import_focus_external_annotations(
        focus_csv=focus_csv,
        external_annotations=external_jsonl,
        output_csv=tmp_path / "annotated_focus.csv",
        output_json=tmp_path / "summary.json",
        preflight_output_csv=tmp_path / "preflight.csv",
        preflight_output_json=tmp_path / "preflight.json",
        expected_focus_rows=1,
    )

    assert summary["inputs"]["input_format"] == "jsonl"
    assert summary["decision"] == "ready_for_focus_verdict_pipeline"
    assert summary["counts"]["imported_rows"] == 1


def test_loop111_blocks_identity_and_model_fields(tmp_path: Path):
    row = _external_row("blind-00001")
    row["source_sha256"] = "a" * 64
    row["loop57_final_prob"] = "0.95"
    summary = _run(
        tmp_path,
        focus_rows=[_focus_row("blind-00001")],
        external_rows=[row],
        external_fields=[*EXTERNAL_FIELDS, "source_sha256", "loop57_final_prob"],
    )

    assert summary["decision"] == "blocked_invalid_external_annotations"
    assert summary["ready_for_loop110_focus_pipeline"] is False
    assert "external_contains_unapproved_fields" in summary["blockers"]
    assert "external_contains_identity_or_model_fields" in summary["blockers"]
    assert "source_sha256" in summary["external"]["identity_or_model_fields"]
    assert "loop57_final_prob" in summary["external"]["identity_or_model_fields"]
    assert summary["post_import_preflight"] is None


def test_loop111_blocks_unknown_and_duplicate_blind_ids(tmp_path: Path):
    summary = _run(
        tmp_path,
        focus_rows=[_focus_row("blind-00001")],
        external_rows=[
            _external_row("blind-99999"),
            _external_row("blind-99999"),
        ],
    )

    assert "external_duplicate_blind_review_id" in summary["blockers"]
    assert "external_ids_missing_from_focus_csv" in summary["blockers"]
    assert summary["external"]["unknown_blind_review_id_count"] == 1
    assert summary["external"]["duplicate_blind_review_id_rows"] == 1


def test_loop111_blocks_identity_or_model_score_only_note_via_post_preflight(tmp_path: Path):
    summary = _run(
        tmp_path,
        focus_rows=[_focus_row("blind-00001")],
        external_rows=[
            _external_row(
                "blind-00001",
                manual_label_verdict="label_correct",
                manual_verdict_note="filename and loop57 probability prove this",
                recommended_action="model_blindspot",
            )
        ],
    )

    assert summary["decision"] == "blocked_invalid_external_annotations"
    assert "post_import_focus_preflight_not_ready" in summary["blockers"]
    assert summary["post_import_preflight"]["ready_for_focus_merge"] is False
    assert summary["post_import_preflight"]["manual_quality"]["evidence_note_identity_or_score_only_rows"] == 1


def test_loop111_blocks_blank_external_manual_fields(tmp_path: Path):
    summary = _run(
        tmp_path,
        focus_rows=[_focus_row("blind-00001")],
        external_rows=[
            {
                "blind_review_id": "blind-00001",
                "manual_label_verdict": "",
                "manual_verdict_note": "",
                "recommended_action": "",
            }
        ],
    )

    assert "external_annotation_rows_have_blank_manual_fields" in summary["blockers"]
    assert summary["ready_for_loop110_focus_pipeline"] is False
