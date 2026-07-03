from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_loop113_external_focus_annotation_package import (  # noqa: E402
    ANNOTATION_FIELDS,
    export_external_focus_annotation_package,
)


FOCUS_FIELDS = [
    "blind_review_id",
    "current_label",
    "loop106_focus_rank",
    "loop106_focus_score",
    "loop106_focus_bucket",
    "loop106_focus_reasons",
    "review_tags",
    "content_evidence_fields",
    "source_size_bytes",
    "file_entropy",
    "pe_parse_status",
    "pe_number_of_sections",
    "pe_section_names",
    "pe_has_import_directory",
    "pe_import_directory_size",
    "pe_has_resource_directory",
    "pe_resource_directory_size",
    "pe_has_security_directory",
    "pe_security_directory_size",
    "overlay_size",
    "overlay_entropy",
    "overlay_after_security_size",
    "overlay_after_security_entropy",
    "duplicate_manifest_sha_group",
    "manifest_duplicate_group_size",
    "objective_issue_count",
    "objective_issue_flags",
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


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _focus_row(blind_id: str, **overrides: str) -> dict:
    row = {
        "blind_review_id": blind_id,
        "current_label": "0",
        "loop106_focus_rank": "1",
        "loop106_focus_score": "114.000000",
        "loop106_focus_bucket": "benign_label_content_review",
        "loop106_focus_reasons": "overlay_present|high_overlay_entropy",
        "review_tags": "overlay_present|high_overlay_entropy",
        "content_evidence_fields": "source_size_bytes|file_entropy|pe_parse_status|overlay_size",
        "source_size_bytes": "23430594",
        "file_entropy": "7.993970",
        "pe_parse_status": "ok",
        "pe_number_of_sections": "11",
        "pe_section_names": ".text|.rdata|.rsrc",
        "pe_has_import_directory": "true",
        "pe_import_directory_size": "4160",
        "pe_has_resource_directory": "true",
        "pe_resource_directory_size": "18576",
        "pe_has_security_directory": "false",
        "pe_security_directory_size": "0",
        "overlay_size": "22600130",
        "overlay_entropy": "7.999991",
        "overlay_after_security_size": "22600130",
        "overlay_after_security_entropy": "7.999991",
        "duplicate_manifest_sha_group": "false",
        "manifest_duplicate_group_size": "",
        "objective_issue_count": "0",
        "objective_issue_flags": "",
        "manual_label_verdict": "",
        "manual_verdict_note": "",
        "recommended_action": "",
    }
    row.update(overrides)
    return row


def _run(tmp_path: Path, rows: list[dict], *, fieldnames: list[str] | None = None):
    focus_csv = _write_csv(tmp_path / "focus.csv", rows, fieldnames or FOCUS_FIELDS)
    return export_external_focus_annotation_package(
        focus_csv=focus_csv,
        context_csv=tmp_path / "context.csv",
        annotation_template_csv=tmp_path / "template.csv",
        reviewer_guide_json=tmp_path / "guide.json",
        output_json=tmp_path / "summary.json",
        expected_focus_rows=len(rows),
    )


def test_loop113_exports_context_and_header_only_template(tmp_path: Path):
    summary = _run(tmp_path, [_focus_row("blind-00001"), _focus_row("blind-00002", current_label="1")])
    context_rows, context_fields = _read_csv(tmp_path / "context.csv")
    template_rows, template_fields = _read_csv(tmp_path / "template.csv")
    guide = json.loads((tmp_path / "guide.json").read_text(encoding="utf-8"))

    assert summary["decision"] == "ready_for_external_content_annotation"
    assert summary["blockers"] == []
    assert len(context_rows) == 2
    assert template_rows == []
    assert template_fields == ANNOTATION_FIELDS
    assert guide["annotation_fields"] == ANNOTATION_FIELDS
    assert "blind_review_id" in context_fields
    assert "loop106_focus_reasons" in context_fields
    assert "loop106_focus_rank" not in context_fields
    assert "loop106_focus_score" not in context_fields
    assert "manual_label_verdict" not in context_fields
    for forbidden in ["source_path", "source_sha256", "sample_index", "split", "loop57_final_prob", "prediction", "threshold"]:
        assert forbidden not in context_fields
    assert summary["decisions"]["training_allowed"] is False
    assert summary["decisions"]["test10k_allowed"] is False


def test_loop113_blocks_focus_with_identity_or_model_columns(tmp_path: Path):
    row = _focus_row("blind-00001")
    row["source_sha256"] = "a" * 64
    row["loop57_final_prob"] = "0.7"
    summary = _run(
        tmp_path,
        [row],
        fieldnames=[*FOCUS_FIELDS, "source_sha256", "loop57_final_prob"],
    )

    assert summary["decision"] == "blocked_invalid_focus_package"
    assert "focus_contains_identity_or_model_columns" in summary["blockers"]
    assert "source_sha256" in summary["field_audit"]["forbidden_focus_columns"]
    assert "loop57_final_prob" in summary["field_audit"]["forbidden_focus_columns"]


def test_loop113_blocks_context_values_referencing_identity_or_model_terms(tmp_path: Path):
    summary = _run(
        tmp_path,
        [
            _focus_row(
                "blind-00001",
                loop106_focus_reasons="overlay_present|loop57_probability",
            )
        ],
    )

    assert summary["decision"] == "blocked_invalid_focus_package"
    assert "context_output_values_reference_identity_or_model_terms" in summary["blockers"]
    assert summary["field_audit"]["context_value_violation_count"] == 1
    assert summary["field_audit"]["context_value_violation_examples"][0]["field"] == "loop106_focus_reasons"


def test_loop113_detects_duplicate_blind_ids(tmp_path: Path):
    summary = _run(tmp_path, [_focus_row("blind-00001"), _focus_row("blind-00001")])

    assert "duplicate_blind_review_id" in summary["blockers"]
    assert summary["field_audit"]["duplicate_blind_review_id_rows"] == 1
