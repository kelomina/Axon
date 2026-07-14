from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_loop157_current_best_val_external_annotation_package import (  # noqa: E402
    ANNOTATION_FIELDS,
    export_loop157_package,
)


FIELDS = [
    "review_focus_id",
    "focus_rank",
    "priority_band",
    "current_label",
    "error_type",
    "review_lane",
    "content_signal_count",
    "content_tags",
    "recommended_review_action",
    "content_is_dll",
    "content_export_count_log",
    "content_dir_export_log_size",
    "content_dir_security_log_size",
    "content_overlay_log_size",
    "content_resource_entry_count_log",
    "content_resource_type_count_log",
    "content_dir_resource_size_ratio",
    "content_dir_resource_log_size",
    "content_overlay_entropy",
    "content_import_api_count_log",
    "content_avg_imports_per_dll",
    "content_image_base_log",
    "v2_resource_data_entry_count_log",
    "v2_resource_type_icon_count_log",
    "v2_resource_type_version_count_log",
    "v2_resource_type_manifest_count_log",
    "v2_resource_type_dialog_count_log",
    "v2_last_section_entropy",
    "v2_section_max_virtual_raw_ratio_log",
    "v2_api_file_mutation_ratio",
    "v2_import_dll_version_api_ratio",
    "string_benign_vendor_count_log",
    "string_version_resource_count_log",
    "string_script_exec_count_log",
    "string_script_exec_present",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
]


def _row(review_id: str, **overrides: str) -> dict[str, str]:
    row = {
        "review_focus_id": review_id,
        "focus_rank": "1",
        "priority_band": "high",
        "current_label": "0",
        "error_type": "fp",
        "review_lane": "benign_trust_or_label_quality_review",
        "content_signal_count": "2",
        "content_tags": "resource_rich|version_resource_present",
        "recommended_review_action": "review_content_or_external_evidence_without_identity_fields",
        "content_is_dll": "0",
        "content_export_count_log": "0",
        "content_dir_export_log_size": "0",
        "content_dir_security_log_size": "1",
        "content_overlay_log_size": "0",
        "content_resource_entry_count_log": "4",
        "content_resource_type_count_log": "2",
        "content_dir_resource_size_ratio": "0.1",
        "content_dir_resource_log_size": "7",
        "content_overlay_entropy": "0",
        "content_import_api_count_log": "5",
        "content_avg_imports_per_dll": "3",
        "content_image_base_log": "0",
        "v2_resource_data_entry_count_log": "4",
        "v2_resource_type_icon_count_log": "3",
        "v2_resource_type_version_count_log": "1",
        "v2_resource_type_manifest_count_log": "1",
        "v2_resource_type_dialog_count_log": "0",
        "v2_last_section_entropy": "0.5",
        "v2_section_max_virtual_raw_ratio_log": "2",
        "v2_api_file_mutation_ratio": "0.01",
        "v2_import_dll_version_api_ratio": "0",
        "string_benign_vendor_count_log": "2",
        "string_version_resource_count_log": "1",
        "string_script_exec_count_log": "0",
        "string_script_exec_present": "0",
        "manual_label_verdict": "",
        "manual_verdict_note": "",
        "recommended_action": "",
    }
    row.update(overrides)
    return row


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _run(tmp_path: Path, rows: list[dict[str, str]], *, fieldnames: list[str] | None = None):
    review_csv = _write_csv(tmp_path / "review.csv", rows, fieldnames or FIELDS)
    return export_loop157_package(
        review_csv=review_csv,
        context_csv=tmp_path / "context.csv",
        annotation_template_csv=tmp_path / "template.csv",
        reviewer_guide_json=tmp_path / "guide.json",
        output_json=tmp_path / "summary.json",
        expected_rows=len(rows),
    )


def test_loop157_exports_context_and_header_only_template(tmp_path: Path):
    summary = _run(tmp_path, [_row("loop156_000001"), _row("loop156_000002", current_label="1", error_type="fn")])
    context_rows, context_fields = _read_csv(tmp_path / "context.csv")
    template_rows, template_fields = _read_csv(tmp_path / "template.csv")
    guide = json.loads((tmp_path / "guide.json").read_text(encoding="utf-8"))

    assert summary["decision"] == "ready_for_external_content_annotation"
    assert summary["blockers"] == []
    assert len(context_rows) == 2
    assert template_rows == []
    assert template_fields == ANNOTATION_FIELDS
    assert guide["annotation_fields"] == ANNOTATION_FIELDS
    assert "review_focus_id" in context_fields
    assert "content_tags" in context_fields
    assert "focus_rank" not in context_fields
    assert "manual_label_verdict" not in context_fields
    for forbidden in ["source_path", "source_sha256", "sample_index", "prediction", "prob_malicious", "nearest_similarity"]:
        assert forbidden not in context_fields
    assert summary["decisions"]["training_allowed"] is False
    assert summary["decisions"]["test10k_allowed"] is False


def test_loop157_blocks_identity_or_model_input_columns(tmp_path: Path):
    row = _row("loop156_000001")
    row["source_sha256"] = "a" * 64
    row["prob_malicious"] = "0.9"
    summary = _run(tmp_path, [row], fieldnames=[*FIELDS, "source_sha256", "prob_malicious"])

    assert summary["decision"] == "blocked_invalid_external_package"
    assert "review_csv_contains_identity_or_model_columns" in summary["blockers"]
    assert "source_sha256" in summary["field_audit"]["forbidden_input_columns"]
    assert "prob_malicious" in summary["field_audit"]["forbidden_input_columns"]


def test_loop157_blocks_context_values_with_identity_or_model_terms(tmp_path: Path):
    summary = _run(tmp_path, [_row("loop156_000001", content_tags="overlay|model score")])

    assert summary["decision"] == "blocked_invalid_external_package"
    assert "context_output_values_reference_identity_or_model_terms" in summary["blockers"]
    assert summary["field_audit"]["context_value_violation_count"] == 1
    assert summary["field_audit"]["context_value_violation_examples"][0]["field"] == "content_tags"


def test_loop157_detects_duplicate_review_ids(tmp_path: Path):
    summary = _run(tmp_path, [_row("loop156_000001"), _row("loop156_000001")])

    assert "duplicate_review_focus_id" in summary["blockers"]
    assert summary["field_audit"]["duplicate_review_focus_id_rows"] == 1
