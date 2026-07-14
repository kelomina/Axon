import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop126_review_package import build_loop126_review_template  # noqa: E402
from preflight_loop126_review_annotations import preflight_loop126_review_annotations  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_focus(path: Path, rows: list[dict], *, include_forbidden: bool = False) -> None:
    fieldnames = [
        "review_focus_id",
        "focus_rank",
        "priority_band",
        "split",
        "current_label",
        "error_type",
        "error_transition",
        "triage_confidence_bucket",
        "review_lane",
        "content_signal_count",
        "content_tags",
        "recommended_review_action",
        "pe_schema_version",
        "section_entropy_max",
        "rwx_sections_ratio",
        "api_network_ratio",
        "stat_byte_entropy",
    ]
    if include_forbidden:
        fieldnames.append("source_sha256")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _focus_row(**overrides):
    row = {
        "review_focus_id": "loop126_val_focus_000001",
        "focus_rank": "1",
        "priority_band": "critical",
        "split": "val",
        "current_label": "1",
        "error_type": "FN",
        "error_transition": "persistent_error",
        "triage_confidence_bucket": "very_high",
        "review_lane": "model_blindspot_review",
        "content_signal_count": "3",
        "content_tags": "high_section_entropy|rwx_section_present|mixed_executable_writable_sections",
        "recommended_review_action": "review_content_evidence_without_identity_fields",
        "pe_schema_version": "fixed_v2",
        "section_entropy_max": "0.91",
        "rwx_sections_ratio": "0.5",
        "api_network_ratio": "0.2",
        "stat_byte_entropy": "0.82",
    }
    row.update(overrides)
    return row


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_build_loop126_review_template_adds_manual_fields_without_identity_columns():
    with _case_dir("loop126_review_template") as tmp_path:
        focus = tmp_path / "focus.csv"
        annotations = tmp_path / "annotations.csv"
        summary = tmp_path / "template.json"
        _write_focus(focus, [_focus_row()])

        payload = build_loop126_review_template(
            focus_blinded_csv=focus,
            output_annotations_csv=annotations,
            output_json=summary,
            expected_rows=1,
        )
        rows = _read_rows(annotations)

    assert payload["template_ready"] is True
    assert rows[0]["manual_label_verdict"] == ""
    assert rows[0]["manual_verdict_note"] == ""
    assert rows[0]["recommended_action"] == ""
    assert "source_sha256" not in rows[0]
    assert "sample_index" not in rows[0]


def test_build_loop126_review_template_rejects_public_identity_columns():
    with _case_dir("loop126_review_template_forbidden") as tmp_path:
        focus = tmp_path / "focus.csv"
        annotations = tmp_path / "annotations.csv"
        summary = tmp_path / "template.json"
        _write_focus(focus, [_focus_row(source_sha256="a" * 64)], include_forbidden=True)

        payload = build_loop126_review_template(
            focus_blinded_csv=focus,
            output_annotations_csv=annotations,
            output_json=summary,
            expected_rows=1,
        )

    assert payload["template_ready"] is False
    assert "focus_contains_identity_or_model_columns" in payload["blockers"]


def test_build_loop126_review_template_allows_resource_feature_columns():
    with _case_dir("loop126_review_template_resource_columns") as tmp_path:
        focus = tmp_path / "focus.csv"
        annotations = tmp_path / "annotations.csv"
        summary = tmp_path / "template.json"
        row = _focus_row(
            content_resource_entry_count_log="4.2",
            v2_resource_type_version_count_log="1.0",
            string_version_resource_count_log="2.0",
        )
        fieldnames = [
            "review_focus_id",
            "focus_rank",
            "priority_band",
            "current_label",
            "error_type",
            "review_lane",
            "content_resource_entry_count_log",
            "v2_resource_type_version_count_log",
            "string_version_resource_count_log",
        ]
        with focus.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)

        payload = build_loop126_review_template(
            focus_blinded_csv=focus,
            output_annotations_csv=annotations,
            output_json=summary,
            expected_rows=1,
        )

    assert payload["template_ready"] is True
    assert payload["forbidden_columns"] == []


def test_loop126_preflight_accepts_content_evidence_verdicts():
    with _case_dir("loop126_review_preflight_good") as tmp_path:
        annotations = tmp_path / "annotations.csv"
        validated = tmp_path / "validated.csv"
        summary = tmp_path / "preflight.json"
        row = _focus_row(
            manual_label_verdict="label_correct",
            manual_verdict_note="PE section entropy and rwx section evidence match malicious content; API surface is sparse.",
            recommended_action="model_blindspot",
        )
        _write_focus_with_manual(annotations, [row])

        payload = preflight_loop126_review_annotations(
            annotations_csv=annotations,
            output_csv=validated,
            output_json=summary,
            expected_rows=1,
        )
        rows = _read_rows(validated)

    assert payload["ready_for_private_mapping"] is True
    assert payload["actionable_rows"] == 1
    assert rows[0]["loop126_status"] == "label_correct_model_blindspot"
    assert rows[0]["loop126_training_policy_allowed"] == "false"


def test_loop126_preflight_allows_resource_feature_columns():
    with _case_dir("loop126_review_preflight_resource_columns") as tmp_path:
        annotations = tmp_path / "annotations.csv"
        validated = tmp_path / "validated.csv"
        summary = tmp_path / "preflight.json"
        row = _focus_row(
            content_resource_entry_count_log="4.2",
            v2_resource_type_version_count_log="1.0",
            manual_label_verdict="label_correct",
            manual_verdict_note="PE resource table and section entropy evidence support the current label.",
            recommended_action="model_blindspot",
        )
        fieldnames = [
            "review_focus_id",
            "focus_rank",
            "priority_band",
            "current_label",
            "error_type",
            "review_lane",
            "content_resource_entry_count_log",
            "v2_resource_type_version_count_log",
            "manual_label_verdict",
            "manual_verdict_note",
            "recommended_action",
        ]
        with annotations.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)

        payload = preflight_loop126_review_annotations(
            annotations_csv=annotations,
            output_csv=validated,
            output_json=summary,
            expected_rows=1,
        )

    assert payload["ready_for_private_mapping"] is True
    assert payload["forbidden_columns"] == []


def test_loop126_preflight_rejects_identity_or_score_only_notes():
    with _case_dir("loop126_review_preflight_bad_note") as tmp_path:
        annotations = tmp_path / "annotations.csv"
        validated = tmp_path / "validated.csv"
        summary = tmp_path / "preflight.json"
        row = _focus_row(
            manual_label_verdict="label_wrong",
            manual_verdict_note="Wrong because source_sha256 and model probability are suspicious.",
            recommended_action="replace_with_fresh_same_label_candidate",
        )
        _write_focus_with_manual(annotations, [row])

        payload = preflight_loop126_review_annotations(
            annotations_csv=annotations,
            output_csv=validated,
            output_json=summary,
            expected_rows=1,
        )
        rows = _read_rows(validated)

    assert payload["ready_for_private_mapping"] is False
    assert "manual_annotation_quality_issues" in payload["blockers"]
    assert "manual_verdict_note_identity_or_score_only" in rows[0]["loop126_issue_flags"]


def test_loop126_preflight_replacement_uses_current_label_only():
    with _case_dir("loop126_review_preflight_replacement") as tmp_path:
        annotations = tmp_path / "annotations.csv"
        validated = tmp_path / "validated.csv"
        summary = tmp_path / "preflight.json"
        row = _focus_row(
            current_label="0",
            manual_label_verdict="feature_broken",
            manual_verdict_note="NPZ feature shape mismatch and PE parse evidence show broken extracted content.",
            recommended_action="replace_with_fresh_same_label_candidate",
        )
        _write_focus_with_manual(annotations, [row])

        payload = preflight_loop126_review_annotations(
            annotations_csv=annotations,
            output_csv=validated,
            output_json=summary,
            expected_rows=1,
        )
        rows = _read_rows(validated)

    assert payload["ready_for_private_mapping"] is True
    assert payload["replacement_required_rows"] == 1
    assert payload["replacement_label_counts"] == {"0": 1}
    assert rows[0]["loop126_plan_action"] == "quarantine_and_replace_fresh_same_original_label"


def _write_focus_with_manual(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "review_focus_id",
        "focus_rank",
        "priority_band",
        "split",
        "current_label",
        "error_type",
        "error_transition",
        "triage_confidence_bucket",
        "review_lane",
        "content_signal_count",
        "content_tags",
        "recommended_review_action",
        "pe_schema_version",
        "section_entropy_max",
        "rwx_sections_ratio",
        "api_network_ratio",
        "stat_byte_entropy",
        "manual_label_verdict",
        "manual_verdict_note",
        "recommended_action",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
