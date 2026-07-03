import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop106_content_review_focus import build_focus  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _base_row(blind_id: str, label: str, **overrides: str) -> dict[str, str]:
    row = {
        "blind_review_id": blind_id,
        "current_label": label,
        "review_tags": "",
        "source_size_bytes": "1000000",
        "file_entropy": "4.0",
        "pe_parse_status": "ok",
        "pe_number_of_sections": "5",
        "pe_section_names": ".text|.rdata|.data",
        "pe_has_import_directory": "true",
        "pe_import_directory_size": "2048",
        "pe_has_resource_directory": "false",
        "pe_resource_directory_size": "0",
        "pe_has_security_directory": "false",
        "pe_security_directory_size": "0",
        "overlay_size": "0",
        "overlay_entropy": "",
        "overlay_after_security_size": "0",
        "overlay_after_security_entropy": "",
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


def test_loop106_focus_ranks_content_without_identity_fields():
    with _case_dir("loop106_focus") as tmp_path:
        input_csv = tmp_path / "blinded.csv"
        output_csv = tmp_path / "focus.csv"
        output_json = tmp_path / "focus.json"
        _write_csv(
            input_csv,
            [
                _base_row("blind-00001", "0"),
                _base_row(
                    "blind-00002",
                    "0",
                    review_tags="overlay_present|high_overlay_entropy",
                    content_evidence_fields="source_size_bytes|source_sha256|cache_path|file_entropy|sample_index|overlay_size",
                    file_entropy="7.6",
                    overlay_size="200000",
                    overlay_entropy="7.8",
                    pe_number_of_sections="9",
                    pe_has_security_directory="true",
                    pe_security_directory_size="4096",
                    overlay_after_security_size="1024",
                    overlay_after_security_entropy="7.2",
                ),
                _base_row(
                    "blind-00003",
                    "1",
                    duplicate_manifest_sha_group="true",
                    manifest_duplicate_group_size="2",
                ),
            ],
        )

        summary = build_focus(
            blinded_csv=input_csv,
            output_csv=output_csv,
            output_json=output_json,
            max_rows=2,
            require_expected_rows=3,
        )
        rows, fields = _read_csv(output_csv)

    assert summary["blockers"] == []
    assert summary["decisions"]["ready_for_independent_content_review"] is True
    assert len(rows) == 2
    assert rows[0]["blind_review_id"] == "blind-00002"
    assert "benign_label_malware_like_static_shape" in rows[0]["loop106_focus_reasons"]
    assert rows[0]["content_evidence_fields"] == "source_size_bytes|file_entropy|overlay_size"
    assert "source_path" not in fields
    assert "source_sha256" not in fields
    assert "sample_index" not in fields
    assert "loop57_final_prob" not in fields
    for row in rows:
        for forbidden in ["source_sha256", "cache_path", "source_path", "sample_index"]:
            assert forbidden not in row.get("content_evidence_fields", "")
    assert summary["selected_rows"] == 2


def test_loop106_focus_blocks_identity_or_model_columns():
    with _case_dir("loop106_focus_forbidden") as tmp_path:
        input_csv = tmp_path / "bad_blinded.csv"
        output_csv = tmp_path / "focus.csv"
        output_json = tmp_path / "focus.json"
        row = _base_row("blind-00001", "0")
        row["source_sha256"] = "a" * 64
        row["loop57_final_prob"] = "0.91"
        _write_csv(input_csv, [row])

        summary = build_focus(
            blinded_csv=input_csv,
            output_csv=output_csv,
            output_json=output_json,
            max_rows=1,
            require_expected_rows=1,
        )

    assert "input_contains_identity_or_model_columns" in summary["blockers"]
    assert "source_sha256" in summary["forbidden_input_columns"]
    assert "loop57_final_prob" in summary["forbidden_input_columns"]
    assert summary["decisions"]["ready_for_independent_content_review"] is False


def test_loop106_focus_detects_row_count_mismatch():
    with _case_dir("loop106_focus_row_count") as tmp_path:
        input_csv = tmp_path / "blinded.csv"
        output_csv = tmp_path / "focus.csv"
        output_json = tmp_path / "focus.json"
        _write_csv(input_csv, [_base_row("blind-00001", "1")])

        summary = build_focus(
            blinded_csv=input_csv,
            output_csv=output_csv,
            output_json=output_json,
            max_rows=1,
            require_expected_rows=2,
        )

    assert "input_row_count_mismatch_expected" in summary["blockers"]
