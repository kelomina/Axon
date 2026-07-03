import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from merge_loop106_focus_annotations import merge_focus_annotations  # noqa: E402


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


def _full_row(blind_id: str, label: str = "0") -> dict[str, str]:
    return {
        "blind_review_id": blind_id,
        "current_label": label,
        "review_tags": "",
        "file_entropy": "4.0",
        "pe_has_import_directory": "true",
        "manual_label_verdict": "",
        "manual_verdict_note": "",
        "recommended_action": "",
    }


def _focus_row(blind_id: str, **manual: str) -> dict[str, str]:
    row = {
        "blind_review_id": blind_id,
        "current_label": "0",
        "loop106_focus_rank": "1",
        "loop106_focus_score": "12.0",
        "loop106_focus_bucket": "benign_label_content_review",
        "loop106_focus_reasons": "overlay_present",
        "file_entropy": "7.7",
        "pe_has_import_directory": "true",
        "manual_label_verdict": "",
        "manual_verdict_note": "",
        "recommended_action": "",
    }
    row.update(manual)
    return row


def test_merge_loop106_focus_annotations_merges_only_manual_fields():
    with _case_dir("loop107_merge") as tmp_path:
        full_csv = tmp_path / "full.csv"
        focus_csv = tmp_path / "focus.csv"
        output_csv = tmp_path / "merged.csv"
        output_json = tmp_path / "summary.json"
        _write_csv(full_csv, [_full_row("blind-00001"), _full_row("blind-00002", "1")])
        _write_csv(
            focus_csv,
            [
                _focus_row(
                    "blind-00002",
                    manual_label_verdict="feature_broken",
                    manual_verdict_note="PE section header content is corrupt in external parser",
                    recommended_action="replace_with_fresh_same_label_candidate",
                )
            ],
        )

        summary = merge_focus_annotations(
            full_blinded_csv=full_csv,
            focus_annotations_csv=focus_csv,
            output_csv=output_csv,
            output_json=output_json,
            expected_full_rows=2,
            expected_focus_rows=1,
        )
        rows, fields = _read_csv(output_csv)

    assert summary["blockers"] == []
    assert summary["rows"]["merged_annotated_rows"] == 1
    assert "loop106_focus_score" not in fields
    assert rows[0]["manual_label_verdict"] == ""
    assert rows[1]["manual_label_verdict"] == "feature_broken"
    assert rows[1]["manual_verdict_note"] == "PE section header content is corrupt in external parser"
    assert rows[1]["recommended_action"] == "replace_with_fresh_same_label_candidate"


def test_merge_loop106_focus_annotations_blocks_identity_and_model_columns():
    with _case_dir("loop107_merge_forbidden") as tmp_path:
        full_csv = tmp_path / "full.csv"
        focus_csv = tmp_path / "focus.csv"
        output_csv = tmp_path / "merged.csv"
        output_json = tmp_path / "summary.json"
        _write_csv(full_csv, [_full_row("blind-00001")])
        row = _focus_row("blind-00001")
        row["source_sha256"] = "a" * 64
        row["loop57_final_prob"] = "0.9"
        _write_csv(focus_csv, [row])

        summary = merge_focus_annotations(
            full_blinded_csv=full_csv,
            focus_annotations_csv=focus_csv,
            output_csv=output_csv,
            output_json=output_json,
            expected_full_rows=1,
            expected_focus_rows=1,
        )

    assert "focus_contains_identity_or_model_columns" in summary["blockers"]
    assert "source_sha256" in summary["forbidden_focus_columns"]
    assert "loop57_final_prob" in summary["forbidden_focus_columns"]
    assert summary["decisions"]["ready_for_loop96_unblind"] is False


def test_merge_loop106_focus_annotations_blocks_duplicate_and_unknown_ids():
    with _case_dir("loop107_merge_bad_ids") as tmp_path:
        full_csv = tmp_path / "full.csv"
        focus_csv = tmp_path / "focus.csv"
        output_csv = tmp_path / "merged.csv"
        output_json = tmp_path / "summary.json"
        _write_csv(full_csv, [_full_row("blind-00001")])
        _write_csv(focus_csv, [_focus_row("blind-99999"), _focus_row("blind-99999")])

        summary = merge_focus_annotations(
            full_blinded_csv=full_csv,
            focus_annotations_csv=focus_csv,
            output_csv=output_csv,
            output_json=output_json,
            expected_full_rows=1,
            expected_focus_rows=2,
        )

    assert any(item.startswith("focus_duplicate_blind_review_id:blind-99999") for item in summary["blockers"])
    assert "focus_ids_missing_from_full_blinded_csv" in summary["blockers"]
    assert summary["unknown_focus_id_count"] == 1
