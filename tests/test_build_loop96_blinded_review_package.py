from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_loop96_blinded_review_package import build_blinded_package, unblind_verdicts
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
    "review_tags",
    "content_evidence_fields",
    "source_exists",
    "source_size_bytes",
    "source_sha256_match",
    "file_entropy",
    "mz_signature",
    "pe_parse_status",
    "pe_number_of_sections",
    "pe_section_names",
    "overlay_size",
    "overlay_entropy",
    "identity_columns_are_not_evidence",
    "model_score_columns_are_not_verdict_evidence",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(sample_index: str, *, label: str = "1") -> dict:
    return {
        "review_batch_rank": sample_index,
        "review_category": "c_high_conflict_persistent_error",
        "source_path": f"data/named/{sample_index}.exe",
        "cache_path": f"data/.cache/{sample_index}.npz",
        "source_sha256": sample_index.zfill(64)[-64:],
        "sample_index": sample_index,
        "split": "test",
        "label": label,
        "loop57_error_type": "FN" if label == "1" else "FP",
        "loop57_final_prob": "0.01",
        "review_tags": "overlay_present|high_overlay_entropy",
        "content_evidence_fields": "source_size_bytes|pe_parse_status|overlay_entropy",
        "source_exists": "true",
        "source_size_bytes": "12345",
        "source_sha256_match": "true",
        "file_entropy": "6.5",
        "mz_signature": "true",
        "pe_parse_status": "ok",
        "pe_number_of_sections": "5",
        "pe_section_names": ".text|.rdata|.rsrc",
        "overlay_size": "1024",
        "overlay_entropy": "7.4",
        "identity_columns_are_not_evidence": "true",
        "model_score_columns_are_not_verdict_evidence": "true",
        "manual_label_verdict": "",
        "manual_verdict_note": "",
        "recommended_action": "",
    }


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def test_loop96_builds_blinded_review_csv_without_identity_or_scores(tmp_path: Path):
    input_csv = _write_csv(tmp_path / "intake.csv", [_row("101"), _row("102", label="0")])
    summary = build_blinded_package(
        input_csv=input_csv,
        blinded_csv=tmp_path / "blinded.csv",
        private_map_csv=tmp_path / "private.csv",
        output_json=tmp_path / "summary.json",
        expected_rows=2,
        seed=1,
    )
    blinded_rows, blinded_fields = _read_rows(tmp_path / "blinded.csv")
    private_rows, private_fields = _read_rows(tmp_path / "private.csv")

    forbidden = {
        "source_path",
        "cache_path",
        "source_sha256",
        "sample_index",
        "split",
        "review_batch_rank",
        "loop57_error_type",
        "loop57_final_prob",
    }
    assert summary["blockers"] == []
    assert summary["decisions"]["ready_for_blinded_review"] is True
    assert not forbidden.intersection(blinded_fields)
    assert {"blind_review_id", "current_label", "pe_parse_status", "manual_label_verdict"}.issubset(blinded_fields)
    assert {"source_path", "source_sha256", "sample_index", "loop57_final_prob"}.issubset(private_fields)
    assert len(blinded_rows) == 2
    assert len(private_rows) == 2
    assert all(row["manual_label_verdict"] == "" for row in blinded_rows)


def test_loop96_unblind_restores_loop87_ready_csv(tmp_path: Path):
    input_csv = _write_csv(tmp_path / "intake.csv", [_row("101"), _row("102", label="0")])
    build_blinded_package(
        input_csv=input_csv,
        blinded_csv=tmp_path / "blinded.csv",
        private_map_csv=tmp_path / "private.csv",
        output_json=tmp_path / "build.json",
        expected_rows=2,
        seed=1,
    )
    blinded_rows, blinded_fields = _read_rows(tmp_path / "blinded.csv")
    blinded_rows[0]["manual_label_verdict"] = "label_correct"
    blinded_rows[0]["manual_verdict_note"] = "PE content evidence shows valid overlay and resource structure; external review kept label"
    blinded_rows[0]["recommended_action"] = "model_blindspot"
    _write_csv(tmp_path / "annotated.csv", blinded_rows, fieldnames=blinded_fields)

    unblind_summary = unblind_verdicts(
        annotated_blinded_csv=tmp_path / "annotated.csv",
        private_map_csv=tmp_path / "private.csv",
        output_csv=tmp_path / "loop87_ready.csv",
        output_json=tmp_path / "unblind.json",
        expected_rows=2,
    )
    loop87_summary = validate_loop86_verdicts(
        evidence_csv=tmp_path / "loop87_ready.csv",
        output_csv=tmp_path / "validated.csv",
        output_json=tmp_path / "loop87.json",
        expected_rows=2,
    )

    assert unblind_summary["blockers"] == []
    assert unblind_summary["decisions"]["ready_for_loop87_import"] is True
    assert loop87_summary["import_ready"] is True
    assert loop87_summary["decision"] == "ready_for_redraw_plan_review_only"
    assert loop87_summary["status_counts"] == {"label_correct_model_blindspot": 1, "no_decision": 1}
    assert loop87_summary["replacement_required_rows"] == 0
    assert loop87_summary["training_policy_rows"] == 0


def test_loop96_unblind_blocks_missing_private_mapping(tmp_path: Path):
    input_csv = _write_csv(tmp_path / "intake.csv", [_row("101"), _row("102")])
    build_blinded_package(
        input_csv=input_csv,
        blinded_csv=tmp_path / "blinded.csv",
        private_map_csv=tmp_path / "private.csv",
        output_json=tmp_path / "build.json",
        expected_rows=2,
        seed=1,
    )
    blinded_rows, blinded_fields = _read_rows(tmp_path / "blinded.csv")
    private_rows, private_fields = _read_rows(tmp_path / "private.csv")
    private_rows.pop()
    _write_csv(tmp_path / "private_short.csv", private_rows, fieldnames=private_fields)

    summary = unblind_verdicts(
        annotated_blinded_csv=tmp_path / "blinded.csv",
        private_map_csv=tmp_path / "private_short.csv",
        output_csv=tmp_path / "loop87_ready.csv",
        output_json=tmp_path / "unblind.json",
        expected_rows=2,
    )

    assert "annotated_private_row_count_mismatch" in summary["blockers"]
    assert "annotated_ids_missing_from_private_map" in summary["blockers"]
    assert summary["decisions"]["ready_for_loop87_import"] is False
