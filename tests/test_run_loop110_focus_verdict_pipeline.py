from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop96_blinded_review_package import build_blinded_package  # noqa: E402
from run_loop110_focus_verdict_pipeline import run_focus_verdict_pipeline  # noqa: E402


INTAKE_FIELDS = [
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


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _intake_row(sample_index: str, *, label: str = "1") -> dict:
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


def _case(tmp_path: Path):
    intake_csv = _write_csv(tmp_path / "intake.csv", [_intake_row("101"), _intake_row("102", label="0")], INTAKE_FIELDS)
    build_blinded_package(
        input_csv=intake_csv,
        blinded_csv=tmp_path / "full_blinded.csv",
        private_map_csv=tmp_path / "private_map.csv",
        output_json=tmp_path / "build.json",
        expected_rows=2,
        seed=1,
    )
    blinded_rows, _fields = _read_rows(tmp_path / "full_blinded.csv")
    return {
        "full_blinded_csv": tmp_path / "full_blinded.csv",
        "private_map_csv": tmp_path / "private_map.csv",
        "blind_ids": [row["blind_review_id"] for row in blinded_rows],
    }


def _run(tmp_path: Path, focus_rows: list[dict], *, fieldnames: list[str] | None = None):
    case = _case(tmp_path)
    focus_csv = _write_csv(tmp_path / "focus.csv", focus_rows, fieldnames or FOCUS_FIELDS)
    return run_focus_verdict_pipeline(
        full_blinded_csv=case["full_blinded_csv"],
        focus_annotations_csv=focus_csv,
        private_map_csv=case["private_map_csv"],
        output_dir=tmp_path / "loop110",
        output_json=tmp_path / "loop110_summary.json",
        expected_full_rows=2,
        expected_focus_rows=len(focus_rows),
    )


def test_loop110_runs_noop_focus_pipeline(tmp_path: Path):
    case = _case(tmp_path)
    focus_csv = _write_csv(
        tmp_path / "focus.csv",
        [_focus_row(case["blind_ids"][0]), _focus_row(case["blind_ids"][1])],
        FOCUS_FIELDS,
    )
    summary = run_focus_verdict_pipeline(
        full_blinded_csv=case["full_blinded_csv"],
        focus_annotations_csv=focus_csv,
        private_map_csv=case["private_map_csv"],
        output_dir=tmp_path / "loop110",
        output_json=tmp_path / "loop110_summary.json",
        expected_full_rows=2,
        expected_focus_rows=2,
    )
    saved = json.loads((tmp_path / "loop110_summary.json").read_text(encoding="utf-8"))

    assert summary == saved
    assert summary["decision"] == "ready_noop_no_actionable_verdicts"
    assert summary["blockers"] == []
    assert summary["stages"]["loop87_import"]["passed"] is True
    assert summary["counts"]["loop87_actionable_rows"] == 0
    assert summary["decisions"]["training_allowed"] is False
    assert Path(summary["outputs"]["loop87_json"]).exists()


def test_loop110_stops_before_merge_when_focus_preflight_blocks(tmp_path: Path):
    case = _case(tmp_path)
    focus_csv = _write_csv(
        tmp_path / "focus.csv",
        [
            _focus_row(
                case["blind_ids"][0],
                manual_label_verdict="label_correct",
                manual_verdict_note="filename and loop57 probability prove this",
                recommended_action="model_blindspot",
            )
        ],
        FOCUS_FIELDS,
    )
    summary = run_focus_verdict_pipeline(
        full_blinded_csv=case["full_blinded_csv"],
        focus_annotations_csv=focus_csv,
        private_map_csv=case["private_map_csv"],
        output_dir=tmp_path / "loop110",
        output_json=tmp_path / "loop110_summary.json",
        expected_full_rows=2,
        expected_focus_rows=1,
    )

    assert summary["decision"] == "blocked_before_redraw_preflight"
    assert "focus_annotation_preflight_not_ready" in summary["blockers"]
    assert summary["stages"]["focus_merge"]["ran"] is False
    assert summary["stages"]["loop96_unblind"]["ran"] is False
    assert summary["stages"]["loop87_import"]["ran"] is False


def test_loop110_actionable_content_verdict_reaches_loop87_without_training_authorization(tmp_path: Path):
    case = _case(tmp_path)
    focus_csv = _write_csv(
        tmp_path / "focus.csv",
        [
            _focus_row(
                case["blind_ids"][0],
                manual_label_verdict="feature_broken",
                manual_verdict_note="PE parse evidence and npz feature mismatch confirm broken extraction",
                recommended_action="replace_with_fresh_same_label_candidate",
            )
        ],
        FOCUS_FIELDS,
    )
    summary = run_focus_verdict_pipeline(
        full_blinded_csv=case["full_blinded_csv"],
        focus_annotations_csv=focus_csv,
        private_map_csv=case["private_map_csv"],
        output_dir=tmp_path / "loop110",
        output_json=tmp_path / "loop110_summary.json",
        expected_full_rows=2,
        expected_focus_rows=1,
    )

    assert summary["decision"] == "ready_for_redraw_preflight_review_only"
    assert summary["blockers"] == []
    assert summary["counts"]["loop87_actionable_rows"] == 1
    assert summary["counts"]["loop87_replacement_required_rows"] == 1
    assert summary["decisions"]["ready_for_redraw_preflight"] is True
    assert summary["decisions"]["training_allowed"] is False
    assert summary["decisions"]["test10k_allowed"] is False
