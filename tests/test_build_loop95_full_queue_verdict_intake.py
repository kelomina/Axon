from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_loop95_full_queue_verdict_intake import build_intake


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


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> Path:
    if fieldnames is None:
        fieldnames = FIELDS
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _row(sample_index: str, *, wave_rank: str = "", source_sha256: str | None = None) -> dict:
    return {
        "review_batch_rank": wave_rank,
        "review_category": "c_high_conflict_persistent_error",
        "source_path": f"data/{sample_index}.bin",
        "cache_path": f"data/.cache/{sample_index}.npz",
        "source_sha256": source_sha256 or (sample_index.zfill(64)[-64:]),
        "sample_index": sample_index,
        "split": "test",
        "label": "1",
        "loop57_error_type": "FN",
        "loop57_final_prob": "0.01",
        "manual_label_verdict": "",
        "manual_verdict_note": "",
        "recommended_action": "",
    }


def _fixtures(tmp_path: Path):
    loop72 = _write_csv(
        tmp_path / "loop72.csv",
        [
            {"review_wave_id": "1", "sample_index": "101"},
            {"review_wave_id": "1", "sample_index": "102"},
            {"review_wave_id": "2", "sample_index": "201"},
        ],
        fieldnames=["review_wave_id", "sample_index"],
    )
    summary = _write_json(
        tmp_path / "multiwave.json",
        {
            "combined": {"rows": 3, "queue_rows": 3},
            "wave_reports": [
                {"wave_id": 1, "rows": 2},
                {"wave_id": 2, "rows": 1},
            ],
        },
    )
    wave1 = _write_csv(tmp_path / "wave1.csv", [_row("101"), _row("102")])
    wave2 = _write_csv(tmp_path / "wave2.csv", [_row("201")])
    return loop72, summary, wave1, wave2


def test_loop95_builds_full_queue_intake_without_authorizing_training(tmp_path: Path):
    loop72, summary_json, wave1, wave2 = _fixtures(tmp_path)
    summary = build_intake(
        loop72_plan_csv=loop72,
        multiwave_summary_json=summary_json,
        waves=[(1, wave1), (2, wave2)],
        output_csv=tmp_path / "intake.csv",
        output_json=tmp_path / "summary.json",
    )
    rows = list(csv.DictReader((tmp_path / "intake.csv").open("r", encoding="utf-8-sig", newline="")))

    assert summary["blockers"] == []
    assert summary["rows"] == 3
    assert summary["decisions"]["ready_for_loop87_full_queue_import"] is True
    assert summary["decisions"]["training_allowed"] is False
    assert [row["loop95_wave_id"] for row in rows] == ["1", "1", "2"]
    assert [row["loop95_intake_row_number"] for row in rows] == ["1", "2", "3"]
    assert "not model evidence" in summary["identity_feature_policy"]


def test_loop95_blocks_duplicate_sample_index_across_waves(tmp_path: Path):
    loop72, summary_json, wave1, _wave2 = _fixtures(tmp_path)
    duplicate_wave2 = _write_csv(tmp_path / "duplicate_wave2.csv", [_row("102")])
    summary = build_intake(
        loop72_plan_csv=loop72,
        multiwave_summary_json=summary_json,
        waves=[(1, wave1), (2, duplicate_wave2)],
        output_csv=tmp_path / "intake.csv",
        output_json=tmp_path / "summary.json",
    )

    assert "duplicate_sample_index_across_intake" in summary["blockers"]
    assert "wave2_missing_loop72_sample_index" in summary["blockers"]
    assert "wave2_unexpected_sample_index" in summary["blockers"]
    assert summary["decisions"]["ready_for_loop87_full_queue_import"] is False


def test_loop95_blocks_wave_count_mismatch(tmp_path: Path):
    loop72, summary_json, _wave1, wave2 = _fixtures(tmp_path)
    short_wave1 = _write_csv(tmp_path / "short_wave1.csv", [_row("101")])
    summary = build_intake(
        loop72_plan_csv=loop72,
        multiwave_summary_json=summary_json,
        waves=[(1, short_wave1), (2, wave2)],
        output_csv=tmp_path / "intake.csv",
        output_json=tmp_path / "summary.json",
    )

    assert "wave1_row_count_mismatch_loop72" in summary["blockers"]
    assert "wave1_row_count_mismatch_multiwave_summary" in summary["blockers"]
    assert "wave1_missing_loop72_sample_index" in summary["blockers"]
    assert "combined_row_count_mismatch_expected" in summary["blockers"]
