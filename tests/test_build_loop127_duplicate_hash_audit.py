from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_loop127_duplicate_hash_audit import build_loop127_duplicate_hash_audit


def _write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_sha256", "sample_index", "label", "prob_malicious", "prediction", "correct"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_duplicate_hash_audit_reports_duplicate_groups_and_redraw_policy(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    sha = "a" * 64
    _write_predictions(
        train_csv,
        [
            {"source_sha256": sha, "sample_index": "1", "label": "1", "prob_malicious": "0.8", "prediction": "1", "correct": "True"},
            {"source_sha256": sha, "sample_index": "2", "label": "1", "prob_malicious": "0.8", "prediction": "1", "correct": "True"},
        ],
    )
    _write_predictions(val_csv, [])

    payload = build_loop127_duplicate_hash_audit(
        train_predictions=train_csv,
        val_predictions=val_csv,
        output_csv=tmp_path / "duplicates.csv",
        output_json=tmp_path / "duplicates.json",
    )
    rows = list(csv.DictReader((tmp_path / "duplicates.csv").open("r", encoding="utf-8-sig", newline="")))

    assert payload["duplicate_groups"] == 1
    assert payload["duplicate_rows"] == 2
    assert payload["split_row_counts"] == {"train": 2}
    assert payload["ready_without_redraw"] is False
    assert rows[0]["recommended_action"] == "quarantine_duplicate_group_and_redraw_full_replacement_batch"
