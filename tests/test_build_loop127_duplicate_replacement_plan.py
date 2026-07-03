from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_loop127_duplicate_replacement_plan import build_loop127_duplicate_replacement_plan


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_duplicate_replacement_plan_excludes_noncanonical_duplicate(tmp_path: Path):
    sha = "a" * 64
    duplicate_csv = tmp_path / "duplicates.csv"
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    duplicate_fields = ["split", "source_sha256", "sample_index", "label", "prob_malicious", "prediction", "correct"]
    prediction_fields = ["source_path", "source_sha256", "cache_path", "label", "split", "sample_index"]
    _write_csv(
        duplicate_csv,
        duplicate_fields,
        [
            {"split": "train", "source_sha256": sha, "sample_index": "10", "label": "1"},
            {"split": "train", "source_sha256": sha, "sample_index": "20", "label": "1"},
        ],
    )
    _write_csv(
        train_csv,
        prediction_fields,
        [
            {
                "source_path": str(tmp_path / "canonical.exe"),
                "source_sha256": sha,
                "cache_path": str(tmp_path / "canonical.npz"),
                "label": "1",
                "split": "train",
                "sample_index": "10",
            },
            {
                "source_path": str(tmp_path / "duplicate.exe"),
                "source_sha256": sha,
                "cache_path": str(tmp_path / "duplicate.npz"),
                "label": "1",
                "split": "train",
                "sample_index": "20",
            },
        ],
    )
    _write_csv(val_csv, prediction_fields, [])

    payload = build_loop127_duplicate_replacement_plan(
        duplicate_audit_csv=duplicate_csv,
        train_predictions=train_csv,
        val_predictions=val_csv,
        output_plan_csv=tmp_path / "plan.csv",
        output_json=tmp_path / "plan.json",
    )
    rows = list(csv.DictReader((tmp_path / "plan.csv").open("r", encoding="utf-8-sig", newline="")))

    assert payload["plan_ready"] is True
    assert payload["plan_rows"] == 1
    assert payload["replacement_counts_by_label"] == {"1": 1}
    assert rows[0]["sample_index"] == "20"
    assert rows[0]["plan_action"] == "exclude_and_replace"
    assert rows[0]["replacement_required"] == "true"
