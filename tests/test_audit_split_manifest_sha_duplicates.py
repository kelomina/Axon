from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.audit_split_manifest_sha_duplicates import audit_manifest_sha_duplicates


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_sha_duplicate_audit_finds_cache_sha_duplicates_without_split_sha(tmp_path: Path):
    split_csv = tmp_path / "split.csv"
    manifest_json = tmp_path / "manifest.json"
    focus_csv = tmp_path / "focus.csv"
    details_csv = tmp_path / "details.csv"

    split_rows = [
        {"source_path": "data/a.exe", "label": "1", "split": "test", "sample_index": "1"},
        {"source_path": "data/a_copy.exe", "label": "1", "split": "test", "sample_index": "2"},
        {"source_path": "data/b.exe", "label": "0", "split": "train", "sample_index": "3"},
    ]
    _write_csv(split_csv, split_rows)
    manifest_json.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "source_path": "data/a.exe",
                        "label": 1,
                        "cache_path": "cache/a.npz",
                        "source_sha256": "sha-same",
                    },
                    {
                        "source_path": "data/a_copy.exe",
                        "label": 1,
                        "cache_path": "cache/a-copy.npz",
                        "source_sha256": "sha-same",
                    },
                    {
                        "source_path": "data/b.exe",
                        "label": 0,
                        "cache_path": "cache/b.npz",
                        "source_sha256": "sha-b",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        focus_csv,
        [{"source_path": "data/a.exe", "source_sha256": "", "label": "1", "sample_index": "1"}],
        ["source_path", "source_sha256", "label", "sample_index"],
    )

    report = audit_manifest_sha_duplicates(
        split_csv=split_csv,
        manifest_json=manifest_json,
        output_csv=details_csv,
        focus_queue_csv=focus_csv,
    )
    details = list(csv.DictReader(details_csv.open("r", encoding="utf-8-sig", newline="")))

    assert report["split_rows"] == 3
    assert report["matched_rows"] == 3
    assert report["duplicate_groups"] == 1
    assert report["duplicate_extra_rows"] == 1
    assert report["focus_duplicate_groups"] == 1
    assert report["focus_duplicate_detail_rows"] == 2
    assert report["cross_label_groups"] == 0
    assert {row["manifest_source_sha256"] for row in details} == {"sha-same"}
