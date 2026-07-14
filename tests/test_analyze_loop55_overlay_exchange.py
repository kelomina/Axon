from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts import analyze_loop55_overlay_exchange as audit
from scripts.train_loop55_overlay_boundary import OVERLAY_BOUNDARY_FEATURE_NAMES


def _write_predictions(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "source_path",
        "cache_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "stage2_prob_malicious",
        "prediction",
        "correct",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_loop55_exchange_groups_and_feature_summary(tmp_path: Path):
    rows28 = [
        {"source_path": "a", "cache_path": "", "source_sha256": "sha-a", "label": 1, "split": "val", "sample_index": 1, "stage2_prob_malicious": 0.4, "prediction": 0, "correct": False},
        {"source_path": "b", "cache_path": "", "source_sha256": "sha-b", "label": 0, "split": "val", "sample_index": 2, "stage2_prob_malicious": 0.2, "prediction": 0, "correct": True},
    ]
    rows55 = [
        {"source_path": "a", "cache_path": "", "source_sha256": "sha-a", "label": 1, "split": "val", "sample_index": 1, "stage2_prob_malicious": 0.8, "prediction": 1, "correct": True},
        {"source_path": "b", "cache_path": "", "source_sha256": "sha-b", "label": 0, "split": "val", "sample_index": 2, "stage2_prob_malicious": 0.9, "prediction": 1, "correct": False},
    ]
    loop28 = tmp_path / "loop28.csv"
    loop55 = tmp_path / "loop55.csv"
    _write_predictions(loop28, rows28)
    _write_predictions(loop55, rows55)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    first = np.zeros(len(OVERLAY_BOUNDARY_FEATURE_NAMES), dtype=np.float32)
    first[0] = 1.0
    second = np.ones(len(OVERLAY_BOUNDARY_FEATURE_NAMES), dtype=np.float32)
    np.savez(cache_dir / "sha-a.npz", features=first)
    np.savez(cache_dir / "sha-b.npz", features=second)

    output_json = tmp_path / "report.json"
    output_csv = tmp_path / "details.csv"
    audit.main(
        [
            "--loop28-predictions",
            str(loop28),
            "--loop55-predictions",
            str(loop55),
            "--overlay-cache-dir",
            str(cache_dir),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ]
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["exchange_counts"]["loop28_only_error"] == 1
    assert report["exchange_counts"]["loop55_only_error"] == 1
    assert report["feature_summary"]["groups"]["loop28_only_error"]["rows"] == 1
    assert output_csv.read_text(encoding="utf-8").count("\n") == 3
