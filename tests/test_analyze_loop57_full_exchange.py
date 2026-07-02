from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts import analyze_loop57_full_exchange as audit
from scripts.train_loop55_overlay_boundary import OVERLAY_BOUNDARY_FEATURE_NAMES


def _write_predictions(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_loop57_full_exchange_counts_overrides_and_feature_deltas(tmp_path: Path):
    loop28_rows = [
        {"source_path": "a", "source_sha256": "sha-a", "label": 1, "split": "test", "sample_index": 1, "stage2_prob_malicious": 0.4, "prediction": 0, "correct": False},
        {"source_path": "b", "source_sha256": "sha-b", "label": 0, "split": "test", "sample_index": 2, "stage2_prob_malicious": 0.2, "prediction": 0, "correct": True},
    ]
    loop57_rows = [
        {"source_path": "a", "source_sha256": "sha-a", "label": 1, "split": "test", "sample_index": 1, "base_prob_malicious": 0.4, "candidate_prob_malicious": 0.9, "gate_prob_override": 0.95, "final_prob_malicious": 0.9, "prediction": 1, "correct": True, "fn_override": True},
        {"source_path": "b", "source_sha256": "sha-b", "label": 0, "split": "test", "sample_index": 2, "base_prob_malicious": 0.2, "candidate_prob_malicious": 0.8, "gate_prob_override": 0.91, "final_prob_malicious": 0.8, "prediction": 1, "correct": False, "fn_override": True},
    ]
    loop28 = tmp_path / "loop28.csv"
    loop57 = tmp_path / "loop57.csv"
    _write_predictions(
        loop28,
        loop28_rows,
        ["source_path", "source_sha256", "label", "split", "sample_index", "stage2_prob_malicious", "prediction", "correct"],
    )
    _write_predictions(
        loop57,
        loop57_rows,
        [
            "source_path",
            "source_sha256",
            "label",
            "split",
            "sample_index",
            "base_prob_malicious",
            "candidate_prob_malicious",
            "gate_prob_override",
            "final_prob_malicious",
            "prediction",
            "correct",
            "fn_override",
        ],
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    repaired = np.ones(len(OVERLAY_BOUNDARY_FEATURE_NAMES), dtype=np.float32)
    harmed = np.zeros(len(OVERLAY_BOUNDARY_FEATURE_NAMES), dtype=np.float32)
    np.savez(cache_dir / "sha-a.npz", features=repaired)
    np.savez(cache_dir / "sha-b.npz", features=harmed)

    output_json = tmp_path / "report.json"
    output_csv = tmp_path / "details.csv"
    audit.main(
        [
            "--loop28-predictions",
            str(loop28),
            "--loop57-predictions",
            str(loop57),
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
    assert report["exchange_counts"]["loop57_only_error"] == 1
    assert report["override_counts_by_group"]["loop28_only_error"] == 1
    assert report["override_counts_by_group"]["loop57_only_error"] == 1
    assert report["feature_summary"]["top_repair_vs_harm_feature_deltas"][0]["difference"] == 1.0
    assert output_csv.read_text(encoding="utf-8").count("\n") == 3
