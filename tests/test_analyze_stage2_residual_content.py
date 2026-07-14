import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_stage2_residual_content import main
from train_stage2_cache_matrix import CONTENT_PE_FEATURE_NAMES


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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_content_cache(cache_dir: Path, source_sha: str, first_feature: float) -> None:
    features = np.zeros(len(CONTENT_PE_FEATURE_NAMES), dtype=np.float32)
    features[0] = first_feature
    np.savez(cache_dir / f"{source_sha}.npz", features=features)


def test_residual_content_attribution_uses_content_cache_not_filename(tmp_path: Path):
    cache_dir = tmp_path / "content_pe"
    cache_dir.mkdir()
    predictions = tmp_path / "predictions.csv"
    rows = [
        {
            "source_path": str(tmp_path / "benign-looking.exe"),
            "cache_path": "unused.npz",
            "source_sha256": "tn",
            "label": "0",
            "split": "test",
            "sample_index": "1",
            "stage2_prob_malicious": "0.1",
            "prediction": "0",
            "correct": "True",
        },
        {
            "source_path": str(tmp_path / "also-benign-looking.exe"),
            "cache_path": "unused.npz",
            "source_sha256": "fp",
            "label": "0",
            "split": "test",
            "sample_index": "2",
            "stage2_prob_malicious": "0.9",
            "prediction": "1",
            "correct": "False",
        },
        {
            "source_path": str(tmp_path / "malicious-looking.exe"),
            "cache_path": "unused.npz",
            "source_sha256": "tp",
            "label": "1",
            "split": "test",
            "sample_index": "3",
            "stage2_prob_malicious": "0.9",
            "prediction": "1",
            "correct": "True",
        },
        {
            "source_path": str(tmp_path / "renamed-random"),
            "cache_path": "unused.npz",
            "source_sha256": "fn",
            "label": "1",
            "split": "test",
            "sample_index": "4",
            "stage2_prob_malicious": "0.1",
            "prediction": "0",
            "correct": "False",
        },
    ]
    _write_predictions(predictions, rows)
    _write_content_cache(cache_dir, "tn", 0.0)
    _write_content_cache(cache_dir, "fp", 4.0)
    _write_content_cache(cache_dir, "tp", 5.0)
    _write_content_cache(cache_dir, "fn", 0.0)

    output_dir = tmp_path / "out"
    result = main(
        [
            "--predictions",
            str(predictions),
            "--output-dir",
            str(output_dir),
            "--content-pe-cache-dir",
            str(cache_dir),
            "--top-k",
            "5",
            "--min-slice-support",
            "1",
        ]
    )

    assert result == 0
    report = json.loads((output_dir / "residual_content_attribution_report.json").read_text(encoding="utf-8"))
    assert report["confusion"]["false_positive"] == 1
    assert report["confusion"]["false_negative"] == 1
    assert report["diagnostic_path_slices_enabled"] is False
    fp_top = report["top_feature_attribution"]["FP_vs_TN"][0]
    assert fp_top["feature"] == CONTENT_PE_FEATURE_NAMES[0]
    assert fp_top["mean_delta"] == 4.0
    assert "diagnostic_path_slices_csv" in report["outputs"]
    assert report["outputs"]["diagnostic_path_slices_csv"] is None
