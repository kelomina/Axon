from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop153_current_best_val_noise_focus import build_loop153_focus  # noqa: E402


PRED_FIELDS = ["source_path", "source_sha256", "label", "split", "sample_index", "prediction", "trusted_signer_guard_prediction"]
NEIGHBOR_FIELDS = [
    "support_bucket",
    "priority",
    "reason",
    "error_type",
    "source_path",
    "source_sha256",
    "label",
    "prediction",
    "prob_malicious",
    "opposite_label_ratio",
    "nearest_similarity",
]
CONTENT_FIELDS = [
    "source_sha256",
    "source_path",
    "cache_path",
    "content_dir_security_log_size",
    "content_overlay_log_size",
    "content_overlay_entropy",
    "content_resource_entry_count_log",
    "v2_resource_type_version_count_log",
    "string_benign_vendor_count_log",
    "string_script_exec_present",
]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_loop153_filters_to_current_best_val_errors_and_blinds_public_output(tmp_path: Path):
    predictions = _write_csv(
        tmp_path / "predictions.csv",
        [
            {"source_path": "data/a.exe", "source_sha256": "a" * 64, "label": "0", "split": "val", "sample_index": "1", "prediction": "1", "trusted_signer_guard_prediction": "1"},
            {"source_path": "data/b.exe", "source_sha256": "b" * 64, "label": "1", "split": "val", "sample_index": "2", "prediction": "0", "trusted_signer_guard_prediction": "0"},
            {"source_path": "data/c.exe", "source_sha256": "c" * 64, "label": "1", "split": "val", "sample_index": "3", "prediction": "1", "trusted_signer_guard_prediction": "1"},
        ],
        PRED_FIELDS,
    )
    neighbor = _write_csv(
        tmp_path / "neighbor.csv",
        [
            {"support_bucket": "neighbors_support_model_prediction", "priority": "10", "reason": "x", "error_type": "fp", "source_path": "data/a.exe", "source_sha256": "a" * 64, "label": "0", "prediction": "1", "prob_malicious": "0.9", "opposite_label_ratio": "1.0", "nearest_similarity": "0.99"},
            {"support_bucket": "neighbors_mixed", "priority": "20", "reason": "y", "error_type": "fn", "source_path": "data/b.exe", "source_sha256": "b" * 64, "label": "1", "prediction": "0", "prob_malicious": "0.1", "opposite_label_ratio": "0.5", "nearest_similarity": "0.75"},
            {"support_bucket": "neighbors_support_model_prediction", "priority": "10", "reason": "z", "error_type": "fn", "source_path": "data/c.exe", "source_sha256": "c" * 64, "label": "1", "prediction": "0", "prob_malicious": "0.1", "opposite_label_ratio": "1.0", "nearest_similarity": "0.99"},
        ],
        NEIGHBOR_FIELDS,
    )
    content = _write_csv(
        tmp_path / "content.csv",
        [
            {"source_sha256": "a" * 64, "source_path": "data/a.exe", "cache_path": "cache/a.npz", "content_dir_security_log_size": "1", "content_overlay_log_size": "1", "content_overlay_entropy": "0.9", "content_resource_entry_count_log": "4", "v2_resource_type_version_count_log": "1", "string_benign_vendor_count_log": "1", "string_script_exec_present": "0"},
            {"source_sha256": "b" * 64, "source_path": "data/b.exe", "cache_path": "cache/b.npz", "content_dir_security_log_size": "0", "content_overlay_log_size": "0", "content_overlay_entropy": "0", "content_resource_entry_count_log": "0", "v2_resource_type_version_count_log": "0", "string_benign_vendor_count_log": "0", "string_script_exec_present": "1"},
            {"source_sha256": "c" * 64, "source_path": "data/c.exe", "cache_path": "cache/c.npz", "content_dir_security_log_size": "1", "content_overlay_log_size": "1", "content_overlay_entropy": "0.9", "content_resource_entry_count_log": "4", "v2_resource_type_version_count_log": "1", "string_benign_vendor_count_log": "1", "string_script_exec_present": "0"},
        ],
        CONTENT_FIELDS,
    )

    payload = build_loop153_focus(
        predictions_csv=predictions,
        prediction_column="trusted_signer_guard_prediction",
        neighbor_csv=neighbor,
        content_review_csv=content,
        filtered_neighbor_csv=tmp_path / "filtered_neighbor.csv",
        filtered_content_review_csv=tmp_path / "filtered_content.csv",
        output_focus_csv=tmp_path / "focus.csv",
        output_private_map_csv=tmp_path / "private.csv",
        output_json=tmp_path / "summary.json",
    )

    assert payload["prediction_summary"]["current_error_rows"] == 2
    assert payload["source_filter_summary"]["filtered_neighbor_rows"] == 2
    assert payload["focus_rows"] == 1
    assert payload["error_counts"] == {"fp": 1}
    focus_rows = list(csv.DictReader((tmp_path / "focus.csv").open(encoding="utf-8-sig")))
    assert focus_rows[0]["review_focus_id"] == "loop153_val_focus_000001"
    assert "source_sha256" not in focus_rows[0]
    assert "prediction" not in focus_rows[0]
    private_rows = list(csv.DictReader((tmp_path / "private.csv").open(encoding="utf-8-sig")))
    assert private_rows[0]["source_sha256"] == "a" * 64
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["schema"] == "axon_loop153_current_best_val_noise_focus_v1"
