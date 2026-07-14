from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop156_current_best_val_full_error_review import build_loop156_review  # noqa: E402


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
    "content_resource_type_count_log",
    "content_dir_resource_size_ratio",
    "content_dir_resource_log_size",
    "v2_resource_data_entry_count_log",
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


def test_loop156_exports_all_current_errors_without_identity_public_columns(tmp_path: Path):
    neighbor = _write_csv(
        tmp_path / "neighbor.csv",
        [
            {"support_bucket": "neighbors_mixed", "priority": "20", "reason": "mixed", "error_type": "fn", "source_path": "a.exe", "source_sha256": "a" * 64, "label": "1", "prediction": "0", "prob_malicious": "0.1", "opposite_label_ratio": "0.5", "nearest_similarity": "0.75"},
            {"support_bucket": "neighbors_support_model_prediction", "priority": "10", "reason": "support", "error_type": "fp", "source_path": "b.exe", "source_sha256": "b" * 64, "label": "0", "prediction": "1", "prob_malicious": "0.9", "opposite_label_ratio": "1.0", "nearest_similarity": "0.99"},
            {"support_bucket": "neighbors_mixed", "priority": "30", "reason": "no-content", "error_type": "fp", "source_path": "c.exe", "source_sha256": "c" * 64, "label": "0", "prediction": "1", "prob_malicious": "0.8", "opposite_label_ratio": "0.1", "nearest_similarity": "0.1"},
        ],
        NEIGHBOR_FIELDS,
    )
    content = _write_csv(
        tmp_path / "content.csv",
        [
            {"source_sha256": "a" * 64, "source_path": "a.exe", "cache_path": "cache/a.npz", "content_dir_security_log_size": "0", "content_overlay_log_size": "1", "content_overlay_entropy": "0.9", "content_resource_entry_count_log": "1", "content_resource_type_count_log": "1", "content_dir_resource_size_ratio": "0.1", "content_dir_resource_log_size": "1", "v2_resource_data_entry_count_log": "1", "v2_resource_type_version_count_log": "0", "string_benign_vendor_count_log": "0", "string_script_exec_present": "1"},
            {"source_sha256": "b" * 64, "source_path": "b.exe", "cache_path": "cache/b.npz", "content_dir_security_log_size": "1", "content_overlay_log_size": "0", "content_overlay_entropy": "0", "content_resource_entry_count_log": "4", "content_resource_type_count_log": "2", "content_dir_resource_size_ratio": "0.2", "content_dir_resource_log_size": "1", "v2_resource_data_entry_count_log": "3", "v2_resource_type_version_count_log": "1", "string_benign_vendor_count_log": "2", "string_script_exec_present": "0"},
        ],
        CONTENT_FIELDS,
    )

    payload = build_loop156_review(
        neighbor_csv=neighbor,
        content_review_csv=content,
        output_review_csv=tmp_path / "review.csv",
        output_private_map_csv=tmp_path / "private.csv",
        output_json=tmp_path / "summary.json",
    )

    assert payload["review_rows"] == 2
    assert payload["error_counts"] == {"fn": 1, "fp": 1}
    assert payload["skipped_counts"] == {"missing_content_row": 1}
    review_rows = list(csv.DictReader((tmp_path / "review.csv").open(encoding="utf-8-sig")))
    assert review_rows[0]["review_focus_id"] == "loop156_val_error_000001"
    assert review_rows[0]["error_type"] == "fp"
    assert "source_sha256" not in review_rows[0]
    assert "prediction" not in review_rows[0]
    private_rows = list(csv.DictReader((tmp_path / "private.csv").open(encoding="utf-8-sig")))
    assert private_rows[0]["source_sha256"] == "b" * 64
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["schema"] == (
        "axon_loop156_current_best_val_full_error_review_v1"
    )
