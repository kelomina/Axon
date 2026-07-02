from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.analyze_loop66_val_blindspots import audit_val_blindspots
from scripts.train_loop55_overlay_boundary import OVERLAY_BOUNDARY_FEATURE_NAMES
from src.kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES


def _write_predictions(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "source_path",
        "cache_path",
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
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_feature_cache(path: Path, features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features.astype(np.float32, copy=False))


def test_loop66_val_blindspot_audit_reports_content_and_overlay_deltas(tmp_path: Path):
    predictions = tmp_path / "loop57_val.csv"
    content_cache = tmp_path / "content"
    overlay_cache = tmp_path / "overlay"
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "deltas.csv"

    rows = [
        {
            "source_path": "tp.exe",
            "cache_path": "unused",
            "source_sha256": "sha-tp",
            "label": "1",
            "split": "val",
            "sample_index": "1",
            "base_prob_malicious": "0.90",
            "candidate_prob_malicious": "0.95",
            "gate_prob_override": "0.10",
            "final_prob_malicious": "0.90",
            "prediction": "1",
            "correct": "True",
            "fn_override": "False",
        },
        {
            "source_path": "fn.exe",
            "cache_path": "unused",
            "source_sha256": "sha-fn",
            "label": "1",
            "split": "val",
            "sample_index": "2",
            "base_prob_malicious": "0.10",
            "candidate_prob_malicious": "0.20",
            "gate_prob_override": "0.10",
            "final_prob_malicious": "0.10",
            "prediction": "0",
            "correct": "False",
            "fn_override": "False",
        },
        {
            "source_path": "tn.exe",
            "cache_path": "unused",
            "source_sha256": "sha-tn",
            "label": "0",
            "split": "val",
            "sample_index": "3",
            "base_prob_malicious": "0.10",
            "candidate_prob_malicious": "0.20",
            "gate_prob_override": "0.10",
            "final_prob_malicious": "0.10",
            "prediction": "0",
            "correct": "True",
            "fn_override": "False",
        },
        {
            "source_path": "fp.exe",
            "cache_path": "unused",
            "source_sha256": "sha-fp",
            "label": "0",
            "split": "val",
            "sample_index": "4",
            "base_prob_malicious": "0.10",
            "candidate_prob_malicious": "0.90",
            "gate_prob_override": "0.95",
            "final_prob_malicious": "0.90",
            "prediction": "1",
            "correct": "False",
            "fn_override": "True",
        },
        {
            "source_path": "repaired.exe",
            "cache_path": "unused",
            "source_sha256": "sha-repaired",
            "label": "1",
            "split": "val",
            "sample_index": "5",
            "base_prob_malicious": "0.10",
            "candidate_prob_malicious": "0.90",
            "gate_prob_override": "0.95",
            "final_prob_malicious": "0.90",
            "prediction": "1",
            "correct": "True",
            "fn_override": "True",
        },
    ]
    _write_predictions(predictions, rows)

    for row in rows:
        content = np.zeros(len(CONTENT_PE_V1_FEATURE_NAMES), dtype=np.float32)
        overlay = np.zeros(len(OVERLAY_BOUNDARY_FEATURE_NAMES), dtype=np.float32)
        if row["source_sha256"] == "sha-fp":
            content[0] = 5.0
        if row["source_sha256"] == "sha-fn":
            overlay[1] = 7.0
        if row["source_sha256"] == "sha-repaired":
            content[2] = 9.0
        _write_feature_cache(content_cache / f"{row['source_sha256']}.npz", content)
        _write_feature_cache(overlay_cache / f"{row['source_sha256']}.npz", overlay)

    report = audit_val_blindspots(
        loop57_val_predictions=predictions,
        content_pe_cache_dir=content_cache,
        overlay_boundary_cache_dir=overlay_cache,
        output_json=output_json,
        output_csv=output_csv,
        top_k=3,
    )
    csv_text = output_csv.read_text(encoding="utf-8-sig")

    assert report["rows"] == 5
    assert report["final_group_counts"] == {"tp": 2, "tn": 1, "fp": 1, "fn": 1}
    assert report["exchange_group_counts"]["base_error_final_repaired"] == 1
    assert report["exchange_group_counts"]["base_correct_final_harmed"] == 1
    assert report["feature_shapes"]["content_pe_v1"] == [5, len(CONTENT_PE_V1_FEATURE_NAMES)]
    assert report["feature_shapes"]["overlay_boundary"] == [5, len(OVERLAY_BOUNDARY_FEATURE_NAMES)]
    assert "final_fp_minus_final_tn" in csv_text
    assert "content_file_log_size" in csv_text
    assert "final_fn_minus_final_tp" in csv_text
    assert "overlay_boundary_security_log_size" in csv_text
    assert json.loads(output_json.read_text(encoding="utf-8"))["protocol"].startswith("read-only Val")
