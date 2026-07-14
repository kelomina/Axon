from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.probe_loop67_val_content_strata_rules import probe_val_content_strata_rules
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


def _feature_vector(names: list[str], values: dict[str, float]) -> np.ndarray:
    vector = np.zeros(len(names), dtype=np.float32)
    index = {name: idx for idx, name in enumerate(names)}
    for name, value in values.items():
        vector[index[name]] = float(value)
    return vector


def _write_npz(path: Path, features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features.astype(np.float32, copy=False))


def test_loop67_rule_probe_rejects_or_accepts_by_configured_val_margin(tmp_path: Path):
    pred_csv = tmp_path / "loop57_val.csv"
    content_dir = tmp_path / "content"
    overlay_dir = tmp_path / "overlay"
    out_json = tmp_path / "report.json"
    out_csv = tmp_path / "candidates.csv"

    rows = [
        {
            "source_path": "repair.exe",
            "cache_path": "unused",
            "source_sha256": "sha-repair",
            "label": "1",
            "split": "val",
            "sample_index": "1",
            "base_prob_malicious": "0.10",
            "candidate_prob_malicious": "0.70",
            "gate_prob_override": "0.90",
            "final_prob_malicious": "0.10",
            "prediction": "0",
            "correct": "False",
            "fn_override": "False",
        },
        {
            "source_path": "tp.exe",
            "cache_path": "unused",
            "source_sha256": "sha-tp",
            "label": "1",
            "split": "val",
            "sample_index": "2",
            "base_prob_malicious": "0.90",
            "candidate_prob_malicious": "0.90",
            "gate_prob_override": "0.10",
            "final_prob_malicious": "0.90",
            "prediction": "1",
            "correct": "True",
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
            "candidate_prob_malicious": "0.10",
            "gate_prob_override": "0.10",
            "final_prob_malicious": "0.10",
            "prediction": "0",
            "correct": "True",
            "fn_override": "False",
        },
    ]
    _write_predictions(pred_csv, rows)

    for row in rows:
        values = {}
        if row["source_sha256"] == "sha-repair":
            values = {
                "content_dir_security_log_size": 4.0,
                "content_overlay_log_size": 5.0,
                "content_dir_basereloc_log_size": 4.0,
                "content_dir_export_log_size": 1.0,
                "content_dir_exception_log_size": 1.0,
                "content_dir_iat_log_size": 5.0,
            }
        _write_npz(content_dir / f"{row['source_sha256']}.npz", _feature_vector(CONTENT_PE_V1_FEATURE_NAMES, values))
        _write_npz(
            overlay_dir / f"{row['source_sha256']}.npz",
            _feature_vector(OVERLAY_BOUNDARY_FEATURE_NAMES, {}),
        )

    rejected = probe_val_content_strata_rules(
        loop57_val_predictions=pred_csv,
        content_pe_cache_dir=content_dir,
        overlay_boundary_cache_dir=overlay_dir,
        output_json=out_json,
        output_csv=out_csv,
        min_error_reduction_for_test10k=2,
    )
    accepted = probe_val_content_strata_rules(
        loop57_val_predictions=pred_csv,
        content_pe_cache_dir=content_dir,
        overlay_boundary_cache_dir=overlay_dir,
        output_json=out_json,
        output_csv=out_csv,
        min_error_reduction_for_test10k=1,
    )

    assert rejected["baseline_loop57"]["errors"] == 1
    assert rejected["selected_by_val"]["metrics"]["errors"] == 0
    assert rejected["overall_decision"] == "reject_no_candidate_with_sufficient_val_margin"
    assert accepted["overall_decision"] == "eligible_for_test10k"
    assert "repair_signed_overlay_complex" in out_csv.read_text(encoding="utf-8-sig")
    assert json.loads(out_json.read_text(encoding="utf-8"))["protocol"].startswith("read-only Val-only")
