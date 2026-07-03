import argparse
import csv

import numpy as np
import pytest

from scripts.train_loop43_content_cross import (
    CONTENT_CROSS_FEATURE_NAMES,
    CONTENT_PE_FEATURE_NAMES,
    CONTENT_PE_V2_FEATURE_NAMES,
    content_cross_features_from_arrays,
    run_strict_readiness_preflight,
)


def _vector(names, values):
    mapping = {name: index for index, name in enumerate(names)}
    result = np.zeros(len(names), dtype=np.float32)
    for name, value in values.items():
        result[mapping[name]] = value
    return result


def test_content_cross_features_have_stable_width_and_no_nan():
    pe1 = _vector(
        CONTENT_PE_FEATURE_NAMES,
        {
            "content_is_dll": 1.0,
            "content_dir_security_present": 1.0,
            "content_dir_security_log_size": 3.0,
            "content_overlay_present": 1.0,
            "content_overlay_log_size": 2.0,
            "content_overlay_entropy": 0.8,
            "content_export_count_log": 1.5,
            "content_section_high_entropy_ratio": 0.5,
            "content_section_combo_rwx_ratio": 0.25,
            "content_section_zero_raw_ratio": 0.2,
            "content_section_name_packer_hit_ratio": 0.1,
            "content_system_dll_ratio": 0.9,
            "content_import_api_count_log": 4.0,
        },
    )
    pe2 = _vector(
        CONTENT_PE_V2_FEATURE_NAMES,
        {
            "v2_api_driver_present": 1.0,
            "v2_api_driver_count_log": 2.0,
            "v2_export_pattern_service_present": 1.0,
            "v2_section_exec_write_count_log": 1.0,
            "v2_section_exec_high_entropy_ratio": 0.7,
            "v2_last_section_entropy": 0.9,
        },
    )

    features = content_cross_features_from_arrays(pe1, pe2)

    assert features.shape == (len(CONTENT_CROSS_FEATURE_NAMES),)
    assert np.isfinite(features).all()
    assert features[CONTENT_CROSS_FEATURE_NAMES.index("cross_dll_security_log_size")] == 3.0
    assert features[CONTENT_CROSS_FEATURE_NAMES.index("cross_system_dll_high_import")] == 3.6


def test_content_cross_features_do_not_fire_for_unsigned_overlay_security_cross():
    pe1 = _vector(
        CONTENT_PE_FEATURE_NAMES,
        {
            "content_dir_security_present": 0.0,
            "content_overlay_present": 1.0,
            "content_overlay_log_size": 5.0,
            "content_overlay_entropy": 0.75,
        },
    )
    pe2 = np.zeros(len(CONTENT_PE_V2_FEATURE_NAMES), dtype=np.float32)

    features = content_cross_features_from_arrays(pe1, pe2)

    assert features[CONTENT_CROSS_FEATURE_NAMES.index("cross_security_overlay_log_size")] == 0.0
    assert features[CONTENT_CROSS_FEATURE_NAMES.index("cross_unsigned_overlay_log_size")] == 5.0


def _write_predictions_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path",
                "source_sha256",
                "cache_path",
                "label",
                "split",
                "sample_index",
                "prob_malicious",
                "prediction",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_main_cache(path, *, label, sha):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, label=np.asarray(label), source_sha256=np.asarray(sha))


def _write_sidecar(path, *, dim):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=np.zeros(dim, dtype=np.float32))


def test_strict_readiness_preflight_blocks_missing_sidecar_before_training(tmp_path):
    train_sha = "a" * 64
    val_sha = "b" * 64
    train_cache = tmp_path / "cache" / "train.npz"
    val_cache = tmp_path / "cache" / "val.npz"
    _write_main_cache(train_cache, label=0, sha=train_sha)
    _write_main_cache(val_cache, label=1, sha=val_sha)
    _write_sidecar(tmp_path / "v1" / f"{train_sha}.npz", dim=len(CONTENT_PE_FEATURE_NAMES))
    _write_sidecar(tmp_path / "v2" / f"{train_sha}.npz", dim=len(CONTENT_PE_V2_FEATURE_NAMES))
    _write_sidecar(tmp_path / "v1" / f"{val_sha}.npz", dim=len(CONTENT_PE_FEATURE_NAMES))

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _write_predictions_csv(
        train_csv,
        [
            {
                "source_path": str(tmp_path / "train.exe"),
                "source_sha256": train_sha,
                "cache_path": str(train_cache),
                "label": "0",
                "split": "train",
                "sample_index": "1",
                "prob_malicious": "0.1",
                "prediction": "0",
                "correct": "True",
            }
        ],
    )
    _write_predictions_csv(
        val_csv,
        [
            {
                "source_path": str(tmp_path / "val.exe"),
                "source_sha256": val_sha,
                "cache_path": str(val_cache),
                "label": "1",
                "split": "val",
                "sample_index": "2",
                "prob_malicious": "0.9",
                "prediction": "1",
                "correct": "True",
            }
        ],
    )
    args = argparse.Namespace(
        train_predictions=train_csv,
        val_predictions=val_csv,
        content_pe_cache_dir=tmp_path / "v1",
        content_pe_v2_cache_dir=tmp_path / "v2",
        expected_train_rows=1,
        expected_val_rows=1,
        expected_test_rows=0,
        expected_total_rows=2,
    )

    with pytest.raises(RuntimeError, match="preflight blocked training"):
        run_strict_readiness_preflight(args, tmp_path / "out")
