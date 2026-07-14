from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import build_content_pe_v2_feature_cache
from scripts.train_stage2_cache_matrix import CONTENT_PE_V2_FEATURE_NAMES


def test_content_pe_v2_cache_limit_requires_smoke(tmp_path: Path):
    with pytest.raises(ValueError, match="--limit requires --smoke"):
        build_content_pe_v2_feature_cache.main(
            [
                "--predictions",
                str(tmp_path / "missing.csv"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--limit",
                "1",
                "--output-json",
                str(tmp_path / "report.json"),
            ]
        )


def test_content_pe_v2_cache_negative_limit_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="--limit must be non-negative"):
        build_content_pe_v2_feature_cache.main(
            [
                "--predictions",
                str(tmp_path / "missing.csv"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--smoke",
                "--limit",
                "-1",
                "--output-json",
                str(tmp_path / "report.json"),
            ]
        )


def test_content_pe_v2_cache_rejects_too_many_workers(tmp_path: Path):
    with pytest.raises(ValueError, match="--workers must be <= 8"):
        build_content_pe_v2_feature_cache.main(
            [
                "--predictions",
                str(tmp_path / "missing.csv"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--workers",
                "9",
                "--output-json",
                str(tmp_path / "report.json"),
            ]
        )


def test_content_pe_v2_cache_rejects_bad_max_pending(tmp_path: Path):
    with pytest.raises(ValueError, match="--max-pending must be at least 1"):
        build_content_pe_v2_feature_cache.main(
            [
                "--predictions",
                str(tmp_path / "missing.csv"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--max-pending",
                "0",
                "--output-json",
                str(tmp_path / "report.json"),
            ]
        )


def test_load_valid_cached_features_rejects_bad_shape(tmp_path: Path):
    cache_path = tmp_path / "bad.npz"
    np.savez(cache_path, features=np.zeros(3, dtype=np.float32))

    assert build_content_pe_v2_feature_cache._load_valid_cached_features(cache_path) is None


def test_load_valid_cached_features_rejects_nonfinite(tmp_path: Path):
    cache_path = tmp_path / "bad.npz"
    features = np.zeros(len(CONTENT_PE_V2_FEATURE_NAMES), dtype=np.float32)
    features[0] = np.nan
    np.savez(cache_path, features=features)

    assert build_content_pe_v2_feature_cache._load_valid_cached_features(cache_path) is None


def test_load_valid_cached_features_accepts_content_pe_v2_shape(tmp_path: Path):
    cache_path = tmp_path / "good.npz"
    features = np.ones(len(CONTENT_PE_V2_FEATURE_NAMES), dtype=np.float32)
    np.savez(cache_path, features=features)

    loaded = build_content_pe_v2_feature_cache._load_valid_cached_features(cache_path)

    assert loaded is not None
    np.testing.assert_array_equal(loaded, features)


def test_content_pe_v2_cache_streams_unique_rows_and_applies_limit(tmp_path: Path, monkeypatch):
    predictions = tmp_path / "predictions.csv"
    output_json = tmp_path / "report.json"
    rows = [
        {"source_path": "a.exe", "source_sha256": "a" * 64},
        {"source_path": "a-duplicate.exe", "source_sha256": "a" * 64},
        {"source_path": "b.exe", "source_sha256": "b" * 64},
        {"source_path": "c.exe", "source_sha256": "c" * 64},
    ]
    with predictions.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "source_sha256"])
        writer.writeheader()
        writer.writerows(rows)

    processed = []

    def fake_build_one(payload):
        row, cache_dir_text = payload
        processed.append((row["source_sha256"], cache_dir_text))
        return {"status": "created", "zero": False}

    monkeypatch.setattr(build_content_pe_v2_feature_cache, "_build_one", fake_build_one)

    exit_code = build_content_pe_v2_feature_cache.main(
        [
            "--predictions",
            str(predictions),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--workers",
            "1",
            "--smoke",
            "--limit",
            "2",
            "--output-json",
            str(output_json),
        ]
    )
    report = json.loads(output_json.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert [item[0] for item in processed] == ["a" * 64, "b" * 64]
    assert report["input_rows"] == 4
    assert report["deduplicated_rows_before_limit"] == 3
    assert report["unique_rows"] == 2


def test_content_pe_v2_cache_records_bounded_failures(tmp_path: Path, monkeypatch):
    predictions = tmp_path / "predictions.csv"
    output_json = tmp_path / "report.json"
    rows = [
        {"source_path": f"bad-{idx}.exe", "source_sha256": f"{idx:064x}"}
        for idx in range(25)
    ]
    with predictions.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "source_sha256"])
        writer.writeheader()
        writer.writerows(rows)

    def failing_build_one(_payload):
        raise RuntimeError("simulated extraction failure")

    monkeypatch.setattr(build_content_pe_v2_feature_cache, "_build_one", failing_build_one)

    exit_code = build_content_pe_v2_feature_cache.main(
        [
            "--predictions",
            str(predictions),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--workers",
            "1",
            "--output-json",
            str(output_json),
        ]
    )
    report = json.loads(output_json.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert report["counts"]["failed"] == len(rows)
    assert len(report["failure_examples"]) == 20
