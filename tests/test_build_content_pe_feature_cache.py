import csv
from concurrent.futures import Future
import inspect
import json

import pytest
import numpy as np

from scripts import build_content_pe_feature_cache
from kvd_features.content_pe_v1 import CONTENT_PE_FEATURE_NAMES


def test_content_pe_cache_limit_requires_smoke(tmp_path):
    with pytest.raises(ValueError, match="--limit requires --smoke"):
        build_content_pe_feature_cache.main(
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


def test_content_pe_cache_negative_limit_rejected(tmp_path):
    with pytest.raises(ValueError, match="--limit must be non-negative"):
        build_content_pe_feature_cache.main(
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


def test_content_pe_cache_rejects_too_many_workers(tmp_path):
    with pytest.raises(ValueError, match="--workers must be <= 8"):
        build_content_pe_feature_cache.main(
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


def test_content_pe_cache_rejects_bad_max_pending(tmp_path):
    with pytest.raises(ValueError, match="--max-pending must be at least 1"):
        build_content_pe_feature_cache.main(
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


def test_load_valid_cached_features_rejects_bad_shape(tmp_path):
    cache_path = tmp_path / "bad.npz"
    np.savez(cache_path, features=np.zeros(3, dtype=np.float32))

    assert build_content_pe_feature_cache._load_valid_cached_features(cache_path) is None


def test_load_valid_cached_features_rejects_nonfinite(tmp_path):
    cache_path = tmp_path / "bad.npz"
    features = np.zeros(len(CONTENT_PE_FEATURE_NAMES), dtype=np.float32)
    features[0] = np.inf
    np.savez(cache_path, features=features)

    assert build_content_pe_feature_cache._load_valid_cached_features(cache_path) is None


def test_load_valid_cached_features_accepts_content_pe_v1_shape(tmp_path):
    cache_path = tmp_path / "good.npz"
    features = np.ones(len(CONTENT_PE_FEATURE_NAMES), dtype=np.float32)
    np.savez(cache_path, features=features)

    loaded = build_content_pe_feature_cache._load_valid_cached_features(cache_path)

    assert loaded is not None
    np.testing.assert_array_equal(loaded, features)


def test_content_pe_cache_streams_unique_rows_and_applies_limit(tmp_path, monkeypatch):
    predictions = tmp_path / "predictions.csv"
    cache_dir = tmp_path / "cache"
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

    monkeypatch.setattr(build_content_pe_feature_cache, "_build_one", fake_build_one)

    build_content_pe_feature_cache.main(
        [
            "--predictions",
            str(predictions),
            "--cache-dir",
            str(cache_dir),
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

    assert [item[0] for item in processed] == ["a" * 64, "b" * 64]
    assert report["input_rows"] == 4
    assert report["deduplicated_rows_before_limit"] == 3
    assert report["unique_rows"] == 2
    assert report["counts"]["created"] == 2


def test_content_pe_cache_records_bounded_failures(tmp_path, monkeypatch):
    predictions = tmp_path / "predictions.csv"
    cache_dir = tmp_path / "cache"
    output_json = tmp_path / "report.json"
    rows = [
        {"source_path": f"bad-{idx}.exe", "source_sha256": f"{idx:064x}"}
        for idx in range(build_content_pe_feature_cache.MAX_FAILURE_EXAMPLES + 5)
    ]
    with predictions.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "source_sha256"])
        writer.writeheader()
        writer.writerows(rows)

    def failing_build_one(payload):
        row, _cache_dir_text = payload
        raise RuntimeError(f"cannot build {row['source_sha256']}")

    monkeypatch.setattr(build_content_pe_feature_cache, "_build_one", failing_build_one)

    exit_code = build_content_pe_feature_cache.main(
        [
            "--predictions",
            str(predictions),
            "--cache-dir",
            str(cache_dir),
            "--workers",
            "1",
            "--output-json",
            str(output_json),
        ]
    )
    report = json.loads(output_json.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert report["counts"]["failed"] == len(rows)
    assert len(report["failure_examples"]) == build_content_pe_feature_cache.MAX_FAILURE_EXAMPLES


def test_content_pe_cache_caps_pending_window(tmp_path, monkeypatch):
    predictions = tmp_path / "predictions.csv"
    cache_dir = tmp_path / "cache"
    output_json = tmp_path / "report.json"
    rows = [
        {"source_path": f"ok-{idx}.exe", "source_sha256": f"{idx:064x}"}
        for idx in range(10)
    ]
    with predictions.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "source_sha256"])
        writer.writeheader()
        writer.writerows(rows)

    class FakeExecutor:
        max_observed = 0
        current_pending = 0

        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, payload):
            future = Future()
            future.set_result({"status": "created", "zero": False})
            FakeExecutor.current_pending += 1
            FakeExecutor.max_observed = max(FakeExecutor.max_observed, FakeExecutor.current_pending)
            return future

    original_drain = build_content_pe_feature_cache._drain_completed

    def tracking_drain(*args, **kwargs):
        pending = original_drain(*args, **kwargs)
        FakeExecutor.current_pending = len(pending)
        return pending

    monkeypatch.setattr(build_content_pe_feature_cache, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(build_content_pe_feature_cache, "_drain_completed", tracking_drain)

    exit_code = build_content_pe_feature_cache.main(
        [
            "--predictions",
            str(predictions),
            "--cache-dir",
            str(cache_dir),
            "--workers",
            "2",
            "--max-pending",
            "3",
            "--output-json",
            str(output_json),
        ]
    )
    report = json.loads(output_json.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert FakeExecutor.max_observed <= 3
    assert report["max_pending_tasks"] == 3
    assert report["counts"]["created"] == len(rows)


def test_content_pe_cache_main_avoids_payload_and_executor_map_lists():
    source = inspect.getsource(build_content_pe_feature_cache.main)

    assert "payloads =" not in source
    assert "executor.map" not in source
    assert "read_prediction_rows" not in source
    assert "rows.extend" not in source
    assert "pending[future]" in source
