from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import build_content_cert_feature_cache, build_content_string_feature_cache
from scripts import materialize_loop127_content_pe_sidecars as loop127_sidecars
from scripts import train_stage2_cache_matrix as stage2


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_content_cache_path_rejects_invalid_source_sha256(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    row = {"source_path": str(tmp_path / "sample.exe"), "source_sha256": "../escape"}

    with pytest.raises(ValueError, match="invalid source_sha256"):
        stage2.content_cache_path_for_row(row, cache_dir)

    assert not (tmp_path / "escape.npz").exists()


def test_content_builder_rejects_source_sha256_mismatch_before_writing(tmp_path: Path, monkeypatch):
    source_path = tmp_path / "sample.exe"
    source_path.write_bytes(b"actual-content")
    wrong_sha = _sha256_bytes(b"different-content")
    cache_dir = tmp_path / "cache"
    row = {"source_path": str(source_path), "source_sha256": wrong_sha}

    def fail_if_extractor_is_called(_path):
        raise AssertionError("extractor should not run when source_sha256 mismatches source_path bytes")

    monkeypatch.setattr(build_content_string_feature_cache, "_content_string_features_from_path", fail_if_extractor_is_called)

    with pytest.raises(ValueError, match="source_sha256_mismatch"):
        build_content_string_feature_cache._build_one((row, str(cache_dir)))

    assert not (cache_dir / f"{wrong_sha}.npz").exists()


@pytest.mark.parametrize(
    ("module", "feature_names_attr", "extractor_name"),
    [
        (build_content_string_feature_cache, "CONTENT_STRING_FEATURE_NAMES", "_content_string_features_from_path"),
        (build_content_cert_feature_cache, "CONTENT_CERT_FEATURE_NAMES", "_content_cert_features_from_path"),
    ],
)
def test_string_and_cert_builders_refresh_invalid_existing_cache(
    tmp_path: Path,
    monkeypatch,
    module,
    feature_names_attr: str,
    extractor_name: str,
):
    source_path = tmp_path / "sample.exe"
    source_path.write_bytes(b"MZcontent")
    source_sha = _sha256_bytes(b"MZcontent")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_path = cache_dir / f"{source_sha}.npz"
    np.savez(cache_path, features=np.zeros(3, dtype=np.float32))
    feature_dim = len(getattr(module, feature_names_attr))
    replacement = np.ones(feature_dim, dtype=np.float32)
    monkeypatch.setattr(module, extractor_name, lambda _path: replacement)

    result = module._build_one(({"source_path": str(source_path), "source_sha256": source_sha}, str(cache_dir)))

    assert result == {"status": "refreshed_invalid", "zero": False}
    with np.load(cache_path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["features"], replacement)


@pytest.mark.parametrize("module", [build_content_string_feature_cache, build_content_cert_feature_cache])
def test_string_and_cert_builders_stream_unique_rows_and_report_failures(tmp_path: Path, monkeypatch, module):
    predictions = tmp_path / "predictions.csv"
    output_json = tmp_path / "report.json"
    rows = [
        {"source_path": "a.exe", "source_sha256": "a" * 64},
        {"source_path": "a-dup.exe", "source_sha256": "a" * 64},
        {"source_path": "bad.exe", "source_sha256": "b" * 64},
    ]
    with predictions.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "source_sha256"])
        writer.writeheader()
        writer.writerows(rows)

    processed = []

    def fake_build_one(payload):
        row, _cache_dir_text = payload
        processed.append(row["source_sha256"])
        if row["source_sha256"] == "b" * 64:
            raise RuntimeError("simulated sidecar failure")
        return {"status": "created", "zero": False}

    monkeypatch.setattr(module, "_build_one", fake_build_one)

    exit_code = module.main(
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
    assert processed == ["a" * 64, "b" * 64]
    assert report["input_rows"] == 3
    assert report["deduplicated_rows_before_limit"] == 2
    assert report["unique_rows"] == 2
    assert report["counts"]["created"] == 1
    assert report["counts"]["failed"] == 1
    assert len(report["failure_examples"]) == 1


@pytest.mark.parametrize("module", [build_content_string_feature_cache, build_content_cert_feature_cache])
def test_string_and_cert_cache_limit_requires_smoke(tmp_path: Path, module):
    with pytest.raises(ValueError, match="--limit requires --smoke"):
        module.main(
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


@pytest.mark.parametrize("module", [build_content_string_feature_cache, build_content_cert_feature_cache])
def test_string_and_cert_cache_negative_limit_rejected(tmp_path: Path, module):
    with pytest.raises(ValueError, match="--limit must be non-negative"):
        module.main(
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


@pytest.mark.parametrize("module", [build_content_string_feature_cache, build_content_cert_feature_cache])
def test_string_and_cert_builders_apply_smoke_limit_after_dedup(tmp_path: Path, monkeypatch, module):
    predictions = tmp_path / "predictions.csv"
    output_json = tmp_path / "report.json"
    rows = [
        {"source_path": "a.exe", "source_sha256": "a" * 64},
        {"source_path": "a-dup.exe", "source_sha256": "a" * 64},
        {"source_path": "b.exe", "source_sha256": "b" * 64},
        {"source_path": "c.exe", "source_sha256": "c" * 64},
    ]
    with predictions.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "source_sha256"])
        writer.writeheader()
        writer.writerows(rows)

    processed = []

    def fake_build_one(payload):
        row, _cache_dir_text = payload
        processed.append(row["source_sha256"])
        return {"status": "created", "zero": False}

    monkeypatch.setattr(module, "_build_one", fake_build_one)

    exit_code = module.main(
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
    assert processed == ["a" * 64, "b" * 64]
    assert report["input_rows"] == 4
    assert report["deduplicated_rows_before_limit"] == 3
    assert report["unique_rows"] == 2


@pytest.mark.parametrize("module", [build_content_string_feature_cache, build_content_cert_feature_cache])
def test_string_and_cert_builders_reject_unbounded_worker_settings(tmp_path: Path, module):
    with pytest.raises(ValueError, match="--workers must be <= 8"):
        module.main(
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

    with pytest.raises(ValueError, match="--max-pending must be at least 1"):
        module.main(
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


def test_loop127_sidecar_build_one_rejects_source_sha256_mismatch(tmp_path: Path):
    source_path = tmp_path / "sample.exe"
    source_path.write_bytes(b"actual-content")
    wrong_sha = _sha256_bytes(b"different-content")
    row = {
        "source_path": str(source_path),
        "source_sha256": wrong_sha,
        "sample_index": "1",
        "split": "train",
        "label": "1",
    }

    result = loop127_sidecars._build_one((row, str(tmp_path / "v1"), str(tmp_path / "v2"), True))

    assert result["failed"] is True
    assert result["failure_reason"].startswith("source_sha256_mismatch for " + str(source_path))
    assert not (tmp_path / "v1" / f"{wrong_sha}.npz").exists()
    assert not (tmp_path / "v2" / f"{wrong_sha}.npz").exists()


def test_save_feature_npz_atomic_removes_temp_file_on_save_failure(tmp_path: Path, monkeypatch):
    cache_path = tmp_path / "features.npz"

    def failing_savez(path, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(stage2.np, "savez", failing_savez)

    with pytest.raises(OSError, match="simulated write failure"):
        stage2.save_feature_npz_atomic(cache_path, np.ones(4, dtype=np.float32))

    assert not cache_path.exists()
    assert list(tmp_path.glob("*.tmp.npz")) == []


def test_save_feature_npz_atomic_retries_transient_replace_failure(tmp_path: Path, monkeypatch):
    cache_path = tmp_path / "features.npz"
    original_replace = Path.replace
    attempts = {"count": 0}

    def flaky_replace(self, target):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise PermissionError("simulated transient replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)

    stage2.save_feature_npz_atomic(cache_path, np.ones(4, dtype=np.float32))

    assert attempts["count"] == 2
    assert cache_path.exists()
    assert list(tmp_path.glob("*.tmp.npz")) == []
