import csv
import json
import shutil
import sys
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from recover_missing_feature_cache import read_missing_rows, recover_one, save_feature_cache_npz, update_manifest  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_csv(path: Path, rows: list[dict]) -> Path:
    fieldnames = ["source_path", "source_sha256", "cache_path", "label", "split", "sample_index"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_read_missing_rows_deduplicates_by_source_and_label():
    with _case_dir("recover_missing_rows") as tmp_path:
        source = tmp_path / "sample.exe"
        source.write_bytes(b"MZ")
        first = _write_csv(
            tmp_path / "first.csv",
            [{"source_path": str(source), "source_sha256": "a" * 64, "cache_path": "a.npz", "label": "1", "split": "test", "sample_index": "1"}],
        )
        second = _write_csv(
            tmp_path / "second.csv",
            [{"source_path": str(source), "source_sha256": "a" * 64, "cache_path": "b.npz", "label": "1", "split": "test", "sample_index": "2"}],
        )

        rows = read_missing_rows([first, second])

    assert len(rows) == 1
    assert rows[0]["label"] == 1
    assert rows[0]["sample_index"] == "1"
    assert rows[0]["source_sha256"] == "a" * 64


def test_read_missing_rows_prefers_original_source_path_for_worktree_rows():
    with _case_dir("recover_missing_rows_original_source") as tmp_path:
        original = tmp_path / "data" / "sample.exe"
        materialized = tmp_path / "data" / "random_20w_worktree" / "sample.exe"
        original.parent.mkdir(parents=True)
        materialized.parent.mkdir(parents=True)
        original.write_bytes(b"MZpayload")
        materialized.write_bytes(b"MZpayload")
        missing_csv = tmp_path / "missing.csv"
        with missing_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["source_path", "original_source_path", "source_sha256", "label", "split", "sample_index"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "source_path": str(materialized),
                    "original_source_path": str(original),
                    "label": "1",
                    "source_sha256": "b" * 64,
                    "split": "test",
                    "sample_index": "3",
                }
            )

        rows = read_missing_rows([missing_csv])

    assert len(rows) == 1
    assert rows[0]["source_path"] == str(original.absolute())
    assert rows[0]["materialized_source_path"] == str(materialized.absolute())
    assert rows[0]["source_sha256"] == "b" * 64


def test_update_manifest_appends_only_successful_unique_rows():
    with _case_dir("recover_manifest") as tmp_path:
        manifest_path = tmp_path / "manifest_hash.json"
        config = type(
            "Config",
            (),
            {
                "max_byte_length": 8192,
                "pe_feature_dim": 256,
                "stat_feature_dim": 49,
                "lightweight_feature_dim": 256,
                "strict_pe_parsing": True,
                "allow_pe_fallback": False,
                "pe_schema_version": "fixed_v2",
                "pe_fixed_section_slots": 32,
            },
        )()
        results = [
            {
                "status": "extracted",
                "source_path": str(tmp_path / "a.exe"),
                "cache_path": str(tmp_path / "a.npz"),
                "label": 1,
                "source_sha256": "aaa",
            },
            {
                "status": "feature_extract_failed",
                "source_path": str(tmp_path / "b.exe"),
                "cache_path": str(tmp_path / "b.npz"),
            },
            {
                "status": "cache_hit",
                "source_path": str(tmp_path / "a.exe"),
                "cache_path": str(tmp_path / "a.npz"),
                "label": 1,
                "source_sha256": "aaa",
            },
        ]

        added = update_manifest(manifest_path, config, "hash", results, dry_run=False)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert added == 1
    assert payload["cache_config_hash"] == "hash"
    assert payload["max_byte_length"] == 8192
    assert payload["cache_storage_format"] == "compressed"
    assert len(payload["samples"]) == 1
    assert payload["samples"][0]["source_sha256"] == "aaa"


def test_save_feature_cache_npz_can_write_uncompressed_npz():
    with _case_dir("recover_uncompressed_npz") as tmp_path:
        cache_path = tmp_path / "sample.npz"

        save_feature_cache_npz(
            cache_path,
            {"byte_sequence": np.array([77, 90], dtype=np.uint8), "label": 1},
            "uncompressed",
        )

        with zipfile.ZipFile(cache_path) as archive:
            compression_types = {item.compress_type for item in archive.infolist()}

    assert compression_types == {zipfile.ZIP_STORED}


def test_recover_one_rejects_source_sha256_mismatch_before_writing_cache():
    with _case_dir("recover_hash_mismatch") as tmp_path:
        source = tmp_path / "sample.exe"
        source.write_bytes(b"MZactual-content")
        cache_dir = tmp_path / "cache"
        payload = {
            "source_path": str(source),
            "label": 1,
            "expected_source_sha256": "0" * 64,
            "cache_dir": str(cache_dir),
            "config_hash": "abc12345",
            "storage_format": "uncompressed",
            "config_dict": {
                "max_file_size": 1024,
                "max_byte_length": 8,
                "pe_feature_dim": 143,
                "stat_feature_dim": 49,
                "lightweight_feature_dim": 2,
                "strict_pe_parsing": True,
                "allow_pe_fallback": False,
                "pe_schema_version": "fixed_v2",
                "pe_fixed_section_slots": 32,
            },
        }

        result = recover_one(payload)

    assert result["status"] == "source_sha256_mismatch"
    assert result["expected_source_sha256"] == "0" * 64
    assert not any(cache_dir.glob("*.npz"))


def test_recover_one_requires_expected_source_sha256():
    with _case_dir("recover_hash_missing") as tmp_path:
        source = tmp_path / "sample.exe"
        source.write_bytes(b"MZactual-content")
        cache_dir = tmp_path / "cache"
        payload = {
            "source_path": str(source),
            "label": 1,
            "expected_source_sha256": "",
            "cache_dir": str(cache_dir),
            "config_hash": "abc12345",
            "storage_format": "uncompressed",
            "config_dict": {
                "max_file_size": 1024,
                "max_byte_length": 8,
                "pe_feature_dim": 143,
                "stat_feature_dim": 49,
                "lightweight_feature_dim": 2,
                "strict_pe_parsing": True,
                "allow_pe_fallback": False,
                "pe_schema_version": "fixed_v2",
                "pe_fixed_section_slots": 32,
            },
        }

        result = recover_one(payload)

    assert result["status"] == "missing_expected_source_sha256"
    assert result["source_sha256"]
    assert not any(cache_dir.glob("*.npz"))
