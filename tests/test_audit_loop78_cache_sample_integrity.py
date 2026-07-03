from __future__ import annotations

import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_loop78_cache_sample_integrity import audit_cache_sample_integrity  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "source_sha256", "label", "sample_index", "split"])
        writer.writeheader()
        writer.writerows(rows)


def _write_cache(path: Path, *, label: int, sha: str, pe_dim: int = 2, stat_dim: int = 1, light_dim: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        byte_sequence=np.array([77, 90], dtype=np.uint8),
        pe_features=np.ones(pe_dim, dtype=np.float32),
        stat_features=np.ones(stat_dim, dtype=np.float32),
        lightweight_features=np.ones(light_dim, dtype=np.float32),
        label=np.array(label, dtype=np.int64),
        source_sha256=np.array(sha),
    )


def _write_manifest(path: Path, samples: list[dict], *, pe_dim: int = 2, stat_dim: int = 1, light_dim: int = 3) -> None:
    path.write_text(
        json.dumps(
            {
                "max_byte_length": 2,
                "pe_feature_dim": pe_dim,
                "stat_feature_dim": stat_dim,
                "lightweight_feature_dim": light_dim,
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )


def _case_payload(tmp_path: Path, *, cache_label: int = 0, cache_sha: str | None = None, pe_dim: int = 2, cache_exists: bool = True):
    source_path = tmp_path / "data" / "sample.exe"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"MZ")
    sha = "a" * 64
    cache_path = tmp_path / "cache" / "sample.npz"
    if cache_exists:
        _write_cache(cache_path, label=cache_label, sha=cache_sha or sha, pe_dim=pe_dim)
    split_csv = tmp_path / "split.csv"
    manifest_json = tmp_path / "manifest.json"
    rows = [{"source_path": str(source_path), "source_sha256": sha, "label": "0", "sample_index": "0", "split": "train"}]
    _write_csv(split_csv, rows)
    _write_manifest(
        manifest_json,
        [{"source_path": str(source_path), "source_sha256": sha, "cache_path": str(cache_path), "label": 0}],
    )
    return split_csv, manifest_json


def test_cache_sample_integrity_passes_valid_npz():
    with _case_dir("loop78_valid") as tmp_path:
        split_csv, manifest_json = _case_payload(tmp_path)
        payload = audit_cache_sample_integrity(
            split_csv=split_csv,
            manifest_json=manifest_json,
            sample_size=1,
            enforce_20w=False,
            enforce_label_balance=False,
            detail_output_csv=tmp_path / "detail.csv",
        )

    assert payload["audit_ready"] is True
    assert payload["sampled_rows"] == 1
    assert payload["failed_rows"] == 0
    assert payload["status_counts"] == {"pass": 1}


def test_cache_sample_integrity_detects_missing_cache():
    with _case_dir("loop78_missing_cache") as tmp_path:
        split_csv, manifest_json = _case_payload(tmp_path, cache_exists=False)
        payload = audit_cache_sample_integrity(
            split_csv=split_csv,
            manifest_json=manifest_json,
            sample_size=1,
            enforce_20w=False,
            enforce_label_balance=False,
        )

    assert payload["audit_ready"] is False
    assert payload["issue_counts"] == {"cache_file_missing": 1}


def test_cache_sample_integrity_detects_shape_mismatch():
    with _case_dir("loop78_shape_mismatch") as tmp_path:
        split_csv, manifest_json = _case_payload(tmp_path, pe_dim=1)
        payload = audit_cache_sample_integrity(
            split_csv=split_csv,
            manifest_json=manifest_json,
            sample_size=1,
            enforce_20w=False,
            enforce_label_balance=False,
        )

    assert payload["audit_ready"] is False
    assert payload["issue_counts"] == {"shape_mismatch": 1}


def test_cache_sample_integrity_detects_label_mismatch():
    with _case_dir("loop78_label_mismatch") as tmp_path:
        split_csv, manifest_json = _case_payload(tmp_path, cache_label=1)
        payload = audit_cache_sample_integrity(
            split_csv=split_csv,
            manifest_json=manifest_json,
            sample_size=1,
            enforce_20w=False,
            enforce_label_balance=False,
        )

    assert payload["audit_ready"] is False
    assert payload["issue_counts"] == {"label_mismatch": 1}


def test_cache_sample_integrity_detects_source_sha_mismatch():
    with _case_dir("loop78_sha_mismatch") as tmp_path:
        split_csv, manifest_json = _case_payload(tmp_path, cache_sha="b" * 64)
        payload = audit_cache_sample_integrity(
            split_csv=split_csv,
            manifest_json=manifest_json,
            sample_size=1,
            enforce_20w=False,
            enforce_label_balance=False,
        )

    assert payload["audit_ready"] is False
    assert payload["issue_counts"] == {"source_sha256_mismatch_npz_manifest": 1}


def test_cache_sample_integrity_blocks_non_20w_when_enforced():
    with _case_dir("loop78_shape_enforced") as tmp_path:
        split_csv, manifest_json = _case_payload(tmp_path)
        payload = audit_cache_sample_integrity(
            split_csv=split_csv,
            manifest_json=manifest_json,
            sample_size=1,
            enforce_20w=True,
            enforce_label_balance=True,
        )

    assert payload["audit_ready"] is False
    assert payload["shape_failures"]
    assert payload["failed_rows"] == 0
