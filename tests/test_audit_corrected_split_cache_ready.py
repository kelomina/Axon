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

from audit_corrected_split_cache_ready import audit_corrected_split_cache_ready  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_split(path: Path, rows: list[dict]) -> None:
    fieldnames = ["source_path", "source_sha256", "label", "sample_index", "split"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_cache_npz(
    path: Path,
    *,
    label: int = 0,
    source_sha256: str = "a" * 64,
    byte_length: int = 8,
    pe_dim: int = 4,
    stat_dim: int = 3,
    include_stat_features: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "byte_sequence": np.zeros(byte_length, dtype=np.uint8),
        "pe_features": np.zeros(pe_dim, dtype=np.float32),
        "label": np.asarray(label, dtype=np.int64),
        "source_sha256": np.asarray(source_sha256),
    }
    if include_stat_features:
        arrays["stat_features"] = np.zeros(stat_dim, dtype=np.float32)
    np.savez(path, **arrays)


def _manifest_payload(samples: list[dict], *, byte_length: int = 8, pe_dim: int = 4, stat_dim: int = 3) -> dict:
    return {
        "max_byte_length": byte_length,
        "pe_feature_dim": pe_dim,
        "stat_feature_dim": stat_dim,
        "samples": samples,
    }


def test_cache_ready_when_all_rows_have_existing_cache_with_shape_disabled():
    with _case_dir("corrected_cache_ready") as tmp_path:
        cache_path = tmp_path / "cache" / "a.npz"
        _write_cache_npz(cache_path, label=0, source_sha256="a" * 64)
        source_path = tmp_path / "data" / ("a" * 64)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                _manifest_payload(
                    [{"source_path": str(source_path), "source_sha256": "a" * 64, "label": 0, "cache_path": str(cache_path)}]
                )
            ),
            encoding="utf-8",
        )
        split_csv = tmp_path / "split.csv"
        _write_split(
            split_csv,
            [{"source_path": str(source_path), "source_sha256": "a" * 64, "label": "0", "sample_index": "0", "split": "train"}],
        )

        payload = audit_corrected_split_cache_ready(
            split_csv=split_csv,
            manifest_json=manifest_path,
            enforce_shape=False,
        )

    assert payload["total_rows"] == 1
    assert payload["covered_rows"] == 1
    assert payload["missing_rows"] == 0
    assert payload["metadata_checked_rows"] == 1
    assert payload["metadata_failure_rows"] == 0
    assert payload["cache_ready"] is True


def test_missing_cache_rows_are_reported():
    with _case_dir("corrected_cache_missing") as tmp_path:
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"samples": []}), encoding="utf-8")
        split_csv = tmp_path / "split.csv"
        missing_source = tmp_path / "data" / ("b" * 64)
        _write_split(split_csv, [{"source_path": str(missing_source), "label": "1", "sample_index": "3", "split": "val"}])
        missing_csv = tmp_path / "missing.csv"

        payload = audit_corrected_split_cache_ready(
            split_csv=split_csv,
            manifest_json=manifest_path,
            missing_cache_output=missing_csv,
            enforce_shape=False,
        )
        missing_rows = list(csv.DictReader(missing_csv.open("r", encoding="utf-8-sig", newline="")))

    assert payload["cache_ready"] is False
    assert payload["missing_rows"] == 1
    assert payload["missing_label_counts"] == {"1": 1}
    assert payload["missing_split_counts"] == {"val": 1}
    assert missing_rows[0]["reason"] == "manifest_missing"


def test_shape_enforcement_flags_non_20w_split():
    with _case_dir("corrected_cache_shape") as tmp_path:
        cache_path = tmp_path / "cache" / "a.npz"
        _write_cache_npz(cache_path, label=0, source_sha256="a" * 64)
        source_path = tmp_path / "data" / ("a" * 64)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                _manifest_payload(
                    [{"source_path": str(source_path), "source_sha256": "a" * 64, "label": 0, "cache_path": str(cache_path)}]
                )
            ),
            encoding="utf-8",
        )
        split_csv = tmp_path / "split.csv"
        _write_split(
            split_csv,
            [{"source_path": str(source_path), "source_sha256": "a" * 64, "label": "0", "sample_index": "0", "split": "train"}],
        )

        payload = audit_corrected_split_cache_ready(
            split_csv=split_csv,
            manifest_json=manifest_path,
        )

    assert payload["covered_rows"] == 1
    assert payload["missing_rows"] == 0
    assert payload["shape_failures"]
    assert payload["cache_ready"] is False


def test_label_balance_drift_is_reported_but_not_blocking_when_not_enforced(monkeypatch):
    with _case_dir("corrected_cache_label_drift_report") as tmp_path:
        import audit_corrected_split_cache_ready as module

        monkeypatch.setattr(module, "EXPECTED_TOTAL", 2)
        monkeypatch.setattr(module, "EXPECTED_SPLIT_COUNTS", {"train": 2})
        monkeypatch.setattr(module, "EXPECTED_LABEL_SPLIT_COUNTS", {"train": {"0": 1, "1": 1}, "val": {}, "test": {}})

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        source_a = tmp_path / "data" / ("a" * 64)
        source_b = tmp_path / "data" / ("b" * 64)
        cache_a = cache_dir / "a.npz"
        cache_b = cache_dir / "b.npz"
        _write_cache_npz(cache_a, label=1, source_sha256="a" * 64)
        _write_cache_npz(cache_b, label=1, source_sha256="b" * 64)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                _manifest_payload(
                    [
                        {"source_path": str(source_a), "source_sha256": "a" * 64, "label": 1, "cache_path": str(cache_a)},
                        {"source_path": str(source_b), "source_sha256": "b" * 64, "label": 1, "cache_path": str(cache_b)},
                    ]
                )
            ),
            encoding="utf-8",
        )
        split_csv = tmp_path / "split.csv"
        _write_split(
            split_csv,
            [
                {"source_path": str(source_a), "source_sha256": "a" * 64, "label": "1", "sample_index": "0", "split": "train"},
                {"source_path": str(source_b), "source_sha256": "b" * 64, "label": "1", "sample_index": "1", "split": "train"},
            ],
        )

        payload = audit_corrected_split_cache_ready(
            split_csv=split_csv,
            manifest_json=manifest_path,
        )

    assert payload["shape_failures"] == []
    assert payload["cache_ready"] is True
    assert payload["label_balance_enforced"] is False
    assert payload["label_balance_drift"] == ["train:{'1': 2}"]


def test_label_balance_drift_blocks_when_enforced(monkeypatch):
    with _case_dir("corrected_cache_label_drift_enforced") as tmp_path:
        import audit_corrected_split_cache_ready as module

        monkeypatch.setattr(module, "EXPECTED_TOTAL", 2)
        monkeypatch.setattr(module, "EXPECTED_SPLIT_COUNTS", {"train": 2})
        monkeypatch.setattr(module, "EXPECTED_LABEL_SPLIT_COUNTS", {"train": {"0": 1, "1": 1}, "val": {}, "test": {}})

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        source_a = tmp_path / "data" / ("a" * 64)
        source_b = tmp_path / "data" / ("b" * 64)
        cache_a = cache_dir / "a.npz"
        cache_b = cache_dir / "b.npz"
        _write_cache_npz(cache_a, label=1, source_sha256="a" * 64)
        _write_cache_npz(cache_b, label=1, source_sha256="b" * 64)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                _manifest_payload(
                    [
                        {"source_path": str(source_a), "source_sha256": "a" * 64, "label": 1, "cache_path": str(cache_a)},
                        {"source_path": str(source_b), "source_sha256": "b" * 64, "label": 1, "cache_path": str(cache_b)},
                    ]
                )
            ),
            encoding="utf-8",
        )
        split_csv = tmp_path / "split.csv"
        _write_split(
            split_csv,
            [
                {"source_path": str(source_a), "source_sha256": "a" * 64, "label": "1", "sample_index": "0", "split": "train"},
                {"source_path": str(source_b), "source_sha256": "b" * 64, "label": "1", "sample_index": "1", "split": "train"},
            ],
        )

        payload = audit_corrected_split_cache_ready(
            split_csv=split_csv,
            manifest_json=manifest_path,
            enforce_label_balance=True,
        )

    assert payload["label_balance_enforced"] is True
    assert payload["shape_failures"]
    assert payload["cache_ready"] is False


def test_cache_metadata_label_mismatch_blocks():
    with _case_dir("corrected_cache_label_mismatch") as tmp_path:
        cache_path = tmp_path / "cache" / "a.npz"
        _write_cache_npz(cache_path, label=1, source_sha256="a" * 64)
        source_path = tmp_path / "data" / ("a" * 64)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                _manifest_payload(
                    [{"source_path": str(source_path), "source_sha256": "a" * 64, "label": 0, "cache_path": str(cache_path)}]
                )
            ),
            encoding="utf-8",
        )
        split_csv = tmp_path / "split.csv"
        _write_split(
            split_csv,
            [{"source_path": str(source_path), "source_sha256": "a" * 64, "label": "0", "sample_index": "0", "split": "train"}],
        )

        payload = audit_corrected_split_cache_ready(
            split_csv=split_csv,
            manifest_json=manifest_path,
            enforce_shape=False,
        )

    assert payload["cache_ready"] is False
    assert payload["metadata_failure_rows"] == 1
    assert payload["metadata_issue_counts"]["label_mismatch"] == 1


def test_cache_metadata_source_sha_mismatch_blocks():
    with _case_dir("corrected_cache_sha_mismatch") as tmp_path:
        cache_path = tmp_path / "cache" / "a.npz"
        _write_cache_npz(cache_path, label=0, source_sha256="b" * 64)
        source_path = tmp_path / "data" / ("a" * 64)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                _manifest_payload(
                    [{"source_path": str(source_path), "source_sha256": "a" * 64, "label": 0, "cache_path": str(cache_path)}]
                )
            ),
            encoding="utf-8",
        )
        split_csv = tmp_path / "split.csv"
        _write_split(
            split_csv,
            [{"source_path": str(source_path), "source_sha256": "a" * 64, "label": "0", "sample_index": "0", "split": "train"}],
        )

        payload = audit_corrected_split_cache_ready(
            split_csv=split_csv,
            manifest_json=manifest_path,
            enforce_shape=False,
        )

    assert payload["cache_ready"] is False
    assert payload["metadata_issue_counts"]["source_sha256_mismatch_npz_manifest"] == 1


def test_cache_metadata_missing_required_field_blocks():
    with _case_dir("corrected_cache_missing_field") as tmp_path:
        cache_path = tmp_path / "cache" / "a.npz"
        _write_cache_npz(cache_path, label=0, source_sha256="a" * 64, include_stat_features=False)
        source_path = tmp_path / "data" / ("a" * 64)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                _manifest_payload(
                    [{"source_path": str(source_path), "source_sha256": "a" * 64, "label": 0, "cache_path": str(cache_path)}]
                )
            ),
            encoding="utf-8",
        )
        split_csv = tmp_path / "split.csv"
        _write_split(
            split_csv,
            [{"source_path": str(source_path), "source_sha256": "a" * 64, "label": "0", "sample_index": "0", "split": "train"}],
        )

        payload = audit_corrected_split_cache_ready(
            split_csv=split_csv,
            manifest_json=manifest_path,
            enforce_shape=False,
        )

    assert payload["cache_ready"] is False
    assert payload["metadata_issue_counts"]["npz_missing_fields"] == 1


def test_cache_metadata_shape_mismatch_blocks():
    with _case_dir("corrected_cache_shape_mismatch") as tmp_path:
        cache_path = tmp_path / "cache" / "a.npz"
        _write_cache_npz(cache_path, label=0, source_sha256="a" * 64, pe_dim=5)
        source_path = tmp_path / "data" / ("a" * 64)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                _manifest_payload(
                    [{"source_path": str(source_path), "source_sha256": "a" * 64, "label": 0, "cache_path": str(cache_path)}],
                    pe_dim=4,
                )
            ),
            encoding="utf-8",
        )
        split_csv = tmp_path / "split.csv"
        _write_split(
            split_csv,
            [{"source_path": str(source_path), "source_sha256": "a" * 64, "label": "0", "sample_index": "0", "split": "train"}],
        )

        payload = audit_corrected_split_cache_ready(
            split_csv=split_csv,
            manifest_json=manifest_path,
            enforce_shape=False,
        )

    assert payload["cache_ready"] is False
    assert payload["metadata_issue_counts"]["shape_mismatch"] == 1


def test_cache_metadata_validation_can_be_explicitly_disabled():
    with _case_dir("corrected_cache_metadata_disabled") as tmp_path:
        cache_path = tmp_path / "cache" / "a.npz"
        cache_path.parent.mkdir()
        cache_path.write_bytes(b"npz-placeholder")
        source_path = tmp_path / "data" / ("a" * 64)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                _manifest_payload(
                    [{"source_path": str(source_path), "source_sha256": "a" * 64, "label": 0, "cache_path": str(cache_path)}]
                )
            ),
            encoding="utf-8",
        )
        split_csv = tmp_path / "split.csv"
        _write_split(
            split_csv,
            [{"source_path": str(source_path), "source_sha256": "a" * 64, "label": "0", "sample_index": "0", "split": "train"}],
        )

        payload = audit_corrected_split_cache_ready(
            split_csv=split_csv,
            manifest_json=manifest_path,
            enforce_shape=False,
            validate_cache_metadata=False,
        )

    assert payload["cache_ready"] is True
    assert payload["cache_metadata_validation_enabled"] is False
    assert payload["metadata_checked_rows"] == 0
