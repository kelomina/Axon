from __future__ import annotations

import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "label", "sample_index", "split"])
        writer.writeheader()
        writer.writerows(rows)


def test_cache_ready_when_all_rows_have_existing_cache_with_shape_disabled():
    with _case_dir("corrected_cache_ready") as tmp_path:
        cache_path = tmp_path / "cache" / "a.npz"
        cache_path.parent.mkdir()
        cache_path.write_bytes(b"npz-placeholder")
        source_path = tmp_path / "data" / ("a" * 64)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps({"samples": [{"source_path": str(source_path), "source_sha256": "a" * 64, "cache_path": str(cache_path)}]}),
            encoding="utf-8",
        )
        split_csv = tmp_path / "split.csv"
        _write_split(split_csv, [{"source_path": str(source_path), "label": "0", "sample_index": "0", "split": "train"}])

        payload = audit_corrected_split_cache_ready(
            split_csv=split_csv,
            manifest_json=manifest_path,
            enforce_shape=False,
        )

    assert payload["total_rows"] == 1
    assert payload["covered_rows"] == 1
    assert payload["missing_rows"] == 0
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
        cache_path.parent.mkdir()
        cache_path.write_bytes(b"npz-placeholder")
        source_path = tmp_path / "data" / ("a" * 64)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps({"samples": [{"source_path": str(source_path), "source_sha256": "a" * 64, "cache_path": str(cache_path)}]}),
            encoding="utf-8",
        )
        split_csv = tmp_path / "split.csv"
        _write_split(split_csv, [{"source_path": str(source_path), "label": "0", "sample_index": "0", "split": "train"}])

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
        cache_a.write_bytes(b"npz-placeholder")
        cache_b.write_bytes(b"npz-placeholder")
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "samples": [
                        {"source_path": str(source_a), "source_sha256": "a" * 64, "cache_path": str(cache_a)},
                        {"source_path": str(source_b), "source_sha256": "b" * 64, "cache_path": str(cache_b)},
                    ]
                }
            ),
            encoding="utf-8",
        )
        split_csv = tmp_path / "split.csv"
        _write_split(
            split_csv,
            [
                {"source_path": str(source_a), "label": "1", "sample_index": "0", "split": "train"},
                {"source_path": str(source_b), "label": "1", "sample_index": "1", "split": "train"},
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
        cache_a.write_bytes(b"npz-placeholder")
        cache_b.write_bytes(b"npz-placeholder")
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "samples": [
                        {"source_path": str(source_a), "source_sha256": "a" * 64, "cache_path": str(cache_a)},
                        {"source_path": str(source_b), "source_sha256": "b" * 64, "cache_path": str(cache_b)},
                    ]
                }
            ),
            encoding="utf-8",
        )
        split_csv = tmp_path / "split.csv"
        _write_split(
            split_csv,
            [
                {"source_path": str(source_a), "label": "1", "sample_index": "0", "split": "train"},
                {"source_path": str(source_b), "label": "1", "sample_index": "1", "split": "train"},
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
