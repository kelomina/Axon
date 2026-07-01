from __future__ import annotations

import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_split_duplicate_sources import audit_split_duplicate_sources  # noqa: E402


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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_duplicate_audit_reports_clean_split():
    with _case_dir("duplicate_audit_clean") as tmp_path:
        split_csv = tmp_path / "split.csv"
        _write_split(
            split_csv,
            [
                {"source_path": "data/a.exe", "source_sha256": "", "label": "0", "sample_index": "0", "split": "train"},
                {"source_path": "data/b.exe", "source_sha256": "", "label": "1", "sample_index": "1", "split": "val"},
            ],
        )

        payload = audit_split_duplicate_sources(split_csv=split_csv)

    assert payload["has_duplicates"] is False
    assert payload["duplicate_groups"] == 0
    assert payload["duplicate_extra_rows"] == 0


def test_duplicate_audit_flags_same_path_duplicate():
    with _case_dir("duplicate_audit_same_path") as tmp_path:
        split_csv = tmp_path / "split.csv"
        _write_split(
            split_csv,
            [
                {"source_path": "data/a.exe", "source_sha256": "", "label": "0", "sample_index": "0", "split": "train"},
                {"source_path": "data/a.exe", "source_sha256": "", "label": "0", "sample_index": "1", "split": "train"},
            ],
        )

        payload = audit_split_duplicate_sources(split_csv=split_csv)

    assert payload["has_duplicates"] is True
    assert payload["duplicate_groups"] == 1
    assert payload["duplicate_extra_rows"] == 1
    assert payload["same_path_duplicate_groups"] == 1
    assert payload["cross_split_groups"] == 0
    assert payload["cross_label_groups"] == 0


def test_duplicate_audit_flags_cross_split_and_cross_label_conflict():
    with _case_dir("duplicate_audit_conflict") as tmp_path:
        split_csv = tmp_path / "split.csv"
        sample_sha = "a" * 64
        _write_split(
            split_csv,
            [
                {"source_path": "data/train/a.exe", "source_sha256": sample_sha, "label": "0", "sample_index": "0", "split": "train"},
                {"source_path": "data/test/a-copy.exe", "source_sha256": sample_sha, "label": "1", "sample_index": "1", "split": "test"},
            ],
        )

        payload = audit_split_duplicate_sources(split_csv=split_csv)

    assert payload["has_duplicates"] is True
    assert payload["has_cross_split_duplicates"] is True
    assert payload["has_cross_label_duplicates"] is True
    assert payload["cross_split_groups"] == 1
    assert payload["cross_label_groups"] == 1
    assert payload["cross_split_pattern_counts"] == {"test|train": 1}
    assert payload["label_pattern_counts"] == {"0|1": 1}


def test_duplicate_audit_infers_sha_from_sha_like_path_stem():
    with _case_dir("duplicate_audit_path_sha") as tmp_path:
        split_csv = tmp_path / "split.csv"
        sample_sha = "b" * 64
        _write_split(
            split_csv,
            [
                {"source_path": f"data/benign/{sample_sha}.exe", "source_sha256": "", "label": "0", "sample_index": "0", "split": "val"},
                {"source_path": f"data/benign/copy/{sample_sha}.bin", "source_sha256": "", "label": "0", "sample_index": "1", "split": "val"},
            ],
        )

        payload = audit_split_duplicate_sources(split_csv=split_csv, include_path=True, include_sha=True)
        path_only = audit_split_duplicate_sources(split_csv=split_csv, include_path=True, include_sha=False)

    assert payload["has_duplicates"] is True
    assert payload["duplicate_groups"] == 1
    assert path_only["has_duplicates"] is False


def test_duplicate_audit_writes_detail_rows():
    with _case_dir("duplicate_audit_details") as tmp_path:
        split_csv = tmp_path / "split.csv"
        output_csv = tmp_path / "details.csv"
        _write_split(
            split_csv,
            [
                {"source_path": "data/a.exe", "source_sha256": "", "label": "0", "sample_index": "0", "split": "train"},
                {"source_path": "data/a.exe", "source_sha256": "", "label": "0", "sample_index": "1", "split": "val"},
            ],
        )

        payload = audit_split_duplicate_sources(split_csv=split_csv, output_csv=output_csv)
        rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8-sig", newline="")))

    assert payload["detail_rows"] == 2
    assert len(rows) == 2
    assert {row["cross_split"] for row in rows} == {"true"}
