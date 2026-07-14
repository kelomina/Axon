import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_split_cache_coverage import audit_split_cache_coverage  # noqa: E402


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(row)
        return rows


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_audit_split_cache_coverage_uses_original_source_path_and_writes_missing_rows():
    with _case_dir("audit_split_cache_coverage") as tmp_path:
        cache_dir = tmp_path / "data" / ".cache"
        cache_dir.mkdir(parents=True)
        cache_path = cache_dir / "hit_38672ba0.npz"
        np.savez_compressed(cache_path, byte_sequence=np.array([77, 90]), pe_features=np.array([1.0]), label=1)

        original = tmp_path / "data" / "待拉黑" / "hit.exe"
        materialized = tmp_path / "data" / "random_20w_worktree" / "待拉黑" / "hit.exe"
        missing = tmp_path / "data" / "random_20w_worktree" / "待加入白名单" / "missing.exe"
        manifest_path = cache_dir / "manifest_38672ba0.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "source_path": str(original),
                            "cache_path": str(cache_path),
                            "label": 1,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        split_csv = tmp_path / "split.csv"
        with split_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["source_path", "original_source_path", "label", "sample_index", "split"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "source_path": str(materialized),
                    "original_source_path": str(original),
                    "label": "1",
                    "sample_index": "0",
                    "split": "test",
                }
            )
            writer.writerow(
                {
                    "source_path": str(missing),
                    "original_source_path": "",
                    "label": "0",
                    "sample_index": "1",
                    "split": "test",
                }
            )

        output_json = tmp_path / "coverage.json"
        missing_csv = tmp_path / "missing.csv"
        payload = audit_split_cache_coverage(
            split_csv=split_csv,
            manifest_path=manifest_path,
            split="test",
            output_json=output_json,
            missing_cache_output=missing_csv,
        )
        missing_rows = _read_csv_rows(missing_csv)

    assert payload["total_rows"] == 2
    assert payload["covered_rows"] == 1
    assert payload["missing_rows"] == 1
    assert payload["coverage_ratio"] == 0.5
    assert payload["manifest_match_counts"] == {"source_path": 1}
    assert payload["missing_label_counts"] == {"0": 1}
    assert missing_rows[0]["source_path"] == str(missing)


def test_audit_split_cache_coverage_handles_all_split_without_materializing_rows():
    with _case_dir("audit_split_cache_coverage_all") as tmp_path:
        cache_dir = tmp_path / "data" / ".cache"
        cache_dir.mkdir(parents=True)
        cache_path = cache_dir / "hit_38672ba0.npz"
        np.savez_compressed(cache_path, byte_sequence=np.array([77, 90]), pe_features=np.array([1.0]), label=1)
        source = tmp_path / "data" / "hit.exe"
        manifest_path = cache_dir / "manifest_38672ba0.json"
        manifest_path.write_text(
            json.dumps({"samples": [{"source_path": str(source), "cache_path": str(cache_path), "label": 1}]}),
            encoding="utf-8",
        )

        split_csv = tmp_path / "split.csv"
        with split_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["source_path", "original_source_path", "label", "sample_index", "split"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "source_path": str(source),
                    "original_source_path": "",
                    "label": "1",
                    "sample_index": "0",
                    "split": "train",
                }
            )
            writer.writerow(
                {
                    "source_path": str(tmp_path / "data" / "missing.exe"),
                    "original_source_path": "",
                    "label": "0",
                    "sample_index": "1",
                    "split": "val",
                }
            )

        payload = audit_split_cache_coverage(
            split_csv=split_csv,
            manifest_path=manifest_path,
            split="all",
            output_json=tmp_path / "coverage.json",
            missing_cache_output=tmp_path / "missing.csv",
        )
        missing_rows = _read_csv_rows(tmp_path / "missing.csv")

    assert payload["total_rows"] == 2
    assert payload["covered_rows"] == 1
    assert payload["missing_rows"] == 1
    assert payload["missing_split_counts"] == {"val": 1}
    assert missing_rows[0]["split"] == "val"


def test_audit_split_cache_coverage_handles_empty_filtered_split():
    with _case_dir("audit_split_cache_coverage_empty") as tmp_path:
        cache_dir = tmp_path / "data" / ".cache"
        cache_dir.mkdir(parents=True)
        manifest_path = cache_dir / "manifest_38672ba0.json"
        manifest_path.write_text(json.dumps({"samples": []}), encoding="utf-8")
        split_csv = tmp_path / "split.csv"
        with split_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["source_path", "original_source_path", "label", "sample_index", "split"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "source_path": str(tmp_path / "data" / "sample.exe"),
                    "original_source_path": "",
                    "label": "0",
                    "sample_index": "0",
                    "split": "train",
                }
            )

        payload = audit_split_cache_coverage(
            split_csv=split_csv,
            manifest_path=manifest_path,
            split="test",
            output_json=tmp_path / "coverage.json",
            missing_cache_output=tmp_path / "missing.csv",
        )
        missing_rows = _read_csv_rows(tmp_path / "missing.csv")

    assert payload["total_rows"] == 0
    assert payload["covered_rows"] == 0
    assert payload["missing_rows"] == 0
    assert payload["coverage_ratio"] == 0.0
    assert missing_rows == []
