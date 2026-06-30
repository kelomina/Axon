from __future__ import annotations

import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_corrected_split_cache_recovery_plan import build_plan, write_markdown  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_missing_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "label", "sample_index", "split", "reason", "expected_cache_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_empty_missing_csv_builds_no_recovery_needed_plan():
    with _case_dir("corrected_cache_recovery_empty") as tmp_path:
        missing_csv = tmp_path / "missing.csv"
        _write_missing_csv(missing_csv, [])

        plan = build_plan(
            missing_csv=missing_csv,
            checkpoint=Path("models/base.pt"),
            cache_dir=Path("data/.cache"),
            recovery_output_json=Path("reports/recovery.json"),
            audit_command="audit again",
        )

    assert plan["needs_recovery"] is False
    assert plan["missing_summary"]["rows"] == 0
    assert "--dry-run" in plan["commands"]["dry_run"]
    assert "--dry-run" not in plan["commands"]["recover"]
    assert "--storage-format uncompressed" in plan["commands"]["recover"]


def test_missing_rows_are_summarized_and_markdown_is_written():
    with _case_dir("corrected_cache_recovery_missing") as tmp_path:
        missing_csv = tmp_path / "missing.csv"
        _write_missing_csv(
            missing_csv,
            [
                {
                    "source_path": r"E:\data\a.exe",
                    "label": "1",
                    "sample_index": "10",
                    "split": "train",
                    "reason": "manifest_missing",
                    "expected_cache_path": "",
                },
                {
                    "source_path": r"E:\data\b.dll",
                    "label": "0",
                    "sample_index": "11",
                    "split": "val",
                    "reason": "cache_file_missing",
                    "expected_cache_path": r"E:\cache\b.npz",
                },
            ],
        )

        plan = build_plan(
            missing_csv=missing_csv,
            checkpoint=Path("models/base.pt"),
            cache_dir=Path("data/.cache"),
            recovery_output_json=Path("reports/recovery.json"),
            audit_command="audit again",
            workers=2,
            backend="thread",
            storage_format="compressed",
        )
        md_path = tmp_path / "plan.md"
        write_markdown(plan, md_path)
        md_text = md_path.read_text(encoding="utf-8")

    assert plan["needs_recovery"] is True
    assert plan["missing_summary"]["label_counts"] == {"0": 1, "1": 1}
    assert plan["missing_summary"]["split_counts"] == {"train": 1, "val": 1}
    assert plan["missing_summary"]["reason_counts"] == {"cache_file_missing": 1, "manifest_missing": 1}
    assert "--workers 2" in plan["commands"]["recover"]
    assert "--backend thread" in plan["commands"]["recover"]
    assert "--storage-format compressed" in plan["commands"]["recover"]
    assert "Post-recovery strict audit" in md_text
