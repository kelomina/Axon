import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_replacement_plan_from_cache_recovery_failures import (  # noqa: E402
    build_replacement_plan_from_failures,
)


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
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "source_sha256", "label", "sample_index", "split"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_recovery(path: Path, failed_examples: list[dict]) -> None:
    path.write_text(json.dumps({"failed_examples": failed_examples}), encoding="utf-8")


def test_builds_same_label_replacement_plan_for_feature_extract_failure():
    with _case_dir("cache_failure_plan") as tmp_path:
        split_csv = tmp_path / "split.csv"
        recovery_json = tmp_path / "recovery.json"
        _write_csv(
            split_csv,
            [
                {
                    "source_path": "looks-benign-name.exe",
                    "source_sha256": "a" * 64,
                    "label": "1",
                    "sample_index": "9",
                    "split": "val",
                }
            ],
        )
        _write_recovery(
            recovery_json,
            [{"status": "feature_extract_failed", "source_path": "looks-benign-name.exe"}],
        )

        rows, summary = build_replacement_plan_from_failures(split_csv=split_csv, recovery_json=recovery_json)

    assert summary["plan_ready"] is True
    assert summary["replacement_counts_by_split_label"] == {"val:1": 1}
    assert "never inferred from names" in summary["identity_feature_policy"]
    assert rows[0]["replacement_label"] == "1"
    assert rows[0]["planned_label"] == "1"
    assert rows[0]["cache_recovery_status"] == "feature_extract_failed"


def test_ignores_successful_cache_recovery_rows():
    with _case_dir("cache_failure_plan_success_only") as tmp_path:
        split_csv = tmp_path / "split.csv"
        recovery_json = tmp_path / "recovery.json"
        _write_csv(
            split_csv,
            [{"source_path": "sample.exe", "source_sha256": "a" * 64, "label": "0", "sample_index": "1", "split": "train"}],
        )
        _write_recovery(recovery_json, [{"status": "extracted", "source_path": "sample.exe"}])

        rows, summary = build_replacement_plan_from_failures(split_csv=split_csv, recovery_json=recovery_json)

    assert rows == []
    assert summary["plan_ready"] is False
    assert summary["failed_rows"] == 0


def test_blocks_ambiguous_hash_match():
    with _case_dir("cache_failure_plan_ambiguous") as tmp_path:
        split_csv = tmp_path / "split.csv"
        recovery_json = tmp_path / "recovery.json"
        _write_csv(
            split_csv,
            [
                {"source_path": "a.exe", "source_sha256": "b" * 64, "label": "0", "sample_index": "1", "split": "train"},
                {"source_path": "b.exe", "source_sha256": "b" * 64, "label": "0", "sample_index": "2", "split": "val"},
            ],
        )
        _write_recovery(
            recovery_json,
            [{"status": "feature_extract_failed", "source_sha256": "b" * 64}],
        )

        rows, summary = build_replacement_plan_from_failures(split_csv=split_csv, recovery_json=recovery_json)

    assert rows == []
    assert summary["plan_ready"] is False
    assert summary["blocked_rows"] == 1
    assert summary["match_counts"]["ambiguous_source_sha256"] == 1
