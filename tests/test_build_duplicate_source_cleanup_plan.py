from __future__ import annotations

import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_duplicate_source_cleanup_plan import build_duplicate_cleanup_plan  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_duplicate_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "duplicate_group_id",
        "duplicate_key",
        "group_size",
        "labels",
        "splits",
        "cross_label",
        "cross_split",
        "same_path_rows",
        "source_path",
        "source_sha256",
        "label",
        "sample_index",
        "split",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dup_row(group: str, path: str, label: str, sample_index: str, split: str, *, cross_label: str = "false") -> dict:
    return {
        "duplicate_group_id": group,
        "duplicate_key": f"sha:{group.rjust(64, 'a')[-64:]}",
        "group_size": "2",
        "labels": "0|1" if cross_label == "true" else label,
        "splits": split,
        "cross_label": cross_label,
        "cross_split": "false",
        "same_path_rows": "false",
        "source_path": path,
        "source_sha256": "",
        "label": label,
        "sample_index": sample_index,
        "split": split,
    }


def test_same_label_duplicate_creates_one_replace_plan_row_when_test_not_mutated():
    with _case_dir("duplicate_cleanup_same_label") as tmp_path:
        duplicate_csv = tmp_path / "duplicates.csv"
        plan_csv = tmp_path / "plan.csv"
        review_csv = tmp_path / "review.csv"
        output_json = tmp_path / "summary.json"
        _write_duplicate_rows(
            duplicate_csv,
            [
                _dup_row("1", "data/a.exe", "1", "10", "train"),
                _dup_row("1", "data/a-copy.exe", "1", "20", "val"),
            ],
        )

        summary = build_duplicate_cleanup_plan(
            duplicate_csv=duplicate_csv,
            output_plan_csv=plan_csv,
            output_review_csv=review_csv,
            output_json=output_json,
        )
        plan_rows = list(csv.DictReader(plan_csv.open("r", encoding="utf-8-sig", newline="")))
        review_rows = list(csv.DictReader(review_csv.open("r", encoding="utf-8-sig", newline="")))

    assert summary["auto_plan_rows"] == 1
    assert summary["manual_review_rows"] == 0
    assert len(plan_rows) == 1
    assert plan_rows[0]["source_path"] == "data/a.exe"
    assert plan_rows[0]["plan_action"] == "exclude_and_replace"
    assert plan_rows[0]["replacement_required"] == "true"
    assert plan_rows[0]["replacement_label"] == "1"
    assert review_rows == []


def test_cross_label_duplicate_goes_to_review_only():
    with _case_dir("duplicate_cleanup_cross_label") as tmp_path:
        duplicate_csv = tmp_path / "duplicates.csv"
        plan_csv = tmp_path / "plan.csv"
        review_csv = tmp_path / "review.csv"
        output_json = tmp_path / "summary.json"
        _write_duplicate_rows(
            duplicate_csv,
            [
                _dup_row("2", "data/white.exe", "0", "11", "val", cross_label="true"),
                _dup_row("2", "data/black.exe", "1", "12", "test", cross_label="true"),
            ],
        )

        summary = build_duplicate_cleanup_plan(
            duplicate_csv=duplicate_csv,
            output_plan_csv=plan_csv,
            output_review_csv=review_csv,
            output_json=output_json,
        )
        plan_rows = list(csv.DictReader(plan_csv.open("r", encoding="utf-8-sig", newline="")))
        review_rows = list(csv.DictReader(review_csv.open("r", encoding="utf-8-sig", newline="")))

    assert summary["auto_plan_rows"] == 0
    assert summary["manual_review_rows"] == 2
    assert plan_rows == []
    assert len(review_rows) == 2
    assert all(row["manual_label_verdict"] == "" for row in review_rows)
    assert all("requires human label adjudication" in row["review_reason"] for row in review_rows)


def test_keep_policy_can_prefer_train_for_same_label_duplicates():
    with _case_dir("duplicate_cleanup_prefer_train") as tmp_path:
        duplicate_csv = tmp_path / "duplicates.csv"
        plan_csv = tmp_path / "plan.csv"
        review_csv = tmp_path / "review.csv"
        output_json = tmp_path / "summary.json"
        _write_duplicate_rows(
            duplicate_csv,
            [
                _dup_row("3", "data/test.exe", "0", "1", "test"),
                _dup_row("3", "data/train.exe", "0", "2", "train"),
            ],
        )

        build_duplicate_cleanup_plan(
            duplicate_csv=duplicate_csv,
            output_plan_csv=plan_csv,
            output_review_csv=review_csv,
            output_json=output_json,
            keep_policy="prefer_train",
            freeze_test=False,
        )
        plan_rows = list(csv.DictReader(plan_csv.open("r", encoding="utf-8-sig", newline="")))

    assert len(plan_rows) == 1
    assert plan_rows[0]["source_path"] == "data/test.exe"
    assert plan_rows[0]["split"] == "test"


def test_same_label_duplicate_that_would_mutate_test_goes_to_review_by_default():
    with _case_dir("duplicate_cleanup_freeze_test") as tmp_path:
        duplicate_csv = tmp_path / "duplicates.csv"
        plan_csv = tmp_path / "plan.csv"
        review_csv = tmp_path / "review.csv"
        output_json = tmp_path / "summary.json"
        _write_duplicate_rows(
            duplicate_csv,
            [
                _dup_row("5", "data/train.exe", "1", "1", "train"),
                _dup_row("5", "data/test.exe", "1", "2", "test"),
            ],
        )

        summary = build_duplicate_cleanup_plan(
            duplicate_csv=duplicate_csv,
            output_plan_csv=plan_csv,
            output_review_csv=review_csv,
            output_json=output_json,
            keep_policy="prefer_train",
        )
        plan_rows = list(csv.DictReader(plan_csv.open("r", encoding="utf-8-sig", newline="")))
        review_rows = list(csv.DictReader(review_csv.open("r", encoding="utf-8-sig", newline="")))

    assert summary["auto_plan_rows"] == 0
    assert summary["manual_review_rows"] == 2
    assert summary["group_action_counts"] == {"manual_review_required_frozen_test": 1}
    assert plan_rows == []
    assert all("frozen test split" in row["review_reason"] for row in review_rows)


def test_summary_json_is_written():
    with _case_dir("duplicate_cleanup_json") as tmp_path:
        duplicate_csv = tmp_path / "duplicates.csv"
        plan_csv = tmp_path / "plan.csv"
        review_csv = tmp_path / "review.csv"
        output_json = tmp_path / "summary.json"
        _write_duplicate_rows(duplicate_csv, [_dup_row("4", "data/a.exe", "1", "1", "train"), _dup_row("4", "data/b.exe", "1", "2", "train")])

        summary = build_duplicate_cleanup_plan(
            duplicate_csv=duplicate_csv,
            output_plan_csv=plan_csv,
            output_review_csv=review_csv,
            output_json=output_json,
        )
        persisted = json.loads(output_json.read_text(encoding="utf-8"))

    assert persisted["schema"] == "axon_duplicate_source_cleanup_plan_v1"
    assert persisted["auto_plan_rows"] == summary["auto_plan_rows"]
