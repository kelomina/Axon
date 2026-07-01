from __future__ import annotations

import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from apply_manual_review_verdicts import build_plan  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_empty_manual_verdicts_create_no_actions():
    with _case_dir("manual_verdict_empty") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        _write_csv(
            split_csv,
            ["source_path", "label", "sample_index", "split"],
            [{"source_path": "data/a.exe", "label": "0", "sample_index": "1", "split": "train"}],
        )
        _write_csv(
            review_csv,
            ["source_path", "source_sha256", "label", "manual_label_verdict", "recommended_action"],
            [{"source_path": "data/a.exe", "source_sha256": "", "label": "0", "manual_label_verdict": "", "recommended_action": ""}],
        )

        rows, summary = build_plan(review_csv=review_csv, split_csv=split_csv)

    assert rows == []
    assert summary["planned_rows"] == 0
    assert summary["ignored_rows"] == 1
    assert summary["review_split_counts"] == {"train": 1}
    assert summary["review_rows_in_test_split"] == 0


def test_relabel_verdict_without_target_label_requires_manual_target():
    with _case_dir("manual_verdict_relabel") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        _write_csv(
            split_csv,
            ["source_path", "label", "sample_index", "split"],
            [{"source_path": "data/a.exe", "label": "0", "sample_index": "1", "split": "train"}],
        )
        _write_csv(
            review_csv,
            ["source_path", "source_sha256", "label", "manual_label_verdict", "recommended_action"],
            [{
                "source_path": "data/a.exe",
                "source_sha256": "",
                "label": "0",
                "manual_label_verdict": "label_wrong",
                "recommended_action": "relabel_train_only",
            }],
        )

        rows, summary = build_plan(review_csv=review_csv, split_csv=split_csv)

    assert rows[0]["plan_action"] == "needs_manual_target_label"
    assert rows[0]["original_label"] == 0
    assert rows[0]["planned_label"] == 0
    assert rows[0]["replacement_required"] == "false"
    assert rows[0]["usable_for_training_policy"] == "false"
    assert summary["training_policy_rows"] == 0


def test_relabel_verdict_uses_explicit_corrected_label_without_replacement():
    with _case_dir("manual_verdict_relabel_target") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        _write_csv(
            split_csv,
            ["source_path", "label", "sample_index", "split"],
            [{"source_path": "data/a.exe", "label": "0", "sample_index": "1", "split": "train"}],
        )
        _write_csv(
            review_csv,
            ["source_path", "source_sha256", "label", "corrected_label", "manual_label_verdict", "recommended_action"],
            [{
                "source_path": "data/a.exe",
                "source_sha256": "",
                "label": "0",
                "corrected_label": "1",
                "manual_label_verdict": "label_wrong",
                "recommended_action": "relabel_train_only",
            }],
        )

        rows, summary = build_plan(review_csv=review_csv, split_csv=split_csv)

    assert rows[0]["plan_action"] == "relabel"
    assert rows[0]["original_label"] == 0
    assert rows[0]["planned_label"] == 1
    assert rows[0]["replacement_required"] == "false"
    assert rows[0]["usable_for_training_policy"] == "true"
    assert summary["training_policy_rows"] == 1


def test_feature_broken_row_requires_replacement_instead_of_self_fill():
    with _case_dir("manual_verdict_replace") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        _write_csv(
            split_csv,
            ["source_path", "label", "sample_index", "split"],
            [{"source_path": "data/bad.exe", "label": "1", "sample_index": "7", "split": "val"}],
        )
        _write_csv(
            review_csv,
            ["source_path", "source_sha256", "label", "manual_label_verdict", "recommended_action"],
            [{
                "source_path": "data/bad.exe",
                "source_sha256": "",
                "label": "1",
                "manual_label_verdict": "feature_broken",
                "recommended_action": "replace_sample",
            }],
        )

        rows, summary = build_plan(review_csv=review_csv, split_csv=split_csv)

    assert rows[0]["plan_action"] == "exclude_and_replace"
    assert rows[0]["replacement_required"] == "true"
    assert rows[0]["replacement_label"] == "1"
    assert rows[0]["usable_for_training_policy"] == "false"
    assert summary["replacement_required"] == 1
    assert summary["replacement_counts_by_original_label"] == {"1": 1}


def test_exclude_verdict_takes_priority_over_conflicting_relabel_action():
    with _case_dir("manual_verdict_conflicting_replace") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        _write_csv(
            split_csv,
            ["source_path", "label", "sample_index", "split"],
            [{"source_path": "data/bad.exe", "label": "0", "sample_index": "8", "split": "train"}],
        )
        _write_csv(
            review_csv,
            ["source_path", "source_sha256", "label", "manual_label_verdict", "recommended_action"],
            [{
                "source_path": "data/bad.exe",
                "source_sha256": "",
                "label": "0",
                "manual_label_verdict": "feature_broken",
                "recommended_action": "relabel_train_only",
            }],
        )

        rows, summary = build_plan(review_csv=review_csv, split_csv=split_csv)

    assert rows[0]["plan_action"] == "exclude_and_replace"
    assert rows[0]["replacement_required"] == "true"
    assert rows[0]["replacement_label"] == "0"
    assert rows[0]["planned_label"] == 0
    assert summary["replacement_required"] == 1


def test_test_split_verdict_is_withheld_by_default():
    with _case_dir("manual_verdict_test") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        _write_csv(
            split_csv,
            ["source_path", "label", "sample_index", "split"],
            [{"source_path": "data/test.exe", "label": "0", "sample_index": "9", "split": "test"}],
        )
        _write_csv(
            review_csv,
            ["source_path", "source_sha256", "label", "manual_label_verdict", "recommended_action"],
            [{
                "source_path": "data/test.exe",
                "source_sha256": "",
                "label": "0",
                "manual_label_verdict": "label_wrong",
                "recommended_action": "relabel_train_only",
            }],
        )

        rows, summary = build_plan(review_csv=review_csv, split_csv=split_csv)

    assert rows[0]["plan_action"] == "held_out_test_verdict_only"
    assert rows[0]["planned_label"] == 0
    assert rows[0]["usable_for_training_policy"] == "false"
    assert summary["training_policy_rows"] == 0
    assert summary["review_split_counts"] == {"test": 1}
    assert summary["review_label_split_counts"] == {"test:0": 1}
    assert summary["review_rows_in_test_split"] == 1


def test_blank_test_split_review_is_reported_even_without_action():
    with _case_dir("manual_verdict_blank_test") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        _write_csv(
            split_csv,
            ["source_path", "label", "sample_index", "split"],
            [{"source_path": "data/test.exe", "label": "1", "sample_index": "10", "split": "test"}],
        )
        _write_csv(
            review_csv,
            ["source_path", "source_sha256", "label", "manual_label_verdict", "recommended_action"],
            [{
                "source_path": "data/test.exe",
                "source_sha256": "",
                "label": "1",
                "manual_label_verdict": "",
                "recommended_action": "",
            }],
        )

        rows, summary = build_plan(review_csv=review_csv, split_csv=split_csv)

    assert rows == []
    assert summary["ignored_rows"] == 1
    assert summary["training_policy_rows"] == 0
    assert summary["review_split_counts"] == {"test": 1}
    assert summary["review_label_split_counts"] == {"test:1": 1}
    assert summary["review_rows_in_test_split"] == 1


def test_review_sha_can_match_split_path_filename():
    sample_sha = "a" * 64
    with _case_dir("manual_verdict_sha_match") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        _write_csv(
            split_csv,
            ["source_path", "label", "sample_index", "split"],
            [{"source_path": f"data/white/{sample_sha}.exe", "label": "0", "sample_index": "11", "split": "train"}],
        )
        _write_csv(
            review_csv,
            ["source_path", "source_sha256", "label", "corrected_label", "manual_label_verdict", "recommended_action"],
            [{
                "source_path": "",
                "source_sha256": sample_sha,
                "label": "0",
                "corrected_label": "1",
                "manual_label_verdict": "label_wrong",
                "recommended_action": "relabel_train_only",
            }],
        )

        rows, summary = build_plan(review_csv=review_csv, split_csv=split_csv)

    assert rows[0]["sample_index"] == "11"
    assert rows[0]["plan_action"] == "relabel"
    assert summary["missing_split_rows"] == 0


def test_explicit_sha_match_wins_over_path_stem_sha_collision():
    sample_sha = "b" * 64
    with _case_dir("manual_verdict_sha_collision") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        _write_csv(
            split_csv,
            ["source_path", "source_sha256", "label", "sample_index", "split"],
            [
                {
                    "source_path": f"data/path_alias/{sample_sha}.exe",
                    "source_sha256": "",
                    "label": "0",
                    "sample_index": "21",
                    "split": "train",
                },
                {
                    "source_path": "data/real_sha/real.exe",
                    "source_sha256": sample_sha,
                    "label": "1",
                    "sample_index": "22",
                    "split": "val",
                },
            ],
        )
        _write_csv(
            review_csv,
            ["source_path", "source_sha256", "label", "manual_label_verdict", "recommended_action"],
            [{
                "source_path": "",
                "source_sha256": sample_sha,
                "label": "1",
                "manual_label_verdict": "",
                "recommended_action": "",
            }],
        )

        rows, summary = build_plan(review_csv=review_csv, split_csv=split_csv)

    assert rows == []
    assert summary["ignored_rows"] == 1
    assert summary["missing_split_rows"] == 0
    assert summary["review_split_counts"] == {"val": 1}
    assert summary["review_label_split_counts"] == {"val:1": 1}


def test_path_stem_sha_alias_is_ignored_when_split_row_has_explicit_sha():
    alias_sha = "c" * 64
    real_sha = "d" * 64
    with _case_dir("manual_verdict_stem_ignored_with_real_sha") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        _write_csv(
            split_csv,
            ["source_path", "source_sha256", "label", "sample_index", "split"],
            [{
                "source_path": f"data/has_real_sha/{alias_sha}.exe",
                "source_sha256": real_sha,
                "label": "0",
                "sample_index": "31",
                "split": "train",
            }],
        )
        _write_csv(
            review_csv,
            ["source_path", "source_sha256", "label", "manual_label_verdict", "recommended_action"],
            [{
                "source_path": "",
                "source_sha256": alias_sha,
                "label": "0",
                "manual_label_verdict": "",
                "recommended_action": "",
            }],
        )

        rows, summary = build_plan(review_csv=review_csv, split_csv=split_csv)

    assert rows == []
    assert summary["ignored_rows"] == 0
    assert summary["missing_split_rows"] == 1
    assert summary["review_split_counts"] == {}
