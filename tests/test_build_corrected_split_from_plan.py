from __future__ import annotations

import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_corrected_split_from_plan import build_corrected_split  # noqa: E402


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


SPLIT_FIELDS = ["source_path", "label", "sample_index", "split"]
PLAN_FIELDS = [
    "source_path",
    "source_sha256",
    "sample_index",
    "split",
    "original_label",
    "planned_label",
    "plan_action",
    "replacement_required",
    "replacement_label",
    "usable_for_training_policy",
]
CANDIDATE_FIELDS = ["source_path", "label", "source_sha256"]


def test_empty_plan_preserves_split_size_and_labels():
    with _case_dir("corrected_split_empty") as tmp_path:
        split_csv = tmp_path / "split.csv"
        plan_csv = tmp_path / "plan.csv"
        candidates_csv = tmp_path / "candidates.csv"
        _write_csv(
            split_csv,
            SPLIT_FIELDS,
            [
                {"source_path": "data/a.exe", "label": "0", "sample_index": "0", "split": "train"},
                {"source_path": "data/b.exe", "label": "1", "sample_index": "1", "split": "val"},
            ],
        )
        _write_csv(plan_csv, PLAN_FIELDS, [])
        _write_csv(candidates_csv, CANDIDATE_FIELDS, [])

        rows, summary = build_corrected_split(
            split_csv=split_csv,
            plan_csv=plan_csv,
            candidate_csv=candidates_csv,
        )

    assert len(rows) == 2
    assert [row["label"] for row in rows] == ["0", "1"]
    assert summary["excluded_rows"] == 0
    assert summary["replacement_summary"]["selected_replacements"] == 0


def test_relabel_and_replacement_keep_total_count():
    with _case_dir("corrected_split_replace") as tmp_path:
        split_csv = tmp_path / "split.csv"
        plan_csv = tmp_path / "plan.csv"
        candidates_csv = tmp_path / "candidates.csv"
        _write_csv(
            split_csv,
            SPLIT_FIELDS,
            [
                {"source_path": "data/good.exe", "label": "0", "sample_index": "0", "split": "train"},
                {"source_path": "data/bad.exe", "label": "1", "sample_index": "1", "split": "val"},
                {"source_path": "data/test.exe", "label": "0", "sample_index": "2", "split": "test"},
            ],
        )
        _write_csv(
            plan_csv,
            PLAN_FIELDS,
            [
                {
                    "source_path": "data/good.exe",
                    "source_sha256": "",
                    "sample_index": "0",
                    "split": "train",
                    "original_label": "0",
                    "planned_label": "1",
                    "plan_action": "relabel",
                    "replacement_required": "false",
                    "replacement_label": "",
                    "usable_for_training_policy": "true",
                },
                {
                    "source_path": "data/bad.exe",
                    "source_sha256": "",
                    "sample_index": "1",
                    "split": "val",
                    "original_label": "1",
                    "planned_label": "1",
                    "plan_action": "exclude_and_replace",
                    "replacement_required": "true",
                    "replacement_label": "1",
                    "usable_for_training_policy": "false",
                },
            ],
        )
        _write_csv(
            candidates_csv,
            CANDIDATE_FIELDS,
            [
                {"source_path": "data/unused-mal.exe", "label": "1", "source_sha256": ""},
                {"source_path": "data/already-used.exe", "label": "0", "source_sha256": ""},
            ],
        )

        rows, summary = build_corrected_split(
            split_csv=split_csv,
            plan_csv=plan_csv,
            candidate_csv=candidates_csv,
            seed=7,
        )

    assert len(rows) == 3
    by_path = {row["source_path"]: row for row in rows}
    assert by_path["data/good.exe"]["label"] == "1"
    assert "data/bad.exe" not in by_path
    assert by_path["data/unused-mal.exe"]["split"] == "val"
    assert by_path["data/unused-mal.exe"]["label"] == "1"
    assert summary["excluded_rows"] == 1
    assert summary["relabeled_rows"] == 1
    assert summary["replacement_summary"]["selected_replacements"] == 1
    assert summary["corrected_summary"]["rows"] == summary["original_summary"]["rows"]


def test_replacement_shortfall_raises_instead_of_emitting_short_split():
    with _case_dir("corrected_split_shortfall") as tmp_path:
        split_csv = tmp_path / "split.csv"
        plan_csv = tmp_path / "plan.csv"
        candidates_csv = tmp_path / "candidates.csv"
        _write_csv(
            split_csv,
            SPLIT_FIELDS,
            [{"source_path": "data/bad.exe", "label": "0", "sample_index": "0", "split": "train"}],
        )
        _write_csv(
            plan_csv,
            PLAN_FIELDS,
            [{
                "source_path": "data/bad.exe",
                "source_sha256": "",
                "sample_index": "0",
                "split": "train",
                "original_label": "0",
                "planned_label": "0",
                "plan_action": "exclude_and_replace",
                "replacement_required": "true",
                "replacement_label": "0",
                "usable_for_training_policy": "false",
            }],
        )
        _write_csv(candidates_csv, CANDIDATE_FIELDS, [])

        with pytest.raises(ValueError, match="Not enough unused same-label replacement candidates"):
            build_corrected_split(
                split_csv=split_csv,
                plan_csv=plan_csv,
                candidate_csv=candidates_csv,
            )


def test_excluded_sample_cannot_be_selected_as_its_own_replacement_from_candidate_csv():
    with _case_dir("corrected_split_self_replacement") as tmp_path:
        split_csv = tmp_path / "split.csv"
        plan_csv = tmp_path / "plan.csv"
        candidates_csv = tmp_path / "candidates.csv"
        _write_csv(
            split_csv,
            SPLIT_FIELDS,
            [{"source_path": "data/bad.exe", "label": "1", "sample_index": "0", "split": "val"}],
        )
        _write_csv(
            plan_csv,
            PLAN_FIELDS,
            [{
                "source_path": "data/bad.exe",
                "source_sha256": "",
                "sample_index": "0",
                "split": "val",
                "original_label": "1",
                "planned_label": "1",
                "plan_action": "exclude_and_replace",
                "replacement_required": "true",
                "replacement_label": "1",
                "usable_for_training_policy": "false",
            }],
        )
        _write_csv(
            candidates_csv,
            CANDIDATE_FIELDS,
            [{"source_path": "data/bad.exe", "label": "1", "source_sha256": ""}],
        )

        with pytest.raises(ValueError, match="Not enough unused same-label replacement candidates"):
            build_corrected_split(
                split_csv=split_csv,
                plan_csv=plan_csv,
                candidate_csv=candidates_csv,
            )


def test_exact_sample_index_plan_does_not_exclude_duplicate_sha_canonical_row():
    sample_sha = "a" * 64
    with _case_dir("corrected_split_duplicate_sha_exact") as tmp_path:
        split_csv = tmp_path / "split.csv"
        plan_csv = tmp_path / "plan.csv"
        candidates_csv = tmp_path / "candidates.csv"
        duplicate_to_replace = f"data/date/{sample_sha}.exe"
        canonical_to_keep = f"data/family/{sample_sha}.exe"
        _write_csv(
            split_csv,
            SPLIT_FIELDS,
            [
                {"source_path": duplicate_to_replace, "label": "1", "sample_index": "0", "split": "train"},
                {"source_path": canonical_to_keep, "label": "1", "sample_index": "1", "split": "test"},
            ],
        )
        _write_csv(
            plan_csv,
            PLAN_FIELDS,
            [{
                "source_path": duplicate_to_replace,
                "source_sha256": "",
                "sample_index": "0",
                "split": "train",
                "original_label": "1",
                "planned_label": "1",
                "plan_action": "exclude_and_replace",
                "replacement_required": "true",
                "replacement_label": "1",
                "usable_for_training_policy": "false",
            }],
        )
        _write_csv(
            candidates_csv,
            CANDIDATE_FIELDS,
            [{"source_path": "data/fresh-mal.exe", "label": "1", "source_sha256": "b" * 64}],
        )

        rows, summary = build_corrected_split(
            split_csv=split_csv,
            plan_csv=plan_csv,
            candidate_csv=candidates_csv,
        )

    by_path = {row["source_path"]: row for row in rows}
    assert duplicate_to_replace not in by_path
    assert canonical_to_keep in by_path
    assert "data/fresh-mal.exe" in by_path
    assert summary["excluded_rows"] == 1
    assert summary["replacement_summary"]["selected_replacements"] == 1
