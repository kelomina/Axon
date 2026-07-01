from __future__ import annotations

import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_automatic_noise_replacement_plan import build_automatic_plan  # noqa: E402


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


SPLIT_FIELDS = ["source_path", "source_sha256", "label", "sample_index", "split"]
REVIEW_FIELDS = [
    "source_path",
    "source_sha256",
    "label",
    "support_bucket",
    "priority",
    "prob_malicious",
    "opposite_label_ratio",
    "nearest_similarity",
]


def test_build_automatic_plan_replaces_only_high_confidence_training_rows():
    with _case_dir("auto_noise_plan_basic") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        _write_csv(
            split_csv,
            SPLIT_FIELDS,
            [
                {"source_path": "data/a.exe", "source_sha256": "a" * 64, "label": "0", "sample_index": "0", "split": "val"},
                {"source_path": "data/b.exe", "source_sha256": "b" * 64, "label": "1", "sample_index": "1", "split": "train"},
            ],
        )
        _write_csv(
            review_csv,
            REVIEW_FIELDS,
            [
                {
                    "source_path": "data/a.exe",
                    "source_sha256": "a" * 64,
                    "label": "0",
                    "support_bucket": "neighbors_support_model_prediction",
                    "priority": "0",
                    "prob_malicious": "0.99",
                    "opposite_label_ratio": "0.92",
                    "nearest_similarity": "0.50",
                },
                {
                    "source_path": "data/b.exe",
                    "source_sha256": "b" * 64,
                    "label": "1",
                    "support_bucket": "neighbors_mixed",
                    "priority": "0",
                    "prob_malicious": "0.01",
                    "opposite_label_ratio": "0.92",
                    "nearest_similarity": "0.50",
                },
            ],
        )

        rows, summary = build_automatic_plan(review_csv=review_csv, split_csv=split_csv)

    assert len(rows) == 1
    assert rows[0]["source_path"] == "data/a.exe"
    assert rows[0]["plan_action"] == "exclude_and_replace"
    assert rows[0]["replacement_required"] == "true"
    assert rows[0]["replacement_label"] == "0"
    assert rows[0]["planned_label"] == "0"
    assert rows[0]["manual_label_verdict"] == "automatic_high_confidence_noise_candidate"
    assert rows[0]["usable_for_training_policy"] == "false"
    assert summary["replacement_counts_by_original_label"] == {"0": 1}
    assert summary["skipped_counts"] == {"not_eligible": 1}
    assert summary["policy"]["relabeling_allowed"] is False


def test_build_automatic_plan_skips_test_rows_and_label_mismatches():
    with _case_dir("auto_noise_plan_skip_test") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        _write_csv(
            split_csv,
            SPLIT_FIELDS,
            [
                {"source_path": "data/test.exe", "source_sha256": "c" * 64, "label": "0", "sample_index": "2", "split": "test"},
                {"source_path": "data/mismatch.exe", "source_sha256": "d" * 64, "label": "1", "sample_index": "3", "split": "val"},
            ],
        )
        _write_csv(
            review_csv,
            REVIEW_FIELDS,
            [
                {
                    "source_path": "data/test.exe",
                    "source_sha256": "c" * 64,
                    "label": "0",
                    "support_bucket": "neighbors_support_model_prediction",
                    "priority": "0",
                    "prob_malicious": "0.99",
                    "opposite_label_ratio": "0.90",
                    "nearest_similarity": "0.50",
                },
                {
                    "source_path": "data/mismatch.exe",
                    "source_sha256": "d" * 64,
                    "label": "0",
                    "support_bucket": "neighbors_support_model_prediction",
                    "priority": "0",
                    "prob_malicious": "0.99",
                    "opposite_label_ratio": "0.90",
                    "nearest_similarity": "0.50",
                },
            ],
        )

        rows, summary = build_automatic_plan(review_csv=review_csv, split_csv=split_csv)

    assert rows == []
    assert summary["skipped_counts"] == {"held_out_test": 1, "split_label_mismatch": 1}
    assert summary["replacement_required"] == 0


def test_build_automatic_plan_honors_thresholds():
    with _case_dir("auto_noise_plan_thresholds") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        _write_csv(
            split_csv,
            SPLIT_FIELDS,
            [{"source_path": "data/a.exe", "source_sha256": "a" * 64, "label": "0", "sample_index": "0", "split": "val"}],
        )
        _write_csv(
            review_csv,
            REVIEW_FIELDS,
            [{
                "source_path": "data/a.exe",
                "source_sha256": "a" * 64,
                "label": "0",
                "support_bucket": "neighbors_support_model_prediction",
                "priority": "1",
                "prob_malicious": "0.99",
                "opposite_label_ratio": "0.92",
                "nearest_similarity": "0.50",
            }],
        )

        rows, summary = build_automatic_plan(review_csv=review_csv, split_csv=split_csv, max_priority=0)

    assert rows == []
    assert summary["skipped_counts"] == {"not_eligible": 1}
