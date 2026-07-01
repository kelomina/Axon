from __future__ import annotations

import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop39_replacement_preflight import build_preflight  # noqa: E402


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
REVIEW_FIELDS = [
    "review_priority_rank",
    "source_path",
    "source_sha256",
    "sample_index",
    "label",
    "manual_label_verdict",
    "recommended_action",
]
CANDIDATE_FIELDS = ["source_path", "label", "source_sha256"]


def test_preflight_blocks_blank_loop39_queue_without_mutation():
    with _case_dir("loop39_preflight_blank") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        _write_csv(
            split_csv,
            SPLIT_FIELDS,
            [{"source_path": "data/a.exe", "label": "0", "sample_index": "0", "split": "train"}],
        )
        _write_csv(
            review_csv,
            REVIEW_FIELDS,
            [
                {
                    "review_priority_rank": "1",
                    "source_path": "data/a.exe",
                    "source_sha256": "",
                    "sample_index": "0",
                    "label": "0",
                    "manual_label_verdict": "",
                    "recommended_action": "",
                }
            ],
        )

        payload = build_preflight(
            review_csv=review_csv,
            split_csv=split_csv,
            enforce_shape=False,
            enforce_label_balance=False,
        )

    assert payload["preflight_ok"] is False
    assert payload["preflight_status"] == "blocked_no_verdicts"
    assert payload["replacement_required"] == 0
    assert payload["blank_manual_rows"] == 1


def test_preflight_requires_candidate_pool_for_replacement_verdicts():
    with _case_dir("loop39_preflight_candidate_required") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        _write_csv(
            split_csv,
            SPLIT_FIELDS,
            [{"source_path": "data/bad.exe", "label": "1", "sample_index": "7", "split": "val"}],
        )
        _write_csv(
            review_csv,
            REVIEW_FIELDS,
            [
                {
                    "review_priority_rank": "1",
                    "source_path": "data/bad.exe",
                    "source_sha256": "",
                    "sample_index": "7",
                    "label": "1",
                    "manual_label_verdict": "feature_broken",
                    "recommended_action": "replace_with_fresh_same_label_candidate",
                }
            ],
        )

        payload = build_preflight(
            review_csv=review_csv,
            split_csv=split_csv,
            enforce_shape=False,
            enforce_label_balance=False,
        )

    assert payload["preflight_ok"] is False
    assert payload["preflight_status"] == "candidate_pool_required"
    assert payload["replacement_required"] == 1
    assert payload["replacement_counts_by_label"] == {"1": 1}


def test_preflight_rejects_self_fill_and_wrong_label_candidate_shortfall():
    with _case_dir("loop39_preflight_candidate_shortfall") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        candidate_csv = tmp_path / "candidates.csv"
        _write_csv(
            split_csv,
            SPLIT_FIELDS,
            [{"source_path": "data/bad.exe", "label": "1", "sample_index": "7", "split": "val"}],
        )
        _write_csv(
            review_csv,
            REVIEW_FIELDS,
            [
                {
                    "review_priority_rank": "1",
                    "source_path": "data/bad.exe",
                    "source_sha256": "",
                    "sample_index": "7",
                    "label": "1",
                    "manual_label_verdict": "out_of_scope",
                    "recommended_action": "replace_with_fresh_same_label_candidate",
                }
            ],
        )
        _write_csv(
            candidate_csv,
            CANDIDATE_FIELDS,
            [
                {"source_path": "data/bad.exe", "label": "1", "source_sha256": ""},
                {"source_path": "data/fresh-benign.exe", "label": "0", "source_sha256": ""},
            ],
        )

        payload = build_preflight(
            review_csv=review_csv,
            split_csv=split_csv,
            candidate_csv=candidate_csv,
            enforce_shape=False,
            enforce_label_balance=False,
        )

    assert payload["preflight_ok"] is False
    assert payload["preflight_status"] == "candidate_pool_shortfall"
    assert payload["candidate_summary"]["self_replacement_rows"] == 1
    assert payload["candidate_summary"]["valid_fresh_label_counts"] == {"0": 1}
    assert payload["candidate_summary"]["replacement_shortfall"] == {"1": 1}


def test_preflight_passes_with_fresh_same_label_candidate_and_complete_verdicts():
    with _case_dir("loop39_preflight_ready") as tmp_path:
        split_csv = tmp_path / "split.csv"
        review_csv = tmp_path / "review.csv"
        candidate_csv = tmp_path / "candidates.csv"
        _write_csv(
            split_csv,
            SPLIT_FIELDS,
            [{"source_path": "data/bad.exe", "label": "0", "sample_index": "3", "split": "train"}],
        )
        _write_csv(
            review_csv,
            REVIEW_FIELDS,
            [
                {
                    "review_priority_rank": "1",
                    "source_path": "data/bad.exe",
                    "source_sha256": "",
                    "sample_index": "3",
                    "label": "0",
                    "manual_label_verdict": "label_wrong",
                    "recommended_action": "replace_with_fresh_same_label_candidate",
                }
            ],
        )
        _write_csv(
            candidate_csv,
            CANDIDATE_FIELDS,
            [{"source_path": "data/fresh-benign.exe", "label": "0", "source_sha256": ""}],
        )

        payload = build_preflight(
            review_csv=review_csv,
            split_csv=split_csv,
            candidate_csv=candidate_csv,
            enforce_shape=False,
            enforce_label_balance=False,
        )

    assert payload["preflight_ok"] is True
    assert payload["preflight_status"] == "ready_for_corrected_split"
    assert payload["candidate_summary"]["enough_fresh_same_label_candidates"] is True
