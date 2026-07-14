from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import audit_stage2_knn_neighbors as audit_knn  # noqa: E402


def test_read_review_rows_filters_and_stops_at_max_rows(tmp_path):
    path = tmp_path / "review.csv"
    path.write_text(
        "source_sha256,priority,reason\n"
        "a,1,keep\n"
        "b,3,skip\n"
        "c,1,keep-but-stop\n"
        "d,not-an-int,should-not-be-read\n",
        encoding="utf-8",
    )

    rows, total = audit_knn._read_review_rows(path, max_priority=1, max_rows=1)

    assert total == 1
    assert len(rows) == 1
    assert rows[0]["source_sha256"] == "a"


def test_align_prediction_rows_by_key_stops_after_matches(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text(
        "source_sha256,sample_index,source_path,label,prob_malicious\n"
        "ignore,,x,0,0.1\n"
        "a,,a.exe,0,0.2\n"
        "b,,b.exe,1,0.8\n"
        "late,,late.exe,1,not-a-number\n",
        encoding="utf-8",
    )

    rows, scanned = audit_knn._align_prediction_rows_by_key(path, {"a", "b"})

    assert scanned == 3
    assert sorted(rows) == ["a", "b"]


def test_top_k_for_similarity_row_returns_sorted_neighbors():
    idx, sim = audit_knn._top_k_for_similarity_row(
        np.asarray([0.1, 0.9, 0.4, 0.7], dtype=np.float32),
        3,
    )

    assert idx.tolist() == [1, 3, 2]
    np.testing.assert_allclose(sim, [0.9, 0.7, 0.4])
