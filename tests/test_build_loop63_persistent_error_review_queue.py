from __future__ import annotations

import csv
from pathlib import Path

from scripts.build_loop63_persistent_error_review_queue import build_queue


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_loop63_prioritizes_persistent_conflict_rows_and_keeps_manual_fields_blank(tmp_path: Path):
    loop57 = tmp_path / "loop57.csv"
    loop28 = tmp_path / "loop28.csv"
    conflict = tmp_path / "conflict.csv"
    output_csv = tmp_path / "queue.csv"
    output_json = tmp_path / "queue.json"

    loop57_fields = [
        "source_path",
        "cache_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "base_prob_malicious",
        "candidate_prob_malicious",
        "gate_prob_override",
        "final_prob_malicious",
        "prediction",
        "correct",
        "fn_override",
    ]
    _write_csv(
        loop57,
        [
            {
                "source_path": "data/a.exe",
                "cache_path": "cache/a.npz",
                "source_sha256": "sha-a",
                "label": 1,
                "split": "test",
                "sample_index": "1",
                "base_prob_malicious": "0.001",
                "candidate_prob_malicious": "0.2",
                "gate_prob_override": "0.1",
                "final_prob_malicious": "0.001",
                "prediction": 0,
                "correct": False,
                "fn_override": False,
            },
            {
                "source_path": "data/b.exe",
                "cache_path": "cache/b.npz",
                "source_sha256": "sha-b",
                "label": 0,
                "split": "test",
                "sample_index": "2",
                "base_prob_malicious": "0.2",
                "candidate_prob_malicious": "0.9",
                "gate_prob_override": "0.9",
                "final_prob_malicious": "0.9",
                "prediction": 1,
                "correct": False,
                "fn_override": True,
            },
            {
                "source_path": "data/c.exe",
                "cache_path": "cache/c.npz",
                "source_sha256": "sha-c",
                "label": 0,
                "split": "test",
                "sample_index": "3",
                "base_prob_malicious": "0.1",
                "candidate_prob_malicious": "0.1",
                "gate_prob_override": "0.1",
                "final_prob_malicious": "0.1",
                "prediction": 0,
                "correct": True,
                "fn_override": False,
            },
        ],
        loop57_fields,
    )

    loop28_fields = [
        "source_path",
        "cache_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "stage2_prob_malicious",
        "prediction",
        "correct",
    ]
    _write_csv(
        loop28,
        [
            {
                "source_path": "data/a.exe",
                "cache_path": "cache/a.npz",
                "source_sha256": "sha-a",
                "label": 1,
                "split": "test",
                "sample_index": "1",
                "stage2_prob_malicious": "0.001",
                "prediction": 0,
                "correct": False,
            },
            {
                "source_path": "data/b.exe",
                "cache_path": "cache/b.npz",
                "source_sha256": "sha-b",
                "label": 0,
                "split": "test",
                "sample_index": "2",
                "stage2_prob_malicious": "0.2",
                "prediction": 0,
                "correct": True,
            },
        ],
        loop28_fields,
    )

    conflict_fields = [
        "review_priority_rank",
        "review_lane",
        "conflict_bucket",
        "source_path",
        "source_sha256",
        "sample_index",
        "corrected_by_any_compared_model",
    ]
    _write_csv(
        conflict,
        [
            {
                "review_priority_rank": "9",
                "review_lane": "A_unfixed_severe_conflict",
                "conflict_bucket": "severe_fn_conflict_prob_le_0.01",
                "source_path": "data/a.exe",
                "source_sha256": "sha-a",
                "sample_index": "1",
                "corrected_by_any_compared_model": "False",
            }
        ],
        conflict_fields,
    )

    summary = build_queue(
        loop57_predictions=loop57,
        loop28_predictions=loop28,
        loop39_conflict_queue=conflict,
        output_csv=output_csv,
        output_json=output_json,
        max_examples=5,
    )
    rows = list(csv.DictReader(output_csv.open("r", encoding="utf-8-sig", newline="")))

    assert summary["loop57_error_rows"] == 2
    assert summary["loop39_intersection_rows"] == 1
    assert rows[0]["source_sha256"] == "sha-a"
    assert rows[0]["review_lane"] == "A_persistent_error_in_high_conflict_queue"
    assert rows[0]["manual_label_verdict"] == ""
    assert rows[0]["recommended_action"] == ""
    assert "fresh valid candidate" in rows[0]["replacement_rule"]
    assert rows[1]["review_lane"] == "D_loop57_new_error"
