import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_combined_manual_review_queue import combine_review_queues  # noqa: E402


FIELDNAMES = [
    "review_rank",
    "support_bucket",
    "priority",
    "reason",
    "error_type",
    "path_hint",
    "source_path",
    "source_sha256",
    "label",
    "prediction",
    "prob_malicious",
    "score_column",
    "stage2_prob_malicious",
    "base_prob_malicious",
    "top_k",
    "neighbor_label_counts",
    "same_label_count",
    "opposite_label_count",
    "opposite_label_ratio",
    "nearest_similarity",
    "top5_neighbor_labels",
    "top5_neighbor_similarities",
    "top5_neighbor_sha256",
    "top5_neighbor_paths",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _base_row(sha: str, **overrides: str) -> dict[str, str]:
    row = {
        "review_rank": "1",
        "support_bucket": "neighbors_support_model_prediction",
        "priority": "0",
        "reason": "severe_fp_prob_ge_0.95",
        "error_type": "FP",
        "path_hint": "data/a.exe",
        "source_path": f"data/{sha}.exe",
        "source_sha256": sha,
        "label": "0",
        "prediction": "1",
        "prob_malicious": "0.99",
        "score_column": "blend_prob_malicious",
        "stage2_prob_malicious": "0.99",
        "base_prob_malicious": "0.5",
        "top_k": "25",
        "neighbor_label_counts": "1:25",
        "same_label_count": "0",
        "opposite_label_count": "25",
        "opposite_label_ratio": "1.0",
        "nearest_similarity": "0.95",
        "top5_neighbor_labels": "1|1|1|1|1",
        "top5_neighbor_similarities": "0.9|0.8|0.7|0.6|0.5",
        "top5_neighbor_sha256": "a | b | c | d | e",
        "top5_neighbor_paths": "a | b | c | d | e",
        "manual_label_verdict": "",
        "manual_verdict_note": "",
        "recommended_action": "",
    }
    row.update(overrides)
    return row


def test_combines_and_deduplicates_blank_manual_rows(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_rows(first, [_base_row("a" * 64), _base_row("b" * 64, priority="1")])
    _write_rows(second, [_base_row("a" * 64, support_bucket="neighbors_mixed"), _base_row("c" * 64)])

    out_csv = tmp_path / "combined.csv"
    out_json = tmp_path / "combined.json"
    summary = combine_review_queues(
        inputs=[("model", first), ("mixed", second)],
        output_csv=out_csv,
        output_json=out_json,
    )

    assert summary["input_rows_total"] == 4
    assert summary["output_rows"] == 3
    assert summary["deduplicated_rows"] == 1
    assert summary["manual_fields_blank_output"] is True
    assert summary["review_source_count_counts"] == {"1": 2, "2": 1}

    with out_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["combined_rank"] for row in rows] == ["1", "2", "3"]
    merged = [row for row in rows if row["source_sha256"] == "a" * 64][0]
    assert merged["review_sources"] == "model|mixed"
    assert merged["review_source_count"] == "2"
    assert merged["dedup_method"] == "source_sha256"
    assert merged["dedup_key"] == "a" * 64
    assert merged["manual_label_verdict"] == ""
    assert merged["recommended_action"] == ""


def test_deduplicates_same_sha_with_different_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_rows(first, [_base_row("a" * 64, source_path="data/one.exe")])
    _write_rows(second, [_base_row("a" * 64, source_path="data/two.exe")])

    summary = combine_review_queues(
        inputs=[("first", first), ("second", second)],
        output_csv=tmp_path / "combined.csv",
        output_json=tmp_path / "combined.json",
    )

    assert summary["input_rows_total"] == 2
    assert summary["output_rows"] == 1
    assert summary["deduplicated_rows"] == 1


def test_rejects_filled_manual_fields_by_default(tmp_path: Path) -> None:
    review_csv = tmp_path / "review.csv"
    _write_rows(review_csv, [_base_row("a" * 64, manual_label_verdict="label_wrong")])

    with pytest.raises(ValueError, match="filled manual fields"):
        combine_review_queues(
            inputs=[("review", review_csv)],
            output_csv=tmp_path / "combined.csv",
            output_json=tmp_path / "combined.json",
        )
