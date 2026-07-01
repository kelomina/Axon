import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_source_aware_adjudication_queue import build_queue  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "review_rank",
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
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _row(source_path: str, error_type: str, prob: str, nearest: str, opposite: str, priority: str = "0") -> dict:
    label = "0" if error_type == "FP" else "1"
    prediction = "1" if error_type == "FP" else "0"
    return {
        "review_rank": "",
        "priority": priority,
        "reason": "test",
        "error_type": error_type,
        "path_hint": "",
        "source_path": source_path,
        "source_sha256": Path(source_path).stem[:64].ljust(64, "0"),
        "label": label,
        "prediction": prediction,
        "prob_malicious": prob,
        "score_column": "blend_prob_malicious",
        "stage2_prob_malicious": prob,
        "base_prob_malicious": prob,
        "top_k": "25",
        "neighbor_label_counts": "",
        "same_label_count": "",
        "opposite_label_count": "",
        "opposite_label_ratio": opposite,
        "nearest_similarity": nearest,
        "top5_neighbor_labels": "",
        "top5_neighbor_similarities": "",
        "top5_neighbor_sha256": "",
        "top5_neighbor_paths": "",
        "manual_label_verdict": "should_be_cleared",
        "manual_verdict_note": "should_be_cleared",
        "recommended_action": "should_be_cleared",
    }


def test_source_aware_queue_prioritizes_whitelist_conflicts_and_clears_manual_fields():
    with _case_dir("source_queue") as tmp_path:
        review_csv = tmp_path / "review.csv"
        output_csv = tmp_path / "queue.csv"
        output_json = tmp_path / "summary.json"
        _write_rows(
            review_csv,
            [
                _row(r"E:\repo\data\待拉黑\2020-11\2020-11-07\a.exe", "FN", "0.10", "0.70", "0.90"),
                _row(r"E:\repo\data\待加入白名单\b.exe", "FP", "0.80", "0.70", "0.85", priority="2"),
                _row(r"E:\repo\data\待加入白名单\c.exe", "FP", "0.99", "0.96", "1.00"),
                _row(r"E:\repo\data\待拉黑\2020-11\2020-11-07\d.exe", "FN", "0.20", "0.65", "0.90"),
                _row(r"E:\repo\data\待加入白名单\e.exe", "FP", "0.91", "0.92", "0.80", priority="1"),
            ],
        )

        summary = build_queue(review_csv=review_csv, output_csv=output_csv, output_json=output_json)
        rows = list(csv.DictReader(output_csv.open(encoding="utf-8-sig")))
        persisted = json.loads(output_json.read_text(encoding="utf-8"))

    assert summary["rows"] == 5
    assert rows[0]["review_lane"] == "A_whitelist_critical_fp"
    assert rows[1]["review_lane"] == "B_whitelist_high_similarity_fp"
    assert rows[2]["review_lane"] == "C_whitelist_remaining_fp"
    assert rows[3]["review_lane"] == "D_malicious_batch_fn"
    assert rows[3]["source_group_size"] == "2"
    assert rows[4]["source_group_size"] == "2"
    assert all(row["manual_label_verdict"] == "" for row in rows)
    assert all(row["recommended_action"] == "" for row in rows)
    assert persisted["lane_counts"] == {
        "A_whitelist_critical_fp": 1,
        "B_whitelist_high_similarity_fp": 1,
        "C_whitelist_remaining_fp": 1,
        "D_malicious_batch_fn": 2,
    }
