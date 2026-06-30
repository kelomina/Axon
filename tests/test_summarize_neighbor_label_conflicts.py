from __future__ import annotations

import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_neighbor_label_conflicts import summarize  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_summarize_neighbor_conflicts_accepts_generic_probability_column():
    with _case_dir("neighbor_conflicts") as tmp_path:
        input_csv = tmp_path / "neighbors.csv"
        with input_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "support_bucket",
                    "priority",
                    "reason",
                    "error_type",
                    "source_path",
                    "source_sha256",
                    "label",
                    "prediction",
                    "prob_malicious",
                    "score_column",
                    "base_prob_malicious",
                    "neighbor_label_counts",
                    "opposite_label_ratio",
                    "nearest_similarity",
                    "top5_neighbor_labels",
                    "top5_neighbor_similarities",
                    "top5_neighbor_sha256",
                    "top5_neighbor_paths",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "support_bucket": "neighbors_support_model_prediction",
                    "priority": 0,
                    "reason": "severe_fp_prob_ge_0.95",
                    "error_type": "FP",
                    "source_path": "data/a.exe",
                    "source_sha256": "sha-a",
                    "label": 0,
                    "prediction": 1,
                    "prob_malicious": "0.9900000000",
                    "score_column": "blend_prob_malicious",
                    "base_prob_malicious": "0.90",
                    "neighbor_label_counts": "0:1|1:24",
                    "opposite_label_ratio": "0.960000",
                    "nearest_similarity": "0.98000000",
                    "top5_neighbor_labels": "1|1|1|1|1",
                    "top5_neighbor_similarities": "0.98|0.97|0.96|0.95|0.94",
                    "top5_neighbor_sha256": "n1 | n2 | n3 | n4 | n5",
                    "top5_neighbor_paths": "p1 | p2 | p3 | p4 | p5",
                }
            )

        summary = summarize(
            input_csv,
            tmp_path / "summary.json",
            tmp_path / "conflicts.csv",
            max_priority=1,
        )
        rows = list(csv.DictReader((tmp_path / "conflicts.csv").open("r", encoding="utf-8-sig", newline="")))

    assert summary["high_similarity_opposite_label_conflict"]["count"] == 1
    assert rows[0]["prob_malicious"] == "0.9900000000"
    assert rows[0]["stage2_prob_malicious"] == "0.9900000000"
    assert rows[0]["score_column"] == "blend_prob_malicious"
