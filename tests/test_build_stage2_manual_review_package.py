from __future__ import annotations

import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_stage2_manual_review_package import build_package  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_manual_review_package_accepts_generic_probability_column():
    with _case_dir("manual_review_package") as tmp_path:
        neighbor_csv = tmp_path / "neighbors.csv"
        with neighbor_csv.open("w", encoding="utf-8-sig", newline="") as handle:
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
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "support_bucket": "neighbors_support_model_prediction",
                    "priority": 0,
                    "reason": "severe_fp_prob_ge_0.95",
                    "error_type": "FP",
                    "source_path": "E:/Project/python/Axon_v2.6Exp/data/benign/a.exe",
                    "source_sha256": "sha-a",
                    "label": 0,
                    "prediction": 1,
                    "prob_malicious": "0.9900000000",
                    "score_column": "blend_prob_malicious",
                    "base_prob_malicious": "0.90",
                    "top_k": 25,
                    "neighbor_label_counts": "0:1|1:24",
                    "same_label_count": 1,
                    "opposite_label_count": 24,
                    "opposite_label_ratio": "0.960000",
                    "nearest_similarity": "0.98000000",
                    "top5_neighbor_labels": "1|1|1|1|1",
                    "top5_neighbor_similarities": "0.98|0.97|0.96|0.95|0.94",
                    "top5_neighbor_sha256": "n1 | n2 | n3 | n4 | n5",
                    "top5_neighbor_paths": "p1 | p2 | p3 | p4 | p5",
                }
            )

        summary = build_package(
            neighbor_csv,
            tmp_path / "package.csv",
            tmp_path / "package.json",
            fp_count=1,
            fn_count=0,
            max_priority=1,
            support_bucket="neighbors_support_model_prediction",
        )
        rows = list(csv.DictReader((tmp_path / "package.csv").open("r", encoding="utf-8-sig", newline="")))

    assert summary["selected_rows"] == 1
    assert rows[0]["prob_malicious"] == "0.9900000000"
    assert rows[0]["stage2_prob_malicious"] == "0.9900000000"
    assert rows[0]["score_column"] == "blend_prob_malicious"
