from __future__ import annotations

import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_stage2_error_review_queue import build_queue  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_error_review_queue_accepts_custom_score_column():
    with _case_dir("blend_error_queue") as tmp_path:
        predictions = tmp_path / "predictions.csv"
        with predictions.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "source_path",
                    "source_sha256",
                    "label",
                    "split",
                    "sample_index",
                    "blend_prob_malicious",
                    "prediction",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "source_path": "data/a.exe",
                    "source_sha256": "sha-a",
                    "label": 0,
                    "split": "test",
                    "sample_index": 1,
                    "blend_prob_malicious": 0.99,
                    "prediction": 1,
                }
            )

        summary = build_queue(
            predictions,
            tmp_path / "queue.csv",
            tmp_path / "queue.json",
            max_examples=5,
            score_column="blend_prob_malicious",
        )
        queue_rows = list(csv.DictReader((tmp_path / "queue.csv").open("r", encoding="utf-8-sig", newline="")))

    assert summary["score_column"] == "blend_prob_malicious"
    assert summary["errors_total"] == 1
    assert summary["reason_counts"] == {"severe_fp_prob_ge_0.95": 1}
    assert queue_rows[0]["prob_malicious"] == "0.9900000000"
    assert queue_rows[0]["score_column"] == "blend_prob_malicious"
