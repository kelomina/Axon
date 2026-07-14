from __future__ import annotations

import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_prediction_blend import evaluate_blend  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_predictions(path: Path, scores: list[float]) -> None:
    labels = [0, 1]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "source_sha256", "label", "split", "sample_index", "score"])
        writer.writeheader()
        for index, (label, score) in enumerate(zip(labels, scores)):
            writer.writerow(
                {
                    "source_path": f"data/{index}.exe",
                    "source_sha256": f"sha-{index}",
                    "label": label,
                    "split": "val",
                    "sample_index": index,
                    "score": score,
                }
            )


def test_evaluate_prediction_blend_applies_frozen_weights_and_threshold():
    with _case_dir("prediction_blend") as tmp_path:
        first = tmp_path / "first.csv"
        second = tmp_path / "second.csv"
        _write_predictions(first, [0.2, 0.4])
        _write_predictions(second, [0.2, 0.9])

        rows, report = evaluate_blend(
            [
                ("first", first, "score", 1.0),
                ("second", second, "score", 2.0),
            ],
            threshold=0.5,
        )

    assert report["rows"] == 2
    assert report["normalized_weights"]["first"] == pytest.approx(1 / 3)
    assert report["normalized_weights"]["second"] == pytest.approx(2 / 3)
    assert report["metrics"]["errors"] == 0
    assert rows[1]["prediction"] == 1


def test_evaluate_prediction_blend_rejects_misaligned_rows():
    with _case_dir("prediction_blend_alignment") as tmp_path:
        first = tmp_path / "first.csv"
        second = tmp_path / "second.csv"
        _write_predictions(first, [0.2, 0.9])
        _write_predictions(second, [0.2, 0.9])
        rows = list(csv.DictReader(second.open("r", encoding="utf-8-sig", newline="")))
        rows[0]["source_sha256"] = "different-sha"
        with second.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["source_path", "source_sha256", "label", "split", "sample_index", "score"],
            )
            writer.writeheader()
            writer.writerows(rows)

        with pytest.raises(ValueError, match="not aligned"):
            evaluate_blend(
                [
                    ("first", first, "score", 1.0),
                    ("second", second, "score", 1.0),
                ],
                threshold=0.5,
            )
