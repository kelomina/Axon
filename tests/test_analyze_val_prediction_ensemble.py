from __future__ import annotations

import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_val_prediction_ensemble import analyze  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_predictions(path: Path, scores: list[float]) -> None:
    labels = [0, 0, 1, 1]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_sha256", "label", "sample_index", "score"])
        writer.writeheader()
        for index, (label, score) in enumerate(zip(labels, scores)):
            writer.writerow({"source_sha256": f"sha-{index}", "label": label, "sample_index": index, "score": score})


def test_analyze_val_prediction_ensemble_scores_average_candidates():
    with _case_dir("val_ensemble") as tmp_path:
        first = tmp_path / "first.csv"
        second = tmp_path / "second.csv"
        _write_predictions(first, [0.1, 0.6, 0.8, 0.7])
        _write_predictions(second, [0.2, 0.3, 0.4, 0.9])

        result = analyze(
            [
                ("first", first, "score"),
                ("second", second, "score"),
            ],
            [0.5],
            [(["first", "second"], [1.0, 2.0])],
        )

    assert result["rows"] == 4
    assert {row["name"] for row in result["single_models"]} == {"first", "second"}
    assert result["average_ensembles"][0]["names"] == ["first", "second"]
    assert result["weighted_ensembles"][0]["weights"] == pytest.approx([1 / 3, 2 / 3])
    assert result["pairwise_error_overlap"][0]["left_errors"] == 1


def test_analyze_val_prediction_ensemble_preserves_duplicate_sha_by_sample_index():
    with _case_dir("val_ensemble_duplicate_sha") as tmp_path:
        first = tmp_path / "first.csv"
        second = tmp_path / "second.csv"
        for path in [first, second]:
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_sha256", "label", "sample_index", "score"])
                writer.writeheader()
                writer.writerow({"source_sha256": "same", "label": 0, "sample_index": 1, "score": 0.1})
                writer.writerow({"source_sha256": "same", "label": 1, "sample_index": 2, "score": 0.9})

        result = analyze(
            [
                ("first", first, "score"),
                ("second", second, "score"),
            ],
            [0.5],
        )

    assert result["rows"] == 2
    assert result["key_column"] == "sample_index"
