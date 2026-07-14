import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_prediction_thresholds import compare_predictions, parse_prediction_arg  # noqa: E402


def _write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_path", "label", "split", "prob_malicious", "prediction"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_compare_predictions_ranks_by_best_threshold_f1(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"

    _write_predictions(
        baseline,
        [
            {"source_path": "a", "label": 1, "split": "val", "prob_malicious": 0.80, "prediction": 1},
            {"source_path": "b", "label": 1, "split": "val", "prob_malicious": 0.40, "prediction": 0},
            {"source_path": "c", "label": 0, "split": "val", "prob_malicious": 0.30, "prediction": 0},
            {"source_path": "d", "label": 0, "split": "val", "prob_malicious": 0.20, "prediction": 0},
        ],
    )
    _write_predictions(
        candidate,
        [
            {"source_path": "a", "label": 1, "split": "val", "prob_malicious": 0.80, "prediction": 1},
            {"source_path": "b", "label": 1, "split": "val", "prob_malicious": 0.60, "prediction": 1},
            {"source_path": "c", "label": 0, "split": "val", "prob_malicious": 0.30, "prediction": 0},
            {"source_path": "d", "label": 0, "split": "val", "prob_malicious": 0.20, "prediction": 0},
        ],
    )

    report = compare_predictions(
        [("baseline", baseline), ("candidate", candidate)],
        thresholds=[0.5],
    )

    assert report["summary"][0]["model"] == "candidate"
    assert report["summary"][0]["best_f1"] == 1.0
    assert report["summary"][1]["model"] == "baseline"


def test_parse_prediction_arg_uses_stem_when_name_is_omitted():
    name, path = parse_prediction_arg("reports/example_predictions.csv")

    assert name == "example_predictions"
    assert path == Path("reports/example_predictions.csv")
