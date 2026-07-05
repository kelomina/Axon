import csv
import json

import pytest

from scripts.evaluate_prediction_override_union import evaluate_override_union, main


FIELDNAMES = [
    "source_path",
    "cache_path",
    "source_sha256",
    "label",
    "split",
    "sample_index",
    "stage2_prob_malicious",
    "prediction",
]


def write_prediction_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def row(index, label, prediction, score=None, split="val"):
    return {
        "source_path": f"data/sample_{index}.bin",
        "cache_path": f"data/.cache/sample_{index}.npz",
        "source_sha256": f"{index:064x}",
        "label": str(label),
        "split": split,
        "sample_index": str(index),
        "stage2_prob_malicious": str(score if score is not None else prediction),
        "prediction": str(prediction),
    }


def test_override_union_applies_frozen_non_conflicting_changes(tmp_path):
    baseline = tmp_path / "baseline.csv"
    override_a = tmp_path / "override_a.csv"
    override_b = tmp_path / "override_b.csv"
    write_prediction_csv(
        baseline,
        [
            row(1, 0, 0, 0.1),
            row(2, 1, 0, 0.2),
            row(3, 0, 1, 0.8),
        ],
    )
    write_prediction_csv(
        override_a,
        [
            row(1, 0, 0, 0.1),
            row(2, 1, 1, 0.9),
            row(3, 0, 1, 0.8),
        ],
    )
    write_prediction_csv(
        override_b,
        [
            row(1, 0, 0, 0.1),
            row(2, 1, 0, 0.2),
            row(3, 0, 0, 0.3),
        ],
    )

    rows, report = evaluate_override_union(
        baseline_csv=baseline,
        overrides=[("a", override_a), ("b", override_b)],
        key_columns=("sample_index", "source_sha256"),
    )

    assert [item["prediction"] for item in rows] == [0, 1, 0]
    assert report["baseline_metrics"]["errors"] == 2
    assert report["metrics"]["errors"] == 0
    assert report["accepted_override_counts"] == {"a": 1, "b": 1}


def test_override_union_rejects_conflicting_override_predictions(tmp_path):
    baseline = tmp_path / "baseline.csv"
    override_a = tmp_path / "override_a.csv"
    override_b = tmp_path / "override_b.csv"
    write_prediction_csv(baseline, [row(1, 0, 0, 0.1)])
    write_prediction_csv(override_a, [row(1, 0, 1, 0.9)])
    write_prediction_csv(override_b, [row(1, 0, 0, 0.1)])

    # This is not a conflict because override_b agrees with the baseline.
    rows, report = evaluate_override_union(
        baseline_csv=baseline,
        overrides=[("a", override_a), ("b", override_b)],
        key_columns=("sample_index", "source_sha256"),
    )
    assert rows[0]["accepted_override"] == "a"
    assert report["changed_rows"] == 1

    write_prediction_csv(override_b, [row(1, 0, 2, 0.5)])
    with pytest.raises(ValueError, match="Conflicting override"):
        evaluate_override_union(
            baseline_csv=baseline,
            overrides=[("a", override_a), ("b", override_b)],
            key_columns=("sample_index", "source_sha256"),
        )


def test_main_writes_outputs(tmp_path):
    baseline = tmp_path / "baseline.csv"
    override = tmp_path / "override.csv"
    output_json = tmp_path / "report.json"
    output_csv = tmp_path / "predictions.csv"
    write_prediction_csv(baseline, [row(1, 1, 0, 0.2)])
    write_prediction_csv(override, [row(1, 1, 1, 0.9)])

    assert main(
        [
            "--baseline-csv",
            str(baseline),
            "--override",
            f"recovery={override}",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ]
    ) == 0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["metrics"]["errors"] == 0
    assert "source_sha256" in output_csv.read_text(encoding="utf-8")
