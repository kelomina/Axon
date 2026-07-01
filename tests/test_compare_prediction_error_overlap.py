from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_prediction_error_overlap import compare_predictions  # noqa: E402


def _write_predictions(path: Path, scores: list[float], labels: list[int] | None = None) -> None:
    labels = labels or [0, 1, 1]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "source_sha256", "label", "split", "sample_index", "score"],
        )
        writer.writeheader()
        for index, (label, score) in enumerate(zip(labels, scores)):
            writer.writerow(
                {
                    "source_path": f"data/{index}.exe",
                    "source_sha256": f"sha-{index}",
                    "label": label,
                    "split": "val",
                    "sample_index": str(index),
                    "score": score,
                }
            )


def test_compare_prediction_error_overlap_reports_fixed_and_new_errors(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_predictions(baseline, [0.9, 0.2, 0.9])
    _write_predictions(candidate, [0.1, 0.8, 0.2])

    report, details = compare_predictions(
        [
            ("baseline", baseline, "score", 0.5),
            ("candidate", candidate, "score", 0.5),
        ],
        key_columns=["source_sha256", "source_path"],
    )

    assert report["single"]["baseline"]["errors"] == 2
    assert report["single"]["candidate"]["errors"] == 1
    assert report["versus_baseline"][0]["fixed_baseline_errors"] == 2
    assert report["versus_baseline"][0]["new_candidate_errors"] == 1
    assert report["rows"]["any_error_on_common"] == 3
    assert {row["error_pattern"] for row in details} == {"baseline", "candidate"}


def test_compare_prediction_error_overlap_uses_identity_key_not_sample_index(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_predictions(baseline, [0.9], labels=[0])
    _write_predictions(candidate, [0.9], labels=[0])

    # Same sample index but different identity; this should not be treated as
    # a common row when source identity keys are requested.
    rows = list(csv.DictReader(candidate.open("r", encoding="utf-8-sig")))
    rows[0]["source_sha256"] = "new-sha"
    rows[0]["source_path"] = "data/new.exe"
    with candidate.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report, details = compare_predictions(
        [
            ("baseline", baseline, "score", 0.5),
            ("candidate", candidate, "score", 0.5),
        ],
        key_columns=["source_sha256", "source_path"],
    )

    assert report["rows"]["common"] == 0
    assert report["pattern_counts"]["missing:baseline"] == 1
    assert report["pattern_counts"]["missing:candidate"] == 1
    assert details == []
