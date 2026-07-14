import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_loop83_calibrator_rescue_profile import build_summary  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_overlap_csv(path: Path, rows: list[dict]) -> Path:
    fieldnames = [
        "join_key",
        "join_key_name",
        "sample_index",
        "label",
        "split",
        "loop57_score",
        "loop57_prediction",
        "loop57_correct",
        "calibrator_score",
        "calibrator_prediction",
        "calibrator_correct",
        "overlap_group",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_loop83_profile_counts_groups_and_rule_capture():
    with _case_dir("loop83_profile") as tmp_path:
        overlap = _write_overlap_csv(
            tmp_path / "overlap.csv",
            [
                {
                    "join_key": "a",
                    "join_key_name": "source_sha256",
                    "sample_index": "1",
                    "label": "1",
                    "split": "val",
                    "loop57_score": "0.2",
                    "loop57_prediction": "0",
                    "loop57_correct": "False",
                    "calibrator_score": "0.9",
                    "calibrator_prediction": "1",
                    "calibrator_correct": "True",
                    "overlap_group": "calibrator_only_correct",
                },
                {
                    "join_key": "b",
                    "join_key_name": "source_sha256",
                    "sample_index": "2",
                    "label": "0",
                    "split": "val",
                    "loop57_score": "0.2",
                    "loop57_prediction": "0",
                    "loop57_correct": "True",
                    "calibrator_score": "0.9",
                    "calibrator_prediction": "1",
                    "calibrator_correct": "False",
                    "overlap_group": "loop57_only_correct",
                },
                {
                    "join_key": "c",
                    "join_key_name": "source_sha256",
                    "sample_index": "3",
                    "label": "0",
                    "split": "val",
                    "loop57_score": "0.1",
                    "loop57_prediction": "0",
                    "loop57_correct": "True",
                    "calibrator_score": "0.2",
                    "calibrator_prediction": "0",
                    "calibrator_correct": "True",
                    "overlap_group": "both_correct",
                },
            ],
        )

        report = build_summary(overlap_csv=overlap, thresholds=[0.5])

    assert report["rows"] == 3
    assert report["group_counts"]["calibrator_only_correct"] == 1
    assert report["group_counts"]["loop57_only_correct"] == 1
    assert report["rule_scan"]["best"]["calibrator_only_correct_captured"] == 1
    assert report["rule_scan"]["best"]["loop57_only_correct_harmed"] == 1
    assert report["identity_feature_policy"]["rule_features_used"] == ["abs_score_delta"]


def test_loop83_profile_reports_incomplete_overlap_blocker():
    with _case_dir("loop83_profile_blocker") as tmp_path:
        overlap = _write_overlap_csv(
            tmp_path / "overlap.csv",
            [
                {
                    "join_key": "a",
                    "join_key_name": "source_sha256",
                    "sample_index": "1",
                    "label": "1",
                    "split": "val",
                    "loop57_score": "0.9",
                    "loop57_prediction": "1",
                    "loop57_correct": "True",
                    "calibrator_score": "0.8",
                    "calibrator_prediction": "1",
                    "calibrator_correct": "True",
                    "overlap_group": "both_correct",
                }
            ],
        )

        report = build_summary(overlap_csv=overlap, thresholds=[0.1])

    assert report["blockers"] == ["Expected complete 20000-row Val overlap"]
