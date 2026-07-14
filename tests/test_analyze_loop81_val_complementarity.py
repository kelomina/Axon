import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_loop81_val_complementarity import build_summary  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_val_complementarity_joins_by_sample_index_not_row_order():
    with _case_dir("loop81_overlap") as tmp_path:
        loop57 = _write_csv(
            tmp_path / "loop57.csv",
            [
                {
                    "source_path": "name-a.exe",
                    "source_sha256": "a" * 64,
                    "label": "1",
                    "split": "val",
                    "sample_index": "11",
                    "final_prob_malicious": "0.9",
                    "prediction": "1",
                },
                {
                    "source_path": "name-b.exe",
                    "source_sha256": "b" * 64,
                    "label": "1",
                    "split": "val",
                    "sample_index": "10",
                    "final_prob_malicious": "0.2",
                    "prediction": "0",
                },
            ],
            [
                "source_path",
                "source_sha256",
                "label",
                "split",
                "sample_index",
                "final_prob_malicious",
                "prediction",
            ],
        )
        calibrator = _write_csv(
            tmp_path / "cal.csv",
            [
                {
                    "source_path": "renamed-b.bin",
                    "source_sha256": "different",
                    "label": "1",
                    "split": "val",
                    "sample_index": "10",
                    "prob_malicious": "0.8",
                    "prediction": "1",
                },
                {
                    "source_path": "renamed-a.bin",
                    "source_sha256": "different",
                    "label": "1",
                    "split": "val",
                    "sample_index": "11",
                    "prob_malicious": "0.1",
                    "prediction": "0",
                },
            ],
            ["source_path", "source_sha256", "label", "split", "sample_index", "prob_malicious", "prediction"],
        )

        report = build_summary(
            loop57_predictions=loop57,
            calibrator_predictions=calibrator,
            output_overlap_csv=tmp_path / "overlap.csv",
        )
        overlap_rows = list(csv.DictReader((tmp_path / "overlap.csv").open("r", encoding="utf-8-sig")))

    assert report["join_summary"]["common_rows"] == 2
    assert report["overlap_counts"]["loop57_only_correct"] == 1
    assert report["overlap_counts"]["calibrator_only_correct"] == 1
    assert report["oracle_gain_vs_loop57_errors"] == 1
    assert report["calibrator_regression_vs_loop57_errors"] == 1
    assert {row["sample_index"] for row in overlap_rows} == {"10", "11"}
    assert {row["join_key_name"] for row in overlap_rows} == {"sample_index"}


def test_val_complementarity_can_join_by_source_sha256_when_sample_index_differs():
    with _case_dir("loop81_overlap_sha") as tmp_path:
        sha_a = "a" * 64
        sha_b = "b" * 64
        loop57 = _write_csv(
            tmp_path / "loop57.csv",
            [
                {
                    "source_path": "train-name-a.exe",
                    "source_sha256": sha_a,
                    "label": "1",
                    "split": "val",
                    "sample_index": "501",
                    "final_prob_malicious": "0.9",
                    "prediction": "1",
                },
                {
                    "source_path": "train-name-b.exe",
                    "source_sha256": sha_b,
                    "label": "1",
                    "split": "val",
                    "sample_index": "502",
                    "final_prob_malicious": "0.2",
                    "prediction": "0",
                },
            ],
            [
                "source_path",
                "source_sha256",
                "label",
                "split",
                "sample_index",
                "final_prob_malicious",
                "prediction",
            ],
        )
        calibrator = _write_csv(
            tmp_path / "cal.csv",
            [
                {
                    "source_path": "renamed-b.bin",
                    "source_sha256": sha_b,
                    "label": "1",
                    "split": "val",
                    "sample_index": "12",
                    "prob_malicious": "0.8",
                    "prediction": "1",
                },
                {
                    "source_path": "renamed-a.bin",
                    "source_sha256": sha_a,
                    "label": "1",
                    "split": "val",
                    "sample_index": "11",
                    "prob_malicious": "0.1",
                    "prediction": "0",
                },
            ],
            ["source_path", "source_sha256", "label", "split", "sample_index", "prob_malicious", "prediction"],
        )

        report = build_summary(
            loop57_predictions=loop57,
            calibrator_predictions=calibrator,
            join_key="source_sha256",
            output_overlap_csv=tmp_path / "overlap.csv",
        )
        overlap_rows = list(csv.DictReader((tmp_path / "overlap.csv").open("r", encoding="utf-8-sig")))

    assert report["join_key"] == "source_sha256"
    assert report["join_summary"]["join_key"] == "source_sha256"
    assert report["join_summary"]["common_rows"] == 2
    assert report["join_summary"]["label_mismatches"] == 0
    assert report["overlap_counts"]["loop57_only_correct"] == 1
    assert report["overlap_counts"]["calibrator_only_correct"] == 1
    assert {row["join_key_name"] for row in overlap_rows} == {"source_sha256"}
    assert {row["join_key"] for row in overlap_rows} == {sha_a, sha_b}
    assert report["identity_feature_policy"]["source_sha256"] == "alignment/cache-audit only"


def test_val_complementarity_reports_blockers_for_missing_overlap():
    with _case_dir("loop81_overlap_block") as tmp_path:
        loop57 = _write_csv(
            tmp_path / "loop57.csv",
            [{"label": "0", "split": "val", "sample_index": "1", "final_prob_malicious": "0.1", "prediction": "0"}],
            ["label", "split", "sample_index", "final_prob_malicious", "prediction"],
        )
        calibrator = _write_csv(
            tmp_path / "cal.csv",
            [{"label": "0", "split": "val", "sample_index": "2", "prob_malicious": "0.1", "prediction": "0"}],
            ["label", "split", "sample_index", "prob_malicious", "prediction"],
        )

        report = build_summary(loop57_predictions=loop57, calibrator_predictions=calibrator)

    assert report["blockers"]
    assert report["ready_for_val_fusion_probe"] is False


def test_val_complementarity_requires_source_sha256_when_selected():
    with _case_dir("loop81_overlap_missing_sha") as tmp_path:
        loop57 = _write_csv(
            tmp_path / "loop57.csv",
            [{"label": "0", "split": "val", "sample_index": "1", "final_prob_malicious": "0.1", "prediction": "0"}],
            ["label", "split", "sample_index", "final_prob_malicious", "prediction"],
        )
        calibrator = _write_csv(
            tmp_path / "cal.csv",
            [{"label": "0", "split": "val", "sample_index": "1", "prob_malicious": "0.1", "prediction": "0"}],
            ["label", "split", "sample_index", "prob_malicious", "prediction"],
        )

        try:
            build_summary(loop57_predictions=loop57, calibrator_predictions=calibrator, join_key="source_sha256")
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected missing source_sha256 to fail")

    assert "missing source_sha256" in message


def test_val_complementarity_reports_duplicate_source_sha256_as_blocker():
    with _case_dir("loop81_overlap_duplicate_sha") as tmp_path:
        duplicate_sha = "d" * 64
        loop57 = _write_csv(
            tmp_path / "loop57.csv",
            [
                {
                    "source_sha256": duplicate_sha,
                    "label": "1",
                    "split": "val",
                    "sample_index": "1",
                    "final_prob_malicious": "0.9",
                    "prediction": "1",
                }
            ],
            ["source_sha256", "label", "split", "sample_index", "final_prob_malicious", "prediction"],
        )
        calibrator = _write_csv(
            tmp_path / "cal.csv",
            [
                {
                    "source_sha256": duplicate_sha,
                    "label": "1",
                    "split": "val",
                    "sample_index": "11",
                    "prob_malicious": "0.9",
                    "prediction": "1",
                },
                {
                    "source_sha256": duplicate_sha,
                    "label": "0",
                    "split": "val",
                    "sample_index": "12",
                    "prob_malicious": "0.9",
                    "prediction": "1",
                },
            ],
            ["source_sha256", "label", "split", "sample_index", "prob_malicious", "prediction"],
        )

        report = build_summary(
            loop57_predictions=loop57,
            calibrator_predictions=calibrator,
            join_key="source_sha256",
        )

    assert report["ready_for_val_fusion_probe"] is False
    assert "Calibrator prediction file has duplicate source_sha256 values" in report["blockers"]
    assert "Common source_sha256 set contains ambiguous duplicate keys" in report["blockers"]
    duplicate_summary = report["join_summary"]["duplicate_keys"]["calibrator"]
    assert duplicate_summary["duplicate_key_count"] == 1
    assert duplicate_summary["duplicate_examples"][0]["labels"] == ["0", "1"]
