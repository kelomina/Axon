import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from search_strict_val_prediction_blend import search_blend  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path",
                "source_sha256",
                "label",
                "split",
                "sample_index",
                "prob_malicious",
            ],
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_search_blend_aligns_by_hash_and_ignores_path_names():
    with _case_dir("strict_blend_hash_only") as tmp_path:
        first_csv = tmp_path / "first.csv"
        second_csv = tmp_path / "second.csv"
        rows_first = [
            {
                "source_path": "benign-looking-name.exe",
                "source_sha256": "b" * 64,
                "label": "1",
                "split": "val",
                "sample_index": "2",
                "prob_malicious": "0.20",
            },
            {
                "source_path": "malware-looking-name.exe",
                "source_sha256": "a" * 64,
                "label": "0",
                "split": "val",
                "sample_index": "1",
                "prob_malicious": "0.80",
            },
        ]
        rows_second = [
            {
                "source_path": "renamed-anything.bin",
                "source_sha256": "a" * 64,
                "label": "0",
                "split": "val",
                "sample_index": "1",
                "prob_malicious": "0.10",
            },
            {
                "source_path": "another-renamed-anything.bin",
                "source_sha256": "b" * 64,
                "label": "1",
                "split": "val",
                "sample_index": "2",
                "prob_malicious": "0.90",
            },
        ]
        _write_predictions(first_csv, rows_first)
        _write_predictions(second_csv, rows_second)

        payload = search_blend(
            first_csv=first_csv,
            second_csv=second_csv,
            first_score_column="prob_malicious",
            second_score_column="prob_malicious",
            weights=[0.0, 1.0],
            thresholds=[0.5],
        )

    assert payload["rows"] == 2
    assert payload["best"]["second_weight"] == 1.0
    assert payload["best"]["f1"] == 1.0
    assert "path/name/directory/extension" in payload["identity_feature_policy"]


def test_search_blend_rejects_partial_overlap():
    with _case_dir("strict_blend_partial_overlap") as tmp_path:
        first_csv = tmp_path / "first.csv"
        second_csv = tmp_path / "second.csv"
        _write_predictions(
            first_csv,
            [
                {"source_path": "a.exe", "source_sha256": "a" * 64, "label": "0", "split": "val", "sample_index": "1", "prob_malicious": "0.1"},
                {"source_path": "b.exe", "source_sha256": "b" * 64, "label": "1", "split": "val", "sample_index": "2", "prob_malicious": "0.9"},
            ],
        )
        _write_predictions(
            second_csv,
            [
                {"source_path": "a-renamed.exe", "source_sha256": "a" * 64, "label": "0", "split": "val", "sample_index": "1", "prob_malicious": "0.2"},
            ],
        )

        with pytest.raises(ValueError, match="exact same source_sha256/sample_index set"):
            search_blend(
                first_csv=first_csv,
                second_csv=second_csv,
                first_score_column="prob_malicious",
                second_score_column="prob_malicious",
                weights=[0.0],
                thresholds=[0.5],
            )


def test_search_blend_rejects_label_and_split_mismatch():
    with _case_dir("strict_blend_label_split_mismatch") as tmp_path:
        first_csv = tmp_path / "first.csv"
        second_csv = tmp_path / "second.csv"
        _write_predictions(
            first_csv,
            [
                {"source_path": "a.exe", "source_sha256": "a" * 64, "label": "0", "split": "val", "sample_index": "1", "prob_malicious": "0.1"},
                {"source_path": "c.exe", "source_sha256": "c" * 64, "label": "1", "split": "val", "sample_index": "3", "prob_malicious": "0.8"},
            ],
        )
        _write_predictions(
            second_csv,
            [
                {"source_path": "a.exe", "source_sha256": "a" * 64, "label": "1", "split": "val", "sample_index": "1", "prob_malicious": "0.2"},
                {"source_path": "c.exe", "source_sha256": "c" * 64, "label": "1", "split": "test", "sample_index": "3", "prob_malicious": "0.7"},
            ],
        )

        with pytest.raises(ValueError) as excinfo:
            search_blend(
                first_csv=first_csv,
                second_csv=second_csv,
                first_score_column="prob_malicious",
                second_score_column="prob_malicious",
                weights=[0.0],
                thresholds=[0.5],
            )

    message = str(excinfo.value)
    assert "label_mismatch" in message
    assert "split_mismatch" in message


def test_search_blend_rejects_sample_index_set_mismatch():
    with _case_dir("strict_blend_sample_index_set_mismatch") as tmp_path:
        first_csv = tmp_path / "first.csv"
        second_csv = tmp_path / "second.csv"
        _write_predictions(
            first_csv,
            [
                {"source_path": "b.exe", "source_sha256": "b" * 64, "label": "1", "split": "val", "sample_index": "2", "prob_malicious": "0.9"},
            ],
        )
        _write_predictions(
            second_csv,
            [
                {"source_path": "b.exe", "source_sha256": "b" * 64, "label": "1", "split": "val", "sample_index": "99", "prob_malicious": "0.8"},
            ],
        )

        with pytest.raises(ValueError, match="exact same source_sha256/sample_index set"):
            search_blend(
                first_csv=first_csv,
                second_csv=second_csv,
                first_score_column="prob_malicious",
                second_score_column="prob_malicious",
                weights=[0.0],
                thresholds=[0.5],
            )


def test_search_blend_allows_duplicate_source_sha256_with_distinct_sample_index():
    with _case_dir("strict_blend_duplicate_sha_distinct_rows") as tmp_path:
        first_csv = tmp_path / "first.csv"
        second_csv = tmp_path / "second.csv"
        first_rows = [
            {"source_path": "a.exe", "source_sha256": "a" * 64, "label": "0", "split": "val", "sample_index": "1", "prob_malicious": "0.1"},
            {"source_path": "a-copy.exe", "source_sha256": "a" * 64, "label": "1", "split": "val", "sample_index": "2", "prob_malicious": "0.8"},
        ]
        second_rows = [
            {"source_path": "renamed-a.exe", "source_sha256": "a" * 64, "label": "0", "split": "val", "sample_index": "1", "prob_malicious": "0.1"},
            {"source_path": "renamed-a-copy.exe", "source_sha256": "a" * 64, "label": "1", "split": "val", "sample_index": "2", "prob_malicious": "0.8"},
        ]
        _write_predictions(first_csv, first_rows)
        _write_predictions(second_csv, second_rows)

        payload = search_blend(
            first_csv=first_csv,
            second_csv=second_csv,
            first_score_column="prob_malicious",
            second_score_column="prob_malicious",
            weights=[0.0],
            thresholds=[0.5],
        )

    assert payload["rows"] == 2
    assert payload["duplicate_source_sha256_rows"] == 1


def test_search_blend_rejects_duplicate_source_sha256_sample_index_key():
    with _case_dir("strict_blend_duplicate_row_key") as tmp_path:
        first_csv = tmp_path / "first.csv"
        second_csv = tmp_path / "second.csv"
        duplicate_rows = [
            {"source_path": "a.exe", "source_sha256": "a" * 64, "label": "0", "split": "val", "sample_index": "1", "prob_malicious": "0.1"},
            {"source_path": "a-copy.exe", "source_sha256": "a" * 64, "label": "0", "split": "val", "sample_index": "1", "prob_malicious": "0.2"},
        ]
        _write_predictions(first_csv, duplicate_rows)
        _write_predictions(
            second_csv,
            [
                {"source_path": "a.exe", "source_sha256": "a" * 64, "label": "0", "split": "val", "sample_index": "1", "prob_malicious": "0.1"},
            ],
        )

        with pytest.raises(ValueError, match="duplicate source_sha256/sample_index"):
            search_blend(
                first_csv=first_csv,
                second_csv=second_csv,
                first_score_column="prob_malicious",
                second_score_column="prob_malicious",
                weights=[0.0],
                thresholds=[0.5],
            )


def test_search_blend_rejects_invalid_source_sha256():
    with _case_dir("strict_blend_invalid_sha") as tmp_path:
        first_csv = tmp_path / "first.csv"
        second_csv = tmp_path / "second.csv"
        rows = [
            {"source_path": "a.exe", "source_sha256": "not-a-sha", "label": "0", "split": "val", "sample_index": "1", "prob_malicious": "0.1"},
        ]
        _write_predictions(first_csv, rows)
        _write_predictions(second_csv, rows)

        with pytest.raises(ValueError, match="invalid source_sha256"):
            search_blend(
                first_csv=first_csv,
                second_csv=second_csv,
                first_score_column="prob_malicious",
                second_score_column="prob_malicious",
                weights=[0.0],
                thresholds=[0.5],
            )


@pytest.mark.parametrize(
    "score_column",
    ["source_path", "filename", "directory", "extension", "source_sha256", "sample_index", "split", "label"],
)
def test_search_blend_rejects_identity_or_leakage_score_columns(score_column: str):
    with _case_dir("strict_blend_denied_score_column") as tmp_path:
        first_csv = tmp_path / "first.csv"
        second_csv = tmp_path / "second.csv"
        rows = [
            {
                "source_path": "a.exe",
                "source_sha256": "a" * 64,
                "label": "0",
                "split": "val",
                "sample_index": "1",
                "prob_malicious": "0.1",
                score_column: "0.5",
            },
        ]
        _write_predictions(first_csv, rows)
        _write_predictions(second_csv, rows)

        with pytest.raises(ValueError, match="identity/leakage column"):
            search_blend(
                first_csv=first_csv,
                second_csv=second_csv,
                first_score_column=score_column,
                second_score_column="prob_malicious",
                weights=[0.0],
                thresholds=[0.5],
            )
