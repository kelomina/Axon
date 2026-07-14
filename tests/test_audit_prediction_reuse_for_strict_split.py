import csv
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_prediction_reuse_for_strict_split import audit_reuse  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "source_sha256", "label", "split", "sample_index", "prob_malicious"],
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_audit_reuse_ignores_path_names_and_accepts_exact_hash_rows():
    with _case_dir("prediction_reuse_exact") as tmp_path:
        split_csv = tmp_path / "split.csv"
        predictions_csv = tmp_path / "predictions.csv"
        split_rows = [
            {"source_path": "bad-name.exe", "source_sha256": "a" * 64, "label": "0", "split": "train", "sample_index": "1"},
            {"source_path": "good-name.exe", "source_sha256": "b" * 64, "label": "1", "split": "train", "sample_index": "2"},
        ]
        prediction_rows = [
            {"source_path": "renamed-1.bin", "source_sha256": "a" * 64, "label": "0", "split": "train", "sample_index": "1", "prob_malicious": "0.1"},
            {"source_path": "renamed-2.bin", "source_sha256": "b" * 64, "label": "1", "split": "train", "sample_index": "2", "prob_malicious": "0.9"},
        ]
        _write_rows(split_csv, split_rows)
        _write_rows(predictions_csv, prediction_rows)

        payload = audit_reuse(strict_split_csv=split_csv, predictions_csv=predictions_csv, split="train")

    assert payload["decision"] == "reusable_exact"
    assert payload["missing_key_count"] == 0
    assert payload["extra_key_count"] == 0
    assert payload["reusable_by_unique_source_sha256_label"] == 2


def test_audit_reuse_rejects_non_exact_row_set_even_with_high_sha_overlap():
    with _case_dir("prediction_reuse_not_exact") as tmp_path:
        split_csv = tmp_path / "split.csv"
        predictions_csv = tmp_path / "predictions.csv"
        _write_rows(
            split_csv,
            [
                {"source_path": "a.exe", "source_sha256": "a" * 64, "label": "0", "split": "train", "sample_index": "1"},
                {"source_path": "b.exe", "source_sha256": "b" * 64, "label": "1", "split": "train", "sample_index": "2"},
            ],
        )
        _write_rows(
            predictions_csv,
            [
                {"source_path": "a-renamed.exe", "source_sha256": "a" * 64, "label": "0", "split": "train", "sample_index": "1", "prob_malicious": "0.1"},
                {"source_path": "b-renamed.exe", "source_sha256": "b" * 64, "label": "1", "split": "train", "sample_index": "99", "prob_malicious": "0.9"},
            ],
        )

        payload = audit_reuse(strict_split_csv=split_csv, predictions_csv=predictions_csv, split="train")

    assert payload["decision"] == "not_reusable_exact"
    assert payload["missing_key_count"] == 1
    assert payload["extra_key_count"] == 1
