import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_strict_val_errors import summarize_errors  # noqa: E402


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
                "cache_path",
                "label",
                "split",
                "sample_index",
                "prob_malicious",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_errors_uses_probability_buckets_not_identity_groups():
    with _case_dir("strict_val_error_summary") as tmp_path:
        predictions = tmp_path / "predictions.csv"
        _write_predictions(
            predictions,
            [
                {
                    "source_path": "data/looks_bad/malicious_name.exe",
                    "source_sha256": "a" * 64,
                    "cache_path": "a.npz",
                    "label": "0",
                    "split": "val",
                    "sample_index": "1",
                    "prob_malicious": "0.95",
                },
                {
                    "source_path": "data/looks_good/benign_name.exe",
                    "source_sha256": "b" * 64,
                    "cache_path": "b.npz",
                    "label": "1",
                    "split": "val",
                    "sample_index": "2",
                    "prob_malicious": "0.05",
                },
                {
                    "source_path": "data/whatever.exe",
                    "source_sha256": "c" * 64,
                    "cache_path": "c.npz",
                    "label": "1",
                    "split": "val",
                    "sample_index": "3",
                    "prob_malicious": "0.85",
                },
            ],
        )

        payload = summarize_errors(predictions, threshold=0.5)

    assert payload["error_count"] == 2
    assert payload["false_positive_count"] == 1
    assert payload["false_negative_count"] == 1
    assert payload["confidence_bucket_counts"] == {
        "fn_high_conf_lt_0.10": 1,
        "fp_high_conf_ge_0.90": 1,
    }
    assert "directory" in payload["identity_feature_policy"]
    assert "top_breakdowns" not in payload
    assert payload["error_examples"][0]["source_path"] == "data/looks_bad/malicious_name.exe"
