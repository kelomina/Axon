import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_phase2_error_intrinsics import build_phase2_error_intrinsics  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_cache(
    path: Path,
    pe: list[float],
    stat: list[float],
    *,
    orig_key: str | None = None,
    orig_len: int | None = None,
) -> None:
    payload = {
        "pe_features": np.asarray(pe, dtype=np.float32),
        "stat_features": np.asarray(stat, dtype=np.float32),
    }
    if orig_key is not None:
        payload[orig_key] = np.asarray(orig_len, dtype=np.int64)
    np.savez(path, **payload)


def test_phase2_error_intrinsics_uses_numeric_features_and_writes_review_queue():
    with _case_dir("phase2_error_intrinsics") as tmp_path:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        rows = []
        cases = [
            ("correct0", 0, 0.10, [0.0, 0.0], [0.0, 0.0]),
            ("correct1", 1, 0.90, [1.0, 1.0], [1.0, 1.0]),
            ("fp", 0, 0.95, [10.0, 0.0], [5.0, 0.0]),
            ("fn", 1, 0.05, [-5.0, 1.0], [-2.0, 1.0]),
        ]
        for index, (name, label, prob, pe, stat) in enumerate(cases):
            cache_path = cache_dir / f"{name}.npz"
            _write_cache(cache_path, pe, stat)
            rows.append(
                {
                    "source_path": f"{name}.exe",
                    "source_sha256": str(index) * 64,
                    "cache_path": str(cache_path),
                    "label": str(label),
                    "split": "test",
                    "sample_index": str(index),
                    "calibrated_prob_malicious": str(prob),
                }
            )

        predictions = tmp_path / "predictions.csv"
        with predictions.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        output_json = tmp_path / "report.json"
        review_csv = tmp_path / "review.csv"
        payload = build_phase2_error_intrinsics(
            predictions_csv=predictions,
            output_json=output_json,
            output_review_csv=review_csv,
            threshold=0.44,
            prob_column="calibrated_prob_malicious",
            background_per_label=5,
            seed=1,
            top_k_features=2,
        )

        with review_csv.open("r", encoding="utf-8", newline="") as handle:
            review_rows = list(csv.DictReader(handle))
        stored = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["error_counts"] == {"FN": 1, "FP": 1}
    assert stored["confidence_bucket_counts"] == {
        "fn_high_conf_lt_0.10": 1,
        "fp_high_conf_ge_0.90": 1,
    }
    assert len(review_rows) == 2
    assert {row["error_type"] for row in review_rows} == {"FP", "FN"}
    assert {row["audit_queue"] for row in review_rows} == {"label_noise_high_fp", "label_noise_high_fn"}
    assert all(row["review_status"] == "pending" for row in review_rows)
    assert all("feature_anomaly_flags" in row for row in review_rows)
    assert all("orig_len_missing_or_zero" not in row["feature_anomaly_flags"] for row in review_rows)
    assert all(row["orig_len"] == "" for row in review_rows)
    assert all(row["orig_len_key"] == "" for row in review_rows)
    assert {row["pe_feature_dim"] for row in review_rows} == {"2"}
    assert {row["stat_feature_dim"] for row in review_rows} == {"2"}
    assert "source_path" in review_rows[0]
    assert payload["audit_queue_counts"] == {
        "label_noise_high_fn": 1,
        "label_noise_high_fp": 1,
    }
    assert "directory" in payload["identity_feature_policy"]
    assert "path_group" not in payload
    assert "missing orig_len is recorded as unavailable" in stored["cache_schema_notes"][0]


def test_phase2_error_intrinsics_flags_non_positive_orig_len_only_when_present():
    with _case_dir("phase2_error_orig_len") as tmp_path:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        rows = []
        cases = [
            ("correct0", 0, 0.10, [0.0], [0.0], None, None),
            ("correct1", 1, 0.90, [1.0], [1.0], None, None),
            ("fp_missing_orig", 0, 0.80, [2.0], [2.0], None, None),
            ("fn_zero_orig", 1, 0.20, [3.0], [3.0], "orig_length", 0),
        ]
        for index, (name, label, prob, pe, stat, orig_key, orig_len) in enumerate(cases):
            cache_path = cache_dir / f"{name}.npz"
            _write_cache(cache_path, pe, stat, orig_key=orig_key, orig_len=orig_len)
            rows.append(
                {
                    "source_path": f"{name}.exe",
                    "source_sha256": str(index) * 64,
                    "cache_path": str(cache_path),
                    "label": str(label),
                    "split": "test",
                    "sample_index": str(index),
                    "calibrated_prob_malicious": str(prob),
                }
            )

        predictions = tmp_path / "predictions.csv"
        with predictions.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        review_csv = tmp_path / "review.csv"
        build_phase2_error_intrinsics(
            predictions_csv=predictions,
            output_json=tmp_path / "report.json",
            output_review_csv=review_csv,
            threshold=0.44,
            prob_column="calibrated_prob_malicious",
            background_per_label=5,
            seed=1,
            top_k_features=1,
        )

        with review_csv.open("r", encoding="utf-8", newline="") as handle:
            review_rows = {row["source_path"]: row for row in csv.DictReader(handle)}

    missing = review_rows["fp_missing_orig.exe"]
    zero = review_rows["fn_zero_orig.exe"]
    assert missing["orig_len"] == ""
    assert missing["orig_len_key"] == ""
    assert "orig_len_missing_or_zero" not in missing["feature_anomaly_flags"]
    assert zero["orig_len"] == "0"
    assert zero["orig_len_key"] == "orig_length"
    assert "orig_len_missing_or_zero" in zero["feature_anomaly_flags"]
