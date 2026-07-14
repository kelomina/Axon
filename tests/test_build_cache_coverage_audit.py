import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_cache_coverage_audit import build_audit  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_cache_coverage_audit_marks_blocked_recommendations():
    with _case_dir("cache_coverage_audit") as tmp_path:
        test_subset = _write_json(
            tmp_path / "test_subset.json",
            {"rows": {"total": 100, "kept": 80, "skipped_missing_cache": 20, "missing_cache_output": "missing.csv"}},
        )
        hard_fn = _write_json(
            tmp_path / "hard_fn.json",
            {"raw_samples": 10, "predicted_samples": 4, "missing_cache_samples": 6},
        )
        hard_error = _write_json(
            tmp_path / "hard_error.json",
            {"raw_samples": 12, "predicted_samples": 12, "missing_cache_samples": 0},
        )

        audit = build_audit(
            test_current_subset=test_subset,
            hard_fn_summary=hard_fn,
            hard_error_summary=hard_error,
        )

    by_name = {row["name"]: row for row in audit["checks"]}
    assert by_name["official_test_current_cache_subset"]["coverage_ratio"] == 0.8
    assert by_name["official_test_current_cache_subset"]["missing_output"] == "missing.csv"
    assert by_name["hard_fn_holdout_current_cache_subset"]["coverage_ratio"] == 0.4
    assert audit["all_full_coverage"] is False
    assert audit["blocked_recommendations"] == ["ga_feature_mask", "probability_calibration"]


def test_build_cache_coverage_audit_can_include_high_value_benign_checks():
    with _case_dir("cache_coverage_audit_high_value_benign") as tmp_path:
        test_subset = _write_json(
            tmp_path / "test_subset.json",
            {"rows": {"total": 100, "kept": 100, "skipped_missing_cache": 0}},
        )
        hard_fn = _write_json(
            tmp_path / "hard_fn.json",
            {"raw_samples": 10, "predicted_samples": 10, "missing_cache_samples": 0},
        )
        hard_error = _write_json(
            tmp_path / "hard_error.json",
            {"raw_samples": 12, "predicted_samples": 12, "missing_cache_samples": 0},
        )
        high_value_calibrator = _write_json(
            tmp_path / "high_value_calibrator.json",
            {"rows": {"total": 8127, "kept": 8127, "skipped_missing_cache": 0}},
        )
        high_value_full = _write_json(
            tmp_path / "high_value_full.json",
            {"raw_samples": 8127, "predicted_samples": 8127, "missing_cache_samples": 0},
        )
        high_value_mask = _write_json(
            tmp_path / "high_value_mask.json",
            {"raw_samples": 8127, "predicted_samples": 8100, "missing_cache_samples": 27},
        )
        ga_full_hard_fn = _write_json(
            tmp_path / "ga_full_hard_fn.json",
            {"raw_samples": 201, "predicted_samples": 201, "missing_cache_samples": 0},
        )
        ga_mask_hard_fn = _write_json(
            tmp_path / "ga_mask_hard_fn.json",
            {"raw_samples": 201, "predicted_samples": 201, "missing_cache_samples": 0},
        )
        ga_full_hard_error = _write_json(
            tmp_path / "ga_full_hard_error.json",
            {"raw_samples": 317, "predicted_samples": 317, "missing_cache_samples": 0},
        )
        ga_mask_hard_error = _write_json(
            tmp_path / "ga_mask_hard_error.json",
            {"raw_samples": 317, "predicted_samples": 300, "missing_cache_samples": 17},
        )

        audit = build_audit(
            test_current_subset=test_subset,
            hard_fn_summary=hard_fn,
            hard_error_summary=hard_error,
            high_value_benign_calibrator_eval=high_value_calibrator,
            high_value_benign_full_summary=high_value_full,
            high_value_benign_mask_summary=high_value_mask,
            ga_full_hard_fn_summary=ga_full_hard_fn,
            ga_mask_hard_fn_summary=ga_mask_hard_fn,
            ga_full_hard_error_summary=ga_full_hard_error,
            ga_mask_hard_error_summary=ga_mask_hard_error,
        )

    by_name = {row["name"]: row for row in audit["checks"]}
    assert by_name["high_value_benign_probability_calibrator_strict_full"]["coverage_ratio"] == 1.0
    assert by_name["high_value_benign_full_feature_cache_subset"]["missing"] == 0
    assert by_name["high_value_benign_ga_mask_cache_subset"]["missing"] == 27
    assert by_name["ga_full_hard_fn_holdout_cache_subset"]["missing"] == 0
    assert by_name["ga_mask_hard_error_holdout_cache_subset"]["missing"] == 17
    assert audit["all_full_coverage"] is False
    assert audit["blocked_recommendations"] == ["ga_feature_mask"]
