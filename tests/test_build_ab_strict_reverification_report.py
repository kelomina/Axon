import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_ab_strict_reverification_report import build_report, render_markdown  # noqa: E402


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


def _calibrator_eval(total: int, baseline_errors: int, calibrated_errors: int) -> dict:
    return {
        "rows": {"total": total, "kept": total, "skipped_missing_cache": 0},
        "baseline": {
            "threshold": 0.53,
            "f1": 0.9,
            "auc": 0.95,
            "false_positive": baseline_errors // 2,
            "false_negative": baseline_errors - baseline_errors // 2,
            "errors": baseline_errors,
        },
        "calibrator_metrics": {
            "threshold": 0.24,
            "f1": 0.95,
            "auc": 0.98,
            "false_positive": calibrated_errors // 2,
            "false_negative": calibrated_errors - calibrated_errors // 2,
            "errors": calibrated_errors,
        },
    }


def test_build_ab_strict_reverification_report_summarizes_a_b_results():
    with _case_dir("ab_strict_reverification") as tmp_path:
        cache = _write_json(
            tmp_path / "cache.json",
            {
                "all_full_coverage": True,
                "blocked_recommendations": [],
                "checks": [
                    {"name": "official", "total": 10, "covered": 10, "missing": 0, "coverage_ratio": 1.0},
                    {"name": "high_value", "total": 5, "covered": 5, "missing": 0, "coverage_ratio": 1.0},
                ],
            },
        )
        training = _write_json(
            tmp_path / "train.json",
            {
                "protocol": "train split trains calibrator; val split selects model and threshold; no test used",
                "train_rows": {"total": 8, "kept": 8},
                "val_rows": {"total": 4, "kept": 4},
                "selected": {"val_best": {"threshold": 0.24}},
            },
        )
        test_eval = _write_json(tmp_path / "test.json", _calibrator_eval(10, 4, 1))
        hard_fn = _write_json(tmp_path / "hard_fn.json", _calibrator_eval(3, 2, 0))
        hard_error = _write_json(tmp_path / "hard_error.json", _calibrator_eval(4, 3, 1))
        high_value = _write_json(tmp_path / "high_value.json", _calibrator_eval(5, 2, 1))
        feature_mask_20k = _write_json(
            tmp_path / "mask20k.json",
            {
                "samples": 20,
                "summary": {
                    "baseline_full": {
                        "metrics": {
                            "threshold": 0.5,
                            "f1": 0.9,
                            "false_positive": 3,
                            "false_negative": 5,
                            "errors": 8,
                        }
                    },
                    "best_mask_errors": {
                        "threshold": 0.525,
                        "metrics": {
                            "f1": 0.92,
                            "false_positive": 4,
                            "false_negative": 2,
                            "errors": 6,
                        },
                        "delta_vs_baseline_full": {
                            "false_positive": 1,
                            "false_negative": -3,
                            "errors": -2,
                        },
                    },
                },
            },
        )
        feature_mask_holdout = _write_json(
            tmp_path / "holdout.json",
            {
                "sections": {
                    "hard_fn": {
                        "full": {"errors": 3},
                        "mask": {"errors": 2},
                        "delta_mask_minus_full": {"false_positive": 0, "false_negative": -1, "errors": -1},
                    }
                }
            },
        )
        high_value_baseline = _write_json(
            tmp_path / "high_value_baseline.json",
            {"total_predictions": 5, "threshold": 0.53, "false_positive_count": 2, "false_negative_count": 0, "error_count": 2},
        )
        high_value_mask = _write_json(
            tmp_path / "high_value_mask.json",
            {"total_predictions": 5, "threshold": 0.525, "false_positive_count": 3, "false_negative_count": 0, "error_count": 3},
        )

        report = build_report(
            cache_audit_path=cache,
            calibrator_training_path=training,
            calibrator_test_path=test_eval,
            calibrator_hard_fn_path=hard_fn,
            calibrator_hard_error_path=hard_error,
            calibrator_high_value_path=high_value,
            feature_mask_20k_path=feature_mask_20k,
            feature_mask_holdout_path=feature_mask_holdout,
            high_value_baseline_path=high_value_baseline,
            high_value_mask_path=high_value_mask,
        )
        markdown = render_markdown(report)

    assert report["probability_calibration"]["no_test_used_for_training"] is True
    assert report["probability_calibration"]["all_strict_rows_kept"] is True
    assert report["probability_calibration"]["strict_evaluations"][0]["delta"]["errors"] == -3
    assert report["ga_feature_mask"]["high_value_benign"]["delta_mask_minus_baseline"]["false_positive"] == 1
    assert report["conclusion"]["probability_calibration"] == "strictly_reverified_useful"
    assert "A/B Strict Reverification Report" in markdown
