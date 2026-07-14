import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_model_review_report as review_report  # noqa: E402
from build_model_review_report import build_review, render_markdown  # noqa: E402


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


def _selection_payload() -> dict:
    return {
        "decision_rule": "same protocol",
        "baseline": {
            "model": "baseline",
            "threshold": 0.55,
            "original_hard_family_test_f1": 0.91,
            "original_hard_family_test_fp": 10,
            "original_hard_family_test_fn": 12,
            "original_hard_family_test_errors": 22,
            "hard_error_holdout_f1": 0.5,
            "hard_error_holdout_fp": 4,
            "hard_error_holdout_fn": 5,
            "hard_error_holdout_errors": 9,
            "hard_fn_holdout_f1": 0.8,
            "hard_fn_holdout_fp": 0,
            "hard_fn_holdout_fn": 3,
            "hard_fn_holdout_errors": 3,
        },
        "recommendation": {
            "model": "candidate",
            "threshold": 0.63,
            "original_hard_family_test_f1": 0.94,
            "original_hard_family_test_fp": 8,
            "original_hard_family_test_fn": 9,
            "original_hard_family_test_errors": 17,
            "hard_error_holdout_f1": 0.55,
            "hard_error_holdout_fp": 3,
            "hard_error_holdout_fn": 4,
            "hard_error_holdout_errors": 7,
            "hard_fn_holdout_f1": 0.86,
            "hard_fn_holdout_fp": 0,
            "hard_fn_holdout_fn": 2,
            "hard_fn_holdout_errors": 2,
        },
        "candidate_summary": [
            {
                "model": "candidate",
                "threshold": 0.63,
                "original_hard_family_test_f1": 0.94,
                "original_hard_family_test_fp": 8,
                "original_hard_family_test_fn": 9,
                "original_hard_family_test_errors": 17,
                "hard_error_holdout_errors": 7,
                "hard_fn_holdout_errors": 2,
            }
        ],
    }


def test_build_review_marks_complete_gate_package_usable():
    with _case_dir("model_review_complete") as tmp_path:
        selection = _write_json(tmp_path / "selection.json", _selection_payload())
        val_threshold = _write_json(
            tmp_path / "val.json",
            {
                "clean_baseline": {
                    "val_selected": {"threshold": 0.53, "f1": 0.95},
                    "test_at_val_selected": {"threshold": 0.53, "f1": 0.951, "fp": 20, "fn": 21},
                }
            },
        )
        error_summary = _write_json(
            tmp_path / "errors.json",
            {
                "threshold": 0.55,
                "total_predictions": 100,
                "error_count": 6,
                "false_positive_count": 2,
                "false_negative_count": 4,
                "top_error_groups": [{"group_id": "g1", "error_count": 4, "fp_count": 1, "fn_count": 3}],
            },
        )
        group_summary = _write_json(
            tmp_path / "groups.json",
            {
                "overall": {"groups": 10, "predicted_samples": 100, "error_count": 6, "accuracy": 0.94},
                "rare_groups": {"groups": 8, "predicted_samples": 70, "error_count": 5, "accuracy": 0.93},
                "singleton_groups": {"groups": 6, "predicted_samples": 50, "error_count": 4, "accuracy": 0.92},
                "worst_groups": [{"group_id": "g1"}],
            },
        )

        report = build_review(
            title="Test Review",
            selection_report_path=selection,
            val_threshold_report_path=val_threshold,
            error_summary_path=error_summary,
            group_summary_path=group_summary,
        )
        markdown = render_markdown(report)

    assert report["review_status"] == "usable"
    assert all(gate["status"] == "PASS" for gate in report["gates"])
    assert "## Gate Audit" in markdown
    assert "candidate" in markdown


def test_build_review_includes_probability_calibrator_evaluation():
    with _case_dir("model_review_calibrator") as tmp_path:
        selection = _write_json(tmp_path / "selection.json", _selection_payload())
        calibrator_eval = _write_json(
            tmp_path / "calibrator.json",
            {
                "rows": {"total": 100, "kept": 100, "skipped_missing_cache": 0},
                "baseline": {
                    "threshold": 0.53,
                    "f1": 0.91,
                    "false_positive": 12,
                    "false_negative": 8,
                    "errors": 20,
                    "auc": 0.97,
                },
                "calibrator_metrics": {
                    "threshold": 0.24,
                    "f1": 0.95,
                    "false_positive": 7,
                    "false_negative": 4,
                    "errors": 11,
                    "auc": 0.99,
                },
                "slices": {
                    "whitelist_benign_label_0": {
                        "rows": 40,
                        "baseline": {
                            "threshold": 0.53,
                            "f1": 0.0,
                            "false_positive": 12,
                            "false_negative": 0,
                            "errors": 12,
                            "auc": None,
                        },
                        "calibrator_metrics": {
                            "threshold": 0.24,
                            "f1": 0.0,
                            "false_positive": 7,
                            "false_negative": 0,
                            "errors": 7,
                            "auc": None,
                        },
                    }
                },
            },
        )
        hard_fn_eval = _write_json(
            tmp_path / "hard_fn.json",
            {
                "predictions": "hard_fn_holdout_predictions.csv",
                "rows": {"total": 10, "kept": 4, "skipped_missing_cache": 6},
                "baseline": {
                    "threshold": 0.55,
                    "f1": 0.2,
                    "false_positive": 0,
                    "false_negative": 3,
                    "errors": 3,
                    "auc": None,
                },
                "calibrator_metrics": {
                    "threshold": 0.24,
                    "f1": 0.8,
                    "false_positive": 0,
                    "false_negative": 1,
                    "errors": 1,
                    "auc": None,
                },
            },
        )
        diagnostic_eval = _write_json(
            tmp_path / "diagnostic_current_cache.json",
            {
                "predictions": "test_current_cache_predictions.csv",
                "rows": {"total": 1000, "kept": 600, "skipped_missing_cache": 400},
                "baseline": {
                    "threshold": 0.53,
                    "f1": 0.90,
                    "false_positive": 30,
                    "false_negative": 20,
                    "errors": 50,
                    "auc": 0.96,
                },
                "calibrator_metrics": {
                    "threshold": 0.24,
                    "f1": 0.94,
                    "false_positive": 24,
                    "false_negative": 12,
                    "errors": 36,
                    "auc": 0.98,
                },
                "slices": {
                    "benign_label_0": {
                        "rows": 280,
                        "baseline": {
                            "threshold": 0.53,
                            "f1": 0.0,
                            "false_positive": 30,
                            "false_negative": 0,
                            "errors": 30,
                            "auc": None,
                        },
                        "calibrator_metrics": {
                            "threshold": 0.24,
                            "f1": 0.0,
                            "false_positive": 24,
                            "false_negative": 0,
                            "errors": 24,
                            "auc": None,
                        },
                    }
                },
            },
        )

        report = build_review(
            title="Test Review",
            selection_report_path=selection,
            calibrator_evaluation_path=calibrator_eval,
            calibrator_holdout_evaluation_paths=[hard_fn_eval],
            calibrator_diagnostic_evaluation_paths=[diagnostic_eval],
        )
        markdown = render_markdown(report)

    assert report["calibrator_evaluation"]["rows"]["kept"] == 100
    assert report["inputs"]["calibrator_diagnostic_evaluations"] == [str(diagnostic_eval)]
    assert "## Probability Calibrator" in markdown
    assert "| test | baseline | 100/100 | 0 | 0.530 | 0.9100 | 12 | 8 | 20 | 0.9700 |" in markdown
    assert "| test | calibrator | 100/100 | 0 | 0.240 | 0.9500 | 7 | 4 | 11 | 0.9900 |" in markdown
    assert "| test::whitelist_benign_label_0 | calibrator | 40 |  | 0.240 | 0.0000 | 7 | 0 | 7 |  |" in markdown
    assert "| hard_fn_holdout_predictions | calibrator | 4/10 | 6 | 0.240 | 0.8000 | 0 | 1 | 1 |  |" in markdown
    assert "| diagnostic:test_current_cache_predictions | calibrator | 600/1000 | 400 | 0.240 | 0.9400 | 24 | 12 | 36 | 0.9800 |" in markdown
    assert "| diagnostic:test_current_cache_predictions::benign_label_0 | calibrator | 280 |  | 0.240 | 0.0000 | 24 | 0 | 24 |  |" in markdown


def test_build_review_includes_feature_mask_tradeoff_evidence():
    with _case_dir("model_review_feature_mask") as tmp_path:
        selection = _write_json(tmp_path / "selection.json", _selection_payload())
        feature_mask_eval = _write_json(
            tmp_path / "feature_mask.json",
            {
                "feature_mask": "config/feature_masks/ga_recall_guard_2000.json",
                "feature_mask_summary": {"kept_total": 125, "kept_pe": 95, "kept_stat": 30},
                "samples": 20000,
                "summary": {
                    "baseline_full": {
                        "metrics": {
                            "threshold": 0.5,
                            "f1": 0.931,
                            "false_positive": 382,
                            "false_negative": 958,
                            "errors": 1340,
                        }
                    },
                    "best_mask_f1": {
                        "metrics": {
                            "threshold": 0.5,
                            "f1": 0.9392,
                            "false_positive": 602,
                            "false_negative": 614,
                            "errors": 1216,
                        },
                        "delta_vs_baseline_full": {"false_positive": 220, "false_negative": -344, "errors": -124},
                    },
                    "best_mask_errors": {
                        "metrics": {
                            "threshold": 0.525,
                            "f1": 0.9391,
                            "false_positive": 540,
                            "false_negative": 670,
                            "errors": 1210,
                        },
                        "delta_vs_baseline_full": {"false_positive": 158, "false_negative": -288, "errors": -130},
                    },
                },
            },
        )
        feature_mask_groups = _write_json(
            tmp_path / "feature_mask_groups.json",
            {
                "groups": [
                    {
                        "group": "benign",
                        "full_050": {"sample_count": 10000},
                        "delta_mask0525_minus_full050": {"fp": 158, "fn": 0, "errors": 158},
                    },
                    {
                        "group": "malicious",
                        "full_050": {"sample_count": 10000},
                        "delta_mask0525_minus_full050": {"fp": 0, "fn": -288, "errors": -288},
                    },
                ]
            },
        )
        feature_mask_holdout = _write_json(
            tmp_path / "feature_mask_holdout.json",
            {
                "comparison": "full baseline threshold 0.50 vs GA mask threshold 0.525",
                "sections": {
                    "hard_fn_current_subset": {
                        "full": {
                            "total_predictions": 39,
                            "threshold": 0.5,
                            "false_positive": 0,
                            "false_negative": 12,
                            "errors": 12,
                        },
                        "mask": {
                            "total_predictions": 39,
                            "threshold": 0.525,
                            "false_positive": 0,
                            "false_negative": 11,
                            "errors": 11,
                        },
                        "delta_mask_minus_full": {"false_positive": 0, "false_negative": -1, "errors": -1},
                    }
                },
            },
        )

        report = build_review(
            title="Test Review",
            selection_report_path=selection,
            feature_mask_evaluation_path=feature_mask_eval,
            feature_mask_groups_path=feature_mask_groups,
            feature_mask_holdout_path=feature_mask_holdout,
        )
        markdown = render_markdown(report)

    optional_gates = {gate["name"]: gate["status"] for gate in report["gates"] if gate["severity"] == "optional"}
    assert optional_gates["feature-mask threshold evidence"] == "PASS"
    assert optional_gates["feature-mask source-group evidence"] == "PASS"
    assert "## GA Feature Mask" in markdown
    assert "| mask lowest errors | 0.525 | 0.9391 | 540 | 670 | 1210 | 158 | -288 | -130 |" in markdown
    assert "| benign | 10000 | 158 | 0 | 158 |" in markdown
    assert "### Feature Mask Hard Holdouts" in markdown
    assert "| hard_fn_current_subset | mask | 39 | 0.525 | 0 | 11 | 11 | 0 | -1 | -1 |" in markdown


def test_build_review_warns_when_gate_artifacts_are_missing():
    with _case_dir("model_review_missing") as tmp_path:
        selection = _write_json(tmp_path / "selection.json", _selection_payload())

        report = build_review(title="Test Review", selection_report_path=selection)

    assert report["review_status"] == "usable_with_warnings"
    warning_names = {gate["name"] for gate in report["gates"] if gate["status"] == "WARN"}
    assert "val-selected threshold evidence" in warning_names
    assert "error analysis evidence" in warning_names
    assert "group evaluation evidence" in warning_names


def test_build_review_projects_large_lists_and_omits_raw_blobs():
    with _case_dir("model_review_projection") as tmp_path:
        payload = _selection_payload()
        payload["candidate_summary"] = [
            {
                "model": f"candidate_{index}",
                "threshold": 0.5,
                "original_hard_family_test_f1": 0.9,
                "raw_predictions": ["do-not-copy"] * 100,
            }
            for index in range(review_report.MAX_REPORT_ROWS + 5)
        ]
        selection = _write_json(tmp_path / "selection.json", payload)
        feature_mask_groups = _write_json(
            tmp_path / "feature_mask_groups.json",
            {
                "groups": [
                    {
                        "group": f"group_{index}",
                        "full_050": {"sample_count": index},
                        "delta_mask0525_minus_full050": {"errors": -index},
                        "raw_rows": ["do-not-copy"] * 100,
                    }
                    for index in range(review_report.MAX_REPORT_ROWS + 3)
                ]
            },
        )

        report = build_review(
            title="Projection Review",
            selection_report_path=selection,
            feature_mask_groups_path=feature_mask_groups,
        )

    assert len(report["candidate_summary"]) == review_report.MAX_REPORT_ROWS
    assert "raw_predictions" not in report["candidate_summary"][0]
    assert len(report["feature_mask_groups"]["groups"]) == review_report.MAX_REPORT_ROWS
    assert "raw_rows" not in report["feature_mask_groups"]["groups"][0]


def test_load_json_rejects_oversized_input(tmp_path, monkeypatch):
    path = tmp_path / "large.json"
    path.write_text('{"payload":"' + ("x" * 128) + '"}', encoding="utf-8")
    monkeypatch.setattr(review_report, "MAX_JSON_INPUT_BYTES", 32)

    try:
        review_report.load_json(path)
    except ValueError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("Expected oversized review input to be rejected")
