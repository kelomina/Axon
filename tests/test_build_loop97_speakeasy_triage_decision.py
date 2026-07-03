from __future__ import annotations

import json
from pathlib import Path

from scripts.build_loop97_speakeasy_triage_decision import build_decision


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(*, split_name: str, total: int, rule_new_fn: int, rule_fn_delta: int, malicious_timeouts: int) -> dict:
    return {
        "split_name": split_name,
        "sample_counts": {"total": total},
        "by_role": {
            "calibrator_FP": {"timeouts": 10},
            "matched_correct_malicious_for_FN": {"timeouts": malicious_timeouts},
            "ordinary_malicious": {"timeouts": 0},
            "rule_risk_correct_malicious": {"timeouts": 0},
        },
        "rule_comparison": {
            "calibrator_fixed_threshold": {
                "f1": 0.7,
                "errors": 20,
                "false_positive": 10,
                "false_negative": 10,
            },
            "timeout_filter_score_lt_0.95": {
                "f1": 0.8,
                "errors": 15,
                "false_positive": 0,
                "false_negative": 15,
                "delta_vs_calibrator": {
                    "errors": -5,
                    "false_positive": -10,
                    "false_negative": rule_fn_delta,
                },
                "fixed_baseline_fp": 10,
                "new_fn_from_baseline_tp": rule_new_fn,
            },
        },
    }


def _random_val_summary() -> dict:
    return {
        "sample_counts": {"total": 30},
        "model_comparison": {
            "existing_probability_calibrator_fixed_threshold_on_val_subset": {
                "f1": 1.0,
                "errors": 0,
            }
        },
    }


def test_loop97_blocks_automatic_merge_when_confirmation_adds_fn(tmp_path: Path):
    val_json = _write_json(tmp_path / "val.json", _summary(split_name="val", total=20, rule_new_fn=1, rule_fn_delta=1, malicious_timeouts=0))
    test_json = _write_json(
        tmp_path / "test.json",
        _summary(split_name="test", total=100, rule_new_fn=3, rule_fn_delta=3, malicious_timeouts=2),
    )
    random_json = _write_json(tmp_path / "random.json", _random_val_summary())

    decision = build_decision(
        val_summary_json=val_json,
        test_confirmation_json=test_json,
        random_val_summary_json=random_json,
        output_json=tmp_path / "decision.json",
        max_allowed_new_fn_confirmation=0,
        max_allowed_new_fn_rate_confirmation=0.01,
    )

    assert "confirmation_new_fn_exceeds_zero_tolerance" in decision["blockers"]
    assert "confirmation_new_fn_rate_too_high" in decision["blockers"]
    assert "confirmation_fn_delta_positive" in decision["blockers"]
    assert "timeout_signal_also_hits_true_malicious_rows" in decision["blockers"]
    assert decision["decisions"]["automatic_classifier_merge_allowed"] is False
    assert decision["decisions"]["manual_external_review_signal_allowed"] is True


def test_loop97_allows_manual_signal_even_when_no_blockers(tmp_path: Path):
    val_json = _write_json(tmp_path / "val.json", _summary(split_name="val", total=20, rule_new_fn=0, rule_fn_delta=0, malicious_timeouts=0))
    test_json = _write_json(tmp_path / "test.json", _summary(split_name="test", total=100, rule_new_fn=0, rule_fn_delta=0, malicious_timeouts=0))
    random_json = _write_json(tmp_path / "random.json", _random_val_summary())

    decision = build_decision(
        val_summary_json=val_json,
        test_confirmation_json=test_json,
        random_val_summary_json=random_json,
        output_json=tmp_path / "decision.json",
    )

    assert decision["blockers"] == []
    assert decision["decisions"]["automatic_classifier_merge_allowed"] is False
    assert decision["decisions"]["manual_external_review_signal_allowed"] is True
    assert decision["confirmation_evidence"]["rule"]["fixed_baseline_fp"] == 10
