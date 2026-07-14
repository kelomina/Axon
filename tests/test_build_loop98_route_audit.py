from __future__ import annotations

import json
from pathlib import Path

from scripts.build_loop98_route_audit import build_audit


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _loop79() -> dict:
    return {
        "decision": "pass",
        "sections": {
            "fixed_v2_replacement_130": {
                "replacement_rows": 130,
                "self_replacements": 0,
                "selection_status_counts": {"strict_extracted": 130},
            },
            "current_split_cache": {
                "total_rows": 200000,
                "covered_rows": 200000,
                "missing_rows": 0,
                "label_balance_enforced": True,
                "cache_metadata_validation_enabled": True,
                "metadata_checked_rows": 200000,
                "metadata_failure_rows": 0,
                "sampled_rows": 2000,
                "sample_failed_rows": 0,
            },
        },
    }


def _loop80() -> dict:
    return {
        "decision": "not_final_candidate",
        "blockers": ["calibrator does not beat current Loop57 full-test best"],
        "rows": {"kept": 160000},
        "calibrator": {"metrics": {"f1": 0.96, "errors": 5000}},
        "loop57_current_best": {"metrics": {"f1": 0.988, "errors": 1868}},
        "deltas": {"calibrator_vs_loop57": {"errors": 3132}},
    }


def _loop85(*, stop_fusion: bool = True) -> dict:
    return {
        "current_best": {
            "model": "Loop57 FN overlay gate",
            "full_test_rows": 160000,
            "f1": 0.9883629658239992,
            "errors": 1868,
            "false_positive": 1195,
            "false_negative": 673,
        },
        "target_gap": {
            "f1_target_error_budget_approx": 160,
            "minimum_error_reduction_needed_approx": 1708,
        },
        "fusion_evidence": {
            "loop82_calibrator_only_correct": 56,
            "loop82_loop57_only_correct": 463,
            "loop83_score_delta_improves_loop57": False,
            "loop84_content_selector_promising": False,
            "stop_current_calibrator_fusion": stop_fusion,
        },
    }


def _loop95() -> dict:
    return {
        "rows": 1868,
        "expected_rows": 1868,
        "blockers": [],
        "decisions": {
            "training_allowed": False,
            "ready_for_loop87_full_queue_import": True,
        },
    }


def _loop96() -> dict:
    return {
        "rows": 1868,
        "blockers": [],
        "decisions": {
            "ready_for_blinded_review": True,
            "training_allowed": False,
        },
    }


def _loop96_import(*, actionable_rows: int = 0, replacement_rows: int = 0, training_rows: int = 0) -> dict:
    decision = "ready_noop_no_actionable_verdicts" if actionable_rows == 0 else "ready_for_redraw_plan_review_only"
    return {
        "rows": 1868,
        "blocking_issues": [],
        "decision": decision,
        "manual_quality": {"blank_verdict_rows": 1868 - actionable_rows},
        "actionable_rows": actionable_rows,
        "replacement_required_rows": replacement_rows,
        "training_policy_rows": training_rows,
    }


def _loop97() -> dict:
    return {
        "blockers": ["confirmation_new_fn_exceeds_zero_tolerance"],
        "confirmation_evidence": {"rule": {"new_fn_from_baseline_tp": 3, "fn_delta": 3}},
        "decisions": {
            "automatic_classifier_merge_allowed": False,
            "automatic_threshold_override_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "manual_external_review_signal_allowed": True,
        },
    }


def _case(tmp_path: Path, *, actionable_rows: int = 0, replacement_rows: int = 0, training_rows: int = 0):
    return {
        "loop79_current_state": _write_json(tmp_path / "loop79.json", _loop79()),
        "loop80_calibrator_fulltest": _write_json(tmp_path / "loop80.json", _loop80()),
        "loop85_noise_strategy": _write_json(tmp_path / "loop85.json", _loop85()),
        "loop95_intake": _write_json(tmp_path / "loop95.json", _loop95()),
        "loop96_blinded_review": _write_json(tmp_path / "loop96.json", _loop96()),
        "loop96_verdict_import": _write_json(
            tmp_path / "loop96_import.json",
            _loop96_import(
                actionable_rows=actionable_rows,
                replacement_rows=replacement_rows,
                training_rows=training_rows,
            ),
        ),
        "loop97_speakeasy": _write_json(tmp_path / "loop97.json", _loop97()),
        "output_json": tmp_path / "loop98.json",
    }


def test_loop98_awaits_independent_verdicts_when_full_queue_is_blank(tmp_path: Path):
    report = build_audit(**_case(tmp_path))

    assert report["decision"] == "await_independent_blinded_verdicts"
    assert report["decisions"]["training_allowed_now"] is False
    assert report["decisions"]["test10k_allowed_now"] is False
    assert report["decisions"]["current_automatic_model_route_open"] is False
    assert report["route_sections"]["fixed_v2_cache_and_redraw"]["status"] == "pass"
    assert report["route_sections"]["fixed_v2_cache_and_redraw"]["evidence"]["label_balance_enforced"] is True
    assert report["route_sections"]["fixed_v2_cache_and_redraw"]["evidence"]["metadata_checked_rows"] == 200000
    assert report["route_sections"]["fixed_v2_cache_and_redraw"]["evidence"]["metadata_failure_rows"] == 0
    assert report["route_sections"]["probability_calibrator"]["status"] == "closed_as_final_candidate"
    assert report["route_sections"]["current_calibrator_fusion"]["status"] == "closed"
    assert report["route_sections"]["speakeasy_dynamic_triage"]["status"] == "manual_context_only"
    assert "authorization only" in report["evidence_semantics"]
    assert "not model features" in report["evidence_semantics"]
    assert "filename" in report["identity_feature_policy"]["forbidden_as_model_or_verdict_evidence"]
    assert report["redraw_policy"]["self_fill_allowed"] is False


def test_loop98_marks_redraw_preflight_ready_only_for_actionable_replacements(tmp_path: Path):
    report = build_audit(**_case(tmp_path, actionable_rows=4, replacement_rows=4, training_rows=0))

    assert report["decision"] == "ready_for_non_destructive_redraw_preflight"
    assert report["decisions"]["ready_for_redraw_preflight"] is True
    assert report["decisions"]["training_allowed_now"] is False


def test_loop98_blocks_training_policy_rows_from_verdict_import(tmp_path: Path):
    report = build_audit(**_case(tmp_path, actionable_rows=4, replacement_rows=4, training_rows=1))

    assert report["decision"] == "await_independent_blinded_verdicts"
    assert "loop96_training_policy_rows_present" in report["route_sections"]["full_queue_review"]["blockers"]
    assert report["decisions"]["ready_for_redraw_preflight"] is False


def test_loop98_blocks_loop79_without_cache_metadata_readiness(tmp_path: Path):
    paths = _case(tmp_path)
    loop79 = _loop79()
    loop79["sections"]["current_split_cache"].pop("cache_metadata_validation_enabled")
    loop79["sections"]["current_split_cache"].pop("metadata_checked_rows")
    loop79["sections"]["current_split_cache"].pop("metadata_failure_rows")
    paths["loop79_current_state"] = _write_json(tmp_path / "loop79_legacy.json", loop79)

    report = build_audit(**paths)

    assert report["route_sections"]["fixed_v2_cache_and_redraw"]["status"] == "block"
    assert "current_cache_metadata_validation_not_enabled" in report["route_sections"]["fixed_v2_cache_and_redraw"]["blockers"]
    assert "current_cache_metadata_not_fully_checked" in report["route_sections"]["fixed_v2_cache_and_redraw"]["blockers"]
    assert "current_cache_metadata_failures_present" in report["route_sections"]["fixed_v2_cache_and_redraw"]["blockers"]


def test_loop98_blocks_loop79_without_label_balance_enforced(tmp_path: Path):
    paths = _case(tmp_path)
    loop79 = _loop79()
    loop79["sections"]["current_split_cache"]["label_balance_enforced"] = False
    paths["loop79_current_state"] = _write_json(tmp_path / "loop79_unbalanced.json", loop79)

    report = build_audit(**paths)

    assert report["route_sections"]["fixed_v2_cache_and_redraw"]["status"] == "block"
    assert "current_cache_label_balance_not_enforced" in report["route_sections"]["fixed_v2_cache_and_redraw"]["blockers"]


def test_loop98_blocks_loop79_with_cache_metadata_failures(tmp_path: Path):
    paths = _case(tmp_path)
    loop79 = _loop79()
    loop79["sections"]["current_split_cache"]["metadata_failure_rows"] = 1
    paths["loop79_current_state"] = _write_json(tmp_path / "loop79_bad_metadata.json", loop79)

    report = build_audit(**paths)

    assert report["route_sections"]["fixed_v2_cache_and_redraw"]["status"] == "block"
    assert "current_cache_metadata_failures_present" in report["route_sections"]["fixed_v2_cache_and_redraw"]["blockers"]
