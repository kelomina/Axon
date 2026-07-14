#!/usr/bin/env python3
"""Build a read-only Loop98 route audit from existing gate reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence


IDENTITY_FIELDS = [
    "filename",
    "path",
    "extension",
    "directory",
    "hash",
    "source_sha256",
    "sample_index",
    "split",
    "row_order",
    "model_score",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _get(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _section_status(blockers: list[str]) -> str:
    return "pass" if not blockers else "block"


def build_audit(
    *,
    loop79_current_state: Path,
    loop80_calibrator_fulltest: Path,
    loop85_noise_strategy: Path,
    loop95_intake: Path,
    loop96_blinded_review: Path,
    loop96_verdict_import: Path,
    loop97_speakeasy: Path,
    output_json: Path,
) -> dict[str, Any]:
    loop79 = read_json(loop79_current_state)
    loop80 = read_json(loop80_calibrator_fulltest)
    loop85 = read_json(loop85_noise_strategy)
    loop95 = read_json(loop95_intake)
    loop96 = read_json(loop96_blinded_review)
    loop96_import = read_json(loop96_verdict_import)
    loop97 = read_json(loop97_speakeasy)

    blockers: list[str] = []

    fixed_v2_blockers = []
    if loop79.get("decision") != "pass":
        fixed_v2_blockers.append("loop79_current_state_gate_not_pass")
    if _int(_get(loop79, "sections", "fixed_v2_replacement_130", "replacement_rows"), -1) != 130:
        fixed_v2_blockers.append("replacement_130_count_not_proven")
    if _int(_get(loop79, "sections", "fixed_v2_replacement_130", "self_replacements"), -1) != 0:
        fixed_v2_blockers.append("replacement_self_fill_detected")
    if _int(_get(loop79, "sections", "current_split_cache", "total_rows"), -1) != 200000:
        fixed_v2_blockers.append("current_split_not_200000")
    if _int(_get(loop79, "sections", "current_split_cache", "covered_rows"), -1) != 200000:
        fixed_v2_blockers.append("current_cache_not_fully_covered")
    if _int(_get(loop79, "sections", "current_split_cache", "missing_rows"), -1) != 0:
        fixed_v2_blockers.append("current_cache_missing_rows")
    if _get(loop79, "sections", "current_split_cache", "label_balance_enforced") is not True:
        fixed_v2_blockers.append("current_cache_label_balance_not_enforced")
    if _get(loop79, "sections", "current_split_cache", "cache_metadata_validation_enabled") is not True:
        fixed_v2_blockers.append("current_cache_metadata_validation_not_enabled")
    if _int(_get(loop79, "sections", "current_split_cache", "metadata_checked_rows"), -1) != 200000:
        fixed_v2_blockers.append("current_cache_metadata_not_fully_checked")
    if _int(_get(loop79, "sections", "current_split_cache", "metadata_failure_rows"), -1) != 0:
        fixed_v2_blockers.append("current_cache_metadata_failures_present")

    current_best = _get(loop85, "current_best", default={}) or {}
    target_gap = _get(loop85, "target_gap", default={}) or {}
    if _int(current_best.get("full_test_rows"), -1) != 160000:
        blockers.append("current_best_full_test_rows_not_160000")
    if _int(current_best.get("errors"), -1) <= 0:
        blockers.append("current_best_errors_missing")

    calibrator_blockers = _list(loop80.get("blockers"))
    calibrator_decision = loop80.get("decision")
    if calibrator_decision != "not_final_candidate":
        blockers.append("loop80_calibrator_decision_unexpected")

    fusion_stop = bool(_get(loop85, "fusion_evidence", "stop_current_calibrator_fusion", default=False))
    if not fusion_stop:
        blockers.append("loop85_does_not_stop_current_calibrator_fusion")

    intake_blockers = _list(loop95.get("blockers"))
    if _int(loop95.get("rows"), -1) != 1868 or _int(loop95.get("expected_rows"), -1) != 1868:
        intake_blockers.append("loop95_full_queue_not_1868")
    if bool(_get(loop95, "decisions", "training_allowed", default=True)):
        intake_blockers.append("loop95_unexpected_training_allowed")

    blinded_blockers = _list(loop96.get("blockers"))
    if _int(loop96.get("rows"), -1) != 1868:
        blinded_blockers.append("loop96_blinded_rows_not_1868")
    if not bool(_get(loop96, "decisions", "ready_for_blinded_review", default=False)):
        blinded_blockers.append("loop96_not_ready_for_blinded_review")
    if bool(_get(loop96, "decisions", "training_allowed", default=True)):
        blinded_blockers.append("loop96_unexpected_training_allowed")

    verdict_blockers = _list(loop96_import.get("blocking_issues"))
    if _int(loop96_import.get("rows"), -1) != 1868:
        verdict_blockers.append("loop96_verdict_import_rows_not_1868")
    if _int(loop96_import.get("training_policy_rows"), -1) != 0:
        verdict_blockers.append("loop96_training_policy_rows_present")
    if loop96_import.get("decision") not in {
        "ready_noop_no_actionable_verdicts",
        "ready_for_redraw_plan_review_only",
    }:
        verdict_blockers.append("loop96_verdict_import_not_ready")

    speakeasy_blockers = _list(loop97.get("blockers"))
    if bool(_get(loop97, "decisions", "automatic_classifier_merge_allowed", default=True)):
        speakeasy_blockers.append("loop97_unexpected_automatic_merge_allowed")
    if bool(_get(loop97, "decisions", "training_allowed", default=True)):
        speakeasy_blockers.append("loop97_unexpected_training_allowed")
    if bool(_get(loop97, "decisions", "test10k_allowed", default=True)):
        speakeasy_blockers.append("loop97_unexpected_test10k_allowed")

    route_sections = {
        "fixed_v2_cache_and_redraw": {
            "status": _section_status(fixed_v2_blockers),
            "blockers": fixed_v2_blockers,
            "evidence": {
                "loop79_decision": loop79.get("decision"),
                "replacement_rows": _get(loop79, "sections", "fixed_v2_replacement_130", "replacement_rows"),
                "self_replacements": _get(loop79, "sections", "fixed_v2_replacement_130", "self_replacements"),
                "selection_status_counts": _get(loop79, "sections", "fixed_v2_replacement_130", "selection_status_counts"),
                "current_total_rows": _get(loop79, "sections", "current_split_cache", "total_rows"),
                "current_covered_rows": _get(loop79, "sections", "current_split_cache", "covered_rows"),
                "current_missing_rows": _get(loop79, "sections", "current_split_cache", "missing_rows"),
                "label_balance_enforced": _get(loop79, "sections", "current_split_cache", "label_balance_enforced"),
                "cache_metadata_validation_enabled": _get(loop79, "sections", "current_split_cache", "cache_metadata_validation_enabled"),
                "metadata_checked_rows": _get(loop79, "sections", "current_split_cache", "metadata_checked_rows"),
                "metadata_failure_rows": _get(loop79, "sections", "current_split_cache", "metadata_failure_rows"),
                "sampled_rows": _get(loop79, "sections", "current_split_cache", "sampled_rows"),
                "sample_failed_rows": _get(loop79, "sections", "current_split_cache", "sample_failed_rows"),
            },
        },
        "probability_calibrator": {
            "status": "closed_as_final_candidate" if calibrator_decision == "not_final_candidate" else "review",
            "blockers": calibrator_blockers,
            "evidence": {
                "decision": calibrator_decision,
                "full_test_rows": _get(loop80, "rows", "kept"),
                "calibrator_f1": _get(loop80, "calibrator", "metrics", "f1"),
                "calibrator_errors": _get(loop80, "calibrator", "metrics", "errors"),
                "loop57_f1": _get(loop80, "loop57_current_best", "metrics", "f1"),
                "loop57_errors": _get(loop80, "loop57_current_best", "metrics", "errors"),
                "delta_errors_vs_loop57": _get(loop80, "deltas", "calibrator_vs_loop57", "errors"),
            },
        },
        "current_calibrator_fusion": {
            "status": "closed" if fusion_stop else "review",
            "blockers": [] if fusion_stop else ["loop85_did_not_stop_current_fusion_route"],
            "evidence": {
                "loop82_calibrator_only_correct": _get(loop85, "fusion_evidence", "loop82_calibrator_only_correct"),
                "loop82_loop57_only_correct": _get(loop85, "fusion_evidence", "loop82_loop57_only_correct"),
                "loop83_score_delta_improves_loop57": _get(loop85, "fusion_evidence", "loop83_score_delta_improves_loop57"),
                "loop84_content_selector_promising": _get(loop85, "fusion_evidence", "loop84_content_selector_promising"),
                "stop_current_calibrator_fusion": fusion_stop,
            },
        },
        "full_queue_review": {
            "status": _section_status(intake_blockers + blinded_blockers + verdict_blockers),
            "blockers": intake_blockers + blinded_blockers + verdict_blockers,
            "evidence": {
                "intake_rows": loop95.get("rows"),
                "intake_ready_for_loop87": _get(loop95, "decisions", "ready_for_loop87_full_queue_import"),
                "blinded_rows": loop96.get("rows"),
                "ready_for_blinded_review": _get(loop96, "decisions", "ready_for_blinded_review"),
                "verdict_import_decision": loop96_import.get("decision"),
                "blank_verdict_rows": _get(loop96_import, "manual_quality", "blank_verdict_rows"),
                "actionable_rows": loop96_import.get("actionable_rows"),
                "replacement_required_rows": loop96_import.get("replacement_required_rows"),
                "training_policy_rows": loop96_import.get("training_policy_rows"),
            },
        },
        "speakeasy_dynamic_triage": {
            "status": "manual_context_only" if speakeasy_blockers else "review",
            "blockers": speakeasy_blockers,
            "evidence": {
                "automatic_classifier_merge_allowed": _get(loop97, "decisions", "automatic_classifier_merge_allowed"),
                "automatic_threshold_override_allowed": _get(loop97, "decisions", "automatic_threshold_override_allowed"),
                "training_allowed": _get(loop97, "decisions", "training_allowed"),
                "test10k_allowed": _get(loop97, "decisions", "test10k_allowed"),
                "manual_external_review_signal_allowed": _get(loop97, "decisions", "manual_external_review_signal_allowed"),
                "confirmation_new_fn": _get(loop97, "confirmation_evidence", "rule", "new_fn_from_baseline_tp"),
                "confirmation_fn_delta": _get(loop97, "confirmation_evidence", "rule", "fn_delta"),
            },
        },
    }

    automatic_route_open = (
        not fixed_v2_blockers
        and not intake_blockers
        and not blinded_blockers
        and not verdict_blockers
        and calibrator_decision != "not_final_candidate"
        and not fusion_stop
        and not speakeasy_blockers
    )
    if automatic_route_open:
        blockers.append("unexpected_automatic_route_open")

    full_queue_has_actionable_verdicts = _int(loop96_import.get("actionable_rows"), 0) > 0
    ready_for_redraw = (
        full_queue_has_actionable_verdicts
        and _int(loop96_import.get("replacement_required_rows"), 0) > 0
        and _int(loop96_import.get("training_policy_rows"), 0) == 0
        and not verdict_blockers
    )

    final_decision = (
        "await_independent_blinded_verdicts"
        if not ready_for_redraw
        else "ready_for_non_destructive_redraw_preflight"
    )

    report = {
        "schema": "axon_loop98_identity_safe_route_audit_v1",
        "protocol": (
            "read-only route audit; no training, no model loading, no threshold tuning, "
            "no NPZ array loading, no split/cache mutation"
        ),
        "evidence_semantics": (
            "Fields named evidence in this route report are audit/source-of-truth summaries "
            "for authorization only. They are not model features, verdict evidence, threshold "
            "signals, fusion inputs, relabel evidence, replacement-sampling evidence, or "
            "production inference evidence."
        ),
        "inputs": {
            "loop79_current_state": str(loop79_current_state),
            "loop80_calibrator_fulltest": str(loop80_calibrator_fulltest),
            "loop85_noise_strategy": str(loop85_noise_strategy),
            "loop95_intake": str(loop95_intake),
            "loop96_blinded_review": str(loop96_blinded_review),
            "loop96_verdict_import": str(loop96_verdict_import),
            "loop97_speakeasy": str(loop97_speakeasy),
        },
        "decision": final_decision,
        "blockers": sorted(set(blockers)),
        "current_best": {
            "model": current_best.get("model", "Loop57 FN overlay gate"),
            "full_test_rows": _int(current_best.get("full_test_rows"), 0),
            "f1": _float(current_best.get("f1"), 0.0),
            "errors": _int(current_best.get("errors"), 0),
            "false_positive": _int(current_best.get("false_positive"), 0),
            "false_negative": _int(current_best.get("false_negative"), 0),
        },
        "target_gap": {
            "target_f1": 0.999,
            "approx_error_budget": _int(target_gap.get("f1_target_error_budget_approx"), 160),
            "minimum_error_reduction_needed": _int(target_gap.get("minimum_error_reduction_needed_approx"), 0),
        },
        "route_sections": route_sections,
        "identity_feature_policy": {
            "forbidden_as_model_or_verdict_evidence": IDENTITY_FIELDS,
            "allowed_uses": [
                "loading",
                "alignment",
                "cache audit",
                "duplicate detection",
                "manual/external review indexing",
            ],
            "strict_boundary": (
                "Real deployment names do not match training-corpus names. Filename, path, directory, extension, "
                "hash, source_sha256, sample_index, split, row order, and model score fields must not be used as "
                "training, fusion, threshold, relabel, replacement-sampling, or production inference evidence."
            ),
            "label_boundary": (
                "If an original corpus used human-curated directories to bootstrap labels, that step ends at the "
                "locked split/manifest label. The 20w protocol uses the locked manifest label pool for redraws, "
                "not names or paths."
            ),
        },
        "redraw_policy": {
            "bad_row_actions": ["label_wrong", "feature_broken", "out_of_scope"],
            "required_action": "quarantine_then_fresh_same_original_label_redraw",
            "self_fill_allowed": False,
            "replacement_pool": "locked_manifest_original_label_pool",
            "exact_split_required": {
                "total": 200000,
                "train": 20000,
                "val": 20000,
                "test": 160000,
            },
        },
        "decisions": {
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed_without_verdict": False,
            "training_allowed_now": False,
            "test10k_allowed_now": False,
            "full_test_allowed_now": False,
            "current_automatic_model_route_open": False,
            "ready_for_redraw_preflight": ready_for_redraw,
            "next_allowed_step": (
                "Fill Loop96 blinded review with independent content/external verdicts, unblind, run Loop87, "
                "then build non-destructive fresh same-original-label redraw preflight."
                if not ready_for_redraw
                else "Run redraw readiness/preflight before any Train/Val-only retraining."
            ),
        },
    }
    write_json(output_json, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop98 identity-safe route audit.")
    parser.add_argument("--loop79-current-state", type=Path, default=Path("reports/random_20w_split/loop79_current_state_gate.json"))
    parser.add_argument("--loop80-calibrator-fulltest", type=Path, default=Path("reports/random_20w_split/loop80_calibrator_fulltest_summary.json"))
    parser.add_argument("--loop85-noise-strategy", type=Path, default=Path("reports/random_20w_split/loop85_noise_strategy_gate.json"))
    parser.add_argument("--loop95-intake", type=Path, default=Path("reports/random_20w_split/loop95_full_queue_review_evidence_intake.json"))
    parser.add_argument("--loop96-blinded-review", type=Path, default=Path("reports/random_20w_split/loop96_full_queue_blinded_review.json"))
    parser.add_argument("--loop96-verdict-import", type=Path, default=Path("reports/random_20w_split/loop96_full_queue_verdict_import.json"))
    parser.add_argument("--loop97-speakeasy", type=Path, default=Path("reports/random_20w_split/loop97_speakeasy_triage_decision.json"))
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_audit(
        loop79_current_state=args.loop79_current_state,
        loop80_calibrator_fulltest=args.loop80_calibrator_fulltest,
        loop85_noise_strategy=args.loop85_noise_strategy,
        loop95_intake=args.loop95_intake,
        loop96_blinded_review=args.loop96_blinded_review,
        loop96_verdict_import=args.loop96_verdict_import,
        loop97_speakeasy=args.loop97_speakeasy,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
