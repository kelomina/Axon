#!/usr/bin/env python3
"""Build a read-only Loop85 strategy gate from existing noise/fusion evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_field(payload: dict[str, Any], key: str, *, source: Path) -> Any:
    if key not in payload:
        raise KeyError(f"{source} missing required field: {key}")
    return payload[key]


def build_summary(
    *,
    loop57_full_eval: Path,
    loop63_queue_summary: Path,
    loop63_health_summary: Path,
    loop64_duplicate_summary: Path,
    loop65_review_summary: Path,
    loop82_complementarity: Path,
    loop83_rescue_profile: Path,
    loop84_content_rescue: Path,
) -> dict[str, Any]:
    loop57 = read_json(loop57_full_eval)
    loop63 = read_json(loop63_queue_summary)
    health = read_json(loop63_health_summary)
    dup = read_json(loop64_duplicate_summary)
    loop65 = read_json(loop65_review_summary)
    loop82 = read_json(loop82_complementarity)
    loop83 = read_json(loop83_rescue_profile)
    loop84 = read_json(loop84_content_rescue)

    loop57_metrics = require_field(loop57, "metrics", source=loop57_full_eval)
    loop57_records = require_field(loop57, "records", source=loop57_full_eval)
    loop63_errors = int(require_field(loop63, "loop57_error_rows", source=loop63_queue_summary))
    loop57_errors = int(require_field(loop57_metrics, "errors", source=loop57_full_eval))

    blockers = []
    if int(loop57_records.get("kept", -1)) != 160000:
        blockers.append("Loop57 full-test did not keep exactly 160000 rows")
    if loop57_errors != loop63_errors:
        blockers.append("Loop63 queue does not cover every Loop57 full-test error")
    if int(health.get("objective_issue_row_count", -1)) != 0:
        blockers.append("A-lane health audit found objective cache/source issues")
    if int(dup.get("cross_label_groups", -1)) != 0:
        blockers.append("Manifest duplicate audit found cross-label duplicate groups")
    if int(dup.get("cross_split_groups", -1)) != 0:
        blockers.append("Manifest duplicate audit found cross-split duplicate groups")
    if not bool(loop65.get("manual_fields_blank_output", False)):
        blockers.append("Loop65 review batch already contains manual verdicts; regenerate strategy after adjudication")

    fusion_stop_reasons = []
    loop82_ready = bool(loop82.get("ready_for_val_fusion_probe", False))
    if not loop82_ready:
        fusion_stop_reasons.append("Loop82 did not clear same-manifest Val alignment")
    if bool(loop83.get("rule_scan", {}).get("improves_loop57", True)):
        fusion_stop_reasons.append("Loop83 score-delta result unexpectedly improved Loop57; review before stopping fusion")
    if bool(loop84.get("interpretation", {}).get("is_promising", True)):
        fusion_stop_reasons.append("Loop84 content selector unexpectedly passed; review before stopping fusion")

    automatic_replacement_allowed = False
    if health.get("objective_issue_row_count", 0) or dup.get("cross_label_groups", 0) or dup.get("cross_split_groups", 0):
        automatic_replacement_allowed = False

    next_actions = [
        {
            "priority": 1,
            "action": "manual_or_external_evidence_review",
            "input": str(loop65.get("outputs", {}).get("review_csv", "")),
            "rows": int(loop65.get("selected_rows", 0)),
            "reason": "Highest-priority compact batch already prepared; no automatic label/cache mutation is justified.",
        },
        {
            "priority": 2,
            "action": "expand_persistent_error_review_if_capacity_allows",
            "input": str(loop63.get("outputs", {}).get("queue_csv", "")),
            "rows": loop63_errors,
            "reason": "All Loop57 full-test errors are queued with priority lanes and blank adjudication fields.",
        },
        {
            "priority": 3,
            "action": "new_evidence_source_before_more_fusion",
            "input": "new external/dynamic/content evidence",
            "rows": None,
            "reason": "Loop83 and Loop84 reject current calibrator fusion selectors on Val.",
        },
    ]

    target_gap = {
        "loop57_full_test_errors": loop57_errors,
        "f1_target_error_budget_approx": 160,
        "minimum_error_reduction_needed_approx": max(loop57_errors - 160, 0),
    }

    return {
        "schema": "axon_loop85_noise_strategy_gate_v1",
        "protocol": "read-only strategy gate; no training, no threshold tuning, no relabeling, no split/cache mutation",
        "inputs": {
            "loop57_full_eval": str(loop57_full_eval),
            "loop63_queue_summary": str(loop63_queue_summary),
            "loop63_health_summary": str(loop63_health_summary),
            "loop64_duplicate_summary": str(loop64_duplicate_summary),
            "loop65_review_summary": str(loop65_review_summary),
            "loop82_complementarity": str(loop82_complementarity),
            "loop83_rescue_profile": str(loop83_rescue_profile),
            "loop84_content_rescue": str(loop84_content_rescue),
        },
        "blockers": blockers,
        "current_best": {
            "model": "Loop57 FN overlay gate",
            "full_test_rows": int(loop57_records.get("kept", 0)),
            "f1": float(loop57_metrics.get("f1", 0.0)),
            "errors": loop57_errors,
            "false_positive": int(loop57_metrics.get("false_positive", 0)),
            "false_negative": int(loop57_metrics.get("false_negative", 0)),
        },
        "target_gap": target_gap,
        "noise_evidence": {
            "loop63_queue_covers_errors": loop57_errors == loop63_errors,
            "loop63_review_lane_counts": loop63.get("review_lane_counts", {}),
            "loop63_error_type_counts": loop63.get("error_type_counts", {}),
            "loop63_manual_blank": {
                "manual_label_verdict_blank_count": loop63.get("manual_label_verdict_blank_count"),
                "recommended_action_blank_count": loop63.get("recommended_action_blank_count"),
            },
            "A_lane_health": {
                "rows": health.get("rows"),
                "error_type_counts": health.get("error_type_counts", {}),
                "objective_issue_row_count": health.get("objective_issue_row_count"),
                "issue_counts": health.get("issue_counts", {}),
            },
            "manifest_duplicates": {
                "duplicate_groups": dup.get("duplicate_groups"),
                "duplicate_detail_rows": dup.get("duplicate_detail_rows"),
                "cross_label_groups": dup.get("cross_label_groups"),
                "cross_split_groups": dup.get("cross_split_groups"),
                "focus_duplicate_groups": dup.get("focus_duplicate_groups"),
                "focus_duplicate_detail_rows": dup.get("focus_duplicate_detail_rows"),
            },
            "review_batch": {
                "selected_rows": loop65.get("selected_rows"),
                "category_counts": loop65.get("category_counts", {}),
                "error_type_counts": loop65.get("error_type_counts", {}),
                "manual_fields_blank_output": loop65.get("manual_fields_blank_output"),
            },
        },
        "fusion_evidence": {
            "loop82_ready_for_val_fusion_probe": loop82_ready,
            "loop82_calibrator_only_correct": loop82.get("overlap_counts", {}).get("calibrator_only_correct"),
            "loop82_loop57_only_correct": loop82.get("overlap_counts", {}).get("loop57_only_correct"),
            "loop83_score_delta_improves_loop57": loop83.get("rule_scan", {}).get("improves_loop57"),
            "loop83_best_rule": loop83.get("rule_scan", {}).get("best"),
            "loop84_content_selector_promising": loop84.get("interpretation", {}).get("is_promising"),
            "loop84_best_selector": loop84.get("selector_cv", {}).get("best"),
            "stop_current_calibrator_fusion": not fusion_stop_reasons,
            "fusion_stop_review_reasons": fusion_stop_reasons,
        },
        "decisions": {
            "automatic_replacement_allowed": automatic_replacement_allowed,
            "automatic_relabel_allowed": False,
            "test10k_allowed_for_current_calibrator_fusion": False,
            "next_phase": "manual/external-evidence noise review before more fusion",
            "replacement_rule": (
                "If label_wrong, feature_broken, or out_of_scope is confirmed, quarantine the bad row and "
                "fresh-redraw one valid sample from the same locked-manifest original-label pool. Do not use bad "
                "samples to fill counts, and do not choose replacements by filename/path/directory similarity."
            ),
        },
        "next_actions": next_actions,
        "identity_feature_policy": {
            "forbidden_as_model_evidence": [
                "filename",
                "path",
                "extension",
                "directory",
                "hash",
                "source_sha256",
                "sample_index",
                "split",
                "row_order",
            ],
            "allowed_uses": [
                "loading",
                "alignment",
                "cache audit",
                "duplicate detection",
                "manual/external review indexing",
            ],
            "label_source_boundary": (
                "Filename/path/directory inference is only a bootstrap method for building a one-time human-curated "
                "label manifest when no independent label list exists. In the 20w protocol, labels come from the "
                "locked split/manifest; identity fields must not be used as training, fusion, threshold, relabel, "
                "or production inference evidence."
            ),
            "redraw_boundary": (
                "same original-label pool means the locked manifest label pool, not a filename, extension, path, "
                "directory, hash, sample_index, split, or row-order bucket."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop85 noise strategy gate.")
    parser.add_argument("--loop57-full-eval", type=Path, required=True)
    parser.add_argument("--loop63-queue-summary", type=Path, required=True)
    parser.add_argument("--loop63-health-summary", type=Path, required=True)
    parser.add_argument("--loop64-duplicate-summary", type=Path, required=True)
    parser.add_argument("--loop65-review-summary", type=Path, required=True)
    parser.add_argument("--loop82-complementarity", type=Path, required=True)
    parser.add_argument("--loop83-rescue-profile", type=Path, required=True)
    parser.add_argument("--loop84-content-rescue", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_summary(
        loop57_full_eval=args.loop57_full_eval,
        loop63_queue_summary=args.loop63_queue_summary,
        loop63_health_summary=args.loop63_health_summary,
        loop64_duplicate_summary=args.loop64_duplicate_summary,
        loop65_review_summary=args.loop65_review_summary,
        loop82_complementarity=args.loop82_complementarity,
        loop83_rescue_profile=args.loop83_rescue_profile,
        loop84_content_rescue=args.loop84_content_rescue,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
