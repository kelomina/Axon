#!/usr/bin/env python3
"""Build a read-only coverage gate for full-error evidence review."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence


IDENTITY_FEATURE_POLICY = (
    "source_path/cache_path/source_sha256/sample_index/split/review rank/model score columns are loading, "
    "alignment, priority, cache-audit, duplicate-review, and manual-review fields only; they are not model "
    "evidence, verdict evidence, threshold inputs, or replacement sampling keys"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _key(row: dict[str, Any]) -> str:
    sample_index = str(row.get("sample_index", "")).strip()
    if sample_index:
        return f"sample_index:{sample_index}"
    sha = str(row.get("source_sha256", "")).strip().casefold()
    if sha:
        return f"sha:{sha}"
    path = str(row.get("source_path", "")).strip().casefold()
    return f"path:{path}" if path else ""


def key_set(rows: Sequence[dict[str, Any]]) -> set[str]:
    return {key for row in rows if (key := _key(row))}


def summarize_wave_plan(wave_rows: Sequence[dict[str, str]], target_min_fixed: int) -> dict[str, Any]:
    by_wave: dict[int, list[dict[str, str]]] = {}
    for row in wave_rows:
        wave_id = _int(row.get("review_wave_id"))
        by_wave.setdefault(wave_id, []).append(row)
    wave_sizes = {str(wave_id): len(rows) for wave_id, rows in sorted(by_wave.items())}
    first_wave_reaching_target = None
    cumulative_rows = 0
    for wave_id in sorted(by_wave):
        cumulative_rows += len(by_wave[wave_id])
        if target_min_fixed and cumulative_rows >= target_min_fixed and first_wave_reaching_target is None:
            first_wave_reaching_target = wave_id
    return {
        "wave_count": len(by_wave),
        "wave_sizes": wave_sizes,
        "first_wave_covering_target_gap_by_row_count": first_wave_reaching_target,
    }


def build_coverage_report(
    *,
    queue_csv: Path,
    target_gap_json: Path,
    loop72_wave_csv: Path,
    loop86_summary_json: Path,
    loop87_import_json: Path,
    loop63_health_summary_json: Path,
    loop64_duplicate_summary_json: Path,
    output_json: Path,
) -> dict[str, Any]:
    queue_rows = read_rows(queue_csv)
    wave_rows = read_rows(loop72_wave_csv)
    target_gap = read_json(target_gap_json)
    loop86 = read_json(loop86_summary_json)
    loop87 = read_json(loop87_import_json)
    health = read_json(loop63_health_summary_json)
    duplicate = read_json(loop64_duplicate_summary_json)

    queue_keys = key_set(queue_rows)
    wave_keys = key_set(wave_rows)
    queue_error_type_counts = Counter(row.get("loop57_error_type", "") for row in queue_rows)
    queue_lane_counts = Counter(row.get("review_lane", "") for row in queue_rows)
    queue_priority_counts = Counter(row.get("priority_reason", "") for row in queue_rows)
    current = target_gap.get("current_best", {})
    target_min_fixed = _int(
        target_gap.get("target_gap_best_case", {}).get(
            "minimum_fixed_errors_best_case",
            target_gap.get("error_reduction_needed_best_case", 0),
        )
    )
    loop86_rows = _int(loop86.get("rows"))
    loop87_rows = _int(loop87.get("rows"))
    loop87_actionable = _int(loop87.get("actionable_rows"))
    queue_rows_count = len(queue_rows)
    evidence_coverage_ratio = (loop86_rows / queue_rows_count) if queue_rows_count else 0.0
    target_gap_coverage_ratio = (loop86_rows / target_min_fixed) if target_min_fixed else 0.0
    remaining_rows_without_loop86_package = max(queue_rows_count - loop86_rows, 0)
    remaining_target_gap_after_loop86_package = max(target_min_fixed - loop86_rows, 0)

    blockers = []
    if queue_rows_count != _int(current.get("errors")):
        blockers.append("queue_rows_do_not_match_current_best_errors")
    if queue_keys and wave_keys and queue_keys != wave_keys:
        blockers.append("loop72_wave_plan_does_not_cover_same_queue_rows")
    if loop86_rows <= 0:
        blockers.append("loop86_evidence_package_missing_or_empty")
    if not bool(loop87.get("import_ready", False)):
        blockers.append("loop87_import_gate_not_ready")

    report = {
        "schema": "axon_loop88_full_error_evidence_coverage_v1",
        "protocol": (
            "read-only full-error evidence coverage gate; no model fitting, no threshold selection, no automatic "
            "relabeling, no split/cache mutation, no Test-derived feature engineering"
        ),
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "inputs": {
            "queue_csv": str(queue_csv),
            "target_gap_json": str(target_gap_json),
            "loop72_wave_csv": str(loop72_wave_csv),
            "loop86_summary_json": str(loop86_summary_json),
            "loop87_import_json": str(loop87_import_json),
            "loop63_health_summary_json": str(loop63_health_summary_json),
            "loop64_duplicate_summary_json": str(loop64_duplicate_summary_json),
        },
        "blockers": blockers,
        "current_best": {
            "f1": float(current.get("f1", 0.0) or 0.0),
            "errors": _int(current.get("errors")),
            "fp": _int(current.get("fp")),
            "fn": _int(current.get("fn")),
        },
        "target_gap": {
            "target_f1": float(target_gap.get("target_f1", 0.999) or 0.999),
            "minimum_fixed_errors_best_case": target_min_fixed,
            "required_error_reduction_ratio": float(
                target_gap.get("error_reduction_needed_ratio_of_current_errors", 0.0) or 0.0
            ),
        },
        "queue_coverage": {
            "queue_rows": queue_rows_count,
            "unique_queue_keys": len(queue_keys),
            "loop72_wave_rows": len(wave_rows),
            "unique_wave_keys": len(wave_keys),
            "loop72_covers_queue_keys": queue_keys == wave_keys if queue_keys and wave_keys else False,
            "error_type_counts": dict(sorted(queue_error_type_counts.items())),
            "review_lane_counts": dict(sorted(queue_lane_counts.items())),
            "priority_reason_counts": dict(sorted(queue_priority_counts.items())),
        },
        "evidence_package_coverage": {
            "loop86_rows": loop86_rows,
            "loop86_source_exists_count": _int(loop86.get("source_exists_count")),
            "loop86_cache_exists_count": _int(loop86.get("cache_exists_count")),
            "loop86_source_sha256_mismatch_count": _int(loop86.get("source_sha256_mismatch_count")),
            "loop86_pe_parse_status_counts": loop86.get("pe_parse_status_counts", {}),
            "coverage_of_queue_ratio": evidence_coverage_ratio,
            "coverage_of_target_gap_ratio": target_gap_coverage_ratio,
            "remaining_queue_rows_without_loop86_package": remaining_rows_without_loop86_package,
            "remaining_target_gap_after_loop86_package": remaining_target_gap_after_loop86_package,
        },
        "verdict_gate_status": {
            "loop87_rows": loop87_rows,
            "loop87_import_ready": bool(loop87.get("import_ready", False)),
            "loop87_decision": loop87.get("decision", ""),
            "loop87_blank_verdict_rows": loop87.get("manual_quality", {}).get("blank_verdict_rows"),
            "loop87_actionable_rows": loop87_actionable,
            "loop87_replacement_required_rows": loop87.get("replacement_required_rows"),
            "loop87_training_policy_rows": loop87.get("training_policy_rows"),
        },
        "existing_audits": {
            "loop63_A_lane_health_rows": _int(health.get("rows")),
            "loop63_A_lane_objective_issue_rows": _int(health.get("objective_issue_row_count")),
            "loop63_A_lane_issue_counts": health.get("issue_counts", {}),
            "loop64_duplicate_groups": _int(duplicate.get("duplicate_groups")),
            "loop64_cross_label_groups": _int(duplicate.get("cross_label_groups")),
            "loop64_cross_split_groups": _int(duplicate.get("cross_split_groups")),
            "loop64_focus_duplicate_detail_rows": _int(duplicate.get("focus_duplicate_detail_rows")),
        },
        "wave_plan": summarize_wave_plan(wave_rows, target_min_fixed),
        "decisions": {
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "next_allowed_step": (
                "Expand Loop86-style evidence packaging from the first 62 rows toward the full Loop72 wave plan; "
                "empty verdicts remain no-op, and confirmed bad rows still require strict Loop87-style ingress."
            ),
        },
        "recommendation": {
            "priority": "expand_evidence_package_coverage",
            "reason": (
                "The first Loop86 package covers only "
                f"{loop86_rows}/{queue_rows_count} current-best errors and "
                f"{loop86_rows}/{target_min_fixed} best-case target-gap fixes."
            ),
            "minimum_additional_rows_to_cover_target_gap_best_case": remaining_target_gap_after_loop86_package,
            "minimum_additional_rows_to_cover_all_current_errors": remaining_rows_without_loop86_package,
        },
    }
    write_json(output_json, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop88 full-error evidence coverage gate.")
    parser.add_argument("--queue-csv", type=Path, required=True)
    parser.add_argument("--target-gap-json", type=Path, required=True)
    parser.add_argument("--loop72-wave-csv", type=Path, required=True)
    parser.add_argument("--loop86-summary-json", type=Path, required=True)
    parser.add_argument("--loop87-import-json", type=Path, required=True)
    parser.add_argument("--loop63-health-summary-json", type=Path, required=True)
    parser.add_argument("--loop64-duplicate-summary-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_coverage_report(
        queue_csv=args.queue_csv,
        target_gap_json=args.target_gap_json,
        loop72_wave_csv=args.loop72_wave_csv,
        loop86_summary_json=args.loop86_summary_json,
        loop87_import_json=args.loop87_import_json,
        loop63_health_summary_json=args.loop63_health_summary_json,
        loop64_duplicate_summary_json=args.loop64_duplicate_summary_json,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
