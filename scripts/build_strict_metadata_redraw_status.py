#!/usr/bin/env python3
"""Summarize strict metadata redraw readiness for the Val-first funnel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _bool(value: object) -> bool:
    return bool(value) if isinstance(value, bool) else str(value).strip().casefold() == "true"


def build_status(
    *,
    enrichment_json: Path,
    metadata_plan_json: Path,
    metadata_corrected_json: Path,
    metadata_replacement_audit_json: Path,
    first_cache_recovery_json: Path,
    cache_failure_plan_json: Path,
    cache_failure_corrected_json: Path,
    cache_failure_replacement_audit_json: Path,
    final_cache_ready_json: Path,
    final_split_metadata_json: Path,
) -> dict[str, Any]:
    enrichment = read_json(enrichment_json)
    metadata_plan = read_json(metadata_plan_json)
    metadata_corrected = read_json(metadata_corrected_json)
    metadata_replacement = read_json(metadata_replacement_audit_json)
    first_recovery = read_json(first_cache_recovery_json)
    cache_failure_plan = read_json(cache_failure_plan_json)
    cache_failure_corrected = read_json(cache_failure_corrected_json)
    cache_failure_replacement = read_json(cache_failure_replacement_audit_json)
    final_cache = read_json(final_cache_ready_json)
    final_meta = read_json(final_split_metadata_json)

    blockers: list[str] = []
    if _int(enrichment.get("rows")) != 200000:
        blockers.append("enrichment_rows_not_200000")
    if enrichment.get("shape_failures"):
        blockers.append("enrichment_shape_failures")
    if _int(enrichment.get("row_issue_count")) != _int(metadata_plan.get("plan_rows")):
        blockers.append("metadata_issue_plan_count_mismatch")
    if not _bool(metadata_plan.get("plan_ready")):
        blockers.append("metadata_replacement_plan_not_ready")
    if _int(metadata_corrected.get("excluded_rows")) != _int(metadata_plan.get("plan_rows")):
        blockers.append("metadata_corrected_excluded_count_mismatch")
    if not _bool(metadata_replacement.get("replacement_integrity_ok")):
        blockers.append("metadata_replacement_integrity_failed")
    if not _bool(metadata_replacement.get("label_balance_enforced")):
        blockers.append("metadata_replacement_label_balance_not_enforced")

    first_status_counts = dict(first_recovery.get("status_counts", {}))
    first_failed = sum(_int(count) for status, count in first_status_counts.items() if status not in {"extracted", "cache_hit"})
    if first_failed != _int(cache_failure_plan.get("plan_rows")):
        blockers.append("cache_failure_plan_count_mismatch")
    if first_failed and not _bool(cache_failure_plan.get("plan_ready")):
        blockers.append("cache_failure_plan_not_ready")
    if _int(cache_failure_corrected.get("excluded_rows")) != _int(cache_failure_plan.get("plan_rows")):
        blockers.append("cache_failure_corrected_excluded_count_mismatch")
    if not _bool(cache_failure_replacement.get("replacement_integrity_ok")):
        blockers.append("cache_failure_replacement_integrity_failed")
    if not _bool(cache_failure_replacement.get("label_balance_enforced")):
        blockers.append("cache_failure_replacement_label_balance_not_enforced")

    if not _bool(final_cache.get("cache_ready")):
        blockers.append("final_cache_not_ready")
    if not _bool(final_cache.get("cache_metadata_validation_enabled")):
        blockers.append("final_cache_metadata_validation_not_enabled")
    if _int(final_cache.get("total_rows")) != 200000 or _int(final_cache.get("covered_rows")) != 200000:
        blockers.append("final_cache_coverage_not_200000")
    if _int(final_cache.get("missing_rows")) != 0:
        blockers.append("final_cache_missing_rows")
    if _int(final_cache.get("metadata_failure_rows")) != 0:
        blockers.append("final_cache_metadata_failures")
    if not _bool(final_cache.get("label_balance_enforced")):
        blockers.append("final_cache_label_balance_not_enforced")
    if final_cache.get("shape_failures"):
        blockers.append("final_cache_shape_failures")

    if not _bool(final_meta.get("audit_ready")):
        blockers.append("final_split_metadata_not_ready")
    if not _bool(final_meta.get("validate_npz")):
        blockers.append("final_split_metadata_npz_not_validated")
    if not _bool(final_meta.get("expect_20w")):
        blockers.append("final_split_metadata_20w_not_checked")
    if _int(final_meta.get("rows")) != 200000:
        blockers.append("final_split_metadata_rows_not_200000")
    if _int(final_meta.get("row_issue_count")) != 0:
        blockers.append("final_split_metadata_row_issues")
    if final_meta.get("metadata_issue_counts"):
        blockers.append("final_split_metadata_issue_counts")
    if final_meta.get("shape_failures"):
        blockers.append("final_split_metadata_shape_failures")

    ready = not blockers
    return {
        "schema": "axon_strict_metadata_redraw_status_v1",
        "decision": "ready_for_val_first_reverification" if ready else "blocked_strict_metadata_redraw",
        "blockers": blockers,
        "identity_feature_policy": (
            "source_path/path/name/extension/directory are loading and alignment fields only; "
            "source_sha256 is content identity only; no identity field is model evidence, verdict evidence, "
            "threshold evidence, or feature-selection evidence."
        ),
        "counts": {
            "initial_split_rows": _int(enrichment.get("rows")),
            "initial_metadata_issue_rows": _int(enrichment.get("row_issue_count")),
            "metadata_redraw_plan_rows": _int(metadata_plan.get("plan_rows")),
            "metadata_redraw_selected_replacements": _int(metadata_corrected.get("replacement_summary", {}).get("selected_replacements")),
            "first_cache_recovery_failed_rows": first_failed,
            "cache_failure_redraw_plan_rows": _int(cache_failure_plan.get("plan_rows")),
            "cache_failure_selected_replacements": _int(cache_failure_corrected.get("replacement_summary", {}).get("selected_replacements")),
            "final_cache_total_rows": _int(final_cache.get("total_rows")),
            "final_cache_covered_rows": _int(final_cache.get("covered_rows")),
            "final_metadata_rows": _int(final_meta.get("rows")),
        },
        "final_split_summary": final_cache.get("split_summary", {}),
        "final_cache": {
            "cache_ready": _bool(final_cache.get("cache_ready")),
            "covered_rows": _int(final_cache.get("covered_rows")),
            "missing_rows": _int(final_cache.get("missing_rows")),
            "metadata_failure_rows": _int(final_cache.get("metadata_failure_rows")),
            "manifest_match_counts": dict(final_cache.get("manifest_match_counts", {})),
        },
        "final_split_metadata": {
            "audit_ready": _bool(final_meta.get("audit_ready")),
            "rows": _int(final_meta.get("rows")),
            "match_counts": dict(final_meta.get("match_counts", {})),
            "metadata_issue_counts": dict(final_meta.get("metadata_issue_counts", {})),
            "row_issue_count": _int(final_meta.get("row_issue_count")),
        },
        "ready_for": {
            "train_val_only": ready,
            "test10k": False,
            "full_test": False,
        },
        "next_step": "restart_val_first_funnel" if ready else "fix_strict_metadata_redraw_blockers",
        "notes": [
            "This status authorizes only Train/Val reverification; Test-10k and full-test remain locked.",
            "Rows with manifest label conflicts were quarantined and replaced; feature extraction failures were redrawn again.",
            "The final split keeps exactly 200000 rows with 20000/20000/160000 split counts.",
        ],
        "inputs": {
            "enrichment_json": str(resolve_path(enrichment_json)),
            "metadata_plan_json": str(resolve_path(metadata_plan_json)),
            "metadata_corrected_json": str(resolve_path(metadata_corrected_json)),
            "metadata_replacement_audit_json": str(resolve_path(metadata_replacement_audit_json)),
            "first_cache_recovery_json": str(resolve_path(first_cache_recovery_json)),
            "cache_failure_plan_json": str(resolve_path(cache_failure_plan_json)),
            "cache_failure_corrected_json": str(resolve_path(cache_failure_corrected_json)),
            "cache_failure_replacement_audit_json": str(resolve_path(cache_failure_replacement_audit_json)),
            "final_cache_ready_json": str(resolve_path(final_cache_ready_json)),
            "final_split_metadata_json": str(resolve_path(final_split_metadata_json)),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize strict metadata redraw status.")
    parser.add_argument("--enrichment-json", type=Path, required=True)
    parser.add_argument("--metadata-plan-json", type=Path, required=True)
    parser.add_argument("--metadata-corrected-json", type=Path, required=True)
    parser.add_argument("--metadata-replacement-audit-json", type=Path, required=True)
    parser.add_argument("--first-cache-recovery-json", type=Path, required=True)
    parser.add_argument("--cache-failure-plan-json", type=Path, required=True)
    parser.add_argument("--cache-failure-corrected-json", type=Path, required=True)
    parser.add_argument("--cache-failure-replacement-audit-json", type=Path, required=True)
    parser.add_argument("--final-cache-ready-json", type=Path, required=True)
    parser.add_argument("--final-split-metadata-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_status(
        enrichment_json=args.enrichment_json,
        metadata_plan_json=args.metadata_plan_json,
        metadata_corrected_json=args.metadata_corrected_json,
        metadata_replacement_audit_json=args.metadata_replacement_audit_json,
        first_cache_recovery_json=args.first_cache_recovery_json,
        cache_failure_plan_json=args.cache_failure_plan_json,
        cache_failure_corrected_json=args.cache_failure_corrected_json,
        cache_failure_replacement_audit_json=args.cache_failure_replacement_audit_json,
        final_cache_ready_json=args.final_cache_ready_json,
        final_split_metadata_json=args.final_split_metadata_json,
    )
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["decision"] == "ready_for_val_first_reverification" or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
