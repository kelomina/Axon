#!/usr/bin/env python3
"""Summarize Loop89 Wave1 evidence-package expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def build_summary(
    *,
    loop72_summary_json: Path,
    loop88_coverage_json: Path,
    wave1_evidence_json: Path,
    wave1_verdict_json: Path,
    output_json: Path,
) -> dict[str, Any]:
    loop72 = read_json(loop72_summary_json)
    loop88 = read_json(loop88_coverage_json)
    evidence = read_json(wave1_evidence_json)
    verdict = read_json(wave1_verdict_json)

    queue_rows = _int(loop88.get("queue_coverage", {}).get("queue_rows"))
    target_gap = _int(loop88.get("target_gap", {}).get("minimum_fixed_errors_best_case"))
    evidence_rows = _int(evidence.get("rows"))
    coverage_queue_ratio = evidence_rows / queue_rows if queue_rows else 0.0
    coverage_target_ratio = evidence_rows / target_gap if target_gap else 0.0
    wave_summaries = loop72.get("wave_summaries", [])
    wave1_summary = next((row for row in wave_summaries if _int(row.get("review_wave_id")) == 1), {})

    blockers = []
    if evidence_rows != _int(wave1_summary.get("rows")):
        blockers.append("wave1_evidence_rows_do_not_match_loop72_wave1")
    if not bool(verdict.get("import_ready", False)):
        blockers.append("wave1_verdict_import_not_ready")
    if _int(verdict.get("replacement_required_rows")) != 0:
        blockers.append("wave1_has_replacement_requests_without_external_planning")

    summary = {
        "schema": "axon_loop89_wave1_evidence_summary_v1",
        "protocol": (
            "read-only Wave1 evidence expansion summary; no model fitting, no threshold selection, no automatic "
            "relabeling, no split/cache mutation"
        ),
        "inputs": {
            "loop72_summary_json": str(loop72_summary_json),
            "loop88_coverage_json": str(loop88_coverage_json),
            "wave1_evidence_json": str(wave1_evidence_json),
            "wave1_verdict_json": str(wave1_verdict_json),
        },
        "blockers": blockers,
        "wave1": {
            "rows": evidence_rows,
            "loop72_wave_rows": _int(wave1_summary.get("rows")),
            "error_type_counts": evidence.get("error_type_counts", {}),
            "category_counts": evidence.get("category_counts", {}),
            "source_exists_count": _int(evidence.get("source_exists_count")),
            "cache_exists_count": _int(evidence.get("cache_exists_count")),
            "source_sha256_mismatch_count": _int(evidence.get("source_sha256_mismatch_count")),
            "pe_parse_status_counts": evidence.get("pe_parse_status_counts", {}),
            "review_tag_counts": evidence.get("review_tag_counts", {}),
            "verdict_decision": verdict.get("decision", ""),
            "blank_verdict_rows": verdict.get("manual_quality", {}).get("blank_verdict_rows"),
            "actionable_rows": _int(verdict.get("actionable_rows")),
            "replacement_required_rows": _int(verdict.get("replacement_required_rows")),
            "training_policy_rows": _int(verdict.get("training_policy_rows")),
        },
        "coverage_after_wave1": {
            "queue_rows": queue_rows,
            "target_gap_minimum_fixed_errors_best_case": target_gap,
            "coverage_of_queue_ratio": coverage_queue_ratio,
            "coverage_of_target_gap_ratio": coverage_target_ratio,
            "remaining_queue_rows_without_evidence_package": max(queue_rows - evidence_rows, 0),
            "remaining_target_gap_rows_without_evidence_package": max(target_gap - evidence_rows, 0),
        },
        "decisions": {
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "next_allowed_step": (
                "Continue packaging Loop72 Wave2 or import external/manual verdicts for Wave1 through the same "
                "Loop87 gate. Empty verdicts remain no-op."
            ),
        },
    }
    write_json(output_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop89 Wave1 evidence summary.")
    parser.add_argument("--loop72-summary-json", type=Path, required=True)
    parser.add_argument("--loop88-coverage-json", type=Path, required=True)
    parser.add_argument("--wave1-evidence-json", type=Path, required=True)
    parser.add_argument("--wave1-verdict-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_summary(
        loop72_summary_json=args.loop72_summary_json,
        loop88_coverage_json=args.loop88_coverage_json,
        wave1_evidence_json=args.wave1_evidence_json,
        wave1_verdict_json=args.wave1_verdict_json,
        output_json=args.output_json,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
