#!/usr/bin/env python3
"""Summarize multi-wave evidence package coverage."""

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


def _merge_counts(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = target.get(str(key), 0) + _int(value)


def _parse_wave_spec(spec: str) -> tuple[int, Path, Path]:
    parts = spec.split("=", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid --wave spec, expected WAVE=EVIDENCE_JSON,VERDICT_JSON: {spec}")
    wave_id = _int(parts[0], -1)
    paths = [Path(item.strip()) for item in parts[1].split(",") if item.strip()]
    if wave_id <= 0 or len(paths) != 2:
        raise ValueError(f"Invalid --wave spec, expected WAVE=EVIDENCE_JSON,VERDICT_JSON: {spec}")
    return wave_id, paths[0], paths[1]


IDENTITY_FEATURE_POLICY = (
    "filename/path/extension/directory/source_sha256/cache_path/sample_index/split/row order are loading, "
    "alignment, cache-audit, duplicate-review, and manual-index fields only; they are not model evidence, "
    "verdict evidence, replacement sampling keys, or threshold/fusion inputs"
)


def build_summary(
    *,
    loop72_summary_json: Path,
    loop88_coverage_json: Path,
    waves: Sequence[tuple[int, Path, Path]],
    output_json: Path,
) -> dict[str, Any]:
    loop72 = read_json(loop72_summary_json)
    loop88 = read_json(loop88_coverage_json)
    wave_summaries = {
        _int(item.get("review_wave_id")): item
        for item in loop72.get("wave_summaries", [])
    }
    queue_rows = _int(loop88.get("queue_coverage", {}).get("queue_rows"))
    target_gap = _int(loop88.get("target_gap", {}).get("minimum_fixed_errors_best_case"))

    blockers = []
    seen_waves = set()
    total_rows = 0
    total_blank = 0
    total_actionable = 0
    total_replacement = 0
    total_training_policy = 0
    error_type_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    wave_reports = []

    for wave_id, evidence_path, verdict_path in sorted(waves, key=lambda item: item[0]):
        if wave_id in seen_waves:
            blockers.append(f"duplicate_wave_id:{wave_id}")
        seen_waves.add(wave_id)
        evidence = read_json(evidence_path)
        verdict = read_json(verdict_path)
        expected_rows = _int(wave_summaries.get(wave_id, {}).get("rows"))
        evidence_rows = _int(evidence.get("rows"))
        verdict_rows = _int(verdict.get("rows"))
        if evidence_rows != expected_rows:
            blockers.append(f"wave{wave_id}_evidence_rows_do_not_match_loop72")
        if verdict_rows != evidence_rows:
            blockers.append(f"wave{wave_id}_verdict_rows_do_not_match_evidence")
        if not bool(verdict.get("import_ready", False)):
            blockers.append(f"wave{wave_id}_verdict_import_not_ready")

        blank = _int(verdict.get("manual_quality", {}).get("blank_verdict_rows"))
        actionable = _int(verdict.get("actionable_rows"))
        replacement = _int(verdict.get("replacement_required_rows"))
        training_policy = _int(verdict.get("training_policy_rows"))
        total_rows += evidence_rows
        total_blank += blank
        total_actionable += actionable
        total_replacement += replacement
        total_training_policy += training_policy
        _merge_counts(error_type_counts, evidence.get("error_type_counts", {}))
        _merge_counts(category_counts, evidence.get("category_counts", {}))
        _merge_counts(tag_counts, evidence.get("review_tag_counts", {}))
        wave_reports.append(
            {
                "wave_id": wave_id,
                "rows": evidence_rows,
                "loop72_wave_rows": expected_rows,
                "error_type_counts": evidence.get("error_type_counts", {}),
                "category_counts": evidence.get("category_counts", {}),
                "source_exists_count": _int(evidence.get("source_exists_count")),
                "cache_exists_count": _int(evidence.get("cache_exists_count")),
                "source_sha256_mismatch_count": _int(evidence.get("source_sha256_mismatch_count")),
                "pe_parse_status_counts": evidence.get("pe_parse_status_counts", {}),
                "verdict_decision": verdict.get("decision", ""),
                "blank_verdict_rows": blank,
                "actionable_rows": actionable,
                "replacement_required_rows": replacement,
                "training_policy_rows": training_policy,
            }
        )

    summary = {
        "schema": "axon_loop90_multiwave_evidence_summary_v1",
        "protocol": (
            "read-only multi-wave evidence coverage summary; no model fitting, no threshold selection, no automatic "
            "relabeling, no split/cache mutation"
        ),
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "inputs": {
            "loop72_summary_json": str(loop72_summary_json),
            "loop88_coverage_json": str(loop88_coverage_json),
            "waves": [
                {"wave_id": wave_id, "evidence_json": str(evidence), "verdict_json": str(verdict)}
                for wave_id, evidence, verdict in sorted(waves, key=lambda item: item[0])
            ],
        },
        "blockers": blockers,
        "covered_waves": sorted(seen_waves),
        "wave_reports": wave_reports,
        "combined": {
            "rows": total_rows,
            "queue_rows": queue_rows,
            "target_gap_minimum_fixed_errors_best_case": target_gap,
            "coverage_of_queue_ratio": total_rows / queue_rows if queue_rows else 0.0,
            "coverage_of_target_gap_ratio": total_rows / target_gap if target_gap else 0.0,
            "remaining_queue_rows_without_evidence_package": max(queue_rows - total_rows, 0),
            "remaining_target_gap_rows_without_evidence_package": max(target_gap - total_rows, 0),
            "error_type_counts": dict(sorted(error_type_counts.items())),
            "category_counts": dict(sorted(category_counts.items())),
            "review_tag_counts": dict(sorted(tag_counts.items())),
            "blank_verdict_rows": total_blank,
            "actionable_rows": total_actionable,
            "replacement_required_rows": total_replacement,
            "training_policy_rows": total_training_policy,
        },
        "decisions": {
            "automatic_relabel_allowed": False,
            "automatic_replacement_allowed": False,
            "training_allowed": False,
            "test10k_allowed": False,
            "next_allowed_step": (
                "Continue packaging the next Loop72 wave or import external/manual verdicts through Loop87. "
                "Empty verdicts remain no-op."
            ),
        },
    }
    write_json(output_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop90 multi-wave evidence summary.")
    parser.add_argument("--loop72-summary-json", type=Path, required=True)
    parser.add_argument("--loop88-coverage-json", type=Path, required=True)
    parser.add_argument(
        "--wave",
        action="append",
        default=[],
        help="Wave spec in the form WAVE_ID=evidence_summary.json,verdict_import.json",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_summary(
        loop72_summary_json=args.loop72_summary_json,
        loop88_coverage_json=args.loop88_coverage_json,
        waves=[_parse_wave_spec(spec) for spec in args.wave],
        output_json=args.output_json,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
