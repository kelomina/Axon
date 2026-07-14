#!/usr/bin/env python3
"""Build a full review-wave plan for current-best full-test errors.

This is a planning artifact for manual or external-evidence adjudication. It
does not train, tune thresholds, relabel automatically, mutate splits, or turn
full-test identity fields into model rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence


ALLOWED_VERDICTS = "label_correct|label_wrong|feature_broken|out_of_scope|uncertain"
ALLOWED_ACTIONS = "keep_label|replace_sample|quarantine_source_group|needs_more_evidence|model_blindspot"
REPLACEMENT_RULE = (
    "Confirmed label_wrong/feature_broken/out_of_scope rows must trigger fresh same-original-label redraw. "
    "Do not fill from the bad rows, and preserve exactly 200000 split rows."
)
IDENTITY_FEATURE_POLICY = (
    "filename/path/extension/directory/source hash/cache_path/sample_index/split/row order are loading, "
    "alignment, cache-audit, duplicate-review, and manual-review fields only; they are not model evidence "
    "and must not drive thresholds, relabeling, feature engineering, or production inference"
)

FIELDNAMES = [
    "review_wave_id",
    "review_wave_rank",
    "global_review_rank",
    "cumulative_review_rows",
    "cumulative_fixed_fp_if_all_confirmed",
    "cumulative_fixed_fn_if_all_confirmed",
    "cumulative_f1_if_all_confirmed_fixed",
    "target_gap_coverage_ratio",
    "target_reached_if_all_confirmed_by_this_row",
    "review_category",
    "evidence_lane",
    "suggested_review_question",
    "review_lane",
    "priority_reason",
    "exchange_group",
    "loop39_review_lane",
    "loop39_conflict_bucket",
    "loop39_corrected_by_any_compared_model",
    "loop57_error_type",
    "label",
    "loop57_final_prob",
    "loop57_base_prob",
    "loop57_candidate_prob",
    "loop57_gate_prob",
    "loop57_prediction",
    "loop57_fn_override",
    "loop28_prob",
    "loop28_prediction",
    "loop28_correct",
    "duplicate_manifest_sha_group",
    "manifest_duplicate_group_id",
    "manifest_duplicate_group_size",
    "manifest_duplicate_group_focus_rows",
    "objective_issue_count",
    "objective_issue_flags",
    "pe_has_imports",
    "pe_has_exports",
    "pe_has_resources",
    "pe_has_security_directory",
    "pe_is_dll",
    "pe_overlay_size",
    "pe_sections",
    "source_path",
    "cache_path",
    "source_sha256",
    "sample_index",
    "split",
    "allowed_manual_label_verdicts",
    "allowed_recommended_actions",
    "replacement_rule",
    "manual_label_verdict",
    "manual_verdict_note",
    "recommended_action",
    "corrected_label",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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


def _row_keys(row: dict) -> list[str]:
    candidates = [
        row.get("source_sha256"),
        row.get("split_source_sha256"),
        row.get("manifest_source_sha256"),
        row.get("source_path"),
        row.get("sample_index"),
    ]
    keys = []
    seen = set()
    for value in candidates:
        key = str(value or "").strip().casefold()
        if key and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def _index_by_keys(rows: Sequence[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        for key in _row_keys(row):
            indexed.setdefault(key, row)
    return indexed


def _duplicate_by_key(rows: Sequence[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        payload = {
            "duplicate_manifest_sha_group": "true",
            "manifest_duplicate_group_id": row.get("duplicate_group_id", ""),
            "manifest_duplicate_group_size": row.get("group_size", ""),
            "manifest_duplicate_group_focus_rows": row.get("focus_queue_rows", ""),
        }
        for key in _row_keys(row):
            indexed[key] = payload
    return indexed


def _lookup(mapping: dict[str, dict], row: dict) -> dict:
    for key in _row_keys(row):
        found = mapping.get(key)
        if found is not None:
            return found
    return {}


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def _review_rank(row: dict) -> int:
    return _int(row.get("review_priority_rank"), 999999)


def _review_category(row: dict, health: dict, duplicate: dict) -> str:
    if _int(health.get("objective_issue_count")) > 0:
        return "a_objective_data_issue"
    if duplicate:
        return "b_duplicate_content_group"
    if row.get("review_lane") == "A_persistent_error_in_high_conflict_queue":
        return "c_high_conflict_persistent_error"
    if row.get("exchange_group") == "loop57_new_error":
        return "d_loop57_new_error"
    if row.get("loop57_error_type") == "FN":
        return "e_persistent_fn"
    return "f_persistent_fp"


def _evidence_lane(row: dict, health: dict, duplicate: dict) -> str:
    if _int(health.get("objective_issue_count")) > 0:
        return "objective_cache_or_feature_issue"
    if duplicate:
        return "same-content-group-review"
    if row.get("loop57_error_type") == "FN":
        return "external-maliciousness-confirmation"
    return "external-benign-provenance-confirmation"


def _suggested_question(row: dict) -> str:
    if row.get("loop57_error_type") == "FN":
        return (
            "Confirm whether the sample is truly malicious and in-scope using content, sandbox, vendor, "
            "or provenance evidence; do not use filename or directory naming."
        )
    return (
        "Confirm whether the sample is truly benign and in-scope using content, signature, vendor, "
        "or provenance evidence; do not use filename or directory naming."
    )


def _duplicate_group_id(row: dict, duplicate: dict) -> str:
    return str(duplicate.get("manifest_duplicate_group_id") or "")


def _build_ordered_units(rows: Sequence[dict], duplicate_lookup: dict[str, dict]) -> list[list[tuple[dict, dict]]]:
    sorted_rows = sorted(rows, key=_review_rank)
    group_to_rows: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for row in sorted_rows:
        duplicate = _lookup(duplicate_lookup, row)
        group_id = _duplicate_group_id(row, duplicate)
        if group_id:
            group_to_rows[group_id].append((row, duplicate))

    units: list[list[tuple[dict, dict]]] = []
    seen_row_keys: set[str] = set()
    seen_groups: set[str] = set()
    for row in sorted_rows:
        key = "|".join(_row_keys(row)) or str(id(row))
        if key in seen_row_keys:
            continue
        duplicate = _lookup(duplicate_lookup, row)
        group_id = _duplicate_group_id(row, duplicate)
        if group_id:
            if group_id in seen_groups:
                continue
            unit = group_to_rows[group_id]
            seen_groups.add(group_id)
        else:
            unit = [(row, duplicate)]
        for item, _ in unit:
            seen_row_keys.add("|".join(_row_keys(item)) or str(id(item)))
        units.append(unit)
    return units


def _assign_waves(units: Sequence[list[tuple[dict, dict]]], wave_size: int) -> list[tuple[int, dict, dict]]:
    assignments: list[tuple[int, dict, dict]] = []
    wave_id = 1
    wave_count = 0
    limit = max(1, int(wave_size))
    for unit in units:
        unit_size = len(unit)
        if wave_count > 0 and wave_count + unit_size > limit:
            wave_id += 1
            wave_count = 0
        for row, duplicate in unit:
            assignments.append((wave_id, row, duplicate))
            wave_count += 1
        if wave_count >= limit:
            wave_id += 1
            wave_count = 0
    return assignments


def _wave_summaries(rows: Sequence[dict]) -> list[dict[str, Any]]:
    summaries = []
    by_wave: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_wave[str(row["review_wave_id"])].append(row)
    for wave_id in sorted(by_wave, key=lambda item: int(item)):
        wave_rows = by_wave[wave_id]
        last = wave_rows[-1]
        summaries.append(
            {
                "review_wave_id": int(wave_id),
                "rows": len(wave_rows),
                "wave_error_type_counts": dict(sorted(Counter(row["loop57_error_type"] for row in wave_rows).items())),
                "wave_category_counts": dict(sorted(Counter(row["review_category"] for row in wave_rows).items())),
                "cumulative_review_rows": int(last["cumulative_review_rows"]),
                "cumulative_fixed_fp_if_all_confirmed": int(last["cumulative_fixed_fp_if_all_confirmed"]),
                "cumulative_fixed_fn_if_all_confirmed": int(last["cumulative_fixed_fn_if_all_confirmed"]),
                "cumulative_f1_if_all_confirmed_fixed": float(last["cumulative_f1_if_all_confirmed_fixed"]),
                "target_gap_coverage_ratio": float(last["target_gap_coverage_ratio"]),
                "target_reached_if_all_confirmed": last["target_reached_if_all_confirmed_by_this_row"] == "true",
            }
        )
    return summaries


def build_wave_plan(
    *,
    queue_csv: Path,
    target_gap_json: Path,
    output_csv: Path,
    output_json: Path,
    health_audit_csv: Optional[Path] = None,
    duplicate_details_csv: Optional[Path] = None,
    wave_size: int = 200,
) -> dict[str, Any]:
    queue_rows = read_rows(queue_csv)
    health_lookup = _index_by_keys(read_rows(health_audit_csv)) if health_audit_csv and health_audit_csv.exists() else {}
    duplicate_lookup = _duplicate_by_key(read_rows(duplicate_details_csv)) if duplicate_details_csv and duplicate_details_csv.exists() else {}
    target = read_json(target_gap_json)
    current = target["current_best"]
    target_f1 = float(target["target_f1"])
    target_min_fixed = int(target["error_reduction_needed_best_case"])
    current_tp = int(current["tp"])
    current_fp = int(current["fp"])
    current_fn = int(current["fn"])

    assignments = _assign_waves(_build_ordered_units(queue_rows, duplicate_lookup), wave_size)
    fixed_fp = 0
    fixed_fn = 0
    wave_ranks: Counter[int] = Counter()
    output_rows: list[dict] = []
    for global_rank, (wave_id, row, duplicate) in enumerate(assignments, start=1):
        wave_ranks[wave_id] += 1
        health = _lookup(health_lookup, row)
        if row.get("loop57_error_type") == "FP":
            fixed_fp += 1
        elif row.get("loop57_error_type") == "FN":
            fixed_fn += 1
        new_tp = current_tp + fixed_fn
        new_fp = max(0, current_fp - fixed_fp)
        new_fn = max(0, current_fn - fixed_fn)
        cumulative_f1 = _f1(new_tp, new_fp, new_fn)
        category = _review_category(row, health, duplicate)
        item = {field: "" for field in FIELDNAMES}
        item.update({field: row.get(field, "") for field in FIELDNAMES if field in row})
        item.update({field: health.get(field, "") for field in FIELDNAMES if field in health})
        item.update(duplicate)
        item["review_wave_id"] = str(wave_id)
        item["review_wave_rank"] = str(wave_ranks[wave_id])
        item["global_review_rank"] = str(global_rank)
        item["cumulative_review_rows"] = str(global_rank)
        item["cumulative_fixed_fp_if_all_confirmed"] = str(fixed_fp)
        item["cumulative_fixed_fn_if_all_confirmed"] = str(fixed_fn)
        item["cumulative_f1_if_all_confirmed_fixed"] = f"{cumulative_f1:.10f}"
        item["target_gap_coverage_ratio"] = f"{min(1.0, global_rank / target_min_fixed):.10f}" if target_min_fixed else "1.0000000000"
        item["target_reached_if_all_confirmed_by_this_row"] = "true" if cumulative_f1 >= target_f1 else "false"
        item["review_category"] = category
        item["evidence_lane"] = _evidence_lane(row, health, duplicate)
        item["suggested_review_question"] = _suggested_question(row)
        item["duplicate_manifest_sha_group"] = duplicate.get("duplicate_manifest_sha_group", "false")
        item["objective_issue_count"] = str(_int(health.get("objective_issue_count")))
        item["allowed_manual_label_verdicts"] = ALLOWED_VERDICTS
        item["allowed_recommended_actions"] = ALLOWED_ACTIONS
        item["replacement_rule"] = REPLACEMENT_RULE
        item["manual_label_verdict"] = ""
        item["manual_verdict_note"] = ""
        item["recommended_action"] = ""
        item["corrected_label"] = ""
        output_rows.append(item)

    write_rows(output_csv, output_rows)
    wave_summaries = _wave_summaries(output_rows)
    first_wave_reaching_target = next(
        (wave["review_wave_id"] for wave in wave_summaries if wave["target_reached_if_all_confirmed"]),
        None,
    )
    duplicate_group_ids = {
        row["manifest_duplicate_group_id"]
        for row in output_rows
        if str(row.get("duplicate_manifest_sha_group", "")).casefold() == "true" and row.get("manifest_duplicate_group_id")
    }
    summary = {
        "schema": "axon_loop72_review_wave_plan_v1",
        "protocol": (
            "manual/external-evidence review wave plan only; no model fitting, no threshold selection, "
            "no automatic relabeling, no split mutation, and no Test-derived feature engineering"
        ),
        "identity_feature_policy": IDENTITY_FEATURE_POLICY,
        "queue_csv": str(queue_csv),
        "target_gap_json": str(target_gap_json),
        "health_audit_csv": str(health_audit_csv) if health_audit_csv else "",
        "duplicate_details_csv": str(duplicate_details_csv) if duplicate_details_csv else "",
        "wave_size": max(1, int(wave_size)),
        "rows": len(output_rows),
        "wave_count": len(wave_summaries),
        "first_wave_reaching_target_if_all_actionable": first_wave_reaching_target,
        "target": {
            "target_f1": target_f1,
            "current_f1": float(current["f1"]),
            "current_errors": int(current["errors"]),
            "current_fp": current_fp,
            "current_fn": current_fn,
            "minimum_fixed_errors_best_case": target_min_fixed,
        },
        "full_plan_best_case": {
            "fixed_errors": len(output_rows),
            "fixed_fp": fixed_fp,
            "fixed_fn": fixed_fn,
            "f1": float(output_rows[-1]["cumulative_f1_if_all_confirmed_fixed"]) if output_rows else float(current["f1"]),
            "target_reached_if_all_confirmed": bool(first_wave_reaching_target is not None),
        },
        "review_lane_counts": dict(sorted(Counter(row["review_lane"] for row in output_rows).items())),
        "review_category_counts": dict(sorted(Counter(row["review_category"] for row in output_rows).items())),
        "error_type_counts": dict(sorted(Counter(row["loop57_error_type"] for row in output_rows).items())),
        "evidence_lane_counts": dict(sorted(Counter(row["evidence_lane"] for row in output_rows).items())),
        "duplicate_manifest_group_count": len(duplicate_group_ids),
        "duplicate_manifest_group_rows": sum(1 for row in output_rows if row["duplicate_manifest_sha_group"] == "true"),
        "manual_fields_blank_output": all(
            not row["manual_label_verdict"] and not row["manual_verdict_note"] and not row["recommended_action"] and not row["corrected_label"]
            for row in output_rows
        ),
        "replacement_rule": REPLACEMENT_RULE,
        "wave_summaries": wave_summaries,
        "decision": {
            "next_action": (
                "Run external-evidence adjudication wave by wave. Empty or uncertain verdicts remain no-op; "
                "confirmed bad rows require fresh same-original-label redraw before any corrected split is trained."
            ),
            "model_gate": "no_model_candidate_and_no_test_tuning",
        },
        "artifacts": {
            "review_wave_csv": str(output_csv),
            "summary_json": str(output_json),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Loop72 full-error review wave plan.")
    parser.add_argument("--queue-csv", type=Path, required=True)
    parser.add_argument("--target-gap-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--health-audit-csv", type=Path)
    parser.add_argument("--duplicate-details-csv", type=Path)
    parser.add_argument("--wave-size", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_wave_plan(
        queue_csv=args.queue_csv,
        target_gap_json=args.target_gap_json,
        health_audit_csv=args.health_audit_csv,
        duplicate_details_csv=args.duplicate_details_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        wave_size=args.wave_size,
    )
    print(
        json.dumps(
            {
                "rows": summary["rows"],
                "wave_count": summary["wave_count"],
                "first_wave_reaching_target_if_all_actionable": summary["first_wave_reaching_target_if_all_actionable"],
                "full_plan_best_case": summary["full_plan_best_case"],
                "artifacts": summary["artifacts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
