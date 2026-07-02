#!/usr/bin/env python3
"""Build a compact review batch from Loop63 A-lane persistent conflicts.

The batch is for manual or external-evidence adjudication only. It does not
train, tune thresholds, relabel automatically, or mutate the split.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


ALLOWED_VERDICTS = "label_correct|label_wrong|feature_broken|out_of_scope|uncertain"
ALLOWED_ACTIONS = "keep_sample|replace_with_fresh_same_label_candidate|quarantine_for_more_evidence|model_blindspot"
REPLACEMENT_RULE = (
    "If feature_broken/out_of_scope/label_wrong is confirmed, do not fill from this row. "
    "Re-sample one fresh valid candidate from the same original-label pool and preserve the exact 200000-row split."
)

FIELDNAMES = [
    "review_batch_rank",
    "review_category",
    "review_priority_rank",
    "review_lane",
    "priority_reason",
    "loop57_error_type",
    "label",
    "loop57_final_prob",
    "loop57_base_prob",
    "loop57_candidate_prob",
    "loop57_gate_prob",
    "loop39_review_lane",
    "loop39_conflict_bucket",
    "loop39_corrected_by_any_compared_model",
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
]


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(str(row.get(key, "")).strip())
    except (TypeError, ValueError):
        return default


def _int(row: dict, key: str, default: int = 999999) -> int:
    try:
        return int(float(str(row.get(key, "")).strip()))
    except (TypeError, ValueError):
        return default


def _key(row: dict) -> str:
    return str(row.get("source_sha256") or row.get("source_path") or row.get("sample_index") or "").casefold()


def _row_keys(row: dict) -> list[str]:
    keys = [
        row.get("source_sha256"),
        row.get("split_source_sha256"),
        row.get("manifest_source_sha256"),
        row.get("source_path"),
        row.get("sample_index"),
    ]
    normalized = []
    seen = set()
    for value in keys:
        key = str(value or "").casefold()
        if not key or key in seen:
            continue
        normalized.append(key)
        seen.add(key)
    return normalized


def _audit_by_key(rows: Sequence[dict]) -> dict[str, dict]:
    return {_key(row): row for row in rows if _key(row)}


def _duplicate_info(rows: Sequence[dict]) -> dict[str, dict]:
    info: dict[str, dict] = {}
    for row in rows:
        payload = {
            "duplicate_manifest_sha_group": "true",
            "manifest_duplicate_group_id": row.get("duplicate_group_id", ""),
            "manifest_duplicate_group_size": row.get("group_size", ""),
            "manifest_duplicate_group_focus_rows": row.get("focus_queue_rows", ""),
        }
        for key in _row_keys(row):
            info[key] = payload
    return info


def _lookup_by_keys(mapping: dict[str, dict], row: dict) -> dict:
    for key in _row_keys(row):
        item = mapping.get(key)
        if item is not None:
            return item
    return {}


def _sort_severity(row: dict) -> tuple:
    error_type = row.get("loop57_error_type", "")
    prob = _float(row, "loop57_final_prob")
    severity = 1.0 - prob if error_type == "FN" else prob
    return (
        _int(row, "review_priority_rank"),
        0 if error_type == "FN" else 1,
        -severity,
        row.get("source_sha256", ""),
    )


def _select_category(
    *,
    selected: list[dict],
    seen: set[str],
    rows: Sequence[dict],
    category: str,
    limit: int,
) -> None:
    count = 0
    for row in rows:
        if count >= limit:
            return
        key = _key(row)
        if key and key in seen:
            continue
        item = dict(row)
        item["review_category"] = category
        selected.append(item)
        if key:
            seen.add(key)
        count += 1


def _normalized_output_row(row: dict, audit: dict, dup: dict) -> dict:
    output = {field: "" for field in FIELDNAMES}
    output.update({field: row.get(field, "") for field in FIELDNAMES})
    output.update({field: audit.get(field, "") for field in FIELDNAMES if field in audit})
    output.update(dup)
    output["duplicate_manifest_sha_group"] = dup.get("duplicate_manifest_sha_group", "false")
    output["allowed_manual_label_verdicts"] = ALLOWED_VERDICTS
    output["allowed_recommended_actions"] = ALLOWED_ACTIONS
    output["replacement_rule"] = REPLACEMENT_RULE
    output["manual_label_verdict"] = ""
    output["manual_verdict_note"] = ""
    output["recommended_action"] = ""
    return output


def build_review_batch(
    *,
    queue_csv: Path,
    health_audit_csv: Path,
    duplicate_details_csv: Path,
    output_csv: Path,
    output_json: Path,
    severe_fn_count: int,
    severe_fp_count: int,
    duplicate_group_count: int,
    corrected_by_other_count: int,
) -> dict:
    queue_rows = [row for row in read_rows(queue_csv) if row.get("review_lane") == "A_persistent_error_in_high_conflict_queue"]
    health_by_key = _audit_by_key(read_rows(health_audit_csv))
    duplicate_by_key = _duplicate_info(read_rows(duplicate_details_csv)) if duplicate_details_csv.exists() else {}
    duplicate_group_ids = sorted({info.get("manifest_duplicate_group_id", "") for info in duplicate_by_key.values() if info})

    selected: list[dict] = []
    seen: set[str] = set()

    severe_fn = sorted(
        [row for row in queue_rows if row.get("priority_reason") == "severe_fn_prob_le_0.01"],
        key=_sort_severity,
    )
    _select_category(
        selected=selected,
        seen=seen,
        rows=severe_fn,
        category="a_severe_persistent_fn",
        limit=severe_fn_count,
    )

    severe_fp = sorted(
        [row for row in queue_rows if row.get("priority_reason") == "severe_fp_prob_ge_0.99"],
        key=_sort_severity,
    )
    _select_category(
        selected=selected,
        seen=seen,
        rows=severe_fp,
        category="b_severe_persistent_fp",
        limit=severe_fp_count,
    )

    duplicate_rows = []
    for group_id in duplicate_group_ids[: max(0, int(duplicate_group_count))]:
        group_rows = [
            row
            for row in queue_rows
            if _lookup_by_keys(duplicate_by_key, row).get("manifest_duplicate_group_id") == group_id
        ]
        duplicate_rows.extend(sorted(group_rows, key=_sort_severity))
    _select_category(
        selected=selected,
        seen=seen,
        rows=duplicate_rows,
        category="c_duplicate_content_group",
        limit=len(duplicate_rows),
    )

    corrected_by_other = sorted(
        [
            row
            for row in queue_rows
            if str(row.get("loop39_corrected_by_any_compared_model", "")).casefold() == "true"
        ],
        key=_sort_severity,
    )
    _select_category(
        selected=selected,
        seen=seen,
        rows=corrected_by_other,
        category="d_corrected_by_other_model",
        limit=corrected_by_other_count,
    )

    output_rows = []
    for rank, row in enumerate(selected, start=1):
        key = _key(row)
        item = _normalized_output_row(row, health_by_key.get(key, {}), _lookup_by_keys(duplicate_by_key, row))
        item["review_batch_rank"] = str(rank)
        output_rows.append(item)

    write_rows(output_csv, output_rows)
    selected_duplicate_rows = [
        row for row in output_rows if str(row.get("duplicate_manifest_sha_group", "")).casefold() == "true"
    ]
    summary = {
        "schema": "axon_loop65_review_batch_v1",
        "protocol": "manual/external-evidence review batch only; no training, threshold tuning, automatic relabeling, or split mutation",
        "identity_feature_policy": (
            "source_path/source_sha256/cache_path/sample_index/split are review and cache alignment fields only; "
            "they are not model evidence"
        ),
        "queue_csv": str(queue_csv),
        "health_audit_csv": str(health_audit_csv),
        "duplicate_details_csv": str(duplicate_details_csv),
        "selected_rows": len(output_rows),
        "category_counts": dict(sorted(Counter(row["review_category"] for row in output_rows).items())),
        "error_type_counts": dict(sorted(Counter(row["loop57_error_type"] for row in output_rows).items())),
        "requested_duplicate_group_count": max(0, int(duplicate_group_count)),
        "requested_duplicate_group_rows_in_queue": len(duplicate_rows),
        "selected_duplicate_group_rows": len(selected_duplicate_rows),
        "selected_duplicate_group_category_counts": dict(
            sorted(Counter(row["review_category"] for row in selected_duplicate_rows).items())
        ),
        "manual_fields_blank_output": all(
            not row.get("manual_label_verdict") and not row.get("manual_verdict_note") and not row.get("recommended_action")
            for row in output_rows
        ),
        "replacement_rule": REPLACEMENT_RULE,
        "outputs": {
            "review_csv": str(output_csv),
            "summary_json": str(output_json),
        },
        "examples": output_rows[:20],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build Loop65 compact review batch from Loop63 A-lane rows.")
    parser.add_argument("--queue-csv", type=Path, required=True)
    parser.add_argument("--health-audit-csv", type=Path, required=True)
    parser.add_argument("--duplicate-details-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--severe-fn-count", type=int, default=20)
    parser.add_argument("--severe-fp-count", type=int, default=20)
    parser.add_argument("--duplicate-group-count", type=int, default=2)
    parser.add_argument("--corrected-by-other-count", type=int, default=20)
    args = parser.parse_args(argv)
    summary = build_review_batch(
        queue_csv=args.queue_csv,
        health_audit_csv=args.health_audit_csv,
        duplicate_details_csv=args.duplicate_details_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        severe_fn_count=args.severe_fn_count,
        severe_fp_count=args.severe_fp_count,
        duplicate_group_count=args.duplicate_group_count,
        corrected_by_other_count=args.corrected_by_other_count,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
