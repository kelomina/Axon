#!/usr/bin/env python3
"""Audit replacement integrity for a corrected split.

This is a post-build guard. It does not change the original split, corrected
split, or manual plan. It verifies that excluded rows disappeared and that the
replacement rows are fresh rows from outside the original split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOTAL = 200000
EXPECTED_SPLIT_COUNTS = {"train": 20000, "val": 20000, "test": 160000}
EXPECTED_LABEL_SPLIT_COUNTS = {
    "train": {"0": 10000, "1": 10000},
    "val": {"0": 10000, "1": 10000},
    "test": {"0": 80000, "1": 80000},
}

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from apply_manual_review_verdicts import source_keys  # noqa: E402


DETAIL_FIELDNAMES = [
    "record_type",
    "source_path",
    "label",
    "sample_index",
    "split",
    "status",
    "reason",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_detail_rows(path: Path, rows: Sequence[dict]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDNAMES, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def row_keys(row: dict) -> set[str]:
    keys = set(source_keys(row))
    source_path = str(row.get("source_path") or "").strip()
    if source_path:
        keys.add(f"path:{source_path.casefold()}")
    return keys


def exact_row_key(row: dict) -> str:
    return f"{str(row.get('sample_index', '')).strip()}|{str(row.get('source_path', '')).strip().casefold()}"


def has_exact_row_identity(row: dict) -> bool:
    return bool(str(row.get("sample_index", "")).strip() and str(row.get("source_path", "")).strip())


def build_exact_lookup(rows: Sequence[dict]) -> dict[str, dict]:
    return {exact_row_key(row): row for row in rows if has_exact_row_identity(row)}


def build_key_lookup(rows: Sequence[dict]) -> dict[str, list[dict]]:
    lookup: dict[str, list[dict]] = {}
    for row in rows:
        for key in row_keys(row):
            lookup.setdefault(key, []).append(row)
    return lookup


def any_key_in(keys: set[str], lookup_or_set: dict[str, list[dict]] | set[str]) -> bool:
    if isinstance(lookup_or_set, set):
        return bool(keys & lookup_or_set)
    return any(key in lookup_or_set for key in keys)


def lookup_rows(row: dict, lookup: dict[str, list[dict]]) -> list[dict]:
    matched: list[dict] = []
    seen = set()
    for key in row_keys(row):
        for item in lookup.get(key, []):
            ident = id(item)
            if ident not in seen:
                matched.append(item)
                seen.add(ident)
    return matched


def lookup_request_rows(row: dict, exact_lookup: dict[str, dict], loose_lookup: dict[str, list[dict]]) -> list[dict]:
    if has_exact_row_identity(row):
        exact = exact_lookup.get(exact_row_key(row))
        return [exact] if exact is not None else []
    return lookup_rows(row, loose_lookup)


def split_summary(rows: Sequence[dict]) -> dict:
    return {
        "rows": len(rows),
        "split_counts": dict(sorted(Counter(row.get("split", "") for row in rows).items())),
        "label_split_counts": {
            split: dict(sorted(Counter(str(row.get("label", "")) for row in rows if row.get("split") == split).items()))
            for split in ["train", "val", "test"]
        },
    }


def duplicate_key_count(rows: Sequence[dict]) -> int:
    seen: set[str] = set()
    duplicate_rows = 0
    for row in rows:
        keys = row_keys(row)
        if keys & seen:
            duplicate_rows += 1
        seen.update(keys)
    return duplicate_rows


def row_present_exact(row: dict, exact_lookup: dict[str, dict], loose_lookup: dict[str, list[dict]]) -> bool:
    if has_exact_row_identity(row):
        return exact_row_key(row) in exact_lookup
    return any_key_in(row_keys(row), loose_lookup)


def count_by_split_label(rows: Sequence[dict], *, label_field: str = "label") -> dict[str, int]:
    counts = Counter()
    for row in rows:
        split = str(row.get("split", ""))
        label = str(row.get(label_field, row.get("label", "")))
        counts[f"{split}:{label}"] += 1
    return dict(sorted(counts.items()))


def replacement_label(row: dict) -> str:
    return str(row.get("replacement_label") or row.get("original_label") or row.get("label") or "").strip()


def request_group_counts(rows: Sequence[dict]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        split = str(row.get("split", ""))
        counts[f"{split}:{replacement_label(row)}"] += 1
    return dict(sorted(counts.items()))


def shape_failures(
    rows: Sequence[dict],
    *,
    enforce_shape: bool,
    enforce_label_balance: bool,
) -> list[str]:
    if not enforce_shape:
        return []
    summary = split_summary(rows)
    failures = []
    if summary["rows"] != EXPECTED_TOTAL:
        failures.append(f"expected {EXPECTED_TOTAL} rows, got {summary['rows']}")
    if summary["split_counts"] != EXPECTED_SPLIT_COUNTS:
        failures.append(f"split_counts mismatch: {summary['split_counts']}")
    if enforce_label_balance and summary["label_split_counts"] != EXPECTED_LABEL_SPLIT_COUNTS:
        failures.append(f"label_split_counts mismatch: {summary['label_split_counts']}")
    return failures


def _detail(row: dict, record_type: str, status: str, reason: str) -> dict:
    return {
        "record_type": record_type,
        "source_path": row.get("source_path", ""),
        "label": row.get("label", row.get("original_label", "")),
        "sample_index": row.get("sample_index", ""),
        "split": row.get("split", ""),
        "status": status,
        "reason": reason,
    }


def audit_corrected_split_replacements(
    *,
    original_split_csv: Path,
    corrected_split_csv: Path,
    plan_csv: Path,
    detail_output_csv: Optional[Path] = None,
    enforce_shape: bool = True,
    enforce_label_balance: bool = False,
) -> dict:
    original_rows = read_csv_rows(original_split_csv)
    corrected_rows = read_csv_rows(corrected_split_csv)
    plan_rows = read_csv_rows(plan_csv)

    original_lookup = build_key_lookup(original_rows)
    corrected_lookup = build_key_lookup(corrected_rows)
    original_exact_lookup = build_exact_lookup(original_rows)
    corrected_exact_lookup = build_exact_lookup(corrected_rows)
    original_key_set = set(original_lookup)
    corrected_key_set = set(corrected_lookup)

    replacement_requests = [
        row
        for row in plan_rows
        if str(row.get("replacement_required", "")).strip().casefold() == "true"
        or str(row.get("plan_action", "")).strip().casefold() == "exclude_and_replace"
    ]
    relabel_requests = [
        row
        for row in plan_rows
        if str(row.get("plan_action", "")).strip().casefold() == "relabel"
        and str(row.get("usable_for_training_policy", "")).strip().casefold() == "true"
    ]

    detail_rows: list[dict] = []
    failures: list[str] = []

    excluded_original_rows: list[dict] = []
    missing_excluded_in_original = 0
    test_replacement_requests = 0
    planned_excluded_removed_count = 0
    for request in replacement_requests:
        if str(request.get("split", "")) == "test":
            test_replacement_requests += 1
        matches = lookup_request_rows(request, original_exact_lookup, original_lookup)
        if not matches:
            missing_excluded_in_original += 1
            detail_rows.append(_detail(request, "replacement_request", "missing_original", "replacement request did not match original split"))
            continue
        for original in matches:
            excluded_original_rows.append(original)
            status = "still_present" if row_present_exact(original, corrected_exact_lookup, corrected_lookup) else "removed"
            reason = "excluded row still appears in corrected split" if status == "still_present" else "excluded row removed"
            if status == "removed":
                planned_excluded_removed_count += 1
            detail_rows.append(_detail(original, "excluded_original", status, reason))

    excluded_key_set: set[str] = set()
    excluded_exact_key_set: set[str] = set()
    for row in excluded_original_rows:
        excluded_key_set.update(row_keys(row))
        if has_exact_row_identity(row):
            excluded_exact_key_set.add(exact_row_key(row))

    excluded_present_after = [
        row
        for row in corrected_rows
        if (
            (excluded_exact_key_set and exact_row_key(row) in excluded_exact_key_set)
            or (
                not excluded_exact_key_set
                and excluded_key_set
                and any_key_in(row_keys(row), excluded_key_set)
            )
        )
    ]
    for row in excluded_present_after:
        detail_rows.append(_detail(row, "corrected_row", "excluded_key_present", "corrected row matches an excluded source"))

    fresh_replacement_rows = [
        row for row in corrected_rows if not any_key_in(row_keys(row), original_key_set)
    ]
    for row in fresh_replacement_rows:
        detail_rows.append(_detail(row, "fresh_replacement", "fresh", "corrected row was not present in the original split"))

    original_missing_rows = [
        row for row in original_rows if not any_key_in(row_keys(row), corrected_key_set)
    ]
    planned_excluded_missing = [
        row
        for row in original_missing_rows
        if (
            (excluded_exact_key_set and exact_row_key(row) in excluded_exact_key_set)
            or (
                not excluded_exact_key_set
                and excluded_key_set
                and any_key_in(row_keys(row), excluded_key_set)
            )
        )
    ]
    unplanned_removed_rows = [
        row
        for row in original_missing_rows
        if not (
            (excluded_exact_key_set and exact_row_key(row) in excluded_exact_key_set)
            or (
                not excluded_exact_key_set
                and excluded_key_set
                and any_key_in(row_keys(row), excluded_key_set)
            )
        )
    ]
    for row in unplanned_removed_rows:
        detail_rows.append(_detail(row, "original_row", "unplanned_removed", "original row disappeared without replacement request"))

    relabel_missing_original = 0
    relabel_missing_corrected = 0
    relabel_label_mismatch = 0
    test_relabel_requests = 0
    for request in relabel_requests:
        if str(request.get("split", "")) == "test":
            test_relabel_requests += 1
        original_matches = lookup_request_rows(request, original_exact_lookup, original_lookup)
        corrected_matches = lookup_request_rows(request, corrected_exact_lookup, corrected_lookup)
        if not original_matches:
            relabel_missing_original += 1
            detail_rows.append(_detail(request, "relabel_request", "missing_original", "relabel request did not match original split"))
            continue
        if not corrected_matches:
            relabel_missing_corrected += 1
            detail_rows.append(_detail(request, "relabel_request", "missing_corrected", "relabel source missing from corrected split"))
            continue
        planned_label = str(request.get("planned_label", "")).strip()
        if not any(str(row.get("label", "")).strip() == planned_label for row in corrected_matches):
            relabel_label_mismatch += 1
            detail_rows.append(_detail(request, "relabel_request", "label_mismatch", "corrected split does not contain planned label"))
        else:
            detail_rows.append(_detail(request, "relabel_request", "applied", "corrected split contains planned label"))

    request_counts = request_group_counts(replacement_requests)
    fresh_counts = count_by_split_label(fresh_replacement_rows)
    replacement_count_mismatches = {
        key: {"requested": request_counts.get(key, 0), "fresh": fresh_counts.get(key, 0)}
        for key in sorted(set(request_counts) | set(fresh_counts))
        if request_counts.get(key, 0) != fresh_counts.get(key, 0)
    }

    row_count_ok = len(original_rows) == len(corrected_rows)
    if not row_count_ok:
        failures.append(f"corrected row count changed: {len(corrected_rows)} != {len(original_rows)}")
    if missing_excluded_in_original:
        failures.append(f"replacement requests missing original rows: {missing_excluded_in_original}")
    if excluded_present_after:
        failures.append(f"excluded rows still present after correction: {len(excluded_present_after)}")
    if unplanned_removed_rows:
        failures.append(f"unplanned original rows removed: {len(unplanned_removed_rows)}")
    if replacement_count_mismatches:
        failures.append("fresh replacement counts do not match replacement requests")
    original_duplicate_rows = duplicate_key_count(original_rows)
    corrected_duplicate_rows = duplicate_key_count(corrected_rows)
    duplicate_key_row_delta = corrected_duplicate_rows - original_duplicate_rows
    if duplicate_key_row_delta > 0:
        failures.append(f"corrected split introduced duplicate source keys: +{duplicate_key_row_delta}")
    if test_replacement_requests:
        failures.append(f"test split replacement requests are not allowed: {test_replacement_requests}")
    if test_relabel_requests:
        failures.append(f"test split relabel requests are not allowed: {test_relabel_requests}")
    if relabel_missing_original:
        failures.append(f"relabel requests missing original rows: {relabel_missing_original}")
    if relabel_missing_corrected:
        failures.append(f"relabel requests missing corrected rows: {relabel_missing_corrected}")
    if relabel_label_mismatch:
        failures.append(f"relabel requests not applied: {relabel_label_mismatch}")

    failures.extend(
        shape_failures(
            corrected_rows,
            enforce_shape=enforce_shape,
            enforce_label_balance=enforce_label_balance,
        )
    )

    if detail_output_csv is not None:
        write_detail_rows(detail_output_csv, detail_rows)

    payload = {
        "schema": "axon_corrected_split_replacement_integrity_v1",
        "original_split_csv": str(resolve_path(original_split_csv)),
        "corrected_split_csv": str(resolve_path(corrected_split_csv)),
        "plan_csv": str(resolve_path(plan_csv)),
        "original_summary": split_summary(original_rows),
        "corrected_summary": split_summary(corrected_rows),
        "plan_rows": len(plan_rows),
        "replacement_requests": len(replacement_requests),
        "relabel_requests": len(relabel_requests),
        "row_count_ok": row_count_ok,
        "shape_enforced": bool(enforce_shape),
        "label_balance_enforced": bool(enforce_label_balance),
        "original_duplicate_key_rows": original_duplicate_rows,
        "corrected_duplicate_key_rows": corrected_duplicate_rows,
        "duplicate_key_row_delta": duplicate_key_row_delta,
        "excluded_rows_found_in_original": len(excluded_original_rows),
        "excluded_rows_missing_in_original": missing_excluded_in_original,
        "excluded_rows_present_after_correction": len(excluded_present_after),
        "planned_excluded_rows_removed": planned_excluded_removed_count,
        "unplanned_original_rows_removed": len(unplanned_removed_rows),
        "fresh_replacement_rows": len(fresh_replacement_rows),
        "replacement_request_counts_by_split_label": request_counts,
        "fresh_replacement_counts_by_split_label": fresh_counts,
        "replacement_count_mismatches": replacement_count_mismatches,
        "test_replacement_requests": test_replacement_requests,
        "test_relabel_requests": test_relabel_requests,
        "relabel_missing_original": relabel_missing_original,
        "relabel_missing_corrected": relabel_missing_corrected,
        "relabel_label_mismatch": relabel_label_mismatch,
        "detail_output_csv": str(resolve_path(detail_output_csv)) if detail_output_csv is not None else None,
        "integrity_failures": failures,
        "replacement_integrity_ok": not failures,
        "notes": [
            "Excluded rows must disappear from the corrected split.",
            "Fresh replacement rows are rows whose source keys were not present in the original split.",
            "Replacement counts must match replacement requests by split and label.",
            "This audit does not extract features; run the corrected split cache readiness audit after this check.",
        ],
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit replacement integrity for a corrected split.")
    parser.add_argument("--original-split-csv", type=Path, required=True)
    parser.add_argument("--corrected-split-csv", type=Path, required=True)
    parser.add_argument("--plan-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--detail-output-csv", type=Path, default=None)
    parser.add_argument("--no-enforce-shape", action="store_true")
    parser.add_argument("--enforce-label-balance", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless replacement integrity is clean.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_corrected_split_replacements(
        original_split_csv=args.original_split_csv,
        corrected_split_csv=args.corrected_split_csv,
        plan_csv=args.plan_csv,
        detail_output_csv=args.detail_output_csv,
        enforce_shape=not bool(args.no_enforce_shape),
        enforce_label_balance=bool(args.enforce_label_balance),
    )
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.strict and not payload["replacement_integrity_ok"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
