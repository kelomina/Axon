#!/usr/bin/env python3
"""Build a non-destructive cleanup plan for duplicate source identities."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_split_duplicate_sources import read_csv_rows, resolve_path  # noqa: E402


PLAN_FIELDNAMES = [
    "source_path",
    "source_sha256",
    "sample_index",
    "split",
    "original_label",
    "planned_label",
    "plan_action",
    "reason",
    "manual_label_verdict",
    "recommended_action",
    "manual_verdict_note",
    "replacement_required",
    "replacement_label",
    "usable_for_training_policy",
]

QUEUE_FIELDNAMES = [
    "duplicate_group_id",
    "duplicate_key",
    "group_size",
    "labels",
    "splits",
    "cross_label",
    "cross_split",
    "source_path",
    "source_sha256",
    "label",
    "sample_index",
    "split",
    "review_reason",
    "manual_label_verdict",
    "recommended_action",
    "manual_verdict_note",
]

SPLIT_PROTECTION_RANK = {"test": 0, "val": 1, "train": 2}


def write_csv_rows(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize_bool(value: object) -> bool:
    return str(value or "").strip().casefold() in {"true", "1", "yes"}


def group_detail_rows(rows: Sequence[dict]) -> list[list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("duplicate_group_id", ""))].append(row)
    return [groups[key] for key in sorted(groups, key=lambda item: int(item) if item.isdigit() else item)]


def _sample_index(row: dict) -> int:
    try:
        return int(row.get("sample_index", 0))
    except (TypeError, ValueError):
        return 0


def choose_canonical_row(rows: Sequence[dict], policy: str) -> dict:
    if policy == "lowest_sample_index":
        return sorted(rows, key=_sample_index)[0]
    if policy == "prefer_train":
        return sorted(rows, key=lambda row: ({"train": 0, "val": 1, "test": 2}.get(str(row.get("split", "")), 99), _sample_index(row)))[0]
    if policy == "protect_test_then_val":
        return sorted(rows, key=lambda row: (SPLIT_PROTECTION_RANK.get(str(row.get("split", "")), 99), _sample_index(row)))[0]
    raise ValueError(f"Unsupported keep policy: {policy}")


def rows_requiring_test_mutation(rows: Sequence[dict], canonical: dict) -> list[dict]:
    canonical_sample_index = str(canonical.get("sample_index", ""))
    return [
        row
        for row in rows
        if str(row.get("sample_index", "")) != canonical_sample_index
        and str(row.get("split", "")).strip().casefold() == "test"
    ]


def plan_row(row: dict, *, reason: str) -> dict:
    label = str(row.get("label", "")).strip()
    return {
        "source_path": row.get("source_path", ""),
        "source_sha256": row.get("source_sha256", ""),
        "sample_index": row.get("sample_index", ""),
        "split": row.get("split", ""),
        "original_label": label,
        "planned_label": label,
        "plan_action": "exclude_and_replace",
        "reason": reason,
        "manual_label_verdict": "out_of_scope",
        "recommended_action": "replace_sample",
        "manual_verdict_note": "duplicate source identity; keep one canonical same-label row and replace duplicate",
        "replacement_required": "true",
        "replacement_label": label,
        "usable_for_training_policy": "false",
    }


def queue_row(row: dict, *, reason: str) -> dict:
    return {
        "duplicate_group_id": row.get("duplicate_group_id", ""),
        "duplicate_key": row.get("duplicate_key", ""),
        "group_size": row.get("group_size", ""),
        "labels": row.get("labels", ""),
        "splits": row.get("splits", ""),
        "cross_label": row.get("cross_label", ""),
        "cross_split": row.get("cross_split", ""),
        "source_path": row.get("source_path", ""),
        "source_sha256": row.get("source_sha256", ""),
        "label": row.get("label", ""),
        "sample_index": row.get("sample_index", ""),
        "split": row.get("split", ""),
        "review_reason": reason,
        "manual_label_verdict": "",
        "recommended_action": "",
        "manual_verdict_note": "",
    }


def build_duplicate_cleanup_plan(
    *,
    duplicate_csv: Path,
    output_plan_csv: Path,
    output_review_csv: Path,
    output_json: Path,
    keep_policy: str = "protect_test_then_val",
    freeze_test: bool = True,
) -> dict:
    detail_rows = read_csv_rows(duplicate_csv)
    groups = group_detail_rows(detail_rows)
    plan_rows: list[dict] = []
    review_rows: list[dict] = []
    group_actions: Counter[str] = Counter()
    plan_split_counts: Counter[str] = Counter()
    plan_label_counts: Counter[str] = Counter()

    for rows in groups:
        if not rows:
            continue
        cross_label = any(normalize_bool(row.get("cross_label")) for row in rows)
        group_id = str(rows[0].get("duplicate_group_id", ""))
        if cross_label:
            group_actions["manual_review_required"] += 1
            reason = "cross-label duplicate source identity requires human label adjudication"
            for row in rows:
                review_rows.append(queue_row(row, reason=reason))
            continue

        canonical = choose_canonical_row(rows, keep_policy)
        if freeze_test and rows_requiring_test_mutation(rows, canonical):
            group_actions["manual_review_required_frozen_test"] += 1
            reason = "same-label duplicate would require mutating frozen test split; review in next data version"
            for row in rows:
                review_rows.append(queue_row(row, reason=reason))
            continue

        canonical_sample_index = str(canonical.get("sample_index", ""))
        group_actions["auto_replace_duplicates"] += 1
        for row in rows:
            if str(row.get("sample_index", "")) == canonical_sample_index:
                continue
            reason = f"duplicate source identity group {group_id}; canonical sample_index={canonical_sample_index}"
            planned = plan_row(row, reason=reason)
            plan_rows.append(planned)
            plan_split_counts[str(row.get("split", ""))] += 1
            plan_label_counts[str(row.get("label", ""))] += 1

    write_csv_rows(output_plan_csv, plan_rows, PLAN_FIELDNAMES)
    write_csv_rows(output_review_csv, review_rows, QUEUE_FIELDNAMES)

    summary = {
        "schema": "axon_duplicate_source_cleanup_plan_v1",
        "duplicate_csv": str(resolve_path(duplicate_csv)),
        "keep_policy": keep_policy,
        "duplicate_groups": len(groups),
        "freeze_test": bool(freeze_test),
        "detail_rows": len(detail_rows),
        "auto_plan_rows": len(plan_rows),
        "manual_review_rows": len(review_rows),
        "group_action_counts": dict(sorted(group_actions.items())),
        "planned_replacement_counts_by_split": dict(sorted(plan_split_counts.items())),
        "planned_replacement_counts_by_label": dict(sorted(plan_label_counts.items())),
        "outputs": {
            "plan_csv": str(resolve_path(output_plan_csv)),
            "review_csv": str(resolve_path(output_review_csv)),
            "json": str(resolve_path(output_json)),
        },
        "notes": [
            "Same-label duplicate groups are converted to exclude-and-replace plan rows.",
            "Cross-label duplicate groups are sent to manual review and are not auto-replaced.",
            "When freeze_test is true, duplicate groups that require test replacement are sent to review.",
            "The plan is non-destructive; use build_corrected_split_from_plan.py to materialize a corrected split.",
            "Fresh replacement and exact 20w shape must be verified by downstream audits.",
        ],
    }
    resolved_json = resolve_path(output_json)
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a non-destructive duplicate source cleanup plan.")
    parser.add_argument("--duplicate-csv", type=Path, required=True)
    parser.add_argument("--output-plan-csv", type=Path, required=True)
    parser.add_argument("--output-review-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--keep-policy",
        choices=["protect_test_then_val", "prefer_train", "lowest_sample_index"],
        default="protect_test_then_val",
    )
    parser.add_argument(
        "--allow-test-replacements",
        action="store_true",
        help="Allow automatic duplicate cleanup plan rows for test split. Default freezes test and queues those groups for review.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_duplicate_cleanup_plan(
        duplicate_csv=args.duplicate_csv,
        output_plan_csv=args.output_plan_csv,
        output_review_csv=args.output_review_csv,
        output_json=args.output_json,
        keep_policy=args.keep_policy,
        freeze_test=not bool(args.allow_test_replacements),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
