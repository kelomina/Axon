#!/usr/bin/env python3
"""Build a replacement plan for rows whose bounded cache recovery failed."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_FIELDNAMES = [
    "source_path",
    "source_sha256",
    "sample_index",
    "split",
    "original_label",
    "planned_label",
    "plan_action",
    "replacement_required",
    "replacement_label",
    "usable_for_training_policy",
    "cache_recovery_status",
]
SUCCESS_STATUSES = {"extracted", "cache_hit"}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_path(value: object) -> str:
    return str(value or "").strip().casefold()


def normalize_sha(value: object) -> str:
    return str(value or "").strip().casefold()


def split_index(rows: Sequence[dict]) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    by_path: dict[str, dict] = {}
    by_sha: dict[str, list[dict]] = {}
    for row in rows:
        source_path = normalize_path(row.get("source_path"))
        source_sha = normalize_sha(row.get("source_sha256"))
        if source_path:
            by_path.setdefault(source_path, row)
        if source_sha:
            by_sha.setdefault(source_sha, []).append(row)
    return by_path, by_sha


def failed_recovery_rows(recovery_json: Path) -> list[dict]:
    payload = json.loads(resolve_path(recovery_json).read_text(encoding="utf-8"))
    examples = payload.get("failed_examples", [])
    if not isinstance(examples, list):
        raise ValueError("Recovery JSON failed_examples must be a list")
    return [
        dict(row)
        for row in examples
        if str(row.get("status", "")).strip() not in SUCCESS_STATUSES
    ]


def locate_failed_row(failure: dict, by_path: dict[str, dict], by_sha: dict[str, list[dict]]) -> tuple[Optional[dict], str]:
    source_sha = normalize_sha(failure.get("source_sha256") or failure.get("expected_source_sha256"))
    if source_sha:
        matches = by_sha.get(source_sha, [])
        if len(matches) == 1:
            return matches[0], "source_sha256"
        if len(matches) > 1:
            return None, "ambiguous_source_sha256"
    source_path = normalize_path(failure.get("source_path"))
    if source_path:
        row = by_path.get(source_path)
        if row is not None:
            return row, "source_path"
    return None, "missing_split_row"


def build_replacement_plan_from_failures(
    *,
    split_csv: Path,
    recovery_json: Path,
) -> tuple[list[dict], dict]:
    split_rows = read_csv_rows(split_csv)
    by_path, by_sha = split_index(split_rows)
    failures = failed_recovery_rows(recovery_json)

    plan_rows: list[dict] = []
    blocked_rows: list[dict] = []
    status_counts: Counter = Counter()
    match_counts: Counter = Counter()
    replacement_counts: Counter = Counter()

    for failure in failures:
        status = str(failure.get("status", "")).strip()
        status_counts[status] += 1
        row, match_reason = locate_failed_row(failure, by_path, by_sha)
        match_counts[match_reason] += 1
        if row is None:
            blocked_rows.append({**failure, "match_reason": match_reason})
            continue
        split = str(row.get("split", "")).strip()
        label = str(row.get("label", "")).strip()
        if split not in {"train", "val", "test"} or label not in {"0", "1"}:
            blocked_rows.append({**failure, "match_reason": match_reason, "split_row": row})
            continue
        replacement_counts[f"{split}:{label}"] += 1
        plan_rows.append(
            {
                "source_path": row.get("source_path", ""),
                "source_sha256": normalize_sha(row.get("source_sha256")),
                "sample_index": row.get("sample_index", ""),
                "split": split,
                "original_label": label,
                "planned_label": label,
                "plan_action": "exclude_and_replace",
                "replacement_required": "true",
                "replacement_label": label,
                "usable_for_training_policy": "false",
                "cache_recovery_status": status,
            }
        )

    summary = {
        "schema": "axon_replacement_plan_from_cache_recovery_failures_v1",
        "split_csv": str(resolve_path(split_csv)),
        "recovery_json": str(resolve_path(recovery_json)),
        "failed_rows": len(failures),
        "plan_rows": len(plan_rows),
        "blocked_rows": len(blocked_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "match_counts": dict(sorted(match_counts.items())),
        "replacement_counts_by_split_label": dict(sorted(replacement_counts.items())),
        "blocked_examples": blocked_rows[:50],
        "plan_ready": bool(failures) and not blocked_rows and len(plan_rows) == len(failures),
        "identity_feature_policy": (
            "source_path/source_sha256 are used only to locate failed cache rows in the split; "
            "replacement_label is copied from the explicit split label, never inferred from names, paths, directories, or extensions."
        ),
        "notes": [
            "Failed cache recovery rows are treated as feature-broken rows and must be replaced with fresh same-label candidates.",
            "This tool does not change cache or split files; run corrected split build and audits after this plan.",
        ],
    }
    return plan_rows, summary


def write_plan_csv(path: Path, rows: Sequence[dict]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDNAMES, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build replacement plan from cache recovery failures.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--recovery-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows, summary = build_replacement_plan_from_failures(
        split_csv=args.split_csv,
        recovery_json=args.recovery_json,
    )
    write_plan_csv(args.output_csv, rows)
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    summary["outputs"] = {
        "csv": str(resolve_path(args.output_csv)),
        "json": str(output_json),
    }
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["plan_ready"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
