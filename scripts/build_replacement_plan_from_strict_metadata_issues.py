#!/usr/bin/env python3
"""Build an exclude-and-replace plan from strict metadata issue rows."""

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
    "metadata_issue_flags",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_issue_rows(summary_json: Path) -> list[dict]:
    payload = json.loads(resolve_path(summary_json).read_text(encoding="utf-8"))
    rows = payload.get("row_issue_examples", [])
    row_issue_count = int(payload.get("row_issue_count", 0) or 0)
    if row_issue_count != len(rows):
        raise ValueError(
            "Strict metadata summary does not contain every issue row; rerun the source tool "
            "with full issue export before building a replacement plan."
        )
    return [dict(row) for row in rows]


def build_replacement_plan_from_issues(
    *,
    summary_json: Path,
    allowed_issues: Optional[set[str]] = None,
) -> tuple[list[dict], dict]:
    issue_rows = read_issue_rows(summary_json)
    allowed = allowed_issues or {"manifest_conflicting_labels_for_source_sha256"}
    plan_rows: list[dict] = []
    blocked_rows: list[dict] = []
    issue_counts: Counter = Counter()
    replacement_counts: Counter = Counter()

    for row in issue_rows:
        issues = [str(issue) for issue in row.get("issues", [])]
        for issue in issues:
            issue_counts[issue] += 1
        unsupported = sorted(set(issues) - allowed)
        label = str(row.get("label", "")).strip()
        split = str(row.get("split", "")).strip()
        if unsupported or label not in {"0", "1"} or split not in {"train", "val", "test"}:
            blocked_rows.append({**row, "unsupported_issues": unsupported})
            continue
        replacement_counts[f"{split}:{label}"] += 1
        plan_rows.append(
            {
                "source_path": row.get("source_path", ""),
                "source_sha256": str(row.get("source_sha256", "")).strip().casefold(),
                "sample_index": row.get("sample_index", ""),
                "split": split,
                "original_label": label,
                "planned_label": label,
                "plan_action": "exclude_and_replace",
                "replacement_required": "true",
                "replacement_label": label,
                "usable_for_training_policy": "false",
                "metadata_issue_flags": "|".join(issues),
            }
        )

    summary = {
        "schema": "axon_replacement_plan_from_strict_metadata_issues_v1",
        "source_summary_json": str(resolve_path(summary_json)),
        "allowed_issues": sorted(allowed),
        "issue_rows": len(issue_rows),
        "plan_rows": len(plan_rows),
        "blocked_rows": len(blocked_rows),
        "issue_counts": dict(sorted(issue_counts.items())),
        "replacement_counts_by_split_label": dict(sorted(replacement_counts.items())),
        "blocked_examples": blocked_rows[:50],
        "plan_ready": bool(issue_rows) and not blocked_rows and len(plan_rows) == len(issue_rows),
        "identity_feature_policy": (
            "source_path and source_sha256 are used only to identify rows for quarantine/replacement; "
            "the plan does not relabel and does not use names, paths, directories, or extensions as verdict evidence."
        ),
        "notes": [
            "Every issue row becomes exclude_and_replace with replacement_label equal to the original explicit split label.",
            "This plan is for metadata-noise quarantine only; run replacement integrity and cache readiness audits after building the corrected split.",
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
    parser = argparse.ArgumentParser(description="Build replacement plan from strict metadata issue summary.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--allow-issue", action="append", default=["manifest_conflicting_labels_for_source_sha256"])
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows, summary = build_replacement_plan_from_issues(
        summary_json=args.summary_json,
        allowed_issues=set(args.allow_issue or []),
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
