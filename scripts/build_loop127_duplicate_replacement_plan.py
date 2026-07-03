#!/usr/bin/env python3
"""Build a non-destructive Loop127 replacement plan for duplicate content hashes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
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
    "reason",
    "manual_label_verdict",
    "recommended_action",
    "manual_verdict_note",
    "replacement_required",
    "replacement_label",
    "usable_for_training_policy",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("split") or "").strip(),
        str(row.get("sample_index") or "").strip(),
        str(row.get("source_sha256") or "").strip().casefold(),
    )


def _sample_index(row: dict[str, str]) -> int:
    try:
        return int(str(row.get("sample_index") or ""))
    except ValueError:
        return 0


def _prediction_index(rows: Sequence[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    index = {}
    for row in rows:
        index.setdefault(_row_key(row), row)
    return index


def _plan_row(row: dict[str, str], *, canonical_sample_index: str) -> dict[str, str]:
    label = str(row.get("label") or "").strip()
    return {
        "source_path": str(row.get("source_path") or ""),
        "source_sha256": str(row.get("source_sha256") or ""),
        "sample_index": str(row.get("sample_index") or ""),
        "split": str(row.get("split") or ""),
        "original_label": label,
        "planned_label": label,
        "plan_action": "exclude_and_replace",
        "reason": (
            "loop127_duplicate_source_sha256; "
            f"canonical_sample_index={canonical_sample_index}; fresh_same_label_redraw_required"
        ),
        "manual_label_verdict": "out_of_scope",
        "recommended_action": "replace_sample",
        "manual_verdict_note": "duplicate content identity; replacement must come from fresh unused same-label candidate",
        "replacement_required": "true",
        "replacement_label": label,
        "usable_for_training_policy": "false",
    }


def build_loop127_duplicate_replacement_plan(
    *,
    duplicate_audit_csv: Path,
    train_predictions: Path,
    val_predictions: Path,
    output_plan_csv: Path,
    output_json: Path,
) -> dict:
    duplicate_rows = read_csv_rows(duplicate_audit_csv)
    prediction_rows = read_csv_rows(train_predictions) + read_csv_rows(val_predictions)
    predictions = _prediction_index(prediction_rows)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in duplicate_rows:
        split = str(row.get("split") or "").strip()
        sha = str(row.get("source_sha256") or "").strip().casefold()
        if split and sha:
            groups[(split, sha)].append(row)

    blockers: list[str] = []
    plan_rows: list[dict[str, str]] = []
    missing_prediction_rows = 0
    label_conflict_groups = 0
    for (split, sha), rows in sorted(groups.items()):
        labels = {str(row.get("label") or "").strip() for row in rows}
        if len(labels) > 1:
            label_conflict_groups += 1
            continue
        canonical = sorted(rows, key=_sample_index)[0]
        canonical_sample_index = str(canonical.get("sample_index") or "").strip()
        for duplicate in rows:
            sample_index = str(duplicate.get("sample_index") or "").strip()
            if sample_index == canonical_sample_index:
                continue
            prediction = predictions.get((split, sample_index, sha))
            if prediction is None:
                missing_prediction_rows += 1
                continue
            plan_rows.append(_plan_row(prediction, canonical_sample_index=canonical_sample_index))

    if label_conflict_groups:
        blockers.append("duplicate_label_conflict_groups")
    if missing_prediction_rows:
        blockers.append("duplicate_rows_missing_from_predictions")

    output_plan = resolve_path(output_plan_csv)
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    with output_plan.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(plan_rows)

    split_counts = Counter(row["split"] for row in plan_rows)
    label_counts = Counter(row["replacement_label"] for row in plan_rows)
    payload = {
        "schema": "axon_loop127_duplicate_replacement_plan_v1",
        "protocol": "read-only plan; duplicate content hashes are identity-only data quality blockers, not model evidence",
        "duplicate_audit_csv": str(resolve_path(duplicate_audit_csv)),
        "train_predictions": str(resolve_path(train_predictions)),
        "val_predictions": str(resolve_path(val_predictions)),
        "output_plan_csv": str(output_plan),
        "duplicate_groups": len(groups),
        "duplicate_audit_rows": len(duplicate_rows),
        "plan_rows": len(plan_rows),
        "replacement_counts_by_split": dict(sorted(split_counts.items())),
        "replacement_counts_by_label": dict(sorted(label_counts.items())),
        "label_conflict_groups": label_conflict_groups,
        "missing_prediction_rows": missing_prediction_rows,
        "blockers": blockers,
        "plan_ready": not blockers,
        "notes": [
            "Canonical row is the lowest sample_index within each split/source_sha256 group.",
            "Every non-canonical duplicate becomes exclude_and_replace with the original locked label.",
            "This script does not sample replacements and does not mutate the split.",
            "Run corrected split builder with fresh unused same-label candidates, then rerun cache/readiness audits.",
        ],
    }
    output_json_path = resolve_path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop127 duplicate replacement plan.")
    parser.add_argument("--duplicate-audit-csv", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--output-plan-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_loop127_duplicate_replacement_plan(
        duplicate_audit_csv=args.duplicate_audit_csv,
        train_predictions=args.train_predictions,
        val_predictions=args.val_predictions,
        output_plan_csv=args.output_plan_csv,
        output_json=args.output_json,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["plan_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
