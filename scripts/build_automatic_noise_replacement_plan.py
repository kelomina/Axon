#!/usr/bin/env python3
"""Build an automatic, non-relabel replacement plan for high-confidence noise candidates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SPLITS = {"train", "val"}
FIELDNAMES = [
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

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from apply_manual_review_verdicts import find_split_row, load_split_index, normalize_text, read_csv_rows  # noqa: E402


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_csv_rows(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def safe_int(value: object, default: int = 999) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def model_confidence(row: dict) -> float:
    prob = safe_float(row.get("prob_malicious"))
    label = normalize_text(row.get("label"))
    if label == "0":
        return prob
    if label == "1":
        return 1.0 - prob
    return 0.0


def row_is_eligible(
    row: dict,
    *,
    max_priority: int,
    min_confidence: float,
    min_opposite_ratio: float,
    min_nearest_similarity: float,
    support_bucket: str,
) -> bool:
    if normalize_text(row.get("support_bucket")) != normalize_text(support_bucket):
        return False
    if safe_int(row.get("priority")) > max_priority:
        return False
    if model_confidence(row) < min_confidence:
        return False
    if safe_float(row.get("opposite_label_ratio")) < min_opposite_ratio:
        return False
    if safe_float(row.get("nearest_similarity")) < min_nearest_similarity:
        return False
    if normalize_text(row.get("label")) not in {"0", "1"}:
        return False
    return True


def plan_row(review_row: dict, split_row: dict, reason: str) -> dict:
    label = str(split_row.get("label", review_row.get("label", ""))).strip()
    return {
        "source_path": split_row.get("source_path", review_row.get("source_path", "")),
        "source_sha256": review_row.get("source_sha256", ""),
        "sample_index": split_row.get("sample_index", review_row.get("sample_index", "")),
        "split": split_row.get("split", ""),
        "original_label": label,
        "planned_label": label,
        "plan_action": "exclude_and_replace",
        "reason": reason,
        "manual_label_verdict": "automatic_high_confidence_noise_candidate",
        "recommended_action": "replace_sample",
        "manual_verdict_note": (
            "Automatic evidence-only action: excluded from Train/Val policy and replaced with a fresh same-label "
            "candidate. No relabel was applied."
        ),
        "replacement_required": "true",
        "replacement_label": label,
        "usable_for_training_policy": "false",
    }


def build_automatic_plan(
    *,
    review_csv: Path,
    split_csv: Path,
    max_priority: int = 0,
    min_confidence: float = 0.95,
    min_opposite_ratio: float = 0.80,
    min_nearest_similarity: float = 0.0,
    support_bucket: str = "neighbors_support_model_prediction",
) -> tuple[list[dict], dict]:
    review_rows = read_csv_rows(review_csv)
    split_index, split_summary = load_split_index(split_csv)
    planned_rows: list[dict] = []
    skipped_counts: Counter[str] = Counter()
    replacement_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for row in review_rows:
        if not row_is_eligible(
            row,
            max_priority=max_priority,
            min_confidence=min_confidence,
            min_opposite_ratio=min_opposite_ratio,
            min_nearest_similarity=min_nearest_similarity,
            support_bucket=support_bucket,
        ):
            skipped_counts["not_eligible"] += 1
            continue
        split_row = find_split_row(row, split_index)
        if split_row is None:
            skipped_counts["missing_split_row"] += 1
            continue
        split = str(split_row.get("split", "")).strip()
        if split not in TRAINING_SPLITS:
            skipped_counts[f"held_out_{split or 'unknown'}"] += 1
            continue
        label = str(split_row.get("label", row.get("label", ""))).strip()
        if label != str(row.get("label", "")).strip():
            skipped_counts["split_label_mismatch"] += 1
            continue
        reason = (
            f"auto_noise_replace:support={support_bucket};priority<={max_priority};"
            f"confidence>={min_confidence};opposite_ratio>={min_opposite_ratio};"
            f"nearest_similarity>={min_nearest_similarity}"
        )
        item = plan_row(row, split_row, reason)
        planned_rows.append(item)
        replacement_counts[label] += 1
        split_counts[split] += 1
        label_counts[label] += 1

    summary = {
        "schema": "axon_automatic_noise_replacement_plan_v1",
        "review_csv": str(resolve_path(review_csv)),
        "split_csv": str(resolve_path(split_csv)),
        "policy": {
            "action": "exclude_and_replace_same_label_only",
            "support_bucket": support_bucket,
            "max_priority": int(max_priority),
            "min_confidence": float(min_confidence),
            "min_opposite_ratio": float(min_opposite_ratio),
            "min_nearest_similarity": float(min_nearest_similarity),
            "relabeling_allowed": False,
            "test_split_actions_allowed": False,
        },
        "split_summary": split_summary,
        "review_rows": len(review_rows),
        "planned_rows": len(planned_rows),
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "planned_split_counts": dict(sorted(split_counts.items())),
        "planned_label_counts": dict(sorted(label_counts.items())),
        "replacement_required": len(planned_rows),
        "replacement_counts_by_original_label": dict(sorted(replacement_counts.items())),
        "training_policy_rows": 0,
        "notes": [
            "This plan never relabels rows; it only excludes high-confidence Train/Val noise candidates.",
            "Every excluded row requires one fresh unused valid same-label replacement.",
            "Test split rows are ignored even if they match the evidence policy.",
        ],
    }
    return planned_rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build automatic same-label replacement plan for high-confidence noise.")
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-priority", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.95)
    parser.add_argument("--min-opposite-ratio", type=float, default=0.80)
    parser.add_argument("--min-nearest-similarity", type=float, default=0.0)
    parser.add_argument("--support-bucket", default="neighbors_support_model_prediction")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows, summary = build_automatic_plan(
        review_csv=args.review_csv,
        split_csv=args.split_csv,
        max_priority=args.max_priority,
        min_confidence=args.min_confidence,
        min_opposite_ratio=args.min_opposite_ratio,
        min_nearest_similarity=args.min_nearest_similarity,
        support_bucket=args.support_bucket,
    )
    write_csv_rows(args.output_csv, rows, FIELDNAMES)
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    summary["outputs"] = {"csv": str(resolve_path(args.output_csv)), "json": str(output_json)}
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
