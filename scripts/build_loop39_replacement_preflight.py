#!/usr/bin/env python3
"""Preflight Loop39 manual verdicts before any corrected split is built.

This script is read-only. It validates the manual adjudication queue, checks the
active split shape, and optionally audits a fresh replacement candidate pool. It
does not edit the split, cache, or raw files.
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

ALLOWED_VERDICTS = {"", "label_correct", "label_wrong", "feature_broken", "out_of_scope", "uncertain"}
ALLOWED_ACTIONS = {
    "",
    "keep_sample",
    "replace_with_fresh_same_label_candidate",
    "quarantine_for_more_evidence",
    "model_blindspot",
}
REPLACEMENT_VERDICTS = {"label_wrong", "feature_broken", "out_of_scope"}
KEEP_ACTIONS = {"keep_sample", "model_blindspot"}

DETAIL_FIELDNAMES = [
    "review_priority_rank",
    "source_path",
    "source_sha256",
    "sample_index",
    "split",
    "label",
    "manual_label_verdict",
    "recommended_action",
    "row_status",
    "replacement_required",
    "replacement_label",
    "reasons",
]

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from apply_manual_review_verdicts import source_keys, source_path_stem_sha  # noqa: E402


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_detail_csv(path: Path, rows: Sequence[dict]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDNAMES, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize(value: object) -> str:
    return str(value or "").strip().casefold()


def normalize_path(value: object) -> str:
    return str(value or "").strip().replace("/", "\\").casefold()


def exact_row_key(row: dict) -> str:
    return f"{str(row.get('sample_index', '')).strip()}|{normalize_path(row.get('source_path'))}"


def has_exact_identity(row: dict) -> bool:
    return bool(str(row.get("sample_index", "")).strip() and str(row.get("source_path", "")).strip())


def row_keys(row: dict) -> set[str]:
    keys = set(source_keys(row))
    source_path = normalize_path(row.get("source_path"))
    if source_path:
        keys.add(f"path:{source_path}")
    stem_sha = source_path_stem_sha(row)
    if stem_sha:
        keys.add(f"sha:{stem_sha}")
    return keys


def split_summary(rows: Sequence[dict]) -> dict:
    return {
        "rows": len(rows),
        "split_counts": dict(sorted(Counter(row.get("split", "") for row in rows).items())),
        "label_counts": dict(sorted(Counter(str(row.get("label", "")) for row in rows).items())),
        "label_split_counts": {
            split: dict(sorted(Counter(str(row.get("label", "")) for row in rows if row.get("split") == split).items()))
            for split in ["train", "val", "test"]
        },
    }


def split_shape_failures(
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


def build_split_index(split_rows: Sequence[dict]) -> dict:
    by_exact = {}
    by_sample_index = {}
    by_path = {}
    by_key: dict[str, list[dict]] = {}
    for row in split_rows:
        if has_exact_identity(row):
            by_exact.setdefault(exact_row_key(row), row)
        sample_index = str(row.get("sample_index", "")).strip()
        if sample_index:
            by_sample_index.setdefault(sample_index, row)
        path = normalize_path(row.get("source_path"))
        if path:
            by_path.setdefault(path, row)
        for key in row_keys(row):
            by_key.setdefault(key, []).append(row)
    return {
        "by_exact": by_exact,
        "by_sample_index": by_sample_index,
        "by_path": by_path,
        "by_key": by_key,
    }


def find_split_row(review_row: dict, split_index: dict) -> Optional[dict]:
    if has_exact_identity(review_row):
        row = split_index["by_exact"].get(exact_row_key(review_row))
        if row is not None:
            return row
    sample_index = str(review_row.get("sample_index", "")).strip()
    if sample_index:
        row = split_index["by_sample_index"].get(sample_index)
        if row is not None:
            return row
    path = normalize_path(review_row.get("source_path"))
    if path:
        row = split_index["by_path"].get(path)
        if row is not None:
            return row
    for key in row_keys(review_row):
        rows = split_index["by_key"].get(key)
        if rows:
            return rows[0]
    return None


def classify_review_row(review_row: dict, split_row: Optional[dict]) -> dict:
    verdict = normalize(review_row.get("manual_label_verdict"))
    action = normalize(review_row.get("recommended_action"))
    reasons: list[str] = []
    replacement_required = False
    row_status = "ready_no_replacement"

    if split_row is None:
        reasons.append("review row does not match the split")
        row_status = "missing_split_row"
    if verdict not in ALLOWED_VERDICTS:
        reasons.append(f"invalid manual_label_verdict={verdict or '<blank>'}")
        row_status = "invalid_manual_fields"
    if action not in ALLOWED_ACTIONS:
        reasons.append(f"invalid recommended_action={action or '<blank>'}")
        row_status = "invalid_manual_fields"

    if not reasons:
        if not verdict and not action:
            row_status = "blank_no_verdict"
        elif not verdict or not action:
            reasons.append("manual verdict/action pair is incomplete")
            row_status = "incomplete_manual_fields"
        elif verdict in REPLACEMENT_VERDICTS:
            if action != "replace_with_fresh_same_label_candidate":
                reasons.append(f"{verdict} must use replace_with_fresh_same_label_candidate")
                row_status = "inconsistent_manual_fields"
            else:
                replacement_required = True
                row_status = "replacement_required"
        elif verdict == "label_correct":
            if action not in KEEP_ACTIONS:
                reasons.append("label_correct must use keep_sample or model_blindspot")
                row_status = "inconsistent_manual_fields"
        elif verdict == "uncertain":
            if action != "quarantine_for_more_evidence":
                reasons.append("uncertain must use quarantine_for_more_evidence")
                row_status = "inconsistent_manual_fields"
            else:
                row_status = "quarantine_for_more_evidence"
        elif action == "replace_with_fresh_same_label_candidate":
            reasons.append("replacement action requires label_wrong, feature_broken, or out_of_scope")
            row_status = "inconsistent_manual_fields"

    label = str((split_row or review_row).get("label", "")).strip()
    split = str((split_row or review_row).get("split", "")).strip()
    return {
        "review_priority_rank": review_row.get("review_priority_rank", ""),
        "source_path": (split_row or review_row).get("source_path", review_row.get("source_path", "")),
        "source_sha256": review_row.get("source_sha256", ""),
        "sample_index": (split_row or review_row).get("sample_index", review_row.get("sample_index", "")),
        "split": split,
        "label": label,
        "manual_label_verdict": review_row.get("manual_label_verdict", ""),
        "recommended_action": review_row.get("recommended_action", ""),
        "row_status": row_status,
        "replacement_required": "true" if replacement_required else "false",
        "replacement_label": label if replacement_required else "",
        "reasons": "|".join(reasons),
    }


def candidate_pool_summary(
    *,
    candidate_csv: Path,
    split_rows: Sequence[dict],
    replacement_detail_rows: Sequence[dict],
) -> dict:
    candidate_rows = read_csv_rows(candidate_csv)
    original_keys: set[str] = set()
    for row in split_rows:
        original_keys.update(row_keys(row))
    excluded_keys: set[str] = set()
    for row in replacement_detail_rows:
        excluded_keys.update(row_keys(row))

    valid_by_label: Counter[str] = Counter()
    invalid_label_rows = 0
    missing_path_rows = 0
    already_used_rows = 0
    self_replacement_rows = 0
    duplicate_candidate_rows = 0
    seen_candidate_keys: set[str] = set()

    for row in candidate_rows:
        label = str(row.get("label", "")).strip()
        path = str(row.get("source_path", "")).strip()
        keys = row_keys(row)
        if label not in {"0", "1"}:
            invalid_label_rows += 1
            continue
        if not path:
            missing_path_rows += 1
            continue
        if keys & excluded_keys:
            self_replacement_rows += 1
            continue
        if keys & original_keys:
            already_used_rows += 1
            continue
        if keys & seen_candidate_keys:
            duplicate_candidate_rows += 1
            continue
        seen_candidate_keys.update(keys)
        valid_by_label[label] += 1

    required_by_label = Counter(row["replacement_label"] for row in replacement_detail_rows if row["replacement_required"] == "true")
    shortfall = {
        label: max(0, required_by_label.get(label, 0) - valid_by_label.get(label, 0))
        for label in sorted(set(required_by_label) | set(valid_by_label))
    }
    shortfall = {label: count for label, count in shortfall.items() if count > 0}
    return {
        "candidate_csv": str(resolve_path(candidate_csv)),
        "candidate_rows": len(candidate_rows),
        "valid_fresh_label_counts": dict(sorted(valid_by_label.items())),
        "required_replacement_label_counts": dict(sorted(required_by_label.items())),
        "replacement_shortfall": shortfall,
        "enough_fresh_same_label_candidates": not shortfall,
        "invalid_label_rows": invalid_label_rows,
        "missing_path_rows": missing_path_rows,
        "already_used_rows": already_used_rows,
        "self_replacement_rows": self_replacement_rows,
        "duplicate_candidate_rows": duplicate_candidate_rows,
    }


def choose_status(blocking_issues: Sequence[str]) -> str:
    if not blocking_issues:
        return "ready_for_corrected_split"
    priority = [
        "split_shape_invalid",
        "missing_split_rows",
        "invalid_manual_fields",
        "inconsistent_manual_fields",
        "incomplete_manual_fields",
        "blocked_no_verdicts",
        "candidate_pool_required",
        "candidate_pool_shortfall",
    ]
    for item in priority:
        if item in blocking_issues:
            return item
    return blocking_issues[0]


def build_preflight(
    *,
    review_csv: Path,
    split_csv: Path,
    candidate_csv: Optional[Path] = None,
    detail_output_csv: Optional[Path] = None,
    enforce_shape: bool = True,
    enforce_label_balance: bool = True,
    allow_partial_adjudication: bool = False,
) -> dict:
    review_rows = read_csv_rows(review_csv)
    split_rows = read_csv_rows(split_csv)
    split_index = build_split_index(split_rows)
    detail_rows = [classify_review_row(row, find_split_row(row, split_index)) for row in review_rows]

    status_counts = Counter(row["row_status"] for row in detail_rows)
    replacement_rows = [row for row in detail_rows if row["replacement_required"] == "true"]
    replacement_counts_by_label = Counter(row["replacement_label"] for row in replacement_rows)
    replacement_counts_by_split_label = Counter(f"{row['split']}:{row['replacement_label']}" for row in replacement_rows)
    split_failures = split_shape_failures(
        split_rows,
        enforce_shape=enforce_shape,
        enforce_label_balance=enforce_label_balance,
    )

    blocking_issues = []
    if split_failures:
        blocking_issues.append("split_shape_invalid")
    if status_counts["missing_split_row"]:
        blocking_issues.append("missing_split_rows")
    if status_counts["invalid_manual_fields"]:
        blocking_issues.append("invalid_manual_fields")
    if status_counts["inconsistent_manual_fields"]:
        blocking_issues.append("inconsistent_manual_fields")
    if status_counts["incomplete_manual_fields"]:
        blocking_issues.append("incomplete_manual_fields")
    if status_counts["blank_no_verdict"] == len(detail_rows) and detail_rows:
        blocking_issues.append("blocked_no_verdicts")
    elif status_counts["blank_no_verdict"] and not allow_partial_adjudication:
        blocking_issues.append("incomplete_manual_fields")

    candidate_summary = None
    if replacement_rows:
        if candidate_csv is None:
            blocking_issues.append("candidate_pool_required")
        else:
            candidate_summary = candidate_pool_summary(
                candidate_csv=candidate_csv,
                split_rows=split_rows,
                replacement_detail_rows=replacement_rows,
            )
            if candidate_summary["replacement_shortfall"]:
                blocking_issues.append("candidate_pool_shortfall")

    # Preserve first occurrence order while removing duplicates.
    blocking_issues = list(dict.fromkeys(blocking_issues))

    if detail_output_csv is not None:
        write_detail_csv(detail_output_csv, detail_rows)

    return {
        "schema": "axon_loop39_replacement_preflight_v1",
        "review_csv": str(resolve_path(review_csv)),
        "split_csv": str(resolve_path(split_csv)),
        "candidate_csv": str(resolve_path(candidate_csv)) if candidate_csv is not None else None,
        "enforce_shape": bool(enforce_shape),
        "enforce_label_balance": bool(enforce_label_balance),
        "allow_partial_adjudication": bool(allow_partial_adjudication),
        "review_rows": len(review_rows),
        "split_summary": split_summary(split_rows),
        "split_shape_failures": split_failures,
        "row_status_counts": dict(sorted(status_counts.items())),
        "blank_manual_rows": int(status_counts["blank_no_verdict"]),
        "replacement_required": len(replacement_rows),
        "replacement_counts_by_label": dict(sorted(replacement_counts_by_label.items())),
        "replacement_counts_by_split_label": dict(sorted(replacement_counts_by_split_label.items())),
        "candidate_summary": candidate_summary,
        "blocking_issues": blocking_issues,
        "preflight_status": choose_status(blocking_issues),
        "preflight_ok": not blocking_issues,
        "detail_output_csv": str(resolve_path(detail_output_csv)) if detail_output_csv is not None else None,
        "notes": [
            "This preflight is read-only and does not edit the split, cache, or raw files.",
            "Loop39 feature_broken/out_of_scope/label_wrong rows require fresh same-label replacement candidates.",
            "Paths, filenames, extensions, hashes, sample ids, split names, and row order are identity/audit fields only.",
            "A corrected split must still be exactly 200000 rows with the 20000/20000/160000 split.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight Loop39 replacement readiness.")
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--detail-output-csv", type=Path, default=None)
    parser.add_argument("--no-enforce-shape", action="store_true")
    parser.add_argument("--no-enforce-label-balance", action="store_true")
    parser.add_argument("--allow-partial-adjudication", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless preflight_ok is true.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_preflight(
        review_csv=args.review_csv,
        split_csv=args.split_csv,
        candidate_csv=args.candidate_csv,
        detail_output_csv=args.detail_output_csv,
        enforce_shape=not bool(args.no_enforce_shape),
        enforce_label_balance=not bool(args.no_enforce_label_balance),
        allow_partial_adjudication=bool(args.allow_partial_adjudication),
    )
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.strict and not payload["preflight_ok"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
