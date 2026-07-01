#!/usr/bin/env python3
"""Strict cache readiness audit for a corrected 20w split."""

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


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_manifest_lookup(manifest_json: Path) -> dict[str, dict]:
    manifest = json.loads(resolve_path(manifest_json).read_text(encoding="utf-8"))
    lookup: dict[str, dict] = {}
    for sample in manifest.get("samples", []):
        row = {
            "source_path": sample.get("source_path", ""),
            "source_sha256": sample.get("source_sha256", ""),
        }
        for key in source_keys(row):
            lookup.setdefault(key, sample)
    return lookup


def lookup_sample(row: dict, manifest_lookup: dict[str, dict]) -> tuple[Optional[dict], str]:
    for key in source_keys(row):
        sample = manifest_lookup.get(key)
        if sample is not None:
            if key.startswith("sha:"):
                return sample, "source_sha256"
            return sample, "source_path"
    return None, "manifest_missing"


def split_summary(rows: Sequence[dict]) -> dict:
    return {
        "rows": len(rows),
        "split_counts": dict(sorted(Counter(row.get("split", "") for row in rows).items())),
        "label_split_counts": {
            split: dict(sorted(Counter(str(row.get("label", "")) for row in rows if row.get("split") == split).items()))
            for split in ["train", "val", "test"]
        },
    }


def write_missing_rows(path: Path, rows: Sequence[dict]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "label",
        "sample_index",
        "split",
        "reason",
        "expected_cache_path",
    ]
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_split_shape(
    rows: Sequence[dict],
    *,
    expected_total: int = EXPECTED_TOTAL,
    expected_split_counts: Optional[dict[str, int]] = None,
    expected_label_split_counts: Optional[dict[str, dict[str, int]]] = None,
) -> list[str]:
    summary = split_summary(rows)
    split_targets = EXPECTED_SPLIT_COUNTS if expected_split_counts is None else expected_split_counts
    failures = []
    if summary["rows"] != expected_total:
        failures.append(f"expected {expected_total} rows, got {summary['rows']}")
    if summary["split_counts"] != split_targets:
        failures.append(f"split_counts mismatch: {summary['split_counts']}")
    if expected_label_split_counts is not None and summary["label_split_counts"] != expected_label_split_counts:
        failures.append(f"label_split_counts mismatch: {summary['label_split_counts']}")
    return failures


def audit_corrected_split_cache_ready(
    *,
    split_csv: Path,
    manifest_json: Path,
    missing_cache_output: Optional[Path] = None,
    enforce_shape: bool = True,
    enforce_label_balance: bool = False,
) -> dict:
    rows = read_csv_rows(split_csv)
    manifest_lookup = load_manifest_lookup(manifest_json)
    manifest_dir = resolve_path(manifest_json).parent
    missing_rows: list[dict] = []
    match_counts: Counter[str] = Counter()
    missing_label_counts: Counter[str] = Counter()
    missing_split_counts: Counter[str] = Counter()
    missing_reason_counts: Counter[str] = Counter()

    for row in rows:
        sample, reason = lookup_sample(row, manifest_lookup)
        if sample is None:
            missing = {**row, "reason": reason, "expected_cache_path": ""}
            missing_rows.append(missing)
            missing_label_counts[str(row.get("label", ""))] += 1
            missing_split_counts[str(row.get("split", ""))] += 1
            missing_reason_counts[reason] += 1
            continue

        cache_path = Path(sample.get("cache_path", ""))
        if not cache_path.is_absolute():
            cache_path = manifest_dir / cache_path.name
        if not cache_path.exists():
            missing = {**row, "reason": "cache_file_missing", "expected_cache_path": str(cache_path)}
            missing_rows.append(missing)
            missing_label_counts[str(row.get("label", ""))] += 1
            missing_split_counts[str(row.get("split", ""))] += 1
            missing_reason_counts["cache_file_missing"] += 1
            continue

        match_counts[reason] += 1

    if missing_cache_output is not None:
        write_missing_rows(missing_cache_output, missing_rows)

    covered = len(rows) - len(missing_rows)
    shape_failures = (
        validate_split_shape(
            rows,
            expected_total=EXPECTED_TOTAL,
            expected_split_counts=EXPECTED_SPLIT_COUNTS,
            expected_label_split_counts=EXPECTED_LABEL_SPLIT_COUNTS if enforce_label_balance else None,
        )
        if enforce_shape
        else []
    )
    label_balance_drift = []
    if not enforce_label_balance:
        actual_label_split_counts = split_summary(rows)["label_split_counts"]
        if actual_label_split_counts != EXPECTED_LABEL_SPLIT_COUNTS:
            label_balance_drift = [
                f"{split}:{actual_label_split_counts.get(split, {})}"
                for split in ["train", "val", "test"]
                if actual_label_split_counts.get(split, {}) != EXPECTED_LABEL_SPLIT_COUNTS.get(split, {})
            ]
    payload = {
        "schema": "axon_corrected_split_cache_ready_v1",
        "split_csv": str(resolve_path(split_csv)),
        "manifest_json": str(resolve_path(manifest_json)),
        "split_summary": split_summary(rows),
        "expected_total": EXPECTED_TOTAL,
        "expected_split_counts": EXPECTED_SPLIT_COUNTS,
        "expected_label_split_counts": EXPECTED_LABEL_SPLIT_COUNTS,
        "total_rows": len(rows),
        "covered_rows": covered,
        "missing_rows": len(missing_rows),
        "coverage_ratio": float(covered / len(rows)) if rows else 0.0,
        "manifest_match_counts": dict(sorted(match_counts.items())),
        "missing_label_counts": dict(sorted(missing_label_counts.items())),
        "missing_split_counts": dict(sorted(missing_split_counts.items())),
        "missing_reason_counts": dict(sorted(missing_reason_counts.items())),
        "missing_cache_output": str(resolve_path(missing_cache_output)) if missing_cache_output is not None else None,
        "label_balance_enforced": bool(enforce_label_balance),
        "label_balance_drift": label_balance_drift,
        "shape_failures": shape_failures,
        "cache_ready": not shape_failures and not missing_rows,
        "notes": [
            "cache_ready=true is required before training from a corrected split.",
            "Missing rows should be passed to the cache recovery/extraction flow before rerunning training.",
            "Label balance is reported separately unless --enforce-label-balance is set.",
        ],
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit corrected split cache readiness.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--missing-cache-output", type=Path, default=None)
    parser.add_argument("--no-enforce-shape", action="store_true")
    parser.add_argument(
        "--enforce-label-balance",
        action="store_true",
        help="Also require the corrected split to preserve the original per-split class balance.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless shape and cache coverage are complete.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_corrected_split_cache_ready(
        split_csv=args.split_csv,
        manifest_json=args.manifest_json,
        missing_cache_output=args.missing_cache_output,
        enforce_shape=not bool(args.no_enforce_shape),
        enforce_label_balance=bool(args.enforce_label_balance),
    )
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.strict and not payload["cache_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
