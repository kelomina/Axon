#!/usr/bin/env python3
"""Audit feature-cache coverage for a split CSV without running the model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_split_from_cache import (  # noqa: E402
    MISSING_CACHE_FIELDNAMES,
    iter_split_rows,
    load_manifest_lookup,
    lookup_manifest_sample,
    resolve_path,
)


def audit_split_cache_coverage(
    *,
    split_csv: Path,
    manifest_path: Path,
    split: Optional[str],
    output_json: Path,
    missing_cache_output: Optional[Path],
) -> dict:
    split_csv = resolve_path(split_csv)
    manifest_path = resolve_path(manifest_path)
    output_json = resolve_path(output_json)
    missing_cache_output = resolve_path(missing_cache_output) if missing_cache_output is not None else None

    manifest_by_source, manifest_by_sha = load_manifest_lookup(manifest_path)
    rows = iter_split_rows(split_csv, split)
    cache_dir = manifest_path.parent

    total_rows = 0
    missing_rows = 0
    match_counts: Counter[str] = Counter()
    missing_label_counts: Counter[str] = Counter()
    missing_split_counts: Counter[str] = Counter()
    missing_reason_counts: Counter[str] = Counter()

    missing_handle = None
    missing_writer = None
    if missing_cache_output is not None:
        missing_cache_output.parent.mkdir(parents=True, exist_ok=True)
        missing_handle = missing_cache_output.open("w", encoding="utf-8-sig", newline="")
        missing_writer = csv.DictWriter(missing_handle, fieldnames=MISSING_CACHE_FIELDNAMES, extrasaction="ignore")
        missing_writer.writeheader()

    try:
        for row in rows:
            total_rows += 1
            sample, match_reason = lookup_manifest_sample(row, manifest_by_source, manifest_by_sha)
            missing_reason = None
            if sample is None:
                missing_reason = match_reason
            else:
                cache_path = Path(sample["cache_path"])
                if not cache_path.is_absolute():
                    cache_path = cache_dir / cache_path.name
                if not cache_path.exists():
                    missing_reason = "cache_file_missing"

            if missing_reason is not None:
                missing_rows += 1
                missing = {**row, "reason": missing_reason}
                missing_label_counts[str(row.get("label", ""))] += 1
                missing_split_counts[str(row.get("split", ""))] += 1
                missing_reason_counts[missing_reason] += 1
                if missing_writer is not None:
                    missing_writer.writerow(missing)
                continue

            match_counts[match_reason] += 1
    finally:
        if missing_handle is not None:
            missing_handle.close()

    covered = total_rows - missing_rows
    payload = {
        "schema": "axon_split_cache_coverage_audit_v1",
        "split_csv": str(split_csv),
        "manifest": str(manifest_path),
        "split": split or "all",
        "total_rows": total_rows,
        "covered_rows": covered,
        "missing_rows": missing_rows,
        "coverage_ratio": float(covered / total_rows) if total_rows else 0.0,
        "manifest_match_counts": dict(sorted(match_counts.items())),
        "missing_label_counts": dict(sorted(missing_label_counts.items())),
        "missing_split_counts": dict(sorted(missing_split_counts.items())),
        "missing_reason_counts": dict(sorted(missing_reason_counts.items())),
        "missing_cache_output": str(missing_cache_output) if missing_cache_output is not None else None,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit split CSV coverage in a feature-cache manifest.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--missing-cache-output", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_split_cache_coverage(
        split_csv=args.split_csv,
        manifest_path=args.manifest,
        split=args.split,
        output_json=args.output_json,
        missing_cache_output=args.missing_cache_output,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"JSON: {resolve_path(args.output_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
