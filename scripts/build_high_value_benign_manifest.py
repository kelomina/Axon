#!/usr/bin/env python3
"""Build a non-destructive manifest for high-value benign validation rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


def _is_high_value_benign(row: dict) -> bool:
    if str(row.get("label", "")).strip() != "0":
        return False
    source_path = str(row.get("source_path", ""))
    return "待加入白名单" in source_path or "whitelist" in source_path.casefold()


def _read_csv_rows(path: Path, source_name: str) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _is_high_value_benign(row):
                clean = {
                    "source_path": row.get("source_path", ""),
                    "cache_path": row.get("cache_path", ""),
                    "label": row.get("label", ""),
                    "split": row.get("split", ""),
                    "sample_index": row.get("sample_index", ""),
                    "manifest_sources": source_name,
                }
                rows.append(clean)
    return rows


def _dedupe_rows(groups: Sequence[tuple[Path, str]]) -> list[dict]:
    by_key: dict[tuple[str, str], dict] = {}
    for path, source_name in groups:
        for row in _read_csv_rows(path, source_name):
            key = (row["source_path"], row.get("cache_path", ""))
            if key in by_key:
                sources = set(filter(None, by_key[key]["manifest_sources"].split(";")))
                sources.add(source_name)
                by_key[key]["manifest_sources"] = ";".join(sorted(sources))
            else:
                by_key[key] = row
    return sorted(by_key.values(), key=lambda row: (row["source_path"], row.get("cache_path", "")))


def _summarize(rows: list[dict]) -> dict:
    source_counts = Counter()
    parent_counts = Counter()
    cache_present = 0
    cache_missing = 0
    for row in rows:
        for source in filter(None, row["manifest_sources"].split(";")):
            source_counts[source] += 1
        parent_counts[str(Path(row["source_path"]).parent)] += 1
        cache_path = row.get("cache_path")
        if cache_path:
            if Path(cache_path).exists():
                cache_present += 1
            else:
                cache_missing += 1

    return {
        "rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "top_parent_dirs": [
            {"parent_dir": parent, "count": count}
            for parent, count in parent_counts.most_common(10)
        ],
        "cache_path_present": cache_present,
        "cache_path_missing": cache_missing,
    }


def build_manifest(
    *,
    official_test_missing: Path,
    hard_error_missing: Path,
    hard_error_predictions: Path,
) -> tuple[list[dict], dict]:
    rows = _dedupe_rows(
        [
            (official_test_missing, "official_test_missing_cache"),
            (hard_error_missing, "hard_error_missing_cache"),
            (hard_error_predictions, "hard_error_current_subset_predictions"),
        ]
    )
    summary = {
        "schema": "axon_high_value_benign_manifest_v1",
        "inputs": {
            "official_test_missing": str(official_test_missing),
            "hard_error_missing": str(hard_error_missing),
            "hard_error_predictions": str(hard_error_predictions),
        },
        "summary": _summarize(rows),
    }
    return rows, summary


def write_manifest_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_path", "cache_path", "label", "split", "sample_index", "manifest_sources"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build high-value benign manifest for validation planning.")
    parser.add_argument("--official-test-missing", type=Path, required=True)
    parser.add_argument("--hard-error-missing", type=Path, required=True)
    parser.add_argument("--hard-error-predictions", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows, summary = build_manifest(
        official_test_missing=args.official_test_missing,
        hard_error_missing=args.hard_error_missing,
        hard_error_predictions=args.hard_error_predictions,
    )
    write_manifest_csv(args.output_csv, rows)
    summary["outputs"] = {"csv": str(args.output_csv), "json": str(args.output_json)}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"CSV: {args.output_csv}")
    print(f"JSON: {args.output_json}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
