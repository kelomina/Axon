#!/usr/bin/env python3
"""Audit duplicate source identities inside a split CSV.

The split CSV only guarantees rows. This guard checks whether rows point to the
same source identity by path or by sha-like source identity inferred from the
path/optional source_sha256 columns.
"""

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

from apply_manual_review_verdicts import source_keys  # noqa: E402


DETAIL_FIELDNAMES = [
    "duplicate_group_id",
    "duplicate_key",
    "group_size",
    "labels",
    "splits",
    "cross_label",
    "cross_split",
    "same_path_rows",
    "source_path",
    "source_sha256",
    "label",
    "sample_index",
    "split",
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


def normalize_path(path_text: str) -> str:
    return str(Path(path_text)).replace("/", "\\").casefold()


def identity_keys(row: dict, *, include_path: bool = True, include_sha: bool = True) -> set[str]:
    keys = set(source_keys(row))
    if not include_path:
        keys = {key for key in keys if key.startswith("sha:")}
    if not include_sha:
        keys = {key for key in keys if key.startswith("path:")}
    source_path = str(row.get("source_path") or "").strip()
    if include_path and source_path:
        keys.add(f"path:{normalize_path(source_path)}")
    return keys


def path_key(row: dict) -> str:
    return f"path:{normalize_path(str(row.get('source_path') or '').strip())}"


def build_groups(rows: Sequence[dict], *, include_path: bool, include_sha: bool) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for key in identity_keys(row, include_path=include_path, include_sha=include_sha):
            grouped[key].append(row)
    return {key: value for key, value in grouped.items() if len(value) > 1}


def canonical_duplicate_groups(duplicate_key_groups: dict[str, list[dict]]) -> list[tuple[str, list[dict]]]:
    seen_row_sets: set[tuple[int, ...]] = set()
    groups: list[tuple[str, list[dict]]] = []
    for key in sorted(duplicate_key_groups):
        rows = duplicate_key_groups[key]
        row_ids = tuple(sorted(id(row) for row in rows))
        if row_ids in seen_row_sets:
            continue
        seen_row_sets.add(row_ids)
        groups.append((key, rows))
    return groups


def group_flags(rows: Sequence[dict]) -> dict:
    labels = sorted({str(row.get("label", "")) for row in rows})
    splits = sorted({str(row.get("split", "")) for row in rows})
    path_counts = Counter(path_key(row) for row in rows)
    return {
        "labels": labels,
        "splits": splits,
        "cross_label": len(labels) > 1,
        "cross_split": len(splits) > 1,
        "same_path_rows": any(count > 1 for count in path_counts.values()),
    }


def summarize_groups(groups: Sequence[tuple[str, list[dict]]]) -> dict:
    group_count = len(groups)
    duplicate_rows = sum(len(rows) - 1 for _key, rows in groups)
    cross_label_groups = 0
    cross_split_groups = 0
    same_path_groups = 0
    split_pair_counts: Counter[str] = Counter()
    label_pattern_counts: Counter[str] = Counter()
    split_pattern_counts: Counter[str] = Counter()
    max_group_size = 0
    for _key, rows in groups:
        flags = group_flags(rows)
        max_group_size = max(max_group_size, len(rows))
        if flags["cross_label"]:
            cross_label_groups += 1
        if flags["cross_split"]:
            cross_split_groups += 1
        if flags["same_path_rows"]:
            same_path_groups += 1
        label_pattern_counts["|".join(flags["labels"])] += 1
        split_pattern_counts["|".join(flags["splits"])] += 1
        if flags["cross_split"]:
            split_pair_counts["|".join(flags["splits"])] += 1
    return {
        "duplicate_groups": group_count,
        "duplicate_extra_rows": duplicate_rows,
        "max_group_size": max_group_size,
        "cross_label_groups": cross_label_groups,
        "cross_split_groups": cross_split_groups,
        "same_path_duplicate_groups": same_path_groups,
        "label_pattern_counts": dict(sorted(label_pattern_counts.items())),
        "split_pattern_counts": dict(sorted(split_pattern_counts.items())),
        "cross_split_pattern_counts": dict(sorted(split_pair_counts.items())),
    }


def build_detail_rows(groups: Sequence[tuple[str, list[dict]]]) -> list[dict]:
    details: list[dict] = []
    for group_index, (key, rows) in enumerate(groups, start=1):
        flags = group_flags(rows)
        for row in rows:
            details.append(
                {
                    "duplicate_group_id": group_index,
                    "duplicate_key": key,
                    "group_size": len(rows),
                    "labels": "|".join(flags["labels"]),
                    "splits": "|".join(flags["splits"]),
                    "cross_label": str(flags["cross_label"]).lower(),
                    "cross_split": str(flags["cross_split"]).lower(),
                    "same_path_rows": str(flags["same_path_rows"]).lower(),
                    "source_path": row.get("source_path", ""),
                    "source_sha256": row.get("source_sha256", ""),
                    "label": row.get("label", ""),
                    "sample_index": row.get("sample_index", ""),
                    "split": row.get("split", ""),
                }
            )
    return details


def split_summary(rows: Sequence[dict]) -> dict:
    return {
        "rows": len(rows),
        "split_counts": dict(sorted(Counter(row.get("split", "") for row in rows).items())),
        "label_split_counts": {
            split: dict(sorted(Counter(str(row.get("label", "")) for row in rows if row.get("split") == split).items()))
            for split in ["train", "val", "test"]
        },
    }


def audit_split_duplicate_sources(
    *,
    split_csv: Path,
    output_csv: Optional[Path] = None,
    include_path: bool = True,
    include_sha: bool = True,
) -> dict:
    rows = read_csv_rows(split_csv)
    key_groups = build_groups(rows, include_path=include_path, include_sha=include_sha)
    groups = canonical_duplicate_groups(key_groups)
    detail_rows = build_detail_rows(groups)
    if output_csv is not None:
        write_detail_rows(output_csv, detail_rows)

    summary = summarize_groups(groups)
    payload = {
        "schema": "axon_split_duplicate_source_audit_v1",
        "split_csv": str(resolve_path(split_csv)),
        "include_path": bool(include_path),
        "include_sha": bool(include_sha),
        "split_summary": split_summary(rows),
        **summary,
        "detail_rows": len(detail_rows),
        "detail_csv": str(resolve_path(output_csv)) if output_csv is not None else None,
        "has_duplicates": summary["duplicate_groups"] > 0,
        "has_cross_split_duplicates": summary["cross_split_groups"] > 0,
        "has_cross_label_duplicates": summary["cross_label_groups"] > 0,
        "notes": [
            "Duplicate groups are based on normalized source_path and sha-like identity keys.",
            "cross_split duplicates are potential validation/test leakage risks.",
            "cross_label duplicates are potential label-noise conflicts.",
            "This audit is read-only and does not modify the split.",
        ],
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit duplicate source identities inside a split CSV.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--path-only", action="store_true", help="Only use normalized source_path identity keys.")
    parser.add_argument("--sha-only", action="store_true", help="Only use explicit/sha-like source identity keys.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when duplicate groups exist.")
    parser.add_argument(
        "--strict-cross-split",
        action="store_true",
        help="Exit non-zero only when duplicates cross split boundaries.",
    )
    parser.add_argument(
        "--strict-cross-label",
        action="store_true",
        help="Exit non-zero only when duplicates cross labels.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.path_only and args.sha_only:
        raise ValueError("--path-only and --sha-only are mutually exclusive")
    payload = audit_split_duplicate_sources(
        split_csv=args.split_csv,
        output_csv=args.output_csv,
        include_path=not bool(args.sha_only),
        include_sha=not bool(args.path_only),
    )
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.strict and payload["has_duplicates"]:
        return 2
    if args.strict_cross_split and payload["has_cross_split_duplicates"]:
        return 2
    if args.strict_cross_label and payload["has_cross_label_duplicates"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
