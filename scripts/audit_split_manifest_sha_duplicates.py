#!/usr/bin/env python3
"""Audit split rows for duplicate content SHA using the feature-cache manifest.

This is stricter than source-path/stem checks because the split CSV may not
carry an explicit source_sha256 column. Identity fields are used only for
alignment and duplicate/content-group auditing, never as model evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DETAIL_FIELDNAMES = [
    "duplicate_group_id",
    "manifest_source_sha256",
    "group_size",
    "labels",
    "splits",
    "cross_label",
    "cross_split",
    "in_focus_queue",
    "focus_queue_rows",
    "source_path",
    "split_source_sha256",
    "manifest_cache_path",
    "label",
    "sample_index",
    "split",
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_manifest_rows(path: Path) -> list[dict]:
    data = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    rows = data.get("samples", data if isinstance(data, list) else [])
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported manifest format: {path}")
    return [dict(row) for row in rows]


def normalize_path(value: object) -> str:
    return str(value or "").strip().replace("/", "\\").casefold()


def normalize_sha(value: object) -> str:
    return str(value or "").strip().casefold()


def path_stem_sha(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    name = Path(text).name.casefold()
    stem = Path(name).stem if "." in name else name
    if len(stem) == 64 and all(char in "0123456789abcdef" for char in stem):
        return stem
    return ""


def split_keys(row: dict) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    sha = normalize_sha(row.get("source_sha256"))
    if sha:
        keys.append(("sha", sha))
    path = normalize_path(row.get("source_path"))
    if path:
        keys.append(("path", path))
    stem = path_stem_sha(row.get("source_path"))
    if stem:
        keys.append(("sha", stem))
    return keys


def manifest_keys(row: dict) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    sha = normalize_sha(row.get("source_sha256"))
    if sha:
        keys.append(("sha", sha))
    path = normalize_path(row.get("source_path"))
    if path:
        keys.append(("path", path))
    stem = path_stem_sha(row.get("source_path"))
    if stem:
        keys.append(("sha", stem))
    return keys


def build_manifest_lookup(rows: Sequence[dict]) -> dict[tuple[str, str], dict]:
    lookup: dict[tuple[str, str], dict] = {}
    for row in rows:
        for key in manifest_keys(row):
            lookup.setdefault(key, row)
    return lookup


def focus_keys(rows: Sequence[dict]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        keys.update(split_keys(row))
    return keys


def group_flags(rows: Sequence[dict]) -> dict:
    labels = sorted({str(row.get("label", "")) for row in rows})
    splits = sorted({str(row.get("split", "")) for row in rows})
    return {
        "labels": labels,
        "splits": splits,
        "cross_label": len(labels) > 1,
        "cross_split": len(splits) > 1,
    }


def write_detail_rows(path: Path, rows: Sequence[dict]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDNAMES, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit_manifest_sha_duplicates(
    *,
    split_csv: Path,
    manifest_json: Path,
    output_csv: Optional[Path] = None,
    focus_queue_csv: Optional[Path] = None,
) -> dict:
    split_rows = read_csv_rows(split_csv)
    manifest_rows = load_manifest_rows(manifest_json)
    manifest_lookup = build_manifest_lookup(manifest_rows)
    focus = focus_keys(read_csv_rows(focus_queue_csv)) if focus_queue_csv is not None else set()

    enriched = []
    missing_manifest = []
    match_methods: Counter[str] = Counter()
    for row in split_rows:
        manifest_row = None
        match_method = "missing"
        for key in split_keys(row):
            manifest_row = manifest_lookup.get(key)
            if manifest_row is not None:
                match_method = key[0]
                break
        if manifest_row is None:
            missing_manifest.append(row)
            continue
        match_methods[match_method] += 1
        content_sha = normalize_sha(manifest_row.get("source_sha256"))
        item = {
            **row,
            "manifest_source_sha256": content_sha,
            "manifest_cache_path": manifest_row.get("cache_path", ""),
            "in_focus_queue": any(key in focus for key in split_keys(row)),
        }
        enriched.append(item)

    groups_by_sha: dict[str, list[dict]] = defaultdict(list)
    for row in enriched:
        sha = row.get("manifest_source_sha256", "")
        if sha:
            groups_by_sha[sha].append(row)
    duplicate_groups = [(sha, rows) for sha, rows in sorted(groups_by_sha.items()) if len(rows) > 1]

    detail_rows = []
    for group_index, (sha, rows) in enumerate(duplicate_groups, start=1):
        flags = group_flags(rows)
        focus_count = sum(1 for row in rows if row.get("in_focus_queue"))
        for row in rows:
            detail_rows.append(
                {
                    "duplicate_group_id": group_index,
                    "manifest_source_sha256": sha,
                    "group_size": len(rows),
                    "labels": "|".join(flags["labels"]),
                    "splits": "|".join(flags["splits"]),
                    "cross_label": str(flags["cross_label"]).lower(),
                    "cross_split": str(flags["cross_split"]).lower(),
                    "in_focus_queue": str(bool(row.get("in_focus_queue"))).lower(),
                    "focus_queue_rows": focus_count,
                    "source_path": row.get("source_path", ""),
                    "split_source_sha256": row.get("source_sha256", ""),
                    "manifest_cache_path": row.get("manifest_cache_path", ""),
                    "label": row.get("label", ""),
                    "sample_index": row.get("sample_index", ""),
                    "split": row.get("split", ""),
                }
            )
    if output_csv is not None:
        write_detail_rows(output_csv, detail_rows)

    group_summaries = [group_flags(rows) | {"sha": sha, "size": len(rows)} for sha, rows in duplicate_groups]
    focus_duplicate_groups = [
        (sha, rows) for sha, rows in duplicate_groups if any(row.get("in_focus_queue") for row in rows)
    ]
    payload = {
        "schema": "axon_split_manifest_sha_duplicate_audit_v1",
        "split_csv": str(resolve_path(split_csv)),
        "manifest_json": str(resolve_path(manifest_json)),
        "focus_queue_csv": str(resolve_path(focus_queue_csv)) if focus_queue_csv is not None else None,
        "split_rows": len(split_rows),
        "manifest_rows": len(manifest_rows),
        "matched_rows": len(enriched),
        "missing_manifest_rows": len(missing_manifest),
        "manifest_match_methods": dict(sorted(match_methods.items())),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_extra_rows": sum(len(rows) - 1 for _sha, rows in duplicate_groups),
        "duplicate_detail_rows": len(detail_rows),
        "cross_label_groups": sum(1 for item in group_summaries if item["cross_label"]),
        "cross_split_groups": sum(1 for item in group_summaries if item["cross_split"]),
        "focus_duplicate_groups": len(focus_duplicate_groups),
        "focus_duplicate_detail_rows": sum(len(rows) for _sha, rows in focus_duplicate_groups),
        "focus_cross_label_groups": sum(1 for _sha, rows in focus_duplicate_groups if group_flags(rows)["cross_label"]),
        "focus_cross_split_groups": sum(1 for _sha, rows in focus_duplicate_groups if group_flags(rows)["cross_split"]),
        "label_pattern_counts": dict(
            sorted(Counter("|".join(item["labels"]) for item in group_summaries).items())
        ),
        "split_pattern_counts": dict(
            sorted(Counter("|".join(item["splits"]) for item in group_summaries).items())
        ),
        "detail_csv": str(resolve_path(output_csv)) if output_csv is not None else None,
        "has_duplicates": bool(duplicate_groups),
        "has_cross_label_duplicates": any(item["cross_label"] for item in group_summaries),
        "has_cross_split_duplicates": any(item["cross_split"] for item in group_summaries),
        "notes": [
            "Duplicate groups are based on manifest/cache source_sha256.",
            "This audit is read-only and does not change labels, split rows, or cache files.",
            "focus_queue overlap is for manual/data triage only, not model evidence.",
        ],
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit split duplicate content SHA using cache manifest.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--focus-queue-csv", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when duplicate groups exist.")
    parser.add_argument("--strict-cross-label", action="store_true")
    parser.add_argument("--strict-cross-split", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_manifest_sha_duplicates(
        split_csv=args.split_csv,
        manifest_json=args.manifest_json,
        output_csv=args.output_csv,
        focus_queue_csv=args.focus_queue_csv,
    )
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.strict and payload["has_duplicates"]:
        return 2
    if args.strict_cross_label and payload["has_cross_label_duplicates"]:
        return 2
    if args.strict_cross_split and payload["has_cross_split_duplicates"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
