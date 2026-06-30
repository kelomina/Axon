#!/usr/bin/env python3
"""Build an unused raw-PE candidate pool for manual-review replacements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELDNAMES = ["source_path", "label", "source_sha256", "cache_present", "cache_path"]

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from apply_manual_review_verdicts import source_keys  # noqa: E402
from build_random_20w_split import is_valid_pe_sample, iter_sorted_files  # noqa: E402


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha_from_path(path: Path) -> str:
    name = path.name.casefold()
    stem = Path(name).stem if "." in name else name
    if len(stem) == 64 and all(char in "0123456789abcdef" for char in stem):
        return stem
    return ""


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_keys(row: dict) -> set[str]:
    return set(source_keys(row))


def used_split_keys(split_csv: Path) -> set[str]:
    keys: set[str] = set()
    for row in read_csv_rows(split_csv):
        keys.update(_row_keys(row))
    return keys


def manifest_cache_index(manifest_json: Optional[Path]) -> dict[str, dict]:
    if manifest_json is None:
        return {}
    manifest = json.loads(resolve_path(manifest_json).read_text(encoding="utf-8"))
    index: dict[str, dict] = {}
    for sample in manifest.get("samples", []):
        row = {
            "source_path": sample.get("source_path", ""),
            "source_sha256": sample.get("source_sha256", ""),
        }
        for key in _row_keys(row):
            index.setdefault(key, sample)
    return index


def _cache_lookup(row: dict, cache_index: dict[str, dict]) -> tuple[bool, str]:
    for key in _row_keys(row):
        sample = cache_index.get(key)
        if sample is not None:
            return True, str(sample.get("cache_path", ""))
    return False, ""


def _scan_label_root(
    root: Path,
    *,
    label: int,
    used_keys: set[str],
    cache_index: dict[str, dict],
    max_file_size: int,
    hash_files: bool,
    max_candidates: Optional[int],
) -> tuple[list[dict], dict]:
    candidates: list[dict] = []
    files_seen = 0
    valid_pe = 0
    already_used = 0
    duplicate_candidates = 0
    seen_candidate_keys: set[str] = set()
    for path in iter_sorted_files(root):
        files_seen += 1
        if not path.is_file() or not is_valid_pe_sample(path, max_file_size):
            continue
        valid_pe += 1
        source_sha256 = _sha_from_path(path)
        if not source_sha256 and hash_files:
            source_sha256 = _hash_file(path)
        row = {
            "source_path": str(path),
            "label": str(label),
            "source_sha256": source_sha256,
        }
        keys = _row_keys(row)
        if keys & used_keys:
            already_used += 1
            continue
        if keys & seen_candidate_keys:
            duplicate_candidates += 1
            continue
        cache_present, cache_path = _cache_lookup(row, cache_index)
        row["cache_present"] = "true" if cache_present else "false"
        row["cache_path"] = cache_path
        candidates.append(row)
        seen_candidate_keys.update(keys)
        if max_candidates is not None and len(candidates) >= max_candidates:
            break
    summary = {
        "files_seen": files_seen,
        "valid_pe": valid_pe,
        "already_used_in_split": already_used,
        "duplicate_candidates": duplicate_candidates,
        "available_candidates": len(candidates),
        "cache_present_candidates": int(sum(row["cache_present"] == "true" for row in candidates)),
    }
    return candidates, summary


def build_candidate_pool(
    *,
    data_dir: Path,
    split_csv: Path,
    manifest_json: Optional[Path] = None,
    max_file_size: int = 1 * 1024 * 1024 * 1024,
    hash_files: bool = False,
    max_candidates_per_label: Optional[int] = None,
    required_label0: int = 0,
    required_label1: int = 0,
) -> tuple[list[dict], dict]:
    data_dir = resolve_path(data_dir)
    used_keys = used_split_keys(split_csv)
    cache_index = manifest_cache_index(manifest_json)
    roots = {
        "0": data_dir / "待加入白名单",
        "1": data_dir / "待拉黑",
    }
    rows: list[dict] = []
    per_label = {}
    for label, root in roots.items():
        if not root.exists():
            per_label[label] = {
                "root": str(root),
                "missing_root": True,
                "files_seen": 0,
                "valid_pe": 0,
                "already_used_in_split": 0,
                "duplicate_candidates": 0,
                "available_candidates": 0,
                "cache_present_candidates": 0,
            }
            continue
        label_rows, label_summary = _scan_label_root(
            root,
            label=int(label),
            used_keys=used_keys,
            cache_index=cache_index,
            max_file_size=max_file_size,
            hash_files=hash_files,
            max_candidates=max_candidates_per_label,
        )
        rows.extend(label_rows)
        per_label[label] = {"root": str(root), "missing_root": False, **label_summary}

    required = {"0": int(required_label0), "1": int(required_label1)}
    shortfall = {
        label: max(0, required[label] - int(per_label[label]["available_candidates"]))
        for label in ["0", "1"]
    }
    shortfall = {label: count for label, count in shortfall.items() if count > 0}
    summary = {
        "schema": "axon_replacement_candidate_pool_v1",
        "data_dir": str(data_dir),
        "split_csv": str(resolve_path(split_csv)),
        "manifest_json": str(resolve_path(manifest_json)) if manifest_json is not None else None,
        "hash_files": bool(hash_files),
        "max_candidates_per_label": max_candidates_per_label,
        "used_split_key_count": len(used_keys),
        "rows": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "per_label": per_label,
        "required_replacements": required,
        "replacement_shortfall": shortfall,
        "enough_for_required_replacements": not shortfall,
        "notes": [
            "Candidates are valid raw PE files that are not already present in the current split.",
            "cache_present=false means the raw file needs feature extraction before it can be used from cache.",
            "This script does not modify the split, cache, or raw files.",
        ],
    }
    return rows, summary


def write_candidate_csv(path: Path, rows: Sequence[dict]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build unused same-label replacement candidate pool.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-file-size", type=int, default=1 * 1024 * 1024 * 1024)
    parser.add_argument("--hash-files", action="store_true", help="Hash files whose filename is not a SHA-256 value.")
    parser.add_argument("--max-candidates-per-label", type=int, default=None)
    parser.add_argument("--required-label0", type=int, default=0)
    parser.add_argument("--required-label1", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows, summary = build_candidate_pool(
        data_dir=args.data_dir,
        split_csv=args.split_csv,
        manifest_json=args.manifest_json,
        max_file_size=int(args.max_file_size),
        hash_files=bool(args.hash_files),
        max_candidates_per_label=args.max_candidates_per_label,
        required_label0=int(args.required_label0),
        required_label1=int(args.required_label1),
    )
    write_candidate_csv(args.output_csv, rows)
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    summary["outputs"] = {"csv": str(resolve_path(args.output_csv)), "json": str(output_json)}
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
