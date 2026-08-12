#!/usr/bin/env python3
"""Derive deterministic nested 1k/5k/10k manifests from the frozen 712 split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("manifests/roadmap_9997/corpus_712_split/split_712.csv")
DEFAULT_OUTPUT_DIR = Path("manifests/roadmap_9997/corpus_712_funnel")
DEFAULT_STAGE_SIZES = (1000, 5000, 10000)
SPLITS = ("train", "val", "test")
LABELS = (0, 1)
OUTPUT_FIELDS = ("sample_index", "source_sha256", "label", "split", "date", "source_path")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def parse_stage_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size <= 0 or size % 2 for size in sizes):
        raise ValueError("stage sizes must be positive even integers")
    if tuple(sorted(set(sizes))) != sizes:
        raise ValueError("stage sizes must be unique and strictly increasing")
    return sizes


def read_excluded_sha256(paths: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            first_line = handle.readline()
            handle.seek(0)
            if "," not in first_line:
                for line in handle:
                    value = line.strip().casefold()
                    if value:
                        if not is_sha256(value):
                            raise ValueError(f"invalid excluded SHA-256 in {path}: {value}")
                        excluded.add(value)
                continue
            reader = csv.DictReader(handle)
            for row in reader:
                value = str(row.get("source_sha256") or row.get("sha256") or "").strip().casefold()
                if not is_sha256(value):
                    raise ValueError(f"excluded CSV lacks a valid SHA-256 in {path}")
                excluded.add(value)
    return excluded


def load_source_rows(path: Path, excluded: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_sha: set[str] = set()
    seen_paths: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"split CSV has no header: {path}")
        if "source_sha256" not in reader.fieldnames and "sha256" not in reader.fieldnames:
            raise ValueError("split CSV requires source_sha256 or legacy sha256")
        for source in reader:
            source_sha = str(source.get("source_sha256") or source.get("sha256") or "").strip().casefold()
            source_path = str(source.get("source_path") or "").strip()
            split = str(source.get("split") or "").strip().casefold()
            label_text = str(source.get("label") or "").strip()
            if not is_sha256(source_sha):
                raise ValueError(f"invalid source SHA-256: {source_sha}")
            if source_sha in excluded:
                continue
            if source_sha in seen_sha:
                raise ValueError(f"duplicate source SHA-256: {source_sha}")
            normalized_path = source_path.casefold()
            if not source_path or normalized_path in seen_paths:
                raise ValueError(f"missing or duplicate source path: {source_path}")
            if split not in SPLITS:
                raise ValueError(f"invalid split for {source_path}: {split}")
            if label_text not in {"0", "1"}:
                raise ValueError(f"invalid label for {source_path}: {label_text}")
            seen_sha.add(source_sha)
            seen_paths.add(normalized_path)
            rows.append(
                {
                    "sample_index": "",
                    "source_sha256": source_sha,
                    "label": label_text,
                    "split": split,
                    "date": str(source.get("date") or "").strip(),
                    "source_path": source_path,
                }
            )
    if not rows:
        raise ValueError(f"split CSV has no usable rows: {path}")
    return rows


def rank_rows(rows: Iterable[dict[str, str]], seed: int) -> dict[tuple[str, int], list[dict[str, str]]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], int(row["label"]))].append(row)
    for (split, label), group_rows in groups.items():
        group_rows.sort(
            key=lambda row: hashlib.sha256(
                f"axon-712-funnel-v1\0{seed}\0{split}\0{label}\0{row['source_sha256']}".encode()
            ).digest()
        )
    return groups


def per_class_quotas(total_rows: int) -> dict[str, int]:
    per_class = total_rows // 2
    train = int(per_class * 0.7)
    val = int(per_class * 0.1)
    return {"train": train, "val": val, "test": per_class - train - val}


def select_stage(
    groups: dict[tuple[str, int], list[dict[str, str]]], total_rows: int
) -> list[dict[str, str]]:
    quotas = per_class_quotas(total_rows)
    selected: list[dict[str, str]] = []
    for split in SPLITS:
        for label in LABELS:
            available = groups.get((split, label), [])
            quota = quotas[split]
            if len(available) < quota:
                raise ValueError(
                    f"insufficient rows for stage {total_rows}: split={split}, label={label}, "
                    f"required={quota}, available={len(available)}"
                )
            selected.extend(available[:quota])
    return selected


def select_full(groups: dict[tuple[str, int], list[dict[str, str]]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for split in SPLITS:
        for label in LABELS:
            selected.extend(groups.get((split, label), []))
    return selected


def write_split(path: Path, rows: Sequence[dict[str, str]]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for sample_index, source in enumerate(rows):
            row = dict(source)
            row["sample_index"] = str(sample_index)
            writer.writerow(row)
            counts[row["split"]] += 1
            counts[f"{row['split']}_label{row['label']}"] += 1
            counts[f"label{row['label']}"] += 1
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
    }


def build_funnel(
    *,
    source_split: Path,
    output_dir: Path,
    stage_sizes: Sequence[int],
    selection_seed: int,
    excluded_sha256: set[str],
) -> dict:
    rows = load_source_rows(source_split, excluded_sha256)
    groups = rank_rows(rows, selection_seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, dict] = {}
    memberships: dict[str, set[str]] = {}
    for size in stage_sizes:
        name = f"{size // 1000}k" if size % 1000 == 0 else str(size)
        stage_rows = select_stage(groups, size)
        artifacts[name] = write_split(output_dir / f"split_{name}.csv", stage_rows)
        memberships[name] = {row["source_sha256"] for row in stage_rows}

    full_rows = select_full(groups)
    artifacts["full"] = write_split(output_dir / "split_full.csv", full_rows)
    memberships["full"] = {row["source_sha256"] for row in full_rows}

    ordered_names = [f"{size // 1000}k" if size % 1000 == 0 else str(size) for size in stage_sizes]
    ordered_names.append("full")
    nesting = []
    for child, parent in zip(ordered_names, ordered_names[1:]):
        missing = memberships[child] - memberships[parent]
        nesting.append({"child": child, "parent": parent, "missing_rows": len(missing)})
        if missing:
            raise ValueError(f"funnel is not nested: {child} is not a subset of {parent}")

    payload = {
        "schema": "axon_712_nested_funnel_v1",
        "source_split": str(source_split),
        "source_split_sha256": sha256_file(source_split),
        "source_rows_after_exclusions": len(rows),
        "selection_seed": selection_seed,
        "selection_algorithm": "sha256(seed,split,label,source_sha256)_prefix",
        "stage_sizes_are_total_rows": True,
        "split_ratios": {"train": 0.7, "val": 0.1, "test": 0.2},
        "class_balance": "1:1",
        "excluded_sha256_count": len(excluded_sha256),
        "artifacts": artifacts,
        "nesting": nesting,
        "test_policy": "1k_5k_10k_test_rows_reserved_unopened; full_test_once_after_freeze",
    }
    receipt_path = output_dir / "funnel_receipt.json"
    receipt_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["receipt_path"] = str(receipt_path)
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage-sizes", default=",".join(str(size) for size in DEFAULT_STAGE_SIZES))
    parser.add_argument("--selection-seed", type=int, default=9997)
    parser.add_argument("--exclude-csv", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = build_funnel(
        source_split=resolve_project_path(args.source_split),
        output_dir=resolve_project_path(args.output_dir),
        stage_sizes=parse_stage_sizes(args.stage_sizes),
        selection_seed=args.selection_seed,
        excluded_sha256=read_excluded_sha256(resolve_project_path(path) for path in args.exclude_csv),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
