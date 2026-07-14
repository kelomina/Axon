#!/usr/bin/env python3
"""Build a reproducible 20w raw-file split from the Axon data directory.

This script only scans raw files under the configured benign/malicious roots,
filters to valid PE samples, and writes:

- split CSV with source_path/label/sample_index/split
- summary JSON with counts, seed, and any shortfalls

It does not train models, rebuild cache, or delete files.
The benign/malicious roots are a human labeling source for the split CSV only;
path text, filenames, extensions, source hashes, split names, and row order must
not be used as model features.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {".cache", "__pycache__", ".git", ".pytest_cache", "reports", "models", "swanlog"}


@dataclass(frozen=True)
class SplitCounts:
    train: int
    val: int
    test: int


@dataclass(frozen=True)
class ScanResult:
    selected_rows: list[dict]
    files_seen: int
    valid_pe_count: int


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def iter_sorted_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        filtered_dirs = []
        for dirname in sorted(dirnames):
            child = Path(dirpath) / dirname
            if dirname in SKIP_DIRS or child.is_symlink():
                continue
            filtered_dirs.append(dirname)
        dirnames[:] = filtered_dirs
        for filename in sorted(filenames):
            file_path = Path(dirpath) / filename
            if file_path.is_symlink():
                continue
            yield file_path


def is_valid_pe_sample(path: Path, max_file_size: int) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    if stat.st_size <= 0 or stat.st_size > max_file_size:
        return False
    try:
        with path.open("rb") as f:
            return f.read(2) == b"MZ"
    except OSError:
        return False


def scan_valid_raw_samples(
    root: Path,
    *,
    label: int,
    max_file_size: int,
    seed: int,
    sample_limit: Optional[int] = None,
) -> ScanResult:
    rng = random.Random((seed << 1) ^ (label * 1000003))
    reservoir: list[dict] = []
    valid_count = 0
    seen_files = 0
    for path in iter_sorted_files(root):
        seen_files += 1
        if not path.is_file():
            continue
        if not is_valid_pe_sample(path, max_file_size):
            continue
        valid_count += 1
        row = {
            "source_path": str(path),
            "label": int(label),
        }
        if sample_limit is None:
            reservoir.append(row)
            continue
        if len(reservoir) < sample_limit:
            reservoir.append(row)
            continue
        replacement_index = rng.randrange(valid_count)
        if replacement_index < sample_limit:
            reservoir[replacement_index] = row
    return ScanResult(reservoir, seen_files, valid_count)


def pick_balanced_samples(
    benign_rows: list[dict],
    malicious_rows: list[dict],
    *,
    total_samples: int,
    seed: int,
) -> tuple[list[dict], dict]:
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if total_samples % 2 != 0:
        raise ValueError("total_samples must be even so labels can be balanced")

    per_class = total_samples // 2
    if len(benign_rows) < per_class or len(malicious_rows) < per_class:
        raise ValueError(
            "Not enough valid PE samples for balanced sampling: "
            f"benign={len(benign_rows)}, malicious={len(malicious_rows)}, per_class={per_class}"
        )

    rng = random.Random(seed)
    benign_selected = rng.sample(benign_rows, per_class)
    malicious_selected = rng.sample(malicious_rows, per_class)

    train_per_class = per_class // 10
    val_per_class = per_class // 10
    test_per_class = per_class - train_per_class - val_per_class
    counts = SplitCounts(
        train=train_per_class * 2,
        val=val_per_class * 2,
        test=test_per_class * 2,
    )

    def split_one_class(rows: list[dict]) -> dict[str, list[dict]]:
        shuffled = rows.copy()
        rng.shuffle(shuffled)
        return {
            "train": shuffled[:train_per_class],
            "val": shuffled[train_per_class:train_per_class + val_per_class],
            "test": shuffled[train_per_class + val_per_class:],
        }

    benign_split = split_one_class(benign_selected)
    malicious_split = split_one_class(malicious_selected)

    assigned = []
    for split in ["train", "val", "test"]:
        split_rows = benign_split[split] + malicious_split[split]
        rng.shuffle(split_rows)
        for row in split_rows:
            assigned.append(
                {
                    "source_path": row["source_path"],
                    "label": int(row["label"]),
                    "split": split,
                }
            )

    for sample_index, row in enumerate(assigned):
        row["sample_index"] = sample_index

    summary = {
        "total_samples": total_samples,
        "samples_per_class": per_class,
        "split_counts": {
            "train": counts.train,
            "val": counts.val,
            "test": counts.test,
        },
        "label_counts": dict(Counter(str(row["label"]) for row in assigned)),
        "label_split_counts": {
            split: dict(Counter(str(row["label"]) for row in assigned if row["split"] == split))
            for split in ["train", "val", "test"]
        },
    }
    return assigned, summary


def write_split_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_path", "label", "sample_index", "split"])
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    *,
    seed: int,
    total_samples: int,
    benign_scan: ScanResult,
    malicious_scan: ScanResult,
    rows: list[dict],
    output_csv: Path,
    shortfall: Optional[dict] = None,
) -> dict:
    label_counts = Counter(str(row["label"]) for row in rows)
    split_counts = Counter(str(row["split"]) for row in rows)
    summary = {
        "schema": "axon_random_20w_split_v1",
        "seed": seed,
        "total_samples": total_samples,
        "label_target_counts": {
            "0": total_samples // 2,
            "1": total_samples // 2,
        },
        "label_counts": dict(sorted(label_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "label_split_counts": {
            split: dict(
                sorted(
                    Counter(str(row["label"]) for row in rows if row["split"] == split).items()
                )
            )
            for split in ["train", "val", "test"]
        },
        "source_directory_counts": {
            "benign_files_seen": benign_scan.files_seen,
            "malicious_files_seen": malicious_scan.files_seen,
            "benign_valid_pe": benign_scan.valid_pe_count,
            "malicious_valid_pe": malicious_scan.valid_pe_count,
        },
        "shortfall": shortfall or {},
        "output_csv": str(output_csv),
        "notes": [
            "Balanced 1:1 label sampling from raw PE files only.",
            "Split ratio is fixed at 1:1:8 over the selected 200000-sample set.",
            "Directory roots provide labels for the split CSV only; path text, filenames, extensions, hashes, split names, and row order are forbidden as model features.",
            "This script does not touch model training or cache rebuilding.",
        ],
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a reproducible 20w balanced raw split.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/random_20w_split"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-samples", type=int, default=200000)
    parser.add_argument("--max-file-size", type=int, default=1 * 1024 * 1024 * 1024)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    benign_root = data_dir / "待加入白名单"
    malicious_root = data_dir / "待拉黑"

    if not benign_root.exists():
        raise FileNotFoundError(f"Benign root not found: {benign_root}")
    if not malicious_root.exists():
        raise FileNotFoundError(f"Malicious root not found: {malicious_root}")

    benign_scan = scan_valid_raw_samples(
        benign_root,
        label=0,
        max_file_size=args.max_file_size,
        seed=args.seed,
        sample_limit=args.total_samples // 2,
    )
    malicious_scan = scan_valid_raw_samples(
        malicious_root,
        label=1,
        max_file_size=args.max_file_size,
        seed=args.seed,
        sample_limit=args.total_samples // 2,
    )
    benign_rows = benign_scan.selected_rows
    malicious_rows = malicious_scan.selected_rows

    output_dir.mkdir(parents=True, exist_ok=True)
    split_csv = output_dir / "random_20w_split.csv"

    shortfall = {}
    if benign_scan.valid_pe_count < args.total_samples // 2 or malicious_scan.valid_pe_count < args.total_samples // 2:
        shortfall = {
            "benign_shortfall": max(0, args.total_samples // 2 - benign_scan.valid_pe_count),
            "malicious_shortfall": max(0, args.total_samples // 2 - malicious_scan.valid_pe_count),
            "message": (
                "Requested 200000 balanced samples cannot be fully satisfied from the current raw PE set."
            ),
        }
        summary_path = output_dir / "random_20w_split_summary.json"
        summary = build_summary(
            seed=args.seed,
            total_samples=args.total_samples,
            benign_scan=benign_scan,
            malicious_scan=malicious_scan,
            rows=[],
            output_csv=split_csv,
            shortfall=shortfall,
        )
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        raise ValueError(
            f"{shortfall['message']} benign_valid={benign_scan.valid_pe_count}, malicious_valid={malicious_scan.valid_pe_count}"
        )

    rows, split_summary = pick_balanced_samples(
        benign_rows,
        malicious_rows,
        total_samples=args.total_samples,
        seed=args.seed,
    )
    write_split_csv(split_csv, rows)

    summary = build_summary(
        seed=args.seed,
        total_samples=args.total_samples,
        benign_scan=benign_scan,
        malicious_scan=malicious_scan,
        rows=rows,
        output_csv=split_csv,
        shortfall=shortfall,
    )
    summary["split_counts"] = split_summary["split_counts"]
    summary["label_counts"] = dict(sorted(Counter(str(row["label"]) for row in rows).items()))

    summary_path = output_dir / "random_20w_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Split CSV: {split_csv}")
    print(f"Summary JSON: {summary_path}")
    print(f"Valid benign PE samples: {len(benign_rows)}")
    print(f"Valid malicious PE samples: {len(malicious_rows)}")
    print(f"Selected samples: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
