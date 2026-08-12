#!/usr/bin/env python3
"""Build a reproducible 7:1:2 train/val/test split over the full raw corpora.

Unlike ``build_random_20w_split.py`` (fixed 1:1:8 over 200k samples), this builds
a 7:1:2 split over the whole usable corpus so the training set scales with the
data actually on disk.

Both corpora are SHA256-named, so exact-duplicate removal and cross-label
conflict detection are done on filenames alone -- no content hashing required.

Guarantees:
- exact SHA256 dedup within and across both roots
- cross-label conflicts (same hash on both sides) are dropped from both classes
- non-PE payloads are excluded by extension allowlist plus an MZ magic check
- classes are balanced 1:1, then split 7:1:2 with a seeded deterministic shuffle
- the MalwareBazaar date directory is carried into the CSV so a temporal
  holdout slice can be derived later without rescanning

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
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Windows PE payload extensions. Benign files are extensionless; the malicious
# corpus mixes in ELF, Office documents, archives and scripts that this project
# does not model.
PE_EXTENSIONS = frozenset(
    {"", ".exe", ".dll", ".sys", ".scr", ".ocx", ".cpl", ".drv", ".efi", ".mui", ".unknown"}
)

DATE_DIR_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

SPLIT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}


@dataclass(frozen=True)
class Candidate:
    sha256: str
    path: str
    date: str


def scan_root(root: Path) -> tuple[dict[str, Candidate], Counter]:
    """Enumerate a corpus root, keyed by SHA256 taken from the filename."""
    by_hash: dict[str, Candidate] = {}
    stats: Counter = Counter()
    for dirpath, _dirnames, filenames in os.walk(root):
        date_match = DATE_DIR_RE.search(dirpath)
        date = date_match.group(1) if date_match else ""
        for name in filenames:
            stats["files_seen"] += 1
            stem, ext = os.path.splitext(name)
            key = stem.lower()
            if not SHA256_RE.match(key):
                stats["non_sha_named"] += 1
                continue
            if ext.lower() not in PE_EXTENSIONS:
                stats["non_pe_extension"] += 1
                continue
            if key in by_hash:
                stats["duplicate_hash"] += 1
                continue
            by_hash[key] = Candidate(key, os.path.join(dirpath, name), date)
            stats["pe_extension_candidates"] += 1
    return by_hash, stats


def is_valid_pe_sample(path: str, max_file_size: int) -> bool:
    """Same acceptance rule the dataset loader uses: non-empty, bounded, MZ."""
    try:
        size = os.stat(path).st_size
    except OSError:
        return False
    if size <= 0 or size > max_file_size:
        return False
    try:
        with open(path, "rb") as handle:
            return handle.read(2) == b"MZ"
    except OSError:
        return False


def take_verified(
    candidates: list[Candidate],
    quota: int,
    max_file_size: int,
    label: str,
    workers: int = 64,
    batch_size: int = 8192,
) -> tuple[list[Candidate], Counter]:
    """Walk the shuffled candidates in order, keeping the first `quota` real PEs.

    Verifying lazily avoids opening the whole corpus: only enough files to fill
    the quota (plus rejects encountered on the way) are ever touched.

    The MZ probe is pure I/O latency on a slow external volume, so batches are
    checked by a thread pool while results are consumed in the original shuffled
    order -- parallel throughput, identical output to the serial version.
    """
    accepted: list[Candidate] = []
    stats: Counter = Counter()
    start = time.time()
    examined = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(candidates), batch_size):
            if len(accepted) >= quota:
                break
            batch = candidates[offset:offset + batch_size]
            verdicts = list(pool.map(lambda c: is_valid_pe_sample(c.path, max_file_size), batch))
            for candidate, is_pe in zip(batch, verdicts):
                examined += 1
                if is_pe:
                    accepted.append(candidate)
                    stats["verified_pe"] += 1
                    if len(accepted) >= quota:
                        break
                else:
                    stats["rejected_not_pe"] += 1
            rate = examined / max(time.time() - start, 1e-6)
            print(
                f"  [{label}] examined {examined:,}, accepted {len(accepted):,}/{quota:,} "
                f"({time.time() - start:.0f}s, {rate:,.0f} files/s)",
                flush=True,
            )
    stats["examined"] = examined
    return accepted, stats


def assign_splits(items: list[Candidate], rng: random.Random) -> dict[str, list[Candidate]]:
    shuffled = list(items)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * SPLIT_RATIOS["train"])
    val_end = train_end + int(total * SPLIT_RATIOS["val"])
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benign-root", type=Path, default=Path(r"G:\私人\良性文件\待加入白名单"))
    parser.add_argument("--malicious-root", type=Path, default=Path(r"G:\私人\恶意\MB\unziped"))
    parser.add_argument("--output-dir", type=Path, default=Path("manifests/roadmap_9997/corpus_712_split"))
    parser.add_argument("--seed", type=int, default=9997)
    parser.add_argument("--max-file-size", type=int, default=256 * 1024 * 1024)
    parser.add_argument(
        "--per-class",
        type=int,
        default=0,
        help="samples per class; 0 means use every usable benign PE and match it",
    )
    args = parser.parse_args()

    t0 = time.time()
    print(f"[scan] benign root: {args.benign_root}", flush=True)
    benign, benign_stats = scan_root(args.benign_root)
    print(f"  {len(benign):,} PE-extension candidates  ({time.time() - t0:.0f}s)", flush=True)

    print(f"[scan] malicious root: {args.malicious_root}", flush=True)
    malicious, malicious_stats = scan_root(args.malicious_root)
    print(f"  {len(malicious):,} PE-extension candidates  ({time.time() - t0:.0f}s)", flush=True)

    conflicts = sorted(set(benign) & set(malicious))
    for sha in conflicts:
        del benign[sha]
        del malicious[sha]
    print(f"[dedup] dropped {len(conflicts)} cross-label hash conflicts from both classes", flush=True)

    rng = random.Random(args.seed)
    benign_pool = sorted(benign.values(), key=lambda c: c.sha256)
    malicious_pool = sorted(malicious.values(), key=lambda c: c.sha256)
    rng.shuffle(benign_pool)
    rng.shuffle(malicious_pool)

    print("[verify] benign MZ check", flush=True)
    benign_quota = args.per_class or len(benign_pool)
    benign_ok, benign_verify = take_verified(benign_pool, benign_quota, args.max_file_size, "benign")

    per_class = args.per_class or len(benign_ok)
    print(f"[balance] per-class target = {per_class:,}", flush=True)
    benign_ok = benign_ok[:per_class]

    print("[verify] malicious MZ check", flush=True)
    malicious_ok, malicious_verify = take_verified(
        malicious_pool, per_class, args.max_file_size, "malicious"
    )

    per_class = min(len(benign_ok), len(malicious_ok))
    benign_ok = benign_ok[:per_class]
    malicious_ok = malicious_ok[:per_class]
    print(f"[balance] final per-class = {per_class:,}", flush=True)

    benign_splits = assign_splits(benign_ok, random.Random(args.seed + 1))
    malicious_splits = assign_splits(malicious_ok, random.Random(args.seed + 2))

    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "split_712.csv"

    counts: Counter = Counter()
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        # `source_sha256` is the column name the strict split-metadata check in
        # dataset._strict_split_metadata_failure requires; do not rename it.
        writer.writerow(["sample_index", "source_sha256", "label", "split", "date", "source_path"])
        index = 0
        for split in ("train", "val", "test"):
            for label, groups in ((0, benign_splits), (1, malicious_splits)):
                for candidate in groups[split]:
                    writer.writerow(
                        [index, candidate.sha256, label, split, candidate.date, candidate.path]
                    )
                    counts[f"{split}_label{label}"] += 1
                    counts[split] += 1
                    index += 1

    summary = {
        "schema": "axon_corpus_712_split_v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "ratios": SPLIT_RATIOS,
        "roots": {"benign": str(args.benign_root), "malicious": str(args.malicious_root)},
        "per_class": per_class,
        "total_rows": index,
        "counts": dict(counts),
        "benign_scan": dict(benign_stats),
        "malicious_scan": dict(malicious_stats),
        "benign_verify": dict(benign_verify),
        "malicious_verify": dict(malicious_verify),
        "cross_label_conflicts_dropped": len(conflicts),
        "conflict_hashes": conflicts,
        "max_file_size": args.max_file_size,
        "pe_extensions": sorted(PE_EXTENSIONS),
        "elapsed_seconds": round(time.time() - t0, 1),
        "notes": [
            "Classes are balanced 1:1, then split 7:1:2 independently per class.",
            "Exact SHA256 duplicates and cross-label conflicts are removed before splitting.",
            "Split is random over the pooled corpus; the `date` column supports a "
            "temporal-holdout evaluation slice for measuring family/campaign leakage.",
            "Path text, filenames, extensions and row order must not be used as features.",
        ],
    }
    summary_path = output_dir / "split_712_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== 7:1:2 split ===")
    for split in ("train", "val", "test"):
        print(
            f"  {split:<5} {counts[split]:>8,}  "
            f"(benign {counts[f'{split}_label0']:,} / malicious {counts[f'{split}_label1']:,})"
        )
    print(f"  TOTAL {index:,}")
    print(f"\n  csv     : {csv_path}")
    print(f"  summary : {summary_path}")
    print(f"  elapsed : {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
