#!/usr/bin/env python3
"""Attach shortcut strata and training sample weights to the 7:1:2 split.

Error attribution showed the champion learned a shortcut -- "packed / high
entropy => malicious, signed / low entropy => benign" -- and that the new corpus
reproduces the same composition (benign 1.8% high-entropy vs malicious 61.5%;
benign 61.2% signed vs malicious 9.6%). Scaling that composition teaches the
same shortcut harder, so training rows are reweighted to flatten it.

Each row is assigned a stratum from two PE properties the shortcut rides on:

    high_entropy : max section entropy >= 7.0
    signed       : a non-empty certificate data directory

giving 4 cells x 2 labels. Train weights are

    w(cell, label) proportional to (1 / n[cell][label]) ** alpha

normalised to mean 1.0 over the training rows and clipped to bound variance.
alpha = 0 reproduces the unweighted baseline (the control arm); alpha = 1 fully
equalises every cell x label group so the shortcut carries no information.

Validation and test rows always keep weight 1.0: metrics must be read on the
natural distribution, with the per-cell breakdown reported alongside.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pefile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTROPY_THRESHOLD = 7.0
CELLS = ("lowent_unsigned", "lowent_signed", "highent_unsigned", "highent_signed")


def cell_name(high_entropy: bool, signed: bool) -> str:
    return f"{'highent' if high_entropy else 'lowent'}_{'signed' if signed else 'unsigned'}"


def probe(path: str) -> tuple[str, bool, bool, bool]:
    """Return (path, high_entropy, signed, parse_ok) for one sample."""
    try:
        pe = pefile.PE(path, fast_load=True)
    except Exception:
        return path, False, False, False
    try:
        entropies = [s.get_entropy() for s in pe.sections]
        high_entropy = bool(entropies and max(entropies) >= ENTROPY_THRESHOLD)
        security = pe.OPTIONAL_HEADER.DATA_DIRECTORY[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_SECURITY"]
        ]
        signed = bool(security.VirtualAddress and security.Size)
        return path, high_entropy, signed, True
    except Exception:
        return path, False, False, False
    finally:
        try:
            pe.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path("manifests/roadmap_9997/corpus_712_split/split_712.csv"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="0 = unweighted control, 1 = fully flatten every cell x label group",
    )
    parser.add_argument(
        "--max-weight-ratio",
        type=float,
        default=25.0,
        help="clip weights to [1/r, r] around the mean to bound gradient variance",
    )
    parser.add_argument("--workers", type=int, default=min(24, (os.cpu_count() or 8)))
    parser.add_argument(
        "--probe-cache",
        type=Path,
        default=None,
        help="where the PE probe result is cached so other alpha arms reuse it",
    )
    args = parser.parse_args()

    split_csv = args.split_csv if args.split_csv.is_absolute() else PROJECT_ROOT / args.split_csv
    if not split_csv.exists():
        sys.exit(f"split csv not found: {split_csv}")

    with split_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    print(f"loaded {len(rows):,} rows from {split_csv}")

    # The strict split-metadata check reads `source_sha256`; older split files
    # emitted the column as `sha256`.
    if rows and "source_sha256" not in rows[0] and "sha256" in rows[0]:
        for row in rows:
            row["source_sha256"] = row.pop("sha256")
        print("  renamed column sha256 -> source_sha256")

    paths = [row["source_path"] for row in rows]

    # Probing 300k PE files costs ~25 min, and every alpha arm needs the same
    # answers, so the probe result is cached next to the split.
    probe_cache = args.probe_cache or split_csv.with_name("shortcut_probe.json")
    probe_cache = probe_cache if probe_cache.is_absolute() else PROJECT_ROOT / probe_cache
    properties: dict[str, tuple[bool, bool, bool]] = {}
    if probe_cache.exists():
        cached = json.loads(probe_cache.read_text(encoding="utf-8"))
        properties = {k: tuple(v) for k, v in cached["properties"].items()}
        missing = [p for p in paths if p not in properties]
        print(f"reusing probe cache {probe_cache} ({len(properties):,} entries, {len(missing):,} missing)")
    else:
        missing = paths

    if missing:
        print(f"probing {len(missing):,} PE files with {args.workers} workers ...", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index, (path, high_entropy, signed, ok) in enumerate(
                pool.map(probe, missing, chunksize=64)
            ):
                properties[path] = (high_entropy, signed, ok)
                if index and index % 20000 == 0:
                    print(f"  {index:,}/{len(missing):,}", flush=True)
        probe_cache.parent.mkdir(parents=True, exist_ok=True)
        probe_cache.write_text(
            json.dumps({"entropy_threshold": ENTROPY_THRESHOLD, "properties": properties}),
            encoding="utf-8",
        )
        print(f"  wrote probe cache: {probe_cache}")

    parse_failures = sum(1 for path in paths if not properties[path][2])
    print(f"  parse failures: {parse_failures:,}")

    # --- stratum counts on the training rows ---------------------------------
    train_counts: dict[tuple[str, int], int] = defaultdict(int)
    for row in rows:
        if row["split"] != "train":
            continue
        high_entropy, signed, _ = properties[row["source_path"]]
        train_counts[(cell_name(high_entropy, signed), int(row["label"]))] += 1

    raw: dict[tuple[str, int], float] = {}
    for cell in CELLS:
        for label in (0, 1):
            count = train_counts.get((cell, label), 0)
            raw[(cell, label)] = (1.0 / count) ** args.alpha if count else 0.0

    train_total = sum(train_counts.values())
    weighted_sum = sum(raw[(cell, label)] * count for (cell, label), count in train_counts.items())
    scale = train_total / weighted_sum if weighted_sum else 1.0
    weights = {key: value * scale for key, value in raw.items()}

    low, high = 1.0 / args.max_weight_ratio, args.max_weight_ratio
    clipped = {key: min(max(value, low), high) for key, value in weights.items()}

    # --- report ---------------------------------------------------------------
    print(f"\n=== training strata (alpha={args.alpha}, clip=+/-{args.max_weight_ratio}x) ===")
    print(f"{'cell':<20} {'benign':>9} {'malicious':>10} {'P(mal)':>8} "
          f"{'w_benign':>9} {'w_malic':>9} {'P(mal) after':>13}")
    print("-" * 84)
    for cell in CELLS:
        n0 = train_counts.get((cell, 0), 0)
        n1 = train_counts.get((cell, 1), 0)
        if not (n0 or n1):
            continue
        w0 = clipped[(cell, 0)]
        w1 = clipped[(cell, 1)]
        before = n1 / (n0 + n1)
        mass0, mass1 = n0 * w0, n1 * w1
        after = mass1 / (mass0 + mass1) if (mass0 + mass1) else 0.0
        print(f"{cell:<20} {n0:>9,} {n1:>10,} {before:>8.1%} "
              f"{w0:>9.3f} {w1:>9.3f} {after:>13.1%}")

    # --- write ----------------------------------------------------------------
    output = args.output or split_csv.with_name(f"split_712_round0_alpha{args.alpha:g}.csv")
    output = output if output.is_absolute() else PROJECT_ROOT / output
    fieldnames = list(rows[0].keys()) + ["high_entropy", "signed", "stratum", "sample_weight"]
    cell_totals: Counter = Counter()
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            high_entropy, signed, _ = properties[row["source_path"]]
            cell = cell_name(high_entropy, signed)
            label = int(row["label"])
            row["high_entropy"] = int(high_entropy)
            row["signed"] = int(signed)
            row["stratum"] = f"{cell}_label{label}"
            row["sample_weight"] = (
                f"{clipped.get((cell, label), 1.0):.6f}" if row["split"] == "train" else "1.000000"
            )
            cell_totals[(row["split"], cell, label)] += 1
            writer.writerow(row)

    summary = {
        "schema": "axon_round0_strata_v1",
        "source_split": str(split_csv),
        "alpha": args.alpha,
        "max_weight_ratio": args.max_weight_ratio,
        "entropy_threshold": ENTROPY_THRESHOLD,
        "parse_failures": parse_failures,
        "train_counts": {f"{c}_label{l}": n for (c, l), n in sorted(train_counts.items())},
        "train_weights": {f"{c}_label{l}": w for (c, l), w in sorted(clipped.items())},
        "cell_totals": {f"{s}_{c}_label{l}": n for (s, c, l), n in sorted(cell_totals.items())},
        "notes": [
            "Weights apply to train rows only; val/test stay at 1.0 so metrics are "
            "read on the natural distribution.",
            "Report F1 per stratum as well as overall -- an aggregate F1 hides the shortcut.",
            "alpha=0 is the unweighted control arm.",
        ],
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  split  : {output}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
