#!/usr/bin/env python3
"""Build stage2 input CSVs (train/val) by joining a split CSV with a cache manifest.

Usage:
    python scripts/build_stage2_input.py \
        --split-csv manifests/roadmap_9997/corpus_712_funnel/split_5k.csv \
        --manifest-json data/.cache_712_fixedv2/manifest_38672ba0.json \
        --cache-dir data/.cache_712_fixedv2 \
        --output-train reports/roadmap_9997/funnel_712/stage2_input_5k_train.csv \
        --output-val reports/roadmap_9997/funnel_712/stage2_input_5k_val.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--manifest-json", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-train", required=True, type=Path)
    parser.add_argument("--output-val", required=True, type=Path)
    args = parser.parse_args(argv)

    # Load manifest: source_sha256 -> relative cache_path
    with args.manifest_json.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    sha_to_cache: dict[str, str] = {}
    for sample in manifest["samples"]:
        sha_to_cache[sample["source_sha256"]] = sample["cache_path"]

    cache_dir = args.cache_dir.resolve()

    # Read split CSV and join
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    missing_count = 0

    with args.split_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            split = row["split"]
            if split not in ("train", "val"):
                continue
            sha = row["source_sha256"]
            cache_rel = sha_to_cache.get(sha)
            if cache_rel is None:
                missing_count += 1
                continue
            out_row = {
                "source_path": row["source_path"],
                "cache_path": str(cache_dir / cache_rel),
                "source_sha256": sha,
                "label": row["label"],
                "split": split,
                "sample_index": row["sample_index"],
                "prob_malicious": "0.5000000000",
            }
            if split == "train":
                train_rows.append(out_row)
            else:
                val_rows.append(out_row)

    if missing_count > 0:
        print(f"ERROR: {missing_count} split rows not found in manifest", file=sys.stderr)
        return 1

    fieldnames = [
        "source_path",
        "cache_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "prob_malicious",
    ]

    for output_path, rows in [(args.output_train, train_rows), (args.output_val, val_rows)]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)

    print(f"train rows: {len(train_rows)} -> {args.output_train}")
    print(f"val rows:   {len(val_rows)} -> {args.output_val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
