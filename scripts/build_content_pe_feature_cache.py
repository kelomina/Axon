#!/usr/bin/env python3
"""Precompute content-only PE metadata features for Stage-2 experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from kvd_features.content_pe_v1 import CONTENT_PE_FEATURE_NAMES, _content_pe_features_from_path  # noqa: E402
from train_stage2_cache_matrix import (  # noqa: E402
    read_prediction_rows,
    resolve_path,
    save_feature_npz_atomic,
)


def _deduplicate_rows(rows: Sequence[dict]) -> list[dict]:
    seen = set()
    unique = []
    for row in rows:
        key = (row.get("source_sha256") or row.get("source_path") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _cache_path_for_row(row: dict, cache_dir: Path) -> Path:
    key = (row.get("source_sha256") or "").strip().lower()
    if not key:
        key = hashlib.sha256(str(resolve_path(Path(row["source_path"]))).encode("utf-8", errors="ignore")).hexdigest()
    return cache_dir / f"{key}.npz"


def _build_one(payload: tuple[dict, str]) -> dict:
    row, cache_dir_text = payload
    cache_dir = Path(cache_dir_text)
    cache_path = _cache_path_for_row(row, cache_dir)
    if cache_path.exists():
        return {"status": "exists", "zero": False}

    source_path = resolve_path(Path(row["source_path"]))
    features = _content_pe_features_from_path(source_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_feature_npz_atomic(cache_path, features)
    return {
        "status": "created",
        "zero": bool(np.count_nonzero(features) == 0),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sidecar cache for content-only PE metadata features.")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit applied after de-duplicating prediction rows.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Required when --limit is used, so production cache builds cannot be truncated accidentally.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.limit is not None:
        if not args.smoke:
            raise ValueError("--limit requires --smoke; production cache builds must not be truncated.")
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")

    cache_dir = resolve_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for prediction_path in args.predictions:
        rows.extend(read_prediction_rows(prediction_path))
    unique_rows = _deduplicate_rows(rows)
    total_unique_rows = len(unique_rows)
    if args.limit is not None:
        unique_rows = unique_rows[: args.limit]

    start = time.perf_counter()
    counts = {"exists": 0, "created": 0, "zero_features": 0}
    worker_count = max(1, int(args.workers))
    payloads = [(row, str(cache_dir)) for row in unique_rows]
    if worker_count == 1:
        iterator = map(_build_one, payloads)
        for result in iterator:
            counts[result["status"]] += 1
            counts["zero_features"] += int(result["zero"])
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for index, result in enumerate(executor.map(_build_one, payloads, chunksize=32), start=1):
                counts[result["status"]] += 1
                counts["zero_features"] += int(result["zero"])
                if index % 1000 == 0:
                    print(
                        f"[content-cache] processed={index}/{len(payloads)} created={counts['created']} "
                        f"exists={counts['exists']} zero={counts['zero_features']}",
                        flush=True,
                    )

    elapsed = time.perf_counter() - start
    report = {
        "schema": "axon_content_pe_feature_cache_v1",
        "protocol": "content-only PE metadata; filename/path/extension are not encoded as model features",
        "predictions": [str(resolve_path(path)) for path in args.predictions],
        "cache_dir": str(cache_dir),
        "workers": worker_count,
        "smoke": bool(args.smoke),
        "input_rows": len(rows),
        "deduplicated_rows_before_limit": total_unique_rows,
        "limit": args.limit,
        "unique_rows": len(unique_rows),
        "feature_dim": len(CONTENT_PE_FEATURE_NAMES),
        "counts": counts,
        "elapsed_sec": elapsed,
    }
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
