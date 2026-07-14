#!/usr/bin/env python3
"""Precompute content-only PE metadata features for Stage-2 experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from kvd_features.content_pe_v1 import CONTENT_PE_FEATURE_NAMES, _content_pe_features_from_path  # noqa: E402
from train_stage2_cache_matrix import (  # noqa: E402
    content_cache_path_for_row,
    load_valid_feature_npz,
    resolve_path,
    save_feature_npz_atomic,
    source_sha256_for_row,
    verify_content_row_source_sha256,
)

MAX_WORKERS = 8
DEFAULT_PENDING_MULTIPLIER = 4
MAX_PENDING_TASKS = 64
MAX_FAILURE_EXAMPLES = 20
MAX_ERROR_TEXT = 500


def _iter_prediction_rows(paths: Sequence[Path]) -> Iterable[dict]:
    for path in paths:
        with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)


def _project_row(row: dict) -> dict:
    return {
        "source_path": str(row.get("source_path") or "").strip(),
        "source_sha256": source_sha256_for_row(row),
    }


def _cache_path_for_row(row: dict, cache_dir: Path) -> Path:
    cache_path = content_cache_path_for_row(row, cache_dir)
    if cache_path is None:
        raise ValueError("cache_dir is required")
    return cache_path


def _load_valid_cached_features(cache_path: Path) -> np.ndarray | None:
    return load_valid_feature_npz(cache_path, len(CONTENT_PE_FEATURE_NAMES))


def _build_one(payload: tuple[dict, str]) -> dict:
    row, cache_dir_text = payload
    cache_dir = Path(cache_dir_text)
    cache_path = _cache_path_for_row(row, cache_dir)
    existed_before = cache_path.exists()
    if existed_before:
        features = _load_valid_cached_features(cache_path)
        if features is not None:
            return {
                "status": "exists",
                "zero": bool(np.count_nonzero(features) == 0),
            }

    source_path, _source_sha = verify_content_row_source_sha256(row)
    features = _content_pe_features_from_path(source_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_feature_npz_atomic(cache_path, features)
    return {
        "status": "refreshed_invalid" if existed_before else "created",
        "zero": bool(np.count_nonzero(features) == 0),
    }


def _update_counts(counts: dict, result: dict) -> int:
    counts[result["status"]] += 1
    counts["zero_features"] += int(result["zero"])
    return _processed_count(counts)


def _processed_count(counts: dict) -> int:
    return counts["exists"] + counts["created"] + counts["refreshed_invalid"] + counts["failed"]


def _record_failure(counts: dict, failure_examples: list[dict], row: dict, exc: BaseException) -> int:
    counts["failed"] += 1
    if len(failure_examples) < MAX_FAILURE_EXAMPLES:
        failure_examples.append(
            {
                "source_sha256": str(row.get("source_sha256") or "").strip(),
                "source_path": str(row.get("source_path") or "").strip(),
                "error": str(exc)[:MAX_ERROR_TEXT],
            }
        )
    return _processed_count(counts)


def _print_progress(counts: dict, processed: int, total_to_process: Optional[int]) -> None:
    if processed % 1000 == 0:
        total_text = f"/{total_to_process}" if total_to_process is not None else ""
        print(
            f"[content-cache] processed={processed}{total_text} created={counts['created']} "
            f"exists={counts['exists']} failed={counts['failed']} zero={counts['zero_features']}",
            flush=True,
        )


def _drain_completed(
    pending: dict,
    counts: dict,
    failure_examples: list[dict],
    *,
    total_to_process: Optional[int] = None,
) -> dict:
    done, _remaining = wait(set(pending), return_when=FIRST_COMPLETED)
    for future in done:
        row = pending.pop(future)
        try:
            result = future.result()
        except Exception as exc:
            processed = _record_failure(counts, failure_examples, row, exc)
        else:
            processed = _update_counts(counts, result)
        _print_progress(counts, processed, total_to_process)
    return pending


def _drain_all(
    pending: dict,
    counts: dict,
    failure_examples: list[dict],
    *,
    total_to_process: Optional[int] = None,
) -> None:
    while pending:
        pending = _drain_completed(
            pending,
            counts,
            failure_examples,
            total_to_process=total_to_process,
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sidecar cache for content-only PE metadata features.")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-pending",
        type=int,
        default=None,
        help=f"Maximum queued futures for multi-process builds; capped at {MAX_PENDING_TASKS}.",
    )
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
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.workers > MAX_WORKERS:
        raise ValueError(f"--workers must be <= {MAX_WORKERS} on this 8GB workflow")
    if args.max_pending is not None and args.max_pending < 1:
        raise ValueError("--max-pending must be at least 1")

    cache_dir = resolve_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    counts = {"exists": 0, "created": 0, "refreshed_invalid": 0, "failed": 0, "zero_features": 0}
    failure_examples: list[dict] = []
    worker_count = int(args.workers)
    seen_sha = set()
    input_rows = 0
    total_unique_rows = 0
    submitted_rows = 0
    limit_reached = False
    total_to_process = args.limit if args.limit is not None else None

    def should_submit() -> bool:
        return args.limit is None or submitted_rows < args.limit

    default_max_pending = min(MAX_PENDING_TASKS, max(1, worker_count * DEFAULT_PENDING_MULTIPLIER))
    max_pending = default_max_pending if args.max_pending is None else min(args.max_pending, MAX_PENDING_TASKS)
    if worker_count == 1:
        for row in _iter_prediction_rows(args.predictions):
            input_rows += 1
            try:
                projected_row = _project_row(row)
            except Exception as exc:
                projected_row = {
                    "source_path": str(row.get("source_path") or "").strip(),
                    "source_sha256": str(row.get("source_sha256") or "").strip(),
                }
                _record_failure(counts, failure_examples, projected_row, exc)
                continue
            key = projected_row["source_sha256"]
            if key in seen_sha:
                continue
            seen_sha.add(key)
            total_unique_rows += 1
            if should_submit():
                submitted_rows += 1
                try:
                    result = _build_one((projected_row, str(cache_dir)))
                except Exception as exc:
                    _record_failure(counts, failure_examples, projected_row, exc)
                else:
                    _update_counts(counts, result)
            elif not limit_reached:
                limit_reached = True
    else:
        pending = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for row in _iter_prediction_rows(args.predictions):
                input_rows += 1
                try:
                    projected_row = _project_row(row)
                except Exception as exc:
                    projected_row = {
                        "source_path": str(row.get("source_path") or "").strip(),
                        "source_sha256": str(row.get("source_sha256") or "").strip(),
                    }
                    _record_failure(counts, failure_examples, projected_row, exc)
                    continue
                key = projected_row["source_sha256"]
                if key in seen_sha:
                    continue
                seen_sha.add(key)
                total_unique_rows += 1
                if should_submit():
                    submitted_rows += 1
                    future = executor.submit(_build_one, (projected_row, str(cache_dir)))
                    pending[future] = projected_row
                    if len(pending) >= max_pending:
                        pending = _drain_completed(
                            pending,
                            counts,
                            failure_examples,
                            total_to_process=total_to_process,
                        )
                elif not limit_reached:
                    limit_reached = True
            _drain_all(pending, counts, failure_examples, total_to_process=total_to_process)

    elapsed = time.perf_counter() - start
    report = {
        "schema": "axon_content_pe_feature_cache_v1",
        "protocol": "content-only PE metadata; filename/path/extension are not encoded as model features",
        "predictions": [str(resolve_path(path)) for path in args.predictions],
        "cache_dir": str(cache_dir),
        "workers": worker_count,
        "max_pending_tasks": max_pending if worker_count > 1 else 0,
        "smoke": bool(args.smoke),
        "input_rows": input_rows,
        "deduplicated_rows_before_limit": total_unique_rows,
        "limit": args.limit,
        "unique_rows": submitted_rows,
        "feature_dim": len(CONTENT_PE_FEATURE_NAMES),
        "counts": counts,
        "failure_examples": failure_examples,
        "elapsed_sec": elapsed,
    }
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
