"""Shared bounded runner for content sidecar cache builders."""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from train_stage2_cache_matrix import resolve_path, source_sha256_for_row

MAX_WORKERS = 8
DEFAULT_PENDING_MULTIPLIER = 4
MAX_PENDING_TASKS = 64
MAX_FAILURE_EXAMPLES = 20
MAX_ERROR_TEXT = 500


def iter_prediction_rows(paths: Sequence[Path]) -> Iterable[dict]:
    for path in paths:
        with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)


def project_row(row: dict) -> dict:
    return {
        "source_path": str(row.get("source_path") or "").strip(),
        "source_sha256": source_sha256_for_row(row),
    }


def processed_count(counts: dict) -> int:
    return counts["exists"] + counts["created"] + counts["refreshed_invalid"] + counts["failed"]


def update_counts(counts: dict, result: dict) -> int:
    counts[result["status"]] += 1
    counts["zero_features"] += int(result["zero"])
    return processed_count(counts)


def record_failure(counts: dict, failure_examples: list[dict], row: dict, exc: BaseException) -> int:
    counts["failed"] += 1
    if len(failure_examples) < MAX_FAILURE_EXAMPLES:
        failure_examples.append(
            {
                "source_sha256": str(row.get("source_sha256") or "").strip(),
                "source_path": str(row.get("source_path") or "").strip(),
                "error": str(exc)[:MAX_ERROR_TEXT],
            }
        )
    return processed_count(counts)


def _print_progress(counts: dict, processed: int, total_to_process: Optional[int], label: str) -> None:
    if processed % 1000 == 0:
        total_text = f"/{total_to_process}" if total_to_process is not None else ""
        print(
            f"[{label}] processed={processed}{total_text} created={counts['created']} "
            f"exists={counts['exists']} failed={counts['failed']} zero={counts['zero_features']}",
            flush=True,
        )


def _drain_completed(
    pending: dict,
    counts: dict,
    failure_examples: list[dict],
    progress_label: str,
    *,
    total_to_process: Optional[int] = None,
) -> dict:
    done, _remaining = wait(set(pending), return_when=FIRST_COMPLETED)
    for future in done:
        row = pending.pop(future)
        try:
            result = future.result()
        except Exception as exc:
            processed = record_failure(counts, failure_examples, row, exc)
        else:
            processed = update_counts(counts, result)
        _print_progress(counts, processed, total_to_process, progress_label)
    return pending


def _drain_all(
    pending: dict,
    counts: dict,
    failure_examples: list[dict],
    progress_label: str,
    *,
    total_to_process: Optional[int] = None,
) -> None:
    while pending:
        pending = _drain_completed(
            pending,
            counts,
            failure_examples,
            progress_label,
            total_to_process=total_to_process,
        )


def validate_worker_window(workers: int, max_pending: Optional[int]) -> tuple[int, int]:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if workers > MAX_WORKERS:
        raise ValueError(f"--workers must be <= {MAX_WORKERS} on this 8GB workflow")
    if max_pending is not None and max_pending < 1:
        raise ValueError("--max-pending must be at least 1")
    worker_count = int(workers)
    default_max_pending = min(MAX_PENDING_TASKS, max(1, worker_count * DEFAULT_PENDING_MULTIPLIER))
    bounded_max_pending = default_max_pending if max_pending is None else min(max_pending, MAX_PENDING_TASKS)
    return worker_count, bounded_max_pending


def run_feature_cache_build(
    *,
    args,
    schema: str,
    protocol: str,
    feature_dim: int,
    build_one: Callable[[tuple[dict, str]], dict],
    progress_label: str,
) -> int:
    worker_count, max_pending = validate_worker_window(int(args.workers), getattr(args, "max_pending", None))
    limit = getattr(args, "limit", None)
    smoke = bool(getattr(args, "smoke", False))

    cache_dir = resolve_path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    counts = {"exists": 0, "created": 0, "refreshed_invalid": 0, "failed": 0, "zero_features": 0}
    failure_examples: list[dict] = []
    seen_sha = set()
    input_rows = 0
    total_unique_rows = 0
    submitted_rows = 0
    limit_reached = False
    total_to_process = limit if limit is not None else None

    def should_submit() -> bool:
        return limit is None or submitted_rows < limit

    if worker_count == 1:
        for row in iter_prediction_rows(args.predictions):
            input_rows += 1
            try:
                projected_row = project_row(row)
            except Exception as exc:
                fallback_row = {
                    "source_path": str(row.get("source_path") or "").strip(),
                    "source_sha256": str(row.get("source_sha256") or "").strip(),
                }
                record_failure(counts, failure_examples, fallback_row, exc)
                continue
            key = projected_row["source_sha256"]
            if key in seen_sha:
                continue
            seen_sha.add(key)
            total_unique_rows += 1
            if should_submit():
                submitted_rows += 1
                try:
                    result = build_one((projected_row, str(cache_dir)))
                except Exception as exc:
                    record_failure(counts, failure_examples, projected_row, exc)
                else:
                    update_counts(counts, result)
            elif not limit_reached:
                limit_reached = True
    else:
        pending = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for row in iter_prediction_rows(args.predictions):
                input_rows += 1
                try:
                    projected_row = project_row(row)
                except Exception as exc:
                    fallback_row = {
                        "source_path": str(row.get("source_path") or "").strip(),
                        "source_sha256": str(row.get("source_sha256") or "").strip(),
                    }
                    record_failure(counts, failure_examples, fallback_row, exc)
                    continue
                key = projected_row["source_sha256"]
                if key in seen_sha:
                    continue
                seen_sha.add(key)
                total_unique_rows += 1
                if should_submit():
                    submitted_rows += 1
                    future = executor.submit(build_one, (projected_row, str(cache_dir)))
                    pending[future] = projected_row
                    if len(pending) >= max_pending:
                        pending = _drain_completed(
                            pending,
                            counts,
                            failure_examples,
                            progress_label,
                            total_to_process=total_to_process,
                        )
                elif not limit_reached:
                    limit_reached = True
            _drain_all(
                pending,
                counts,
                failure_examples,
                progress_label,
                total_to_process=total_to_process,
            )

    elapsed = time.perf_counter() - start
    report = {
        "schema": schema,
        "protocol": protocol,
        "predictions": [str(resolve_path(path)) for path in args.predictions],
        "cache_dir": str(cache_dir),
        "workers": worker_count,
        "max_pending_tasks": max_pending if worker_count > 1 else 0,
        "smoke": smoke,
        "input_rows": input_rows,
        "deduplicated_rows_before_limit": total_unique_rows,
        "limit": limit,
        "unique_rows": submitted_rows,
        "feature_dim": feature_dim,
        "counts": counts,
        "failure_examples": failure_examples,
        "elapsed_sec": elapsed,
    }
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if counts["failed"] else 0
