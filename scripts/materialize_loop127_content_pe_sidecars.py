#!/usr/bin/env python3
"""Materialize Loop127 Train/Val content PE sidecars with strict identity checks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
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
    CONTENT_PE_V2_FEATURE_NAMES,
    _content_pe_v2_features_from_path,
    is_valid_source_sha256,
    resolve_path,
    save_feature_npz_atomic,
    verify_content_row_source_sha256,
)


REQUIRED_COLUMNS = ["source_path", "source_sha256", "cache_path", "label", "split", "sample_index"]
MAX_FAILURE_EXAMPLES = 10


def is_valid_sha256(value: object) -> bool:
    return is_valid_source_sha256(value)


def iter_prediction_rows(path: Path) -> Iterable[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def read_prediction_fieldnames(path: Path) -> list[str]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def _load_valid_cached_features(cache_path: Path, expected_dim: int) -> np.ndarray | None:
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if "features" not in data.files:
                return None
            features = data["features"].astype(np.float32, copy=False)
    except Exception:
        return None
    if features.shape != (expected_dim,):
        return None
    if not np.isfinite(features).all():
        return None
    return features


def _audit_prediction_rows(path: Path, expected_split: str) -> tuple[dict, set[str]]:
    fieldnames = read_prediction_fieldnames(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    issue_counts: Counter[str] = Counter()
    seen_sha: dict[str, str] = {}
    seen_index: set[str] = set()
    source_sha_set: set[str] = set()
    row_count = 0
    for row in iter_prediction_rows(path):
        row_count += 1
        split = str(row.get("split") or "").strip()
        label = str(row.get("label") or "").strip()
        sha = str(row.get("source_sha256") or "").strip().casefold()
        sample_index = str(row.get("sample_index") or "").strip()
        if split != expected_split:
            issue_counts["unexpected_split"] += 1
        if label not in {"0", "1"}:
            issue_counts["invalid_label"] += 1
        if not is_valid_sha256(sha):
            issue_counts["invalid_source_sha256"] += 1
        else:
            source_sha_set.add(sha)
        if not sample_index.isdigit():
            issue_counts["invalid_sample_index"] += 1
        elif sample_index in seen_index:
            issue_counts["duplicate_sample_index"] += 1
        else:
            seen_index.add(sample_index)
        if is_valid_sha256(sha):
            previous_index = seen_sha.get(sha)
            if previous_index is None:
                seen_sha[sha] = sample_index
            elif previous_index != sample_index:
                issue_counts["duplicate_source_sha256"] += 1
    if missing_columns:
        issue_counts["missing_required_columns"] += 1
    return (
        {
            "rows": row_count,
            "missing_columns": missing_columns,
            "issue_counts": dict(sorted(issue_counts.items())),
            "ready_for_materialization": not any(
                key in issue_counts
                for key in [
                    "unexpected_split",
                    "invalid_label",
                    "invalid_source_sha256",
                    "invalid_sample_index",
                    "duplicate_sample_index",
                    "missing_required_columns",
                ]
            ),
        },
        source_sha_set,
    )


def _overlap_summary(train_sha: set[str], val_sha: set[str]) -> tuple[int, list[str], int]:
    smaller, larger = (train_sha, val_sha) if len(train_sha) <= len(val_sha) else (val_sha, train_sha)
    overlap_count = 0
    examples: list[str] = []
    for sha in smaller:
        if sha not in larger:
            continue
        overlap_count += 1
        if len(examples) < MAX_FAILURE_EXAMPLES:
            examples.append(sha)
    return overlap_count, sorted(examples), len(train_sha) + len(val_sha) - overlap_count


def _build_one(payload: tuple[dict[str, str], str, str, bool]) -> dict:
    row, v1_dir_text, v2_dir_text, refresh_invalid = payload
    sha = str(row.get("source_sha256") or "").strip().casefold()
    source_path = resolve_path(Path(str(row.get("source_path") or "")))
    result = {
        "source_sha256": sha,
        "sample_index": str(row.get("sample_index") or ""),
        "split": str(row.get("split") or ""),
        "label": str(row.get("label") or ""),
        "v1_status": "not_run",
        "v2_status": "not_run",
        "v1_zero": False,
        "v2_zero": False,
        "failed": False,
        "failure_reason": "",
    }
    if not source_path.is_file():
        result["failed"] = True
        result["failure_reason"] = "source_path_missing_or_not_file"
        return result
    try:
        source_path, sha = verify_content_row_source_sha256(row)
        result["source_sha256"] = sha
    except ValueError as exc:
        result["failed"] = True
        result["failure_reason"] = str(exc)
        return result

    specs = [
        ("v1", Path(v1_dir_text), len(CONTENT_PE_FEATURE_NAMES), _content_pe_features_from_path),
        ("v2", Path(v2_dir_text), len(CONTENT_PE_V2_FEATURE_NAMES), _content_pe_v2_features_from_path),
    ]
    for name, cache_dir, expected_dim, extractor in specs:
        cache_path = cache_dir / f"{sha}.npz"
        existed_before = cache_path.exists()
        cached_features = _load_valid_cached_features(cache_path, expected_dim) if existed_before else None
        if cached_features is not None:
            result[f"{name}_status"] = "exists"
            result[f"{name}_zero"] = bool(np.count_nonzero(cached_features) == 0)
            continue
        if existed_before and not refresh_invalid:
            result[f"{name}_status"] = "invalid_existing"
            result["failed"] = True
            result["failure_reason"] = f"{name}_invalid_existing"
            continue
        try:
            features = extractor(source_path)
        except Exception as exc:
            result[f"{name}_status"] = "extract_failed"
            result["failed"] = True
            result["failure_reason"] = f"{name}_extract_failed:{type(exc).__name__}"
            continue
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (expected_dim,) or not np.isfinite(features).all():
            result[f"{name}_status"] = "extracted_invalid"
            result["failed"] = True
            result["failure_reason"] = f"{name}_extracted_invalid"
            continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_feature_npz_atomic(cache_path, features)
        result[f"{name}_status"] = "refreshed_invalid" if existed_before else "created"
        result[f"{name}_zero"] = bool(np.count_nonzero(features) == 0)
    return result


def _record_result(
    result: dict,
    counts: Counter[str],
    zero_counts: Counter[str],
    failure_examples: list[dict],
) -> bool:
    counts[f"v1_{result['v1_status']}"] += 1
    counts[f"v2_{result['v2_status']}"] += 1
    zero_counts["v1_zero"] += int(bool(result["v1_zero"]))
    zero_counts["v2_zero"] += int(bool(result["v2_zero"]))
    if result["failed"] and len(failure_examples) < MAX_FAILURE_EXAMPLES:
        failure_examples.append(
            {
                "split": result["split"],
                "sample_index": result["sample_index"],
                "label": result["label"],
                "source_sha256": result["source_sha256"],
                "failure_reason": result["failure_reason"],
            }
        )
    return bool(result["failed"])


def _failure_result(row: dict[str, str], reason: str) -> dict:
    return {
        "source_sha256": str(row.get("source_sha256") or "").strip().casefold(),
        "sample_index": str(row.get("sample_index") or ""),
        "split": str(row.get("split") or ""),
        "label": str(row.get("label") or ""),
        "v1_status": "not_run",
        "v2_status": "not_run",
        "v1_zero": False,
        "v2_zero": False,
        "failed": True,
        "failure_reason": reason,
    }


def _drain_completed(
    pending: dict,
    counts: Counter[str],
    zero_counts: Counter[str],
    failure_examples: list[dict],
) -> tuple[dict, bool]:
    done, _remaining = wait(set(pending), return_when=FIRST_COMPLETED)
    had_failure = False
    for future in done:
        row = pending.pop(future)
        try:
            result = future.result()
        except Exception as exc:
            result = _failure_result(row, f"worker_exception:{type(exc).__name__}:{str(exc)[:300]}")
        had_failure = _record_result(result, counts, zero_counts, failure_examples) or had_failure
    return pending, had_failure


def _drain_all(
    pending: dict,
    counts: Counter[str],
    zero_counts: Counter[str],
    failure_examples: list[dict],
) -> bool:
    had_failure = False
    while pending:
        pending, batch_failed = _drain_completed(pending, counts, zero_counts, failure_examples)
        had_failure = had_failure or batch_failed
    return had_failure


def materialize_loop127_content_pe_sidecars(
    *,
    train_predictions: Path,
    val_predictions: Path,
    content_pe_cache_dir: Path,
    content_pe_v2_cache_dir: Path,
    output_json: Path,
    workers: int = 4,
    refresh_invalid: bool = True,
) -> dict:
    train_audit, train_sha = _audit_prediction_rows(train_predictions, "train")
    val_audit, val_sha = _audit_prediction_rows(val_predictions, "val")
    cross_split_overlap_count, cross_split_overlap_examples, unique_source_sha256_rows = _overlap_summary(
        train_sha,
        val_sha,
    )
    blockers = []
    if not train_audit["ready_for_materialization"]:
        blockers.append("train_inputs_not_materializable")
    if not val_audit["ready_for_materialization"]:
        blockers.append("val_inputs_not_materializable")
    if cross_split_overlap_count:
        blockers.append("train_val_source_sha256_overlap")

    content_pe_cache_dir = resolve_path(content_pe_cache_dir)
    content_pe_v2_cache_dir = resolve_path(content_pe_v2_cache_dir)
    counts: Counter[str] = Counter()
    zero_counts: Counter[str] = Counter()
    failure_examples = []
    start = time.perf_counter()
    if blockers:
        pass
    else:
        worker_count = max(1, int(workers))
        max_pending = max(1, worker_count * 8)
        seen_materialized: set[str] = set()
        materialization_failed = False
        if worker_count == 1:
            for path in (train_predictions, val_predictions):
                for row in iter_prediction_rows(path):
                    sha = str(row.get("source_sha256") or "").strip().casefold()
                    if not is_valid_sha256(sha) or sha in seen_materialized:
                        continue
                    seen_materialized.add(sha)
                    try:
                        result = _build_one(
                            (row, str(content_pe_cache_dir), str(content_pe_v2_cache_dir), bool(refresh_invalid))
                        )
                    except Exception as exc:
                        result = _failure_result(row, f"worker_exception:{type(exc).__name__}:{str(exc)[:300]}")
                    if _record_result(
                        result,
                        counts,
                        zero_counts,
                        failure_examples,
                    ):
                        materialization_failed = True
        else:
            pending = {}
            had_failure = False
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                for path in (train_predictions, val_predictions):
                    for row in iter_prediction_rows(path):
                        sha = str(row.get("source_sha256") or "").strip().casefold()
                        if not is_valid_sha256(sha) or sha in seen_materialized:
                            continue
                        seen_materialized.add(sha)
                        future = executor.submit(
                            _build_one,
                            (row, str(content_pe_cache_dir), str(content_pe_v2_cache_dir), bool(refresh_invalid)),
                        )
                        pending[future] = row
                        if len(pending) >= max_pending:
                            pending, batch_failed = _drain_completed(pending, counts, zero_counts, failure_examples)
                            had_failure = had_failure or batch_failed
                had_failure = _drain_all(pending, counts, zero_counts, failure_examples) or had_failure
            materialization_failed = had_failure
        if materialization_failed:
            blockers.append("sidecar_materialization_failures")
    elapsed = time.perf_counter() - start

    report = {
        "schema": "axon_loop127_content_pe_sidecar_materialization_v1",
        "protocol": (
            "Loop127 Train/Val only; source_path is used only to open bytes for content extraction; "
            "cache keys require valid source_sha256; no Test rows, no model fitting, no threshold selection"
        ),
        "train_predictions": str(resolve_path(train_predictions)),
        "val_predictions": str(resolve_path(val_predictions)),
        "content_pe_cache_dir": str(content_pe_cache_dir),
        "content_pe_v2_cache_dir": str(content_pe_v2_cache_dir),
        "workers": max(1, int(workers)),
        "refresh_invalid": bool(refresh_invalid),
        "train": train_audit,
        "val": val_audit,
        "cross_split_source_sha256_overlap_count": cross_split_overlap_count,
        "cross_split_source_sha256_overlap_examples": cross_split_overlap_examples,
        "input_rows": int(train_audit["rows"]) + int(val_audit["rows"]),
        "unique_source_sha256_rows": unique_source_sha256_rows,
        "counts": dict(sorted(counts.items())),
        "zero_counts": dict(sorted(zero_counts.items())),
        "failure_examples": failure_examples,
        "elapsed_sec": elapsed,
        "blockers": blockers,
        "ready_for_readiness_recheck": not blockers,
    }
    output_path = resolve_path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize strict Loop127 Train/Val content PE sidecars.")
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--content-pe-cache-dir", type=Path, required=True)
    parser.add_argument("--content-pe-v2-cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-refresh-invalid", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = materialize_loop127_content_pe_sidecars(
        train_predictions=args.train_predictions,
        val_predictions=args.val_predictions,
        content_pe_cache_dir=args.content_pe_cache_dir,
        content_pe_v2_cache_dir=args.content_pe_v2_cache_dir,
        output_json=args.output_json,
        workers=args.workers,
        refresh_invalid=not args.no_refresh_invalid,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready_for_readiness_recheck"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
