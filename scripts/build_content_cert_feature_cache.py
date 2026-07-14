#!/usr/bin/env python3
"""Precompute content-only Authenticode certificate features for Stage-2 experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for item in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from train_stage2_cache_matrix import (  # noqa: E402
    CONTENT_CERT_FEATURE_NAMES,
    _content_cert_features_from_path,
    content_cache_path_for_row,
    load_valid_feature_npz,
    resolve_path,
    save_feature_npz_atomic,
    verify_content_row_source_sha256,
)
from content_cache_build_runner import MAX_PENDING_TASKS, run_feature_cache_build  # noqa: E402


def _cache_path_for_row(row: dict, cache_dir: Path) -> Path:
    cache_path = content_cache_path_for_row(row, cache_dir)
    if cache_path is None:
        raise ValueError("cache_dir is required")
    return cache_path


def _load_valid_cached_features(cache_path: Path) -> np.ndarray | None:
    return load_valid_feature_npz(cache_path, len(CONTENT_CERT_FEATURE_NAMES))


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
    features = _content_cert_features_from_path(source_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_feature_npz_atomic(cache_path, features)
    return {
        "status": "refreshed_invalid" if existed_before else "created",
        "zero": bool(np.count_nonzero(features) == 0),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sidecar cache for content-only certificate features.")
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

    cache_dir = resolve_path(args.cache_dir)
    args.cache_dir = cache_dir
    return run_feature_cache_build(
        args=args,
        schema="axon_content_cert_feature_cache_v1",
        protocol="content-only Authenticode certificate blob features; filename/path/extension are not encoded",
        feature_dim=len(CONTENT_CERT_FEATURE_NAMES),
        build_one=_build_one,
        progress_label="cert-cache",
    )


if __name__ == "__main__":
    raise SystemExit(main())
