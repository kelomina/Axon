#!/usr/bin/env python3
"""Audit a deterministic 1% sample of the 20w feature cache for integrity.

This script is read-only. It opens only the sampled NPZ files, verifies manifest
alignment, shape, label, finite numeric arrays, and source_sha256 consistency.
It does not train, evaluate, mutate cache, or scan the full raw data tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOTAL = 200000
EXPECTED_SPLIT_COUNTS = {"train": 20000, "val": 20000, "test": 160000}
EXPECTED_LABEL_SPLIT_COUNTS = {
    "train": {"0": 10000, "1": 10000},
    "val": {"0": 10000, "1": 10000},
    "test": {"0": 80000, "1": 80000},
}
REQUIRED_NPZ_FIELDS = [
    "byte_sequence",
    "pe_features",
    "stat_features",
    "lightweight_features",
    "label",
    "source_sha256",
]
DETAIL_FIELDNAMES = [
    "sample_rank",
    "sample_index",
    "split",
    "label",
    "source_path",
    "manifest_match_reason",
    "cache_path",
    "status",
    "issue_flags",
]

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from apply_manual_review_verdicts import source_keys  # noqa: E402


def resolve_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    resolved = resolve_path(path)
    assert resolved is not None
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    resolved = resolve_path(path)
    assert resolved is not None
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    assert resolved is not None
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def split_summary(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "split_counts": dict(sorted(Counter(row.get("split", "") for row in rows).items())),
        "label_split_counts": {
            split: dict(sorted(Counter(str(row.get("label", "")) for row in rows if row.get("split") == split).items()))
            for split in ["train", "val", "test"]
        },
    }


def validate_split_shape(rows: Sequence[dict[str, str]], *, enforce_20w: bool, enforce_label_balance: bool) -> list[str]:
    if not enforce_20w:
        return []
    summary = split_summary(rows)
    failures = []
    if summary["rows"] != EXPECTED_TOTAL:
        failures.append(f"expected {EXPECTED_TOTAL} rows, got {summary['rows']}")
    if summary["split_counts"] != EXPECTED_SPLIT_COUNTS:
        failures.append(f"split_counts mismatch: {summary['split_counts']}")
    if enforce_label_balance and summary["label_split_counts"] != EXPECTED_LABEL_SPLIT_COUNTS:
        failures.append(f"label_split_counts mismatch: {summary['label_split_counts']}")
    return failures


def sample_split_rows(
    rows: Sequence[dict[str, str]],
    *,
    sample_fraction: float,
    sample_size: Optional[int],
    seed: int,
) -> list[dict[str, str]]:
    if sample_size is None:
        sample_size = max(1, int(round(len(rows) * float(sample_fraction))))
    sample_size = min(int(sample_size), len(rows))
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[f"{row.get('split', '')}:{row.get('label', '')}"].append(row)

    selected: list[dict[str, str]] = []
    remaining_pool: list[dict[str, str]] = []
    remaining_slots = sample_size
    for key in sorted(groups):
        group_rows = list(groups[key])
        group_target = int(round(len(group_rows) / len(rows) * sample_size)) if rows else 0
        group_target = min(group_target, len(group_rows), remaining_slots)
        selected.extend(rng.sample(group_rows, group_target))
        selected_ids = {id(row) for row in selected}
        remaining_pool.extend(row for row in group_rows if id(row) not in selected_ids)
        remaining_slots = sample_size - len(selected)

    if remaining_slots > 0 and remaining_pool:
        selected.extend(rng.sample(remaining_pool, min(remaining_slots, len(remaining_pool))))
    selected.sort(key=lambda row: int(float(row.get("sample_index", 0) or 0)))
    return selected[:sample_size]


def load_manifest(manifest_json: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    resolved = resolve_path(manifest_json)
    assert resolved is not None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    lookup: dict[str, dict[str, Any]] = {}
    for sample in payload.get("samples", []):
        row = {
            "source_path": sample.get("source_path", ""),
            "source_sha256": sample.get("source_sha256", ""),
        }
        for key in source_keys(row):
            lookup.setdefault(key, sample)
    return payload, lookup


def lookup_manifest_sample(row: dict[str, str], lookup: dict[str, dict[str, Any]]) -> tuple[Optional[dict[str, Any]], str]:
    for key in source_keys(row):
        sample = lookup.get(key)
        if sample is not None:
            return sample, "source_sha256" if key.startswith("sha:") else "source_path"
    return None, "manifest_missing"


def resolve_cache_path(sample: dict[str, Any], manifest_dir: Path) -> Path:
    cache_path = Path(str(sample.get("cache_path", "")))
    if cache_path.is_absolute():
        return cache_path
    return manifest_dir / cache_path.name


def scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(value)


def expected_shapes(manifest: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        "byte_sequence": (int(manifest.get("max_byte_length", 0) or 0),),
        "pe_features": (int(manifest.get("pe_feature_dim", 0) or 0),),
        "stat_features": (int(manifest.get("stat_feature_dim", 0) or 0),),
        "lightweight_features": (int(manifest.get("lightweight_feature_dim", 0) or 0),),
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_npz_row(
    *,
    row: dict[str, str],
    sample: dict[str, Any],
    manifest: dict[str, Any],
    manifest_dir: Path,
    verify_source_hash: bool,
    max_source_hash_bytes: int,
) -> tuple[list[str], str]:
    issues: list[str] = []
    cache_path = resolve_cache_path(sample, manifest_dir)
    if not cache_path.exists():
        return ["cache_file_missing"], str(cache_path)

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            missing = [field for field in REQUIRED_NPZ_FIELDS if field not in data.files]
            if missing:
                issues.append("npz_missing_fields:" + "|".join(missing))
                return issues, str(cache_path)

            split_label = int(str(row.get("label", "")).strip())
            manifest_label = int(sample.get("label", split_label))
            npz_label = int(np.asarray(data["label"]).item())
            if not (split_label == manifest_label == npz_label):
                issues.append(f"label_mismatch:split={split_label}:manifest={manifest_label}:npz={npz_label}")

            manifest_sha = str(sample.get("source_sha256", "")).casefold()
            npz_sha = scalar_text(data["source_sha256"]).casefold()
            split_sha = str(row.get("source_sha256") or "").casefold()
            if manifest_sha and npz_sha != manifest_sha:
                issues.append("source_sha256_mismatch_npz_manifest")
            if split_sha and manifest_sha and split_sha != manifest_sha:
                issues.append("source_sha256_mismatch_split_manifest")

            for field, expected_shape in expected_shapes(manifest).items():
                if expected_shape == (0,):
                    continue
                actual_shape = tuple(data[field].shape)
                if actual_shape != expected_shape:
                    issues.append(f"shape_mismatch:{field}:actual={actual_shape}:expected={expected_shape}")
                if field != "byte_sequence" and not np.isfinite(data[field]).all():
                    issues.append(f"non_finite_values:{field}")
    except Exception as exc:  # pragma: no cover - operational detail is emitted in reports.
        issues.append(f"npz_load_error:{type(exc).__name__}")
        return issues, str(cache_path)

    source_path = Path(str(sample.get("source_path") or row.get("source_path") or ""))
    if verify_source_hash:
        if not source_path.exists():
            issues.append("source_file_missing")
        elif source_path.stat().st_size > max_source_hash_bytes:
            issues.append("source_hash_skipped_file_too_large")
        else:
            actual_sha = sha256_file(source_path)
            if manifest_sha and actual_sha != manifest_sha:
                issues.append("source_sha256_mismatch_actual_manifest")
    return issues, str(cache_path)


def audit_cache_sample_integrity(
    *,
    split_csv: Path,
    manifest_json: Path,
    sample_fraction: float = 0.01,
    sample_size: Optional[int] = None,
    seed: int = 42,
    enforce_20w: bool = True,
    enforce_label_balance: bool = True,
    verify_source_hash: bool = False,
    max_source_hash_bytes: int = 64 * 1024 * 1024,
    detail_output_csv: Optional[Path] = None,
) -> dict[str, Any]:
    split_rows = read_csv_rows(split_csv)
    manifest, manifest_lookup = load_manifest(manifest_json)
    manifest_dir = resolve_path(manifest_json).parent  # type: ignore[union-attr]
    shape_failures = validate_split_shape(split_rows, enforce_20w=enforce_20w, enforce_label_balance=enforce_label_balance)
    sample_rows = sample_split_rows(split_rows, sample_fraction=sample_fraction, sample_size=sample_size, seed=seed)

    status_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    sampled_split_counts: Counter[str] = Counter()
    sampled_label_counts: Counter[str] = Counter()
    detail_rows: list[dict[str, Any]] = []

    for rank, row in enumerate(sample_rows, start=1):
        sampled_split_counts[str(row.get("split", ""))] += 1
        sampled_label_counts[str(row.get("label", ""))] += 1
        sample, match_reason = lookup_manifest_sample(row, manifest_lookup)
        if sample is None:
            issues = [match_reason]
            cache_path = ""
        else:
            match_counts[match_reason] += 1
            issues, cache_path = audit_npz_row(
                row=row,
                sample=sample,
                manifest=manifest,
                manifest_dir=manifest_dir,
                verify_source_hash=verify_source_hash,
                max_source_hash_bytes=max_source_hash_bytes,
            )
        status = "pass" if not issues else "fail"
        status_counts[status] += 1
        for issue in issues:
            issue_counts[issue.split(":", 1)[0]] += 1
        detail_rows.append(
            {
                "sample_rank": rank,
                "sample_index": row.get("sample_index", ""),
                "split": row.get("split", ""),
                "label": row.get("label", ""),
                "source_path": row.get("source_path", ""),
                "manifest_match_reason": match_reason,
                "cache_path": cache_path,
                "status": status,
                "issue_flags": "|".join(issues),
            }
        )

    if detail_output_csv is not None:
        write_csv_rows(detail_output_csv, detail_rows, DETAIL_FIELDNAMES)

    failed_rows = int(status_counts.get("fail", 0))
    payload = {
        "schema": "axon_loop78_cache_sample_integrity_v1",
        "protocol": "read-only deterministic cache sample integrity audit; no model fitting, no threshold selection, no cache mutation",
        "split_csv": str(resolve_path(split_csv)),
        "manifest_json": str(resolve_path(manifest_json)),
        "seed": int(seed),
        "sample_fraction": float(sample_fraction),
        "requested_sample_size": sample_size,
        "sampled_rows": len(sample_rows),
        "split_summary": split_summary(split_rows),
        "enforce_20w": bool(enforce_20w),
        "enforce_label_balance": bool(enforce_label_balance),
        "shape_failures": shape_failures,
        "verify_source_hash": bool(verify_source_hash),
        "max_source_hash_bytes": int(max_source_hash_bytes),
        "source_hash_policy": (
            "actual source hashing enabled for sampled rows within max_source_hash_bytes"
            if verify_source_hash
            else "actual source hashing skipped; source_sha256 consistency checked between manifest and NPZ"
        ),
        "manifest_declared_shapes": {
            key: list(value) for key, value in expected_shapes(manifest).items()
        },
        "sampled_split_counts": dict(sorted(sampled_split_counts.items())),
        "sampled_label_counts": dict(sorted(sampled_label_counts.items())),
        "manifest_match_counts": dict(sorted(match_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "failed_rows": failed_rows,
        "detail_output_csv": str(resolve_path(detail_output_csv)) if detail_output_csv is not None else None,
        "audit_ready": not shape_failures and failed_rows == 0,
        "memory_leak_profile": {
            "loads_model": False,
            "uses_cuda": False,
            "opens_npz_files": True,
            "npz_scope": "sample_only",
            "scans_raw_data": False,
            "hashes_source_files": bool(verify_source_hash),
        },
        "notes": [
            "This is an integrity sample audit, not full cache coverage.",
            "Run full coverage/readiness audits separately before training.",
            "Identity fields are used only for manifest/cache alignment, not as model evidence.",
        ],
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit sampled feature-cache integrity.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--detail-output-csv", type=Path, default=None)
    parser.add_argument("--sample-fraction", type=float, default=0.01)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-enforce-20w", action="store_true")
    parser.add_argument("--no-enforce-label-balance", action="store_true")
    parser.add_argument("--verify-source-hash", action="store_true")
    parser.add_argument("--max-source-hash-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_cache_sample_integrity(
        split_csv=args.split_csv,
        manifest_json=args.manifest_json,
        sample_fraction=float(args.sample_fraction),
        sample_size=args.sample_size,
        seed=int(args.seed),
        enforce_20w=not bool(args.no_enforce_20w),
        enforce_label_balance=not bool(args.no_enforce_label_balance),
        verify_source_hash=bool(args.verify_source_hash),
        max_source_hash_bytes=int(args.max_source_hash_bytes),
        detail_output_csv=args.detail_output_csv,
    )
    write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["audit_ready"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
