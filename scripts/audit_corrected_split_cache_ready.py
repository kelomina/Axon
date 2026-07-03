#!/usr/bin/env python3
"""Strict cache readiness audit for a corrected 20w split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter
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
    "label",
    "source_sha256",
]
METADATA_DETAIL_FIELDNAMES = [
    "source_path",
    "source_sha256",
    "label",
    "sample_index",
    "split",
    "manifest_match_reason",
    "expected_cache_path",
    "issue_flags",
]

if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from apply_manual_review_verdicts import source_keys  # noqa: E402


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_manifest(manifest_json: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads(resolve_path(manifest_json).read_text(encoding="utf-8"))
    lookup: dict[str, dict] = {}
    for sample in manifest.get("samples", []):
        row = {
            "source_path": sample.get("source_path", ""),
            "source_sha256": sample.get("source_sha256", ""),
        }
        for key in source_keys(row):
            lookup.setdefault(key, sample)
    return manifest, lookup


def lookup_sample(row: dict, manifest_lookup: dict[str, dict]) -> tuple[Optional[dict], str]:
    for key in source_keys(row):
        sample = manifest_lookup.get(key)
        if sample is not None:
            if key.startswith("sha:"):
                return sample, "source_sha256"
            return sample, "source_path"
    return None, "manifest_missing"


def scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(value)


def parse_label(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def expected_shapes(manifest: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for field, manifest_key in [
        ("byte_sequence", "max_byte_length"),
        ("pe_features", "pe_feature_dim"),
        ("stat_features", "stat_feature_dim"),
        ("lightweight_features", "lightweight_feature_dim"),
    ]:
        value = int(manifest.get(manifest_key, 0) or 0)
        if value > 0:
            shapes[field] = (value,)
    return shapes


def read_npz_array_header(zf: zipfile.ZipFile, field: str) -> tuple[tuple[int, ...], np.dtype]:
    member = f"{field}.npy"
    with zf.open(member, "r") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported_npy_header_version:{version}")
    return tuple(int(part) for part in shape), np.dtype(dtype)


def audit_npz_metadata(
    *,
    row: dict,
    sample: dict[str, Any],
    manifest: dict[str, Any],
    cache_path: Path,
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    shape_skipped: list[str] = []

    try:
        with zipfile.ZipFile(cache_path, "r") as zf:
            npz_fields = {
                Path(info.filename).stem
                for info in zf.infolist()
                if info.filename.endswith(".npy")
            }
            missing_fields = [field for field in REQUIRED_NPZ_FIELDS if field not in npz_fields]
            if missing_fields:
                issues.append("npz_missing_fields:" + "|".join(missing_fields))

            for field, expected_shape in expected_shapes(manifest).items():
                if field not in npz_fields:
                    if field in REQUIRED_NPZ_FIELDS:
                        issues.append(f"shape_missing_required_field:{field}")
                    else:
                        shape_skipped.append(field)
                    continue
                actual_shape, _dtype = read_npz_array_header(zf, field)
                if actual_shape != expected_shape:
                    issues.append(f"shape_mismatch:{field}:actual={actual_shape}:expected={expected_shape}")
    except Exception as exc:
        return [f"npz_header_error:{type(exc).__name__}"], shape_skipped

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if "label" in data.files:
                split_label = parse_label(row.get("label"))
                manifest_label = parse_label(sample.get("label"))
                npz_label = parse_label(np.asarray(data["label"]).item())
                if split_label not in {0, 1}:
                    issues.append(f"split_label_invalid:{row.get('label', '')}")
                if manifest_label not in {0, 1}:
                    issues.append(f"manifest_label_invalid:{sample.get('label', '')}")
                if npz_label not in {0, 1}:
                    issues.append(f"npz_label_invalid:{npz_label}")
                if (
                    split_label in {0, 1}
                    and manifest_label in {0, 1}
                    and npz_label in {0, 1}
                    and not (split_label == manifest_label == npz_label)
                ):
                    issues.append(f"label_mismatch:split={split_label}:manifest={manifest_label}:npz={npz_label}")

            manifest_sha = str(sample.get("source_sha256") or "").strip().casefold()
            split_sha = str(row.get("source_sha256") or "").strip().casefold()
            if not manifest_sha:
                issues.append("manifest_missing_source_sha256")
            if "source_sha256" in data.files:
                npz_sha = scalar_text(data["source_sha256"]).strip().casefold()
                if not npz_sha:
                    issues.append("npz_missing_source_sha256_value")
                if manifest_sha and npz_sha and npz_sha != manifest_sha:
                    issues.append("source_sha256_mismatch_npz_manifest")
            if split_sha and manifest_sha and split_sha != manifest_sha:
                issues.append("source_sha256_mismatch_split_manifest")
    except Exception as exc:
        issues.append(f"npz_scalar_error:{type(exc).__name__}")

    return issues, shape_skipped


def split_row_sha_issue(row: dict) -> str:
    split_sha = str(row.get("source_sha256") or "").strip().casefold()
    if not split_sha:
        return "split_missing_source_sha256"
    if len(split_sha) != 64 or any(char not in "0123456789abcdef" for char in split_sha):
        return "split_invalid_source_sha256"
    return ""


def split_summary(rows: Sequence[dict]) -> dict:
    return {
        "rows": len(rows),
        "split_counts": dict(sorted(Counter(row.get("split", "") for row in rows).items())),
        "label_split_counts": {
            split: dict(sorted(Counter(str(row.get("label", "")) for row in rows if row.get("split") == split).items()))
            for split in ["train", "val", "test"]
        },
    }


def write_missing_rows(path: Path, rows: Sequence[dict]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "source_sha256",
        "label",
        "sample_index",
        "split",
        "reason",
        "expected_cache_path",
    ]
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_metadata_issue_rows(path: Path, rows: Sequence[dict]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_DETAIL_FIELDNAMES, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_split_shape(
    rows: Sequence[dict],
    *,
    expected_total: int = EXPECTED_TOTAL,
    expected_split_counts: Optional[dict[str, int]] = None,
    expected_label_split_counts: Optional[dict[str, dict[str, int]]] = None,
) -> list[str]:
    summary = split_summary(rows)
    split_targets = EXPECTED_SPLIT_COUNTS if expected_split_counts is None else expected_split_counts
    failures = []
    if summary["rows"] != expected_total:
        failures.append(f"expected {expected_total} rows, got {summary['rows']}")
    if summary["split_counts"] != split_targets:
        failures.append(f"split_counts mismatch: {summary['split_counts']}")
    if expected_label_split_counts is not None and summary["label_split_counts"] != expected_label_split_counts:
        failures.append(f"label_split_counts mismatch: {summary['label_split_counts']}")
    return failures


def audit_corrected_split_cache_ready(
    *,
    split_csv: Path,
    manifest_json: Path,
    missing_cache_output: Optional[Path] = None,
    metadata_issue_output: Optional[Path] = None,
    enforce_shape: bool = True,
    enforce_label_balance: bool = False,
    validate_cache_metadata: bool = True,
) -> dict:
    rows = read_csv_rows(split_csv)
    manifest, manifest_lookup = load_manifest(manifest_json)
    manifest_dir = resolve_path(manifest_json).parent
    missing_rows: list[dict] = []
    metadata_issue_rows: list[dict] = []
    match_counts: Counter[str] = Counter()
    missing_label_counts: Counter[str] = Counter()
    missing_split_counts: Counter[str] = Counter()
    missing_reason_counts: Counter[str] = Counter()
    metadata_issue_counts: Counter[str] = Counter()
    shape_check_skipped_counts: Counter[str] = Counter()
    metadata_checked_rows = 0

    for row in rows:
        sample, reason = lookup_sample(row, manifest_lookup)
        split_sha_issue = split_row_sha_issue(row)
        if sample is None:
            missing = {**row, "reason": reason, "expected_cache_path": ""}
            missing_rows.append(missing)
            missing_label_counts[str(row.get("label", ""))] += 1
            missing_split_counts[str(row.get("split", ""))] += 1
            missing_reason_counts[reason] += 1
            continue

        cache_path = Path(sample.get("cache_path", ""))
        if not cache_path.is_absolute():
            cache_path = manifest_dir / cache_path.name
        if not cache_path.exists():
            missing = {**row, "reason": "cache_file_missing", "expected_cache_path": str(cache_path)}
            missing_rows.append(missing)
            missing_label_counts[str(row.get("label", ""))] += 1
            missing_split_counts[str(row.get("split", ""))] += 1
            missing_reason_counts["cache_file_missing"] += 1
            continue

        match_counts[reason] += 1
        if validate_cache_metadata:
            metadata_checked_rows += 1
            pre_issues = [split_sha_issue] if split_sha_issue else []
            issues, skipped_shapes = audit_npz_metadata(
                row=row,
                sample=sample,
                manifest=manifest,
                cache_path=cache_path,
            )
            issues = pre_issues + issues
            for field in skipped_shapes:
                shape_check_skipped_counts[field] += 1
            if issues:
                metadata_issue_rows.append(
                    {
                        **row,
                        "manifest_match_reason": reason,
                        "expected_cache_path": str(cache_path),
                        "issue_flags": "|".join(issues),
                    }
                )
                for issue in issues:
                    metadata_issue_counts[issue.split(":", 1)[0]] += 1

    if missing_cache_output is not None:
        write_missing_rows(missing_cache_output, missing_rows)
    if metadata_issue_output is not None:
        write_metadata_issue_rows(metadata_issue_output, metadata_issue_rows)

    covered = len(rows) - len(missing_rows)
    shape_failures = (
        validate_split_shape(
            rows,
            expected_total=EXPECTED_TOTAL,
            expected_split_counts=EXPECTED_SPLIT_COUNTS,
            expected_label_split_counts=EXPECTED_LABEL_SPLIT_COUNTS if enforce_label_balance else None,
        )
        if enforce_shape
        else []
    )
    label_balance_drift = []
    if not enforce_label_balance:
        actual_label_split_counts = split_summary(rows)["label_split_counts"]
        if actual_label_split_counts != EXPECTED_LABEL_SPLIT_COUNTS:
            label_balance_drift = [
                f"{split}:{actual_label_split_counts.get(split, {})}"
                for split in ["train", "val", "test"]
                if actual_label_split_counts.get(split, {}) != EXPECTED_LABEL_SPLIT_COUNTS.get(split, {})
            ]
    payload = {
        "schema": "axon_corrected_split_cache_ready_v1",
        "split_csv": str(resolve_path(split_csv)),
        "manifest_json": str(resolve_path(manifest_json)),
        "split_summary": split_summary(rows),
        "expected_total": EXPECTED_TOTAL,
        "expected_split_counts": EXPECTED_SPLIT_COUNTS,
        "expected_label_split_counts": EXPECTED_LABEL_SPLIT_COUNTS,
        "total_rows": len(rows),
        "covered_rows": covered,
        "missing_rows": len(missing_rows),
        "coverage_ratio": float(covered / len(rows)) if rows else 0.0,
        "manifest_match_counts": dict(sorted(match_counts.items())),
        "cache_metadata_validation_enabled": bool(validate_cache_metadata),
        "metadata_checked_rows": metadata_checked_rows,
        "metadata_failure_rows": len(metadata_issue_rows),
        "metadata_issue_counts": dict(sorted(metadata_issue_counts.items())),
        "metadata_issue_examples": metadata_issue_rows[:20],
        "metadata_issue_output": str(resolve_path(metadata_issue_output)) if metadata_issue_output is not None else None,
        "manifest_declared_shapes": {key: list(value) for key, value in expected_shapes(manifest).items()},
        "shape_check_skipped_counts": dict(sorted(shape_check_skipped_counts.items())),
        "missing_label_counts": dict(sorted(missing_label_counts.items())),
        "missing_split_counts": dict(sorted(missing_split_counts.items())),
        "missing_reason_counts": dict(sorted(missing_reason_counts.items())),
        "missing_cache_output": str(resolve_path(missing_cache_output)) if missing_cache_output is not None else None,
        "label_balance_enforced": bool(enforce_label_balance),
        "label_balance_drift": label_balance_drift,
        "shape_failures": shape_failures,
        "cache_ready": not shape_failures and not missing_rows and (not validate_cache_metadata or not metadata_issue_rows),
        "memory_leak_profile": {
            "loads_model": False,
            "uses_cuda": False,
            "opens_npz_files": bool(validate_cache_metadata),
            "npz_scope": "full_split_metadata_only" if validate_cache_metadata else "disabled",
            "reads_large_arrays": False,
            "scans_raw_data": False,
        },
        "notes": [
            "cache_ready=true is required before training from a corrected split.",
            "Missing rows should be passed to the cache recovery/extraction flow before rerunning training.",
            "Cache metadata validation checks NPZ fields, label, source_sha256, and declared array shapes without full numeric array scans.",
            "Identity fields are used only for loading/alignment/cache audit, never as model or verdict evidence.",
            "Label balance is reported separately unless --enforce-label-balance is set.",
        ],
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit corrected split cache readiness.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--missing-cache-output", type=Path, default=None)
    parser.add_argument("--metadata-issue-output", type=Path, default=None)
    parser.add_argument("--no-enforce-shape", action="store_true")
    parser.add_argument(
        "--no-validate-cache-metadata",
        action="store_true",
        help="Compatibility escape hatch. Strict audits should keep metadata validation enabled.",
    )
    parser.add_argument(
        "--enforce-label-balance",
        action="store_true",
        help="Also require the corrected split to preserve the original per-split class balance.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless shape and cache coverage are complete.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_corrected_split_cache_ready(
        split_csv=args.split_csv,
        manifest_json=args.manifest_json,
        missing_cache_output=args.missing_cache_output,
        metadata_issue_output=args.metadata_issue_output,
        enforce_shape=not bool(args.no_enforce_shape),
        enforce_label_balance=bool(args.enforce_label_balance),
        validate_cache_metadata=not bool(args.no_validate_cache_metadata),
    )
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.strict and not payload["cache_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
