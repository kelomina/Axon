#!/usr/bin/env python3
"""Audit Loop127 content-cross Val-only inputs without building matrices."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = ["source_path", "source_sha256", "cache_path", "label", "split", "sample_index", "prob_malicious"]
FORBIDDEN_EXTRA_COLUMN_TOKENS = [
    "filename",
    "file_name",
    "basename",
    "extension",
    "suffix",
    "directory",
    "dirname",
    "folder",
    "path_token",
    "verdict",
    "threshold",
]
DEFAULT_CONTENT_PE_V1_DIM = 100
DEFAULT_CONTENT_PE_V2_DIM = 182


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def is_valid_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def sidecar_cache_path(cache_dir: Path, row: dict[str, str]) -> Optional[Path]:
    sha = str(row.get("source_sha256") or "").strip().casefold()
    if not is_valid_sha256(sha):
        return None
    return resolve_path(cache_dir) / f"{sha}.npz"


def _npz_scalar_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            return ""
        return str(value.item())
    return str(value)


def _audit_main_cache(
    *,
    cache_path: Path,
    expected_label: str,
    expected_sha256: str,
    validate_npz_contents: bool,
) -> list[str]:
    issues: list[str] = []
    if not cache_path.exists():
        return ["missing_cache_path"]
    if not cache_path.is_file():
        return ["cache_path_not_file"]
    if cache_path.suffix.casefold() != ".npz":
        issues.append("cache_path_not_npz")
    if not validate_npz_contents:
        return issues

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            files = set(data.files)
            if "label" not in files:
                issues.append("cache_missing_label")
            else:
                cache_label = _npz_scalar_text(data["label"]).strip()
                if cache_label != expected_label:
                    issues.append("cache_label_mismatch")
            if "source_sha256" not in files:
                issues.append("cache_missing_source_sha256")
            else:
                cache_sha = _npz_scalar_text(data["source_sha256"]).strip().casefold()
                if cache_sha != expected_sha256:
                    issues.append("cache_source_sha256_mismatch")
    except Exception:
        issues.append("cache_npz_unreadable")
    return issues


def _audit_sidecar(
    *,
    sidecar_path: Path,
    expected_dim: int,
    validate_npz_contents: bool,
) -> list[str]:
    issues: list[str] = []
    if not sidecar_path.exists():
        return ["missing"]
    if not sidecar_path.is_file():
        return ["not_file"]
    if sidecar_path.suffix.casefold() != ".npz":
        issues.append("not_npz")
    if not validate_npz_contents:
        return issues

    try:
        with np.load(sidecar_path, allow_pickle=False) as data:
            if "features" not in set(data.files):
                issues.append("missing_features")
            else:
                features = data["features"]
                if features.shape != (expected_dim,):
                    issues.append("feature_shape_mismatch")
                if not np.isfinite(features).all():
                    issues.append("feature_nonfinite")
    except Exception:
        issues.append("npz_unreadable")
    return issues


def _count_issue(issue_counts: Counter[str], issue: str) -> None:
    issue_counts[issue] += 1


def forbidden_extra_columns(fieldnames: Sequence[str]) -> list[str]:
    required = set(REQUIRED_COLUMNS)
    violations = []
    for column in fieldnames:
        normalized = str(column).strip().casefold()
        if normalized in required:
            continue
        if any(token in normalized for token in FORBIDDEN_EXTRA_COLUMN_TOKENS):
            violations.append(str(column))
    return violations


def audit_rows(
    *,
    rows: Sequence[dict[str, str]],
    fieldnames: Sequence[str],
    expected_rows: Optional[int],
    expected_split: str,
    content_pe_cache_dir: Path,
    content_pe_v2_cache_dir: Path,
    expected_content_pe_v1_dim: int,
    expected_content_pe_v2_dim: int,
    validate_npz_contents: bool,
    sample_missing_examples: int,
) -> dict:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    forbidden_columns = forbidden_extra_columns(fieldnames)
    issue_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    missing_examples: dict[str, list[dict[str, str]]] = {"cache_path": [], "content_pe_v1": [], "content_pe_v2": []}
    seen_sha: dict[str, dict[str, str]] = {}
    seen_sample_index: set[str] = set()

    for row in rows:
        split = str(row.get("split") or "").strip()
        label = str(row.get("label") or "").strip()
        sha = str(row.get("source_sha256") or "").strip().casefold()
        sample_index = str(row.get("sample_index") or "").strip()
        split_counts[split] += 1
        label_counts[label] += 1
        if split != expected_split:
            issue_counts["unexpected_split"] += 1
        if label not in {"0", "1"}:
            issue_counts["invalid_label"] += 1
        valid_sha = is_valid_sha256(sha)
        if not valid_sha:
            issue_counts["invalid_source_sha256"] += 1
        if not sample_index.isdigit():
            issue_counts["invalid_sample_index"] += 1
        elif sample_index in seen_sample_index:
            issue_counts["duplicate_sample_index"] += 1
        else:
            seen_sample_index.add(sample_index)
        if valid_sha:
            previous_label = seen_sha.get(sha)
            if previous_label is None:
                seen_sha[sha] = {"label": label, "sample_index": sample_index}
            else:
                if previous_label.get("sample_index") != sample_index:
                    issue_counts["duplicate_source_sha256"] += 1
                if previous_label.get("label") != label:
                    issue_counts["source_sha256_label_conflict"] += 1
        try:
            probability = float(row.get("prob_malicious", ""))
        except ValueError:
            issue_counts["invalid_prob_malicious"] += 1
        else:
            if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
                issue_counts["prob_malicious_out_of_range"] += 1

        cache_path_text = str(row.get("cache_path") or "").strip()
        if not cache_path_text:
            issue_counts["blank_cache_path"] += 1
            if len(missing_examples["cache_path"]) < sample_missing_examples:
                missing_examples["cache_path"].append(_example(row))
        else:
            cache_path = resolve_path(Path(cache_path_text))
            cache_issues = _audit_main_cache(
                cache_path=cache_path,
                expected_label=label,
                expected_sha256=sha,
                validate_npz_contents=validate_npz_contents and valid_sha and label in {"0", "1"},
            )
            for issue in cache_issues:
                _count_issue(issue_counts, issue)
            if cache_issues and len(missing_examples["cache_path"]) < sample_missing_examples:
                missing_examples["cache_path"].append(_example(row))

        sidecar_specs = [
            ("content_pe_v1", content_pe_cache_dir, expected_content_pe_v1_dim),
            ("content_pe_v2", content_pe_v2_cache_dir, expected_content_pe_v2_dim),
        ]
        for key, cache_dir, expected_dim in sidecar_specs:
            path = sidecar_cache_path(cache_dir, row)
            if path is None:
                continue
            sidecar_issues = _audit_sidecar(
                sidecar_path=path,
                expected_dim=expected_dim,
                validate_npz_contents=validate_npz_contents,
            )
            for issue in sidecar_issues:
                _count_issue(issue_counts, f"{key}_{issue}")
            if sidecar_issues and len(missing_examples[key]) < sample_missing_examples:
                missing_examples[key].append(_example(row))

    if expected_rows is not None and len(rows) != expected_rows:
        issue_counts["row_count_mismatch_expected"] += 1
    if missing_columns:
        issue_counts["missing_required_columns"] += 1
    if forbidden_columns:
        issue_counts["forbidden_extra_columns"] += 1

    return {
        "rows": len(rows),
        "expected_rows": expected_rows,
        "expected_split": expected_split,
        "missing_columns": missing_columns,
        "forbidden_extra_columns": forbidden_columns,
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "missing_examples": missing_examples,
        "ready": not issue_counts,
    }


def audit_cross_split_identity(train_rows: Sequence[dict[str, str]], val_rows: Sequence[dict[str, str]]) -> dict:
    train_sha = {
        str(row.get("source_sha256") or "").strip().casefold()
        for row in train_rows
        if is_valid_sha256(row.get("source_sha256"))
    }
    val_sha = {
        str(row.get("source_sha256") or "").strip().casefold()
        for row in val_rows
        if is_valid_sha256(row.get("source_sha256"))
    }
    train_sample_indexes = {
        str(row.get("sample_index") or "").strip()
        for row in train_rows
        if str(row.get("sample_index") or "").strip().isdigit()
    }
    val_sample_indexes = {
        str(row.get("sample_index") or "").strip()
        for row in val_rows
        if str(row.get("sample_index") or "").strip().isdigit()
    }
    sha_overlap = sorted(train_sha & val_sha)
    sample_index_overlap = sorted(train_sample_indexes & val_sample_indexes, key=lambda value: int(value))
    issue_counts: Counter[str] = Counter()
    if sha_overlap:
        issue_counts["train_val_source_sha256_overlap"] += len(sha_overlap)
    if sample_index_overlap:
        issue_counts["train_val_sample_index_overlap"] += len(sample_index_overlap)
    return {
        "issue_counts": dict(sorted(issue_counts.items())),
        "source_sha256_overlap_examples": sha_overlap[:5],
        "sample_index_overlap_examples": sample_index_overlap[:5],
        "ready": not issue_counts,
    }


def _example(row: dict[str, str]) -> dict[str, str]:
    return {
        "source_sha256": str(row.get("source_sha256") or ""),
        "sample_index": str(row.get("sample_index") or ""),
        "split": str(row.get("split") or ""),
        "label": str(row.get("label") or ""),
    }


def audit_loop127_content_cross_readiness(
    *,
    train_predictions: Path,
    val_predictions: Path,
    content_pe_cache_dir: Path,
    content_pe_v2_cache_dir: Path,
    output_json: Path,
    expected_train_rows: int = 20000,
    expected_val_rows: int = 20000,
    expected_test_rows: int = 160000,
    expected_total_rows: int = 200000,
    expected_content_pe_v1_dim: int = DEFAULT_CONTENT_PE_V1_DIM,
    expected_content_pe_v2_dim: int = DEFAULT_CONTENT_PE_V2_DIM,
    validate_npz_contents: bool = True,
    sample_missing_examples: int = 5,
) -> dict:
    train_rows, train_fields = read_csv_rows(train_predictions)
    val_rows, val_fields = read_csv_rows(val_predictions)
    train = audit_rows(
        rows=train_rows,
        fieldnames=train_fields,
        expected_rows=expected_train_rows,
        expected_split="train",
        content_pe_cache_dir=content_pe_cache_dir,
        content_pe_v2_cache_dir=content_pe_v2_cache_dir,
        expected_content_pe_v1_dim=expected_content_pe_v1_dim,
        expected_content_pe_v2_dim=expected_content_pe_v2_dim,
        validate_npz_contents=validate_npz_contents,
        sample_missing_examples=sample_missing_examples,
    )
    val = audit_rows(
        rows=val_rows,
        fieldnames=val_fields,
        expected_rows=expected_val_rows,
        expected_split="val",
        content_pe_cache_dir=content_pe_cache_dir,
        content_pe_v2_cache_dir=content_pe_v2_cache_dir,
        expected_content_pe_v1_dim=expected_content_pe_v1_dim,
        expected_content_pe_v2_dim=expected_content_pe_v2_dim,
        validate_npz_contents=validate_npz_contents,
        sample_missing_examples=sample_missing_examples,
    )
    cross_split = audit_cross_split_identity(train_rows, val_rows)
    blockers = []
    if not train["ready"]:
        blockers.append("train_inputs_not_ready")
    if not val["ready"]:
        blockers.append("val_inputs_not_ready")
    if not cross_split["ready"]:
        blockers.append("train_val_identity_overlap")
    split_contract = {
        "expected_train_rows": expected_train_rows,
        "expected_val_rows": expected_val_rows,
        "expected_test_rows": expected_test_rows,
        "expected_total_rows": expected_total_rows,
        "known_train_val_rows": len(train_rows) + len(val_rows),
        "implied_total_rows": len(train_rows) + len(val_rows) + expected_test_rows,
        "matches_20w_contract": (
            len(train_rows) == expected_train_rows
            and len(val_rows) == expected_val_rows
            and len(train_rows) + len(val_rows) + expected_test_rows == expected_total_rows
        ),
        "note": "This gate validates Loop127 Train/Val inputs and preserves the locked 20k/20k/160k contract by count; it does not inspect full test rows.",
    }
    if not split_contract["matches_20w_contract"]:
        blockers.append("split_contract_count_mismatch")
    payload = {
        "schema": "axon_loop127_content_cross_readiness_v1",
        "protocol": "read-only input readiness audit; no matrix build, no model fitting, no threshold selection",
        "identity_policy": "source_sha256/cache_path/source_path/sample_index are used only for cache lookup and row alignment, not as model evidence.",
        "npz_validation_policy": (
            "Main cache validation reads label/source_sha256 only; sidecar validation reads small features arrays for "
            "shape/finite checks. No byte matrices or model inputs are built."
        ),
        "train_predictions": str(resolve_path(train_predictions)),
        "val_predictions": str(resolve_path(val_predictions)),
        "content_pe_cache_dir": str(resolve_path(content_pe_cache_dir)),
        "content_pe_v2_cache_dir": str(resolve_path(content_pe_v2_cache_dir)),
        "expected_sidecar_dims": {
            "content_pe_v1": expected_content_pe_v1_dim,
            "content_pe_v2": expected_content_pe_v2_dim,
        },
        "split_contract": split_contract,
        "train": train,
        "val": val,
        "cross_split_identity": cross_split,
        "blockers": blockers,
        "ready_for_loop43_val_only": not blockers,
    }
    resolved_output = resolve_path(output_json)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Loop127 content-cross input readiness.")
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--content-pe-cache-dir", type=Path, required=True)
    parser.add_argument("--content-pe-v2-cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-train-rows", type=int, default=20000)
    parser.add_argument("--expected-val-rows", type=int, default=20000)
    parser.add_argument("--expected-test-rows", type=int, default=160000)
    parser.add_argument("--expected-total-rows", type=int, default=200000)
    parser.add_argument("--expected-content-pe-v1-dim", type=int, default=DEFAULT_CONTENT_PE_V1_DIM)
    parser.add_argument("--expected-content-pe-v2-dim", type=int, default=DEFAULT_CONTENT_PE_V2_DIM)
    parser.add_argument("--skip-npz-content-validation", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_loop127_content_cross_readiness(
        train_predictions=args.train_predictions,
        val_predictions=args.val_predictions,
        content_pe_cache_dir=args.content_pe_cache_dir,
        content_pe_v2_cache_dir=args.content_pe_v2_cache_dir,
        output_json=args.output_json,
        expected_train_rows=args.expected_train_rows,
        expected_val_rows=args.expected_val_rows,
        expected_test_rows=args.expected_test_rows,
        expected_total_rows=args.expected_total_rows,
        expected_content_pe_v1_dim=args.expected_content_pe_v1_dim,
        expected_content_pe_v2_dim=args.expected_content_pe_v2_dim,
        validate_npz_contents=not args.skip_npz_content_validation,
    )
    print(json.dumps({"ready_for_loop43_val_only": payload["ready_for_loop43_val_only"], "blockers": payload["blockers"]}, indent=2, ensure_ascii=False))
    return 0 if payload["ready_for_loop43_val_only"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
