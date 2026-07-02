#!/usr/bin/env python3
"""Build a read-only content-health audit for Loop39 conflict rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pefile

    PEFILE_AVAILABLE = True
except ImportError:
    PEFILE_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]


EXPECTED_CACHE_SHAPES = {
    "byte_sequence": (8192,),
    "pe_features": (256,),
    "stat_features": (49,),
    "lightweight_features": (256,),
}


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], Counter[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("samples", data if isinstance(data, list) else [])
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    sha_counts: Counter[str] = Counter()
    for row in rows:
        source_path = str(row.get("source_path", ""))
        label = str(row.get("label", ""))
        source_sha256 = str(row.get("source_sha256", "")).lower()
        by_key[(source_path, label)] = row
        if source_sha256:
            sha_counts[source_sha256] += 1
    return by_key, sha_counts


def check_cache_npz(cache_path: Path, expected_label: str, expected_sha256: str) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    facts: dict[str, Any] = {}
    if not cache_path.exists():
        return ["cache_missing"], facts

    try:
        npz = np.load(cache_path, allow_pickle=False)
    except Exception as exc:
        return [f"cache_load_failed:{type(exc).__name__}"], facts

    with npz:
        for key, expected_shape in EXPECTED_CACHE_SHAPES.items():
            if key not in npz.files:
                issues.append(f"cache_missing_key:{key}")
                continue
            arr = npz[key]
            facts[f"{key}_shape"] = "x".join(str(v) for v in arr.shape)
            facts[f"{key}_dtype"] = str(arr.dtype)
            if tuple(arr.shape) != expected_shape:
                issues.append(f"cache_bad_shape:{key}:{arr.shape}")
            if key == "byte_sequence":
                if arr.dtype != np.uint8:
                    issues.append(f"cache_bad_dtype:{key}:{arr.dtype}")
            elif not np.isfinite(arr).all():
                issues.append(f"cache_nonfinite:{key}")

        if "label" not in npz.files:
            issues.append("cache_missing_key:label")
        else:
            cache_label = str(int(np.asarray(npz["label"]).item()))
            facts["cache_label"] = cache_label
            if cache_label != expected_label:
                issues.append(f"cache_label_mismatch:{cache_label}!={expected_label}")

        if "source_sha256" not in npz.files:
            issues.append("cache_missing_key:source_sha256")
        else:
            cache_sha = str(np.asarray(npz["source_sha256"]).item()).lower()
            facts["cache_source_sha256"] = cache_sha
            if expected_sha256 and cache_sha != expected_sha256.lower():
                issues.append("cache_source_sha256_mismatch")

    return issues, facts


def pe_content_facts(source_path: Path) -> tuple[list[str], dict[str, Any]]:
    if not source_path.exists():
        return ["source_missing"], {}
    if not PEFILE_AVAILABLE:
        return ["pefile_unavailable"], {}
    try:
        pe = pefile.PE(str(source_path), fast_load=True)
    except Exception as exc:
        return [f"strict_pe_parse_failed:{type(exc).__name__}"], {}

    try:
        try:
            pe.parse_data_directories(
                directories=[
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"],
                    pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_SECURITY"],
                ]
            )
        except Exception:
            pass
        security_dir = pe.OPTIONAL_HEADER.DATA_DIRECTORY[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_SECURITY"]
        ]
        facts = {
            "strict_pe_parse_ok": True,
            "pe_machine": int(pe.FILE_HEADER.Machine),
            "pe_sections": int(pe.FILE_HEADER.NumberOfSections),
            "pe_is_dll": bool(pe.FILE_HEADER.Characteristics & 0x2000),
            "pe_has_imports": hasattr(pe, "DIRECTORY_ENTRY_IMPORT"),
            "pe_has_exports": hasattr(pe, "DIRECTORY_ENTRY_EXPORT"),
            "pe_has_resources": hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"),
            "pe_has_security_directory": bool(security_dir.VirtualAddress and security_dir.Size),
            "pe_overlay_size": int(len(pe.get_overlay() or b"")),
        }
        return [], facts
    finally:
        pe.close()


def build_audit(
    *,
    queue_csv: Path,
    split_csv: Path,
    manifest_json: Path,
    output_csv: Path,
    output_json: Path,
    lane: str | None,
    limit: int | None,
) -> dict[str, Any]:
    queue_rows = read_csv_rows(queue_csv)
    if lane:
        queue_rows = [row for row in queue_rows if row.get("review_lane") == lane]
    if limit is not None:
        queue_rows = queue_rows[:limit]

    split_rows = read_csv_rows(split_csv)
    split_by_key = {(row["source_path"], row["label"]): row for row in split_rows}
    manifest_by_key, manifest_sha_counts = load_manifest(manifest_json)
    queue_sha_counts = Counter(str(row.get("source_sha256", "")).lower() for row in queue_rows)

    audited_rows: list[dict[str, Any]] = []
    summary_counts: Counter[str] = Counter()
    for row in queue_rows:
        source_path_text = row.get("source_path", "")
        label = str(row.get("label", ""))
        source_sha256 = str(row.get("source_sha256", "")).lower()
        key = (source_path_text, label)
        split_row = split_by_key.get(key)
        manifest_row = manifest_by_key.get(key)

        issues: list[str] = []
        facts: dict[str, Any] = {}
        if split_row is None:
            issues.append("not_in_active_split")
        else:
            facts["active_split"] = split_row.get("split", "")
            facts["active_sample_index"] = split_row.get("sample_index", "")

        if manifest_row is None:
            issues.append("not_in_manifest")
        else:
            cache_path = Path(str(manifest_row.get("cache_path", "")))
            cache_issues, cache_facts = check_cache_npz(cache_path, label, source_sha256)
            issues.extend(cache_issues)
            facts.update(cache_facts)
            facts["cache_path_exists"] = cache_path.exists()

        source_path = Path(source_path_text)
        if source_path.exists():
            try:
                actual_sha = file_sha256(source_path)
                facts["source_file_sha256"] = actual_sha
                if source_sha256 and actual_sha.lower() != source_sha256:
                    issues.append("source_file_sha256_mismatch")
            except Exception as exc:
                issues.append(f"source_sha256_failed:{type(exc).__name__}")
        else:
            issues.append("source_missing")

        pe_issues, pe_facts = pe_content_facts(source_path)
        issues.extend(pe_issues)
        facts.update(pe_facts)

        facts["queue_source_sha256_group_size"] = queue_sha_counts[source_sha256]
        facts["manifest_source_sha256_group_size"] = manifest_sha_counts[source_sha256]
        if queue_sha_counts[source_sha256] > 1 or manifest_sha_counts[source_sha256] > 1:
            issues.append("duplicate_source_sha256_group")

        objective_issue_flags = [
            issue
            for issue in issues
            if issue.startswith(
                (
                    "source_missing",
                    "source_file_sha256_mismatch",
                    "cache_",
                    "not_in_manifest",
                    "strict_pe_parse_failed",
                )
            )
            or issue == "not_in_active_split"
        ]

        for issue in issues:
            summary_counts[issue.split(":", 1)[0]] += 1

        audited_rows.append(
            {
                "review_priority_rank": row.get("review_priority_rank", ""),
                "review_lane": row.get("review_lane", ""),
                "conflict_bucket": row.get("conflict_bucket", ""),
                "label": label,
                "loop28_error_type": row.get("loop28_error_type", ""),
                "loop28_score": row.get("loop28_score", ""),
                "source_sha256": source_sha256,
                "source_path": source_path_text,
                "manual_label_verdict": row.get("manual_label_verdict", ""),
                "recommended_action": row.get("recommended_action", ""),
                "objective_issue_count": len(objective_issue_flags),
                "objective_issue_flags": "|".join(objective_issue_flags),
                "all_issue_flags": "|".join(issues),
                **facts,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in audited_rows for key in row})
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audited_rows)

    lane_counts = Counter(row.get("review_lane", "") for row in audited_rows)
    error_type_counts = Counter(row.get("loop28_error_type", "") for row in audited_rows)
    objective_issue_rows = [row for row in audited_rows if int(row["objective_issue_count"]) > 0]
    report = {
        "schema": "axon_loop50_conflict_content_audit_v1",
        "policy": (
            "Read-only content/cache health audit. Filename/path/extension/directory/hash/sample id/split/row "
            "order are not model evidence or relabel evidence."
        ),
        "queue_csv": str(queue_csv),
        "split_csv": str(split_csv),
        "manifest_json": str(manifest_json),
        "lane_filter": lane,
        "limit": limit,
        "rows": len(audited_rows),
        "lane_counts": dict(sorted(lane_counts.items())),
        "error_type_counts": dict(sorted(error_type_counts.items())),
        "manual_label_verdict_blank_count": sum(1 for row in audited_rows if not row.get("manual_label_verdict")),
        "recommended_action_blank_count": sum(1 for row in audited_rows if not row.get("recommended_action")),
        "objective_issue_row_count": len(objective_issue_rows),
        "issue_counts": dict(sorted(summary_counts.items())),
        "outputs": {
            "audit_csv": str(output_csv),
            "summary_json": str(output_json),
        },
        "replacement_rule": (
            "If an objective content/cache issue is independently confirmed, replace with a fresh same-label "
            "candidate and preserve the exact 200000-row split. This audit does not alter labels or splits."
        ),
    }
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue-csv",
        default="reports/random_20w_split/loop39_loop28_conflict_adjudication/loop28_conflict_adjudication_queue.csv",
    )
    parser.add_argument("--split-csv", default="reports/random_20w_split/loop27_corrected_split.csv")
    parser.add_argument("--manifest-json", default="data/.cache/manifest_38672ba0.json")
    parser.add_argument(
        "--output-csv",
        default="reports/random_20w_split/loop50_conflict_content_audit/loop50_conflict_content_audit.csv",
    )
    parser.add_argument(
        "--output-json",
        default="reports/random_20w_split/loop50_conflict_content_audit/loop50_conflict_content_audit_summary.json",
    )
    parser.add_argument("--lane", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_audit(
        queue_csv=resolve_path(args.queue_csv),
        split_csv=resolve_path(args.split_csv),
        manifest_json=resolve_path(args.manifest_json),
        output_csv=resolve_path(args.output_csv),
        output_json=resolve_path(args.output_json),
        lane=args.lane,
        limit=args.limit,
    )
    print(
        "[loop50] rows=",
        report["rows"],
        "objective_issue_rows=",
        report["objective_issue_row_count"],
        "issue_counts=",
        report["issue_counts"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
