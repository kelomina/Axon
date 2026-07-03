#!/usr/bin/env python3
"""Enrich a split CSV with content SHA-256 metadata without using names as labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_20W_SPLIT_COUNTS = {"train": 20000, "val": 20000, "test": 160000}
EXPECTED_20W_LABEL_SPLIT_COUNTS = {
    "train": {"0": 10000, "1": 10000},
    "val": {"0": 10000, "1": 10000},
    "test": {"0": 80000, "1": 80000},
}
DEFAULT_FIELDNAMES = ["source_path", "source_sha256", "label", "sample_index", "split"]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def is_valid_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split_rows(path: Path) -> tuple[list[dict], list[str]]:
    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = {"source_path", "label", "split"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"Split CSV missing required columns: {missing}")
        return [dict(row) for row in reader], fieldnames


def read_manifest_samples(path: Path) -> list[dict]:
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Manifest must contain a samples list")
    return [dict(sample) for sample in samples]


def build_manifest_by_sha(samples: Sequence[dict]) -> tuple[dict[str, list[dict]], Counter]:
    by_sha: dict[str, list[dict]] = defaultdict(list)
    issue_counts: Counter = Counter()
    for sample in samples:
        source_sha = str(sample.get("source_sha256") or "").strip().casefold()
        if not is_valid_sha256(source_sha):
            issue_counts["manifest_invalid_source_sha256"] += 1
            continue
        label_text = str(sample.get("label", "")).strip()
        if label_text not in {"0", "1"}:
            issue_counts["manifest_invalid_label"] += 1
            continue
        normalized = dict(sample)
        normalized["source_sha256"] = source_sha
        normalized["label"] = label_text
        by_sha[source_sha].append(normalized)
    return by_sha, issue_counts


def output_fieldnames(input_fieldnames: Sequence[str]) -> list[str]:
    fields = list(input_fieldnames)
    if "source_sha256" not in fields:
        if "source_path" in fields:
            insert_at = fields.index("source_path") + 1
            fields.insert(insert_at, "source_sha256")
        else:
            fields.insert(0, "source_sha256")
    ordered = []
    for field in DEFAULT_FIELDNAMES:
        if field in fields and field not in ordered:
            ordered.append(field)
    for field in fields:
        if field not in ordered:
            ordered.append(field)
    return ordered


def split_shape_failures(rows: Sequence[dict]) -> tuple[list[str], dict]:
    split_counts: Counter = Counter()
    label_counts: Counter = Counter()
    label_split_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        split = str(row.get("split", "")).strip()
        label = str(row.get("label", "")).strip()
        split_counts[split] += 1
        label_counts[label] += 1
        label_split_counts[split][label] += 1

    normalized_label_split_counts = {
        split: dict(sorted(counts.items())) for split, counts in sorted(label_split_counts.items())
    }
    summary = {
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "label_split_counts": normalized_label_split_counts,
    }
    failures = []
    if dict(split_counts) != EXPECTED_20W_SPLIT_COUNTS:
        failures.append(f"split_counts:{dict(sorted(split_counts.items()))}")
    if normalized_label_split_counts != EXPECTED_20W_LABEL_SPLIT_COUNTS:
        failures.append(f"label_split_counts:{normalized_label_split_counts}")
    return failures, summary


def enrich_strict_split_metadata(
    *,
    split_csv: Path,
    manifest_json: Path,
    verify_existing_hash_from_source: bool = False,
    expect_20w: bool = False,
) -> tuple[list[dict], dict]:
    split_rows, input_fields = read_split_rows(split_csv)
    manifest_samples = read_manifest_samples(manifest_json)
    manifest_by_sha, manifest_issue_counts = build_manifest_by_sha(manifest_samples)

    enriched_rows: list[dict] = []
    row_issues: list[dict] = []
    issue_counts: Counter = Counter(manifest_issue_counts)
    hash_source_counts: Counter = Counter()
    manifest_match_counts: Counter = Counter()

    for row_index, row in enumerate(split_rows):
        enriched = dict(row)
        issues: list[str] = []
        split = str(enriched.get("split", "")).strip()
        label = str(enriched.get("label", "")).strip()
        source_path_text = str(enriched.get("source_path", "")).strip()
        source_sha = str(enriched.get("source_sha256", "")).strip().casefold()

        if split not in {"train", "val", "test"}:
            issues.append("split_invalid")
        if label not in {"0", "1"}:
            issues.append("split_label_invalid")

        if source_sha:
            if not is_valid_sha256(source_sha):
                issues.append("split_invalid_source_sha256")
                hash_source_counts["invalid_existing_source_sha256"] += 1
                source_sha = ""
            else:
                hash_source_counts["existing_split_source_sha256"] += 1
                if verify_existing_hash_from_source and source_path_text:
                    source_path = resolve_path(Path(source_path_text))
                    if source_path.exists() and source_path.is_file():
                        actual_sha = sha256_file(source_path)
                        hash_source_counts["verified_existing_hash_from_source_file"] += 1
                        if actual_sha != source_sha:
                            issues.append("source_sha256_mismatch_split_file")
                    else:
                        issues.append("source_file_missing_for_existing_hash_verification")
        else:
            hash_source_counts["missing_split_source_sha256"] += 1
            if not source_path_text:
                issues.append("source_path_missing_for_hash")
            else:
                source_path = resolve_path(Path(source_path_text))
                if not source_path.exists() or not source_path.is_file():
                    issues.append("source_file_missing_for_hash")
                else:
                    source_sha = sha256_file(source_path)
                    hash_source_counts["computed_from_source_file"] += 1

        enriched["source_sha256"] = source_sha
        if source_sha and is_valid_sha256(source_sha):
            manifest_matches = manifest_by_sha.get(source_sha, [])
            if not manifest_matches:
                issues.append("manifest_missing_source_sha256")
            else:
                manifest_labels = {str(sample.get("label", "")).strip() for sample in manifest_matches}
                if len(manifest_labels) > 1:
                    issues.append("manifest_conflicting_labels_for_source_sha256")
                if label in {"0", "1"} and label not in manifest_labels:
                    issues.append("label_mismatch_split_manifest")
                else:
                    manifest_match_counts["source_sha256"] += 1
        elif "split_invalid_source_sha256" not in issues:
            issues.append("split_missing_source_sha256")

        enriched_rows.append(enriched)
        if issues:
            for issue in issues:
                issue_counts[issue] += 1
            row_issues.append(
                {
                    "row_index": row_index,
                    "sample_index": enriched.get("sample_index", ""),
                    "split": split,
                    "label": label,
                    "source_path": source_path_text,
                    "source_sha256": source_sha,
                    "issues": issues,
                }
            )

    shape_failures = []
    split_summary = {
        "rows": len(enriched_rows),
        "split_counts": {},
        "label_counts": {},
        "label_split_counts": {},
    }
    if expect_20w:
        shape_failures, split_summary = split_shape_failures(enriched_rows)
    else:
        _unused_failures, split_summary = split_shape_failures(enriched_rows)

    enrichment_ready = not row_issues and not shape_failures
    summary = {
        "schema": "axon_strict_split_metadata_enrichment_v1",
        "identity_feature_policy": (
            "source_path/path/name/extension/directory are locating and alignment fields only; "
            "this tool never infers labels from names, paths, directories, or extensions. "
            "source_sha256 is content identity only, not malware or benign evidence."
        ),
        "split_csv": str(resolve_path(split_csv)),
        "manifest_json": str(resolve_path(manifest_json)),
        "verify_existing_hash_from_source": bool(verify_existing_hash_from_source),
        "expect_20w": bool(expect_20w),
        "rows": len(enriched_rows),
        "manifest_samples": len(manifest_samples),
        "split_summary": split_summary,
        "hash_source_counts": dict(sorted(hash_source_counts.items())),
        "manifest_match_counts": dict(sorted(manifest_match_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "row_issue_count": len(row_issues),
        "row_issue_examples": row_issues[:50],
        "shape_failures": shape_failures,
        "output_fieldnames": output_fieldnames(input_fields),
        "enrichment_ready": enrichment_ready,
        "memory_leak_profile": {
            "loads_model": False,
            "uses_cuda": False,
            "opens_npz_files": False,
            "reads_raw_files_for_sha256": True,
            "stores_file_bytes_in_memory": False,
        },
        "notes": [
            "Rows without source_sha256 are enriched only by hashing the referenced file content.",
            "Manifest matching is by source_sha256 only; path/name/extension/directory fallback is intentionally forbidden.",
            "Passing this enrichment only prepares strict metadata; run audit_strict_split_metadata.py before training.",
        ],
    }
    return enriched_rows, summary


def write_split_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add strict source_sha256 metadata to a split CSV.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--verify-existing-hash-from-source", action="store_true")
    parser.add_argument("--expect-20w", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows, summary = enrich_strict_split_metadata(
        split_csv=args.split_csv,
        manifest_json=args.manifest_json,
        verify_existing_hash_from_source=bool(args.verify_existing_hash_from_source),
        expect_20w=bool(args.expect_20w),
    )
    write_split_csv(args.output_csv, rows, summary["output_fieldnames"])
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    summary["outputs"] = {
        "csv": str(resolve_path(args.output_csv)),
        "json": str(output_json),
    }
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["enrichment_ready"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
