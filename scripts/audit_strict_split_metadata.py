#!/usr/bin/env python3
"""Audit split/manifest metadata without using names as label evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence


EXPECTED_20W_SPLIT_COUNTS = {"train": 20000, "val": 20000, "test": 160000}
EXPECTED_20W_LABEL_SPLIT_COUNTS = {
    "train": {"0": 10000, "1": 10000},
    "val": {"0": 10000, "1": 10000},
    "test": {"0": 80000, "1": 80000},
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def is_valid_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def scalar_text(value: Any) -> str:
    import numpy as np

    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(value)


def read_split_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"source_path", "split", "label", "source_sha256"}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"Split CSV missing required strict metadata columns: {missing}")
        return [dict(row) for row in reader if row.get("source_path")]


def read_manifest_samples(path: Path) -> list[dict]:
    payload = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Manifest must contain a samples list")
    return [dict(sample) for sample in samples]


def resolve_cache_path(cache_path_text: str, manifest_json: Path) -> Path:
    raw = Path(cache_path_text)
    if raw.is_absolute():
        return raw
    manifest_dir = resolve_path(manifest_json).parent
    candidates = [manifest_dir / raw, manifest_dir / raw.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_npz_metadata(cache_path: Path) -> tuple[Optional[int], Optional[str], list[str]]:
    import numpy as np

    issues = []
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if "label" not in data.files:
                issues.append("npz_missing_label")
                label = None
            else:
                label = int(np.asarray(data["label"]).item())
            if "source_sha256" not in data.files:
                issues.append("npz_missing_source_sha256")
                source_sha256 = None
            else:
                source_sha256 = scalar_text(data["source_sha256"]).strip().casefold()
            for field in ["byte_sequence", "pe_features"]:
                if field not in data.files:
                    issues.append(f"npz_missing_{field}")
    except Exception as exc:
        return None, None, [f"npz_read_failed:{type(exc).__name__}"]
    return label, source_sha256, issues


def validate_manifest(samples: Sequence[dict]) -> tuple[dict[str, list[dict]], Counter]:
    by_sha: dict[str, list[dict]] = defaultdict(list)
    issue_counts: Counter = Counter()
    for sample in samples:
        source_sha = str(sample.get("source_sha256") or "").strip().casefold()
        if not is_valid_sha256(source_sha):
            issue_counts["manifest_invalid_source_sha256"] += 1
            continue
        try:
            label = int(sample.get("label"))
        except (TypeError, ValueError):
            issue_counts["manifest_invalid_label"] += 1
            continue
        if label not in {0, 1}:
            issue_counts["manifest_invalid_label"] += 1
            continue
        normalized = dict(sample)
        normalized["label"] = label
        normalized["source_sha256"] = source_sha
        by_sha[source_sha].append(normalized)
    return by_sha, issue_counts


def audit_strict_split_metadata(
    *,
    split_csv: Path,
    manifest_json: Path,
    validate_npz: bool = True,
    expect_20w: bool = False,
    strict_unique_source_sha256: bool = False,
) -> dict:
    split_rows = read_split_rows(split_csv)
    manifest_samples = read_manifest_samples(manifest_json)
    manifest_by_sha, manifest_issue_counts = validate_manifest(manifest_samples)

    row_issues = []
    split_counts: Counter = Counter()
    label_counts: Counter = Counter()
    label_split_counts: dict[str, Counter] = defaultdict(Counter)
    split_sha_counts: Counter = Counter()
    match_counts: Counter = Counter()
    metadata_issue_counts: Counter = Counter(manifest_issue_counts)

    for row_index, row in enumerate(split_rows):
        issues = []
        split = str(row.get("split") or "").strip()
        label_text = str(row.get("label") or "").strip()
        source_sha = str(row.get("source_sha256") or "").strip().casefold()

        if split not in {"train", "val", "test"}:
            issues.append("split_invalid")
        else:
            split_counts[split] += 1
        if label_text not in {"0", "1"}:
            issues.append("split_label_invalid")
            label = None
        else:
            label = int(label_text)
            label_counts[label_text] += 1
            if split in {"train", "val", "test"}:
                label_split_counts[split][label_text] += 1
        if not is_valid_sha256(source_sha):
            issues.append("split_invalid_source_sha256")
        else:
            split_sha_counts[source_sha] += 1

        manifest_matches = manifest_by_sha.get(source_sha, []) if is_valid_sha256(source_sha) else []
        if not manifest_matches:
            issues.append("manifest_missing_source_sha256")
        else:
            manifest_labels = {int(sample["label"]) for sample in manifest_matches}
            if len(manifest_labels) > 1:
                issues.append("manifest_conflicting_labels_for_source_sha256")
            if label is not None and label not in manifest_labels:
                issues.append("label_mismatch_split_manifest")
            else:
                match_counts["source_sha256"] += 1
                sample = next(
                    item for item in manifest_matches if label is None or int(item["label"]) == label
                )
                if validate_npz:
                    cache_path = resolve_cache_path(str(sample.get("cache_path", "")), manifest_json)
                    if not cache_path.exists():
                        issues.append("cache_file_missing")
                    else:
                        npz_label, npz_sha, npz_issues = read_npz_metadata(cache_path)
                        issues.extend(npz_issues)
                        if label is not None and npz_label is not None and npz_label != label:
                            issues.append("label_mismatch_split_npz")
                        if npz_sha is not None and npz_sha != source_sha:
                            issues.append("source_sha256_mismatch_split_npz")

        if issues:
            for issue in issues:
                metadata_issue_counts[issue] += 1
            row_issues.append({
                "row_index": row_index,
                "sample_index": row.get("sample_index", ""),
                "split": split,
                "label": label_text,
                "source_sha256": source_sha,
                "issues": issues,
            })

    shape_failures = []
    if expect_20w:
        if dict(split_counts) != EXPECTED_20W_SPLIT_COUNTS:
            shape_failures.append(f"split_counts:{dict(split_counts)}")
        normalized_label_split_counts = {
            split: dict(sorted(counts.items())) for split, counts in label_split_counts.items()
        }
        if normalized_label_split_counts != EXPECTED_20W_LABEL_SPLIT_COUNTS:
            shape_failures.append(f"label_split_counts:{normalized_label_split_counts}")
    if strict_unique_source_sha256:
        duplicate_hashes = sum(1 for count in split_sha_counts.values() if count > 1)
        if duplicate_hashes:
            shape_failures.append(f"duplicate_source_sha256:{duplicate_hashes}")

    audit_ready = not row_issues and not shape_failures
    return {
        "schema": "axon_strict_split_metadata_audit_v1",
        "identity_feature_policy": (
            "source_path/path/name/extension/directory are alignment and loading fields only; "
            "source_sha256 is content identity only; none are malware or benign evidence"
        ),
        "split_csv": str(resolve_path(split_csv)),
        "manifest_json": str(resolve_path(manifest_json)),
        "validate_npz": bool(validate_npz),
        "expect_20w": bool(expect_20w),
        "strict_unique_source_sha256": bool(strict_unique_source_sha256),
        "rows": len(split_rows),
        "manifest_samples": len(manifest_samples),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "label_split_counts": {
            split: dict(sorted(counts.items())) for split, counts in sorted(label_split_counts.items())
        },
        "match_counts": dict(sorted(match_counts.items())),
        "metadata_issue_counts": dict(sorted(metadata_issue_counts.items())),
        "row_issue_count": len(row_issues),
        "row_issue_examples": row_issues[:50],
        "shape_failures": shape_failures,
        "audit_ready": audit_ready,
        "ready_for": {
            "train_val_only": audit_ready,
            "test10k": False,
            "full_test": False,
        },
        "notes": [
            "This audit never infers labels from file names, paths, directories, or extensions.",
            "Passing this audit only proves split/cache metadata consistency; it is not Test-10k or full-test authorization.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit strict split label/source-hash metadata.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--no-validate-npz", action="store_true")
    parser.add_argument("--expect-20w", action="store_true")
    parser.add_argument("--strict-unique-source-sha256", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_strict_split_metadata(
        split_csv=args.split_csv,
        manifest_json=args.manifest_json,
        validate_npz=not bool(args.no_validate_npz),
        expect_20w=bool(args.expect_20w),
        strict_unique_source_sha256=bool(args.strict_unique_source_sha256),
    )
    output_path = resolve_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["audit_ready"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
