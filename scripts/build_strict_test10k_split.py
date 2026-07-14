#!/usr/bin/env python3
"""Build a locked strict Test-10k split from the corrected 20w split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def is_valid_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = {"source_path", "source_sha256", "label", "sample_index", "split"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"Strict split missing columns: {missing}")
        return list(reader), fieldnames


def build_test10k(*, split_csv: Path, output_csv: Path, per_label: int = 5000) -> dict:
    rows, fieldnames = read_rows(split_csv)
    test_rows = [row for row in rows if str(row.get("split", "")).strip() == "test"]
    selected: list[dict[str, str]] = []
    label_counts: Counter = Counter()
    issue_counts: Counter = Counter()
    seen_keys: set[tuple[str, str]] = set()

    for row in test_rows:
        source_sha = str(row.get("source_sha256") or "").strip().casefold()
        label = str(row.get("label") or "").strip()
        sample_index = str(row.get("sample_index") or "").strip()
        if not is_valid_sha256(source_sha):
            issue_counts["invalid_source_sha256"] += 1
            continue
        if label not in {"0", "1"}:
            issue_counts["invalid_label"] += 1
            continue
        if not sample_index:
            issue_counts["missing_sample_index"] += 1
            continue
        key = (source_sha, sample_index)
        if key in seen_keys:
            issue_counts["duplicate_source_sha256_sample_index"] += 1
            continue
        if label_counts[label] >= per_label:
            continue
        normalized = dict(row)
        normalized["source_sha256"] = source_sha
        normalized["label"] = label
        normalized["split"] = "test10k"
        selected.append(normalized)
        seen_keys.add(key)
        label_counts[label] += 1
        if label_counts["0"] >= per_label and label_counts["1"] >= per_label:
            break

    if label_counts["0"] != per_label or label_counts["1"] != per_label:
        raise ValueError(f"Could not build balanced Test-10k: {dict(label_counts)}")
    if issue_counts:
        raise ValueError(f"Strict Test-10k source rows had issues: {dict(issue_counts)}")

    resolved_output = resolve_path(output_csv)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)

    return {
        "schema": "axon_strict_test10k_split_v1",
        "identity_feature_policy": "Rows are selected from frozen test split order; source_sha256/sample_index validate identity only; path/name/directory/extension are ignored.",
        "source_split_csv": str(resolve_path(split_csv)),
        "output_csv": str(resolved_output),
        "source_test_rows": len(test_rows),
        "selected_rows": len(selected),
        "label_counts": dict(sorted(label_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "ready_for": {"test10k": True, "full_test": False},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a strict balanced Test-10k split.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=5000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_test10k(split_csv=args.split_csv, output_csv=args.output_csv, per_label=int(args.per_label))
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected_rows": payload["selected_rows"], "label_counts": payload["label_counts"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
