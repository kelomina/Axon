#!/usr/bin/env python3
"""Audit whether an existing prediction CSV can be reused for a strict split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def is_valid_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def read_rows(path: Path) -> list[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str]:
    return (
        str(row.get("source_sha256") or "").strip().casefold(),
        str(row.get("sample_index") or "").strip(),
    )


def audit_reuse(*, strict_split_csv: Path, predictions_csv: Path, split: str) -> dict:
    split_rows = [row for row in read_rows(strict_split_csv) if str(row.get("split", "")).strip() == split]
    prediction_rows = [row for row in read_rows(predictions_csv) if str(row.get("split", "")).strip() == split]
    split_keys = {row_key(row) for row in split_rows}
    prediction_keys = {row_key(row) for row in prediction_rows}
    split_shas = [key[0] for key in split_keys]
    prediction_shas = [key[0] for key in prediction_keys]
    split_sha_set = set(split_shas)
    prediction_sha_set = set(prediction_shas)

    issue_counts: Counter = Counter()
    for source_sha, sample_index in split_keys:
        if not is_valid_sha256(source_sha):
            issue_counts["strict_split_invalid_source_sha256"] += 1
        if not sample_index:
            issue_counts["strict_split_missing_sample_index"] += 1
    for source_sha, sample_index in prediction_keys:
        if not is_valid_sha256(source_sha):
            issue_counts["prediction_invalid_source_sha256"] += 1
        if not sample_index:
            issue_counts["prediction_missing_sample_index"] += 1

    duplicate_split_shas = sum(count - 1 for count in Counter(split_shas).values() if count > 1)
    duplicate_prediction_shas = sum(count - 1 for count in Counter(prediction_shas).values() if count > 1)

    by_sha_predictions: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        source_sha = str(row.get("source_sha256") or "").strip().casefold()
        by_sha_predictions[source_sha].append(row)
    reusable_by_sha = 0
    ambiguous_by_sha = 0
    label_mismatch_by_sha = 0
    for split_row in split_rows:
        source_sha = str(split_row.get("source_sha256") or "").strip().casefold()
        label = str(split_row.get("label") or "").strip()
        candidates = by_sha_predictions.get(source_sha, [])
        if len(candidates) == 1:
            if str(candidates[0].get("label") or "").strip() == label:
                reusable_by_sha += 1
            else:
                label_mismatch_by_sha += 1
        elif len(candidates) > 1:
            ambiguous_by_sha += 1

    missing_keys = split_keys - prediction_keys
    extra_keys = prediction_keys - split_keys
    missing_shas = split_sha_set - prediction_sha_set
    extra_shas = prediction_sha_set - split_sha_set
    decision = "reusable_exact" if not missing_keys and not extra_keys and not issue_counts else "not_reusable_exact"
    return {
        "schema": "axon_prediction_reuse_for_strict_split_audit_v1",
        "identity_feature_policy": (
            "source_sha256/sample_index are used only to audit row reuse; path/name/directory/extension are ignored."
        ),
        "strict_split_csv": str(resolve_path(strict_split_csv)),
        "predictions_csv": str(resolve_path(predictions_csv)),
        "split": split,
        "decision": decision,
        "strict_rows": len(split_rows),
        "prediction_rows": len(prediction_rows),
        "strict_unique_keys": len(split_keys),
        "prediction_unique_keys": len(prediction_keys),
        "strict_unique_source_sha256": len(split_sha_set),
        "prediction_unique_source_sha256": len(prediction_sha_set),
        "duplicate_strict_source_sha256_rows": duplicate_split_shas,
        "duplicate_prediction_source_sha256_rows": duplicate_prediction_shas,
        "missing_key_count": len(missing_keys),
        "extra_key_count": len(extra_keys),
        "missing_source_sha256_count": len(missing_shas),
        "extra_source_sha256_count": len(extra_shas),
        "reusable_by_unique_source_sha256_label": reusable_by_sha,
        "ambiguous_by_source_sha256": ambiguous_by_sha,
        "label_mismatch_by_source_sha256": label_mismatch_by_sha,
        "issue_counts": dict(sorted(issue_counts.items())),
        "examples": {
            "missing_keys": sorted(missing_keys)[:10],
            "extra_keys": sorted(extra_keys)[:10],
            "missing_source_sha256": sorted(missing_shas)[:10],
            "extra_source_sha256": sorted(extra_shas)[:10],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit prediction CSV reuse against a strict split.")
    parser.add_argument("--strict-split-csv", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_reuse(
        strict_split_csv=args.strict_split_csv,
        predictions_csv=args.predictions_csv,
        split=args.split,
    )
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ["decision", "strict_rows", "prediction_rows", "missing_key_count", "extra_key_count", "reusable_by_unique_source_sha256_label"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
