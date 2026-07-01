#!/usr/bin/env python3
"""Summarize Loop28 residual strata from existing audit CSVs."""

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


def read_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def identity_key(row: dict) -> str:
    sha = str(row.get("source_sha256") or "").casefold()
    path = str(row.get("source_path") or "").casefold()
    return f"{sha}\x1f{path}"


def loop28_error_type(row: dict) -> str:
    return str(row.get("loop28_error_type") or "")


def is_loop28_error(row: dict) -> bool:
    return bool(loop28_error_type(row))


def corrected_by(row: dict, model_name: str) -> bool:
    return is_loop28_error(row) and not str(row.get(f"{model_name}_error_type") or "")


def count_by(rows: Sequence[dict], column: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(column) or "") for row in rows).items()))


def summarize(
    *,
    overlap_csv: Path,
    noise_csv: Path,
    output_json: Path,
    output_csv: Path,
) -> dict:
    overlap_rows = read_rows(overlap_csv)
    noise_rows = read_rows(noise_csv)
    noise_by_key = {identity_key(row): row for row in noise_rows}

    loop28_errors = [row for row in overlap_rows if is_loop28_error(row)]
    for row in loop28_errors:
        noise_row = noise_by_key.get(identity_key(row), {})
        row["noise_bucket"] = noise_row.get("noise_bucket", "not_suspected")
        row["loop28_noise_bucket"] = row["noise_bucket"]
        row["loop28_error_type"] = loop28_error_type(row)
        row["corrected_by_loop37"] = corrected_by(row, "loop37")
        row["corrected_by_byte_ngram"] = corrected_by(row, "byte_ngram")
        row["corrected_by_loop26_blend"] = corrected_by(row, "loop26_blend")
        row["corrected_by_any_compared_model"] = any(
            row[f"corrected_by_{name}"] for name in ["loop37", "byte_ngram", "loop26_blend"]
        )

    corrected_any = [row for row in loop28_errors if row["corrected_by_any_compared_model"]]
    not_corrected = [row for row in loop28_errors if not row["corrected_by_any_compared_model"]]
    suspected_noise = [row for row in loop28_errors if row["noise_bucket"] != "not_suspected"]
    severe_conflict = [
        row
        for row in loop28_errors
        if row["noise_bucket"].startswith("severe_") or row["noise_bucket"].startswith("high_")
    ]
    near_threshold = [row for row in loop28_errors if row["noise_bucket"].startswith("near_threshold")]

    fieldnames = [
        "source_path",
        "source_sha256",
        "sample_index",
        "label",
        "loop28_error_type",
        "loop28_score",
        "loop37_score",
        "byte_ngram_score",
        "loop26_blend_score",
        "noise_bucket",
        "corrected_by_loop37",
        "corrected_by_byte_ngram",
        "corrected_by_loop26_blend",
        "corrected_by_any_compared_model",
    ]
    resolved_csv = resolve_path(output_csv)
    resolved_csv.parent.mkdir(parents=True, exist_ok=True)
    with resolved_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(loop28_errors)

    summary = {
        "schema": "axon_loop28_residual_strata_v1",
        "overlap_csv": str(resolve_path(overlap_csv)),
        "noise_csv": str(resolve_path(noise_csv)),
        "loop28_errors": len(loop28_errors),
        "loop28_error_type_counts": count_by(loop28_errors, "loop28_error_type"),
        "noise_bucket_counts_on_loop28_errors": count_by(loop28_errors, "noise_bucket"),
        "suspected_noise_or_hard_count": len(suspected_noise),
        "severe_or_high_conflict_count": len(severe_conflict),
        "near_threshold_count": len(near_threshold),
        "corrected_by": {
            "loop37": sum(1 for row in loop28_errors if row["corrected_by_loop37"]),
            "byte_ngram": sum(1 for row in loop28_errors if row["corrected_by_byte_ngram"]),
            "loop26_blend": sum(1 for row in loop28_errors if row["corrected_by_loop26_blend"]),
            "any_compared_model": len(corrected_any),
        },
        "corrected_by_any_error_type_counts": count_by(corrected_any, "loop28_error_type"),
        "not_corrected_by_any_error_type_counts": count_by(not_corrected, "loop28_error_type"),
        "corrected_by_any_noise_bucket_counts": count_by(corrected_any, "noise_bucket"),
        "not_corrected_by_any_noise_bucket_counts": count_by(not_corrected, "noise_bucket"),
        "outputs": {
            "detail_csv": str(resolved_csv),
            "summary_json": str(resolve_path(output_json)),
        },
    }
    resolved_json = resolve_path(output_json)
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Loop28 residual strata.")
    parser.add_argument("--overlap-csv", type=Path, required=True)
    parser.add_argument("--noise-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = summarize(
        overlap_csv=args.overlap_csv,
        noise_csv=args.noise_csv,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
