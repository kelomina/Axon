#!/usr/bin/env python3
"""Build a manual review package from Stage2 neighbor-audit rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _probability(row: dict) -> float:
    return _float(row, "prob_malicious", _float(row, "stage2_prob_malicious"))


def _score_column(row: dict) -> str:
    if row.get("score_column"):
        return row["score_column"]
    if row.get("stage2_prob_malicious"):
        return "stage2_prob_malicious"
    if row.get("prob_malicious"):
        return "prob_malicious"
    return ""


def _sort_key(row: dict) -> tuple:
    error_type = row.get("error_type", "")
    prob = _probability(row)
    confidence = 1.0 - prob if error_type == "FN" else prob
    return (
        _int(row, "priority", 999),
        -_float(row, "opposite_label_ratio"),
        -_float(row, "nearest_similarity"),
        -confidence,
        row.get("source_path", ""),
    )


def _select(rows: Sequence[dict], error_type: str, count: int) -> list[dict]:
    selected = [row for row in rows if row.get("error_type") == error_type]
    selected.sort(key=_sort_key)
    return selected[: max(0, int(count))]


def _path_hint(source_path: str) -> str:
    parts = Path(source_path).parts
    lowered = [part.casefold() for part in parts]
    if "data" in lowered:
        parts = parts[lowered.index("data") + 1 :]
    return "/".join(parts[:4])


def _dedupe_key(row: dict) -> str:
    return str(row.get("source_sha256") or row.get("source_path") or "").casefold()


def _dedupe_rows(rows: Sequence[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for row in rows:
        key = _dedupe_key(row)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(row)
    return deduped


def build_package(
    neighbor_csv: Path,
    output_csv: Path,
    output_json: Path,
    fp_count: int,
    fn_count: int,
    max_priority: int,
    support_bucket: str,
) -> dict:
    rows = [
        row for row in read_rows(neighbor_csv)
        if row.get("support_bucket") == support_bucket and _int(row, "priority", 999) <= int(max_priority)
    ]
    selected = _select(_dedupe_rows(rows), "FP", fp_count) + _select(_dedupe_rows(rows), "FN", fn_count)
    for rank, row in enumerate(selected, start=1):
        row["review_rank"] = rank
        row["path_hint"] = _path_hint(row.get("source_path", ""))
        row["prob_malicious"] = f"{_probability(row):.10f}"
        row["stage2_prob_malicious"] = row.get("stage2_prob_malicious") or row["prob_malicious"]
        row["score_column"] = _score_column(row)
        row["manual_label_verdict"] = ""
        row["manual_verdict_note"] = ""
        row["recommended_action"] = ""

    fieldnames = [
        "review_rank",
        "support_bucket",
        "priority",
        "reason",
        "error_type",
        "path_hint",
        "source_path",
        "source_sha256",
        "label",
        "prediction",
        "prob_malicious",
        "score_column",
        "stage2_prob_malicious",
        "base_prob_malicious",
        "top_k",
        "neighbor_label_counts",
        "same_label_count",
        "opposite_label_count",
        "opposite_label_ratio",
        "nearest_similarity",
        "top5_neighbor_labels",
        "top5_neighbor_similarities",
        "top5_neighbor_sha256",
        "top5_neighbor_paths",
        "manual_label_verdict",
        "manual_verdict_note",
        "recommended_action",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "schema": "axon_stage2_manual_review_package_v1",
        "neighbor_csv": str(neighbor_csv),
        "support_bucket": support_bucket,
        "max_priority": int(max_priority),
        "available_rows": len(rows),
        "selected_rows": len(selected),
        "selected_counts": dict(sorted(Counter(row["error_type"] for row in selected).items())),
        "reason_counts": dict(sorted(Counter(row["reason"] for row in selected).items())),
        "output_csv": str(output_csv),
        "examples": selected[:10],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build manual review package from Stage2 neighbor audit.")
    parser.add_argument("--neighbor-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--fp-count", type=int, default=40)
    parser.add_argument("--fn-count", type=int, default=40)
    parser.add_argument("--max-priority", type=int, default=1)
    parser.add_argument("--support-bucket", default="neighbors_support_model_prediction")
    args = parser.parse_args(argv)
    summary = build_package(
        neighbor_csv=args.neighbor_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        fp_count=args.fp_count,
        fn_count=args.fn_count,
        max_priority=args.max_priority,
        support_bucket=args.support_bucket,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
