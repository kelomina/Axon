#!/usr/bin/env python3
"""Summarize manual-review rows by source path and neighbor-conflict evidence."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Callable, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parts(path_text: str) -> list[str]:
    return [part for part in re.split(r"[\\/]+", str(path_text or "")) if part]


def _after_data_parts(path_text: str) -> list[str]:
    parts = _parts(path_text)
    lowered = [part.casefold() for part in parts]
    if "data" in lowered:
        return parts[lowered.index("data") + 1 :]
    return parts


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_match(path_text: str, pattern: str, default: str = "<none>") -> str:
    match = re.search(pattern, str(path_text or ""))
    return match.group(0) if match else default


def data_dir(row: dict) -> str:
    parts = _after_data_parts(row.get("source_path", ""))
    return parts[0] if parts else "<unknown>"


def month(row: dict) -> str:
    return _first_match(row.get("source_path", ""), r"20\d{2}-\d{2}")


def date_dir(row: dict) -> str:
    return _first_match(row.get("source_path", ""), r"20\d{2}-\d{2}-\d{2}")


def parent_dir(row: dict) -> str:
    path = Path(str(row.get("source_path", "")))
    return path.parent.name or "<none>"


def extension(row: dict) -> str:
    return Path(str(row.get("source_path", ""))).suffix.casefold() or "<none>"


def source_prefix(row: dict, depth: int) -> str:
    parts = _after_data_parts(row.get("source_path", ""))
    if not parts:
        return "<unknown>"
    if data_dir(row) in {"待加入白名单", "benign"} and month(row) == "<none>":
        return f"{parts[0]}/<flat>"
    return "/".join(parts[: max(1, min(int(depth), len(parts)))])


def _probability(row: dict) -> float:
    for key in ("prob_malicious", "stage2_prob_malicious", "blend_prob_malicious"):
        value = row.get(key)
        if value not in (None, ""):
            return _safe_float(value)
    return 0.0


def _similarity(row: dict) -> float:
    return _safe_float(row.get("nearest_similarity"))


def _opposite_ratio(row: dict) -> float:
    return _safe_float(row.get("opposite_label_ratio"))


def _compact_counter(counter: Counter[str]) -> str:
    return "|".join(f"{key}:{counter[key]}" for key in sorted(counter))


def _group_summary(dimension: str, value: str, rows: Sequence[dict], example_limit: int) -> dict:
    probs = [_probability(row) for row in rows]
    similarities = [_similarity(row) for row in rows]
    opposite_ratios = [_opposite_ratio(row) for row in rows]
    error_types = Counter(str(row.get("error_type", "")) for row in rows)
    priorities = Counter(str(row.get("priority", "")) for row in rows)
    labels = Counter(str(row.get("label", "")) for row in rows)
    high_similarity = [
        row for row in rows
        if _similarity(row) >= 0.90 and _opposite_ratio(row) >= 0.80
    ]
    critical = [
        row for row in rows
        if _similarity(row) >= 0.95 and _opposite_ratio(row) >= 0.80
    ]
    return {
        "dimension": dimension,
        "value": value,
        "count": len(rows),
        "fp_count": error_types.get("FP", 0),
        "fn_count": error_types.get("FN", 0),
        "priority_counts": _compact_counter(priorities),
        "label_counts": _compact_counter(labels),
        "avg_prob_malicious": mean(probs) if probs else None,
        "min_prob_malicious": min(probs) if probs else None,
        "max_prob_malicious": max(probs) if probs else None,
        "avg_nearest_similarity": mean(similarities) if similarities else None,
        "max_nearest_similarity": max(similarities) if similarities else None,
        "avg_opposite_label_ratio": mean(opposite_ratios) if opposite_ratios else None,
        "max_opposite_label_ratio": max(opposite_ratios) if opposite_ratios else None,
        "high_similarity_conflicts_ge_0.90": len(high_similarity),
        "critical_conflicts_ge_0.95": len(critical),
        "example_sha256": " | ".join(str(row.get("source_sha256", "")) for row in rows[:example_limit]),
        "example_paths": " | ".join(str(row.get("source_path", "")) for row in rows[:example_limit]),
    }


def _dimension_functions(prefix_depth: int) -> dict[str, Callable[[dict], str]]:
    return {
        "source_prefix": lambda row: source_prefix(row, prefix_depth),
        "data_dir": data_dir,
        "month": month,
        "date_dir": date_dir,
        "parent_dir": parent_dir,
        "extension": extension,
        "error_type": lambda row: str(row.get("error_type", "")) or "<none>",
        "priority": lambda row: str(row.get("priority", "")) or "<none>",
        "label": lambda row: str(row.get("label", "")) or "<none>",
    }


def build_source_summary(
    *,
    review_csv: Path,
    output_csv: Path,
    output_json: Path,
    prefix_depth: int = 3,
    example_limit: int = 3,
) -> dict:
    rows = read_rows(review_csv)
    dimensions = _dimension_functions(prefix_depth)
    grouped_rows: list[dict] = []
    for dimension, key_fn in dimensions.items():
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[str(key_fn(row))].append(row)
        for value, items in groups.items():
            grouped_rows.append(_group_summary(dimension, value, items, example_limit))
    grouped_rows.sort(
        key=lambda row: (
            row["dimension"],
            -int(row["count"]),
            -int(row["critical_conflicts_ge_0.95"]),
            -int(row["high_similarity_conflicts_ge_0.90"]),
            row["value"],
        )
    )

    fieldnames = [
        "dimension",
        "value",
        "count",
        "fp_count",
        "fn_count",
        "priority_counts",
        "label_counts",
        "avg_prob_malicious",
        "min_prob_malicious",
        "max_prob_malicious",
        "avg_nearest_similarity",
        "max_nearest_similarity",
        "avg_opposite_label_ratio",
        "max_opposite_label_ratio",
        "high_similarity_conflicts_ge_0.90",
        "critical_conflicts_ge_0.95",
        "example_sha256",
        "example_paths",
    ]
    output_csv = resolve_path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(grouped_rows)

    top_by_dimension = {}
    for dimension in dimensions:
        top_by_dimension[dimension] = [
            row for row in grouped_rows if row["dimension"] == dimension
        ][:10]

    summary = {
        "schema": "axon_manual_review_source_summary_v1",
        "review_csv": str(resolve_path(review_csv)),
        "rows": len(rows),
        "prefix_depth": int(prefix_depth),
        "error_type_counts": dict(sorted(Counter(str(row.get("error_type", "")) for row in rows).items())),
        "priority_counts": dict(sorted(Counter(str(row.get("priority", "")) for row in rows).items())),
        "data_dir_counts": dict(sorted(Counter(data_dir(row) for row in rows).items())),
        "month_counts": dict(sorted(Counter(month(row) for row in rows).items())),
        "source_prefix_counts": dict(sorted(Counter(source_prefix(row, prefix_depth) for row in rows).items())),
        "high_similarity_conflicts_ge_0.90": sum(
            1 for row in rows if _similarity(row) >= 0.90 and _opposite_ratio(row) >= 0.80
        ),
        "critical_conflicts_ge_0.95": sum(
            1 for row in rows if _similarity(row) >= 0.95 and _opposite_ratio(row) >= 0.80
        ),
        "top_by_dimension": top_by_dimension,
        "outputs": {
            "groups_csv": str(output_csv),
            "summary_json": str(resolve_path(output_json)),
        },
    }
    output_json = resolve_path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize manual-review rows by source and neighbor evidence.")
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prefix-depth", type=int, default=3)
    parser.add_argument("--example-limit", type=int, default=3)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_source_summary(
        review_csv=args.review_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        prefix_depth=args.prefix_depth,
        example_limit=args.example_limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
