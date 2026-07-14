#!/usr/bin/env python3
"""Summarize Stage2 error-review queues by path, extension, time, and confidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def path_parts(source_path: str) -> list[str]:
    parts = list(Path(source_path).parts)
    lowered = [part.casefold() for part in parts]
    if "data" in lowered:
        index = lowered.index("data")
        return parts[index + 1 :]
    return parts[-5:]


def ext_name(source_path: str) -> str:
    suffix = Path(source_path).suffix.casefold()
    return suffix if suffix else "<none>"


def month_key(parts: Sequence[str]) -> str:
    for part in parts:
        if len(part) == 7 and part[4] == "-" and part[:4].isdigit() and part[5:].isdigit():
            return part
    return "<none>"


def confidence_bucket(label: int, probability: float) -> str:
    if label == 1:
        if probability <= 0.01:
            return "fn_prob_le_0.01"
        if probability <= 0.05:
            return "fn_prob_le_0.05"
        if probability <= 0.15:
            return "fn_prob_le_0.15"
        if probability <= 0.35:
            return "fn_prob_le_0.35"
        return "fn_near_threshold"
    if probability >= 0.99:
        return "fp_prob_ge_0.99"
    if probability >= 0.95:
        return "fp_prob_ge_0.95"
    if probability >= 0.85:
        return "fp_prob_ge_0.85"
    if probability >= 0.65:
        return "fp_prob_ge_0.65"
    return "fp_near_threshold"


def top_counter(counter: Counter[str], limit: int) -> list[dict]:
    return [{"key": key, "count": int(count)} for key, count in counter.most_common(limit)]


def cross_tab(rows: Sequence[dict], key_name: str, limit: int) -> list[dict]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for row in rows:
        key = str(row[key_name])
        totals[key] += 1
        buckets[key][row["error_type"]] += 1
    output = []
    for key, total in totals.most_common(limit):
        output.append(
            {
                key_name: key,
                "total": int(total),
                "fp": int(buckets[key].get("FP", 0)),
                "fn": int(buckets[key].get("FN", 0)),
            }
        )
    return output


def summarize(input_csv: Path, output_json: Path, output_csv: Path, max_priority: int, top_n: int) -> dict:
    raw_rows = read_rows(input_csv)
    rows = [row for row in raw_rows if int(row.get("priority", 999)) <= int(max_priority)]
    enriched = []
    for row in rows:
        parts = path_parts(row.get("source_path", ""))
        top_dir = parts[0] if len(parts) > 0 else "<none>"
        mid_dir = parts[1] if len(parts) > 1 else "<none>"
        third_dir = parts[2] if len(parts) > 2 else "<none>"
        probability = float(row["stage2_prob_malicious"])
        label = int(row["label"])
        enriched.append(
            {
                **row,
                "top_dir": top_dir,
                "mid_dir": mid_dir,
                "third_dir": third_dir,
                "month": month_key(parts),
                "extension": ext_name(row.get("source_path", "")),
                "confidence_bucket": confidence_bucket(label, probability),
            }
        )

    counters = {
        "error_type": Counter(row["error_type"] for row in enriched),
        "priority": Counter(str(row["priority"]) for row in enriched),
        "reason": Counter(row["reason"] for row in enriched),
        "top_dir": Counter(row["top_dir"] for row in enriched),
        "mid_dir": Counter(row["mid_dir"] for row in enriched),
        "month": Counter(row["month"] for row in enriched),
        "extension": Counter(row["extension"] for row in enriched),
        "confidence_bucket": Counter(row["confidence_bucket"] for row in enriched),
    }
    summary = {
        "schema": "axon_stage2_error_queue_summary_v1",
        "input_csv": str(input_csv),
        "rows_total": len(raw_rows),
        "selected_max_priority": int(max_priority),
        "selected_rows": len(enriched),
        "top_n": int(top_n),
        "counts": {name: top_counter(counter, top_n) for name, counter in counters.items()},
        "top_dir_by_error_type": cross_tab(enriched, "top_dir", top_n),
        "month_by_error_type": cross_tab(enriched, "month", top_n),
        "extension_by_error_type": cross_tab(enriched, "extension", top_n),
        "confidence_by_error_type": cross_tab(enriched, "confidence_bucket", top_n),
        "examples": enriched[: min(30, len(enriched))],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "priority",
        "error_type",
        "reason",
        "top_dir",
        "mid_dir",
        "third_dir",
        "month",
        "extension",
        "confidence_bucket",
        "source_path",
        "source_sha256",
        "label",
        "stage2_prob_malicious",
        "prediction",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(enriched)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a Stage2 error review queue.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-priority", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args(argv)
    summary = summarize(args.input_csv, args.output_json, args.output_csv, args.max_priority, args.top_n)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
