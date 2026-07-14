#!/usr/bin/env python3
"""Build the first manual review package from a prioritized error queue."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional, Sequence


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _prob(row: dict) -> float:
    return float(row.get("prob_malicious") or 0.0)


def _key(row: dict) -> str:
    return str(row.get("source_sha256") or row.get("source_path") or row.get("sample_index")).casefold()


def _add_rows(
    *,
    selected: list[dict],
    seen: set[str],
    rows: Sequence[dict],
    category: str,
    limit: int,
) -> None:
    for row in rows:
        if len([item for item in selected if item["review_category"] == category]) >= limit:
            return
        key = _key(row)
        if key in seen:
            continue
        item = dict(row)
        item["review_category"] = category
        selected.append(item)
        seen.add(key)


def _path_has_2020_08_2022_08(row: dict) -> bool:
    return bool(re.search(r"2020-08[\\/]+2022-08-", row.get("source_path", ""), flags=re.IGNORECASE))


def _is_family_path(row: dict) -> bool:
    text = row.get("source_path", "").casefold()
    return "待拉黑" in text and "黑文件1" in text and "samples" in text


def build_package(queue_csv: Path, threshold: float, output_csv: Path, output_json: Path) -> dict:
    queue = _read_csv(queue_csv)
    selected: list[dict] = []
    seen: set[str] = set()

    severe_fn = sorted(
        [row for row in queue if row.get("reason") == "severe_fn_label1_prob_le_0.01"],
        key=lambda row: (_prob(row), row.get("source_path", "")),
    )
    _add_rows(selected=selected, seen=seen, rows=severe_fn, category="all_severe_fn_prob_le_0.01", limit=8)

    severe_fp_all = [row for row in queue if row.get("reason") == "severe_fp_label0_prob_ge_0.99"]
    severe_fp_exe = sorted(
        [row for row in severe_fp_all if row.get("extension") == ".exe"],
        key=lambda row: (-_prob(row), row.get("source_path", "")),
    )
    _add_rows(selected=selected, seen=seen, rows=severe_fp_exe, category="severe_fp_prob_ge_0.99_top", limit=2)

    severe_fp_remaining = sorted(
        severe_fp_all,
        key=lambda row: (-_prob(row), row.get("extension") != ".exe", row.get("source_path", "")),
    )
    _add_rows(selected=selected, seen=seen, rows=severe_fp_remaining, category="severe_fp_prob_ge_0.99_top", limit=8)

    path_anomaly_fn = sorted(
        [row for row in queue if row.get("error_type") == "FN" and _path_has_2020_08_2022_08(row)],
        key=lambda row: (_prob(row), row.get("source_path", "")),
    )
    _add_rows(selected=selected, seen=seen, rows=path_anomaly_fn, category="path_anomaly_2020_08_contains_2022_08_fn", limit=2)

    family_fn = sorted(
        [row for row in queue if row.get("error_type") == "FN" and _is_family_path(row)],
        key=lambda row: (abs(_prob(row) - threshold), row.get("source_path", "")),
    )
    _add_rows(selected=selected, seen=seen, rows=family_fn, category="family_path_fn_near_threshold", limit=2)

    for index, row in enumerate(selected, start=1):
        row["review_rank"] = index
        row["manual_label_verdict"] = ""
        row["manual_verdict_note"] = ""
        row["recommended_action"] = ""

    fieldnames = [
        "review_rank",
        "review_category",
        "priority",
        "reason",
        "error_type",
        "source_path",
        "cache_path",
        "source_sha256",
        "label",
        "prediction",
        "prob_malicious",
        "distance_to_threshold",
        "split",
        "sample_index",
        "data_dir",
        "extension",
        "month",
        "file_size",
        "manual_label_verdict",
        "manual_verdict_note",
        "recommended_action",
    ]
    _write_csv(output_csv, selected, fieldnames)
    summary = {
        "schema": "axon_top_error_review_package_v1",
        "queue_csv": str(queue_csv),
        "threshold": threshold,
        "selected_count": len(selected),
        "category_counts": {
            category: sum(1 for row in selected if row["review_category"] == category)
            for category in sorted({row["review_category"] for row in selected})
        },
        "rules": [
            "all severe FN with prob <= 0.01, up to 8",
            "severe FP with prob >= 0.99, top 8 total with at least 2 .exe when available",
            "path-anomaly FN under 2020-08/2022-08-*, lowest probability 2",
            "family-path FN under 黑文件1/samples, closest to threshold 2",
        ],
        "rows": selected,
        "outputs": {
            "review_csv": str(output_csv),
            "summary_json": str(output_json),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build Top20 manual review package from a validation error queue.")
    parser.add_argument("--queue-csv", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build_package(args.queue_csv, args.threshold, args.output_csv, args.output_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
