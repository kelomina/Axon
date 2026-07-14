"""Combine manual review CSVs into one deduplicated human queue."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


MANUAL_FIELDS = ("manual_label_verdict", "manual_verdict_note", "recommended_action")
BASE_FIELDNAMES = [
    "combined_rank",
    "review_sources",
    "review_source_count",
    "dedup_key",
    "dedup_method",
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _has_manual_verdict(row: dict[str, str]) -> bool:
    return any(str(row.get(field, "")).strip() for field in MANUAL_FIELDS)


def _normalized_path(value: str) -> str:
    return value.strip().replace("\\", "/").casefold()


def _dedup_key(row: dict[str, str]) -> tuple[str, str]:
    source_sha256 = str(row.get("source_sha256", "")).strip().casefold()
    if source_sha256:
        return "source_sha256", source_sha256
    source_path = _normalized_path(str(row.get("source_path", "")))
    if source_path:
        return "source_path", source_path
    return "", ""


def _float_value(row: dict[str, str], column: str) -> float:
    try:
        return float(str(row.get(column, "")).strip())
    except ValueError:
        return 0.0


def _int_value(row: dict[str, str], column: str) -> int:
    try:
        return int(float(str(row.get(column, "")).strip()))
    except ValueError:
        return 999


def _sort_key(row: dict[str, str]) -> tuple[int, int, float, float, str]:
    error_type = str(row.get("error_type", "")).strip().upper()
    probability = _float_value(row, "prob_malicious")
    severity = probability if error_type == "FP" else 1.0 - probability
    support_priority = {
        "neighbors_support_model_prediction": 0,
        "neighbors_mixed": 1,
        "neighbors_support_dataset_label": 2,
    }.get(str(row.get("support_bucket", "")).strip(), 3)
    return (
        _int_value(row, "priority"),
        support_priority,
        -severity,
        -_float_value(row, "nearest_similarity"),
        str(row.get("source_sha256") or row.get("source_path") or ""),
    )


def combine_review_queues(
    inputs: list[tuple[str, Path]],
    output_csv: Path,
    output_json: Path,
    allow_filled_manual_fields: bool = False,
) -> dict[str, object]:
    combined: dict[tuple[str, str], dict[str, str]] = {}
    sources_by_key: dict[tuple[str, str], list[str]] = {}
    input_rows: dict[str, int] = {}
    blank_manual_rows = 0
    filled_manual_rows = 0

    for name, path in inputs:
        rows = _read_csv(path)
        input_rows[name] = len(rows)
        for row in rows:
            if _has_manual_verdict(row):
                filled_manual_rows += 1
                if not allow_filled_manual_fields:
                    raise ValueError(
                        f"{path} contains filled manual fields; combine before adjudication "
                        "or pass --allow-filled-manual-fields explicitly."
                    )
            else:
                blank_manual_rows += 1

            key = _dedup_key(row)
            if not key[0] or not key[1]:
                raise ValueError(f"{path} has a row without source_sha256 or source_path.")

            sources_by_key.setdefault(key, [])
            if name not in sources_by_key[key]:
                sources_by_key[key].append(name)

            if key not in combined or _sort_key(row) < _sort_key(combined[key]):
                combined[key] = dict(row)

    output_rows = []
    for row in sorted(combined.values(), key=_sort_key):
        key = _dedup_key(row)
        item = {field: row.get(field, "") for field in BASE_FIELDNAMES if field != "combined_rank"}
        item["review_sources"] = "|".join(sources_by_key[key])
        item["review_source_count"] = str(len(sources_by_key[key]))
        item["dedup_key"] = key[1]
        item["dedup_method"] = key[0]
        for field in MANUAL_FIELDS:
            item[field] = ""
        output_rows.append(item)

    for rank, row in enumerate(output_rows, start=1):
        row["combined_rank"] = str(rank)

    fieldnames = list(BASE_FIELDNAMES)
    _write_csv(output_csv, output_rows, fieldnames)

    error_counts = Counter(row.get("error_type", "") for row in output_rows)
    support_counts = Counter(row.get("support_bucket", "") for row in output_rows)
    priority_counts = Counter(row.get("priority", "") for row in output_rows)
    label_counts = Counter(row.get("label", "") for row in output_rows)
    review_source_count_counts = Counter(row.get("review_source_count", "") for row in output_rows)

    summary: dict[str, object] = {
        "schema": "axon_combined_manual_review_queue_v1",
        "inputs": [{"name": name, "path": str(path), "rows": input_rows[name]} for name, path in inputs],
        "dedup_policy": "source_sha256_then_normalized_source_path",
        "input_rows_total": sum(input_rows.values()),
        "output_rows": len(output_rows),
        "deduplicated_rows": sum(input_rows.values()) - len(output_rows),
        "blank_manual_rows_input": blank_manual_rows,
        "filled_manual_rows_input": filled_manual_rows,
        "manual_fields_blank_output": all(not _has_manual_verdict(row) for row in output_rows),
        "error_type_counts": dict(sorted(error_counts.items())),
        "support_bucket_counts": dict(sorted(support_counts.items())),
        "priority_counts": dict(sorted(priority_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "review_source_count_counts": dict(sorted(review_source_count_counts.items())),
        "output_csv": str(output_csv),
        "output_json": str(output_json),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("Input name cannot be blank.")
    return name.strip(), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine manual review CSVs into one deduplicated queue.")
    parser.add_argument("--input", action="append", required=True, type=_parse_input, help="name=path")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--allow-filled-manual-fields", action="store_true")
    args = parser.parse_args()

    summary = combine_review_queues(
        inputs=args.input,
        output_csv=args.output_csv,
        output_json=args.output_json,
        allow_filled_manual_fields=args.allow_filled_manual_fields,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
