#!/usr/bin/env python3
"""Build a Loop127 duplicate source_sha256 audit from Train/Val prediction CSVs."""

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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _duplicate_rows(split: str, rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        sha = str(row.get("source_sha256") or "").strip().casefold()
        if sha:
            groups[sha].append(row)
    output = []
    for sha, group in sorted(groups.items()):
        sample_indexes = {str(row.get("sample_index") or "").strip() for row in group}
        if len(sample_indexes) <= 1:
            continue
        labels = {str(row.get("label") or "").strip() for row in group}
        predictions = {str(row.get("prediction") or "").strip() for row in group}
        for row in group:
            output.append(
                {
                    "split": split,
                    "source_sha256": sha,
                    "sample_index": str(row.get("sample_index") or ""),
                    "label": str(row.get("label") or ""),
                    "prob_malicious": str(row.get("prob_malicious") or ""),
                    "prediction": str(row.get("prediction") or ""),
                    "correct": str(row.get("correct") or ""),
                    "group_size": str(len(group)),
                    "label_conflict": str(len(labels) > 1),
                    "prediction_conflict": str(len(predictions) > 1),
                    "recommended_action": "quarantine_duplicate_group_and_redraw_full_replacement_batch",
                }
            )
    return output


def build_loop127_duplicate_hash_audit(
    *,
    train_predictions: Path,
    val_predictions: Path,
    output_csv: Path,
    output_json: Path,
) -> dict:
    rows = _duplicate_rows("train", _read_rows(train_predictions)) + _duplicate_rows("val", _read_rows(val_predictions))
    split_counts = Counter(row["split"] for row in rows)
    group_counts = Counter((row["split"], row["source_sha256"]) for row in rows)
    label_conflict_groups = {
        (row["split"], row["source_sha256"])
        for row in rows
        if row["label_conflict"].casefold() == "true"
    }
    prediction_conflict_groups = {
        (row["split"], row["source_sha256"])
        for row in rows
        if row["prediction_conflict"].casefold() == "true"
    }

    output_csv_path = resolve_path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "source_sha256",
        "sample_index",
        "label",
        "prob_malicious",
        "prediction",
        "correct",
        "group_size",
        "label_conflict",
        "prediction_conflict",
        "recommended_action",
    ]
    with output_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema": "axon_loop127_duplicate_hash_audit_v1",
        "protocol": "source_sha256 is identity-only; duplicates are data-quality blockers, not model evidence",
        "train_predictions": str(resolve_path(train_predictions)),
        "val_predictions": str(resolve_path(val_predictions)),
        "output_csv": str(output_csv_path),
        "duplicate_groups": len(group_counts),
        "duplicate_rows": len(rows),
        "split_row_counts": dict(sorted(split_counts.items())),
        "label_conflict_groups": len(label_conflict_groups),
        "prediction_conflict_groups": len(prediction_conflict_groups),
        "recommended_policy": (
            "Do not fill from duplicate samples. Quarantine duplicate groups and redraw a full replacement batch "
            "under the locked 20k/20k/160k split contract."
        ),
        "ready_without_redraw": len(rows) == 0,
    }
    output_json_path = resolve_path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop127 duplicate source_sha256 audit.")
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_loop127_duplicate_hash_audit(
        train_predictions=args.train_predictions,
        val_predictions=args.val_predictions,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ready_without_redraw"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
