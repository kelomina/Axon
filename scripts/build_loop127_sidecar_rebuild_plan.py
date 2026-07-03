#!/usr/bin/env python3
"""Build a read-only Loop127 sidecar rebuild plan from readiness output."""

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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _index_rows(rows: Sequence[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    indexed = {}
    for row in rows:
        key = (
            str(row.get("split") or "").strip(),
            str(row.get("sample_index") or "").strip(),
        )
        if key not in indexed:
            indexed[key] = row
    return indexed


def _required_examples(readiness: dict) -> list[dict[str, str]]:
    rows = []
    for split in ["train", "val"]:
        for sidecar, examples in readiness.get(split, {}).get("missing_examples", {}).items():
            if sidecar == "cache_path":
                continue
            for example in examples:
                rows.append(
                    {
                        "split": str(example.get("split") or ""),
                        "sample_index": str(example.get("sample_index") or ""),
                        "source_sha256": str(example.get("source_sha256") or ""),
                        "label": str(example.get("label") or ""),
                        "sidecar": sidecar,
                        "reason": f"{sidecar}_missing",
                    }
                )
    return rows


def build_loop127_sidecar_rebuild_plan(
    *,
    readiness_json: Path,
    train_predictions: Path,
    val_predictions: Path,
    output_csv: Path,
    output_json: Path,
) -> dict:
    readiness = json.loads(resolve_path(readiness_json).read_text(encoding="utf-8"))
    indexed_rows = _index_rows(_read_rows(train_predictions) + _read_rows(val_predictions))
    plan_rows = []
    missing_prediction_rows = 0
    seen = set()
    for item in _required_examples(readiness):
        key = (item["split"], item["sample_index"], item["sidecar"])
        if key in seen:
            continue
        seen.add(key)
        prediction_row = indexed_rows.get((item["split"], item["sample_index"]))
        if prediction_row is None:
            missing_prediction_rows += 1
            source_path = ""
            cache_path = ""
        else:
            source_path = str(prediction_row.get("source_path") or "")
            cache_path = str(prediction_row.get("cache_path") or "")
        plan_rows.append(
            {
                **item,
                "source_path": source_path,
                "cache_path": cache_path,
                "action": "rebuild_sidecar_from_source_content",
            }
        )

    sidecar_counts = Counter(row["sidecar"] for row in plan_rows)
    split_counts = Counter(row["split"] for row in plan_rows)
    output_csv_path = resolve_path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "sample_index",
        "label",
        "source_sha256",
        "sidecar",
        "reason",
        "action",
        "source_path",
        "cache_path",
    ]
    with output_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(plan_rows)

    payload = {
        "schema": "axon_loop127_sidecar_rebuild_plan_v1",
        "protocol": "read-only plan; source_path is retained only as the file location for content extraction, not as evidence",
        "readiness_json": str(resolve_path(readiness_json)),
        "train_predictions": str(resolve_path(train_predictions)),
        "val_predictions": str(resolve_path(val_predictions)),
        "output_csv": str(output_csv_path),
        "rows": len(plan_rows),
        "sidecar_counts": dict(sorted(sidecar_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "missing_prediction_rows": missing_prediction_rows,
        "ready_to_rebuild": missing_prediction_rows == 0 and len(plan_rows) > 0,
    }
    output_json_path = resolve_path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop127 sidecar rebuild plan from readiness JSON.")
    parser.add_argument("--readiness-json", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_loop127_sidecar_rebuild_plan(
        readiness_json=args.readiness_json,
        train_predictions=args.train_predictions,
        val_predictions=args.val_predictions,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["missing_prediction_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
