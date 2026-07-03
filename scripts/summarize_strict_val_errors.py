#!/usr/bin/env python3
"""Summarize Val errors without path/name/directory grouping."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path) -> list[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def error_type(row: dict[str, str], threshold: float) -> str:
    label = _int(row.get("label"))
    prob = _float(row.get("prob_malicious"))
    pred = 1 if prob >= threshold else 0
    if label == 0 and pred == 1:
        return "FP"
    if label == 1 and pred == 0:
        return "FN"
    return ""


def confidence_bucket(row: dict[str, str], kind: str, threshold: float) -> str:
    prob = _float(row.get("prob_malicious"))
    if kind == "FP":
        if prob >= 0.90:
            return "fp_high_conf_ge_0.90"
        if prob >= 0.75:
            return "fp_mid_conf_0.75_0.90"
        return f"fp_near_threshold_{threshold:.2f}_0.75"
    if prob < 0.10:
        return "fn_high_conf_lt_0.10"
    if prob < 0.30:
        return "fn_mid_conf_0.10_0.30"
    return f"fn_near_threshold_0.30_{threshold:.2f}"


def summarize_errors(predictions_csv: Path, threshold: float) -> dict:
    rows = read_rows(predictions_csv)
    fp_probs: list[float] = []
    fn_probs: list[float] = []
    bucket_counts: Counter = Counter()
    examples = []
    for row in rows:
        kind = error_type(row, threshold)
        if not kind:
            continue
        prob = _float(row.get("prob_malicious"))
        if kind == "FP":
            fp_probs.append(prob)
        else:
            fn_probs.append(prob)
        bucket_counts[confidence_bucket(row, kind, threshold)] += 1
        if len(examples) < 50:
            examples.append(
                {
                    "error_type": kind,
                    "sample_index": row.get("sample_index", ""),
                    "split": row.get("split", ""),
                    "label": row.get("label", ""),
                    "source_sha256": row.get("source_sha256", ""),
                    "prob_malicious": prob,
                    "margin_to_threshold": abs(prob - threshold),
                    "source_path": row.get("source_path", ""),
                }
            )
    total_errors = len(fp_probs) + len(fn_probs)
    return {
        "schema": "axon_strict_val_error_summary_v1",
        "predictions_csv": str(resolve_path(predictions_csv)),
        "threshold": float(threshold),
        "total_predictions": len(rows),
        "error_count": total_errors,
        "false_positive_count": len(fp_probs),
        "false_negative_count": len(fn_probs),
        "fp_prob": {
            "avg": mean(fp_probs) if fp_probs else None,
            "min": min(fp_probs) if fp_probs else None,
            "max": max(fp_probs) if fp_probs else None,
        },
        "fn_prob": {
            "avg": mean(fn_probs) if fn_probs else None,
            "min": min(fn_probs) if fn_probs else None,
            "max": max(fn_probs) if fn_probs else None,
        },
        "confidence_bucket_counts": dict(sorted(bucket_counts.items())),
        "error_examples": examples,
        "identity_feature_policy": (
            "source_path and source_sha256 appear only so humans can locate rows after the summary; "
            "this summary does not group, rank, or decide by file name, path, directory, extension, hash, or sample_index."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize strict Val errors without identity grouping.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = summarize_errors(args.predictions, float(args.threshold))
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ["error_count", "false_positive_count", "false_negative_count", "confidence_bucket_counts"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
