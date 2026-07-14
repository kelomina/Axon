#!/usr/bin/env python3
"""Evaluate a frozen byte n-gram SGD model on cache-backed split rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_byte_ngram_sgd import (  # noqa: E402
    ByteHashConfig,
    load_records,
    metrics_at_threshold,
    predict_scores,
    write_predictions,
)

_ = ByteHashConfig


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a frozen byte n-gram SGD model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = joblib.load(resolve_path(args.model))
    model = payload["model"]
    hash_config = payload["hash_config"]
    threshold = float(payload["threshold"] if args.threshold is None else args.threshold)

    records = load_records(args.split_csv, args.manifest, args.split, args.max_rows)
    labels, scores = predict_scores(model, records, hash_config, max(1, int(args.batch_size)))
    metrics = metrics_at_threshold(labels, scores, threshold)

    output_csv = resolve_path(args.output_csv)
    write_predictions(output_csv, records, labels, scores, threshold)

    report = {
        "schema": "axon_byte_ngram_sgd_frozen_eval_v1",
        "protocol": "frozen byte n-gram SGD model only; no fitting and no threshold sweep",
        "model": str(resolve_path(args.model)),
        "split_csv": str(resolve_path(args.split_csv)),
        "manifest": str(resolve_path(args.manifest)),
        "split": args.split,
        "max_rows": args.max_rows,
        "records": len(records),
        "threshold": threshold,
        "metrics": metrics,
        "output_predictions_csv": str(output_csv),
    }
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
