#!/usr/bin/env python3
"""Summarize GA feature-mask holdout checks from error-analysis artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(summary: dict) -> dict:
    return {
        "predictions": summary.get("predictions"),
        "threshold": summary.get("threshold"),
        "total_predictions": summary.get("total_predictions"),
        "false_positive": summary.get("false_positive_count"),
        "false_negative": summary.get("false_negative_count"),
        "errors": summary.get("error_count"),
    }


def _delta(mask: dict, full: dict) -> dict:
    return {
        "false_positive": int(mask["false_positive"]) - int(full["false_positive"]),
        "false_negative": int(mask["false_negative"]) - int(full["false_negative"]),
        "errors": int(mask["errors"]) - int(full["errors"]),
    }


def build_summary(
    *,
    feature_mask: Path,
    hard_fn_full: Path,
    hard_fn_mask: Path,
    hard_error_full: Path,
    hard_error_mask: Path,
) -> dict:
    sections = {}
    for name, full_path, mask_path in [
        ("hard_fn_current_subset", hard_fn_full, hard_fn_mask),
        ("hard_error_current_subset", hard_error_full, hard_error_mask),
    ]:
        full = _row(load_json(full_path))
        mask = _row(load_json(mask_path))
        if full["total_predictions"] != mask["total_predictions"]:
            raise ValueError(
                f"{name} full/mask sample counts differ: "
                f"{full['total_predictions']} != {mask['total_predictions']}"
            )
        sections[name] = {
            "full": full,
            "mask": mask,
            "delta_mask_minus_full": _delta(mask, full),
        }

    return {
        "schema": "axon_feature_mask_holdout_summary_v1",
        "feature_mask": str(feature_mask),
        "comparison": "full baseline threshold 0.50 vs GA mask threshold 0.525",
        "sections": sections,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a GA feature-mask holdout summary JSON.")
    parser.add_argument("--feature-mask", type=Path, required=True)
    parser.add_argument("--hard-fn-full", type=Path, required=True)
    parser.add_argument("--hard-fn-mask", type=Path, required=True)
    parser.add_argument("--hard-error-full", type=Path, required=True)
    parser.add_argument("--hard-error-mask", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_summary(
        feature_mask=args.feature_mask,
        hard_fn_full=args.hard_fn_full,
        hard_fn_mask=args.hard_fn_mask,
        hard_error_full=args.hard_error_full,
        hard_error_mask=args.hard_error_mask,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
