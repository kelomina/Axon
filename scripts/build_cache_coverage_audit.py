#!/usr/bin/env python3
"""Build a cache-coverage audit for current model-review artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ratio(kept: int, total: int) -> float:
    return float(kept / total) if total else 0.0


def _from_export_summary(name: str, path: Path, blocked_items: list[str]) -> dict:
    payload = load_json(path)
    total = int(payload["raw_samples"])
    kept = int(payload["predicted_samples"])
    missing = int(payload["missing_cache_samples"])
    return {
        "name": name,
        "source": str(path),
        "total": total,
        "covered": kept,
        "missing": missing,
        "coverage_ratio": _ratio(kept, total),
        "missing_output": payload.get("missing_cache_output"),
        "blocked_recommendations": blocked_items,
    }


def _from_calibrator_eval(name: str, path: Path, blocked_items: list[str]) -> dict:
    payload = load_json(path)
    rows = payload.get("rows") or {}
    total = int(rows["total"])
    kept = int(rows["kept"])
    missing = int(rows.get("skipped_missing_cache", 0))
    return {
        "name": name,
        "source": str(path),
        "total": total,
        "covered": kept,
        "missing": missing,
        "coverage_ratio": _ratio(kept, total),
        "missing_examples": rows.get("missing_cache_examples", []),
        "missing_output": rows.get("missing_cache_output"),
        "blocked_recommendations": blocked_items,
    }


def build_audit(
    *,
    test_current_subset: Path,
    hard_fn_summary: Path,
    hard_error_summary: Path,
    high_value_benign_calibrator_eval: Optional[Path] = None,
    high_value_benign_full_summary: Optional[Path] = None,
    high_value_benign_mask_summary: Optional[Path] = None,
    ga_full_hard_fn_summary: Optional[Path] = None,
    ga_mask_hard_fn_summary: Optional[Path] = None,
    ga_full_hard_error_summary: Optional[Path] = None,
    ga_mask_hard_error_summary: Optional[Path] = None,
) -> dict:
    checks = [
        _from_calibrator_eval(
            "official_test_current_cache_subset",
            test_current_subset,
            ["probability_calibration"],
        ),
        _from_export_summary(
            "hard_fn_holdout_current_cache_subset",
            hard_fn_summary,
            ["probability_calibration", "ga_feature_mask"],
        ),
        _from_export_summary(
            "hard_error_holdout_current_cache_subset",
            hard_error_summary,
            ["probability_calibration", "ga_feature_mask"],
        ),
    ]
    if high_value_benign_calibrator_eval is not None:
        checks.append(
            _from_calibrator_eval(
                "high_value_benign_probability_calibrator_strict_full",
                high_value_benign_calibrator_eval,
                ["probability_calibration"],
            )
        )
    if high_value_benign_full_summary is not None:
        checks.append(
            _from_export_summary(
                "high_value_benign_full_feature_cache_subset",
                high_value_benign_full_summary,
                ["ga_feature_mask"],
            )
        )
    if high_value_benign_mask_summary is not None:
        checks.append(
            _from_export_summary(
                "high_value_benign_ga_mask_cache_subset",
                high_value_benign_mask_summary,
                ["ga_feature_mask"],
            )
        )
    for name, path in [
        ("ga_full_hard_fn_holdout_cache_subset", ga_full_hard_fn_summary),
        ("ga_mask_hard_fn_holdout_cache_subset", ga_mask_hard_fn_summary),
        ("ga_full_hard_error_holdout_cache_subset", ga_full_hard_error_summary),
        ("ga_mask_hard_error_holdout_cache_subset", ga_mask_hard_error_summary),
    ]:
        if path is not None:
            checks.append(_from_export_summary(name, path, ["ga_feature_mask"]))
    blocked = sorted({item for check in checks for item in check["blocked_recommendations"] if check["missing"] > 0})
    return {
        "schema": "axon_cache_coverage_audit_v1",
        "checks": checks,
        "blocked_recommendations": blocked,
        "all_full_coverage": all(check["missing"] == 0 for check in checks),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cache coverage audit JSON.")
    parser.add_argument("--test-current-subset", type=Path, required=True)
    parser.add_argument("--hard-fn-summary", type=Path, required=True)
    parser.add_argument("--hard-error-summary", type=Path, required=True)
    parser.add_argument("--high-value-benign-calibrator-eval", type=Path, default=None)
    parser.add_argument("--high-value-benign-full-summary", type=Path, default=None)
    parser.add_argument("--high-value-benign-mask-summary", type=Path, default=None)
    parser.add_argument("--ga-full-hard-fn-summary", type=Path, default=None)
    parser.add_argument("--ga-mask-hard-fn-summary", type=Path, default=None)
    parser.add_argument("--ga-full-hard-error-summary", type=Path, default=None)
    parser.add_argument("--ga-mask-hard-error-summary", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    audit = build_audit(
        test_current_subset=args.test_current_subset,
        hard_fn_summary=args.hard_fn_summary,
        hard_error_summary=args.hard_error_summary,
        high_value_benign_calibrator_eval=args.high_value_benign_calibrator_eval,
        high_value_benign_full_summary=args.high_value_benign_full_summary,
        high_value_benign_mask_summary=args.high_value_benign_mask_summary,
        ga_full_hard_fn_summary=args.ga_full_hard_fn_summary,
        ga_mask_hard_fn_summary=args.ga_mask_hard_fn_summary,
        ga_full_hard_error_summary=args.ga_full_hard_error_summary,
        ga_mask_hard_error_summary=args.ga_mask_hard_error_summary,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
