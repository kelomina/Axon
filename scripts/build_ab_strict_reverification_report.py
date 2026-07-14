#!/usr/bin/env python3
"""Build the A/B strict reverification report from existing evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_subset(row: dict) -> dict:
    return {
        "threshold": row.get("threshold"),
        "f1": row.get("f1"),
        "auc": row.get("auc"),
        "false_positive": row.get("false_positive"),
        "false_negative": row.get("false_negative"),
        "errors": row.get("errors"),
    }


def _calibrator_eval(name: str, path: Path) -> dict:
    payload = load_json(path)
    rows = payload.get("rows") or {}
    baseline = _metric_subset(payload.get("baseline") or {})
    calibrated = _metric_subset(payload.get("calibrator_metrics") or {})
    return {
        "name": name,
        "source": str(path),
        "rows": {
            "total": rows.get("total"),
            "kept": rows.get("kept"),
            "skipped_missing_cache": rows.get("skipped_missing_cache", 0),
        },
        "baseline": baseline,
        "calibrator": calibrated,
        "delta": {
            "f1": (calibrated.get("f1") or 0.0) - (baseline.get("f1") or 0.0),
            "errors": int(calibrated.get("errors") or 0) - int(baseline.get("errors") or 0),
            "false_positive": int(calibrated.get("false_positive") or 0)
            - int(baseline.get("false_positive") or 0),
            "false_negative": int(calibrated.get("false_negative") or 0)
            - int(baseline.get("false_negative") or 0),
        },
    }


def _cache_summary(cache_audit: dict) -> dict:
    checks = cache_audit.get("checks") or []
    return {
        "all_full_coverage": bool(cache_audit.get("all_full_coverage")),
        "blocked_recommendations": list(cache_audit.get("blocked_recommendations") or []),
        "checks": [
            {
                "name": row.get("name"),
                "total": row.get("total"),
                "covered": row.get("covered"),
                "missing": row.get("missing"),
                "coverage_ratio": row.get("coverage_ratio"),
            }
            for row in checks
        ],
    }


def _ga_20k_summary(path: Path) -> dict:
    payload = load_json(path)
    summary = payload.get("summary") or {}
    baseline = (summary.get("baseline_full") or {}).get("metrics") or {}
    best_mask_errors = summary.get("best_mask_errors") or {}
    mask_metrics = best_mask_errors.get("metrics") or {}
    delta = best_mask_errors.get("delta_vs_baseline_full") or {}
    return {
        "source": str(path),
        "sample_count": payload.get("samples"),
        "baseline_full": {
            "threshold": baseline.get("threshold"),
            "f1": baseline.get("f1"),
            "false_positive": baseline.get("false_positive"),
            "false_negative": baseline.get("false_negative"),
            "errors": baseline.get("errors"),
        },
        "mask_lowest_errors": {
            "threshold": best_mask_errors.get("threshold"),
            "f1": mask_metrics.get("f1"),
            "false_positive": mask_metrics.get("false_positive"),
            "false_negative": mask_metrics.get("false_negative"),
            "errors": mask_metrics.get("errors"),
            "delta_false_positive": delta.get("false_positive"),
            "delta_false_negative": delta.get("false_negative"),
            "delta_errors": delta.get("errors"),
        },
    }


def _ga_high_value_summary(baseline_path: Path, mask_path: Path) -> dict:
    baseline = load_json(baseline_path)
    mask = load_json(mask_path)
    return {
        "baseline_source": str(baseline_path),
        "mask_source": str(mask_path),
        "rows": baseline.get("total_predictions"),
        "baseline": {
            "threshold": baseline.get("threshold"),
            "false_positive": baseline.get("false_positive_count"),
            "false_negative": baseline.get("false_negative_count"),
            "errors": baseline.get("error_count"),
        },
        "mask": {
            "threshold": mask.get("threshold"),
            "false_positive": mask.get("false_positive_count"),
            "false_negative": mask.get("false_negative_count"),
            "errors": mask.get("error_count"),
        },
        "delta_mask_minus_baseline": {
            "false_positive": int(mask.get("false_positive_count") or 0)
            - int(baseline.get("false_positive_count") or 0),
            "false_negative": int(mask.get("false_negative_count") or 0)
            - int(baseline.get("false_negative_count") or 0),
            "errors": int(mask.get("error_count") or 0) - int(baseline.get("error_count") or 0),
        },
    }


def build_report(
    *,
    cache_audit_path: Path,
    calibrator_training_path: Path,
    calibrator_test_path: Path,
    calibrator_hard_fn_path: Path,
    calibrator_hard_error_path: Path,
    calibrator_high_value_path: Path,
    feature_mask_20k_path: Path,
    feature_mask_holdout_path: Path,
    high_value_baseline_path: Path,
    high_value_mask_path: Path,
) -> dict:
    training = load_json(calibrator_training_path)
    cache = _cache_summary(load_json(cache_audit_path))
    calibrator = {
        "training_source": str(calibrator_training_path),
        "protocol": training.get("protocol"),
        "train_rows": training.get("train_rows"),
        "val_rows": training.get("val_rows"),
        "selected": training.get("selected"),
        "strict_evaluations": [
            _calibrator_eval("official_test", calibrator_test_path),
            _calibrator_eval("hard_fn_holdout", calibrator_hard_fn_path),
            _calibrator_eval("hard_error_holdout", calibrator_hard_error_path),
            _calibrator_eval("high_value_benign", calibrator_high_value_path),
        ],
    }
    calibrator["all_strict_rows_kept"] = all(
        (row["rows"].get("total") == row["rows"].get("kept"))
        and int(row["rows"].get("skipped_missing_cache") or 0) == 0
        for row in calibrator["strict_evaluations"]
    )
    calibrator["no_test_used_for_training"] = "no test used" in str(training.get("protocol", "")).casefold()

    ga = {
        "feature_mask_20k": _ga_20k_summary(feature_mask_20k_path),
        "hard_holdouts": load_json(feature_mask_holdout_path),
        "high_value_benign": _ga_high_value_summary(high_value_baseline_path, high_value_mask_path),
    }

    return {
        "schema": "axon_ab_strict_reverification_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cache_coverage": cache,
        "probability_calibration": calibrator,
        "ga_feature_mask": ga,
        "conclusion": {
            "probability_calibration": (
                "strictly_reverified_useful"
                if cache["all_full_coverage"]
                and calibrator["all_strict_rows_kept"]
                and calibrator["no_test_used_for_training"]
                else "needs_attention"
            ),
            "ga_feature_mask": "strictly_reverified_high_security_candidate_not_default",
        },
    }


def _fmt(value, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict) -> str:
    lines = [
        "# A/B Strict Reverification Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Cache full coverage: `{report['cache_coverage']['all_full_coverage']}`",
        f"- Probability calibration conclusion: `{report['conclusion']['probability_calibration']}`",
        f"- GA feature-mask conclusion: `{report['conclusion']['ga_feature_mask']}`",
        "",
        "## Cache Coverage",
        "",
        "| check | covered | missing | coverage |",
        "|---|---:|---:|---:|",
    ]
    for row in report["cache_coverage"]["checks"]:
        lines.append(
            "| {name} | {covered}/{total} | {missing} | {ratio} |".format(
                name=row["name"],
                covered=_fmt(row["covered"]),
                total=_fmt(row["total"]),
                missing=_fmt(row["missing"]),
                ratio=_fmt(row["coverage_ratio"]),
            )
        )

    lines.extend(
        [
            "",
            "## Probability Calibration",
            "",
            f"- Training protocol: {report['probability_calibration']['protocol']}",
            f"- No test used for training: `{report['probability_calibration']['no_test_used_for_training']}`",
            f"- All strict rows kept: `{report['probability_calibration']['all_strict_rows_kept']}`",
            "",
            "| slice | rows | baseline F1 | calibrated F1 | baseline FP/FN | calibrated FP/FN | error delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["probability_calibration"]["strict_evaluations"]:
        baseline = row["baseline"]
        calibrated = row["calibrator"]
        lines.append(
            "| {name} | {kept}/{total} | {bf1} | {cf1} | {bfp}/{bfn} | {cfp}/{cfn} | {derr} |".format(
                name=row["name"],
                kept=_fmt(row["rows"]["kept"]),
                total=_fmt(row["rows"]["total"]),
                bf1=_fmt(baseline["f1"]),
                cf1=_fmt(calibrated["f1"]),
                bfp=_fmt(baseline["false_positive"]),
                bfn=_fmt(baseline["false_negative"]),
                cfp=_fmt(calibrated["false_positive"]),
                cfn=_fmt(calibrated["false_negative"]),
                derr=_fmt(row["delta"]["errors"]),
            )
        )

    ga_20k = report["ga_feature_mask"]["feature_mask_20k"]
    ga_hv = report["ga_feature_mask"]["high_value_benign"]
    lines.extend(
        [
            "",
            "## GA Feature Mask",
            "",
            "| check | baseline | mask | delta |",
            "|---|---:|---:|---:|",
            "| 20k errors | {base} | {mask} | {delta} |".format(
                base=_fmt(ga_20k["baseline_full"]["errors"]),
                mask=_fmt(ga_20k["mask_lowest_errors"]["errors"]),
                delta=_fmt(ga_20k["mask_lowest_errors"]["delta_errors"]),
            ),
            "| 20k FP/FN | {bfp}/{bfn} | {mfp}/{mfn} | {dfp}/{dfn} |".format(
                bfp=_fmt(ga_20k["baseline_full"]["false_positive"]),
                bfn=_fmt(ga_20k["baseline_full"]["false_negative"]),
                mfp=_fmt(ga_20k["mask_lowest_errors"]["false_positive"]),
                mfn=_fmt(ga_20k["mask_lowest_errors"]["false_negative"]),
                dfp=_fmt(ga_20k["mask_lowest_errors"]["delta_false_positive"]),
                dfn=_fmt(ga_20k["mask_lowest_errors"]["delta_false_negative"]),
            ),
            "| high-value benign FP | {base} | {mask} | {delta} |".format(
                base=_fmt(ga_hv["baseline"]["false_positive"]),
                mask=_fmt(ga_hv["mask"]["false_positive"]),
                delta=_fmt(ga_hv["delta_mask_minus_baseline"]["false_positive"]),
            ),
            "",
            "### Hard Holdouts",
            "",
            "| slice | full errors | mask errors | delta FP | delta FN | delta errors |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, section in report["ga_feature_mask"]["hard_holdouts"].get("sections", {}).items():
        delta = section["delta_mask_minus_full"]
        lines.append(
            "| {name} | {full} | {mask} | {dfp} | {dfn} | {derr} |".format(
                name=name,
                full=_fmt(section["full"]["errors"]),
                mask=_fmt(section["mask"]["errors"]),
                dfp=_fmt(delta["false_positive"]),
                dfn=_fmt(delta["false_negative"]),
                derr=_fmt(delta["errors"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build A/B strict reverification JSON and Markdown reports.")
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--calibrator-training", type=Path, required=True)
    parser.add_argument("--calibrator-test", type=Path, required=True)
    parser.add_argument("--calibrator-hard-fn", type=Path, required=True)
    parser.add_argument("--calibrator-hard-error", type=Path, required=True)
    parser.add_argument("--calibrator-high-value", type=Path, required=True)
    parser.add_argument("--feature-mask-20k", type=Path, required=True)
    parser.add_argument("--feature-mask-holdout", type=Path, required=True)
    parser.add_argument("--high-value-baseline", type=Path, required=True)
    parser.add_argument("--high-value-mask", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        cache_audit_path=args.cache_audit,
        calibrator_training_path=args.calibrator_training,
        calibrator_test_path=args.calibrator_test,
        calibrator_hard_fn_path=args.calibrator_hard_fn,
        calibrator_hard_error_path=args.calibrator_hard_error,
        calibrator_high_value_path=args.calibrator_high_value,
        feature_mask_20k_path=args.feature_mask_20k,
        feature_mask_holdout_path=args.feature_mask_holdout,
        high_value_baseline_path=args.high_value_baseline,
        high_value_mask_path=args.high_value_mask,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"JSON: {args.output_json}")
    print(f"Markdown: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
