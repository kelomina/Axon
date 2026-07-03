#!/usr/bin/env python3
"""Summarize Loop80 probability-calibrator full-test evidence.

This is a read-only report builder. It does not train, evaluate, open NPZ
files, load pickle models, or inspect raw samples.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence


TARGET_F1 = 0.999


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_subset(metrics: dict[str, Any], *, samples_fallback: Optional[int] = None) -> dict[str, Any]:
    return {
        "samples": metrics.get("samples", samples_fallback),
        "threshold": metrics.get("threshold"),
        "f1": metrics.get("f1"),
        "auc": metrics.get("auc"),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "false_positive": metrics.get("false_positive"),
        "false_negative": metrics.get("false_negative"),
        "errors": metrics.get("errors"),
    }


def _errors_at_target_for_balanced_test(samples: int, target_f1: float) -> int:
    """Best-case FP-only error allowance for a balanced binary test set."""
    positives = samples // 2
    # F1 = 2TP / (2TP + FP + FN). Best case sets FN=0 and TP=positives.
    # Solve FP <= 2TP * (1 / target_f1 - 1).
    return int((2 * positives) * (1.0 / target_f1 - 1.0))


def build_summary(
    *,
    fulltest_eval: Path,
    loop57_eval: Path,
    baseline_eval: Path,
    target_f1: float = TARGET_F1,
) -> dict[str, Any]:
    fulltest = load_json(fulltest_eval)
    loop57 = load_json(loop57_eval)
    baseline = load_json(baseline_eval)

    rows = fulltest.get("rows") or {}
    calibrator_metrics = fulltest.get("calibrator_metrics") or {}
    baseline_metrics = fulltest.get("baseline") or (baseline.get("metrics") or {})
    loop57_metrics = loop57.get("metrics") or {}
    samples = int(rows.get("total") or calibrator_metrics.get("samples") or 0)
    allowed_errors = _errors_at_target_for_balanced_test(samples, target_f1) if samples else 0
    calibrator_errors = int(calibrator_metrics.get("errors") or 0)
    loop57_errors = int(loop57_metrics.get("errors") or 0)

    blockers = []
    if int(rows.get("total") or -1) != 160000:
        blockers.append("full-test rows are not 160000")
    if rows.get("total") != rows.get("kept"):
        blockers.append("not every full-test row was kept")
    if int(rows.get("skipped_missing_cache") or 0) != 0:
        blockers.append("full-test evaluation skipped missing cache rows")
    if float(calibrator_metrics.get("f1") or 0.0) < target_f1:
        blockers.append("calibrator full-test F1 is below target")
    if calibrator_errors > allowed_errors:
        blockers.append("calibrator full-test errors exceed best-case target allowance")
    if calibrator_errors >= loop57_errors:
        blockers.append("calibrator does not beat current Loop57 full-test best")

    return {
        "schema": "axon_loop80_calibrator_fulltest_summary_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": (
            "read-only summary of fixed train-split probability calibrator on "
            "160000-row full test; threshold selected on Val and Test-10k used "
            "only as confirmation"
        ),
        "target_f1": float(target_f1),
        "decision": "target_met" if not blockers else "not_final_candidate",
        "blockers": blockers,
        "sources": {
            "fulltest_eval": str(fulltest_eval),
            "loop57_fulltest_eval": str(loop57_eval),
            "baseline_fulltest_eval": str(baseline_eval),
        },
        "rows": {
            "total": rows.get("total"),
            "kept": rows.get("kept"),
            "skipped_missing_cache": rows.get("skipped_missing_cache"),
        },
        "calibrator": {
            "model": (fulltest.get("calibrator") or {}).get("model"),
            "features": (fulltest.get("calibrator") or {}).get("features"),
            "threshold": (fulltest.get("calibrator") or {}).get("threshold"),
            "metrics": _metric_subset(calibrator_metrics, samples_fallback=samples),
        },
        "baseline_8192": {
            "metrics": _metric_subset(baseline_metrics, samples_fallback=samples),
        },
        "loop57_current_best": {
            "metrics": _metric_subset(loop57_metrics),
        },
        "deltas": {
            "calibrator_vs_8192_baseline": {
                "f1": float(calibrator_metrics.get("f1") or 0.0) - float(baseline_metrics.get("f1") or 0.0),
                "errors": calibrator_errors - int(baseline_metrics.get("errors") or 0),
                "false_positive": int(calibrator_metrics.get("false_positive") or 0)
                - int(baseline_metrics.get("false_positive") or 0),
                "false_negative": int(calibrator_metrics.get("false_negative") or 0)
                - int(baseline_metrics.get("false_negative") or 0),
            },
            "calibrator_vs_loop57": {
                "f1": float(calibrator_metrics.get("f1") or 0.0) - float(loop57_metrics.get("f1") or 0.0),
                "errors": calibrator_errors - loop57_errors,
                "false_positive": int(calibrator_metrics.get("false_positive") or 0)
                - int(loop57_metrics.get("false_positive") or 0),
                "false_negative": int(calibrator_metrics.get("false_negative") or 0)
                - int(loop57_metrics.get("false_negative") or 0),
            },
        },
        "target_gap": {
            "allowed_errors_best_case_at_target": allowed_errors,
            "current_calibrator_errors": calibrator_errors,
            "errors_to_remove_best_case": max(0, calibrator_errors - allowed_errors),
        },
        "identity_feature_policy": {
            "prediction_csv_identity_fields": "source_path/cache_path/sample_index/split are alignment and audit fields only",
            "calibrator_features": "probability+stat_features+pe_features only",
            "forbidden_as_model_evidence": [
                "filename",
                "path",
                "extension",
                "directory",
                "hash",
                "source_sha256",
                "sample_index",
                "split",
                "row_order",
            ],
        },
        "next_step": (
            "Do not promote this calibrator as final. Use its large FN reduction "
            "as evidence for future Val-first fusion with stronger content models, "
            "while preserving Loop57 as current full-test best."
        ),
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return "" if value is None else str(value)


def render_markdown(report: dict[str, Any]) -> str:
    cal = report["calibrator"]["metrics"]
    base = report["baseline_8192"]["metrics"]
    best = report["loop57_current_best"]["metrics"]
    lines = [
        "# Loop80 Calibrator Full-Test Summary",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Generated at: `{report['generated_at']}`",
        f"- Target F1: `{report['target_f1']}`",
        f"- Rows kept: `{report['rows']['kept']}/{report['rows']['total']}`",
        f"- Missing cache skipped: `{report['rows']['skipped_missing_cache']}`",
        "",
        "## Metrics",
        "",
        "| candidate | F1 | AUC | FP | FN | errors |",
        "|---|---:|---:|---:|---:|---:|",
        "| 8192 baseline | {f1} | {auc} | {fp} | {fn} | {err} |".format(
            f1=_fmt(base["f1"]),
            auc=_fmt(base["auc"]),
            fp=_fmt(base["false_positive"]),
            fn=_fmt(base["false_negative"]),
            err=_fmt(base["errors"]),
        ),
        "| Loop80 calibrator | {f1} | {auc} | {fp} | {fn} | {err} |".format(
            f1=_fmt(cal["f1"]),
            auc=_fmt(cal["auc"]),
            fp=_fmt(cal["false_positive"]),
            fn=_fmt(cal["false_negative"]),
            err=_fmt(cal["errors"]),
        ),
        "| Loop57 current best | {f1} | {auc} | {fp} | {fn} | {err} |".format(
            f1=_fmt(best["f1"]),
            auc=_fmt(best["auc"]),
            fp=_fmt(best["false_positive"]),
            fn=_fmt(best["false_negative"]),
            err=_fmt(best["errors"]),
        ),
        "",
        "## Deltas",
        "",
        f"- Calibrator vs 8192 baseline errors: `{report['deltas']['calibrator_vs_8192_baseline']['errors']}`",
        f"- Calibrator vs Loop57 errors: `{report['deltas']['calibrator_vs_loop57']['errors']}`",
        f"- Best-case errors still to remove for F1 target: `{report['target_gap']['errors_to_remove_best_case']}`",
        "",
        "## Blockers",
        "",
    ]
    if report["blockers"]:
        for blocker in report["blockers"]:
            lines.append(f"- {blocker}")
    else:
        lines.append("None.")
    lines.extend(["", report["next_step"], ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Loop80 calibrator full-test summary.")
    parser.add_argument("--fulltest-eval", type=Path, default=Path("reports/random_20w_split/loop80_calibrator_fulltest_eval.json"))
    parser.add_argument("--loop57-eval", type=Path, default=Path("reports/random_20w_split/loop57_fn_overlay_gate_frozen_full_test_eval.json"))
    parser.add_argument("--baseline-eval", type=Path, default=Path("reports/random_20w_split/random_20w_8192_replaced_test_eval.json"))
    parser.add_argument("--target-f1", type=float, default=TARGET_F1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_summary(
        fulltest_eval=args.fulltest_eval,
        loop57_eval=args.loop57_eval,
        baseline_eval=args.baseline_eval,
        target_f1=float(args.target_f1),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["decision"] == "target_met" or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
