#!/usr/bin/env python3
"""Build a compact evidence summary for training-trick experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


NEGATIVE_BY_CACHE = {"exp2_swa", "exp3_ema", "exp5_all_combined"}
CANDIDATE_BY_GROUP = {"exp1_byte_noise", "exp4_near_threshold"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_rows(results: list[dict]) -> list[dict]:
    baseline = next(row for row in results if row["experiment"] == "exp0_baseline")
    baseline_f1 = float(baseline["f1"])
    rows = []
    for row in results:
        f1 = float(row["f1"])
        delta = f1 - baseline_f1
        if row["experiment"] in NEGATIVE_BY_CACHE:
            decision = "negative_do_not_prioritize"
        elif row["experiment"] in CANDIDATE_BY_GROUP:
            decision = "needs_group_isolated_confirmation"
        else:
            decision = "baseline"
        rows.append(
            {
                "experiment": row["experiment"],
                "f1": f1,
                "delta_vs_baseline_f1": delta,
                "status": row.get("status"),
                "decision": decision,
            }
        )
    return rows


def _group_rows(results_payload: dict) -> list[dict]:
    results = results_payload.get("results", [])
    baseline = next(row for row in results if row["experiment"] == "exp0_baseline")
    baseline_val = float(baseline["val_f1"])
    baseline_test = float(baseline["test_f1"])
    rows = []
    for row in results:
        test = row.get("test") or {}
        val_f1 = float(row["val_f1"])
        test_f1 = float(row["test_f1"])
        delta_test = test_f1 - baseline_test
        delta_val = val_f1 - baseline_val
        if row["experiment"] in CANDIDATE_BY_GROUP and delta_test > 0 and delta_val <= 0:
            decision = "small_test_gain_but_not_confirmed"
        elif row["experiment"] == "exp0_baseline":
            decision = "baseline"
        else:
            decision = "not_improved"
        rows.append(
            {
                "experiment": row["experiment"],
                "val_f1": val_f1,
                "test_f1": test_f1,
                "delta_vs_baseline_val_f1": delta_val,
                "delta_vs_baseline_test_f1": delta_test,
                "test_false_positive": test.get("false_positive"),
                "test_false_negative": test.get("false_negative"),
                "status": row.get("status"),
                "decision": decision,
            }
        )
    return rows


def _multiseed_rows(results_payload: dict) -> list[dict]:
    aggregate = results_payload.get("multiseed_summary", {}).get("aggregate_by_base_experiment", {})
    rows = []
    baseline = aggregate.get("exp0_baseline", {})
    baseline_test = float(baseline.get("test_f1_mean", 0.0))
    for experiment, data in aggregate.items():
        test_f1_mean = float(data.get("test_f1_mean", 0.0))
        delta_test = test_f1_mean - baseline_test if baseline_test else 0.0
        if experiment == "exp0_baseline":
            decision = "baseline"
        elif experiment in CANDIDATE_BY_GROUP and delta_test < 0:
            decision = "negative_do_not_prioritize"
        else:
            decision = "needs_further_confirmation"
        rows.append(
            {
                "experiment": experiment,
                "runs": data.get("runs"),
                "seeds": data.get("seeds"),
                "val_f1_mean": data.get("val_f1_mean"),
                "val_f1_stdev": data.get("val_f1_stdev"),
                "test_f1_mean": data.get("test_f1_mean"),
                "test_f1_stdev": data.get("test_f1_stdev"),
                "delta_vs_baseline_test_f1_mean": data.get("delta_test_f1_mean_vs_baseline", delta_test),
                "test_false_positive_mean": data.get("test_fp_mean"),
                "test_false_negative_mean": data.get("test_fn_mean"),
                "delta_vs_baseline_fp_mean": data.get("delta_test_fp_mean_vs_baseline"),
                "delta_vs_baseline_fn_mean": data.get("delta_test_fn_mean_vs_baseline"),
                "decision": decision,
            }
        )
    return rows


def build_summary(
    cache_results_path: Path,
    group_results_path: Path,
    multiseed_results_path: Path | None = None,
) -> dict:
    cache_results = load_json(cache_results_path)
    group_results = load_json(group_results_path)
    cache_rows = _cache_rows(cache_results)
    group_rows = _group_rows(group_results)
    multiseed_rows = _multiseed_rows(load_json(multiseed_results_path)) if multiseed_results_path else []
    multiseed_negative = sorted(
        row["experiment"] for row in multiseed_rows if row["decision"] == "negative_do_not_prioritize"
    )

    return {
        "schema": "axon_training_trick_summary_v1",
        "inputs": {
            "cache_sample_results": str(cache_results_path),
            "group_isolated_results": str(group_results_path),
            "group_isolated_multiseed_results": str(multiseed_results_path) if multiseed_results_path else None,
        },
        "cache_sample_note": (
            "Older 20k cache-sample comparison; useful for negative screening, "
            "but group-isolated results are stronger evidence for candidates."
        ),
        "cache_sample_rows": cache_rows,
        "group_isolated_rows": group_rows,
        "group_isolated_multiseed_rows": multiseed_rows,
        "decisions": {
            "negative_do_not_prioritize": sorted(
                row["experiment"] for row in cache_rows if row["decision"] == "negative_do_not_prioritize"
            )
            + multiseed_negative,
            "small_gain_needs_multiseed": sorted(
                row["experiment"] for row in group_rows if row["decision"] == "small_test_gain_but_not_confirmed"
            )
            if not multiseed_rows
            else [],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build training-trick evidence summary JSON.")
    parser.add_argument("--cache-results", type=Path, required=True)
    parser.add_argument("--group-results", type=Path, required=True)
    parser.add_argument("--multiseed-results", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_summary(args.cache_results, args.group_results, args.multiseed_results)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
