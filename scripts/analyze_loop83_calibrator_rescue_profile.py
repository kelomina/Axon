#!/usr/bin/env python3
"""Profile Loop82 calibrator rescue/regression rows on Val only.

This is a read-only diagnostic. Identity fields are kept only as row audit
metadata and are not used to derive thresholds or model evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


GROUPS = (
    "both_correct",
    "both_wrong",
    "loop57_only_correct",
    "calibrator_only_correct",
)

FORBIDDEN_IDENTITY_EVIDENCE = [
    "filename",
    "path",
    "extension",
    "directory",
    "hash",
    "source_sha256",
    "sample_index",
    "split",
    "row_order",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_int(value: Any) -> int:
    return int(float(str(value)))


def to_float(value: Any) -> float:
    return float(str(value))


def safe_logit(value: float) -> float:
    clipped = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return float(np.log(clipped / (1.0 - clipped)))


def metrics_from_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "samples": int(labels.shape[0]),
        "accuracy": float((tp + tn) / max(labels.shape[0], 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "errors": int(fp + fn),
    }


def describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def enrich_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        loop57_score = to_float(row["loop57_score"])
        calibrator_score = to_float(row["calibrator_score"])
        label = to_int(row["label"])
        item = dict(row)
        item["label"] = label
        item["loop57_score"] = loop57_score
        item["calibrator_score"] = calibrator_score
        item["loop57_prediction"] = to_int(row["loop57_prediction"])
        item["calibrator_prediction"] = to_int(row["calibrator_prediction"])
        item["loop57_correct"] = str(row["loop57_correct"]).casefold() == "true"
        item["calibrator_correct"] = str(row["calibrator_correct"]).casefold() == "true"
        item["calibrator_minus_loop57"] = calibrator_score - loop57_score
        item["abs_score_delta"] = abs(calibrator_score - loop57_score)
        item["loop57_confidence"] = abs(loop57_score - 0.5) * 2.0
        item["calibrator_confidence"] = abs(calibrator_score - 0.5) * 2.0
        item["calibrator_logit_minus_loop57"] = safe_logit(calibrator_score) - safe_logit(loop57_score)
        enriched.append(item)
    return enriched


def build_group_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for group in GROUPS:
        group_rows = [row for row in rows if row["overlap_group"] == group]
        summary[group] = {
            "rows": len(group_rows),
            "labels": dict(Counter(str(row["label"]) for row in group_rows)),
            "loop57_score": describe([row["loop57_score"] for row in group_rows]),
            "calibrator_score": describe([row["calibrator_score"] for row in group_rows]),
            "calibrator_minus_loop57": describe([row["calibrator_minus_loop57"] for row in group_rows]),
            "abs_score_delta": describe([row["abs_score_delta"] for row in group_rows]),
            "loop57_confidence": describe([row["loop57_confidence"] for row in group_rows]),
            "calibrator_confidence": describe([row["calibrator_confidence"] for row in group_rows]),
            "calibrator_logit_minus_loop57": describe(
                [row["calibrator_logit_minus_loop57"] for row in group_rows]
            ),
        }
    return summary


def apply_candidate_rule(rows: Sequence[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    loop57_pred = np.asarray([row["loop57_prediction"] for row in rows], dtype=np.int64)
    cal_pred = np.asarray([row["calibrator_prediction"] for row in rows], dtype=np.int64)
    use_calibrator = np.asarray([row["abs_score_delta"] >= threshold for row in rows], dtype=bool)
    final_pred = np.where(use_calibrator, cal_pred, loop57_pred).astype(np.int64)
    metrics = metrics_from_predictions(labels, final_pred)
    metrics.update(
        {
            "rule": "use_calibrator_when_abs_score_delta_ge",
            "threshold": float(threshold),
            "use_calibrator_count": int(use_calibrator.sum()),
            "use_calibrator_label0": int(((labels == 0) & use_calibrator).sum()),
            "use_calibrator_label1": int(((labels == 1) & use_calibrator).sum()),
            "calibrator_only_correct_captured": int(
                sum(row["overlap_group"] == "calibrator_only_correct" and row["abs_score_delta"] >= threshold for row in rows)
            ),
            "loop57_only_correct_harmed": int(
                sum(row["overlap_group"] == "loop57_only_correct" and row["abs_score_delta"] >= threshold for row in rows)
            ),
        }
    )
    return metrics


def scan_candidate_rules(rows: Sequence[dict[str, Any]], thresholds: Sequence[float]) -> list[dict[str, Any]]:
    results = [apply_candidate_rule(rows, threshold=float(threshold)) for threshold in thresholds]
    results.sort(
        key=lambda row: (
            row["f1"],
            -row["errors"],
            -row["loop57_only_correct_harmed"],
            row["calibrator_only_correct_captured"],
            -row["use_calibrator_count"],
            row["threshold"],
        ),
        reverse=True,
    )
    return results


def build_summary(
    *,
    overlap_csv: Path,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    rows = enrich_rows(read_rows(overlap_csv))
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    loop57_pred = np.asarray([row["loop57_prediction"] for row in rows], dtype=np.int64)
    cal_pred = np.asarray([row["calibrator_prediction"] for row in rows], dtype=np.int64)
    group_counts = dict(Counter(row["overlap_group"] for row in rows))
    blocker = []
    if len(rows) != 20000:
        blocker.append("Expected complete 20000-row Val overlap")
    for group in GROUPS:
        group_counts.setdefault(group, 0)

    rules = scan_candidate_rules(rows, thresholds)
    best_rule = rules[0] if rules else None
    loop57_metrics = metrics_from_predictions(labels, loop57_pred)
    calibrator_metrics = metrics_from_predictions(labels, cal_pred)
    best_rule_improves_loop57 = bool(best_rule and best_rule["errors"] < loop57_metrics["errors"])
    return {
        "schema": "axon_loop83_calibrator_rescue_profile_v1",
        "protocol": "Val-only diagnostic; no training, no Test/Test-10k access, no identity evidence",
        "overlap_csv": str(overlap_csv),
        "rows": len(rows),
        "blockers": blocker,
        "group_counts": group_counts,
        "metrics": {
            "loop57": loop57_metrics,
            "calibrator": calibrator_metrics,
        },
        "group_summary": build_group_summary(rows),
        "rule_scan": {
            "feature": "abs_score_delta",
            "thresholds": [float(threshold) for threshold in thresholds],
            "best": best_rule,
            "top5": rules[:5],
            "improves_loop57": best_rule_improves_loop57,
        },
        "identity_feature_policy": {
            "forbidden_as_model_evidence": FORBIDDEN_IDENTITY_EVIDENCE,
            "allowed_identity_use": "row audit only; not used in rule features",
            "rule_features_used": ["abs_score_delta"],
        },
        "next_step": (
            "Only attempt a learned Val-only fusion probe if non-identity score/content "
            "features show a way to capture calibrator-only-correct rows without harming "
            "many Loop57-only-correct rows."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Loop82 calibrator rescue rows.")
    parser.add_argument("--overlap-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
        help="Comma-separated abs_score_delta thresholds for a diagnostic rule scan.",
    )
    return parser


def parse_thresholds(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_summary(overlap_csv=args.overlap_csv, thresholds=parse_thresholds(args.thresholds))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
