#!/usr/bin/env python3
"""Content-only residual attribution for frozen stage-2 predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from train_stage2_cache_matrix import (  # noqa: E402
    CONTENT_CERT_FEATURE_NAMES,
    CONTENT_PE_FEATURE_NAMES,
    CONTENT_STRING_FEATURE_NAMES,
    content_cert_features_for_row,
    content_pe_features_for_row,
    content_string_features_for_row,
    resolve_path,
)


PROBABILITY_COLUMNS = ("stage2_prob_malicious", "prob_malicious", "base_prob_malicious")
CONFUSION_ORDER = ("TP", "TN", "FP", "FN")
ATTRIBUTION_PAIRS = (("FP", "TN"), ("FN", "TP"))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze stage-2 residuals using content-derived features only.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--content-pe-cache-dir", type=Path, default=None)
    parser.add_argument("--content-string-cache-dir", type=Path, default=None)
    parser.add_argument("--content-cert-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--feature-sets",
        default="content_pe",
        help="Comma-separated content feature sets: content_pe,content_string,content_cert.",
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-slice-support", type=int, default=100)
    parser.add_argument(
        "--diagnostic-path-slices",
        action="store_true",
        help="Write path/extension slices for diagnostics only; never use them as model features.",
    )
    return parser.parse_args(argv)


def read_prediction_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_from_row(row: dict, columns: Sequence[str]) -> float:
    for column in columns:
        value = row.get(column)
        if value not in (None, ""):
            return float(value)
    raise ValueError(f"None of the probability columns were found: {', '.join(columns)}")


def _prediction_from_row(row: dict, score: float, threshold: Optional[float]) -> int:
    value = row.get("prediction")
    if value not in (None, ""):
        return int(value)
    if threshold is None:
        raise ValueError("Prediction column is missing; pass --threshold to recompute predictions from probability.")
    return int(score >= threshold)


def _confusion_name(label: int, prediction: int) -> str:
    if label == 1 and prediction == 1:
        return "TP"
    if label == 0 and prediction == 0:
        return "TN"
    if label == 0 and prediction == 1:
        return "FP"
    if label == 1 and prediction == 0:
        return "FN"
    raise ValueError(f"Unsupported label/prediction pair: label={label} prediction={prediction}")


def normalize_feature_sets(text: str) -> list[str]:
    selected = [item.strip() for item in text.split(",") if item.strip()]
    allowed = {"content_pe", "content_string", "content_cert"}
    unknown = sorted(set(selected) - allowed)
    if unknown:
        raise ValueError(f"Unknown feature set(s): {unknown}. Allowed: {sorted(allowed)}")
    if not selected:
        raise ValueError("At least one content feature set is required")
    return selected


def load_content_feature_matrix(
    rows: Sequence[dict],
    feature_sets: Sequence[str],
    *,
    content_pe_cache_dir: Optional[Path],
    content_string_cache_dir: Optional[Path],
    content_cert_cache_dir: Optional[Path],
) -> tuple[np.ndarray, list[str]]:
    parts: list[np.ndarray] = []
    names: list[str] = []

    if "content_pe" in feature_sets:
        pe_rows = [
            content_pe_features_for_row(
                row,
                str(resolve_path(content_pe_cache_dir)) if content_pe_cache_dir is not None else None,
            )
            for row in rows
        ]
        parts.append(np.vstack(pe_rows).astype(np.float32, copy=False))
        names.extend(CONTENT_PE_FEATURE_NAMES)

    if "content_string" in feature_sets:
        string_rows = [
            content_string_features_for_row(
                row,
                str(resolve_path(content_string_cache_dir)) if content_string_cache_dir is not None else None,
            )
            for row in rows
        ]
        parts.append(np.vstack(string_rows).astype(np.float32, copy=False))
        names.extend(CONTENT_STRING_FEATURE_NAMES)

    if "content_cert" in feature_sets:
        cert_rows = [
            content_cert_features_for_row(
                row,
                str(resolve_path(content_cert_cache_dir)) if content_cert_cache_dir is not None else None,
            )
            for row in rows
        ]
        parts.append(np.vstack(cert_rows).astype(np.float32, copy=False))
        names.extend(CONTENT_CERT_FEATURE_NAMES)

    if not parts:
        raise ValueError("No feature parts were loaded")
    return np.hstack(parts).astype(np.float32, copy=False), names


def _quantiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def summarize_confusion(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, dict]:
    buckets = np.asarray([_confusion_name(int(label), int(pred)) for label, pred in zip(labels, predictions)])
    counts = {name: int(np.count_nonzero(buckets == name)) for name in CONFUSION_ORDER}
    tp = counts["TP"]
    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1.0e-12)
    total = int(labels.shape[0])
    summary = {
        "total": total,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "errors": int(fp + fn),
        "accuracy": float((tp + tn) / max(total, 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "score_quantiles_by_bucket": {
            name: _quantiles(scores[buckets == name]) for name in CONFUSION_ORDER
        },
        "high_confidence_wrong": {
            "fp_score_ge_0_95": int(np.count_nonzero((buckets == "FP") & (scores >= 0.95))),
            "fp_score_ge_0_99": int(np.count_nonzero((buckets == "FP") & (scores >= 0.99))),
            "fn_score_le_0_05": int(np.count_nonzero((buckets == "FN") & (scores <= 0.05))),
            "fn_score_le_0_01": int(np.count_nonzero((buckets == "FN") & (scores <= 0.01))),
        },
    }
    return buckets, summary


def feature_attribution_rows(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    buckets: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    for residual_name, control_name in ATTRIBUTION_PAIRS:
        residual_mask = buckets == residual_name
        control_mask = buckets == control_name
        if not np.any(residual_mask) or not np.any(control_mask):
            continue
        residual = matrix[residual_mask]
        control = matrix[control_mask]
        for index, name in enumerate(feature_names):
            residual_values = residual[:, index]
            control_values = control[:, index]
            residual_mean = float(np.mean(residual_values))
            control_mean = float(np.mean(control_values))
            diff = residual_mean - control_mean
            pooled_std = math.sqrt(
                max(float(np.var(residual_values)) + float(np.var(control_values)), 0.0) / 2.0
            )
            effect = diff / max(pooled_std, 1.0e-9)
            residual_present = float(np.mean(residual_values > 0.0))
            control_present = float(np.mean(control_values > 0.0))
            rows.append(
                {
                    "comparison": f"{residual_name}_vs_{control_name}",
                    "residual_bucket": residual_name,
                    "control_bucket": control_name,
                    "feature": name,
                    "residual_count": int(residual_values.shape[0]),
                    "control_count": int(control_values.shape[0]),
                    "residual_mean": residual_mean,
                    "control_mean": control_mean,
                    "mean_delta": float(diff),
                    "effect_size": float(effect),
                    "abs_effect_size": float(abs(effect)),
                    "residual_present_rate": residual_present,
                    "control_present_rate": control_present,
                    "present_rate_delta": float(residual_present - control_present),
                }
            )
    rows.sort(key=lambda row: (row["comparison"], -row["abs_effect_size"], row["feature"]))
    return rows


def slice_summary_rows(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    labels: np.ndarray,
    buckets: np.ndarray,
    *,
    min_support: int,
) -> list[dict]:
    rows: list[dict] = []
    global_error_rate = float(np.mean((buckets == "FP") | (buckets == "FN")))
    for index, name in enumerate(feature_names):
        values = matrix[:, index]
        candidates: list[tuple[str, np.ndarray]] = [(f"{name} > 0", values > 0.0)]
        q90 = float(np.quantile(values, 0.90))
        q10 = float(np.quantile(values, 0.10))
        if q90 > q10:
            candidates.append((f"{name} >= p90({q90:.6g})", values >= q90))
            candidates.append((f"{name} <= p10({q10:.6g})", values <= q10))
        for slice_name, mask in candidates:
            support = int(np.count_nonzero(mask))
            if support < min_support or support == labels.shape[0]:
                continue
            fp = int(np.count_nonzero(mask & (buckets == "FP")))
            fn = int(np.count_nonzero(mask & (buckets == "FN")))
            tp = int(np.count_nonzero(mask & (buckets == "TP")))
            tn = int(np.count_nonzero(mask & (buckets == "TN")))
            errors = fp + fn
            error_rate = errors / max(support, 1)
            rows.append(
                {
                    "slice": slice_name,
                    "feature": name,
                    "support": support,
                    "support_ratio": float(support / max(labels.shape[0], 1)),
                    "label0": int(np.count_nonzero(mask & (labels == 0))),
                    "label1": int(np.count_nonzero(mask & (labels == 1))),
                    "true_positive": tp,
                    "true_negative": tn,
                    "false_positive": fp,
                    "false_negative": fn,
                    "errors": errors,
                    "error_rate": float(error_rate),
                    "error_rate_lift": float(error_rate / max(global_error_rate, 1.0e-12)),
                }
            )
    rows.sort(key=lambda row: (-row["error_rate_lift"], -row["errors"], row["slice"]))
    return rows


def diagnostic_path_slice_rows(rows: Sequence[dict], buckets: np.ndarray) -> list[dict]:
    grouped: dict[str, dict] = {}
    for row, bucket in zip(rows, buckets):
        suffix = Path(row.get("source_path", "")).suffix.lower() or "<none>"
        entry = grouped.setdefault(
            suffix,
            {"slice": suffix, "support": 0, "true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0},
        )
        entry["support"] += 1
        if bucket == "TP":
            entry["true_positive"] += 1
        elif bucket == "TN":
            entry["true_negative"] += 1
        elif bucket == "FP":
            entry["false_positive"] += 1
        elif bucket == "FN":
            entry["false_negative"] += 1
    output = []
    for entry in grouped.values():
        errors = entry["false_positive"] + entry["false_negative"]
        output.append(
            {
                **entry,
                "errors": errors,
                "error_rate": float(errors / max(entry["support"], 1)),
                "diagnostic_only": True,
            }
        )
    output.sort(key=lambda row: (-row["errors"], row["slice"]))
    return output


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    prediction_path = resolve_path(args.predictions)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_prediction_rows(prediction_path)
    if not rows:
        raise ValueError(f"No rows found in {prediction_path}")

    scores = np.asarray([_float_from_row(row, PROBABILITY_COLUMNS) for row in rows], dtype=np.float32)
    predictions = np.asarray([_prediction_from_row(row, float(score), args.threshold) for row, score in zip(rows, scores)])
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    buckets, confusion_summary = summarize_confusion(labels, predictions, scores)

    feature_sets = normalize_feature_sets(args.feature_sets)
    matrix, feature_names = load_content_feature_matrix(
        rows,
        feature_sets,
        content_pe_cache_dir=args.content_pe_cache_dir,
        content_string_cache_dir=args.content_string_cache_dir,
        content_cert_cache_dir=args.content_cert_cache_dir,
    )

    attribution = feature_attribution_rows(matrix, feature_names, buckets)
    slices = slice_summary_rows(
        matrix,
        feature_names,
        labels,
        buckets,
        min_support=max(1, int(args.min_slice_support)),
    )

    attribution_path = output_dir / "content_feature_attribution.csv"
    slices_path = output_dir / "content_feature_slices.csv"
    write_csv(attribution_path, attribution)
    write_csv(slices_path, slices)

    path_slices_path = None
    path_slices = []
    if args.diagnostic_path_slices:
        path_slices = diagnostic_path_slice_rows(rows, buckets)
        path_slices_path = output_dir / "diagnostic_path_slices.csv"
        write_csv(path_slices_path, path_slices)

    top_k = max(1, int(args.top_k))
    top_by_comparison = {}
    for comparison in sorted({row["comparison"] for row in attribution}):
        selected = [row for row in attribution if row["comparison"] == comparison]
        top_by_comparison[comparison] = selected[:top_k]

    report = {
        "schema": "axon_stage2_residual_content_attribution_v1",
        "protocol": (
            "Frozen prediction residual analysis only; no fitting, no threshold sweep. "
            "source_path/cache_path/source_sha256 are used only to load samples or optional diagnostic slices, "
            "never as model features."
        ),
        "predictions": str(prediction_path),
        "feature_sets": list(feature_sets),
        "feature_count": int(matrix.shape[1]),
        "threshold_argument": args.threshold,
        "confusion": confusion_summary,
        "outputs": {
            "content_feature_attribution_csv": str(attribution_path),
            "content_feature_slices_csv": str(slices_path),
            "diagnostic_path_slices_csv": str(path_slices_path) if path_slices_path is not None else None,
        },
        "top_feature_attribution": top_by_comparison,
        "top_error_lift_slices": slices[:top_k],
        "top_error_count_slices": sorted(slices, key=lambda row: (-row["errors"], -row["error_rate_lift"], row["slice"]))[
            :top_k
        ],
        "diagnostic_path_slices_enabled": bool(args.diagnostic_path_slices),
        "diagnostic_path_slices_top": path_slices[:top_k],
    }

    report_path = output_dir / "residual_content_attribution_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["confusion"], indent=2, ensure_ascii=False))
    print(f"JSON: {report_path}")
    print(f"Attribution CSV: {attribution_path}")
    print(f"Slice CSV: {slices_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
