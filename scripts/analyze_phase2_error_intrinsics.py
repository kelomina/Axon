#!/usr/bin/env python3
"""Build Phase 2 intrinsic error and noise audit queues.

Identity columns are kept only so humans can locate samples. The grouping and
ranking logic uses labels, probabilities, and cached numeric PE/stat features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPS = 1.0e-6


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


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


def iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def error_type(label: int, probability: float, threshold: float) -> str:
    prediction = int(probability >= threshold)
    if label == 0 and prediction == 1:
        return "FP"
    if label == 1 and prediction == 0:
        return "FN"
    return ""


def confidence_bucket(kind: str, probability: float, threshold: float) -> str:
    if kind == "FP":
        if probability >= 0.90:
            return "fp_high_conf_ge_0.90"
        if probability >= 0.75:
            return "fp_mid_conf_0.75_0.90"
        return f"fp_near_threshold_{threshold:.2f}_0.75"
    if probability < 0.10:
        return "fn_high_conf_lt_0.10"
    if probability < 0.30:
        return "fn_mid_conf_0.10_0.30"
    return f"fn_near_threshold_0.30_{threshold:.2f}"


@dataclass
class FeatureBundle:
    pe: np.ndarray
    stat: np.ndarray
    byte_len: int
    orig_len: Optional[int]
    orig_len_key: str


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum: Optional[np.ndarray] = None
        self.sumsq: Optional[np.ndarray] = None

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if self.sum is None:
            self.sum = np.zeros(arr.shape, dtype=np.float64)
            self.sumsq = np.zeros(arr.shape, dtype=np.float64)
        self.count += 1
        self.sum += arr
        self.sumsq += arr * arr

    def mean(self) -> Optional[np.ndarray]:
        if self.count == 0 or self.sum is None:
            return None
        return self.sum / float(self.count)

    def std(self) -> Optional[np.ndarray]:
        if self.count == 0 or self.sum is None or self.sumsq is None:
            return None
        mean = self.mean()
        assert mean is not None
        variance = np.maximum((self.sumsq / float(self.count)) - mean * mean, 0.0)
        return np.sqrt(variance)


def reservoir_add(
    reservoir: list[dict[str, str]],
    row: dict[str, str],
    seen_count: int,
    limit: int,
    rng: random.Random,
) -> None:
    if limit <= 0:
        return
    if len(reservoir) < limit:
        reservoir.append(dict(row))
        return
    replace_index = rng.randrange(seen_count)
    if replace_index < limit:
        reservoir[replace_index] = dict(row)


def load_feature_bundle(cache_path: Path) -> FeatureBundle:
    with np.load(resolve_path(cache_path), allow_pickle=False) as data:
        pe = np.asarray(data["pe_features"], dtype=np.float32).reshape(-1)
        stat = np.asarray(data["stat_features"], dtype=np.float32).reshape(-1)
        byte_len = int(np.asarray(data["byte_sequence"]).reshape(-1).shape[0]) if "byte_sequence" in data.files else 0
        orig_len: Optional[int] = None
        orig_len_key = ""
        for candidate_key in ("orig_length", "orig_len", "original_length"):
            if candidate_key in data.files:
                orig_len = int(np.asarray(data[candidate_key]).reshape(-1)[0])
                orig_len_key = candidate_key
                break
    return FeatureBundle(pe=pe, stat=stat, byte_len=byte_len, orig_len=orig_len, orig_len_key=orig_len_key)


def audit_queue(kind: str, probability: float, margin: float) -> tuple[str, int]:
    if kind == "FN" and probability <= 0.01:
        return "label_noise_extreme_fn", 0
    if kind == "FP" and probability >= 0.99:
        return "label_noise_extreme_fp", 1
    if kind == "FN" and probability <= 0.05:
        return "label_noise_high_fn", 2
    if kind == "FP" and probability >= 0.95:
        return "label_noise_high_fp", 3
    if margin <= 0.02:
        return "calibration_near_threshold", 4
    if margin <= 0.05:
        return "calibration_broad_near_threshold", 5
    return "model_behavior_review", 9


def feature_anomaly_flags(features: FeatureBundle) -> list[str]:
    flags: list[str] = []
    if features.byte_len <= 0:
        flags.append("byte_sequence_missing_or_empty")
    if features.orig_len is not None and features.orig_len <= 0:
        flags.append("orig_len_missing_or_zero")
    if features.pe.size == 0:
        flags.append("pe_empty")
    elif not np.any(features.pe):
        flags.append("pe_all_zero")
    if features.stat.size == 0:
        flags.append("stat_empty")
    elif not np.any(features.stat):
        flags.append("stat_all_zero")
    if int(np.isnan(features.pe).sum()):
        flags.append("pe_has_nan")
    if int(np.isnan(features.stat).sum()):
        flags.append("stat_has_nan")
    if int(np.isinf(features.pe).sum()):
        flags.append("pe_has_inf")
    if int(np.isinf(features.stat).sum()):
        flags.append("stat_has_inf")
    return flags


def update_group_stats(
    stats: dict[str, dict[str, RunningStats]],
    group: str,
    features: FeatureBundle,
) -> None:
    stats[group]["pe"].update(features.pe)
    stats[group]["stat"].update(features.stat)


def top_feature_shifts(
    group_stats: dict[str, RunningStats],
    background_stats: dict[str, RunningStats],
    *,
    prefix: str,
    top_k: int,
) -> list[dict]:
    output: list[dict] = []
    group_mean = group_stats[prefix].mean()
    bg_mean = background_stats[prefix].mean()
    bg_std = background_stats[prefix].std()
    if group_mean is None or bg_mean is None or bg_std is None:
        return output
    denom = np.maximum(bg_std, EPS)
    z = (group_mean - bg_mean) / denom
    order = np.argsort(np.abs(z))[::-1][:top_k]
    for index in order:
        output.append(
            {
                "feature": f"{prefix}_{int(index)}",
                "mean": float(group_mean[index]),
                "background_mean": float(bg_mean[index]),
                "background_std": float(bg_std[index]),
                "z_score": float(z[index]),
            }
        )
    return output


def max_abs_z(features: FeatureBundle, background: dict[str, RunningStats]) -> tuple[float, str]:
    best_score = 0.0
    best_feature = ""
    for prefix, arr in (("pe", features.pe), ("stat", features.stat)):
        mean = background[prefix].mean()
        std = background[prefix].std()
        if mean is None or std is None:
            continue
        z = np.abs((arr.astype(np.float64) - mean) / np.maximum(std, EPS))
        if z.size == 0:
            continue
        index = int(np.argmax(z))
        score = float(z[index])
        if score > best_score:
            best_score = score
            best_feature = f"{prefix}_{index}"
    return best_score, best_feature


def build_phase2_error_intrinsics(
    *,
    predictions_csv: Path,
    output_json: Path,
    output_review_csv: Path,
    threshold: float,
    prob_column: str,
    background_per_label: int,
    seed: int,
    top_k_features: int,
) -> dict:
    rng = random.Random(seed)
    errors: list[dict[str, str]] = []
    background_rows: dict[int, list[dict[str, str]]] = {0: [], 1: []}
    background_seen: Counter = Counter()
    total_rows = 0
    label_counts: Counter = Counter()
    prediction_counts: Counter = Counter()
    error_counts: Counter = Counter()
    bucket_counts: Counter = Counter()

    for row in iter_csv_rows(predictions_csv):
        total_rows += 1
        label = _int(row.get("label"))
        probability = _float(row.get(prob_column))
        kind = error_type(label, probability, threshold)
        label_counts[str(label)] += 1
        prediction_counts[str(int(probability >= threshold))] += 1
        if kind:
            copied = dict(row)
            copied["_error_type"] = kind
            copied["_confidence_bucket"] = confidence_bucket(kind, probability, threshold)
            copied["_probability"] = str(probability)
            errors.append(copied)
            error_counts[kind] += 1
            bucket_counts[copied["_confidence_bucket"]] += 1
        else:
            background_seen[label] += 1
            reservoir_add(
                background_rows[label],
                row,
                background_seen[label],
                background_per_label,
                rng,
            )

    group_stats: dict[str, dict[str, RunningStats]] = defaultdict(lambda: {"pe": RunningStats(), "stat": RunningStats()})
    background_stats: dict[int, dict[str, RunningStats]] = {
        0: {"pe": RunningStats(), "stat": RunningStats()},
        1: {"pe": RunningStats(), "stat": RunningStats()},
    }
    review_rows: list[dict] = []
    feature_load_failures = 0

    for label, rows in background_rows.items():
        for row in rows:
            try:
                features = load_feature_bundle(Path(row["cache_path"]))
                background_stats[label]["pe"].update(features.pe)
                background_stats[label]["stat"].update(features.stat)
            except Exception:
                feature_load_failures += 1

    for row in errors:
        label = _int(row.get("label"))
        kind = row["_error_type"]
        bucket = row["_confidence_bucket"]
        probability = _float(row.get("_probability"))
        try:
            features = load_feature_bundle(Path(row["cache_path"]))
            update_group_stats(group_stats, kind, features)
            update_group_stats(group_stats, bucket, features)
            anomaly_score, anomaly_feature = max_abs_z(features, background_stats[label])
            flags = feature_anomaly_flags(features)
            feature_status = "loaded"
            byte_seq_len = features.byte_len
            orig_len = "" if features.orig_len is None else features.orig_len
            orig_len_key = features.orig_len_key
            pe_feature_dim = int(features.pe.size)
            stat_feature_dim = int(features.stat.size)
        except Exception as exc:
            feature_load_failures += 1
            anomaly_score = math.nan
            anomaly_feature = ""
            flags = [f"feature_load_failed:{type(exc).__name__}"]
            feature_status = f"load_failed:{type(exc).__name__}"
            byte_seq_len = ""
            orig_len = ""
            orig_len_key = ""
            pe_feature_dim = ""
            stat_feature_dim = ""
        margin = abs(probability - threshold)
        queue, priority = audit_queue(kind, probability, margin)
        if flags and queue == "model_behavior_review":
            queue = "feature_anomaly_review"
            priority = 6
        review_rows.append(
            {
                "audit_priority": priority,
                "audit_queue": queue,
                "error_type": kind,
                "confidence_bucket": bucket,
                "sample_index": row.get("sample_index", ""),
                "split": row.get("split", ""),
                "label": str(label),
                "source_sha256": row.get("source_sha256", ""),
                prob_column: probability,
                "threshold": float(threshold),
                "margin_to_threshold": margin,
                "feature_anomaly_max_abs_z": anomaly_score,
                "feature_anomaly_feature": anomaly_feature,
                "feature_anomaly_flags": ";".join(flags),
                "feature_status": feature_status,
                "byte_seq_len": byte_seq_len,
                "orig_len": orig_len,
                "orig_len_key": orig_len_key,
                "pe_feature_dim": pe_feature_dim,
                "stat_feature_dim": stat_feature_dim,
                "review_status": "pending",
                "reviewed_label": "",
                "label_change_recommendation": "",
                "reviewer_note": "",
                "evidence_source": "",
                "source_path": row.get("source_path", ""),
                "cache_path": row.get("cache_path", ""),
            }
        )

    review_rows.sort(
        key=lambda item: (
            int(item["audit_priority"]),
            -float(item["feature_anomaly_max_abs_z"]) if not math.isnan(float(item["feature_anomaly_max_abs_z"])) else -1.0,
            str(item["source_sha256"]),
        )
    )
    resolved_review = resolve_path(output_review_csv)
    resolved_review.parent.mkdir(parents=True, exist_ok=True)
    review_fieldnames = [
        "audit_priority",
        "audit_queue",
        "error_type",
        "confidence_bucket",
        "sample_index",
        "split",
        "label",
        "source_sha256",
        prob_column,
        "threshold",
        "margin_to_threshold",
        "feature_anomaly_max_abs_z",
        "feature_anomaly_feature",
        "feature_anomaly_flags",
        "feature_status",
        "byte_seq_len",
        "orig_len",
        "orig_len_key",
        "pe_feature_dim",
        "stat_feature_dim",
        "review_status",
        "reviewed_label",
        "label_change_recommendation",
        "reviewer_note",
        "evidence_source",
        "source_path",
        "cache_path",
    ]
    with resolved_review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(review_rows)

    group_reports = {}
    for group, stats in sorted(group_stats.items()):
        bg_label = 0 if group.startswith("FP") or group.startswith("fp_") else 1
        group_reports[group] = {
            "count": stats["pe"].count,
            "top_pe_shifts_vs_same_label_correct_background": top_feature_shifts(
                stats, background_stats[bg_label], prefix="pe", top_k=top_k_features
            ),
            "top_stat_shifts_vs_same_label_correct_background": top_feature_shifts(
                stats, background_stats[bg_label], prefix="stat", top_k=top_k_features
            ),
        }

    payload = {
        "schema": "axon_phase2_error_intrinsics_v1",
        "predictions_csv": str(resolve_path(predictions_csv)),
        "review_csv": str(resolved_review),
        "threshold": float(threshold),
        "probability_column": prob_column,
        "total_rows": total_rows,
        "label_counts": dict(sorted(label_counts.items())),
        "prediction_counts": dict(sorted(prediction_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "confidence_bucket_counts": dict(sorted(bucket_counts.items())),
        "audit_queue_counts": dict(sorted(Counter(str(row["audit_queue"]) for row in review_rows).items())),
        "background_sampling": {
            "seed": int(seed),
            "requested_per_label": int(background_per_label),
            "seen_correct_by_label": {str(key): int(value) for key, value in sorted(background_seen.items())},
            "sampled_correct_by_label": {str(key): len(value) for key, value in sorted(background_rows.items())},
        },
        "feature_load_failures": feature_load_failures,
        "cache_schema_notes": [
            "Current fixed-v2 training cache may omit original file length; missing orig_len is recorded as unavailable, not as a feature anomaly.",
            "orig_len_missing_or_zero is emitted only when an orig_length/orig_len/original_length field exists and is non-positive.",
        ],
        "group_reports": group_reports,
        "phase2_priority": [
            "Review high-confidence FP/FN first; threshold tuning cannot explain them.",
            "Use source_sha256/cache_path only for locating samples and verifying cache identity.",
            "Do not group or decide by source_path, filename, directory, or extension.",
            "Feature anomaly z-scores are triage signals, not relabel evidence by themselves.",
        ],
        "identity_feature_policy": (
            "Path, filename, directory, extension, sample_index, cache_path, and source_sha256 are alignment/location fields only. "
            "This report groups by label, probability bucket, and cached numeric PE/stat features."
        ),
    }
    resolved_output = resolve_path(output_json)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phase 2 intrinsic error/noise audit queues.")
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-review-csv", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--prob-column", default="calibrated_prob_malicious")
    parser.add_argument("--background-per-label", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-features", type=int, default=8)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_phase2_error_intrinsics(
        predictions_csv=args.predictions_csv,
        output_json=args.output_json,
        output_review_csv=args.output_review_csv,
        threshold=float(args.threshold),
        prob_column=str(args.prob_column),
        background_per_label=int(args.background_per_label),
        seed=int(args.seed),
        top_k_features=int(args.top_k_features),
    )
    print(
        json.dumps(
            {
                "total_rows": payload["total_rows"],
                "error_counts": payload["error_counts"],
                "confidence_bucket_counts": payload["confidence_bucket_counts"],
                "audit_queue_counts": payload["audit_queue_counts"],
                "review_csv": payload["review_csv"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
