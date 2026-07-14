#!/usr/bin/env python3
"""Build Loop133 error and noise audit artifacts for the current R5 best.

Identity columns are kept only for audit/review lookup. Ranking and summaries
use labels, predictions, probabilities, guard status, and numeric content
features.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from evaluate_loop129_content_fp_guard_rules import metrics, read_rows, resolve_path  # noqa: E402
from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES  # noqa: E402
from train_stage2_cache_matrix import (  # noqa: E402
    CONTENT_PE_V2_FEATURE_NAMES,
    CONTENT_STRING_FEATURE_NAMES,
    load_valid_feature_npz,
)


FEATURE_NAMES = [
    "content_is_dll",
    "content_export_count_log",
    "content_dir_export_log_size",
    "content_dir_security_log_size",
    "content_overlay_log_size",
    "content_resource_entry_count_log",
    "content_resource_type_count_log",
    "content_dir_resource_size_ratio",
    "content_dir_resource_log_size",
    "content_overlay_entropy",
    "content_import_api_count_log",
    "content_avg_imports_per_dll",
    "content_image_base_log",
    "v2_resource_data_entry_count_log",
    "v2_resource_type_icon_count_log",
    "v2_resource_type_version_count_log",
    "v2_resource_type_manifest_count_log",
    "v2_resource_type_dialog_count_log",
    "v2_last_section_entropy",
    "v2_section_max_virtual_raw_ratio_log",
    "v2_api_file_mutation_ratio",
    "v2_import_dll_version_api_ratio",
    "string_benign_vendor_count_log",
    "string_version_resource_count_log",
    "string_script_exec_count_log",
    "string_script_exec_present",
]


def row_key(row: dict, key_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")) for column in key_columns)


def align_rows(rows: list[dict], other_rows: list[dict] | None, key_columns: Sequence[str], label: str) -> list[dict | None]:
    if other_rows is None:
        return [None] * len(rows)
    by_key = {row_key(row, key_columns): row for row in other_rows}
    if len(by_key) != len(other_rows):
        raise ValueError(f"{label} predictions contain duplicate alignment keys")
    aligned = []
    missing = []
    for row in rows:
        key = row_key(row, key_columns)
        other = by_key.get(key)
        if other is None:
            missing.append(key)
        aligned.append(other)
    if missing:
        preview = ", ".join(str(item) for item in missing[:5])
        raise ValueError(f"Missing {len(missing)} {label} rows; first keys: {preview}")
    return aligned


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def error_kind(label: int, prediction: int) -> str:
    if label == 0 and prediction == 1:
        return "fp"
    if label == 1 and prediction == 0:
        return "fn"
    return ""


def priority_and_reasons(
    *,
    label: int,
    prediction: int,
    probability: float,
    guard_flip: bool,
    extra_flip_over_reference: bool,
) -> tuple[int, list[str]]:
    kind = error_kind(label, prediction)
    reasons: list[str] = []
    priority = 999
    if kind == "fn" and guard_flip:
        reasons.append("guard_harmful_fn")
        priority = min(priority, 0)
    if kind == "fn" and extra_flip_over_reference:
        reasons.append("r5_extra_harmful_fn")
        priority = min(priority, 1)
    if kind == "fn" and probability < 0.10:
        reasons.append("high_conf_fn_lt_0.10")
        priority = min(priority, 10)
    if kind == "fp" and probability >= 0.90:
        reasons.append("high_conf_fp_ge_0.90")
        priority = min(priority, 20)
    if kind and abs(probability - 0.5) <= 0.05:
        reasons.append("near_primary_prob_0.45_0.55")
        priority = min(priority, 30)
    if kind:
        reasons.append(f"model_{kind}")
        priority = min(priority, 90)
    return priority, reasons


def _cache_key(row: dict) -> str:
    source_sha = str(row.get("source_sha256") or "").strip().casefold()
    if not source_sha:
        raise ValueError("source_sha256 is required for sidecar cache lookup")
    return source_sha


def _load_sidecar(row: dict, cache_dir: Path, feature_names: Sequence[str], family: str) -> np.ndarray:
    cache_path = resolve_path(cache_dir) / f"{_cache_key(row)}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing {family} sidecar cache: {cache_path}")
    features = load_valid_feature_npz(cache_path, len(feature_names))
    if features is None:
        raise ValueError(f"Bad {family} sidecar cache: {cache_path}")
    return features.astype(np.float32, copy=False)


def build_feature_table(
    rows: list[dict],
    content_pe_cache_dir: Path,
    content_pe_v2_cache_dir: Path,
    content_string_cache_dir: Path,
) -> dict[str, np.ndarray]:
    assert_no_identity_feature_names(FEATURE_NAMES, context="Loop133 R5 error audit features")
    v1_index = {name: index for index, name in enumerate(CONTENT_PE_V1_FEATURE_NAMES)}
    v2_index = {name: index for index, name in enumerate(CONTENT_PE_V2_FEATURE_NAMES)}
    string_index = {name: index for index, name in enumerate(CONTENT_STRING_FEATURE_NAMES)}
    missing = sorted(
        name
        for name in FEATURE_NAMES
        if name not in v1_index and name not in v2_index and name not in string_index
    )
    if missing:
        raise ValueError(f"Missing audit feature names: {missing}")

    table = {name: np.zeros(len(rows), dtype=np.float32) for name in FEATURE_NAMES}
    for index, row in enumerate(rows):
        pe1 = _load_sidecar(row, content_pe_cache_dir, CONTENT_PE_V1_FEATURE_NAMES, "content_pe_v1")
        pe2 = _load_sidecar(row, content_pe_v2_cache_dir, CONTENT_PE_V2_FEATURE_NAMES, "content_pe_v2")
        string_features = _load_sidecar(row, content_string_cache_dir, CONTENT_STRING_FEATURE_NAMES, "content_string")
        for name in FEATURE_NAMES:
            if name in v1_index:
                table[name][index] = pe1[v1_index[name]]
            elif name in v2_index:
                table[name][index] = pe2[v2_index[name]]
            else:
                table[name][index] = string_features[string_index[name]]
    return table


def feature_summary(feature_table: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    output = {"count": int(np.count_nonzero(mask))}
    if output["count"] == 0:
        return output
    for name in FEATURE_NAMES:
        values = feature_table[name][mask]
        output[name] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return output


def build_audit(
    *,
    rows: list[dict],
    primary_rows: Sequence[dict | None],
    reference_rows: Sequence[dict | None],
    feature_table: dict[str, np.ndarray],
    max_review_rows: int,
) -> tuple[dict, list[dict], list[dict]]:
    labels = np.asarray([_int(row.get("label")) for row in rows], dtype=np.int64)
    predictions = np.asarray([_int(row.get("prediction")) for row in rows], dtype=np.int64)
    probabilities = np.asarray([_float(row.get("stage2_prob_malicious")) for row in rows], dtype=np.float32)
    primary_predictions = np.asarray(
        [
            _int(primary_row.get("prediction")) if primary_row is not None else _int(row.get("prediction"))
            for row, primary_row in zip(rows, primary_rows)
        ],
        dtype=np.int64,
    )
    reference_predictions = np.asarray(
        [
            _int(reference_row.get("prediction")) if reference_row is not None else _int(row.get("prediction"))
            for row, reference_row in zip(rows, reference_rows)
        ],
        dtype=np.int64,
    )
    guard_flips = predictions != primary_predictions
    extra_flips = guard_flips & (reference_predictions == primary_predictions)
    fp_mask = (labels == 0) & (predictions == 1)
    fn_mask = (labels == 1) & (predictions == 0)
    guard_repaired_fp = guard_flips & (labels == 0) & (predictions == 0)
    guard_harmful_fn = guard_flips & (labels == 1) & (predictions == 0)
    extra_repaired_fp = extra_flips & (labels == 0) & (predictions == 0)
    extra_harmful_fn = extra_flips & (labels == 1) & (predictions == 0)

    summary = {
        "schema": "axon_loop133_r5_error_audit_v1",
        "protocol": "post-hoc audit only; identity fields are retained for review lookup but not used as model evidence",
        "metrics": metrics(labels, predictions),
        "counts": {
            "total": int(labels.shape[0]),
            "errors": int(np.count_nonzero(fp_mask | fn_mask)),
            "fp": int(np.count_nonzero(fp_mask)),
            "fn": int(np.count_nonzero(fn_mask)),
            "guard_flips": int(np.count_nonzero(guard_flips)),
            "guard_repaired_fp": int(np.count_nonzero(guard_repaired_fp)),
            "guard_harmful_fn": int(np.count_nonzero(guard_harmful_fn)),
            "extra_flips_over_reference": int(np.count_nonzero(extra_flips)),
            "extra_repaired_fp": int(np.count_nonzero(extra_repaired_fp)),
            "extra_harmful_fn": int(np.count_nonzero(extra_harmful_fn)),
            "high_conf_fp_ge_0_90": int(np.count_nonzero(fp_mask & (probabilities >= 0.90))),
            "high_conf_fn_lt_0_10": int(np.count_nonzero(fn_mask & (probabilities < 0.10))),
            "near_primary_prob_errors_0_45_0_55": int(
                np.count_nonzero((fp_mask | fn_mask) & (np.abs(probabilities - 0.5) <= 0.05))
            ),
        },
        "feature_summary": {
            "fp": feature_summary(feature_table, fp_mask),
            "fn": feature_summary(feature_table, fn_mask),
            "guard_repaired_fp": feature_summary(feature_table, guard_repaired_fp),
            "guard_harmful_fn": feature_summary(feature_table, guard_harmful_fn),
            "extra_repaired_fp": feature_summary(feature_table, extra_repaired_fp),
            "extra_harmful_fn": feature_summary(feature_table, extra_harmful_fn),
            "high_conf_fp_ge_0_90": feature_summary(feature_table, fp_mask & (probabilities >= 0.90)),
            "high_conf_fn_lt_0_10": feature_summary(feature_table, fn_mask & (probabilities < 0.10)),
        },
    }

    review_rows: list[dict] = []
    flip_rows: list[dict] = []
    lane_counts: Counter = Counter()
    for index, row in enumerate(rows):
        label = int(labels[index])
        prediction = int(predictions[index])
        probability = float(probabilities[index])
        guard_flip = bool(guard_flips[index])
        extra_flip = bool(extra_flips[index])
        kind = error_kind(label, prediction)
        priority, reasons = priority_and_reasons(
            label=label,
            prediction=prediction,
            probability=probability,
            guard_flip=guard_flip,
            extra_flip_over_reference=extra_flip,
        )
        base_output = {
            "priority": priority,
            "reasons": ";".join(reasons),
            "sample_index": row.get("sample_index", ""),
            "source_sha256": row.get("source_sha256", ""),
            "label": str(label),
            "prediction": str(prediction),
            "error_type": kind,
            "guard_flip": "1" if guard_flip else "0",
            "extra_flip_over_reference": "1" if extra_flip else "0",
            "primary_prob": f"{probability:.10f}",
            "primary_pred": str(int(primary_predictions[index])),
            "reference_pred": str(int(reference_predictions[index])),
        }
        for name in FEATURE_NAMES:
            base_output[name] = f"{float(feature_table[name][index]):.10g}"
        base_output["source_path"] = row.get("source_path", "")
        base_output["cache_path"] = row.get("cache_path", "")
        if kind:
            lane_counts.update(reasons)
            review_rows.append(base_output)
        if guard_flip:
            flip_output = dict(base_output)
            if label == 0 and prediction == 0:
                flip_output["outcome"] = "repaired_fp"
            elif label == 1 and prediction == 0:
                flip_output["outcome"] = "harmful_fn"
            else:
                flip_output["outcome"] = "other"
            flip_rows.append(flip_output)
    review_rows.sort(key=lambda item: (int(item["priority"]), item["error_type"], item["sample_index"]))
    if max_review_rows > 0:
        review_rows = review_rows[:max_review_rows]
    flip_rows.sort(key=lambda item: (item["outcome"], item["sample_index"]))
    summary["review_lane_counts"] = dict(lane_counts)
    summary["review_rows_written"] = len(review_rows)
    summary["flip_rows_written"] = len(flip_rows)
    return summary, review_rows, flip_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build Loop133 R5 full-test error audit artifacts.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--primary-predictions", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, default=None)
    parser.add_argument("--content-pe-cache-dir", type=Path, required=True)
    parser.add_argument("--content-pe-v2-cache-dir", type=Path, required=True)
    parser.add_argument("--content-string-cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-review-csv", type=Path, required=True)
    parser.add_argument("--output-flips-csv", type=Path, required=True)
    parser.add_argument("--max-review-rows", type=int, default=0, help="0 writes all error rows")
    parser.add_argument("--key-columns", default="sample_index,source_sha256")
    args = parser.parse_args(argv)

    if args.max_review_rows < 0:
        raise ValueError("--max-review-rows must be non-negative")
    key_columns = tuple(item.strip() for item in args.key_columns.split(",") if item.strip())
    rows = read_rows(args.predictions)
    primary_rows = align_rows(rows, read_rows(args.primary_predictions), key_columns, "primary")
    reference_rows = align_rows(
        rows,
        read_rows(args.reference_predictions) if args.reference_predictions else None,
        key_columns,
        "reference",
    )
    feature_table = build_feature_table(
        rows,
        args.content_pe_cache_dir,
        args.content_pe_v2_cache_dir,
        args.content_string_cache_dir,
    )
    summary, review_rows, flip_rows = build_audit(
        rows=rows,
        primary_rows=primary_rows,
        reference_rows=reference_rows,
        feature_table=feature_table,
        max_review_rows=args.max_review_rows,
    )
    summary.update(
        {
            "predictions": str(resolve_path(args.predictions)),
            "primary_predictions": str(resolve_path(args.primary_predictions)),
            "reference_predictions": str(resolve_path(args.reference_predictions)) if args.reference_predictions else None,
            "content_pe_cache_dir": str(resolve_path(args.content_pe_cache_dir)),
            "content_pe_v2_cache_dir": str(resolve_path(args.content_pe_v2_cache_dir)),
            "content_string_cache_dir": str(resolve_path(args.content_string_cache_dir)),
            "output_review_csv": str(resolve_path(args.output_review_csv)),
            "output_flips_csv": str(resolve_path(args.output_flips_csv)),
            "feature_names": FEATURE_NAMES,
        }
    )

    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.output_review_csv, review_rows)
    write_csv(args.output_flips_csv, flip_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
