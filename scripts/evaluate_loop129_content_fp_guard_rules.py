#!/usr/bin/env python3
"""Evaluate predeclared content-structure FP guard rules.

Rules may only flip the current primary prediction from 1 to 0. Identity fields
are used for row alignment and cache lookup only, never as model features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES  # noqa: E402
from train_stage2_cache_matrix import CONTENT_PE_V2_FEATURE_NAMES  # noqa: E402


RULE_FEATURE_NAMES = [
    "primary_prob_malicious",
    "conservative_prob_malicious",
    "old_content_cross_prob_malicious",
    "v2_resource_data_entry_count_log",
    "v2_resource_type_icon_count_log",
    "content_resource_entry_count_log",
    "content_resource_type_count_log",
    "content_dir_resource_size_ratio",
    "content_is_dll",
    "content_export_count_log",
    "content_dir_export_log_size",
    "v2_export_ordinal_span_log",
    "v2_export_pattern_com_present",
]


@dataclass(frozen=True)
class RuleSpec:
    name: str
    description: str


RULES = [
    RuleSpec(
        name="R2_resource_icon_lowconf",
        description=(
            "primary=1 and any conservative=0 and primary_prob<=0.65 and "
            "v2_resource_data_entry_count_log>=2.0 and v2_resource_type_icon_count_log>=1.5"
        ),
    ),
    RuleSpec(
        name="R3_resource_icon_lowconf_not_dll",
        description="R2 plus content_is_dll==0",
    ),
    RuleSpec(
        name="R4_resource_icon_lowconf_resource_ratio_floor",
        description="R2 plus content_dir_resource_size_ratio>=0.001",
    ),
]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict, key_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")) for column in key_columns)


def align_optional(rows: list[dict], other_rows: list[dict] | None, key_columns: Sequence[str], label: str) -> list[dict | None]:
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


def _cache_key(row: dict) -> str:
    source_sha = str(row.get("source_sha256") or "").strip().casefold()
    if source_sha:
        return source_sha
    source_path = str(row.get("source_path") or "")
    return hashlib.sha256(str(resolve_path(Path(source_path))).encode("utf-8", errors="ignore")).hexdigest()


def load_feature_npz(row: dict, cache_dir: Path, expected_dim: int, family: str) -> np.ndarray:
    cache_path = resolve_path(cache_dir) / f"{_cache_key(row)}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing {family} sidecar cache: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as data:
        if "features" not in data.files:
            raise ValueError(f"{family} cache missing features array: {cache_path}")
        features = data["features"].astype(np.float32, copy=False)
    if features.shape != (expected_dim,):
        raise ValueError(f"Bad {family} shape for {cache_path}: {features.shape} != {(expected_dim,)}")
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite {family} features: {cache_path}")
    return features


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    labels = labels.astype(np.int64, copy=False)
    predictions = predictions.astype(np.int64, copy=False)
    tp = int(np.count_nonzero((labels == 1) & (predictions == 1)))
    tn = int(np.count_nonzero((labels == 0) & (predictions == 0)))
    fp = int(np.count_nonzero((labels == 0) & (predictions == 1)))
    fn = int(np.count_nonzero((labels == 1) & (predictions == 0)))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "accuracy": float((tp + tn) / max(labels.shape[0], 1)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "errors": fp + fn,
    }


def build_feature_table(rows: list[dict], content_pe_cache_dir: Path, content_pe_v2_cache_dir: Path) -> dict[str, np.ndarray]:
    assert_no_identity_feature_names(RULE_FEATURE_NAMES, context="Loop129 content FP guard rules")
    c = {name: index for index, name in enumerate(CONTENT_PE_V1_FEATURE_NAMES)}
    v2 = {name: index for index, name in enumerate(CONTENT_PE_V2_FEATURE_NAMES)}
    required_v1 = [
        "content_resource_entry_count_log",
        "content_resource_type_count_log",
        "content_dir_resource_size_ratio",
        "content_is_dll",
        "content_export_count_log",
        "content_dir_export_log_size",
    ]
    required_v2 = [
        "v2_resource_data_entry_count_log",
        "v2_resource_type_icon_count_log",
        "v2_export_ordinal_span_log",
        "v2_export_pattern_com_present",
    ]
    missing_v1 = sorted(set(required_v1) - set(c))
    missing_v2 = sorted(set(required_v2) - set(v2))
    if missing_v1 or missing_v2:
        raise ValueError(f"Missing feature names: v1={missing_v1} v2={missing_v2}")
    table = {name: np.zeros(len(rows), dtype=np.float32) for name in required_v1 + required_v2}
    for index, row in enumerate(rows):
        pe1 = load_feature_npz(row, content_pe_cache_dir, len(CONTENT_PE_V1_FEATURE_NAMES), "content_pe_v1")
        pe2 = load_feature_npz(row, content_pe_v2_cache_dir, len(CONTENT_PE_V2_FEATURE_NAMES), "content_pe_v2")
        for name in required_v1:
            table[name][index] = pe1[c[name]]
        for name in required_v2:
            table[name][index] = pe2[v2[name]]
    return table


def evaluate_rules(
    *,
    rows: list[dict],
    conservative_rows: list[dict | None],
    old_rows: list[dict | None],
    feature_table: dict[str, np.ndarray],
    rule_names: Sequence[str],
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    primary_prob = np.asarray([float(row["stage2_prob_malicious"]) for row in rows], dtype=np.float32)
    primary_pred = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    conservative_pred = np.asarray(
        [int(row["prediction"]) if row is not None else 1 for row in conservative_rows], dtype=np.int64
    )
    old_pred = np.asarray([int(row["prediction"]) if row is not None else 1 for row in old_rows], dtype=np.int64)
    possible = (primary_pred == 1) & ((conservative_pred == 0) | (old_pred == 0))

    results = {}
    masks = {}
    predictions_by_rule = {}
    for rule_name in rule_names:
        if rule_name == "R2_resource_icon_lowconf":
            mask = (
                possible
                & (primary_prob <= 0.65)
                & (feature_table["v2_resource_data_entry_count_log"] >= 2.0)
                & (feature_table["v2_resource_type_icon_count_log"] >= 1.5)
            )
        elif rule_name == "R3_resource_icon_lowconf_not_dll":
            mask = (
                possible
                & (primary_prob <= 0.65)
                & (feature_table["v2_resource_data_entry_count_log"] >= 2.0)
                & (feature_table["v2_resource_type_icon_count_log"] >= 1.5)
                & (feature_table["content_is_dll"] == 0.0)
            )
        elif rule_name == "R4_resource_icon_lowconf_resource_ratio_floor":
            mask = (
                possible
                & (primary_prob <= 0.65)
                & (feature_table["v2_resource_data_entry_count_log"] >= 2.0)
                & (feature_table["v2_resource_type_icon_count_log"] >= 1.5)
                & (feature_table["content_dir_resource_size_ratio"] >= 0.001)
            )
        else:
            raise ValueError(f"Unknown rule: {rule_name}")
        predictions = primary_pred.copy()
        predictions[mask] = 0
        results[rule_name] = {
            "rule": rule_name,
            "description": next(rule.description for rule in RULES if rule.name == rule_name),
            "metrics": metrics(labels, predictions),
            "flips": int(np.count_nonzero(mask)),
            "flipped_label0": int(np.count_nonzero(mask & (labels == 0))),
            "flipped_label1": int(np.count_nonzero(mask & (labels == 1))),
            "possible_guard_rows": int(np.count_nonzero(possible)),
        }
        masks[rule_name] = mask
        predictions_by_rule[rule_name] = predictions
    return results, masks, predictions_by_rule, labels, primary_pred, primary_prob


def write_predictions(path: Path, rows: list[dict], predictions: np.ndarray, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["guard_flip"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            output = dict(row)
            output["guard_flip"] = "1" if bool(mask[index]) else "0"
            output["prediction"] = str(int(predictions[index]))
            output["correct"] = "True" if int(predictions[index]) == int(row["label"]) else "False"
            writer.writerow(output)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Loop129 content-structure FP guard rules.")
    parser.add_argument("--primary-predictions", type=Path, required=True)
    parser.add_argument("--conservative-predictions", type=Path, required=True)
    parser.add_argument("--old-predictions", type=Path, default=None)
    parser.add_argument("--content-pe-cache-dir", type=Path, required=True)
    parser.add_argument("--content-pe-v2-cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-predictions-csv", type=Path, required=True)
    parser.add_argument("--select-rule", default=None, help="Use an already selected rule; otherwise select best rule on this split.")
    parser.add_argument("--key-columns", default="sample_index,source_sha256")
    args = parser.parse_args(argv)

    key_columns = tuple(item.strip() for item in args.key_columns.split(",") if item.strip())
    primary_rows = read_rows(args.primary_predictions)
    conservative_rows = align_optional(primary_rows, read_rows(args.conservative_predictions), key_columns, "conservative")
    old_rows = align_optional(primary_rows, read_rows(args.old_predictions) if args.old_predictions else None, key_columns, "old")
    feature_table = build_feature_table(primary_rows, args.content_pe_cache_dir, args.content_pe_v2_cache_dir)
    rule_names = [rule.name for rule in RULES]
    results, masks, predictions_by_rule, labels, primary_pred, _primary_prob = evaluate_rules(
        rows=primary_rows,
        conservative_rows=conservative_rows,
        old_rows=old_rows,
        feature_table=feature_table,
        rule_names=rule_names,
    )
    primary_metrics = metrics(labels, primary_pred)
    if args.select_rule:
        selected_rule = args.select_rule
    else:
        selected_rule = min(rule_names, key=lambda name: (results[name]["metrics"]["errors"], -results[name]["metrics"]["f1"]))
    if selected_rule not in results:
        raise ValueError(f"Unknown selected rule: {selected_rule}")
    output_json = resolve_path(args.output_json)
    output_predictions_csv = resolve_path(args.output_predictions_csv)
    write_predictions(output_predictions_csv, primary_rows, predictions_by_rule[selected_rule], masks[selected_rule])
    report = {
        "schema": "axon_loop129_content_fp_guard_rules_v1",
        "protocol": "predeclared content numeric FP guard rules; only primary 1->0 flips are allowed; no fitting",
        "primary_predictions": str(resolve_path(args.primary_predictions)),
        "conservative_predictions": str(resolve_path(args.conservative_predictions)),
        "old_predictions": str(resolve_path(args.old_predictions)) if args.old_predictions else None,
        "content_pe_cache_dir": str(resolve_path(args.content_pe_cache_dir)),
        "content_pe_v2_cache_dir": str(resolve_path(args.content_pe_v2_cache_dir)),
        "output_predictions_csv": str(output_predictions_csv),
        "feature_names": RULE_FEATURE_NAMES,
        "primary_metrics": primary_metrics,
        "selected_rule": selected_rule,
        "selected": results[selected_rule],
        "rules": results,
        "records": {"total": len(primary_rows), "kept": len(primary_rows)},
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"primary_metrics": primary_metrics, "selected_rule": selected_rule, "selected": results[selected_rule]}, ensure_ascii=False, indent=2))
    print(f"JSON: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
