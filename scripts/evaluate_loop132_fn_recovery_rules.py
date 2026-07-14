#!/usr/bin/env python3
"""Evaluate predeclared FN recovery rules over the current Loop130 R5 best.

Rules may only flip the current base prediction from 0 to 1. Identity fields
are used for row alignment and cache lookup only, never as model features.
"""

from __future__ import annotations

import argparse
import csv
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

from evaluate_loop129_content_fp_guard_rules import metrics, read_rows, resolve_path  # noqa: E402
from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from train_stage2_cache_matrix import CONTENT_PE_V2_FEATURE_NAMES, load_valid_feature_npz  # noqa: E402


PRIMARY_PROB_FLOOR = 0.20
FILE_MUTATION_RATIO_FLOOR = 0.04
VERSION_API_RATIO_FLOOR = 0.02
VIRTUAL_RAW_RATIO_LOG_FLOOR = 3.5

RULE_FEATURE_NAMES = [
    "primary_prob_malicious",
    "v2_api_file_mutation_ratio",
    "v2_import_dll_version_api_ratio",
    "v2_section_max_virtual_raw_ratio_log",
]


@dataclass(frozen=True)
class RuleSpec:
    name: str
    description: str


RULES = [
    RuleSpec(
        name="R9_file_or_version_api_recovery",
        description=(
            "base prediction=0 and primary_prob>=0.20 and "
            "(v2_api_file_mutation_ratio>=0.04 or v2_import_dll_version_api_ratio>=0.02)"
        ),
    ),
    RuleSpec(
        name="R10_virtual_raw_ratio_recovery",
        description=(
            "base prediction=0 and primary_prob>=0.20 and "
            "v2_section_max_virtual_raw_ratio_log>=3.5"
        ),
    ),
    RuleSpec(
        name="R11_union_file_version_or_virtual_raw",
        description="R9 or R10",
    ),
]


def row_key(row: dict, key_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")) for column in key_columns)


def align_rows(rows: list[dict], other_rows: list[dict], key_columns: Sequence[str], label: str) -> list[dict]:
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
    if not source_sha:
        raise ValueError("source_sha256 is required for content PE v2 cache lookup")
    return source_sha


def load_pe_v2_features(row: dict, cache_dir: Path) -> np.ndarray:
    cache_path = resolve_path(cache_dir) / f"{_cache_key(row)}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing content PE v2 sidecar cache: {cache_path}")
    features = load_valid_feature_npz(cache_path, len(CONTENT_PE_V2_FEATURE_NAMES))
    if features is None:
        raise ValueError(f"Bad content PE v2 sidecar cache: {cache_path}")
    return features.astype(np.float32, copy=False)


def build_feature_table(
    rows: list[dict],
    primary_rows: Sequence[dict],
    content_pe_v2_cache_dir: Path,
) -> dict[str, np.ndarray]:
    assert_no_identity_feature_names(RULE_FEATURE_NAMES, context="Loop132 FN recovery rules")
    v2 = {name: index for index, name in enumerate(CONTENT_PE_V2_FEATURE_NAMES)}
    required_v2 = [
        "v2_api_file_mutation_ratio",
        "v2_import_dll_version_api_ratio",
        "v2_section_max_virtual_raw_ratio_log",
    ]
    missing_v2 = sorted(set(required_v2) - set(v2))
    if missing_v2:
        raise ValueError(f"Missing content PE v2 feature names: {missing_v2}")

    table = {
        "primary_prob_malicious": np.asarray(
            [float(row["stage2_prob_malicious"]) for row in primary_rows],
            dtype=np.float32,
        )
    }
    for name in required_v2:
        table[name] = np.zeros(len(rows), dtype=np.float32)
    for index, row in enumerate(rows):
        pe2 = load_pe_v2_features(row, content_pe_v2_cache_dir)
        for name in required_v2:
            table[name][index] = pe2[v2[name]]
    return table


def _r9_mask(base_pred: np.ndarray, feature_table: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (base_pred == 0)
        & (feature_table["primary_prob_malicious"] >= PRIMARY_PROB_FLOOR)
        & (
            (feature_table["v2_api_file_mutation_ratio"] >= FILE_MUTATION_RATIO_FLOOR)
            | (feature_table["v2_import_dll_version_api_ratio"] >= VERSION_API_RATIO_FLOOR)
        )
    )


def _r10_mask(base_pred: np.ndarray, feature_table: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (base_pred == 0)
        & (feature_table["primary_prob_malicious"] >= PRIMARY_PROB_FLOOR)
        & (feature_table["v2_section_max_virtual_raw_ratio_log"] >= VIRTUAL_RAW_RATIO_LOG_FLOOR)
    )


def rule_mask(rule_name: str, base_pred: np.ndarray, feature_table: dict[str, np.ndarray]) -> np.ndarray:
    if rule_name == "R9_file_or_version_api_recovery":
        return _r9_mask(base_pred, feature_table)
    if rule_name == "R10_virtual_raw_ratio_recovery":
        return _r10_mask(base_pred, feature_table)
    if rule_name == "R11_union_file_version_or_virtual_raw":
        return _r9_mask(base_pred, feature_table) | _r10_mask(base_pred, feature_table)
    raise ValueError(f"Unknown rule: {rule_name}")


def evaluate_rules(
    rows: list[dict],
    feature_table: dict[str, np.ndarray],
    rule_names: Sequence[str],
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    base_pred = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)
    results = {}
    masks = {}
    predictions_by_rule = {}
    for rule_name in rule_names:
        mask = rule_mask(rule_name, base_pred, feature_table)
        predictions = base_pred.copy()
        predictions[mask] = 1
        results[rule_name] = {
            "rule": rule_name,
            "description": next(rule.description for rule in RULES if rule.name == rule_name),
            "metrics": metrics(labels, predictions),
            "flips": int(np.count_nonzero(mask)),
            "flipped_label1": int(np.count_nonzero(mask & (labels == 1))),
            "flipped_label0": int(np.count_nonzero(mask & (labels == 0))),
        }
        masks[rule_name] = mask
        predictions_by_rule[rule_name] = predictions
    return results, masks, predictions_by_rule, labels, base_pred


def write_predictions(path: Path, rows: list[dict], predictions: np.ndarray, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["fn_recovery_flip"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            output = dict(row)
            output["fn_recovery_flip"] = "1" if bool(mask[index]) else "0"
            output["prediction"] = str(int(predictions[index]))
            output["correct"] = "True" if int(predictions[index]) == int(row["label"]) else "False"
            writer.writerow(output)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Loop132 FN recovery rules over a base prediction CSV.")
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--primary-predictions", type=Path, required=True)
    parser.add_argument("--content-pe-v2-cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-predictions-csv", type=Path, required=True)
    parser.add_argument("--select-rule", default=None, help="Use an already selected rule; otherwise select best rule on this split.")
    parser.add_argument("--key-columns", default="sample_index,source_sha256")
    args = parser.parse_args(argv)

    key_columns = tuple(item.strip() for item in args.key_columns.split(",") if item.strip())
    base_rows = read_rows(args.base_predictions)
    primary_rows = align_rows(base_rows, read_rows(args.primary_predictions), key_columns, "primary")
    feature_table = build_feature_table(base_rows, primary_rows, args.content_pe_v2_cache_dir)
    rule_names = [rule.name for rule in RULES]
    results, masks, predictions_by_rule, labels, base_pred = evaluate_rules(base_rows, feature_table, rule_names)
    base_metrics = metrics(labels, base_pred)
    if args.select_rule:
        if args.select_rule not in results:
            raise ValueError(f"Unknown selected rule: {args.select_rule}")
        selected_rule = args.select_rule
    else:
        selected_rule = max(
            results,
            key=lambda name: (results[name]["metrics"]["f1"], -results[name]["metrics"]["errors"]),
        )
    write_predictions(args.output_predictions_csv, base_rows, predictions_by_rule[selected_rule], masks[selected_rule])

    report = {
        "schema": "axon_loop132_fn_recovery_rules_v1",
        "protocol": "predeclared content PE v2 FN recovery rules; only base 0->1 flips are allowed; no fitting",
        "base_predictions": str(resolve_path(args.base_predictions)),
        "primary_predictions": str(resolve_path(args.primary_predictions)),
        "content_pe_v2_cache_dir": str(resolve_path(args.content_pe_v2_cache_dir)),
        "output_predictions_csv": str(resolve_path(args.output_predictions_csv)),
        "feature_names": RULE_FEATURE_NAMES,
        "base_metrics": base_metrics,
        "selected_rule": selected_rule,
        "selected": results[selected_rule],
        "rules": results,
        "records": {"total": len(base_rows), "kept": len(base_rows)},
    }
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
