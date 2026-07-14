#!/usr/bin/env python3
"""Evaluate content/string FP guard refinements over Loop129 R4.

Rules may only flip the current primary prediction from 1 to 0. Identity fields
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

from evaluate_loop129_content_fp_guard_rules import (  # noqa: E402
    align_optional,
    build_feature_table as build_content_feature_table,
    metrics,
    read_rows,
    resolve_path,
)
from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from train_stage2_cache_matrix import (  # noqa: E402
    CONTENT_PE_V2_FEATURE_NAMES,
    CONTENT_STRING_FEATURE_NAMES,
    load_valid_feature_npz,
)


VENDOR_STRING_LOG_THRESHOLD = 3.0
VERSION_STRING_LOG_THRESHOLD = float(np.log1p(2.0))
RESOURCE_ENTRY_LOG_THRESHOLD = float(np.log1p(39.0))
DIALOG_PROTECTOR_LOG_THRESHOLD = 3.0

RULE_FEATURE_NAMES = [
    "primary_prob_malicious",
    "v2_resource_data_entry_count_log",
    "v2_resource_type_icon_count_log",
    "content_dir_resource_size_ratio",
    "content_resource_entry_count_log",
    "v2_resource_type_dialog_count_log",
    "string_benign_vendor_count_log",
    "string_version_resource_count_log",
]


@dataclass(frozen=True)
class RuleSpec:
    name: str
    description: str


RULES = [
    RuleSpec(
        name="R4_resource_icon_lowconf_resource_ratio_floor",
        description=(
            "primary=1 and any conservative=0 and primary_prob<=0.65 and "
            "v2_resource_data_entry_count_log>=2.0 and v2_resource_type_icon_count_log>=1.5 and "
            "content_dir_resource_size_ratio>=0.001"
        ),
    ),
    RuleSpec(
        name="R5_r4_plus_vendor_strings",
        description=(
            "R4 plus remaining possible_guard rows with string_benign_vendor_count_log>=3.0"
        ),
    ),
    RuleSpec(
        name="R6_r4_plus_version_resource_strings",
        description=(
            "R4 plus remaining possible_guard rows with string_version_resource_count_log>=log(3)"
        ),
    ),
    RuleSpec(
        name="R7_r4_plus_resource_entry_40",
        description=(
            "R4 plus remaining possible_guard rows with content_resource_entry_count_log>=log(40)"
        ),
    ),
    RuleSpec(
        name="R8_r5_dialog_protector",
        description=(
            "R5, except R5 flips with v2_resource_type_dialog_count_log>=3.0 are restored to primary=1"
        ),
    ),
]


def _cache_key(row: dict) -> str:
    source_sha = str(row.get("source_sha256") or "").strip().casefold()
    if not source_sha:
        raise ValueError("source_sha256 is required for content string cache lookup")
    return source_sha


def load_string_feature_npz(row: dict, cache_dir: Path) -> np.ndarray:
    cache_path = resolve_path(cache_dir) / f"{_cache_key(row)}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing string sidecar cache: {cache_path}")
    features = load_valid_feature_npz(cache_path, len(CONTENT_STRING_FEATURE_NAMES))
    if features is None:
        raise ValueError(f"Bad string sidecar cache: {cache_path}")
    return features.astype(np.float32, copy=False)


def load_pe_v2_feature_npz(row: dict, cache_dir: Path) -> np.ndarray:
    cache_path = resolve_path(cache_dir) / f"{_cache_key(row)}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing content PE v2 sidecar cache: {cache_path}")
    features = load_valid_feature_npz(cache_path, len(CONTENT_PE_V2_FEATURE_NAMES))
    if features is None:
        raise ValueError(f"Bad content PE v2 sidecar cache: {cache_path}")
    return features.astype(np.float32, copy=False)


def build_feature_table(
    rows: list[dict],
    content_pe_cache_dir: Path,
    content_pe_v2_cache_dir: Path,
    content_string_cache_dir: Path,
) -> dict[str, np.ndarray]:
    feature_table = build_content_feature_table(rows, content_pe_cache_dir, content_pe_v2_cache_dir)
    if "v2_resource_type_dialog_count_log" in RULE_FEATURE_NAMES:
        v2_index = CONTENT_PE_V2_FEATURE_NAMES.index("v2_resource_type_dialog_count_log")
        feature_table["v2_resource_type_dialog_count_log"] = np.asarray(
            [load_pe_v2_feature_npz(row, content_pe_v2_cache_dir)[v2_index] for row in rows],
            dtype=np.float32,
        )
    string_features = np.vstack([load_string_feature_npz(row, content_string_cache_dir) for row in rows])
    for index, name in enumerate(CONTENT_STRING_FEATURE_NAMES):
        if name in RULE_FEATURE_NAMES:
            feature_table[name] = string_features[:, index]
    return feature_table


def _base_r4_mask(
    possible: np.ndarray,
    primary_prob: np.ndarray,
    feature_table: dict[str, np.ndarray],
) -> np.ndarray:
    return (
        possible
        & (primary_prob <= 0.65)
        & (feature_table["v2_resource_data_entry_count_log"] >= 2.0)
        & (feature_table["v2_resource_type_icon_count_log"] >= 1.5)
        & (feature_table["content_dir_resource_size_ratio"] >= 0.001)
    )


def _rule_mask(
    rule_name: str,
    possible: np.ndarray,
    primary_prob: np.ndarray,
    feature_table: dict[str, np.ndarray],
) -> np.ndarray:
    r4_mask = _base_r4_mask(possible, primary_prob, feature_table)
    remaining_possible = possible & ~r4_mask
    if rule_name == "R4_resource_icon_lowconf_resource_ratio_floor":
        return r4_mask
    if rule_name == "R5_r4_plus_vendor_strings":
        return r4_mask | (
            remaining_possible
            & (feature_table["string_benign_vendor_count_log"] >= VENDOR_STRING_LOG_THRESHOLD)
        )
    if rule_name == "R6_r4_plus_version_resource_strings":
        return r4_mask | (
            remaining_possible
            & (feature_table["string_version_resource_count_log"] >= VERSION_STRING_LOG_THRESHOLD)
        )
    if rule_name == "R7_r4_plus_resource_entry_40":
        return r4_mask | (
            remaining_possible
            & (feature_table["content_resource_entry_count_log"] >= RESOURCE_ENTRY_LOG_THRESHOLD)
        )
    if rule_name == "R8_r5_dialog_protector":
        r5_mask = r4_mask | (
            remaining_possible
            & (feature_table["string_benign_vendor_count_log"] >= VENDOR_STRING_LOG_THRESHOLD)
        )
        protector = r5_mask & (
            feature_table["v2_resource_type_dialog_count_log"] >= DIALOG_PROTECTOR_LOG_THRESHOLD
        )
        return r5_mask & ~protector
    raise ValueError(f"Unknown rule: {rule_name}")


def evaluate_rules(
    rows: list[dict],
    conservative_rows: Sequence[dict | None],
    old_rows: Sequence[dict | None],
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
        mask = _rule_mask(rule_name, possible, primary_prob, feature_table)
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
            "extra_flips_over_r4": int(
                np.count_nonzero(mask & ~_base_r4_mask(possible, primary_prob, feature_table))
            ),
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
    parser = argparse.ArgumentParser(description="Evaluate Loop130 content/string FP guard rules.")
    parser.add_argument("--primary-predictions", type=Path, required=True)
    parser.add_argument("--conservative-predictions", type=Path, required=True)
    parser.add_argument("--old-predictions", type=Path, default=None)
    parser.add_argument("--content-pe-cache-dir", type=Path, required=True)
    parser.add_argument("--content-pe-v2-cache-dir", type=Path, required=True)
    parser.add_argument("--content-string-cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-predictions-csv", type=Path, required=True)
    parser.add_argument("--select-rule", default=None, help="Use an already selected rule; otherwise select best rule on this split.")
    parser.add_argument("--key-columns", default="sample_index,source_sha256")
    args = parser.parse_args(argv)

    assert_no_identity_feature_names(RULE_FEATURE_NAMES, context="Loop130 content/string FP guard rules")
    key_columns = tuple(item.strip() for item in args.key_columns.split(",") if item.strip())
    primary_rows = read_rows(args.primary_predictions)
    conservative_rows = align_optional(primary_rows, read_rows(args.conservative_predictions), key_columns, "conservative")
    old_rows = align_optional(primary_rows, read_rows(args.old_predictions) if args.old_predictions else None, key_columns, "old")
    feature_table = build_feature_table(
        primary_rows,
        args.content_pe_cache_dir,
        args.content_pe_v2_cache_dir,
        args.content_string_cache_dir,
    )

    rule_names = [rule.name for rule in RULES]
    results, masks, predictions_by_rule, labels, primary_pred, _primary_prob = evaluate_rules(
        primary_rows, conservative_rows, old_rows, feature_table, rule_names
    )
    primary_metrics = metrics(labels, primary_pred)
    if args.select_rule:
        if args.select_rule not in results:
            raise ValueError(f"Unknown selected rule: {args.select_rule}")
        selected_rule = args.select_rule
    else:
        selected_rule = max(
            results,
            key=lambda name: (results[name]["metrics"]["f1"], -results[name]["metrics"]["errors"]),
        )
    write_predictions(args.output_predictions_csv, primary_rows, predictions_by_rule[selected_rule], masks[selected_rule])

    report = {
        "schema": "axon_loop130_content_string_guard_rules_v1",
        "protocol": (
            "predeclared content/string numeric FP guard rules; only primary 1->0 flips are allowed; no fitting"
        ),
        "primary_predictions": str(resolve_path(args.primary_predictions)),
        "conservative_predictions": str(resolve_path(args.conservative_predictions)),
        "old_predictions": str(resolve_path(args.old_predictions)) if args.old_predictions else None,
        "content_pe_cache_dir": str(resolve_path(args.content_pe_cache_dir)),
        "content_pe_v2_cache_dir": str(resolve_path(args.content_pe_v2_cache_dir)),
        "content_string_cache_dir": str(resolve_path(args.content_string_cache_dir)),
        "output_predictions_csv": str(resolve_path(args.output_predictions_csv)),
        "feature_names": RULE_FEATURE_NAMES,
        "primary_metrics": primary_metrics,
        "selected_rule": selected_rule,
        "selected": results[selected_rule],
        "rules": results,
        "records": {"total": len(primary_rows), "kept": len(primary_rows)},
    }
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
