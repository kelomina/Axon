#!/usr/bin/env python3
"""Analyze Loop55 overlay-boundary error exchanges against Loop28.

This script is read-only analysis. Identity fields are used only to align rows
and find sidecar caches; they are not model features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for item in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from train_loop55_overlay_boundary import OVERLAY_BOUNDARY_FEATURE_NAMES  # noqa: E402
from train_stage2_cache_matrix import resolve_path  # noqa: E402


GROUPS = ("both_correct", "loop28_only_error", "loop55_only_error", "both_error")


def _read_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: dict) -> str:
    sample_index = str(row.get("sample_index") or "").strip()
    if sample_index:
        return f"sample:{sample_index}"
    source_sha = str(row.get("source_sha256") or "").strip().lower()
    if source_sha:
        return f"sha:{source_sha}"
    source_path = str(row.get("source_path") or "").strip()
    return "path:" + hashlib.sha256(source_path.encode("utf-8", errors="ignore")).hexdigest()


def _to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _to_int(value: object) -> int:
    return int(float(str(value).strip()))


def _to_float(value: object) -> float:
    return float(str(value).strip())


def _cache_path_for_row(row: dict, cache_dir: Path) -> Path:
    key = str(row.get("source_sha256") or "").strip().lower()
    if not key:
        key = hashlib.sha256(str(resolve_path(Path(row["source_path"]))).encode("utf-8", errors="ignore")).hexdigest()
    return cache_dir / f"{key}.npz"


def _load_overlay_features(row: dict, cache_dir: Path) -> np.ndarray:
    cache_path = _cache_path_for_row(row, cache_dir)
    with np.load(cache_path, allow_pickle=False) as data:
        features = data["features"].astype(np.float32, copy=False)
    if features.shape != (len(OVERLAY_BOUNDARY_FEATURE_NAMES),):
        raise ValueError(f"Bad overlay feature shape for {cache_path}: {features.shape}")
    return features


def _group_name(loop28_correct: bool, loop55_correct: bool) -> str:
    if loop28_correct and loop55_correct:
        return "both_correct"
    if not loop28_correct and loop55_correct:
        return "loop28_only_error"
    if loop28_correct and not loop55_correct:
        return "loop55_only_error"
    return "both_error"


def summarize_feature_means(rows: Sequence[dict], cache_dir: Path, top_k: int) -> dict:
    by_group = {group: [] for group in GROUPS}
    labels_by_group = {group: Counter() for group in GROUPS}
    for row in rows:
        group = row["exchange_group"]
        by_group[group].append(_load_overlay_features(row, cache_dir))
        labels_by_group[group][_to_int(row["label"])] += 1

    summary = {}
    means = {}
    for group, arrays in by_group.items():
        if arrays:
            matrix = np.vstack(arrays).astype(np.float32, copy=False)
            mean = matrix.mean(axis=0)
        else:
            mean = np.zeros(len(OVERLAY_BOUNDARY_FEATURE_NAMES), dtype=np.float32)
        means[group] = mean
        nonzero = int(sum(1 for item in arrays if np.count_nonzero(item) > 0))
        summary[group] = {
            "rows": len(arrays),
            "label_counts": {str(label): count for label, count in sorted(labels_by_group[group].items())},
            "nonzero_overlay_features": nonzero,
            "mean_by_feature": {
                name: float(mean[index]) for index, name in enumerate(OVERLAY_BOUNDARY_FEATURE_NAMES)
            },
        }

    repair_minus_harm = means["loop28_only_error"] - means["loop55_only_error"]
    top_repair_features = sorted(
        (
            {
                "feature": name,
                "repair_mean": float(means["loop28_only_error"][index]),
                "harm_mean": float(means["loop55_only_error"][index]),
                "difference": float(repair_minus_harm[index]),
            }
            for index, name in enumerate(OVERLAY_BOUNDARY_FEATURE_NAMES)
        ),
        key=lambda item: abs(item["difference"]),
        reverse=True,
    )[:top_k]
    return {"groups": summary, "top_repair_vs_harm_feature_deltas": top_repair_features}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Loop55 vs Loop28 Val error exchanges.")
    parser.add_argument("--loop28-predictions", type=Path, required=True)
    parser.add_argument("--loop55-predictions", type=Path, required=True)
    parser.add_argument("--overlay-cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    loop28_rows = _read_rows(args.loop28_predictions)
    loop55_rows = _read_rows(args.loop55_predictions)
    if len(loop28_rows) != len(loop55_rows):
        raise ValueError(f"Prediction row mismatch: {len(loop28_rows)} != {len(loop55_rows)}")

    loop55_by_key = {_row_key(row): row for row in loop55_rows}
    merged = []
    group_counts = Counter()
    transition_counts = Counter()
    for row28 in loop28_rows:
        key = _row_key(row28)
        row55 = loop55_by_key.get(key)
        if row55 is None:
            raise ValueError(f"Missing Loop55 row for key {key}")
        label28 = _to_int(row28["label"])
        label55 = _to_int(row55["label"])
        if label28 != label55:
            raise ValueError(f"Label mismatch for key {key}: {label28} != {label55}")
        correct28 = _to_bool(row28["correct"])
        correct55 = _to_bool(row55["correct"])
        pred28 = _to_int(row28["prediction"])
        pred55 = _to_int(row55["prediction"])
        group = _group_name(correct28, correct55)
        group_counts[group] += 1
        transition_counts[f"{pred28}->{pred55}|label={label28}"] += 1
        merged.append(
            {
                **row55,
                "loop28_prob": _to_float(row28["stage2_prob_malicious"]),
                "loop55_prob": _to_float(row55["stage2_prob_malicious"]),
                "loop28_prediction": pred28,
                "loop55_prediction": pred55,
                "loop28_correct": correct28,
                "loop55_correct": correct55,
                "exchange_group": group,
            }
        )

    cache_dir = resolve_path(args.overlay_cache_dir)
    feature_summary = summarize_feature_means(merged, cache_dir, max(1, int(args.top_k)))
    report = {
        "schema": "axon_loop55_overlay_exchange_v1",
        "protocol": "read-only Val residual exchange audit; identity fields used only for alignment/cache lookup",
        "loop28_predictions": str(resolve_path(args.loop28_predictions)),
        "loop55_predictions": str(resolve_path(args.loop55_predictions)),
        "overlay_cache_dir": str(cache_dir),
        "rows": len(merged),
        "exchange_counts": {group: int(group_counts[group]) for group in GROUPS},
        "transition_counts": dict(sorted(transition_counts.items())),
        "feature_summary": feature_summary,
    }

    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    output_csv = resolve_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "source_sha256",
        "label",
        "exchange_group",
        "loop28_prob",
        "loop55_prob",
        "loop28_prediction",
        "loop55_prediction",
        "source_path",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged:
            if row["exchange_group"] == "both_correct":
                continue
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    print(json.dumps({key: report[key] for key in ("rows", "exchange_counts", "transition_counts")}, indent=2))
    print(f"JSON: {output_json}")
    print(f"CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
