#!/usr/bin/env python3
"""Analyze Loop61 override classifier exchanges against Loop57.

Read-only attribution. Identity fields are used only for row alignment and
cache lookup; they are not features or model evidence.
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


GROUPS = ("both_correct", "loop57_only_error", "loop61_only_error", "both_error")


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


def _prob(row: dict, *columns: str) -> float:
    for column in columns:
        if column in row and str(row[column]).strip() != "":
            return _to_float(row[column])
    raise ValueError(f"Missing probability column; tried {columns}")


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


def _group_name(loop57_correct: bool, loop61_correct: bool) -> str:
    if loop57_correct and loop61_correct:
        return "both_correct"
    if not loop57_correct and loop61_correct:
        return "loop57_only_error"
    if loop57_correct and not loop61_correct:
        return "loop61_only_error"
    return "both_error"


def _summarize_scores(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"rows": 0}
    base = np.asarray([_to_float(row["base_prob_malicious"]) for row in rows], dtype=np.float32)
    candidate = np.asarray([_to_float(row["candidate_prob_malicious"]) for row in rows], dtype=np.float32)
    loop57_gate = np.asarray([_to_float(row["loop57_gate_prob"]) for row in rows], dtype=np.float32)
    loop61_gate = np.asarray([_to_float(row["loop61_gate_prob"]) for row in rows], dtype=np.float32)
    return {
        "rows": len(rows),
        "base_prob_mean": float(base.mean()),
        "candidate_prob_mean": float(candidate.mean()),
        "loop57_gate_prob_mean": float(loop57_gate.mean()),
        "loop61_gate_prob_mean": float(loop61_gate.mean()),
        "gate_delta_mean": float((loop61_gate - loop57_gate).mean()),
    }


def summarize_feature_means(rows: Sequence[dict], cache_dir: Path, top_k: int) -> dict:
    by_group = {group: [] for group in GROUPS}
    labels_by_group = {group: Counter() for group in GROUPS}
    score_rows = {group: [] for group in GROUPS}
    for row in rows:
        group = row["exchange_group"]
        by_group[group].append(_load_overlay_features(row, cache_dir))
        labels_by_group[group][_to_int(row["label"])] += 1
        score_rows[group].append(row)

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
            "score_summary": _summarize_scores(score_rows[group]),
            "mean_by_feature": {
                name: float(mean[index]) for index, name in enumerate(OVERLAY_BOUNDARY_FEATURE_NAMES)
            },
        }

    repair_minus_harm = means["loop57_only_error"] - means["loop61_only_error"]
    top_repair_features = sorted(
        (
            {
                "feature": name,
                "loop61_repair_mean": float(means["loop57_only_error"][index]),
                "loop61_harm_mean": float(means["loop61_only_error"][index]),
                "difference": float(repair_minus_harm[index]),
            }
            for index, name in enumerate(OVERLAY_BOUNDARY_FEATURE_NAMES)
        ),
        key=lambda item: abs(item["difference"]),
        reverse=True,
    )[:top_k]
    return {"groups": summary, "top_loop61_repair_vs_harm_feature_deltas": top_repair_features}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Loop61 vs Loop57 prediction exchanges.")
    parser.add_argument("--loop57-predictions", type=Path, required=True)
    parser.add_argument("--loop61-predictions", type=Path, required=True)
    parser.add_argument("--overlay-cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=16)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    loop57_rows = _read_rows(args.loop57_predictions)
    loop61_rows = _read_rows(args.loop61_predictions)
    if len(loop57_rows) != len(loop61_rows):
        raise ValueError(f"Prediction row mismatch: {len(loop57_rows)} != {len(loop61_rows)}")

    loop61_by_key = {_row_key(row): row for row in loop61_rows}
    merged = []
    group_counts = Counter()
    transition_counts = Counter()
    override_counts_57 = Counter()
    override_counts_61 = Counter()
    for row57 in loop57_rows:
        key = _row_key(row57)
        row61 = loop61_by_key.get(key)
        if row61 is None:
            raise ValueError(f"Missing Loop61 row for key {key}")
        label57 = _to_int(row57["label"])
        label61 = _to_int(row61["label"])
        if label57 != label61:
            raise ValueError(f"Label mismatch for key {key}: {label57} != {label61}")
        pred57 = _to_int(row57["prediction"])
        pred61 = _to_int(row61["prediction"])
        correct57 = _to_bool(row57["correct"])
        correct61 = _to_bool(row61["correct"])
        group = _group_name(correct57, correct61)
        group_counts[group] += 1
        transition_counts[f"{pred57}->{pred61}|label={label57}"] += 1
        if _to_bool(row57.get("fn_override", False)):
            override_counts_57[group] += 1
        if _to_bool(row61.get("fn_override", False)):
            override_counts_61[group] += 1
        merged.append(
            {
                **row61,
                "base_prob_malicious": _prob(row61, "base_prob_malicious"),
                "candidate_prob_malicious": _prob(row61, "candidate_prob_malicious"),
                "loop57_gate_prob": _prob(row57, "gate_prob_override"),
                "loop61_gate_prob": _prob(row61, "gate_prob_override"),
                "loop57_prediction": pred57,
                "loop61_prediction": pred61,
                "loop57_correct": correct57,
                "loop61_correct": correct61,
                "loop57_override": _to_bool(row57.get("fn_override", False)),
                "loop61_override": _to_bool(row61.get("fn_override", False)),
                "exchange_group": group,
            }
        )

    cache_dir = resolve_path(args.overlay_cache_dir)
    feature_summary = summarize_feature_means(merged, cache_dir, max(1, int(args.top_k)))
    report = {
        "schema": "axon_loop61_exchange_v1",
        "protocol": "read-only exchange audit; identity fields used only for alignment/cache lookup",
        "identity_feature_policy": (
            "sample_index/source_sha256/source_path are alignment and cache lookup fields only; "
            "do not use them as model, threshold, gate, GA, noise, or replacement evidence"
        ),
        "loop57_predictions": str(resolve_path(args.loop57_predictions)),
        "loop61_predictions": str(resolve_path(args.loop61_predictions)),
        "overlay_cache_dir": str(cache_dir),
        "rows": len(merged),
        "exchange_counts": {group: int(group_counts[group]) for group in GROUPS},
        "transition_counts": dict(sorted(transition_counts.items())),
        "loop57_override_counts_by_group": {group: int(override_counts_57[group]) for group in GROUPS},
        "loop61_override_counts_by_group": {group: int(override_counts_61[group]) for group in GROUPS},
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
        "loop57_override",
        "loop61_override",
        "base_prob_malicious",
        "candidate_prob_malicious",
        "loop57_gate_prob",
        "loop61_gate_prob",
        "loop57_prediction",
        "loop61_prediction",
        "source_path",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged:
            if row["exchange_group"] == "both_correct":
                continue
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "rows",
                    "exchange_counts",
                    "transition_counts",
                    "loop57_override_counts_by_group",
                    "loop61_override_counts_by_group",
                )
            },
            indent=2,
        )
    )
    print(f"JSON: {output_json}")
    print(f"CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
