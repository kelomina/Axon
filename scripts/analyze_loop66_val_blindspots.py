#!/usr/bin/env python3
"""Analyze Loop57 validation blindspots with content-derived features only.

This is a read-only Val audit. It does not train, tune thresholds, touch
Test-10k/full-test, relabel samples, or mutate the split. Identity fields are
used only for CSV alignment and sidecar cache lookup.
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
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES  # noqa: E402
from train_loop55_overlay_boundary import OVERLAY_BOUNDARY_FEATURE_NAMES  # noqa: E402


FINAL_GROUPS = ("tp", "tn", "fp", "fn")
EXCHANGE_GROUPS = ("both_correct", "base_error_final_repaired", "base_correct_final_harmed", "both_error")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _to_int(value: object) -> int:
    return int(float(str(value).strip()))


def _cache_key(row: dict) -> str:
    source_sha = str(row.get("source_sha256") or "").strip().casefold()
    if source_sha:
        return source_sha
    source_path = str(row.get("source_path") or "")
    return hashlib.sha256(str(resolve_path(Path(source_path))).encode("utf-8", errors="ignore")).hexdigest()


def _cache_path(row: dict, cache_dir: Path) -> Path:
    return resolve_path(cache_dir) / f"{_cache_key(row)}.npz"


def _load_feature_vector(
    row: dict,
    *,
    cache_dir: Path,
    expected_dim: int,
    feature_family: str,
) -> np.ndarray:
    path = _cache_path(row, cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing {feature_family} sidecar cache: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "features" not in data.files:
            raise ValueError(f"{feature_family} cache missing features array: {path}")
        features = data["features"].astype(np.float32, copy=False)
    if features.shape != (expected_dim,):
        raise ValueError(f"Bad {feature_family} feature shape for {path}: {features.shape} != {(expected_dim,)}")
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite {feature_family} features: {path}")
    return features


def _final_group(label: int, prediction: int) -> str:
    if label == 1 and prediction == 1:
        return "tp"
    if label == 0 and prediction == 0:
        return "tn"
    if label == 0 and prediction == 1:
        return "fp"
    return "fn"


def _exchange_group(label: int, base_prediction: int, final_prediction: int) -> str:
    base_correct = int(base_prediction) == int(label)
    final_correct = int(final_prediction) == int(label)
    if base_correct and final_correct:
        return "both_correct"
    if not base_correct and final_correct:
        return "base_error_final_repaired"
    if base_correct and not final_correct:
        return "base_correct_final_harmed"
    return "both_error"


def _row_snapshot(row: dict, *, base_threshold: float) -> dict:
    label = _to_int(row["label"])
    final_prediction = _to_int(row["prediction"])
    base_prob = _to_float(row.get("base_prob_malicious"))
    base_prediction = int(base_prob >= base_threshold)
    final_prob = _to_float(row.get("final_prob_malicious", row.get("stage2_prob_malicious", base_prob)))
    candidate_prob = _to_float(row.get("candidate_prob_malicious"))
    gate_prob = _to_float(row.get("gate_prob_override"))
    return {
        "label": label,
        "base_prob": base_prob,
        "base_prediction": base_prediction,
        "candidate_prob": candidate_prob,
        "gate_prob": gate_prob,
        "final_prob": final_prob,
        "final_prediction": final_prediction,
        "fn_override": _to_bool(row.get("fn_override", False)),
        "final_group": _final_group(label, final_prediction),
        "exchange_group": _exchange_group(label, base_prediction, final_prediction),
    }


def _score_summary(snapshots: Sequence[dict]) -> dict:
    if not snapshots:
        return {"rows": 0}
    keys = ("base_prob", "candidate_prob", "gate_prob", "final_prob")
    output = {"rows": len(snapshots)}
    for key in keys:
        values = np.asarray([float(row[key]) for row in snapshots], dtype=np.float32)
        output[f"{key}_mean"] = float(values.mean())
        output[f"{key}_min"] = float(values.min())
        output[f"{key}_max"] = float(values.max())
    output["override_count"] = int(sum(1 for row in snapshots if row["fn_override"]))
    return output


def _summarize_groups(
    *,
    matrix: np.ndarray,
    feature_names: Sequence[str],
    group_labels: Sequence[str],
    snapshots: Sequence[dict],
    all_groups: Sequence[str],
) -> dict:
    summary = {}
    group_labels_arr = np.asarray(group_labels)
    for group in all_groups:
        mask = group_labels_arr == group
        count = int(mask.sum())
        if count:
            mean = matrix[mask].mean(axis=0)
            nonzero_rows = int((np.count_nonzero(matrix[mask], axis=1) > 0).sum())
        else:
            mean = np.zeros(len(feature_names), dtype=np.float32)
            nonzero_rows = 0
        group_snapshots = [snap for snap, item_group in zip(snapshots, group_labels) if item_group == group]
        summary[group] = {
            "rows": count,
            "nonzero_feature_rows": nonzero_rows,
            "score_summary": _score_summary(group_snapshots),
            "mean_by_feature": {
                name: float(mean[index]) for index, name in enumerate(feature_names)
            },
        }
    return summary


def _contrast_rows(
    *,
    matrix: np.ndarray,
    feature_names: Sequence[str],
    group_labels: Sequence[str],
    left_group: str,
    right_group: str,
    contrast: str,
    feature_family: str,
    top_k: int,
) -> list[dict]:
    labels = np.asarray(group_labels)
    left_mask = labels == left_group
    right_mask = labels == right_group
    if int(left_mask.sum()) == 0 or int(right_mask.sum()) == 0:
        return []
    left_mean = matrix[left_mask].mean(axis=0)
    right_mean = matrix[right_mask].mean(axis=0)
    diff = left_mean - right_mean
    order = np.argsort(-np.abs(diff))[:top_k]
    rows = []
    for index in order:
        rows.append(
            {
                "contrast": contrast,
                "feature_family": feature_family,
                "feature": feature_names[int(index)],
                "left_group": left_group,
                "right_group": right_group,
                "left_rows": int(left_mask.sum()),
                "right_rows": int(right_mask.sum()),
                "left_mean": float(left_mean[int(index)]),
                "right_mean": float(right_mean[int(index)]),
                "difference": float(diff[int(index)]),
                "abs_difference": float(abs(diff[int(index)])),
            }
        )
    return rows


def _write_delta_csv(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = [
        "contrast",
        "feature_family",
        "feature",
        "left_group",
        "right_group",
        "left_rows",
        "right_rows",
        "left_mean",
        "right_mean",
        "difference",
        "abs_difference",
    ]
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit_val_blindspots(
    *,
    loop57_val_predictions: Path,
    content_pe_cache_dir: Path,
    overlay_boundary_cache_dir: Path,
    output_json: Path,
    output_csv: Path,
    base_threshold: float = 0.5,
    top_k: int = 20,
) -> dict:
    assert_no_identity_feature_names(CONTENT_PE_V1_FEATURE_NAMES, context="Loop66 content PE v1 features")
    assert_no_identity_feature_names(OVERLAY_BOUNDARY_FEATURE_NAMES, context="Loop66 overlay boundary features")

    rows = read_rows(loop57_val_predictions)
    if not rows:
        raise ValueError("No Loop57 validation prediction rows found")

    snapshots = []
    content_features = []
    overlay_features = []
    split_counts: Counter[str] = Counter()
    final_counts: Counter[str] = Counter()
    exchange_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for row in rows:
        split = str(row.get("split", "")).strip()
        split_counts[split] += 1
        if split != "val":
            raise ValueError(f"Loop66 is Val-only, got split={split!r}")
        snap = _row_snapshot(row, base_threshold=base_threshold)
        snapshots.append(snap)
        final_counts[snap["final_group"]] += 1
        exchange_counts[snap["exchange_group"]] += 1
        label_counts[str(snap["label"])] += 1
        content_features.append(
            _load_feature_vector(
                row,
                cache_dir=content_pe_cache_dir,
                expected_dim=len(CONTENT_PE_V1_FEATURE_NAMES),
                feature_family="content_pe_v1",
            )
        )
        overlay_features.append(
            _load_feature_vector(
                row,
                cache_dir=overlay_boundary_cache_dir,
                expected_dim=len(OVERLAY_BOUNDARY_FEATURE_NAMES),
                feature_family="overlay_boundary",
            )
        )

    content_matrix = np.vstack(content_features).astype(np.float32, copy=False)
    overlay_matrix = np.vstack(overlay_features).astype(np.float32, copy=False)
    final_groups = [snap["final_group"] for snap in snapshots]
    exchange_groups = [snap["exchange_group"] for snap in snapshots]
    top_k = max(1, int(top_k))

    delta_rows: list[dict] = []
    for feature_family, matrix, names in (
        ("content_pe_v1", content_matrix, CONTENT_PE_V1_FEATURE_NAMES),
        ("overlay_boundary", overlay_matrix, OVERLAY_BOUNDARY_FEATURE_NAMES),
    ):
        delta_rows.extend(
            _contrast_rows(
                matrix=matrix,
                feature_names=names,
                group_labels=final_groups,
                left_group="fp",
                right_group="tn",
                contrast="final_fp_minus_final_tn",
                feature_family=feature_family,
                top_k=top_k,
            )
        )
        delta_rows.extend(
            _contrast_rows(
                matrix=matrix,
                feature_names=names,
                group_labels=final_groups,
                left_group="fn",
                right_group="tp",
                contrast="final_fn_minus_final_tp",
                feature_family=feature_family,
                top_k=top_k,
            )
        )
        delta_rows.extend(
            _contrast_rows(
                matrix=matrix,
                feature_names=names,
                group_labels=exchange_groups,
                left_group="base_error_final_repaired",
                right_group="base_correct_final_harmed",
                contrast="repaired_minus_harmed",
                feature_family=feature_family,
                top_k=top_k,
            )
        )

    _write_delta_csv(output_csv, delta_rows)
    report = {
        "schema": "axon_loop66_val_blindspot_content_audit_v1",
        "protocol": (
            "read-only Val blindspot audit; no training, no threshold selection, "
            "no Test-10k/full-test use, no relabeling, no split/cache mutation"
        ),
        "identity_feature_policy": (
            "source_path/source_sha256/cache_path/sample_index/split are only used for row alignment "
            "and sidecar cache lookup; they are not model evidence"
        ),
        "loop57_val_predictions": str(resolve_path(loop57_val_predictions)),
        "content_pe_cache_dir": str(resolve_path(content_pe_cache_dir)),
        "overlay_boundary_cache_dir": str(resolve_path(overlay_boundary_cache_dir)),
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "base_threshold": float(base_threshold),
        "final_group_counts": {group: int(final_counts[group]) for group in FINAL_GROUPS},
        "exchange_group_counts": {group: int(exchange_counts[group]) for group in EXCHANGE_GROUPS},
        "feature_shapes": {
            "content_pe_v1": list(content_matrix.shape),
            "overlay_boundary": list(overlay_matrix.shape),
        },
        "group_summaries": {
            "final": {
                "content_pe_v1": _summarize_groups(
                    matrix=content_matrix,
                    feature_names=CONTENT_PE_V1_FEATURE_NAMES,
                    group_labels=final_groups,
                    snapshots=snapshots,
                    all_groups=FINAL_GROUPS,
                ),
                "overlay_boundary": _summarize_groups(
                    matrix=overlay_matrix,
                    feature_names=OVERLAY_BOUNDARY_FEATURE_NAMES,
                    group_labels=final_groups,
                    snapshots=snapshots,
                    all_groups=FINAL_GROUPS,
                ),
            },
            "exchange": {
                "content_pe_v1": _summarize_groups(
                    matrix=content_matrix,
                    feature_names=CONTENT_PE_V1_FEATURE_NAMES,
                    group_labels=exchange_groups,
                    snapshots=snapshots,
                    all_groups=EXCHANGE_GROUPS,
                ),
                "overlay_boundary": _summarize_groups(
                    matrix=overlay_matrix,
                    feature_names=OVERLAY_BOUNDARY_FEATURE_NAMES,
                    group_labels=exchange_groups,
                    snapshots=snapshots,
                    all_groups=EXCHANGE_GROUPS,
                ),
            },
        },
        "top_feature_deltas": delta_rows,
        "outputs": {
            "delta_csv": str(resolve_path(output_csv)),
            "summary_json": str(resolve_path(output_json)),
        },
    }
    output_json_resolved = resolve_path(output_json)
    output_json_resolved.parent.mkdir(parents=True, exist_ok=True)
    output_json_resolved.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Loop66 Val-only content blindspot audit for Loop57.")
    parser.add_argument("--loop57-val-predictions", type=Path, required=True)
    parser.add_argument("--content-pe-cache-dir", type=Path, required=True)
    parser.add_argument("--overlay-boundary-cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--base-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    report = audit_val_blindspots(
        loop57_val_predictions=args.loop57_val_predictions,
        content_pe_cache_dir=args.content_pe_cache_dir,
        overlay_boundary_cache_dir=args.overlay_boundary_cache_dir,
        output_json=args.output_json,
        output_csv=args.output_csv,
        base_threshold=args.base_threshold,
        top_k=args.top_k,
    )
    print(
        json.dumps(
            {
                "rows": report["rows"],
                "final_group_counts": report["final_group_counts"],
                "exchange_group_counts": report["exchange_group_counts"],
                "outputs": report["outputs"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
