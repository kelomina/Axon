#!/usr/bin/env python3
"""Select/evaluate a Loop136 kNN prediction-support gate.

The gate uses a frozen Train-memory kNN reference from Stage-2 numeric features.
It can flip Loop136 predictions only when nearby Train examples strongly support
the opposite class. Paths/hashes/sample ids are alignment/cache metadata only.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
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

from config import AxonExperimentConfig  # noqa: E402
from train_stage2_cache_matrix import (  # noqa: E402
    FeatureConfig,
    append_frozen_knn_features,
    build_matrix,
    load_stage2_knn_reference_from_payload,
    read_prediction_rows,
    resolve_path,
)


SCHEMA = "axon_loop146_knn_prediction_support_gate_v1"


def _metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
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
        "errors": int(fp + fn),
    }


def _prepare_rows(rows: Sequence[dict]) -> list[dict]:
    prepared = []
    for row in rows:
        item = dict(row)
        if not str(item.get("prob_malicious", "")).strip():
            item["prob_malicious"] = item.get("stage2_prob_malicious", "")
        prepared.append(item)
    return prepared


def _load_stage2_payload(model_path: Path) -> tuple[dict, FeatureConfig, AxonExperimentConfig, dict]:
    resolved = resolve_path(model_path)
    with resolved.open("rb") as handle:
        payload = pickle.load(handle)
    feature_config = payload["feature_config"]
    if not isinstance(feature_config, FeatureConfig):
        feature_config = FeatureConfig(**dict(feature_config))
    checkpoint_config = AxonExperimentConfig.from_dict(dict(payload["checkpoint_config"]))
    knn_payload = payload.get("knn") or {}
    if not knn_payload.get("enabled"):
        raise ValueError("Stage2 model payload does not contain an enabled kNN reference")
    reference = load_stage2_knn_reference_from_payload(resolved, knn_payload)
    return payload, feature_config, checkpoint_config, reference


def _columns(feature_names: Sequence[str], ref_k: int) -> dict[str, int]:
    mapping = {name: index for index, name in enumerate(feature_names)}
    required = [
        f"knn{ref_k}_mal_ratio",
        f"knn{ref_k}_weighted_mal_ratio",
        f"knn{ref_k}_mean_similarity",
        "knn_top1_label",
        "knn_top1_similarity",
        "knn_top1_top2_gap",
    ]
    missing = [name for name in required if name not in mapping]
    if missing:
        raise ValueError(f"Missing kNN feature names: {missing}")
    return {name: mapping[name] for name in required}


def _base_predictions(rows: Sequence[dict]) -> np.ndarray:
    return np.asarray([int(row["prediction"]) for row in rows], dtype=np.int64)


def _apply_rule(
    base_predictions: np.ndarray,
    knn_features: np.ndarray,
    feature_names: Sequence[str],
    *,
    ref_k: int,
    min_mal_for_0to1: float,
    max_mal_for_1to0: float,
    min_weighted_agree: float,
    min_top1_similarity: float,
    min_top1_gap: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    columns = _columns(feature_names, ref_k)
    mal_ratio = knn_features[:, columns[f"knn{ref_k}_mal_ratio"]]
    weighted_mal_ratio = knn_features[:, columns[f"knn{ref_k}_weighted_mal_ratio"]]
    top1_label = np.rint(knn_features[:, columns["knn_top1_label"]]).astype(np.int64)
    top1_similarity = knn_features[:, columns["knn_top1_similarity"]]
    top1_gap = knn_features[:, columns["knn_top1_top2_gap"]]

    allow_0to1 = mode in ("both", "fn_only")
    allow_1to0 = mode in ("both", "fp_only")
    flip_0to1 = (
        allow_0to1
        & (base_predictions == 0)
        & (mal_ratio >= float(min_mal_for_0to1))
        & (weighted_mal_ratio >= float(min_weighted_agree))
        & (top1_label == 1)
        & (top1_similarity >= float(min_top1_similarity))
        & (top1_gap >= float(min_top1_gap))
    )
    flip_1to0 = (
        allow_1to0
        & (base_predictions == 1)
        & (mal_ratio <= float(max_mal_for_1to0))
        & (weighted_mal_ratio <= (1.0 - float(min_weighted_agree)))
        & (top1_label == 0)
        & (top1_similarity >= float(min_top1_similarity))
        & (top1_gap >= float(min_top1_gap))
    )
    flips = flip_0to1 | flip_1to0
    predictions = base_predictions.copy()
    predictions[flip_0to1] = 1
    predictions[flip_1to0] = 0
    return predictions, flips


def _build_knn_features(
    rows: Sequence[dict],
    *,
    model_path: Path,
    batch_size: int,
    similarity_memory_mib: float,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    payload, feature_config, checkpoint_config, reference = _load_stage2_payload(model_path)
    top_ks = [int(item) for item in payload["knn"]["top_ks"]]
    rows = _prepare_rows(rows)
    matrix, labels, _base_probs, kept_rows, counts = build_matrix(rows, checkpoint_config, feature_config)
    with_knn = append_frozen_knn_features(
        matrix,
        reference,
        top_ks,
        batch_size=int(batch_size),
        max_similarity_mib=float(similarity_memory_mib),
    )
    knn_features = with_knn[:, matrix.shape[1] :].astype(np.float32, copy=False)
    return kept_rows, labels, _base_predictions(kept_rows), knn_features, list(payload["knn"]["feature_names"]), counts


def _write_predictions(
    path: Path,
    rows: Sequence[dict],
    labels: np.ndarray,
    base_predictions: np.ndarray,
    predictions: np.ndarray,
    flips: np.ndarray,
) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "cache_path",
        "source_sha256",
        "label",
        "split",
        "sample_index",
        "baseline_prediction",
        "stage2_prob_malicious",
        "prediction",
        "support_gate_flip",
        "correct",
    ]
    with resolved.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row, label, base_prediction, prediction, flip in zip(rows, labels, base_predictions, predictions, flips):
            writer.writerow(
                {
                    "source_path": row.get("source_path", ""),
                    "cache_path": row.get("cache_path", ""),
                    "source_sha256": row.get("source_sha256", ""),
                    "label": int(label),
                    "split": row.get("split", ""),
                    "sample_index": row.get("sample_index", ""),
                    "baseline_prediction": int(base_prediction),
                    "stage2_prob_malicious": row.get("stage2_prob_malicious", row.get("prob_malicious", "")),
                    "prediction": int(prediction),
                    "support_gate_flip": "1" if bool(flip) else "0",
                    "correct": "True" if int(prediction) == int(label) else "False",
                }
            )


def _candidate_grid(args: argparse.Namespace):
    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        for min_mal in np.arange(args.min_mal_start, args.min_mal_stop + 1.0e-9, args.min_mal_step):
            for max_mal in np.arange(args.max_mal_start, args.max_mal_stop + 1.0e-9, args.max_mal_step):
                for min_weighted in np.arange(
                    args.min_weighted_start,
                    args.min_weighted_stop + 1.0e-9,
                    args.min_weighted_step,
                ):
                    for min_sim in np.arange(args.min_similarity_start, args.min_similarity_stop + 1.0e-9, args.min_similarity_step):
                        for min_gap in np.arange(args.min_gap_start, args.min_gap_stop + 1.0e-9, args.min_gap_step):
                            yield {
                                "mode": mode,
                                "ref_k": int(args.ref_k),
                                "min_mal_for_0to1": round(float(min_mal), 6),
                                "max_mal_for_1to0": round(float(max_mal), 6),
                                "min_weighted_agree": round(float(min_weighted), 6),
                                "min_top1_similarity": round(float(min_sim), 6),
                                "min_top1_gap": round(float(min_gap), 6),
                            }


def _passes_gate(row: dict, args: argparse.Namespace) -> bool:
    metric = row["metrics"]
    return (
        int(metric["errors"]) <= int(args.gate_max_errors)
        and int(metric["false_positive"]) <= int(args.gate_max_fp)
        and int(metric["false_negative"]) <= int(args.gate_max_fn)
        and float(metric["f1"]) > float(args.gate_min_f1)
    )


def select(args: argparse.Namespace) -> int:
    rows = read_prediction_rows(args.predictions, args.max_rows)
    kept_rows, labels, base_predictions, knn_features, feature_names, counts = _build_knn_features(
        rows,
        model_path=args.support_stage2_model,
        batch_size=args.knn_batch_size,
        similarity_memory_mib=args.knn_similarity_memory_mib,
    )
    baseline_metrics = _metric(labels, base_predictions)
    candidates = []
    for rule in _candidate_grid(args):
        predictions, flips = _apply_rule(base_predictions, knn_features, feature_names, **rule)
        metric = _metric(labels, predictions)
        row = {
            **rule,
            "metrics": metric,
            "changed_rows": int(np.count_nonzero(flips)),
            "changed_label0": int(np.count_nonzero(flips & (labels == 0))),
            "changed_label1": int(np.count_nonzero(flips & (labels == 1))),
        }
        row["passes_gate"] = _passes_gate(row, args)
        candidates.append(row)
    candidates.sort(
        key=lambda row: (
            bool(row["passes_gate"]),
            float(row["metrics"]["f1"]),
            -int(row["metrics"]["errors"]),
            -int(row["metrics"]["false_negative"]),
            -int(row["changed_rows"]),
        ),
        reverse=True,
    )
    selected = candidates[0]
    predictions, flips = _apply_rule(base_predictions, knn_features, feature_names, **{key: selected[key] for key in [
        "ref_k",
        "min_mal_for_0to1",
        "max_mal_for_1to0",
        "min_weighted_agree",
        "min_top1_similarity",
        "min_top1_gap",
        "mode",
    ]})
    output_json = resolve_path(args.output_json)
    output_csv = resolve_path(args.output_predictions_csv)
    _write_predictions(output_csv, kept_rows, labels, base_predictions, predictions, flips)
    report = {
        "schema": SCHEMA,
        "protocol": "Val-only rule selection over frozen Train-memory kNN support; no test used",
        "identity_feature_policy": "identity columns are alignment/cache metadata only, never model evidence",
        "predictions": str(resolve_path(args.predictions)),
        "support_stage2_model": str(resolve_path(args.support_stage2_model)),
        "output_predictions_csv": str(output_csv),
        "records": counts,
        "baseline_metrics": baseline_metrics,
        "selected_by_val": selected,
        "top_candidates": candidates[:50],
        "knn_feature_names": feature_names,
        "gate": {
            "max_errors": args.gate_max_errors,
            "max_fp": args.gate_max_fp,
            "max_fn": args.gate_max_fn,
            "min_f1_exclusive": args.gate_min_f1,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["baseline_metrics", "selected_by_val", "records"]}, indent=2, ensure_ascii=False))
    return 0


def evaluate(args: argparse.Namespace) -> int:
    selector = json.loads(resolve_path(args.selector_json).read_text(encoding="utf-8"))
    selected = selector["selected_by_val"]
    rows = read_prediction_rows(args.predictions, args.max_rows)
    kept_rows, labels, base_predictions, knn_features, feature_names, counts = _build_knn_features(
        rows,
        model_path=args.support_stage2_model,
        batch_size=args.knn_batch_size,
        similarity_memory_mib=args.knn_similarity_memory_mib,
    )
    rule = {key: selected[key] for key in [
        "ref_k",
        "min_mal_for_0to1",
        "max_mal_for_1to0",
        "min_weighted_agree",
        "min_top1_similarity",
        "min_top1_gap",
        "mode",
    ]}
    predictions, flips = _apply_rule(base_predictions, knn_features, feature_names, **rule)
    output_csv = resolve_path(args.output_predictions_csv)
    _write_predictions(output_csv, kept_rows, labels, base_predictions, predictions, flips)
    report = {
        "schema": f"{SCHEMA}_eval",
        "protocol": "Frozen kNN support gate evaluation; no fitting and no threshold sweep",
        "selector_json": str(resolve_path(args.selector_json)),
        "predictions": str(resolve_path(args.predictions)),
        "support_stage2_model": str(resolve_path(args.support_stage2_model)),
        "output_predictions_csv": str(output_csv),
        "records": counts,
        "selected_from_val": selected,
        "baseline_metrics": _metric(labels, base_predictions),
        "metrics": _metric(labels, predictions),
        "changed_rows": int(np.count_nonzero(flips)),
        "changed_label0": int(np.count_nonzero(flips & (labels == 0))),
        "changed_label1": int(np.count_nonzero(flips & (labels == 1))),
    }
    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--support-stage2-model", type=Path, required=True)
    parser.add_argument("--knn-batch-size", type=int, default=256)
    parser.add_argument("--knn-similarity-memory-mib", type=float, default=128.0)
    parser.add_argument("--max-rows", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Loop146 kNN prediction-support gate.")
    sub = parser.add_subparsers(dest="command", required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--predictions", type=Path, required=True)
    sel.add_argument("--output-json", type=Path, required=True)
    sel.add_argument("--output-predictions-csv", type=Path, required=True)
    sel.add_argument("--ref-k", type=int, default=25)
    sel.add_argument("--modes", default="both,fp_only,fn_only")
    sel.add_argument("--min-mal-start", type=float, default=0.70)
    sel.add_argument("--min-mal-stop", type=float, default=0.98)
    sel.add_argument("--min-mal-step", type=float, default=0.02)
    sel.add_argument("--max-mal-start", type=float, default=0.02)
    sel.add_argument("--max-mal-stop", type=float, default=0.30)
    sel.add_argument("--max-mal-step", type=float, default=0.02)
    sel.add_argument("--min-weighted-start", type=float, default=0.70)
    sel.add_argument("--min-weighted-stop", type=float, default=0.98)
    sel.add_argument("--min-weighted-step", type=float, default=0.04)
    sel.add_argument("--min-similarity-start", type=float, default=0.80)
    sel.add_argument("--min-similarity-stop", type=float, default=0.98)
    sel.add_argument("--min-similarity-step", type=float, default=0.04)
    sel.add_argument("--min-gap-start", type=float, default=0.0)
    sel.add_argument("--min-gap-stop", type=float, default=0.20)
    sel.add_argument("--min-gap-step", type=float, default=0.05)
    sel.add_argument("--gate-max-errors", type=int, default=169)
    sel.add_argument("--gate-max-fp", type=int, default=122)
    sel.add_argument("--gate-max-fn", type=int, default=57)
    sel.add_argument("--gate-min-f1", type=float, default=0.9910789932718664)
    _add_common(sel)
    sel.set_defaults(func=select)

    ev = sub.add_parser("eval")
    ev.add_argument("--selector-json", type=Path, required=True)
    ev.add_argument("--predictions", type=Path, required=True)
    ev.add_argument("--output-json", type=Path, required=True)
    ev.add_argument("--output-predictions-csv", type=Path, required=True)
    _add_common(ev)
    ev.set_defaults(func=evaluate)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
