#!/usr/bin/env python3
"""Verify and summarize the completed local Loop164 OOF diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP_DIR = PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164"
DEFAULT_REPORT = LOOP_DIR / "local_whole_file_oof_report.json"
DEFAULT_PREDICTIONS = LOOP_DIR / "local_whole_file_oof_predictions.jsonl"
DEFAULT_FOLDS = LOOP_DIR / "local_train_diagnostic_folds.jsonl"
DEFAULT_OUTPUT = LOOP_DIR / "local_whole_file_oof_analysis.json"
REPORT_SCHEMA = "axon_loop164_local_whole_file_oof_report_v1"
PREDICTION_SCHEMA = "axon_loop164_local_whole_file_oof_prediction_v1"
ANALYSIS_SCHEMA = "axon_loop164_local_whole_file_oof_analysis_v1"
CLAIM_SCOPE = "local_train_content_group_oof_diagnostic_not_production"
MAX_REPORT_BYTES = 4 * 1024 * 1024
MAX_ROWS_BYTES = 32 * 1024 * 1024
TARGET_F1_NUMERATOR = 9997
TARGET_F1_DENOMINATOR = 10000
PREDICTION_KEYS = {
    "schema",
    "loop_id",
    "claim_scope",
    "split_role",
    "train_row_index",
    "sample_index",
    "source_sha256",
    "content_component_id",
    "diagnostic_fold",
    "label",
    "whole_file_probability",
    "whole_file_score",
    "whole_file_uncertainty",
    "whole_file_missingness",
    "missing_reason",
    "fixed_threshold_prediction",
    "identity_metadata_not_model_features",
}


class OOFAnalysisError(ValueError):
    """The completed OOF artifacts do not satisfy their recorded contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise OOFAnalysisError(f"Duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite(value: str) -> object:
    raise OOFAnalysisError(f"Non-finite JSON value: {value}")


def _read_bounded(path: Path, max_bytes: int) -> bytes:
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise OOFAnalysisError(f"Artifact is too large: {path}")
    return raw


def _parse_object(raw: bytes, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OOFAnalysisError(f"Invalid JSON: {context}") from exc
    if not isinstance(payload, dict):
        raise OOFAnalysisError(f"Expected JSON object: {context}")
    return payload


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _quantile(values: list[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise OOFAnalysisError("Quantile input is invalid")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0 or len(labels) != len(scores):
        raise OOFAnalysisError("ROC AUC requires aligned rows from both labels")
    ordered = sorted(zip(scores, labels), key=lambda row: row[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _max_target_errors(positive_rows: int) -> int:
    budget = 2 * positive_rows * (TARGET_F1_DENOMINATOR - TARGET_F1_NUMERATOR)
    false_negative_cost = 2 * TARGET_F1_DENOMINATOR - TARGET_F1_NUMERATOR
    best = 0
    for false_negative in range(positive_rows + 1):
        remaining = budget - false_negative_cost * false_negative
        if remaining < 0:
            break
        false_positive = remaining // TARGET_F1_NUMERATOR
        best = max(best, false_negative + false_positive)
    return best


def _calibration(labels: list[int], scores: list[float]) -> dict[str, float]:
    epsilon = 1e-15
    brier = sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels)
    log_loss = -sum(
        label * math.log(min(max(score, epsilon), 1.0 - epsilon))
        + (1 - label) * math.log(min(max(1.0 - score, epsilon), 1.0 - epsilon))
        for label, score in zip(labels, scores)
    ) / len(labels)
    ece = 0.0
    for bin_index in range(10):
        lower = bin_index / 10.0
        upper = (bin_index + 1) / 10.0
        rows = [
            (label, score)
            for label, score in zip(labels, scores)
            if lower <= score < upper or (bin_index == 9 and score == 1.0)
        ]
        if rows:
            accuracy = sum(label for label, _ in rows) / len(rows)
            confidence = sum(score for _, score in rows) / len(rows)
            ece += len(rows) / len(labels) * abs(accuracy - confidence)
    return {"brier_score": brier, "log_loss": log_loss, "ece_10_equal_width": ece}


def _stratum(rows: list[tuple[int, int]]) -> dict[str, float | int]:
    errors = sum(label != prediction for label, prediction in rows)
    return {
        "rows": len(rows),
        "errors": errors,
        "error_rate": errors / len(rows) if rows else 0.0,
    }


def analyze(
    *,
    report_path: Path,
    predictions_path: Path,
    folds_path: Path,
) -> dict[str, Any]:
    report_path = report_path.resolve(strict=True)
    predictions_path = predictions_path.resolve(strict=True)
    folds_path = folds_path.resolve(strict=True)
    report_raw = _read_bounded(report_path, MAX_REPORT_BYTES)
    prediction_raw = _read_bounded(predictions_path, MAX_ROWS_BYTES)
    folds_raw = _read_bounded(folds_path, MAX_ROWS_BYTES)
    report = _parse_object(report_raw, "OOF report")
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status") != "completed"
        or report.get("claim_scope") != CLAIM_SCOPE
        or report.get("decision") != "local_supported_only_oof_observed_not_promotable"
        or report.get("fatal") is not None
    ):
        raise OOFAnalysisError("OOF report completion or scope is invalid")
    if report.get("forbidden_outputs") != {
        "checkpoint_written": False,
        "model_state_serialized": False,
        "threshold_sweep": False,
        "val_test_or_full_predictions_read": False,
    }:
        raise OOFAnalysisError("OOF forbidden-output record drifted")
    prediction_binding = report.get("predictions")
    folds_binding = report.get("bindings", {}).get("folds")
    if not isinstance(prediction_binding, dict) or (
        Path(str(prediction_binding.get("path") or "")).resolve(strict=True) != predictions_path
        or prediction_binding.get("sha256") != _sha256(prediction_raw)
        or prediction_binding.get("record_count") != 20000
        or prediction_binding.get("schema") != PREDICTION_SCHEMA
    ):
        raise OOFAnalysisError("OOF prediction binding is invalid")
    if not isinstance(folds_binding, dict) or (
        Path(str(folds_binding.get("path") or "")).resolve(strict=True) != folds_path
        or folds_binding.get("sha256") != _sha256(folds_raw)
    ):
        raise OOFAnalysisError("OOF folds binding is invalid")

    prediction_lines = prediction_raw.splitlines()
    fold_lines = folds_raw.splitlines()
    if len(prediction_lines) != 20000 or len(fold_lines) != 20000:
        raise OOFAnalysisError("OOF row count drifted")
    predictions = [_parse_object(line, "prediction row") for line in prediction_lines]
    folds = [_parse_object(line, "fold row") for line in fold_lines]
    fold_by_index = {int(row["train_row_index"]): row for row in folds}
    if set(fold_by_index) != set(range(20000)):
        raise OOFAnalysisError("Fold train-row coverage drifted")

    supported: list[dict[str, Any]] = []
    missing = Counter()
    for expected_index, row in enumerate(predictions):
        if set(row) != PREDICTION_KEYS or (
            row.get("schema") != PREDICTION_SCHEMA
            or row.get("claim_scope") != CLAIM_SCOPE
            or row.get("split_role") != "train"
            or row.get("train_row_index") != expected_index
            or row.get("sample_index") != expected_index
            or row.get("label") not in {0, 1}
        ):
            raise OOFAnalysisError("Prediction row identity or schema drifted")
        fold_row = fold_by_index[expected_index]
        if (
            row.get("source_sha256") != fold_row.get("source_sha256")
            or row.get("content_component_id") != fold_row.get("content_component_id")
            or row.get("diagnostic_fold") != fold_row.get("diagnostic_fold")
            or row.get("label") != fold_row.get("label")
        ):
            raise OOFAnalysisError("Prediction/fold identity binding drifted")
        if row.get("whole_file_missingness") == 1:
            if (
                row.get("whole_file_probability") is not None
                or row.get("fixed_threshold_prediction") is not None
                or row.get("whole_file_score") != 0.5
                or row.get("whole_file_uncertainty") != 1.0
                or row.get("missing_reason") not in {"oversize", "read_failure"}
            ):
                raise OOFAnalysisError("Missing prediction semantics drifted")
            missing[str(row["missing_reason"])] += 1
            continue
        score = row.get("whole_file_score")
        probability = row.get("whole_file_probability")
        prediction = row.get("fixed_threshold_prediction")
        if (
            row.get("whole_file_missingness") != 0
            or row.get("missing_reason") is not None
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            or probability != score
            or prediction != int(float(score) >= 0.5)
        ):
            raise OOFAnalysisError("Supported prediction semantics drifted")
        supported.append(row)

    labels = [int(row["label"]) for row in supported]
    scores = [float(row["whole_file_score"]) for row in supported]
    decisions = [int(row["fixed_threshold_prediction"]) for row in supported]
    true_positive = sum(label == 1 and decision == 1 for label, decision in zip(labels, decisions))
    true_negative = sum(label == 0 and decision == 0 for label, decision in zip(labels, decisions))
    false_positive = sum(label == 0 and decision == 1 for label, decision in zip(labels, decisions))
    false_negative = sum(label == 1 and decision == 0 for label, decision in zip(labels, decisions))
    errors = false_positive + false_negative
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    recorded_metrics = report.get("oof", {}).get("supported_only_fixed_threshold_metrics")
    if not isinstance(recorded_metrics, dict) or any(
        recorded_metrics.get(key) != value
        for key, value in {
            "samples": len(supported),
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "errors": errors,
            "f1": f1,
        }.items()
    ):
        raise OOFAnalysisError("Recorded supported metrics drifted")

    label_scores = {
        str(label): [score for row, score in zip(supported, scores) if int(row["label"]) == label]
        for label in (0, 1)
    }
    quantiles = {
        label: {
            str(quantile): _quantile(values, quantile)
            for quantile in (0.01, 0.1, 0.5, 0.9, 0.99)
        }
        for label, values in label_scores.items()
    }
    high_confidence_false_positive = sum(
        label == 0 and score >= 0.9 for label, score in zip(labels, scores)
    )
    high_confidence_false_negative = sum(
        label == 1 and score <= 0.1 for label, score in zip(labels, scores)
    )
    fold_f1 = [
        float(row["supported_only_fixed_threshold_metrics"]["f1"])
        for row in report.get("fold_reports", [])
    ]
    if len(fold_f1) != 5:
        raise OOFAnalysisError("Fold report count drifted")

    component_strata: dict[str, list[tuple[int, int]]] = {
        "singleton": [],
        "non_singleton": [],
    }
    size_strata: dict[str, list[tuple[int, int]]] = {
        "le_64_kib": [],
        "64_kib_to_256_kib": [],
        "256_kib_to_1_mib": [],
        "1_mib_to_8_mib": [],
    }
    for row, label, decision in zip(supported, labels, decisions):
        fold_row = fold_by_index[int(row["train_row_index"])]
        component_key = "singleton" if int(fold_row["content_component_size"]) == 1 else "non_singleton"
        component_strata[component_key].append((label, decision))
        size = int(fold_row["source_size_bytes"])
        if size <= 64 * 1024:
            size_key = "le_64_kib"
        elif size <= 256 * 1024:
            size_key = "64_kib_to_256_kib"
        elif size <= 1024 * 1024:
            size_key = "256_kib_to_1_mib"
        else:
            size_key = "1_mib_to_8_mib"
        size_strata[size_key].append((label, decision))

    positive_rows = sum(labels)
    max_target_errors = _max_target_errors(positive_rows)
    conservative_metrics = report["oof"][
        "canonical_denominator_conservative_all_missing_wrong_metrics"
    ]
    return {
        "schema": ANALYSIS_SCHEMA,
        "loop_id": "loop164_whole_file_residual_expert",
        "claim_scope": "posthoc_local_train_oof_description_not_model_selection_or_promotion",
        "inputs": {
            "report": {"path": str(report_path), "sha256": _sha256(report_raw)},
            "predictions": {"path": str(predictions_path), "sha256": _sha256(prediction_raw)},
            "folds": {"path": str(folds_path), "sha256": _sha256(folds_raw)},
        },
        "fixed_threshold_result": {
            "threshold": 0.5,
            "threshold_selection_performed": False,
            "supported_rows": len(supported),
            "missing_rows": sum(missing.values()),
            "coverage": len(supported) / 20000,
            "f1": f1,
            "errors": errors,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "conservative_full_denominator_f1": conservative_metrics["f1"],
            "conservative_full_denominator_errors": conservative_metrics["errors"],
        },
        "posthoc_descriptive_metrics": {
            "roc_auc": _roc_auc(labels, scores),
            **_calibration(labels, scores),
            "score_quantiles_by_label": quantiles,
            "high_confidence_false_positive_ge_0_9": high_confidence_false_positive,
            "high_confidence_false_negative_le_0_1": high_confidence_false_negative,
            "fold_f1": fold_f1,
            "fold_f1_mean": statistics.fmean(fold_f1),
            "fold_f1_population_stddev": statistics.pstdev(fold_f1),
            "fold_f1_min": min(fold_f1),
            "fold_f1_max": max(fold_f1),
            "component_size_strata": {
                name: _stratum(rows) for name, rows in component_strata.items()
            },
            "file_size_strata": {name: _stratum(rows) for name, rows in size_strata.items()},
        },
        "target_gap": {
            "target_f1": TARGET_F1_NUMERATOR / TARGET_F1_DENOMINATOR,
            "supported_positive_rows": positive_rows,
            "maximum_supported_errors_at_target": max_target_errors,
            "observed_supported_errors": errors,
            "minimum_error_reduction_required": errors - max_target_errors,
            "standalone_target_met": f1 >= TARGET_F1_NUMERATOR / TARGET_F1_DENOMINATOR,
        },
        "limitations": [
            "one_seed_one_epoch_local_content_group_diagnostic",
            "three_oversized_lsh_buckets_were_not_component_resolved",
            "no_authoritative_family_campaign_source_or_first_seen_time",
            "no_decision_aligned_loop151_train_oof_for_complementarity",
            "posthoc_auc_calibration_and_slice_metrics_are_descriptive_only",
        ],
        "decision": "stop_current_standalone_scale_preserve_oof_for_future_complementarity_audit_only",
        "next_gate": (
            "Do not add seeds, epochs, threshold search, or heldout access to this standalone lineage. "
            "First reconstruct decision-aligned Loop151 Train OOF; then run one frozen cross-fitted "
            "complementarity gate. Close Loop164 if it cannot repair champion errors without excess breaks."
        ),
        "ready_for": {
            "more_standalone_seeds_or_epochs": False,
            "threshold_selection": False,
            "candidate_promotion": False,
            "val_test_or_full_access": False,
            "future_train_oof_complementarity_audit": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze completed local Loop164 OOF artifacts.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = analyze(
        report_path=args.report,
        predictions_path=args.predictions,
        folds_path=args.folds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
