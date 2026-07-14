#!/usr/bin/env python3
"""Run the frozen Train-only Loop69 x Loop164 surrogate complementarity gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from analyze_loop164_local_oof_result import analyze as analyze_loop164

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP69_DIR = (
    PROJECT_ROOT
    / "reports"
    / "random_20w_split"
    / "loop69_nested_oof_override_full_train"
)
LOOP164_DIR = PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164"
DEFAULT_LOOP69_REPORT = LOOP69_DIR / "loop69_nested_oof_override_report.json"
DEFAULT_LOOP69_READINESS = LOOP69_DIR / "loop68_readiness_on_loop69_full_train.json"
DEFAULT_LOOP69_PREDICTIONS = LOOP69_DIR / "loop69_nested_oof_override_train_predictions.csv"
DEFAULT_LOOP164_REPORT = LOOP164_DIR / "local_whole_file_oof_report.json"
DEFAULT_LOOP164_PREDICTIONS = LOOP164_DIR / "local_whole_file_oof_predictions.jsonl"
DEFAULT_LOOP164_FOLDS = LOOP164_DIR / "local_train_diagnostic_folds.jsonl"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "roadmap_9997"
    / "loop165"
    / "loop69_loop164_surrogate_complementarity.json"
)

SCHEMA = "axon_loop165_loop69_loop164_surrogate_complementarity_v1"
CLAIM_SCOPE = "local_train_cross_snapshot_surrogate_oracle_not_loop151_equivalent"
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_ROWS_BYTES = 32 * 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

CANONICAL_SHA256 = {
    "loop69_report": "0eb98433fc756e568e7bc0e66fd1e43c23788fe37bfd026168007972afc4ebb5",
    "loop69_readiness": "78f1452dd8157bdab6a2a2cf58bb1d4566a222324eba3f8edd45cafcaeb76ebd",
    "loop69_predictions": "9f942d68d523a0663ff1f5e9e03e6a0e47feffea34050cdfde96728d1e524a9a",
    "loop164_report": "da55531d39b628a2a02ec008451b7ad0455f6876cabd91dcb8c56f7e18c3e07f",
    "loop164_predictions": "4f706788d812987714ebd9f717b77f75b10997309dbe7991c083b9928ad3d4df",
    "loop164_folds": "00a31a1bd86d7b887447f3e86e5e753ebcaaee45be74311199332e073a3880a5",
}

LOOP69_COLUMNS = (
    "source_path",
    "cache_path",
    "source_sha256",
    "label",
    "split",
    "sample_index",
    "oof_fold",
    "base_oof_prob_malicious",
    "candidate_oof_prob_malicious",
    "allow_oof_prob",
    "final_oof_prob_malicious",
    "final_oof_prediction",
    "oof_override_flag",
    "possible_override_flag",
    "candidate_threshold",
    "allow_threshold",
    "selected_candidate",
    "selected_override_model",
)

PREREGISTERED_GATE = {
    "minimum_supported_disagreements": 100,
    "minimum_base_error_repairs": 30,
    "minimum_blind_switch_precision": 0.80,
    "minimum_net_error_reduction": 1,
}


class SurrogateAuditError(ValueError):
    """An input or alignment contract is invalid."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise SurrogateAuditError(f"Duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite(value: str) -> object:
    raise SurrogateAuditError(f"Non-finite JSON value: {value}")


def _read_bounded(path: Path, max_bytes: int) -> bytes:
    path = path.resolve(strict=True)
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise SurrogateAuditError(f"Artifact is too large: {path}")
    return raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _parse_json_object(raw: bytes, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SurrogateAuditError(f"Invalid JSON: {context}") from exc
    if not isinstance(payload, dict):
        raise SurrogateAuditError(f"Expected JSON object: {context}")
    return payload


def _bind_raw(raw: bytes, expected_sha256: str, context: str) -> str:
    observed = _sha256(raw)
    if observed != expected_sha256:
        raise SurrogateAuditError(
            f"{context} SHA-256 drifted: expected {expected_sha256}, observed {observed}"
        )
    return observed


def _parse_int(value: str, context: str, allowed: set[int] | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SurrogateAuditError(f"Invalid integer for {context}: {value!r}") from exc
    if allowed is not None and parsed not in allowed:
        raise SurrogateAuditError(f"Unexpected integer for {context}: {parsed}")
    return parsed


def _parse_probability(value: str, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SurrogateAuditError(f"Invalid probability for {context}: {value!r}") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise SurrogateAuditError(f"Out-of-range probability for {context}: {parsed}")
    return parsed


def _parse_bool(value: str, context: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise SurrogateAuditError(f"Invalid boolean for {context}: {value!r}")


def _metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, int | float]:
    if not labels or len(labels) != len(predictions):
        raise SurrogateAuditError("Metrics require non-empty aligned labels and predictions")
    true_positive = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    true_negative = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    false_positive = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    false_negative = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-15)
    return {
        "samples": len(labels),
        "accuracy": (true_positive + true_negative) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "errors": false_positive + false_negative,
    }


def _load_loop69(
    *,
    report_path: Path,
    readiness_path: Path,
    predictions_path: Path,
    expected_sha256: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report_raw = _read_bounded(report_path, MAX_JSON_BYTES)
    readiness_raw = _read_bounded(readiness_path, MAX_JSON_BYTES)
    predictions_raw = _read_bounded(predictions_path, MAX_ROWS_BYTES)
    bindings = {
        "report": _bind_raw(report_raw, expected_sha256["loop69_report"], "Loop69 report"),
        "readiness": _bind_raw(
            readiness_raw,
            expected_sha256["loop69_readiness"],
            "Loop69 readiness",
        ),
        "predictions": _bind_raw(
            predictions_raw,
            expected_sha256["loop69_predictions"],
            "Loop69 predictions",
        ),
    }
    report = _parse_json_object(report_raw, "Loop69 report")
    readiness = _parse_json_object(readiness_raw, "Loop69 readiness")
    if (
        report.get("schema") != "axon_loop69_nested_oof_override_v1"
        or report.get("outer_folds") != 5
        or report.get("inner_folds") != 5
        or report.get("base_model") != "hgb_lr0.06_leaf31_l2_0"
        or report.get("candidate_model") != "extra_trees_300_leaf1"
        or report.get("override_model") != "override_logreg_balanced_c1"
        or report.get("base_threshold") != 0.5
        or report.get("records")
        != {"total": 20000, "kept": 20000, "skipped_missing_cache": 0}
    ):
        raise SurrogateAuditError("Loop69 frozen recipe or record contract drifted")
    if (
        readiness.get("schema") != "axon_loop68_residual_oof_readiness_v1"
        or readiness.get("expected_train_rows") != 20000
        or readiness.get("expected_val_rows") != 0
        or readiness.get("candidate_count") != 1
        or readiness.get("ready_candidate_count") != 1
        or readiness.get("overall_decision") != "third_layer_residual_training_allowed"
    ):
        raise SurrogateAuditError("Loop69 readiness contract drifted")

    try:
        decoded = predictions_raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SurrogateAuditError("Loop69 predictions are not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(decoded, newline=""))
    if tuple(reader.fieldnames or ()) != LOOP69_COLUMNS:
        raise SurrogateAuditError("Loop69 prediction columns drifted")
    raw_rows = list(reader)
    if len(raw_rows) != 20000:
        raise SurrogateAuditError("Loop69 prediction row count drifted")

    rows: list[dict[str, Any]] = []
    seen_sha256: set[str] = set()
    seen_indices: set[int] = set()
    for row_number, row in enumerate(raw_rows):
        if tuple(row) != LOOP69_COLUMNS:
            raise SurrogateAuditError(f"Loop69 row {row_number} has an invalid CSV shape")
        source_sha256 = str(row["source_sha256"]).lower()
        if not SHA256_PATTERN.fullmatch(source_sha256) or source_sha256 in seen_sha256:
            raise SurrogateAuditError(f"Loop69 row {row_number} has an invalid or duplicate SHA-256")
        sample_index = _parse_int(row["sample_index"], "Loop69 sample_index")
        if sample_index in seen_indices:
            raise SurrogateAuditError(f"Loop69 sample_index is duplicated: {sample_index}")
        label = _parse_int(row["label"], "Loop69 label", {0, 1})
        prediction = _parse_int(row["final_oof_prediction"], "Loop69 prediction", {0, 1})
        oof_fold = _parse_int(row["oof_fold"], "Loop69 OOF fold", {1, 2, 3, 4, 5})
        for field in (
            "base_oof_prob_malicious",
            "candidate_oof_prob_malicious",
            "allow_oof_prob",
            "final_oof_prob_malicious",
            "candidate_threshold",
            "allow_threshold",
        ):
            _parse_probability(row[field], f"Loop69 {field}")
        _parse_bool(row["oof_override_flag"], "Loop69 override flag")
        _parse_bool(row["possible_override_flag"], "Loop69 possible override flag")
        if (
            row["split"] != "train"
            or row["selected_candidate"] != "hgb_lr0.06_leaf31_l2_0 -> extra_trees_300_leaf1"
            or row["selected_override_model"] != "override_logreg_balanced_c1"
        ):
            raise SurrogateAuditError(f"Loop69 row {row_number} recipe identity drifted")
        seen_sha256.add(source_sha256)
        seen_indices.add(sample_index)
        rows.append(
            {
                "source_sha256": source_sha256,
                "sample_index": sample_index,
                "label": label,
                "prediction": prediction,
                "oof_fold": oof_fold,
            }
        )
    if seen_indices != set(range(20000)):
        raise SurrogateAuditError("Loop69 sample_index coverage drifted")
    rows.sort(key=lambda row: int(row["sample_index"]))

    observed_metrics = _metrics(
        [int(row["label"]) for row in rows],
        [int(row["prediction"]) for row in rows],
    )
    recorded_metrics = report.get("metrics")
    if not isinstance(recorded_metrics, dict):
        raise SurrogateAuditError("Loop69 report metrics are missing")
    for key in (
        "samples",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
        "errors",
    ):
        if recorded_metrics.get(key) != observed_metrics[key]:
            raise SurrogateAuditError(f"Loop69 recorded metric drifted: {key}")
    if not math.isclose(
        float(recorded_metrics.get("f1", math.nan)),
        float(observed_metrics["f1"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise SurrogateAuditError("Loop69 recorded metric drifted: f1")
    return rows, {"sha256": bindings, "metrics": observed_metrics}


def _load_loop164(
    *,
    report_path: Path,
    predictions_path: Path,
    folds_path: Path,
    expected_sha256: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report_raw = _read_bounded(report_path, MAX_JSON_BYTES)
    predictions_raw = _read_bounded(predictions_path, MAX_ROWS_BYTES)
    folds_raw = _read_bounded(folds_path, MAX_ROWS_BYTES)
    bindings = {
        "report": _bind_raw(report_raw, expected_sha256["loop164_report"], "Loop164 report"),
        "predictions": _bind_raw(
            predictions_raw,
            expected_sha256["loop164_predictions"],
            "Loop164 predictions",
        ),
        "folds": _bind_raw(folds_raw, expected_sha256["loop164_folds"], "Loop164 folds"),
    }
    analysis = analyze_loop164(
        report_path=report_path,
        predictions_path=predictions_path,
        folds_path=folds_path,
    )
    rows = [
        _parse_json_object(line, f"Loop164 prediction row {index}")
        for index, line in enumerate(predictions_raw.splitlines())
    ]
    if len({str(row["source_sha256"]) for row in rows}) != len(rows):
        raise SurrogateAuditError("Loop164 source SHA-256 values are not unique")
    return rows, {"sha256": bindings, "analysis": analysis}


def _replacement_receipt(
    loop69_rows: Sequence[dict[str, Any]],
    loop164_rows: Sequence[dict[str, Any]],
    common_sha256: set[str],
) -> list[dict[str, Any]]:
    by_loop69_index = {int(row["sample_index"]): row for row in loop69_rows}
    by_loop164_index = {int(row["sample_index"]): row for row in loop164_rows}
    replacements: list[dict[str, Any]] = []
    for sample_index in range(20000):
        old_row = by_loop69_index[sample_index]
        current_row = by_loop164_index[sample_index]
        old_sha256 = str(old_row["source_sha256"])
        current_sha256 = str(current_row["source_sha256"])
        if old_sha256 == current_sha256:
            if old_sha256 not in common_sha256:
                raise SurrogateAuditError("Equal indexed SHA-256 is absent from the common set")
            continue
        replacements.append(
            {
                "sample_index": sample_index,
                "loop69_source_sha256": old_sha256,
                "loop69_label": int(old_row["label"]),
                "loop164_source_sha256": current_sha256,
                "loop164_label": int(current_row["label"]),
                "loop164_missingness": int(current_row["whole_file_missingness"]),
                "loop164_missing_reason": current_row["missing_reason"],
            }
        )
    return replacements


def _build_complementarity(
    loop69_rows: Sequence[dict[str, Any]],
    loop164_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_loop69 = {str(row["source_sha256"]): row for row in loop69_rows}
    by_loop164 = {str(row["source_sha256"]): row for row in loop164_rows}
    common_sha256 = set(by_loop69) & set(by_loop164)
    only_loop69 = set(by_loop69) - set(by_loop164)
    only_loop164 = set(by_loop164) - set(by_loop69)
    if len(by_loop69) != len(loop69_rows) or len(by_loop164) != len(loop164_rows):
        raise SurrogateAuditError("Ambiguous duplicate SHA-256 alignment")

    # SHA 是唯一允许的跨快照连接键；sample_index 只用于证明共同样本没有错位。
    common_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for source_sha256 in sorted(common_sha256, key=lambda value: int(by_loop69[value]["sample_index"])):
        loop69_row = by_loop69[source_sha256]
        loop164_row = by_loop164[source_sha256]
        if int(loop69_row["label"]) != int(loop164_row["label"]):
            raise SurrogateAuditError(f"Label mismatch for common SHA-256: {source_sha256}")
        if int(loop69_row["sample_index"]) != int(loop164_row["sample_index"]):
            raise SurrogateAuditError(f"Index mismatch for common SHA-256: {source_sha256}")
        common_rows.append((loop69_row, loop164_row))

    replacements = _replacement_receipt(loop69_rows, loop164_rows, common_sha256)
    if len(replacements) != len(only_loop69) or len(replacements) != len(only_loop164):
        raise SurrogateAuditError("Snapshot drift is not an exact same-index replacement set")

    supported = [pair for pair in common_rows if int(pair[1]["whole_file_missingness"]) == 0]
    missing = [pair for pair in common_rows if int(pair[1]["whole_file_missingness"]) == 1]
    normalized_fold_matches = sum(
        int(loop69_row["oof_fold"]) - 1 == int(loop164_row["diagnostic_fold"])
        for loop69_row, loop164_row in common_rows
    )
    component_rows: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for pair in common_rows:
        component_rows.setdefault(str(pair[1]["content_component_id"]), []).append(pair)
    non_singleton_components = [rows for rows in component_rows.values() if len(rows) > 1]
    cross_loop69_fold_components = [
        rows
        for rows in non_singleton_components
        if len({int(loop69_row["oof_fold"]) for loop69_row, _ in rows}) > 1
    ]
    repairs = sum(
        int(loop69_row["prediction"]) != int(loop69_row["label"])
        and int(loop164_row["fixed_threshold_prediction"]) == int(loop69_row["label"])
        for loop69_row, loop164_row in supported
    )
    breaks = sum(
        int(loop69_row["prediction"]) == int(loop69_row["label"])
        and int(loop164_row["fixed_threshold_prediction"]) != int(loop69_row["label"])
        for loop69_row, loop164_row in supported
    )
    both_wrong = sum(
        int(loop69_row["prediction"]) != int(loop69_row["label"])
        and int(loop164_row["fixed_threshold_prediction"]) != int(loop69_row["label"])
        for loop69_row, loop164_row in supported
    )
    both_correct = len(supported) - repairs - breaks - both_wrong
    disagreement_support = repairs + breaks
    blind_switch_precision = repairs / disagreement_support if disagreement_support else 0.0
    net_error_reduction = repairs - breaks

    labels = [int(loop69_row["label"]) for loop69_row, _ in common_rows]
    loop69_predictions = [int(loop69_row["prediction"]) for loop69_row, _ in common_rows]
    blind_switch_predictions: list[int] = []
    oracle_predictions: list[int] = []
    missing_base_errors = 0
    for loop69_row, loop164_row in common_rows:
        label = int(loop69_row["label"])
        base_prediction = int(loop69_row["prediction"])
        if int(loop164_row["whole_file_missingness"]) == 1:
            blind_switch_predictions.append(base_prediction)
            oracle_predictions.append(base_prediction)
            missing_base_errors += base_prediction != label
            continue
        whole_file_prediction = int(loop164_row["fixed_threshold_prediction"])
        blind_switch_predictions.append(whole_file_prediction)
        # 这个 oracle 使用真值选择专家，只描述理论上限，绝不是可部署规则。
        oracle_predictions.append(
            label
            if base_prediction == label or whole_file_prediction == label
            else base_prediction
        )

    gate_checks = {
        "supported_disagreements": disagreement_support
        >= PREREGISTERED_GATE["minimum_supported_disagreements"],
        "base_error_repairs": repairs >= PREREGISTERED_GATE["minimum_base_error_repairs"],
        "blind_switch_precision": blind_switch_precision
        >= PREREGISTERED_GATE["minimum_blind_switch_precision"],
        "net_error_reduction": net_error_reduction
        >= PREREGISTERED_GATE["minimum_net_error_reduction"],
    }
    gate_passed = all(gate_checks.values())
    by_loop164_fold: dict[str, dict[str, int | float]] = {}
    for diagnostic_fold in range(5):
        fold_rows = [
            pair for pair in supported if int(pair[1]["diagnostic_fold"]) == diagnostic_fold
        ]
        fold_repairs = sum(
            int(loop69_row["prediction"]) != int(loop69_row["label"])
            and int(loop164_row["fixed_threshold_prediction"]) == int(loop69_row["label"])
            for loop69_row, loop164_row in fold_rows
        )
        fold_breaks = sum(
            int(loop69_row["prediction"]) == int(loop69_row["label"])
            and int(loop164_row["fixed_threshold_prediction"]) != int(loop69_row["label"])
            for loop69_row, loop164_row in fold_rows
        )
        by_loop164_fold[str(diagnostic_fold)] = {
            "supported_rows": len(fold_rows),
            "repairs": fold_repairs,
            "breaks": fold_breaks,
            "decision_changes": fold_repairs + fold_breaks,
            "blind_switch_precision": fold_repairs / max(fold_repairs + fold_breaks, 1),
            "net_error_reduction": fold_repairs - fold_breaks,
        }
    return {
        "alignment_receipt": {
            "loop69_rows": len(loop69_rows),
            "loop164_rows": len(loop164_rows),
            "common_sha256_rows": len(common_rows),
            "loop69_only_rows": len(only_loop69),
            "loop164_only_rows": len(only_loop164),
            "same_index_replacements": replacements,
            "join_key": "source_sha256",
            "sample_index_role": "same-row drift audit only; never a fallback join key",
            "silent_row_drop": False,
        },
        "partition_compatibility": {
            "loop69_partition": "random_stratified_five_fold_seed_69_one_based",
            "loop164_partition": "content_component_five_fold_seed_164_zero_based",
            "common_rows_with_same_numeric_fold_after_normalization": normalized_fold_matches,
            "common_rows_with_different_fold_after_normalization": len(common_rows)
            - normalized_fold_matches,
            "non_singleton_loop164_components_on_common_rows": len(non_singleton_components),
            "non_singleton_components_crossing_loop69_folds": len(
                cross_loop69_fold_components
            ),
            "rows_in_crossing_components": sum(len(rows) for rows in cross_loop69_fold_components),
            "shared_outer_partition": False,
        },
        "coverage": {
            "common_rows": len(common_rows),
            "supported_common_rows": len(supported),
            "missing_common_rows": len(missing),
            "supported_common_fraction": len(supported) / len(common_rows),
            "loop164_total_missing_rows": sum(
                int(row["whole_file_missingness"]) == 1 for row in loop164_rows
            ),
            "unmatched_loop164_rows_without_loop69_baseline": len(only_loop164),
        },
        "overlap": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "base_error_repairs": repairs,
            "base_correct_breaks": breaks,
            "supported_disagreements": disagreement_support,
            "blind_switch_precision": blind_switch_precision,
            "net_error_reduction": net_error_reduction,
            "repair_rate_over_all_common_base_errors": repairs
            / max(sum(label != prediction for label, prediction in zip(labels, loop69_predictions)), 1),
            "missing_common_base_errors_retained": missing_base_errors,
            "by_loop164_diagnostic_fold": by_loop164_fold,
        },
        "metrics": {
            "loop69_common_denominator": _metrics(labels, loop69_predictions),
            "blind_switch_to_loop164_on_supported_disagreement_retain_base_on_missing": _metrics(
                labels,
                blind_switch_predictions,
            ),
            "unattainable_oracle_choose_correct_expert_retain_base_on_missing": _metrics(
                labels,
                oracle_predictions,
            ),
            "oracle_current_snapshot_conservative_errors_if_all_unmatched_are_wrong": (
                _metrics(labels, oracle_predictions)["errors"] + len(only_loop164)
            ),
        },
        "surrogate_cost_fuse": {
            "thresholds": dict(PREREGISTERED_GATE),
            "checks": gate_checks,
            "passed": gate_passed,
            "evidence_role": "surrogate_negative_supporting_evidence_not_formal_loop151_gate",
        },
    }


def audit(
    *,
    loop69_report: Path = DEFAULT_LOOP69_REPORT,
    loop69_readiness: Path = DEFAULT_LOOP69_READINESS,
    loop69_predictions: Path = DEFAULT_LOOP69_PREDICTIONS,
    loop164_report: Path = DEFAULT_LOOP164_REPORT,
    loop164_predictions: Path = DEFAULT_LOOP164_PREDICTIONS,
    loop164_folds: Path = DEFAULT_LOOP164_FOLDS,
    expected_sha256: dict[str, str] = CANONICAL_SHA256,
) -> dict[str, Any]:
    required_bindings = set(CANONICAL_SHA256)
    if set(expected_sha256) != required_bindings:
        raise SurrogateAuditError("Expected SHA-256 binding set is incomplete")
    loop69_rows, loop69_evidence = _load_loop69(
        report_path=loop69_report,
        readiness_path=loop69_readiness,
        predictions_path=loop69_predictions,
        expected_sha256=expected_sha256,
    )
    loop164_rows, loop164_evidence = _load_loop164(
        report_path=loop164_report,
        predictions_path=loop164_predictions,
        folds_path=loop164_folds,
        expected_sha256=expected_sha256,
    )
    result = _build_complementarity(loop69_rows, loop164_rows)
    gate_passed = bool(result["surrogate_cost_fuse"]["passed"])
    return {
        "schema": SCHEMA,
        "loop_id": "loop165_loop69_loop164_surrogate_complementarity",
        "claim_scope": CLAIM_SCOPE,
        "protocol": {
            "training_performed": False,
            "threshold_sweep_performed": False,
            "val_test_or_full_access": False,
            "identity_fields_used_as_model_features": False,
            "frozen_loop69_hard_predictions": True,
            "frozen_loop164_threshold": 0.5,
        },
        "inputs": {
            "loop69": {
                "report": str(loop69_report.resolve()),
                "readiness": str(loop69_readiness.resolve()),
                "predictions": str(loop69_predictions.resolve()),
                **loop69_evidence,
            },
            "loop164": {
                "report": str(loop164_report.resolve()),
                "predictions": str(loop164_predictions.resolve()),
                "folds": str(loop164_folds.resolve()),
                "sha256": loop164_evidence["sha256"],
                "validated_claim_scope": loop164_evidence["analysis"]["claim_scope"],
            },
        },
        **result,
        "limitations": [
            "loop69_is_a_strong_train_only_surrogate_not_loop151_equivalent",
            "loop69_and_loop164_do_not_share_one_joint_nested_outer_partition",
            "loop164_content_components_cross_loop69_random_folds",
            "four_current_snapshot_rows_have_no_loop69_baseline",
            "oracle_uses_labels_and_is_not_a_learned_or_deployable_router",
            "no_val_test_test10k_or_full_test_evidence",
        ],
        "formal_loop151_complementarity_gate": {
            "status": "not_run",
            "decision": "blocked_wrong_base_lineage_and_fold_scope",
            "blockers": [
                "loop69_is_loop61_style_not_decision_aligned_loop151",
                "loop69_random_folds_do_not_match_loop164_content_component_folds",
                "four_current_snapshot_rows_lack_loop69_baseline",
                "loop69_report_lacks_complete_recipe_input_and_output_sha_provenance",
            ],
        },
        "decision": (
            "surrogate_supports_exact_loop151_investment_but_formal_gate_remains_blocked"
            if gate_passed
            else "park_current_loop164_recipe_surrogate_negative_exact_loop151_gate_not_run"
        ),
        "loop151_exact_oof": {
            "authorized_for_loop164": gate_passed,
            "still_required_before_any_future_learned_multi_expert_router": True,
        },
        "current_loop164_recipe": {
            "compute_status": "parked_no_further_standalone_or_exact_oof_spend",
            "formal_lineage_closed": False,
            "resurrection_condition": (
                "Only a materially stronger independently validated whole-file expert may restart a new "
                "proposal; this surrogate does not adjudicate formal Loop151 complementarity."
            ),
        },
        "target_status": {
            "full_test_f1_target": 0.9997,
            "target_achieved": False,
            "promotion_evidence": False,
        },
        "next_action": (
            "Preserve Loop151 as champion; park the current Loop164 recipe without claiming a formal "
            "Loop151 gate. Build independent "
            "foundation, structural, behavior, and label-quality evidence before paying for a "
            "new decision-aligned multi-expert OOF router."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen Loop69 x Loop164 Train-only surrogate complementarity gate."
    )
    parser.add_argument("--loop69-report", type=Path, default=DEFAULT_LOOP69_REPORT)
    parser.add_argument("--loop69-readiness", type=Path, default=DEFAULT_LOOP69_READINESS)
    parser.add_argument("--loop69-predictions", type=Path, default=DEFAULT_LOOP69_PREDICTIONS)
    parser.add_argument("--loop164-report", type=Path, default=DEFAULT_LOOP164_REPORT)
    parser.add_argument("--loop164-predictions", type=Path, default=DEFAULT_LOOP164_PREDICTIONS)
    parser.add_argument("--loop164-folds", type=Path, default=DEFAULT_LOOP164_FOLDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit(
        loop69_report=args.loop69_report,
        loop69_readiness=args.loop69_readiness,
        loop69_predictions=args.loop69_predictions,
        loop164_report=args.loop164_report,
        loop164_predictions=args.loop164_predictions,
        loop164_folds=args.loop164_folds,
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
