"""Validation and fixed metrics for the local Loop164 OOF diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

LOOP_ID = "loop164_whole_file_residual_expert"
FOLD_SUMMARY_SCHEMA = "axon_loop164_local_train_diagnostic_folds_v1"
FOLD_RECORD_SCHEMA = "axon_loop164_local_train_diagnostic_fold_record_v1"
FOLD_CLAIM_SCOPE = "local_train_content_similarity_diagnostic_not_family_or_time_isolation"
REQUIRED_LIMITATIONS = [
    "no_authoritative_first_seen_time",
    "no_family_or_campaign_group",
    "no_custodian_source_group",
    "bounded_chunk_sketch_is_not_a_complete_near_duplicate_oracle",
    "not_purged_forward_oof",
]
MAX_FOLD_BUNDLE_BYTES = 32 * 1024 * 1024
FOLD_RECORD_KEYS = {
    "schema",
    "loop_id",
    "claim_scope",
    "split_role",
    "train_row_index",
    "sample_index",
    "source_path",
    "source_sha256",
    "source_size_bytes",
    "label",
    "availability",
    "missing_reason",
    "content_component_id",
    "content_component_size",
    "diagnostic_fold",
    "identity_metadata_not_model_features",
}
IDENTITY_METADATA_FIELDS = [
    "train_row_index",
    "sample_index",
    "source_path",
    "source_sha256",
    "content_component_id",
    "diagnostic_fold",
]
AVAILABILITY_REASONS = {
    "supported": None,
    "oversize": "oversize",
    "read_failure": "read_failure",
    "parse_failure": "parse_failure",
}


class LocalOOFContractError(ValueError):
    """The diagnostic folds or OOF outputs violate their frozen contract."""


@dataclass(frozen=True)
class LocalOOFRecord:
    train_row_index: int
    sample_index: int
    source_path: Path
    source_sha256: str
    source_size_bytes: Optional[int]
    label: int
    availability: str
    missing_reason: Optional[str]
    component_id: str
    component_size: int
    fold: int


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LocalOOFContractError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise LocalOOFContractError(f"Non-finite JSON value: {value}")


def _parse_json_object(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalOOFContractError(f"Invalid JSON for {context}") from exc
    if not isinstance(payload, dict):
        raise LocalOOFContractError(f"Expected JSON object for {context}")
    return payload


def _read_bounded(path: Path, *, max_bytes: int) -> bytes:
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise LocalOOFContractError(f"Bounded artifact is too large: {path}")
    return raw


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().casefold()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _lexical_relative_to(path: Path, root: Path) -> Path:
    absolute_path = path.absolute()
    absolute_root = root.absolute()
    try:
        return absolute_path.relative_to(absolute_root)
    except ValueError:
        path_parts = absolute_path.parts
        root_parts = absolute_root.parts
        if len(path_parts) < len(root_parts) or tuple(
            part.casefold() for part in path_parts[: len(root_parts)]
        ) != tuple(part.casefold() for part in root_parts):
            raise
        return Path(*path_parts[len(root_parts) :])


def _parse_record(
    payload: dict[str, Any],
    *,
    data_root: Path,
    fold_count: int,
    max_supported_file_bytes: int,
) -> LocalOOFRecord:
    if set(payload) != FOLD_RECORD_KEYS:
        raise LocalOOFContractError("Diagnostic fold record fields drifted")
    if (
        payload.get("schema") != FOLD_RECORD_SCHEMA
        or payload.get("loop_id") != LOOP_ID
        or payload.get("claim_scope") != FOLD_CLAIM_SCOPE
        or payload.get("split_role") != "train"
    ):
        raise LocalOOFContractError("Diagnostic fold record scope is invalid")
    for key in (
        "train_row_index",
        "sample_index",
        "content_component_size",
        "diagnostic_fold",
    ):
        if not isinstance(payload.get(key), int) or isinstance(payload.get(key), bool):
            raise LocalOOFContractError(f"Diagnostic fold record {key} is invalid")
    if payload["train_row_index"] < 0 or payload["sample_index"] < 0:
        raise LocalOOFContractError("Diagnostic fold identity index is negative")
    if payload["content_component_size"] < 1:
        raise LocalOOFContractError("Diagnostic fold component size is invalid")
    if not 0 <= payload["diagnostic_fold"] < fold_count:
        raise LocalOOFContractError("Diagnostic fold index is out of range")
    if payload.get("label") not in {0, 1}:
        raise LocalOOFContractError("Diagnostic fold label is invalid")
    source_sha256 = str(payload.get("source_sha256") or "").strip().casefold()
    if not _is_sha256(source_sha256):
        raise LocalOOFContractError("Diagnostic fold source SHA is invalid")
    source_path = Path(str(payload.get("source_path") or ""))
    try:
        relative = _lexical_relative_to(source_path, data_root)
    except ValueError as exc:
        raise LocalOOFContractError("Diagnostic source path escapes the data root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise LocalOOFContractError("Diagnostic source path has invalid components")
    availability = str(payload.get("availability") or "")
    if availability not in AVAILABILITY_REASONS:
        raise LocalOOFContractError("Diagnostic availability is invalid")
    if payload.get("missing_reason") != AVAILABILITY_REASONS[availability]:
        raise LocalOOFContractError("Diagnostic availability/missingness drifted")
    size = payload.get("source_size_bytes")
    if availability == "read_failure":
        if size is not None:
            raise LocalOOFContractError("Read-failure rows must not declare a size")
    elif not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise LocalOOFContractError("Available diagnostic row size is invalid")
    elif availability == "supported" and not 1 <= size <= max_supported_file_bytes:
        raise LocalOOFContractError("Supported-row size is outside the frozen input cap")
    elif availability == "oversize" and size <= max_supported_file_bytes:
        raise LocalOOFContractError("Oversize-row size does not exceed the frozen input cap")
    elif availability == "parse_failure" and size != 0:
        raise LocalOOFContractError("Parse-failure rows must declare zero bytes")
    component_id = str(payload.get("content_component_id") or "")
    if len(component_id) != 24 or any(character not in "0123456789abcdef" for character in component_id):
        raise LocalOOFContractError("Diagnostic component id is invalid")
    if payload.get("identity_metadata_not_model_features") != IDENTITY_METADATA_FIELDS:
        raise LocalOOFContractError("Diagnostic identity feature declaration drifted")
    return LocalOOFRecord(
        train_row_index=int(payload["train_row_index"]),
        sample_index=int(payload["sample_index"]),
        source_path=source_path,
        source_sha256=source_sha256,
        source_size_bytes=size,
        label=int(payload["label"]),
        availability=availability,
        missing_reason=payload.get("missing_reason"),
        component_id=component_id,
        component_size=int(payload["content_component_size"]),
        fold=int(payload["diagnostic_fold"]),
    )


def load_local_diagnostic_folds(
    *,
    folds_path: Path,
    summary_path: Path,
    data_root: Path,
    expected_rows: int,
    fold_count: int,
    expected_seed: int,
    max_supported_file_bytes: int,
    expected_rows_per_fold: int,
    expected_rows_per_label_per_fold: int,
) -> tuple[list[LocalOOFRecord], dict[str, Any]]:
    if (
        expected_rows < 1
        or fold_count < 2
        or expected_seed < 0
        or max_supported_file_bytes < 1
        or expected_rows_per_fold < 2
        or expected_rows_per_label_per_fold < 1
        or expected_rows_per_fold != 2 * expected_rows_per_label_per_fold
        or expected_rows != fold_count * expected_rows_per_fold
    ):
        raise LocalOOFContractError("Expected row/fold counts are invalid")
    folds_path = folds_path.resolve(strict=True)
    summary_path = summary_path.resolve(strict=True)
    data_root = data_root.resolve(strict=True)
    summary = _parse_json_object(
        _read_bounded(summary_path, max_bytes=1024 * 1024),
        context="diagnostic fold summary",
    )
    if (
        summary.get("schema") != FOLD_SUMMARY_SCHEMA
        or summary.get("loop_id") != LOOP_ID
        or summary.get("claim_scope") != FOLD_CLAIM_SCOPE
        or summary.get("decision")
        != "local_content_group_diagnostic_folds_ready_not_production_scope"
    ):
        raise LocalOOFContractError("Diagnostic fold summary scope is invalid")
    ready_for = summary.get("ready_for")
    if ready_for != {
        "a2_training_authority": False,
        "candidate_promotion": False,
        "local_whole_file_randomized_oof_diagnostic": True,
        "loop164_production_oof": False,
        "val_or_test_access": False,
    }:
        raise LocalOOFContractError("Diagnostic fold readiness claims are invalid")
    if summary.get("limitations") != REQUIRED_LIMITATIONS:
        raise LocalOOFContractError("Diagnostic fold limitations drifted")
    parameters = summary.get("parameters")
    if not isinstance(parameters, dict) or (
        parameters.get("fold_count") != fold_count
        or parameters.get("seed") != expected_seed
        or parameters.get("max_supported_file_bytes") != max_supported_file_bytes
    ):
        raise LocalOOFContractError("Diagnostic fold count binding drifted")
    inputs = summary.get("inputs")
    split_prefix = inputs.get("canonical_split_train_prefix") if isinstance(inputs, dict) else None
    if not isinstance(split_prefix, dict) or (
        split_prefix.get("heldout_rows_read") != 0
        or split_prefix.get("stopped_before_next_line") is not True
        or split_prefix.get("train_rows") != expected_rows
    ):
        raise LocalOOFContractError("Diagnostic Train-only input boundary drifted")
    time_stress = summary.get("time_stress_metadata")
    if not isinstance(time_stress, dict) or time_stress.get("used_for_fold_assignment") is not False:
        raise LocalOOFContractError("Diagnostic time-metadata limitation drifted")
    output = summary.get("output")
    if not isinstance(output, dict):
        raise LocalOOFContractError("Diagnostic fold output binding is missing")
    if Path(str(output.get("path") or "")).resolve(strict=True) != folds_path:
        raise LocalOOFContractError("Diagnostic fold path binding drifted")
    folds_raw = _read_bounded(folds_path, max_bytes=MAX_FOLD_BUNDLE_BYTES)
    if (
        output.get("sha256") != hashlib.sha256(folds_raw).hexdigest()
        or output.get("record_count") != expected_rows
        or output.get("record_schema") != FOLD_RECORD_SCHEMA
    ):
        raise LocalOOFContractError("Diagnostic fold hash/count/schema binding drifted")
    lines = folds_raw.splitlines()
    if len(lines) != expected_rows or any(not line.strip() for line in lines):
        raise LocalOOFContractError("Diagnostic fold JSONL line count is invalid")
    records = [
        _parse_record(
            _parse_json_object(line, context="diagnostic fold record"),
            data_root=data_root,
            fold_count=fold_count,
            max_supported_file_bytes=max_supported_file_bytes,
        )
        for line in lines
    ]
    records.sort(key=lambda record: record.train_row_index)
    if [record.train_row_index for record in records] != list(range(expected_rows)):
        raise LocalOOFContractError("Diagnostic train_row_index coverage is not exact")
    if [record.sample_index for record in records] != list(range(expected_rows)):
        raise LocalOOFContractError("Diagnostic sample_index coverage is not exact")
    if len({record.source_sha256 for record in records}) != expected_rows:
        raise LocalOOFContractError("Diagnostic source SHA coverage is not unique")
    component_folds: dict[str, set[int]] = defaultdict(set)
    component_labels: dict[str, set[int]] = defaultdict(set)
    component_counts: Counter[str] = Counter()
    for record in records:
        component_folds[record.component_id].add(record.fold)
        component_labels[record.component_id].add(record.label)
        component_counts[record.component_id] += 1
    if any(len(folds) != 1 for folds in component_folds.values()):
        raise LocalOOFContractError("A content component crosses diagnostic folds")
    if any(len(labels) != 1 for labels in component_labels.values()):
        raise LocalOOFContractError("A content component crosses locked labels")
    if any(component_counts[record.component_id] != record.component_size for record in records):
        raise LocalOOFContractError("Diagnostic component size binding drifted")
    actual_fold_labels = {
        str(fold): {
            str(label): sum(record.fold == fold and record.label == label for record in records)
            for label in (0, 1)
        }
        for fold in range(fold_count)
    }
    expected_fold_labels = {
        str(fold): {
            "0": expected_rows_per_label_per_fold,
            "1": expected_rows_per_label_per_fold,
        }
        for fold in range(fold_count)
    }
    if actual_fold_labels != expected_fold_labels:
        raise LocalOOFContractError("Diagnostic folds are not exactly preregistered and balanced")
    actual_fold_totals = {
        str(fold): sum(record.fold == fold for record in records) for fold in range(fold_count)
    }
    if actual_fold_totals != {
        str(fold): expected_rows_per_fold for fold in range(fold_count)
    }:
        raise LocalOOFContractError("Diagnostic fold totals drifted")
    folds_summary = summary.get("folds")
    if not isinstance(folds_summary, dict) or (
        folds_summary.get("component_cross_fold_count") != 0
        or folds_summary.get("fold_label_counts") != actual_fold_labels
        or folds_summary.get("fold_total_counts") != actual_fold_totals
    ):
        raise LocalOOFContractError("Diagnostic aggregate fold counts drifted")
    availability_counts = Counter(record.availability for record in records)
    label_counts = Counter(record.label for record in records)
    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, dict) or (
        aggregate.get("canonical_train_rows") != expected_rows
        or aggregate.get("availability_counts") != dict(sorted(availability_counts.items()))
        or aggregate.get("label_counts") != {str(label): label_counts[label] for label in (0, 1)}
        or aggregate.get("cross_label_components") != 0
        or aggregate.get("split_rows_by_role_read") != {"train": expected_rows}
    ):
        raise LocalOOFContractError("Diagnostic aggregate Train-only counts drifted")
    return records, summary


def fixed_binary_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float,
) -> dict[str, float | int]:
    if len(labels) != len(scores) or not labels:
        raise LocalOOFContractError("Metric labels/scores must be non-empty and aligned")
    if not 0.0 <= threshold <= 1.0:
        raise LocalOOFContractError("Metric threshold is invalid")
    true_positive = true_negative = false_positive = false_negative = 0
    for label, score in zip(labels, scores):
        if label not in {0, 1} or not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise LocalOOFContractError("Metric row is invalid")
        prediction = int(score >= threshold)
        if label == 1 and prediction == 1:
            true_positive += 1
        elif label == 0 and prediction == 0:
            true_negative += 1
        elif label == 0 and prediction == 1:
            false_positive += 1
        else:
            false_negative += 1
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": len(labels),
        "threshold": threshold,
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
