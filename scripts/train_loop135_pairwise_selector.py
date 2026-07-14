#!/usr/bin/env python3
"""Train a Val-gated pairwise selector between two frozen predictors.

The selector is deliberately narrow: the baseline prediction is kept unless a
model trained on Train-only disagreement rows decides to accept the candidate
prediction. Identity columns are used only for alignment, cache lookup, and
auditable output.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from kvd_features.content_pe_v1 import CONTENT_PE_V1_FEATURE_NAMES  # noqa: E402
from train_stage2_cache_matrix import (  # noqa: E402
    CONTENT_PE_V2_FEATURE_NAMES,
    CONTENT_STRING_FEATURE_NAMES,
    FeatureConfig,
    _knn_feature_names,
    append_frozen_knn_features,
    build_matrix,
    build_oof_knn_features,
    load_stage2_knn_reference_from_payload,
    load_valid_feature_npz,
)
from config import AxonExperimentConfig  # noqa: E402


SCORE_FEATURE_NAMES = [
    "baseline_prob_malicious",
    "candidate_prob_malicious",
    "prob_delta_candidate_minus_baseline",
    "baseline_abs_margin_from_half",
    "candidate_abs_margin_from_half",
    "margin_delta_candidate_minus_baseline",
    "baseline_predicted_malicious",
    "candidate_predicted_malicious",
    "direction_baseline0_candidate1",
    "direction_baseline1_candidate0",
]

CONTENT_FEATURE_NAMES = [
    "content_is_dll",
    "content_export_count_log",
    "content_dir_security_log_size",
    "content_overlay_log_size",
    "content_resource_entry_count_log",
    "content_resource_type_count_log",
    "content_dir_resource_size_ratio",
    "content_overlay_entropy",
    "content_import_api_count_log",
    "content_avg_imports_per_dll",
    "v2_resource_data_entry_count_log",
    "v2_resource_type_icon_count_log",
    "v2_resource_type_dialog_count_log",
    "v2_last_section_entropy",
    "v2_section_max_virtual_raw_ratio_log",
    "v2_api_file_mutation_ratio",
    "v2_import_dll_version_api_ratio",
    "string_benign_vendor_count_log",
    "string_version_resource_count_log",
    "string_script_exec_count_log",
    "string_script_exec_present",
]


def support_feature_names(top_ks: Sequence[int]) -> list[str]:
    names = [f"support_{name}" for name in _knn_feature_names(top_ks)]
    assert_no_identity_feature_names(names, context="Loop135 pairwise support features")
    return names


@dataclass(frozen=True)
class AlignedPair:
    rows: list[dict]
    labels: np.ndarray
    baseline_prob: np.ndarray
    candidate_prob: np.ndarray
    baseline_pred: np.ndarray
    candidate_pred: np.ndarray


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_rows(path: Path) -> list[dict]:
    with resolve_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict, key_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")) for column in key_columns)


def _float(row: dict, column: str) -> float:
    return float(str(row[column]).strip())


def align_pair(
    baseline_path: Path,
    candidate_path: Path,
    *,
    key_columns: Sequence[str],
    baseline_score_column: str,
    candidate_score_column: str,
) -> AlignedPair:
    baseline_rows = read_rows(baseline_path)
    candidate_rows = read_rows(candidate_path)
    candidate_by_key = {row_key(row, key_columns): row for row in candidate_rows}
    if len(candidate_by_key) != len(candidate_rows):
        raise ValueError("Candidate predictions contain duplicate alignment keys")

    rows: list[dict] = []
    labels = []
    baseline_prob = []
    candidate_prob = []
    baseline_pred = []
    candidate_pred = []
    missing = []
    for baseline_row in baseline_rows:
        key = row_key(baseline_row, key_columns)
        candidate_row = candidate_by_key.get(key)
        if candidate_row is None:
            missing.append(key)
            continue
        label = int(baseline_row["label"])
        if label != int(candidate_row["label"]):
            raise ValueError(f"Label mismatch for alignment key={key}")
        rows.append(baseline_row)
        labels.append(label)
        baseline_prob.append(_float(baseline_row, baseline_score_column))
        candidate_prob.append(_float(candidate_row, candidate_score_column))
        baseline_pred.append(int(baseline_row["prediction"]))
        candidate_pred.append(int(candidate_row["prediction"]))
    if missing:
        preview = ", ".join(str(item) for item in missing[:5])
        raise ValueError(f"Missing {len(missing)} candidate rows; first keys: {preview}")

    return AlignedPair(
        rows=rows,
        labels=np.asarray(labels, dtype=np.int64),
        baseline_prob=np.asarray(baseline_prob, dtype=np.float32),
        candidate_prob=np.asarray(candidate_prob, dtype=np.float32),
        baseline_pred=np.asarray(baseline_pred, dtype=np.int64),
        candidate_pred=np.asarray(candidate_pred, dtype=np.int64),
    )


def disagreement_mask(pair: AlignedPair) -> np.ndarray:
    return pair.baseline_pred != pair.candidate_pred


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


def passes_val_improvement_constraint(
    candidate_metrics: dict,
    baseline_metrics: dict,
    *,
    require_val_improvement: bool,
    min_val_error_reduction: int,
    min_val_f1_delta: float,
) -> bool:
    if not require_val_improvement:
        return True
    baseline_errors = int(baseline_metrics["errors"])
    candidate_errors = int(candidate_metrics["errors"])
    baseline_f1 = float(baseline_metrics["f1"])
    candidate_f1 = float(candidate_metrics["f1"])
    return (
        candidate_errors <= baseline_errors - int(min_val_error_reduction)
        and candidate_f1 > baseline_f1 + float(min_val_f1_delta)
    )


def build_score_features(pair: AlignedPair, indices: np.ndarray) -> np.ndarray:
    assert_no_identity_feature_names(SCORE_FEATURE_NAMES, context="Loop135 pairwise score features")
    baseline_prob = pair.baseline_prob[indices]
    candidate_prob = pair.candidate_prob[indices]
    baseline_margin = np.abs(baseline_prob - 0.5)
    candidate_margin = np.abs(candidate_prob - 0.5)
    baseline_pred = pair.baseline_pred[indices].astype(np.float32, copy=False)
    candidate_pred = pair.candidate_pred[indices].astype(np.float32, copy=False)
    matrix = np.column_stack(
        [
            baseline_prob,
            candidate_prob,
            candidate_prob - baseline_prob,
            baseline_margin,
            candidate_margin,
            candidate_margin - baseline_margin,
            baseline_pred,
            candidate_pred,
            ((baseline_pred == 0) & (candidate_pred == 1)).astype(np.float32),
            ((baseline_pred == 1) & (candidate_pred == 0)).astype(np.float32),
        ]
    ).astype(np.float32, copy=False)
    if matrix.shape[1] != len(SCORE_FEATURE_NAMES):
        raise ValueError("Score feature shape mismatch")
    return matrix


def _cache_key(row: dict) -> str:
    source_sha = str(row.get("source_sha256") or "").strip().casefold()
    if not source_sha:
        raise ValueError("source_sha256 is required for content sidecar lookup")
    return source_sha


def _load_sidecar(row: dict, cache_dir: Path, feature_names: Sequence[str], family: str) -> np.ndarray:
    cache_path = resolve_path(cache_dir) / f"{_cache_key(row)}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing {family} sidecar cache: {cache_path}")
    features = load_valid_feature_npz(cache_path, len(feature_names))
    if features is None:
        raise ValueError(f"Bad {family} sidecar cache: {cache_path}")
    return features.astype(np.float32, copy=False)


def build_content_features(
    rows: Sequence[dict],
    indices: np.ndarray,
    *,
    content_pe_cache_dir: Optional[Path],
    content_pe_v2_cache_dir: Optional[Path],
    content_string_cache_dir: Optional[Path],
) -> tuple[np.ndarray, list[str]]:
    if not (content_pe_cache_dir and content_pe_v2_cache_dir and content_string_cache_dir):
        return np.empty((len(indices), 0), dtype=np.float32), []
    assert_no_identity_feature_names(CONTENT_FEATURE_NAMES, context="Loop135 pairwise content features")
    v1_index = {name: idx for idx, name in enumerate(CONTENT_PE_V1_FEATURE_NAMES)}
    v2_index = {name: idx for idx, name in enumerate(CONTENT_PE_V2_FEATURE_NAMES)}
    string_index = {name: idx for idx, name in enumerate(CONTENT_STRING_FEATURE_NAMES)}
    missing = [
        name
        for name in CONTENT_FEATURE_NAMES
        if name not in v1_index and name not in v2_index and name not in string_index
    ]
    if missing:
        raise ValueError(f"Missing content feature names: {missing}")

    matrix = np.empty((len(indices), len(CONTENT_FEATURE_NAMES)), dtype=np.float32)
    for output_index, row_index in enumerate(indices.tolist()):
        row = rows[row_index]
        pe1 = _load_sidecar(row, content_pe_cache_dir, CONTENT_PE_V1_FEATURE_NAMES, "content_pe_v1")
        pe2 = _load_sidecar(row, content_pe_v2_cache_dir, CONTENT_PE_V2_FEATURE_NAMES, "content_pe_v2")
        string_features = _load_sidecar(row, content_string_cache_dir, CONTENT_STRING_FEATURE_NAMES, "content_string")
        for feature_index, name in enumerate(CONTENT_FEATURE_NAMES):
            if name in v1_index:
                matrix[output_index, feature_index] = pe1[v1_index[name]]
            elif name in v2_index:
                matrix[output_index, feature_index] = pe2[v2_index[name]]
            else:
                matrix[output_index, feature_index] = string_features[string_index[name]]
    return matrix, list(CONTENT_FEATURE_NAMES)


def _load_support_payload(model_path: Path) -> tuple[FeatureConfig, AxonExperimentConfig, dict, Path]:
    resolved_model_path = resolve_path(model_path)
    with resolved_model_path.open("rb") as handle:
        payload = pickle.load(handle)
    feature_config = payload["feature_config"]
    if not isinstance(feature_config, FeatureConfig):
        feature_config = FeatureConfig(**dict(feature_config))
    checkpoint_config = AxonExperimentConfig.from_dict(dict(payload["checkpoint_config"]))
    knn_payload = payload.get("knn") or {}
    if not knn_payload.get("enabled"):
        raise ValueError("Support model must contain enabled kNN memory")
    return feature_config, checkpoint_config, knn_payload, resolved_model_path


def _parse_top_ks(text: Optional[str], knn_payload: dict) -> list[int]:
    if text:
        top_ks = [int(item.strip()) for item in text.split(",") if item.strip()]
    else:
        top_ks = [int(item) for item in knn_payload.get("top_ks") or []]
    if not top_ks:
        raise ValueError("Support kNN top-k list is empty")
    return sorted(dict.fromkeys(top_ks))


def _support_fallback_row(pair_row: dict) -> dict:
    row = dict(pair_row)
    if "prob_malicious" not in row or str(row.get("prob_malicious", "")).strip() == "":
        for column in ("baseline_prob_malicious", "stage2_prob_malicious", "candidate_prob_malicious"):
            if str(row.get(column, "")).strip() != "":
                row["prob_malicious"] = row[column]
                break
    if "prob_malicious" not in row or str(row.get("prob_malicious", "")).strip() == "":
        raise ValueError("Cannot build support fallback row without a probability column")
    return row


def _aligned_support_rows(
    pair: AlignedPair,
    support_predictions: Path,
    key_columns: Sequence[str],
    *,
    allow_pair_fallback: bool = False,
) -> tuple[list[dict], int]:
    support_rows = read_rows(support_predictions)
    support_by_key: dict[tuple[str, ...], dict] = {}
    for row in support_rows:
        key = row_key(row, key_columns)
        previous = support_by_key.get(key)
        if previous is None:
            support_by_key[key] = row
            continue
        comparable_columns = ("label", "cache_path", "source_sha256", "prob_malicious", "stage2_prob_malicious")
        if any(str(previous.get(column, "")) != str(row.get(column, "")) for column in comparable_columns):
            raise ValueError(f"Support predictions contain conflicting duplicate alignment key: {key}")
    aligned = []
    missing = []
    fallback_count = 0
    for pair_row, label in zip(pair.rows, pair.labels.tolist()):
        key = row_key(pair_row, key_columns)
        support_row = support_by_key.get(key)
        if support_row is None:
            if allow_pair_fallback:
                aligned.append(_support_fallback_row(pair_row))
                fallback_count += 1
                continue
            missing.append(key)
            continue
        if int(support_row["label"]) != int(label):
            raise ValueError(f"Support label mismatch for alignment key={key}")
        if "prob_malicious" not in support_row or str(support_row.get("prob_malicious", "")).strip() == "":
            support_row = _support_fallback_row(support_row)
            fallback_count += 1
        aligned.append(support_row)
    if missing:
        preview = ", ".join(str(item) for item in missing[:5])
        raise ValueError(f"Missing {len(missing)} support rows; first keys: {preview}")
    return aligned, fallback_count


def build_support_feature_blocks(
    *,
    support_stage2_model: Path,
    support_train_predictions: Path,
    support_val_predictions: Path,
    train_pair: AlignedPair,
    val_pair: AlignedPair,
    key_columns: Sequence[str],
    support_key_columns: Sequence[str],
    top_ks_text: Optional[str],
    oof_folds: int,
    seed: int,
    batch_size: int,
    similarity_memory_mib: float,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    feature_config, checkpoint_config, knn_payload, resolved_model_path = _load_support_payload(support_stage2_model)
    top_ks = _parse_top_ks(top_ks_text, knn_payload)
    feature_names = support_feature_names(top_ks)
    reference = load_stage2_knn_reference_from_payload(resolved_model_path, knn_payload)
    raw_train_rows = read_rows(support_train_predictions)
    raw_train_labels = np.asarray([int(row["label"]) for row in raw_train_rows], dtype=np.int64)
    memory_labels = np.asarray(reference["memory_labels"], dtype=np.int64)
    if memory_labels.shape[0] != len(raw_train_rows):
        raise ValueError(
            f"Support kNN memory and train predictions disagree: memory={memory_labels.shape[0]} rows={len(raw_train_rows)}"
        )
    if not np.array_equal(memory_labels, raw_train_labels):
        raise ValueError("Support kNN memory labels do not match support train prediction row order")
    train_rows, train_fallback_count = _aligned_support_rows(
        train_pair,
        support_train_predictions,
        support_key_columns,
        allow_pair_fallback=True,
    )
    val_rows, val_fallback_count = _aligned_support_rows(val_pair, support_val_predictions, support_key_columns)
    train_matrix, train_labels, _train_base_probs, _train_kept_rows, train_counts = build_matrix(
        train_rows,
        checkpoint_config,
        feature_config,
    )
    if train_counts["kept"] != len(train_rows):
        raise ValueError(f"Support train rows must be fully covered: {train_counts}")
    if not np.array_equal(train_labels.astype(np.int64, copy=False), train_pair.labels.astype(np.int64, copy=False)):
        raise ValueError("Support train labels do not match pair labels")

    train_support, train_oof_info = build_oof_knn_features(
        train_matrix,
        train_labels,
        top_ks=top_ks,
        folds=int(oof_folds),
        seed=int(seed),
        batch_size=int(batch_size),
        max_similarity_mib=float(similarity_memory_mib),
    )
    val_matrix, val_labels, _val_base_probs, _val_kept_rows, val_counts = build_matrix(
        val_rows,
        checkpoint_config,
        feature_config,
    )
    if val_counts["kept"] != len(val_rows):
        raise ValueError(f"Support val rows must be fully covered: {val_counts}")
    if not np.array_equal(val_labels.astype(np.int64, copy=False), val_pair.labels.astype(np.int64, copy=False)):
        raise ValueError("Support val labels do not match pair labels")
    val_with_support = append_frozen_knn_features(
        val_matrix,
        reference,
        top_ks,
        batch_size=int(batch_size),
        max_similarity_mib=float(similarity_memory_mib),
    )
    val_support = val_with_support[:, val_matrix.shape[1] :]
    info = {
        "support_stage2_model": str(resolved_model_path),
        "support_train_predictions": str(resolve_path(support_train_predictions)),
        "support_val_predictions": str(resolve_path(support_val_predictions)),
        "support_key_columns": list(support_key_columns),
        "top_ks": top_ks,
        "oof_folds": int(oof_folds),
        "batch_size": int(batch_size),
        "similarity_memory_mib": float(similarity_memory_mib),
        "feature_config": feature_config.__dict__,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "train_support_fallback_rows": int(train_fallback_count),
        "val_support_fallback_rows": int(val_fallback_count),
        "train_oof_info": train_oof_info,
        "feature_names": feature_names,
        "leakage_policy": "train selector rows use OOF kNN support; val/test rows use frozen train-memory kNN support",
    }
    return train_support.astype(np.float32, copy=False), val_support.astype(np.float32, copy=False), feature_names, info


def build_eval_support_feature_block(
    *,
    support_stage2_model: Path,
    support_predictions: Path,
    pair: AlignedPair,
    support_key_columns: Sequence[str],
    top_ks: Sequence[int],
    batch_size: int,
    similarity_memory_mib: float,
) -> tuple[np.ndarray, list[str], dict]:
    feature_config, checkpoint_config, knn_payload, resolved_model_path = _load_support_payload(support_stage2_model)
    reference = load_stage2_knn_reference_from_payload(resolved_model_path, knn_payload)
    support_rows, support_fallback_count = _aligned_support_rows(pair, support_predictions, support_key_columns)
    matrix, labels, _base_probs, _kept_rows, counts = build_matrix(
        support_rows,
        checkpoint_config,
        feature_config,
    )
    if counts["kept"] != len(support_rows):
        raise ValueError(f"Support eval rows must be fully covered: {counts}")
    if not np.array_equal(labels.astype(np.int64, copy=False), pair.labels.astype(np.int64, copy=False)):
        raise ValueError("Support eval labels do not match pair labels")
    with_support = append_frozen_knn_features(
        matrix,
        reference,
        top_ks,
        batch_size=int(batch_size),
        max_similarity_mib=float(similarity_memory_mib),
    )
    support_matrix = with_support[:, matrix.shape[1] :]
    info = {
        "support_stage2_model": str(resolved_model_path),
        "support_predictions": str(resolve_path(support_predictions)),
        "support_key_columns": list(support_key_columns),
        "top_ks": [int(item) for item in top_ks],
        "batch_size": int(batch_size),
        "similarity_memory_mib": float(similarity_memory_mib),
        "records": counts,
        "support_fallback_rows": int(support_fallback_count),
        "feature_names": support_feature_names(top_ks),
    }
    return support_matrix.astype(np.float32, copy=False), support_feature_names(top_ks), info


def build_selector_features(
    pair: AlignedPair,
    indices: np.ndarray,
    *,
    content_pe_cache_dir: Optional[Path],
    content_pe_v2_cache_dir: Optional[Path],
    content_string_cache_dir: Optional[Path],
    support_matrix: Optional[np.ndarray] = None,
    support_names: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    score_matrix = build_score_features(pair, indices)
    content_matrix, content_names = build_content_features(
        pair.rows,
        indices,
        content_pe_cache_dir=content_pe_cache_dir,
        content_pe_v2_cache_dir=content_pe_v2_cache_dir,
        content_string_cache_dir=content_string_cache_dir,
    )
    matrices = [score_matrix]
    names = list(SCORE_FEATURE_NAMES)
    if content_matrix.shape[1] > 0:
        matrices.append(content_matrix)
        names.extend(content_names)
    if support_matrix is not None:
        if support_matrix.shape[0] != pair.labels.shape[0]:
            raise ValueError("Support feature row count must match pair rows")
        selected_support = support_matrix[indices].astype(np.float32, copy=False)
        names.extend(list(support_names or []))
        matrices.append(selected_support)
    matrix = np.concatenate(matrices, axis=1).astype(np.float32, copy=False)
    if len(names) != matrix.shape[1]:
        raise ValueError("Selector feature names do not match feature matrix width")
    assert_no_identity_feature_names(names, context="Loop135 pairwise selector features")
    return matrix, names


def model_candidates(seed: int) -> list[tuple[str, object]]:
    return [
        (
            "selector_logreg_l2_c0.1",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="liblinear", C=0.1)),
        ),
        (
            "selector_logreg_balanced_c0.1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, solver="liblinear", C=0.1, class_weight="balanced"),
            ),
        ),
        (
            "selector_hgb_leaf3",
            HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.04,
                max_leaf_nodes=3,
                l2_regularization=0.05,
                random_state=seed,
            ),
        ),
        (
            "selector_extra_trees_leaf2",
            ExtraTreesClassifier(
                n_estimators=100,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=1,
                class_weight="balanced",
            ),
        ),
    ]


def predict_scores(model, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float32)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(matrix), dtype=np.float32)
        return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)
    return np.asarray(model.predict(matrix), dtype=np.float32)


def parse_thresholds(text: str) -> list[float]:
    if ":" in text:
        start, stop, step = (float(part) for part in text.split(":"))
        values = []
        current = start
        while current <= stop + 1.0e-12:
            values.append(round(float(current), 10))
            current += step
        return values
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def apply_selector(pair: AlignedPair, diff_indices: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    accept_candidate = np.zeros(pair.labels.shape[0], dtype=bool)
    accept_candidate[diff_indices] = scores >= float(threshold)
    predictions = pair.baseline_pred.copy()
    predictions[accept_candidate] = pair.candidate_pred[accept_candidate]
    return predictions, accept_candidate


def apply_selector_directional(
    pair: AlignedPair,
    diff_indices: np.ndarray,
    scores: np.ndarray,
    threshold_0to1: float,
    threshold_1to0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply separate accept thresholds for the two disagreement directions."""
    accept_candidate = np.zeros(pair.labels.shape[0], dtype=bool)
    if diff_indices.size:
        baseline_diff = pair.baseline_pred[diff_indices]
        candidate_diff = pair.candidate_pred[diff_indices]
        accept_diff = np.zeros(diff_indices.shape[0], dtype=bool)
        direction_0to1 = (baseline_diff == 0) & (candidate_diff == 1)
        direction_1to0 = (baseline_diff == 1) & (candidate_diff == 0)
        accept_diff[direction_0to1] = scores[direction_0to1] >= float(threshold_0to1)
        accept_diff[direction_1to0] = scores[direction_1to0] >= float(threshold_1to0)
        accept_candidate[diff_indices] = accept_diff
    predictions = pair.baseline_pred.copy()
    predictions[accept_candidate] = pair.candidate_pred[accept_candidate]
    return predictions, accept_candidate


def write_predictions(
    path: Path,
    pair: AlignedPair,
    scores: np.ndarray,
    predictions: np.ndarray,
    accept_candidate: np.ndarray,
) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(pair.rows[0].keys())
    for fieldname in [
        "baseline_prob_malicious",
        "candidate_prob_malicious",
        "selector_score",
        "selector_accept_candidate",
        "stage2_prob_malicious",
        "prediction",
        "correct",
    ]:
        if fieldname not in fieldnames:
            fieldnames.append(fieldname)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(pair.rows):
            output = dict(row)
            output["baseline_prob_malicious"] = f"{float(pair.baseline_prob[index]):.10f}"
            output["candidate_prob_malicious"] = f"{float(pair.candidate_prob[index]):.10f}"
            output["selector_score"] = f"{float(scores[index]):.10f}"
            output["selector_accept_candidate"] = "1" if bool(accept_candidate[index]) else "0"
            output["stage2_prob_malicious"] = f"{float(pair.candidate_prob[index] if accept_candidate[index] else pair.baseline_prob[index]):.10f}"
            output["prediction"] = str(int(predictions[index]))
            output["correct"] = "True" if int(predictions[index]) == int(pair.labels[index]) else "False"
            writer.writerow(output)


def train_command(args: argparse.Namespace) -> int:
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    key_columns = tuple(item.strip() for item in args.key_columns.split(",") if item.strip())
    support_key_columns = tuple(item.strip() for item in args.support_key_columns.split(",") if item.strip())
    train_pair = align_pair(
        args.baseline_train_predictions,
        args.candidate_train_predictions,
        key_columns=key_columns,
        baseline_score_column=args.baseline_score_column,
        candidate_score_column=args.candidate_score_column,
    )
    val_pair = align_pair(
        args.baseline_val_predictions,
        args.candidate_val_predictions,
        key_columns=key_columns,
        baseline_score_column=args.baseline_score_column,
        candidate_score_column=args.candidate_score_column,
    )
    train_diff = np.flatnonzero(disagreement_mask(train_pair))
    val_diff = np.flatnonzero(disagreement_mask(val_pair))
    train_targets = (train_pair.candidate_pred[train_diff] == train_pair.labels[train_diff]).astype(np.int64)
    if train_targets.size < 8 or len(set(int(item) for item in train_targets.tolist())) < 2:
        raise ValueError(
            f"Not enough mixed train disagreement rows: rows={train_targets.size}, classes={sorted(set(train_targets.tolist()))}"
        )

    train_support_matrix = None
    val_support_matrix = None
    support_names: list[str] = []
    support_info = None
    if args.support_stage2_model or args.support_train_predictions or args.support_val_predictions:
        if not (args.support_stage2_model and args.support_train_predictions and args.support_val_predictions):
            raise ValueError(
                "--support-stage2-model, --support-train-predictions, and --support-val-predictions must be passed together"
            )
        train_support_matrix, val_support_matrix, support_names, support_info = build_support_feature_blocks(
            support_stage2_model=args.support_stage2_model,
            support_train_predictions=args.support_train_predictions,
            support_val_predictions=args.support_val_predictions,
            train_pair=train_pair,
            val_pair=val_pair,
            key_columns=key_columns,
            support_key_columns=support_key_columns,
            top_ks_text=args.support_top_ks,
            oof_folds=int(args.support_oof_folds),
            seed=int(args.seed),
            batch_size=int(args.support_knn_batch_size),
            similarity_memory_mib=float(args.support_knn_similarity_memory_mib),
        )

    train_x, feature_names = build_selector_features(
        train_pair,
        train_diff,
        content_pe_cache_dir=args.content_pe_cache_dir,
        content_pe_v2_cache_dir=args.content_pe_v2_cache_dir,
        content_string_cache_dir=args.content_string_cache_dir,
        support_matrix=train_support_matrix,
        support_names=support_names,
    )
    val_x, _ = build_selector_features(
        val_pair,
        val_diff,
        content_pe_cache_dir=args.content_pe_cache_dir,
        content_pe_v2_cache_dir=args.content_pe_v2_cache_dir,
        content_string_cache_dir=args.content_string_cache_dir,
        support_matrix=val_support_matrix,
        support_names=support_names,
    )
    thresholds = parse_thresholds(args.thresholds)
    threshold_mode = str(args.threshold_mode)
    baseline_val_metrics = metrics(val_pair.labels, val_pair.baseline_pred)
    candidate_val_metrics = metrics(val_pair.labels, val_pair.candidate_pred)
    if int(args.min_val_error_reduction) < 0:
        raise ValueError("--min-val-error-reduction must be non-negative")
    if float(args.min_val_f1_delta) < 0.0:
        raise ValueError("--min-val-f1-delta must be non-negative")

    candidates = []
    selected = None
    selected_model = None
    selected_predictions = None
    selected_accept = None
    selected_scores_full = None
    best_key = None
    baseline_val_fn = int(baseline_val_metrics["false_negative"])
    max_allowed_val_fn = (
        baseline_val_fn + int(args.max_val_fn_increase)
        if args.max_val_fn_increase is not None
        else None
    )
    start_all = time.perf_counter()
    for model_name, prototype in model_candidates(int(args.seed)):
        model = clone(prototype)
        start = time.perf_counter()
        model.fit(train_x, train_targets)
        fit_sec = time.perf_counter() - start
        val_scores_diff = predict_scores(model, val_x) if val_diff.size else np.asarray([], dtype=np.float32)
        threshold_pairs = (
            [(threshold, threshold) for threshold in thresholds]
            if threshold_mode == "global"
            else [(threshold_0to1, threshold_1to0) for threshold_0to1 in thresholds for threshold_1to0 in thresholds]
        )
        for threshold_0to1, threshold_1to0 in threshold_pairs:
            scores_full = np.zeros(val_pair.labels.shape[0], dtype=np.float32)
            scores_full[val_diff] = val_scores_diff
            if threshold_mode == "directional":
                predictions, accept = apply_selector_directional(
                    val_pair,
                    val_diff,
                    val_scores_diff,
                    threshold_0to1,
                    threshold_1to0,
                )
            else:
                predictions, accept = apply_selector(val_pair, val_diff, val_scores_diff, threshold_0to1)
            candidate_metrics = metrics(val_pair.labels, predictions)
            result = {
                "name": model_name,
                "threshold_mode": threshold_mode,
                "threshold": float(threshold_0to1) if threshold_mode == "global" else None,
                "threshold_0to1": float(threshold_0to1),
                "threshold_1to0": float(threshold_1to0),
                "thresholds_by_direction": {
                    "baseline0_candidate1": float(threshold_0to1),
                    "baseline1_candidate0": float(threshold_1to0),
                },
                "fit_sec": float(fit_sec),
                "val": candidate_metrics,
                "passes_val_fn_constraint": (
                    True
                    if max_allowed_val_fn is None
                    else int(candidate_metrics["false_negative"]) <= int(max_allowed_val_fn)
                ),
                "passes_val_improvement_constraint": passes_val_improvement_constraint(
                    candidate_metrics,
                    baseline_val_metrics,
                    require_val_improvement=bool(args.require_val_improvement),
                    min_val_error_reduction=int(args.min_val_error_reduction),
                    min_val_f1_delta=float(args.min_val_f1_delta),
                ),
                "val_error_delta_vs_baseline": int(candidate_metrics["errors"]) - int(baseline_val_metrics["errors"]),
                "val_f1_delta_vs_baseline": float(candidate_metrics["f1"]) - float(baseline_val_metrics["f1"]),
                "val_accepts": int(np.count_nonzero(accept)),
                "val_accept_label0": int(np.count_nonzero(accept & (val_pair.labels == 0))),
                "val_accept_label1": int(np.count_nonzero(accept & (val_pair.labels == 1))),
                "train_disagreements": int(train_diff.size),
                "train_candidate_correct": int(np.count_nonzero(train_targets == 1)),
                "train_candidate_wrong": int(np.count_nonzero(train_targets == 0)),
                "val_disagreements": int(val_diff.size),
            }
            candidates.append(result)
            if not bool(result["passes_val_fn_constraint"]):
                continue
            if not bool(result["passes_val_improvement_constraint"]):
                continue
            key = (float(candidate_metrics["f1"]), -int(candidate_metrics["errors"]), -int(candidate_metrics["false_negative"]))
            if best_key is None or key > best_key:
                best_key = key
                selected = result
                selected_model = model
                selected_predictions = predictions.copy()
                selected_accept = accept.copy()
                selected_scores_full = scores_full.copy()
        if selected_model is not model:
            del model

    if selected is None or selected_model is None or selected_predictions is None or selected_accept is None:
        raise ValueError("No selector candidate was selected")
    if selected_scores_full is None:
        raise ValueError("Selected scores were not retained")

    model_path = output_dir / "loop135_pairwise_selector.pkl"
    payload = {
        "schema": "axon_loop135_pairwise_selector_payload_v1",
        "protocol": "Train-only selector over frozen predictor disagreements; Val selects model/threshold; no test used",
        "model": selected_model,
        "selected": selected,
        "feature_names": feature_names,
        "threshold_mode": threshold_mode,
        "support_info": support_info,
        "key_columns": key_columns,
        "baseline_score_column": args.baseline_score_column,
        "candidate_score_column": args.candidate_score_column,
        "identity_feature_policy": (
            "source_path/source_sha256/cache_path/sample_index/split/path/name/extension/directory/hash are alignment, "
            "loading, and audit fields only; they are not model evidence"
        ),
    }
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle)
    output_predictions = output_dir / "loop135_pairwise_selector_val_predictions.csv"
    write_predictions(output_predictions, val_pair, selected_scores_full, selected_predictions, selected_accept)
    report = {
        "schema": "axon_loop135_pairwise_selector_v1",
        "protocol": payload["protocol"],
        "baseline_train_predictions": str(resolve_path(args.baseline_train_predictions)),
        "candidate_train_predictions": str(resolve_path(args.candidate_train_predictions)),
        "baseline_val_predictions": str(resolve_path(args.baseline_val_predictions)),
        "candidate_val_predictions": str(resolve_path(args.candidate_val_predictions)),
        "model_path": str(model_path),
        "val_predictions": str(output_predictions),
        "feature_names": feature_names,
        "support_info": support_info,
        "baseline_val": baseline_val_metrics,
        "candidate_val": candidate_val_metrics,
        "selected_by_val": selected,
        "selection_constraints": {
            "max_val_fn_increase": args.max_val_fn_increase,
            "baseline_val_fn": baseline_val_fn,
            "max_allowed_val_fn": max_allowed_val_fn,
            "require_val_improvement": bool(args.require_val_improvement),
            "baseline_val_errors": int(baseline_val_metrics["errors"]),
            "baseline_val_f1": float(baseline_val_metrics["f1"]),
            "min_val_error_reduction": int(args.min_val_error_reduction),
            "min_val_f1_delta": float(args.min_val_f1_delta),
            "threshold_mode": threshold_mode,
        },
        "candidates": candidates,
        "disagreement_summary": {
            "train": int(train_diff.size),
            "train_candidate_correct": int(np.count_nonzero(train_targets == 1)),
            "train_candidate_wrong": int(np.count_nonzero(train_targets == 0)),
            "val": int(val_diff.size),
            "val_candidate_correct": int(np.count_nonzero(val_pair.candidate_pred[val_diff] == val_pair.labels[val_diff])),
            "val_candidate_wrong": int(np.count_nonzero(val_pair.candidate_pred[val_diff] != val_pair.labels[val_diff])),
        },
        "elapsed_sec": float(time.perf_counter() - start_all),
    }
    report_path = output_dir / "loop135_pairwise_selector_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected_by_val": selected, "baseline_val": baseline_val_metrics, "candidate_val": candidate_val_metrics}, ensure_ascii=False, indent=2))
    print(f"JSON: {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Loop135 pairwise frozen-predictor selector.")
    parser.add_argument("--baseline-train-predictions", type=Path, required=True)
    parser.add_argument("--candidate-train-predictions", type=Path, required=True)
    parser.add_argument("--baseline-val-predictions", type=Path, required=True)
    parser.add_argument("--candidate-val-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-score-column", default="stage2_prob_malicious")
    parser.add_argument("--candidate-score-column", default="stage2_prob_malicious")
    parser.add_argument("--content-pe-cache-dir", type=Path, default=None)
    parser.add_argument("--content-pe-v2-cache-dir", type=Path, default=None)
    parser.add_argument("--content-string-cache-dir", type=Path, default=None)
    parser.add_argument("--support-stage2-model", type=Path, default=None)
    parser.add_argument("--support-train-predictions", type=Path, default=None)
    parser.add_argument("--support-val-predictions", type=Path, default=None)
    parser.add_argument("--support-key-columns", default="source_sha256")
    parser.add_argument("--support-top-ks", default=None)
    parser.add_argument("--support-oof-folds", type=int, default=5)
    parser.add_argument("--support-knn-batch-size", type=int, default=256)
    parser.add_argument("--support-knn-similarity-memory-mib", type=float, default=128.0)
    parser.add_argument("--thresholds", default="0.05:0.95:0.01")
    parser.add_argument("--threshold-mode", choices=("global", "directional"), default="global")
    parser.add_argument("--key-columns", default="sample_index,source_sha256")
    parser.add_argument("--seed", type=int, default=135)
    parser.add_argument("--max-val-fn-increase", type=int, default=None)
    parser.add_argument("--require-val-improvement", action="store_true")
    parser.add_argument("--min-val-error-reduction", type=int, default=1)
    parser.add_argument("--min-val-f1-delta", type=float, default=0.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return train_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
