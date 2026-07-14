#!/usr/bin/env python3
"""Train Loop55 overlay/security-boundary Stage-2 candidates.

The added features separate the PE Security Directory certificate blob from
true overlay payload bytes. Paths and hashes are used only to open files and
align sidecar caches; they are not model features.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from config import AxonExperimentConfig  # noqa: E402
from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402
from train_stage2_cache_matrix import (  # noqa: E402
    DATA_DIRECTORY_INDEXES,
    PEFILE_AVAILABLE,
    FeatureConfig,
    _entropy_from_bytes,
    append_feature_columns,
    assert_stage2_feature_names_safe,
    build_matrix,
    clean_slice_metrics,
    content_cache_path_for_row,
    filter_model_candidates,
    load_valid_feature_npz,
    model_candidates,
    parse_thresholds,
    predict_scores,
    read_prediction_rows,
    resolve_path,
    sample_weights,
    save_feature_npz_atomic,
    select_best_threshold,
    source_sha256_for_row,
    summarize_noise,
    summarize_weights,
    verify_content_row_source_sha256,
    write_predictions,
)

try:
    import pefile
except ImportError:  # pragma: no cover - guarded by PEFILE_AVAILABLE
    pefile = None


LOOP28_VAL_F1 = 0.9919048570857486
LOOP28_VAL_ERRORS = 162
LOOP55_TEST10K_ERROR_GATE = 152


OVERLAY_BOUNDARY_FEATURE_NAMES = [
    "overlay_boundary_security_present",
    "overlay_boundary_security_log_size",
    "overlay_boundary_security_ratio",
    "overlay_boundary_overlay_present",
    "overlay_boundary_overlay_log_size",
    "overlay_boundary_overlay_ratio",
    "overlay_boundary_payload_present",
    "overlay_boundary_payload_log_size",
    "overlay_boundary_payload_ratio",
    "overlay_boundary_payload_entropy",
    "overlay_boundary_payload_head_entropy",
    "overlay_boundary_payload_tail_entropy",
    "overlay_boundary_payload_entropy_delta",
    "overlay_boundary_payload_high_entropy",
    "overlay_boundary_payload_after_security",
    "overlay_boundary_payload_before_security",
    "overlay_boundary_payload_segment_count_norm",
    "overlay_boundary_largest_payload_segment_ratio",
    "overlay_boundary_security_covers_overlay",
    "overlay_boundary_security_inside_overlay",
    "overlay_boundary_security_starts_at_overlay",
    "overlay_boundary_security_ends_at_eof",
    "overlay_boundary_gap_last_section_to_overlay_log",
    "overlay_boundary_gap_last_section_to_security_log",
    "overlay_boundary_overlay_touches_last_section",
    "overlay_boundary_payload_touches_last_section",
    "overlay_boundary_last_section_entropy",
    "overlay_boundary_payload_entropy_minus_last_section",
    "overlay_boundary_last_section_raw_virtual_delta",
    "overlay_boundary_overlay_slack_ratio",
    "overlay_boundary_security_declared_beyond_eof",
    "overlay_boundary_payload_after_cert_log_size",
]

assert_no_identity_feature_names(OVERLAY_BOUNDARY_FEATURE_NAMES, context="Loop55 overlay boundary features")


@dataclass(frozen=True)
class OverlayBoundaryConfig:
    cache_dir: Optional[str]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _safe_log1p(value: float) -> float:
    return math.log1p(max(float(value), 0.0))


def _clamp_span(start: int, size: int, file_size: int) -> Optional[tuple[int, int, bool]]:
    if file_size <= 0 or size <= 0:
        return None
    raw_start = int(start)
    raw_end = raw_start + int(size)
    clipped_start = max(0, min(raw_start, file_size))
    clipped_end = max(clipped_start, min(raw_end, file_size))
    if clipped_end <= clipped_start:
        return None
    return clipped_start, clipped_end, raw_end > file_size


def _subtract_span(
    segments: Sequence[tuple[int, int]],
    cut: Optional[tuple[int, int]],
) -> list[tuple[int, int]]:
    if cut is None:
        return [(start, end) for start, end in segments if end > start]
    cut_start, cut_end = cut
    result: list[tuple[int, int]] = []
    for start, end in segments:
        if cut_start > start:
            result.append((start, min(cut_start, end)))
        if cut_end < end:
            result.append((max(cut_end, start), end))
    return [(start, end) for start, end in result if end > start]


def _read_segments(file_path: Path, segments: Sequence[tuple[int, int]], limit: int = 131072) -> bytes:
    chunks: list[bytes] = []
    remaining = int(limit)
    if remaining <= 0:
        return b""
    try:
        with file_path.open("rb") as handle:
            for start, end in segments:
                if remaining <= 0:
                    break
                size = max(0, min(end - start, remaining))
                if size <= 0:
                    continue
                handle.seek(start)
                chunks.append(handle.read(size))
                remaining -= size
    except OSError:
        return b""
    return b"".join(chunks)


def _section_raw_span(section, file_size: int) -> Optional[tuple[int, int]]:
    start = int(getattr(section, "PointerToRawData", 0) or 0)
    size = int(getattr(section, "SizeOfRawData", 0) or 0)
    span = _clamp_span(start, size, file_size)
    if span is None:
        return None
    return span[0], span[1]


def _read_span_prefix(file_path: Path, span: Optional[tuple[int, int]], limit: int = 4096) -> bytes:
    if span is None or limit <= 0:
        return b""
    start, end = span
    size = max(0, min(end - start, int(limit)))
    if size <= 0:
        return b""
    try:
        with file_path.open("rb") as handle:
            handle.seek(start)
            return handle.read(size)
    except OSError:
        return b""


def overlay_boundary_features_from_path(file_path: Path) -> np.ndarray:
    if not PEFILE_AVAILABLE:
        return np.zeros(len(OVERLAY_BOUNDARY_FEATURE_NAMES), dtype=np.float32)
    file_path = Path(file_path)
    try:
        file_size = int(file_path.stat().st_size)
    except OSError:
        return np.zeros(len(OVERLAY_BOUNDARY_FEATURE_NAMES), dtype=np.float32)
    if file_size <= 0:
        return np.zeros(len(OVERLAY_BOUNDARY_FEATURE_NAMES), dtype=np.float32)

    try:
        pe = pefile.PE(str(file_path), fast_load=True)
    except Exception:
        return np.zeros(len(OVERLAY_BOUNDARY_FEATURE_NAMES), dtype=np.float32)

    try:
        optional = getattr(pe, "OPTIONAL_HEADER", None)
        directories = getattr(optional, "DATA_DIRECTORY", []) if optional is not None else []
        security = None
        security_index = DATA_DIRECTORY_INDEXES["security"]
        if 0 <= security_index < len(directories):
            security = directories[security_index]
        security_start = int(getattr(security, "VirtualAddress", 0) if security is not None else 0)
        security_size = int(getattr(security, "Size", 0) if security is not None else 0)
        security_span_full = _clamp_span(security_start, security_size, file_size)
        security_span = (
            (security_span_full[0], security_span_full[1]) if security_span_full is not None else None
        )
        security_beyond_eof = bool(security_span_full[2]) if security_span_full is not None else False
        security_len = (security_span[1] - security_span[0]) if security_span is not None else 0

        overlay_offset = pe.get_overlay_data_start_offset()
        overlay_start = int(overlay_offset) if overlay_offset is not None else file_size
        overlay_start = max(0, min(overlay_start, file_size))
        overlay_span = (overlay_start, file_size) if overlay_start < file_size else None
        overlay_len = (file_size - overlay_start) if overlay_span is not None else 0
        payload_segments = _subtract_span([overlay_span] if overlay_span is not None else [], security_span)
        payload_len = sum(end - start for start, end in payload_segments)

        sections = list(getattr(pe, "sections", []) or [])
        last_section_span = _section_raw_span(sections[-1], file_size) if sections else None
        last_section_end = last_section_span[1] if last_section_span is not None else 0
        last_section_entropy = 0.0
        last_raw_virtual_delta = 0.0
        if sections:
            last = sections[-1]
            raw_size = float(getattr(last, "SizeOfRawData", 0) or 0)
            virt_size = float(getattr(last, "Misc_VirtualSize", 0) or 0)
            last_raw_virtual_delta = _safe_ratio(abs(raw_size - virt_size), max(raw_size, virt_size, 1.0))
            last_section_entropy = _entropy_from_bytes(_read_span_prefix(file_path, last_section_span, 4096))

        payload_bytes = _read_segments(file_path, payload_segments)
        payload_entropy = _entropy_from_bytes(payload_bytes) if payload_bytes else 0.0
        if payload_bytes:
            midpoint = max(1, len(payload_bytes) // 2)
            payload_head_entropy = _entropy_from_bytes(payload_bytes[:midpoint])
            payload_tail_entropy = _entropy_from_bytes(payload_bytes[midpoint:])
        else:
            payload_head_entropy = 0.0
            payload_tail_entropy = 0.0

        security_present = 1.0 if security_len > 0 else 0.0
        overlay_present = 1.0 if overlay_len > 0 else 0.0
        payload_present = 1.0 if payload_len > 0 else 0.0
        security_inside_overlay = (
            1.0
            if security_span is not None
            and overlay_span is not None
            and security_span[0] >= overlay_span[0]
            and security_span[1] <= overlay_span[1]
            else 0.0
        )
        security_covers_overlay = (
            1.0
            if security_span is not None
            and overlay_span is not None
            and security_span[0] <= overlay_span[0]
            and security_span[1] >= overlay_span[1]
            else 0.0
        )
        security_starts_at_overlay = (
            1.0
            if security_span is not None and overlay_span is not None and security_span[0] == overlay_span[0]
            else 0.0
        )
        security_ends_at_eof = 1.0 if security_span is not None and security_span[1] >= file_size else 0.0
        payload_after_security = (
            1.0
            if security_span is not None and any(start >= security_span[1] for start, _end in payload_segments)
            else 0.0
        )
        payload_before_security = (
            1.0
            if security_span is not None and any(end <= security_span[0] for _start, end in payload_segments)
            else 0.0
        )
        largest_payload = max((end - start for start, end in payload_segments), default=0)
        first_payload_start = min((start for start, _end in payload_segments), default=file_size)
        gap_last_to_overlay = max(0, overlay_start - last_section_end) if overlay_span is not None else 0
        gap_last_to_security = (
            max(0, security_span[0] - last_section_end) if security_span is not None else 0
        )
        overlay_touches_last = 1.0 if overlay_span is not None and gap_last_to_overlay <= 16 else 0.0
        payload_touches_last = (
            1.0 if payload_present and max(0, first_payload_start - last_section_end) <= 16 else 0.0
        )
        payload_after_cert_size = (
            sum(end - start for start, end in payload_segments if security_span is not None and start >= security_span[1])
            if security_span is not None
            else 0
        )

        features = np.asarray(
            [
                security_present,
                _safe_log1p(security_len),
                _safe_ratio(security_len, file_size),
                overlay_present,
                _safe_log1p(overlay_len),
                _safe_ratio(overlay_len, file_size),
                payload_present,
                _safe_log1p(payload_len),
                _safe_ratio(payload_len, file_size),
                payload_entropy,
                payload_head_entropy,
                payload_tail_entropy,
                abs(payload_head_entropy - payload_tail_entropy),
                1.0 if payload_entropy >= 0.80 else 0.0,
                payload_after_security,
                payload_before_security,
                min(len(payload_segments), 8) / 8.0,
                _safe_ratio(largest_payload, payload_len),
                security_covers_overlay,
                security_inside_overlay,
                security_starts_at_overlay,
                security_ends_at_eof,
                _safe_log1p(gap_last_to_overlay),
                _safe_log1p(gap_last_to_security),
                overlay_touches_last,
                payload_touches_last,
                last_section_entropy,
                payload_entropy - last_section_entropy,
                last_raw_virtual_delta,
                _safe_ratio(max(0, overlay_len - payload_len - security_len), overlay_len),
                1.0 if security_beyond_eof else 0.0,
                _safe_log1p(payload_after_cert_size),
            ],
            dtype=np.float32,
        )
        if features.shape != (len(OVERLAY_BOUNDARY_FEATURE_NAMES),):
            raise ValueError(
                f"Overlay boundary feature length mismatch: {features.shape[0]} != "
                f"{len(OVERLAY_BOUNDARY_FEATURE_NAMES)}"
            )
        return np.nan_to_num(features, copy=False)
    except Exception:
        return np.zeros(len(OVERLAY_BOUNDARY_FEATURE_NAMES), dtype=np.float32)
    finally:
        pe.close()


def _overlay_cache_path(row: dict, cache_dir: Optional[str]) -> Optional[Path]:
    cache_path = content_cache_path_for_row(row, cache_dir)
    if cache_path is None:
        return None
    return cache_path.with_name(f"overlay_boundary_v1_{cache_path.name}")


def overlay_boundary_features_for_row(row: dict, cache_dir: Optional[str]) -> np.ndarray:
    source_path, _source_sha = verify_content_row_source_sha256(row)
    cache_path = _overlay_cache_path(row, cache_dir)
    if cache_path is not None and cache_path.exists():
        features = load_valid_feature_npz(cache_path, len(OVERLAY_BOUNDARY_FEATURE_NAMES))
        if features is not None:
            return features

    features = overlay_boundary_features_from_path(source_path)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_feature_npz_atomic(cache_path, features)
    return features


def build_overlay_boundary_matrix(rows: Sequence[dict], config: OverlayBoundaryConfig) -> np.ndarray:
    if not rows:
        raise ValueError("No overlay boundary rows were loaded")
    matrix = np.empty((len(rows), len(OVERLAY_BOUNDARY_FEATURE_NAMES)), dtype=np.float32)
    for index, row in enumerate(rows):
        matrix[index] = overlay_boundary_features_for_row(row, config.cache_dir)
    return matrix


def _build_overlay_cache_one(payload: tuple[dict, str]) -> dict:
    row, cache_dir = payload
    features = overlay_boundary_features_for_row(row, cache_dir)
    return {"zero": bool(np.count_nonzero(features) == 0)}


def build_overlay_boundary_cache(
    rows: Sequence[dict],
    *,
    cache_dir: Path,
    workers: int,
) -> dict:
    cache_dir = resolve_path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    worker_count = max(1, int(workers))
    if worker_count != 1:
        raise ValueError("Loop55 overlay cache build is intentionally single-process to avoid Windows worker RSS copies")
    counts = {"processed": 0, "zero_features": 0}
    start = time.perf_counter()
    for row in rows:
        result = _build_overlay_cache_one((row, str(cache_dir)))
        counts["processed"] += 1
        counts["zero_features"] += int(result["zero"])
        if counts["processed"] % 1000 == 0:
            print(
                f"[overlay-cache] processed={counts['processed']}/{len(rows)} "
                f"zero={counts['zero_features']}",
                flush=True,
            )
    counts["elapsed_sec"] = time.perf_counter() - start
    counts["cache_dir"] = str(cache_dir)
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Loop55 overlay/security-boundary Stage-2 candidates.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--thresholds", default="0.35:0.65:0.005")
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--chunk-count", type=int, default=16)
    parser.add_argument("--feature-set", choices=["tabular", "extended"], default="extended")
    parser.add_argument("--content-pe-cache-dir", type=Path, required=True)
    parser.add_argument("--overlay-boundary-cache-dir", type=Path, default=None)
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--cache-workers", type=int, default=1)
    parser.add_argument("--cache-report-json", type=Path, default=None)
    parser.add_argument("--noise-modes", default="none,soft_conflict_downweight,trim_extreme_conflict")
    parser.add_argument("--model-candidates", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-val-errors", type=int, default=LOOP28_VAL_ERRORS)
    parser.add_argument("--baseline-val-f1", type=float, default=LOOP28_VAL_F1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = load_safe_checkpoint(resolve_path(args.checkpoint), map_location="cpu")
    checkpoint_config = AxonExperimentConfig.from_dict(dict(checkpoint["config"]))
    del checkpoint
    gc.collect()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_cache_dir = resolve_path(args.overlay_boundary_cache_dir or (output_dir / "overlay_boundary_cache_v1"))

    if args.build_cache_only:
        rows = []
        rows.extend(read_prediction_rows(args.train_predictions, args.max_train_rows))
        rows.extend(read_prediction_rows(args.val_predictions, args.max_val_rows))
        seen = set()
        unique_rows = []
        invalid_source_sha256 = 0
        for row in rows:
            try:
                key = source_sha256_for_row(row)
            except ValueError:
                invalid_source_sha256 += 1
                continue
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(row)
        cache_counts = build_overlay_boundary_cache(
            unique_rows,
            cache_dir=overlay_cache_dir,
            workers=int(args.cache_workers),
        )
        report = {
            "schema": "axon_loop55_overlay_boundary_cache_v1",
            "protocol": "content-only overlay/security-boundary features; filename/path/extension are not encoded",
            "train_predictions": str(resolve_path(args.train_predictions)),
            "val_predictions": str(resolve_path(args.val_predictions)),
            "input_rows": len(rows),
            "unique_rows": len(unique_rows),
            "invalid_source_sha256_rows": invalid_source_sha256,
            "feature_dim": len(OVERLAY_BOUNDARY_FEATURE_NAMES),
            "cache_counts": cache_counts,
        }
        report_path = resolve_path(args.cache_report_json or (output_dir / "loop55_overlay_boundary_cache_report.json"))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    feature_config = FeatureConfig(
        prefix_len=max(0, int(args.prefix_len)),
        chunk_count=max(1, int(args.chunk_count)),
        include_pe=True,
        include_stat=True,
        include_lightweight=args.feature_set == "extended",
        include_byte_summary=args.feature_set == "extended",
        include_content_pe=True,
        content_cache_dir=str(resolve_path(args.content_pe_cache_dir)),
    )
    safe_feature_name_groups = assert_stage2_feature_names_safe(feature_config, checkpoint_config=checkpoint_config)
    assert_no_identity_feature_names(OVERLAY_BOUNDARY_FEATURE_NAMES, context="Loop55 overlay boundary features")

    train_rows = read_prediction_rows(args.train_predictions, args.max_train_rows)
    val_rows = read_prediction_rows(args.val_predictions, args.max_val_rows)
    train_x, train_y, train_base, train_kept_rows, train_counts = build_matrix(
        train_rows, checkpoint_config, feature_config
    )
    val_x, val_y, val_base, val_kept_rows, val_counts = build_matrix(val_rows, checkpoint_config, feature_config)
    del train_rows
    del val_rows

    overlay_config = OverlayBoundaryConfig(cache_dir=str(overlay_cache_dir))
    train_overlay = build_overlay_boundary_matrix(train_kept_rows, overlay_config)
    val_overlay = build_overlay_boundary_matrix(val_kept_rows, overlay_config)
    base_feature_dim = int(train_x.shape[1])
    overlay_boundary_feature_dim = int(train_overlay.shape[1])
    train_overlay_rows = int(train_overlay.shape[0])
    val_overlay_rows = int(val_overlay.shape[0])
    train_present = int((train_overlay[:, 6] > 0).sum())
    val_present = int((val_overlay[:, 6] > 0).sum())
    train_x = append_feature_columns(train_x, train_overlay)
    val_x = append_feature_columns(val_x, val_overlay)
    del train_overlay
    del val_overlay
    gc.collect()
    print(f"[matrix] train={train_x.shape} val={val_x.shape}", flush=True)

    thresholds = parse_thresholds(args.thresholds)
    baseline_val_best = select_best_threshold(val_base, val_y, thresholds)
    candidates = filter_model_candidates(model_candidates(int(args.seed)), args.model_candidates)
    noise_modes = [item.strip() for item in args.noise_modes.split(",") if item.strip()]
    results = []
    best_key = None
    selected = None
    selected_model = None
    selected_val_scores = None
    for noise_mode in noise_modes:
        weights = sample_weights(train_y, train_base, noise_mode)
        weight_summary = summarize_weights(train_y, weights)
        for model_name, model_template in candidates:
            model = clone(model_template)
            start = time.perf_counter()
            fit_kwargs = {}
            if model_name != "logreg_l2_c1":
                fit_kwargs["sample_weight"] = weights
            try:
                model.fit(train_x, train_y, **fit_kwargs)
            except TypeError:
                model.fit(train_x, train_y)
            fit_sec = time.perf_counter() - start
            val_scores = predict_scores(model, val_x)
            val_best = select_best_threshold(val_scores, val_y, thresholds)
            clean_val = clean_slice_metrics(val_scores, val_y, val_base, float(val_best["threshold"]))
            result = {
                "name": f"{model_name}__noise_{noise_mode}",
                "base_model": model_name,
                "noise_mode": noise_mode,
                "fit_sec": fit_sec,
                "effective_train_rows": int(weight_summary["effective_train_rows"]),
                "weight_summary": weight_summary,
                "val_best": val_best,
                "clean_val_at_val_threshold": clean_val,
                "delta_val_errors_vs_loop28": int(val_best["errors"]) - int(args.baseline_val_errors),
                "delta_val_f1_vs_loop28": float(val_best["f1"]) - float(args.baseline_val_f1),
            }
            results.append(result)
            candidate_key = (float(val_best["f1"]), -int(val_best["errors"]))
            if best_key is None or candidate_key > best_key:
                if selected_model is not None:
                    del selected_model
                if selected_val_scores is not None:
                    del selected_val_scores
                best_key = candidate_key
                selected = result
                selected_model = model
                selected_val_scores = val_scores.astype(np.float32, copy=True)
            else:
                del model
            del val_scores
            gc.collect()
            print(
                f"[val] {result['name']} f1={val_best['f1']:.6f} "
                f"errors={val_best['errors']} threshold={val_best['threshold']:.4f}",
                flush=True,
            )

    if selected is None or selected_model is None or selected_val_scores is None:
        raise ValueError("No fitted Loop55 candidate was available for selection")
    selected_threshold = float(selected["val_best"]["threshold"])
    val_predictions_path = output_dir / "loop55_overlay_boundary_val_predictions.csv"
    write_predictions(val_predictions_path, val_kept_rows, val_y, selected_val_scores, selected_threshold)
    model_path = output_dir / "loop55_overlay_boundary_selected_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "schema": "axon_loop55_overlay_boundary_payload_v1",
                "model": selected_model,
                "selected": selected,
                "threshold": selected_threshold,
                "feature_config": feature_config,
                "checkpoint_config": checkpoint_config.to_dict(),
                "overlay_boundary_feature_names": OVERLAY_BOUNDARY_FEATURE_NAMES,
                "identity_feature_policy": (
                    "source_path/source_sha256/cache_path/sample_index/split/filename/extension/directory "
                    "are audit or loading fields only and are forbidden as model features"
                ),
            },
            handle,
        )

    val_kept_count = int(val_counts.get("kept", 0)) if isinstance(val_counts, dict) else int(len(val_y))
    if val_kept_count < 20000:
        test_gate_decision = "smoke_only_not_eligible_for_test10k"
    elif int(selected["val_best"]["errors"]) <= LOOP55_TEST10K_ERROR_GATE:
        test_gate_decision = "eligible_for_test10k"
    else:
        test_gate_decision = "reject_val_margin_too_small"

    report = {
        "schema": "axon_loop55_overlay_boundary_v1",
        "protocol": "train fits overlay/security-boundary Stage-2 candidates; Val selects model and threshold; no Test-10k/full-test used",
        "checkpoint": str(resolve_path(args.checkpoint)),
        "train_predictions": str(resolve_path(args.train_predictions)),
        "val_predictions": str(resolve_path(args.val_predictions)),
        "identity_feature_policy": (
            "filename/path/extension/directory/source hash/sample id/split/row order are audit/loading fields only "
            "and are not model features"
        ),
        "records": {"train": train_counts, "val": val_counts},
        "feature_config": feature_config.__dict__,
        "feature_name_groups": {
            **safe_feature_name_groups,
            "overlay_boundary_feature_names": OVERLAY_BOUNDARY_FEATURE_NAMES,
        },
        "base_feature_dim": base_feature_dim,
        "overlay_boundary_feature_dim": overlay_boundary_feature_dim,
        "feature_dim": int(train_x.shape[1]),
        "overlay_boundary_cache_dir": str(overlay_cache_dir),
        "overlay_boundary_coverage": {
            "train_payload_present": train_present,
            "val_payload_present": val_present,
            "train_payload_zero": int(train_overlay_rows - train_present),
            "val_payload_zero": int(val_overlay_rows - val_present),
        },
        "baseline_val_best": baseline_val_best,
        "loop28_reference": {
            "locked_val_f1": float(args.baseline_val_f1),
            "locked_val_errors": int(args.baseline_val_errors),
        },
        "noise_summary": {
            "train": summarize_noise(train_y, train_base),
            "val": summarize_noise(val_y, val_base),
        },
        "models": sorted(results, key=lambda row: (row["val_best"]["f1"], -row["val_best"]["errors"]), reverse=True),
        "selected_by_val": selected,
        "model_path": str(model_path),
        "val_predictions_csv": str(val_predictions_path),
        "test_ran": False,
        "test10k_error_gate": LOOP55_TEST10K_ERROR_GATE,
        "test_gate_decision": test_gate_decision,
    }
    report_path = output_dir / "loop55_overlay_boundary_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected_by_val": selected, "test_gate_decision": report["test_gate_decision"]}, indent=2))
    print(f"JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
