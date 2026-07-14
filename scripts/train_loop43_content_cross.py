#!/usr/bin/env python3
"""Train Loop43 content-cross Stage-2 candidates.

The feature additions here are explicit content-derived interactions over
already cached PE v1/v2 sidecar features. Paths, filenames, extensions, hashes,
sample indexes, split names, and row order are not model inputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from config import AxonExperimentConfig  # noqa: E402
from audit_loop127_content_cross_readiness import audit_loop127_content_cross_readiness  # noqa: E402
from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402
from train_stage2_cache_matrix import (  # noqa: E402
    CONTENT_PE_FEATURE_NAMES,
    CONTENT_PE_V2_FEATURE_NAMES,
    FeatureConfig,
    assert_stage2_feature_names_safe,
    build_matrix,
    clean_slice_metrics,
    append_feature_columns,
    content_pe_features_for_row,
    content_pe_v2_features_for_row,
    filter_model_candidates,
    metrics_at_threshold,
    model_candidates,
    parse_thresholds,
    predict_scores,
    read_prediction_rows,
    resolve_path,
    sample_weights,
    select_best_threshold,
    summarize_noise,
    summarize_weights,
    write_predictions,
)


LOOP28_VAL_F1 = 0.9919048570857486
LOOP28_VAL_ERRORS = 162
LOOP43_TEST10K_ERROR_GATE = 152


def loop43_local_hgb_candidates(seed: int) -> list[tuple[str, object]]:
    """Local HGB refinements for the content-cross candidate.

    These names are intentionally kept out of the shared Stage-2 candidate pool
    so other experiments do not become slower by default.
    """

    specs = [
        ("hgb_l43_lr0.05_leaf31_iter320_l2_0", 0.05, 31, 320, 0.0),
        ("hgb_l43_lr0.06_leaf23_iter320_l2_0", 0.06, 23, 320, 0.0),
        ("hgb_l43_lr0.06_leaf31_iter340_l2_0", 0.06, 31, 340, 0.0),
        ("hgb_l43_lr0.06_leaf47_iter260_l2_0", 0.06, 47, 260, 0.0),
        ("hgb_l43_lr0.07_leaf31_iter280_l2_1e-4", 0.07, 31, 280, 1.0e-4),
        ("hgb_l43_lr0.07_leaf47_iter260_l2_1e-4", 0.07, 47, 260, 1.0e-4),
        ("hgb_l43_lr0.08_leaf23_iter280_l2_1e-3", 0.08, 23, 280, 1.0e-3),
        ("hgb_l43_lr0.08_leaf47_iter240_l2_1e-3", 0.08, 47, 240, 1.0e-3),
    ]
    return [
        (
            name,
            HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_leaf_nodes=max_leaf_nodes,
                max_iter=max_iter,
                l2_regularization=l2_regularization,
                random_state=seed,
            ),
        )
        for name, learning_rate, max_leaf_nodes, max_iter, l2_regularization in specs
    ]


CONTENT_CROSS_FEATURE_NAMES = [
    # DLL/driver-like content profile.
    "cross_dll_export_log",
    "cross_dll_security_present",
    "cross_dll_security_log_size",
    "cross_dll_overlay_present",
    "cross_dll_overlay_entropy",
    "cross_dll_exception_present",
    "cross_dll_debug_present",
    "cross_dll_tls_present",
    "cross_dll_large_address_aware",
    "cross_dll_driver_api_present",
    "cross_dll_service_api_present",
    "cross_dll_driver_export_pattern",
    "cross_dll_service_export_pattern",
    "cross_dll_native_subsystem_like",
    # Security directory and overlay disentangling.
    "cross_security_overlay_present",
    "cross_security_overlay_log_size",
    "cross_security_overlay_entropy",
    "cross_security_export_log",
    "cross_security_exception_present",
    "cross_security_debug_present",
    "cross_unsigned_overlay_log_size",
    "cross_unsigned_overlay_entropy",
    "cross_unsigned_high_entropy_section",
    "cross_overlay_entropy_ratio",
    "cross_overlay_last_section_entropy",
    "cross_overlay_ep_section_entropy",
    "cross_overlay_resource_weak",
    # Section, entrypoint, entropy, and packer-like structure.
    "cross_exec_write_high_entropy",
    "cross_exec_zero_raw_high_entropy",
    "cross_write_zero_raw_high_entropy",
    "cross_ep_write_high_entropy",
    "cross_ep_entropy_overlay",
    "cross_ep_raw_virtual_delta_overlay",
    "cross_last_section_entropy_overlay",
    "cross_packer_rwx",
    "cross_packer_zero_raw",
    "cross_packer_overlay",
    "cross_raw_virtual_overlay",
    "cross_virtual_raw_overlay",
    # Import/API semantic combinations.
    "cross_system_dll_high_import",
    "cross_system_dll_network",
    "cross_system_dll_filesystem",
    "cross_system_dll_registry",
    "cross_system_dll_injection",
    "cross_network_overlay_entropy",
    "cross_filesystem_overlay_entropy",
    "cross_registry_overlay_entropy",
    "cross_injection_exec_write",
    "cross_process_exec_write",
    "cross_driver_api_export",
    "cross_driver_api_security",
    "cross_driver_api_overlay",
    "cross_service_api_export",
    "cross_service_api_security",
    "cross_service_api_resource_weak",
    "cross_crypto_cert_security",
    "cross_crypto_cert_overlay",
    # Resource/export/version-ish structure using content-only parsed PE fields.
    "cross_export_resource_weak",
    "cross_export_overlay",
    "cross_export_security",
    "cross_resource_low_signed",
    "cross_resource_low_overlay",
    "cross_resource_high_entropy",
    "cross_resource_icon_overlay",
    "cross_resource_version_signed",
    "cross_manifest_signed_overlay",
]

assert_no_identity_feature_names(CONTENT_CROSS_FEATURE_NAMES, context="Loop43 content-cross features")


@dataclass(frozen=True)
class CrossConfig:
    content_pe_cache_dir: Optional[str]
    content_pe_v2_cache_dir: Optional[str]


def _feature_lookup(names: Sequence[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(names)}


PE1 = _feature_lookup(CONTENT_PE_FEATURE_NAMES)
PE2 = _feature_lookup(CONTENT_PE_V2_FEATURE_NAMES)


def _v(values: np.ndarray, mapping: dict[str, int], name: str) -> float:
    index = mapping.get(name)
    if index is None:
        return 0.0
    return float(values[index])


def _present(value: float, threshold: float = 0.0) -> float:
    return 1.0 if float(value) > threshold else 0.0


def _inv_present(value: float, threshold: float = 0.0) -> float:
    return 1.0 if float(value) <= threshold else 0.0


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def content_cross_features_from_arrays(pe1: np.ndarray, pe2: np.ndarray) -> np.ndarray:
    """Return content-only cross features from existing PE sidecar arrays."""

    is_dll = _clip01(_v(pe1, PE1, "content_is_dll"))
    security_present = _present(_v(pe1, PE1, "content_dir_security_present"))
    security_log = _v(pe1, PE1, "content_dir_security_log_size")
    overlay_present = _present(_v(pe1, PE1, "content_overlay_present"))
    overlay_log = _v(pe1, PE1, "content_overlay_log_size")
    overlay_entropy = _v(pe1, PE1, "content_overlay_entropy")
    overlay_ratio = _v(pe1, PE1, "content_overlay_ratio")
    export_log = _v(pe1, PE1, "content_export_count_log")
    export_name_ratio = _v(pe1, PE1, "content_export_name_ratio")
    exception_present = _present(_v(pe1, PE1, "content_dir_exception_present"))
    debug_present = _present(_v(pe1, PE1, "content_dir_debug_present"))
    tls_present = _present(_v(pe1, PE1, "content_dir_tls_present"))
    large_address = _clip01(_v(pe1, PE1, "content_large_address_aware"))
    subsystem = _v(pe1, PE1, "content_subsystem")
    high_entropy_section_ratio = _v(pe1, PE1, "content_section_high_entropy_ratio")
    rwx_ratio = _v(pe1, PE1, "content_section_combo_rwx_ratio")
    rw_ratio = _v(pe1, PE1, "content_section_combo_rw_ratio")
    zero_raw_ratio = _v(pe1, PE1, "content_section_zero_raw_ratio")
    raw_virtual_mismatch = _v(pe1, PE1, "content_section_raw_virtual_mismatch_ratio")
    packer_ratio = _v(pe1, PE1, "content_section_name_packer_hit_ratio")
    system_dll_ratio = _v(pe1, PE1, "content_system_dll_ratio")
    import_api_log = _v(pe1, PE1, "content_import_api_count_log")
    import_dll_log = _v(pe1, PE1, "content_import_dll_count_log")
    network_ratio = _v(pe1, PE1, "content_api_network_ratio")
    filesystem_ratio = _v(pe1, PE1, "content_api_filesystem_ratio")
    registry_ratio = _v(pe1, PE1, "content_api_registry_ratio")
    injection_ratio = _v(pe1, PE1, "content_api_injection_ratio")
    crypto_ratio = _v(pe1, PE1, "content_api_crypto_ratio")
    resource_log = _v(pe1, PE1, "content_resource_entry_count_log")
    resource_type_log = _v(pe1, PE1, "content_resource_type_count_log")

    driver_api = _present(_v(pe2, PE2, "v2_api_driver_present"))
    service_api = _present(_v(pe2, PE2, "v2_api_service_present"))
    process_api = _present(_v(pe2, PE2, "v2_api_process_enum_present")) or _present(
        _v(pe2, PE2, "v2_api_memory_present")
    )
    crypto_cert_api = _present(_v(pe2, PE2, "v2_api_crypto_cert_present"))
    resource_api = _present(_v(pe2, PE2, "v2_api_resource_present"))
    driver_count = _v(pe2, PE2, "v2_api_driver_count_log")
    service_count = _v(pe2, PE2, "v2_api_service_count_log")
    export_service_pattern = _present(_v(pe2, PE2, "v2_export_pattern_service_present"))
    export_plugin_pattern = _present(_v(pe2, PE2, "v2_export_pattern_plugin_present"))
    resource_icon = _present(_v(pe2, PE2, "v2_resource_type_icon_present"))
    resource_version = _present(_v(pe2, PE2, "v2_resource_type_version_present"))
    resource_manifest = _present(_v(pe2, PE2, "v2_resource_type_manifest_present"))
    resource_max_entropy = _v(pe2, PE2, "v2_resource_max_entropy")
    exec_write_log = _v(pe2, PE2, "v2_section_exec_write_count_log")
    exec_high_entropy = _v(pe2, PE2, "v2_section_exec_high_entropy_ratio")
    write_high_entropy = _v(pe2, PE2, "v2_section_write_high_entropy_ratio")
    zero_raw_exec = _v(pe2, PE2, "v2_section_zero_raw_exec_ratio")
    zero_raw_write = _v(pe2, PE2, "v2_section_zero_raw_write_ratio")
    max_raw_virtual_delta = _v(pe2, PE2, "v2_section_max_raw_virtual_delta")
    max_virtual_raw_log = _v(pe2, PE2, "v2_section_max_virtual_raw_ratio_log")
    ep_write = _present(_v(pe2, PE2, "v2_ep_in_write_section"))
    ep_entropy = _v(pe2, PE2, "v2_ep_section_entropy")
    ep_raw_virtual_delta = _v(pe2, PE2, "v2_ep_section_raw_virtual_delta")
    last_section_entropy = _v(pe2, PE2, "v2_last_section_entropy")

    resource_weak = _inv_present(resource_log) * _inv_present(resource_type_log)
    native_subsystem_like = 1.0 if subsystem <= 0.02 or driver_api > 0.0 else 0.0
    export_like = _present(export_log) or export_service_pattern or export_plugin_pattern

    features = np.asarray(
        [
            is_dll * export_log,
            is_dll * security_present,
            is_dll * security_log,
            is_dll * overlay_present,
            is_dll * overlay_entropy,
            is_dll * exception_present,
            is_dll * debug_present,
            is_dll * tls_present,
            is_dll * large_address,
            is_dll * driver_api,
            is_dll * service_api,
            is_dll * driver_api * float(export_like),
            is_dll * service_api * export_service_pattern,
            is_dll * native_subsystem_like,
            security_present * overlay_present,
            security_present * overlay_log,
            security_present * overlay_entropy,
            security_present * export_log,
            security_present * exception_present,
            security_present * debug_present,
            (1.0 - security_present) * overlay_log,
            (1.0 - security_present) * overlay_entropy,
            (1.0 - security_present) * high_entropy_section_ratio,
            overlay_entropy * overlay_ratio,
            overlay_entropy * last_section_entropy,
            overlay_entropy * ep_entropy,
            overlay_present * resource_weak,
            exec_write_log * max(exec_high_entropy, write_high_entropy),
            zero_raw_exec * high_entropy_section_ratio,
            zero_raw_write * high_entropy_section_ratio,
            ep_write * ep_entropy,
            ep_entropy * overlay_entropy,
            ep_raw_virtual_delta * overlay_present,
            last_section_entropy * overlay_entropy,
            packer_ratio * rwx_ratio,
            packer_ratio * zero_raw_ratio,
            packer_ratio * overlay_present,
            raw_virtual_mismatch * overlay_present,
            max_virtual_raw_log * overlay_present,
            system_dll_ratio * import_api_log,
            system_dll_ratio * network_ratio,
            system_dll_ratio * filesystem_ratio,
            system_dll_ratio * registry_ratio,
            system_dll_ratio * injection_ratio,
            network_ratio * overlay_entropy,
            filesystem_ratio * overlay_entropy,
            registry_ratio * overlay_entropy,
            injection_ratio * exec_write_log,
            float(process_api) * exec_write_log,
            driver_count * export_log,
            driver_api * security_present,
            driver_api * overlay_present,
            service_count * export_log,
            service_api * security_present,
            service_api * resource_weak,
            crypto_cert_api * security_present,
            crypto_ratio * overlay_present,
            export_log * resource_weak,
            export_name_ratio * overlay_present,
            export_name_ratio * security_present,
            resource_weak * security_present,
            resource_weak * overlay_present,
            resource_max_entropy * resource_api,
            resource_icon * overlay_present,
            resource_version * security_present,
            resource_manifest * security_present * overlay_present,
        ],
        dtype=np.float32,
    )
    if features.shape != (len(CONTENT_CROSS_FEATURE_NAMES),):
        raise ValueError(
            f"Content cross feature length mismatch: {features.shape[0]} != {len(CONTENT_CROSS_FEATURE_NAMES)}"
        )
    return np.nan_to_num(features, copy=False)


def build_content_cross_matrix(rows: Sequence[dict], config: CrossConfig) -> np.ndarray:
    if not rows:
        raise ValueError("No content cross rows were loaded")
    matrix = np.empty((len(rows), len(CONTENT_CROSS_FEATURE_NAMES)), dtype=np.float32)
    for index, row in enumerate(rows):
        pe1 = content_pe_features_for_row(row, config.content_pe_cache_dir)
        pe2 = content_pe_v2_features_for_row(row, config.content_pe_v2_cache_dir)
        matrix[index] = content_cross_features_from_arrays(pe1, pe2)
    return matrix


def run_strict_readiness_preflight(args: argparse.Namespace, output_dir: Path) -> dict:
    """Block training before matrix build if strict Loop127 inputs are not ready."""

    report_path = output_dir / "loop43_content_cross_preflight.json"
    payload = audit_loop127_content_cross_readiness(
        train_predictions=args.train_predictions,
        val_predictions=args.val_predictions,
        content_pe_cache_dir=args.content_pe_cache_dir,
        content_pe_v2_cache_dir=args.content_pe_v2_cache_dir,
        output_json=report_path,
        expected_train_rows=int(args.expected_train_rows),
        expected_val_rows=int(args.expected_val_rows),
        expected_test_rows=int(args.expected_test_rows),
        expected_total_rows=int(args.expected_total_rows),
        validate_npz_contents=True,
    )
    if not payload.get("ready_for_loop43_val_only"):
        blockers = ", ".join(payload.get("blockers", [])) or "unknown"
        raise RuntimeError(f"Loop43 content-cross preflight blocked training: {blockers}. See {report_path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Loop43 content-cross Stage-2 candidates.")
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
    parser.add_argument("--content-pe-v2-cache-dir", type=Path, required=True)
    parser.add_argument("--noise-modes", default="none,soft_conflict_downweight,trim_extreme_conflict")
    parser.add_argument("--model-candidates", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-val-errors", type=int, default=LOOP28_VAL_ERRORS)
    parser.add_argument("--baseline-val-f1", type=float, default=LOOP28_VAL_F1)
    parser.add_argument("--expected-train-rows", type=int, default=20000)
    parser.add_argument("--expected-val-rows", type=int, default=20000)
    parser.add_argument("--expected-test-rows", type=int, default=160000)
    parser.add_argument("--expected-total-rows", type=int, default=200000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = run_strict_readiness_preflight(args, output_dir)

    checkpoint = load_safe_checkpoint(resolve_path(args.checkpoint), map_location="cpu")
    checkpoint_config = AxonExperimentConfig.from_dict(dict(checkpoint["config"]))
    del checkpoint
    gc.collect()

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
    assert_no_identity_feature_names(CONTENT_CROSS_FEATURE_NAMES, context="Loop43 content-cross features")

    train_rows = read_prediction_rows(args.train_predictions, args.max_train_rows)
    val_rows = read_prediction_rows(args.val_predictions, args.max_val_rows)
    train_x, train_y, train_base, train_kept_rows, train_counts = build_matrix(
        train_rows, checkpoint_config, feature_config
    )
    val_x, val_y, val_base, val_kept_rows, val_counts = build_matrix(val_rows, checkpoint_config, feature_config)
    del train_rows
    del val_rows

    cross_config = CrossConfig(
        content_pe_cache_dir=str(resolve_path(args.content_pe_cache_dir)),
        content_pe_v2_cache_dir=str(resolve_path(args.content_pe_v2_cache_dir)),
    )
    train_cross = build_content_cross_matrix(train_kept_rows, cross_config)
    val_cross = build_content_cross_matrix(val_kept_rows, cross_config)
    base_feature_dim = int(train_x.shape[1])
    content_cross_feature_dim = int(train_cross.shape[1])
    train_x = append_feature_columns(train_x, train_cross)
    val_x = append_feature_columns(val_x, val_cross)
    del train_cross
    del val_cross
    gc.collect()
    print(f"[matrix] train={train_x.shape} val={val_x.shape}", flush=True)

    thresholds = parse_thresholds(args.thresholds)
    baseline_val_best = select_best_threshold(val_base, val_y, thresholds)
    candidates = filter_model_candidates(
        model_candidates(int(args.seed)) + loop43_local_hgb_candidates(int(args.seed)),
        args.model_candidates,
    )
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
        raise ValueError("No fitted Loop43 candidate was available for selection")
    selected_threshold = float(selected["val_best"]["threshold"])
    val_predictions_path = output_dir / "loop43_content_cross_val_predictions.csv"
    write_predictions(val_predictions_path, val_kept_rows, val_y, selected_val_scores, selected_threshold)
    model_path = output_dir / "loop43_content_cross_selected_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "schema": "axon_loop43_content_cross_payload_v1",
                "model": selected_model,
                "selected": selected,
                "threshold": selected_threshold,
                "feature_config": feature_config,
                "checkpoint_config": checkpoint_config.to_dict(),
                "content_cross_feature_names": CONTENT_CROSS_FEATURE_NAMES,
                "identity_feature_policy": (
                    "source_path/source_sha256/cache_path/sample_index/split/filename/extension/directory "
                    "are audit or loading fields only and are forbidden as model features"
                ),
            },
            handle,
        )

    report = {
        "schema": "axon_loop43_content_cross_v1",
        "protocol": "train fits content-cross Stage-2 candidates; Val selects model and threshold; no Test-10k/full-test used",
        "checkpoint": str(resolve_path(args.checkpoint)),
        "train_predictions": str(resolve_path(args.train_predictions)),
        "val_predictions": str(resolve_path(args.val_predictions)),
        "identity_feature_policy": (
            "filename/path/extension/directory/source hash/sample id/split/row order are audit/loading fields only "
            "and are not model features"
        ),
        "records": {"train": train_counts, "val": val_counts},
        "feature_config": feature_config.__dict__,
        "readiness_preflight": {
            "ready_for_loop43_val_only": bool(preflight.get("ready_for_loop43_val_only")),
            "report_json": str(output_dir / "loop43_content_cross_preflight.json"),
            "split_contract": preflight.get("split_contract", {}),
        },
        "feature_name_groups": {
            **safe_feature_name_groups,
            "content_cross_feature_names": CONTENT_CROSS_FEATURE_NAMES,
        },
        "base_feature_dim": base_feature_dim,
        "content_cross_feature_dim": content_cross_feature_dim,
        "feature_dim": int(train_x.shape[1]),
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
        "test10k_error_gate": LOOP43_TEST10K_ERROR_GATE,
        "test_gate_decision": (
            "eligible_for_test10k"
            if int(selected["val_best"]["errors"]) <= LOOP43_TEST10K_ERROR_GATE
            else "reject_val_margin_too_small"
        ),
    }
    report_path = output_dir / "loop43_content_cross_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected_by_val": selected, "test_gate_decision": report["test_gate_decision"]}, indent=2))
    print(f"JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
