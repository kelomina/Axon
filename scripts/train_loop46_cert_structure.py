#!/usr/bin/env python3
"""Train Loop46 certificate-structure Stage-2 candidates.

The added features parse the PE Security Directory / WIN_CERTIFICATE blob as
content. Paths and hashes are used only to open files and align caches; they are
not model features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
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
from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402
from train_stage2_cache_matrix import (  # noqa: E402
    FeatureConfig,
    assert_stage2_feature_names_safe,
    build_matrix,
    clean_slice_metrics,
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
    _read_certificate_blob,
)


LOOP28_VAL_F1 = 0.9919048570857486
LOOP28_VAL_ERRORS = 162
LOOP46_TEST10K_ERROR_GATE = 152


CERT_STRUCTURE_OIDS = {
    "pkcs7_data": "1.2.840.113549.1.7.1",
    "pkcs7_signed_data": "1.2.840.113549.1.7.2",
    "spc_indirect_data": "1.3.6.1.4.1.311.2.1.4",
    "spc_statement_type": "1.3.6.1.4.1.311.2.1.11",
    "microsoft_individual_code_signing": "1.3.6.1.4.1.311.2.1.21",
    "microsoft_commercial_code_signing": "1.3.6.1.4.1.311.2.1.22",
    "code_signing": "1.3.6.1.5.5.7.3.3",
    "timestamping": "1.3.6.1.5.5.7.3.8",
    "sha1": "1.3.14.3.2.26",
    "sha256": "2.16.840.1.101.3.4.2.1",
    "sha384": "2.16.840.1.101.3.4.2.2",
    "sha512": "2.16.840.1.101.3.4.2.3",
    "md5": "1.2.840.113549.2.5",
    "rsa_encryption": "1.2.840.113549.1.1.1",
    "sha1_rsa": "1.2.840.113549.1.1.5",
    "sha256_rsa": "1.2.840.113549.1.1.11",
    "sha384_rsa": "1.2.840.113549.1.1.12",
    "sha512_rsa": "1.2.840.113549.1.1.13",
    "ecdsa_sha256": "1.2.840.10045.4.3.2",
    "common_name": "2.5.4.3",
    "country_name": "2.5.4.6",
    "locality_name": "2.5.4.7",
    "state_name": "2.5.4.8",
    "organization_name": "2.5.4.10",
    "organizational_unit_name": "2.5.4.11",
    "email_address": "1.2.840.113549.1.9.1",
    "signing_time": "1.2.840.113549.1.9.5",
    "message_digest": "1.2.840.113549.1.9.4",
}

CERT_STRUCTURE_FEATURE_NAMES = [
    "cert_struct_present",
    "cert_struct_payload_log_size",
    "cert_struct_payload_ratio",
    "cert_struct_root_sequence",
    "cert_struct_parse_ok",
    "cert_struct_malformed_count_log",
    "cert_struct_trailing_ratio",
    "cert_struct_node_count_log",
    "cert_struct_sequence_count_log",
    "cert_struct_set_count_log",
    "cert_struct_oid_count_log",
    "cert_struct_unique_oid_count_log",
    "cert_struct_integer_count_log",
    "cert_struct_octet_string_count_log",
    "cert_struct_bit_string_count_log",
    "cert_struct_context_tag_count_log",
    "cert_struct_string_count_log",
    "cert_struct_printable_string_count_log",
    "cert_struct_utf8_string_count_log",
    "cert_struct_bmp_string_count_log",
    "cert_struct_ia5_string_count_log",
    "cert_struct_utc_time_count_log",
    "cert_struct_generalized_time_count_log",
    "cert_struct_max_depth_norm",
    "cert_struct_constructed_ratio",
    "cert_struct_mean_value_len_norm",
    "cert_struct_max_value_len_ratio",
    "cert_struct_large_octet_count_log",
    "cert_struct_min_year_norm",
    "cert_struct_max_year_norm",
    "cert_struct_year_span_norm",
    "cert_struct_string_total_len_log",
    "cert_struct_string_ascii_ratio",
    "cert_struct_string_digit_ratio",
    "cert_struct_string_dot_ratio",
]
for oid_name in CERT_STRUCTURE_OIDS:
    CERT_STRUCTURE_FEATURE_NAMES.append(f"cert_struct_oid_{oid_name}_present")

assert_no_identity_feature_names(CERT_STRUCTURE_FEATURE_NAMES, context="Loop46 cert structure features")


@dataclass(frozen=True)
class CertStructureConfig:
    cache_dir: Optional[str]


@dataclass(frozen=True)
class Asn1Node:
    tag_class: int
    constructed: bool
    tag: int
    depth: int
    value_start: int
    value_end: int


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _decode_asn1_length(data: bytes, offset: int, end: int) -> tuple[Optional[int], int, bool]:
    if offset >= end:
        return None, offset, False
    first = data[offset]
    offset += 1
    if first < 0x80:
        return int(first), offset, False
    count = first & 0x7F
    if count == 0:
        return None, offset, True
    if count > 4 or offset + count > end:
        return None, offset, False
    length = int.from_bytes(data[offset : offset + count], "big", signed=False)
    return length, offset + count, False


def _decode_oid(value: bytes) -> Optional[str]:
    if not value:
        return None
    first = value[0]
    arcs = [first // 40, first % 40]
    current = 0
    for byte in value[1:]:
        current = (current << 7) | (byte & 0x7F)
        if not byte & 0x80:
            arcs.append(current)
            current = 0
    if current:
        return None
    return ".".join(str(arc) for arc in arcs)


def _parse_asn1_nodes(data: bytes, *, max_nodes: int = 8192) -> tuple[list[Asn1Node], int, int]:
    nodes: list[Asn1Node] = []
    malformed = 0
    trailing = 0

    def parse_range(start: int, end: int, depth: int) -> int:
        nonlocal malformed, trailing
        offset = start
        while offset < end and len(nodes) < max_nodes:
            node_start = offset
            first = data[offset]
            offset += 1
            tag_class = (first >> 6) & 0x03
            constructed = bool(first & 0x20)
            tag = first & 0x1F
            if tag == 0x1F:
                tag = 0
                while offset < end:
                    byte = data[offset]
                    offset += 1
                    tag = (tag << 7) | (byte & 0x7F)
                    if not byte & 0x80:
                        break
                else:
                    malformed += 1
                    return node_start
            length, value_start, indefinite = _decode_asn1_length(data, offset, end)
            if indefinite or length is None:
                malformed += 1
                return node_start
            value_end = value_start + int(length)
            if value_end > end or value_end < value_start:
                malformed += 1
                return node_start
            nodes.append(
                Asn1Node(
                    tag_class=tag_class,
                    constructed=constructed,
                    tag=int(tag),
                    depth=int(depth),
                    value_start=int(value_start),
                    value_end=int(value_end),
                )
            )
            if constructed and value_start < value_end:
                child_end = parse_range(value_start, value_end, depth + 1)
                if child_end < value_end:
                    malformed += 1
            offset = value_end
        if offset < end:
            trailing += end - offset
        return offset

    final_offset = parse_range(0, len(data), 0)
    if final_offset < len(data):
        trailing += len(data) - final_offset
    return nodes, malformed, trailing


def _parse_asn1_year(tag: int, value: bytes) -> Optional[int]:
    try:
        text = value.decode("ascii", errors="strict")
    except UnicodeDecodeError:
        return None
    if tag == 23 and len(text) >= 2 and text[:2].isdigit():
        year = int(text[:2])
        return 1900 + year if year >= 50 else 2000 + year
    if tag == 24 and len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return None


def _string_bytes(tag: int, value: bytes) -> bytes:
    if tag == 30:
        try:
            return value.decode("utf-16be", errors="ignore").encode("utf-8", errors="ignore")
        except Exception:
            return b""
    return value


def cert_structure_features_from_blob(blob: bytes, declared_size: int, file_size: int) -> np.ndarray:
    if not blob:
        return np.zeros(len(CERT_STRUCTURE_FEATURE_NAMES), dtype=np.float32)

    payload = blob[8:] if len(blob) >= 8 else blob
    nodes, malformed, trailing = _parse_asn1_nodes(payload)
    root_sequence = 1.0 if payload[:1] == b"\x30" else 0.0
    parse_ok = 1.0 if nodes and malformed == 0 else 0.0
    value_lengths = [max(0, node.value_end - node.value_start) for node in nodes]
    node_count = len(nodes)
    constructed_count = sum(1 for node in nodes if node.constructed)
    sequence_count = sum(1 for node in nodes if node.tag_class == 0 and node.tag == 16)
    set_count = sum(1 for node in nodes if node.tag_class == 0 and node.tag == 17)
    integer_count = sum(1 for node in nodes if node.tag_class == 0 and node.tag == 2)
    octet_count = sum(1 for node in nodes if node.tag_class == 0 and node.tag == 4)
    bit_count = sum(1 for node in nodes if node.tag_class == 0 and node.tag == 3)
    context_count = sum(1 for node in nodes if node.tag_class == 2)
    string_tags = {12, 19, 20, 22, 28, 30}
    string_nodes = [node for node in nodes if node.tag_class == 0 and node.tag in string_tags]
    years = []
    oids = []
    string_payloads = []
    for node in nodes:
        value = payload[node.value_start : node.value_end]
        if node.tag_class == 0 and node.tag == 6:
            decoded = _decode_oid(value)
            if decoded:
                oids.append(decoded)
        if node.tag_class == 0 and node.tag in (23, 24):
            year = _parse_asn1_year(node.tag, value)
            if year is not None:
                years.append(year)
        if node in string_nodes:
            string_payloads.append(_string_bytes(node.tag, value))

    string_blob = b"\n".join(string_payloads)
    ascii_count = sum(1 for byte in string_blob if 32 <= byte <= 126)
    digit_count = sum(1 for byte in string_blob if 48 <= byte <= 57)
    dot_count = string_blob.count(b".")
    min_year = min(years) if years else 0
    max_year = max(years) if years else 0
    year_span = max_year - min_year if years else 0
    unique_oids = set(oids)
    payload_len = len(payload)
    features = [
        1.0,
        math.log1p(payload_len),
        _safe_ratio(payload_len, file_size),
        root_sequence,
        parse_ok,
        math.log1p(malformed),
        _safe_ratio(trailing, payload_len),
        math.log1p(node_count),
        math.log1p(sequence_count),
        math.log1p(set_count),
        math.log1p(len(oids)),
        math.log1p(len(unique_oids)),
        math.log1p(integer_count),
        math.log1p(octet_count),
        math.log1p(bit_count),
        math.log1p(context_count),
        math.log1p(len(string_nodes)),
        math.log1p(sum(1 for node in string_nodes if node.tag == 19)),
        math.log1p(sum(1 for node in string_nodes if node.tag == 12)),
        math.log1p(sum(1 for node in string_nodes if node.tag == 30)),
        math.log1p(sum(1 for node in string_nodes if node.tag == 22)),
        math.log1p(sum(1 for node in nodes if node.tag_class == 0 and node.tag == 23)),
        math.log1p(sum(1 for node in nodes if node.tag_class == 0 and node.tag == 24)),
        min((max((node.depth for node in nodes), default=0)), 32) / 32.0,
        _safe_ratio(constructed_count, node_count),
        min(float(np.mean(value_lengths)) if value_lengths else 0.0, 65536.0) / 65536.0,
        _safe_ratio(max(value_lengths) if value_lengths else 0, payload_len),
        math.log1p(sum(1 for length in value_lengths if length >= 4096)),
        _safe_ratio(min_year - 1970, 160.0) if min_year else 0.0,
        _safe_ratio(max_year - 1970, 160.0) if max_year else 0.0,
        min(max(year_span, 0), 160) / 160.0,
        math.log1p(len(string_blob)),
        _safe_ratio(ascii_count, len(string_blob)),
        _safe_ratio(digit_count, len(string_blob)),
        _safe_ratio(dot_count, len(string_blob)),
    ]
    for oid in CERT_STRUCTURE_OIDS.values():
        features.append(1.0 if oid in unique_oids else 0.0)

    if len(features) != len(CERT_STRUCTURE_FEATURE_NAMES):
        raise ValueError(
            f"Cert structure feature length mismatch: {len(features)} != {len(CERT_STRUCTURE_FEATURE_NAMES)}"
        )
    return np.nan_to_num(np.asarray(features, dtype=np.float32), copy=False)


def cert_structure_features_from_path(file_path: Path) -> np.ndarray:
    try:
        file_size = file_path.stat().st_size
    except OSError:
        file_size = 0
    blob, declared_size, _revision, _cert_type = _read_certificate_blob(file_path)
    return cert_structure_features_from_blob(blob, declared_size, file_size)


def _cert_structure_cache_path(row: dict, cache_dir: Optional[str]) -> Optional[Path]:
    if not cache_dir:
        return None
    key = (row.get("source_sha256") or "").strip().lower()
    if not key:
        source_path = row.get("source_path", "")
        key = hashlib.sha256(str(resolve_path(Path(source_path))).encode("utf-8", errors="ignore")).hexdigest()
    return resolve_path(Path(cache_dir)) / f"{key}.npz"


def _save_feature_npz_atomic(cache_path: Path, features: np.ndarray) -> None:
    temp_path = cache_path.with_name(f"{cache_path.stem}.{time.time_ns()}.tmp.npz")
    np.savez(temp_path, features=features.astype(np.float32, copy=False))
    temp_path.replace(cache_path)


def cert_structure_features_for_row(row: dict, cache_dir: Optional[str]) -> np.ndarray:
    cache_path = _cert_structure_cache_path(row, cache_dir)
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            features = data["features"].astype(np.float32, copy=False)
        if features.shape == (len(CERT_STRUCTURE_FEATURE_NAMES),):
            return features

    source_path = resolve_path(Path(row["source_path"]))
    features = cert_structure_features_from_path(source_path)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _save_feature_npz_atomic(cache_path, features)
    return features


def build_cert_structure_matrix(rows: Sequence[dict], config: CertStructureConfig) -> np.ndarray:
    features = [cert_structure_features_for_row(row, config.cache_dir) for row in rows]
    if not features:
        raise ValueError("No cert structure rows were loaded")
    return np.vstack(features).astype(np.float32, copy=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Loop46 certificate-structure Stage-2 candidates.")
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
    parser.add_argument("--content-cert-cache-dir", type=Path, required=True)
    parser.add_argument("--cert-structure-cache-dir", type=Path, default=None)
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
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_config = FeatureConfig(
        prefix_len=max(0, int(args.prefix_len)),
        chunk_count=max(1, int(args.chunk_count)),
        include_pe=True,
        include_stat=True,
        include_lightweight=args.feature_set == "extended",
        include_byte_summary=args.feature_set == "extended",
        include_content_pe=True,
        content_cache_dir=str(resolve_path(args.content_pe_cache_dir)),
        include_content_cert=True,
        content_cert_cache_dir=str(resolve_path(args.content_cert_cache_dir)),
    )
    safe_feature_name_groups = assert_stage2_feature_names_safe(feature_config, checkpoint_config=checkpoint_config)
    assert_no_identity_feature_names(CERT_STRUCTURE_FEATURE_NAMES, context="Loop46 cert structure features")

    train_rows = read_prediction_rows(args.train_predictions, args.max_train_rows)
    val_rows = read_prediction_rows(args.val_predictions, args.max_val_rows)
    train_x, train_y, train_base, train_kept_rows, train_counts = build_matrix(
        train_rows, checkpoint_config, feature_config
    )
    val_x, val_y, val_base, val_kept_rows, val_counts = build_matrix(val_rows, checkpoint_config, feature_config)

    cert_structure_cache_dir = resolve_path(args.cert_structure_cache_dir or (output_dir / "cert_structure_cache_v1"))
    structure_config = CertStructureConfig(cache_dir=str(cert_structure_cache_dir))
    train_struct = build_cert_structure_matrix(train_kept_rows, structure_config)
    val_struct = build_cert_structure_matrix(val_kept_rows, structure_config)
    train_x = np.hstack([train_x, train_struct]).astype(np.float32, copy=False)
    val_x = np.hstack([val_x, val_struct]).astype(np.float32, copy=False)
    print(f"[matrix] train={train_x.shape} val={val_x.shape}", flush=True)

    thresholds = parse_thresholds(args.thresholds)
    baseline_val_best = select_best_threshold(val_base, val_y, thresholds)
    candidates = filter_model_candidates(model_candidates(int(args.seed)), args.model_candidates)
    noise_modes = [item.strip() for item in args.noise_modes.split(",") if item.strip()]
    results = []
    fitted = []
    for noise_mode in noise_modes:
        weights = sample_weights(train_y, train_base, noise_mode)
        weight_summary = summarize_weights(train_y, weights)
        for model_name, model in candidates:
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
            fitted.append((float(val_best["f1"]), -int(val_best["errors"]), result, model, val_scores))
            print(
                f"[val] {result['name']} f1={val_best['f1']:.6f} "
                f"errors={val_best['errors']} threshold={val_best['threshold']:.4f}",
                flush=True,
            )

    fitted.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _selected_f1, _neg_errors, selected, selected_model, selected_val_scores = fitted[0]
    selected_threshold = float(selected["val_best"]["threshold"])
    val_predictions_path = output_dir / "loop46_cert_structure_val_predictions.csv"
    write_predictions(val_predictions_path, val_kept_rows, val_y, selected_val_scores, selected_threshold)
    model_path = output_dir / "loop46_cert_structure_selected_model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "schema": "axon_loop46_cert_structure_payload_v1",
                "model": selected_model,
                "selected": selected,
                "threshold": selected_threshold,
                "feature_config": feature_config,
                "checkpoint_config": checkpoint_config.to_dict(),
                "cert_structure_feature_names": CERT_STRUCTURE_FEATURE_NAMES,
                "identity_feature_policy": (
                    "source_path/source_sha256/cache_path/sample_index/split/filename/extension/directory "
                    "are audit or loading fields only and are forbidden as model features"
                ),
            },
            handle,
        )

    signed_train = int((train_struct[:, 0] > 0).sum())
    signed_val = int((val_struct[:, 0] > 0).sum())
    val_kept_count = int(val_counts.get("kept", 0)) if isinstance(val_counts, dict) else int(len(val_y))
    if val_kept_count < 20000:
        test_gate_decision = "smoke_only_not_eligible_for_test10k"
    elif int(selected["val_best"]["errors"]) <= LOOP46_TEST10K_ERROR_GATE:
        test_gate_decision = "eligible_for_test10k"
    else:
        test_gate_decision = "reject_val_margin_too_small"

    report = {
        "schema": "axon_loop46_cert_structure_v1",
        "protocol": "train fits certificate-structure Stage-2 candidates; Val selects model and threshold; no Test-10k/full-test used",
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
            "cert_structure_feature_names": CERT_STRUCTURE_FEATURE_NAMES,
        },
        "base_feature_dim": int(train_x.shape[1] - train_struct.shape[1]),
        "cert_structure_feature_dim": int(train_struct.shape[1]),
        "feature_dim": int(train_x.shape[1]),
        "cert_structure_cache_dir": str(cert_structure_cache_dir),
        "cert_structure_coverage": {
            "train_present": signed_train,
            "val_present": signed_val,
            "train_zero": int(train_struct.shape[0] - signed_train),
            "val_zero": int(val_struct.shape[0] - signed_val),
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
        "test10k_error_gate": LOOP46_TEST10K_ERROR_GATE,
        "test_gate_decision": test_gate_decision,
    }
    report_path = output_dir / "loop46_cert_structure_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected_by_val": selected, "test_gate_decision": report["test_gate_decision"]}, indent=2))
    print(f"JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
