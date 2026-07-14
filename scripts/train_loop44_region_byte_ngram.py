#!/usr/bin/env python3
"""Train a regionized byte n-gram SGD probe.

Loop44 is a validation-first experiment. It uses paths only to open the binary
content and to align rows with the fixed cache manifest. Filename, extension,
directory text, path text, hashes, sample ids, split names, and row order are
not model features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle as sklearn_shuffle

try:
    import pefile

    PEFILE_AVAILABLE = True
except ImportError:
    pefile = None
    PEFILE_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"
for item in (PROJECT_ROOT, SCRIPTS_DIR, SRC_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from dataset import _load_cached_feature_npz  # noqa: E402
from identity_feature_guard import assert_no_identity_feature_names  # noqa: E402
from train_byte_ngram_sgd import (  # noqa: E402
    load_checkpoint_config,
    load_records,
    metrics_at_threshold,
    parse_float_list,
    parse_thresholds,
    resolve_path,
    select_threshold,
    sigmoid,
    write_predictions,
    _hashed_byte_hist_features,
    _hashed_dense_features,
    _hashed_ngram_features,
    _hashed_position_byte_features,
)


LOOP28_VAL_ERRORS = 162
LOOP44_TEST10K_ERROR_GATE = 152

REGION_NAMES = [
    "head",
    "tail",
    "entrypoint",
    "overlay_payload",
    "security_directory",
    "resource_directory",
    "import_directory",
    "export_directory",
    "first_exec_section",
    "last_section",
    "max_entropy_section",
]

REGION_SCALAR_FEATURE_NAMES = [
    f"region_{region_name}_{suffix}"
    for region_name in REGION_NAMES
    for suffix in ("present", "log_size", "entropy")
]
assert_no_identity_feature_names(REGION_SCALAR_FEATURE_NAMES, context="Loop44 region scalar features")


@dataclass(frozen=True)
class RegionSlice:
    name: str
    start: int
    size: int


@dataclass(frozen=True)
class RegionHashConfig:
    n_features: int
    prefix_len: int
    ngram_min: int
    ngram_max: int
    ngram_stride: int
    include_prefix_features: bool
    include_full_ngram_features: bool
    include_region_ngram_features: bool
    include_region_scalar_features: bool
    include_byte_hist: bool
    include_cache_features: bool
    region_window: int
    tail_window: int
    max_byte_length: int
    pe_feature_dim: int
    stat_feature_dim: int
    lightweight_feature_dim: int


def _entropy_from_bytes(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float64)
    probs = counts[counts > 0] / float(len(data))
    return float(-(probs * np.log2(probs)).sum() / 8.0)


def _read_region(file_path: Path, start: int, size: int) -> bytes:
    if size <= 0 or start < 0:
        return b""
    try:
        file_size = file_path.stat().st_size
        if start >= file_size:
            return b""
        with file_path.open("rb") as handle:
            handle.seek(max(0, int(start)))
            return handle.read(max(0, min(int(size), file_size - int(start))))
    except OSError:
        return b""


def _stable_col(name: str, n_features: int) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % int(n_features)


def _region_salt(region_name: str, n_features: int) -> np.uint64:
    return np.uint64(_stable_col(f"loop44_region_salt::{region_name}", n_features))


def _valid_slice(name: str, start: int, size: int, file_size: int, window: int) -> Optional[RegionSlice]:
    if file_size <= 0 or size <= 0:
        return None
    start = max(0, min(int(start), max(file_size - 1, 0)))
    size = max(0, min(int(size), int(window), file_size - start))
    if size <= 0:
        return None
    return RegionSlice(name=name, start=start, size=size)


def _rva_region(pe, name: str, rva: int, size: int, file_size: int, window: int) -> Optional[RegionSlice]:
    if int(rva or 0) <= 0 or int(size or 0) <= 0:
        return None
    try:
        offset = int(pe.get_offset_from_rva(int(rva)))
    except Exception:
        return None
    return _valid_slice(name, offset, int(size), file_size, window)


def _section_region(name: str, section, file_size: int, window: int) -> Optional[RegionSlice]:
    start = int(getattr(section, "PointerToRawData", 0) or 0)
    size = int(getattr(section, "SizeOfRawData", 0) or 0)
    return _valid_slice(name, start, size, file_size, window)


def _overlay_payload_region(
    overlay_offset: Optional[int],
    file_size: int,
    window: int,
    security_span: Optional[tuple[int, int]],
) -> Optional[RegionSlice]:
    if overlay_offset is None:
        return None
    overlay_start = max(0, min(int(overlay_offset), file_size))
    segments = [(overlay_start, file_size)]
    if security_span is not None:
        security_start, security_end = security_span
        security_start = max(overlay_start, min(int(security_start), file_size))
        security_end = max(security_start, min(int(security_end), file_size))
        next_segments = []
        for start, end in segments:
            if security_start > start:
                next_segments.append((start, min(security_start, end)))
            if security_end < end:
                next_segments.append((max(security_end, start), end))
        segments = [(start, end) for start, end in next_segments if end > start]
    if not segments:
        return None
    start, end = max(segments, key=lambda item: item[1] - item[0])
    return _valid_slice("overlay_payload", start, end - start, file_size, window)


def _section_entropy(section) -> float:
    size = int(getattr(section, "SizeOfRawData", 0) or 0)
    if size <= 0:
        return 0.0
    try:
        return _entropy_from_bytes(section.get_data()[:4096])
    except Exception:
        return 0.0


def region_slices_from_path(file_path: Path, *, region_window: int, tail_window: int) -> list[RegionSlice]:
    """Return fixed semantic binary regions derived only from file content."""

    try:
        file_size = file_path.stat().st_size
    except OSError:
        return []
    if file_size <= 0:
        return []

    regions: list[RegionSlice] = []
    head = _valid_slice("head", 0, min(region_window, file_size), file_size, region_window)
    if head is not None:
        regions.append(head)
    tail_start = max(file_size - int(tail_window), 0)
    tail = _valid_slice("tail", tail_start, min(tail_window, file_size), file_size, tail_window)
    if tail is not None:
        regions.append(tail)

    if not PEFILE_AVAILABLE:
        return regions

    try:
        pe = pefile.PE(str(file_path), fast_load=True)
    except Exception:
        return regions

    try:
        optional = getattr(pe, "OPTIONAL_HEADER", None)
        security_span: Optional[tuple[int, int]] = None
        if optional is not None:
            entry_rva = int(getattr(optional, "AddressOfEntryPoint", 0) or 0)
            if entry_rva > 0:
                try:
                    entry_offset = int(pe.get_offset_from_rva(entry_rva))
                    entry_start = max(entry_offset - region_window // 4, 0)
                    entry = _valid_slice("entrypoint", entry_start, region_window, file_size, region_window)
                    if entry is not None:
                        regions.append(entry)
                except Exception:
                    pass

            directories = getattr(optional, "DATA_DIRECTORY", []) or []
            directory_map = {
                "export_directory": 0,
                "import_directory": 1,
                "resource_directory": 2,
            }
            for region_name, directory_index in directory_map.items():
                if len(directories) <= directory_index:
                    continue
                directory = directories[directory_index]
                directory_region = _rva_region(
                    pe,
                    region_name,
                    int(getattr(directory, "VirtualAddress", 0) or 0),
                    int(getattr(directory, "Size", 0) or 0),
                    file_size,
                    region_window,
                )
                if directory_region is not None:
                    regions.append(directory_region)

            security_index = 4
            if len(directories) > security_index:
                security = directories[security_index]
                security_start = int(getattr(security, "VirtualAddress", 0) or 0)
                security_size = int(getattr(security, "Size", 0) or 0)
                security_region = _valid_slice(
                    "security_directory",
                    security_start,
                    security_size,
                    file_size,
                    region_window,
                )
                if security_region is not None:
                    regions.append(security_region)
                    security_span = (
                        security_start,
                        min(file_size, security_start + max(0, security_size)),
                    )

        overlay_offset = pe.get_overlay_data_start_offset()
        overlay = _overlay_payload_region(
            int(overlay_offset) if overlay_offset is not None else None,
            file_size,
            region_window,
            security_span,
        )
        if overlay is not None:
            regions.append(overlay)

        sections = list(getattr(pe, "sections", []) or [])
        if sections:
            last = _section_region("last_section", sections[-1], file_size, region_window)
            if last is not None:
                regions.append(last)

            exec_sections = [
                section
                for section in sections
                if int(getattr(section, "Characteristics", 0) or 0) & 0x20000000
            ]
            if exec_sections:
                first_exec = _section_region("first_exec_section", exec_sections[0], file_size, region_window)
                if first_exec is not None:
                    regions.append(first_exec)

            entropy_section = max(sections, key=_section_entropy)
            max_entropy = _section_region("max_entropy_section", entropy_section, file_size, region_window)
            if max_entropy is not None:
                regions.append(max_entropy)
    finally:
        pe.close()

    known = set(REGION_NAMES)
    return [region for region in regions if region.name in known]


def _record_source_path(record: dict) -> Optional[Path]:
    for key in ("source_path", "original_source_path"):
        text = str(record.get(key) or "").strip()
        if not text:
            continue
        path = resolve_path(Path(text))
        if path.exists():
            return path
    return None


def region_payloads_for_record(record: dict, config: RegionHashConfig, *, allow_missing_source: bool) -> list[tuple[str, bytes]]:
    cached = record.get("_loop44_region_payloads")
    if cached is not None:
        return cached

    source_path = _record_source_path(record)
    if source_path is None:
        if allow_missing_source:
            record["_loop44_region_payloads"] = []
            return []
        raise FileNotFoundError(f"Cannot open source content for sample_index={record.get('sample_index', '')}")

    payloads = []
    for region in region_slices_from_path(source_path, region_window=config.region_window, tail_window=config.tail_window):
        data = _read_region(source_path, region.start, region.size)
        if data:
            payloads.append((region.name, data))
    if not payloads and not allow_missing_source:
        raise ValueError(f"No readable content regions for sample_index={record.get('sample_index', '')}")
    record["_loop44_region_payloads"] = payloads
    return payloads


def _hashed_region_ngram_features(
    payloads: Sequence[tuple[str, bytes]],
    *,
    ngram_min: int,
    ngram_max: int,
    stride: int,
    n_features: int,
) -> np.ndarray:
    parts = []
    modulus = np.uint64(n_features)
    for region_name, data in payloads:
        if not data:
            continue
        arr = np.frombuffer(data, dtype=np.uint8)
        cols = _hashed_ngram_features(arr, ngram_min, ngram_max, stride, n_features).astype(np.uint64, copy=False)
        if cols.size:
            cols = (cols + _region_salt(region_name, n_features)) % modulus
            parts.append(cols.astype(np.int64, copy=False))
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def _hashed_region_scalar_features(payloads: Sequence[tuple[str, bytes]], n_features: int) -> tuple[np.ndarray, np.ndarray]:
    payload_by_name = {name: data for name, data in payloads}
    cols = []
    values = []
    for region_name in REGION_NAMES:
        data = payload_by_name.get(region_name, b"")
        stats = {
            "present": 1.0 if data else 0.0,
            "log_size": math.log1p(float(len(data))) if data else 0.0,
            "entropy": _entropy_from_bytes(data) if data else 0.0,
        }
        for suffix, value in stats.items():
            if value == 0.0:
                continue
            cols.append(_stable_col(f"region::{region_name}::{suffix}", n_features))
            values.append(float(value))
    if not cols:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    return np.asarray(cols, dtype=np.int64), np.asarray(values, dtype=np.float32)


def transform_batch(
    records: Sequence[dict],
    config: RegionHashConfig,
    *,
    allow_missing_source: bool,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    row_indices: list[np.ndarray] = []
    col_indices: list[np.ndarray] = []
    data_values: list[np.ndarray] = []
    labels = np.empty(len(records), dtype=np.int64)

    for row_idx, record in enumerate(records):
        byte_seq, pe_feat, stat_feat, lightweight_feat, label = _load_cached_feature_npz(
            Path(record["cache_path"]),
            config.max_byte_length,
            config.pe_feature_dim,
            config.stat_feature_dim,
            config.lightweight_feature_dim,
            expected_label=int(record["label"]),
            expected_source_sha256=record.get("source_sha256") or None,
            allow_missing_source_sha256=False,
        )
        labels[row_idx] = label

        cols_parts: list[np.ndarray] = []
        values_parts: list[np.ndarray] = []
        if config.include_prefix_features:
            prefix_cols = _hashed_position_byte_features(byte_seq, config.prefix_len, config.n_features)
            cols_parts.append(prefix_cols)
            values_parts.append(np.ones(prefix_cols.shape[0], dtype=np.float32))
        if config.include_full_ngram_features:
            full_cols = _hashed_ngram_features(
                byte_seq,
                config.ngram_min,
                config.ngram_max,
                config.ngram_stride,
                config.n_features,
            )
            cols_parts.append(full_cols)
            values_parts.append(np.ones(full_cols.shape[0], dtype=np.float32))

        payloads = region_payloads_for_record(record, config, allow_missing_source=allow_missing_source)
        if config.include_region_ngram_features:
            region_cols = _hashed_region_ngram_features(
                payloads,
                ngram_min=config.ngram_min,
                ngram_max=config.ngram_max,
                stride=config.ngram_stride,
                n_features=config.n_features,
            )
            cols_parts.append(region_cols)
            values_parts.append(np.ones(region_cols.shape[0], dtype=np.float32))
        if config.include_region_scalar_features:
            region_scalar_cols, region_scalar_values = _hashed_region_scalar_features(payloads, config.n_features)
            cols_parts.append(region_scalar_cols)
            values_parts.append(region_scalar_values)
        if config.include_byte_hist:
            hist_cols, hist_values = _hashed_byte_hist_features(byte_seq, config.n_features)
            cols_parts.append(hist_cols)
            values_parts.append(hist_values)
        if config.include_cache_features:
            dense_cols, dense_values = _hashed_dense_features(pe_feat, stat_feat, lightweight_feat, config.n_features)
            cols_parts.append(dense_cols)
            values_parts.append(dense_values)

        cols = np.concatenate(cols_parts) if cols_parts else np.empty(0, dtype=np.int64)
        values = np.concatenate(values_parts) if values_parts else np.empty(0, dtype=np.float32)
        if cols.size:
            cols, inverse = np.unique(cols, return_inverse=True)
            summed = np.zeros(cols.shape[0], dtype=np.float32)
            np.add.at(summed, inverse, values)
            norm = float(np.linalg.norm(summed))
            if norm > 0:
                summed /= norm
            row_indices.append(np.full(cols.shape[0], row_idx, dtype=np.int32))
            col_indices.append(cols.astype(np.int32, copy=False))
            data_values.append(summed)

    if row_indices:
        rows = np.concatenate(row_indices)
        cols = np.concatenate(col_indices)
        data = np.concatenate(data_values)
    else:
        rows = np.empty(0, dtype=np.int32)
        cols = np.empty(0, dtype=np.int32)
        data = np.empty(0, dtype=np.float32)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(records), config.n_features), dtype=np.float32)
    return matrix, labels


def batched(records: Sequence[dict], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield list(records[start : start + batch_size])


def summarize_region_coverage(records: Sequence[dict], config: RegionHashConfig, *, allow_missing_source: bool) -> dict:
    counts = {name: 0 for name in REGION_NAMES}
    zero_region_rows = 0
    total_regions = 0
    for record in records:
        payloads = region_payloads_for_record(record, config, allow_missing_source=allow_missing_source)
        names = {name for name, data in payloads if data}
        if not names:
            zero_region_rows += 1
        total_regions += len(names)
        for name in names:
            counts[name] += 1
    return {
        "rows": int(len(records)),
        "zero_region_rows": int(zero_region_rows),
        "avg_regions_per_row": float(total_regions / len(records)) if records else 0.0,
        "region_counts": {name: int(count) for name, count in counts.items()},
    }


def train_candidate(
    train_records: Sequence[dict],
    config: RegionHashConfig,
    *,
    alpha: float,
    l1_ratio: float,
    epochs: int,
    batch_size: int,
    seed: int,
    allow_missing_source: bool,
) -> SGDClassifier:
    model = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet" if l1_ratio > 0 else "l2",
        alpha=alpha,
        l1_ratio=l1_ratio,
        learning_rate="optimal",
        average=True,
        class_weight=None,
        random_state=seed,
    )
    classes = np.asarray([0, 1], dtype=np.int64)
    first_batch = True
    rng = np.random.default_rng(seed)
    records = list(train_records)
    for epoch in range(epochs):
        order = rng.permutation(len(records))
        shuffled = [records[int(i)] for i in order]
        processed = 0
        for batch in batched(shuffled, batch_size):
            x_batch, y_batch = transform_batch(batch, config, allow_missing_source=allow_missing_source)
            if first_batch:
                model.partial_fit(x_batch, y_batch, classes=classes)
                first_batch = False
            else:
                model.partial_fit(x_batch, y_batch)
            processed += len(batch)
        print(f"[train] alpha={alpha:g} epoch={epoch + 1}/{epochs} rows={processed}", flush=True)
    return model


def predict_scores(
    model: SGDClassifier,
    records: Sequence[dict],
    config: RegionHashConfig,
    batch_size: int,
    *,
    allow_missing_source: bool,
) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    scores = []
    for batch in batched(records, batch_size):
        x_batch, y_batch = transform_batch(batch, config, allow_missing_source=allow_missing_source)
        batch_scores = sigmoid(model.decision_function(x_batch))
        labels.append(y_batch)
        scores.append(batch_scores.astype(np.float32, copy=False))
    return np.concatenate(labels), np.concatenate(scores)


def write_loop44_predictions(path: Path, records: Sequence[dict], labels: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    write_predictions(path, records, labels, scores, threshold)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train regionized byte n-gram SGD candidates from content regions.")
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-features", type=int, default=2**21)
    parser.add_argument("--prefix-len", type=int, default=4096)
    parser.add_argument("--region-window", type=int, default=1024)
    parser.add_argument("--tail-window", type=int, default=1024)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--ngram-stride", type=int, default=2)
    parser.add_argument("--alphas", default="3e-6,1e-5")
    parser.add_argument("--l1-ratio", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--thresholds", default="0.20:0.80:0.005")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=10000)
    parser.add_argument("--skip-test-eval", action="store_true")
    parser.add_argument("--include-prefix-features", action="store_true")
    parser.add_argument("--include-full-ngram-features", action="store_true")
    parser.add_argument("--include-byte-hist", action="store_true")
    parser.add_argument("--include-cache-features", action="store_true")
    parser.add_argument("--no-region-ngram-features", action="store_true")
    parser.add_argument("--no-region-scalar-features", action="store_true")
    parser.add_argument("--allow-missing-source-regions", action="store_true")
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args(argv)

    checkpoint_config = load_checkpoint_config(args.checkpoint)
    hash_config = RegionHashConfig(
        n_features=int(args.n_features),
        prefix_len=int(args.prefix_len),
        ngram_min=int(args.ngram_min),
        ngram_max=int(args.ngram_max),
        ngram_stride=int(args.ngram_stride),
        include_prefix_features=bool(args.include_prefix_features),
        include_full_ngram_features=bool(args.include_full_ngram_features),
        include_region_ngram_features=not bool(args.no_region_ngram_features),
        include_region_scalar_features=not bool(args.no_region_scalar_features),
        include_byte_hist=bool(args.include_byte_hist),
        include_cache_features=bool(args.include_cache_features),
        region_window=max(1, int(args.region_window)),
        tail_window=max(1, int(args.tail_window)),
        max_byte_length=checkpoint_config.max_byte_length,
        pe_feature_dim=checkpoint_config.pe_feature_dim,
        stat_feature_dim=checkpoint_config.stat_feature_dim,
        lightweight_feature_dim=checkpoint_config.lightweight_feature_dim,
    )

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records = load_records(args.split_csv, args.manifest, "train", args.max_train_rows)
    val_records = load_records(args.split_csv, args.manifest, "val", args.max_val_rows)
    test_records = []
    if not args.skip_test_eval:
        test_records = load_records(args.split_csv, args.manifest, "test", args.max_test_rows)

    train_records = list(sklearn_shuffle(train_records, random_state=args.seed))
    allow_missing_source = bool(args.allow_missing_source_regions)
    print("[regions] preloading train region payloads", flush=True)
    train_region_coverage = summarize_region_coverage(train_records, hash_config, allow_missing_source=allow_missing_source)
    print("[regions] preloading val region payloads", flush=True)
    val_region_coverage = summarize_region_coverage(val_records, hash_config, allow_missing_source=allow_missing_source)
    test_region_coverage = None
    if test_records:
        print("[regions] preloading test region payloads", flush=True)
        test_region_coverage = summarize_region_coverage(test_records, hash_config, allow_missing_source=allow_missing_source)

    thresholds = parse_thresholds(args.thresholds)
    alpha_values = parse_float_list(args.alphas)
    candidates = []
    best = None
    best_model = None
    for alpha in alpha_values:
        model = train_candidate(
            train_records,
            hash_config,
            alpha=float(alpha),
            l1_ratio=float(args.l1_ratio),
            epochs=max(1, int(args.epochs)),
            batch_size=max(1, int(args.batch_size)),
            seed=int(args.seed),
            allow_missing_source=allow_missing_source,
        )
        val_labels, val_scores = predict_scores(
            model,
            val_records,
            hash_config,
            max(1, int(args.batch_size)),
            allow_missing_source=allow_missing_source,
        )
        val_best = select_threshold(val_labels, val_scores, thresholds)
        candidate = {"alpha": float(alpha), "val": val_best}
        candidates.append(candidate)
        print(
            f"[val] alpha={alpha:g} f1={val_best['f1']:.6f} "
            f"errors={val_best['errors']} threshold={val_best['threshold']:.4f}",
            flush=True,
        )
        if best is None or (val_best["f1"], -val_best["errors"]) > (best["val"]["f1"], -best["val"]["errors"]):
            best = candidate
            best_model = model

    if best is None or best_model is None:
        raise ValueError("No candidate model was trained")

    threshold = float(best["val"]["threshold"])
    val_labels, val_scores = predict_scores(
        best_model,
        val_records,
        hash_config,
        max(1, int(args.batch_size)),
        allow_missing_source=allow_missing_source,
    )
    val_metrics = metrics_at_threshold(val_labels, val_scores, threshold)
    test_metrics = None
    test_predictions = None
    if test_records:
        test_labels, test_scores = predict_scores(
            best_model,
            test_records,
            hash_config,
            max(1, int(args.batch_size)),
            allow_missing_source=allow_missing_source,
        )
        test_metrics = metrics_at_threshold(test_labels, test_scores, threshold)

    model_path = output_dir / "loop44_region_byte_ngram_selected_model.joblib"
    joblib.dump(
        {
            "model": best_model,
            "hash_config": hash_config,
            "threshold": threshold,
            "selected": best,
            "checkpoint_config": checkpoint_config.to_dict(),
        },
        model_path,
    )

    val_predictions = output_dir / "loop44_region_byte_ngram_val_predictions.csv"
    write_loop44_predictions(val_predictions, val_records, val_labels, val_scores, threshold)
    if test_records:
        test_predictions = output_dir / "loop44_region_byte_ngram_test_predictions.csv"
        write_loop44_predictions(test_predictions, test_records, test_labels, test_scores, threshold)

    test_gate_decision = "not_run_val_only"
    if test_records:
        test_gate_decision = "ran_by_explicit_request"
    elif len(val_records) < 20000:
        test_gate_decision = "smoke_only_not_eligible_for_test10k"
    elif int(val_metrics["errors"]) <= LOOP44_TEST10K_ERROR_GATE:
        test_gate_decision = "eligible_for_test10k"
    else:
        test_gate_decision = "reject_val_margin_too_small"

    report = {
        "schema": "axon_loop44_region_byte_ngram_v1",
        "protocol": (
            "train split fits regionized byte n-gram SGD candidates; Val selects alpha and threshold; "
            "Test-10k/full-test are not used unless explicitly provided after Val gate"
        ),
        "identity_feature_policy": (
            "source_path/original_source_path/cache_path/source_sha256/sample_index/split are loading, "
            "alignment, and audit fields only; model features are byte n-grams/statistics read from file content"
        ),
        "pefile_available": bool(PEFILE_AVAILABLE),
        "split_csv": str(resolve_path(args.split_csv)),
        "manifest": str(resolve_path(args.manifest)),
        "checkpoint": str(resolve_path(args.checkpoint)),
        "records": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "region_coverage": {
            "train": train_region_coverage,
            "val": val_region_coverage,
            "test": test_region_coverage,
        },
        "region_names": REGION_NAMES,
        "region_scalar_feature_names": REGION_SCALAR_FEATURE_NAMES
        if hash_config.include_region_scalar_features
        else [],
        "hash_config": hash_config.__dict__,
        "candidate_alphas": alpha_values,
        "candidates": candidates,
        "selected": best,
        "selected_threshold": threshold,
        "val_metrics_at_selected_threshold": val_metrics,
        "loop28_val_errors_reference": LOOP28_VAL_ERRORS,
        "test10k_error_gate": LOOP44_TEST10K_ERROR_GATE,
        "test_gate_decision": test_gate_decision,
        "test_metrics_at_selected_threshold": test_metrics,
        "model_path": str(model_path),
        "val_predictions": str(val_predictions),
        "test_predictions": str(test_predictions) if test_predictions is not None else None,
    }
    report_path = output_dir / "loop44_region_byte_ngram_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
