#!/usr/bin/env python3
"""Benchmark the Axon ONNX DLL on an explicit manifest split."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Optional, Sequence


class KvdConfig(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("model_normal_path", ctypes.c_char_p),
        ("model_packed_path", ctypes.c_char_p),
        ("family_classifier_json_path", ctypes.c_char_p),
        ("allowed_scan_root", ctypes.c_char_p),
        ("max_file_size", ctypes.c_uint),
        ("prediction_threshold", ctypes.c_float),
        ("onnx_model_path", ctypes.c_char_p),
        ("onnx_model_normal_path", ctypes.c_char_p),
        ("onnx_model_packed_path", ctypes.c_char_p),
        ("stage2_model_json_path", ctypes.c_char_p),
        ("archive_scanner_path", ctypes.c_char_p),
        ("scan_nested", ctypes.c_int),
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def select_balanced_rows(rows: Sequence[dict[str, str]], split: str, count: int) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("split", "").strip().casefold() == split]
    by_label = {
        label: [row for row in selected if int(row["label"]) == label]
        for label in (0, 1)
    }
    if count <= 0:
        if len(by_label[0]) != len(by_label[1]):
            raise ValueError(f"split {split!r} is not balanced")
        limit = len(by_label[0])
    else:
        if count % 2:
            raise ValueError("count must be even for balanced selection")
        limit = count // 2
        if any(len(by_label[label]) < limit for label in (0, 1)):
            raise ValueError(f"split {split!r} does not contain {limit} rows per class")
    return [row for pair in zip(by_label[0][:limit], by_label[1][:limit]) for row in pair]


def binary_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float | int]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    true_positive = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    true_negative = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    false_positive = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    false_negative = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (true_positive + true_negative) / len(labels) if labels else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


class KvdLibrary:
    def __init__(
        self,
        dll_path: Path,
        onnx_path: Path,
        allowed_root: Path,
        threshold: float,
        stage2_path: Optional[Path] = None,
    ):
        self.library = ctypes.CDLL(str(dll_path))
        self.library.kvd_create.argtypes = [ctypes.POINTER(KvdConfig)]
        self.library.kvd_create.restype = ctypes.c_void_p
        self.library.kvd_destroy.argtypes = [ctypes.c_void_p]
        self.library.kvd_scan_path.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.library.kvd_scan_path.restype = ctypes.c_int
        self.library.kvd_free.argtypes = [ctypes.c_void_p]
        self._onnx_bytes = str(onnx_path).encode("utf-8")
        self._root_bytes = str(allowed_root).encode("utf-8")
        self._stage2_bytes = str(stage2_path).encode("utf-8") if stage2_path is not None else None
        config = KvdConfig()
        config.onnx_model_path = self._onnx_bytes
        config.allowed_scan_root = self._root_bytes
        config.prediction_threshold = threshold
        config.stage2_model_json_path = self._stage2_bytes
        started = time.perf_counter()
        self.handle = self.library.kvd_create(ctypes.byref(config))
        self.init_ms = (time.perf_counter() - started) * 1000.0
        if not self.handle:
            raise RuntimeError("kvd_create failed")

    def close(self) -> None:
        if self.handle:
            self.library.kvd_destroy(self.handle)
            self.handle = None

    def scan(self, path: Path) -> tuple[dict, float]:
        pointer = ctypes.c_void_p()
        length = ctypes.c_size_t()
        started = time.perf_counter()
        return_code = self.library.kvd_scan_path(
            self.handle,
            str(path).encode("utf-8"),
            ctypes.byref(pointer),
            ctypes.byref(length),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if return_code != 0:
            raise RuntimeError(f"kvd_scan_path returned {return_code}")
        try:
            payload = json.loads(ctypes.string_at(pointer, length.value).decode("utf-8"))
        finally:
            self.library.kvd_free(pointer)
        if not payload.get("ok"):
            raise RuntimeError(json.dumps(payload, ensure_ascii=False))
        return payload, elapsed_ms


def run_benchmark(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_heldout:
        raise ValueError("held-out split access requires --allow-heldout")
    with args.split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = select_balanced_rows(rows, args.split, args.count)
    if not selected:
        raise ValueError(f"split {args.split!r} contains no rows")

    scanner = KvdLibrary(
        args.dll.resolve(),
        args.onnx.resolve(),
        args.allowed_root.resolve(),
        args.threshold,
        args.stage2.resolve() if args.stage2 is not None else None,
    )
    errors: list[dict] = []
    samples: list[dict] = []
    try:
        for index in range(args.warmup):
            scanner.scan(Path(selected[index % len(selected)]["source_path"]))
        for row in selected:
            source_path = Path(row["source_path"])
            try:
                payload, elapsed_ms = scanner.scan(source_path)
                samples.append(
                    {
                        "source_sha256": row.get("source_sha256") or row.get("sha256"),
                        "label": int(row["label"]),
                        "prediction": int(payload["prediction"]),
                        "prob_malicious": float(payload["prob_malicious"]),
                        "elapsed_ms": elapsed_ms,
                        "file_size": source_path.stat().st_size,
                        "timing_ms": payload.get("timing_ms"),
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "source_sha256": row.get("source_sha256") or row.get("sha256"),
                        "source_path": str(source_path),
                        "error": str(exc),
                    }
                )
    finally:
        scanner.close()

    latencies = [sample["elapsed_ms"] for sample in samples]
    metrics = binary_metrics(
        [sample["label"] for sample in samples],
        [sample["prediction"] for sample in samples],
    )
    sidecar_path = args.onnx.with_suffix(args.onnx.suffix + ".data")
    report = {
        "schema": "axon_onnx_dll_benchmark_v1",
        "split": args.split,
        "requested_count": args.count,
        "sample_count": len(samples),
        "warmup_count": args.warmup,
        "threshold": args.threshold,
        "scan_error_count": len(errors),
        "metrics": metrics,
        "latency_ms": {
            "init": scanner.init_ms,
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50) if latencies else None,
            "p95": percentile(latencies, 0.95) if latencies else None,
            "p99": percentile(latencies, 0.99) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "gates": {
            "scan_errors_zero": not errors,
            "max_latency_below_500ms": bool(latencies) and not errors and max(latencies) < 500.0,
        },
        "runtime": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "axon_ort_intra_op_threads_env": os.environ.get("AXON_ORT_INTRA_OP_THREADS"),
            "axon_native_student_only_env": os.environ.get("AXON_NATIVE_STUDENT_ONLY"),
        },
        "artifacts": {
            "dll": str(args.dll.resolve()),
            "dll_sha256": file_sha256(args.dll),
            "onnx": str(args.onnx.resolve()),
            "onnx_sha256": file_sha256(args.onnx),
            "onnx_data": str(sidecar_path.resolve()) if sidecar_path.exists() else None,
            "onnx_data_sha256": file_sha256(sidecar_path) if sidecar_path.exists() else None,
            "split_csv": str(args.split_csv.resolve()),
            "split_csv_sha256": file_sha256(args.split_csv),
            "stage2": str(args.stage2.resolve()) if args.stage2 is not None else None,
            "stage2_sha256": file_sha256(args.stage2) if args.stage2 is not None else None,
        },
        "errors": errors,
        "samples": samples,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, default=None)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--count", type=int, default=0, help="Balanced row count; 0 means the full split")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--allow-heldout", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = run_benchmark(args)
    except Exception as exc:
        print(f"[Error] {exc}")
        return 1
    print(json.dumps({key: report[key] for key in ("sample_count", "scan_error_count", "metrics", "latency_ms", "gates")}, indent=2))
    return 0 if report["gates"]["scan_errors_zero"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
