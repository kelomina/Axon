#!/usr/bin/env python3
"""Run the non-promotable Loop164 train-only whole-file engineering probe."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "loop164_whole_file.toml"
DEFAULT_BUNDLE = PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_probe_bundle.jsonl"
DEFAULT_BUNDLE_SUMMARY = (
    PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_probe_bundle_summary.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_feasibility_probe_receipt.json"
)
RESOURCE_GUARD = (
    PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_feasibility_resource_guard.json"
)
CONFIG_SCHEMA = "axon_loop164_local_feasibility_config_v1"
RECEIPT_SCHEMA = "axon_loop164_local_feasibility_probe_receipt_v1"
FAILURE_SCHEMA = "axon_loop164_local_feasibility_failure_observation_v1"
CLAIM_SCOPE = "user_directed_local_custody_engineering_only"
MISSING_REASONS = ("timeout", "unsupported", "read_failure", "parse_failure", "oversize")
MAX_CONFIG_BYTES = 128 * 1024


class ProbeContractError(ValueError):
    """The local feasibility run does not match its frozen config."""


class ProbeNonfiniteError(RuntimeError):
    """A loss, gradient, or model tensor became non-finite."""


class ProbeResourceError(RuntimeError):
    """The local process exceeded a frozen resource cap."""


@dataclass(frozen=True)
class ProbeConfig:
    seed: int
    records_per_class: int
    epochs: int
    wall_timeout_seconds: int
    per_file_timeout_seconds: int
    device: str
    precision: str
    embedding_dim: int
    channels: int
    receptive_field_bytes: int
    output_stride_bytes: int
    chunk_bytes: int
    max_outputs_per_chunk: int
    num_classes: int
    data_root: Path
    max_supported_file_bytes: int
    bounded_read_bytes: int
    workers: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    gradient_accumulation_steps: int
    max_optimizer_steps: int
    gradient_clip_norm: float
    max_process_rss_bytes: int
    max_cuda_allocated_bytes: int
    max_cuda_reserved_bytes: int


@dataclass
class ProbeCounters:
    denominator: int
    success: int = 0
    missing: int = 0
    completed_scans: int = 0
    verified_sha_passes: int = 0
    raw_bytes_read: int = 0
    backward_microbatches: int = 0
    optimizer_steps: int = 0
    discarded_accumulation: int = 0
    timeout_events: int = 0
    oom_events: int = 0
    nonfinite_events: int = 0

    def __post_init__(self) -> None:
        self.missing_by_reason = {reason: 0 for reason in MISSING_REASONS}

    def add_missing(self, reason: str, count: int = 1) -> None:
        if reason not in self.missing_by_reason or count < 0:
            raise ProbeContractError("Invalid missingness accounting")
        self.missing += count
        self.missing_by_reason[reason] += count
        if reason == "timeout":
            self.timeout_events += count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bounded(path: Path, max_bytes: int) -> bytes:
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ProbeContractError(f"Input exceeds its bounded size: {path}")
    return raw


def _require_exact_keys(payload: object, expected: set[str], *, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ProbeContractError(f"{context} fields do not match the frozen config")
    return payload


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ProbeContractError(f"{key} must be a positive integer")
    return value


def _nonnegative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProbeContractError(f"{key} must be a non-negative integer")
    return value


def _positive_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProbeContractError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ProbeContractError(f"{key} must be finite and positive")
    return number


def _resolve_project_path(value: object, *, context: str, require_file: bool = False) -> Path:
    path = Path(str(value or ""))
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    resolved_root = PROJECT_ROOT.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ProbeContractError(f"{context} escapes the project root") from exc
    if any(ancestor.is_symlink() for ancestor in (candidate.absolute(), *candidate.absolute().parents)):
        raise ProbeContractError(f"{context} cannot use symbolic links")
    if require_file and not resolved.is_file():
        raise ProbeContractError(f"{context} must be a file")
    return resolved


def load_config(path: Path) -> ProbeConfig:
    config_path = path.resolve(strict=True)
    if config_path != DEFAULT_CONFIG.resolve(strict=True) and PROJECT_ROOT == Path(__file__).resolve().parents[1]:
        # Tests may monkeypatch DEFAULT_CONFIG; production use is pinned to the canonical file.
        raise ProbeContractError("Only the canonical Loop164 feasibility config is accepted")
    try:
        payload = tomllib.loads(_read_bounded(config_path, MAX_CONFIG_BYTES).decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProbeContractError("Loop164 feasibility config is unreadable") from exc
    root = _require_exact_keys(
        payload,
        {"probe", "model", "input", "training", "resources"},
        context="config root",
    )
    probe = _require_exact_keys(
        root["probe"],
        {
            "schema",
            "claim_scope",
            "seed",
            "records_per_class",
            "epochs",
            "wall_timeout_seconds",
            "per_file_timeout_seconds",
            "device",
            "precision",
        },
        context="probe config",
    )
    model = _require_exact_keys(
        root["model"],
        {
            "embedding_dim",
            "channels",
            "receptive_field_bytes",
            "output_stride_bytes",
            "chunk_bytes",
            "max_outputs_per_chunk",
            "num_classes",
        },
        context="model config",
    )
    input_config = _require_exact_keys(
        root["input"],
        {"data_root", "max_supported_file_bytes", "bounded_read_bytes", "workers"},
        context="input config",
    )
    training = _require_exact_keys(
        root["training"],
        {
            "optimizer",
            "learning_rate",
            "weight_decay",
            "gradient_accumulation_steps",
            "max_optimizer_steps",
            "gradient_clip_norm",
        },
        context="training config",
    )
    resources = _require_exact_keys(
        root["resources"],
        {"max_process_rss_bytes", "max_cuda_allocated_bytes", "max_cuda_reserved_bytes"},
        context="resource config",
    )
    if probe.get("schema") != CONFIG_SCHEMA or probe.get("claim_scope") != CLAIM_SCOPE:
        raise ProbeContractError("Probe config identity or claim scope drifted")
    if probe.get("device") not in {"cpu", "cuda"} or probe.get("precision") != "fp32":
        raise ProbeContractError("The first feasibility probe requires CPU/CUDA FP32")
    if probe.get("epochs") != 1:
        raise ProbeContractError("The feasibility probe is fixed to one epoch")
    if model.get("num_classes") != 2:
        raise ProbeContractError("The feasibility probe requires two classes")
    if input_config.get("workers") != 0:
        raise ProbeContractError("The feasibility probe requires a single-process loader")
    if training.get("optimizer") != "adamw":
        raise ProbeContractError("The feasibility probe requires AdamW")

    receptive_field = _positive_int(model, "receptive_field_bytes")
    stride = _positive_int(model, "output_stride_bytes")
    chunk_bytes = _positive_int(model, "chunk_bytes")
    max_outputs = _positive_int(model, "max_outputs_per_chunk")
    if stride > receptive_field:
        raise ProbeContractError("output_stride_bytes cannot exceed receptive_field_bytes")
    expected_chunk_bytes = (max_outputs - 1) * stride + receptive_field
    if chunk_bytes != expected_chunk_bytes:
        raise ProbeContractError("chunk_bytes does not match output-coordinate geometry")
    bounded_read_bytes = _positive_int(input_config, "bounded_read_bytes")
    if bounded_read_bytes != chunk_bytes:
        raise ProbeContractError("bounded_read_bytes must equal chunk_bytes")

    return ProbeConfig(
        seed=_positive_int(probe, "seed"),
        records_per_class=_positive_int(probe, "records_per_class"),
        epochs=_positive_int(probe, "epochs"),
        wall_timeout_seconds=_positive_int(probe, "wall_timeout_seconds"),
        per_file_timeout_seconds=_positive_int(probe, "per_file_timeout_seconds"),
        device=str(probe["device"]),
        precision=str(probe["precision"]),
        embedding_dim=_positive_int(model, "embedding_dim"),
        channels=_positive_int(model, "channels"),
        receptive_field_bytes=receptive_field,
        output_stride_bytes=stride,
        chunk_bytes=chunk_bytes,
        max_outputs_per_chunk=max_outputs,
        num_classes=_positive_int(model, "num_classes"),
        data_root=_resolve_project_path(input_config["data_root"], context="data_root"),
        max_supported_file_bytes=_positive_int(input_config, "max_supported_file_bytes"),
        bounded_read_bytes=bounded_read_bytes,
        workers=_nonnegative_int(input_config, "workers"),
        optimizer=str(training["optimizer"]),
        learning_rate=_positive_float(training, "learning_rate"),
        weight_decay=_positive_float(training, "weight_decay"),
        gradient_accumulation_steps=_positive_int(training, "gradient_accumulation_steps"),
        max_optimizer_steps=_positive_int(training, "max_optimizer_steps"),
        gradient_clip_norm=_positive_float(training, "gradient_clip_norm"),
        max_process_rss_bytes=_positive_int(resources, "max_process_rss_bytes"),
        max_cuda_allocated_bytes=_positive_int(resources, "max_cuda_allocated_bytes"),
        max_cuda_reserved_bytes=_positive_int(resources, "max_cuda_reserved_bytes"),
    )


def _peak_process_rss_bytes() -> int:
    if platform.system().casefold() == "windows":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        get_current_process = ctypes.windll.kernel32.GetCurrentProcess  # type: ignore[attr-defined]
        get_current_process.argtypes = []
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo  # type: ignore[attr-defined]
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_process_memory_info.restype = ctypes.c_int
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = get_current_process()
        if not get_process_memory_info(process, ctypes.byref(counters), counters.cb):
            raise ProbeResourceError("Unable to read peak process memory")
        return int(counters.PeakWorkingSetSize)

    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if platform.system().casefold() == "darwin" else peak * 1024


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _source_bindings(config_path: Path) -> dict[str, dict[str, object]]:
    paths = {
        "config": config_path,
        "controller": Path(__file__).resolve(),
        "model": PROJECT_ROOT / "src" / "loop164" / "whole_file_gcg.py",
        "loader": PROJECT_ROOT / "src" / "loop164" / "authorized_input.py",
        "runtime_lock": PROJECT_ROOT / "requirements.txt",
        "resource_guard": RESOURCE_GUARD,
    }
    return {
        name: {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _runtime_payload(torch: Any, device: Any) -> dict[str, Any]:
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "device": str(device),
        "precision": "fp32",
    }
    if device.type == "cuda":
        runtime.update(
            {
                "cuda": str(torch.version.cuda),
                "cudnn": int(torch.backends.cudnn.version() or 0),
                "gpu_name": torch.cuda.get_device_name(device),
            }
        )
    return runtime


def _base_observation(
    *,
    schema: str,
    status: str,
    config_path: Path,
    bundle_path: Path,
    bundle_summary_path: Path,
    bindings: dict[str, dict[str, object]],
    argv: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": schema,
        "loop_id": "loop164_whole_file_residual_expert",
        "claim_scope": CLAIM_SCOPE,
        "status": status,
        "created_at_unix_seconds": int(time.time()),
        "argv": list(argv),
        "bindings": {
            **bindings,
            "bundle": {"path": str(bundle_path), "sha256": sha256_file(bundle_path)},
            "bundle_summary": {
                "path": str(bundle_summary_path),
                "sha256": sha256_file(bundle_summary_path),
            },
        },
        "forbidden_outputs": {
            "checkpoint_written": False,
            "model_state_serialized": False,
            "predictions_written": False,
            "metrics_computed": [],
            "threshold_operations": 0,
        },
        "ready_for": {
            "loop164_candidate_promotion": False,
            "val_or_test_access": False,
            "external_certification": False,
            "f1_claim": False,
        },
    }


def run_probe(
    *,
    config_path: Path,
    bundle_path: Path,
    bundle_summary_path: Path,
    output_path: Path,
    argv: Sequence[str],
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("Probe output already exists; refusing to overwrite")
    config_path = config_path.resolve(strict=True)
    bundle_path = bundle_path.resolve(strict=True)
    bundle_summary_path = bundle_summary_path.resolve(strict=True)
    config = load_config(config_path)
    bindings = _source_bindings(config_path)
    start = clock()
    global_deadline = start + config.wall_timeout_seconds

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    import torch
    import torch.nn.functional as functional

    from loop164.authorized_input import (
        SourceIntegrityError,
        SourceTimeoutError,
        StreamingWholeFileByteSource,
        load_local_probe_bundle,
    )
    from loop164.whole_file_gcg import WholeFileGCGClassifier

    records, bundle_summary = load_local_probe_bundle(
        bundle_path=bundle_path,
        summary_path=bundle_summary_path,
        data_root=config.data_root,
        expected_records_per_class=config.records_per_class,
    )
    denominator = config.records_per_class * 2
    if len(records) != denominator:
        raise ProbeContractError("Probe denominator drifted after bundle validation")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ProbeResourceError("CUDA is required by the canonical probe config")
    device = torch.device(config.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = WholeFileGCGClassifier(
        embedding_dim=config.embedding_dim,
        channels=config.channels,
        receptive_field_bytes=config.receptive_field_bytes,
        output_stride_bytes=config.output_stride_bytes,
        max_outputs_per_chunk=config.max_outputs_per_chunk,
        num_classes=config.num_classes,
    ).to(device=device, dtype=torch.float32)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    optimizer.zero_grad(set_to_none=True)
    ordered_records = list(records)
    random.Random(config.seed).shuffle(ordered_records)
    counters = ProbeCounters(denominator=denominator)
    accumulation_count = 0
    peak_rss = _peak_process_rss_bytes()
    fatal_code: Optional[str] = None
    fatal_type: Optional[str] = None

    def update_resource_peaks() -> tuple[int, int, int]:
        nonlocal peak_rss
        peak_rss = max(peak_rss, _peak_process_rss_bytes())
        cuda_allocated = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        cuda_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        if peak_rss > config.max_process_rss_bytes:
            raise ProbeResourceError("Process RSS exceeded the frozen cap")
        if cuda_allocated > config.max_cuda_allocated_bytes:
            raise ProbeResourceError("CUDA allocated memory exceeded the frozen cap")
        if cuda_reserved > config.max_cuda_reserved_bytes:
            raise ProbeResourceError("CUDA reserved memory exceeded the frozen cap")
        return peak_rss, cuda_allocated, cuda_reserved

    def optimizer_step() -> None:
        nonlocal accumulation_count
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            counters.nonfinite_events += 1
            raise ProbeNonfiniteError("Gradient norm became non-finite")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        counters.optimizer_steps += 1
        accumulation_count = 0
        if counters.optimizer_steps > config.max_optimizer_steps:
            raise ProbeContractError("Optimizer step cap was exceeded")

    try:
        for record_index, record in enumerate(ordered_records):
            if clock() > global_deadline:
                remaining = denominator - record_index
                counters.add_missing("timeout", remaining)
                counters.discarded_accumulation += accumulation_count
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                break
            if record.source_size_bytes > config.max_supported_file_bytes:
                counters.add_missing("oversize")
                continue
            if record.source_size_bytes < 1:
                counters.add_missing("parse_failure")
                continue
            try:
                source = StreamingWholeFileByteSource(
                    record,
                    data_root=config.data_root,
                    receptive_field_bytes=config.receptive_field_bytes,
                    output_stride_bytes=config.output_stride_bytes,
                    max_outputs_per_chunk=config.max_outputs_per_chunk,
                    bounded_read_bytes=config.bounded_read_bytes,
                    max_supported_file_bytes=config.max_supported_file_bytes,
                    timeout_seconds=config.per_file_timeout_seconds,
                    absolute_deadline=global_deadline,
                    clock=clock,
                )
                result = model.forward_from_source(source, return_features=False)
                source.assert_complete()
                logits = result["logits"]
                if not torch.isfinite(logits).all():
                    counters.nonfinite_events += 1
                    raise ProbeNonfiniteError("Model logits became non-finite")
                labels = torch.tensor([record.label], dtype=torch.long, device=device)
                loss = functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    counters.nonfinite_events += 1
                    raise ProbeNonfiniteError("Training loss became non-finite")
                (loss / config.gradient_accumulation_steps).backward()
                counters.backward_microbatches += 1
                accumulation_count += 1
                counters.success += 1
                counters.completed_scans += len(source.scan_receipts)
                counters.verified_sha_passes += sum(
                    receipt.sha256 == record.source_sha256 for receipt in source.scan_receipts
                )
                counters.raw_bytes_read += sum(
                    receipt.bytes_read for receipt in source.scan_receipts
                )
                if accumulation_count == config.gradient_accumulation_steps:
                    optimizer_step()
            except SourceTimeoutError:
                counters.add_missing("timeout")
            except FileNotFoundError:
                counters.add_missing("read_failure")
            except PermissionError:
                counters.add_missing("read_failure")
            except SourceIntegrityError:
                raise
            update_resource_peaks()

        if counters.success + counters.missing != counters.denominator:
            raise ProbeContractError("Probe denominator accounting is incomplete")
        if counters.missing != sum(counters.missing_by_reason.values()):
            raise ProbeContractError("Probe missingness accounting is inconsistent")
        if accumulation_count:
            optimizer_step()
        if counters.completed_scans != counters.success * 2:
            raise ProbeContractError("Successful rows did not complete exactly two scans")
        if counters.verified_sha_passes != counters.completed_scans:
            raise ProbeContractError("Not every completed scan verified the source SHA")
        update_resource_peaks()
    except torch.OutOfMemoryError as exc:
        counters.oom_events += 1
        fatal_code = "cuda_or_host_oom"
        fatal_type = type(exc).__name__
    except ProbeNonfiniteError as exc:
        fatal_code = "nonfinite_abort"
        fatal_type = type(exc).__name__
    except ProbeResourceError as exc:
        fatal_code = "resource_cap_abort"
        fatal_type = type(exc).__name__
    except SourceIntegrityError as exc:
        fatal_code = "source_integrity_abort"
        fatal_type = type(exc).__name__
    except Exception as exc:
        fatal_code = "unexpected_abort"
        fatal_type = type(exc).__name__

    elapsed = max(0.0, clock() - start)
    cuda_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    cuda_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    peak_rss = max(peak_rss, _peak_process_rss_bytes())
    base = _base_observation(
        schema=FAILURE_SCHEMA if fatal_code else RECEIPT_SCHEMA,
        status="failed" if fatal_code else "completed",
        config_path=config_path,
        bundle_path=bundle_path,
        bundle_summary_path=bundle_summary_path,
        bindings=bindings,
        argv=argv,
    )
    base.update(
        {
            "runtime": _runtime_payload(torch, device),
            "probe_contract": {
                "records_per_class": config.records_per_class,
                "epochs": config.epochs,
                "batch_size": 1,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "max_optimizer_steps": config.max_optimizer_steps,
                "wall_timeout_seconds": config.wall_timeout_seconds,
                "per_file_timeout_seconds": config.per_file_timeout_seconds,
                "source_content_passes": 2,
                "source_content_third_hash_pass": False,
                "bundle_decision": bundle_summary["decision"],
            },
            "aggregate": {
                "denominator": counters.denominator,
                "success": counters.success,
                "missing": counters.missing,
                "missing_by_reason": counters.missing_by_reason,
                "completed_scans": counters.completed_scans,
                "verified_sha_passes": counters.verified_sha_passes,
                "raw_bytes_read": counters.raw_bytes_read,
                "backward_microbatches": counters.backward_microbatches,
                "optimizer_steps": counters.optimizer_steps,
                "discarded_accumulation": counters.discarded_accumulation,
                "timeout_events": counters.timeout_events,
                "oom_events": counters.oom_events,
                "nonfinite_events": counters.nonfinite_events,
            },
            "resources": {
                "peak_process_rss_bytes": peak_rss,
                "peak_cuda_allocated_bytes": cuda_allocated,
                "peak_cuda_reserved_bytes": cuda_reserved,
                "elapsed_seconds": elapsed,
                "successful_files_per_second": counters.success / elapsed if elapsed else 0.0,
                "raw_megabytes_per_second": (
                    counters.raw_bytes_read / (1024 * 1024) / elapsed if elapsed else 0.0
                ),
            },
            "fatal": {"code": fatal_code, "exception_type": fatal_type} if fatal_code else None,
            "decision": (
                "engineering_probe_failed_not_promotable"
                if fatal_code
                else "engineering_feasibility_observed_not_promotable"
            ),
        }
    )
    _write_exclusive_json(output_path, base)
    if fatal_code:
        raise RuntimeError(f"Loop164 feasibility probe aborted: {fatal_code}")
    return base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local train-only Loop164 whole-file engineering probe."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--bundle-summary", type=Path, default=DEFAULT_BUNDLE_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(parsed_argv)
    receipt = run_probe(
        config_path=args.config,
        bundle_path=args.bundle,
        bundle_summary_path=args.bundle_summary,
        output_path=args.output,
        argv=parsed_argv,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
