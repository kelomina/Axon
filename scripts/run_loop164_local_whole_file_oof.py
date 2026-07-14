#!/usr/bin/env python3
"""Run one-seed five-fold whole-file OOF on local train-only diagnostic folds."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "loop164_local_oof.toml"
DEFAULT_FOLDS = (
    PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_train_diagnostic_folds.jsonl"
)
DEFAULT_FOLDS_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "roadmap_9997"
    / "loop164"
    / "local_train_diagnostic_folds_summary.json"
)
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "reports"
    / "roadmap_9997"
    / "loop164"
    / "local_whole_file_oof_predictions.jsonl"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_whole_file_oof_report.json"
)
RESOURCE_GUARD = (
    PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_whole_file_oof_resource_guard.json"
)
LOCAL_AUTHORIZATION = (
    PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_whole_file_oof_authorization.json"
)
RUN_LEASE = (
    PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_whole_file_oof_run_lease.json"
)
CONFIG_SCHEMA = "axon_loop164_local_whole_file_oof_config_v1"
REPORT_SCHEMA = "axon_loop164_local_whole_file_oof_report_v1"
FAILURE_SCHEMA = "axon_loop164_local_whole_file_oof_failure_v1"
PREDICTION_SCHEMA = "axon_loop164_local_whole_file_oof_prediction_v1"
CLAIM_SCOPE = "local_train_content_group_oof_diagnostic_not_production"
MAX_CONFIG_BYTES = 128 * 1024
MAX_RESOURCE_GUARD_BYTES = 1024 * 1024
MAX_AUTHORIZATION_BYTES = 1024 * 1024
RESOURCE_GUARD_MAX_AGE_SECONDS = 3600.0
FROZEN_PROTOCOL_SHA256 = {
    "config": "9cfef5fd37f17a9b90c010de181d228f8cb214e8bb573cf1d18e224283a4abc8",
    "folds": "00a31a1bd86d7b887447f3e86e5e753ebcaaee45be74311199332e073a3880a5",
    "folds_summary": "2b2a39a60ddf6b2713e03fef00f2c374db4f028f1b1fd3d16071d55cf6cfaddd",
    "model": "b8e946549ef996daf028f558cfbc4b662b946f317e970d705f8b4aa4d7a78101",
    "loader": "5e877b54fc1473cdb47b07c771ae925e0b5f0f504577dcac8c09c973e22175b2",
    "oof_contract": "c261f5fc34f3580ad87f10f29b5120c50e619daedeccfd84ebd0672367f90724",
    "runtime_lock": "d0269e1a86a1cc488b7520642c23ce8d05db1a97741138ff960d0b93b5de7578",
}


class OOFRunError(RuntimeError):
    """The local OOF run cannot preserve its frozen execution contract."""


@dataclass(frozen=True)
class OOFConfig:
    seed: int
    fold_count: int
    expected_rows: int
    epochs: int
    decision_threshold: float
    neutral_missing_score: float
    wall_timeout_seconds: int
    per_file_timeout_seconds: int
    device: str
    embedding_dim: int
    channels: int
    receptive_field_bytes: int
    output_stride_bytes: int
    chunk_bytes: int
    max_outputs_per_chunk: int
    data_root: Path
    max_supported_file_bytes: int
    bounded_read_bytes: int
    learning_rate: float
    weight_decay: float
    gradient_accumulation_steps: int
    gradient_clip_norm: float
    max_process_rss_bytes: int
    max_cuda_allocated_bytes: int
    max_cuda_reserved_bytes: int


@dataclass
class RunCounters:
    fit_source_calls: int = 0
    fit_scans: int = 0
    holdout_source_calls: int = 0
    holdout_scans: int = 0
    verified_sha_passes: int = 0
    raw_bytes_read: int = 0
    backward_microbatches: int = 0
    optimizer_steps: int = 0
    nonfinite_events: int = 0
    oom_events: int = 0


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
        raise OOFRunError(f"Bounded input is too large: {path}")
    return raw


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise OOFRunError(f"Duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite_json(value: str) -> object:
    raise OOFRunError(f"Non-finite JSON value: {value}")


def _expected_resource_guard_command(argv: Sequence[str]) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), *argv]


def _resource_guard_targets() -> list[Path]:
    return [
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "loop164" / "whole_file_gcg.py",
        PROJECT_ROOT / "src" / "loop164" / "authorized_input.py",
        PROJECT_ROOT / "src" / "loop164" / "local_oof.py",
    ]


def _frozen_protocol_paths(
    config_path: Path,
    folds_path: Path,
    folds_summary_path: Path,
) -> dict[str, Path]:
    return {
        "config": config_path,
        "folds": folds_path,
        "folds_summary": folds_summary_path,
        "model": PROJECT_ROOT / "src" / "loop164" / "whole_file_gcg.py",
        "loader": PROJECT_ROOT / "src" / "loop164" / "authorized_input.py",
        "oof_contract": PROJECT_ROOT / "src" / "loop164" / "local_oof.py",
        "runtime_lock": PROJECT_ROOT / "requirements.txt",
    }


def _validate_frozen_protocol(
    *,
    config_path: Path,
    folds_path: Path,
    folds_summary_path: Path,
    predictions_path: Path,
    report_path: Path,
) -> None:
    canonical_inputs = {
        config_path.resolve(strict=True): DEFAULT_CONFIG.resolve(strict=True),
        folds_path.resolve(strict=True): DEFAULT_FOLDS.resolve(strict=True),
        folds_summary_path.resolve(strict=True): DEFAULT_FOLDS_SUMMARY.resolve(strict=True),
    }
    if any(actual != expected for actual, expected in canonical_inputs.items()):
        raise OOFRunError("Local OOF inputs must use the canonical frozen paths")
    if (
        predictions_path.resolve(strict=False) != DEFAULT_PREDICTIONS.resolve(strict=False)
        or report_path.resolve(strict=False) != DEFAULT_REPORT.resolve(strict=False)
    ):
        raise OOFRunError("Local OOF outputs must use the canonical frozen paths")
    paths = _frozen_protocol_paths(config_path, folds_path, folds_summary_path)
    actual = {name: sha256_file(path.resolve(strict=True)) for name, path in paths.items()}
    if actual != FROZEN_PROTOCOL_SHA256:
        drifted = sorted(name for name in actual if actual[name] != FROZEN_PROTOCOL_SHA256[name])
        raise OOFRunError(f"Frozen local OOF protocol drifted: {', '.join(drifted)}")


def _validate_resource_guard(
    path: Path,
    *,
    argv: Sequence[str],
    now: Optional[float] = None,
) -> dict[str, Any]:
    resolved_path = path.resolve(strict=True)
    if resolved_path != RESOURCE_GUARD.resolve(strict=True):
        raise OOFRunError("Only the canonical local OOF resource guard is accepted")
    try:
        payload = json.loads(
            _read_bounded(resolved_path, MAX_RESOURCE_GUARD_BYTES).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OOFRunError("Local OOF resource guard is unreadable") from exc
    if not isinstance(payload, dict):
        raise OOFRunError("Local OOF resource guard must be a JSON object")
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from pre_run_resource_leak_guard import validate_guard_receipt

    validation = validate_guard_receipt(
        payload,
        expected_target_scripts=_resource_guard_targets(),
        expected_command=_expected_resource_guard_command(argv),
        expected_cwd=PROJECT_ROOT,
        max_age_seconds=RESOURCE_GUARD_MAX_AGE_SECONDS,
        now=now,
    )
    if not validation["valid"]:
        failures = ", ".join(str(item) for item in validation["failures"])
        raise OOFRunError(f"Local OOF resource guard was rejected: {failures}")
    return {
        "path": str(resolved_path),
        "sha256": sha256_file(resolved_path),
        "max_age_seconds": RESOURCE_GUARD_MAX_AGE_SECONDS,
        "validation": validation,
    }


def _validate_local_authorization(
    path: Path,
    *,
    config_path: Path,
    folds_path: Path,
    folds_summary_path: Path,
    resource_guard_path: Path,
    argv: Sequence[str],
) -> dict[str, Any]:
    resolved_path = path.resolve(strict=True)
    if resolved_path != LOCAL_AUTHORIZATION.resolve(strict=True):
        raise OOFRunError("Only the canonical local OOF authorization is accepted")
    try:
        payload = json.loads(
            _read_bounded(resolved_path, MAX_AUTHORIZATION_BYTES).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OOFRunError("Local OOF authorization is unreadable") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "loop_id",
        "claim_scope",
        "authorization",
        "command",
        "bindings",
        "outputs",
        "decision",
    }:
        raise OOFRunError("Local OOF authorization fields drifted")
    if (
        payload.get("schema") != "axon_loop164_local_whole_file_oof_authorization_v1"
        or payload.get("loop_id") != "loop164_whole_file_residual_expert"
        or payload.get("claim_scope") != CLAIM_SCOPE
        or payload.get("decision") != "authorized_local_train_diagnostic_only"
        or payload.get("command") != _expected_resource_guard_command(argv)
        or payload.get("authorization")
        != {
            "authorized": True,
            "authority_type": "user_explicit_local_custody_delegation",
            "authorization_date": "2026-07-13",
            "public_key_required": False,
            "external_a2_training_authority": False,
            "val_test_or_full_access": False,
            "candidate_promotion": False,
            "checkpoint_or_model_state_write": False,
            "threshold_selection": False,
        }
        or payload.get("outputs")
        != {
            "predictions": str(DEFAULT_PREDICTIONS.resolve(strict=False)),
            "report": str(DEFAULT_REPORT.resolve(strict=False)),
        }
    ):
        raise OOFRunError("Local OOF authorization scope drifted")
    expected_paths = {
        **_frozen_protocol_paths(config_path, folds_path, folds_summary_path),
        "controller": Path(__file__).resolve(),
        "resource_guard": resource_guard_path,
    }
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(expected_paths):
        raise OOFRunError("Local OOF authorization binding set drifted")
    for name, expected_path in expected_paths.items():
        binding = bindings.get(name)
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise OOFRunError(f"Local OOF authorization binding is invalid: {name}")
        bound_path = Path(str(binding.get("path") or "")).resolve(strict=True)
        if (
            bound_path != expected_path.resolve(strict=True)
            or binding.get("sha256") != sha256_file(expected_path.resolve(strict=True))
        ):
            raise OOFRunError(f"Local OOF authorization binding drifted: {name}")
    return {
        "path": str(resolved_path),
        "sha256": sha256_file(resolved_path),
        "authority_type": payload["authorization"]["authority_type"],
        "public_key_required": False,
        "external_a2_training_authority": False,
    }


def _load_canonical_loop164_modules() -> tuple[Any, Any, Any]:
    resolved_src = SRC_DIR.resolve(strict=True)
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or os.curdir).resolve(strict=False) != resolved_src
    ]
    sys.path.insert(0, str(resolved_src))
    authorized_module = importlib.import_module("loop164.authorized_input")
    contract_module = importlib.import_module("loop164.local_oof")
    model_module = importlib.import_module("loop164.whole_file_gcg")
    expected_modules = {
        authorized_module: PROJECT_ROOT / "src" / "loop164" / "authorized_input.py",
        contract_module: PROJECT_ROOT / "src" / "loop164" / "local_oof.py",
        model_module: PROJECT_ROOT / "src" / "loop164" / "whole_file_gcg.py",
    }
    for module, expected_path in expected_modules.items():
        module_path = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
        if module_path != expected_path.resolve(strict=True):
            raise OOFRunError(f"Imported non-canonical Loop164 module: {module.__name__}")
    return authorized_module, contract_module, model_module


def _exact_object(payload: object, keys: set[str], *, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != keys:
        raise OOFRunError(f"{context} fields drifted")
    return payload


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise OOFRunError(f"{key} must be a positive integer")
    return value


def _positive_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise OOFRunError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise OOFRunError(f"{key} must be finite and positive")
    return number


def _resolve_project_path(value: object, *, context: str) -> Path:
    raw_path = Path(str(value or ""))
    candidate = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(PROJECT_ROOT.resolve(strict=True))
    if any(path.is_symlink() for path in (candidate.absolute(), *candidate.absolute().parents)):
        raise OOFRunError(f"{context} cannot use symbolic links")
    return resolved


def load_config(path: Path) -> OOFConfig:
    path = path.resolve(strict=True)
    if path != DEFAULT_CONFIG.resolve(strict=True):
        raise OOFRunError("Only the canonical local OOF config is accepted")
    try:
        payload = tomllib.loads(_read_bounded(path, MAX_CONFIG_BYTES).decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise OOFRunError("Local OOF config is unreadable") from exc
    root = _exact_object(
        payload,
        {"diagnostic", "model", "input", "training", "resources"},
        context="config root",
    )
    diagnostic = _exact_object(
        root["diagnostic"],
        {
            "schema",
            "claim_scope",
            "seed",
            "fold_count",
            "expected_rows",
            "epochs",
            "decision_threshold",
            "neutral_missing_score",
            "wall_timeout_seconds",
            "per_file_timeout_seconds",
            "device",
            "precision",
        },
        context="diagnostic config",
    )
    model = _exact_object(
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
    input_config = _exact_object(
        root["input"],
        {"data_root", "max_supported_file_bytes", "bounded_read_bytes", "workers"},
        context="input config",
    )
    training = _exact_object(
        root["training"],
        {
            "optimizer",
            "learning_rate",
            "weight_decay",
            "gradient_accumulation_steps",
            "gradient_clip_norm",
        },
        context="training config",
    )
    resources = _exact_object(
        root["resources"],
        {"max_process_rss_bytes", "max_cuda_allocated_bytes", "max_cuda_reserved_bytes"},
        context="resources config",
    )
    if (
        diagnostic.get("schema") != CONFIG_SCHEMA
        or diagnostic.get("claim_scope") != CLAIM_SCOPE
        or diagnostic.get("precision") != "fp32"
        or diagnostic.get("device") not in {"cpu", "cuda"}
        or diagnostic.get("epochs") != 1
        or float(diagnostic.get("decision_threshold", -1)) != 0.5
        or float(diagnostic.get("neutral_missing_score", -1)) != 0.5
    ):
        raise OOFRunError("Diagnostic identity or fixed decision semantics drifted")
    if model.get("num_classes") != 2 or input_config.get("workers") != 0:
        raise OOFRunError("Local OOF requires binary output and a single-process loader")
    if training.get("optimizer") != "adamw":
        raise OOFRunError("Local OOF requires AdamW")
    receptive_field = _positive_int(model, "receptive_field_bytes")
    output_stride = _positive_int(model, "output_stride_bytes")
    max_outputs = _positive_int(model, "max_outputs_per_chunk")
    chunk_bytes = _positive_int(model, "chunk_bytes")
    if output_stride > receptive_field:
        raise OOFRunError("Output stride cannot exceed the receptive field")
    if chunk_bytes != (max_outputs - 1) * output_stride + receptive_field:
        raise OOFRunError("Output-coordinate chunk geometry drifted")
    if _positive_int(input_config, "bounded_read_bytes") != chunk_bytes:
        raise OOFRunError("bounded_read_bytes must equal chunk_bytes")
    return OOFConfig(
        seed=_positive_int(diagnostic, "seed"),
        fold_count=_positive_int(diagnostic, "fold_count"),
        expected_rows=_positive_int(diagnostic, "expected_rows"),
        epochs=_positive_int(diagnostic, "epochs"),
        decision_threshold=float(diagnostic["decision_threshold"]),
        neutral_missing_score=float(diagnostic["neutral_missing_score"]),
        wall_timeout_seconds=_positive_int(diagnostic, "wall_timeout_seconds"),
        per_file_timeout_seconds=_positive_int(diagnostic, "per_file_timeout_seconds"),
        device=str(diagnostic["device"]),
        embedding_dim=_positive_int(model, "embedding_dim"),
        channels=_positive_int(model, "channels"),
        receptive_field_bytes=receptive_field,
        output_stride_bytes=output_stride,
        chunk_bytes=chunk_bytes,
        max_outputs_per_chunk=max_outputs,
        data_root=_resolve_project_path(input_config["data_root"], context="data root"),
        max_supported_file_bytes=_positive_int(input_config, "max_supported_file_bytes"),
        bounded_read_bytes=_positive_int(input_config, "bounded_read_bytes"),
        learning_rate=_positive_float(training, "learning_rate"),
        weight_decay=_positive_float(training, "weight_decay"),
        gradient_accumulation_steps=_positive_int(training, "gradient_accumulation_steps"),
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

        current_process = ctypes.windll.kernel32.GetCurrentProcess  # type: ignore[attr-defined]
        current_process.argtypes = []
        current_process.restype = ctypes.c_void_p
        memory_info = ctypes.windll.psapi.GetProcessMemoryInfo  # type: ignore[attr-defined]
        memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        memory_info.restype = ctypes.c_int
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not memory_info(current_process(), ctypes.byref(counters), counters.cb):
            raise OOFRunError("Unable to read peak process memory")
        return int(counters.PeakWorkingSetSize)
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if platform.system().casefold() == "darwin" else peak * 1024


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode()
    _write_exclusive(path, raw)


def _acquire_run_lease(argv: Sequence[str]) -> dict[str, Any]:
    payload = {
        "schema": "axon_loop164_local_whole_file_oof_run_lease_v1",
        "pid": os.getpid(),
        "created_at_unix_seconds": int(time.time()),
        "cwd": str(Path.cwd().resolve(strict=False)),
        "command": _expected_resource_guard_command(argv),
        "controller_sha256": sha256_file(Path(__file__).resolve()),
    }
    _write_json(RUN_LEASE, payload)
    return payload


def _release_run_lease() -> None:
    RUN_LEASE.unlink()


def _bindings(config_path: Path, folds_path: Path, folds_summary_path: Path) -> dict[str, Any]:
    paths = {
        "config": config_path,
        "controller": Path(__file__).resolve(),
        "model": PROJECT_ROOT / "src" / "loop164" / "whole_file_gcg.py",
        "loader": PROJECT_ROOT / "src" / "loop164" / "authorized_input.py",
        "oof_contract": PROJECT_ROOT / "src" / "loop164" / "local_oof.py",
        "folds": folds_path,
        "folds_summary": folds_summary_path,
        "resource_guard": RESOURCE_GUARD,
        "local_authorization": LOCAL_AUTHORIZATION,
        "runtime_lock": PROJECT_ROOT / "requirements.txt",
    }
    return {
        name: {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _runtime(torch: Any, device: Any) -> dict[str, Any]:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "device": str(device),
        "precision": "fp32",
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
    if device.type == "cuda":
        payload.update(
            {
                "cuda": str(torch.version.cuda),
                "cudnn": int(torch.backends.cudnn.version() or 0),
                "gpu_name": torch.cuda.get_device_name(device),
            }
        )
    return payload


def run_local_oof(
    *,
    config_path: Path,
    folds_path: Path,
    folds_summary_path: Path,
    predictions_path: Path,
    report_path: Path,
    argv: Sequence[str],
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    if predictions_path.exists() or report_path.exists():
        raise FileExistsError("Local OOF outputs already exist; refusing to overwrite")
    config_path = config_path.resolve(strict=True)
    folds_path = folds_path.resolve(strict=True)
    folds_summary_path = folds_summary_path.resolve(strict=True)
    _validate_frozen_protocol(
        config_path=config_path,
        folds_path=folds_path,
        folds_summary_path=folds_summary_path,
        predictions_path=predictions_path,
        report_path=report_path,
    )
    resource_guard_validation = _validate_resource_guard(RESOURCE_GUARD, argv=argv)
    authorization_validation = _validate_local_authorization(
        LOCAL_AUTHORIZATION,
        config_path=config_path,
        folds_path=folds_path,
        folds_summary_path=folds_summary_path,
        resource_guard_path=RESOURCE_GUARD,
        argv=argv,
    )
    config = load_config(config_path)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    authorized_module, contract_module, model_module = _load_canonical_loop164_modules()
    local_probe_record_class = authorized_module.LocalProbeRecord
    streaming_source_class = authorized_module.StreamingWholeFileByteSource
    fixed_binary_metrics = contract_module.fixed_binary_metrics
    load_local_diagnostic_folds = contract_module.load_local_diagnostic_folds

    records, folds_summary = load_local_diagnostic_folds(
        folds_path=folds_path,
        summary_path=folds_summary_path,
        data_root=config.data_root,
        expected_rows=config.expected_rows,
        fold_count=config.fold_count,
        expected_seed=config.seed,
        max_supported_file_bytes=config.max_supported_file_bytes,
        expected_rows_per_fold=config.expected_rows // config.fold_count,
        expected_rows_per_label_per_fold=config.expected_rows // config.fold_count // 2,
    )
    bindings = _bindings(config_path, folds_path, folds_summary_path)
    import torch
    import torch.nn.functional as functional
    whole_file_classifier_class = model_module.WholeFileGCGClassifier

    if config.device == "cuda" and not torch.cuda.is_available():
        raise OOFRunError("CUDA is required by the canonical local OOF config")
    device = torch.device(config.device)
    run_start = clock()
    deadline = run_start + config.wall_timeout_seconds
    counters = RunCounters()
    peak_rss = _peak_process_rss_bytes()
    predictions: dict[int, dict[str, Any]] = {}
    fold_reports: list[dict[str, Any]] = []
    fatal_code: Optional[str] = None
    fatal_type: Optional[str] = None
    fatal_message: Optional[str] = None
    expected_optimizer_steps = 0

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")

    def check_deadline() -> None:
        if clock() > deadline:
            raise OOFRunError("Local OOF exceeded its global wall timeout")

    def update_resources() -> None:
        nonlocal peak_rss
        peak_rss = max(peak_rss, _peak_process_rss_bytes())
        cuda_allocated = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        cuda_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        if peak_rss > config.max_process_rss_bytes:
            raise OOFRunError("Local OOF exceeded the process RSS cap")
        if cuda_allocated > config.max_cuda_allocated_bytes:
            raise OOFRunError("Local OOF exceeded the CUDA allocated cap")
        if cuda_reserved > config.max_cuda_reserved_bytes:
            raise OOFRunError("Local OOF exceeded the CUDA reserved cap")

    def make_source(record: Any) -> Any:
        if record.source_size_bytes is None:
            raise OOFRunError("Supported OOF row has no source size")
        return streaming_source_class(
            local_probe_record_class(
                source_path=record.source_path,
                source_sha256=record.source_sha256,
                source_size_bytes=record.source_size_bytes,
                label=record.label,
            ),
            data_root=config.data_root,
            receptive_field_bytes=config.receptive_field_bytes,
            output_stride_bytes=config.output_stride_bytes,
            max_outputs_per_chunk=config.max_outputs_per_chunk,
            bounded_read_bytes=config.bounded_read_bytes,
            max_supported_file_bytes=config.max_supported_file_bytes,
            timeout_seconds=config.per_file_timeout_seconds,
            absolute_deadline=deadline,
            clock=clock,
        )

    def account_source(source: Any, *, phase: str) -> None:
        receipts = source.scan_receipts
        if len(receipts) != 2:
            raise OOFRunError("Every successful model call must complete two source passes")
        if phase == "fit":
            counters.fit_source_calls += 1
            counters.fit_scans += len(receipts)
        else:
            counters.holdout_source_calls += 1
            counters.holdout_scans += len(receipts)
        counters.verified_sha_passes += sum(
            receipt.sha256 == source.record.source_sha256 for receipt in receipts
        )
        counters.raw_bytes_read += sum(receipt.bytes_read for receipt in receipts)

    run_lease = _acquire_run_lease(argv)
    try:
        for fold in range(config.fold_count):
            check_deadline()
            fold_start = clock()
            fold_seed = int.from_bytes(
                hashlib.sha256(f"{config.seed}:{fold}".encode("ascii")).digest()[:4],
                "big",
            )
            random.seed(fold_seed)
            torch.manual_seed(fold_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(fold_seed)
            model = whole_file_classifier_class(
                embedding_dim=config.embedding_dim,
                channels=config.channels,
                receptive_field_bytes=config.receptive_field_bytes,
                output_stride_bytes=config.output_stride_bytes,
                max_outputs_per_chunk=config.max_outputs_per_chunk,
                num_classes=2,
            ).to(device=device, dtype=torch.float32)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
            optimizer.zero_grad(set_to_none=True)
            fit_records = [
                record
                for record in records
                if record.fold != fold and record.availability == "supported"
            ]
            holdout_records = [record for record in records if record.fold == fold]
            fold_expected_optimizer_steps = math.ceil(
                len(fit_records) / config.gradient_accumulation_steps
            )
            expected_optimizer_steps += fold_expected_optimizer_steps
            optimizer_steps_before_fold = counters.optimizer_steps
            fit_sha256 = {record.source_sha256 for record in fit_records}
            holdout_sha256 = {record.source_sha256 for record in holdout_records}
            fit_components = {record.component_id for record in fit_records}
            holdout_components = {record.component_id for record in holdout_records}
            if fit_sha256 & holdout_sha256 or fit_components & holdout_components:
                raise OOFRunError("Fit/holdout identity or content-component overlap detected")
            random.Random(fold_seed).shuffle(fit_records)
            accumulation = 0
            model.train()
            for fit_index, record in enumerate(fit_records):
                check_deadline()
                source = make_source(record)
                result = model.forward_from_source(source)
                source.assert_complete()
                account_source(source, phase="fit")
                logits = result["logits"]
                if not torch.isfinite(logits).all():
                    counters.nonfinite_events += 1
                    raise OOFRunError("Fit logits became non-finite")
                label = torch.tensor([record.label], dtype=torch.long, device=device)
                loss = functional.cross_entropy(logits, label)
                if not torch.isfinite(loss):
                    counters.nonfinite_events += 1
                    raise OOFRunError("Fit loss became non-finite")
                (loss / config.gradient_accumulation_steps).backward()
                counters.backward_microbatches += 1
                accumulation += 1
                if accumulation == config.gradient_accumulation_steps:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_norm
                    )
                    if not torch.isfinite(gradient_norm):
                        counters.nonfinite_events += 1
                        raise OOFRunError("Fit gradient norm became non-finite")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    counters.optimizer_steps += 1
                    accumulation = 0
                if fit_index % 64 == 0:
                    update_resources()
            if accumulation:
                correction = config.gradient_accumulation_steps / accumulation
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
                if not torch.isfinite(gradient_norm):
                    counters.nonfinite_events += 1
                    raise OOFRunError("Final fit gradient norm became non-finite")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                counters.optimizer_steps += 1
            if counters.optimizer_steps - optimizer_steps_before_fold != fold_expected_optimizer_steps:
                raise OOFRunError("Fold optimizer-step accounting drifted")

            model.eval()
            fold_labels: list[int] = []
            fold_scores: list[float] = []
            fold_missing = Counter()
            for holdout_index, record in enumerate(holdout_records):
                check_deadline()
                if record.availability != "supported":
                    fold_missing[record.missing_reason or "unknown"] += 1
                    predictions[record.train_row_index] = {
                        "schema": PREDICTION_SCHEMA,
                        "loop_id": "loop164_whole_file_residual_expert",
                        "claim_scope": CLAIM_SCOPE,
                        "split_role": "train",
                        "train_row_index": record.train_row_index,
                        "sample_index": record.sample_index,
                        "source_sha256": record.source_sha256,
                        "content_component_id": record.component_id,
                        "diagnostic_fold": record.fold,
                        "label": record.label,
                        "whole_file_probability": None,
                        "whole_file_score": config.neutral_missing_score,
                        "whole_file_uncertainty": 1.0,
                        "whole_file_missingness": 1,
                        "missing_reason": record.missing_reason,
                        "fixed_threshold_prediction": None,
                        "identity_metadata_not_model_features": [
                            "train_row_index",
                            "sample_index",
                            "source_sha256",
                            "content_component_id",
                            "diagnostic_fold",
                        ],
                    }
                    continue
                source = make_source(record)
                with torch.inference_mode():
                    logits = model.forward_from_source(source)["logits"]
                    score = float(torch.softmax(logits.float(), dim=1)[0, 1].item())
                source.assert_complete()
                account_source(source, phase="holdout")
                if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                    counters.nonfinite_events += 1
                    raise OOFRunError("Holdout score became non-finite")
                prediction: Optional[int] = int(score >= config.decision_threshold)
                missingness = 0
                missing_reason = None
                uncertainty = 1.0 - 2.0 * abs(score - 0.5)
                fold_labels.append(record.label)
                fold_scores.append(score)
                predictions[record.train_row_index] = {
                    "schema": PREDICTION_SCHEMA,
                    "loop_id": "loop164_whole_file_residual_expert",
                    "claim_scope": CLAIM_SCOPE,
                    "split_role": "train",
                    "train_row_index": record.train_row_index,
                    "sample_index": record.sample_index,
                    "source_sha256": record.source_sha256,
                    "content_component_id": record.component_id,
                    "diagnostic_fold": record.fold,
                    "label": record.label,
                    "whole_file_probability": score,
                    "whole_file_score": score,
                    "whole_file_uncertainty": uncertainty,
                    "whole_file_missingness": missingness,
                    "missing_reason": missing_reason,
                    "fixed_threshold_prediction": prediction,
                    "identity_metadata_not_model_features": [
                        "train_row_index",
                        "sample_index",
                        "source_sha256",
                        "content_component_id",
                        "diagnostic_fold",
                    ],
                }
                if holdout_index % 64 == 0:
                    update_resources()
            fold_metrics = fixed_binary_metrics(
                fold_labels,
                fold_scores,
                threshold=config.decision_threshold,
            )
            fold_report = {
                "fold": fold,
                "fit_supported_rows": len(fit_records),
                "holdout_rows": len(holdout_records),
                "holdout_supported_rows": len(fold_scores),
                "holdout_missing_by_reason": dict(sorted(fold_missing.items())),
                "supported_only_fixed_threshold_metrics": fold_metrics,
                "optimizer_steps": fold_expected_optimizer_steps,
                "elapsed_seconds": clock() - fold_start,
            }
            fold_reports.append(fold_report)
            print(json.dumps({"fold_completed": fold, **fold_report}, sort_keys=True), flush=True)
            del optimizer, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            update_resources()
        if set(predictions) != set(range(config.expected_rows)):
            raise OOFRunError("OOF predictions do not cover every train row exactly once")
        supported_row_count = sum(record.availability == "supported" for record in records)
        if counters.fit_source_calls != supported_row_count * (config.fold_count - 1):
            raise OOFRunError("Fit source-call accounting drifted")
        if counters.holdout_source_calls != supported_row_count:
            raise OOFRunError("Holdout source-call accounting drifted")
        if counters.verified_sha_passes != counters.fit_scans + counters.holdout_scans:
            raise OOFRunError("Not every OOF source pass verified its SHA")
        if counters.backward_microbatches != counters.fit_source_calls:
            raise OOFRunError("Not every fit source call contributed one backward microbatch")
        if counters.optimizer_steps != expected_optimizer_steps:
            raise OOFRunError("OOF optimizer-step accounting drifted")
    except torch.OutOfMemoryError as exc:
        counters.oom_events += 1
        fatal_code = "oom_abort"
        fatal_type = type(exc).__name__
        fatal_message = str(exc)[:1000]
    except Exception as exc:
        fatal_code = "contract_or_runtime_abort"
        fatal_type = type(exc).__name__
        fatal_message = str(exc)[:1000]

    peak_cuda_allocated = 0
    peak_cuda_reserved = 0
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_cuda_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_cuda_reserved = int(torch.cuda.max_memory_reserved(device))
    except Exception as exc:
        if fatal_code is None:
            fatal_code = "cuda_finalization_abort"
            fatal_type = type(exc).__name__
            fatal_message = str(exc)[:1000]
        else:
            suffix = f"; CUDA finalization: {type(exc).__name__}: {str(exc)[:500]}"
            fatal_message = ((fatal_message or "") + suffix)[:1000]
    elapsed = clock() - run_start
    peak_rss = max(peak_rss, _peak_process_rss_bytes())
    base_report: dict[str, Any] = {
        "schema": FAILURE_SCHEMA if fatal_code else REPORT_SCHEMA,
        "loop_id": "loop164_whole_file_residual_expert",
        "claim_scope": CLAIM_SCOPE,
        "status": "failed" if fatal_code else "completed",
        "created_at_unix_seconds": int(time.time()),
        "argv": list(argv),
        "bindings": bindings,
        "resource_guard": resource_guard_validation,
        "local_authorization": authorization_validation,
        "run_lease": run_lease,
        "runtime": _runtime(torch, device),
        "config_contract": {
            "seed": config.seed,
            "fold_count": config.fold_count,
            "epochs": config.epochs,
            "decision_threshold": config.decision_threshold,
            "threshold_selection_performed": False,
            "neutral_missing_score": config.neutral_missing_score,
            "fold_claim_scope": folds_summary["claim_scope"],
        },
        "fold_reports": fold_reports,
        "execution": {
            **counters.__dict__,
            "expected_optimizer_steps": expected_optimizer_steps,
            "elapsed_seconds": elapsed,
            "peak_process_rss_bytes": peak_rss,
            "peak_cuda_allocated_bytes": peak_cuda_allocated,
            "peak_cuda_reserved_bytes": peak_cuda_reserved,
        },
        "forbidden_outputs": {
            "checkpoint_written": False,
            "model_state_serialized": False,
            "val_test_or_full_predictions_read": False,
            "threshold_sweep": False,
        },
        "ready_for": {
            "loop164_production_oof": False,
            "candidate_promotion": False,
            "val_or_test_access": False,
            "external_certification": False,
            "f1_claim_on_full_denominator": False,
        },
        "fatal": (
            {"code": fatal_code, "exception_type": fatal_type, "message": fatal_message}
            if fatal_code
            else None
        ),
    }
    if fatal_code:
        base_report["decision"] = "local_oof_failed_not_promotable"
        _write_json(report_path, base_report)
        _release_run_lease()
        raise RuntimeError(f"Loop164 local OOF aborted: {fatal_code}")

    ordered_predictions = [predictions[index] for index in range(config.expected_rows)]
    supported_predictions = [
        prediction
        for prediction in ordered_predictions
        if prediction["whole_file_missingness"] == 0
    ]
    supported_metrics = fixed_binary_metrics(
        [int(prediction["label"]) for prediction in supported_predictions],
        [float(prediction["whole_file_score"]) for prediction in supported_predictions],
        threshold=config.decision_threshold,
    )
    missing_counts = Counter(
        str(prediction["missing_reason"])
        for prediction in ordered_predictions
        if prediction["whole_file_missingness"] == 1
    )
    missing_by_label_reason: dict[str, dict[str, int]] = {}
    for label in (0, 1):
        label_counts = Counter(
            str(prediction["missing_reason"])
            for prediction in ordered_predictions
            if prediction["whole_file_missingness"] == 1
            and int(prediction["label"]) == label
        )
        missing_by_label_reason[str(label)] = dict(sorted(label_counts.items()))
    conservative_scores = [
        (
            float(prediction["whole_file_score"])
            if prediction["whole_file_missingness"] == 0
            else (1.0 if int(prediction["label"]) == 0 else 0.0)
        )
        for prediction in ordered_predictions
    ]
    conservative_metrics = fixed_binary_metrics(
        [int(prediction["label"]) for prediction in ordered_predictions],
        conservative_scores,
        threshold=config.decision_threshold,
    )
    prediction_lines = [
        json.dumps(prediction, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        for prediction in ordered_predictions
    ]
    prediction_raw = ("\n".join(prediction_lines) + "\n").encode()
    base_report.update(
        {
            "oof": {
                "denominator": config.expected_rows,
                "supported_rows": len(supported_predictions),
                "missing_rows": config.expected_rows - len(supported_predictions),
                "missing_by_reason": dict(sorted(missing_counts.items())),
                "missing_by_label_and_reason": missing_by_label_reason,
                "coverage": len(supported_predictions) / config.expected_rows,
                "supported_only_fixed_threshold_metrics": supported_metrics,
                "supported_subset_fixed_0_5_metrics": supported_metrics,
                "supported_subset_fixed_0_5_f1": supported_metrics["f1"],
                "canonical_denominator_conservative_all_missing_wrong_metrics": (
                    conservative_metrics
                ),
                "full_denominator_f1": None,
                "full_denominator_f1_reason": (
                    "whole-file missing rows defer to a future OOF base expert; no standalone fallback "
                    "hard decision is fabricated"
                ),
            },
            "predictions": {
                "path": str(predictions_path),
                "sha256": hashlib.sha256(prediction_raw).hexdigest(),
                "record_count": len(ordered_predictions),
                "schema": PREDICTION_SCHEMA,
            },
            "decision": "local_supported_only_oof_observed_not_promotable",
        }
    )
    _write_exclusive(predictions_path, prediction_raw)
    try:
        _write_json(report_path, base_report)
    except BaseException:
        predictions_path.unlink(missing_ok=True)
        raise
    _release_run_lease()
    return base_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local train-only Loop164 whole-file OOF.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--folds-summary", type=Path, default=DEFAULT_FOLDS_SUMMARY)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(parsed_argv)
    report = run_local_oof(
        config_path=args.config,
        folds_path=args.folds,
        folds_summary_path=args.folds_summary,
        predictions_path=args.predictions,
        report_path=args.report,
        argv=parsed_argv,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
