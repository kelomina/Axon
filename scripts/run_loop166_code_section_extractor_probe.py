#!/usr/bin/env python3
"""Run the non-promotable Loop166 Train-only code-section extractor gate."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loop166.code_sections import MISSING_REASONS, extract_executable_code  # noqa: E402

DEFAULT_BUNDLE = PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_probe_bundle.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_probe_bundle_summary.json"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "random_20w_worktree"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "roadmap_9997" / "loop166" / "code_section_extractor_probe.json"
PROPOSAL = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop166_code_section_foundation" / "proposal.json"
EXPECTED_SHA256 = {
    "bundle": "90961bfed0460787e261965a3180e1b0569df0f9d275f9693daad1ccf53dc233",
    "summary": "3ab978be18a3fa6a91dc34bded3c51dee337e48903a457bd49c1616066c6db91",
    "proposal": "0bffb0fe0e5990758017014ce3ff14d43f33ef9a59b5e3feb662aaaa0fdcd5ae",
}
EXPECTED_ROWS_PER_CLASS = 128
EXPECTED_ROWS = EXPECTED_ROWS_PER_CLASS * 2
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_WALL_SECONDS = 180.0
MAX_PEAK_RSS_BYTES = 1024 * 1024 * 1024
MINIMUM_COVERAGE = 0.85
SCHEMA = "axon_loop166_code_section_extractor_probe_v1"
CLAIM_SCOPE = "local_train_only_extractor_resource_diagnostic_not_model_quality"
RECORD_KEYS = {
    "schema",
    "loop_id",
    "bundle_role",
    "split_role",
    "label",
    "source_path",
    "source_sha256",
    "source_size_bytes",
    "metadata_not_model_features",
    "source_path_usage",
    "source_sha256_usage",
}


class ExtractorProbeError(ValueError):
    """The Phase A input or execution contract is invalid."""


@dataclass(frozen=True)
class ProbeRecord:
    source_path: Path
    source_sha256: str
    source_size_bytes: int
    label: int


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExtractorProbeError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ExtractorProbeError(f"Non-finite JSON value: {value}")


def _parse_object(raw: bytes, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtractorProbeError(f"Invalid JSON: {context}") from exc
    if not isinstance(payload, dict):
        raise ExtractorProbeError(f"Expected JSON object: {context}")
    return payload


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_bounded(path: Path, max_bytes: int) -> bytes:
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ExtractorProbeError(f"Bounded input is too large: {path}")
    return raw


def _bind_file(path: Path, expected_sha256: str, max_bytes: int) -> bytes:
    path = path.resolve(strict=True)
    raw = _read_bounded(path, max_bytes)
    observed = _sha256(raw)
    if observed != expected_sha256:
        raise ExtractorProbeError(
            f"Input SHA-256 drifted for {path}: expected {expected_sha256}, observed {observed}"
        )
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


def _resolve_source(path: Path, data_root: Path) -> Path:
    try:
        relative = _lexical_relative_to(path, data_root)
    except ValueError as exc:
        raise ExtractorProbeError("Source path escapes the materialized Train root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ExtractorProbeError("Source path has invalid relative components")
    cursor = data_root.absolute()
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ExtractorProbeError("Source path cannot contain symlinks")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(data_root.resolve(strict=True))
    except ValueError as exc:
        raise ExtractorProbeError("Resolved source escapes the materialized Train root") from exc
    if not resolved.is_file():
        raise ExtractorProbeError("Source path is not a regular file")
    return resolved


def _fingerprint(path: Path) -> tuple[int, int, int, int]:
    stat_result = os.stat(path, follow_symlinks=False)
    return (
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_dev),
        int(stat_result.st_ino),
    )


def _load_records() -> tuple[list[ProbeRecord], dict[str, str]]:
    bundle_raw = _bind_file(DEFAULT_BUNDLE, EXPECTED_SHA256["bundle"], 4 * 1024 * 1024)
    summary_raw = _bind_file(DEFAULT_SUMMARY, EXPECTED_SHA256["summary"], 4 * 1024 * 1024)
    _bind_file(PROPOSAL, EXPECTED_SHA256["proposal"], 4 * 1024 * 1024)
    summary = _parse_object(summary_raw, "bundle summary")
    if (
        summary.get("schema") != "axon_loop164_local_probe_bundle_summary_v1"
        or summary.get("decision") != "local_train_only_probe_bundle_ready"
        or summary.get("ready_for")
        != {
            "local_runtime_probe_bundle": True,
            "loop164_whole_file_training": False,
            "val_or_test_access": False,
            "f1_claim": False,
        }
    ):
        raise ExtractorProbeError("Bundle summary readiness contract drifted")
    bound_bundle = summary.get("bundle")
    if not isinstance(bound_bundle, dict) or (
        bound_bundle.get("sha256") != EXPECTED_SHA256["bundle"]
        or bound_bundle.get("record_count") != EXPECTED_ROWS
    ):
        raise ExtractorProbeError("Bundle summary binding drifted")
    lines = bundle_raw.splitlines()
    if len(lines) != EXPECTED_ROWS:
        raise ExtractorProbeError("Bundle row count drifted")
    records: list[ProbeRecord] = []
    seen_sha256: set[str] = set()
    seen_paths: set[str] = set()
    for index, line in enumerate(lines):
        payload = _parse_object(line, f"bundle row {index}")
        source_sha256 = str(payload.get("source_sha256") or "").strip().casefold()
        source_path = Path(str(payload.get("source_path") or ""))
        source_size = payload.get("source_size_bytes")
        if set(payload) != RECORD_KEYS or (
            payload.get("schema") != "axon_loop164_local_probe_record_v1"
            or payload.get("bundle_role") != "local_train_only_runtime_probe"
            or payload.get("split_role") != "train"
            or payload.get("label") not in {0, 1}
            or not _is_sha256(source_sha256)
            or not isinstance(source_size, int)
            or isinstance(source_size, bool)
            or not 0 < source_size <= MAX_INPUT_BYTES
            or payload.get("metadata_not_model_features")
            != ["source_path", "source_sha256", "source_size_bytes"]
            or payload.get("source_path_usage") != "loader_identity_only_not_model_feature"
            or payload.get("source_sha256_usage")
            != "integrity_binding_only_not_model_feature"
        ):
            raise ExtractorProbeError(f"Bundle row {index} identity or role drifted")
        path_key = str(source_path).casefold()
        if source_sha256 in seen_sha256 or path_key in seen_paths:
            raise ExtractorProbeError("Bundle repeats a source identity")
        seen_sha256.add(source_sha256)
        seen_paths.add(path_key)
        records.append(
            ProbeRecord(source_path, source_sha256, int(source_size), int(payload["label"]))
        )
    if {label: sum(record.label == label for record in records) for label in (0, 1)} != {
        0: EXPECTED_ROWS_PER_CLASS,
        1: EXPECTED_ROWS_PER_CLASS,
    }:
        raise ExtractorProbeError("Bundle label balance drifted")
    return records, {
        "bundle": EXPECTED_SHA256["bundle"],
        "summary": EXPECTED_SHA256["summary"],
        "proposal": EXPECTED_SHA256["proposal"],
    }


def _read_verified(record: ProbeRecord, data_root: Path) -> bytes:
    source = _resolve_source(record.source_path, data_root)
    before = _fingerprint(source)
    if before[0] != record.source_size_bytes:
        raise ExtractorProbeError("Source size does not match the bundle")
    raw = _read_bounded(source, MAX_INPUT_BYTES)
    after = _fingerprint(source)
    if before != after:
        raise ExtractorProbeError("Source fingerprint changed during extraction")
    if len(raw) != record.source_size_bytes or _sha256(raw) != record.source_sha256:
        raise ExtractorProbeError("Source bytes do not match the bundle")
    return raw


def _quantiles(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("min", "p50", "p90", "p95", "p99", "max")}
    ordered = sorted(values)

    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min": float(ordered[0]),
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": float(ordered[-1]),
    }


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

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
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
        process = get_current_process()
        ok = get_process_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            raise ExtractorProbeError("Unable to read Windows process memory counters")
        return int(counters.PeakWorkingSetSize)
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if platform.system().casefold() == "darwin" else usage * 1024)


def run_probe() -> dict[str, Any]:
    started = time.perf_counter()
    records, bindings = _load_records()
    data_root = DEFAULT_DATA_ROOT.resolve(strict=True)
    missing = Counter()
    success_by_label = Counter()
    missing_by_label = Counter()
    code_sizes: list[int] = []
    span_counts: list[int] = []
    executable_section_counts: list[int] = []
    warning_counts: list[int] = []
    raw_bytes_read = 0
    code_bytes_observed = 0
    overlap_bytes_removed = 0
    commitment = hashlib.sha256()
    for row_index, record in enumerate(records):
        if time.perf_counter() - started > MAX_WALL_SECONDS:
            raise ExtractorProbeError("Extractor probe exceeded its wall-time contract")
        raw = _read_verified(record, data_root)
        raw_bytes_read += len(raw)
        extraction = extract_executable_code(raw)
        if extraction.missing_reason is not None:
            if extraction.missing_reason not in MISSING_REASONS:
                raise ExtractorProbeError("Extractor emitted an unknown missing reason")
            missing[extraction.missing_reason] += 1
            missing_by_label[record.label] += 1
            commitment.update(
                f"{row_index}:{record.source_sha256}:missing:{extraction.missing_reason}\n".encode(
                    "ascii"
                )
            )
            continue
        success_by_label[record.label] += 1
        code_size = len(extraction.code_bytes)
        code_sizes.append(code_size)
        span_counts.append(len(extraction.spans))
        executable_section_counts.append(extraction.declared_executable_sections)
        warning_counts.append(extraction.parser_warning_count)
        code_bytes_observed += code_size
        overlap_bytes_removed += extraction.overlap_bytes_removed
        span_text = ",".join(f"{start}-{end}" for start, end in extraction.spans)
        commitment.update(
            (
                f"{row_index}:{record.source_sha256}:available:{span_text}:"
                f"{_sha256(extraction.code_bytes)}\n"
            ).encode("ascii")
        )

    success = sum(success_by_label.values())
    missing_count = sum(missing.values())
    elapsed = time.perf_counter() - started
    peak_rss = _peak_process_rss_bytes()
    coverage = success / EXPECTED_ROWS
    gates = {
        "denominator_conserved": success + missing_count == EXPECTED_ROWS,
        "minimum_coverage": coverage >= MINIMUM_COVERAGE,
        "wall_time": elapsed <= MAX_WALL_SECONDS,
        "peak_rss": peak_rss < MAX_PEAK_RSS_BYTES,
        "silent_drop_zero": success + missing_count == len(records),
        "raw_code_output_zero": True,
    }
    return {
        "schema": SCHEMA,
        "loop_id": "loop166_code_section_foundation",
        "claim_scope": CLAIM_SCOPE,
        "protocol": {
            "split_role": "train",
            "training_performed": False,
            "model_constructed": False,
            "tokenizer_trained": False,
            "quality_metrics_computed": False,
            "threshold_operations": False,
            "val_test_or_full_access": False,
            "identity_fields_used_as_model_features": False,
            "public_key_required": False,
        },
        "input_bindings": {
            "bundle": {"path": str(DEFAULT_BUNDLE.resolve()), "sha256": bindings["bundle"]},
            "summary": {"path": str(DEFAULT_SUMMARY.resolve()), "sha256": bindings["summary"]},
            "proposal": {"path": str(PROPOSAL.resolve()), "sha256": bindings["proposal"]},
            "extractor_source": {
                "path": str((SRC_DIR / "loop166" / "code_sections.py").resolve()),
                "sha256": _sha256((SRC_DIR / "loop166" / "code_sections.py").read_bytes()),
            },
        },
        "extractor_contract": {
            "section_selector": "IMAGE_SCN_MEM_EXECUTE",
            "span_order": "ascending_raw_file_offset",
            "overlap_policy": "merge_before_concatenation_no_duplicate_raw_bytes",
            "invalid_span_policy": "whole_sample_missing",
            "section_name_used": False,
            "raw_code_persisted": False,
        },
        "counts": {
            "denominator": EXPECTED_ROWS,
            "success": success,
            "missing": missing_count,
            "missing_by_reason": {reason: missing[reason] for reason in MISSING_REASONS},
            "success_by_label": {str(label): success_by_label[label] for label in (0, 1)},
            "missing_by_label": {str(label): missing_by_label[label] for label in (0, 1)},
            "silent_drop": EXPECTED_ROWS - success - missing_count,
        },
        "coverage": coverage,
        "distributions": {
            "code_bytes": _quantiles(code_sizes),
            "merged_span_count": _quantiles(span_counts),
            "declared_executable_section_count": _quantiles(executable_section_counts),
            "parser_warning_count": _quantiles(warning_counts),
            "mean_code_bytes": statistics.fmean(code_sizes) if code_sizes else 0.0,
        },
        "resources": {
            "elapsed_seconds": elapsed,
            "peak_process_rss_bytes": peak_rss,
            "raw_bytes_read": raw_bytes_read,
            "code_bytes_observed_not_persisted": code_bytes_observed,
            "overlap_bytes_removed": overlap_bytes_removed,
            "raw_code_artifact_bytes": 0,
        },
        "extraction_commitment_sha256": commitment.hexdigest(),
        "gates": gates,
        "decision": "phase_a_extractor_gate_pass" if all(gates.values()) else "phase_a_extractor_gate_fail",
        "ready_for": {
            "one_outer_tiny_mlm_resource_cell": all(gates.values()),
            "five_fold_oof": False,
            "val_test_or_full": False,
            "promotion": False,
        },
        "target_status": {"target_f1": 0.9997, "target_achieved": False},
    }


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen Loop166 Phase A extractor gate.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    try:
        output.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ExtractorProbeError("Output must remain inside the project root") from exc
    payload = run_probe()
    _write_new_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["decision"] == "phase_a_extractor_gate_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
