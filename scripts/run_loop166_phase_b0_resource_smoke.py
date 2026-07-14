#!/usr/bin/env python3
"""Run the Train-only Loop166 Phase B0 tiny-MLM resource gate."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import random
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loop164.authorized_input import load_local_probe_bundle  # noqa: E402
from loop164.local_oof import load_local_diagnostic_folds  # noqa: E402
from loop166.byte_bpe import (  # noqa: E402
    chunk_token_ids_losslessly,
    encode_bytes,
    select_even_windows,
    tokenizer_vocab_size,
    train_byte_bpe_tokenizer,
)
from loop166.code_sections import extract_executable_code  # noqa: E402
from loop166.mlm_model import (  # noqa: E402
    TinyMaskedLanguageModel,
    TinyMLMConfig,
    count_parameters,
)

DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "manifests"
    / "roadmap_9997"
    / "loop166_code_section_foundation"
    / "phase_b0_resource_smoke.json"
)
DEFAULT_BUNDLE = (
    PROJECT_ROOT / "reports" / "roadmap_9997" / "loop164" / "local_probe_bundle.jsonl"
)
DEFAULT_BUNDLE_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "roadmap_9997"
    / "loop164"
    / "local_probe_bundle_summary.json"
)
DEFAULT_FOLDS = (
    PROJECT_ROOT
    / "reports"
    / "roadmap_9997"
    / "loop164"
    / "local_train_diagnostic_folds.jsonl"
)
DEFAULT_FOLDS_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "roadmap_9997"
    / "loop164"
    / "local_train_diagnostic_folds_summary.json"
)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "random_20w_worktree"
DEFAULT_TOKENIZER = (
    PROJECT_ROOT / "reports" / "roadmap_9997" / "loop166" / "phase_b0_tokenizer.json"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "models" / "roadmap_9997" / "loop166" / "phase_b0_tiny_mlm.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports"
    / "roadmap_9997"
    / "loop166"
    / "phase_b0_resource_smoke.json"
)

SCHEMA = "axon_loop166_phase_b0_resource_smoke_report_v1"
CLAIM_SCOPE = "local_train_only_outer_fit_subset_resource_smoke_not_model_quality"
MAX_BOUNDED_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_SOURCE_BYTES = 8 * 1024 * 1024
EXPECTED_BUNDLE_ROWS = 256
EXPECTED_FOLD_ROWS = 20_000
EXPECTED_FOLDS = 5
EXPECTED_FOLD_SEED = 164


class PhaseB0Error(ValueError):
    """The Phase B0 contract, input, or execution failed closed."""


class NonfiniteTrainingError(RuntimeError):
    """The resource smoke encountered a non-finite training value."""


@dataclass(frozen=True)
class JoinedFitRecord:
    source_path: Path
    source_sha256: str
    source_size_bytes: int
    diagnostic_fold: int


@dataclass(frozen=True)
class PreparedSequence:
    input_ids: tuple[int, ...]
    valid_tokens: int
    original_bytes: int


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PhaseB0Error(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise PhaseB0Error(f"Non-finite JSON value: {value}")


def _parse_object(raw: bytes, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseB0Error(f"Invalid JSON: {context}") from exc
    if not isinstance(payload, dict):
        raise PhaseB0Error(f"Expected JSON object: {context}")
    return payload


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_bounded(path: Path, maximum_bytes: int) -> bytes:
    with path.open("rb") as handle:
        raw = handle.read(maximum_bytes + 1)
    if len(raw) > maximum_bytes:
        raise PhaseB0Error(f"Bounded input is too large: {path}")
    return raw


def _resolve_project_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    return candidate.resolve(strict=True)


def _resolve_output_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    absolute = candidate.absolute()
    try:
        absolute.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise PhaseB0Error("Phase B0 outputs must remain inside the project root") from exc
    return absolute


def _validate_contract(contract_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path = contract_path.resolve(strict=True)
    raw = _read_bounded(contract_path, 1024 * 1024)
    contract = _parse_object(raw, "Phase B0 contract")
    authority = contract.get("authority")
    data_scope = contract.get("data_scope")
    tokenizer = contract.get("tokenizer")
    model = contract.get("model")
    training = contract.get("training")
    gates = contract.get("resource_gates")
    forbidden = contract.get("forbidden")
    if (
        contract.get("schema") != "axon_loop166_phase_b0_resource_smoke_contract_v1"
        or contract.get("loop_id") != "loop166_code_section_foundation"
        or contract.get("claim_scope") != CLAIM_SCOPE
        or not isinstance(authority, dict)
        or authority.get("user_directed_local_custody") is not True
        or authority.get("public_key_required") is not False
        or authority.get("a2_or_a3_authority") is not False
        or not isinstance(data_scope, dict)
        or data_scope.get("outer_holdout_fold") != 0
        or data_scope.get("holdout_raw_opens_allowed") != 0
        or data_scope.get("maximum_fit_records_opened") != 64
        or data_scope.get("selection")
        != "probe_bundle_order_filtered_to_diagnostic_fold_not_equal_0_then_first_64"
        or data_scope.get("sequence_overflow_policy")
        != "lossless_bpe_token_chunking"
        or data_scope.get("raw_code_persistence") is not False
        or not isinstance(tokenizer, dict)
        or tokenizer.get("algorithm") != "byte_bijective_bpe"
        or tokenizer.get("base_alphabet") != 256
        or tokenizer.get("expected_total_vocabulary") != 1029
        or tokenizer.get("fit_scope") != "selected_outer_fit_subset_only"
        or not isinstance(model, dict)
        or model.get("sequence_tokens") != 512
        or model.get("global_token_index") != 0
        or model.get("gradient_checkpointing") is not True
        or model.get("tied_input_output_embeddings") is not True
        or not isinstance(training, dict)
        or training.get("objective") != "masked_language_modeling_resource_only"
        or training.get("microbatch") != 2
        or training.get("gradient_accumulation_steps") != 2
        or training.get("optimizer_steps") != 8
        or training.get("precision") != "amp_fp16"
        or training.get("gradient_scaler_initial_scale") != 128.0
        or training.get("gradient_scaler_growth_interval") != 1000
        or training.get("quality_metric_allowed") is not False
        or training.get("threshold_operation_allowed") is not False
        or not isinstance(gates, dict)
        or gates.get("original_byte_coverage_exact_required") is not True
        or gates.get("atomic_checkpoint_required") is not True
        or gates.get("checkpoint_roundtrip_exact_logits_required") is not True
        or not isinstance(forbidden, dict)
        or forbidden.get("outer_holdout_raw_access") is not True
        or forbidden.get("labels_as_model_inputs") is not True
        or forbidden.get("identity_as_model_inputs") is not True
        or forbidden.get("quality_or_f1_claim") is not True
    ):
        raise PhaseB0Error("Phase B0 contract scope or frozen parameters drifted")

    dependency = contract.get("dependency")
    if not isinstance(dependency, dict) or (
        dependency.get("package") != "tokenizers" or dependency.get("version") != "0.22.2"
    ):
        raise PhaseB0Error("Pinned tokenizer dependency contract drifted")
    import tokenizers as tokenizers_package

    if tokenizers_package.__version__ != dependency["version"]:
        raise PhaseB0Error(
            "tokenizers runtime version drifted: "
            f"expected {dependency['version']}, observed {tokenizers_package.__version__}"
        )

    bindings = contract.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != {
        "phase_a_decision",
        "probe_bundle",
        "diagnostic_folds",
        "extractor",
    }:
        raise PhaseB0Error("Phase B0 input bindings drifted")
    observed_bindings: dict[str, Any] = {}
    for name, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise PhaseB0Error(f"Invalid binding shape: {name}")
        path = _resolve_project_path(Path(str(binding["path"])))
        bound_raw = _read_bounded(path, MAX_BOUNDED_ARTIFACT_BYTES)
        observed = _sha256(bound_raw)
        if observed != binding["sha256"]:
            raise PhaseB0Error(
                f"Binding drifted for {name}: expected {binding['sha256']}, observed {observed}"
            )
        observed_bindings[name] = {"path": str(path), "sha256": observed}
    observed_bindings["contract"] = {"path": str(contract_path), "sha256": _sha256(raw)}
    return contract, observed_bindings


def _join_outer_fit_records(
    bundle_records: Sequence[Any],
    fold_records: Sequence[Any],
    *,
    outer_holdout_fold: int,
    maximum_fit_records: int,
) -> tuple[list[JoinedFitRecord], dict[str, int]]:
    if maximum_fit_records < 1:
        raise PhaseB0Error("maximum_fit_records must be positive")
    fold_by_sha: dict[str, Any] = {}
    for fold_record in fold_records:
        sha256 = str(fold_record.source_sha256).casefold()
        if sha256 in fold_by_sha:
            raise PhaseB0Error("Diagnostic folds repeat a source SHA")
        fold_by_sha[sha256] = fold_record

    joined: list[JoinedFitRecord] = []
    outer_holdout_rows = 0
    for bundle_record in bundle_records:
        fold_record = fold_by_sha.get(str(bundle_record.source_sha256).casefold())
        if fold_record is None:
            raise PhaseB0Error("Probe bundle row is absent from diagnostic folds")
        if (
            str(bundle_record.source_path).casefold()
            != str(fold_record.source_path).casefold()
            or bundle_record.source_size_bytes != fold_record.source_size_bytes
            or bundle_record.label != fold_record.label
            or fold_record.availability != "supported"
            or fold_record.missing_reason is not None
        ):
            raise PhaseB0Error("Probe/fold join identity or availability drifted")
        if fold_record.fold == outer_holdout_fold:
            outer_holdout_rows += 1
            continue
        if len(joined) < maximum_fit_records:
            joined.append(
                JoinedFitRecord(
                    source_path=bundle_record.source_path,
                    source_sha256=bundle_record.source_sha256,
                    source_size_bytes=bundle_record.source_size_bytes,
                    diagnostic_fold=fold_record.fold,
                )
            )
    if len(joined) != maximum_fit_records:
        raise PhaseB0Error("Insufficient outer-fit rows after the fail-closed fold join")
    return joined, {
        "bundle_rows": len(bundle_records),
        "fold_rows": len(fold_records),
        "outer_holdout_metadata_rows": outer_holdout_rows,
        "outer_fit_metadata_rows": len(bundle_records) - outer_holdout_rows,
        "selected_outer_fit_rows": len(joined),
    }


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
        raise PhaseB0Error("Raw source escapes the materialized Train root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise PhaseB0Error("Raw source has invalid path components")
    cursor = data_root.absolute()
    if cursor.is_symlink():
        raise PhaseB0Error("Materialized Train root cannot be a symlink")
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise PhaseB0Error("Raw source path cannot contain symlinks")
    resolved_root = data_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise PhaseB0Error("Resolved raw source escapes the materialized Train root") from exc
    if not resolved.is_file():
        raise PhaseB0Error("Raw source is not a regular file")
    return resolved


def _fingerprint(path: Path) -> tuple[int, int, int, int]:
    stat_result = os.stat(path, follow_symlinks=False)
    return (
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_dev),
        int(stat_result.st_ino),
    )


def _read_verified_source(record: JoinedFitRecord, data_root: Path) -> bytes:
    source = _resolve_source(record.source_path, data_root)
    before = _fingerprint(source)
    if before[0] != record.source_size_bytes:
        raise PhaseB0Error("Raw source size does not match the joined metadata")
    raw = _read_bounded(source, MAX_SOURCE_BYTES)
    after = _fingerprint(source)
    if before != after:
        raise PhaseB0Error("Raw source changed during the verified open")
    if len(raw) != record.source_size_bytes or _sha256(raw) != record.source_sha256:
        raise PhaseB0Error("Raw source bytes do not match the joined SHA binding")
    return raw


def _extract_fit_windows(
    records: Sequence[JoinedFitRecord],
    *,
    data_root: Path,
    window_bytes: int,
    maximum_windows: int,
) -> tuple[list[bytes], dict[str, Any]]:
    windows: list[bytes] = []
    missing = Counter()
    raw_bytes = 0
    code_bytes = 0
    commitment = hashlib.sha256()
    for record_index, record in enumerate(records):
        raw = _read_verified_source(record, data_root)
        raw_bytes += len(raw)
        extraction = extract_executable_code(raw)
        del raw
        if extraction.missing_reason is not None:
            missing[extraction.missing_reason] += 1
            commitment.update(
                f"{record_index}:missing:{extraction.missing_reason}\n".encode("ascii")
            )
            continue
        selected_windows = select_even_windows(
            extraction.code_bytes,
            window_bytes=window_bytes,
            max_windows=maximum_windows,
        )
        if not selected_windows:
            raise PhaseB0Error("Available code extraction produced no training windows")
        code_bytes += len(extraction.code_bytes)
        for window in selected_windows:
            commitment.update(hashlib.sha256(window).digest())
        windows.extend(selected_windows)
    if not windows:
        raise PhaseB0Error("The selected outer-fit subset produced no code windows")
    return windows, {
        "fit_raw_opens": len(records),
        "outer_holdout_raw_opens": 0,
        "fit_raw_bytes_verified": raw_bytes,
        "available_code_records": len(records) - sum(missing.values()),
        "missing_code_records": sum(missing.values()),
        "missing_by_reason": dict(sorted(missing.items())),
        "code_bytes_observed_not_persisted": code_bytes,
        "selected_windows": len(windows),
        "selected_window_original_bytes": sum(map(len, windows)),
        "window_commitment_sha256": commitment.hexdigest(),
    }


def _atomic_tokenizer_save(tokenizer: Any, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        tokenizer.save(str(temporary))
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(torch_module: Any, payload: dict[str, Any], path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as handle:
            torch_module.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_sequences(
    tokenizer: Any,
    windows: Sequence[bytes],
    *,
    sequence_tokens: int,
) -> tuple[list[PreparedSequence], dict[str, int]]:
    pad_id = tokenizer.token_to_id("[PAD]")
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    if None in {pad_id, cls_id, sep_id}:
        raise PhaseB0Error("Tokenizer is missing required framing tokens")
    prepared: list[PreparedSequence] = []
    maximum_payload_tokens = sequence_tokens - 2
    if maximum_payload_tokens < 1:
        raise PhaseB0Error("sequence_tokens must leave room for CLS and SEP")
    original_window_bytes = 0
    prepared_original_bytes = 0
    split_window_count = 0
    sequence_expansion_count = 0
    for window in windows:
        chunks = chunk_token_ids_losslessly(
            tokenizer,
            window,
            max_content_tokens=maximum_payload_tokens,
        )
        if len(chunks) > 1:
            split_window_count += 1
            sequence_expansion_count += len(chunks) - 1
        original_window_bytes += len(window)
        window_prepared_bytes = 0
        for chunk in chunks:
            framed = [int(cls_id), *chunk.token_ids, int(sep_id)]
            valid_tokens = len(framed)
            framed.extend([int(pad_id)] * (sequence_tokens - valid_tokens))
            prepared.append(
                PreparedSequence(tuple(framed), valid_tokens, chunk.original_byte_length)
            )
            window_prepared_bytes += chunk.original_byte_length
        if window_prepared_bytes != len(window):
            raise PhaseB0Error("Lossless BPE token chunking did not conserve original bytes")
        prepared_original_bytes += window_prepared_bytes
    return prepared, {
        "prepared_sequences": len(prepared),
        "original_window_bytes": original_window_bytes,
        "prepared_original_bytes": prepared_original_bytes,
        "split_window_count": split_window_count,
        "sequence_expansion_count": sequence_expansion_count,
        "overlength_windows_excluded": 0,
    }


def _masked_batch(
    torch_module: Any,
    rows: Sequence[PreparedSequence],
    *,
    mask_token_id: int,
    mask_ratio: float,
    generator: Any,
) -> tuple[Any, Any, Any, int]:
    input_ids = torch_module.tensor([row.input_ids for row in rows], dtype=torch_module.long)
    attention_mask = torch_module.zeros_like(input_ids, dtype=torch_module.bool)
    labels = torch_module.full_like(input_ids, -100)
    for row_index, row in enumerate(rows):
        attention_mask[row_index, : row.valid_tokens] = True
        candidate_count = row.valid_tokens - 2
        if candidate_count < 1:
            raise PhaseB0Error("Prepared sequence contains no byte-derived tokens")
        masked_count = max(1, int(candidate_count * mask_ratio + 0.5))
        selected = torch_module.randperm(candidate_count, generator=generator)[:masked_count] + 1
        labels[row_index, selected] = input_ids[row_index, selected]
        input_ids[row_index, selected] = mask_token_id
    return input_ids, attention_mask, labels, sum(row.original_bytes for row in rows)


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
        if not get_process_memory_info(
            get_current_process(), ctypes.byref(counters), counters.cb
        ):
            raise PhaseB0Error("Unable to read Windows process memory counters")
        return int(counters.PeakWorkingSetSize)
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if platform.system().casefold() == "darwin" else usage * 1024)


def run_resource_smoke(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    bundle_path: Path = DEFAULT_BUNDLE,
    bundle_summary_path: Path = DEFAULT_BUNDLE_SUMMARY,
    folds_path: Path = DEFAULT_FOLDS,
    folds_summary_path: Path = DEFAULT_FOLDS_SUMMARY,
    data_root: Path = DEFAULT_DATA_ROOT,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
) -> dict[str, Any]:
    started = time.perf_counter()
    contract, bindings = _validate_contract(contract_path)
    data_scope = contract["data_scope"]
    tokenizer_contract = contract["tokenizer"]
    model_contract = contract["model"]
    training_contract = contract["training"]
    resource_contract = contract["resource_gates"]

    bundle_path = bundle_path.resolve(strict=True)
    folds_path = folds_path.resolve(strict=True)
    if bundle_path != Path(bindings["probe_bundle"]["path"]):
        raise PhaseB0Error("CLI probe bundle does not match the frozen binding")
    if folds_path != Path(bindings["diagnostic_folds"]["path"]):
        raise PhaseB0Error("CLI diagnostic folds do not match the frozen binding")
    data_root = data_root.resolve(strict=True)
    bundle_records, _ = load_local_probe_bundle(
        bundle_path=bundle_path,
        summary_path=bundle_summary_path,
        data_root=data_root,
        expected_records_per_class=EXPECTED_BUNDLE_ROWS // 2,
    )
    fold_records, _ = load_local_diagnostic_folds(
        folds_path=folds_path,
        summary_path=folds_summary_path,
        data_root=data_root,
        expected_rows=EXPECTED_FOLD_ROWS,
        fold_count=EXPECTED_FOLDS,
        expected_seed=EXPECTED_FOLD_SEED,
        max_supported_file_bytes=MAX_SOURCE_BYTES,
        expected_rows_per_fold=EXPECTED_FOLD_ROWS // EXPECTED_FOLDS,
        expected_rows_per_label_per_fold=EXPECTED_FOLD_ROWS // EXPECTED_FOLDS // 2,
    )
    selected_records, selection_counts = _join_outer_fit_records(
        bundle_records,
        fold_records,
        outer_holdout_fold=int(data_scope["outer_holdout_fold"]),
        maximum_fit_records=int(data_scope["maximum_fit_records_opened"]),
    )
    windows, raw_access = _extract_fit_windows(
        selected_records,
        data_root=data_root,
        window_bytes=int(data_scope["window_original_bytes"]),
        maximum_windows=int(data_scope["maximum_windows_per_file"]),
    )

    tokenizer = train_byte_bpe_tokenizer(
        windows,
        vocab_size=int(tokenizer_contract["expected_total_vocabulary"]),
        special_tokens=tuple(tokenizer_contract["special_tokens"]),
    )
    observed_vocab_size = tokenizer_vocab_size(tokenizer)
    tokenizer_path = _resolve_output_path(tokenizer_path)
    tokenizer_atomic = _atomic_tokenizer_save(tokenizer, tokenizer_path)
    from tokenizers import Tokenizer

    restored_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_roundtrip_exact = (
        tokenizer_vocab_size(restored_tokenizer) == observed_vocab_size
        and all(
            encode_bytes(tokenizer, window) == encode_bytes(restored_tokenizer, window)
            for window in windows[:8]
        )
    )
    prepared, preparation_counts = _prepare_sequences(
        restored_tokenizer,
        windows,
        sequence_tokens=int(model_contract["sequence_tokens"]),
    )
    required_training_rows = (
        int(training_contract["optimizer_steps"])
        * int(training_contract["gradient_accumulation_steps"])
        * int(training_contract["microbatch"])
    )
    if len(prepared) < required_training_rows:
        raise PhaseB0Error(
            f"Insufficient encodable windows: need {required_training_rows}, got {len(prepared)}"
        )

    import torch

    if not torch.cuda.is_available():
        raise PhaseB0Error("Phase B0 canonical AMP resource smoke requires CUDA")
    device = torch.device("cuda")
    random.seed(int(training_contract["seed"]))
    torch.manual_seed(int(training_contract["seed"]))
    torch.cuda.manual_seed_all(int(training_contract["seed"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    pad_token_id = restored_tokenizer.token_to_id("[PAD]")
    mask_token_id = restored_tokenizer.token_to_id("[MASK]")
    if pad_token_id is None or mask_token_id is None:
        raise PhaseB0Error("Recovered tokenizer is missing PAD or MASK")
    model_config = TinyMLMConfig(
        vocab_size=observed_vocab_size,
        sequence_tokens=int(model_contract["sequence_tokens"]),
        layers=int(model_contract["layers"]),
        hidden_dim=int(model_contract["hidden_dim"]),
        heads=int(model_contract["heads"]),
        ffn_dim=int(model_contract["ffn_dim"]),
        local_attention_window=int(model_contract["local_attention_window"]),
        global_token_index=int(model_contract["global_token_index"]),
        dropout=float(model_contract["dropout"]),
        activation=str(model_contract["activation"]),
        gradient_checkpointing=bool(model_contract["gradient_checkpointing"]),
        tied_input_output_embeddings=bool(model_contract["tied_input_output_embeddings"]),
        pad_token_id=int(pad_token_id),
    )
    model = TinyMaskedLanguageModel(model_config).to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_contract["learning_rate"]),
        weight_decay=float(training_contract["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=float(training_contract["gradient_scaler_initial_scale"]),
        growth_interval=int(training_contract["gradient_scaler_growth_interval"]),
        enabled=True,
    )
    optimizer.zero_grad(set_to_none=True)
    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(int(training_contract["seed"]))

    optimizer_steps = 0
    backward_microbatches = 0
    nonfinite_events = 0
    oom_events = 0
    original_training_bytes = 0
    fatal_reason: Optional[str] = None
    peak_rss = _peak_process_rss_bytes()
    training_started = time.perf_counter()
    training_deadline = started + float(resource_contract["maximum_wall_seconds"])
    model.train()
    try:
        cursor = 0
        for _step in range(int(training_contract["optimizer_steps"])):
            for _accumulation in range(int(training_contract["gradient_accumulation_steps"])):
                if time.perf_counter() > training_deadline:
                    fatal_reason = "wall_timeout"
                    break
                rows = prepared[cursor : cursor + int(training_contract["microbatch"])]
                cursor += len(rows)
                input_ids, attention_mask, labels, batch_original_bytes = _masked_batch(
                    torch,
                    rows,
                    mask_token_id=int(mask_token_id),
                    mask_ratio=float(training_contract["mask_ratio"]),
                    generator=mask_generator,
                )
                input_ids = input_ids.to(device, non_blocking=False)
                attention_mask = attention_mask.to(device, non_blocking=False)
                labels = labels.to(device, non_blocking=False)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    output = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = output["loss"]
                if not torch.isfinite(loss) or not torch.isfinite(output["logits"]).all():
                    nonfinite_events += 1
                    raise NonfiniteTrainingError("Non-finite MLM loss or logits")
                scaler.scale(
                    loss / int(training_contract["gradient_accumulation_steps"])
                ).backward()
                backward_microbatches += 1
                original_training_bytes += batch_original_bytes
                peak_rss = max(peak_rss, _peak_process_rss_bytes())
            if fatal_reason is not None:
                break
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_contract["gradient_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                nonfinite_events += 1
                raise NonfiniteTrainingError("Non-finite MLM gradient norm")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
    except torch.OutOfMemoryError:
        oom_events += 1
        fatal_reason = "cuda_out_of_memory"
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    except NonfiniteTrainingError:
        fatal_reason = "nonfinite_training"
        optimizer.zero_grad(set_to_none=True)
    training_seconds = time.perf_counter() - training_started
    throughput = original_training_bytes / training_seconds if training_seconds > 0 else 0.0

    evaluation_rows = prepared[: int(training_contract["microbatch"])]
    eval_input_ids = torch.tensor(
        [row.input_ids for row in evaluation_rows], dtype=torch.long, device=device
    )
    eval_attention_mask = torch.zeros_like(eval_input_ids, dtype=torch.bool)
    for row_index, row in enumerate(evaluation_rows):
        eval_attention_mask[row_index, : row.valid_tokens] = True
    model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        reference_logits = model(
            eval_input_ids, attention_mask=eval_attention_mask
        )["logits"].detach().cpu()

    checkpoint_path = _resolve_output_path(checkpoint_path)
    checkpoint_payload = {
        "schema": "axon_loop166_phase_b0_tiny_mlm_checkpoint_v1",
        "model_config": asdict(model_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "optimizer_steps": optimizer_steps,
        "tokenizer_sha256": _sha256(tokenizer_path.read_bytes()),
    }
    checkpoint_atomic = _atomic_torch_save(torch, checkpoint_payload, checkpoint_path)
    restored_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(restored_payload, dict) or (
        restored_payload.get("schema") != "axon_loop166_phase_b0_tiny_mlm_checkpoint_v1"
        or not isinstance(restored_payload.get("model_config"), dict)
        or not isinstance(restored_payload.get("model_state_dict"), dict)
    ):
        raise PhaseB0Error("weights_only checkpoint recovery contract failed")
    restored_config = TinyMLMConfig(**restored_payload["model_config"])
    restored_model = TinyMaskedLanguageModel(restored_config).to(device)
    restored_model.load_state_dict(restored_payload["model_state_dict"], strict=True)
    restored_model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        restored_logits = restored_model(
            eval_input_ids, attention_mask=eval_attention_mask
        )["logits"].detach().cpu()
    exact_eval_logits = torch.equal(reference_logits, restored_logits)

    elapsed = time.perf_counter() - started
    peak_rss = max(peak_rss, _peak_process_rss_bytes())
    peak_cuda_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_cuda_reserved = int(torch.cuda.max_memory_reserved(device))
    gates = {
        "contract_bindings_exact": True,
        "selected_outer_fit_rows_exact": len(selected_records)
        == int(data_scope["maximum_fit_records_opened"]),
        "fit_raw_open_cap": raw_access["fit_raw_opens"]
        <= int(data_scope["maximum_fit_records_opened"]),
        "outer_holdout_raw_opens_zero": raw_access["outer_holdout_raw_opens"]
        == int(data_scope["holdout_raw_opens_allowed"]),
        "tokenizer_vocabulary_exact": observed_vocab_size
        == int(tokenizer_contract["expected_total_vocabulary"]),
        "tokenizer_atomic": tokenizer_atomic,
        "tokenizer_roundtrip_exact": tokenizer_roundtrip_exact,
        "sequence_byte_coverage_exact": preparation_counts["original_window_bytes"]
        == raw_access["selected_window_original_bytes"]
        == preparation_counts["prepared_original_bytes"]
        and preparation_counts["overlength_windows_excluded"] == 0,
        "model_parameter_range": int(model_contract["minimum_parameters"])
        <= parameter_count
        <= int(model_contract["maximum_parameters"]),
        "minimum_optimizer_steps": optimizer_steps
        >= int(resource_contract["minimum_optimizer_steps"]),
        "nonfinite_events": nonfinite_events <= int(resource_contract["nonfinite_allowed"]),
        "oom_events": oom_events <= int(resource_contract["oom_allowed"]),
        "wall_time": elapsed <= float(resource_contract["maximum_wall_seconds"]),
        "cuda_allocated": peak_cuda_allocated
        < int(resource_contract["maximum_cuda_allocated_bytes"]),
        "cuda_reserved": peak_cuda_reserved
        < int(resource_contract["maximum_cuda_reserved_bytes"]),
        "process_rss": peak_rss < int(resource_contract["maximum_process_rss_bytes"]),
        "minimum_original_byte_throughput": throughput
        >= float(resource_contract["minimum_original_bytes_per_training_second"]),
        "checkpoint_atomic": checkpoint_atomic,
        "checkpoint_weights_only_roundtrip": True,
        "checkpoint_roundtrip_exact_logits": exact_eval_logits,
        "raw_code_artifact_bytes_zero": True,
        "quality_metrics_not_computed": True,
        "threshold_operations_not_performed": True,
    }
    passed = all(gates.values()) and fatal_reason is None
    return {
        "schema": SCHEMA,
        "loop_id": "loop166_code_section_foundation",
        "claim_scope": CLAIM_SCOPE,
        "input_bindings": {
            **bindings,
            "controller_source": {
                "path": str(Path(__file__).resolve(strict=True)),
                "sha256": _sha256(Path(__file__).resolve(strict=True).read_bytes()),
            },
            "bundle_summary": {
                "path": str(bundle_summary_path.resolve(strict=True)),
                "sha256": _sha256(bundle_summary_path.resolve(strict=True).read_bytes()),
            },
            "folds_summary": {
                "path": str(folds_summary_path.resolve(strict=True)),
                "sha256": _sha256(folds_summary_path.resolve(strict=True).read_bytes()),
            },
            "byte_bpe_source": {
                "path": str((SRC_DIR / "loop166" / "byte_bpe.py").resolve(strict=True)),
                "sha256": _sha256((SRC_DIR / "loop166" / "byte_bpe.py").read_bytes()),
            },
            "mlm_model_source": {
                "path": str((SRC_DIR / "loop166" / "mlm_model.py").resolve(strict=True)),
                "sha256": _sha256((SRC_DIR / "loop166" / "mlm_model.py").read_bytes()),
            },
        },
        "selection": selection_counts,
        "raw_access": raw_access,
        "tokenizer": {
            "algorithm": tokenizer_contract["algorithm"],
            "fit_scope": tokenizer_contract["fit_scope"],
            "vocabulary_size": observed_vocab_size,
            "special_tokens": tokenizer_contract["special_tokens"],
            "normalizer": tokenizer_contract["normalizer"],
            "pre_tokenizer": tokenizer_contract["pre_tokenizer"],
            "roundtrip_exact": tokenizer_roundtrip_exact,
        },
        "sequence_preparation": {
            "overflow_policy": data_scope["sequence_overflow_policy"],
            **preparation_counts,
        },
        "model": {
            "config": asdict(model_config),
            "parameter_count": parameter_count,
            "tied_input_output_embeddings": (
                model.lm_head.weight.data_ptr() == model.token_embeddings.weight.data_ptr()
            ),
        },
        "training": {
            "objective": training_contract["objective"],
            "precision": training_contract["precision"],
            "microbatch": training_contract["microbatch"],
            "gradient_accumulation_steps": training_contract[
                "gradient_accumulation_steps"
            ],
            "gradient_scaler_initial_scale": training_contract[
                "gradient_scaler_initial_scale"
            ],
            "gradient_scaler_growth_interval": training_contract[
                "gradient_scaler_growth_interval"
            ],
            "backward_microbatches": backward_microbatches,
            "optimizer_steps": optimizer_steps,
            "original_code_window_bytes_processed": original_training_bytes,
            "training_seconds": training_seconds,
            "original_bytes_per_training_second": throughput,
            "nonfinite_events": nonfinite_events,
            "oom_events": oom_events,
            "fatal_reason": fatal_reason,
            "labels_used_as_model_inputs": False,
            "identity_used_as_model_inputs": False,
            "quality_metrics_computed": False,
            "threshold_operations": False,
        },
        "resources": {
            "elapsed_seconds": elapsed,
            "peak_process_rss_bytes": peak_rss,
            "peak_cuda_allocated_bytes": peak_cuda_allocated,
            "peak_cuda_reserved_bytes": peak_cuda_reserved,
        },
        "artifacts": {
            "tokenizer": {
                "path": str(tokenizer_path),
                "sha256": _sha256(tokenizer_path.read_bytes()),
                "atomic": tokenizer_atomic,
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path.read_bytes()),
                "atomic": checkpoint_atomic,
                "loaded_with_weights_only": True,
                "roundtrip_exact_eval_logits": exact_eval_logits,
            },
            "raw_code_artifact_bytes": 0,
        },
        "gates": gates,
        "decision": "phase_b0_resource_gate_pass" if passed else "phase_b0_resource_gate_fail",
        "ready_for": {
            "one_full_outer_fit_resource_cell": passed,
            "five_fold_oof": False,
            "val_test_or_full": False,
            "promotion": False,
        },
        "target_status": {"target_f1": 0.9997, "target_achieved": False},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen Loop166 Phase B0 tiny-MLM resource gate."
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--bundle-summary", type=Path, default=DEFAULT_BUNDLE_SUMMARY)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--folds-summary", type=Path, default=DEFAULT_FOLDS_SUMMARY)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tokenizer-output", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--checkpoint-output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_resource_smoke(
        contract_path=args.contract,
        bundle_path=args.bundle,
        bundle_summary_path=args.bundle_summary,
        folds_path=args.folds,
        folds_summary_path=args.folds_summary,
        data_root=args.data_root,
        tokenizer_path=args.tokenizer_output,
        checkpoint_path=args.checkpoint_output,
    )
    output = _resolve_output_path(args.output)
    _atomic_json_save(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["decision"] == "phase_b0_resource_gate_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
