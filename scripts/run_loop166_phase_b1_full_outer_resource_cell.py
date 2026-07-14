#!/usr/bin/env python3
"""Loop166 Phase B1 full outer-fit resource-cell controller."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import platform
import secrets
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from struct import pack
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loop164.local_oof import LocalOOFRecord, load_local_diagnostic_folds  # noqa: E402
from loop166.b1_schedule import (  # noqa: E402
    deterministic_permutation,
    iter_optimizer_groups,
    mask_content_batch,
    optimizer_group_from_cursor,
    permutation_commitment_sha256,
    validate_exact_once_schedule,
)
from loop166.byte_bpe import (  # noqa: E402
    chunk_token_ids_losslessly,
    encode_bytes,
    tokenizer_vocab_size,
    train_byte_bpe_tokenizer,
)
from loop166.code_sections import MISSING_REASONS, extract_executable_code  # noqa: E402
from loop166.compact_corpus import (  # noqa: E402
    CompactSequenceCorpus,
    materialize_framed_batch,
)
from loop166.mlm_model import TinyMaskedLanguageModel, TinyMLMConfig, count_parameters  # noqa: E402

DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "manifests"
    / "roadmap_9997"
    / "loop166_code_section_foundation"
    / "phase_b1_full_outer_resource_cell.json"
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
    PROJECT_ROOT / "reports" / "roadmap_9997" / "loop166" / "phase_b1_tokenizer.json"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "models" / "roadmap_9997" / "loop166" / "phase_b1_tiny_mlm.pt"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "roadmap_9997"
    / "loop166"
    / "phase_b1_full_outer_resource_cell.json"
)
DEFAULT_RUN_AUTH = (
    PROJECT_ROOT
    / "manifests"
    / "roadmap_9997"
    / "loop166_code_section_foundation"
    / "phase_b1_run_authorization.json"
)
DEFAULT_MARKER = (
    PROJECT_ROOT
    / "reports"
    / "roadmap_9997"
    / "loop166"
    / "phase_b1_execution_consumed.json"
)
DEFAULT_FINAL_VERIFY_RECEIPT = (
    PROJECT_ROOT
    / "reports"
    / "roadmap_9997"
    / "loop166"
    / "phase_b1_final_verify_receipt.json"
)

CONTRACT_SCHEMA = "axon_loop166_phase_b1_full_outer_resource_cell_contract_v1"
REPORT_SCHEMA = "axon_loop166_phase_b1_full_outer_resource_cell_report_v1"
CLAIM_SCOPE = "local_train_only_one_full_outer_fit_resource_cell_not_model_quality"
EXPECTED_FOLD_ROWS = 20_000
EXPECTED_FOLDS = 5
EXPECTED_FOLD_SEED = 164
FOLD_SOURCE_SIZE_LIMIT = 8 * 1024 * 1024
STATIC_ARTIFACT_LIMIT = 64 * 1024 * 1024
READ_CHUNK_BYTES = 4 * 1024 * 1024
LOWER_HEX = frozenset("0123456789abcdef")
ALLOWED_SCAN_MISSING_REASONS = frozenset({"source_unavailable", *MISSING_REASONS})
IMPORT_CLOSURE_BINDING_NAMES = frozenset(
    {
        "extractor",
        "compact_corpus",
        "b1_schedule",
        "local_oof",
        "loop164_package",
        "loop166_package",
        "byte_bpe",
        "mlm_model",
        "contract_tests",
    }
)


class B1FatalError(RuntimeError):
    """A Phase B1 integrity or frozen-contract violation."""


@dataclass(frozen=True)
class VerifiedSource:
    raw_bytes: bytes
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class OuterFitScope:
    records: tuple[LocalOOFRecord, ...]
    audit: dict[str, Any]


@dataclass(frozen=True)
class OuterFitScan:
    windows: tuple[bytes, ...]
    accounting: dict[str, Any]
    outer_fit_corpus_commitment_sha256: str


@dataclass(frozen=True)
class CompactCorpusBuild:
    corpus: CompactSequenceCorpus
    accounting: dict[str, Any]


@dataclass(frozen=True)
class RunHandoff:
    authorization_sha256: str
    marker_sha256: str
    handoff_nonce: str
    parent_pid: int
    canonical_parent_argv_sha256: str

    @property
    def handoff_nonce_sha256(self) -> str:
        return _sha256(self.handoff_nonce.encode("ascii"))


@dataclass(frozen=True)
class ValidatedSchedule:
    groups: tuple[Any, ...]
    build_seconds: float
    permutation_commitment_sha256: str


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise B1FatalError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise B1FatalError(f"Non-finite JSON value: {value}")


def _parse_json_object(raw: bytes, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise B1FatalError(f"Invalid JSON: {context}") from exc
    if not isinstance(payload, dict):
        raise B1FatalError(f"Expected JSON object: {context}")
    return payload


def _is_lower_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= LOWER_HEX


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_bounded(path: Path, maximum_bytes: int) -> bytes:
    try:
        with path.open("rb") as handle:
            raw = handle.read(maximum_bytes + 1)
    except OSError as exc:
        raise B1FatalError(f"Unable to read bound artifact: {path}") from exc
    if len(raw) > maximum_bytes:
        raise B1FatalError(f"Bound artifact is too large: {path}")
    return raw


def _resolve_bound_path(path_text: object) -> Path:
    path = Path(str(path_text or ""))
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise B1FatalError(f"Bound path is unavailable: {candidate}") from exc


def _verify_binding(name: str, binding: object) -> dict[str, str]:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise B1FatalError(f"Invalid static binding: {name}")
    expected = binding.get("sha256")
    if not _is_lower_sha256(expected):
        raise B1FatalError(f"Static binding is pending or malformed: {name}")
    path = _resolve_bound_path(binding.get("path"))
    observed = _sha256(_read_bounded(path, STATIC_ARTIFACT_LIMIT))
    if observed != expected:
        raise B1FatalError(
            f"Static binding drifted for {name}: expected {expected}, observed {observed}"
        )
    return {"path": str(path), "sha256": observed}


def _validate_import_source_paths(observed: dict[str, dict[str, str]]) -> None:
    module_names = {
        "local_oof": LocalOOFRecord.__module__,
        "loop164_package": "loop164",
        "loop166_package": "loop166",
        "byte_bpe": chunk_token_ids_losslessly.__module__,
        "extractor": extract_executable_code.__module__,
        "compact_corpus": CompactSequenceCorpus.__module__,
        "b1_schedule": deterministic_permutation.__module__,
        "mlm_model": TinyMaskedLanguageModel.__module__,
    }
    for binding_name, module_name in module_names.items():
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve(strict=True) != Path(
            observed[binding_name]["path"]
        ):
            raise B1FatalError(
                f"Imported module source is shadowed or unbound: {binding_name}"
            )


def _validate_b0_gate(report_path: Path, final_bindings: dict[str, Any]) -> None:
    report = _parse_json_object(_read_bounded(report_path, 8 * 1024 * 1024), "bound B0 report")
    sequence = report.get("sequence_preparation")
    gates = report.get("gates")
    training = report.get("training")
    if (
        report.get("decision") != final_bindings.get("required_report_decision")
        or not isinstance(gates, dict)
        or gates.get("sequence_byte_coverage_exact")
        is not final_bindings.get("required_sequence_byte_coverage_gate")
        or not isinstance(sequence, dict)
        or sequence.get("overlength_windows_excluded")
        != final_bindings.get("required_overlength_windows_excluded")
        or sequence.get("original_window_bytes") != sequence.get("prepared_original_bytes")
        or not isinstance(training, dict)
        or training.get("quality_metrics_computed") is not False
        or training.get("threshold_operations") is not False
    ):
        raise B1FatalError("Bound B0 zero-drop report no longer satisfies the B1 gate")


def _validate_b0_decision(path: Path) -> None:
    decision = _parse_json_object(_read_bounded(path, 4 * 1024 * 1024), "bound B0 decision")
    protocol = decision.get("protocol")
    ready = decision.get("ready_for")
    remediation = decision.get("zero_drop_remediation")
    if (
        decision.get("schema") != "axon_loop166_phase_b0_decision_v1"
        or decision.get("decision")
        != "phase_b0_zero_drop_resource_gate_pass_allow_one_b1_full_outer_resource_cell"
        or not isinstance(protocol, dict)
        or protocol.get("outer_holdout_raw_access") is not False
        or protocol.get("quality_metrics_computed") is not False
        or protocol.get("threshold_operations_performed") is not False
        or not isinstance(ready, dict)
        or ready.get("phase_b1_full_outer_resource_cell") is not True
        or not isinstance(remediation, dict)
        or remediation.get("byte_coverage_exact") is not True
        or remediation.get("overlength_windows_excluded") != 0
    ):
        raise B1FatalError("Bound B0 decision does not authorize exactly one B1 resource cell")


def validate_static_preflight(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    controller_path: Optional[Path] = None,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Validate every executable binding before any raw source can be opened."""
    contract_path = contract_path.resolve(strict=True)
    contract_raw = _read_bounded(contract_path, 2 * 1024 * 1024)
    contract = _parse_json_object(contract_raw, "Phase B1 contract")
    authority = contract.get("authority")
    data_scope = contract.get("data_scope")
    extraction = contract.get("extraction")
    tokenizer = contract.get("tokenizer")
    sequence = contract.get("lossless_sequence_preparation")
    storage = contract.get("compact_sequence_storage")
    model = contract.get("model")
    training = contract.get("training")
    checkpoint = contract.get("checkpoint_and_resume")
    resources = contract.get("resource_gates")
    forbidden = contract.get("forbidden")
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("loop_id") != "loop166_code_section_foundation"
        or contract.get("claim_scope") != CLAIM_SCOPE
        or not isinstance(authority, dict)
        or authority.get("user_directed_local_custody") is not True
        or authority.get("public_key_required") is not False
        or authority.get("val_test_or_full_authority") is not False
        or not isinstance(data_scope, dict)
        or data_scope.get("outer_holdout_fold") != 0
        or data_scope.get("outer_holdout_raw_opens_allowed") != 0
        or data_scope.get("outer_fit_folds") != [1, 2, 3, 4]
        or data_scope.get("outer_fit_metadata_rows") != 16_000
        or data_scope.get("known_size_records_to_attempt") != 15_988
        or data_scope.get("prior_source_unavailable_records_to_retry") != 12
        or data_scope.get("maximum_source_file_bytes") != 256 * 1024 * 1024
        or not isinstance(extraction, dict)
        or extraction.get("maximum_windows_per_file") != 8
        or extraction.get("window_original_bytes") != 512
        or extraction.get("raw_code_persistence") is not False
        or set(extraction.get("allowed_missing_reasons", ())) != ALLOWED_SCAN_MISSING_REASONS
        or not isinstance(tokenizer, dict)
        or tokenizer.get("algorithm") != "byte_bijective_bpe"
        or tokenizer.get("fit_scope") != "all_successful_outer_fit_windows_only"
        or tokenizer.get("fresh_fit_required") is not True
        or tokenizer.get("reuse_phase_b0_tokenizer") is not False
        or tokenizer.get("expected_total_vocabulary") != 1029
        or tokenizer.get("unknown_content_token_allowed") is not False
        or not isinstance(sequence, dict)
        or sequence.get("helper") != "chunk_token_ids_losslessly"
        or sequence.get("sequence_tokens") != 512
        or sequence.get("max_content_tokens") != 510
        or sequence.get("per_window_maximum_sequences_after_split") != 2
        or not isinstance(storage, dict)
        or storage.get("required") is not True
        or storage.get("scope") != "in_memory_only"
        or storage.get("flat_token_buffer_plus_offsets_required") is not True
        or storage.get("python_int_tuple_per_padded_sequence_forbidden") is not True
        or storage.get("durable_token_cache_allowed") is not False
        or not isinstance(model, dict)
        or model.get("sequence_tokens") != 512
        or model.get("layers") != 6
        or model.get("hidden_dim") != 384
        or model.get("heads") != 6
        or model.get("ffn_dim") != 1536
        or model.get("fresh_initialization_required") is not True
        or model.get("phase_b0_checkpoint_initialization_allowed") is not False
        or not isinstance(training, dict)
        or training.get("seed") != 166
        or training.get("epochs") != 1
        or training.get("deterministic_shuffle") is not True
        or training.get("shuffle_seed") != 166
        or training.get("microbatch") != 2
        or training.get("gradient_accumulation_steps") != 2
        or training.get("partial_final_step_repeat_or_padding_with_samples_allowed")
        is not False
        or training.get("partial_final_gradient_normalized_by_actual_sequence_count")
        is not True
        or training.get("each_sequence_visit_count") != 1
        or training.get("mask_ratio") != 0.25
        or training.get("optimizer") != "adamw"
        or training.get("learning_rate") != 0.0002
        or training.get("weight_decay") != 0.01
        or training.get("precision") != "amp_fp16"
        or training.get("grad_scaler")
        != {
            "init_scale": 128,
            "growth_interval": 1000,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
        }
        or training.get("training_loss_reporting_allowed") is not False
        or training.get("quality_metric_allowed") is not False
        or not isinstance(checkpoint, dict)
        or checkpoint.get("fresh_process_resume_required") is not True
        or checkpoint.get("fresh_process_resume_at_optimizer_step") != 4096
        or checkpoint.get("checkpoint_interval_optimizer_steps") != 4096
        or checkpoint.get("weights_only_load_required") is not True
        or checkpoint.get("resume_rebuilds_compact_corpus_from_bound_outer_fit_raw")
        is not True
        or checkpoint.get("checkpoint_roundtrip_exact_synthetic_logits_required")
        is not True
        or not isinstance(resources, dict)
        or resources.get("maximum_cumulative_wall_seconds") != 28_800
        or resources.get("maximum_cuda_allocated_bytes_exclusive") != 8_000_000_000
        or resources.get("maximum_cuda_reserved_bytes_exclusive") != 8_000_000_000
        or resources.get("maximum_process_rss_bytes_exclusive") != 12_000_000_000
        or resources.get("minimum_original_bytes_per_training_second") != 2000
        or resources.get("minimum_free_disk_bytes_before_raw_open") != 1_073_741_824
        or resources.get("minimum_free_disk_bytes_during_run") != 536_870_912
        or resources.get("maximum_total_durable_output_bytes") != 536_870_912
        or resources.get("nonfinite_events_allowed") != 0
        or resources.get("oom_events_allowed") != 0
        or not isinstance(forbidden, dict)
        or forbidden.get("outer_holdout_raw_access") is not True
        or forbidden.get("labels_as_model_inputs") is not True
        or forbidden.get("identity_as_model_inputs") is not True
        or forbidden.get("threshold_search_or_sweep") is not True
    ):
        raise B1FatalError("Phase B1 frozen execution contract drifted")

    observed: dict[str, dict[str, str]] = {
        "contract": {"path": str(contract_path), "sha256": _sha256(contract_raw)}
    }
    static_bindings = contract.get("static_bindings")
    if not isinstance(static_bindings, dict) or set(static_bindings) != {
        "proposal",
        "diagnostic_folds",
        "diagnostic_folds_summary",
        *IMPORT_CLOSURE_BINDING_NAMES,
    }:
        raise B1FatalError("Phase B1 static binding set drifted")
    for name, binding in static_bindings.items():
        observed[name] = _verify_binding(name, binding)
    _validate_import_source_paths(observed)

    final_b0 = contract.get("final_b0_zero_drop_bindings")
    if not isinstance(final_b0, dict) or final_b0.get("status") != "bound_zero_drop_gate_pass":
        raise B1FatalError("Final B0 zero-drop closure is not bound")
    b0_binding_names = (
        "decision_manifest",
        "report",
        "contract",
        "controller_source",
        "byte_bpe_source",
        "mlm_model_source",
        "contract_tests",
    )
    for name in b0_binding_names:
        observed[f"final_b0_{name}"] = _verify_binding(f"final_b0_{name}", final_b0.get(name))
    _validate_b0_decision(Path(observed["final_b0_decision_manifest"]["path"]))
    _validate_b0_gate(Path(observed["final_b0_report"]["path"]), final_b0)

    closure = contract.get("execution_closure")
    if not isinstance(closure, dict):
        raise B1FatalError("Phase B1 execution closure is missing")
    expected_controller_sha = closure.get("phase_b1_controller_sha256")
    if not _is_lower_sha256(expected_controller_sha):
        raise B1FatalError("Phase B1 controller binding is still pending")
    if closure.get("run_allowed_with_pending_binding") is not True:
        raise B1FatalError("Phase B1 execution closure is not enabled")
    controller = (controller_path or Path(__file__)).resolve(strict=True)
    expected_controller = _resolve_bound_path(closure.get("phase_b1_controller_path"))
    if controller != expected_controller:
        raise B1FatalError("Executed B1 controller path differs from the frozen closure")
    controller_sha = _sha256(_read_bounded(controller, 8 * 1024 * 1024))
    if controller_sha != expected_controller_sha:
        raise B1FatalError("Executed B1 controller source SHA differs from the frozen closure")
    observed["phase_b1_controller"] = {"path": str(controller), "sha256": controller_sha}
    return contract, observed


def canonical_parent_argv() -> tuple[str, ...]:
    return (
        str(Path(sys.executable).resolve(strict=True)),
        str(Path(__file__).resolve(strict=True)),
        "--contract",
        str(DEFAULT_CONTRACT.resolve(strict=True)),
        "--folds",
        str(DEFAULT_FOLDS.resolve(strict=True)),
        "--folds-summary",
        str(DEFAULT_FOLDS_SUMMARY.resolve(strict=True)),
        "--data-root",
        str(DEFAULT_DATA_ROOT.resolve(strict=True)),
        "--tokenizer-output",
        str(DEFAULT_TOKENIZER.absolute()),
        "--checkpoint-output",
        str(DEFAULT_CHECKPOINT.absolute()),
        "--report-output",
        str(DEFAULT_REPORT.absolute()),
    )


def _argv_sha256(argv: Sequence[str]) -> str:
    raw = json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256(raw)


def _runtime_binding() -> dict[str, Any]:
    import pefile
    import tokenizers
    import torch

    return {
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "torch_version": str(torch.__version__),
        "tokenizers_version": str(tokenizers.__version__),
        "pefile_version": str(pefile.__version__),
        "version_match_policy": "exact",
        "runtime_drift_action": "fail_before_lease_consumption_or_raw_open",
    }


def validate_run_authorization(
    contract: dict[str, Any],
    observed_bindings: dict[str, dict[str, str]],
    *,
    authorization_path: Path = DEFAULT_RUN_AUTH,
) -> tuple[dict[str, Any], str]:
    """Bind the one-shot run to code, runtime, canonical paths, and canonical argv."""
    try:
        authorization_path = authorization_path.resolve(strict=True)
    except OSError as exc:
        raise B1FatalError("Phase B1 run authorization is absent or still pending") from exc
    raw = _read_bounded(authorization_path, 4 * 1024 * 1024)
    authorization = _parse_json_object(raw, "Phase B1 run authorization")
    bindings = authorization.get("bindings")
    authority = authorization.get("authority")
    if (
        authorization.get("schema") != "axon_loop166_phase_b1_run_authorization_v1"
        or authorization.get("loop_id") != "loop166_code_section_foundation"
        or authorization.get("claim_scope")
        != "local_train_only_one_full_outer_fit_resource_cell_run_authorization"
        or authorization.get("status") != "granted_source_closure"
        or authorization.get("authorization_granted") is not True
        or authorization.get("decision")
        != "authorize_one_phase_b1_full_outer_resource_cell"
        or authorization.get("research_champion") != "Loop151"
        or authority
        != {
            "user_directed_local_custody": True,
            "public_key_required": False,
            "a2_or_a3_authority": False,
            "val_test_or_full_authority": False,
            "promotion_authority": False,
        }
        or not isinstance(bindings, dict)
    ):
        raise B1FatalError("Phase B1 run authorization identity or decision drifted")
    expected_direct_bindings = {
        "phase_b0_decision": observed_bindings["final_b0_decision_manifest"],
        "phase_b1_contract": observed_bindings["contract"],
        "phase_b1_controller": observed_bindings["phase_b1_controller"],
        "phase_b1_tests": observed_bindings["contract_tests"],
    }
    if bindings != expected_direct_bindings:
        raise B1FatalError("Phase B1 authorization direct bindings drifted")
    if authorization.get("runtime") != _runtime_binding():
        raise B1FatalError("Phase B1 authorization runtime binding drifted")
    canonical_invocation = authorization.get("canonical_invocation")
    parent_invocation = (
        canonical_invocation.get("parent") if isinstance(canonical_invocation, dict) else None
    )
    if not isinstance(parent_invocation, dict) or (
        parent_invocation.get("mode") != "parent"
        or parent_invocation.get("argv") != list(canonical_parent_argv())
        or canonical_invocation.get("unlisted_arguments_allowed") is not False
        or canonical_invocation.get("path_alias_or_symlink_allowed") is not False
        or canonical_invocation.get("working_directory")
        != str(PROJECT_ROOT.resolve(strict=True))
    ):
        raise B1FatalError("Phase B1 authorization canonical argv drifted")
    canonical_paths = authorization.get("canonical_paths")
    if canonical_paths != {
        "project_root": str(PROJECT_ROOT.resolve(strict=True)),
        "contract": str(DEFAULT_CONTRACT.resolve(strict=True)),
        "folds": str(DEFAULT_FOLDS.resolve(strict=True)),
        "folds_summary": str(DEFAULT_FOLDS_SUMMARY.resolve(strict=True)),
        "data_root": str(DEFAULT_DATA_ROOT.resolve(strict=True)),
        "tokenizer_output": str(DEFAULT_TOKENIZER.absolute()),
        "checkpoint_output": str(DEFAULT_CHECKPOINT.absolute()),
        "report_output": str(DEFAULT_REPORT.absolute()),
        "controller": str(Path(__file__).resolve(strict=True)),
        "tests": str(Path(observed_bindings["contract_tests"]["path"])),
        "run_authorization": str(DEFAULT_RUN_AUTH.resolve(strict=True)),
        "consumption_marker": str(DEFAULT_MARKER.absolute()),
        "final_verify_receipt": str(DEFAULT_FINAL_VERIFY_RECEIPT.absolute()),
    }:
        raise B1FatalError("Phase B1 authorization canonical path binding drifted")
    closure = authorization.get("runtime_source_closure")
    sources = closure.get("sources") if isinstance(closure, dict) else None
    auth_to_contract_names = {
        "loop164_package": "loop164_package",
        "loop164_local_oof": "local_oof",
        "loop166_package": "loop166_package",
        "loop166_byte_bpe": "byte_bpe",
        "loop166_mlm_model": "mlm_model",
        "loop166_code_sections": "extractor",
        "loop166_compact_corpus": "compact_corpus",
        "loop166_b1_schedule": "b1_schedule",
    }
    expected_sources = {
        auth_name: observed_bindings[contract_name]
        for auth_name, contract_name in auth_to_contract_names.items()
    }
    if not isinstance(closure, dict) or (
        closure.get("status") != "complete"
        or closure.get("pending_bindings") != []
        or closure.get("all_sources_must_match_before_authorization") is not True
        or sources != expected_sources
    ):
        raise B1FatalError("Phase B1 authorization runtime source closure drifted")
    lease = authorization.get("one_shot_lease")
    if not isinstance(lease, dict) or (
        lease.get("lease_id") != "loop166-b1-outer0-resource-cell-v1"
        or lease.get("consumption_marker_absolute") != str(DEFAULT_MARKER.absolute())
        or lease.get("marker_must_not_exist_before_parent_start") is not True
        or lease.get("marker_creation")
        != "atomic_O_EXCL_then_file_fsync_then_parent_directory_flush_best_effort_on_windows"
        or lease.get("consume_before_any_raw_open") is not True
        or lease.get("overwrite_or_delete_marker_allowed") is not False
        or lease.get("parent_mode_consumes_lease") is not True
        or lease.get("resume_or_final_verify_may_consume_new_lease") is not False
        or lease.get("all_modes_must_bind_same_marker_content_sha256") is not True
    ):
        raise B1FatalError("Phase B1 one-shot lease contract drifted")
    handoff = authorization.get("handoff_nonce")
    if not isinstance(handoff, dict) or (
        handoff.get("required") is not True
        or handoff.get("minimum_entropy_bits") != 256
        or handoff.get("direct_cli_value_allowed") is not False
        or handoff.get("plaintext_persistence_allowed") is not False
        or handoff.get("checkpoint_contains_sha256_commitment_only") is not True
        or handoff.get("resume_and_final_verify_must_prove_nonce_possession") is not True
    ):
        raise B1FatalError("Phase B1 handoff nonce contract drifted")
    forbidden = authorization.get("forbidden")
    if not isinstance(forbidden, dict) or any(
        forbidden.get(name) is not True
        for name in (
            "outer_holdout_raw_access",
            "val_test_or_full_access",
            "heldout_tokenizer_or_model_fit",
            "quality_metrics",
            "threshold_operations",
            "f1_accuracy_auc_or_perplexity_claim",
            "hard_decisions_or_predictions",
            "five_fold_oof_claim",
            "promotion_claim",
            "public_key_dependency",
        )
    ):
        raise B1FatalError("Phase B1 authorization forbidden-scope contract drifted")
    prerequisites = authorization.get("grant_prerequisites")
    if not isinstance(prerequisites, dict) or not prerequisites or any(
        value is not True for value in prerequisites.values()
    ):
        raise B1FatalError("Phase B1 authorization grant prerequisites are incomplete")
    if authorization.get("ready_for") != {
        "parent_execution": True,
        "resume_execution": True,
        "final_verify_execution": True,
        "five_fold_oof": False,
        "val_test_or_full": False,
        "promotion": False,
    }:
        raise B1FatalError("Phase B1 authorization readiness drifted")
    if Path.cwd().resolve(strict=True) != PROJECT_ROOT.resolve(strict=True):
        raise B1FatalError("Phase B1 working directory is not canonical")
    return authorization, _sha256(raw)


def consume_run_authorization(
    authorization: dict[str, Any],
    authorization_sha256: str,
) -> RunHandoff:
    """Consume the authorization exactly once with O_EXCL before raw access."""
    handoff_nonce = secrets.token_hex(32)
    parent_pid = os.getpid()
    parent_argv_sha = _argv_sha256(canonical_parent_argv())
    payload = {
        "schema": "axon_loop166_phase_b1_execution_consumption_v1",
        "loop_id": "loop166_code_section_foundation",
        "authorization_sha256": authorization_sha256,
        "lease_id": authorization["one_shot_lease"]["lease_id"],
        "handoff_nonce_sha256": _sha256(handoff_nonce.encode("ascii")),
        "parent_pid": parent_pid,
        "canonical_parent_argv_sha256": parent_argv_sha,
        "status": "consumed_before_raw_access",
    }
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    DEFAULT_MARKER.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(DEFAULT_MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise B1FatalError("Phase B1 one-shot authorization marker already exists") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    _fsync_parent_directory(DEFAULT_MARKER.parent)
    return RunHandoff(
        authorization_sha256=authorization_sha256,
        marker_sha256=_sha256(raw),
        handoff_nonce=handoff_nonce,
        parent_pid=parent_pid,
        canonical_parent_argv_sha256=parent_argv_sha,
    )


def _fsync_parent_directory(path: Path) -> bool:
    if platform.system().casefold() == "windows":
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            create_file = kernel32.CreateFileW
            create_file.argtypes = [
                ctypes.c_wchar_p,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_void_p,
            ]
            create_file.restype = ctypes.c_void_p
            flush_file_buffers = kernel32.FlushFileBuffers
            flush_file_buffers.argtypes = [ctypes.c_void_p]
            flush_file_buffers.restype = ctypes.c_int
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = [ctypes.c_void_p]
            close_handle.restype = ctypes.c_int
            handle = create_file(
                str(path),
                0x80000000,
                0x00000001 | 0x00000002 | 0x00000004,
                None,
                3,
                0x02000000,
                None,
            )
        except (AttributeError, OSError, TypeError, ValueError):
            return False
        invalid_handle = ctypes.c_void_p(-1).value
        if handle in {None, invalid_handle}:
            return False
        try:
            return bool(flush_file_buffers(handle))
        except (OSError, TypeError, ValueError):
            return False
        finally:
            try:
                close_handle(handle)
            except (OSError, TypeError, ValueError):
                pass
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def validate_consumption_marker(
    handoff: RunHandoff,
    *,
    require_direct_parent_pid: Optional[int] = None,
) -> dict[str, Any]:
    raw = _read_bounded(DEFAULT_MARKER.resolve(strict=True), 1024 * 1024)
    if _sha256(raw) != handoff.marker_sha256:
        raise B1FatalError("Phase B1 execution-consumption marker SHA drifted")
    marker = _parse_json_object(raw, "Phase B1 execution-consumption marker")
    if (
        marker.get("schema") != "axon_loop166_phase_b1_execution_consumption_v1"
        or marker.get("authorization_sha256") != handoff.authorization_sha256
        or marker.get("lease_id") != "loop166-b1-outer0-resource-cell-v1"
        or marker.get("handoff_nonce_sha256") != handoff.handoff_nonce_sha256
        or marker.get("parent_pid") != handoff.parent_pid
        or marker.get("canonical_parent_argv_sha256")
        != handoff.canonical_parent_argv_sha256
        or marker.get("status") != "consumed_before_raw_access"
    ):
        raise B1FatalError("Phase B1 execution-consumption marker content drifted")
    if require_direct_parent_pid is not None and os.getppid() != require_direct_parent_pid:
        raise B1FatalError("Phase B1 worker was not spawned by its bound parent process")
    return marker


def load_and_select_outer_fit_scope(
    contract: dict[str, Any],
    *,
    folds_path: Path = DEFAULT_FOLDS,
    folds_summary_path: Path = DEFAULT_FOLDS_SUMMARY,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> OuterFitScope:
    records, _summary = load_local_diagnostic_folds(
        folds_path=folds_path,
        summary_path=folds_summary_path,
        data_root=data_root,
        expected_rows=EXPECTED_FOLD_ROWS,
        fold_count=EXPECTED_FOLDS,
        expected_seed=EXPECTED_FOLD_SEED,
        max_supported_file_bytes=FOLD_SOURCE_SIZE_LIMIT,
        expected_rows_per_fold=EXPECTED_FOLD_ROWS // EXPECTED_FOLDS,
        expected_rows_per_label_per_fold=EXPECTED_FOLD_ROWS // EXPECTED_FOLDS // 2,
    )
    return select_outer_fit_records(records, contract)


def select_outer_fit_records(
    records: Sequence[LocalOOFRecord],
    contract: dict[str, Any],
) -> OuterFitScope:
    """Select all folds 1-4 while retaining fold 0 as metadata-only audit rows."""
    data_scope = contract["data_scope"]
    outer_fold = int(data_scope["outer_holdout_fold"])
    fit_folds = set(int(value) for value in data_scope["outer_fit_folds"])
    ordered = sorted(records, key=lambda record: record.train_row_index)
    if len(ordered) != EXPECTED_FOLD_ROWS or len(
        {record.train_row_index for record in ordered}
    ) != EXPECTED_FOLD_ROWS:
        raise B1FatalError("Diagnostic fold row identity coverage drifted")
    holdout = [record for record in ordered if record.fold == outer_fold]
    fit = [record for record in ordered if record.fold in fit_folds]
    if len(holdout) + len(fit) != len(ordered):
        raise B1FatalError("A diagnostic row belongs to neither holdout nor outer-fit scope")
    label_counts = Counter(record.label for record in fit)
    holdout_label_counts = Counter(record.label for record in holdout)
    known_size = sum(record.source_size_bytes is not None for record in fit)
    retry = sum(record.availability == "read_failure" for record in fit)
    prior_oversize = sum(record.availability == "oversize" for record in fit)
    if (
        len(fit) != data_scope["outer_fit_metadata_rows"]
        or len(holdout) != data_scope["outer_holdout_metadata_rows"]
        or {str(key): label_counts[key] for key in (0, 1)}
        != data_scope["outer_fit_label_counts"]
        or {str(key): holdout_label_counts[key] for key in (0, 1)}
        != data_scope["outer_holdout_label_counts"]
        or known_size != data_scope["known_size_records_to_attempt"]
        or retry != data_scope["prior_source_unavailable_records_to_retry"]
        or prior_oversize != data_scope["prior_oversize_records_that_must_not_be_excluded"]
    ):
        raise B1FatalError("Outer-fit metadata scope counts drifted")
    return OuterFitScope(
        tuple(fit),
        {
            "fit_metadata_rows": len(fit),
            "fit_label_counts": {str(key): label_counts[key] for key in (0, 1)},
            "outer_holdout_metadata_rows": len(holdout),
            "outer_holdout_label_counts": {
                str(key): holdout_label_counts[key] for key in (0, 1)
            },
            "known_size_records": known_size,
            "prior_source_unavailable_records": retry,
            "prior_oversize_records": prior_oversize,
            "outer_holdout_raw_opens": 0,
            "outer_holdout_raw_bytes": 0,
        },
    )


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


def _source_candidate(path: Path, data_root: Path) -> tuple[Path, Path]:
    try:
        resolved_root = data_root.resolve(strict=True)
    except OSError as exc:
        raise B1FatalError("Materialized Train root is unavailable") from exc
    try:
        relative = _lexical_relative_to(path, data_root)
    except ValueError as exc:
        raise B1FatalError("Raw source escapes the materialized Train root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise B1FatalError("Raw source path has invalid components")
    cursor = data_root.absolute()
    if cursor.is_symlink():
        raise B1FatalError("Materialized Train root cannot be a symlink")
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise B1FatalError("Raw source path cannot contain symlinks")
    return path.absolute(), resolved_root


def _fingerprint(path: Path) -> tuple[int, int, int, int]:
    stat_result = os.stat(path, follow_symlinks=False)
    return (
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_dev),
        int(stat_result.st_ino),
    )


def read_verified_outer_fit_source(
    record: LocalOOFRecord,
    *,
    data_root: Path,
    maximum_source_bytes: int,
    audit: Optional[dict[str, int]] = None,
) -> Optional[VerifiedSource]:
    """Return verified bytes, or None only when the source is genuinely unavailable."""
    candidate, resolved_root = _source_candidate(record.source_path, data_root)
    declared_size = record.source_size_bytes
    if declared_size is not None and declared_size > maximum_source_bytes:
        raise B1FatalError("Declared source exceeds the B1 cap before source read")
    try:
        before = _fingerprint(candidate)
    except (FileNotFoundError, PermissionError, OSError):
        return None
    if before[0] > maximum_source_bytes:
        raise B1FatalError("Observed source exceeds the B1 cap before source read")
    if declared_size is not None and before[0] != declared_size:
        raise B1FatalError("Known source size drifted before source read")
    try:
        resolved_candidate = candidate.resolve(strict=True)
        resolved_candidate.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise B1FatalError("Resolved raw source escapes the materialized Train root") from exc
    if _fingerprint(resolved_candidate) != before:
        raise B1FatalError("Raw source changed while resolving root confinement")
    candidate = resolved_candidate
    digest = hashlib.sha256()
    raw = bytearray()
    if audit is not None:
        audit["raw_open_attempts"] = audit.get("raw_open_attempts", 0) + 1
    try:
        with candidate.open("rb") as handle:
            if audit is not None:
                audit["raw_open_successes"] = audit.get("raw_open_successes", 0) + 1
            while True:
                chunk = handle.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                raw.extend(chunk)
                digest.update(chunk)
                if audit is not None:
                    audit["raw_bytes_read"] = audit.get("raw_bytes_read", 0) + len(chunk)
                if len(raw) > maximum_source_bytes:
                    raise B1FatalError("Source exceeded the B1 cap during read")
        after = _fingerprint(candidate)
    except B1FatalError:
        raise
    except (FileNotFoundError, PermissionError, OSError):
        return None
    if before != after:
        raise B1FatalError("Source fingerprint changed during verified read")
    observed_size = len(raw)
    observed_sha = digest.hexdigest()
    if observed_size != before[0]:
        raise B1FatalError("Source byte count differs from its stable fingerprint")
    if declared_size is not None and observed_size != declared_size:
        raise B1FatalError("Known source size drifted during verified read")
    if observed_sha != record.source_sha256:
        raise B1FatalError("Source SHA-256 drifted during verified read")
    return VerifiedSource(bytes(raw), observed_size, observed_sha)


def _free_disk_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


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
        if not get_process_memory_info(process, ctypes.byref(counters), counters.cb):
            raise B1FatalError("Unable to read process RSS")
        return int(counters.PeakWorkingSetSize)
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if platform.system().casefold() == "darwin" else usage * 1024)


def guard_non_cuda_phase(
    contract: dict[str, Any],
    *,
    disk_path: Path,
    cumulative_wall_seconds: float,
    state: Optional[dict[str, Any]] = None,
) -> dict[str, int]:
    resources = contract["resource_gates"]
    rss = _peak_process_rss_bytes()
    free_disk = _free_disk_bytes(disk_path)
    if cumulative_wall_seconds > resources["maximum_cumulative_wall_seconds"]:
        raise B1FatalError("B1 cumulative wall cap expired outside CUDA training")
    if rss >= resources["maximum_process_rss_bytes_exclusive"]:
        raise B1FatalError("B1 process RSS cap was exceeded outside CUDA training")
    if free_disk < resources["minimum_free_disk_bytes_during_run"]:
        raise B1FatalError("B1 disk floor was crossed outside CUDA training")
    if state is not None:
        state["peak_process_rss_bytes"] = max(state["peak_process_rss_bytes"], rss)
        state["minimum_free_disk_bytes"] = min(
            state["minimum_free_disk_bytes"], free_disk
        )
    return {"process_rss_bytes": rss, "free_disk_bytes": free_disk}


def scan_outer_fit_corpus(
    scope: OuterFitScope,
    contract: dict[str, Any],
    *,
    data_root: Path,
    disk_probe_path: Path = PROJECT_ROOT,
    cumulative_wall_seconds_before: float = 0.0,
) -> OuterFitScan:
    """Attempt every fit record and retain only bounded selected code windows in memory."""
    data_scope = contract["data_scope"]
    extraction_contract = contract["extraction"]
    resources = contract["resource_gates"]
    if _free_disk_bytes(disk_probe_path) < resources["minimum_free_disk_bytes_before_raw_open"]:
        raise B1FatalError("Insufficient free disk before raw access")
    windows: list[bytes] = []
    missing = Counter()
    source_verified = 0
    source_unavailable = 0
    extraction_success = 0
    raw_bytes = 0
    code_bytes = 0
    known_attempted = 0
    retry_attempted = 0
    prior_oversize_attempted = 0
    per_file_window_sum = 0
    commitment = hashlib.sha256(b"axon_loop166_phase_b1_outer_fit_corpus_v1\x00")
    started = time.perf_counter()
    read_audit = {"raw_open_attempts": 0, "raw_open_successes": 0, "raw_bytes_read": 0}
    peak_rss = _peak_process_rss_bytes()
    minimum_free_disk = _free_disk_bytes(disk_probe_path)
    for record in scope.records:
        try:
            if (
                cumulative_wall_seconds_before + time.perf_counter() - started
                > resources["maximum_cumulative_wall_seconds"]
            ):
                raise B1FatalError("B1 cumulative wall-time cap expired during raw scan")
            if record.source_size_bytes is None:
                retry_attempted += 1
            else:
                known_attempted += 1
            if record.availability == "oversize":
                prior_oversize_attempted += 1
            verified = read_verified_outer_fit_source(
                record,
                data_root=data_root,
                maximum_source_bytes=int(data_scope["maximum_source_file_bytes"]),
                audit=read_audit,
            )
            if verified is None:
                source_unavailable += 1
                missing["source_unavailable"] += 1
                commitment.update(pack("<Q", record.train_row_index))
                commitment.update(b"source_unavailable\x00")
                continue
            source_verified += 1
            raw_bytes += verified.size_bytes
            extracted = extract_executable_code(verified.raw_bytes)
            if extracted.missing_reason is not None:
                if extracted.missing_reason not in ALLOWED_SCAN_MISSING_REASONS:
                    raise B1FatalError("Extractor emitted a non-contract missing reason")
                missing[extracted.missing_reason] += 1
                commitment.update(pack("<Q", record.train_row_index))
                commitment.update(extracted.missing_reason.encode("ascii") + b"\x00")
                continue
            extraction_success += 1
            code_bytes += len(extracted.code_bytes)
            selected = [
                extracted.code_bytes[
                    start : start + extraction_contract["window_original_bytes"]
                ]
                for start in range(
                    0,
                    len(extracted.code_bytes),
                    extraction_contract["window_original_bytes"],
                )
            ]
            maximum_windows = int(extraction_contract["maximum_windows_per_file"])
            if len(selected) > maximum_windows:
                denominator = maximum_windows - 1
                span = len(selected) - 1
                indices = [
                    (selection_index * span + denominator // 2) // denominator
                    for selection_index in range(maximum_windows)
                ]
                selected = [selected[index] for index in indices]
            if not selected:
                raise B1FatalError("Successful extraction produced no selected windows")
            commitment.update(pack("<Q", record.train_row_index))
            commitment.update(b"available\x00")
            commitment.update(pack("<Q", len(selected)))
            for window in selected:
                commitment.update(pack("<Q", len(window)))
                commitment.update(hashlib.sha256(window).digest())
            windows.extend(selected)
            per_file_window_sum += len(selected)
        finally:
            peak_rss = max(peak_rss, _peak_process_rss_bytes())
            if peak_rss >= resources["maximum_process_rss_bytes_exclusive"]:
                raise B1FatalError("Process RSS exceeded the frozen B1 cap during scan")
            current_free_disk = _free_disk_bytes(disk_probe_path)
            minimum_free_disk = min(minimum_free_disk, current_free_disk)
            if current_free_disk < resources["minimum_free_disk_bytes_during_run"]:
                raise B1FatalError("Free disk fell below the frozen B1 runtime floor")

    fit_rows = len(scope.records)
    extraction_missing = sum(missing.values())
    coverage = extraction_success / fit_rows
    if (
        source_verified + source_unavailable != fit_rows
        or extraction_success + extraction_missing != fit_rows
        or sum(missing.values()) != extraction_missing
        or per_file_window_sum != len(windows)
        or coverage
        < extraction_contract["accounting_invariants"]["minimum_extraction_success_coverage"]
        or known_attempted != data_scope["known_size_records_to_attempt"]
        or retry_attempted != data_scope["prior_source_unavailable_records_to_retry"]
        or prior_oversize_attempted
        != data_scope["prior_oversize_records_that_must_not_be_excluded"]
    ):
        raise B1FatalError("B1 scan accounting invariant failed")
    return OuterFitScan(
        tuple(windows),
        {
            "fit_metadata_rows": fit_rows,
            "known_size_records_attempted": known_attempted,
            "prior_source_unavailable_records_retried": retry_attempted,
            "prior_oversize_records_attempted": prior_oversize_attempted,
            "source_verified": source_verified,
            "source_unavailable": source_unavailable,
            "extraction_success": extraction_success,
            "extraction_missing": extraction_missing,
            "missing_by_reason": {
                reason: missing[reason] for reason in sorted(ALLOWED_SCAN_MISSING_REASONS)
            },
            "selected_windows": len(windows),
            "selected_window_original_bytes": sum(map(len, windows)),
            "per_file_window_count_sum": per_file_window_sum,
            "raw_bytes_verified": raw_bytes,
            "fit_raw_open_attempts": read_audit["raw_open_attempts"],
            "fit_raw_open_successes": read_audit["raw_open_successes"],
            "fit_raw_bytes_actually_read": read_audit["raw_bytes_read"],
            "code_bytes_observed_not_persisted": code_bytes,
            "extraction_success_coverage": coverage,
            "outer_holdout_raw_opens": 0,
            "outer_holdout_raw_bytes": 0,
            "raw_code_artifact_bytes": 0,
            "durable_token_artifact_bytes": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_process_rss_bytes": peak_rss,
            "minimum_free_disk_bytes": minimum_free_disk,
        },
        commitment.hexdigest(),
    )


def build_compact_corpus(
    tokenizer: Any,
    windows: Sequence[bytes],
    contract: dict[str, Any],
) -> CompactCorpusBuild:
    """Losslessly encode every window into the required compact in-memory corpus."""
    tokenizer_contract = contract["tokenizer"]
    sequence_contract = contract["lossless_sequence_preparation"]
    vocab_size = tokenizer_vocab_size(tokenizer)
    if vocab_size != tokenizer_contract["expected_total_vocabulary"]:
        raise B1FatalError("Fresh tokenizer vocabulary size drifted")
    maximum_tokens = int(sequence_contract["max_content_tokens"])
    corpus = CompactSequenceCorpus(vocab_size, maximum_tokens)
    original_window_bytes = 0
    prepared_original_bytes = 0
    split_window_count = 0
    sequence_expansion_count = 0
    for window in windows:
        chunks = chunk_token_ids_losslessly(
            tokenizer,
            window,
            max_content_tokens=maximum_tokens,
        )
        if len(chunks) > sequence_contract["per_window_maximum_sequences_after_split"]:
            raise B1FatalError("A window exceeded the frozen maximum sequence expansion")
        if len(chunks) > 1:
            split_window_count += 1
            sequence_expansion_count += len(chunks) - 1
        original_window_bytes += len(window)
        for chunk in chunks:
            corpus.append(chunk)
            prepared_original_bytes += chunk.original_byte_length
    if original_window_bytes != prepared_original_bytes or corpus.total_original_bytes != original_window_bytes:
        raise B1FatalError("Compact corpus did not conserve every original window byte")
    return CompactCorpusBuild(
        corpus,
        {
            "original_window_count": len(windows),
            "original_window_bytes": original_window_bytes,
            "prepared_sequence_count": len(corpus),
            "prepared_original_bytes": prepared_original_bytes,
            "split_window_count": split_window_count,
            "sequence_expansion_count": sequence_expansion_count,
            "content_token_count": corpus.total_tokens,
            "estimated_compact_storage_bytes": corpus.estimated_storage_bytes,
            "compact_corpus_commitment_sha256": corpus.commitment_sha256(),
            "dropped_content_tokens": 0,
            "dropped_original_bytes": 0,
            "overlength_windows_excluded": 0,
            "durable_token_artifact_bytes": 0,
        },
    )


def _resolve_output_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    absolute = candidate.absolute()
    try:
        absolute.relative_to(PROJECT_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise B1FatalError("B1 outputs must remain inside the project root") from exc
    return absolute


def _atomic_tokenizer_save(tokenizer: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        tokenizer.save(str(temporary))
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(torch_module: Any, payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as handle:
            torch_module.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
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


def _artifact_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fit_fresh_tokenizer_and_compact_corpus(
    scan: OuterFitScan,
    contract: dict[str, Any],
    *,
    tokenizer_path: Path,
) -> tuple[Any, CompactCorpusBuild, dict[str, Any]]:
    """Fit one fresh tokenizer, atomically persist it, then compact all windows."""
    tokenizer_contract = contract["tokenizer"]
    tokenizer = train_byte_bpe_tokenizer(
        scan.windows,
        vocab_size=int(tokenizer_contract["expected_total_vocabulary"]),
        special_tokens=tuple(tokenizer_contract["special_tokens"]),
    )
    if tokenizer_vocab_size(tokenizer) != tokenizer_contract["expected_total_vocabulary"]:
        raise B1FatalError("Fresh B1 tokenizer vocabulary size drifted")
    tokenizer_path = _resolve_output_path(tokenizer_path)
    _atomic_tokenizer_save(tokenizer, tokenizer_path)
    from tokenizers import Tokenizer

    restored = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer_vocab_size(restored) != tokenizer_vocab_size(tokenizer):
        raise B1FatalError("Atomic tokenizer recovery changed its vocabulary")
    probes = scan.windows[:8] + scan.windows[-8:]
    if any(encode_bytes(tokenizer, value) != encode_bytes(restored, value) for value in probes):
        raise B1FatalError("Atomic tokenizer recovery changed byte encodings")
    compact = build_compact_corpus(restored, scan.windows, contract)
    return restored, compact, {
        "path": str(tokenizer_path),
        "sha256": _artifact_sha(tokenizer_path),
        "atomic": True,
        "fresh_fit": True,
        "roundtrip_exact": True,
        "corpus_commitment_sha256": scan.outer_fit_corpus_commitment_sha256,
    }


def rebuild_compact_corpus_from_tokenizer(
    scan: OuterFitScan,
    contract: dict[str, Any],
    *,
    tokenizer_path: Path,
    expected_tokenizer_sha256: str,
) -> tuple[Any, CompactCorpusBuild]:
    from tokenizers import Tokenizer

    tokenizer_path = tokenizer_path.resolve(strict=True)
    if _artifact_sha(tokenizer_path) != expected_tokenizer_sha256:
        raise B1FatalError("Resume tokenizer SHA-256 drifted")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer, build_compact_corpus(tokenizer, scan.windows, contract)


def _build_model_config(contract: dict[str, Any], tokenizer: Any) -> TinyMLMConfig:
    model = contract["model"]
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise B1FatalError("B1 tokenizer is missing PAD")
    return TinyMLMConfig(
        vocab_size=tokenizer_vocab_size(tokenizer),
        sequence_tokens=int(model["sequence_tokens"]),
        layers=int(model["layers"]),
        hidden_dim=int(model["hidden_dim"]),
        heads=int(model["heads"]),
        ffn_dim=int(model["ffn_dim"]),
        local_attention_window=int(model["local_attention_window"]),
        global_token_index=int(model["global_token_index"]),
        dropout=float(model["dropout"]),
        activation=str(model["activation"]),
        gradient_checkpointing=bool(model["gradient_checkpointing"]),
        tied_input_output_embeddings=bool(model["tied_input_output_embeddings"]),
        pad_token_id=int(pad_id),
    )


def _tokenizer_ids(tokenizer: Any) -> dict[str, int]:
    ids = {
        name: tokenizer.token_to_id(token)
        for name, token in {
            "pad": "[PAD]",
            "cls": "[CLS]",
            "sep": "[SEP]",
            "mask": "[MASK]",
        }.items()
    }
    if any(value is None for value in ids.values()) or len(set(ids.values())) != len(ids):
        raise B1FatalError("Tokenizer framing ids are missing or collide")
    return {name: int(value) for name, value in ids.items() if value is not None}


def _synthetic_eval_batch(torch_module: Any, tokenizer: Any, config: TinyMLMConfig) -> tuple[Any, Any]:
    ids = _tokenizer_ids(tokenizer)
    content = encode_bytes(tokenizer, b"Loop166-B1-resume-fidelity")
    if not content or len(content) > config.sequence_tokens - 2:
        raise B1FatalError("Unable to construct bounded synthetic resume input")
    input_ids = torch_module.full(
        (1, config.sequence_tokens), ids["pad"], dtype=torch_module.long
    )
    valid = len(content) + 2
    input_ids[0, 0] = ids["cls"]
    input_ids[0, 1 : valid - 1] = torch_module.tensor(content, dtype=torch_module.long)
    input_ids[0, valid - 1] = ids["sep"]
    attention = torch_module.zeros_like(input_ids, dtype=torch_module.bool)
    attention[0, :valid] = True
    return input_ids, attention


def _new_training_state() -> dict[str, Any]:
    return {
        "completed_optimizer_steps": 0,
        "completed_sequence_count": 0,
        "next_permutation_cursor": 0,
        "training_original_bytes": 0,
        "training_seconds": 0.0,
        "post_warmup_original_bytes": 0,
        "post_warmup_seconds": 0.0,
        "throughput_window_original_bytes": 0,
        "throughput_window_seconds": 0.0,
        "throughput_windows": 0,
        "low_throughput_windows": 0,
        "consecutive_low_throughput_windows": 0,
        "maximum_consecutive_low_throughput_windows": 0,
        "nonfinite_events": 0,
        "oom_events": 0,
        "peak_process_rss_bytes": 0,
        "peak_cuda_allocated_bytes": 0,
        "peak_cuda_reserved_bytes": 0,
        "minimum_free_disk_bytes": 1 << 63,
        "checkpoint_writes": 0,
    }


def prepare_validated_schedule(
    permutation: Sequence[int],
    contract: dict[str, Any],
) -> ValidatedSchedule:
    started = time.perf_counter()
    groups = tuple(
        iter_optimizer_groups(
            permutation,
            microbatch_size=int(contract["training"]["microbatch"]),
            gradient_accumulation_steps=int(
                contract["training"]["gradient_accumulation_steps"]
            ),
        )
    )
    validate_exact_once_schedule(permutation, groups, len(permutation))
    return ValidatedSchedule(
        groups=groups,
        build_seconds=time.perf_counter() - started,
        permutation_commitment_sha256=permutation_commitment_sha256(permutation),
    )


def _sample_runtime_resources(
    torch_module: Any,
    device: Any,
    state: dict[str, Any],
    contract: dict[str, Any],
    *,
    disk_path: Path,
    cumulative_wall_seconds: float,
) -> None:
    gates = contract["resource_gates"]
    state["peak_process_rss_bytes"] = max(
        int(state["peak_process_rss_bytes"]), _peak_process_rss_bytes()
    )
    state["peak_cuda_allocated_bytes"] = max(
        int(state["peak_cuda_allocated_bytes"]),
        int(torch_module.cuda.max_memory_allocated(device)),
    )
    state["peak_cuda_reserved_bytes"] = max(
        int(state["peak_cuda_reserved_bytes"]),
        int(torch_module.cuda.max_memory_reserved(device)),
    )
    free_disk = _free_disk_bytes(disk_path)
    state["minimum_free_disk_bytes"] = min(int(state["minimum_free_disk_bytes"]), free_disk)
    if cumulative_wall_seconds > gates["maximum_cumulative_wall_seconds"]:
        raise B1FatalError("B1 cumulative wall-time cap expired")
    if state["peak_process_rss_bytes"] >= gates["maximum_process_rss_bytes_exclusive"]:
        raise B1FatalError("B1 process RSS cap was exceeded")
    if state["peak_cuda_allocated_bytes"] >= gates["maximum_cuda_allocated_bytes_exclusive"]:
        raise B1FatalError("B1 CUDA allocated cap was exceeded")
    if state["peak_cuda_reserved_bytes"] >= gates["maximum_cuda_reserved_bytes_exclusive"]:
        raise B1FatalError("B1 CUDA reserved cap was exceeded")
    if free_disk < gates["minimum_free_disk_bytes_during_run"]:
        raise B1FatalError("B1 runtime free-disk floor was crossed")


def _update_throughput_state(
    state: dict[str, Any],
    contract: dict[str, Any],
    *,
    group_original_bytes: int,
    group_seconds: float,
    step_before: int,
) -> None:
    gates = contract["resource_gates"]
    state["training_original_bytes"] += int(group_original_bytes)
    state["training_seconds"] += float(group_seconds)
    if step_before < gates["throughput_warmup_optimizer_steps"]:
        return
    state["post_warmup_original_bytes"] += int(group_original_bytes)
    state["post_warmup_seconds"] += float(group_seconds)
    state["throughput_window_original_bytes"] += int(group_original_bytes)
    state["throughput_window_seconds"] += float(group_seconds)
    if state["throughput_window_seconds"] < gates["throughput_window_seconds"]:
        return
    _close_throughput_window(state, contract)


def _close_throughput_window(state: dict[str, Any], contract: dict[str, Any]) -> None:
    seconds = float(state["throughput_window_seconds"])
    if seconds <= 0:
        return
    rate = state["throughput_window_original_bytes"] / seconds
    state["throughput_windows"] += 1
    if rate < contract["resource_gates"]["minimum_original_bytes_per_training_second"]:
        state["low_throughput_windows"] += 1
        state["consecutive_low_throughput_windows"] += 1
        state["maximum_consecutive_low_throughput_windows"] = max(
            state["maximum_consecutive_low_throughput_windows"],
            state["consecutive_low_throughput_windows"],
        )
    else:
        state["consecutive_low_throughput_windows"] = 0
    state["throughput_window_original_bytes"] = 0
    state["throughput_window_seconds"] = 0.0
    if (
        state["maximum_consecutive_low_throughput_windows"]
        > contract["resource_gates"]["consecutive_low_throughput_windows_allowed"]
    ):
        raise B1FatalError("B1 throughput failed too many consecutive windows")


def build_checkpoint_payload(
    *,
    torch_module: Any,
    model: Any,
    optimizer: Any,
    scaler: Any,
    model_config: TinyMLMConfig,
    tokenizer: Any,
    tokenizer_sha256: str,
    mask_generator: Any,
    permutation_commitment: str,
    corpus_commitment: str,
    compact_commitment: str,
    state: dict[str, Any],
    cumulative_wall_seconds: float,
    parent_pid: int,
    run_context: Optional[dict[str, Any]] = None,
    handoff: Optional[RunHandoff] = None,
    canonical_child_argv_sha256: Optional[str] = None,
    resume_pid: int = 0,
) -> dict[str, Any]:
    synthetic_ids, synthetic_attention = _synthetic_eval_batch(
        torch_module, tokenizer, model_config
    )
    device = next(model.parameters()).device
    model.eval()
    autocast_context = (
        torch_module.amp.autocast("cuda", dtype=torch_module.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch_module.inference_mode(), autocast_context:
        synthetic_logits = model(
            synthetic_ids.to(device), attention_mask=synthetic_attention.to(device)
        )["logits"].detach().cpu()
    model.train()
    return {
        "schema": "axon_loop166_phase_b1_tiny_mlm_checkpoint_v1",
        "model_config": asdict(model_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "completed_optimizer_steps": int(state["completed_optimizer_steps"]),
        "completed_sequence_count": int(state["completed_sequence_count"]),
        "shuffle_seed": 166,
        "shuffle_commitment_sha256": permutation_commitment,
        "next_permutation_cursor": int(state["next_permutation_cursor"]),
        "torch_cpu_rng_state": torch_module.get_rng_state(),
        "torch_cuda_rng_state": torch_module.cuda.get_rng_state(device).cpu(),
        "mask_generator_state": mask_generator.get_state(),
        "tokenizer_sha256": tokenizer_sha256,
        "outer_fit_corpus_commitment_sha256": corpus_commitment,
        "compact_corpus_commitment_sha256": compact_commitment,
        "cumulative_wall_seconds": float(cumulative_wall_seconds),
        "training_state": dict(state),
        "synthetic_input_ids": synthetic_ids,
        "synthetic_attention_mask": synthetic_attention,
        "synthetic_logits": synthetic_logits,
        "parent_pid": int(parent_pid),
        "resume_pid": int(resume_pid),
        "authorization_sha256": (
            handoff.authorization_sha256 if handoff is not None else "0" * 64
        ),
        "marker_sha256": handoff.marker_sha256 if handoff is not None else "0" * 64,
        "handoff_nonce_sha256": (
            handoff.handoff_nonce_sha256 if handoff is not None else "0" * 64
        ),
        "canonical_parent_argv_sha256": (
            handoff.canonical_parent_argv_sha256 if handoff is not None else "0" * 64
        ),
        "canonical_child_argv_sha256": canonical_child_argv_sha256 or "0" * 64,
        "permutation_prefix_original_bytes": int(
            state.get("training_original_bytes", 0)
        ),
        "run_context": dict(run_context or {}),
    }


def verify_checkpoint_payload(
    payload: object,
    contract: dict[str, Any],
    *,
    expected_handoff: Optional[RunHandoff] = None,
    expected_child_argv_sha256: Optional[str] = None,
    corpus: Optional[CompactSequenceCorpus] = None,
    permutation: Optional[Sequence[int]] = None,
    require_final: bool = False,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != (
        "axon_loop166_phase_b1_tiny_mlm_checkpoint_v1"
    ):
        raise B1FatalError("B1 checkpoint schema is invalid")
    required = set(contract["checkpoint_and_resume"]["payload_requires"])
    missing = sorted(required - set(payload))
    if missing:
        raise B1FatalError(f"B1 checkpoint is missing required fields: {missing}")
    model_config_payload = payload.get("model_config")
    if not isinstance(model_config_payload, dict):
        raise B1FatalError("B1 checkpoint model config is missing or invalid")
    try:
        model_config = TinyMLMConfig(**model_config_payload)
    except (TypeError, ValueError) as exc:
        raise B1FatalError("B1 checkpoint model config is invalid") from exc
    frozen_model = contract["model"]
    if model_config != TinyMLMConfig(
        vocab_size=int(contract["tokenizer"]["expected_total_vocabulary"]),
        sequence_tokens=int(frozen_model["sequence_tokens"]),
        layers=int(frozen_model["layers"]),
        hidden_dim=int(frozen_model["hidden_dim"]),
        heads=int(frozen_model["heads"]),
        ffn_dim=int(frozen_model["ffn_dim"]),
        local_attention_window=int(frozen_model["local_attention_window"]),
        global_token_index=int(frozen_model["global_token_index"]),
        dropout=float(frozen_model["dropout"]),
        activation=str(frozen_model["activation"]),
        gradient_checkpointing=bool(frozen_model["gradient_checkpointing"]),
        tied_input_output_embeddings=bool(
            frozen_model["tied_input_output_embeddings"]
        ),
        pad_token_id=0,
    ):
        raise B1FatalError("B1 checkpoint model config drifted from the frozen contract")
    if payload["completed_sequence_count"] != payload["next_permutation_cursor"]:
        raise B1FatalError("B1 checkpoint cursor and completed sequence count differ")
    cursor = payload.get("next_permutation_cursor")
    steps = payload.get("completed_optimizer_steps")
    prefix_bytes = payload.get("permutation_prefix_original_bytes")
    if (
        not isinstance(cursor, int)
        or isinstance(cursor, bool)
        or cursor < 0
        or not isinstance(steps, int)
        or isinstance(steps, bool)
        or steps < 0
        or steps != (cursor + 3) // 4
        or not isinstance(prefix_bytes, int)
        or isinstance(prefix_bytes, bool)
        or prefix_bytes < 0
    ):
        raise B1FatalError("B1 checkpoint step/cursor/prefix accounting is invalid")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise B1FatalError("B1 checkpoint deep training state is missing or invalid")
    if (
        training_state.get("completed_optimizer_steps") != steps
        or training_state.get("completed_sequence_count") != cursor
        or training_state.get("next_permutation_cursor") != cursor
        or training_state.get("training_original_bytes", 0) != prefix_bytes
    ):
        raise B1FatalError("B1 checkpoint deep training state diverged")
    _assert_finite_numeric_tree(training_state, "training_state")
    _assert_finite_tensor_tree(training_state, "training_state")
    for rng_name in (
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
        "mask_generator_state",
    ):
        rng = payload.get(rng_name)
        if (
            not hasattr(rng, "dtype")
            or str(rng.dtype) != "torch.uint8"
            or getattr(rng, "ndim", 0) != 1
            or getattr(rng, "numel", lambda: 0)() < 1
        ):
            raise B1FatalError(f"B1 checkpoint RNG state is invalid: {rng_name}")
    for name in (
        "shuffle_commitment_sha256",
        "tokenizer_sha256",
        "outer_fit_corpus_commitment_sha256",
        "compact_corpus_commitment_sha256",
        "authorization_sha256",
        "marker_sha256",
        "handoff_nonce_sha256",
        "canonical_parent_argv_sha256",
        "canonical_child_argv_sha256",
    ):
        if not _is_lower_sha256(payload.get(name)):
            raise B1FatalError(f"B1 checkpoint commitment is invalid: {name}")
    if not math.isfinite(float(payload.get("cumulative_wall_seconds", math.nan))):
        raise B1FatalError("B1 checkpoint cumulative wall time is non-finite")
    for state_name in ("model_state_dict", "optimizer_state_dict", "scaler_state_dict"):
        state_value = payload.get(state_name)
        if not isinstance(state_value, dict):
            raise B1FatalError(f"B1 checkpoint state is missing or invalid: {state_name}")
        _assert_finite_numeric_tree(state_value, state_name)
        _assert_finite_tensor_tree(state_value, state_name)
    synthetic_input_ids = payload.get("synthetic_input_ids")
    synthetic_attention = payload.get("synthetic_attention_mask")
    synthetic_logits = payload.get("synthetic_logits")
    expected_input_shape = (1, model_config.sequence_tokens)
    expected_logit_shape = (*expected_input_shape, model_config.vocab_size)
    if (
        getattr(synthetic_input_ids, "shape", None) != expected_input_shape
        or str(getattr(synthetic_input_ids, "dtype", "")) != "torch.int64"
        or getattr(synthetic_attention, "shape", None) != expected_input_shape
        or str(getattr(synthetic_attention, "dtype", "")) != "torch.bool"
        or getattr(synthetic_logits, "shape", None) != expected_logit_shape
        or not hasattr(synthetic_logits, "isfinite")
    ):
        raise B1FatalError("B1 checkpoint synthetic fidelity tensors are missing or invalid")
    _assert_finite_tensor_tree(synthetic_logits, "synthetic_logits")
    if expected_handoff is not None and (
        payload["authorization_sha256"] != expected_handoff.authorization_sha256
        or payload["marker_sha256"] != expected_handoff.marker_sha256
        or payload["handoff_nonce_sha256"] != expected_handoff.handoff_nonce_sha256
        or payload["canonical_parent_argv_sha256"]
        != expected_handoff.canonical_parent_argv_sha256
        or payload.get("parent_pid") != expected_handoff.parent_pid
    ):
        raise B1FatalError("B1 checkpoint authorization handoff binding drifted")
    if (
        expected_child_argv_sha256 is not None
        and payload["canonical_child_argv_sha256"] != expected_child_argv_sha256
    ):
        raise B1FatalError("B1 checkpoint canonical child argv binding drifted")
    if corpus is not None and permutation is not None:
        if cursor > len(permutation):
            raise B1FatalError("B1 checkpoint cursor exceeds rebuilt permutation")
        recomputed_prefix = sum(
            corpus[index].original_byte_length for index in permutation[:cursor]
        )
        if recomputed_prefix != prefix_bytes:
            raise B1FatalError("B1 checkpoint permutation-prefix byte accounting drifted")
    if require_final:
        run_context = payload.get("run_context")
        expected_count = (
            len(permutation)
            if permutation is not None
            else run_context.get("prepared_sequence_count")
            if isinstance(run_context, dict)
            else None
        )
        if not isinstance(expected_count, int) or cursor != expected_count:
            raise B1FatalError("Final B1 checkpoint does not close the exact-once epoch")
        if isinstance(run_context, dict) and steps != run_context.get("total_optimizer_steps"):
            raise B1FatalError("Final B1 checkpoint optimizer-step count drifted")
    return payload


def _assert_finite_numeric_tree(value: Any, context: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, bytes)):
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise B1FatalError(f"B1 checkpoint contains non-finite numeric state: {context}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_numeric_tree(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_numeric_tree(child, f"{context}[{index}]")


def _assert_finite_tensor_tree(value: Any, context: str) -> None:
    if hasattr(value, "is_floating_point") and callable(value.is_floating_point):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            value.isfinite().all().item()
        ):
            raise B1FatalError(f"B1 checkpoint tensor is non-finite: {context}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tensor_tree(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_tensor_tree(child, f"{context}[{index}]")


def _write_training_checkpoint(
    *,
    torch_module: Any,
    checkpoint_path: Path,
    payload: dict[str, Any],
    state: dict[str, Any],
) -> None:
    _atomic_torch_save(torch_module, payload, checkpoint_path)


def train_segment(
    *,
    torch_module: Any,
    model: Any,
    optimizer: Any,
    scaler: Any,
    model_config: TinyMLMConfig,
    tokenizer: Any,
    tokenizer_sha256: str,
    corpus: CompactSequenceCorpus,
    corpus_commitment: str,
    permutation: Sequence[int],
    schedule: ValidatedSchedule,
    mask_generator: Any,
    state: dict[str, Any],
    contract: dict[str, Any],
    checkpoint_path: Path,
    cumulative_wall_seconds_before: float,
    phase_started: float,
    parent_pid: int,
    handoff: Optional[RunHandoff] = None,
    canonical_child_argv_sha256: Optional[str] = None,
    resume_pid: int = 0,
    run_context: Optional[dict[str, Any]] = None,
    stop_after_optimizer_step: Optional[int] = None,
) -> dict[str, Any]:
    training = contract["training"]
    checkpoint_contract = contract["checkpoint_and_resume"]
    ids = _tokenizer_ids(tokenizer)
    device = next(model.parameters()).device
    permutation_commitment = schedule.permutation_commitment_sha256
    compact_commitment = corpus.commitment_sha256()
    next_sample = time.perf_counter()
    pending_schedule_seconds = schedule.build_seconds
    for group in schedule.groups[int(state["completed_optimizer_steps"]) :]:
        if group.cursor_start != state["next_permutation_cursor"]:
            raise B1FatalError("Validated schedule diverged from the checkpoint cursor")
        step_started = time.perf_counter()
        step_original_bytes = 0
        optimizer.zero_grad(set_to_none=True)
        try:
            for microbatch in group.microbatches:
                framed = materialize_framed_batch(
                    corpus,
                    microbatch.indices,
                    pad_id=ids["pad"],
                    cls_id=ids["cls"],
                    sep_id=ids["sep"],
                    sequence_tokens=model_config.sequence_tokens,
                    torch_module=torch_module,
                )
                masked = mask_content_batch(
                    torch_module,
                    framed.input_ids,
                    framed.attention_mask,
                    cls_id=ids["cls"],
                    sep_id=ids["sep"],
                    mask_token_id=ids["mask"],
                    mask_ratio=float(training["mask_ratio"]),
                    generator=mask_generator,
                )
                input_ids = masked.masked_input_ids.to(device)
                attention_mask = framed.attention_mask.to(device)
                labels = masked.labels.to(device)
                with torch_module.amp.autocast("cuda", dtype=torch_module.float16):
                    logits = model(input_ids, attention_mask=attention_mask)["logits"]
                    token_objectives = torch_module.nn.functional.cross_entropy(
                        logits.reshape(-1, model_config.vocab_size),
                        labels.reshape(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).reshape(labels.shape)
                    selected = labels.ne(-100)
                    sequence_objectives = token_objectives.sum(dim=1) / selected.sum(dim=1)
                    objective = sequence_objectives.mean() * microbatch.loss_weight
                if not torch_module.isfinite(logits).all() or not torch_module.isfinite(objective):
                    state["nonfinite_events"] += 1
                    raise B1FatalError("B1 training produced a non-finite tensor")
                scaler.scale(objective).backward()
                step_original_bytes += int(framed.original_byte_lengths.sum().item())
            scaler.unscale_(optimizer)
            gradient_norm = torch_module.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            if not torch_module.isfinite(gradient_norm):
                state["nonfinite_events"] += 1
                raise B1FatalError("B1 training produced a non-finite gradient norm")
            scaler.step(optimizer)
            scaler.update()
        except torch_module.OutOfMemoryError as exc:
            state["oom_events"] += 1
            optimizer.zero_grad(set_to_none=True)
            torch_module.cuda.empty_cache()
            raise B1FatalError("B1 CUDA out of memory; recipe fallback is forbidden") from exc
        state["completed_optimizer_steps"] += 1
        state["completed_sequence_count"] += group.sequence_count
        state["next_permutation_cursor"] = group.cursor_end
        step_seconds = time.perf_counter() - step_started + pending_schedule_seconds
        pending_schedule_seconds = 0.0
        _update_throughput_state(
            state,
            contract,
            group_original_bytes=step_original_bytes,
            group_seconds=step_seconds,
            step_before=state["completed_optimizer_steps"] - 1,
        )
        cumulative_wall = (
            cumulative_wall_seconds_before + time.perf_counter() - phase_started
        )
        if time.perf_counter() >= next_sample:
            _sample_runtime_resources(
                torch_module,
                device,
                state,
                contract,
                disk_path=checkpoint_path.parent,
                cumulative_wall_seconds=cumulative_wall,
            )
            next_sample = time.perf_counter() + contract["resource_gates"][
                "resource_sample_interval_seconds"
            ]
        should_checkpoint = (
            state["completed_optimizer_steps"]
            % checkpoint_contract["checkpoint_interval_optimizer_steps"]
            == 0
            or state["next_permutation_cursor"] == len(permutation)
        )
        if should_checkpoint:
            state["checkpoint_writes"] += 1
            payload = build_checkpoint_payload(
                torch_module=torch_module,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                model_config=model_config,
                tokenizer=tokenizer,
                tokenizer_sha256=tokenizer_sha256,
                mask_generator=mask_generator,
                permutation_commitment=permutation_commitment,
                corpus_commitment=corpus_commitment,
                compact_commitment=compact_commitment,
                state=state,
                cumulative_wall_seconds=cumulative_wall,
                parent_pid=parent_pid,
                run_context=run_context,
                handoff=handoff,
                canonical_child_argv_sha256=canonical_child_argv_sha256,
                resume_pid=resume_pid,
            )
            _write_training_checkpoint(
                torch_module=torch_module,
                checkpoint_path=checkpoint_path,
                payload=payload,
                state=state,
            )
        if (
            stop_after_optimizer_step is not None
            and state["completed_optimizer_steps"] == stop_after_optimizer_step
        ):
            return state
        if (
            stop_after_optimizer_step is not None
            and state["completed_optimizer_steps"] > stop_after_optimizer_step
        ):
            raise B1FatalError("B1 training crossed the frozen fresh-process resume step")
    return state


def _initialize_training_runtime(
    contract: dict[str, Any], tokenizer: Any
) -> tuple[Any, Any, Any, TinyMLMConfig, Any, Any]:
    import torch

    if not torch.cuda.is_available():
        raise B1FatalError("B1 canonical resource cell requires CUDA")
    training = contract["training"]
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    config = _build_model_config(contract, tokenizer)
    model = TinyMaskedLanguageModel(config).to(device)
    if not (
        contract["model"]["minimum_parameters"]
        <= count_parameters(model)
        <= contract["model"]["maximum_parameters"]
    ):
        raise B1FatalError("B1 model parameter count drifted")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    grad_scaler = training["grad_scaler"]
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=float(grad_scaler["init_scale"]),
        growth_factor=float(grad_scaler["growth_factor"]),
        backoff_factor=float(grad_scaler["backoff_factor"]),
        growth_interval=int(grad_scaler["growth_interval"]),
        enabled=True,
    )
    mask_generator = torch.Generator(device="cpu")
    mask_generator.manual_seed(seed)
    return torch, model, optimizer, config, scaler, mask_generator


def _restore_training_runtime(
    contract: dict[str, Any],
    tokenizer: Any,
    payload: dict[str, Any],
) -> tuple[Any, Any, Any, TinyMLMConfig, Any, Any, bool]:
    import torch

    if not torch.cuda.is_available():
        raise B1FatalError("B1 resume requires CUDA")
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    config = TinyMLMConfig(**payload["model_config"])
    if config != _build_model_config(contract, tokenizer):
        raise B1FatalError("B1 checkpoint model config drifted from the frozen contract")
    model = TinyMaskedLanguageModel(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(contract["training"]["learning_rate"]),
        weight_decay=float(contract["training"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    grad_scaler = contract["training"]["grad_scaler"]
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=float(grad_scaler["init_scale"]),
        growth_factor=float(grad_scaler["growth_factor"]),
        backoff_factor=float(grad_scaler["backoff_factor"]),
        growth_interval=int(grad_scaler["growth_interval"]),
        enabled=True,
    )
    scaler.load_state_dict(payload["scaler_state_dict"])
    model.eval()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        restored_logits = model(
            payload["synthetic_input_ids"].to(device),
            attention_mask=payload["synthetic_attention_mask"].to(device),
        )["logits"].detach().cpu()
    exact_logits = torch.equal(restored_logits, payload["synthetic_logits"])
    if not exact_logits:
        raise B1FatalError("Fresh-process checkpoint synthetic logits are not exact")
    torch.set_rng_state(payload["torch_cpu_rng_state"])
    torch.cuda.set_rng_state(payload["torch_cuda_rng_state"], device)
    mask_generator = torch.Generator(device="cpu")
    mask_generator.set_state(payload["mask_generator_state"])
    model.train()
    return torch, model, optimizer, config, scaler, mask_generator, exact_logits


def assert_report_has_no_quality_metrics(payload: dict[str, Any]) -> None:
    forbidden_keys = {
        "loss",
        "loss_curve",
        "perplexity",
        "bits_per_byte",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "threshold",
        "predictions",
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).casefold() in forbidden_keys:
                    raise B1FatalError(f"Forbidden quality-result field in B1 report: {key}")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)


def _durable_output_bytes(paths: Sequence[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.exists() and path.is_file())


def _build_final_report(
    *,
    bindings: dict[str, Any],
    scope: OuterFitScope,
    initial_scan: dict[str, Any],
    resume_scan: dict[str, Any],
    compact: CompactCorpusBuild,
    tokenizer_artifact: dict[str, Any],
    checkpoint_path: Path,
    report_path: Path,
    state: dict[str, Any],
    contract: dict[str, Any],
    cumulative_wall_seconds: float,
    parent_pid: int,
    resume_pid: int,
    exact_logits: bool,
    final_verify_receipt: dict[str, Any],
) -> dict[str, Any]:
    if state["throughput_window_seconds"] > 0:
        _close_throughput_window(state, contract)
    training_rate = (
        state["training_original_bytes"] / state["training_seconds"]
        if state["training_seconds"] > 0
        else 0.0
    )
    sequence_count = len(compact.corpus)
    required = contract["required_report_invariants"]
    durable_bytes = _durable_output_bytes(
        [Path(tokenizer_artifact["path"]), checkpoint_path, report_path]
    )
    gates = {
        "outer_holdout_raw_zero": scope.audit["outer_holdout_raw_opens"] == 0
        and scope.audit["outer_holdout_raw_bytes"] == 0,
        "fit_scope_exact": scope.audit["fit_metadata_rows"] == required["fit_metadata_rows"],
        "initial_scan_denominator_exact": initial_scan["fit_metadata_rows"]
        == required["fit_metadata_rows"],
        "known_size_attempts_exact": initial_scan["known_size_records_attempted"]
        == required["known_size_records_attempted"],
        "source_unavailable_retries_exact": initial_scan[
            "prior_source_unavailable_records_retried"
        ]
        == required["prior_source_unavailable_records_retried"],
        "resume_scan_commitment_exact": resume_scan["commitment_match"] is True,
        "sequence_byte_coverage_exact": compact.accounting["original_window_bytes"]
        == compact.accounting["prepared_original_bytes"],
        "sequence_drop_zero": compact.accounting["dropped_content_tokens"]
        == required["dropped_content_tokens"]
        and compact.accounting["dropped_original_bytes"]
        == required["dropped_original_bytes"]
        and compact.accounting["overlength_windows_excluded"]
        == required["overlength_windows_excluded"],
        "exact_once_epoch": state["completed_sequence_count"] == sequence_count,
        "exact_once_original_bytes": state["training_original_bytes"]
        == compact.corpus.total_original_bytes,
        "fresh_process_resume": parent_pid != resume_pid and exact_logits,
        "independent_final_checkpoint_verification": final_verify_receipt.get("decision")
        == "phase_b1_final_checkpoint_verification_pass",
        "cumulative_wall": cumulative_wall_seconds
        <= contract["resource_gates"]["maximum_cumulative_wall_seconds"],
        "cuda_allocated": state["peak_cuda_allocated_bytes"]
        < contract["resource_gates"]["maximum_cuda_allocated_bytes_exclusive"],
        "cuda_reserved": state["peak_cuda_reserved_bytes"]
        < contract["resource_gates"]["maximum_cuda_reserved_bytes_exclusive"],
        "process_rss": state["peak_process_rss_bytes"]
        < contract["resource_gates"]["maximum_process_rss_bytes_exclusive"],
        "disk_floor": state["minimum_free_disk_bytes"]
        >= contract["resource_gates"]["minimum_free_disk_bytes_during_run"],
        "durable_output_cap": durable_bytes
        <= contract["resource_gates"]["maximum_total_durable_output_bytes"],
        "epoch_average_throughput": training_rate
        >= contract["resource_gates"]["minimum_original_bytes_per_training_second"],
        "throughput_windows": state["maximum_consecutive_low_throughput_windows"]
        <= contract["resource_gates"]["consecutive_low_throughput_windows_allowed"],
        "nonfinite_zero": state["nonfinite_events"]
        <= contract["resource_gates"]["nonfinite_events_allowed"],
        "oom_zero": state["oom_events"] <= contract["resource_gates"]["oom_events_allowed"],
        "quality_results_absent": True,
    }
    passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "loop_id": "loop166_code_section_foundation",
        "claim_scope": CLAIM_SCOPE,
        "input_bindings": bindings,
        "scope": scope.audit,
        "raw_access": {
            "initial_fit_raw_open_attempts": initial_scan["fit_raw_open_attempts"],
            "initial_fit_raw_open_successes": initial_scan["fit_raw_open_successes"],
            "initial_fit_raw_bytes_actually_read": initial_scan[
                "fit_raw_bytes_actually_read"
            ],
            "resume_fit_raw_open_attempts": resume_scan["fit_raw_open_attempts"],
            "resume_fit_raw_open_successes": resume_scan["fit_raw_open_successes"],
            "resume_fit_raw_bytes_actually_read": resume_scan[
                "fit_raw_bytes_actually_read"
            ],
            "outer_holdout_raw_opens": 0,
            "outer_holdout_raw_bytes": 0,
        },
        "initial_scan": initial_scan,
        "resume_scan": resume_scan,
        "tokenizer": tokenizer_artifact,
        "sequence_preparation": compact.accounting,
        "training": {
            "epochs_completed": 1 if state["completed_sequence_count"] == sequence_count else 0,
            "prepared_sequence_count": sequence_count,
            "completed_sequence_count": state["completed_sequence_count"],
            "completed_optimizer_steps": state["completed_optimizer_steps"],
            "original_bytes_processed": state["training_original_bytes"],
            "training_seconds": state["training_seconds"],
            "original_bytes_per_training_second": training_rate,
            "throughput_windows": state["throughput_windows"],
            "low_throughput_windows": state["low_throughput_windows"],
            "nonfinite_events": state["nonfinite_events"],
            "oom_events": state["oom_events"],
            "quality_metrics_computed": False,
            "threshold_operations_performed": False,
        },
        "checkpoint_resume": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _artifact_sha(checkpoint_path),
            "checkpoint_writes": state["checkpoint_writes"],
            "fresh_process_resume_completed": parent_pid != resume_pid,
            "parent_pid": parent_pid,
            "resume_pid": resume_pid,
            "weights_only_load": True,
            "rng_and_cursor_restored": True,
            "exact_synthetic_logits": exact_logits,
            "final_verify_receipt": {
                "path": str(DEFAULT_FINAL_VERIFY_RECEIPT),
                "sha256": _artifact_sha(DEFAULT_FINAL_VERIFY_RECEIPT),
                "verifier_pid": final_verify_receipt.get("verifier_pid"),
                "decision": final_verify_receipt.get("decision"),
            },
        },
        "resources": {
            "cumulative_wall_seconds": cumulative_wall_seconds,
            "peak_process_rss_bytes": state["peak_process_rss_bytes"],
            "peak_cuda_allocated_bytes": state["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes": state["peak_cuda_reserved_bytes"],
            "minimum_free_disk_bytes": state["minimum_free_disk_bytes"],
            "durable_output_bytes_before_report": durable_bytes,
        },
        "artifacts": {
            "raw_code_artifact_bytes": 0,
            "durable_token_artifact_bytes": 0,
            "tokenizer_atomic": True,
            "checkpoint_atomic": True,
            "report_atomic": True,
        },
        "gates": gates,
        "decision": "phase_b1_full_outer_resource_gate_pass"
        if passed
        else "phase_b1_full_outer_resource_gate_fail",
        "ready_for": {
            "phase_b1_complete": passed,
            "five_fold_oof": False,
            "val_test_or_full": False,
            "promotion": False,
        },
    }
    assert_report_has_no_quality_metrics(report)
    return report


def _load_checkpoint_weights_only(
    path: Path,
    contract: dict[str, Any],
    **verification_kwargs: Any,
) -> dict[str, Any]:
    import torch

    payload = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=True)
    return verify_checkpoint_payload(payload, contract, **verification_kwargs)


def _merge_scan_resource_peak(state: dict[str, Any], scan: OuterFitScan) -> None:
    state["peak_process_rss_bytes"] = max(
        int(state["peak_process_rss_bytes"]),
        int(scan.accounting["peak_process_rss_bytes"]),
    )


def _internal_worker_command(mode: str, args: argparse.Namespace) -> tuple[str, ...]:
    if mode not in {"resume", "final_verify"}:
        raise ValueError("Unknown B1 internal worker mode")
    mode_flag = "--resume-worker" if mode == "resume" else "--final-verify-worker"
    return (
        sys.executable,
        str(Path(__file__).resolve(strict=True)),
        mode_flag,
        "--contract",
        str(args.contract),
        "--folds",
        str(args.folds),
        "--folds-summary",
        str(args.folds_summary),
        "--data-root",
        str(args.data_root),
        "--tokenizer-output",
        str(args.tokenizer_output),
        "--checkpoint-output",
        str(args.checkpoint_output),
        "--report-output",
        str(args.report_output),
    )


def _handoff_environment(handoff: RunHandoff, *, resume_pid: int = 0) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "AXON_B1_HANDOFF_NONCE": handoff.handoff_nonce,
            "AXON_B1_AUTHORIZATION_SHA256": handoff.authorization_sha256,
            "AXON_B1_MARKER_SHA256": handoff.marker_sha256,
            "AXON_B1_PARENT_PID": str(handoff.parent_pid),
            "AXON_B1_PARENT_ARGV_SHA256": handoff.canonical_parent_argv_sha256,
            "AXON_B1_RESUME_PID": str(resume_pid),
        }
    )
    return environment


def _handoff_from_environment() -> tuple[RunHandoff, int]:
    nonce = os.environ.get("AXON_B1_HANDOFF_NONCE", "")
    if len(nonce) != 64 or not set(nonce) <= LOWER_HEX:
        raise B1FatalError("B1 internal worker has no valid in-memory handoff nonce")
    try:
        parent_pid = int(os.environ["AXON_B1_PARENT_PID"])
        resume_pid = int(os.environ.get("AXON_B1_RESUME_PID", "0"))
    except (KeyError, ValueError) as exc:
        raise B1FatalError("B1 internal worker PID handoff is invalid") from exc
    handoff = RunHandoff(
        authorization_sha256=os.environ.get("AXON_B1_AUTHORIZATION_SHA256", ""),
        marker_sha256=os.environ.get("AXON_B1_MARKER_SHA256", ""),
        handoff_nonce=nonce,
        parent_pid=parent_pid,
        canonical_parent_argv_sha256=os.environ.get("AXON_B1_PARENT_ARGV_SHA256", ""),
    )
    if not all(
        _is_lower_sha256(value)
        for value in (
            handoff.authorization_sha256,
            handoff.marker_sha256,
            handoff.canonical_parent_argv_sha256,
        )
    ):
        raise B1FatalError("B1 internal worker commitment handoff is invalid")
    return handoff, resume_pid


def _spawn_internal_worker(
    mode: str,
    args: argparse.Namespace,
    handoff: RunHandoff,
    *,
    cumulative_wall_seconds: float,
    contract: dict[str, Any],
    resume_pid: int = 0,
) -> int:
    command = _internal_worker_command(mode, args)
    remaining = contract["resource_gates"]["maximum_cumulative_wall_seconds"] - float(
        cumulative_wall_seconds
    )
    if remaining <= 0:
        raise B1FatalError("No B1 wall-time budget remains for a fresh worker")
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            env=_handoff_environment(handoff, resume_pid=resume_pid),
            timeout=remaining,
        )
    except subprocess.TimeoutExpired as exc:
        raise B1FatalError(f"B1 {mode} worker exceeded the remaining 8-hour budget") from exc
    return int(completed.returncode)


def _validate_runtime_paths(
    args: argparse.Namespace,
    bindings: dict[str, dict[str, str]],
) -> None:
    if args.folds.resolve(strict=True) != Path(bindings["diagnostic_folds"]["path"]):
        raise B1FatalError("Runtime folds path differs from the frozen static binding")
    if args.folds_summary.resolve(strict=True) != Path(
        bindings["diagnostic_folds_summary"]["path"]
    ):
        raise B1FatalError("Runtime folds summary differs from the frozen static binding")
    summary = _parse_json_object(
        _read_bounded(args.folds_summary.resolve(strict=True), 2 * 1024 * 1024),
        "bound diagnostic folds summary",
    )
    inputs = summary.get("inputs")
    expected_root = (
        Path(str(inputs.get("materialized_data_root"))).resolve(strict=True)
        if isinstance(inputs, dict)
        else None
    )
    if expected_root is None or args.data_root.resolve(strict=True) != expected_root:
        raise B1FatalError("Runtime data root differs from the bound fold summary")
    output_paths = {
        args.tokenizer_output.absolute(),
        args.checkpoint_output.absolute(),
        args.report_output.absolute(),
    }
    if len(output_paths) != 3:
        raise B1FatalError("Tokenizer, checkpoint, and report outputs must be distinct")


def validate_canonical_preflight(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, Any], str]:
    """Validate the complete parent closure without consuming the lease or opening raw data."""
    contract, bindings = validate_static_preflight(args.contract)
    _validate_runtime_paths(args, bindings)
    authorization, authorization_sha = validate_run_authorization(contract, bindings)
    if DEFAULT_MARKER.exists():
        raise B1FatalError("Phase B1 one-shot authorization marker already exists")
    if (
        _free_disk_bytes(args.checkpoint_output.parent)
        < contract["resource_gates"]["minimum_free_disk_bytes_before_raw_open"]
    ):
        raise B1FatalError("Insufficient free disk before B1 lease consumption")
    guard_non_cuda_phase(
        contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=0.0,
    )
    return contract, bindings, authorization, authorization_sha


def run_initial_controller(args: argparse.Namespace) -> int:
    phase_started = time.perf_counter()
    contract, bindings, authorization, authorization_sha = validate_canonical_preflight(args)
    handoff = consume_run_authorization(authorization, authorization_sha)
    validate_consumption_marker(handoff)
    pre_scan_guard = guard_non_cuda_phase(
        contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=time.perf_counter() - phase_started,
    )
    scope = load_and_select_outer_fit_scope(
        contract,
        folds_path=args.folds,
        folds_summary_path=args.folds_summary,
        data_root=args.data_root,
    )
    before_scan = time.perf_counter() - phase_started
    scan = scan_outer_fit_corpus(
        scope,
        contract,
        data_root=args.data_root,
        disk_probe_path=args.checkpoint_output.parent,
        cumulative_wall_seconds_before=before_scan,
    )
    post_scan_guard = guard_non_cuda_phase(
        contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=time.perf_counter() - phase_started,
    )
    tokenizer, compact, tokenizer_artifact = fit_fresh_tokenizer_and_compact_corpus(
        scan,
        contract,
        tokenizer_path=args.tokenizer_output,
    )
    post_compact_guard = guard_non_cuda_phase(
        contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=time.perf_counter() - phase_started,
    )
    scan_accounting = dict(scan.accounting)
    scan_commitment = scan.outer_fit_corpus_commitment_sha256
    permutation = deterministic_permutation(
        len(compact.corpus), int(contract["training"]["shuffle_seed"])
    )
    schedule = prepare_validated_schedule(permutation, contract)
    total_optimizer_steps = (
        len(permutation) + contract["training"]["effective_full_step_sequences"] - 1
    ) // contract["training"]["effective_full_step_sequences"]
    resume_step = contract["checkpoint_and_resume"]["fresh_process_resume_at_optimizer_step"]
    if total_optimizer_steps <= resume_step:
        raise B1FatalError("B1 corpus cannot reach the frozen fresh-process resume boundary")
    torch, model, optimizer, model_config, scaler, mask_generator = (
        _initialize_training_runtime(contract, tokenizer)
    )
    state = _new_training_state()
    _merge_scan_resource_peak(state, scan)
    state["peak_process_rss_bytes"] = max(
        state["peak_process_rss_bytes"],
        pre_scan_guard["process_rss_bytes"],
        post_scan_guard["process_rss_bytes"],
        post_compact_guard["process_rss_bytes"],
    )
    state["minimum_free_disk_bytes"] = min(
        state["minimum_free_disk_bytes"],
        scan.accounting["minimum_free_disk_bytes"],
        pre_scan_guard["free_disk_bytes"],
        post_scan_guard["free_disk_bytes"],
        post_compact_guard["free_disk_bytes"],
    )
    run_context = {
        "input_bindings": bindings,
        "scope_audit": scope.audit,
        "initial_scan": scan_accounting,
        "tokenizer_artifact": tokenizer_artifact,
        "compact_accounting": compact.accounting,
        "authorization_sha256": handoff.authorization_sha256,
        "marker_sha256": handoff.marker_sha256,
        "handoff_nonce_sha256": handoff.handoff_nonce_sha256,
        "canonical_parent_argv_sha256": handoff.canonical_parent_argv_sha256,
        "prepared_sequence_count": len(permutation),
        "total_optimizer_steps": total_optimizer_steps,
        "resume_argv_sha256": _argv_sha256(_internal_worker_command("resume", args)),
        "final_verify_argv_sha256": _argv_sha256(
            _internal_worker_command("final_verify", args)
        ),
        "parent_non_cuda_resource_guards": {
            "pre_scan": pre_scan_guard,
            "post_scan": post_scan_guard,
            "post_compact": post_compact_guard,
        },
    }
    del scan
    gc.collect()
    initial_cumulative_before_training = time.perf_counter() - phase_started
    train_segment(
        torch_module=torch,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        model_config=model_config,
        tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_artifact["sha256"],
        corpus=compact.corpus,
        corpus_commitment=scan_commitment,
        permutation=permutation,
        schedule=schedule,
        mask_generator=mask_generator,
        state=state,
        contract=contract,
        checkpoint_path=args.checkpoint_output,
        cumulative_wall_seconds_before=initial_cumulative_before_training,
        phase_started=time.perf_counter(),
        parent_pid=os.getpid(),
        handoff=handoff,
        canonical_child_argv_sha256=run_context["resume_argv_sha256"],
        run_context=run_context,
        stop_after_optimizer_step=resume_step,
    )
    if state["completed_optimizer_steps"] != resume_step:
        raise B1FatalError("Initial B1 process did not stop exactly at optimizer step 4096")
    del model, optimizer, scaler, compact, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return_code = _spawn_internal_worker(
        "resume",
        args,
        handoff,
        cumulative_wall_seconds=time.perf_counter() - phase_started,
        contract=contract,
    )
    if return_code != 0:
        raise B1FatalError(f"Fresh B1 resume process failed with exit code {return_code}")
    return 0


def run_final_verify_worker(args: argparse.Namespace) -> dict[str, Any]:
    verifier_started = time.perf_counter()
    handoff, resume_pid = _handoff_from_environment()
    if resume_pid <= 0:
        raise B1FatalError("Final verifier has no bound resume-process PID")
    validate_consumption_marker(handoff, require_direct_parent_pid=resume_pid)
    contract, bindings = validate_static_preflight(args.contract)
    _validate_runtime_paths(args, bindings)
    _authorization, authorization_sha = validate_run_authorization(contract, bindings)
    if authorization_sha != handoff.authorization_sha256:
        raise B1FatalError("Final verifier authorization SHA handoff drifted")
    final_argv_sha = _argv_sha256(_internal_worker_command("final_verify", args))
    payload = _load_checkpoint_weights_only(
        args.checkpoint_output,
        contract,
        expected_handoff=handoff,
        expected_child_argv_sha256=final_argv_sha,
        require_final=True,
    )
    if payload.get("resume_pid") != resume_pid or payload.get("parent_pid") != handoff.parent_pid:
        raise B1FatalError("Final checkpoint process lineage drifted")
    run_context = payload.get("run_context")
    if not isinstance(run_context, dict) or (
        run_context.get("input_bindings") != bindings
        or run_context.get("final_verify_argv_sha256") != final_argv_sha
    ):
        raise B1FatalError("Final checkpoint run context drifted")
    from tokenizers import Tokenizer

    tokenizer_path = args.tokenizer_output.resolve(strict=True)
    if _artifact_sha(tokenizer_path) != payload["tokenizer_sha256"]:
        raise B1FatalError("Final verifier tokenizer SHA drifted")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    torch, model, optimizer, _config, _scaler, _generator, exact_logits = (
        _restore_training_runtime(contract, tokenizer, payload)
    )
    _assert_finite_tensor_tree(model.state_dict(), "final_model_state")
    _assert_finite_tensor_tree(optimizer.state_dict(), "final_optimizer_state")
    state = dict(payload["training_state"])
    guard_non_cuda_phase(
        contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=float(payload["cumulative_wall_seconds"])
        + time.perf_counter()
        - verifier_started,
        state=state,
    )
    if not exact_logits:
        raise B1FatalError("Final verifier synthetic logits are not bit exact")
    receipt = {
        "schema": "axon_loop166_phase_b1_final_checkpoint_verification_v1",
        "loop_id": "loop166_code_section_foundation",
        "authorization_sha256": handoff.authorization_sha256,
        "marker_sha256": handoff.marker_sha256,
        "handoff_nonce_sha256": handoff.handoff_nonce_sha256,
        "checkpoint_sha256": _artifact_sha(args.checkpoint_output),
        "completed_optimizer_steps": payload["completed_optimizer_steps"],
        "completed_sequence_count": payload["completed_sequence_count"],
        "next_permutation_cursor": payload["next_permutation_cursor"],
        "model_tensors_finite": True,
        "optimizer_tensors_finite": True,
        "rng_state_validated": True,
        "synthetic_logits_bit_exact": True,
        "parent_pid": handoff.parent_pid,
        "resume_pid": resume_pid,
        "verifier_pid": os.getpid(),
        "quality_metrics_computed": False,
        "threshold_operations_performed": False,
        "decision": "phase_b1_final_checkpoint_verification_pass",
    }
    assert_report_has_no_quality_metrics(receipt)
    _atomic_json_save(DEFAULT_FINAL_VERIFY_RECEIPT, receipt)
    del model, optimizer, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def _load_final_verify_receipt(
    handoff: RunHandoff,
    *,
    checkpoint_path: Path,
    resume_pid: int,
) -> dict[str, Any]:
    raw = _read_bounded(DEFAULT_FINAL_VERIFY_RECEIPT.resolve(strict=True), 2 * 1024 * 1024)
    receipt = _parse_json_object(raw, "Phase B1 final verification receipt")
    verifier_pid = receipt.get("verifier_pid")
    if (
        receipt.get("schema")
        != "axon_loop166_phase_b1_final_checkpoint_verification_v1"
        or receipt.get("decision") != "phase_b1_final_checkpoint_verification_pass"
        or receipt.get("authorization_sha256") != handoff.authorization_sha256
        or receipt.get("marker_sha256") != handoff.marker_sha256
        or receipt.get("handoff_nonce_sha256") != handoff.handoff_nonce_sha256
        or receipt.get("checkpoint_sha256") != _artifact_sha(checkpoint_path)
        or receipt.get("parent_pid") != handoff.parent_pid
        or receipt.get("resume_pid") != resume_pid
        or not isinstance(verifier_pid, int)
        or isinstance(verifier_pid, bool)
        or verifier_pid <= 0
        or verifier_pid in {handoff.parent_pid, resume_pid}
        or receipt.get("model_tensors_finite") is not True
        or receipt.get("optimizer_tensors_finite") is not True
        or receipt.get("rng_state_validated") is not True
        or receipt.get("synthetic_logits_bit_exact") is not True
    ):
        raise B1FatalError("Independent final checkpoint receipt drifted")
    assert_report_has_no_quality_metrics(receipt)
    return receipt


def run_resume_worker(args: argparse.Namespace) -> dict[str, Any]:
    worker_started = time.perf_counter()
    handoff, unexpected_resume_pid = _handoff_from_environment()
    if unexpected_resume_pid != 0:
        raise B1FatalError("Resume worker received a final-verifier PID handoff")
    validate_consumption_marker(handoff, require_direct_parent_pid=handoff.parent_pid)
    contract, current_bindings = validate_static_preflight(args.contract)
    _validate_runtime_paths(args, current_bindings)
    _authorization, authorization_sha = validate_run_authorization(
        contract, current_bindings
    )
    if authorization_sha != handoff.authorization_sha256:
        raise B1FatalError("Resume worker authorization SHA handoff drifted")
    resume_argv_sha = _argv_sha256(_internal_worker_command("resume", args))
    payload = _load_checkpoint_weights_only(
        args.checkpoint_output,
        contract,
        expected_handoff=handoff,
        expected_child_argv_sha256=resume_argv_sha,
    )
    parent_pid = int(payload["parent_pid"])
    if parent_pid == os.getpid():
        raise B1FatalError("B1 resume did not enter a fresh operating-system process")
    if payload["completed_optimizer_steps"] != contract["checkpoint_and_resume"][
        "fresh_process_resume_at_optimizer_step"
    ]:
        raise B1FatalError("Fresh B1 resume checkpoint is not at optimizer step 4096")
    run_context = payload.get("run_context")
    if not isinstance(run_context, dict):
        raise B1FatalError("B1 checkpoint is missing its initial run context")
    if run_context.get("input_bindings") != current_bindings:
        raise B1FatalError("B1 static bindings changed between parent and resume processes")
    if run_context.get("resume_argv_sha256") != resume_argv_sha:
        raise B1FatalError("Resume worker argv commitment drifted from run context")
    expected_final_argv_sha = _argv_sha256(_internal_worker_command("final_verify", args))
    if run_context.get("final_verify_argv_sha256") != expected_final_argv_sha:
        raise B1FatalError("Final verifier argv commitment drifted from run context")
    pre_resume_scan_guard = guard_non_cuda_phase(
        contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=float(payload["cumulative_wall_seconds"])
        + time.perf_counter()
        - worker_started,
    )

    scope = load_and_select_outer_fit_scope(
        contract,
        folds_path=args.folds,
        folds_summary_path=args.folds_summary,
        data_root=args.data_root,
    )
    scan = scan_outer_fit_corpus(
        scope,
        contract,
        data_root=args.data_root,
        disk_probe_path=args.checkpoint_output.parent,
        cumulative_wall_seconds_before=float(payload["cumulative_wall_seconds"])
        + time.perf_counter()
        - worker_started,
    )
    if (
        scan.outer_fit_corpus_commitment_sha256
        != payload["outer_fit_corpus_commitment_sha256"]
    ):
        raise B1FatalError("Fresh-process outer-fit raw corpus commitment drifted")
    tokenizer, compact = rebuild_compact_corpus_from_tokenizer(
        scan,
        contract,
        tokenizer_path=args.tokenizer_output,
        expected_tokenizer_sha256=str(payload["tokenizer_sha256"]),
    )
    post_resume_compact_guard = guard_non_cuda_phase(
        contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=float(payload["cumulative_wall_seconds"])
        + time.perf_counter()
        - worker_started,
    )
    if compact.corpus.commitment_sha256() != payload["compact_corpus_commitment_sha256"]:
        raise B1FatalError("Fresh-process compact corpus commitment drifted")
    resume_scan_accounting = dict(scan.accounting)
    resume_scan_commitment = scan.outer_fit_corpus_commitment_sha256
    permutation = deterministic_permutation(
        len(compact.corpus), int(contract["training"]["shuffle_seed"])
    )
    schedule = prepare_validated_schedule(permutation, contract)
    if permutation_commitment_sha256(permutation) != payload["shuffle_commitment_sha256"]:
        raise B1FatalError("Fresh-process deterministic permutation commitment drifted")
    verify_checkpoint_payload(
        payload,
        contract,
        expected_handoff=handoff,
        expected_child_argv_sha256=resume_argv_sha,
        corpus=compact.corpus,
        permutation=permutation,
    )
    torch, model, optimizer, model_config, scaler, mask_generator, exact_logits = (
        _restore_training_runtime(contract, tokenizer, payload)
    )
    state = dict(payload["training_state"])
    if (
        state["completed_optimizer_steps"] != payload["completed_optimizer_steps"]
        or state["next_permutation_cursor"] != payload["next_permutation_cursor"]
        or state["completed_sequence_count"] != payload["completed_sequence_count"]
    ):
        raise B1FatalError("Fresh-process restored training state disagrees with checkpoint cursor")
    _merge_scan_resource_peak(state, scan)
    state["peak_process_rss_bytes"] = max(
        state["peak_process_rss_bytes"],
        pre_resume_scan_guard["process_rss_bytes"],
        post_resume_compact_guard["process_rss_bytes"],
    )
    state["minimum_free_disk_bytes"] = min(
        state["minimum_free_disk_bytes"],
        scan.accounting["minimum_free_disk_bytes"],
        pre_resume_scan_guard["free_disk_bytes"],
        post_resume_compact_guard["free_disk_bytes"],
    )
    del scan
    gc.collect()
    train_segment(
        torch_module=torch,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        model_config=model_config,
        tokenizer=tokenizer,
        tokenizer_sha256=str(payload["tokenizer_sha256"]),
        corpus=compact.corpus,
        corpus_commitment=resume_scan_commitment,
        permutation=permutation,
        schedule=schedule,
        mask_generator=mask_generator,
        state=state,
        contract=contract,
        checkpoint_path=args.checkpoint_output,
        cumulative_wall_seconds_before=float(payload["cumulative_wall_seconds"]),
        phase_started=worker_started,
        parent_pid=parent_pid,
        handoff=handoff,
        canonical_child_argv_sha256=run_context["final_verify_argv_sha256"],
        resume_pid=os.getpid(),
        run_context=run_context,
    )
    if state["next_permutation_cursor"] != len(permutation):
        raise B1FatalError("B1 epoch ended before consuming the full permutation")
    cumulative_wall = float(payload["cumulative_wall_seconds"]) + (
        time.perf_counter() - worker_started
    )
    guard_non_cuda_phase(
        contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=cumulative_wall,
        state=state,
    )
    del model, optimizer, scaler, mask_generator
    gc.collect()
    torch.cuda.empty_cache()
    verify_return_code = _spawn_internal_worker(
        "final_verify",
        args,
        handoff,
        cumulative_wall_seconds=cumulative_wall,
        contract=contract,
        resume_pid=os.getpid(),
    )
    if verify_return_code != 0:
        raise B1FatalError(
            f"Independent final checkpoint verifier failed with exit code {verify_return_code}"
        )
    final_verify_receipt = _load_final_verify_receipt(
        handoff,
        checkpoint_path=args.checkpoint_output,
        resume_pid=os.getpid(),
    )
    cumulative_wall = float(payload["cumulative_wall_seconds"]) + (
        time.perf_counter() - worker_started
    )
    guard_non_cuda_phase(
        contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=cumulative_wall,
        state=state,
    )
    resume_scan_report = {
        **resume_scan_accounting,
        "commitment_match": True,
        "outer_fit_corpus_commitment_sha256": resume_scan_commitment,
    }
    tokenizer_artifact = dict(run_context["tokenizer_artifact"])
    report = _build_final_report(
        bindings=current_bindings,
        scope=scope,
        initial_scan=dict(run_context["initial_scan"]),
        resume_scan=resume_scan_report,
        compact=compact,
        tokenizer_artifact=tokenizer_artifact,
        checkpoint_path=args.checkpoint_output,
        report_path=args.report_output,
        state=state,
        contract=contract,
        cumulative_wall_seconds=cumulative_wall,
        parent_pid=parent_pid,
        resume_pid=os.getpid(),
        exact_logits=exact_logits,
        final_verify_receipt=final_verify_receipt,
    )
    _atomic_json_save(args.report_output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one frozen Loop166 B1 full outer-fit resource cell."
    )
    parser.add_argument("--resume-worker", action="store_true")
    parser.add_argument("--final-verify-worker", action="store_true")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--folds-summary", type=Path, default=DEFAULT_FOLDS_SUMMARY)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tokenizer-output", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--checkpoint-output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser


def _normalize_cli_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.contract = args.contract if args.contract.is_absolute() else PROJECT_ROOT / args.contract
    args.folds = args.folds if args.folds.is_absolute() else PROJECT_ROOT / args.folds
    args.folds_summary = (
        args.folds_summary
        if args.folds_summary.is_absolute()
        else PROJECT_ROOT / args.folds_summary
    )
    args.data_root = (
        args.data_root if args.data_root.is_absolute() else PROJECT_ROOT / args.data_root
    )
    args.tokenizer_output = _resolve_output_path(args.tokenizer_output)
    args.checkpoint_output = _resolve_output_path(args.checkpoint_output)
    args.report_output = _resolve_output_path(args.report_output)
    expected = {
        "contract": DEFAULT_CONTRACT.resolve(strict=True),
        "folds": DEFAULT_FOLDS.resolve(strict=True),
        "folds_summary": DEFAULT_FOLDS_SUMMARY.resolve(strict=True),
        "data_root": DEFAULT_DATA_ROOT.resolve(strict=True),
        "tokenizer_output": DEFAULT_TOKENIZER.absolute(),
        "checkpoint_output": DEFAULT_CHECKPOINT.absolute(),
        "report_output": DEFAULT_REPORT.absolute(),
    }
    for name, expected_path in expected.items():
        if Path(getattr(args, name)).absolute() != expected_path:
            raise B1FatalError(f"B1 runtime path is not canonical: {name}")
    if args.resume_worker and args.final_verify_worker:
        raise B1FatalError("B1 internal worker modes are mutually exclusive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _normalize_cli_paths(build_parser().parse_args(argv))
    actual_argv = (
        str(Path(sys.executable).resolve(strict=True)),
        str(Path(sys.argv[0]).resolve(strict=True)),
        *sys.argv[1:],
    )
    if not args.resume_worker and not args.final_verify_worker:
        if actual_argv != canonical_parent_argv():
            raise B1FatalError("B1 parent argv differs from the authorized canonical invocation")
    else:
        mode = "resume" if args.resume_worker else "final_verify"
        if actual_argv != _internal_worker_command(mode, args):
            raise B1FatalError("B1 internal worker argv differs from its canonical handoff")
    if args.resume_worker:
        report = run_resume_worker(args)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["decision"] == "phase_b1_full_outer_resource_gate_pass" else 2
    if args.final_verify_worker:
        receipt = run_final_verify_worker(args)
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return run_initial_controller(args)


__all__ = [
    "B1FatalError",
    "CompactCorpusBuild",
    "OuterFitScan",
    "OuterFitScope",
    "VerifiedSource",
    "build_compact_corpus",
    "build_checkpoint_payload",
    "build_parser",
    "assert_report_has_no_quality_metrics",
    "deterministic_permutation",
    "fit_fresh_tokenizer_and_compact_corpus",
    "load_and_select_outer_fit_scope",
    "read_verified_outer_fit_source",
    "scan_outer_fit_corpus",
    "select_outer_fit_records",
    "take_step_indices",
    "verify_checkpoint_payload",
    "validate_canonical_preflight",
    "validate_run_authorization",
    "validate_static_preflight",
]


def take_step_indices(
    permutation: Sequence[int],
    cursor: int,
    *,
    maximum_sequences: int = 4,
) -> tuple[tuple[int, ...], int]:
    """Synthetic-test helper: return one non-repeating group and its next cursor."""
    if maximum_sequences != 4:
        raise ValueError("B1 frozen optimizer groups contain at most four sequences")
    group = optimizer_group_from_cursor(
        permutation,
        cursor,
        microbatch_size=2,
        gradient_accumulation_steps=2,
    )
    if group is None:
        return (), cursor
    return group.indices, group.cursor_end


if __name__ == "__main__":
    raise SystemExit(main())
