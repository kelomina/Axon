#!/usr/bin/env python3
"""Recover the fail-closed Loop166 Phase B1 run from its bound step-4096 checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_loop166_phase_b1_full_outer_resource_cell as b1  # noqa: E402

from loop166.b1_schedule import (  # noqa: E402
    deterministic_permutation,
    permutation_commitment_sha256,
)
from loop166.windows_process_lineage import (  # noqa: E402
    ProcessLineageError,
    validate_spawn_lineage,
)

FOUNDATION_DIR = (
    PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop166_code_section_foundation"
)
LOOP166_REPORT_DIR = PROJECT_ROOT / "reports" / "roadmap_9997" / "loop166"
LOOP166_MODEL_DIR = PROJECT_ROOT / "models" / "roadmap_9997" / "loop166"

DEFAULT_CONTRACT = FOUNDATION_DIR / "phase_b1_step4096_recovery.json"
DEFAULT_AUTHORIZATION = FOUNDATION_DIR / "phase_b1_step4096_recovery_authorization.json"
DEFAULT_INCIDENT = FOUNDATION_DIR / "phase_b1_resume_lineage_failure.json"
DEFAULT_FOLDS = b1.DEFAULT_FOLDS
DEFAULT_FOLDS_SUMMARY = b1.DEFAULT_FOLDS_SUMMARY
DEFAULT_DATA_ROOT = b1.DEFAULT_DATA_ROOT
DEFAULT_SOURCE_TOKENIZER = b1.DEFAULT_TOKENIZER
DEFAULT_SOURCE_CHECKPOINT = b1.DEFAULT_CHECKPOINT
DEFAULT_CHECKPOINT_OUTPUT = LOOP166_MODEL_DIR / "phase_b1_step4096_recovered_tiny_mlm.pt"
DEFAULT_REPORT_OUTPUT = LOOP166_REPORT_DIR / "phase_b1_step4096_recovery_report.json"
DEFAULT_MARKER = LOOP166_REPORT_DIR / "phase_b1_step4096_recovery_consumed.json"
DEFAULT_FINAL_RECEIPT = (
    LOOP166_REPORT_DIR / "phase_b1_step4096_recovery_final_verify_receipt.json"
)

ORIGINAL_CONTRACT = b1.DEFAULT_CONTRACT
ORIGINAL_AUTHORIZATION = b1.DEFAULT_RUN_AUTH
ORIGINAL_CONTROLLER = PROJECT_ROOT / "scripts" / "run_loop166_phase_b1_full_outer_resource_cell.py"
ORIGINAL_TESTS = PROJECT_ROOT / "tests" / "test_loop166_phase_b1.py"
ORIGINAL_MARKER = b1.DEFAULT_MARKER

CONTRACT_SCHEMA = "axon_loop166_phase_b1_step4096_recovery_contract_v1"
AUTHORIZATION_SCHEMA = "axon_loop166_phase_b1_step4096_recovery_authorization_v1"
MARKER_SCHEMA = "axon_loop166_phase_b1_step4096_recovery_consumption_v1"
RECEIPT_SCHEMA = "axon_loop166_phase_b1_step4096_recovery_final_verification_v1"
REPORT_SCHEMA = "axon_loop166_phase_b1_step4096_recovery_report_v1"
LEASE_ID = "loop166-b1-step4096-recovery-v1"
PASS_DECISION = "phase_b1_completed_via_authorized_step4096_recovery"
FAILURE_DECISION = "phase_b1_step4096_recovery_resource_gate_fail"

EXPECTED_ORIGINAL_HASHES = {
    "authorization": "ce271be62e335631db39079ade914f9a111dd58b4c220b319260a0c8bc2564f7",
    "contract": "5e4355ef0d37aacd358e7b8323a35d6eb6bc9935620eb9cded6f460d3a61b609",
    "controller": "12be192088b408a31682b1275d56b29a301841a3563f7518c6371afa49c931bb",
    "tests": "0c45a5be3ff077d463c37634bd860bb7f0f59b8228f044a6b69ea0a52d4073ee",
    "marker": "a4b0de3556f2f084736b78c1cdce647464300ac0c9f2c27271373320172cbd75",
    "checkpoint": "82177e6d69ef1a2e17a1ea09dcb3e870a32e3305f92ea15bcff124fdc86f14a6",
    "tokenizer": "73734f28f01b5045ff6f77beb9c07bd62e52102c5153bd94e21c28285659c7f6",
    "incident": "4f012e8730e46261329126f6dd86cc14e87d14c5a5d9a6b1f34e320a0fa99361",
}
EXPECTED_ORIGINAL_STATE = {
    "completed_optimizer_steps": 4096,
    "completed_sequence_count": 16_384,
    "next_permutation_cursor": 16_384,
    "permutation_prefix_original_bytes": 8_200_700,
    "prepared_sequence_count": 115_072,
    "total_optimizer_steps": 28_768,
    "resume_pid": 0,
}
EXPECTED_TOTAL_ORIGINAL_BYTES = 57_614_320
EXPECTED_COMMITMENTS = {
    "outer_fit_corpus_commitment_sha256": (
        "880b9cb7a79f5253b4d7562280baeb9e8543ccae6f89791a8a97a3da91159ffc"
    ),
    "compact_corpus_commitment_sha256": (
        "325119a0a5f7bb548b23112d16ecce7c2d43b9e9986c18e3061eaaa2bd77f13c"
    ),
    "shuffle_commitment_sha256": (
        "aa79b33bdf73ca13006f5b9772f73dd4d4d35fe43c8cd2fe61b07b5b2a9569ad"
    ),
    "old_handoff_nonce_sha256": (
        "05ceed5316aa36b13a6c241450287ca817163f3a504358b73d154c8fb13ead99"
    ),
}
SOURCE_CUMULATIVE_WALL_SECONDS = 2554.2020570000022
CONSERVATIVE_FAILED_HANDOFF_DEBIT_SECONDS = 300.0
RECOVERY_BUDGET_BASE_SECONDS = 2854.2020570000022
MAXIMUM_CUMULATIVE_WALL_SECONDS = 28_800.0
LOWER_HEX = frozenset("0123456789abcdef")


class RecoveryFatalError(RuntimeError):
    """Raised when the bounded recovery cannot prove every frozen invariant."""


@dataclass(frozen=True)
class RecoveryControlClosure:
    contract: dict[str, Any]
    contract_sha256: str
    authorization: dict[str, Any]
    authorization_sha256: str
    runtime: dict[str, str]


@dataclass(frozen=True)
class RecoveryClosure(RecoveryControlClosure):
    original_contract: dict[str, Any]
    original_bindings: dict[str, dict[str, str]]
    source_payload: dict[str, Any]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    try:
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RecoveryFatalError("Recovery audit payload is not canonical JSON") from exc
    return _sha256(raw)


def _artifact_sha(path: Path) -> str:
    return b1._artifact_sha(path.resolve(strict=True))


def _assert_stable_artifact_sha(path: Path, before_sha256: str, context: str) -> str:
    after_sha256 = _artifact_sha(path)
    if after_sha256 != before_sha256:
        raise RecoveryFatalError(f"{context} changed during verification")
    return after_sha256


def _read_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    raw = b1._read_bounded(path.resolve(strict=True), 8 * 1024 * 1024)
    return b1._parse_json_object(raw, context), raw


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= LOWER_HEX


def _binding_path(binding: object, context: str) -> tuple[Path, str]:
    if not isinstance(binding, dict) or set(binding) < {"path", "sha256"}:
        raise RecoveryFatalError(f"Recovery binding is missing: {context}")
    expected_sha = binding.get("sha256")
    if not _is_sha256(expected_sha):
        raise RecoveryFatalError(f"Recovery binding remains pending: {context}")
    value = Path(str(binding.get("path")))
    candidate = value if value.is_absolute() else PROJECT_ROOT / value
    try:
        path = candidate.resolve(strict=True)
    except OSError as exc:
        raise RecoveryFatalError(f"Recovery binding path is unavailable: {context}") from exc
    if _artifact_sha(path) != expected_sha:
        raise RecoveryFatalError(f"Recovery binding SHA drifted: {context}")
    return path, expected_sha


def _canonical_path(path: Path, *, must_exist: bool) -> str:
    if must_exist:
        return str(path.resolve(strict=True))
    return str(path.absolute())


def _common_cli_args(args: argparse.Namespace) -> tuple[str, ...]:
    return (
        "--contract",
        str(args.contract),
        "--authorization",
        str(args.authorization),
        "--folds",
        str(args.folds),
        "--folds-summary",
        str(args.folds_summary),
        "--data-root",
        str(args.data_root),
        "--source-tokenizer",
        str(args.source_tokenizer),
        "--source-checkpoint",
        str(args.source_checkpoint),
        "--checkpoint-output",
        str(args.checkpoint_output),
        "--report-output",
        str(args.report_output),
    )


def canonical_command(mode: str, args: argparse.Namespace) -> tuple[str, ...]:
    if mode not in {"parent", "recovery", "final_verify"}:
        raise ValueError(f"Unknown recovery mode: {mode}")
    flag: tuple[str, ...] = ()
    if mode == "recovery":
        flag = ("--recovery-worker",)
    elif mode == "final_verify":
        flag = ("--final-verify-worker",)
    return (
        str(Path(sys.executable).resolve(strict=True)),
        str(Path(__file__).resolve(strict=True)),
        *flag,
        *_common_cli_args(args),
    )


def _argv_sha256(argv: Sequence[str]) -> str:
    raw = json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256(raw)


def _expected_paths(args: argparse.Namespace) -> dict[str, str]:
    return {
        "project_root": _canonical_path(PROJECT_ROOT, must_exist=True),
        "incident": _canonical_path(DEFAULT_INCIDENT, must_exist=True),
        "contract": _canonical_path(args.contract, must_exist=True),
        "authorization": _canonical_path(args.authorization, must_exist=True),
        "controller": _canonical_path(Path(__file__), must_exist=True),
        "windows_process_lineage": _canonical_path(
            SRC_DIR / "loop166" / "windows_process_lineage.py", must_exist=True
        ),
        "tests": _canonical_path(
            PROJECT_ROOT / "tests" / "test_loop166_phase_b1_step4096_recovery.py",
            must_exist=True,
        ),
        "lineage_tests": _canonical_path(
            PROJECT_ROOT / "tests" / "test_loop166_windows_process_lineage.py",
            must_exist=True,
        ),
        "folds": _canonical_path(args.folds, must_exist=True),
        "folds_summary": _canonical_path(args.folds_summary, must_exist=True),
        "data_root": _canonical_path(args.data_root, must_exist=True),
        "origin_tokenizer": _canonical_path(args.source_tokenizer, must_exist=True),
        "origin_checkpoint": _canonical_path(args.source_checkpoint, must_exist=True),
        "recovery_marker": _canonical_path(DEFAULT_MARKER, must_exist=False),
        "recovered_checkpoint": _canonical_path(
            args.checkpoint_output, must_exist=False
        ),
        "recovery_report": _canonical_path(args.report_output, must_exist=False),
        "final_verify_receipt": _canonical_path(DEFAULT_FINAL_RECEIPT, must_exist=False),
    }


def _validate_runtime(runtime: object) -> dict[str, str]:
    if not isinstance(runtime, dict):
        raise RecoveryFatalError("Recovery authorization runtime binding is missing")
    required_strings = {
        "python_executable",
        "base_executable",
        "python_version",
        "implementation",
        "platform",
        "torch_version",
        "tokenizers_version",
        "pefile_version",
    }
    if any(not isinstance(runtime.get(key), str) for key in required_strings):
        raise RecoveryFatalError("Recovery authorization runtime binding is incomplete")
    frozen = {key: str(runtime[key]) for key in required_strings}
    try:
        launcher = Path(frozen["python_executable"]).resolve(strict=True)
        base = Path(frozen["base_executable"]).resolve(strict=True)
    except OSError as exc:
        raise RecoveryFatalError("Frozen recovery runtime executable is unavailable") from exc
    observed = b1._runtime_binding()
    for key in (
        "python_version",
        "implementation",
        "platform",
        "torch_version",
        "tokenizers_version",
        "pefile_version",
    ):
        if frozen[key] != observed[key]:
            raise RecoveryFatalError(f"Recovery runtime drifted: {key}")
    if launcher != Path(sys.executable).resolve(strict=True):
        raise RecoveryFatalError("Recovery launcher differs from the authorized Python executable")
    observed_base = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    if base != observed_base:
        raise RecoveryFatalError("Recovery base Python executable drifted")
    if runtime.get("version_match_policy") != "exact":
        raise RecoveryFatalError("Recovery runtime version policy drifted")
    return {**frozen, "python_executable": str(launcher), "base_executable": str(base)}


def _validate_original_evidence(
    args: argparse.Namespace,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, Any]]:
    if b1.DEFAULT_REPORT.exists() or b1.DEFAULT_FINAL_VERIFY_RECEIPT.exists():
        raise RecoveryFatalError("Original failed B1 outputs were unexpectedly backfilled")
    expected_paths = {
        "authorization": ORIGINAL_AUTHORIZATION,
        "contract": ORIGINAL_CONTRACT,
        "controller": ORIGINAL_CONTROLLER,
        "tests": ORIGINAL_TESTS,
        "marker": ORIGINAL_MARKER,
        "checkpoint": args.source_checkpoint,
        "tokenizer": args.source_tokenizer,
    }
    for name, path in expected_paths.items():
        if _artifact_sha(path) != EXPECTED_ORIGINAL_HASHES[name]:
            raise RecoveryFatalError(f"Immutable original artifact drifted: {name}")

    incident, incident_raw = _read_json(DEFAULT_INCIDENT, "Phase B1 lineage failure incident")
    incident_binding = contract.get("incident_binding")
    if not isinstance(incident_binding, dict):
        incident_binding = contract.get("failure_incident")
    incident_path, incident_sha = _binding_path(incident_binding, "failure incident")
    if incident_path != DEFAULT_INCIDENT.resolve(strict=True) or incident_sha != _sha256(incident_raw):
        raise RecoveryFatalError("Recovery incident binding drifted")
    if (
        incident.get("schema") != "axon_loop166_phase_b1_resume_lineage_failure_v1"
        or incident.get("status") != "incomplete_fail_closed_at_resume_handoff"
        or incident.get("decision")
        != "phase_b1_incomplete_allow_design_of_one_fail_closed_recovery_continuation"
    ):
        raise RecoveryFatalError("Recovery incident does not authorize a bounded continuation")
    recovery_boundary = incident.get("recovery_boundary")
    if not isinstance(recovery_boundary, dict) or (
        recovery_boundary.get("requires_new_lease_marker_and_nonce") is not True
        or recovery_boundary.get("original_nonce_continuity_claim_allowed") is not False
        or recovery_boundary.get("maximum_additional_full_fit_raw_passes") != 1
        or recovery_boundary.get("resume_cursor") != 16_384
        or recovery_boundary.get("old_budget_must_continue") is not True
    ):
        raise RecoveryFatalError("Recovery boundary drifted from the fail-closed incident")

    original_contract, original_bindings = b1.validate_static_preflight(
        ORIGINAL_CONTRACT,
        controller_path=ORIGINAL_CONTROLLER,
    )
    _original_authorization, original_authorization_sha = b1.validate_run_authorization(
        original_contract,
        original_bindings,
        authorization_path=ORIGINAL_AUTHORIZATION,
    )
    if original_authorization_sha != EXPECTED_ORIGINAL_HASHES["authorization"]:
        raise RecoveryFatalError("Original authorization closure drifted")

    marker, marker_raw = _read_json(ORIGINAL_MARKER, "original Phase B1 marker")
    if _sha256(marker_raw) != EXPECTED_ORIGINAL_HASHES["marker"] or marker != {
        "authorization_sha256": EXPECTED_ORIGINAL_HASHES["authorization"],
        "canonical_parent_argv_sha256": (
            "ae4b12ca187621385f8d6b8fcefb17621364fb42490381b944a51ceea954858c"
        ),
        "handoff_nonce_sha256": EXPECTED_COMMITMENTS["old_handoff_nonce_sha256"],
        "lease_id": "loop166-b1-outer0-resource-cell-v1",
        "loop_id": "loop166_code_section_foundation",
        "parent_pid": 677904,
        "schema": "axon_loop166_phase_b1_execution_consumption_v1",
        "status": "consumed_before_raw_access",
    }:
        raise RecoveryFatalError("Original one-shot marker content drifted")

    source_payload = b1._load_checkpoint_weights_only(
        args.source_checkpoint,
        original_contract,
        expected_child_argv_sha256=(
            "1b9397f47029c78ee61318de6b00422bdbf0cdfa3ab7c46a1fcd2f22298af09d"
        ),
    )
    for key, value in EXPECTED_ORIGINAL_STATE.items():
        observed = source_payload.get(key)
        if key in {"prepared_sequence_count", "total_optimizer_steps"}:
            observed = source_payload.get("run_context", {}).get(key)
        if observed != value:
            raise RecoveryFatalError(f"Source checkpoint state drifted: {key}")
    if not math.isclose(
        float(source_payload.get("cumulative_wall_seconds", math.nan)),
        SOURCE_CUMULATIVE_WALL_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RecoveryFatalError("Source checkpoint cumulative wall time drifted")
    checkpoint_commitments = {
        "outer_fit_corpus_commitment_sha256": source_payload.get(
            "outer_fit_corpus_commitment_sha256"
        ),
        "compact_corpus_commitment_sha256": source_payload.get(
            "compact_corpus_commitment_sha256"
        ),
        "shuffle_commitment_sha256": source_payload.get("shuffle_commitment_sha256"),
        "old_handoff_nonce_sha256": source_payload.get("handoff_nonce_sha256"),
    }
    if checkpoint_commitments != EXPECTED_COMMITMENTS:
        raise RecoveryFatalError("Source checkpoint commitments drifted")
    if (
        source_payload.get("authorization_sha256") != EXPECTED_ORIGINAL_HASHES["authorization"]
        or source_payload.get("marker_sha256") != EXPECTED_ORIGINAL_HASHES["marker"]
        or source_payload.get("tokenizer_sha256") != EXPECTED_ORIGINAL_HASHES["tokenizer"]
        or source_payload.get("parent_pid") != 677904
    ):
        raise RecoveryFatalError("Source checkpoint original lineage drifted")
    run_context = source_payload.get("run_context")
    if not isinstance(run_context, dict) or (
        run_context.get("input_bindings") != original_bindings
        or run_context.get("authorization_sha256")
        != EXPECTED_ORIGINAL_HASHES["authorization"]
        or run_context.get("marker_sha256") != EXPECTED_ORIGINAL_HASHES["marker"]
        or run_context.get("handoff_nonce_sha256")
        != EXPECTED_COMMITMENTS["old_handoff_nonce_sha256"]
    ):
        raise RecoveryFatalError("Source checkpoint run context drifted")
    b1.assert_report_has_no_quality_metrics(run_context)
    return original_contract, original_bindings, source_payload


def _validate_recovery_contract(contract: dict[str, Any]) -> None:
    authority = contract.get("authority")
    source_state = contract.get("origin_checkpoint_state")
    continuation = contract.get("continuation")
    raw_scope = contract.get("raw_scope")
    raw_passes = contract.get("raw_pass_accounting")
    process_lineage = contract.get("process_lineage")
    resources = contract.get("resource_gates")
    lease = contract.get("one_shot_recovery_lease")
    nonce = contract.get("handoff_nonce")
    artifacts = contract.get("artifact_policy")
    forbidden = contract.get("forbidden")
    required_success = contract.get("required_success_invariants")
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("loop_id") != "loop166_code_section_foundation"
        or contract.get("claim_scope")
        != "local_train_only_one_fail_closed_step4096_recovery_continuation_not_model_quality"
        or contract.get("status") != "source_closure_complete"
        or not isinstance(authority, dict)
        or authority.get("user_directed_local_custody") is not True
        or authority.get("public_key_required") is not False
        or authority.get("external_signer_required") is not False
        or authority.get("val_test_or_full_authority") is not False
        or authority.get("promotion_authority") is not False
        or not isinstance(source_state, dict)
        or any(source_state.get(key) != value for key, value in EXPECTED_ORIGINAL_STATE.items())
        or not math.isclose(
            float(source_state.get("checkpoint_cumulative_wall_seconds", math.nan)),
            SOURCE_CUMULATIVE_WALL_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or source_state.get("exact_tensor_optimizer_scaler_rng_and_synthetic_logit_audit_required")
        is not True
        or not isinstance(continuation, dict)
        or continuation.get("resume_cursor") != 16_384
        or continuation.get("first_recovery_sequence_cursor") != 16_384
        or continuation.get("final_sequence_cursor") != 115_072
        or continuation.get("fresh_model_or_optimizer_initialization_allowed") is not False
        or continuation.get("fresh_tokenizer_fit_allowed") is not False
        or continuation.get("prefix_rewind_repeat_or_retrain_allowed") is not False
        or continuation.get("remaining_sequence_repeat_skip_or_padding_allowed") is not False
        or continuation.get("recipe_microbatch_precision_optimizer_or_schedule_change_allowed")
        is not False
        or not isinstance(raw_scope, dict)
        or raw_scope.get("outer_holdout_fold") != 0
        or raw_scope.get("outer_holdout_raw_opens_allowed") != 0
        or raw_scope.get("outer_holdout_raw_bytes_allowed") != 0
        or raw_scope.get("outer_fit_folds") != [1, 2, 3, 4]
        or raw_scope.get("recovery_full_fit_rebuild_passes_allowed") != 1
        or raw_scope.get("same_lease_rescan_or_retry_allowed") is not False
        or raw_scope.get("final_verify_raw_access_allowed") is not False
        or not isinstance(raw_passes, dict)
        or raw_passes.get("confirmed_full_fit_raw_passes_before_recovery") != 1
        or raw_passes.get("maximum_additional_full_fit_raw_passes") != 1
        or raw_passes.get("maximum_total_full_fit_raw_passes") != 2
        or raw_passes.get("recovery_parent_or_preflight_raw_passes_allowed") != 0
        or raw_passes.get("recovery_worker_raw_passes_allowed") != 1
        or raw_passes.get("final_verifier_raw_passes_allowed") != 0
        or not isinstance(process_lineage, dict)
        or process_lineage.get("maximum_allowed_intermediate_processes") != 1
        or process_lineage.get("broad_ancestor_or_name_only_acceptance_allowed") is not False
        or process_lineage.get("child_default_or_self_reported_executable_allowed") is not False
        or not isinstance(resources, dict)
        or not math.isclose(
            float(resources.get("checkpoint_cumulative_wall_seconds", math.nan)),
            SOURCE_CUMULATIVE_WALL_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or resources.get("conservative_failed_handoff_debit_seconds")
        != CONSERVATIVE_FAILED_HANDOFF_DEBIT_SECONDS
        or not math.isclose(
            float(resources.get("recovery_budget_base_seconds", math.nan)),
            RECOVERY_BUDGET_BASE_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or resources.get("maximum_cumulative_wall_seconds")
        != MAXIMUM_CUMULATIVE_WALL_SECONDS
        or resources.get("budget_reset_allowed") is not False
        or resources.get("any_gate_failure_action") != FAILURE_DECISION
        or not isinstance(lease, dict)
        or lease.get("lease_id") != LEASE_ID
        or lease.get("consumption_marker")
        != str(DEFAULT_MARKER.relative_to(PROJECT_ROOT)).replace("\\", "/")
        or lease.get("consume_before_any_raw_open") is not True
        or lease.get("overwrite_delete_rename_or_reuse_marker_allowed") is not False
        or lease.get("failed_attempt_consumes_lease") is not True
        or lease.get("same_lease_process_or_raw_retry_allowed") is not False
        or not isinstance(nonce, dict)
        or nonce.get("minimum_entropy_bits") != 256
        or nonce.get("direct_cli_value_allowed") is not False
        or nonce.get("plaintext_persistence_allowed") is not False
        or nonce.get("final_verify_must_prove_nonce_possession") is not True
        or not isinstance(artifacts, dict)
        or artifacts.get("old_artifact_overwrite_delete_or_reuse_allowed") is not False
        or artifacts.get("origin_tokenizer_read_only") is not True
        or artifacts.get("origin_checkpoint_read_only") is not True
        or not isinstance(forbidden, dict)
        or forbidden.get("outer_holdout_raw_access") is not True
        or forbidden.get("val_test_or_full_access") is not True
        or forbidden.get("quality_metrics") is not True
        or forbidden.get("threshold_operations") is not True
        or forbidden.get("original_artifact_mutation") is not True
        or forbidden.get("public_key_dependency") is not True
        or any(value is not True for value in forbidden.values())
        or not isinstance(required_success, dict)
        or required_success.get("origin_artifact_closure_unchanged") is not True
        or required_success.get("outer_holdout_raw_opens") != 0
        or required_success.get("outer_holdout_raw_bytes") != 0
        or required_success.get("recovery_full_fit_raw_passes") != 1
        or required_success.get("total_full_fit_raw_passes") != 2
        or required_success.get("raw_compact_and_shuffle_commitments_match") is not True
        or required_success.get("resume_cursor_started_at") != 16_384
        or required_success.get("final_cursor") != 115_072
        or required_success.get("final_optimizer_steps") != 28_768
        or required_success.get("dropped_content_tokens") != 0
        or required_success.get("dropped_original_bytes") != 0
        or required_success.get("overlength_windows_excluded") != 0
        or required_success.get("quality_metrics_computed") is not False
        or required_success.get("threshold_operations_performed") is not False
        or required_success.get("final_verify_raw_access") is not False
        or required_success.get("pass_decision") != PASS_DECISION
        or required_success.get("failure_decision") != FAILURE_DECISION
    ):
        raise RecoveryFatalError("Frozen Phase B1 recovery contract drifted")
    if contract.get("decision") != "allow_one_step4096_recovery_continuation":
        raise RecoveryFatalError("Phase B1 recovery contract decision remains pending")
    closure = contract.get("recovery_execution_closure")
    if not isinstance(closure, dict):
        raise RecoveryFatalError("Recovery execution closure is missing")
    expected_closure_paths = {
        "controller": Path(__file__).resolve(strict=True),
        "windows_process_lineage": (
            SRC_DIR / "loop166" / "windows_process_lineage.py"
        ).resolve(strict=True),
        "contract_tests": (
            PROJECT_ROOT / "tests" / "test_loop166_phase_b1_step4096_recovery.py"
        ).resolve(strict=True),
        "lineage_tests": (
            PROJECT_ROOT / "tests" / "test_loop166_windows_process_lineage.py"
        ).resolve(strict=True),
    }
    for name, expected_path in expected_closure_paths.items():
        observed_path, _observed_sha = _binding_path(closure.get(name), name)
        if observed_path != expected_path:
            raise RecoveryFatalError(f"Recovery execution path drifted: {name}")
    if (
        closure.get("run_allowed_with_pending_binding") is not False
        or closure.get("binding_drift_action")
        != "fail_before_recovery_lease_consumption_or_raw_open"
    ):
        raise RecoveryFatalError("Recovery execution closure remains pending")
    if contract.get("ready_for") != {
        "recovery_execution": True,
        "final_verify_execution": True,
        "five_fold_oof": False,
        "val_test_or_full": False,
        "promotion": False,
        "blocked_by": [],
    }:
        raise RecoveryFatalError("Recovery contract readiness drifted")


def _validate_recovery_authorization(
    args: argparse.Namespace,
    contract: dict[str, Any],
    contract_sha256: str,
    authorization: dict[str, Any],
) -> dict[str, str]:
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("loop_id") != "loop166_code_section_foundation"
        or authorization.get("claim_scope")
        != "local_train_only_one_fail_closed_step4096_recovery_run_authorization"
        or authorization.get("status") != "granted_source_closure"
        or authorization.get("authorization_granted") is not True
        or authorization.get("decision") != "authorize_one_step4096_recovery_continuation"
        or authorization.get("research_champion") != "Loop151"
    ):
        raise RecoveryFatalError("Phase B1 recovery authorization is absent or pending")
    authority = authorization.get("authority")
    if not isinstance(authority, dict) or (
        authority.get("user_directed_local_custody") is not True
        or authority.get("user_direction_received") is not True
        or authority.get("run_authority_withheld_until_source_closure") is not False
        or authority.get("public_key_required") is not False
        or authority.get("external_signer_required") is not False
        or authority.get("a2_or_a3_authority") is not False
        or authority.get("val_test_or_full_authority") is not False
        or authority.get("promotion_authority") is not False
    ):
        raise RecoveryFatalError("Phase B1 recovery authority drifted")
    runtime = _validate_runtime(authorization.get("runtime"))
    if authorization.get("canonical_paths") != _expected_paths(args):
        raise RecoveryFatalError("Phase B1 recovery canonical paths drifted")
    invocation = authorization.get("canonical_invocation")
    parent_invocation = invocation.get("parent") if isinstance(invocation, dict) else None
    internal_modes = (
        invocation.get("internal_modes") if isinstance(invocation, dict) else None
    )
    if not isinstance(invocation, dict) or (
        not isinstance(parent_invocation, dict)
        or parent_invocation.get("mode") != "parent"
        or parent_invocation.get("argv") != list(canonical_command("parent", args))
        or not isinstance(internal_modes, dict)
        or internal_modes.get("allowed_flags")
        != ["--recovery-worker", "--final-verify-worker"]
        or internal_modes.get("direct_user_invocation_allowed") is not False
        or internal_modes.get("must_inherit_new_authorization_marker_and_nonce")
        is not True
        or internal_modes.get("must_revalidate_runtime_paths_sources_lineage_and_lease")
        is not True
        or invocation.get("unlisted_arguments_allowed") is not False
        or invocation.get("path_alias_or_symlink_allowed") is not False
        or invocation.get("working_directory") != str(PROJECT_ROOT.resolve(strict=True))
    ):
        raise RecoveryFatalError("Phase B1 recovery canonical invocation drifted")
    lease = authorization.get("one_shot_recovery_lease")
    if not isinstance(lease, dict) or (
        lease.get("lease_id") != LEASE_ID
        or lease.get("consumption_marker_absolute") != str(DEFAULT_MARKER.absolute())
        or lease.get("marker_must_not_exist_before_recovery_start") is not True
        or lease.get("consume_before_any_raw_open") is not True
        or lease.get("overwrite_delete_rename_or_reuse_marker_allowed") is not False
        or lease.get("failed_attempt_consumes_lease") is not True
        or lease.get("same_lease_retry_allowed") is not False
        or lease.get("all_modes_must_bind_same_marker_content_sha256") is not True
    ):
        raise RecoveryFatalError("Phase B1 recovery one-shot lease drifted")
    nonce = authorization.get("handoff_nonce")
    if not isinstance(nonce, dict) or (
        nonce.get("minimum_entropy_bits") != 256
        or nonce.get("direct_cli_value_allowed") is not False
        or nonce.get("plaintext_persistence_allowed") is not False
        or nonce.get("plaintext_child_handoff_channel") != "environment_only"
        or nonce.get("final_verify_must_prove_nonce_possession") is not True
    ):
        raise RecoveryFatalError("Phase B1 recovery nonce protocol drifted")
    origin_authority = authorization.get("origin_checkpoint_authority")
    if not isinstance(origin_authority, dict) or (
        origin_authority.get("load_policy") != "weights_only"
        or origin_authority.get("completed_optimizer_steps") != 4096
        or origin_authority.get("next_permutation_cursor") != 16_384
        or origin_authority.get("origin_nonce_plaintext_or_possession_required") is not False
        or origin_authority.get("raw_commitment_sha256")
        != EXPECTED_COMMITMENTS["outer_fit_corpus_commitment_sha256"]
        or origin_authority.get("compact_commitment_sha256")
        != EXPECTED_COMMITMENTS["compact_corpus_commitment_sha256"]
        or origin_authority.get("shuffle_commitment_sha256")
        != EXPECTED_COMMITMENTS["shuffle_commitment_sha256"]
    ):
        raise RecoveryFatalError("Original nonce recovery boundary drifted")
    bindings = authorization.get("bindings")
    expected_binding_paths = {
        "incident": DEFAULT_INCIDENT,
        "original_authorization": ORIGINAL_AUTHORIZATION,
        "original_contract": ORIGINAL_CONTRACT,
        "original_controller": ORIGINAL_CONTROLLER,
        "original_tests": ORIGINAL_TESTS,
        "original_consumed_marker": ORIGINAL_MARKER,
        "original_tokenizer": args.source_tokenizer,
        "origin_checkpoint": args.source_checkpoint,
        "recovery_contract": args.contract,
        "recovery_controller": Path(__file__),
        "windows_process_lineage": SRC_DIR
        / "loop166"
        / "windows_process_lineage.py",
        "recovery_tests": PROJECT_ROOT
        / "tests"
        / "test_loop166_phase_b1_step4096_recovery.py",
        "lineage_tests": PROJECT_ROOT / "tests" / "test_loop166_windows_process_lineage.py",
    }
    if not isinstance(bindings, dict) or set(bindings) != set(expected_binding_paths):
        raise RecoveryFatalError("Recovery authorization source bindings are missing")
    observed_binding_hashes: dict[str, str] = {}
    for name, expected_path in expected_binding_paths.items():
        observed_path, observed_sha = _binding_path(bindings.get(name), name)
        if observed_path != expected_path.resolve(strict=True):
            raise RecoveryFatalError(f"Recovery authorization path drifted: {name}")
        observed_binding_hashes[name] = observed_sha
    if observed_binding_hashes["recovery_contract"] != contract_sha256:
        raise RecoveryFatalError("Recovery authorization contract SHA drifted")
    closure = authorization.get("runtime_source_closure")
    if not isinstance(closure, dict) or (
        closure.get("status") != "complete"
        or closure.get("pending_bindings") != []
        or closure.get("all_sources_must_match_before_authorization") is not True
        or closure.get("execution_with_pending_value_allowed") is not False
    ):
        raise RecoveryFatalError("Recovery authorization source closure remains pending")
    raw_scope = authorization.get("raw_scope")
    resources = authorization.get("resource_authority")
    artifact_authority = authorization.get("artifact_authority")
    allowed_modes = authorization.get("allowed_modes")
    forbidden = authorization.get("forbidden")
    if (
        not isinstance(raw_scope, dict)
        or raw_scope.get("outer_holdout_raw_opens_allowed") != 0
        or raw_scope.get("outer_holdout_raw_bytes_allowed") != 0
        or raw_scope.get("outer_fit_folds") != [1, 2, 3, 4]
        or raw_scope.get("recovery_full_fit_rebuild_passes_allowed") != 1
        or raw_scope.get("maximum_total_full_fit_raw_passes") != 2
        or raw_scope.get("same_lease_rescan_allowed") is not False
        or raw_scope.get("final_verify_raw_access_allowed") is not False
        or not isinstance(resources, dict)
        or not math.isclose(
            float(resources.get("recovery_budget_base_seconds", math.nan)),
            RECOVERY_BUDGET_BASE_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or resources.get("maximum_cumulative_wall_seconds")
        != MAXIMUM_CUMULATIVE_WALL_SECONDS
        or resources.get("budget_reset_allowed") is not False
        or not isinstance(artifact_authority, dict)
        or artifact_authority.get("old_artifact_overwrite_delete_or_reuse_allowed")
        is not False
        or artifact_authority.get("new_checkpoint_path_only")
        != str(DEFAULT_CHECKPOINT_OUTPUT.relative_to(PROJECT_ROOT)).replace("\\", "/")
        or artifact_authority.get("new_report_path_only")
        != str(DEFAULT_REPORT_OUTPUT.relative_to(PROJECT_ROOT)).replace("\\", "/")
        or artifact_authority.get("new_receipt_path_only")
        != str(DEFAULT_FINAL_RECEIPT.relative_to(PROJECT_ROOT)).replace("\\", "/")
        or not isinstance(allowed_modes, dict)
        or allowed_modes.get("parent", {}).get("may_open_raw") is not False
        or allowed_modes.get("recovery_worker", {}).get("may_open_outer_fit_raw_once")
        is not True
        or allowed_modes.get("recovery_worker", {}).get("may_resume_from_cursor")
        != 16_384
        or allowed_modes.get("final_verify_worker", {}).get("may_open_raw") is not False
        or not isinstance(forbidden, dict)
        or any(value is not True for value in forbidden.values())
    ):
        raise RecoveryFatalError("Recovery authorization scope drifted")
    prerequisites = authorization.get("grant_prerequisites")
    if not isinstance(prerequisites, dict) or not prerequisites or any(
        value is not True for value in prerequisites.values()
    ):
        raise RecoveryFatalError("Recovery authorization prerequisites are incomplete")
    ready = authorization.get("ready_for")
    if ready != {
        "recovery_execution": True,
        "final_verify_execution": True,
        "five_fold_oof": False,
        "val_test_or_full": False,
        "promotion": False,
        "blocked_by": [],
    }:
        raise RecoveryFatalError("Recovery authorization readiness drifted")
    return runtime


def _validate_mode_outputs(args: argparse.Namespace, mode: str) -> None:
    if mode not in {"parent", "recovery", "final_verify"}:
        raise ValueError(f"Unknown recovery preflight mode: {mode}")
    expected_presence = {
        "parent": {
            DEFAULT_MARKER: False,
            DEFAULT_FINAL_RECEIPT: False,
            args.checkpoint_output: False,
            args.report_output: False,
        },
        "recovery": {
            DEFAULT_MARKER: True,
            DEFAULT_FINAL_RECEIPT: False,
            args.checkpoint_output: False,
            args.report_output: False,
        },
        "final_verify": {
            DEFAULT_MARKER: True,
            DEFAULT_FINAL_RECEIPT: False,
            args.checkpoint_output: True,
            args.report_output: False,
        },
    }[mode]
    for path, must_exist in expected_presence.items():
        if path.exists() is not must_exist:
            expected = "exist" if must_exist else "be absent"
            raise RecoveryFatalError(
                f"Recovery {mode} preflight requires {path} to {expected}"
            )


def validate_control_preflight(
    args: argparse.Namespace,
    *,
    mode: str,
) -> RecoveryControlClosure:
    contract, contract_raw = _read_json(args.contract, "Phase B1 recovery contract")
    _validate_recovery_contract(contract)
    authorization, authorization_raw = _read_json(
        args.authorization, "Phase B1 recovery authorization"
    )
    runtime = _validate_recovery_authorization(
        args,
        contract,
        _sha256(contract_raw),
        authorization,
    )
    _validate_mode_outputs(args, mode)
    immutable_paths = {
        ORIGINAL_MARKER.resolve(strict=True),
        ORIGINAL_AUTHORIZATION.resolve(strict=True),
        ORIGINAL_CONTRACT.resolve(strict=True),
        ORIGINAL_CONTROLLER.resolve(strict=True),
        ORIGINAL_TESTS.resolve(strict=True),
        args.source_checkpoint.resolve(strict=True),
        args.source_tokenizer.resolve(strict=True),
    }
    mutable_paths = {
        DEFAULT_MARKER.absolute(),
        DEFAULT_FINAL_RECEIPT.absolute(),
        args.checkpoint_output.absolute(),
        args.report_output.absolute(),
    }
    if immutable_paths & mutable_paths or len(mutable_paths) != 4:
        raise RecoveryFatalError("Recovery outputs overlap immutable original artifacts")
    if Path.cwd().resolve(strict=True) != PROJECT_ROOT.resolve(strict=True):
        raise RecoveryFatalError("Recovery working directory is not canonical")
    return RecoveryControlClosure(
        contract=contract,
        contract_sha256=_sha256(contract_raw),
        authorization=authorization,
        authorization_sha256=_sha256(authorization_raw),
        runtime=runtime,
    )


def complete_preflight(
    args: argparse.Namespace,
    control: RecoveryControlClosure,
) -> RecoveryClosure:
    original_contract, original_bindings, source_payload = _validate_original_evidence(
        args,
        control.contract,
    )
    b1.guard_non_cuda_phase(
        original_contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=RECOVERY_BUDGET_BASE_SECONDS,
    )
    return RecoveryClosure(
        contract=control.contract,
        contract_sha256=control.contract_sha256,
        authorization=control.authorization,
        authorization_sha256=control.authorization_sha256,
        runtime=control.runtime,
        original_contract=original_contract,
        original_bindings=original_bindings,
        source_payload=source_payload,
    )


def validate_preflight(
    args: argparse.Namespace,
    *,
    mode: str,
) -> RecoveryClosure:
    return complete_preflight(args, validate_control_preflight(args, mode=mode))


def _assert_control_closure_unchanged(
    args: argparse.Namespace,
    closure: RecoveryControlClosure,
) -> None:
    contract, contract_raw = _read_json(args.contract, "Phase B1 recovery contract")
    _validate_recovery_contract(contract)
    authorization, authorization_raw = _read_json(
        args.authorization, "Phase B1 recovery authorization"
    )
    runtime = _validate_recovery_authorization(
        args,
        contract,
        _sha256(contract_raw),
        authorization,
    )
    if (
        _sha256(contract_raw) != closure.contract_sha256
        or _sha256(authorization_raw) != closure.authorization_sha256
        or runtime != closure.runtime
    ):
        raise RecoveryFatalError("Recovery control closure changed after preflight")


def _guard_before_marker_consumption(
    args: argparse.Namespace,
    closure: RecoveryClosure,
    *,
    parent_started: float,
) -> float:
    _assert_control_closure_unchanged(args, closure)
    _assert_immutable_originals(args)
    _validate_mode_outputs(args, "parent")
    cumulative_wall = RECOVERY_BUDGET_BASE_SECONDS + time.perf_counter() - parent_started
    if (
        not math.isfinite(cumulative_wall)
        or cumulative_wall < RECOVERY_BUDGET_BASE_SECONDS
        or cumulative_wall >= MAXIMUM_CUMULATIVE_WALL_SECONDS
    ):
        raise RecoveryFatalError("Recovery budget expired before marker consumption")
    resource_gates = closure.original_contract["resource_gates"]
    free_disk = b1._free_disk_bytes(args.checkpoint_output.parent)
    if free_disk < resource_gates["minimum_free_disk_bytes_before_raw_open"]:
        raise RecoveryFatalError("Insufficient free disk before recovery marker consumption")
    b1.guard_non_cuda_phase(
        closure.original_contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=cumulative_wall,
    )
    return cumulative_wall


def _guard_before_recovery_raw_access(
    args: argparse.Namespace,
    closure: RecoveryClosure,
    *,
    cumulative_wall_seconds: float,
    state: dict[str, Any],
) -> dict[str, int]:
    _assert_control_closure_unchanged(args, closure)
    _assert_immutable_originals(args)
    return b1.guard_non_cuda_phase(
        closure.original_contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=cumulative_wall_seconds,
        state=state,
    )


def _write_exclusive_json(path: Path, payload: dict[str, Any], context: str) -> str:
    b1.assert_report_has_no_quality_metrics(payload)
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise RecoveryFatalError(f"{context} already exists") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    b1._fsync_parent_directory(path.parent)
    return _sha256(raw)


def consume_recovery_authorization(
    closure: RecoveryClosure,
    *,
    cumulative_wall_seconds_at_consumption: float,
    args: argparse.Namespace,
) -> b1.RunHandoff:
    if (
        not math.isfinite(cumulative_wall_seconds_at_consumption)
        or not RECOVERY_BUDGET_BASE_SECONDS
        <= cumulative_wall_seconds_at_consumption
        < MAXIMUM_CUMULATIVE_WALL_SECONDS
    ):
        raise RecoveryFatalError("Recovery marker consumption is outside the frozen budget")
    nonce = secrets.token_hex(32)
    nonce_sha256 = _sha256(nonce.encode("ascii"))
    if nonce_sha256 == EXPECTED_COMMITMENTS["old_handoff_nonce_sha256"]:
        raise RecoveryFatalError("Recovery nonce unexpectedly matches the original commitment")
    parent_argv_sha = _argv_sha256(canonical_command("parent", args))
    payload = {
        "schema": MARKER_SCHEMA,
        "loop_id": "loop166_code_section_foundation",
        "lease_id": LEASE_ID,
        "authorization_sha256": closure.authorization_sha256,
        "contract_sha256": closure.contract_sha256,
        "new_handoff_nonce_sha256": nonce_sha256,
        "recovery_parent_pid": os.getpid(),
        "canonical_parent_argv_sha256": parent_argv_sha,
        "canonical_recovery_argv_sha256": _argv_sha256(
            canonical_command("recovery", args)
        ),
        "canonical_final_verify_argv_sha256": _argv_sha256(
            canonical_command("final_verify", args)
        ),
        "source_checkpoint_sha256": EXPECTED_ORIGINAL_HASHES["checkpoint"],
        "source_checkpoint_optimizer_step": 4096,
        "source_checkpoint_cursor": 16_384,
        "original_marker_sha256": EXPECTED_ORIGINAL_HASHES["marker"],
        "old_handoff_nonce_sha256": EXPECTED_COMMITMENTS["old_handoff_nonce_sha256"],
        "old_handoff_nonce_possession": False,
        "recovery_budget_base_seconds": RECOVERY_BUDGET_BASE_SECONDS,
        "cumulative_wall_seconds_at_consumption": cumulative_wall_seconds_at_consumption,
        "parent_elapsed_at_consumption_seconds": (
            cumulative_wall_seconds_at_consumption - RECOVERY_BUDGET_BASE_SECONDS
        ),
        "status": "recovery_consumed_before_additional_raw_access",
    }
    marker_sha = _write_exclusive_json(DEFAULT_MARKER, payload, "Recovery marker")
    return b1.RunHandoff(
        authorization_sha256=closure.authorization_sha256,
        marker_sha256=marker_sha,
        handoff_nonce=nonce,
        parent_pid=os.getpid(),
        canonical_parent_argv_sha256=parent_argv_sha,
    )


def _recovery_environment(
    handoff: b1.RunHandoff,
    *,
    mode: str,
    worker_pid: int,
    cumulative_base_seconds: float,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "AXON_B1_RECOVERY_MODE": mode,
            "AXON_B1_RECOVERY_NONCE": handoff.handoff_nonce,
            "AXON_B1_RECOVERY_AUTHORIZATION_SHA256": handoff.authorization_sha256,
            "AXON_B1_RECOVERY_MARKER_SHA256": handoff.marker_sha256,
            "AXON_B1_RECOVERY_PARENT_PID": str(handoff.parent_pid),
            "AXON_B1_RECOVERY_PARENT_ARGV_SHA256": handoff.canonical_parent_argv_sha256,
            "AXON_B1_RECOVERY_WORKER_PID": str(worker_pid),
            "AXON_B1_RECOVERY_CUMULATIVE_BASE_SECONDS": repr(cumulative_base_seconds),
        }
    )
    return environment


def _handoff_from_environment(expected_mode: str) -> tuple[b1.RunHandoff, int, float]:
    if os.environ.get("AXON_B1_RECOVERY_MODE") != expected_mode:
        raise RecoveryFatalError("Recovery worker mode handoff drifted")
    nonce = os.environ.get("AXON_B1_RECOVERY_NONCE", "")
    if len(nonce) != 64 or not set(nonce) <= LOWER_HEX:
        raise RecoveryFatalError("Recovery worker has no valid in-memory nonce")
    try:
        parent_pid = int(os.environ["AXON_B1_RECOVERY_PARENT_PID"])
        worker_pid = int(os.environ.get("AXON_B1_RECOVERY_WORKER_PID", "0"))
        cumulative_base = float(
            os.environ["AXON_B1_RECOVERY_CUMULATIVE_BASE_SECONDS"]
        )
    except (KeyError, ValueError) as exc:
        raise RecoveryFatalError("Recovery worker numeric handoff is invalid") from exc
    handoff = b1.RunHandoff(
        authorization_sha256=os.environ.get(
            "AXON_B1_RECOVERY_AUTHORIZATION_SHA256", ""
        ),
        marker_sha256=os.environ.get("AXON_B1_RECOVERY_MARKER_SHA256", ""),
        handoff_nonce=nonce,
        parent_pid=parent_pid,
        canonical_parent_argv_sha256=os.environ.get(
            "AXON_B1_RECOVERY_PARENT_ARGV_SHA256", ""
        ),
    )
    if not all(
        _is_sha256(value)
        for value in (
            handoff.authorization_sha256,
            handoff.marker_sha256,
            handoff.canonical_parent_argv_sha256,
        )
    ):
        raise RecoveryFatalError("Recovery commitment handoff is invalid")
    if (
        not math.isfinite(cumulative_base)
        or cumulative_base < RECOVERY_BUDGET_BASE_SECONDS
        or cumulative_base >= MAXIMUM_CUMULATIVE_WALL_SECONDS
    ):
        raise RecoveryFatalError("Recovery cumulative budget handoff is invalid")
    return handoff, worker_pid, cumulative_base


def _validate_recovery_marker(
    handoff: b1.RunHandoff,
    closure: RecoveryControlClosure,
    args: argparse.Namespace,
) -> dict[str, Any]:
    marker, raw = _read_json(DEFAULT_MARKER, "Phase B1 recovery marker")
    if _sha256(raw) != handoff.marker_sha256:
        raise RecoveryFatalError("Recovery marker SHA drifted")
    required = {
        "schema": MARKER_SCHEMA,
        "loop_id": "loop166_code_section_foundation",
        "lease_id": LEASE_ID,
        "authorization_sha256": handoff.authorization_sha256,
        "contract_sha256": closure.contract_sha256,
        "new_handoff_nonce_sha256": handoff.handoff_nonce_sha256,
        "recovery_parent_pid": handoff.parent_pid,
        "canonical_parent_argv_sha256": handoff.canonical_parent_argv_sha256,
        "canonical_recovery_argv_sha256": _argv_sha256(
            canonical_command("recovery", args)
        ),
        "canonical_final_verify_argv_sha256": _argv_sha256(
            canonical_command("final_verify", args)
        ),
        "source_checkpoint_sha256": EXPECTED_ORIGINAL_HASHES["checkpoint"],
        "source_checkpoint_optimizer_step": 4096,
        "source_checkpoint_cursor": 16_384,
        "original_marker_sha256": EXPECTED_ORIGINAL_HASHES["marker"],
        "old_handoff_nonce_sha256": EXPECTED_COMMITMENTS["old_handoff_nonce_sha256"],
        "old_handoff_nonce_possession": False,
        "recovery_budget_base_seconds": RECOVERY_BUDGET_BASE_SECONDS,
        "status": "recovery_consumed_before_additional_raw_access",
    }
    if any(marker.get(key) != value for key, value in required.items()):
        raise RecoveryFatalError("Recovery marker content drifted")
    elapsed = marker.get("parent_elapsed_at_consumption_seconds")
    cumulative_at_consumption = marker.get("cumulative_wall_seconds_at_consumption")
    if (
        not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or elapsed < 0
        or not isinstance(cumulative_at_consumption, (int, float))
        or not math.isfinite(float(cumulative_at_consumption))
        or not RECOVERY_BUDGET_BASE_SECONDS
        <= float(cumulative_at_consumption)
        < MAXIMUM_CUMULATIVE_WALL_SECONDS
        or not math.isclose(
            float(cumulative_at_consumption),
            RECOVERY_BUDGET_BASE_SECONDS + float(elapsed),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise RecoveryFatalError("Recovery marker parent timing is invalid")
    if handoff.handoff_nonce.encode("ascii") in raw:
        raise RecoveryFatalError("Recovery marker persisted the nonce plaintext")
    return marker


def _validate_worker_lineage(
    expected_parent_pid: int,
    runtime: dict[str, str],
) -> dict[str, Any]:
    try:
        return validate_spawn_lineage(
            expected_parent_pid,
            launcher_executable=runtime["python_executable"],
            base_executable=runtime["base_executable"],
        )
    except ProcessLineageError as exc:
        raise RecoveryFatalError("Recovery process lineage is invalid") from exc


def _spawn_worker(
    mode: str,
    args: argparse.Namespace,
    handoff: b1.RunHandoff,
    closure: RecoveryClosure,
    *,
    worker_pid: int,
    cumulative_base_seconds: float,
) -> int:
    remaining = MAXIMUM_CUMULATIVE_WALL_SECONDS - cumulative_base_seconds
    if remaining <= 0:
        raise RecoveryFatalError("No cumulative wall-time budget remains for recovery")
    try:
        completed = subprocess.run(
            canonical_command(mode, args),
            cwd=PROJECT_ROOT,
            env=_recovery_environment(
                handoff,
                mode=mode,
                worker_pid=worker_pid,
                cumulative_base_seconds=cumulative_base_seconds,
            ),
            check=False,
            timeout=remaining,
        )
    except subprocess.TimeoutExpired as exc:
        raise RecoveryFatalError(f"Recovery {mode} worker exceeded the 8-hour cap") from exc
    # Child source closure is revalidated in the child; this parent-side read prevents silent auth swap.
    if _artifact_sha(args.authorization) != closure.authorization_sha256:
        raise RecoveryFatalError("Recovery authorization changed while a worker was running")
    return int(completed.returncode)


def _assert_immutable_originals(args: argparse.Namespace) -> None:
    checks = {
        DEFAULT_INCIDENT: EXPECTED_ORIGINAL_HASHES["incident"],
        ORIGINAL_MARKER: EXPECTED_ORIGINAL_HASHES["marker"],
        ORIGINAL_AUTHORIZATION: EXPECTED_ORIGINAL_HASHES["authorization"],
        ORIGINAL_CONTRACT: EXPECTED_ORIGINAL_HASHES["contract"],
        ORIGINAL_CONTROLLER: EXPECTED_ORIGINAL_HASHES["controller"],
        ORIGINAL_TESTS: EXPECTED_ORIGINAL_HASHES["tests"],
        args.source_checkpoint: EXPECTED_ORIGINAL_HASHES["checkpoint"],
        args.source_tokenizer: EXPECTED_ORIGINAL_HASHES["tokenizer"],
    }
    for path, expected in checks.items():
        if _artifact_sha(path) != expected:
            raise RecoveryFatalError(f"Immutable original artifact changed: {path}")
    if b1.DEFAULT_REPORT.exists() or b1.DEFAULT_FINAL_VERIFY_RECEIPT.exists():
        raise RecoveryFatalError("Original failed B1 outputs were unexpectedly backfilled")


def run_parent(args: argparse.Namespace) -> int:
    parent_started = time.perf_counter()
    closure = validate_preflight(args, mode="parent")
    cumulative_at_consumption = _guard_before_marker_consumption(
        args,
        closure,
        parent_started=parent_started,
    )
    handoff = consume_recovery_authorization(
        closure,
        cumulative_wall_seconds_at_consumption=cumulative_at_consumption,
        args=args,
    )
    _validate_recovery_marker(handoff, closure, args)
    cumulative_base = RECOVERY_BUDGET_BASE_SECONDS + (
        time.perf_counter() - parent_started
    )
    return_code = _spawn_worker(
        "recovery",
        args,
        handoff,
        closure,
        worker_pid=0,
        cumulative_base_seconds=cumulative_base,
    )
    if return_code != 0:
        raise RecoveryFatalError(
            f"Bound Phase B1 recovery worker failed with exit code {return_code}"
        )
    parent_cumulative_wall = RECOVERY_BUDGET_BASE_SECONDS + (
        time.perf_counter() - parent_started
    )
    b1.guard_non_cuda_phase(
        closure.original_contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=parent_cumulative_wall,
    )
    _assert_control_closure_unchanged(args, closure)
    _assert_immutable_originals(args)
    _validate_parent_report_closure(args, closure, handoff, parent_cumulative_wall)
    return 0


def _recovery_run_context(
    closure: RecoveryClosure,
    handoff: b1.RunHandoff,
    args: argparse.Namespace,
    lineage: dict[str, Any],
    *,
    recovery_scan_accounting: dict[str, Any],
    recovery_compact_accounting: dict[str, Any],
) -> dict[str, Any]:
    original = closure.source_payload["run_context"]
    scan_accounting = dict(recovery_scan_accounting)
    compact_accounting = dict(recovery_compact_accounting)
    return {
        "prepared_sequence_count": EXPECTED_ORIGINAL_STATE["prepared_sequence_count"],
        "total_optimizer_steps": EXPECTED_ORIGINAL_STATE["total_optimizer_steps"],
        "original_run_context": original,
        "original_input_bindings": closure.original_bindings,
        "source_checkpoint_sha256": EXPECTED_ORIGINAL_HASHES["checkpoint"],
        "source_checkpoint_cursor": EXPECTED_ORIGINAL_STATE["next_permutation_cursor"],
        "source_checkpoint_optimizer_step": EXPECTED_ORIGINAL_STATE[
            "completed_optimizer_steps"
        ],
        "incident_sha256": EXPECTED_ORIGINAL_HASHES["incident"],
        "original_authorization_sha256": EXPECTED_ORIGINAL_HASHES["authorization"],
        "original_marker_sha256": EXPECTED_ORIGINAL_HASHES["marker"],
        "old_handoff_nonce_sha256": EXPECTED_COMMITMENTS["old_handoff_nonce_sha256"],
        "old_handoff_nonce_possession": False,
        "recovery_contract_sha256": closure.contract_sha256,
        "recovery_authorization_sha256": handoff.authorization_sha256,
        "recovery_marker_sha256": handoff.marker_sha256,
        "new_handoff_nonce_sha256": handoff.handoff_nonce_sha256,
        "canonical_recovery_argv_sha256": _argv_sha256(
            canonical_command("recovery", args)
        ),
        "canonical_final_verify_argv_sha256": _argv_sha256(
            canonical_command("final_verify", args)
        ),
        "recovery_parent_pid": handoff.parent_pid,
        "recovery_worker_pid": os.getpid(),
        "recovery_worker_lineage": lineage,
        "recovery_scan_accounting": scan_accounting,
        "recovery_scan_accounting_sha256": _canonical_json_sha256(scan_accounting),
        "recovery_compact_accounting": compact_accounting,
        "recovery_compact_accounting_sha256": _canonical_json_sha256(
            compact_accounting
        ),
        "source_cumulative_wall_seconds": SOURCE_CUMULATIVE_WALL_SECONDS,
        "conservative_failed_handoff_debit_seconds": (
            CONSERVATIVE_FAILED_HANDOFF_DEBIT_SECONDS
        ),
        "recovery_budget_base_seconds": RECOVERY_BUDGET_BASE_SECONDS,
    }


def _validate_scan_and_compact_ledgers(
    scan_accounting: object,
    compact_accounting: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(scan_accounting, dict) or not isinstance(compact_accounting, dict):
        raise RecoveryFatalError("Recovery checkpoint is missing its scan ledger")
    if (
        scan_accounting.get("fit_metadata_rows") != 16_000
        or scan_accounting.get("fit_raw_open_attempts") != 15_988
        or scan_accounting.get("fit_raw_open_successes") != 15_988
        or scan_accounting.get("fit_raw_bytes_actually_read") != 19_239_582_561
        or scan_accounting.get("source_unavailable") != 12
        or scan_accounting.get("outer_holdout_raw_opens") != 0
        or scan_accounting.get("outer_holdout_raw_bytes") != 0
        or scan_accounting.get("outer_fit_corpus_commitment_sha256")
        != EXPECTED_COMMITMENTS["outer_fit_corpus_commitment_sha256"]
    ):
        raise RecoveryFatalError("Recovery checkpoint scan ledger drifted")
    if (
        compact_accounting.get("compact_corpus_commitment_sha256")
        != EXPECTED_COMMITMENTS["compact_corpus_commitment_sha256"]
        or compact_accounting.get("prepared_sequence_count") != 115_072
        or compact_accounting.get("original_window_bytes")
        != EXPECTED_TOTAL_ORIGINAL_BYTES
        or compact_accounting.get("prepared_original_bytes")
        != EXPECTED_TOTAL_ORIGINAL_BYTES
        or compact_accounting.get("dropped_content_tokens") != 0
        or compact_accounting.get("dropped_original_bytes") != 0
        or compact_accounting.get("overlength_windows_excluded") != 0
    ):
        raise RecoveryFatalError("Recovery checkpoint compact ledger drifted")
    b1.assert_report_has_no_quality_metrics(scan_accounting)
    b1.assert_report_has_no_quality_metrics(compact_accounting)
    return scan_accounting, compact_accounting


def _validate_final_checkpoint_commitments(
    payload: dict[str, Any],
    closure: RecoveryClosure,
) -> dict[str, Any]:
    run_context = payload.get("run_context")
    training_state = payload.get("training_state")
    if not isinstance(run_context, dict) or not isinstance(training_state, dict):
        raise RecoveryFatalError("Recovery final checkpoint audit context is missing")
    scan_accounting, compact_accounting = _validate_scan_and_compact_ledgers(
        run_context.get("recovery_scan_accounting"),
        run_context.get("recovery_compact_accounting"),
    )
    scan_sha256 = _canonical_json_sha256(scan_accounting)
    compact_sha256 = _canonical_json_sha256(compact_accounting)
    if (
        payload.get("tokenizer_sha256") != EXPECTED_ORIGINAL_HASHES["tokenizer"]
        or payload.get("outer_fit_corpus_commitment_sha256")
        != EXPECTED_COMMITMENTS["outer_fit_corpus_commitment_sha256"]
        or payload.get("compact_corpus_commitment_sha256")
        != EXPECTED_COMMITMENTS["compact_corpus_commitment_sha256"]
        or payload.get("shuffle_commitment_sha256")
        != EXPECTED_COMMITMENTS["shuffle_commitment_sha256"]
        or payload.get("permutation_prefix_original_bytes")
        != EXPECTED_TOTAL_ORIGINAL_BYTES
        or training_state.get("training_original_bytes")
        != EXPECTED_TOTAL_ORIGINAL_BYTES
        or payload.get("completed_optimizer_steps") != 28_768
        or payload.get("completed_sequence_count") != 115_072
        or payload.get("next_permutation_cursor") != 115_072
        or run_context.get("prepared_sequence_count") != 115_072
        or run_context.get("total_optimizer_steps") != 28_768
        or run_context.get("incident_sha256") != EXPECTED_ORIGINAL_HASHES["incident"]
        or run_context.get("original_run_context")
        != closure.source_payload.get("run_context")
        or run_context.get("original_input_bindings") != closure.original_bindings
        or run_context.get("recovery_scan_accounting_sha256") != scan_sha256
        or run_context.get("recovery_compact_accounting_sha256") != compact_sha256
    ):
        raise RecoveryFatalError("Recovery final checkpoint commitment drifted")
    b1.assert_report_has_no_quality_metrics(run_context)
    return {
        "tokenizer_sha256": EXPECTED_ORIGINAL_HASHES["tokenizer"],
        "outer_fit_corpus_commitment_sha256": EXPECTED_COMMITMENTS[
            "outer_fit_corpus_commitment_sha256"
        ],
        "compact_corpus_commitment_sha256": EXPECTED_COMMITMENTS[
            "compact_corpus_commitment_sha256"
        ],
        "shuffle_commitment_sha256": EXPECTED_COMMITMENTS[
            "shuffle_commitment_sha256"
        ],
        "total_original_bytes": EXPECTED_TOTAL_ORIGINAL_BYTES,
        "incident_sha256": EXPECTED_ORIGINAL_HASHES["incident"],
        "recovery_scan_accounting_sha256": scan_sha256,
        "recovery_compact_accounting_sha256": compact_sha256,
    }


def _validate_receipt_resource_state(
    value: object,
    contract: dict[str, Any],
) -> dict[str, int]:
    if not isinstance(value, dict):
        raise RecoveryFatalError("Final verifier resource state is missing")
    required = {
        "peak_process_rss_bytes",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "minimum_free_disk_bytes",
    }
    if set(value) != required or any(
        not isinstance(value.get(name), int)
        or isinstance(value.get(name), bool)
        or int(value[name]) < 0
        for name in required
    ):
        raise RecoveryFatalError("Final verifier resource state is invalid")
    normalized = {name: int(value[name]) for name in required}
    gates = contract["resource_gates"]
    if (
        normalized["peak_process_rss_bytes"]
        >= gates["maximum_process_rss_bytes_exclusive"]
        or normalized["peak_cuda_allocated_bytes"]
        >= gates["maximum_cuda_allocated_bytes_exclusive"]
        or normalized["peak_cuda_reserved_bytes"]
        >= gates["maximum_cuda_reserved_bytes_exclusive"]
        or normalized["minimum_free_disk_bytes"]
        < gates["minimum_free_disk_bytes_during_run"]
    ):
        raise RecoveryFatalError("Final verifier resource state crossed a frozen gate")
    return normalized


def _merge_final_verifier_resources(
    state: dict[str, Any],
    receipt: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    resources = _validate_receipt_resource_state(receipt.get("resource_state"), contract)
    state["peak_process_rss_bytes"] = max(
        int(state["peak_process_rss_bytes"]),
        resources["peak_process_rss_bytes"],
    )
    state["peak_cuda_allocated_bytes"] = max(
        int(state["peak_cuda_allocated_bytes"]),
        resources["peak_cuda_allocated_bytes"],
    )
    state["peak_cuda_reserved_bytes"] = max(
        int(state["peak_cuda_reserved_bytes"]),
        resources["peak_cuda_reserved_bytes"],
    )
    state["minimum_free_disk_bytes"] = min(
        int(state["minimum_free_disk_bytes"]),
        resources["minimum_free_disk_bytes"],
    )


def _load_final_receipt(
    args: argparse.Namespace,
    closure: RecoveryClosure,
    handoff: b1.RunHandoff,
    worker_pid: int,
    *,
    not_after_cumulative_wall_seconds: Optional[float] = None,
) -> dict[str, Any]:
    receipt, _raw = _read_json(DEFAULT_FINAL_RECEIPT, "recovery final receipt")
    verifier_pid = receipt.get("verifier_pid")
    cumulative_wall = receipt.get("cumulative_wall_seconds")
    checkpoint_cumulative_wall = receipt.get("checkpoint_cumulative_wall_seconds")
    upper_wall = (
        MAXIMUM_CUMULATIVE_WALL_SECONDS
        if not_after_cumulative_wall_seconds is None
        else min(
            MAXIMUM_CUMULATIVE_WALL_SECONDS,
            float(not_after_cumulative_wall_seconds),
        )
    )
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("decision") != "phase_b1_recovery_final_checkpoint_verified"
        or receipt.get("recovery_authorization_sha256")
        != closure.authorization_sha256
        or receipt.get("recovery_contract_sha256") != closure.contract_sha256
        or receipt.get("recovery_marker_sha256") != handoff.marker_sha256
        or receipt.get("new_handoff_nonce_sha256") != handoff.handoff_nonce_sha256
        or receipt.get("source_checkpoint_sha256")
        != EXPECTED_ORIGINAL_HASHES["checkpoint"]
        or receipt.get("original_marker_sha256") != EXPECTED_ORIGINAL_HASHES["marker"]
        or receipt.get("old_handoff_nonce_sha256")
        != EXPECTED_COMMITMENTS["old_handoff_nonce_sha256"]
        or receipt.get("incident_sha256") != EXPECTED_ORIGINAL_HASHES["incident"]
        or receipt.get("checkpoint_sha256") != _artifact_sha(args.checkpoint_output)
        or receipt.get("tokenizer_sha256") != EXPECTED_ORIGINAL_HASHES["tokenizer"]
        or receipt.get("outer_fit_corpus_commitment_sha256")
        != EXPECTED_COMMITMENTS["outer_fit_corpus_commitment_sha256"]
        or receipt.get("compact_corpus_commitment_sha256")
        != EXPECTED_COMMITMENTS["compact_corpus_commitment_sha256"]
        or receipt.get("shuffle_commitment_sha256")
        != EXPECTED_COMMITMENTS["shuffle_commitment_sha256"]
        or receipt.get("total_original_bytes") != EXPECTED_TOTAL_ORIGINAL_BYTES
        or not _is_sha256(receipt.get("recovery_scan_accounting_sha256"))
        or not _is_sha256(receipt.get("recovery_compact_accounting_sha256"))
        or receipt.get("completed_optimizer_steps") != 28_768
        or receipt.get("completed_sequence_count") != 115_072
        or receipt.get("next_permutation_cursor") != 115_072
        or receipt.get("recovery_parent_pid") != handoff.parent_pid
        or receipt.get("recovery_worker_pid") != worker_pid
        or not isinstance(verifier_pid, int)
        or isinstance(verifier_pid, bool)
        or verifier_pid <= 0
        or verifier_pid in {handoff.parent_pid, worker_pid}
        or receipt.get("model_tensors_finite") is not True
        or receipt.get("optimizer_tensors_finite") is not True
        or receipt.get("rng_state_validated") is not True
        or receipt.get("synthetic_logits_bit_exact") is not True
        or receipt.get("outer_holdout_raw_opens") != 0
        or receipt.get("outer_holdout_raw_bytes") != 0
        or receipt.get("raw_access_performed") is not False
        or receipt.get("quality_metrics_computed") is not False
        or receipt.get("threshold_operations_performed") is not False
        or receipt.get("old_handoff_nonce_possession") is not False
        or not isinstance(cumulative_wall, (int, float))
        or not math.isfinite(float(cumulative_wall))
        or not isinstance(checkpoint_cumulative_wall, (int, float))
        or not math.isfinite(float(checkpoint_cumulative_wall))
        or not RECOVERY_BUDGET_BASE_SECONDS
        <= float(checkpoint_cumulative_wall)
        <= float(cumulative_wall)
        <= upper_wall
    ):
        raise RecoveryFatalError("Independent recovery final receipt drifted")
    _validate_receipt_resource_state(
        receipt.get("resource_state"),
        closure.original_contract,
    )
    b1.assert_report_has_no_quality_metrics(receipt)
    return receipt


def _build_recovery_report(
    *,
    args: argparse.Namespace,
    closure: RecoveryClosure,
    handoff: b1.RunHandoff,
    scope: b1.OuterFitScope,
    scan: dict[str, Any],
    compact: b1.CompactCorpusBuild,
    state: dict[str, Any],
    cumulative_wall_seconds: float,
    lineage: dict[str, Any],
    exact_logits: bool,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    if state["throughput_window_seconds"] > 0:
        b1._close_throughput_window(state, closure.original_contract)
    training_rate = (
        state["training_original_bytes"] / state["training_seconds"]
        if state["training_seconds"] > 0
        else 0.0
    )
    durable_paths = (
        args.source_tokenizer,
        args.source_checkpoint,
        ORIGINAL_MARKER,
        DEFAULT_MARKER,
        args.checkpoint_output,
        DEFAULT_FINAL_RECEIPT,
    )
    durable_bytes = b1._durable_output_bytes(durable_paths)
    resource_gates = closure.original_contract["resource_gates"]
    checkpoint_sha256 = _artifact_sha(args.checkpoint_output)
    receipt_checkpoint_sha256 = receipt.get("checkpoint_sha256")
    receipt_cumulative_wall = float(receipt.get("cumulative_wall_seconds", math.nan))
    scan_sha256 = _canonical_json_sha256(scan)
    compact_sha256 = _canonical_json_sha256(compact.accounting)
    verifier_resources = _validate_receipt_resource_state(
        receipt.get("resource_state"),
        closure.original_contract,
    )
    gates = {
        "old_run_remains_fail_closed": True,
        "old_nonce_continuity_not_claimed": True,
        "immutable_original_artifacts": True,
        "outer_holdout_raw_zero": scope.audit["outer_holdout_raw_opens"] == 0
        and scope.audit["outer_holdout_raw_bytes"] == 0,
        "single_recovery_fit_scan": scan["fit_metadata_rows"] == 16_000
        and scan["fit_raw_open_attempts"] == 15_988
        and scan["fit_raw_open_successes"] == 15_988
        and scan["fit_raw_bytes_actually_read"] == 19_239_582_561
        and scan["source_unavailable"] == 12,
        "recovery_scan_commitment_exact": scan["outer_fit_corpus_commitment_sha256"]
        == EXPECTED_COMMITMENTS["outer_fit_corpus_commitment_sha256"],
        "compact_commitment_exact": compact.corpus.commitment_sha256()
        == EXPECTED_COMMITMENTS["compact_corpus_commitment_sha256"],
        "sequence_byte_coverage_exact": compact.accounting["original_window_bytes"]
        == compact.accounting["prepared_original_bytes"],
        "sequence_drop_zero": compact.accounting["dropped_content_tokens"] == 0
        and compact.accounting["dropped_original_bytes"] == 0
        and compact.accounting["overlength_windows_excluded"] == 0,
        "exact_once_epoch": state["completed_sequence_count"] == len(compact.corpus)
        == 115_072,
        "exact_once_original_bytes": state["training_original_bytes"]
        == compact.corpus.total_original_bytes,
        "fresh_launcher_aware_recovery": handoff.parent_pid != os.getpid()
        and lineage.get("mode") in {"direct_parent", "windows_venv_redirector"},
        "independent_final_checkpoint_verification": receipt.get("decision")
        == "phase_b1_recovery_final_checkpoint_verified",
        "checkpoint_receipt_sha_exact": receipt_checkpoint_sha256 == checkpoint_sha256,
        "receipt_time_precedes_report": math.isfinite(receipt_cumulative_wall)
        and RECOVERY_BUDGET_BASE_SECONDS
        <= receipt_cumulative_wall
        <= cumulative_wall_seconds,
        "receipt_commitments_exact": receipt.get("tokenizer_sha256")
        == EXPECTED_ORIGINAL_HASHES["tokenizer"]
        and receipt.get("outer_fit_corpus_commitment_sha256")
        == EXPECTED_COMMITMENTS["outer_fit_corpus_commitment_sha256"]
        and receipt.get("compact_corpus_commitment_sha256")
        == EXPECTED_COMMITMENTS["compact_corpus_commitment_sha256"]
        and receipt.get("shuffle_commitment_sha256")
        == EXPECTED_COMMITMENTS["shuffle_commitment_sha256"]
        and receipt.get("total_original_bytes") == EXPECTED_TOTAL_ORIGINAL_BYTES
        and receipt.get("incident_sha256") == EXPECTED_ORIGINAL_HASHES["incident"]
        and receipt.get("recovery_scan_accounting_sha256") == scan_sha256
        and receipt.get("recovery_compact_accounting_sha256") == compact_sha256,
        "final_verifier_resources_merged": state["peak_process_rss_bytes"]
        >= verifier_resources["peak_process_rss_bytes"]
        and state["peak_cuda_allocated_bytes"]
        >= verifier_resources["peak_cuda_allocated_bytes"]
        and state["peak_cuda_reserved_bytes"]
        >= verifier_resources["peak_cuda_reserved_bytes"]
        and state["minimum_free_disk_bytes"]
        <= verifier_resources["minimum_free_disk_bytes"],
        "final_verifier_raw_zero": receipt.get("raw_access_performed") is False
        and receipt.get("outer_holdout_raw_opens") == 0
        and receipt.get("outer_holdout_raw_bytes") == 0,
        "synthetic_logits_exact": exact_logits,
        "cumulative_wall": cumulative_wall_seconds <= MAXIMUM_CUMULATIVE_WALL_SECONDS,
        "cuda_allocated": state["peak_cuda_allocated_bytes"]
        < resource_gates["maximum_cuda_allocated_bytes_exclusive"],
        "cuda_reserved": state["peak_cuda_reserved_bytes"]
        < resource_gates["maximum_cuda_reserved_bytes_exclusive"],
        "process_rss": state["peak_process_rss_bytes"]
        < resource_gates["maximum_process_rss_bytes_exclusive"],
        "disk_floor": state["minimum_free_disk_bytes"]
        >= resource_gates["minimum_free_disk_bytes_during_run"],
        "durable_output_cap": durable_bytes
        <= resource_gates["maximum_total_durable_output_bytes"],
        "epoch_average_throughput": training_rate
        >= resource_gates["minimum_original_bytes_per_training_second"],
        "throughput_windows": state["maximum_consecutive_low_throughput_windows"]
        <= resource_gates["consecutive_low_throughput_windows_allowed"],
        "nonfinite_zero": state["nonfinite_events"] == 0,
        "oom_zero": state["oom_events"] == 0,
        "quality_results_absent": True,
    }
    passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "loop_id": "loop166_code_section_foundation",
        "claim_scope": "local_train_only_step4096_recovery_not_model_quality",
        "recovery_authorization_sha256": closure.authorization_sha256,
        "recovery_contract_sha256": closure.contract_sha256,
        "recovery_marker_sha256": handoff.marker_sha256,
        "original_execution_status": "incomplete_fail_closed_at_resume_handoff",
        "original_artifacts": {
            **EXPECTED_ORIGINAL_HASHES,
            "old_handoff_nonce_sha256": EXPECTED_COMMITMENTS[
                "old_handoff_nonce_sha256"
            ],
            "old_handoff_nonce_possession": False,
        },
        "source_checkpoint": {
            **EXPECTED_ORIGINAL_STATE,
            "path": str(args.source_checkpoint),
            "sha256": EXPECTED_ORIGINAL_HASHES["checkpoint"],
            "cumulative_wall_seconds": SOURCE_CUMULATIVE_WALL_SECONDS,
        },
        "scope": scope.audit,
        "raw_access": {
            "confirmed_original_fit_raw_passes": 1,
            "recovery_fit_raw_passes": 1,
            "maximum_total_fit_raw_passes": 2,
            "recovery_fit_raw_open_attempts": scan["fit_raw_open_attempts"],
            "recovery_fit_raw_open_successes": scan["fit_raw_open_successes"],
            "recovery_fit_raw_bytes_actually_read": scan[
                "fit_raw_bytes_actually_read"
            ],
            "outer_holdout_raw_opens": 0,
            "outer_holdout_raw_bytes": 0,
            "final_verifier_raw_opens": 0,
        },
        "recovery_scan": scan,
        "tokenizer": {
            "path": str(args.source_tokenizer),
            "sha256": EXPECTED_ORIGINAL_HASHES["tokenizer"],
            "read_only_reuse": True,
            "refit": False,
        },
        "sequence_preparation": compact.accounting,
        "training": {
            "resumed_from_optimizer_step": 4096,
            "resumed_from_sequence_cursor": 16_384,
            "completed_optimizer_steps": state["completed_optimizer_steps"],
            "completed_sequence_count": state["completed_sequence_count"],
            "prepared_sequence_count": len(compact.corpus),
            "original_bytes_processed": state["training_original_bytes"],
            "training_seconds": state["training_seconds"],
            "original_bytes_per_training_second": training_rate,
            "nonfinite_events": state["nonfinite_events"],
            "oom_events": state["oom_events"],
            "quality_metrics_computed": False,
            "threshold_operations_performed": False,
        },
        "process_lineage": {
            "recovery_parent_pid": handoff.parent_pid,
            "recovery_worker_pid": os.getpid(),
            "worker_lineage": lineage,
            "new_handoff_nonce_sha256": handoff.handoff_nonce_sha256,
            "old_handoff_nonce_possession": False,
        },
        "checkpoint": {
            "path": str(args.checkpoint_output),
            "sha256": checkpoint_sha256,
            "source_checkpoint_overwritten": False,
            "weights_only_restore": True,
            "rng_and_cursor_restored": True,
            "synthetic_logits_bit_exact": exact_logits,
            "final_receipt_path": str(DEFAULT_FINAL_RECEIPT),
            "final_receipt_sha256": _artifact_sha(DEFAULT_FINAL_RECEIPT),
            "verifier_pid": receipt["verifier_pid"],
            "receipt_checkpoint_sha256": receipt_checkpoint_sha256,
            "checkpoint_cumulative_wall_seconds": receipt[
                "checkpoint_cumulative_wall_seconds"
            ],
            "receipt_cumulative_wall_seconds": receipt_cumulative_wall,
            "recovery_scan_accounting_sha256": scan_sha256,
            "recovery_compact_accounting_sha256": compact_sha256,
        },
        "resources": {
            "source_cumulative_wall_seconds": SOURCE_CUMULATIVE_WALL_SECONDS,
            "conservative_failed_handoff_debit_seconds": (
                CONSERVATIVE_FAILED_HANDOFF_DEBIT_SECONDS
            ),
            "recovery_budget_base_seconds": RECOVERY_BUDGET_BASE_SECONDS,
            "cumulative_wall_seconds": cumulative_wall_seconds,
            "peak_process_rss_bytes": state["peak_process_rss_bytes"],
            "peak_cuda_allocated_bytes": state["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes": state["peak_cuda_reserved_bytes"],
            "minimum_free_disk_bytes": state["minimum_free_disk_bytes"],
            "durable_output_bytes_before_report": durable_bytes,
            "final_verifier": verifier_resources,
        },
        "artifacts": {
            "raw_code_artifact_bytes": 0,
            "durable_token_artifact_bytes": 0,
            "source_artifacts_mutated": False,
            "new_checkpoint_atomic": True,
            "new_receipt_exclusive": True,
            "new_report_exclusive": True,
        },
        "gates": gates,
        "decision": PASS_DECISION if passed else FAILURE_DECISION,
        "ready_for": {
            "phase_b1_recovery_complete": passed,
            "five_fold_oof": False,
            "val_test_or_full": False,
            "promotion": False,
        },
        "research_champion": "Loop151",
    }
    b1.assert_report_has_no_quality_metrics(report)
    return report


def _validate_parent_report_closure(
    args: argparse.Namespace,
    closure: RecoveryClosure,
    handoff: b1.RunHandoff,
    parent_cumulative_wall_seconds: float,
) -> dict[str, Any]:
    report, _raw = _read_json(args.report_output, "Phase B1 recovery report")
    gates = report.get("gates")
    ready = report.get("ready_for")
    checkpoint = report.get("checkpoint")
    resources = report.get("resources")
    lineage = report.get("process_lineage")
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("decision") != PASS_DECISION
        or report.get("recovery_authorization_sha256")
        != closure.authorization_sha256
        or report.get("recovery_contract_sha256") != closure.contract_sha256
        or report.get("recovery_marker_sha256") != handoff.marker_sha256
        or not isinstance(gates, dict)
        or not gates
        or any(value is not True for value in gates.values())
        or ready
        != {
            "phase_b1_recovery_complete": True,
            "five_fold_oof": False,
            "val_test_or_full": False,
            "promotion": False,
        }
        or not isinstance(checkpoint, dict)
        or checkpoint.get("path") != str(args.checkpoint_output)
        or checkpoint.get("sha256") != _artifact_sha(args.checkpoint_output)
        or checkpoint.get("receipt_checkpoint_sha256")
        != checkpoint.get("sha256")
        or checkpoint.get("final_receipt_path") != str(DEFAULT_FINAL_RECEIPT)
        or checkpoint.get("final_receipt_sha256") != _artifact_sha(DEFAULT_FINAL_RECEIPT)
        or not isinstance(resources, dict)
        or not isinstance(resources.get("cumulative_wall_seconds"), (int, float))
        or not RECOVERY_BUDGET_BASE_SECONDS
        <= float(resources["cumulative_wall_seconds"])
        <= parent_cumulative_wall_seconds
        <= MAXIMUM_CUMULATIVE_WALL_SECONDS
        or not isinstance(lineage, dict)
        or not isinstance(lineage.get("recovery_worker_pid"), int)
        or isinstance(lineage.get("recovery_worker_pid"), bool)
        or lineage.get("recovery_worker_pid", 0) <= 0
    ):
        raise RecoveryFatalError("Recovery parent report closure drifted")
    receipt = _load_final_receipt(
        args,
        closure,
        handoff,
        int(lineage["recovery_worker_pid"]),
        not_after_cumulative_wall_seconds=float(resources["cumulative_wall_seconds"]),
    )
    if (
        receipt.get("checkpoint_sha256") != checkpoint.get("sha256")
        or receipt.get("cumulative_wall_seconds")
        != checkpoint.get("receipt_cumulative_wall_seconds")
        or receipt.get("checkpoint_cumulative_wall_seconds")
        != checkpoint.get("checkpoint_cumulative_wall_seconds")
        or receipt.get("recovery_scan_accounting_sha256")
        != checkpoint.get("recovery_scan_accounting_sha256")
        or receipt.get("recovery_compact_accounting_sha256")
        != checkpoint.get("recovery_compact_accounting_sha256")
    ):
        raise RecoveryFatalError("Recovery report and final receipt disagree")
    b1.assert_report_has_no_quality_metrics(report)
    return report


def _run_bound_training_continuation(
    *,
    args: argparse.Namespace,
    closure: RecoveryClosure,
    handoff: b1.RunHandoff,
    recovery_context: dict[str, Any],
    torch_module: Any,
    model: Any,
    optimizer: Any,
    model_config: Any,
    scaler: Any,
    mask_generator: Any,
    tokenizer: Any,
    compact: b1.CompactCorpusBuild,
    permutation: Sequence[int],
    schedule: Any,
    state: dict[str, Any],
    cumulative_base: float,
    worker_started: float,
) -> dict[str, Any]:
    if (
        state.get("completed_optimizer_steps") != 4096
        or state.get("completed_sequence_count") != 16_384
        or state.get("next_permutation_cursor") != 16_384
        or state.get("training_original_bytes") != 8_200_700
    ):
        raise RecoveryFatalError("Recovery training source state is not the bound cursor")
    return b1.train_segment(
        torch_module=torch_module,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        model_config=model_config,
        tokenizer=tokenizer,
        tokenizer_sha256=EXPECTED_ORIGINAL_HASHES["tokenizer"],
        corpus=compact.corpus,
        corpus_commitment=EXPECTED_COMMITMENTS["outer_fit_corpus_commitment_sha256"],
        permutation=permutation,
        schedule=schedule,
        mask_generator=mask_generator,
        state=state,
        contract=closure.original_contract,
        checkpoint_path=args.checkpoint_output,
        cumulative_wall_seconds_before=cumulative_base,
        phase_started=worker_started,
        parent_pid=handoff.parent_pid,
        handoff=handoff,
        canonical_child_argv_sha256=_argv_sha256(
            canonical_command("final_verify", args)
        ),
        resume_pid=os.getpid(),
        run_context=recovery_context,
    )


def run_recovery_worker(args: argparse.Namespace) -> dict[str, Any]:
    worker_started = time.perf_counter()
    handoff, unexpected_worker_pid, cumulative_base = _handoff_from_environment(
        "recovery"
    )
    if unexpected_worker_pid != 0:
        raise RecoveryFatalError("Recovery worker received a verifier lineage handoff")
    control = validate_control_preflight(args, mode="recovery")
    if control.authorization_sha256 != handoff.authorization_sha256:
        raise RecoveryFatalError("Recovery worker authorization handoff drifted")
    marker = _validate_recovery_marker(handoff, control, args)
    if cumulative_base < float(marker["cumulative_wall_seconds_at_consumption"]):
        raise RecoveryFatalError("Recovery worker cumulative time precedes marker consumption")
    lineage = _validate_worker_lineage(handoff.parent_pid, control.runtime)
    if _argv_sha256(canonical_command("recovery", args)) != _read_json(
        DEFAULT_MARKER, "recovery marker"
    )[0]["canonical_recovery_argv_sha256"]:
        raise RecoveryFatalError("Recovery worker argv commitment drifted")
    closure = complete_preflight(args, control)

    source_payload = closure.source_payload
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.source_tokenizer.resolve(strict=True)))
    torch, model, optimizer, model_config, scaler, mask_generator, exact_logits = (
        b1._restore_training_runtime(
            closure.original_contract,
            tokenizer,
            source_payload,
        )
    )
    state = dict(source_payload["training_state"])
    scope = b1.load_and_select_outer_fit_scope(
        closure.original_contract,
        folds_path=args.folds,
        folds_summary_path=args.folds_summary,
        data_root=args.data_root,
    )
    pre_scan_guard = _guard_before_recovery_raw_access(
        args,
        closure,
        cumulative_wall_seconds=cumulative_base + time.perf_counter() - worker_started,
        state=state,
    )
    scan = b1.scan_outer_fit_corpus(
        scope,
        closure.original_contract,
        data_root=args.data_root,
        disk_probe_path=args.checkpoint_output.parent,
        cumulative_wall_seconds_before=(
            cumulative_base + time.perf_counter() - worker_started
        ),
    )
    if (
        scan.outer_fit_corpus_commitment_sha256
        != EXPECTED_COMMITMENTS["outer_fit_corpus_commitment_sha256"]
    ):
        raise RecoveryFatalError("Recovery outer-fit corpus commitment drifted")
    compact = b1.build_compact_corpus(
        tokenizer,
        scan.windows,
        closure.original_contract,
    )
    post_compact_guard = b1.guard_non_cuda_phase(
        closure.original_contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=cumulative_base + time.perf_counter() - worker_started,
    )
    if (
        compact.corpus.commitment_sha256()
        != EXPECTED_COMMITMENTS["compact_corpus_commitment_sha256"]
        or len(compact.corpus) != EXPECTED_ORIGINAL_STATE["prepared_sequence_count"]
    ):
        raise RecoveryFatalError("Recovery compact corpus commitment drifted")
    permutation = deterministic_permutation(
        len(compact.corpus), int(closure.original_contract["training"]["shuffle_seed"])
    )
    if (
        permutation_commitment_sha256(permutation)
        != EXPECTED_COMMITMENTS["shuffle_commitment_sha256"]
    ):
        raise RecoveryFatalError("Recovery deterministic permutation commitment drifted")
    prefix_bytes = sum(
        compact.corpus[index].original_byte_length for index in permutation[:16_384]
    )
    if prefix_bytes != EXPECTED_ORIGINAL_STATE["permutation_prefix_original_bytes"]:
        raise RecoveryFatalError("Recovery permutation prefix byte accounting drifted")
    schedule = b1.prepare_validated_schedule(permutation, closure.original_contract)
    b1.verify_checkpoint_payload(
        source_payload,
        closure.original_contract,
        expected_child_argv_sha256=(
            "1b9397f47029c78ee61318de6b00422bdbf0cdfa3ab7c46a1fcd2f22298af09d"
        ),
        corpus=compact.corpus,
        permutation=permutation,
    )
    b1._merge_scan_resource_peak(state, scan)
    state["peak_process_rss_bytes"] = max(
        state["peak_process_rss_bytes"],
        pre_scan_guard["process_rss_bytes"],
        post_compact_guard["process_rss_bytes"],
    )
    state["minimum_free_disk_bytes"] = min(
        state["minimum_free_disk_bytes"],
        scan.accounting["minimum_free_disk_bytes"],
        pre_scan_guard["free_disk_bytes"],
        post_compact_guard["free_disk_bytes"],
    )
    scan_accounting = dict(scan.accounting)
    scan_accounting["outer_fit_corpus_commitment_sha256"] = (
        scan.outer_fit_corpus_commitment_sha256
    )
    _validate_scan_and_compact_ledgers(scan_accounting, compact.accounting)
    recovery_context = _recovery_run_context(
        closure,
        handoff,
        args,
        lineage,
        recovery_scan_accounting=scan_accounting,
        recovery_compact_accounting=compact.accounting,
    )
    del scan
    gc.collect()

    _run_bound_training_continuation(
        args=args,
        closure=closure,
        handoff=handoff,
        recovery_context=recovery_context,
        torch_module=torch,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        scaler=scaler,
        mask_generator=mask_generator,
        tokenizer=tokenizer,
        compact=compact,
        permutation=permutation,
        schedule=schedule,
        state=state,
        cumulative_base=cumulative_base,
        worker_started=worker_started,
    )
    if (
        state["completed_optimizer_steps"] != 28_768
        or state["completed_sequence_count"] != 115_072
        or state["next_permutation_cursor"] != 115_072
    ):
        raise RecoveryFatalError("Recovery training did not close the exact-once epoch")
    cumulative_wall = cumulative_base + time.perf_counter() - worker_started
    b1.guard_non_cuda_phase(
        closure.original_contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=cumulative_wall,
        state=state,
    )
    del model, optimizer, scaler, mask_generator
    gc.collect()
    torch.cuda.empty_cache()

    verify_return_code = _spawn_worker(
        "final_verify",
        args,
        handoff,
        closure,
        worker_pid=os.getpid(),
        cumulative_base_seconds=cumulative_wall,
    )
    if verify_return_code != 0:
        raise RecoveryFatalError(
            f"Recovery final verifier failed with exit code {verify_return_code}"
        )
    cumulative_wall = cumulative_base + time.perf_counter() - worker_started
    receipt = _load_final_receipt(
        args,
        closure,
        handoff,
        os.getpid(),
        not_after_cumulative_wall_seconds=cumulative_wall,
    )
    if (
        receipt.get("recovery_scan_accounting_sha256")
        != recovery_context["recovery_scan_accounting_sha256"]
        or receipt.get("recovery_compact_accounting_sha256")
        != recovery_context["recovery_compact_accounting_sha256"]
    ):
        raise RecoveryFatalError("Final receipt disagrees with the recovery scan ledger")
    _merge_final_verifier_resources(state, receipt, closure.original_contract)
    cumulative_wall = cumulative_base + time.perf_counter() - worker_started
    b1.guard_non_cuda_phase(
        closure.original_contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=cumulative_wall,
        state=state,
    )
    _assert_control_closure_unchanged(args, closure)
    _assert_immutable_originals(args)
    cumulative_wall = cumulative_base + time.perf_counter() - worker_started
    b1.guard_non_cuda_phase(
        closure.original_contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=cumulative_wall,
        state=state,
    )
    report = _build_recovery_report(
        args=args,
        closure=closure,
        handoff=handoff,
        scope=scope,
        scan=scan_accounting,
        compact=compact,
        state=state,
        cumulative_wall_seconds=cumulative_wall,
        lineage=lineage,
        exact_logits=exact_logits,
        receipt=receipt,
    )
    _write_exclusive_json(args.report_output, report, "Recovery report")
    return report


def run_final_verify_worker(args: argparse.Namespace) -> dict[str, Any]:
    verifier_started = time.perf_counter()
    handoff, recovery_worker_pid, cumulative_base = _handoff_from_environment(
        "final_verify"
    )
    if recovery_worker_pid <= 0:
        raise RecoveryFatalError("Final verifier has no bound recovery worker PID")
    control = validate_control_preflight(args, mode="final_verify")
    if control.authorization_sha256 != handoff.authorization_sha256:
        raise RecoveryFatalError("Final verifier authorization handoff drifted")
    marker = _validate_recovery_marker(handoff, control, args)
    if cumulative_base < float(marker["cumulative_wall_seconds_at_consumption"]):
        raise RecoveryFatalError("Final verifier cumulative time precedes marker consumption")
    lineage = _validate_worker_lineage(recovery_worker_pid, control.runtime)
    closure = complete_preflight(args, control)
    if DEFAULT_FINAL_RECEIPT.exists() or args.report_output.exists():
        raise RecoveryFatalError("Recovery final output already exists")
    checkpoint_sha_before_load = _artifact_sha(args.checkpoint_output)
    payload = b1._load_checkpoint_weights_only(
        args.checkpoint_output,
        closure.original_contract,
        expected_handoff=handoff,
        expected_child_argv_sha256=_argv_sha256(
            canonical_command("final_verify", args)
        ),
        require_final=True,
    )
    run_context = payload.get("run_context")
    if not isinstance(run_context, dict) or (
        run_context.get("source_checkpoint_sha256")
        != EXPECTED_ORIGINAL_HASHES["checkpoint"]
        or run_context.get("source_checkpoint_cursor") != 16_384
        or run_context.get("source_checkpoint_optimizer_step") != 4096
        or run_context.get("incident_sha256") != EXPECTED_ORIGINAL_HASHES["incident"]
        or run_context.get("original_authorization_sha256")
        != EXPECTED_ORIGINAL_HASHES["authorization"]
        or run_context.get("original_marker_sha256")
        != EXPECTED_ORIGINAL_HASHES["marker"]
        or run_context.get("old_handoff_nonce_sha256")
        != EXPECTED_COMMITMENTS["old_handoff_nonce_sha256"]
        or run_context.get("old_handoff_nonce_possession") is not False
        or run_context.get("recovery_contract_sha256") != closure.contract_sha256
        or run_context.get("recovery_authorization_sha256")
        != closure.authorization_sha256
        or run_context.get("recovery_marker_sha256") != handoff.marker_sha256
        or run_context.get("new_handoff_nonce_sha256") != handoff.handoff_nonce_sha256
        or run_context.get("recovery_worker_pid") != recovery_worker_pid
        or run_context.get("canonical_recovery_argv_sha256")
        != _argv_sha256(canonical_command("recovery", args))
        or run_context.get("canonical_final_verify_argv_sha256")
        != _argv_sha256(canonical_command("final_verify", args))
    ):
        raise RecoveryFatalError("Recovery final checkpoint run context drifted")
    if (
        payload.get("parent_pid") != handoff.parent_pid
        or payload.get("resume_pid") != recovery_worker_pid
    ):
        raise RecoveryFatalError("Recovery final checkpoint process lineage drifted")
    commitment_audit = _validate_final_checkpoint_commitments(payload, closure)
    checkpoint_cumulative_wall = payload.get("cumulative_wall_seconds")
    if (
        not isinstance(checkpoint_cumulative_wall, (int, float))
        or not math.isfinite(float(checkpoint_cumulative_wall))
        or not RECOVERY_BUDGET_BASE_SECONDS
        <= float(checkpoint_cumulative_wall)
        <= cumulative_base
    ):
        raise RecoveryFatalError("Recovery final checkpoint time lineage drifted")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.source_tokenizer.resolve(strict=True)))
    torch, model, optimizer, _config, _scaler, _generator, exact_logits = (
        b1._restore_training_runtime(
            closure.original_contract,
            tokenizer,
            payload,
        )
    )
    b1._assert_finite_tensor_tree(model.state_dict(), "recovery_final_model_state")
    b1._assert_finite_tensor_tree(
        optimizer.state_dict(), "recovery_final_optimizer_state"
    )
    state = dict(payload["training_state"])
    cumulative_wall = cumulative_base + time.perf_counter() - verifier_started
    device = next(model.parameters()).device
    b1._sample_runtime_resources(
        torch,
        device,
        state,
        closure.original_contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=cumulative_wall,
    )
    cumulative_wall = cumulative_base + time.perf_counter() - verifier_started
    b1.guard_non_cuda_phase(
        closure.original_contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=cumulative_wall,
        state=state,
    )
    if not exact_logits:
        raise RecoveryFatalError("Recovery final synthetic logits are not bit exact")
    _assert_control_closure_unchanged(args, closure)
    _assert_immutable_originals(args)
    cumulative_wall = cumulative_base + time.perf_counter() - verifier_started
    b1.guard_non_cuda_phase(
        closure.original_contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=cumulative_wall,
        state=state,
    )
    checkpoint_sha256 = _assert_stable_artifact_sha(
        args.checkpoint_output,
        checkpoint_sha_before_load,
        "Recovery final checkpoint",
    )
    cumulative_wall = cumulative_base + time.perf_counter() - verifier_started
    b1.guard_non_cuda_phase(
        closure.original_contract,
        disk_path=args.checkpoint_output.parent,
        cumulative_wall_seconds=cumulative_wall,
        state=state,
    )
    resource_state = {
        "peak_process_rss_bytes": int(state["peak_process_rss_bytes"]),
        "peak_cuda_allocated_bytes": int(state["peak_cuda_allocated_bytes"]),
        "peak_cuda_reserved_bytes": int(state["peak_cuda_reserved_bytes"]),
        "minimum_free_disk_bytes": int(state["minimum_free_disk_bytes"]),
    }
    _validate_receipt_resource_state(resource_state, closure.original_contract)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "loop_id": "loop166_code_section_foundation",
        "recovery_authorization_sha256": closure.authorization_sha256,
        "recovery_contract_sha256": closure.contract_sha256,
        "recovery_marker_sha256": handoff.marker_sha256,
        "new_handoff_nonce_sha256": handoff.handoff_nonce_sha256,
        "source_checkpoint_sha256": EXPECTED_ORIGINAL_HASHES["checkpoint"],
        "original_marker_sha256": EXPECTED_ORIGINAL_HASHES["marker"],
        "old_handoff_nonce_sha256": EXPECTED_COMMITMENTS["old_handoff_nonce_sha256"],
        "old_handoff_nonce_possession": False,
        **commitment_audit,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_cumulative_wall_seconds": float(checkpoint_cumulative_wall),
        "completed_optimizer_steps": payload["completed_optimizer_steps"],
        "completed_sequence_count": payload["completed_sequence_count"],
        "next_permutation_cursor": payload["next_permutation_cursor"],
        "model_tensors_finite": True,
        "optimizer_tensors_finite": True,
        "rng_state_validated": True,
        "synthetic_logits_bit_exact": True,
        "recovery_parent_pid": handoff.parent_pid,
        "recovery_worker_pid": recovery_worker_pid,
        "verifier_pid": os.getpid(),
        "verifier_lineage": lineage,
        "resource_state": resource_state,
        "cumulative_wall_seconds": cumulative_wall,
        "outer_holdout_raw_opens": 0,
        "outer_holdout_raw_bytes": 0,
        "raw_access_performed": False,
        "quality_metrics_computed": False,
        "threshold_operations_performed": False,
        "decision": "phase_b1_recovery_final_checkpoint_verified",
    }
    _write_exclusive_json(DEFAULT_FINAL_RECEIPT, receipt, "Recovery final receipt")
    del model, optimizer, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover Loop166 Phase B1 from its immutable step-4096 checkpoint."
    )
    parser.add_argument("--recovery-worker", action="store_true")
    parser.add_argument("--final-verify-worker", action="store_true")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--folds-summary", type=Path, default=DEFAULT_FOLDS_SUMMARY)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-tokenizer", type=Path, default=DEFAULT_SOURCE_TOKENIZER)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--checkpoint-output", type=Path, default=DEFAULT_CHECKPOINT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "contract",
        "authorization",
        "folds",
        "folds_summary",
        "data_root",
        "source_tokenizer",
        "source_checkpoint",
    ):
        value = Path(getattr(args, name))
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        setattr(args, name, value.resolve(strict=True))
    for name in ("checkpoint_output", "report_output"):
        value = Path(getattr(args, name))
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        setattr(args, name, b1._resolve_output_path(value))
    expected = {
        "contract": DEFAULT_CONTRACT.resolve(strict=True),
        "authorization": DEFAULT_AUTHORIZATION.resolve(strict=True),
        "folds": DEFAULT_FOLDS.resolve(strict=True),
        "folds_summary": DEFAULT_FOLDS_SUMMARY.resolve(strict=True),
        "data_root": DEFAULT_DATA_ROOT.resolve(strict=True),
        "source_tokenizer": DEFAULT_SOURCE_TOKENIZER.resolve(strict=True),
        "source_checkpoint": DEFAULT_SOURCE_CHECKPOINT.resolve(strict=True),
        "checkpoint_output": DEFAULT_CHECKPOINT_OUTPUT.absolute(),
        "report_output": DEFAULT_REPORT_OUTPUT.absolute(),
    }
    for name, expected_path in expected.items():
        if Path(getattr(args, name)).absolute() != expected_path:
            raise RecoveryFatalError(f"Recovery runtime path is not canonical: {name}")
    if args.recovery_worker and args.final_verify_worker:
        raise RecoveryFatalError("Recovery internal worker modes are mutually exclusive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _normalize_args(build_parser().parse_args(argv))
    mode = (
        "recovery"
        if args.recovery_worker
        else "final_verify"
        if args.final_verify_worker
        else "parent"
    )
    actual_argv = (
        str(Path(sys.executable).resolve(strict=True)),
        str(Path(sys.argv[0]).resolve(strict=True)),
        *sys.argv[1:],
    )
    if actual_argv != canonical_command(mode, args):
        raise RecoveryFatalError(f"Recovery {mode} argv is not canonical")
    if mode == "recovery":
        report = run_recovery_worker(args)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return (
            0
            if report["decision"] == PASS_DECISION
            else 2
        )
    if mode == "final_verify":
        receipt = run_final_verify_worker(args)
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
