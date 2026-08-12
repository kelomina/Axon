#!/usr/bin/env python3
"""Run the contained Loop167 Phase-B v11 controller route."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loop167_phase_b.child_attestation_v11 import (  # noqa: E402
    build_child_job_attestation_payload_v11,
    write_child_job_attestation_v11,
)
from loop167_phase_b.contracts import (  # noqa: E402
    PhaseBContractError,
    require_canonical_json,
    sha256_file,
)
from loop167_phase_b.execution_authorization_v11 import (  # noqa: E402
    validate_execution_authorization_v11,
)
from loop167_phase_b.execution_contract_v5 import B1_SAMPLING_INDICATORS_CONTRACT  # noqa: E402
from loop167_phase_b.execution_contract_v11 import (  # noqa: E402
    AUTHORIZATION_CLAIM_SCOPE,
    CONTROLLER_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    FIXED_OUTPUT_CATALOG,
    LOOP166_WINDOWS_JOB_RELATIVE_PATH,
    LOOP166_WINDOWS_PROCESS_LINEAGE_RELATIVE_PATH,
    RAW_ROOT_RELATIVE_PATH,
    RUN_AUTHORIZATION_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    SUPERVISOR_RELATIVE_PATH,
    resolve_output_catalog_v11,
    verify_execution_contract_v11,
)
from loop167_phase_b.invocation_v11 import (  # noqa: E402
    bootstrap_thread_environment_v11,
    validate_current_runtime_invocation_v11,
)
from loop167_phase_b.lease_v11 import (  # noqa: E402
    consume_execution_lease_v11,
    verify_consumed_execution_lease_v11,
)
from loop167_phase_b.path_safety_v4 import safe_project_path  # noqa: E402
from loop167_phase_b.preflight_v11 import validate_static_preflight_v11  # noqa: E402
from loop167_phase_b.supervisor_v11 import _write_new_json  # noqa: E402

RECEIPT_SCHEMA = "axon_loop167_phase_b_execution_receipt_v11"


def _binding(path: Path) -> dict[str, str]:
    relative_path = path.relative_to(PROJECT_ROOT).as_posix()
    safe_path = safe_project_path(
        PROJECT_ROOT,
        relative_path,
        require_exists=True,
        require_regular_file=True,
    )
    return {
        "path": relative_path,
        "sha256": sha256_file(safe_path),
    }


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _is_link_or_reparse(stat_result: os.stat_result) -> bool:
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    return stat.S_ISLNK(stat_result.st_mode) or bool(attributes & 0x0400)


def _resolve_fixed_raw_root_adapter(root: Path, raw_root_relative: str) -> Path:
    """Validate the fixed directory boundary only after the lease is consumed."""

    if raw_root_relative != RAW_ROOT_RELATIVE_PATH:
        raise PhaseBContractError("v11 execution contract raw root drifted")
    root = root.resolve(strict=True)
    candidate = root.joinpath(*raw_root_relative.split("/"))
    cursor = root
    for component in raw_root_relative.split("/"):
        cursor = cursor / component
        try:
            stat_result = cursor.lstat()
        except OSError as error:
            raise PhaseBContractError("v11 fixed raw root is unavailable before lease consumption") from error
        if _is_link_or_reparse(stat_result) or not stat.S_ISDIR(stat_result.st_mode):
            raise PhaseBContractError("v11 fixed raw root traverses an unsafe directory")
    if candidate != cursor:
        raise PhaseBContractError("v11 fixed raw root resolution drifted")
    return candidate


def _ensure_output_parent_after_lease(root: Path, output_path: Path) -> None:
    try:
        relative_parent = output_path.parent.relative_to(root)
    except ValueError as error:
        raise PhaseBContractError("v11 sealed output path escapes the project root") from error
    cursor = root
    for component in relative_parent.parts:
        cursor = cursor / component
        try:
            cursor.mkdir(exist_ok=True)
            stat_result = cursor.lstat()
        except OSError as error:
            raise PhaseBContractError("v11 sealed output parent cannot be prepared") from error
        if _is_link_or_reparse(stat_result) or not stat.S_ISDIR(stat_result.st_mode):
            raise PhaseBContractError("v11 sealed output parent is unsafe")


def _assert_authorization_matches_static_preflight(authorization: Any, static_receipt: Any) -> None:
    expected = {
        "source_closure_binding": static_receipt.source_closure_binding,
        "execution_contract_binding": static_receipt.execution_contract_binding,
        "runtime_lock_binding": static_receipt.runtime_lock_binding,
        "controller_binding": static_receipt.controller_binding,
        "supervisor_binding": static_receipt.supervisor_binding,
        "loop166_windows_job_binding": static_receipt.loop166_windows_job_binding,
        "loop166_windows_process_lineage_binding": static_receipt.loop166_windows_process_lineage_binding,
    }
    for name, expected_binding in expected.items():
        if dict(getattr(authorization, name)) != dict(expected_binding):
            raise PhaseBContractError(f"v11 execution authorization {name} drifted from static preflight")


def _assert_static_preflight_is_unchanged(initial: Any, refreshed: Any) -> None:
    fields = (
        "source_closure_binding",
        "source_closure_sha256",
        "execution_contract_binding",
        "execution_contract_sha256",
        "runtime_lock_binding",
        "runtime_lock_sha256",
        "controller_binding",
        "supervisor_binding",
        "loop166_windows_job_binding",
        "loop166_windows_process_lineage_binding",
    )
    if any(getattr(initial, field) != getattr(refreshed, field) for field in fields):
        raise PhaseBContractError("v11 static source closure drifted after lease verification")


def _require_elapsed(started_at: float, maximum_seconds: int, *, phase: str) -> None:
    if time.monotonic() - started_at > maximum_seconds:
        raise PhaseBContractError(f"v11 Phase-B {phase} wall-clock budget was exceeded")


def _execute_after_lease(
    *,
    authorization: Any,
    static_receipt: Any,
    contract: Any,
    raw_root: Path,
    lease: Any,
    started_at: float,
) -> dict[str, Any]:
    """Import the numerical/raw data plane only after a verified v11 lease exists."""

    from loop167_phase_b.evaluation_v4 import evaluate_phase_b_fit
    from loop167_phase_b.feature_cache_v4 import (
        load_phase_b_feature_cache_v4,
        write_phase_b_feature_cache_v4,
    )
    from loop167_phase_b.fit_worker import run_phase_b_fit
    from loop167_phase_b.progress_ledger import (
        FitLedger,
        RawScanLedger,
        validate_fit_ledger,
        validate_raw_scan_ledger,
    )
    from loop167_phase_b.raw_manifest_adapter_v4 import load_train_only_manifest_v4
    from loop167_phase_b.raw_worker import RawFeatureWorker, RawWorkerConfig

    outputs = authorization.output_paths
    required_output_names = {
        "feature_cache",
        "raw_progress_ledger",
        "fit_progress_ledger",
        "execution_receipt",
    }
    if not required_output_names.issubset(outputs):
        raise PhaseBContractError("v11 execution authorization output catalog drifted")
    for name in required_output_names:
        _ensure_output_parent_after_lease(PROJECT_ROOT, outputs[name])

    protocol = require_canonical_json(PROJECT_ROOT / authorization.protocol_binding["path"])
    raw_context_contract = protocol.get("feature_contract", {}).get("raw_context")
    if not isinstance(raw_context_contract, dict):
        raise PhaseBContractError("v11 Phase-B raw-context contract is unavailable")
    maximum_source_file_bytes = raw_context_contract.get("maximum_source_file_bytes")
    reader_chunk_bytes = raw_context_contract.get("reader_chunk_bytes")
    if not isinstance(maximum_source_file_bytes, int) or not isinstance(reader_chunk_bytes, int):
        raise PhaseBContractError("v11 Phase-B raw-context limits drifted")

    manifest = load_train_only_manifest_v4(
        PROJECT_ROOT,
        phase_b_protocol_binding=authorization.protocol_binding,
        data_root=raw_root,
    )
    extraction_started_at = time.monotonic()
    raw_config = RawWorkerConfig(
        maximum_source_file_bytes=maximum_source_file_bytes,
        maximum_raw_open_attempts=int(contract.resource_contract["maximum_raw_open_attempts"]),
        maximum_raw_bytes_read=int(contract.resource_contract["maximum_raw_bytes"]),
        reader_chunk_bytes=reader_chunk_bytes,
    )
    with RawScanLedger.create(outputs["raw_progress_ledger"]) as raw_ledger:
        raw_outcome = RawFeatureWorker(raw_config).scan(
            manifest.raw_scan_plan,
            expected_raw_scope_commitment_sha256=manifest.raw_scan_plan.raw_scope_commitment_sha256,
            ledger=raw_ledger,
        )
    raw_validation = validate_raw_scan_ledger(outputs["raw_progress_ledger"])
    if not raw_validation.complete or raw_validation.final_record_sha256 != raw_outcome.raw_ledger_final_record_sha256:
        raise PhaseBContractError("v11 raw progress ledger did not close the sealed one-pass scan")
    cache_receipt = write_phase_b_feature_cache_v4(
        outputs["feature_cache"],
        raw_outcome,
        expected_raw_scope_commitment_sha256=manifest.raw_scan_plan.raw_scope_commitment_sha256,
    )
    loaded_cache = load_phase_b_feature_cache_v4(
        cache_receipt.cache_path,
        expected_cache_sha256=cache_receipt.cache_sha256,
        expected_raw_scope_commitment_sha256=raw_outcome.raw_scope_commitment_sha256,
        expected_feature_rows_commitment_sha256=raw_outcome.feature_rows_commitment_sha256,
        expected_raw_ledger_final_record_sha256=raw_outcome.raw_ledger_final_record_sha256,
    )
    sampling_contract = dict(B1_SAMPLING_INDICATORS_CONTRACT)
    if sampling_contract.get("receipt_key") != "sampling_audit":
        raise PhaseBContractError("v11 execution contract B1 sampling-audit binding drifted")
    if len(cache_receipt.sampling_audit.indicator_counts) != sampling_contract.get("dimension"):
        raise PhaseBContractError("v11 B1 sampling-audit dimension drifted from the execution contract")
    _require_elapsed(
        extraction_started_at,
        int(contract.resource_contract["maximum_extraction_wall_seconds"]),
        phase="extraction",
    )

    fit_payload = manifest.to_phase_b_fit_payload(loaded_cache.cache)
    fitting_started_at = time.monotonic()
    with FitLedger.create(outputs["fit_progress_ledger"]) as fit_ledger:
        fit_result = run_phase_b_fit(
            fit_payload.cache,
            fit_payload.labels,
            fit_payload.folds,
            fit_ledger,
            fit_protocol_commitment_sha256=manifest.phase_b_protocol_sha256,
            feature_rows_commitment_sha256=raw_outcome.feature_rows_commitment_sha256,
            raw_ledger_final_record_sha256=raw_outcome.raw_ledger_final_record_sha256,
        )
    fit_validation = validate_fit_ledger(outputs["fit_progress_ledger"])
    if not fit_validation.complete or fit_validation.final_record_sha256 != fit_result.fit_ledger_final_record_sha256:
        raise PhaseBContractError("v11 fit progress ledger did not close all fixed units")
    _require_elapsed(
        fitting_started_at,
        int(contract.resource_contract["maximum_training_wall_seconds"]),
        phase="fitting",
    )
    evaluation = evaluate_phase_b_fit(
        fit_result,
        fit_payload.labels,
        fit_payload.folds,
        manifest.component_ids,
        protocol_sha256=manifest.phase_b_protocol_sha256,
    )
    _require_elapsed(
        started_at,
        int(contract.resource_contract["maximum_total_wall_seconds"]),
        phase="total execution",
    )
    return {
        "schema": RECEIPT_SCHEMA,
        "loop_id": "loop167_ember_v3_novel_delta",
        "status": "completed_single_v7_contained_train_only_raw_pass_fixed_oof_not_promotion_or_heldout_evaluation",
        "claim_scope": AUTHORIZATION_CLAIM_SCOPE,
        "source_closure": dict(static_receipt.source_closure_binding),
        "phase_b_execution_contract": dict(authorization.execution_contract_binding),
        "runtime_lock": dict(authorization.runtime_lock_binding),
        "run_authorization": {
            "path": RUN_AUTHORIZATION_RELATIVE_PATH,
            "sha256": authorization.authorization_sha256,
        },
        "execution_lease": {
            "path": lease.marker_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": lease.marker_sha256,
        },
        "pre_resume_launch_receipt": {
            "path": lease.payload["pre_resume_launch_receipt"]["path"],
            "sha256": lease.launch_receipt_sha256,
        },
        "child_job_attestation": {
            "path": lease.payload["child_job_attestation"]["path"],
            "sha256": lease.child_attestation_sha256,
        },
        "fold_manifest_sha256": manifest.fold_manifest_sha256,
        "raw_scope_commitment_sha256": raw_outcome.raw_scope_commitment_sha256,
        "raw_progress_ledger_final_record_sha256": raw_outcome.raw_ledger_final_record_sha256,
        "raw_open_attempts": raw_validation.cumulative_raw_open_attempts,
        "raw_bytes_read": raw_validation.cumulative_raw_bytes_read,
        "feature_rows_commitment_sha256": raw_outcome.feature_rows_commitment_sha256,
        "feature_cache": {
            "sha256": cache_receipt.cache_sha256,
            "bytes": cache_receipt.cache_bytes,
            "sampling_contract": sampling_contract,
            "sampling_audit": cache_receipt.sampling_audit.to_metadata(),
        },
        "fit": {
            "total_fit_units": fit_result.total_fit_units,
            "fit_progress_ledger_final_record_sha256": fit_result.fit_ledger_final_record_sha256,
            "matrix_replay_sha256": fit_result.matrix_replay_sha256,
            "evaluation_replay_sha256": fit_result.evaluation_replay_sha256,
        },
        "evaluation": {str(seed): dict(summary) for seed, summary in evaluation.items()},
        "heldout_access": False,
        "promotion": False,
    }


def _static_preflight(mode: str, *, phase: str) -> Any:
    bootstrap_thread_environment_v11()
    validate_current_runtime_invocation_v11(PROJECT_ROOT, role="controller", mode=mode)
    return validate_static_preflight_v11(
        PROJECT_ROOT,
        source_closure_binding=_binding(PROJECT_ROOT / SOURCE_CLOSURE_RELATIVE_PATH),
        controller_binding=_binding(Path(__file__).resolve()),
        supervisor_binding=_binding(PROJECT_ROOT / SUPERVISOR_RELATIVE_PATH),
        phase=phase,
    )


def _launch_id() -> str:
    launch_id = os.environ.get("AXON_LOOP167_V11_LAUNCH_ID", "")
    if len(launch_id) != 64 or any(character not in "0123456789abcdef" for character in launch_id):
        raise PhaseBContractError("v11 controller launch id is unavailable")
    return launch_id


def _sealed_static_bindings() -> dict[str, dict[str, str]]:
    return {
        "source_closure": _binding(PROJECT_ROOT / SOURCE_CLOSURE_RELATIVE_PATH),
        "execution_contract": _binding(PROJECT_ROOT / EXECUTION_CONTRACT_RELATIVE_PATH),
        "runtime_lock": _binding(PROJECT_ROOT / RUNTIME_LOCK_RELATIVE_PATH),
        "controller": _binding(PROJECT_ROOT / CONTROLLER_RELATIVE_PATH),
        "supervisor": _binding(PROJECT_ROOT / SUPERVISOR_RELATIVE_PATH),
        "loop166_windows_job": _binding(PROJECT_ROOT / LOOP166_WINDOWS_JOB_RELATIVE_PATH),
        "loop166_windows_process_lineage": _binding(
            PROJECT_ROOT / LOOP166_WINDOWS_PROCESS_LINEAGE_RELATIVE_PATH
        ),
    }


def run_execute() -> dict[str, Any]:
    bootstrap_thread_environment_v11()
    validate_current_runtime_invocation_v11(PROJECT_ROOT, role="controller", mode="execute")
    launch_id = _launch_id()
    expected_bindings = _sealed_static_bindings()
    launch_receipt_path = resolve_output_catalog_v11(PROJECT_ROOT, FIXED_OUTPUT_CATALOG)[
        "supervisor_launch_receipt"
    ]
    child_attestation = build_child_job_attestation_payload_v11(
        PROJECT_ROOT,
        launch_receipt_path=launch_receipt_path,
        launch_id=launch_id,
        expected_bindings=expected_bindings,
    )
    write_child_job_attestation_v11(PROJECT_ROOT, child_attestation)
    static_receipt = validate_static_preflight_v11(
        PROJECT_ROOT,
        source_closure_binding=expected_bindings["source_closure"],
        controller_binding=expected_bindings["controller"],
        supervisor_binding=expected_bindings["supervisor"],
        phase="attested_child",
    )
    authorization = validate_execution_authorization_v11(
        PROJECT_ROOT,
        PROJECT_ROOT / RUN_AUTHORIZATION_RELATIVE_PATH,
        now_utc=_utc_now(),
        phase="attested_child",
        launch_id=launch_id,
    )
    _assert_authorization_matches_static_preflight(authorization, static_receipt)
    contract = verify_execution_contract_v11(PROJECT_ROOT, authorization.execution_contract_binding)
    consumed_lease = consume_execution_lease_v11(
        PROJECT_ROOT,
        PROJECT_ROOT / RUN_AUTHORIZATION_RELATIVE_PATH,
        now_utc=_utc_now(),
        launch_id=launch_id,
    )
    if consumed_lease.authorization_sha256 != authorization.authorization_sha256:
        raise PhaseBContractError("v11 consumed lease authorization drifted")
    lease = verify_consumed_execution_lease_v11(
        PROJECT_ROOT,
        authorization,
        launch_id=launch_id,
        now_utc=_utc_now(),
    )
    post_lease_static_receipt = validate_static_preflight_v11(
        PROJECT_ROOT,
        source_closure_binding=expected_bindings["source_closure"],
        controller_binding=expected_bindings["controller"],
        supervisor_binding=expected_bindings["supervisor"],
        phase="leased_child_pre_raw",
    )
    _assert_static_preflight_is_unchanged(static_receipt, post_lease_static_receipt)
    if sha256_file(lease.marker_path) != lease.marker_sha256:
        raise PhaseBContractError("v11 lease marker drifted after post-lease static preflight")
    raw_root = _resolve_fixed_raw_root_adapter(PROJECT_ROOT, RAW_ROOT_RELATIVE_PATH)

    # 租约消耗后才导入并验证数值运行时，避免 prelease 进程扩大数据面。
    from loop167_phase_b.runtime_lock_v11 import validate_runtime_lock_v11

    runtime_lock = require_canonical_json(PROJECT_ROOT / authorization.runtime_lock_binding["path"])
    validate_runtime_lock_v11(
        PROJECT_ROOT,
        runtime_lock,
        controller_binding=authorization.controller_binding,
        supervisor_binding=authorization.supervisor_binding,
        execution_contract_binding=authorization.execution_contract_binding,
        role="controller",
        mode="execute",
    )
    started_at = time.monotonic()
    receipt = _execute_after_lease(
        authorization=authorization,
        static_receipt=static_receipt,
        contract=contract,
        raw_root=raw_root,
        lease=lease,
        started_at=started_at,
    )
    _write_new_json(PROJECT_ROOT, authorization.output_paths["execution_receipt"], receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preflight:
        receipt = _static_preflight("preflight", phase="prelaunch")
        print(
            json.dumps(
                {
                    "schema": "axon_loop167_phase_b_static_preflight_receipt_v11",
                    "decision": "pass_static_preflight_raw_open_attempts_zero",
                    "source_closure_sha256": receipt.source_closure_sha256,
                    "execution_contract_sha256": receipt.execution_contract_sha256,
                    "runtime_lock_sha256": receipt.runtime_lock_sha256,
                    "raw_open_attempts": receipt.raw_open_attempts,
                },
                sort_keys=True,
            )
        )
        return 0
    receipt = run_execute()
    print(json.dumps({"status": receipt["status"], "raw_open_attempts": receipt["raw_open_attempts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
