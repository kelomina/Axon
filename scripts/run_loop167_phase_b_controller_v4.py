#!/usr/bin/env python3
"""Run the only sealed Loop167 Phase-B v4 controller route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
os.environ.update(THREAD_ENVIRONMENT)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loop167_phase_b.contracts import (  # noqa: E402
    PhaseBContractError,
    canonical_json_bytes,
    require_canonical_json,
    sha256_file,
)
from loop167_phase_b.execution_contract_v4 import (  # noqa: E402
    RAW_ROOT_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    verify_execution_contract_v4,
)
from loop167_phase_b.invocation_v4 import (  # noqa: E402
    bootstrap_thread_environment_v4,
    canonical_argv_v4,
    validate_current_runtime_invocation_v4,
)
from loop167_phase_b.preflight_v4 import validate_static_preflight_v4  # noqa: E402

ARTIFACT_ROOT = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta"
SOURCE_CLOSURE_PATH = PROJECT_ROOT / SOURCE_CLOSURE_RELATIVE_PATH
RUN_AUTHORIZATION_PATH = ARTIFACT_ROOT / "phase_b_run_authorization.json"
EXECUTION_RECEIPT_PATH = (
    PROJECT_ROOT / "reports" / "roadmap_9997" / "loop167" / "phase_b_execution_receipt_v4.json"
)
RECEIPT_SCHEMA = "axon_loop167_phase_b_execution_receipt_v4"
_ACTIVE_WINDOWS_JOB: Any | None = None


def _binding(path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _is_link_or_reparse(stat_result: os.stat_result) -> bool:
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    return stat.S_ISLNK(stat_result.st_mode) or bool(attributes & 0x0400)


def _resolve_fixed_raw_root_adapter(root: Path, raw_root_relative: str) -> Path:
    """Check the fixed data-root boundary without opening a manifest or raw file."""

    if raw_root_relative != RAW_ROOT_RELATIVE_PATH:
        raise PhaseBContractError("Execution contract raw root drifted")
    root = root.resolve(strict=True)
    candidate = root.joinpath(*raw_root_relative.split("/"))
    cursor = root
    for component in raw_root_relative.split("/"):
        cursor = cursor / component
        try:
            stat_result = cursor.lstat()
        except OSError as error:
            raise PhaseBContractError("Fixed raw root is unavailable before lease consumption") from error
        if _is_link_or_reparse(stat_result) or not stat.S_ISDIR(stat_result.st_mode):
            raise PhaseBContractError("Fixed raw root traverses an unsafe directory")
    if candidate != cursor:
        raise PhaseBContractError("Fixed raw root resolution drifted")
    return candidate


def _assign_current_process_to_job(memory_limit_bytes: int) -> None:
    """Create the hard process boundary before burning the one-shot lease."""

    global _ACTIVE_WINDOWS_JOB
    from loop167_phase_b.windows_job_v4 import WindowsJob

    job = WindowsJob.create(memory_limit_bytes=memory_limit_bytes)
    try:
        job.assign_current_process()
    except Exception:
        job.close()
        raise
    # 保持最后一个 Job handle 到进程退出，确保所有后代都留在 kill-on-close 边界内。
    _ACTIVE_WINDOWS_JOB = job


def _ensure_output_parent_after_lease(root: Path, output_path: Path) -> None:
    try:
        relative_parent = output_path.parent.relative_to(root)
    except ValueError as error:
        raise PhaseBContractError("Sealed output path escapes the project root") from error
    cursor = root
    for component in relative_parent.parts:
        cursor = cursor / component
        try:
            cursor.mkdir(exist_ok=True)
            stat_result = cursor.lstat()
        except OSError as error:
            raise PhaseBContractError("Sealed output parent cannot be prepared") from error
        if _is_link_or_reparse(stat_result) or not stat.S_ISDIR(stat_result.st_mode):
            raise PhaseBContractError("Sealed output parent is unsafe")


def _write_new_canonical_json(path: Path, payload: Mapping[str, Any]) -> str:
    content = canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(content).hexdigest()


def _assert_authorization_matches_static_preflight(authorization: Any, static_receipt: Any) -> None:
    expected = {
        "source_closure_binding": static_receipt.source_closure_binding,
        "execution_contract_binding": static_receipt.execution_contract_binding,
        "runtime_lock_binding": static_receipt.runtime_lock_binding,
        "controller_binding": static_receipt.controller_binding,
    }
    for name, expected_binding in expected.items():
        if dict(getattr(authorization, name)) != dict(expected_binding):
            raise PhaseBContractError(f"Execution authorization {name} drifted from static preflight")
    if tuple(authorization.canonical_execute_argv) != canonical_argv_v4("execute"):
        raise PhaseBContractError("Execution authorization execute argv drifted")


def _require_elapsed(started_at: float, maximum_seconds: int, *, phase: str) -> None:
    if time.monotonic() - started_at > maximum_seconds:
        raise PhaseBContractError(f"Phase-B {phase} wall-clock budget was exceeded")


def _execute_after_lease(
    *,
    authorization: Any,
    static_receipt: Any,
    contract: Any,
    raw_root: Path,
    lease: Any,
    started_at: float,
) -> dict[str, Any]:
    """Import numerical/raw modules only after a verified one-shot lease exists."""

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
    if set(outputs) != required_output_names:
        raise PhaseBContractError("Execution authorization output catalog drifted")
    for output_path in outputs.values():
        _ensure_output_parent_after_lease(PROJECT_ROOT, output_path)

    protocol = require_canonical_json(PROJECT_ROOT / authorization.protocol_binding["path"])
    raw_context_contract = protocol.get("feature_contract", {}).get("raw_context")
    if not isinstance(raw_context_contract, dict):
        raise PhaseBContractError("Phase-B raw-context contract is unavailable")
    maximum_source_file_bytes = raw_context_contract.get("maximum_source_file_bytes")
    reader_chunk_bytes = raw_context_contract.get("reader_chunk_bytes")
    if not isinstance(maximum_source_file_bytes, int) or not isinstance(reader_chunk_bytes, int):
        raise PhaseBContractError("Phase-B raw-context limits drifted")

    # 租约已复核后才读取 Train-only manifest；此处生成唯一 raw scan plan 与唯一 fit handoff。
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
        raise PhaseBContractError("Raw progress ledger did not close the sealed one-pass scan")
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
    sampling_contract = dict(contract.b1_sampling_indicators)
    if sampling_contract.get("receipt_key") != "sampling_audit":
        raise PhaseBContractError("Execution contract B1 sampling-audit receipt binding drifted")
    if len(cache_receipt.sampling_audit.indicator_counts) != sampling_contract.get("dimension"):
        raise PhaseBContractError("B1 sampling-audit dimension drifted from the execution contract")
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
        raise PhaseBContractError("Fit progress ledger did not close all fixed units")
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
        "status": "completed_single_train_only_raw_pass_fixed_oof_not_promotion_or_heldout_evaluation",
        "claim_scope": "single_train_only_raw_pass_then_fixed_oof_not_promotion_or_heldout_evaluation",
        "source_closure": dict(static_receipt.source_closure_binding),
        "phase_b_execution_contract": dict(authorization.execution_contract_binding),
        "runtime_lock": dict(authorization.runtime_lock_binding),
        "run_authorization": {
            "path": RUN_AUTHORIZATION_PATH.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": authorization.authorization_sha256,
        },
        "execution_lease": {
            "path": lease.marker_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": lease.marker_sha256,
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


def _static_preflight(mode: str) -> Any:
    bootstrap_thread_environment_v4()
    validate_current_runtime_invocation_v4(PROJECT_ROOT, mode=mode)
    return validate_static_preflight_v4(
        PROJECT_ROOT,
        source_closure_binding=_binding(SOURCE_CLOSURE_PATH),
        controller_binding=_binding(Path(__file__).resolve()),
        canonical_preflight_argv=canonical_argv_v4("preflight"),
    )


def run_execute() -> dict[str, Any]:
    static_receipt = _static_preflight("execute")

    from loop167_phase_b.execution_authorization_v4 import validate_execution_authorization_v4
    from loop167_phase_b.lease_v4 import (
        consume_execution_lease_v4,
        verify_consumed_execution_lease_v4,
    )

    now = _utc_now()
    authorization = validate_execution_authorization_v4(
        PROJECT_ROOT,
        RUN_AUTHORIZATION_PATH,
        now_utc=now,
    )
    _assert_authorization_matches_static_preflight(authorization, static_receipt)
    contract = verify_execution_contract_v4(
        PROJECT_ROOT,
        authorization.execution_contract_binding,
        expected_protocol_binding=authorization.protocol_binding,
    )
    raw_root = _resolve_fixed_raw_root_adapter(PROJECT_ROOT, contract.raw_root_relative)
    _assign_current_process_to_job(int(contract.resource_contract["maximum_training_peak_rss_bytes"]))

    consumed_lease = consume_execution_lease_v4(
        PROJECT_ROOT,
        RUN_AUTHORIZATION_PATH,
        now_utc=_utc_now(),
    )
    if consumed_lease.authorization_sha256 != authorization.authorization_sha256:
        raise PhaseBContractError("Consumed execution lease authorization drifted")
    lease = verify_consumed_execution_lease_v4(PROJECT_ROOT, authorization)

    # 运行时包的实际 hash/argv 验证可触发数值库导入，因此必须放在租约之后。
    from loop167_phase_b.runtime_lock_v4 import validate_runtime_lock_v4

    runtime_lock = require_canonical_json(PROJECT_ROOT / authorization.runtime_lock_binding["path"])
    validate_runtime_lock_v4(
        PROJECT_ROOT,
        runtime_lock,
        controller_binding=authorization.controller_binding,
        execution_contract_binding=authorization.execution_contract_binding,
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
    _write_new_canonical_json(authorization.output_paths["execution_receipt"], receipt)
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
        receipt = _static_preflight("preflight")
        print(
            json.dumps(
                {
                    "schema": "axon_loop167_phase_b_static_preflight_receipt_v4",
                    "decision": "pass_static_preflight_raw_open_attempts_zero",
                    "protocol_sha256": receipt.protocol_sha256,
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
    print(
        json.dumps(
            {
                "schema": RECEIPT_SCHEMA,
                "decision": "completed_single_train_only_raw_pass_fixed_oof",
                "execution_receipt_sha256": sha256_file(EXECUTION_RECEIPT_PATH),
                "raw_open_attempts": receipt["raw_open_attempts"],
                "heldout_access": False,
                "promotion": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
