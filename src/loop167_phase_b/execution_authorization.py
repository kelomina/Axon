"""Fail-closed execution-authorization validation for Loop167 Phase B."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import (
    PhaseBContractError,
    canonical_argv_sha256,
    require_canonical_json,
    resolve_project_file,
    sha256_file,
    verify_file_binding,
)
from .resource_guard import validate_resource_contract

AUTHORIZATION_SCHEMA = "axon_loop167_phase_b_run_authorization_v1"
RESOURCE_GUARD_SCHEMA = "axon_loop167_phase_b_resource_guard_v1"


@dataclass(frozen=True)
class VerifiedExecutionAuthorization:
    authorization_sha256: str
    lease_marker: Path
    output_paths: tuple[Path, ...]
    source_closure_sha256: str
    runtime_lock_sha256: str
    resource_guard_sha256: str


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PhaseBContractError(f"{field} must be a UTC Z timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as exc:
        raise PhaseBContractError(f"{field} is invalid") from exc


def _require_paths(root: Path, values: object, *, field: str) -> tuple[Path, ...]:
    if not isinstance(values, list) or not values or any(not isinstance(value, str) for value in values):
        raise PhaseBContractError(f"{field} must be a nonempty list of project-relative paths")
    paths = tuple(resolve_project_file(root, value) for value in values)
    if len(paths) != len(set(paths)):
        raise PhaseBContractError(f"{field} repeats an output path")
    return paths


def _validate_resource_guard(
    root: Path,
    binding: Mapping[str, str],
    *,
    expected_source_closure: Mapping[str, str],
    expected_runtime_lock: Mapping[str, str],
    expected_controller: Mapping[str, str],
    canonical_argv: Sequence[str],
    now_utc: datetime,
) -> str:
    guard_path, guard_sha256 = verify_file_binding(root, dict(binding), label="resource_guard")
    guard = require_canonical_json(guard_path)
    expected_keys = {
        "schema",
        "loop_id",
        "source_closure",
        "phase_b_protocol",
        "runtime_lock",
        "controller",
        "canonical_argv",
        "canonical_argv_sha256",
        "resource_contract",
        "created_at_utc",
        "maximum_age_seconds",
        "snapshot",
        "minimum_available_memory_bytes",
        "guard_ready",
        "failures",
        "decision",
        "raw_open_attempts",
    }
    if set(guard) != expected_keys or guard["schema"] != RESOURCE_GUARD_SCHEMA:
        raise PhaseBContractError("Resource guard schema drifted")
    if guard["loop_id"] != "loop167_ember_v3_novel_delta":
        raise PhaseBContractError("Resource guard loop id drifted")
    if guard["source_closure"] != dict(expected_source_closure):
        raise PhaseBContractError("Resource guard source closure binding drifted")
    if guard["runtime_lock"] != dict(expected_runtime_lock):
        raise PhaseBContractError("Resource guard runtime lock binding drifted")
    if guard["controller"] != dict(expected_controller):
        raise PhaseBContractError("Resource guard controller binding drifted")
    if guard["canonical_argv"] != list(canonical_argv):
        raise PhaseBContractError("Resource guard argv drifted")
    if guard["canonical_argv_sha256"] != canonical_argv_sha256(canonical_argv):
        raise PhaseBContractError("Resource guard argv hash drifted")
    validate_resource_contract(guard["resource_contract"])
    created_at = _parse_utc(guard["created_at_utc"], field="resource_guard.created_at_utc")
    max_age = guard["maximum_age_seconds"]
    if isinstance(max_age, bool) or not isinstance(max_age, int) or max_age < 1:
        raise PhaseBContractError("Resource guard maximum age drifted")
    if now_utc < created_at or (now_utc - created_at).total_seconds() > max_age:
        raise PhaseBContractError("Resource guard is stale")
    if guard["guard_ready"] is not True or guard["decision"] != "pass" or guard["failures"] != []:
        raise PhaseBContractError("Resource guard is not ready")
    if guard["raw_open_attempts"] != 0:
        raise PhaseBContractError("Resource guard has an unexpected raw-open count")
    return guard_sha256


def validate_execution_authorization(
    root: Path,
    authorization_path: Path,
    *,
    expected_source_closure: Mapping[str, str],
    expected_runtime_lock: Mapping[str, str],
    expected_controller: Mapping[str, str],
    canonical_argv: Sequence[str],
    now_utc: datetime,
) -> VerifiedExecutionAuthorization:
    authorization = require_canonical_json(authorization_path)
    expected_keys = {
        "schema",
        "loop_id",
        "claim_scope",
        "status",
        "execution_authorization_granted",
        "source_closure",
        "runtime_lock",
        "controller",
        "canonical_argv",
        "canonical_argv_sha256",
        "resource_guard",
        "lease",
        "outputs",
        "ready_for",
        "forbidden",
    }
    if set(authorization) != expected_keys or authorization["schema"] != AUTHORIZATION_SCHEMA:
        raise PhaseBContractError("Execution authorization schema drifted")
    if authorization["loop_id"] != "loop167_ember_v3_novel_delta":
        raise PhaseBContractError("Execution authorization loop id drifted")
    if authorization["claim_scope"] != "single_train_only_raw_pass_then_fixed_oof_not_promotion_or_heldout_evaluation":
        raise PhaseBContractError("Execution authorization claim scope drifted")
    if authorization["status"] != "authorized_pending_one_shot_lease" or authorization["execution_authorization_granted"] is not True:
        raise PhaseBContractError("Execution authorization is not granted")
    if authorization["source_closure"] != dict(expected_source_closure):
        raise PhaseBContractError("Execution authorization source closure binding drifted")
    if authorization["runtime_lock"] != dict(expected_runtime_lock):
        raise PhaseBContractError("Execution authorization runtime lock binding drifted")
    if authorization["controller"] != dict(expected_controller):
        raise PhaseBContractError("Execution authorization controller binding drifted")
    if authorization["canonical_argv"] != list(canonical_argv):
        raise PhaseBContractError("Execution authorization argv drifted")
    if authorization["canonical_argv_sha256"] != canonical_argv_sha256(canonical_argv):
        raise PhaseBContractError("Execution authorization argv hash drifted")
    resource_guard_sha256 = _validate_resource_guard(
        root,
        authorization["resource_guard"],
        expected_source_closure=expected_source_closure,
        expected_runtime_lock=expected_runtime_lock,
        expected_controller=expected_controller,
        canonical_argv=canonical_argv,
        now_utc=now_utc,
    )

    lease = authorization["lease"]
    if not isinstance(lease, dict) or set(lease) != {"lease_id", "marker_path", "consume_before_first_raw_open", "retry_allowed"}:
        raise PhaseBContractError("Execution authorization lease fields drifted")
    if lease["lease_id"] != "loop167-phase-b-train-oof-v1":
        raise PhaseBContractError("Execution authorization lease id drifted")
    if lease["consume_before_first_raw_open"] is not True or lease["retry_allowed"] is not False:
        raise PhaseBContractError("Execution authorization lease semantics drifted")
    lease_marker = resolve_project_file(root, lease["marker_path"])
    if lease_marker.exists() or lease_marker.is_symlink():
        raise PhaseBContractError("Execution lease marker already exists or is unsafe")

    output_paths = _require_paths(root, authorization["outputs"], field="outputs")
    for output_path in output_paths:
        if output_path.exists() or output_path.is_symlink():
            raise PhaseBContractError("Execution output already exists or is unsafe")
    if lease_marker in output_paths:
        raise PhaseBContractError("Execution lease marker may not be an output artifact")
    if authorization["ready_for"] != {"raw_access": True, "fit": True, "val": False, "test10k": False, "legacy_full_test": False, "promotion": False}:
        raise PhaseBContractError("Execution authorization ready state drifted")
    forbidden = authorization["forbidden"]
    if not isinstance(forbidden, list) or "val_test10k_legacy_full_sentinel_or_sealed_window_access" not in forbidden:
        raise PhaseBContractError("Execution authorization forbidden scope drifted")
    return VerifiedExecutionAuthorization(
        authorization_sha256=sha256_file(authorization_path),
        lease_marker=lease_marker,
        output_paths=output_paths,
        source_closure_sha256=str(expected_source_closure["sha256"]),
        runtime_lock_sha256=str(expected_runtime_lock["sha256"]),
        resource_guard_sha256=resource_guard_sha256,
    )
