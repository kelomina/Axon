"""Immutable, raw-free replacement execution contract for Loop167 Phase-B v5."""

from __future__ import annotations

import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Sequence

from .contracts import (
    PhaseBContractError,
    canonical_argv_sha256,
    canonical_json_bytes,
    require_canonical_json,
    resolve_project_file,
    sha256_bytes,
    sha256_file,
    verify_file_binding,
)
from .path_safety_v4 import (
    canonical_project_relative_path,
    safe_project_path,
    safe_project_relative_path,
    safe_project_root,
)

LOOP_ID = "loop167_ember_v3_novel_delta"
EXECUTION_CONTRACT_SCHEMA = "axon_loop167_phase_b_execution_contract_v5"

PARENT_ARTIFACT_DIRECTORY = "manifests/roadmap_9997/loop167_ember_v3_novel_delta"
ARTIFACT_DIRECTORY = f"{PARENT_ARTIFACT_DIRECTORY}/phase_b_v5_mappingproxy_remediation"
PHASE_B_PROTOCOL_RELATIVE_PATH = f"{PARENT_ARTIFACT_DIRECTORY}/phase_b_protocol.json"
EXECUTION_CONTRACT_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_execution_contract_v5.json"
RUNTIME_LOCK_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_runtime_lock_v5.json"
SOURCE_CLOSURE_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_source_closure_v5.json"
RESOURCE_GUARD_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_resource_guard_v5.json"
RUN_AUTHORIZATION_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_run_authorization_v5.json"
PARENT_V4_PRELEASE_ATTESTATION_RELATIVE_PATH = (
    f"{ARTIFACT_DIRECTORY}/phase_b_v4_prelease_attestation.json"
)
CONTROLLER_RELATIVE_PATH = "scripts/run_loop167_phase_b_controller_v5.py"
RAW_ROOT_RELATIVE_PATH = "data/random_20w_worktree"

PARENT_V4_EXECUTION_CONTRACT_RELATIVE_PATH = f"{PARENT_ARTIFACT_DIRECTORY}/phase_b_execution_contract_v4.json"
PARENT_V4_RUNTIME_LOCK_RELATIVE_PATH = f"{PARENT_ARTIFACT_DIRECTORY}/phase_b_runtime_lock_v4.json"
PARENT_V4_SOURCE_CLOSURE_RELATIVE_PATH = f"{PARENT_ARTIFACT_DIRECTORY}/phase_b_source_closure_v4.json"
PARENT_V4_RESOURCE_GUARD_RELATIVE_PATH = f"{PARENT_ARTIFACT_DIRECTORY}/phase_b_resource_guard_v4.json"
PARENT_V4_RUN_AUTHORIZATION_RELATIVE_PATH = f"{PARENT_ARTIFACT_DIRECTORY}/phase_b_run_authorization.json"
PARENT_V4_EXECUTION_LEASE_MARKER_RELATIVE_PATH = (
    "reports/roadmap_9997/loop167/phase_b_execution_consumed.json"
)

PARENT_V4_PRELEASE_ATTESTATION_SCHEMA = "axon_loop167_phase_b_v4_prelease_attestation_v1"
PARENT_V4_PRELEASE_FAILURE_STAGE = (
    "execution_contract_mappingproxy_boundary_before_raw_root_job_and_lease"
)
PARENT_V4_PRELEASE_REPLACEMENT_SCOPE = "one_new_v5_chain_only"

B1_SAMPLING_INDICATORS_CONTRACT: dict[str, object] = {
    "dimension": 3,
    "role": "audit_only_not_in_fit_cache",
    "receipt_key": "sampling_audit",
}

CANONICAL_EXECUTE_ARGV = (
    "vnev/Scripts/python.exe",
    "-I",
    CONTROLLER_RELATIVE_PATH,
    "--execute",
)

CLAIM_SCOPE = "local_train_only_structural_delta_diagnostic_not_model_quality_promotion_or_full_test"
AUTHORIZATION_CLAIM_SCOPE = (
    "single_replacement_v5_train_only_raw_pass_then_fixed_oof_not_promotion_or_heldout_evaluation"
)

EXPECTED_RESOURCE_CONTRACT: dict[str, Any] = {
    "maximum_raw_open_attempts": 20000,
    "maximum_raw_bytes": 26843545600,
    "maximum_feature_cache_bytes": 1073741824,
    "maximum_extraction_peak_rss_bytes": 4294967296,
    "maximum_training_peak_rss_bytes": 8589934592,
    "maximum_extraction_wall_seconds": 6000,
    "maximum_training_wall_seconds": 18000,
    "reserved_seal_evaluation_wall_seconds": 4800,
    "maximum_total_wall_seconds": 28800,
    "worker_count": 1,
    "thread_count": 1,
    "maximum_gpu_allocated_bytes": 0,
    "kill_conditions": [
        "oom",
        "nonfinite",
        "timeout",
        "unrecoverable_feature_cache",
        "second_raw_pass_requested",
        "source_or_scope_drift",
        "replay_hash_mismatch",
    ],
}

PARENT_V4_EXPECTED_LEASE = {
    "lease_id": "loop167-phase-b-train-oof-v1",
    "marker_path": PARENT_V4_EXECUTION_LEASE_MARKER_RELATIVE_PATH,
    "consume_before_first_raw_open": True,
    "failed_attempt_consumes_lease": True,
    "retry_resume_or_rescan_allowed": False,
}
EXPECTED_LEASE = {
    "lease_id": "loop167-phase-b-v5-mappingproxy-remediation-train-oof-v1",
    "marker_path": (
        "reports/roadmap_9997/loop167/phase_b_v5_mappingproxy_remediation/"
        "phase_b_execution_consumed_v5.json"
    ),
    "consume_before_first_raw_open": True,
    "failed_attempt_consumes_lease": True,
    "retry_resume_or_rescan_allowed": False,
}

EXPECTED_FORBIDDEN = [
    "val_test10k_legacy_full_sentinel_or_sealed_window_access",
    "threshold_search",
    "hyperparameter_search",
    "path_extension_directory_sha_row_fold_or_label_feature",
    "loop151_loop69_loop164_prediction_or_score_input",
    "unlocked_authenticode_parser_or_public_key_requirement",
    "same_lease_retry",
]

FIXED_OUTPUT_CATALOG: tuple[dict[str, str], ...] = (
    {
        "name": "feature_cache",
        "path": "reports/roadmap_9997/loop167/phase_b_v5_mappingproxy_remediation/phase_b_feature_cache_v5.npz",
        "kind": "numeric_feature_cache",
    },
    {
        "name": "raw_progress_ledger",
        "path": "reports/roadmap_9997/loop167/phase_b_v5_mappingproxy_remediation/phase_b_raw_progress_v5.jsonl",
        "kind": "append_only_raw_ledger",
    },
    {
        "name": "fit_progress_ledger",
        "path": "reports/roadmap_9997/loop167/phase_b_v5_mappingproxy_remediation/phase_b_fit_progress_v5.jsonl",
        "kind": "append_only_fit_ledger",
    },
    {
        "name": "execution_receipt",
        "path": "reports/roadmap_9997/loop167/phase_b_v5_mappingproxy_remediation/phase_b_execution_receipt_v5.json",
        "kind": "final_execution_receipt",
    },
)

PARENT_V4_OUTPUT_PATHS = (
    "reports/roadmap_9997/loop167/phase_b_feature_cache_v4.npz",
    "reports/roadmap_9997/loop167/phase_b_raw_progress_v4.jsonl",
    "reports/roadmap_9997/loop167/phase_b_fit_progress_v4.jsonl",
    "reports/roadmap_9997/loop167/phase_b_execution_receipt_v4.json",
)
PARENT_V4_REQUIRED_SOURCE_PATHS = frozenset(
    {
        "scripts/run_loop167_phase_b_controller_v4.py",
        "src/loop167_phase_b/execution_contract_v4.py",
        "src/loop167_phase_b/execution_authorization_v4.py",
        "src/loop167_phase_b/lease_v4.py",
        "src/loop167_phase_b/raw_worker.py",
    }
)


@dataclass(frozen=True)
class VerifiedParentV4PreleaseAttestationV5:
    """The sealed evidence that v4 failed before consuming its execution authority."""

    attestation_path: Path
    attestation_sha256: str
    parent_execution_contract_binding: Mapping[str, str]
    parent_runtime_lock_binding: Mapping[str, str]
    parent_source_closure_binding: Mapping[str, str]
    parent_resource_guard_binding: Mapping[str, str]
    parent_run_authorization_binding: Mapping[str, str]


@dataclass(frozen=True)
class VerifiedExecutionContractV5:
    """Resolved immutable inputs that downstream v5 gates may consume."""

    contract_path: Path
    contract_sha256: str
    protocol_sha256: str
    parent_v4_prelease_attestation_sha256: str
    canonical_execute_argv: tuple[str, ...]
    resource_contract: Mapping[str, Any]
    output_catalog: tuple[Mapping[str, str], ...]
    output_paths: Mapping[str, Path]
    lease: Mapping[str, Any]
    raw_root_relative: str
    b1_sampling_indicators: Mapping[str, object]


def _fixed_binding_snapshot(
    binding: object,
    *,
    label: str,
    expected_path: str,
) -> dict[str, str]:
    """Take one stable mapping snapshot before handing the legacy verifier a dict."""

    if not isinstance(binding, Mapping):
        raise PhaseBContractError(f"{label} binding must be a mapping")
    snapshot = dict(binding)
    if set(snapshot) != {"path", "sha256"}:
        raise PhaseBContractError(f"{label} binding must contain exactly path and sha256")
    if snapshot["path"] != expected_path:
        raise PhaseBContractError(f"{label} path is outside the fixed v5 contract")
    return snapshot


def ensure_v5_static_artifact_parent(root: Path | str, relative_path: str) -> Path:
    """Create only the fixed v5 static-artifact ancestry without following links."""

    root_path = safe_project_root(root)
    canonical_path = canonical_project_relative_path(relative_path)
    if not canonical_path.startswith(f"{ARTIFACT_DIRECTORY}/"):
        raise PhaseBContractError("Static artifact path is outside the fixed v5 artifact root")
    cursor = root_path
    for component in canonical_path.split("/")[:-1]:
        cursor = cursor / component
        try:
            cursor.mkdir(exist_ok=True)
            stat_result = cursor.lstat()
        except OSError as exc:
            raise PhaseBContractError("Static artifact parent cannot be prepared safely") from exc
        attributes = int(getattr(stat_result, "st_file_attributes", 0))
        if stat.S_ISLNK(stat_result.st_mode) or bool(attributes & 0x0400) or not stat.S_ISDIR(stat_result.st_mode):
            raise PhaseBContractError("Static artifact parent is unsafe")
    return safe_project_path(root_path, canonical_path, require_exists=False)


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PhaseBContractError(f"{label} must be an object")
    return value


def _validate_resource_contract(resource_contract: object) -> None:
    if resource_contract != EXPECTED_RESOURCE_CONTRACT:
        raise PhaseBContractError("Phase-B resource contract drifted from the sealed v5 budget")


def _validate_protocol(payload: Mapping[str, Any]) -> None:
    """Keep the immutable v4 protocol as provenance, not as v5 execution authority."""

    if payload.get("schema") != "axon_loop167_phase_b_protocol_v1":
        raise PhaseBContractError("Phase-B protocol schema drifted")
    if payload.get("loop_id") != LOOP_ID or payload.get("claim_scope") != CLAIM_SCOPE:
        raise PhaseBContractError("Phase-B protocol identity or scope drifted")
    _validate_resource_contract(payload.get("resource_contract"))
    if payload.get("one_shot_lease") != PARENT_V4_EXPECTED_LEASE:
        raise PhaseBContractError("Parent Phase-B one-shot lease provenance drifted")
    if payload.get("forbidden") != EXPECTED_FORBIDDEN:
        raise PhaseBContractError("Phase-B forbidden-scope contract drifted")
    if payload.get("ready_for") != {
        "static_source_closure": True,
        "raw_access": False,
        "fit": False,
        "val": False,
        "test10k": False,
        "legacy_full_test": False,
        "promotion": False,
    }:
        raise PhaseBContractError("Phase-B protocol readiness drifted")

    input_contract = _require_mapping(payload.get("input_contract"), label="input_contract")
    folds = _require_mapping(input_contract.get("folds"), label="input_contract.folds")
    if (
        folds.get("split_role") != "train"
        or folds.get("rows") != 20000
        or folds.get("folds") != 5
        or folds.get("rows_per_fold") != 4000
        or folds.get("val_test_or_full_access") is not False
    ):
        raise PhaseBContractError("Phase-B train-only fold contract drifted")

    fit_contract = _require_mapping(payload.get("fit_contract"), label="fit_contract")
    estimator = _require_mapping(fit_contract.get("estimator"), label="fit_contract.estimator")
    if (
        fit_contract.get("arms") != ["B0", "B1", "M", "A", "CF"]
        or fit_contract.get("outer_folds") != [0, 1, 2, 3, 4]
        or fit_contract.get("replay_seeds") != [41, 42, 43]
        or fit_contract.get("maximum_total_fits") != 75
        or estimator.get("threshold") != 0.5
        or estimator.get("threshold_search_allowed") is not False
        or estimator.get("hyperparameter_search_allowed") is not False
    ):
        raise PhaseBContractError("Phase-B fixed fitting contract drifted")


def output_catalog_sha256(catalog: object = FIXED_OUTPUT_CATALOG) -> str:
    """Hash the exact catalog rather than accepting caller-selected outputs."""

    if isinstance(catalog, tuple):
        catalog = list(catalog)
    return sha256_bytes(canonical_json_bytes({"output_catalog": catalog}))


def _validate_output_catalog(catalog: object) -> tuple[Mapping[str, str], ...]:
    if catalog != list(FIXED_OUTPUT_CATALOG):
        raise PhaseBContractError("Output catalog drifted or contains an arbitrary output")
    return tuple(MappingProxyType(dict(entry)) for entry in FIXED_OUTPUT_CATALOG)


def resolve_output_catalog_v5(root: Path, catalog: object) -> Mapping[str, Path]:
    """Resolve only the sealed v5 output paths, without opening or creating them."""

    validated_catalog = _validate_output_catalog(catalog)
    root_path = safe_project_root(root)
    paths: dict[str, Path] = {}
    for entry in validated_catalog:
        name = entry["name"]
        relative_path = canonical_project_relative_path(entry["path"])
        path = safe_project_path(root_path, relative_path, require_exists=False)
        if name in paths:
            raise PhaseBContractError("Output catalog repeats an output name")
        paths[name] = path
    return MappingProxyType(paths)


def assert_output_catalog_is_fresh_v5(root: Path, catalog: object) -> Mapping[str, Path]:
    """Refuse any prior v5 output before a replacement lease can be consumed."""

    paths = resolve_output_catalog_v5(root, catalog)
    for path in paths.values():
        if path.exists() or path.is_symlink():
            raise PhaseBContractError("A sealed v5 output already exists or is unsafe")
    return paths


def _parent_v4_binding(root: Path, relative_path: str) -> dict[str, str]:
    path = safe_project_path(root, relative_path, require_exists=True, require_regular_file=True)
    return {"path": relative_path, "sha256": sha256_file(path)}


def _verify_parent_v4_artifact(
    root: Path,
    binding: object,
    *,
    label: str,
    expected_path: str,
    expected_schema: str,
) -> tuple[dict[str, str], Mapping[str, Any], str]:
    snapshot = _fixed_binding_snapshot(binding, label=label, expected_path=expected_path)
    safe_project_path(root, expected_path, require_exists=True, require_regular_file=True)
    path, sha256 = verify_file_binding(root, snapshot, label=label)
    payload = require_canonical_json(path)
    if payload.get("schema") != expected_schema or payload.get("loop_id") != LOOP_ID:
        raise PhaseBContractError(f"{label} canonical parent artifact drifted")
    return snapshot, payload, sha256


def _assert_parent_v4_prelease_absence(root: Path) -> None:
    marker_path = safe_project_path(
        root,
        PARENT_V4_EXECUTION_LEASE_MARKER_RELATIVE_PATH,
        require_exists=False,
    )
    if marker_path.exists() or marker_path.is_symlink():
        raise PhaseBContractError("The v4 lease marker exists or is unsafe")
    for relative_path in PARENT_V4_OUTPUT_PATHS:
        output_path = safe_project_path(root, relative_path, require_exists=False)
        if output_path.exists() or output_path.is_symlink():
            raise PhaseBContractError("A v4 output exists or is unsafe")


def _validate_parent_v4_source_files(root: Path, source_closure: Mapping[str, Any]) -> None:
    """Recheck every sealed v4 source hash without loading its raw-processing code."""

    source_files = source_closure.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise PhaseBContractError("Parent v4 source closure has no source-file bindings")
    observed_paths: set[str] = set()
    for binding in source_files:
        source_path, _ = verify_file_binding(root, binding, label="parent_v4_source_file")
        relative_path = safe_project_relative_path(
            root,
            source_path,
            require_exists=True,
            require_regular_file=True,
        )
        if relative_path in observed_paths:
            raise PhaseBContractError("Parent v4 source closure repeats a source-file binding")
        observed_paths.add(relative_path)
    if not PARENT_V4_REQUIRED_SOURCE_PATHS.issubset(observed_paths):
        raise PhaseBContractError("Parent v4 source closure omits required execution sources")


def _validate_parent_v4_prelease_payload(
    root: Path,
    payload: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    expected_keys = {
        "schema",
        "loop_id",
        "parent_execution_contract",
        "parent_runtime_lock",
        "parent_source_closure",
        "parent_resource_guard",
        "parent_run_authorization",
        "v4_raw_open_attempts",
        "v4_lease_marker_absent",
        "v4_output_catalog_absent",
        "failure_stage",
        "v4_prelease_fail_closed",
        "replacement_scope",
    }
    if set(payload) != expected_keys or payload.get("schema") != PARENT_V4_PRELEASE_ATTESTATION_SCHEMA:
        raise PhaseBContractError("Parent v4 pre-lease attestation schema drifted")
    if payload.get("loop_id") != LOOP_ID:
        raise PhaseBContractError("Parent v4 pre-lease attestation loop id drifted")
    if payload.get("failure_stage") != PARENT_V4_PRELEASE_FAILURE_STAGE:
        raise PhaseBContractError("Parent v4 pre-lease failure stage drifted")
    if payload.get("v4_prelease_fail_closed") is not True:
        raise PhaseBContractError("Parent v4 pre-lease attestation is not fail-closed")
    if payload.get("replacement_scope") != PARENT_V4_PRELEASE_REPLACEMENT_SCOPE:
        raise PhaseBContractError("Parent v4 replacement scope drifted")
    if payload.get("v4_raw_open_attempts") != 0:
        raise PhaseBContractError("Parent v4 pre-lease attestation records raw access")
    if payload.get("v4_lease_marker_absent") is not True or payload.get("v4_output_catalog_absent") is not True:
        raise PhaseBContractError("Parent v4 pre-lease absence facts drifted")

    parent_contract, parent_contract_payload, _ = _verify_parent_v4_artifact(
        root,
        payload.get("parent_execution_contract"),
        label="parent_v4_execution_contract",
        expected_path=PARENT_V4_EXECUTION_CONTRACT_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_execution_contract_v4",
    )
    parent_runtime_lock, parent_runtime_lock_payload, _ = _verify_parent_v4_artifact(
        root,
        payload.get("parent_runtime_lock"),
        label="parent_v4_runtime_lock",
        expected_path=PARENT_V4_RUNTIME_LOCK_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_runtime_lock_v4",
    )
    parent_source_closure, parent_source_closure_payload, _ = _verify_parent_v4_artifact(
        root,
        payload.get("parent_source_closure"),
        label="parent_v4_source_closure",
        expected_path=PARENT_V4_SOURCE_CLOSURE_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_source_closure_v4",
    )
    parent_resource_guard, parent_resource_guard_payload, _ = _verify_parent_v4_artifact(
        root,
        payload.get("parent_resource_guard"),
        label="parent_v4_resource_guard",
        expected_path=PARENT_V4_RESOURCE_GUARD_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_resource_guard_v4",
    )
    parent_run_authorization, parent_run_authorization_payload, _ = _verify_parent_v4_artifact(
        root,
        payload.get("parent_run_authorization"),
        label="parent_v4_run_authorization",
        expected_path=PARENT_V4_RUN_AUTHORIZATION_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_run_authorization_v4",
    )

    from .execution_contract_v4 import verify_execution_contract_v4

    verify_execution_contract_v4(root, parent_contract)
    _validate_parent_v4_source_files(root, parent_source_closure_payload)
    if parent_resource_guard_payload.get("raw_open_attempts") != 0:
        raise PhaseBContractError("Parent v4 resource guard records raw access")
    if parent_run_authorization_payload.get("raw_open_attempts") != 0:
        raise PhaseBContractError("Parent v4 run authorization records raw access")
    if parent_resource_guard_payload.get("phase_b_execution_contract") != parent_contract:
        raise PhaseBContractError("Parent v4 resource guard contract binding drifted")
    if parent_resource_guard_payload.get("source_closure") != parent_source_closure:
        raise PhaseBContractError("Parent v4 resource guard source-closure binding drifted")
    if parent_resource_guard_payload.get("runtime_lock") != parent_runtime_lock:
        raise PhaseBContractError("Parent v4 resource guard runtime-lock binding drifted")
    if parent_run_authorization_payload.get("phase_b_execution_contract") != parent_contract:
        raise PhaseBContractError("Parent v4 run authorization contract binding drifted")
    if parent_run_authorization_payload.get("source_closure") != parent_source_closure:
        raise PhaseBContractError("Parent v4 run authorization source-closure binding drifted")
    if parent_run_authorization_payload.get("runtime_lock") != parent_runtime_lock:
        raise PhaseBContractError("Parent v4 run authorization runtime-lock binding drifted")
    if parent_run_authorization_payload.get("resource_guard") != parent_resource_guard:
        raise PhaseBContractError("Parent v4 run authorization resource-guard binding drifted")
    if parent_run_authorization_payload.get("phase_b_protocol") != parent_contract_payload.get("phase_b_protocol"):
        raise PhaseBContractError("Parent v4 run authorization protocol provenance drifted")
    if parent_runtime_lock_payload.get("execution_contract") != parent_contract:
        raise PhaseBContractError("Parent v4 runtime lock contract binding drifted")
    _assert_parent_v4_prelease_absence(root)
    return (
        parent_contract,
        parent_runtime_lock,
        parent_source_closure,
        parent_resource_guard,
        parent_run_authorization,
    )


def build_parent_v4_prelease_attestation_payload_v5(root: Path | str) -> dict[str, Any]:
    """Build static proof that the v4 chain failed before lease consumption or raw access."""

    root_path = safe_project_root(root)
    payload = {
        "schema": PARENT_V4_PRELEASE_ATTESTATION_SCHEMA,
        "loop_id": LOOP_ID,
        "parent_execution_contract": _parent_v4_binding(root_path, PARENT_V4_EXECUTION_CONTRACT_RELATIVE_PATH),
        "parent_runtime_lock": _parent_v4_binding(root_path, PARENT_V4_RUNTIME_LOCK_RELATIVE_PATH),
        "parent_source_closure": _parent_v4_binding(root_path, PARENT_V4_SOURCE_CLOSURE_RELATIVE_PATH),
        "parent_resource_guard": _parent_v4_binding(root_path, PARENT_V4_RESOURCE_GUARD_RELATIVE_PATH),
        "parent_run_authorization": _parent_v4_binding(root_path, PARENT_V4_RUN_AUTHORIZATION_RELATIVE_PATH),
        "v4_raw_open_attempts": 0,
        "v4_lease_marker_absent": True,
        "v4_output_catalog_absent": True,
        "failure_stage": PARENT_V4_PRELEASE_FAILURE_STAGE,
        "v4_prelease_fail_closed": True,
        "replacement_scope": PARENT_V4_PRELEASE_REPLACEMENT_SCOPE,
    }
    _validate_parent_v4_prelease_payload(root_path, payload)
    return payload


def verify_parent_v4_prelease_attestation_v5(
    root: Path | str,
    binding: Mapping[str, str],
) -> VerifiedParentV4PreleaseAttestationV5:
    """Verify the sealed parent-failure attestation without refreshing old v4 authority."""

    root_path = safe_project_root(root)
    snapshot = _fixed_binding_snapshot(
        binding,
        label="parent_v4_prelease_attestation",
        expected_path=PARENT_V4_PRELEASE_ATTESTATION_RELATIVE_PATH,
    )
    safe_project_path(
        root_path,
        PARENT_V4_PRELEASE_ATTESTATION_RELATIVE_PATH,
        require_exists=True,
        require_regular_file=True,
    )
    attestation_path, attestation_sha256 = verify_file_binding(
        root_path,
        snapshot,
        label="parent_v4_prelease_attestation",
    )
    payload = require_canonical_json(attestation_path)
    bindings = _validate_parent_v4_prelease_payload(root_path, payload)
    return VerifiedParentV4PreleaseAttestationV5(
        attestation_path=attestation_path,
        attestation_sha256=attestation_sha256,
        parent_execution_contract_binding=MappingProxyType(dict(bindings[0])),
        parent_runtime_lock_binding=MappingProxyType(dict(bindings[1])),
        parent_source_closure_binding=MappingProxyType(dict(bindings[2])),
        parent_resource_guard_binding=MappingProxyType(dict(bindings[3])),
        parent_run_authorization_binding=MappingProxyType(dict(bindings[4])),
    )


def build_execution_contract_payload_v5(
    root: Path,
    *,
    protocol_binding: Mapping[str, str],
    parent_v4_prelease_attestation_binding: Mapping[str, str],
) -> dict[str, Any]:
    """Build the replacement contract from immutable parent provenance and v5-only authority."""

    protocol_snapshot = _fixed_binding_snapshot(
        protocol_binding,
        label="phase_b_protocol",
        expected_path=PHASE_B_PROTOCOL_RELATIVE_PATH,
    )
    _, protocol_sha256 = verify_file_binding(root, protocol_snapshot, label="phase_b_protocol")
    protocol_path = resolve_project_file(root, protocol_snapshot["path"])
    _validate_protocol(require_canonical_json(protocol_path))
    attestation = verify_parent_v4_prelease_attestation_v5(root, parent_v4_prelease_attestation_binding)
    return {
        "schema": EXECUTION_CONTRACT_SCHEMA,
        "loop_id": LOOP_ID,
        "phase_b_protocol": protocol_snapshot,
        "parent_v4_prelease_attestation": {
            "path": PARENT_V4_PRELEASE_ATTESTATION_RELATIVE_PATH,
            "sha256": attestation.attestation_sha256,
        },
        "claim_scope": CLAIM_SCOPE,
        "canonical_execute_argv": list(CANONICAL_EXECUTE_ARGV),
        "canonical_execute_argv_sha256": canonical_argv_sha256(CANONICAL_EXECUTE_ARGV),
        "resource_contract": dict(EXPECTED_RESOURCE_CONTRACT),
        "raw_root_relative": RAW_ROOT_RELATIVE_PATH,
        "b1_sampling_indicators": dict(B1_SAMPLING_INDICATORS_CONTRACT),
        "output_catalog": [dict(entry) for entry in FIXED_OUTPUT_CATALOG],
        "output_catalog_sha256": output_catalog_sha256(),
        "lease": dict(EXPECTED_LEASE),
        "forbidden": list(EXPECTED_FORBIDDEN),
        "execution_authorization_required": True,
        "protocol_sha256": protocol_sha256,
    }


def validate_execution_contract_payload_v5(
    root: Path,
    payload: Mapping[str, Any],
    *,
    expected_protocol_binding: Mapping[str, str] | None = None,
) -> None:
    """Validate the sealed v5 contract before any resource or authorization gate."""

    expected_keys = {
        "schema",
        "loop_id",
        "phase_b_protocol",
        "parent_v4_prelease_attestation",
        "claim_scope",
        "canonical_execute_argv",
        "canonical_execute_argv_sha256",
        "resource_contract",
        "raw_root_relative",
        "b1_sampling_indicators",
        "output_catalog",
        "output_catalog_sha256",
        "lease",
        "forbidden",
        "execution_authorization_required",
        "protocol_sha256",
    }
    if set(payload) != expected_keys or payload.get("schema") != EXECUTION_CONTRACT_SCHEMA:
        raise PhaseBContractError("Execution contract v5 schema drifted")
    if payload.get("loop_id") != LOOP_ID or payload.get("claim_scope") != CLAIM_SCOPE:
        raise PhaseBContractError("Execution contract v5 identity or scope drifted")
    if payload.get("execution_authorization_required") is not True:
        raise PhaseBContractError("Execution contract v5 must require run authorization")

    protocol_snapshot = _fixed_binding_snapshot(
        payload["phase_b_protocol"],
        label="phase_b_protocol",
        expected_path=PHASE_B_PROTOCOL_RELATIVE_PATH,
    )
    if expected_protocol_binding is not None:
        expected_snapshot = _fixed_binding_snapshot(
            expected_protocol_binding,
            label="expected_phase_b_protocol",
            expected_path=PHASE_B_PROTOCOL_RELATIVE_PATH,
        )
        if protocol_snapshot != expected_snapshot:
            raise PhaseBContractError("Execution contract protocol binding drifted")
    protocol_path, protocol_sha256 = verify_file_binding(root, protocol_snapshot, label="phase_b_protocol")
    if payload["protocol_sha256"] != protocol_sha256:
        raise PhaseBContractError("Execution contract protocol digest drifted")
    _validate_protocol(require_canonical_json(protocol_path))

    verify_parent_v4_prelease_attestation_v5(root, payload["parent_v4_prelease_attestation"])
    if payload["canonical_execute_argv"] != list(CANONICAL_EXECUTE_ARGV):
        raise PhaseBContractError("Execution contract execute argv drifted")
    if payload["canonical_execute_argv_sha256"] != canonical_argv_sha256(CANONICAL_EXECUTE_ARGV):
        raise PhaseBContractError("Execution contract execute argv hash drifted")
    _validate_resource_contract(payload["resource_contract"])
    if payload["raw_root_relative"] != RAW_ROOT_RELATIVE_PATH:
        raise PhaseBContractError("Execution contract raw root drifted")
    if canonical_project_relative_path(payload["raw_root_relative"]) != RAW_ROOT_RELATIVE_PATH:
        raise PhaseBContractError("Execution contract raw root is not canonical")
    if payload["b1_sampling_indicators"] != B1_SAMPLING_INDICATORS_CONTRACT:
        raise PhaseBContractError("Execution contract B1 sampling-audit contract drifted")
    _validate_output_catalog(payload["output_catalog"])
    if payload["output_catalog_sha256"] != output_catalog_sha256():
        raise PhaseBContractError("Execution contract output catalog hash drifted")
    if payload["lease"] != EXPECTED_LEASE:
        raise PhaseBContractError("Execution contract lease drifted")
    if payload["forbidden"] != EXPECTED_FORBIDDEN:
        raise PhaseBContractError("Execution contract forbidden scope drifted")
    resolve_output_catalog_v5(root, payload["output_catalog"])


def verify_execution_contract_v5(
    root: Path,
    contract_binding: Mapping[str, str],
    *,
    expected_protocol_binding: Mapping[str, str] | None = None,
) -> VerifiedExecutionContractV5:
    """Verify a fixed-path v5 contract, including MappingProxy-safe file bindings."""

    contract_snapshot = _fixed_binding_snapshot(
        contract_binding,
        label="execution_contract",
        expected_path=EXECUTION_CONTRACT_RELATIVE_PATH,
    )
    contract_path, contract_sha256 = verify_file_binding(
        root,
        contract_snapshot,
        label="execution_contract",
    )
    payload = require_canonical_json(contract_path)
    validate_execution_contract_payload_v5(
        root,
        payload,
        expected_protocol_binding=expected_protocol_binding,
    )
    protocol_sha256 = str(payload["protocol_sha256"])
    attestation_sha256 = str(payload["parent_v4_prelease_attestation"]["sha256"])
    return VerifiedExecutionContractV5(
        contract_path=contract_path,
        contract_sha256=contract_sha256,
        protocol_sha256=protocol_sha256,
        parent_v4_prelease_attestation_sha256=attestation_sha256,
        canonical_execute_argv=tuple(payload["canonical_execute_argv"]),
        resource_contract=MappingProxyType(dict(EXPECTED_RESOURCE_CONTRACT)),
        output_catalog=_validate_output_catalog(payload["output_catalog"]),
        output_paths=resolve_output_catalog_v5(root, payload["output_catalog"]),
        lease=MappingProxyType(dict(EXPECTED_LEASE)),
        raw_root_relative=RAW_ROOT_RELATIVE_PATH,
        b1_sampling_indicators=MappingProxyType(dict(B1_SAMPLING_INDICATORS_CONTRACT)),
    )


def canonical_execute_argv_v5(argv: Sequence[str]) -> tuple[str, ...]:
    """Reject every execute command except the exact sealed v5 argv."""

    normalized = tuple(argv)
    if normalized != CANONICAL_EXECUTE_ARGV:
        raise PhaseBContractError("Execution argv is not the sealed v5 command")
    return normalized
