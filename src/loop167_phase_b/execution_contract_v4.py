"""Immutable, raw-free execution contract for Loop167 Phase-B v4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .contracts import (
    PhaseBContractError,
    canonical_argv_sha256,
    canonical_json_bytes,
    require_canonical_json,
    resolve_project_file,
    sha256_bytes,
    verify_file_binding,
)
from .path_safety_v4 import canonical_project_relative_path, safe_project_path, safe_project_root

LOOP_ID = "loop167_ember_v3_novel_delta"
EXECUTION_CONTRACT_SCHEMA = "axon_loop167_phase_b_execution_contract_v4"

ARTIFACT_DIRECTORY = "manifests/roadmap_9997/loop167_ember_v3_novel_delta"
PHASE_B_PROTOCOL_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_protocol.json"
EXECUTION_CONTRACT_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_execution_contract_v4.json"
RUNTIME_LOCK_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_runtime_lock_v4.json"
SOURCE_CLOSURE_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_source_closure_v4.json"
RESOURCE_GUARD_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_resource_guard_v4.json"
RUN_AUTHORIZATION_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_run_authorization.json"
CONTROLLER_RELATIVE_PATH = "scripts/run_loop167_phase_b_controller_v4.py"
RAW_ROOT_RELATIVE_PATH = "data/random_20w_worktree"

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
    "single_train_only_raw_pass_then_fixed_oof_not_promotion_or_heldout_evaluation"
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

EXPECTED_LEASE = {
    "lease_id": "loop167-phase-b-train-oof-v1",
    "marker_path": "reports/roadmap_9997/loop167/phase_b_execution_consumed.json",
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
        "path": "reports/roadmap_9997/loop167/phase_b_feature_cache_v4.npz",
        "kind": "numeric_feature_cache",
    },
    {
        "name": "raw_progress_ledger",
        "path": "reports/roadmap_9997/loop167/phase_b_raw_progress_v4.jsonl",
        "kind": "append_only_raw_ledger",
    },
    {
        "name": "fit_progress_ledger",
        "path": "reports/roadmap_9997/loop167/phase_b_fit_progress_v4.jsonl",
        "kind": "append_only_fit_ledger",
    },
    {
        "name": "execution_receipt",
        "path": "reports/roadmap_9997/loop167/phase_b_execution_receipt_v4.json",
        "kind": "final_execution_receipt",
    },
)


@dataclass(frozen=True)
class VerifiedExecutionContractV4:
    """Resolved immutable inputs that downstream v4 gates may consume."""

    contract_path: Path
    contract_sha256: str
    protocol_sha256: str
    canonical_execute_argv: tuple[str, ...]
    resource_contract: Mapping[str, Any]
    output_catalog: tuple[Mapping[str, str], ...]
    output_paths: Mapping[str, Path]
    lease: Mapping[str, Any]
    raw_root_relative: str
    b1_sampling_indicators: Mapping[str, object]


def _require_fixed_binding_path(binding: object, *, label: str, expected_path: str) -> None:
    if not isinstance(binding, dict) or binding.get("path") != expected_path:
        raise PhaseBContractError(f"{label} path is outside the fixed v4 contract")


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PhaseBContractError(f"{label} must be an object")
    return value


def _validate_resource_contract(resource_contract: object) -> None:
    if resource_contract != EXPECTED_RESOURCE_CONTRACT:
        raise PhaseBContractError("Phase-B resource contract drifted from the sealed v4 budget")


def _validate_protocol(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "axon_loop167_phase_b_protocol_v1":
        raise PhaseBContractError("Phase-B protocol schema drifted")
    if payload.get("loop_id") != LOOP_ID or payload.get("claim_scope") != CLAIM_SCOPE:
        raise PhaseBContractError("Phase-B protocol identity or scope drifted")
    _validate_resource_contract(payload.get("resource_contract"))
    if payload.get("one_shot_lease") != EXPECTED_LEASE:
        raise PhaseBContractError("Phase-B one-shot lease contract drifted")
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


def resolve_output_catalog_v4(root: Path, catalog: object) -> Mapping[str, Path]:
    """Resolve only the sealed output paths, without opening or creating them."""

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


def assert_output_catalog_is_fresh_v4(root: Path, catalog: object) -> Mapping[str, Path]:
    """Refuse any prior output before a lease can be consumed."""

    paths = resolve_output_catalog_v4(root, catalog)
    for path in paths.values():
        if path.exists() or path.is_symlink():
            raise PhaseBContractError("A sealed v4 output already exists or is unsafe")
    return paths


def build_execution_contract_payload_v4(
    root: Path,
    *,
    protocol_binding: Mapping[str, str],
) -> dict[str, Any]:
    """Build the one-way protocol-to-contract artifact without runtime bindings."""

    _require_fixed_binding_path(
        protocol_binding,
        label="phase_b_protocol",
        expected_path=PHASE_B_PROTOCOL_RELATIVE_PATH,
    )
    _, protocol_sha256 = verify_file_binding(root, dict(protocol_binding), label="phase_b_protocol")
    protocol_path = resolve_project_file(root, protocol_binding["path"])
    _validate_protocol(require_canonical_json(protocol_path))
    return {
        "schema": EXECUTION_CONTRACT_SCHEMA,
        "loop_id": LOOP_ID,
        "phase_b_protocol": dict(protocol_binding),
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


def validate_execution_contract_payload_v4(
    root: Path,
    payload: Mapping[str, Any],
    *,
    expected_protocol_binding: Mapping[str, str] | None = None,
) -> None:
    """Validate the sealed contract before any resource or authorization gate."""

    expected_keys = {
        "schema",
        "loop_id",
        "phase_b_protocol",
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
        raise PhaseBContractError("Execution contract v4 schema drifted")
    if payload.get("loop_id") != LOOP_ID or payload.get("claim_scope") != CLAIM_SCOPE:
        raise PhaseBContractError("Execution contract v4 identity or scope drifted")
    if payload.get("execution_authorization_required") is not True:
        raise PhaseBContractError("Execution contract v4 must require run authorization")

    protocol_binding = payload["phase_b_protocol"]
    _require_fixed_binding_path(
        protocol_binding,
        label="phase_b_protocol",
        expected_path=PHASE_B_PROTOCOL_RELATIVE_PATH,
    )
    if expected_protocol_binding is not None and protocol_binding != dict(expected_protocol_binding):
        raise PhaseBContractError("Execution contract protocol binding drifted")
    protocol_path, protocol_sha256 = verify_file_binding(root, protocol_binding, label="phase_b_protocol")
    if payload["protocol_sha256"] != protocol_sha256:
        raise PhaseBContractError("Execution contract protocol digest drifted")
    _validate_protocol(require_canonical_json(protocol_path))

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
    resolve_output_catalog_v4(root, payload["output_catalog"])


def verify_execution_contract_v4(
    root: Path,
    contract_binding: Mapping[str, str],
    *,
    expected_protocol_binding: Mapping[str, str] | None = None,
) -> VerifiedExecutionContractV4:
    """Verify a fixed-path v4 contract and return its resolved safe values."""

    _require_fixed_binding_path(
        contract_binding,
        label="execution_contract",
        expected_path=EXECUTION_CONTRACT_RELATIVE_PATH,
    )
    contract_path, contract_sha256 = verify_file_binding(
        root,
        dict(contract_binding),
        label="execution_contract",
    )
    payload = require_canonical_json(contract_path)
    validate_execution_contract_payload_v4(
        root,
        payload,
        expected_protocol_binding=expected_protocol_binding,
    )
    protocol_sha256 = str(payload["protocol_sha256"])
    return VerifiedExecutionContractV4(
        contract_path=contract_path,
        contract_sha256=contract_sha256,
        protocol_sha256=protocol_sha256,
        canonical_execute_argv=tuple(payload["canonical_execute_argv"]),
        resource_contract=MappingProxyType(dict(EXPECTED_RESOURCE_CONTRACT)),
        output_catalog=_validate_output_catalog(payload["output_catalog"]),
        output_paths=resolve_output_catalog_v4(root, payload["output_catalog"]),
        lease=MappingProxyType(dict(EXPECTED_LEASE)),
        raw_root_relative=RAW_ROOT_RELATIVE_PATH,
        b1_sampling_indicators=MappingProxyType(dict(B1_SAMPLING_INDICATORS_CONTRACT)),
    )


def canonical_execute_argv_v4(argv: Sequence[str]) -> tuple[str, ...]:
    """Reject every execute command except the exact sealed v4 argv."""

    normalized = tuple(argv)
    if normalized != CANONICAL_EXECUTE_ARGV:
        raise PhaseBContractError("Execution argv is not the sealed v4 command")
    return normalized
