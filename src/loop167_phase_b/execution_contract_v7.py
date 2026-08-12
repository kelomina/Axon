"""Independent v7 authority contract for the Windows Job ABI remediation."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .contracts import (
    PhaseBContractError,
    canonical_argv_sha256,
    canonical_json_bytes,
    require_canonical_json,
    sha256_bytes,
    sha256_file,
)
from .execution_contract_v5 import (
    B1_SAMPLING_INDICATORS_CONTRACT,
    EXPECTED_FORBIDDEN,
    EXPECTED_RESOURCE_CONTRACT,
)
from .path_safety_v4 import (
    canonical_project_relative_path,
    safe_project_path,
    safe_project_relative_path,
    safe_project_root,
    verify_safe_file_binding,
)

LOOP_ID = "loop167_ember_v3_novel_delta"
PARENT_ARTIFACT_DIRECTORY = "manifests/roadmap_9997/loop167_ember_v3_novel_delta"
ARTIFACT_DIRECTORY = f"{PARENT_ARTIFACT_DIRECTORY}/phase_b_v7_relative_controller_argv_remediation"
REPORT_DIRECTORY = "reports/roadmap_9997/loop167/phase_b_v7_relative_controller_argv_remediation"

PHASE_B_PROTOCOL_RELATIVE_PATH = f"{PARENT_ARTIFACT_DIRECTORY}/phase_b_protocol.json"
PARENT_V6_ARTIFACT_DIRECTORY = f"{PARENT_ARTIFACT_DIRECTORY}/phase_b_v6_windows_job_abi_remediation"
PARENT_V6_SOURCE_CLOSURE_RELATIVE_PATH = f"{PARENT_V6_ARTIFACT_DIRECTORY}/phase_b_source_closure_v6.json"
PARENT_V6_EXECUTION_CONTRACT_RELATIVE_PATH = f"{PARENT_V6_ARTIFACT_DIRECTORY}/phase_b_execution_contract_v6.json"
PARENT_V6_RUNTIME_LOCK_RELATIVE_PATH = f"{PARENT_V6_ARTIFACT_DIRECTORY}/phase_b_runtime_lock_v6.json"
PARENT_V6_RESOURCE_GUARD_RELATIVE_PATH = f"{PARENT_V6_ARTIFACT_DIRECTORY}/phase_b_resource_guard_v6.json"
PARENT_V6_RUN_AUTHORIZATION_RELATIVE_PATH = f"{PARENT_V6_ARTIFACT_DIRECTORY}/phase_b_run_authorization_v6.json"
PARENT_V6_EXECUTION_LEASE_RELATIVE_PATH = (
    "reports/roadmap_9997/loop167/phase_b_v6_windows_job_abi_remediation/"
    "phase_b_execution_consumed_v6.json"
)
PARENT_V6_OUTPUT_PATHS = (
    "reports/roadmap_9997/loop167/phase_b_v6_windows_job_abi_remediation/phase_b_supervisor_launch_v6.json",
    "reports/roadmap_9997/loop167/phase_b_v6_windows_job_abi_remediation/phase_b_supervisor_exit_v6.json",
    "reports/roadmap_9997/loop167/phase_b_v6_windows_job_abi_remediation/phase_b_supervisor_failure_v6.json",
    "reports/roadmap_9997/loop167/phase_b_v6_windows_job_abi_remediation/phase_b_child_job_attestation_v6.json",
    "reports/roadmap_9997/loop167/phase_b_v6_windows_job_abi_remediation/phase_b_feature_cache_v6.npz",
    "reports/roadmap_9997/loop167/phase_b_v6_windows_job_abi_remediation/phase_b_raw_progress_v6.jsonl",
    "reports/roadmap_9997/loop167/phase_b_v6_windows_job_abi_remediation/phase_b_fit_progress_v6.jsonl",
    "reports/roadmap_9997/loop167/phase_b_v6_windows_job_abi_remediation/phase_b_execution_receipt_v6.json",
)
PARENT_V6_EXPECTED_LEASE = {
    "lease_id": "loop167-phase-b-v6-windows-job-abi-remediation-train-oof-v1",
    "marker_path": PARENT_V6_EXECUTION_LEASE_RELATIVE_PATH,
    "consume_before_first_raw_open": True,
    "failed_attempt_consumes_lease": True,
    "retry_resume_or_rescan_allowed": False,
}

PARENT_V6_PRELEASE_ATTESTATION_RELATIVE_PATH = (
    f"{ARTIFACT_DIRECTORY}/phase_b_v6_relative_controller_argv_prelease_attestation.json"
)
EXECUTION_CONTRACT_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_execution_contract_v7.json"
SOURCE_CLOSURE_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_source_closure_v7.json"
RUNTIME_LOCK_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_runtime_lock_v7.json"
RESOURCE_GUARD_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_resource_guard_v7.json"
RUN_AUTHORIZATION_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_run_authorization_v7.json"

SUPERVISOR_RELATIVE_PATH = "scripts/run_loop167_phase_b_supervisor_v7.py"
CONTROLLER_RELATIVE_PATH = "scripts/run_loop167_phase_b_controller_v7.py"
VNEV_PYTHON_RELATIVE_PATH = "vnev/Scripts/python.exe"
RAW_ROOT_RELATIVE_PATH = "data/random_20w_worktree"
LOOP166_WINDOWS_JOB_RELATIVE_PATH = "src/loop166/windows_job.py"
LOOP166_WINDOWS_PROCESS_LINEAGE_RELATIVE_PATH = "src/loop166/windows_process_lineage.py"

PARENT_V6_PRELEASE_ATTESTATION_SCHEMA = "axon_loop167_phase_b_v6_relative_controller_argv_prelease_attestation_v7"
EXECUTION_CONTRACT_SCHEMA = "axon_loop167_phase_b_execution_contract_v7"
CLAIM_SCOPE = "v7_windows_job_abi_remediation_train_only_fixed_oof_not_promotion_or_heldout_evaluation"
AUTHORIZATION_CLAIM_SCOPE = "single_v7_windows_job_abi_remediation_train_only_raw_pass_then_fixed_oof"

CANONICAL_SUPERVISOR_EXECUTE_ARGV = (
    VNEV_PYTHON_RELATIVE_PATH,
    "-I",
    SUPERVISOR_RELATIVE_PATH,
    "--execute",
)
CANONICAL_CONTROLLER_EXECUTE_ARGV = (
    VNEV_PYTHON_RELATIVE_PATH,
    "-I",
    CONTROLLER_RELATIVE_PATH,
    "--execute",
)

EXPECTED_LEASE = {
    "lease_id": "loop167-phase-b-v7-windows-job-abi-remediation-train-oof-v1",
    "marker_path": f"{REPORT_DIRECTORY}/phase_b_execution_consumed_v7.json",
    "consume_before_first_raw_open": True,
    "failed_attempt_consumes_lease": True,
    "retry_resume_or_rescan_allowed": False,
}

FIXED_OUTPUT_CATALOG: tuple[dict[str, str], ...] = (
    {
        "name": "supervisor_launch_receipt",
        "path": f"{REPORT_DIRECTORY}/phase_b_supervisor_launch_v7.json",
        "kind": "pre_resume_containment_receipt",
    },
    {
        "name": "supervisor_exit_receipt",
        "path": f"{REPORT_DIRECTORY}/phase_b_supervisor_exit_v7.json",
        "kind": "supervisor_exit_receipt",
    },
    {
        "name": "supervisor_failure_receipt",
        "path": f"{REPORT_DIRECTORY}/phase_b_supervisor_failure_v7.json",
        "kind": "pre_resume_failure_receipt",
    },
    {
        "name": "child_job_attestation",
        "path": f"{REPORT_DIRECTORY}/phase_b_child_job_attestation_v7.json",
        "kind": "child_membership_receipt",
    },
    {
        "name": "feature_cache",
        "path": f"{REPORT_DIRECTORY}/phase_b_feature_cache_v7.npz",
        "kind": "numeric_feature_cache",
    },
    {
        "name": "raw_progress_ledger",
        "path": f"{REPORT_DIRECTORY}/phase_b_raw_progress_v7.jsonl",
        "kind": "append_only_raw_ledger",
    },
    {
        "name": "fit_progress_ledger",
        "path": f"{REPORT_DIRECTORY}/phase_b_fit_progress_v7.jsonl",
        "kind": "append_only_fit_ledger",
    },
    {
        "name": "execution_receipt",
        "path": f"{REPORT_DIRECTORY}/phase_b_execution_receipt_v7.json",
        "kind": "final_execution_receipt",
    },
)


@dataclass(frozen=True)
class VerifiedParentV6PreleaseAttestationV7:
    attestation_path: Path
    attestation_sha256: str
    parent_source_closure_binding: Mapping[str, str]
    parent_execution_contract_binding: Mapping[str, str]
    parent_runtime_lock_binding: Mapping[str, str]
    parent_resource_guard_binding: Mapping[str, str]
    parent_run_authorization_binding: Mapping[str, str]


@dataclass(frozen=True)
class VerifiedExecutionContractV7:
    contract_path: Path
    contract_sha256: str
    protocol_binding: Mapping[str, str]
    canonical_supervisor_execute_argv: tuple[str, ...]
    canonical_controller_execute_argv: tuple[str, ...]
    resource_contract: Mapping[str, Any]
    output_catalog: tuple[Mapping[str, str], ...]
    output_paths: Mapping[str, Path]
    lease: Mapping[str, Any]


def _fixed_binding(value: object, *, label: str, expected_path: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise PhaseBContractError(f"{label} must be a file binding")
    path = value.get("path")
    digest = value.get("sha256")
    if path != expected_path or not isinstance(digest, str) or len(digest) != 64:
        raise PhaseBContractError(f"{label} binding drifted from its fixed v7 path")
    return {"path": path, "sha256": digest}


def _binding(root: Path, relative_path: str) -> dict[str, str]:
    path = safe_project_path(root, relative_path, require_exists=True, require_regular_file=True)
    return {"path": relative_path, "sha256": sha256_file(path)}


def ensure_v7_static_artifact_parent(root: Path | str, relative_path: str) -> Path:
    root_path = safe_project_root(root)
    canonical = canonical_project_relative_path(relative_path)
    if not canonical.startswith(f"{ARTIFACT_DIRECTORY}/"):
        raise PhaseBContractError("v7 static artifact is outside its fixed root")
    cursor = root_path
    for component in canonical.split("/")[:-1]:
        cursor = cursor / component
        try:
            cursor.mkdir(exist_ok=True)
            stat_result = cursor.lstat()
        except OSError as error:
            raise PhaseBContractError("v7 static artifact parent is unavailable") from error
        attributes = int(getattr(stat_result, "st_file_attributes", 0))
        if stat.S_ISLNK(stat_result.st_mode) or bool(attributes & 0x0400) or not stat.S_ISDIR(stat_result.st_mode):
            raise PhaseBContractError("v7 static artifact parent is unsafe")
    return safe_project_path(root_path, canonical, require_exists=False)


def _require_json_binding(
    root: Path,
    binding: object,
    *,
    label: str,
    expected_path: str,
    expected_schema: str,
) -> tuple[dict[str, str], Mapping[str, Any]]:
    normalized = _fixed_binding(binding, label=label, expected_path=expected_path)
    path, _ = verify_safe_file_binding(root, normalized, label=label)
    payload = require_canonical_json(path)
    if payload.get("schema") != expected_schema or payload.get("loop_id") != LOOP_ID:
        raise PhaseBContractError(f"{label} schema or loop identity drifted")
    return normalized, payload


def _assert_absent(root: Path, relative_paths: Sequence[str], *, label: str) -> None:
    for relative_path in relative_paths:
        path = safe_project_path(root, relative_path, require_exists=False)
        if path.exists() or path.is_symlink():
            raise PhaseBContractError(f"{label} exists or is unsafe")


def _parent_v6_source_bindings(root: Path, payload: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    files = payload.get("source_files")
    if not isinstance(files, list):
        raise PhaseBContractError("Parent v6 source closure lacks source files")
    found: dict[str, dict[str, str]] = {}
    required = {
        "scripts/run_loop167_phase_b_supervisor_v6.py",
        "scripts/run_loop167_phase_b_controller_v6.py",
        "src/loop167_phase_b/invocation_v6.py",
        "src/loop167_phase_b/supervisor_v6.py",
    }
    for value in files:
        if not isinstance(value, Mapping):
            raise PhaseBContractError("Parent v6 source closure has an invalid binding")
        path = value.get("path")
        if path in required:
            verified_path, digest = verify_safe_file_binding(root, value, label="parent_v6_source")
            relative = safe_project_relative_path(root, verified_path, require_exists=True, require_regular_file=True)
            found[relative] = {"path": relative, "sha256": digest}
    if set(found) != required:
        raise PhaseBContractError("Parent v6 source closure omits the argv-defect sources")
    if payload.get("controller") != found["scripts/run_loop167_phase_b_controller_v6.py"]:
        raise PhaseBContractError("Parent v6 source closure controller binding drifted")
    if payload.get("supervisor") != found["scripts/run_loop167_phase_b_supervisor_v6.py"]:
        raise PhaseBContractError("Parent v6 source closure supervisor binding drifted")
    return found


def _validate_parent_v6_execution_surface(payload: Mapping[str, Any]) -> None:
    """Bind absence claims to the sealed v6 output and lease surface."""

    catalog = payload.get("output_catalog")
    if not isinstance(catalog, list) or len(catalog) != len(PARENT_V6_OUTPUT_PATHS):
        raise PhaseBContractError("Parent v6 execution contract output catalog drifted")
    paths = [entry.get("path") if isinstance(entry, Mapping) else None for entry in catalog]
    if tuple(paths) != PARENT_V6_OUTPUT_PATHS or len(set(paths)) != len(paths):
        raise PhaseBContractError("Parent v6 execution contract output paths drifted")
    if payload.get("lease") != PARENT_V6_EXPECTED_LEASE:
        raise PhaseBContractError("Parent v6 execution contract lease surface drifted")
    if payload.get("canonical_controller_execute_argv") != [
        VNEV_PYTHON_RELATIVE_PATH,
        "-I",
        "scripts/run_loop167_phase_b_controller_v6.py",
        "--execute",
    ]:
        raise PhaseBContractError("Parent v6 execution contract controller argv drifted")


def _parent_v6_bindings(root: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    source_binding = _binding(root, PARENT_V6_SOURCE_CLOSURE_RELATIVE_PATH)
    _, source_closure = _require_json_binding(
        root,
        source_binding,
        label="parent_v6_source_closure",
        expected_path=PARENT_V6_SOURCE_CLOSURE_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_source_closure_v6",
    )
    _parent_v6_source_bindings(root, source_closure)
    return (
        source_binding,
        _binding(root, PARENT_V6_EXECUTION_CONTRACT_RELATIVE_PATH),
        _binding(root, PARENT_V6_RUNTIME_LOCK_RELATIVE_PATH),
        _binding(root, PARENT_V6_RESOURCE_GUARD_RELATIVE_PATH),
        _binding(root, PARENT_V6_RUN_AUTHORIZATION_RELATIVE_PATH),
    )


def build_parent_v6_prelease_attestation_payload_v7(root: Path | str) -> dict[str, Any]:
    """Record the v6 argv defect before any child launch, lease, or raw access."""

    root_path = safe_project_root(root)
    (
        source_closure,
        execution_contract,
        runtime_lock,
        resource_guard,
        run_authorization,
    ) = _parent_v6_bindings(root_path)
    _, source_payload = _require_json_binding(
        root_path,
        source_closure,
        label="parent_v6_source_closure",
        expected_path=PARENT_V6_SOURCE_CLOSURE_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_source_closure_v6",
    )
    parent_sources = _parent_v6_source_bindings(root_path, source_payload)
    _, execution_contract_payload = _require_json_binding(
        root_path,
        execution_contract,
        label="parent_v6_execution_contract",
        expected_path=PARENT_V6_EXECUTION_CONTRACT_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_execution_contract_v6",
    )
    _validate_parent_v6_execution_surface(execution_contract_payload)
    _require_json_binding(
        root_path,
        runtime_lock,
        label="parent_v6_runtime_lock",
        expected_path=PARENT_V6_RUNTIME_LOCK_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_runtime_lock_v6",
    )
    _require_json_binding(
        root_path,
        resource_guard,
        label="parent_v6_resource_guard",
        expected_path=PARENT_V6_RESOURCE_GUARD_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_resource_guard_v6",
    )
    _require_json_binding(
        root_path,
        run_authorization,
        label="parent_v6_run_authorization",
        expected_path=PARENT_V6_RUN_AUTHORIZATION_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_run_authorization_v6",
    )
    _assert_absent(root_path, (PARENT_V6_EXECUTION_LEASE_RELATIVE_PATH, *PARENT_V6_OUTPUT_PATHS), label="v6 prelease output")
    return {
        "schema": PARENT_V6_PRELEASE_ATTESTATION_SCHEMA,
        "loop_id": LOOP_ID,
        "status": "v6_relative_controller_argv_construction_defect_before_child_launch",
        "parent_v6_source_closure": source_closure,
        "parent_v6_execution_contract": execution_contract,
        "parent_v6_runtime_lock": runtime_lock,
        "parent_v6_resource_guard": resource_guard,
        "parent_v6_run_authorization": run_authorization,
        "v6_supervisor_source": parent_sources["scripts/run_loop167_phase_b_supervisor_v6.py"],
        "v6_controller_source": parent_sources["scripts/run_loop167_phase_b_controller_v6.py"],
        "v6_invocation_source": parent_sources["src/loop167_phase_b/invocation_v6.py"],
        "v6_supervisor_core_source": parent_sources["src/loop167_phase_b/supervisor_v6.py"],
        "v6_controller_canonical_process_argv": ["scripts/run_loop167_phase_b_controller_v6.py", "--execute"],
        "v6_supervisor_constructed_controller_argument": "absolute_project_controller_path",
        "mismatch": "absolute_script_path_rejected_by_sealed_child_process_argv_validator",
        "v6_launch_receipt_absent": True,
        "v6_child_attestation_absent": True,
        "v6_lease_absent": True,
        "v6_data_outputs_absent": True,
        "replacement_scope": "one_new_v7_relative_controller_argv_remediation_chain_only",
    }


def validate_parent_v6_prelease_attestation_payload_v7(root: Path, payload: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "loop_id",
        "status",
        "parent_v6_source_closure",
        "parent_v6_execution_contract",
        "parent_v6_runtime_lock",
        "parent_v6_resource_guard",
        "parent_v6_run_authorization",
        "v6_supervisor_source",
        "v6_controller_source",
        "v6_invocation_source",
        "v6_supervisor_core_source",
        "v6_controller_canonical_process_argv",
        "v6_supervisor_constructed_controller_argument",
        "mismatch",
        "v6_launch_receipt_absent",
        "v6_child_attestation_absent",
        "v6_lease_absent",
        "v6_data_outputs_absent",
        "replacement_scope",
    }
    if set(payload) != expected_keys or payload.get("schema") != PARENT_V6_PRELEASE_ATTESTATION_SCHEMA:
        raise PhaseBContractError("v7 parent v6 attestation schema drifted")
    if payload.get("loop_id") != LOOP_ID or payload.get("status") != "v6_relative_controller_argv_construction_defect_before_child_launch":
        raise PhaseBContractError("v7 parent v6 attestation identity drifted")
    expected = build_parent_v6_prelease_attestation_payload_v7(root)
    if dict(payload) != expected:
        raise PhaseBContractError("v7 parent v5 attestation facts drifted")


def verify_parent_v6_prelease_attestation_v7(
    root: Path | str,
    binding: Mapping[str, str],
) -> VerifiedParentV6PreleaseAttestationV7:
    root_path = safe_project_root(root)
    normalized = _fixed_binding(
        binding,
        label="parent_v6_prelease_attestation",
        expected_path=PARENT_V6_PRELEASE_ATTESTATION_RELATIVE_PATH,
    )
    path, digest = verify_safe_file_binding(root_path, normalized, label="parent_v6_prelease_attestation")
    payload = require_canonical_json(path)
    validate_parent_v6_prelease_attestation_payload_v7(root_path, payload)
    return VerifiedParentV6PreleaseAttestationV7(
        attestation_path=path,
        attestation_sha256=digest,
        parent_source_closure_binding=MappingProxyType(dict(payload["parent_v6_source_closure"])),
        parent_execution_contract_binding=MappingProxyType(dict(payload["parent_v6_execution_contract"])),
        parent_runtime_lock_binding=MappingProxyType(dict(payload["parent_v6_runtime_lock"])),
        parent_resource_guard_binding=MappingProxyType(dict(payload["parent_v6_resource_guard"])),
        parent_run_authorization_binding=MappingProxyType(dict(payload["parent_v6_run_authorization"])),
    )


def output_catalog_sha256(catalog: object = FIXED_OUTPUT_CATALOG) -> str:
    materialized = list(catalog) if isinstance(catalog, tuple) else catalog
    return sha256_bytes(canonical_json_bytes({"output_catalog": materialized}))


def _validated_output_catalog(catalog: object) -> tuple[Mapping[str, str], ...]:
    materialized = [dict(entry) for entry in catalog] if isinstance(catalog, tuple) else catalog
    if materialized != [dict(entry) for entry in FIXED_OUTPUT_CATALOG]:
        raise PhaseBContractError("v7 output catalog drifted")
    return tuple(MappingProxyType(dict(entry)) for entry in FIXED_OUTPUT_CATALOG)


def resolve_output_catalog_v7(root: Path | str, catalog: object = FIXED_OUTPUT_CATALOG) -> Mapping[str, Path]:
    root_path = safe_project_root(root)
    paths: dict[str, Path] = {}
    for entry in _validated_output_catalog(catalog):
        name = entry["name"]
        path = safe_project_path(root_path, entry["path"], require_exists=False)
        if name in paths:
            raise PhaseBContractError("v7 output catalog repeats a name")
        paths[name] = path
    return MappingProxyType(paths)


def assert_output_catalog_is_fresh_v7(root: Path | str, catalog: object = FIXED_OUTPUT_CATALOG) -> Mapping[str, Path]:
    paths = resolve_output_catalog_v7(root, catalog)
    for path in paths.values():
        if path.exists() or path.is_symlink():
            raise PhaseBContractError("v7 output already exists or is unsafe")
    marker = safe_project_path(root, EXPECTED_LEASE["marker_path"], require_exists=False)
    if marker.exists() or marker.is_symlink():
        raise PhaseBContractError("v7 execution lease already exists or is unsafe")
    return paths


def assert_contained_child_prelease_surface_v7(
    root: Path | str,
    catalog: object = FIXED_OUTPUT_CATALOG,
) -> Mapping[str, Path]:
    """Allow exactly the pre-resume launch receipt after containment is proven."""

    paths = resolve_output_catalog_v7(root, catalog)
    launch_path = paths["supervisor_launch_receipt"]
    if not launch_path.is_file() or launch_path.is_symlink():
        raise PhaseBContractError("v7 contained child requires the pre-resume launch receipt")
    for name, path in paths.items():
        if name == "supervisor_launch_receipt":
            continue
        if path.exists() or path.is_symlink():
            raise PhaseBContractError("v7 contained child prelease output surface drifted")
    marker = safe_project_path(root, EXPECTED_LEASE["marker_path"], require_exists=False)
    if marker.exists() or marker.is_symlink():
        raise PhaseBContractError("v7 contained child execution lease already exists or is unsafe")
    return paths


def assert_attested_child_prelease_surface_v7(
    root: Path | str,
    catalog: object = FIXED_OUTPUT_CATALOG,
) -> Mapping[str, Path]:
    paths = resolve_output_catalog_v7(root, catalog)
    for name in ("supervisor_launch_receipt", "child_job_attestation"):
        path = paths[name]
        if not path.is_file() or path.is_symlink():
            raise PhaseBContractError("v7 attested child requires its containment receipts")
    for name, path in paths.items():
        if name in {"supervisor_launch_receipt", "child_job_attestation"}:
            continue
        if path.exists() or path.is_symlink():
            raise PhaseBContractError("v7 attested child prelease output surface drifted")
    marker = safe_project_path(root, EXPECTED_LEASE["marker_path"], require_exists=False)
    if marker.exists() or marker.is_symlink():
        raise PhaseBContractError("v7 attested child execution lease already exists or is unsafe")
    return paths


def assert_leased_child_pre_raw_surface_v7(
    root: Path | str,
    catalog: object = FIXED_OUTPUT_CATALOG,
) -> Mapping[str, Path]:
    """Allow only containment receipts and the canonical consumed lease before raw access."""

    paths = resolve_output_catalog_v7(root, catalog)
    for name in ("supervisor_launch_receipt", "child_job_attestation"):
        path = paths[name]
        if not path.is_file() or path.is_symlink():
            raise PhaseBContractError("v7 leased child requires its containment receipts")
    for name, path in paths.items():
        if name in {"supervisor_launch_receipt", "child_job_attestation"}:
            continue
        if path.exists() or path.is_symlink():
            raise PhaseBContractError("v7 leased child pre-raw output surface drifted")
    marker = safe_project_path(root, EXPECTED_LEASE["marker_path"], require_exists=True, require_regular_file=True)
    require_canonical_json(marker)
    return paths


def build_execution_contract_payload_v7(
    root: Path | str,
    *,
    parent_v6_prelease_attestation_binding: Mapping[str, str],
) -> dict[str, Any]:
    root_path = safe_project_root(root)
    attestation = verify_parent_v6_prelease_attestation_v7(root_path, parent_v6_prelease_attestation_binding)
    protocol_binding = _binding(root_path, PHASE_B_PROTOCOL_RELATIVE_PATH)
    _, protocol = _require_json_binding(
        root_path,
        protocol_binding,
        label="phase_b_protocol",
        expected_path=PHASE_B_PROTOCOL_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_protocol_v1",
    )
    if protocol.get("claim_scope") != "local_train_only_structural_delta_diagnostic_not_model_quality_promotion_or_full_test":
        raise PhaseBContractError("Phase-B protocol scope drifted")
    assert_output_catalog_is_fresh_v7(root_path)
    return {
        "schema": EXECUTION_CONTRACT_SCHEMA,
        "loop_id": LOOP_ID,
        "phase_b_protocol": protocol_binding,
        "parent_v6_prelease_attestation": {
            "path": PARENT_V6_PRELEASE_ATTESTATION_RELATIVE_PATH,
            "sha256": attestation.attestation_sha256,
        },
        "claim_scope": CLAIM_SCOPE,
        "canonical_supervisor_execute_argv": list(CANONICAL_SUPERVISOR_EXECUTE_ARGV),
        "canonical_supervisor_execute_argv_sha256": canonical_argv_sha256(CANONICAL_SUPERVISOR_EXECUTE_ARGV),
        "canonical_controller_execute_argv": list(CANONICAL_CONTROLLER_EXECUTE_ARGV),
        "canonical_controller_execute_argv_sha256": canonical_argv_sha256(CANONICAL_CONTROLLER_EXECUTE_ARGV),
        "windows_job_boundary": {
            "abi": "typed_ctypes_windll_use_last_error_and_handle_signatures",
            "guard_probe": "non_kill_current_process_assignment_and_membership",
            "controller_launch": "create_suspended_assign_verify_persist_pre_resume_receipt_then_resume",
            "child_self_attestation_required_before_lease": True,
            "kill_on_close_required": True,
        },
        "resource_contract": dict(EXPECTED_RESOURCE_CONTRACT),
        "b1_sampling_indicators": dict(B1_SAMPLING_INDICATORS_CONTRACT),
        "raw_root_relative": RAW_ROOT_RELATIVE_PATH,
        "output_catalog": [dict(entry) for entry in FIXED_OUTPUT_CATALOG],
        "output_catalog_sha256": output_catalog_sha256(),
        "lease": dict(EXPECTED_LEASE),
        "forbidden": list(EXPECTED_FORBIDDEN),
        "execution_authorization_required": True,
        "raw_open_attempts_before_lease": 0,
    }


def validate_execution_contract_payload_v7(root: Path, payload: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "loop_id",
        "phase_b_protocol",
        "parent_v6_prelease_attestation",
        "claim_scope",
        "canonical_supervisor_execute_argv",
        "canonical_supervisor_execute_argv_sha256",
        "canonical_controller_execute_argv",
        "canonical_controller_execute_argv_sha256",
        "windows_job_boundary",
        "resource_contract",
        "b1_sampling_indicators",
        "raw_root_relative",
        "output_catalog",
        "output_catalog_sha256",
        "lease",
        "forbidden",
        "execution_authorization_required",
        "raw_open_attempts_before_lease",
    }
    if set(payload) != expected_keys or payload.get("schema") != EXECUTION_CONTRACT_SCHEMA:
        raise PhaseBContractError("v7 execution contract schema drifted")
    if payload.get("loop_id") != LOOP_ID or payload.get("claim_scope") != CLAIM_SCOPE:
        raise PhaseBContractError("v7 execution contract identity drifted")
    _require_json_binding(
        root,
        payload.get("phase_b_protocol"),
        label="phase_b_protocol",
        expected_path=PHASE_B_PROTOCOL_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_protocol_v1",
    )
    verify_parent_v6_prelease_attestation_v7(root, payload["parent_v6_prelease_attestation"])
    if payload.get("canonical_supervisor_execute_argv") != list(CANONICAL_SUPERVISOR_EXECUTE_ARGV):
        raise PhaseBContractError("v7 supervisor argv drifted")
    if payload.get("canonical_supervisor_execute_argv_sha256") != canonical_argv_sha256(CANONICAL_SUPERVISOR_EXECUTE_ARGV):
        raise PhaseBContractError("v7 supervisor argv hash drifted")
    if payload.get("canonical_controller_execute_argv") != list(CANONICAL_CONTROLLER_EXECUTE_ARGV):
        raise PhaseBContractError("v7 controller argv drifted")
    if payload.get("canonical_controller_execute_argv_sha256") != canonical_argv_sha256(CANONICAL_CONTROLLER_EXECUTE_ARGV):
        raise PhaseBContractError("v7 controller argv hash drifted")
    if payload.get("windows_job_boundary") != {
        "abi": "typed_ctypes_windll_use_last_error_and_handle_signatures",
        "guard_probe": "non_kill_current_process_assignment_and_membership",
        "controller_launch": "create_suspended_assign_verify_persist_pre_resume_receipt_then_resume",
        "child_self_attestation_required_before_lease": True,
        "kill_on_close_required": True,
    }:
        raise PhaseBContractError("v7 Windows Job boundary drifted")
    if payload.get("resource_contract") != EXPECTED_RESOURCE_CONTRACT:
        raise PhaseBContractError("v7 resource contract drifted")
    if payload.get("b1_sampling_indicators") != B1_SAMPLING_INDICATORS_CONTRACT:
        raise PhaseBContractError("v7 B1 sampling contract drifted")
    if payload.get("raw_root_relative") != RAW_ROOT_RELATIVE_PATH:
        raise PhaseBContractError("v7 raw root drifted")
    if canonical_project_relative_path(payload["raw_root_relative"]) != RAW_ROOT_RELATIVE_PATH:
        raise PhaseBContractError("v7 raw root is not canonical")
    _validated_output_catalog(payload.get("output_catalog"))
    if payload.get("output_catalog_sha256") != output_catalog_sha256():
        raise PhaseBContractError("v7 output catalog hash drifted")
    if payload.get("lease") != EXPECTED_LEASE or payload.get("forbidden") != EXPECTED_FORBIDDEN:
        raise PhaseBContractError("v7 lease or forbidden scope drifted")
    if payload.get("execution_authorization_required") is not True or payload.get("raw_open_attempts_before_lease") != 0:
        raise PhaseBContractError("v7 prelease contract drifted")
    resolve_output_catalog_v7(root, payload["output_catalog"])


def verify_execution_contract_v7(root: Path | str, binding: Mapping[str, str]) -> VerifiedExecutionContractV7:
    root_path = safe_project_root(root)
    normalized = _fixed_binding(binding, label="execution_contract", expected_path=EXECUTION_CONTRACT_RELATIVE_PATH)
    path, digest = verify_safe_file_binding(root_path, normalized, label="execution_contract")
    payload = require_canonical_json(path)
    validate_execution_contract_payload_v7(root_path, payload)
    return VerifiedExecutionContractV7(
        contract_path=path,
        contract_sha256=digest,
        protocol_binding=MappingProxyType(dict(payload["phase_b_protocol"])),
        canonical_supervisor_execute_argv=tuple(payload["canonical_supervisor_execute_argv"]),
        canonical_controller_execute_argv=tuple(payload["canonical_controller_execute_argv"]),
        resource_contract=MappingProxyType(dict(EXPECTED_RESOURCE_CONTRACT)),
        output_catalog=_validated_output_catalog(payload["output_catalog"]),
        output_paths=resolve_output_catalog_v7(root_path, payload["output_catalog"]),
        lease=MappingProxyType(dict(EXPECTED_LEASE)),
    )


def canonical_controller_execute_argv_v7(argv: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(argv)
    if normalized != CANONICAL_CONTROLLER_EXECUTE_ARGV:
        raise PhaseBContractError("v7 controller execute argv is not sealed")
    return normalized


__all__ = [
    "ARTIFACT_DIRECTORY",
    "AUTHORIZATION_CLAIM_SCOPE",
    "CANONICAL_CONTROLLER_EXECUTE_ARGV",
    "CANONICAL_SUPERVISOR_EXECUTE_ARGV",
    "CLAIM_SCOPE",
    "CONTROLLER_RELATIVE_PATH",
    "EXECUTION_CONTRACT_RELATIVE_PATH",
    "EXECUTION_CONTRACT_SCHEMA",
    "EXPECTED_LEASE",
    "FIXED_OUTPUT_CATALOG",
    "LOOP_ID",
    "LOOP166_WINDOWS_JOB_RELATIVE_PATH",
    "LOOP166_WINDOWS_PROCESS_LINEAGE_RELATIVE_PATH",
    "PARENT_V6_PRELEASE_ATTESTATION_RELATIVE_PATH",
    "RAW_ROOT_RELATIVE_PATH",
    "REPORT_DIRECTORY",
    "RESOURCE_GUARD_RELATIVE_PATH",
    "RUN_AUTHORIZATION_RELATIVE_PATH",
    "RUNTIME_LOCK_RELATIVE_PATH",
    "SOURCE_CLOSURE_RELATIVE_PATH",
    "SUPERVISOR_RELATIVE_PATH",
    "VNEV_PYTHON_RELATIVE_PATH",
    "VerifiedExecutionContractV7",
    "assert_attested_child_prelease_surface_v7",
    "assert_contained_child_prelease_surface_v7",
    "assert_leased_child_pre_raw_surface_v7",
    "assert_output_catalog_is_fresh_v7",
    "build_execution_contract_payload_v7",
    "build_parent_v6_prelease_attestation_payload_v7",
    "canonical_controller_execute_argv_v7",
    "ensure_v7_static_artifact_parent",
    "output_catalog_sha256",
    "resolve_output_catalog_v7",
    "validate_execution_contract_payload_v7",
    "validate_parent_v6_prelease_attestation_payload_v7",
    "verify_execution_contract_v7",
    "verify_parent_v6_prelease_attestation_v7",
]
