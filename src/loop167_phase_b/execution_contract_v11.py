"""Independent v11 authority contract for the user-authorized no-prelaunch-memory-floor remediation."""

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
ARTIFACT_DIRECTORY = f"{PARENT_ARTIFACT_DIRECTORY}/phase_b_v11_dual_identity_job_attestation_remediation"
REPORT_DIRECTORY = "reports/roadmap_9997/loop167/phase_b_v11_dual_identity_job_attestation_remediation"

PHASE_B_PROTOCOL_RELATIVE_PATH = f"{PARENT_ARTIFACT_DIRECTORY}/phase_b_protocol.json"
PARENT_V10_ARTIFACT_DIRECTORY = f"{PARENT_ARTIFACT_DIRECTORY}/phase_b_v10_redirector_lineage_attestation_remediation"
PARENT_V10_SOURCE_CLOSURE_RELATIVE_PATH = f"{PARENT_V10_ARTIFACT_DIRECTORY}/phase_b_source_closure_v10.json"
PARENT_V10_EXECUTION_CONTRACT_RELATIVE_PATH = f"{PARENT_V10_ARTIFACT_DIRECTORY}/phase_b_execution_contract_v10.json"
PARENT_V10_RUNTIME_LOCK_RELATIVE_PATH = f"{PARENT_V10_ARTIFACT_DIRECTORY}/phase_b_runtime_lock_v10.json"
PARENT_V10_RESOURCE_GUARD_RELATIVE_PATH = f"{PARENT_V10_ARTIFACT_DIRECTORY}/phase_b_resource_guard_v10.json"
PARENT_V10_RUN_AUTHORIZATION_RELATIVE_PATH = f"{PARENT_V10_ARTIFACT_DIRECTORY}/phase_b_run_authorization_v10.json"
PARENT_V10_EXECUTION_LEASE_RELATIVE_PATH = (
    "reports/roadmap_9997/loop167/phase_b_v10_redirector_lineage_attestation_remediation/"
    "phase_b_execution_consumed_v10.json"
)
PARENT_V10_OUTPUT_PATHS = (
    "reports/roadmap_9997/loop167/phase_b_v10_redirector_lineage_attestation_remediation/phase_b_supervisor_launch_v10.json",
    "reports/roadmap_9997/loop167/phase_b_v10_redirector_lineage_attestation_remediation/phase_b_supervisor_exit_v10.json",
    "reports/roadmap_9997/loop167/phase_b_v10_redirector_lineage_attestation_remediation/phase_b_supervisor_failure_v10.json",
    "reports/roadmap_9997/loop167/phase_b_v10_redirector_lineage_attestation_remediation/phase_b_child_job_attestation_v10.json",
    "reports/roadmap_9997/loop167/phase_b_v10_redirector_lineage_attestation_remediation/phase_b_feature_cache_v10.npz",
    "reports/roadmap_9997/loop167/phase_b_v10_redirector_lineage_attestation_remediation/phase_b_raw_progress_v10.jsonl",
    "reports/roadmap_9997/loop167/phase_b_v10_redirector_lineage_attestation_remediation/phase_b_fit_progress_v10.jsonl",
    "reports/roadmap_9997/loop167/phase_b_v10_redirector_lineage_attestation_remediation/phase_b_execution_receipt_v10.json",
)
PARENT_V10_EXPECTED_LEASE = {
    "lease_id": "loop167-phase-b-v10-windows-job-abi-remediation-train-oof-v1",
    "marker_path": PARENT_V10_EXECUTION_LEASE_RELATIVE_PATH,
    "consume_before_first_raw_open": True,
    "failed_attempt_consumes_lease": True,
    "retry_resume_or_rescan_allowed": False,
}

PARENT_V10_PRELEASE_ATTESTATION_RELATIVE_PATH = (
    f"{ARTIFACT_DIRECTORY}/phase_b_v10_launcher_controller_creation_identity_failure_attestation.json"
)
EXECUTION_CONTRACT_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_execution_contract_v11.json"
SOURCE_CLOSURE_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_source_closure_v11.json"
RUNTIME_LOCK_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_runtime_lock_v11.json"
RESOURCE_GUARD_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_resource_guard_v11.json"
RUN_AUTHORIZATION_RELATIVE_PATH = f"{ARTIFACT_DIRECTORY}/phase_b_run_authorization_v11.json"

SUPERVISOR_RELATIVE_PATH = "scripts/run_loop167_phase_b_supervisor_v11.py"
CONTROLLER_RELATIVE_PATH = "scripts/run_loop167_phase_b_controller_v11.py"
VNEV_PYTHON_RELATIVE_PATH = "vnev/Scripts/python.exe"
RAW_ROOT_RELATIVE_PATH = "data/random_20w_worktree"
LOOP166_WINDOWS_JOB_RELATIVE_PATH = "src/loop166/windows_job.py"
LOOP166_WINDOWS_PROCESS_LINEAGE_RELATIVE_PATH = "src/loop166/windows_process_lineage.py"

PARENT_V10_PRELEASE_ATTESTATION_SCHEMA = "axon_loop167_phase_b_v10_prelease_no_dynamic_execution_attestation_v11"
EXECUTION_CONTRACT_SCHEMA = "axon_loop167_phase_b_execution_contract_v11"
CLAIM_SCOPE = "v11_user_authorized_no_prelaunch_memory_floor_train_only_fixed_oof_not_promotion_or_heldout_evaluation"
AUTHORIZATION_CLAIM_SCOPE = "single_v11_user_authorized_no_prelaunch_memory_floor_train_only_raw_pass_then_fixed_oof"

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
    "lease_id": "loop167-phase-b-v11-windows-job-abi-remediation-train-oof-v1",
    "marker_path": f"{REPORT_DIRECTORY}/phase_b_execution_consumed_v11.json",
    "consume_before_first_raw_open": True,
    "failed_attempt_consumes_lease": True,
    "retry_resume_or_rescan_allowed": False,
}

FIXED_OUTPUT_CATALOG: tuple[dict[str, str], ...] = (
    {
        "name": "supervisor_launch_receipt",
        "path": f"{REPORT_DIRECTORY}/phase_b_supervisor_launch_v11.json",
        "kind": "pre_resume_containment_receipt",
    },
    {
        "name": "supervisor_exit_receipt",
        "path": f"{REPORT_DIRECTORY}/phase_b_supervisor_exit_v11.json",
        "kind": "supervisor_exit_receipt",
    },
    {
        "name": "supervisor_failure_receipt",
        "path": f"{REPORT_DIRECTORY}/phase_b_supervisor_failure_v11.json",
        "kind": "pre_resume_failure_receipt",
    },
    {
        "name": "child_job_attestation",
        "path": f"{REPORT_DIRECTORY}/phase_b_child_job_attestation_v11.json",
        "kind": "child_membership_receipt",
    },
    {
        "name": "feature_cache",
        "path": f"{REPORT_DIRECTORY}/phase_b_feature_cache_v11.npz",
        "kind": "numeric_feature_cache",
    },
    {
        "name": "raw_progress_ledger",
        "path": f"{REPORT_DIRECTORY}/phase_b_raw_progress_v11.jsonl",
        "kind": "append_only_raw_ledger",
    },
    {
        "name": "fit_progress_ledger",
        "path": f"{REPORT_DIRECTORY}/phase_b_fit_progress_v11.jsonl",
        "kind": "append_only_fit_ledger",
    },
    {
        "name": "execution_receipt",
        "path": f"{REPORT_DIRECTORY}/phase_b_execution_receipt_v11.json",
        "kind": "final_execution_receipt",
    },
)


@dataclass(frozen=True)
class VerifiedParentV10PreleaseAttestationV11:
    attestation_path: Path
    attestation_sha256: str
    parent_source_closure_binding: Mapping[str, str]
    parent_execution_contract_binding: Mapping[str, str]
    parent_runtime_lock_binding: Mapping[str, str]


@dataclass(frozen=True)
class VerifiedExecutionContractV11:
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
        raise PhaseBContractError(f"{label} binding drifted from its fixed v11 path")
    return {"path": path, "sha256": digest}


def _binding(root: Path, relative_path: str) -> dict[str, str]:
    path = safe_project_path(root, relative_path, require_exists=True, require_regular_file=True)
    return {"path": relative_path, "sha256": sha256_file(path)}


def ensure_v11_static_artifact_parent(root: Path | str, relative_path: str) -> Path:
    root_path = safe_project_root(root)
    canonical = canonical_project_relative_path(relative_path)
    if not canonical.startswith(f"{ARTIFACT_DIRECTORY}/"):
        raise PhaseBContractError("v11 static artifact is outside its fixed root")
    cursor = root_path
    for component in canonical.split("/")[:-1]:
        cursor = cursor / component
        try:
            cursor.mkdir(exist_ok=True)
            stat_result = cursor.lstat()
        except OSError as error:
            raise PhaseBContractError("v11 static artifact parent is unavailable") from error
        attributes = int(getattr(stat_result, "st_file_attributes", 0))
        if stat.S_ISLNK(stat_result.st_mode) or bool(attributes & 0x0400) or not stat.S_ISDIR(stat_result.st_mode):
            raise PhaseBContractError("v11 static artifact parent is unsafe")
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


def _parent_v10_source_bindings(root: Path, payload: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    files = payload.get("source_files")
    if not isinstance(files, list):
        raise PhaseBContractError("Parent v10 source closure lacks source files")
    found: dict[str, dict[str, str]] = {}
    required = {
        "scripts/run_loop167_phase_b_supervisor_v10.py",
        "scripts/run_loop167_phase_b_controller_v10.py",
        "src/loop167_phase_b/invocation_v10.py",
        "src/loop167_phase_b/supervisor_v10.py",
        "src/loop167_phase_b/child_attestation_v10.py",
        "src/loop167_phase_b/resource_guard_v10.py",
        "src/loop167_phase_b/execution_authorization_v10.py",
    }
    for value in files:
        if not isinstance(value, Mapping):
            raise PhaseBContractError("Parent v10 source closure has an invalid binding")
        path = value.get("path")
        if path in required:
            verified_path, digest = verify_safe_file_binding(root, value, label="parent_v10_source")
            relative = safe_project_relative_path(root, verified_path, require_exists=True, require_regular_file=True)
            found[relative] = {"path": relative, "sha256": digest}
    if set(found) != required:
        raise PhaseBContractError("Parent v10 source closure omits required control sources")
    if payload.get("controller") != found["scripts/run_loop167_phase_b_controller_v10.py"]:
        raise PhaseBContractError("Parent v10 source closure controller binding drifted")
    if payload.get("supervisor") != found["scripts/run_loop167_phase_b_supervisor_v10.py"]:
        raise PhaseBContractError("Parent v10 source closure supervisor binding drifted")
    return found


def _assert_v10_controller_identity_mismatch(root: Path, binding: Mapping[str, str]) -> None:
    """Prove the exact pre-attestation identity comparison that rejected v10."""

    source_path, _ = verify_safe_file_binding(root, binding, label="parent_v10_child_attestation_source")
    source = source_path.read_text(encoding="utf-8")
    expected_call = '''membership = audit_current_process_job_membership(
            root_path,
            job_proof_binding,
            int(assignment["process_creation_time_filetime"]),
            expected_pid=os.getpid(),
        )'''
    if expected_call not in source:
        raise PhaseBContractError("v10 controller identity mismatch source proof drifted")


def _validate_parent_v10_execution_surface(payload: Mapping[str, Any]) -> None:
    """Bind absence claims to the sealed v10 output and lease surface."""

    catalog = payload.get("output_catalog")
    if not isinstance(catalog, list) or len(catalog) != len(PARENT_V10_OUTPUT_PATHS):
        raise PhaseBContractError("Parent v10 execution contract output catalog drifted")
    paths = [entry.get("path") if isinstance(entry, Mapping) else None for entry in catalog]
    if tuple(paths) != PARENT_V10_OUTPUT_PATHS or len(set(paths)) != len(paths):
        raise PhaseBContractError("Parent v10 execution contract output paths drifted")
    if payload.get("lease") != PARENT_V10_EXPECTED_LEASE:
        raise PhaseBContractError("Parent v10 execution contract lease surface drifted")
    if payload.get("canonical_controller_execute_argv") != [
        VNEV_PYTHON_RELATIVE_PATH,
        "-I",
        "scripts/run_loop167_phase_b_controller_v10.py",
        "--execute",
    ]:
        raise PhaseBContractError("Parent v10 execution contract controller argv drifted")


def _parent_v10_bindings(root: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    source_binding = _binding(root, PARENT_V10_SOURCE_CLOSURE_RELATIVE_PATH)
    _, source_closure = _require_json_binding(
        root,
        source_binding,
        label="parent_v10_source_closure",
        expected_path=PARENT_V10_SOURCE_CLOSURE_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_source_closure_v10",
    )
    _parent_v10_source_bindings(root, source_closure)
    return (
        source_binding,
        _binding(root, PARENT_V10_EXECUTION_CONTRACT_RELATIVE_PATH),
        _binding(root, PARENT_V10_RUNTIME_LOCK_RELATIVE_PATH),
    )


def build_parent_v10_prelease_attestation_payload_v11(root: Path | str) -> dict[str, Any]:
    """Bind the one-shot v10 failure before its child could attest or open raw data."""

    root_path = safe_project_root(root)
    (
        source_closure,
        execution_contract,
        runtime_lock,
    ) = _parent_v10_bindings(root_path)
    _, source_payload = _require_json_binding(
        root_path,
        source_closure,
        label="parent_v10_source_closure",
        expected_path=PARENT_V10_SOURCE_CLOSURE_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_source_closure_v10",
    )
    parent_sources = _parent_v10_source_bindings(root_path, source_payload)
    _assert_v10_controller_identity_mismatch(
        root_path,
        parent_sources["src/loop167_phase_b/child_attestation_v10.py"],
    )
    _, execution_contract_payload = _require_json_binding(
        root_path,
        execution_contract,
        label="parent_v10_execution_contract",
        expected_path=PARENT_V10_EXECUTION_CONTRACT_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_execution_contract_v10",
    )
    _validate_parent_v10_execution_surface(execution_contract_payload)
    _require_json_binding(
        root_path,
        runtime_lock,
        label="parent_v10_runtime_lock",
        expected_path=PARENT_V10_RUNTIME_LOCK_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_runtime_lock_v10",
    )
    resource_guard = _binding(root_path, PARENT_V10_RESOURCE_GUARD_RELATIVE_PATH)
    run_authorization = _binding(root_path, PARENT_V10_RUN_AUTHORIZATION_RELATIVE_PATH)
    launch_receipt = _binding(root_path, PARENT_V10_OUTPUT_PATHS[0])
    exit_receipt = _binding(root_path, PARENT_V10_OUTPUT_PATHS[1])
    failure_receipt = _binding(root_path, PARENT_V10_OUTPUT_PATHS[2])
    launch_payload = require_canonical_json(
        safe_project_path(root_path, launch_receipt["path"], require_exists=True, require_regular_file=True)
    )
    exit_payload = require_canonical_json(
        safe_project_path(root_path, exit_receipt["path"], require_exists=True, require_regular_file=True)
    )
    failure_payload = require_canonical_json(
        safe_project_path(root_path, failure_receipt["path"], require_exists=True, require_regular_file=True)
    )
    if launch_payload.get("status") != "assigned_and_verified_before_child_resume":
        raise PhaseBContractError("v10 launch did not establish the contained child identity")
    assignment = launch_payload.get("pre_resume_assignment")
    if (
        not isinstance(assignment, Mapping)
        or not isinstance(assignment.get("process_pid"), int)
        or not isinstance(assignment.get("process_creation_time_filetime"), int)
    ):
        raise PhaseBContractError("v10 launch receipt lacks its pre-resume PID")
    if exit_payload.get("status") != "controller_nonzero_exit_with_contained_tree_empty":
        raise PhaseBContractError("v10 exit receipt does not prove contained failure")
    if failure_payload.get("stage") != "post_resume" or failure_payload.get("child_raw_access") != "not_attested_by_supervisor":
        raise PhaseBContractError("v10 failure did not stop before child raw access attestation")
    _assert_absent(
        root_path,
        (
            PARENT_V10_EXECUTION_LEASE_RELATIVE_PATH,
            *PARENT_V10_OUTPUT_PATHS[3:],
        ),
        label="v10 post-launch pre-attestation dynamic surface",
    )
    return {
        "schema": PARENT_V10_PRELEASE_ATTESTATION_SCHEMA,
        "loop_id": LOOP_ID,
        "status": "v10_contained_controller_failed_after_resume_on_launcher_controller_creation_identity_mismatch",
        "parent_v10_source_closure": source_closure,
        "parent_v10_execution_contract": execution_contract,
        "parent_v10_runtime_lock": runtime_lock,
        "v10_resource_guard": resource_guard,
        "v10_run_authorization": run_authorization,
        "v10_launch_receipt": launch_receipt,
        "v10_exit_receipt": exit_receipt,
        "v10_failure_receipt": failure_receipt,
        "v10_supervisor_source": parent_sources["scripts/run_loop167_phase_b_supervisor_v10.py"],
        "v10_controller_source": parent_sources["scripts/run_loop167_phase_b_controller_v10.py"],
        "v10_child_attestation_source": parent_sources["src/loop167_phase_b/child_attestation_v10.py"],
        "v10_controller_canonical_process_argv": ["scripts/run_loop167_phase_b_controller_v10.py", "--execute"],
        "v10_pre_resume_child_pid": assignment["process_pid"],
        "v10_launcher_creation_time_filetime": assignment["process_creation_time_filetime"],
        "v10_failure_stage": "post_resume_controller_creation_time_checked_against_launcher_receipt_before_child_attestation",
        "v10_failure_cause": "controller_current_process_creation_time_compared_to_launcher_receipt_creation_time",
        "v10_child_attestation_absent": True,
        "v10_lease_absent": True,
        "v10_data_outputs_absent": True,
        "replacement_scope": "one_new_v11_redirector_lineage_attestation_remediation_chain_only",
    }


def validate_parent_v10_prelease_attestation_payload_v11(root: Path, payload: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "loop_id",
        "status",
        "parent_v10_source_closure",
        "parent_v10_execution_contract",
        "parent_v10_runtime_lock",
        "v10_resource_guard",
        "v10_run_authorization",
        "v10_launch_receipt",
        "v10_exit_receipt",
        "v10_failure_receipt",
        "v10_supervisor_source",
        "v10_controller_source",
        "v10_child_attestation_source",
        "v10_controller_canonical_process_argv",
        "v10_pre_resume_child_pid",
        "v10_launcher_creation_time_filetime",
        "v10_failure_stage",
        "v10_failure_cause",
        "v10_child_attestation_absent",
        "v10_lease_absent",
        "v10_data_outputs_absent",
        "replacement_scope",
    }
    if set(payload) != expected_keys or payload.get("schema") != PARENT_V10_PRELEASE_ATTESTATION_SCHEMA:
        raise PhaseBContractError("v11 parent v10 attestation schema drifted")
    if (
        payload.get("loop_id") != LOOP_ID
        or payload.get("status")
        != "v10_contained_controller_failed_after_resume_on_launcher_controller_creation_identity_mismatch"
    ):
        raise PhaseBContractError("v11 parent v10 attestation identity drifted")
    expected = build_parent_v10_prelease_attestation_payload_v11(root)
    if dict(payload) != expected:
        raise PhaseBContractError("v11 parent v10 pre-execution attestation facts drifted")


def verify_parent_v10_prelease_attestation_v11(
    root: Path | str,
    binding: Mapping[str, str],
) -> VerifiedParentV10PreleaseAttestationV11:
    root_path = safe_project_root(root)
    normalized = _fixed_binding(
        binding,
        label="parent_v10_prelease_attestation",
        expected_path=PARENT_V10_PRELEASE_ATTESTATION_RELATIVE_PATH,
    )
    path, digest = verify_safe_file_binding(root_path, normalized, label="parent_v10_prelease_attestation")
    payload = require_canonical_json(path)
    validate_parent_v10_prelease_attestation_payload_v11(root_path, payload)
    return VerifiedParentV10PreleaseAttestationV11(
        attestation_path=path,
        attestation_sha256=digest,
        parent_source_closure_binding=MappingProxyType(dict(payload["parent_v10_source_closure"])),
        parent_execution_contract_binding=MappingProxyType(dict(payload["parent_v10_execution_contract"])),
        parent_runtime_lock_binding=MappingProxyType(dict(payload["parent_v10_runtime_lock"])),
    )


def output_catalog_sha256(catalog: object = FIXED_OUTPUT_CATALOG) -> str:
    materialized = list(catalog) if isinstance(catalog, tuple) else catalog
    return sha256_bytes(canonical_json_bytes({"output_catalog": materialized}))


def _validated_output_catalog(catalog: object) -> tuple[Mapping[str, str], ...]:
    materialized = [dict(entry) for entry in catalog] if isinstance(catalog, tuple) else catalog
    if materialized != [dict(entry) for entry in FIXED_OUTPUT_CATALOG]:
        raise PhaseBContractError("v11 output catalog drifted")
    return tuple(MappingProxyType(dict(entry)) for entry in FIXED_OUTPUT_CATALOG)


def resolve_output_catalog_v11(root: Path | str, catalog: object = FIXED_OUTPUT_CATALOG) -> Mapping[str, Path]:
    root_path = safe_project_root(root)
    paths: dict[str, Path] = {}
    for entry in _validated_output_catalog(catalog):
        name = entry["name"]
        path = safe_project_path(root_path, entry["path"], require_exists=False)
        if name in paths:
            raise PhaseBContractError("v11 output catalog repeats a name")
        paths[name] = path
    return MappingProxyType(paths)


def assert_output_catalog_is_fresh_v11(root: Path | str, catalog: object = FIXED_OUTPUT_CATALOG) -> Mapping[str, Path]:
    paths = resolve_output_catalog_v11(root, catalog)
    for path in paths.values():
        if path.exists() or path.is_symlink():
            raise PhaseBContractError("v11 output already exists or is unsafe")
    marker = safe_project_path(root, EXPECTED_LEASE["marker_path"], require_exists=False)
    if marker.exists() or marker.is_symlink():
        raise PhaseBContractError("v11 execution lease already exists or is unsafe")
    return paths


def assert_contained_child_prelease_surface_v11(
    root: Path | str,
    catalog: object = FIXED_OUTPUT_CATALOG,
) -> Mapping[str, Path]:
    """Allow exactly the pre-resume launch receipt after containment is proven."""

    paths = resolve_output_catalog_v11(root, catalog)
    launch_path = paths["supervisor_launch_receipt"]
    if not launch_path.is_file() or launch_path.is_symlink():
        raise PhaseBContractError("v11 contained child requires the pre-resume launch receipt")
    for name, path in paths.items():
        if name == "supervisor_launch_receipt":
            continue
        if path.exists() or path.is_symlink():
            raise PhaseBContractError("v11 contained child prelease output surface drifted")
    marker = safe_project_path(root, EXPECTED_LEASE["marker_path"], require_exists=False)
    if marker.exists() or marker.is_symlink():
        raise PhaseBContractError("v11 contained child execution lease already exists or is unsafe")
    return paths


def assert_attested_child_prelease_surface_v11(
    root: Path | str,
    catalog: object = FIXED_OUTPUT_CATALOG,
) -> Mapping[str, Path]:
    paths = resolve_output_catalog_v11(root, catalog)
    for name in ("supervisor_launch_receipt", "child_job_attestation"):
        path = paths[name]
        if not path.is_file() or path.is_symlink():
            raise PhaseBContractError("v11 attested child requires its containment receipts")
    for name, path in paths.items():
        if name in {"supervisor_launch_receipt", "child_job_attestation"}:
            continue
        if path.exists() or path.is_symlink():
            raise PhaseBContractError("v11 attested child prelease output surface drifted")
    marker = safe_project_path(root, EXPECTED_LEASE["marker_path"], require_exists=False)
    if marker.exists() or marker.is_symlink():
        raise PhaseBContractError("v11 attested child execution lease already exists or is unsafe")
    return paths


def assert_leased_child_pre_raw_surface_v11(
    root: Path | str,
    catalog: object = FIXED_OUTPUT_CATALOG,
) -> Mapping[str, Path]:
    """Allow only containment receipts and the canonical consumed lease before raw access."""

    paths = resolve_output_catalog_v11(root, catalog)
    for name in ("supervisor_launch_receipt", "child_job_attestation"):
        path = paths[name]
        if not path.is_file() or path.is_symlink():
            raise PhaseBContractError("v11 leased child requires its containment receipts")
    for name, path in paths.items():
        if name in {"supervisor_launch_receipt", "child_job_attestation"}:
            continue
        if path.exists() or path.is_symlink():
            raise PhaseBContractError("v11 leased child pre-raw output surface drifted")
    marker = safe_project_path(root, EXPECTED_LEASE["marker_path"], require_exists=True, require_regular_file=True)
    require_canonical_json(marker)
    return paths


def build_execution_contract_payload_v11(
    root: Path | str,
    *,
    parent_v10_prelease_attestation_binding: Mapping[str, str],
) -> dict[str, Any]:
    root_path = safe_project_root(root)
    attestation = verify_parent_v10_prelease_attestation_v11(root_path, parent_v10_prelease_attestation_binding)
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
    assert_output_catalog_is_fresh_v11(root_path)
    return {
        "schema": EXECUTION_CONTRACT_SCHEMA,
        "loop_id": LOOP_ID,
        "phase_b_protocol": protocol_binding,
        "parent_v10_prelease_attestation": {
            "path": PARENT_V10_PRELEASE_ATTESTATION_RELATIVE_PATH,
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
        "prelaunch_available_memory_floor": {
            "enforced": False,
            "authority": "explicit_user_authorization",
            "job_process_memory_cap_bytes": EXPECTED_RESOURCE_CONTRACT["maximum_training_peak_rss_bytes"],
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


def validate_execution_contract_payload_v11(root: Path, payload: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema",
        "loop_id",
        "phase_b_protocol",
        "parent_v10_prelease_attestation",
        "claim_scope",
        "canonical_supervisor_execute_argv",
        "canonical_supervisor_execute_argv_sha256",
        "canonical_controller_execute_argv",
        "canonical_controller_execute_argv_sha256",
        "windows_job_boundary",
        "prelaunch_available_memory_floor",
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
        raise PhaseBContractError("v11 execution contract schema drifted")
    if payload.get("loop_id") != LOOP_ID or payload.get("claim_scope") != CLAIM_SCOPE:
        raise PhaseBContractError("v11 execution contract identity drifted")
    _require_json_binding(
        root,
        payload.get("phase_b_protocol"),
        label="phase_b_protocol",
        expected_path=PHASE_B_PROTOCOL_RELATIVE_PATH,
        expected_schema="axon_loop167_phase_b_protocol_v1",
    )
    verify_parent_v10_prelease_attestation_v11(root, payload["parent_v10_prelease_attestation"])
    if payload.get("canonical_supervisor_execute_argv") != list(CANONICAL_SUPERVISOR_EXECUTE_ARGV):
        raise PhaseBContractError("v11 supervisor argv drifted")
    if payload.get("canonical_supervisor_execute_argv_sha256") != canonical_argv_sha256(CANONICAL_SUPERVISOR_EXECUTE_ARGV):
        raise PhaseBContractError("v11 supervisor argv hash drifted")
    if payload.get("canonical_controller_execute_argv") != list(CANONICAL_CONTROLLER_EXECUTE_ARGV):
        raise PhaseBContractError("v11 controller argv drifted")
    if payload.get("canonical_controller_execute_argv_sha256") != canonical_argv_sha256(CANONICAL_CONTROLLER_EXECUTE_ARGV):
        raise PhaseBContractError("v11 controller argv hash drifted")
    if payload.get("windows_job_boundary") != {
        "abi": "typed_ctypes_windll_use_last_error_and_handle_signatures",
        "guard_probe": "non_kill_current_process_assignment_and_membership",
        "controller_launch": "create_suspended_assign_verify_persist_pre_resume_receipt_then_resume",
        "child_self_attestation_required_before_lease": True,
        "kill_on_close_required": True,
    }:
        raise PhaseBContractError("v11 Windows Job boundary drifted")
    if payload.get("prelaunch_available_memory_floor") != {
        "enforced": False,
        "authority": "explicit_user_authorization",
        "job_process_memory_cap_bytes": EXPECTED_RESOURCE_CONTRACT["maximum_training_peak_rss_bytes"],
    }:
        raise PhaseBContractError("v11 user-authorized memory-floor contract drifted")
    if payload.get("resource_contract") != EXPECTED_RESOURCE_CONTRACT:
        raise PhaseBContractError("v11 resource contract drifted")
    if payload.get("b1_sampling_indicators") != B1_SAMPLING_INDICATORS_CONTRACT:
        raise PhaseBContractError("v11 B1 sampling contract drifted")
    if payload.get("raw_root_relative") != RAW_ROOT_RELATIVE_PATH:
        raise PhaseBContractError("v11 raw root drifted")
    if canonical_project_relative_path(payload["raw_root_relative"]) != RAW_ROOT_RELATIVE_PATH:
        raise PhaseBContractError("v11 raw root is not canonical")
    _validated_output_catalog(payload.get("output_catalog"))
    if payload.get("output_catalog_sha256") != output_catalog_sha256():
        raise PhaseBContractError("v11 output catalog hash drifted")
    if payload.get("lease") != EXPECTED_LEASE or payload.get("forbidden") != EXPECTED_FORBIDDEN:
        raise PhaseBContractError("v11 lease or forbidden scope drifted")
    if payload.get("execution_authorization_required") is not True or payload.get("raw_open_attempts_before_lease") != 0:
        raise PhaseBContractError("v11 prelease contract drifted")
    resolve_output_catalog_v11(root, payload["output_catalog"])


def verify_execution_contract_v11(root: Path | str, binding: Mapping[str, str]) -> VerifiedExecutionContractV11:
    root_path = safe_project_root(root)
    normalized = _fixed_binding(binding, label="execution_contract", expected_path=EXECUTION_CONTRACT_RELATIVE_PATH)
    path, digest = verify_safe_file_binding(root_path, normalized, label="execution_contract")
    payload = require_canonical_json(path)
    validate_execution_contract_payload_v11(root_path, payload)
    return VerifiedExecutionContractV11(
        contract_path=path,
        contract_sha256=digest,
        protocol_binding=MappingProxyType(dict(payload["phase_b_protocol"])),
        canonical_supervisor_execute_argv=tuple(payload["canonical_supervisor_execute_argv"]),
        canonical_controller_execute_argv=tuple(payload["canonical_controller_execute_argv"]),
        resource_contract=MappingProxyType(dict(EXPECTED_RESOURCE_CONTRACT)),
        output_catalog=_validated_output_catalog(payload["output_catalog"]),
        output_paths=resolve_output_catalog_v11(root_path, payload["output_catalog"]),
        lease=MappingProxyType(dict(EXPECTED_LEASE)),
    )


def canonical_controller_execute_argv_v11(argv: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(argv)
    if normalized != CANONICAL_CONTROLLER_EXECUTE_ARGV:
        raise PhaseBContractError("v11 controller execute argv is not sealed")
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
    "PARENT_V10_PRELEASE_ATTESTATION_RELATIVE_PATH",
    "RAW_ROOT_RELATIVE_PATH",
    "REPORT_DIRECTORY",
    "RESOURCE_GUARD_RELATIVE_PATH",
    "RUN_AUTHORIZATION_RELATIVE_PATH",
    "RUNTIME_LOCK_RELATIVE_PATH",
    "SOURCE_CLOSURE_RELATIVE_PATH",
    "SUPERVISOR_RELATIVE_PATH",
    "VNEV_PYTHON_RELATIVE_PATH",
    "VerifiedExecutionContractV11",
    "assert_attested_child_prelease_surface_v11",
    "assert_contained_child_prelease_surface_v11",
    "assert_leased_child_pre_raw_surface_v11",
    "assert_output_catalog_is_fresh_v11",
    "build_execution_contract_payload_v11",
    "build_parent_v10_prelease_attestation_payload_v11",
    "canonical_controller_execute_argv_v11",
    "ensure_v11_static_artifact_parent",
    "output_catalog_sha256",
    "resolve_output_catalog_v11",
    "validate_execution_contract_payload_v11",
    "validate_parent_v10_prelease_attestation_payload_v11",
    "verify_execution_contract_v11",
    "verify_parent_v10_prelease_attestation_v11",
]
