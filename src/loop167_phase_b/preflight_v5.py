"""Raw-free static validation for the Loop167 Phase-B v5 controller.

This module deliberately stops before resource authorization, input-manifest
parsing, and numerical-runtime validation.  Those actions occur only in the
authorized execution path after the one-shot lease boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    PhaseBContractError,
    require_canonical_json,
    require_sha256,
    verify_file_binding,
)
from .execution_contract_v5 import (
    EXECUTION_CONTRACT_RELATIVE_PATH,
    EXPECTED_LEASE,
    FIXED_OUTPUT_CATALOG,
    LOOP_ID,
    PARENT_V4_PRELEASE_ATTESTATION_RELATIVE_PATH,
    RESOURCE_GUARD_RELATIVE_PATH,
    RUN_AUTHORIZATION_RELATIVE_PATH,
    RUNTIME_LOCK_RELATIVE_PATH,
    SOURCE_CLOSURE_RELATIVE_PATH,
    VerifiedExecutionContractV5,
    verify_execution_contract_v5,
    verify_parent_v4_prelease_attestation_v5,
)
from .invocation_v5 import (
    CONTROLLER_V5_RELATIVE_PATH,
    THREAD_ENVIRONMENT_V5,
    VNEV_PYTHON_RELATIVE_PATH,
    canonical_argv_hashes_v5,
    canonical_argv_v5,
)

SOURCE_CLOSURE_V4_RELATIVE_PATH = (
    "manifests/roadmap_9997/loop167_ember_v3_novel_delta/phase_b_source_closure_v4.json"
)
PROTOCOL_ADDITION_RELATIVE_PATH = (
    "manifests/roadmap_9997/loop167_ember_v3_novel_delta/phase_b_protocol_addendum.json"
)
V5_SCOPE = "v5_mappingproxy_boundary_remediation_single_authorized_train_only_execution_no_heldout_access"
PHASE_A_BINDING_NAMES = (
    "proposal",
    "authorization",
    "semantic_delta_mapping",
    "frozen_deduplicated_baseline_allowlist",
    "source_semantics_addendum",
    "source_closure",
    "static_decision",
)
EXPECTED_DYNAMIC_GATES = {
    "fresh_resource_guard_required": True,
    "run_authorization_required": True,
    "one_shot_lease_required": True,
}
EXPECTED_BLOCKERS = [
    "fresh_resource_guard_v5_not_sealed",
    "run_authorization_v5_not_sealed",
    "one_shot_lease_v5_not_consumed",
]
FORBIDDEN_CLOSURE_ARTIFACT_PATHS = frozenset(
    {
        RESOURCE_GUARD_RELATIVE_PATH,
        RUN_AUTHORIZATION_RELATIVE_PATH,
        str(EXPECTED_LEASE["marker_path"]),
        *(str(entry["path"]) for entry in FIXED_OUTPUT_CATALOG),
    }
)


@dataclass(frozen=True)
class StaticPreflightV5Receipt:
    """Only immutable bindings that are safe before the one-shot lease."""

    protocol_sha256: str
    source_closure_sha256: str
    execution_contract_sha256: str
    runtime_lock_sha256: str
    source_closure_binding: Mapping[str, str]
    execution_contract_binding: Mapping[str, str]
    runtime_lock_binding: Mapping[str, str]
    controller_binding: Mapping[str, str]
    raw_open_attempts: int


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PhaseBContractError(f"{label} must be an object")
    return value


def _require_binding_path(binding: object, *, label: str, expected_path: str) -> Mapping[str, str]:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise PhaseBContractError(f"{label} must be a file binding")
    if binding.get("path") != expected_path:
        raise PhaseBContractError(f"{label} path drifted from the fixed v5 route")
    require_sha256(binding.get("sha256"), field=f"{label}.sha256")
    return binding


def _verify_phase_a_bindings(root: Path, bindings: object) -> None:
    values = _mapping(bindings, label="phase_a_bindings")
    if set(values) != set(PHASE_A_BINDING_NAMES):
        raise PhaseBContractError("Phase-A binding names drifted")
    for name in PHASE_A_BINDING_NAMES:
        verify_file_binding(root, values[name], label=f"phase_a.{name}")


def _validate_v4_provenance(root: Path, binding: object) -> None:
    _require_binding_path(
        binding,
        label="supersedes_source_closure_v4",
        expected_path=SOURCE_CLOSURE_V4_RELATIVE_PATH,
    )
    path, _ = verify_file_binding(root, binding, label="source_closure_v4")
    payload = require_canonical_json(path)
    if payload.get("schema") != "axon_loop167_phase_b_source_closure_v4":
        raise PhaseBContractError("Superseded source closure is not the sealed v4 provenance")
    if payload.get("loop_id") != LOOP_ID:
        raise PhaseBContractError("Superseded source closure loop id drifted")


def _validate_protocol_and_addendum(
    root: Path,
    *,
    protocol_binding: Mapping[str, str],
    addendum_binding: Mapping[str, str],
    phase_a_bindings: object,
) -> tuple[str, Mapping[str, Any]]:
    protocol_path, protocol_sha256 = verify_file_binding(root, protocol_binding, label="phase_b_protocol")
    protocol = require_canonical_json(protocol_path)
    if protocol.get("schema") != "axon_loop167_phase_b_protocol_v1" or protocol.get("loop_id") != LOOP_ID:
        raise PhaseBContractError("Phase-B protocol schema or loop id drifted")
    _verify_phase_a_bindings(root, protocol.get("phase_a_bindings"))
    if protocol.get("phase_a_bindings") != phase_a_bindings:
        raise PhaseBContractError("Phase-A bindings differ between protocol and v5 closure")

    _require_binding_path(
        addendum_binding,
        label="phase_b_protocol_addendum",
        expected_path=PROTOCOL_ADDITION_RELATIVE_PATH,
    )
    addendum_path, _ = verify_file_binding(root, addendum_binding, label="phase_b_protocol_addendum")
    addendum = require_canonical_json(addendum_path)
    if addendum.get("parent_phase_b_protocol") != dict(protocol_binding):
        raise PhaseBContractError("Phase-B protocol addendum parent binding drifted")
    return protocol_sha256, protocol


def _validate_runtime_lock_static_v5(
    root: Path,
    *,
    runtime_lock_binding: Mapping[str, str],
    controller_binding: Mapping[str, str],
    execution_contract_binding: Mapping[str, str],
    canonical_preflight_argv: Sequence[str],
) -> str:
    """Validate immutable lock facts without importing its pinned packages."""

    _require_binding_path(
        runtime_lock_binding,
        label="runtime_lock_v5",
        expected_path=RUNTIME_LOCK_RELATIVE_PATH,
    )
    lock_path, lock_sha256 = verify_file_binding(root, runtime_lock_binding, label="runtime_lock_v5")
    lock = require_canonical_json(lock_path)
    expected_keys = {
        "schema",
        "loop_id",
        "runtime_platform",
        "cwd_contract",
        "project_root_no_symlink_or_reparse_required",
        "python",
        "packages",
        "controller",
        "execution_contract",
        "canonical_argv",
        "canonical_argv_sha256",
        "thread_environment",
        "thread_environment_bootstrap_before_external_imports_required",
        "isolated_python_required",
        "network_fetch_allowed",
        "dependency_install_allowed",
    }
    if set(lock) != expected_keys:
        raise PhaseBContractError("Runtime lock v5 fields drifted")
    if lock.get("schema") != "axon_loop167_phase_b_runtime_lock_v5" or lock.get("loop_id") != LOOP_ID:
        raise PhaseBContractError("Runtime lock v5 schema or loop id drifted")
    if lock.get("runtime_platform") != "windows":
        raise PhaseBContractError("Runtime lock v5 platform drifted")
    if lock.get("cwd_contract") != "project_root_without_symlink_or_reparse":
        raise PhaseBContractError("Runtime lock v5 cwd contract drifted")
    if lock.get("project_root_no_symlink_or_reparse_required") is not True:
        raise PhaseBContractError("Runtime lock v5 root-safety requirement drifted")
    if lock.get("controller") != dict(controller_binding):
        raise PhaseBContractError("Runtime lock v5 controller binding drifted")
    if lock.get("execution_contract") != dict(execution_contract_binding):
        raise PhaseBContractError("Runtime lock v5 execution-contract binding drifted")
    expected_argv = {mode: list(canonical_argv_v5(mode)) for mode in ("preflight", "execute")}
    if lock.get("canonical_argv") != expected_argv:
        raise PhaseBContractError("Runtime lock v5 canonical argv drifted")
    if lock.get("canonical_argv_sha256") != canonical_argv_hashes_v5():
        raise PhaseBContractError("Runtime lock v5 canonical argv hashes drifted")
    if tuple(canonical_preflight_argv) != canonical_argv_v5("preflight"):
        raise PhaseBContractError("Static preflight argv is not the fixed v5 invocation")
    if lock.get("thread_environment") != THREAD_ENVIRONMENT_V5:
        raise PhaseBContractError("Runtime lock v5 thread environment drifted")
    if (
        lock.get("thread_environment_bootstrap_before_external_imports_required") is not True
        or lock.get("isolated_python_required") is not True
        or lock.get("network_fetch_allowed") is not False
        or lock.get("dependency_install_allowed") is not False
    ):
        raise PhaseBContractError("Runtime lock v5 execution policy drifted")

    python_binding = _mapping(lock.get("python"), label="runtime_lock.python")
    if set(python_binding) != {"relative_path", "sha256", "implementation", "version"}:
        raise PhaseBContractError("Runtime lock v5 Python binding fields drifted")
    if python_binding.get("relative_path") != VNEV_PYTHON_RELATIVE_PATH:
        raise PhaseBContractError("Runtime lock v5 Python path drifted")
    require_sha256(python_binding.get("sha256"), field="runtime_lock.python.sha256")

    packages = lock.get("packages")
    if not isinstance(packages, list) or len(packages) != 5:
        raise PhaseBContractError("Runtime lock v5 package inventory drifted")
    expected_packages = {
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("pefile", "pefile"),
        ("threadpoolctl", "threadpoolctl"),
    }
    observed_packages: set[tuple[str, str]] = set()
    for package in packages:
        package_binding = _mapping(package, label="runtime_lock.package")
        if set(package_binding) != {"distribution", "module", "relative_path", "sha256", "version"}:
            raise PhaseBContractError("Runtime lock v5 package binding fields drifted")
        distribution = package_binding.get("distribution")
        module = package_binding.get("module")
        if not isinstance(distribution, str) or not isinstance(module, str):
            raise PhaseBContractError("Runtime lock v5 package identity is invalid")
        observed_packages.add((distribution, module))
        relative_path = package_binding.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path.startswith("vnev/"):
            raise PhaseBContractError("Runtime lock v5 package escapes the vnev")
        require_sha256(package_binding.get("sha256"), field=f"runtime_lock.{module}.sha256")
        if not isinstance(package_binding.get("version"), str) or not package_binding["version"]:
            raise PhaseBContractError("Runtime lock v5 package version is invalid")
    if observed_packages != expected_packages:
        raise PhaseBContractError("Runtime lock v5 package inventory drifted")
    return lock_sha256


def _validate_source_files(
    root: Path,
    source_files: object,
    *,
    controller_binding: Mapping[str, str],
) -> None:
    if not isinstance(source_files, list) or not source_files:
        raise PhaseBContractError("Phase-B source closure v5 has no source files")
    observed_paths: set[str] = set()
    for source_binding in source_files:
        path, _ = verify_file_binding(root, source_binding, label="source_file_v5")
        relative_path = path.relative_to(root.resolve(strict=True)).as_posix()
        if relative_path in observed_paths:
            raise PhaseBContractError("Phase-B source closure v5 repeats a source file")
        if relative_path in FORBIDDEN_CLOSURE_ARTIFACT_PATHS:
            raise PhaseBContractError("Source closure v5 must not bind dynamic guard, authorization, lease, or outputs")
        observed_paths.add(relative_path)
    if dict(controller_binding) not in source_files:
        raise PhaseBContractError("Controller v5 is not bound by the source closure")
    if "src/loop167_phase_b/preflight_v5.py" not in observed_paths:
        raise PhaseBContractError("Source closure v5 omits the static preflight module")
    if "src/loop167_phase_b/__init__.py" not in observed_paths:
        raise PhaseBContractError("Source closure v5 omits the lazy package boundary")


def validate_static_preflight_v5(
    root: Path,
    *,
    source_closure_binding: Mapping[str, str],
    controller_binding: Mapping[str, str],
    canonical_preflight_argv: Sequence[str],
) -> StaticPreflightV5Receipt:
    """Validate static provenance without touching raw inputs or outputs."""

    _require_binding_path(
        source_closure_binding,
        label="source_closure_v5",
        expected_path=SOURCE_CLOSURE_RELATIVE_PATH,
    )
    _require_binding_path(
        controller_binding,
        label="controller_v5",
        expected_path=CONTROLLER_V5_RELATIVE_PATH,
    )
    closure_path, closure_sha256 = verify_file_binding(root, source_closure_binding, label="source_closure_v5")
    closure = require_canonical_json(closure_path)
    expected_keys = {
        "schema",
        "loop_id",
        "scope",
        "supersedes_source_closure_v4",
        "parent_v4_prelease_attestation",
        "phase_a_bindings",
        "phase_b_protocol",
        "phase_b_protocol_addendum",
        "phase_b_execution_contract",
        "runtime_lock_v5",
        "source_files",
        "static_preflight_ready",
        "phase_b_raw_execution_ready",
        "dynamic_execution_gates",
        "remaining_execution_blockers",
    }
    if set(closure) != expected_keys:
        raise PhaseBContractError("Phase-B source closure v5 fields drifted")
    if closure.get("schema") != "axon_loop167_phase_b_source_closure_v5" or closure.get("loop_id") != LOOP_ID:
        raise PhaseBContractError("Phase-B source closure v5 schema or loop id drifted")
    if closure.get("scope") != V5_SCOPE:
        raise PhaseBContractError("Phase-B source closure v5 scope drifted")
    if closure.get("static_preflight_ready") is not True or closure.get("phase_b_raw_execution_ready") is not False:
        raise PhaseBContractError("Phase-B source closure v5 execution state drifted")
    if closure.get("dynamic_execution_gates") != EXPECTED_DYNAMIC_GATES:
        raise PhaseBContractError("Phase-B source closure v5 dynamic-gate contract drifted")
    if closure.get("remaining_execution_blockers") != EXPECTED_BLOCKERS:
        raise PhaseBContractError("Phase-B source closure v5 blocker contract drifted")

    _validate_v4_provenance(root, closure["supersedes_source_closure_v4"])
    parent_attestation_binding = _require_binding_path(
        closure["parent_v4_prelease_attestation"],
        label="parent_v4_prelease_attestation",
        expected_path=PARENT_V4_PRELEASE_ATTESTATION_RELATIVE_PATH,
    )
    verify_parent_v4_prelease_attestation_v5(root, parent_attestation_binding)
    _verify_phase_a_bindings(root, closure["phase_a_bindings"])
    protocol_sha256, _ = _validate_protocol_and_addendum(
        root,
        protocol_binding=closure["phase_b_protocol"],
        addendum_binding=closure["phase_b_protocol_addendum"],
        phase_a_bindings=closure["phase_a_bindings"],
    )
    execution_contract_binding = _require_binding_path(
        closure["phase_b_execution_contract"],
        label="phase_b_execution_contract",
        expected_path=EXECUTION_CONTRACT_RELATIVE_PATH,
    )
    verified_contract: VerifiedExecutionContractV5 = verify_execution_contract_v5(
        root,
        execution_contract_binding,
        expected_protocol_binding=closure["phase_b_protocol"],
    )
    runtime_lock_binding = _require_binding_path(
        closure["runtime_lock_v5"],
        label="runtime_lock_v5",
        expected_path=RUNTIME_LOCK_RELATIVE_PATH,
    )
    runtime_lock_sha256 = _validate_runtime_lock_static_v5(
        root,
        runtime_lock_binding=runtime_lock_binding,
        controller_binding=controller_binding,
        execution_contract_binding=execution_contract_binding,
        canonical_preflight_argv=canonical_preflight_argv,
    )
    _validate_source_files(root, closure["source_files"], controller_binding=controller_binding)
    _, controller_sha256 = verify_file_binding(root, controller_binding, label="controller_v5")
    return StaticPreflightV5Receipt(
        protocol_sha256=protocol_sha256,
        source_closure_sha256=closure_sha256,
        execution_contract_sha256=verified_contract.contract_sha256,
        runtime_lock_sha256=runtime_lock_sha256,
        source_closure_binding=dict(source_closure_binding),
        execution_contract_binding=dict(execution_contract_binding),
        runtime_lock_binding=dict(runtime_lock_binding),
        controller_binding={"path": CONTROLLER_V5_RELATIVE_PATH, "sha256": controller_sha256},
        raw_open_attempts=0,
    )
