"""Pinned runtime facts for a contained Loop167 Phase-B v12 controller."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import PhaseBContractError, require_canonical_json, sha256_file
from .execution_contract_v12 import (
    CONTROLLER_RELATIVE_PATH,
    EXECUTION_CONTRACT_RELATIVE_PATH,
    LOOP_ID,
    SUPERVISOR_RELATIVE_PATH,
)
from .invocation_v12 import (
    THREAD_ENVIRONMENT_V12,
    _loaded_external_runtime_modules,
    canonical_argv_hashes_v12,
    canonical_argv_v12,
    validate_current_runtime_invocation_v12,
    validate_runtime_envelope_v12,
    validate_thread_environment_v12,
)
from .path_safety_v4 import (
    safe_project_path,
    safe_project_relative_path,
    safe_project_root,
    verify_safe_file_binding,
)

RUNTIME_LOCK_SCHEMA = "axon_loop167_phase_b_runtime_lock_v12"
RUNTIME_PACKAGES = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("pefile", "pefile"),
    ("threadpoolctl", "threadpoolctl"),
)


@dataclass(frozen=True)
class VerifiedRuntimeLockV12:
    role: str
    mode: str
    controller_sha256: str
    supervisor_sha256: str
    execution_contract_sha256: str


def _clean_external_import_state() -> None:
    loaded = _loaded_external_runtime_modules(sys.modules)
    if loaded:
        raise PhaseBContractError("v12 runtime lock checked after external imports: " + ", ".join(loaded))


def _python_binding(root: Path) -> dict[str, str]:
    executable = Path(sys.executable)
    relative = safe_project_relative_path(root, executable, require_exists=True, require_regular_file=True)
    if relative != "vnev/Scripts/python.exe":
        raise PhaseBContractError("v12 runtime Python is outside vnev")
    return {
        "relative_path": relative,
        "sha256": sha256_file(executable),
        "implementation": platform.python_implementation(),
        "version": sys.version,
    }


def _module_binding(root: Path, distribution: str, module_name: str) -> dict[str, str]:
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise PhaseBContractError(f"v12 runtime module lacks a file: {module_name}")
    relative = safe_project_relative_path(root, module_file, require_exists=True, require_regular_file=True)
    if not relative.startswith("vnev/"):
        raise PhaseBContractError(f"v12 runtime module is outside vnev: {module_name}")
    return {
        "distribution": distribution,
        "module": module_name,
        "relative_path": relative,
        "sha256": sha256_file(Path(module_file)),
        "version": importlib.metadata.version(distribution),
    }


def _source_binding(root: Path, binding: Mapping[str, str], *, expected_path: str, label: str) -> dict[str, str]:
    path, digest = verify_safe_file_binding(root, binding, label=label)
    relative = safe_project_relative_path(root, path, require_exists=True, require_regular_file=True)
    if relative != expected_path:
        raise PhaseBContractError(f"v12 runtime lock {label} path drifted")
    return {"path": relative, "sha256": digest}


def build_runtime_lock_payload_v12(
    root: Path | str,
    *,
    controller_binding: Mapping[str, str],
    supervisor_binding: Mapping[str, str],
    execution_contract_binding: Mapping[str, str],
) -> dict[str, Any]:
    root_path = safe_project_root(root)
    validate_runtime_envelope_v12(root_path)
    _clean_external_import_state()
    controller = _source_binding(root_path, controller_binding, expected_path=CONTROLLER_RELATIVE_PATH, label="controller")
    supervisor = _source_binding(root_path, supervisor_binding, expected_path=SUPERVISOR_RELATIVE_PATH, label="supervisor")
    execution_contract = _source_binding(
        root_path,
        execution_contract_binding,
        expected_path=EXECUTION_CONTRACT_RELATIVE_PATH,
        label="execution_contract",
    )
    require_canonical_json(safe_project_path(root_path, EXECUTION_CONTRACT_RELATIVE_PATH, require_exists=True, require_regular_file=True))
    return {
        "schema": RUNTIME_LOCK_SCHEMA,
        "loop_id": LOOP_ID,
        "runtime_platform": "windows",
        "python": _python_binding(root_path),
        "packages": [_module_binding(root_path, distribution, module) for distribution, module in RUNTIME_PACKAGES],
        "controller": controller,
        "supervisor": supervisor,
        "execution_contract": execution_contract,
        "canonical_argv": {
            f"{role}_{mode}": list(canonical_argv_v12(role, mode))
            for role in ("supervisor", "controller")
            for mode in ("preflight", "execute")
        },
        "canonical_argv_sha256": canonical_argv_hashes_v12(),
        "thread_environment": dict(THREAD_ENVIRONMENT_V12),
        "isolated_python_required": True,
        "network_fetch_allowed": False,
        "dependency_install_allowed": False,
    }


def validate_runtime_lock_v12(
    root: Path | str,
    payload: Mapping[str, Any],
    *,
    controller_binding: Mapping[str, str],
    supervisor_binding: Mapping[str, str],
    execution_contract_binding: Mapping[str, str],
    role: str,
    mode: str,
    verify_current_runtime: bool = True,
) -> VerifiedRuntimeLockV12:
    root_path = safe_project_root(root)
    expected_keys = {
        "schema",
        "loop_id",
        "runtime_platform",
        "python",
        "packages",
        "controller",
        "supervisor",
        "execution_contract",
        "canonical_argv",
        "canonical_argv_sha256",
        "thread_environment",
        "isolated_python_required",
        "network_fetch_allowed",
        "dependency_install_allowed",
    }
    if set(payload) != expected_keys or payload.get("schema") != RUNTIME_LOCK_SCHEMA:
        raise PhaseBContractError("v12 runtime lock schema drifted")
    if payload.get("loop_id") != LOOP_ID or payload.get("runtime_platform") != "windows":
        raise PhaseBContractError("v12 runtime lock identity drifted")
    if payload.get("thread_environment") != THREAD_ENVIRONMENT_V12:
        raise PhaseBContractError("v12 runtime lock thread environment drifted")
    if payload.get("isolated_python_required") is not True:
        raise PhaseBContractError("v12 runtime lock must require isolated Python")
    if payload.get("network_fetch_allowed") is not False or payload.get("dependency_install_allowed") is not False:
        raise PhaseBContractError("v12 runtime lock permits unsafe runtime actions")
    if verify_current_runtime:
        _clean_external_import_state()
        validate_current_runtime_invocation_v12(root_path, role=role, mode=mode)
    else:
        validate_thread_environment_v12()
    expected_controller = _source_binding(root_path, controller_binding, expected_path=CONTROLLER_RELATIVE_PATH, label="controller")
    expected_supervisor = _source_binding(root_path, supervisor_binding, expected_path=SUPERVISOR_RELATIVE_PATH, label="supervisor")
    expected_contract = _source_binding(
        root_path,
        execution_contract_binding,
        expected_path=EXECUTION_CONTRACT_RELATIVE_PATH,
        label="execution_contract",
    )
    if payload.get("controller") != expected_controller or payload.get("supervisor") != expected_supervisor:
        raise PhaseBContractError("v12 runtime lock source binding drifted")
    if payload.get("execution_contract") != expected_contract:
        raise PhaseBContractError("v12 runtime lock execution contract binding drifted")
    expected_argv = {
        f"{role_name}_{mode_name}": list(canonical_argv_v12(role_name, mode_name))
        for role_name in ("supervisor", "controller")
        for mode_name in ("preflight", "execute")
    }
    if payload.get("canonical_argv") != expected_argv or payload.get("canonical_argv_sha256") != canonical_argv_hashes_v12():
        raise PhaseBContractError("v12 runtime lock argv drifted")
    if payload.get("python") != _python_binding(root_path):
        raise PhaseBContractError("v12 runtime lock Python binding drifted")
    expected_packages = [_module_binding(root_path, distribution, module) for distribution, module in RUNTIME_PACKAGES]
    if payload.get("packages") != expected_packages:
        raise PhaseBContractError("v12 runtime lock package binding drifted")
    return VerifiedRuntimeLockV12(
        role=role,
        mode=mode,
        controller_sha256=expected_controller["sha256"],
        supervisor_sha256=expected_supervisor["sha256"],
        execution_contract_sha256=expected_contract["sha256"],
    )


__all__ = [
    "RUNTIME_LOCK_SCHEMA",
    "VerifiedRuntimeLockV12",
    "build_runtime_lock_payload_v12",
    "validate_runtime_lock_v12",
]
