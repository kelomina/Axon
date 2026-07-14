"""Runtime lock v5 with a verified controller invocation and execution contract binding."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import PhaseBContractError, require_canonical_json, sha256_file
from .execution_contract_v5 import LOOP_ID
from .invocation_v5 import (
    CONTROLLER_V5_RELATIVE_PATH,
    EXECUTION_CONTRACT_V5_RELATIVE_PATH,
    THREAD_ENVIRONMENT_V5,
    VNEV_PYTHON_RELATIVE_PATH,
    _loaded_external_runtime_modules,
    canonical_argv_hashes_v5,
    canonical_argv_v5,
    validate_current_runtime_invocation_v5,
    validate_runtime_envelope_v5,
    validate_thread_environment_v5,
)
from .path_safety_v4 import (
    canonical_project_relative_path,
    safe_project_path,
    safe_project_relative_path,
    safe_project_root,
    verify_safe_file_binding,
)

RUNTIME_PACKAGES_V5 = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("pefile", "pefile"),
    ("threadpoolctl", "threadpoolctl"),
)
RUNTIME_LOCK_V5_SCHEMA = "axon_loop167_phase_b_runtime_lock_v5"


@dataclass(frozen=True)
class VerifiedRuntimeLockV5:
    """The frozen runtime facts that the v5 execution authorization may bind."""

    mode: str
    controller_sha256: str
    execution_contract_sha256: str
    canonical_argv_sha256: str


def _require_clean_external_import_state() -> None:
    loaded_modules = _loaded_external_runtime_modules(sys.modules)
    if loaded_modules:
        raise PhaseBContractError(
            "Runtime lock v5 was checked after external runtime imports: " + ", ".join(loaded_modules)
        )


def _python_binding(root: Path) -> dict[str, str]:
    executable = Path(sys.executable)
    relative_path = safe_project_relative_path(
        root,
        executable,
        require_exists=True,
        require_regular_file=True,
    )
    if relative_path != VNEV_PYTHON_RELATIVE_PATH:
        raise PhaseBContractError("Runtime lock v5 Python executable is outside the required vnev path")
    return {
        "relative_path": relative_path,
        "sha256": sha256_file(executable),
        "implementation": platform.python_implementation(),
        "version": sys.version,
    }


def _module_binding(root: Path, distribution: str, module_name: str) -> dict[str, str]:
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise PhaseBContractError(f"Runtime module has no stable source path: {module_name}")
    path = Path(module_file)
    relative_path = safe_project_relative_path(
        root,
        path,
        require_exists=True,
        require_regular_file=True,
    )
    if not relative_path.startswith("vnev/"):
        raise PhaseBContractError(f"Runtime module is outside the project vnev: {module_name}")
    return {
        "distribution": distribution,
        "module": module_name,
        "relative_path": relative_path,
        "sha256": sha256_file(path),
        "version": importlib.metadata.version(distribution),
    }


def _runtime_packages(root: Path) -> list[dict[str, str]]:
    return [_module_binding(root, distribution, module_name) for distribution, module_name in RUNTIME_PACKAGES_V5]


def _validate_bindings(
    root: Path,
    *,
    controller_binding: Mapping[str, str],
    execution_contract_binding: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    controller_path, controller_sha256 = verify_safe_file_binding(root, controller_binding, label="controller_v5")
    controller_relative_path = safe_project_relative_path(
        root,
        controller_path,
        require_exists=True,
        require_regular_file=True,
    )
    if controller_relative_path != CONTROLLER_V5_RELATIVE_PATH:
        raise PhaseBContractError("Runtime lock v5 controller path drifted")
    execution_path, execution_sha256 = verify_safe_file_binding(
        root,
        execution_contract_binding,
        label="execution_contract_v5",
    )
    execution_relative_path = safe_project_relative_path(
        root,
        execution_path,
        require_exists=True,
        require_regular_file=True,
    )
    if execution_relative_path != EXECUTION_CONTRACT_V5_RELATIVE_PATH:
        raise PhaseBContractError("Runtime lock v5 execution-contract path drifted")
    require_canonical_json(execution_path)
    return (
        {"path": controller_relative_path, "sha256": controller_sha256},
        {"path": execution_relative_path, "sha256": execution_sha256},
    )


def _canonical_argv_payload() -> tuple[dict[str, list[str]], dict[str, str]]:
    return (
        {mode: list(canonical_argv_v5(mode)) for mode in ("preflight", "execute")},
        canonical_argv_hashes_v5(),
    )


def build_runtime_lock_payload_v5(
    root: Path | str,
    *,
    controller_binding: Mapping[str, str],
    execution_contract_binding: Mapping[str, str],
) -> dict[str, Any]:
    """Build a lock only from the actual isolated Windows vnev runtime."""

    root_path = safe_project_root(root)
    validate_runtime_envelope_v5(root_path)
    _require_clean_external_import_state()
    controller, execution_contract = _validate_bindings(
        root_path,
        controller_binding=controller_binding,
        execution_contract_binding=execution_contract_binding,
    )
    canonical_argv, canonical_argv_sha256 = _canonical_argv_payload()
    return {
        "schema": RUNTIME_LOCK_V5_SCHEMA,
        "loop_id": LOOP_ID,
        "runtime_platform": "windows",
        "cwd_contract": "project_root_without_symlink_or_reparse",
        "project_root_no_symlink_or_reparse_required": True,
        "python": _python_binding(root_path),
        "packages": _runtime_packages(root_path),
        "controller": controller,
        "execution_contract": execution_contract,
        "canonical_argv": canonical_argv,
        "canonical_argv_sha256": canonical_argv_sha256,
        "thread_environment": dict(THREAD_ENVIRONMENT_V5),
        "thread_environment_bootstrap_before_external_imports_required": True,
        "isolated_python_required": True,
        "network_fetch_allowed": False,
        "dependency_install_allowed": False,
    }


def validate_runtime_lock_v5(
    root: Path | str,
    payload: Mapping[str, Any],
    *,
    controller_binding: Mapping[str, str],
    execution_contract_binding: Mapping[str, str],
    mode: str,
    verify_current_runtime: bool = True,
) -> VerifiedRuntimeLockV5:
    """Validate the lock and, by default, the live canonical controller process."""

    root_path = safe_project_root(root)
    canonical_argv_v5(mode)
    if verify_current_runtime:
        _require_clean_external_import_state()
        invocation = validate_current_runtime_invocation_v5(root_path, mode=mode)
    else:
        invocation = None
        validate_thread_environment_v5()
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
    if set(payload) != expected_keys:
        raise PhaseBContractError("Runtime lock v5 fields drifted")
    if payload["schema"] != RUNTIME_LOCK_V5_SCHEMA:
        raise PhaseBContractError("Runtime lock v5 schema drifted")
    if payload["loop_id"] != LOOP_ID or payload["runtime_platform"] != "windows":
        raise PhaseBContractError("Runtime lock v5 identity drifted")
    if payload["cwd_contract"] != "project_root_without_symlink_or_reparse":
        raise PhaseBContractError("Runtime lock v5 cwd contract drifted")
    if payload["project_root_no_symlink_or_reparse_required"] is not True:
        raise PhaseBContractError("Runtime lock v5 root safety requirement drifted")
    if payload["thread_environment"] != THREAD_ENVIRONMENT_V5:
        raise PhaseBContractError("Runtime lock v5 thread environment drifted")
    if payload["thread_environment_bootstrap_before_external_imports_required"] is not True:
        raise PhaseBContractError("Runtime lock v5 thread bootstrap requirement drifted")
    if payload["isolated_python_required"] is not True:
        raise PhaseBContractError("Runtime lock v5 must require isolated Python")
    if payload["network_fetch_allowed"] is not False or payload["dependency_install_allowed"] is not False:
        raise PhaseBContractError("Runtime lock v5 allows an unsafe dependency action")
    expected_controller, expected_execution_contract = _validate_bindings(
        root_path,
        controller_binding=controller_binding,
        execution_contract_binding=execution_contract_binding,
    )
    if payload["controller"] != expected_controller:
        raise PhaseBContractError("Runtime lock v5 controller binding drifted")
    if payload["execution_contract"] != expected_execution_contract:
        raise PhaseBContractError("Runtime lock v5 execution-contract binding drifted")
    expected_argv, expected_argv_sha256 = _canonical_argv_payload()
    if payload["canonical_argv"] != expected_argv or payload["canonical_argv_sha256"] != expected_argv_sha256:
        raise PhaseBContractError("Runtime lock v5 canonical argv drifted")
    expected_python = _python_binding(root_path)
    if payload["python"] != expected_python:
        raise PhaseBContractError("Runtime lock v5 Python binding drifted")
    expected_packages = _runtime_packages(root_path)
    if payload["packages"] != expected_packages:
        raise PhaseBContractError("Runtime lock v5 package binding drifted")
    if invocation is not None and invocation.canonical_argv_sha256 != expected_argv_sha256[mode]:
        raise PhaseBContractError("Runtime lock v5 active invocation hash drifted")
    return VerifiedRuntimeLockV5(
        mode=mode,
        controller_sha256=expected_controller["sha256"],
        execution_contract_sha256=expected_execution_contract["sha256"],
        canonical_argv_sha256=expected_argv_sha256[mode],
    )


def runtime_lock_binding_v5(root: Path | str, relative_path: object) -> dict[str, str]:
    """Return a no-link binding for a sealed v5 runtime-lock artifact."""

    canonical_path = canonical_project_relative_path(relative_path)
    safe_path = safe_project_path(root, canonical_path, require_exists=True, require_regular_file=True)
    path, sha256 = verify_safe_file_binding(
        root,
        {"path": canonical_path, "sha256": sha256_file(safe_path)},
        label="runtime_lock_v5",
    )
    return {
        "path": safe_project_relative_path(root, path, require_exists=True, require_regular_file=True),
        "sha256": sha256,
    }
