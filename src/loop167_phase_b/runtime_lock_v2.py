"""Runtime-lock v2 for isolated Python, without an impossible hash-seed environment requirement."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import PhaseBContractError, canonical_argv_sha256, sha256_file, verify_file_binding

RUNTIME_PACKAGES = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("pefile", "pefile"),
    ("threadpoolctl", "threadpoolctl"),
)
REQUIRED_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
PYTHON_HASH_SEED_POLICY = "isolated_mode_ignores_environment_and_model_contract_must_not_depend_on_hash_seed"


def _module_binding(distribution: str, module_name: str) -> dict[str, str]:
    module = importlib.import_module(module_name)
    module_path = getattr(module, "__file__", None)
    if not isinstance(module_path, str) or not Path(module_path).is_file():
        raise PhaseBContractError(f"Runtime module has no stable source path: {module_name}")
    path = Path(module_path).resolve(strict=True)
    return {
        "distribution": distribution,
        "module": module_name,
        "version": importlib.metadata.version(distribution),
        "module_path": str(path),
        "module_sha256": sha256_file(path),
    }


def build_runtime_lock_payload(
    root: Path,
    *,
    controller_binding: Mapping[str, str],
    isolation_addendum_binding: Mapping[str, str],
    canonical_argv: Sequence[str],
) -> dict[str, Any]:
    controller_path, controller_sha256 = verify_file_binding(root, dict(controller_binding), label="controller")
    verify_file_binding(root, dict(isolation_addendum_binding), label="runtime_isolation_addendum")
    executable = Path(sys.executable).resolve(strict=True)
    return {
        "schema": "axon_loop167_phase_b_runtime_lock_v2",
        "loop_id": "loop167_ember_v3_novel_delta",
        "cwd_contract": "project_root",
        "python": {
            "executable": str(executable),
            "sha256": sha256_file(executable),
            "implementation": platform.python_implementation(),
            "version": sys.version,
        },
        "packages": [_module_binding(distribution, module_name) for distribution, module_name in RUNTIME_PACKAGES],
        "controller": {
            "path": controller_path.relative_to(root.resolve(strict=True)).as_posix(),
            "sha256": controller_sha256,
        },
        "runtime_isolation_addendum": dict(isolation_addendum_binding),
        "canonical_argv": list(canonical_argv),
        "canonical_argv_sha256": canonical_argv_sha256(canonical_argv),
        "required_environment": dict(REQUIRED_ENVIRONMENT),
        "python_hash_seed_policy": PYTHON_HASH_SEED_POLICY,
        "isolated_python_required": True,
        "network_fetch_allowed": False,
        "dependency_install_allowed": False,
    }


def validate_runtime_lock(
    root: Path,
    payload: Mapping[str, Any],
    *,
    controller_binding: Mapping[str, str],
    isolation_addendum_binding: Mapping[str, str],
    canonical_argv: Sequence[str],
    environment: Mapping[str, str] | None = None,
) -> None:
    expected_keys = {
        "schema",
        "loop_id",
        "cwd_contract",
        "python",
        "packages",
        "controller",
        "runtime_isolation_addendum",
        "canonical_argv",
        "canonical_argv_sha256",
        "required_environment",
        "python_hash_seed_policy",
        "isolated_python_required",
        "network_fetch_allowed",
        "dependency_install_allowed",
    }
    if set(payload) != expected_keys:
        raise PhaseBContractError("Runtime lock v2 fields drifted")
    if payload["schema"] != "axon_loop167_phase_b_runtime_lock_v2":
        raise PhaseBContractError("Runtime lock v2 schema drifted")
    if payload["loop_id"] != "loop167_ember_v3_novel_delta" or payload["cwd_contract"] != "project_root":
        raise PhaseBContractError("Runtime lock v2 identity drifted")
    if payload["isolated_python_required"] is not True:
        raise PhaseBContractError("Runtime lock v2 must require isolated Python")
    if payload["python_hash_seed_policy"] != PYTHON_HASH_SEED_POLICY:
        raise PhaseBContractError("Runtime lock v2 hash-seed policy drifted")
    if payload["network_fetch_allowed"] is not False or payload["dependency_install_allowed"] is not False:
        raise PhaseBContractError("Runtime lock v2 allows an unsafe dependency action")
    if payload["canonical_argv"] != list(canonical_argv):
        raise PhaseBContractError("Runtime lock v2 canonical argv drifted")
    if payload["canonical_argv_sha256"] != canonical_argv_sha256(canonical_argv):
        raise PhaseBContractError("Runtime lock v2 argv hash drifted")
    if payload["runtime_isolation_addendum"] != dict(isolation_addendum_binding):
        raise PhaseBContractError("Runtime lock v2 isolation-addendum binding drifted")

    expected_controller_path, expected_controller_sha256 = verify_file_binding(
        root,
        dict(controller_binding),
        label="controller",
    )
    expected_controller = {
        "path": expected_controller_path.relative_to(root.resolve(strict=True)).as_posix(),
        "sha256": expected_controller_sha256,
    }
    if payload["controller"] != expected_controller:
        raise PhaseBContractError("Runtime lock v2 controller binding drifted")

    executable = Path(sys.executable).resolve(strict=True)
    expected_python = {
        "executable": str(executable),
        "sha256": sha256_file(executable),
        "implementation": platform.python_implementation(),
        "version": sys.version,
    }
    if payload["python"] != expected_python:
        raise PhaseBContractError("Runtime lock v2 Python binding drifted")
    expected_packages = [_module_binding(distribution, module_name) for distribution, module_name in RUNTIME_PACKAGES]
    if payload["packages"] != expected_packages:
        raise PhaseBContractError("Runtime lock v2 package binding drifted")
    if payload["required_environment"] != REQUIRED_ENVIRONMENT:
        raise PhaseBContractError("Runtime lock v2 environment contract drifted")
    observed_environment = os.environ if environment is None else environment
    if any(observed_environment.get(name) != value for name, value in REQUIRED_ENVIRONMENT.items()):
        raise PhaseBContractError("Runtime lock v2 environment is not pinned")
