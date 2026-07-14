"""Capture and validate the pinned Python runtime needed by Loop167 Phase B."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    PhaseBContractError,
    canonical_argv_sha256,
    resolve_project_file,
    sha256_file,
    verify_file_binding,
)

RUNTIME_PACKAGES = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("pefile", "pefile"),
    ("threadpoolctl", "threadpoolctl"),
)
REQUIRED_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


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
    canonical_argv: Sequence[str],
) -> dict[str, Any]:
    controller_path, controller_sha256 = verify_file_binding(root, dict(controller_binding), label="controller")
    executable = Path(sys.executable).resolve(strict=True)
    return {
        "schema": "axon_loop167_phase_b_runtime_lock_v1",
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
        "canonical_argv": list(canonical_argv),
        "canonical_argv_sha256": canonical_argv_sha256(canonical_argv),
        "required_environment": dict(REQUIRED_ENVIRONMENT),
        "isolated_python_required": True,
        "network_fetch_allowed": False,
        "dependency_install_allowed": False,
    }


def validate_runtime_lock(
    root: Path,
    payload: Mapping[str, Any],
    *,
    controller_binding: Mapping[str, str],
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
        "canonical_argv",
        "canonical_argv_sha256",
        "required_environment",
        "isolated_python_required",
        "network_fetch_allowed",
        "dependency_install_allowed",
    }
    if set(payload) != expected_keys:
        raise PhaseBContractError("Runtime lock fields drifted")
    if payload["schema"] != "axon_loop167_phase_b_runtime_lock_v1":
        raise PhaseBContractError("Runtime lock schema drifted")
    if payload["loop_id"] != "loop167_ember_v3_novel_delta":
        raise PhaseBContractError("Runtime lock loop id drifted")
    if payload["cwd_contract"] != "project_root":
        raise PhaseBContractError("Runtime lock cwd contract drifted")
    if payload["isolated_python_required"] is not True:
        raise PhaseBContractError("Runtime lock must require isolated Python")
    if payload["network_fetch_allowed"] is not False or payload["dependency_install_allowed"] is not False:
        raise PhaseBContractError("Runtime lock allows an unsafe dependency action")
    if payload["canonical_argv"] != list(canonical_argv):
        raise PhaseBContractError("Runtime lock canonical argv drifted")
    if payload["canonical_argv_sha256"] != canonical_argv_sha256(canonical_argv):
        raise PhaseBContractError("Runtime lock argv hash drifted")

    expected_controller_path, expected_controller_sha256 = verify_file_binding(
        root,
        dict(controller_binding),
        label="controller",
    )
    controller = payload["controller"]
    if not isinstance(controller, dict) or controller != {
        "path": expected_controller_path.relative_to(root.resolve(strict=True)).as_posix(),
        "sha256": expected_controller_sha256,
    }:
        raise PhaseBContractError("Runtime lock controller binding drifted")

    python = payload["python"]
    executable = Path(sys.executable).resolve(strict=True)
    if not isinstance(python, dict) or python != {
        "executable": str(executable),
        "sha256": sha256_file(executable),
        "implementation": platform.python_implementation(),
        "version": sys.version,
    }:
        raise PhaseBContractError("Runtime lock Python binding drifted")

    expected_packages = [_module_binding(distribution, module_name) for distribution, module_name in RUNTIME_PACKAGES]
    if payload["packages"] != expected_packages:
        raise PhaseBContractError("Runtime lock package binding drifted")
    observed_environment = os.environ if environment is None else environment
    if payload["required_environment"] != REQUIRED_ENVIRONMENT:
        raise PhaseBContractError("Runtime lock environment contract drifted")
    if any(observed_environment.get(name) != value for name, value in REQUIRED_ENVIRONMENT.items()):
        raise PhaseBContractError("Runtime lock environment is not pinned")


def runtime_lock_binding(root: Path, relative_path: str) -> dict[str, str]:
    path = resolve_project_file(root, relative_path)
    if not path.is_file() or path.is_symlink():
        raise PhaseBContractError("Runtime lock file is missing or unsafe")
    return {"path": relative_path, "sha256": sha256_file(path)}
