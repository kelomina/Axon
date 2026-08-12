"""Canonical isolated-Python invocations for the Loop167 Phase-B v10 route."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from .contracts import PhaseBContractError, canonical_argv_sha256
from .execution_contract_v10 import (
    CONTROLLER_RELATIVE_PATH,
    SUPERVISOR_RELATIVE_PATH,
    VNEV_PYTHON_RELATIVE_PATH,
)
from .path_safety_v4 import safe_project_path, safe_project_relative_path, safe_project_root

MODE_FLAGS = {"preflight": "--preflight", "execute": "--execute"}
THREAD_ENVIRONMENT_V10 = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
EXTERNAL_RUNTIME_TOP_LEVEL_MODULES = frozenset({"numpy", "scipy", "sklearn", "pefile", "threadpoolctl"})


@dataclass(frozen=True)
class VerifiedRuntimeInvocationV10:
    role: str
    mode: str
    canonical_argv: tuple[str, ...]
    canonical_argv_sha256: str
    executable: Path
    cwd: Path


def _mode(mode: str) -> str:
    if mode not in MODE_FLAGS:
        raise PhaseBContractError("v10 mode must be preflight or execute")
    return mode


def _role(role: str) -> str:
    if role not in {"supervisor", "controller"}:
        raise PhaseBContractError("v10 role must be supervisor or controller")
    return role


def canonical_argv_v10(role: str, mode: str) -> tuple[str, ...]:
    normalized_role = _role(role)
    normalized_mode = _mode(mode)
    script = SUPERVISOR_RELATIVE_PATH if normalized_role == "supervisor" else CONTROLLER_RELATIVE_PATH
    return (VNEV_PYTHON_RELATIVE_PATH, "-I", script, MODE_FLAGS[normalized_mode])


def canonical_process_argv_v10(role: str, mode: str) -> tuple[str, ...]:
    normalized_role = _role(role)
    normalized_mode = _mode(mode)
    script = SUPERVISOR_RELATIVE_PATH if normalized_role == "supervisor" else CONTROLLER_RELATIVE_PATH
    return (script, MODE_FLAGS[normalized_mode])


def canonical_argv_hashes_v10() -> dict[str, str]:
    return {
        f"{role}_{mode}": canonical_argv_sha256(canonical_argv_v10(role, mode))
        for role in ("supervisor", "controller")
        for mode in MODE_FLAGS
    }


def _loaded_external_runtime_modules(modules: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in modules
            if name.split(".", 1)[0] in EXTERNAL_RUNTIME_TOP_LEVEL_MODULES
        )
    )


def bootstrap_thread_environment_v10(
    *,
    environment: MutableMapping[str, str] | None = None,
    modules: Mapping[str, object] | None = None,
) -> dict[str, str]:
    observed_modules = sys.modules if modules is None else modules
    loaded = _loaded_external_runtime_modules(observed_modules)
    if loaded:
        raise PhaseBContractError("v10 thread bootstrap ran after external imports: " + ", ".join(loaded))
    target = os.environ if environment is None else environment
    target.update(THREAD_ENVIRONMENT_V10)
    return dict(THREAD_ENVIRONMENT_V10)


def validate_thread_environment_v10(environment: Mapping[str, str] | None = None) -> None:
    observed = os.environ if environment is None else environment
    if any(observed.get(name) != value for name, value in THREAD_ENVIRONMENT_V10.items()):
        raise PhaseBContractError("v10 thread environment is not pinned")


def validate_runtime_envelope_v10(
    root: Path | str,
    *,
    executable: Path | str | None = None,
    cwd: Path | str | None = None,
    isolated: bool | None = None,
    os_name: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    root_path = safe_project_root(root)
    if (os.name if os_name is None else os_name) != "nt":
        raise PhaseBContractError("v10 runs only under Windows")
    expected_executable = safe_project_path(
        root_path,
        VNEV_PYTHON_RELATIVE_PATH,
        require_exists=True,
        require_regular_file=True,
    )
    observed_executable = Path(sys.executable if executable is None else executable)
    observed_relative = safe_project_relative_path(
        root_path,
        observed_executable,
        require_exists=True,
        require_regular_file=True,
    )
    if observed_relative != VNEV_PYTHON_RELATIVE_PATH:
        raise PhaseBContractError("v10 requires the project vnev Python executable")
    observed_cwd = Path.cwd() if cwd is None else Path(cwd)
    if not observed_cwd.is_absolute() or safe_project_root(observed_cwd) != root_path:
        raise PhaseBContractError("v10 requires the canonical project root cwd")
    if (bool(sys.flags.isolated) if isolated is None else isolated) is not True:
        raise PhaseBContractError("v10 requires Python isolated mode")
    validate_thread_environment_v10(environment)
    return expected_executable, root_path


def validate_current_runtime_invocation_v10(
    root: Path | str,
    *,
    role: str,
    mode: str,
    executable: Path | str | None = None,
    cwd: Path | str | None = None,
    isolated: bool | None = None,
    os_name: str | None = None,
    process_argv: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> VerifiedRuntimeInvocationV10:
    normalized_role = _role(role)
    normalized_mode = _mode(mode)
    executable_path, root_path = validate_runtime_envelope_v10(
        root,
        executable=executable,
        cwd=cwd,
        isolated=isolated,
        os_name=os_name,
        environment=environment,
    )
    observed_argv = tuple(sys.argv if process_argv is None else process_argv)
    if observed_argv != canonical_process_argv_v10(normalized_role, normalized_mode):
        raise PhaseBContractError("v10 process argv differs from the sealed invocation")
    argv = canonical_argv_v10(normalized_role, normalized_mode)
    return VerifiedRuntimeInvocationV10(
        role=normalized_role,
        mode=normalized_mode,
        canonical_argv=argv,
        canonical_argv_sha256=canonical_argv_sha256(argv),
        executable=executable_path,
        cwd=root_path,
    )


__all__ = [
    "CONTROLLER_RELATIVE_PATH",
    "EXTERNAL_RUNTIME_TOP_LEVEL_MODULES",
    "MODE_FLAGS",
    "SUPERVISOR_RELATIVE_PATH",
    "THREAD_ENVIRONMENT_V10",
    "VNEV_PYTHON_RELATIVE_PATH",
    "VerifiedRuntimeInvocationV10",
    "bootstrap_thread_environment_v10",
    "canonical_argv_hashes_v10",
    "canonical_argv_v10",
    "canonical_process_argv_v10",
    "validate_current_runtime_invocation_v10",
    "validate_runtime_envelope_v10",
    "validate_thread_environment_v10",
]
