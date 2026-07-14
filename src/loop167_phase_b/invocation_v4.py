"""Canonical isolated-Windows invocation contract for Loop167 Phase-B v4."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from .contracts import PhaseBContractError, canonical_argv_sha256
from .path_safety_v4 import (
    safe_project_path,
    safe_project_relative_path,
    safe_project_root,
)

VNEV_PYTHON_RELATIVE_PATH = "vnev/Scripts/python.exe"
CONTROLLER_V4_RELATIVE_PATH = "scripts/run_loop167_phase_b_controller_v4.py"
EXECUTION_CONTRACT_V4_RELATIVE_PATH = (
    "manifests/roadmap_9997/loop167_ember_v3_novel_delta/phase_b_execution_contract_v4.json"
)
THREAD_ENVIRONMENT_V4 = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
EXTERNAL_RUNTIME_TOP_LEVEL_MODULES = frozenset(
    {"numpy", "scipy", "sklearn", "pefile", "threadpoolctl", "pandas", "torch"}
)
MODE_FLAGS = {"preflight": "--preflight", "execute": "--execute"}


@dataclass(frozen=True)
class VerifiedRuntimeInvocationV4:
    mode: str
    canonical_argv: tuple[str, ...]
    canonical_argv_sha256: str
    executable: Path
    cwd: Path


def _require_mode(mode: object) -> str:
    if not isinstance(mode, str) or mode not in MODE_FLAGS:
        raise PhaseBContractError("Runtime mode must be exactly preflight or execute")
    return mode


def canonical_argv_v4(mode: str) -> tuple[str, ...]:
    """Return the one permitted supervisor argv for a v4 controller mode."""

    normalized_mode = _require_mode(mode)
    return (
        VNEV_PYTHON_RELATIVE_PATH,
        "-I",
        CONTROLLER_V4_RELATIVE_PATH,
        MODE_FLAGS[normalized_mode],
    )


def canonical_controller_argv_v4(mode: str) -> tuple[str, ...]:
    """Return the exact `sys.argv` expected after the canonical Python launch."""

    normalized_mode = _require_mode(mode)
    return (CONTROLLER_V4_RELATIVE_PATH, MODE_FLAGS[normalized_mode])


def canonical_argv_hashes_v4() -> dict[str, str]:
    return {mode: canonical_argv_sha256(canonical_argv_v4(mode)) for mode in MODE_FLAGS}


def _loaded_external_runtime_modules(modules: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        sorted(
            module_name
            for module_name in modules
            if module_name.split(".", 1)[0] in EXTERNAL_RUNTIME_TOP_LEVEL_MODULES
        )
    )


def bootstrap_thread_environment_v4(
    *,
    environment: MutableMapping[str, str] | None = None,
    modules: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Pin native-library threads before any external numerical package can load."""

    observed_modules = sys.modules if modules is None else modules
    loaded_modules = _loaded_external_runtime_modules(observed_modules)
    if loaded_modules:
        raise PhaseBContractError(
            "Thread environment bootstrap ran after external runtime imports: " + ", ".join(loaded_modules)
        )
    target_environment = os.environ if environment is None else environment
    target_environment.update(THREAD_ENVIRONMENT_V4)
    return dict(THREAD_ENVIRONMENT_V4)


def validate_thread_environment_v4(environment: Mapping[str, str] | None = None) -> None:
    observed_environment = os.environ if environment is None else environment
    if any(observed_environment.get(name) != value for name, value in THREAD_ENVIRONMENT_V4.items()):
        raise PhaseBContractError("Runtime thread environment is not pinned before package import")


def validate_launch_argv_v4(argv: Sequence[str], *, mode: str) -> None:
    if isinstance(argv, (str, bytes)) or tuple(argv) != canonical_argv_v4(mode):
        raise PhaseBContractError("Runtime launch argv does not match the canonical v4 command")


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.normpath(os.fspath(left))) == os.path.normcase(
        os.path.normpath(os.fspath(right))
    )


def validate_runtime_envelope_v4(
    root: Path | str,
    *,
    executable: Path | str | None = None,
    cwd: Path | str | None = None,
    isolated: bool | None = None,
    os_name: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    """Check the active Windows vnev, isolated-Python, cwd, and thread prerequisites."""

    root_path = safe_project_root(root)
    observed_os_name = os.name if os_name is None else os_name
    if observed_os_name != "nt":
        raise PhaseBContractError("Loop167 Phase-B v4 may run only under Windows")
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
    if observed_relative != VNEV_PYTHON_RELATIVE_PATH or not _same_path(
        Path(os.path.abspath(os.fspath(observed_executable))), expected_executable
    ):
        raise PhaseBContractError("Loop167 Phase-B v4 must use the project vnev Python executable")
    observed_cwd = Path.cwd() if cwd is None else Path(cwd)
    if not observed_cwd.is_absolute():
        raise PhaseBContractError("Loop167 Phase-B v4 cwd must be absolute")
    observed_cwd = safe_project_root(observed_cwd)
    if not _same_path(observed_cwd, root_path):
        raise PhaseBContractError("Loop167 Phase-B v4 cwd must be the project root")
    observed_isolated = bool(sys.flags.isolated) if isolated is None else isolated
    if observed_isolated is not True:
        raise PhaseBContractError("Loop167 Phase-B v4 requires Python -I isolated mode")
    validate_thread_environment_v4(environment)
    return expected_executable, root_path


def validate_current_runtime_invocation_v4(
    root: Path | str,
    *,
    mode: str,
    executable: Path | str | None = None,
    cwd: Path | str | None = None,
    isolated: bool | None = None,
    os_name: str | None = None,
    process_argv: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> VerifiedRuntimeInvocationV4:
    """Validate the complete active controller invocation after stdlib-only bootstrap."""

    normalized_mode = _require_mode(mode)
    executable_path, root_path = validate_runtime_envelope_v4(
        root,
        executable=executable,
        cwd=cwd,
        isolated=isolated,
        os_name=os_name,
        environment=environment,
    )
    observed_argv = tuple(sys.argv if process_argv is None else process_argv)
    expected_controller_argv = canonical_controller_argv_v4(normalized_mode)
    if observed_argv != expected_controller_argv:
        raise PhaseBContractError("Controller sys.argv does not match the canonical v4 invocation")
    canonical_argv = canonical_argv_v4(normalized_mode)
    return VerifiedRuntimeInvocationV4(
        mode=normalized_mode,
        canonical_argv=canonical_argv,
        canonical_argv_sha256=canonical_argv_sha256(canonical_argv),
        executable=executable_path,
        cwd=root_path,
    )
