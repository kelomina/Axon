"""Load sealed Loop166 process proofs without initializing its PE extraction package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping

from .contracts import sha256_file
from .path_safety_v4 import verify_safe_file_binding


class JobMembershipV7Error(RuntimeError):
    """The sealed Loop166 Job membership proof could not be completed."""


class ProcessLineageV7Error(RuntimeError):
    """The sealed Loop166 parent-lineage proof could not be completed."""


def _load_loop166_module(
    root: str,
    relative_path: str,
    expected_sha256: str,
) -> Any:
    path, observed_sha256 = verify_safe_file_binding(
        Path(root),
        {"path": relative_path, "sha256": expected_sha256},
        label="v7_loop166_proof",
    )
    module_name = f"_axon_loop167_v7_{path.stem}_{observed_sha256[:16]}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Loop166 v7 bridge cannot load: {relative_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if sha256_file(path) != observed_sha256:
        sys.modules.pop(module_name, None)
        raise RuntimeError(f"Loop166 v7 bridge source changed while loading: {relative_path}")
    return module


def _proof_module(
    root: Path | str,
    binding: Mapping[str, str],
    *,
    expected_path: str,
) -> Any:
    if dict(binding).get("path") != expected_path or set(binding) != {"path", "sha256"}:
        raise RuntimeError(f"Loop166 v7 bridge binding drifted: {expected_path}")
    return _load_loop166_module(
        str(Path(root)),
        expected_path,
        str(binding["sha256"]),
    )


def audit_current_process_job_membership(
    root: Path | str,
    binding: Mapping[str, str],
    expected_creation_time_filetime: int,
    *,
    expected_pid: int,
) -> dict[str, Any]:
    try:
        return _proof_module(
            root,
            binding,
            expected_path="src/loop166/windows_job.py",
        ).audit_current_process_job_membership(
            expected_creation_time_filetime,
            expected_pid=expected_pid,
        )
    except Exception as error:
        raise JobMembershipV7Error("v7 current child Job membership proof failed") from error


def audit_process_job_membership(
    root: Path | str,
    binding: Mapping[str, str],
    pid: int,
    expected_creation_time_filetime: int,
) -> dict[str, Any]:
    try:
        return _proof_module(
            root,
            binding,
            expected_path="src/loop166/windows_job.py",
        ).audit_process_job_membership(
            pid,
            expected_creation_time_filetime,
        )
    except Exception as error:
        raise JobMembershipV7Error("v7 launcher Job membership proof failed") from error


def validate_spawn_lineage(
    root: Path | str,
    binding: Mapping[str, str],
    expected_parent_pid: int,
    *,
    launcher_executable: str | Path,
    base_executable: str | Path,
) -> dict[str, Any]:
    try:
        return _proof_module(
            root,
            binding,
            expected_path="src/loop166/windows_process_lineage.py",
        ).validate_spawn_lineage(
            expected_parent_pid,
            launcher_executable=launcher_executable,
            base_executable=base_executable,
        )
    except Exception as error:
        raise ProcessLineageV7Error("v7 child process lineage proof failed") from error


__all__ = [
    "JobMembershipV7Error",
    "ProcessLineageV7Error",
    "audit_current_process_job_membership",
    "audit_process_job_membership",
    "validate_spawn_lineage",
]
