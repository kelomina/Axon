"""Strict project-root and project-relative path checks for Phase-B v4."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Mapping

from .contracts import PhaseBContractError, require_sha256, sha256_file

WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
WINDOWS_RESERVED_BASENAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)


def _normalized_path_string(path: Path) -> str:
    return os.path.normcase(os.path.normpath(os.fspath(path)))


def _same_path(left: Path, right: Path) -> bool:
    return _normalized_path_string(left) == _normalized_path_string(right)


def _path_chain(path: Path) -> tuple[Path, ...]:
    anchor = Path(path.anchor)
    anchor_parts = anchor.parts
    current = anchor
    result = [current]
    for part in path.parts[len(anchor_parts) :]:
        current = current / part
        result.append(current)
    return tuple(result)


def _lstat_or_error(path: Path, *, label: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as error:
        raise PhaseBContractError(f"{label} cannot be inspected safely: {path}") from error


def _is_link_or_reparse(stat_result: os.stat_result) -> bool:
    attributes = int(getattr(stat_result, "st_file_attributes", 0))
    return stat.S_ISLNK(stat_result.st_mode) or bool(attributes & WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT)


def _require_safe_existing_path(path: Path, *, label: str) -> os.stat_result:
    stat_result = _lstat_or_error(path, label=label)
    if _is_link_or_reparse(stat_result):
        raise PhaseBContractError(f"{label} must not be a symlink or Windows reparse point: {path}")
    return stat_result


def _canonical_relative_parts(relative_path: object) -> tuple[str, ...]:
    if not isinstance(relative_path, str) or not relative_path:
        raise PhaseBContractError("Project path must be a nonempty relative string")
    if "\x00" in relative_path or "\\" in relative_path or ":" in relative_path:
        raise PhaseBContractError("Project path must use canonical forward-slash relative syntax")
    if relative_path.startswith("/") or relative_path.endswith("/"):
        raise PhaseBContractError("Project path must not have an absolute or trailing separator")
    parts = tuple(relative_path.split("/"))
    if any(not part or part in {".", ".."} for part in parts):
        raise PhaseBContractError("Project path contains a non-canonical component")
    for part in parts:
        if part.endswith((".", " ")):
            raise PhaseBContractError("Project path contains a Windows-ambiguous component")
        device_basename = part.split(".", 1)[0].upper()
        if device_basename in WINDOWS_RESERVED_BASENAMES:
            raise PhaseBContractError("Project path contains a reserved Windows device name")
    return parts


def canonical_project_relative_path(relative_path: object) -> str:
    """Return one strict, portable spelling for an in-project path."""

    return "/".join(_canonical_relative_parts(relative_path))


def safe_project_root(root: Path | str) -> Path:
    """Require an existing absolute project directory with no link or reparse traversal."""

    root_path = Path(root)
    if not root_path.is_absolute():
        raise PhaseBContractError("Project root must be an absolute path")
    root_path = Path(os.path.abspath(os.fspath(root_path)))
    for chain_path in _path_chain(root_path):
        stat_result = _require_safe_existing_path(chain_path, label="Project root component")
        if not stat.S_ISDIR(stat_result.st_mode):
            raise PhaseBContractError("Project root traversal contains a non-directory component")
    try:
        resolved_root = root_path.resolve(strict=True)
    except OSError as error:
        raise PhaseBContractError("Project root cannot be resolved safely") from error
    if not _same_path(root_path, resolved_root):
        raise PhaseBContractError("Project root resolves through an unsafe alias")
    return root_path


def safe_project_path(
    root: Path | str,
    relative_path: object,
    *,
    require_exists: bool,
    require_regular_file: bool = False,
) -> Path:
    """Resolve a canonical project-relative path without following links or reparse points."""

    root_path = safe_project_root(root)
    canonical_path = canonical_project_relative_path(relative_path)
    parts = canonical_path.split("/")
    candidate = root_path.joinpath(*parts)
    current = root_path
    final_stat: os.stat_result | None = None
    for index, part in enumerate(parts):
        current = current / part
        try:
            stat_result = current.lstat()
        except FileNotFoundError:
            if require_exists:
                raise PhaseBContractError(f"Project path is missing: {canonical_path}") from None
            break
        except OSError as error:
            raise PhaseBContractError(f"Project path cannot be inspected safely: {canonical_path}") from error
        if _is_link_or_reparse(stat_result):
            raise PhaseBContractError(f"Project path traverses a symlink or reparse point: {canonical_path}")
        if index < len(parts) - 1 and not stat.S_ISDIR(stat_result.st_mode):
            raise PhaseBContractError(f"Project path traverses a non-directory component: {canonical_path}")
        if index == len(parts) - 1:
            final_stat = stat_result
    if require_exists and final_stat is None:
        raise PhaseBContractError(f"Project path is missing: {canonical_path}")
    if require_regular_file:
        if final_stat is None or not stat.S_ISREG(final_stat.st_mode):
            raise PhaseBContractError(f"Project path must be a regular file: {canonical_path}")
    return candidate


def safe_project_relative_path(
    root: Path | str,
    path: Path | str,
    *,
    require_exists: bool,
    require_regular_file: bool = False,
) -> str:
    """Validate an absolute path lies safely under root and return its canonical relative spelling."""

    root_path = safe_project_root(root)
    candidate = Path(path)
    if not candidate.is_absolute():
        raise PhaseBContractError("Observed project path must be absolute")
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    try:
        common_path = os.path.commonpath((_normalized_path_string(root_path), _normalized_path_string(candidate)))
    except ValueError as error:
        raise PhaseBContractError("Observed project path is on another filesystem root") from error
    if common_path != _normalized_path_string(root_path):
        raise PhaseBContractError("Observed project path escapes the project root")
    relative = os.path.relpath(os.fspath(candidate), os.fspath(root_path)).replace(os.sep, "/")
    canonical_path = canonical_project_relative_path(relative)
    safe_path = safe_project_path(
        root_path,
        canonical_path,
        require_exists=require_exists,
        require_regular_file=require_regular_file,
    )
    if not _same_path(candidate, safe_path):
        raise PhaseBContractError("Observed project path is not canonically rooted")
    return canonical_path


def verify_safe_file_binding(root: Path | str, binding: object, *, label: str) -> tuple[Path, str]:
    """Verify a canonical file binding using v4's no-link project traversal."""

    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise PhaseBContractError(f"{label} binding must contain exactly path and sha256")
    relative_path = canonical_project_relative_path(binding["path"])
    expected_sha256 = require_sha256(binding["sha256"], field=f"{label}.sha256")
    path = safe_project_path(root, relative_path, require_exists=True, require_regular_file=True)
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise PhaseBContractError(f"{label} hash mismatch")
    return path, observed_sha256
