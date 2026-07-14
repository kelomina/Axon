from __future__ import annotations

import hashlib
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.loop167_phase_b.contracts import PhaseBContractError
from src.loop167_phase_b.path_safety_v4 import (
    _is_link_or_reparse,
    canonical_project_relative_path,
    safe_project_path,
    safe_project_relative_path,
    safe_project_root,
    verify_safe_file_binding,
)


def test_path_safety_accepts_a_regular_canonical_project_file(tmp_path: Path) -> None:
    root = tmp_path / "project"
    target = root / "scripts" / "controller.py"
    target.parent.mkdir(parents=True)
    target.write_text("print('synthetic')\n", encoding="ascii")

    assert safe_project_root(root) == root
    assert canonical_project_relative_path("scripts/controller.py") == "scripts/controller.py"
    assert safe_project_path(root, "scripts/controller.py", require_exists=True, require_regular_file=True) == target
    assert safe_project_relative_path(root, target, require_exists=True, require_regular_file=True) == "scripts/controller.py"
    binding = {"path": "scripts/controller.py", "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    assert verify_safe_file_binding(root, binding, label="controller")[0] == target


@pytest.mark.parametrize(
    "relative_path",
    ("../outside.py", "scripts/../controller.py", "/absolute.py", "scripts\\controller.py", "C:/outside.py"),
)
def test_path_safety_rejects_noncanonical_or_escaping_paths(tmp_path: Path, relative_path: str) -> None:
    root = tmp_path / "project"
    root.mkdir()

    with pytest.raises(PhaseBContractError):
        safe_project_path(root, relative_path, require_exists=False)


def test_path_safety_rejects_symlink_traversal_and_symlink_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "target.py").write_text("outside\n", encoding="ascii")
    escaped = root / "escaped"
    try:
        escaped.symlink_to(outside, target_is_directory=True)
        root_link = tmp_path / "project-link"
        root_link.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform")

    with pytest.raises(PhaseBContractError):
        safe_project_path(root, "escaped/target.py", require_exists=True, require_regular_file=True)
    with pytest.raises(PhaseBContractError):
        safe_project_root(root_link)


def test_reparse_point_bit_is_treated_as_unsafe() -> None:
    synthetic_stat = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x0400)

    assert _is_link_or_reparse(synthetic_stat) is True
