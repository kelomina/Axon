from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.loop167_phase_b.execution_authorization_v8 as execution_authorization_v8
import src.loop167_phase_b.execution_contract_v8 as execution_contract_v8
import src.loop167_phase_b.preflight_v8 as preflight_v8
import src.loop167_phase_b.resource_guard_v8 as resource_guard_v8
from src.loop167_phase_b.contracts import PhaseBContractError


def _symlink_binding(root: Path, relative_path: str) -> dict[str, str]:
    outside = root.parent / "outside.json"
    outside.write_bytes(b"{}\n")
    link = root / relative_path
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform")
    return {"path": relative_path, "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()}


def _regular_binding(root: Path, relative_path: str) -> tuple[Path, dict[str, str]]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"{}\n")
    return path, {"path": relative_path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _mark_target_as_reparse_point(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    original_lstat = Path.lstat
    target_name = os.path.normcase(os.path.abspath(target))

    def lstat(path: Path):
        result = original_lstat(path)
        if os.path.normcase(os.path.abspath(path)) == target_name:
            return SimpleNamespace(
                st_mode=result.st_mode,
                st_file_attributes=0x0400,
            )
        return result

    monkeypatch.setattr(Path, "lstat", lstat)


@pytest.mark.parametrize("surface", ("contract", "preflight", "authorization", "resource_guard"))
def test_v7_control_planes_reject_a_simulated_windows_reparse_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    relative_path = "manifests/target.json"
    target, binding = _regular_binding(root, relative_path)
    _mark_target_as_reparse_point(monkeypatch, target)

    with pytest.raises(PhaseBContractError, match="symlink or reparse"):
        if surface == "contract":
            execution_contract_v8._require_json_binding(
                root,
                binding,
                label="target",
                expected_path=relative_path,
                expected_schema="synthetic",
            )
        elif surface == "preflight":
            preflight_v8._verify_source_files(root, [binding])
        elif surface == "authorization":
            execution_authorization_v8._binding(
                root,
                binding,
                label="target",
                expected_path=relative_path,
            )
        else:
            resource_guard_v8._binding(
                root,
                binding,
                label="target",
                expected_path=relative_path,
                expected_schema="synthetic",
            )


def test_v7_execution_contract_rejects_an_in_root_symlink_binding(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    relative_path = "manifests/protocol.json"
    binding = _symlink_binding(root, relative_path)

    with pytest.raises(PhaseBContractError, match="symlink or reparse"):
        execution_contract_v8._require_json_binding(
            root,
            binding,
            label="protocol",
            expected_path=relative_path,
            expected_schema="synthetic",
        )


def test_v7_preflight_rejects_an_in_root_symlink_source_file(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    binding = _symlink_binding(root, "src/control.py")

    with pytest.raises(PhaseBContractError, match="symlink or reparse"):
        preflight_v8._verify_source_files(root, [binding])


def test_v7_authorization_rejects_an_in_root_symlink_binding(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    relative_path = "scripts/controller.py"
    binding = _symlink_binding(root, relative_path)

    with pytest.raises(PhaseBContractError, match="symlink or reparse"):
        execution_authorization_v8._binding(
            root,
            binding,
            label="controller",
            expected_path=relative_path,
        )


def test_v7_resource_guard_rejects_an_in_root_symlink_binding(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    relative_path = "manifests/source.json"
    binding = _symlink_binding(root, relative_path)

    with pytest.raises(PhaseBContractError, match="symlink or reparse"):
        resource_guard_v8._binding(
            root,
            binding,
            label="source_closure",
            expected_path=relative_path,
            expected_schema="synthetic",
        )
