from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from src.loop167_phase_b import loop166_v7_bridge
from src.loop167_phase_b.contracts import PhaseBContractError
from src.loop167_phase_b.loop166_v7_bridge import _proof_module


def _write_proof(root: Path, content: bytes) -> Path:
    path = root / "src" / "loop166" / "windows_job.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return path


def test_v7_bridge_rejects_hash_drift_before_executing_the_proof_module(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _write_proof(root, b"raise RuntimeError('proof module must not execute')\n")

    with pytest.raises(PhaseBContractError, match="hash mismatch"):
        _proof_module(
            root,
            {"path": "src/loop166/windows_job.py", "sha256": "0" * 64},
            expected_path="src/loop166/windows_job.py",
        )


def test_v7_bridge_rejects_an_in_root_symlink_before_executing_the_proof_module(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"raise RuntimeError('symlink target must not execute')\n")
    link = root / "src" / "loop166" / "windows_job.py"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform")

    with pytest.raises(PhaseBContractError, match="symlink or reparse"):
        _proof_module(
            root,
            {
                "path": "src/loop166/windows_job.py",
                "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            },
            expected_path="src/loop166/windows_job.py",
        )


def test_v7_bridge_rejects_a_proof_changed_while_its_loader_executes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    original_bytes = b"proof_value = 'original'\n"
    rewritten_bytes = b"proof_value = 'rewritten'\n"
    proof_path = _write_proof(root, original_bytes)
    expected_sha256 = hashlib.sha256(original_bytes).hexdigest()
    module_name = f"_axon_loop167_v7_{proof_path.stem}_{expected_sha256[:16]}"

    class RewritingLoader:
        def create_module(self, specification: object) -> None:
            return None

        def exec_module(self, module: object) -> None:
            proof_path.write_bytes(rewritten_bytes)

    def spec_from_file_location(module_name: str, path: Path) -> importlib.machinery.ModuleSpec:
        assert module_name.startswith("_axon_loop167_v7_")
        assert path == proof_path
        specification = importlib.util.spec_from_loader(module_name, RewritingLoader())
        assert specification is not None
        return specification

    monkeypatch.setattr(
        loop166_v7_bridge.importlib.util,
        "spec_from_file_location",
        spec_from_file_location,
    )

    with pytest.raises(RuntimeError, match="source changed while loading"):
        _proof_module(
            root,
            {"path": "src/loop166/windows_job.py", "sha256": expected_sha256},
            expected_path="src/loop166/windows_job.py",
        )

    assert proof_path.read_bytes() == rewritten_bytes
    assert module_name not in sys.modules
