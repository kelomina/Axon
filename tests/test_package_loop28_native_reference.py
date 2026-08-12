from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import package_loop28_native_reference as package  # noqa: E402


def _write(path: Path, payload: bytes = b"asset") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _project_root(tmp_path: Path) -> Path:
    for _, source in package.ASSETS:
        _write(tmp_path / source)
    return tmp_path


def _abi_receipt(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": package.ABI_SCHEMA,
                "calling_convention": "__cdecl",
                "exports": [{"ordinal": ordinal, "name": name} for ordinal, name in enumerate(range(18), start=1)],
                "kvd_config_x64_size": 96,
                "kvd_parity_diagnostics_options_v1_x64_size": 56,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_package_writes_native_reference_manifest_and_deterministic_zip(tmp_path: Path) -> None:
    root = _project_root(tmp_path / "project")
    output_parent = tmp_path / "dist"
    output_parent.mkdir()
    receipt = _abi_receipt(tmp_path / "abi.json")
    result = package.package_native_reference(
        project_root=root,
        output_directory=output_parent / "reference",
        zip_path=output_parent / "reference.zip",
        abi_receipt=receipt,
    )

    manifest = json.loads((output_parent / "reference" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_classification"] == "native_reference_not_loop151_research_champion"
    assert manifest["abi"]["calling_convention"] == "__cdecl"
    assert len(manifest["assets"]) == len(package.ASSETS)
    assert result["zip_sha256"] == hashlib.sha256((output_parent / "reference.zip").read_bytes()).hexdigest()


def test_package_rejects_existing_output_and_invalid_abi_receipt(tmp_path: Path) -> None:
    root = _project_root(tmp_path / "project")
    output_parent = tmp_path / "dist"
    output_parent.mkdir()
    receipt = _abi_receipt(tmp_path / "abi.json")
    output = output_parent / "reference"
    output.mkdir()
    with pytest.raises(package.PackageError, match="already exists"):
        package.package_native_reference(project_root=root, output_directory=output, zip_path=output_parent / "reference.zip", abi_receipt=receipt)

    output.rmdir()
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(package.PackageError, match="schema"):
        package.package_native_reference(project_root=root, output_directory=output, zip_path=output_parent / "reference.zip", abi_receipt=receipt)
