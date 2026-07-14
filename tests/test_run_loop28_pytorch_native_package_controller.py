from __future__ import annotations

import importlib.util
import json
import stat
import zipfile
from pathlib import Path

import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_loop28_pytorch_native_package_controller.py"
    )
    spec = importlib.util.spec_from_file_location("loop28_native_package_controller", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema": "one", "schema": "two"}', encoding="utf-8")
    with pytest.raises(module.PackageControllerError, match="Duplicate JSON key"):
        module.load_json_strict(path)


def test_activation_script_avoids_nested_cmd_argument_quoting() -> None:
    module = _load_module()
    payload = module._activation_script_bytes().decode("ascii")
    assert 'call "C:\\Program Files\\Microsoft Visual Studio' in payload
    assert "chcp 65001" in payload
    assert "/s /c" not in payload


def test_hardened_archive_accepts_one_safe_precompiled_pyd(tmp_path: Path) -> None:
    module = _load_module()
    package = tmp_path / "safe.pt2"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model/model.pyd", b"precompiled")
        archive.writestr("model/metadata.json", json.dumps({"device": "cpu"}))
    audit = module.audit_aoti_archive(package)
    assert audit["precompiled_pyd_count"] == 1
    assert audit["member_count"] == 2
    assert audit["unsafe_paths"] == 0


@pytest.mark.parametrize(
    "member",
    ["../escape/model.pyd", "/absolute/model.pyd", "C:/drive/model.pyd"],
)
def test_hardened_archive_rejects_unsafe_windows_paths(tmp_path: Path, member: str) -> None:
    module = _load_module()
    package = tmp_path / "unsafe.pt2"
    with zipfile.ZipFile(package, "w") as archive:
        info = zipfile.ZipInfo("placeholder")
        info.filename = member
        archive.writestr(info, b"precompiled")
    with pytest.raises(module.PackageControllerError, match="archive"):
        module.audit_aoti_archive(package)


def test_hardened_archive_rejects_backslash_member(tmp_path: Path) -> None:
    module = _load_module()
    package = tmp_path / "backslash.pt2"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("dir/model.pyd", b"precompiled")
    payload = package.read_bytes().replace(b"dir/model.pyd", b"dir\\model.pyd")
    package.write_bytes(payload)
    with pytest.raises(module.PackageControllerError, match="unsafe path"):
        module.audit_aoti_archive(package)


def test_hardened_archive_rejects_casefold_collision(tmp_path: Path) -> None:
    module = _load_module()
    package = tmp_path / "collision.pt2"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("model/Kernel.pyd", b"one")
        archive.writestr("MODEL/kernel.PYD", b"two")
    with pytest.raises(module.PackageControllerError, match="collision"):
        module.audit_aoti_archive(package)


def test_hardened_archive_rejects_symbolic_link_member(tmp_path: Path) -> None:
    module = _load_module()
    package = tmp_path / "symlink.pt2"
    link = zipfile.ZipInfo("model/model.pyd")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(link, b"target")
    with pytest.raises(module.PackageControllerError, match="symbolic-link"):
        module.audit_aoti_archive(package)


def test_hardened_archive_rejects_multiple_precompiled_pyds(tmp_path: Path) -> None:
    module = _load_module()
    package = tmp_path / "multiple.pt2"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("model/one.pyd", b"one")
        archive.writestr("model/two.pyd", b"two")
    with pytest.raises(module.PackageControllerError, match="exactly one"):
        module.audit_aoti_archive(package)


def test_hardened_archive_rejects_extreme_compression_ratio(tmp_path: Path) -> None:
    module = _load_module()
    package = tmp_path / "compressed.pt2"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model/model.pyd", b"\x00" * (2 * 1024 * 1024))
    with pytest.raises(module.PackageControllerError, match="compression ratio"):
        module.audit_aoti_archive(package)
