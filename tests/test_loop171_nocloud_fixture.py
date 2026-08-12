from __future__ import annotations

import hashlib
import importlib.metadata
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pycdlib = pytest.importorskip("pycdlib")

from build_loop171_nocloud_fixture_iso import (  # noqa: E402
    FILES,
    PYCDLIB_VERSION,
    VOLUME_IDENTIFIER,
    FixtureBuildError,
    build_fixture,
)


def test_pinned_pycdlib_dependency_and_fixture_contents_are_bound() -> None:
    requirements = (PROJECT_ROOT / "requirements-loop171-fixture.txt").read_text(encoding="utf-8")

    assert importlib.metadata.version("pycdlib") == PYCDLIB_VERSION
    assert "pycdlib==1.16.0" in requirements
    assert "17843829c6bf2fd365d3d2e49a94f06da82c999efc313b9f26b0ce59c0070785" in requirements
    assert set(FILES) == {"meta-data", "network-config", "user-data"}
    assert b"#cloud-config" in FILES["user-data"]
    assert b"only_loopback" in FILES["user-data"]
    assert b"write_attempt_blocked" in FILES["user-data"]


def test_fixture_builds_real_iso_and_is_non_overwritable(tmp_path: Path) -> None:
    output = tmp_path / "fixture.iso"

    manifest = build_fixture(output)

    assert manifest["volume_identifier"] == VOLUME_IDENTIFIER
    assert manifest["fixture_iso_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.read_bytes()[32769:32774] == b"CD001"
    with pytest.raises(FixtureBuildError, match="overwrite"):
        build_fixture(output)


def test_fixture_builder_has_no_host_vm_or_sample_operations() -> None:
    source = (SCRIPTS / "build_loop171_nocloud_fixture_iso.py").read_text(encoding="utf-8")
    forbidden = ("subprocess", "New-VM", "Start-VM", "Mount-VHD", "capa", "source_path")

    for token in forbidden:
        assert token not in source
