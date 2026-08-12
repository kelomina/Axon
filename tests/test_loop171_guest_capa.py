from __future__ import annotations

import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop171.capa_aggregate import CapabilityAggregate  # noqa: E402
from src.loop171.guest_capa import (  # noqa: E402
    GUEST_RECEIPT_SCHEMA,
    GuestCapaError,
    _tree_sha256,
    install_linux_capa_zip,
    receipt_payload,
    run_guest_capa,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mountinfo(path: Path, root: Path, *, read_only: bool = True) -> Path:
    mountinfo = path / "mountinfo"
    mode = "ro" if read_only else "rw"
    mountinfo.write_text(f"42 35 0:42 / {root} {mode},relatime - tmpfs tmpfs {mode}\n", encoding="utf-8")
    return mountinfo


def _guest_paths(tmp_path: Path) -> dict[str, Path]:
    input_root = tmp_path / "input"
    input_root.mkdir()
    source = input_root / "fixture.bin"
    source.write_bytes(b"harmless fixture")
    archive = tmp_path / "capa-v9.4.0-linux.zip"
    archive.write_bytes(b"bound archive fixture")
    capa = tmp_path / "capa"
    capa.write_bytes(b"\x7fELFfixture")
    capa.chmod(capa.stat().st_mode | stat.S_IXUSR)
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "collection.yml").write_text("rule: fixture\n", encoding="utf-8")
    return {
        "source": source,
        "archive": archive,
        "capa": capa,
        "rules": rules,
        "mountinfo": _mountinfo(tmp_path, tmp_path),
    }


def test_guest_run_binds_readonly_source_and_persists_only_aggregate(tmp_path: Path) -> None:
    paths = _guest_paths(tmp_path)
    observed: dict[str, object] = {}

    def fake_runner(**kwargs: object) -> bytes:
        observed.update(kwargs)
        return json.dumps(
            {
                "meta": {},
                "rules": {
                    "fixture": {"meta": {"namespace": "collection"}, "source": "private", "matches": [{"address": 7}]}
                },
            }
        ).encode("utf-8")

    result = run_guest_capa(
        source=paths["source"],
        source_sha256=_sha256(paths["source"]),
        expected_size=paths["source"].stat().st_size,
        max_input_bytes=1024,
        capa=paths["capa"],
        capa_sha256=_sha256(paths["capa"]),
        rules=paths["rules"],
        rules_sha256=_tree_sha256(paths["rules"]),
        toolchain_archive=paths["archive"],
        toolchain_archive_sha256=_sha256(paths["archive"]),
        timeout_seconds=30,
        mountinfo_path=paths["mountinfo"],
        runner=fake_runner,
    )

    assert result.aggregate == CapabilityAggregate(1, (("collection", 1),))
    assert observed["capa"] == paths["capa"]
    serialized = json.dumps(receipt_payload(result), sort_keys=True)
    assert GUEST_RECEIPT_SCHEMA in serialized
    assert str(paths["source"]) not in serialized
    assert _sha256(paths["source"]) not in serialized
    assert "address" not in serialized
    assert "fixture" not in serialized


def test_guest_run_rejects_writable_input_mount_before_runner(tmp_path: Path) -> None:
    paths = _guest_paths(tmp_path)
    writable_mountinfo = _mountinfo(tmp_path, paths["source"].parent, read_only=False)
    called = False

    def fake_runner(**_kwargs: object) -> bytes:
        nonlocal called
        called = True
        return b"{}"

    with pytest.raises(GuestCapaError, match="readonly_mount_required"):
        run_guest_capa(
            source=paths["source"],
            source_sha256=_sha256(paths["source"]),
            expected_size=paths["source"].stat().st_size,
            max_input_bytes=1024,
            capa=paths["capa"],
            capa_sha256=_sha256(paths["capa"]),
            rules=paths["rules"],
            rules_sha256=_tree_sha256(paths["rules"]),
            toolchain_archive=paths["archive"],
            toolchain_archive_sha256=_sha256(paths["archive"]),
            timeout_seconds=30,
            mountinfo_path=writable_mountinfo,
            runner=fake_runner,
        )
    assert not called


def test_toolchain_installer_rejects_traversal_before_extracting(tmp_path: Path) -> None:
    archive = tmp_path / "capa-v9.4.0-linux.zip"
    with zipfile.ZipFile(archive, "w") as payload:
        payload.writestr("../escape", b"blocked")

    with pytest.raises(GuestCapaError, match="toolchain_archive_path_invalid"):
        install_linux_capa_zip(
            archive_path=archive,
            archive_sha256=_sha256(archive),
            destination=tmp_path / "toolchain",
            mountinfo_path=_mountinfo(tmp_path, tmp_path),
        )


def test_toolchain_installer_binds_archive_binary_and_rules_without_paths(tmp_path: Path) -> None:
    archive = tmp_path / "capa-v9.4.0-linux.zip"
    executable = zipfile.ZipInfo("capa-v9.4.0-linux/capa")
    executable.external_attr = (stat.S_IFREG | 0o755) << 16
    with zipfile.ZipFile(archive, "w") as payload:
        payload.writestr(executable, b"\x7fELFfixture")
        payload.writestr("capa-v9.4.0-linux/rules/default.yml", b"rule: fixture\n")

    receipt = install_linux_capa_zip(
        archive_path=archive,
        archive_sha256=_sha256(archive),
        destination=tmp_path / "toolchain",
        mountinfo_path=_mountinfo(tmp_path, tmp_path),
    )

    assert receipt["toolchain_archive_sha256"] == _sha256(archive)
    assert set(receipt) == {
        "schema",
        "toolchain_archive_sha256",
        "capa_sha256",
        "rules_sha256",
        "raw_or_match_location_persisted",
        "source_identity_persisted",
    }
    assert str(tmp_path) not in json.dumps(receipt, sort_keys=True)


def test_rules_tree_rejects_nested_symbolic_links(tmp_path: Path) -> None:
    rules = tmp_path / "rules"
    rules.mkdir()
    (rules / "default.yml").write_text("rule: fixture\n", encoding="utf-8")
    target = tmp_path / "outside.yml"
    target.write_text("rule: outside\n", encoding="utf-8")
    link = rules / "linked.yml"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable in this test environment")

    with pytest.raises(GuestCapaError, match="rules_symlink_forbidden"):
        _tree_sha256(rules)


def test_guest_runner_contract_is_linux_only_and_remains_inactive() -> None:
    script = (PROJECT_ROOT / "scripts" / "run_loop171_capa_guest.py").read_text(encoding="utf-8")
    contract = (
        PROJECT_ROOT / "manifests/roadmap_9997/loop171_hyperv_isolation/guest_capa_runner_contract.json"
    ).read_text(encoding="utf-8")

    assert "windows_job" not in script
    assert "Linux capa guest" in script
    assert "windows_binary_or_wine_forbidden" in contract
    assert '"sample_access_allowed": false' in contract
    assert '"source_scope": "future_authorized_train_only_256_rows"' in contract
