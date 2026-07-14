from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_loop28_parity_post_diagnostic_manifest as post  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(root: Path) -> tuple[post.ArtifactSpec, ...]:
    specs = []
    for index in range(8):
        path = Path("evidence") / f"artifact_{index}.bin"
        absolute = root / path
        absolute.parent.mkdir(parents=True, exist_ok=True)
        absolute.write_bytes(f"artifact-{index}\n".encode())
        expected = _sha256(absolute) if index < 6 else None
        specs.append(post.ArtifactSpec(f"artifact_{index}", "test", path, expected))
    return tuple(specs)


def test_build_write_and_verify_fixed_closure(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    manifest = post.build_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:00:00Z",
        artifacts=inventory,
    )
    assert manifest["integrity"]["artifact_count"] == 8
    assert manifest["integrity"]["verified_predeclared_sha256_count"] == 6
    assert manifest["integrity"]["blockers"] == []

    output = tmp_path / "manifest.json"
    post.write_manifest_exclusive(output, manifest)
    verified = post.verify_manifest(tmp_path, Path("manifest.json"), artifacts=inventory)
    assert verified == manifest
    with pytest.raises(post.ManifestError, match="already exists"):
        post.write_manifest_exclusive(output, manifest)


def test_predeclared_hash_drift_blocks_closure(tmp_path: Path) -> None:
    inventory = list(_inventory(tmp_path))
    inventory[0] = post.ArtifactSpec(
        inventory[0].name,
        inventory[0].role,
        inventory[0].path,
        "0" * 64,
    )
    manifest = post.build_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:00:00+00:00",
        artifacts=tuple(inventory),
    )
    assert manifest["decision"] == "diagnostic_closure_blocked"
    assert manifest["integrity"]["blockers"][0]["reason"] == "predeclared_sha256_mismatch"


def test_symlink_artifact_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    link = tmp_path / "link.bin"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    inventory = (post.ArtifactSpec("link", "test", Path("link.bin"), None),)
    manifest = post.build_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:00:00Z",
        artifacts=inventory,
    )
    assert manifest["integrity"]["artifact_count"] == 0
    assert "regular non-symlink" in manifest["integrity"]["blockers"][0]["reason"]
