from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_loop28_parity_remediation_manifest as remediation  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(root: Path) -> tuple[remediation.ArtifactSpec, ...]:
    specs = []
    for index in range(5):
        path = Path("artifacts") / f"artifact_{index}.bin"
        absolute = root / path
        absolute.parent.mkdir(parents=True, exist_ok=True)
        absolute.write_bytes(f"remediation-{index}\n".encode())
        expected = _sha256(absolute) if index < 2 else None
        specs.append(remediation.ArtifactSpec(f"artifact_{index}", "test", path, expected))
    return tuple(specs)


def test_build_and_verify_remediation_closure(tmp_path: Path) -> None:
    artifacts = _inventory(tmp_path)
    manifest = remediation.build_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:00:00Z",
        artifacts=artifacts,
    )
    assert manifest["integrity"]["artifact_count"] == 5
    assert manifest["integrity"]["verified_predeclared_sha256_count"] == 2
    assert manifest["integrity"]["blockers"] == []
    assert manifest["contract"]["declared_runtime"]["verification_status"] == (
        "declared_not_verified_by_manifest_builder"
    )
    assert manifest["claim_scope"]["synthetic_parity_reverified_by_manifest_builder"] is False

    output = tmp_path / "implementation_manifest.json"
    remediation.write_manifest_exclusive(output, manifest)
    assert (
        remediation.verify_manifest(
            tmp_path,
            Path("implementation_manifest.json"),
            artifacts=artifacts,
        )
        == manifest
    )


def test_expected_parent_drift_blocks_manifest(tmp_path: Path) -> None:
    artifacts = list(_inventory(tmp_path))
    artifacts[0] = remediation.ArtifactSpec(
        artifacts[0].name,
        artifacts[0].role,
        artifacts[0].path,
        "f" * 64,
    )
    manifest = remediation.build_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:00:00Z",
        artifacts=tuple(artifacts),
    )
    assert manifest["decision"] == "implementation_manifest_blocked"
    assert manifest["integrity"]["blockers"][0]["reason"] == "predeclared_sha256_mismatch"


def test_manifest_output_is_exclusive(tmp_path: Path) -> None:
    manifest = remediation.build_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:00:00Z",
        artifacts=_inventory(tmp_path),
    )
    output = tmp_path / "implementation_manifest.json"
    remediation.write_manifest_exclusive(output, manifest)
    with pytest.raises(remediation.ManifestError, match="already exists"):
        remediation.write_manifest_exclusive(output, manifest)


def test_default_inventory_reuses_full_diagnostic_python_closure() -> None:
    expected_paths = {
        spec.path.as_posix()
        for spec in remediation.diagnostic_closure.DEFAULT_ARTIFACTS
        if spec.role in remediation.REUSED_DIAGNOSTIC_PYTHON_ROLES
    }
    actual_paths = {spec.path.as_posix() for spec in remediation.ARTIFACTS}
    assert expected_paths <= actual_paths
    assert "scripts/remediate_loop28_parity.py" in actual_paths
    assert "tests/test_remediate_loop28_parity.py" in actual_paths
    assert "scripts/build_loop28_parity_post_remediation_manifest.py" in actual_paths
    assert "tests/test_build_loop28_parity_post_remediation_manifest.py" in actual_paths
    assert (
        "manifests/roadmap_9997/p0_loop28_parity_remediation/post_remediation_manifest.json"
        not in actual_paths
    )


def test_governance_and_synthetic_evidence_hashes_are_predeclared() -> None:
    by_name = {spec.name: spec for spec in remediation.ARTIFACTS}
    for name in (
        "proposal",
        "authorization",
        "preflight",
        "synthetic_discovery",
    ):
        assert by_name[name].expected_sha256 is not None


def test_cli_output_is_fixed_and_project_confined(tmp_path: Path) -> None:
    expected = (tmp_path / remediation.DEFAULT_OUTPUT).resolve()
    assert remediation.resolve_fixed_output(tmp_path, remediation.DEFAULT_OUTPUT) == expected

    with pytest.raises(remediation.ManifestError, match="not fixed"):
        remediation.resolve_fixed_output(tmp_path, Path("other.json"))
    with pytest.raises(remediation.ManifestError, match="project-relative"):
        remediation.resolve_fixed_output(tmp_path, Path("../escape.json"))
    with pytest.raises(remediation.ManifestError, match="project-relative"):
        remediation.resolve_fixed_output(tmp_path, tmp_path.parent / "escape.json")


def test_verify_rejects_manifest_path_escape(tmp_path: Path) -> None:
    with pytest.raises(remediation.ManifestError, match="project-relative"):
        remediation.verify_manifest(
            tmp_path,
            Path("../implementation_manifest.json"),
            artifacts=_inventory(tmp_path),
        )


def test_duplicate_inventory_is_rejected(tmp_path: Path) -> None:
    artifacts = _inventory(tmp_path)
    with pytest.raises(remediation.ManifestError, match="Duplicate"):
        remediation.build_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:00:00Z",
            artifacts=(*artifacts, artifacts[0]),
        )


def test_synthetic_pre_run_block_forbids_implementation_manifest(tmp_path: Path) -> None:
    blocked = tmp_path / remediation.BLOCKED_EVIDENCE
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text(
        json.dumps(
            {
                "schema": remediation.BLOCKED_EVIDENCE_SCHEMA,
                "loop_id": remediation.LOOP_ID,
                "decision": remediation.BLOCKED_DECISION,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(remediation.ManifestError, match="forbids an implementation manifest"):
        remediation.build_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:00:00Z",
            artifacts=_inventory(tmp_path),
        )
    assert not (tmp_path / remediation.DEFAULT_OUTPUT).exists()
