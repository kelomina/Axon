from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_loop28_parity_diagnostic_manifest as implementation_manifest  # noqa: E402


def _materialize_fixed_inventory(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for index, spec in enumerate(implementation_manifest.DEFAULT_ARTIFACTS):
        target = root / spec.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"opaque-artifact-{index}:{spec.name}".encode())
    parent_hashes = {
        name: implementation_manifest.file_sha256(root / path)
        for name, path in implementation_manifest.PARENT_EVIDENCE_PATHS.items()
    }
    monkeypatch.setattr(implementation_manifest, "PARENT_EVIDENCE_SHA256", parent_hashes)


def test_fixed_inventory_is_unique_complete_and_excludes_cyclic_outputs():
    specs = implementation_manifest.DEFAULT_ARTIFACTS
    paths = [spec.path.as_posix() for spec in specs]
    names = [spec.name for spec in specs]

    assert len(paths) == len(set(path.casefold() for path in paths))
    assert len(names) == len(set(names))
    assert set(implementation_manifest.PREREGISTRATION_PATHS) == {
        spec.path for spec in specs if spec.role == "preregistration"
    }
    assert set(implementation_manifest.PARENT_EVIDENCE_PATHS) <= set(names)
    assert not (set(paths) & implementation_manifest.DENIED_ARTIFACT_PATHS)
    assert (
        "reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.provisional.json"
        in implementation_manifest.DENIED_ARTIFACT_PATHS
    )
    assert (
        "reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.final.json"
        in implementation_manifest.DENIED_ARTIFACT_PATHS
    )
    assert "scripts/diagnose_loop28_parity.py" in paths
    assert "tests/test_diagnose_loop28_parity.py" in paths
    assert "tools/axon_onnx_dll/src/axon_onnx_predict.cpp" in paths
    assert "tools/axon_onnx_dll/build/bin/Release/axon_onnx_predict.dll" in paths
    assert "reports/random_20w_split/loop127_full_duplicate_corrected_split.csv" in paths
    assert "manifests/roadmap_9997/p0_raw_replay/authorization.json" in paths
    assert "manifests/roadmap_9997/p0_raw_replay/pickle_sha256_allowlist.json" in paths
    assert "manifests/roadmap_9997/p0_raw_replay/loop28_stage2.metadata.json" in paths


def test_build_and_verify_use_opaque_hashes_only(monkeypatch, tmp_path: Path):
    _materialize_fixed_inventory(tmp_path, monkeypatch)
    split_path = tmp_path / next(
        spec.path
        for spec in implementation_manifest.DEFAULT_ARTIFACTS
        if spec.name == "frozen_split"
    )
    split_path.write_bytes(b"not,a,parseable,split\n\xff\x00")

    manifest = implementation_manifest.build_implementation_manifest(project_root=tmp_path)
    output_path = tmp_path / implementation_manifest.DEFAULT_OUTPUT
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    summary = implementation_manifest.verify_implementation_manifest(
        tmp_path,
        implementation_manifest.DEFAULT_OUTPUT,
    )

    assert persisted == manifest
    assert manifest["schema"] == implementation_manifest.MANIFEST_SCHEMA
    assert manifest["loop_id"] == implementation_manifest.LOOP_ID
    assert manifest["contract"]["prediction_or_metric_payloads_read"] is False
    assert manifest["contract"]["split_rows_read"] is False
    assert manifest["integrity"]["blockers"] == []
    assert manifest["integrity"]["artifact_count"] == len(implementation_manifest.DEFAULT_ARTIFACTS)
    assert manifest["decision"] == implementation_manifest.READY_DECISION
    assert summary["required_artifacts_verified"] == len(implementation_manifest.DEFAULT_ARTIFACTS)
    assert summary["decision"] == implementation_manifest.VERIFIED_DECISION
    assert summary["implementation_manifest_sha256"] == implementation_manifest.file_sha256(
        output_path
    )


def test_verify_recomputes_every_required_hash(monkeypatch, tmp_path: Path):
    _materialize_fixed_inventory(tmp_path, monkeypatch)
    implementation_manifest.build_implementation_manifest(project_root=tmp_path)
    changed = tmp_path / next(
        spec.path
        for spec in implementation_manifest.DEFAULT_ARTIFACTS
        if spec.name == "native_runtime_source"
    )
    changed.write_bytes(b"changed-after-freeze")

    with pytest.raises(
        implementation_manifest.ManifestContractError,
        match="(size|SHA-256) mismatch: native_runtime_source",
    ):
        implementation_manifest.verify_implementation_manifest(
            tmp_path,
            implementation_manifest.DEFAULT_OUTPUT,
        )


def test_missing_artifact_writes_blocked_manifest(monkeypatch, tmp_path: Path):
    _materialize_fixed_inventory(tmp_path, monkeypatch)
    missing = tmp_path / next(
        spec.path
        for spec in implementation_manifest.DEFAULT_ARTIFACTS
        if spec.name == "native_selftest"
    )
    missing.unlink()

    manifest = implementation_manifest.build_implementation_manifest(project_root=tmp_path)

    assert manifest["decision"] == implementation_manifest.BLOCKED_DECISION
    assert (
        manifest["integrity"]["present_required_artifact_count"]
        == len(implementation_manifest.DEFAULT_ARTIFACTS) - 1
    )
    assert any(
        "missing_required_artifact:native_selftest:" in blocker
        for blocker in manifest["integrity"]["blockers"]
    )
    with pytest.raises(implementation_manifest.ManifestContractError, match="not ready"):
        implementation_manifest.verify_implementation_manifest(
            tmp_path,
            implementation_manifest.DEFAULT_OUTPUT,
        )


def test_build_refuses_overwrite_without_explicit_replace(monkeypatch, tmp_path: Path):
    _materialize_fixed_inventory(tmp_path, monkeypatch)
    implementation_manifest.build_implementation_manifest(project_root=tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        implementation_manifest.build_implementation_manifest(project_root=tmp_path)

    replaced = implementation_manifest.build_implementation_manifest(
        project_root=tmp_path,
        replace=True,
    )
    assert replaced["decision"] == implementation_manifest.READY_DECISION


def test_duplicate_and_denied_paths_are_rejected():
    duplicate = (
        implementation_manifest.ArtifactSpec("one", "test", Path("one.bin")),
        implementation_manifest.ArtifactSpec("two", "test", Path("one.bin")),
    )
    with pytest.raises(
        implementation_manifest.ManifestContractError, match="Duplicate artifact path"
    ):
        implementation_manifest._validate_artifact_specs(duplicate)

    denied = (
        implementation_manifest.ArtifactSpec(
            "receipt",
            "test",
            Path("reports/roadmap_9997/p0_loop28_parity_diagnostic/diagnostic_receipt.json"),
        ),
    )
    with pytest.raises(implementation_manifest.ManifestContractError, match="Denied artifact path"):
        implementation_manifest._validate_artifact_specs(denied)


def test_verify_rejects_manifest_record_reinjection(monkeypatch, tmp_path: Path):
    _materialize_fixed_inventory(tmp_path, monkeypatch)
    implementation_manifest.build_implementation_manifest(project_root=tmp_path)
    output_path = tmp_path / implementation_manifest.DEFAULT_OUTPUT
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["artifacts"].append(dict(payload["artifacts"][0]))
    output_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        implementation_manifest.ManifestContractError, match="Duplicate artifact name"
    ):
        implementation_manifest.verify_implementation_manifest(
            tmp_path,
            implementation_manifest.DEFAULT_OUTPUT,
        )


def test_cli_has_no_output_or_supplemental_artifact_override():
    parser = implementation_manifest.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--artifact", "extra=/tmp/extra"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-json", "elsewhere.json"])


def test_verify_rejects_non_frozen_manifest_path(monkeypatch, tmp_path: Path):
    _materialize_fixed_inventory(tmp_path, monkeypatch)
    implementation_manifest.build_implementation_manifest(project_root=tmp_path)

    with pytest.raises(implementation_manifest.ManifestContractError, match="frozen output path"):
        implementation_manifest.verify_implementation_manifest(
            tmp_path,
            Path("other.json"),
        )
