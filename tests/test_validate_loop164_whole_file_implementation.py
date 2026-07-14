from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
TESTS_DIR = Path(__file__).resolve().parent
for directory in (SCRIPTS_DIR, TESTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from loop164_whole_file_contract_fixture import (  # noqa: E402
    create_whole_file_implementation_fixture,
)
from validate_loop164_whole_file_implementation import (  # noqa: E402
    REQUIRED_SOURCE_ROLE_PATHS,
    calculate_source_closure_sha256,
    sha256_file,
    validate_implementation_manifest_path,
)


def test_valid_synthetic_static_contract_passes_without_protected_input(tmp_path: Path):
    manifest_path = tmp_path / "reports/roadmap_9997/loop164/whole_file_expert_implementation_manifest.json"
    create_whole_file_implementation_fixture(tmp_path, manifest_path)

    result = validate_implementation_manifest_path(manifest_path, root=tmp_path)

    assert result.ready is True
    assert result.source_closure_sha256 is not None
    assert result.memory_contract_sha256 is not None


def test_contract_rejects_closure_drift_and_unbounded_reader(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    create_whole_file_implementation_fixture(tmp_path, manifest_path)
    loader_path = tmp_path / REQUIRED_SOURCE_ROLE_PATHS["input_loader"]
    loader_path.write_text("def read_chunk(handle):\n    return handle.read(-1)\n", encoding="utf-8")

    result = validate_implementation_manifest_path(manifest_path, root=tmp_path)

    assert result.ready is False
    assert "implementation_source_closure_sha256_mismatch" in result.blockers
    assert "implementation_source_unbounded_io_detected" in result.blockers


def test_contract_rejects_identity_and_missingness_policy_drift(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    create_whole_file_implementation_fixture(tmp_path, manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["input_contract"]["identity_feature_count"] = 1
    payload["missingness_contract"]["score_fallback_policy"] = "reuse_loop151_score"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_implementation_manifest_path(manifest_path, root=tmp_path)

    assert result.ready is False
    assert "implementation_input_contract_invalid" in result.blockers
    assert "implementation_missingness_contract_invalid" in result.blockers


def test_contract_rejects_stride_that_would_leave_unscanned_bytes(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    payload = create_whole_file_implementation_fixture(tmp_path, manifest_path)
    payload["memory_contract"]["output_stride_bytes"] = 1024
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_implementation_manifest_path(manifest_path, root=tmp_path)

    assert result.ready is False
    assert "implementation_memory_contract_bounds_invalid" in result.blockers


def test_contract_rejects_false_overlap_declaration(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    payload = create_whole_file_implementation_fixture(tmp_path, manifest_path)
    payload["memory_contract"]["overlap_bytes"] = 511
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_implementation_manifest_path(manifest_path, root=tmp_path)

    assert result.ready is False
    assert "implementation_memory_contract_bounds_invalid" in result.blockers


def test_contract_rejects_missing_required_exactness_oracle(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    payload = create_whole_file_implementation_fixture(tmp_path, manifest_path)
    oracle_path = tmp_path / REQUIRED_SOURCE_ROLE_PATHS["dense_equivalence_test"]
    oracle_path.write_text("def test_dense_reference_equivalence():\n    assert True\n", encoding="utf-8")
    for entry in payload["source_closure"]["files"]:
        if entry["role"] == "dense_equivalence_test":
            entry["sha256"] = sha256_file(oracle_path)
            entry["bytes"] = oracle_path.stat().st_size
    payload["source_closure"]["closure_sha256"] = calculate_source_closure_sha256(
        payload["source_closure"]["files"]
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_implementation_manifest_path(manifest_path, root=tmp_path)

    assert result.ready is False
    assert "implementation_source_required_oracle_test_missing" in result.blockers


def test_contract_rejects_trivial_named_exactness_oracle(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    payload = create_whole_file_implementation_fixture(tmp_path, manifest_path)
    oracle_path = tmp_path / REQUIRED_SOURCE_ROLE_PATHS["dense_equivalence_test"]
    oracle_path.write_text(
        "\n".join(
            f"def test_{name}():\n    assert True\n"
            for name in payload["static_safety_audit"]["required_test_classes"]
        ),
        encoding="utf-8",
    )
    for entry in payload["source_closure"]["files"]:
        if entry["role"] == "dense_equivalence_test":
            entry["sha256"] = sha256_file(oracle_path)
            entry["bytes"] = oracle_path.stat().st_size
    payload["source_closure"]["closure_sha256"] = calculate_source_closure_sha256(
        payload["source_closure"]["files"]
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_implementation_manifest_path(manifest_path, root=tmp_path)

    assert result.ready is False
    assert "implementation_source_oracle_test_body_invalid" in result.blockers
