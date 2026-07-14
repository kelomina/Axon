from __future__ import annotations

import json

import pytest

from src.loop167_phase_b.contracts import (
    PhaseBContractError,
    canonical_argv_sha256,
    canonical_json_bytes,
    require_canonical_json,
    resolve_project_file,
    sha256_file,
    verify_file_binding,
)


def test_canonical_artifact_binding_rejects_drift_and_path_escape(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    artifact = root / "artifact.json"
    payload = {"schema": "synthetic", "value": 1}
    artifact.write_bytes(canonical_json_bytes(payload))
    binding = {"path": "artifact.json", "sha256": sha256_file(artifact)}

    path, digest = verify_file_binding(root, binding, label="artifact")
    assert path == artifact
    assert digest == binding["sha256"]
    assert require_canonical_json(artifact) == payload

    artifact.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(PhaseBContractError, match="canonical"):
        require_canonical_json(artifact)
    with pytest.raises(PhaseBContractError, match="escapes"):
        resolve_project_file(root, "../outside.json")


def test_canonical_argv_hash_is_stable_and_rejects_invalid_values() -> None:
    argv = ("vnev/Scripts/python.exe", "-I", "scripts/run_loop167_phase_b_controller.py", "--preflight")
    assert canonical_argv_sha256(argv) == canonical_argv_sha256(list(argv))
    with pytest.raises(PhaseBContractError, match="nonempty"):
        canonical_argv_sha256(("python", ""))
