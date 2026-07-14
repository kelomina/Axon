from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_loop28_onnx_operator_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_loop28_onnx_operator_manifest", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_build_implementation_manifest_binds_static_no_execution_chain() -> None:
    payload = builder.build_implementation_manifest(
        PROJECT_ROOT,
        generated_at_utc="2026-07-11T22:50:00Z",
    )

    assert payload["decision"] == (
        "static_preflight_implementation_complete_candidate_execution_forbidden"
    )
    assert payload["integrity"]["artifact_count"] == 10
    assert payload["integrity"]["candidate_root_absent"] is True
    assert payload["proof_summary"] == {
        "fixture_count": 4,
        "shared_tied_occurrence": 15,
        "shared_tied_node_index": 1739,
        "success_branch_reachable": False,
    }


def test_build_post_manifest_after_implementation_is_frozen() -> None:
    if not (PROJECT_ROOT / builder.DEFAULT_IMPLEMENTATION).is_file():
        pytest.skip("implementation manifest is frozen after the initial focused test pass")

    payload = builder.build_post_manifest(
        PROJECT_ROOT,
        generated_at_utc="2026-07-11T22:55:00Z",
    )
    assert payload["decision"] == (
        "post_operator_preflight_exact_tie_fallback_pytorch_native_no_execution"
    )
    assert payload["outcome"]["candidate_graph_count"] == 0
    assert payload["outcome"]["strict_margin_success_branch_reachable"] is False


def test_assert_execution_artifacts_absent_rejects_candidate_root(tmp_path: Path) -> None:
    (tmp_path / builder.CANDIDATE_ROOT).mkdir(parents=True)

    with pytest.raises(builder.OperatorManifestError, match="Forbidden candidate root"):
        builder.assert_execution_artifacts_absent(tmp_path)


def test_strict_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    artifact = tmp_path / "duplicate.json"
    artifact.write_text('{"schema": "a", "schema": "b"}', encoding="utf-8")

    with pytest.raises(builder.OperatorManifestError, match="Duplicate JSON key"):
        builder.load_json_strict(artifact)
