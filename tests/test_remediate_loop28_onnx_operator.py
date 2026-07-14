from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "remediate_loop28_onnx_operator.py"
SPEC = importlib.util.spec_from_file_location("remediate_loop28_onnx_operator", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
operator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(operator)


def test_build_preflight_proves_shared_exact_tie_blocker() -> None:
    payload = operator.build_preflight(
        PROJECT_ROOT,
        generated_at_utc="2026-07-11T22:45:00Z",
    )

    assert payload["decision"] == (
        "operator_preflight_exact_tie_fallback_pytorch_native_no_execution"
    )
    assert payload["proof"]["fixture_count"] == 4
    assert payload["proof"]["shared_tied_occurrence"] == 15
    assert payload["proof"]["formal_proof"]["success_branch_reachable"] is False
    assert payload["execution_audit"] == {
        "checkpoint_load_count": 0,
        "onnx_graph_load_count": 0,
        "native_probe_subprocess_count": 0,
        "candidate_graph_count": 0,
        "lease_count": 0,
        "raw_split_cache_heldout_access_count": 0,
        "quality_metric_count": 0,
        "f1_computation_count": 0,
    }
    for fixture in payload["proof"]["fixture_proofs"]:
        route = fixture["shared_blocking_route"]
        assert route["minimum_pytorch_support_margin"] == 0.0
        assert route["exact_support_tie_count"] >= 1


def test_proof_rejects_missing_shared_tie() -> None:
    evidence = operator.load_json_strict(PROJECT_ROOT / operator.DEFAULT_LOCALIZATION_EVIDENCE)
    proposal = operator.load_json_strict(PROJECT_ROOT / operator.DEFAULT_PROPOSAL)
    mutated = copy.deepcopy(evidence)
    row = mutated["fixtures"][0]["profiles"]["routing"]["stability"]["rows"][15]
    row["exact_support_tie_count"] = 0
    row["minimum_pytorch_support_margin"] = 1.0e-5

    with pytest.raises(operator.OperatorPreflightError, match="Expected exact support tie"):
        operator.prove_frozen_tie_blocker(mutated, proposal)


def test_strict_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    artifact = tmp_path / "duplicate.json"
    artifact.write_text('{"schema": "a", "schema": "b"}', encoding="utf-8")

    with pytest.raises(operator.OperatorPreflightError, match="Duplicate JSON key"):
        operator.load_json_strict(artifact)


def test_resolver_rejects_parent_escape() -> None:
    with pytest.raises(operator.OperatorPreflightError, match="project-relative"):
        operator._resolve_within(PROJECT_ROOT, Path("../outside.json"))
