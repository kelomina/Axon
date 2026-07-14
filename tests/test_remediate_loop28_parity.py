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

import remediate_loop28_parity as remediation  # noqa: E402


def _component(name: str, count: int, *, matched: bool = True, mismatches=None) -> dict:
    mismatch_indices = list(mismatches or [])
    return {
        "name": name,
        "element_count": count,
        "whole_match": matched,
        "mismatch_count": 0 if matched else len(mismatch_indices),
        "mismatch_indices": mismatch_indices,
    }


def _diagnostic_result(*, stage2_mismatches=None, delta: float = 1.0e-9) -> dict:
    stage2_mismatch_indices = list(stage2_mismatches or [])
    return {
        "component_results": [
            _component("byte_seq", 8192),
            _component("pe_features", 256),
            _component("stat_features", 49),
            _component("base_logits", 2, matched=False),
            _component("base_probabilities", 2, matched=False),
            _component(
                "stage2_features",
                1520,
                matched=not stage2_mismatch_indices,
                mismatches=stage2_mismatch_indices,
            ),
        ],
        "predictions": {
            "decision_match": True,
            "base_decision_match": True,
            "absolute_probability_deltas": {
                "prob_benign": delta,
                "prob_malicious": delta,
                "base_prob_benign": delta,
                "base_prob_malicious": delta,
                "stage2_prob_malicious": delta,
            },
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_success_gate_allows_only_probability_transform_prefix_drift():
    result = remediation.evaluate_success_gate(
        _diagnostic_result(stage2_mismatches=[0, 1, 2, 3, 4, 5])
    )

    assert result["passed"] is True
    assert result["checks"]["stage2_features_6_through_1519_exact"] is True
    assert result["decision"] == "frozen_train_remediation_gate_passed"


@pytest.mark.parametrize(
    ("stage2_mismatches", "delta"),
    [([6], 1.0e-9), ([], 1.000001e-6)],
)
def test_success_gate_rejects_suffix_or_probability_drift(stage2_mismatches, delta):
    result = remediation.evaluate_success_gate(
        _diagnostic_result(stage2_mismatches=stage2_mismatches, delta=delta)
    )

    assert result["passed"] is False
    assert result["decision"] == "frozen_train_remediation_gate_failed"


def test_success_gate_rejects_feature_or_decision_mismatch():
    payload = _diagnostic_result()
    payload["component_results"][1]["whole_match"] = False
    payload["component_results"][1]["mismatch_count"] = 1
    payload["component_results"][1]["mismatch_indices"] = [117]
    payload["predictions"]["base_decision_match"] = False

    result = remediation.evaluate_success_gate(payload)

    assert result["checks"]["pe_features"] is False
    assert result["checks"]["base_decision_match"] is False
    assert result["passed"] is False


def test_authorizations_bind_proposal_parent_manifest_and_output(monkeypatch, tmp_path: Path):
    proposal_path = Path("proposal.json")
    authorization_path = Path("authorization.json")
    run_path = Path("run.json")
    manifest_path = Path("implementation.json")
    output_path = Path("receipt.json")
    lease_path = Path("lease.json")
    parent_paths = {name: Path(f"{name}.json") for name in remediation.PARENT_EVIDENCE_PATHS}
    for path in [proposal_path, manifest_path, *parent_paths.values()]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.as_posix().encode("utf-8"))
    parent_hashes = {
        name: remediation.replay.file_sha256(tmp_path / path) for name, path in parent_paths.items()
    }
    monkeypatch.setattr(remediation, "DEFAULT_PROPOSAL", proposal_path)
    monkeypatch.setattr(remediation, "DEFAULT_AUTHORIZATION", authorization_path)
    monkeypatch.setattr(remediation, "DEFAULT_RUN_AUTHORIZATION", run_path)
    monkeypatch.setattr(remediation, "DEFAULT_IMPLEMENTATION_MANIFEST", manifest_path)
    monkeypatch.setattr(remediation, "DEFAULT_OUTPUT", output_path)
    monkeypatch.setattr(remediation, "DEFAULT_ATTEMPT_LEASE", lease_path)
    monkeypatch.setattr(remediation, "PARENT_EVIDENCE_PATHS", parent_paths)
    monkeypatch.setattr(remediation, "PARENT_EVIDENCE_SHA256", parent_hashes)

    proposal_sha256 = remediation.replay.file_sha256(tmp_path / proposal_path)
    prereg_payload = {
        "schema": remediation.AUTHORIZATION_SCHEMA,
        "loop_id": remediation.LOOP_ID,
        "authorization_level": "A1_scoped_change",
        "proposal": {"path": proposal_path.as_posix(), "sha256": proposal_sha256},
        "parent_evidence": parent_hashes,
        "frozen_sample": remediation.EXPECTED_SAMPLE,
        "allowed_splits": [remediation.FIXED_SPLIT],
        "allowed_logical_raw_root": remediation.FIXED_LOGICAL_RAW_ROOT,
        "allowed_resolved_raw_roots": remediation.FIXED_RESOLVED_RAW_ROOTS,
        "frozen_tolerance": remediation.FIXED_TOLERANCE,
        "budget": remediation.EXPECTED_BUDGET,
        "timeout_enforcement": remediation.EXPECTED_TIMEOUT_ENFORCEMENT,
        "execution_requires_separate_run_authorization": True,
        "success_gate": "feature_components_exact_and_probability_deltas_at_most_1e-6",
        "decision": (
            "allow_scoped_implementation_and_enumerated_synthetic_validation_"
            "train_run_blocked_pending_hash_manifest"
        ),
    }
    _write_json(tmp_path / authorization_path, prereg_payload)
    preregistration = remediation.verify_remediation_authorization(tmp_path)
    manifest_sha256 = remediation.replay.file_sha256(tmp_path / manifest_path)
    monkeypatch.setattr(
        remediation.implementation_manifest,
        "verify_manifest",
        lambda _root, _path: {
            "integrity": {
                "blockers": [],
                "artifact_count": 12,
                "required_artifact_count": 12,
                "present_required_artifact_count": 12,
            }
        },
    )
    run_payload = {
        "schema": remediation.RUN_AUTHORIZATION_SCHEMA,
        "loop_id": remediation.LOOP_ID,
        "issued_at_utc": "2026-07-12T00:00:00Z",
        "prereg_authorization_sha256": preregistration["authorization_sha256"],
        "proposal_sha256": proposal_sha256,
        "parent_evidence": parent_hashes,
        "implementation_manifest": {
            "path": manifest_path.as_posix(),
            "sha256": manifest_sha256,
        },
        "frozen_sample": remediation.EXPECTED_SAMPLE,
        "budget": remediation.EXPECTED_BUDGET,
        "timeout_enforcement": remediation.EXPECTED_TIMEOUT_ENFORCEMENT,
        "frozen_tolerance": remediation.FIXED_TOLERANCE,
        "attempt_id": remediation.FIXED_ATTEMPT_ID,
        "attempt_lease_path": lease_path.as_posix(),
        "generation": "final",
        "output_path": output_path.as_posix(),
        "claim_scope": {
            "train_only": True,
            "raw_identity_count": 1,
            "heldout_access_allowed": False,
            "training_or_fitting_allowed": False,
            "quality_claim_allowed": False,
            "population_parity_claim_allowed": False,
            "certification_claim_allowed": False,
        },
        "decision": "allow_bounded_loop28_parity_remediation_run",
    }
    _write_json(tmp_path / run_path, run_payload)

    verified = remediation.verify_run_authorization(
        tmp_path,
        run_path,
        output_path,
        preregistration,
    )

    assert verified["status"] == "bounded_run_authorized"
    run_payload["proposal_sha256"] = "0" * 64
    _write_json(tmp_path / run_path, run_payload)
    with pytest.raises(remediation.RemediationContractError, match="proposal_sha256"):
        remediation.verify_run_authorization(
            tmp_path,
            run_path,
            output_path,
            preregistration,
        )


def test_authorization_rejects_duplicate_json_keys(tmp_path: Path):
    path = tmp_path / "authorization.json"
    path.write_text(
        '{"schema":"axon_loop28_parity_remediation_authorization_v1","budget":{},"budget":{}}',
        encoding="utf-8",
    )

    with pytest.raises(remediation.RemediationContractError, match="Duplicate JSON key"):
        remediation._read_json_object(path, remediation.AUTHORIZATION_SCHEMA)


def test_attempt_lease_and_receipt_are_exclusive(monkeypatch, tmp_path: Path):
    lease_path = Path("attempt.json")
    monkeypatch.setattr(remediation, "DEFAULT_ATTEMPT_LEASE", lease_path)
    consumed = remediation._consume_attempt_lease(
        tmp_path,
        {"authorization_sha256": "a" * 64},
    )

    assert remediation._verify_attempt_lease(tmp_path, consumed)["sha256"] == consumed["sha256"]
    with pytest.raises(remediation.RemediationContractError, match="already consumed"):
        remediation._consume_attempt_lease(
            tmp_path,
            {"authorization_sha256": "a" * 64},
        )

    receipt_path = tmp_path / "receipt.json"
    remediation._write_receipt_exclusive(receipt_path, {"decision": "test"})
    with pytest.raises(remediation.RemediationContractError, match="already exists"):
        remediation._write_receipt_exclusive(receipt_path, {"decision": "test"})


def test_remediation_rejects_before_manifest_model_or_raw_access(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        remediation,
        "verify_remediation_authorization",
        lambda _root: (_ for _ in ()).throw(
            remediation.RemediationContractError("no remediation authorization")
        ),
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("authorization failure must precede manifest, model, and raw access")

    monkeypatch.setattr(remediation.implementation_manifest, "verify_manifest", forbidden)
    monkeypatch.setattr(remediation.replay, "guard_pickle_before_load", forbidden)
    monkeypatch.setattr(remediation.replay, "snapshot_verified_sample", forbidden)

    with pytest.raises(remediation.RemediationContractError, match="no remediation"):
        remediation.run_remediation(
            tmp_path,
            run_authorization_path=remediation.DEFAULT_RUN_AUTHORIZATION,
            output_path=remediation.DEFAULT_OUTPUT,
        )


def test_budget_audit_fails_closed_on_missing_native_output_accounting():
    diagnostic_result = {
        "execution_counts": {"python": 1, "native": 1, "crossfeed": 1},
        "execution_durations_seconds": {"native": [0.1], "crossfeed": [0.1]},
    }
    audit = remediation._build_budget_audit(
        snapshot_duration=0.1,
        python_duration=0.2,
        diagnostic_result=diagnostic_result,
        total_duration=0.5,
        native_output_sizes=[],
        require_native_output_accounting=True,
    )

    assert audit["output"]["within_budget"] is False
    assert audit["within_budget"] is False
    with pytest.raises(remediation.RemediationContractError, match="exceeded"):
        remediation._enforce_budget(audit)


def test_lease_sha_is_the_exact_persisted_payload(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(remediation, "DEFAULT_ATTEMPT_LEASE", Path("attempt.json"))
    consumed = remediation._consume_attempt_lease(
        tmp_path,
        {"authorization_sha256": "b" * 64},
    )

    assert (
        hashlib.sha256((tmp_path / "attempt.json").read_bytes()).hexdigest() == consumed["sha256"]
    )
