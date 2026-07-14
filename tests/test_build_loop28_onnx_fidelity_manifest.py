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

import build_loop28_onnx_fidelity_manifest as fidelity  # noqa: E402


def _write_bytes(root: Path, path: Path, payload: bytes) -> str:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _write_json(root: Path, path: Path, payload: dict) -> str:
    return _write_bytes(
        root,
        path,
        (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def _file_row(root: Path, name: str, path: Path) -> dict:
    target = root / path
    return {
        "name": name,
        "role": "frozen_parent_artifact",
        "path": path.as_posix(),
        "size_bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def _claim_boundary() -> dict:
    return {
        "synthetic_cross_runtime_localization_allowed": True,
        "population_parity_claim_allowed": False,
        "quality_claim_allowed": False,
        "native_loop28_ready_claim_allowed": False,
        "native_loop151_ready_claim_allowed": False,
        "raw_rerun_allowed": False,
        "certification_claim_allowed": False,
    }


def _fixture_matrix() -> list[dict]:
    return [
        {
            "name": "pe32_numeric_resource_tls_callbacks",
            "pe_plus": False,
            "named_resource": False,
            "tls_callbacks": True,
            "positive_control": "must_fail_base_probability_gate",
        },
        {
            "name": "pe32_named_resource_tls_callbacks",
            "pe_plus": False,
            "named_resource": True,
            "tls_callbacks": True,
            "positive_control": "must_fail_base_probability_gate",
        },
        {
            "name": "pe32_numeric_resource_zero_tls_callbacks",
            "pe_plus": False,
            "named_resource": False,
            "tls_callbacks": False,
            "positive_control": "must_fail_base_probability_gate",
        },
        {
            "name": "pe32_plus_named_resource_zero_tls_callbacks",
            "pe_plus": True,
            "named_resource": True,
            "tls_callbacks": False,
            "positive_control": "must_pass_base_probability_gate",
        },
    ]


def _setup_implementation(root: Path) -> dict[str, object]:
    for path, payload in (
        (fidelity.DIAGNOSTIC_SOURCE, b"diagnostic source\n"),
        (fidelity.DIAGNOSTIC_TEST, b"diagnostic tests\n"),
        (fidelity.PROBE_SOURCE, b"probe source\n"),
        (fidelity.PROBE_CMAKE, b"probe cmake\n"),
        (fidelity.PROBE_BINARY, b"probe binary\n"),
        (fidelity.BUILDER_SOURCE, b"manifest builder\n"),
        (fidelity.BUILDER_TEST, b"manifest builder tests\n"),
    ):
        _write_bytes(root, path, payload)

    parent_artifact = Path("parent/frozen_status.md")
    _write_bytes(root, parent_artifact, b"frozen parent status\n")
    parent_row = _file_row(root, "frozen_status", parent_artifact)
    parent = {
        "schema": fidelity.PARENT_SCHEMA,
        "loop_id": "p0_loop28_parity_remediation_001",
        "contract": {},
        "claim_scope": {},
        "artifacts": [parent_row],
        "integrity": {
            "artifact_count": 1,
            "required_artifact_count": 1,
            "present_required_artifact_count": 1,
            "blockers": [],
        },
        "decision": fidelity.PARENT_DECISION,
    }
    parent_sha256 = _write_json(root, fidelity.PARENT_CLOSURE, parent)

    baseline_paths = dict(fidelity.BASELINE_PATHS)
    baseline_artifacts = {}
    for name, path in baseline_paths.items():
        sha256 = _write_bytes(root, path, f"{name}\n".encode())
        baseline_artifacts[name] = {"path": path.as_posix(), "sha256": sha256}
    _write_bytes(root, fidelity.PROBE_RUNTIME, b"onnxruntime\n")

    proposal = {
        "schema": fidelity.PROPOSAL_SCHEMA,
        "loop_id": fidelity.LOOP_ID,
        "parent_closure": {
            "path": fidelity.PARENT_CLOSURE.as_posix(),
            "sha256": parent_sha256,
            "decision": fidelity.PARENT_DECISION,
        },
        "fixture_matrix": _fixture_matrix(),
        "probe_plan": {"runtime_determinism_repeats": 3},
        "read_allowlist": [path.as_posix() for path in baseline_paths.values()],
        "budget": {
            "max_fixture_count": 4,
            "max_probe_profiles": 2,
            "max_native_subprocesses": 12,
            "total_wall_clock_seconds": 2400,
            "max_retained_output_bytes": 536870912,
            "cpu_only": True,
        },
        "forbidden": [
            "no raw, split, heldout, or training access",
            "do not edit goal.md",
            "do not relax the 1e-6 gate",
        ],
        "exit_decisions": list(fidelity.EXIT_DECISIONS),
        "claim_boundary": _claim_boundary(),
        "decision": fidelity.PROPOSAL_DECISION,
    }
    proposal_sha256 = _write_json(root, fidelity.PROPOSAL, proposal)

    authorization = {
        "schema": fidelity.AUTHORIZATION_SCHEMA,
        "loop_id": fidelity.LOOP_ID,
        "proposal": {"path": fidelity.PROPOSAL.as_posix(), "sha256": proposal_sha256},
        "parent_closure": {
            "path": fidelity.PARENT_CLOSURE.as_posix(),
            "sha256": parent_sha256,
        },
        "baseline_artifacts": baseline_artifacts,
        "authorized_edit_paths": [
            fidelity.PROPOSAL.as_posix(),
            fidelity.AUTHORIZATION.as_posix(),
            fidelity.PREFLIGHT.as_posix(),
            fidelity.IMPLEMENTATION_OUTPUT.as_posix(),
            fidelity.LOCALIZATION_AUTHORIZATION.as_posix(),
            fidelity.LOCALIZATION_LEASE_PENDING.as_posix(),
            fidelity.LOCALIZATION_LEASE.as_posix(),
            fidelity.POST_OUTPUT.as_posix(),
            fidelity.DIAGNOSTIC_SOURCE.as_posix(),
            fidelity.DIAGNOSTIC_TEST.as_posix(),
            fidelity.BUILDER_SOURCE.as_posix(),
            fidelity.BUILDER_TEST.as_posix(),
            "tools/axon_onnx_fidelity/**",
            "reports/roadmap_9997/p0_loop28_onnx_fidelity/**",
        ],
        "not_authorized_before_localization_run_authorization": [
            "do not load the checkpoint",
            "do not execute any ONNX graph",
            "do not create a lease",
            "do not create localization evidence",
        ],
        "never_authorized": [
            "no raw, split, heldout, training, or GPU execution",
            "do not edit goal.md",
            "no F1 computation",
        ],
        "output_policy": {
            "governance_outputs_exclusive_create": True,
            "temporary_probe_outputs_confined": True,
            "temporary_probe_outputs_deleted_after_evidence_freeze": True,
            "baseline_artifacts_rehashed_before_and_after": True,
        },
        "execution_requires_separate_localization_authorization": True,
        "decision": fidelity.AUTHORIZATION_DECISION,
    }
    authorization_sha256 = _write_json(root, fidelity.AUTHORIZATION, authorization)

    preflight = {
        "schema": fidelity.PREFLIGHT_SCHEMA,
        "loop_id": fidelity.LOOP_ID,
        "governance_binding": {
            "proposal": {"path": fidelity.PROPOSAL.as_posix(), "sha256": proposal_sha256},
            "authorization": {
                "path": fidelity.AUTHORIZATION.as_posix(),
                "sha256": authorization_sha256,
            },
            "parent_closure": {
                "path": fidelity.PARENT_CLOSURE.as_posix(),
                "sha256": parent_sha256,
                "verification_result": f"1 artifact; {fidelity.PARENT_DECISION}",
            },
        },
        "baseline_integrity": {
            f"{name}_sha256": record["sha256"] for name, record in baseline_artifacts.items()
        },
        "output_preconditions": {
            "implementation_manifest_present": False,
            "localization_authorization_present": False,
            "localization_lease_present": False,
            "localization_evidence_present": False,
            "post_manifest_present": False,
            "exclusive_create_required": True,
        },
        "access_boundary": {
            "dataset_raw_accessed": False,
            "split_metadata_accessed": False,
            "cache_rows_accessed": False,
            "heldout_accessed": False,
            "prediction_or_metric_payload_accessed": False,
            "training_or_fitting_performed": False,
            "quality_metric_computed": False,
        },
        "implementation_gate": {
            "parent_closure_verified": True,
            "baseline_hashes_match_authorization": True,
            "implementation_allowed": True,
            "model_inference_allowed": False,
            "requires_separate_localization_authorization": True,
        },
        "decision": fidelity.PREFLIGHT_DECISION,
    }
    preflight_sha256 = _write_json(root, fidelity.PREFLIGHT, preflight)
    return {
        "parent": parent,
        "parent_sha256": parent_sha256,
        "parent_artifact": parent_artifact,
        "proposal": proposal,
        "proposal_sha256": proposal_sha256,
        "authorization": authorization,
        "authorization_sha256": authorization_sha256,
        "preflight": preflight,
        "preflight_sha256": preflight_sha256,
        "baseline_paths": baseline_paths,
        "baseline_artifacts": baseline_artifacts,
    }


def _freeze_implementation(root: Path) -> tuple[dict[str, object], dict, str]:
    chain = _setup_implementation(root)
    manifest = fidelity.build_implementation_manifest(
        root,
        generated_at_utc="2026-07-12T00:00:00Z",
    )
    implementation_sha256 = _write_json(root, fidelity.IMPLEMENTATION_OUTPUT, manifest)
    return chain, manifest, implementation_sha256


def _setup_post(root: Path, *, decision: str = "localized_negative_no_raw") -> dict[str, object]:
    chain, implementation, implementation_sha256 = _freeze_implementation(root)
    attempt_id = f"{fidelity.LOOP_ID}_attempt_001"
    localization_authorization = {
        "schema": fidelity.LOCALIZATION_AUTHORIZATION_SCHEMA,
        "loop_id": fidelity.LOOP_ID,
        "issued_at_utc": "2026-07-12T00:01:00Z",
        "proposal_sha256": chain["proposal_sha256"],
        "authorization_sha256": chain["authorization_sha256"],
        "preflight_sha256": chain["preflight_sha256"],
        "parent_closure_sha256": chain["parent_sha256"],
        "implementation_manifest": {
            "path": fidelity.IMPLEMENTATION_OUTPUT.as_posix(),
            "sha256": implementation_sha256,
        },
        "attempt_id": attempt_id,
        "ready_lease_path": fidelity.LOCALIZATION_LEASE_PENDING.as_posix(),
        "consumed_lease_path": fidelity.LOCALIZATION_LEASE.as_posix(),
        "evidence_path": fidelity.LOCALIZATION_EVIDENCE.as_posix(),
        "fixture_names": [row["name"] for row in chain["proposal"]["fixture_matrix"]],
        "baseline_artifacts": chain["baseline_artifacts"],
        "budget": chain["proposal"]["budget"],
        "claim_scope": fidelity.EXPECTED_RUN_CLAIM_SCOPE,
        "decision": fidelity.LOCALIZATION_AUTHORIZATION_DECISION,
    }
    localization_authorization_sha256 = _write_json(
        root,
        fidelity.LOCALIZATION_AUTHORIZATION,
        localization_authorization,
    )
    lease = {
        "schema": fidelity.LOCALIZATION_LEASE_SCHEMA,
        "loop_id": fidelity.LOOP_ID,
        "localization_authorization": {"sha256": localization_authorization_sha256},
        "consumed_at_utc": "2026-07-12T00:02:00Z",
        "original_lease_sha256": "a" * 64,
        "status": fidelity.LEASE_CONSUMED_STATUS,
    }
    lease_sha256 = _write_json(root, fidelity.LOCALIZATION_LEASE, lease)
    execution_baseline_names = ("checkpoint", "onnx_graph", "onnx_data", "fixture_contract")
    execution_hashes = {
        name: chain["baseline_artifacts"][name]["sha256"] for name in execution_baseline_names
    }
    probe_sha256 = next(
        row["sha256"] for row in implementation["artifacts"] if row["name"] == "probe_binary"
    )
    controls_reproduced = decision != "invalid_positive_control_or_lineage_drift"
    localized = decision == "localized_negative_no_raw"
    input_record = {
        "dtype": "<f4",
        "shape": [1, 1],
        "nbytes": 4,
        "sha256": "b" * 64,
    }
    fixtures = []
    first_divergences = []
    for index, fixture in enumerate(chain["proposal"]["fixture_matrix"]):
        expected_control = (
            "fail" if fixture["positive_control"] == "must_fail_base_probability_gate" else "pass"
        )
        fixture_control = controls_reproduced or index > 0
        fixtures.append(
            {
                "fixture": {
                    "name": fixture["name"],
                    "pe_plus": fixture["pe_plus"],
                    "named_resource": fixture["named_resource"],
                    "tls_callbacks": fixture["tls_callbacks"],
                    "expected_control": expected_control,
                },
                "inputs": {
                    "byte_seq": input_record,
                    "pe_features": input_record,
                    "stat_features": input_record,
                },
                "pytorch_determinism": {"bit_exact": True},
                "profiles": {
                    "macro": {"ort_determinism": {"bit_exact": True}},
                    "routing": {"ort_determinism": {"bit_exact": True}},
                },
                "base_probability": {"control_reproduced": fixture_control},
            }
        )
        first_divergences.append(
            {
                "fixture": fixture["name"],
                "macro": {"probe": "logits"} if localized and index == 0 else None,
                "routing": None,
            }
        )
    evidence = {
        "schema": fidelity.LOCALIZATION_EVIDENCE_SCHEMA,
        "loop_id": fidelity.LOOP_ID,
        "generated_at_utc": "2026-07-12T00:03:00Z",
        "governance": {
            "proposal_sha256": chain["proposal_sha256"],
            "authorization_sha256": chain["authorization_sha256"],
            "preflight_sha256": chain["preflight_sha256"],
            "implementation_manifest_sha256": implementation_sha256,
            "localization_authorization_sha256": localization_authorization_sha256,
            "consumed_lease": {
                "path": fidelity.LOCALIZATION_LEASE.as_posix(),
                "sha256": lease_sha256,
                "original_lease_sha256": lease["original_lease_sha256"],
                "status": fidelity.LEASE_CONSUMED_STATUS,
            },
        },
        "scope": {
            "synthetic_only": True,
            "dataset_raw_accessed": False,
            "split_metadata_accessed": False,
            "cache_rows_accessed": False,
            "heldout_accessed": False,
            "training_or_fitting_performed": False,
            "quality_metric_computed": False,
            "f1_computed": False,
        },
        "runtime_contract": {
            "cpu_only": True,
            "graph_optimization": "ORT_DISABLE_ALL",
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL",
            "repeats": 3,
            "probe_sha256": probe_sha256,
        },
        "graph": {
            "onnx_graph_sha256": execution_hashes["onnx_graph"],
            "onnx_data_sha256": execution_hashes["onnx_data"],
        },
        "fixtures": fixtures,
        "first_divergences": first_divergences,
        "determinism_all_passed": True,
        "positive_controls_reproduced": controls_reproduced,
        "baseline_integrity": {
            "before": execution_hashes,
            "after": execution_hashes,
            "stable": True,
        },
        "budget": {
            "fixture_count": 4,
            "profile_count": 2,
            "native_subprocess_count": 13 if decision == "budget_exhausted_no_claim" else 8,
            "wall_clock_seconds": 100.0,
            "retained_probe_output_bytes": 0,
        },
        "claim_boundary": {
            "synthetic_localization_claim_allowed": localized and controls_reproduced,
            "population_parity_claim_allowed": False,
            "quality_claim_allowed": False,
            "native_loop28_ready_claim_allowed": False,
            "native_loop151_ready_claim_allowed": False,
            "raw_rerun_allowed": False,
            "certification_claim_allowed": False,
        },
        "decision": decision,
    }
    evidence_sha256 = _write_json(root, fidelity.LOCALIZATION_EVIDENCE, evidence)
    for path, payload in (
        (fidelity.GOAL_DELTA, b"immutable goal delta\n"),
        (fidelity.JOURNAL_ENTRY, b"immutable journal entry\n"),
        (fidelity.FINAL_STATUS, b"immutable final status\n"),
    ):
        _write_bytes(root, path, payload)
    return {
        **chain,
        "implementation": implementation,
        "implementation_sha256": implementation_sha256,
        "localization_authorization": localization_authorization,
        "localization_authorization_sha256": localization_authorization_sha256,
        "lease": lease,
        "lease_sha256": lease_sha256,
        "evidence": evidence,
        "evidence_sha256": evidence_sha256,
    }


def test_build_and_verify_implementation_manifest(tmp_path: Path) -> None:
    _setup_implementation(tmp_path)
    manifest = fidelity.build_implementation_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:00:00Z",
    )
    assert manifest["decision"] == fidelity.IMPLEMENTATION_DECISION
    assert manifest["integrity"]["artifact_count"] == 21
    assert manifest["integrity"]["blockers"] == []
    assert fidelity.IMPLEMENTATION_OUTPUT.as_posix() not in {
        row["path"] for row in manifest["artifacts"]
    }

    output = fidelity.resolve_fixed_output(
        tmp_path,
        fidelity.IMPLEMENTATION_OUTPUT,
        mode="implementation",
    )
    fidelity._write_exclusive(output, manifest)
    assert fidelity.verify_implementation_manifest(tmp_path) == manifest


def test_implementation_output_is_exclusive_fixed_and_confined(tmp_path: Path) -> None:
    _setup_implementation(tmp_path)
    manifest = fidelity.build_implementation_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:00:00Z",
    )
    output = fidelity.resolve_fixed_output(
        tmp_path,
        fidelity.IMPLEMENTATION_OUTPUT,
        mode="implementation",
    )
    fidelity._write_exclusive(output, manifest)
    with pytest.raises(fidelity.FidelityManifestError, match="already exists"):
        fidelity._write_exclusive(output, manifest)
    with pytest.raises(fidelity.FidelityManifestError, match="not fixed"):
        fidelity.resolve_fixed_output(tmp_path, Path("other.json"), mode="implementation")
    with pytest.raises(fidelity.FidelityManifestError, match="project-relative"):
        fidelity.resolve_fixed_output(tmp_path, Path("../escape.json"), mode="implementation")


def test_missing_implementation_artifact_is_rejected(tmp_path: Path) -> None:
    _setup_implementation(tmp_path)
    (tmp_path / fidelity.PROBE_SOURCE).unlink()
    with pytest.raises(FileNotFoundError):
        fidelity.build_implementation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:00:00Z",
        )


def test_parent_closure_artifact_drift_is_rejected(tmp_path: Path) -> None:
    chain = _setup_implementation(tmp_path)
    _write_bytes(tmp_path, chain["parent_artifact"], b"drift\n")
    with pytest.raises(fidelity.FidelityManifestError, match="SHA-256 mismatch"):
        fidelity.build_implementation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:00:00Z",
        )


def test_governance_hash_link_drift_is_rejected(tmp_path: Path) -> None:
    chain = _setup_implementation(tmp_path)
    preflight = chain["preflight"]
    preflight["governance_binding"]["authorization"]["sha256"] = "f" * 64
    _write_json(tmp_path, fidelity.PREFLIGHT, preflight)
    with pytest.raises(
        fidelity.FidelityManifestError, match="Preflight governance binding drifted"
    ):
        fidelity.build_implementation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:00:00Z",
        )


def test_duplicate_json_keys_are_rejected_in_governance(tmp_path: Path) -> None:
    _setup_implementation(tmp_path)
    path = tmp_path / fidelity.PROPOSAL
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("{\n", '{\n  "schema": "duplicate",\n', 1), encoding="utf-8")
    with pytest.raises(fidelity.FidelityManifestError, match="Duplicate JSON key"):
        fidelity.build_implementation_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:00:00Z",
        )


def test_implementation_verify_detects_source_drift(tmp_path: Path) -> None:
    _chain, manifest, _sha256 = _freeze_implementation(tmp_path)
    assert fidelity.verify_implementation_manifest(tmp_path) == manifest
    _write_bytes(tmp_path, fidelity.DIAGNOSTIC_SOURCE, b"drifted source\n")
    with pytest.raises(fidelity.FidelityManifestError, match="no longer matches"):
        fidelity.verify_implementation_manifest(tmp_path)


@pytest.mark.parametrize("decision", fidelity.EXIT_DECISIONS)
def test_builds_all_four_post_exit_branches(tmp_path: Path, decision: str) -> None:
    chain = _setup_post(tmp_path, decision=decision)
    manifest = fidelity.build_post_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:04:00Z",
    )
    assert manifest["decision"] == fidelity.POST_DECISIONS[decision]
    assert manifest["claim_scope"]["raw_rerun_authorized"] is False
    assert manifest["lineage"]["consumed_lease_sha256"] == chain["lease_sha256"]
    assert manifest["lineage"]["localization_evidence_sha256"] == chain["evidence_sha256"]
    assert manifest["integrity"]["baseline_artifact_count"] == len(fidelity.BASELINE_PATHS)


def test_post_build_and_verify_detects_immutable_document_drift(tmp_path: Path) -> None:
    _setup_post(tmp_path)
    manifest = fidelity.build_post_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:04:00Z",
    )
    output = fidelity.resolve_fixed_output(tmp_path, fidelity.POST_OUTPUT, mode="post")
    fidelity._write_exclusive(output, manifest)
    assert fidelity.verify_post_manifest(tmp_path) == manifest

    _write_bytes(tmp_path, fidelity.GOAL_DELTA, b"drifted goal delta\n")
    with pytest.raises(fidelity.FidelityManifestError, match="no longer matches"):
        fidelity.verify_post_manifest(tmp_path)


def test_unconsumed_lease_is_rejected(tmp_path: Path) -> None:
    chain = _setup_post(tmp_path)
    lease = chain["lease"]
    lease["status"] = "issued_not_consumed"
    _write_json(tmp_path, fidelity.LOCALIZATION_LEASE, lease)
    with pytest.raises(fidelity.FidelityManifestError, match="not consumed"):
        fidelity.build_post_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


def test_pending_lease_is_rejected(tmp_path: Path) -> None:
    _setup_post(tmp_path)
    _write_bytes(tmp_path, fidelity.LOCALIZATION_LEASE_PENDING, b"pending\n")
    with pytest.raises(fidelity.FidelityManifestError, match="Pending localization lease"):
        fidelity.build_post_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


def test_evidence_must_bind_consumed_lease(tmp_path: Path) -> None:
    chain = _setup_post(tmp_path)
    evidence = chain["evidence"]
    evidence["governance"]["consumed_lease"]["sha256"] = "f" * 64
    _write_json(tmp_path, fidelity.LOCALIZATION_EVIDENCE, evidence)
    with pytest.raises(fidelity.FidelityManifestError, match="consumed lease binding drifted"):
        fidelity.build_post_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


def test_evidence_baseline_pre_post_hash_drift_is_rejected(tmp_path: Path) -> None:
    chain = _setup_post(tmp_path)
    evidence = chain["evidence"]
    evidence["baseline_integrity"]["after"]["checkpoint"] = "f" * 64
    _write_json(tmp_path, fidelity.LOCALIZATION_EVIDENCE, evidence)
    with pytest.raises(fidelity.FidelityManifestError, match="baseline evidence inventory drifted"):
        fidelity.build_post_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


def test_current_baseline_hash_drift_is_rejected(tmp_path: Path) -> None:
    chain = _setup_post(tmp_path)
    _write_bytes(tmp_path, chain["baseline_paths"]["checkpoint"], b"drifted checkpoint\n")
    with pytest.raises(fidelity.FidelityManifestError, match="SHA-256 mismatch"):
        fidelity.build_post_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


def test_duplicate_keys_in_localization_evidence_are_rejected(tmp_path: Path) -> None:
    _setup_post(tmp_path)
    path = tmp_path / fidelity.LOCALIZATION_EVIDENCE
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("{\n", '{\n  "schema": "duplicate",\n', 1), encoding="utf-8")
    with pytest.raises(fidelity.FidelityManifestError, match="Duplicate JSON key"):
        fidelity.build_post_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


def test_inconsistent_exit_decision_is_rejected(tmp_path: Path) -> None:
    chain = _setup_post(tmp_path, decision="localized_negative_no_raw")
    evidence = chain["evidence"]
    for row in evidence["first_divergences"]:
        row["macro"] = None
        row["routing"] = None
    _write_json(tmp_path, fidelity.LOCALIZATION_EVIDENCE, evidence)
    with pytest.raises(fidelity.FidelityManifestError, match="inconsistent with decision"):
        fidelity.build_post_manifest(
            tmp_path,
            generated_at_utc="2026-07-12T00:04:00Z",
        )


def test_post_output_is_exclusive_fixed_and_confined(tmp_path: Path) -> None:
    _setup_post(tmp_path)
    manifest = fidelity.build_post_manifest(
        tmp_path,
        generated_at_utc="2026-07-12T00:04:00Z",
    )
    output = fidelity.resolve_fixed_output(tmp_path, fidelity.POST_OUTPUT, mode="post")
    fidelity._write_exclusive(output, manifest)
    with pytest.raises(fidelity.FidelityManifestError, match="already exists"):
        fidelity._write_exclusive(output, manifest)
    with pytest.raises(fidelity.FidelityManifestError, match="not fixed"):
        fidelity.resolve_fixed_output(tmp_path, fidelity.IMPLEMENTATION_OUTPUT, mode="post")
    with pytest.raises(fidelity.FidelityManifestError, match="project-relative"):
        fidelity.verify_post_manifest(tmp_path, Path("../post_manifest.json"))
