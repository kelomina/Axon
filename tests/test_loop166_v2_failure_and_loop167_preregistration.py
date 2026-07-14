from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.loop166.raw_progress_ledger import validate_raw_progress_ledger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOP166_MANIFEST_ROOT = (
    PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop166_code_section_foundation"
)
LOOP167_MANIFEST_ROOT = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta"


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_binding(binding: dict) -> Path:
    path = PROJECT_ROOT / binding["path"]
    assert path.is_file()
    assert _sha256(path) == binding["sha256"]
    return path


def test_loop166_v2_failure_closure_matches_immutable_evidence() -> None:
    incident_path = LOOP166_MANIFEST_ROOT / "phase_b1_step4096_recovery_v2_nonfinite_failure.json"
    incident = _load_json(incident_path)

    assert incident["status"] == (
        "incomplete_fail_closed_nonfinite_gradient_after_complete_authorized_raw_scan_"
        "and_partial_durable_checkpoint"
    )
    assert incident["failure_observation"]["worker_exception_message"] == (
        "B1 training produced a non-finite gradient norm"
    )
    assert incident["frozen_contract_failure_mapping"]["applicable_to_observed_failure"] is False
    for binding in incident["source_closure"].values():
        if isinstance(binding, dict) and "path" in binding:
            _assert_binding(binding)

    _assert_binding(incident["lease_closure"])
    ledger_path = _assert_binding(incident["raw_progress_ledger"])
    validation = validate_raw_progress_ledger(ledger_path)
    assert validation.status == "complete"
    assert validation.line_count == 32002
    assert validation.terminal_record_count == 16000
    assert validation.cumulative_raw_open_attempts == 15988
    assert validation.cumulative_raw_open_successes == 15988
    assert validation.cumulative_raw_bytes_read == 19239582561
    assert validation.final_record_sha256 == (
        "51ee9446ac3af97590b3a940b4114eff8c9b9da05d58dc873b58073ac4fc4dc4"
    )

    _assert_binding(incident["partial_checkpoint"])
    assert incident["partial_checkpoint"]["completed_optimizer_steps"] == 8192
    assert incident["partial_checkpoint"]["total_optimizer_steps"] == 28768
    assert incident["partial_checkpoint"]["final_verified"] is False
    assert incident["partial_checkpoint"]["resume_authorized"] is False
    for binding in incident["supervisor_closure"].values():
        if isinstance(binding, dict) and "path" in binding:
            _assert_binding(binding)
    assert incident["supervisor_closure"]["job_tree_empty_after_exit"] is True
    assert incident["supervisor_closure"]["timeout_termination"] is False

    accounting = incident["raw_pass_accounting"]
    assert accounting["physical_completed_full_fit_raw_passes_minimum"] == 2
    assert accounting["physical_completed_full_fit_raw_passes_maximum"] == 3
    assert accounting["physical_completed_full_fit_raw_passes_exact"] is False
    assert accounting["charged_full_fit_raw_pass_equivalents"] == 3
    for absent in incident["absent_success_artifacts"]:
        assert not (PROJECT_ROOT / absent["path"]).exists()
        assert absent["must_not_be_backfilled"] is True
    assert not any(incident["ready_for"].values())


def test_loop166_decision_retires_recipe_and_selects_loop167() -> None:
    incident_path = LOOP166_MANIFEST_ROOT / "phase_b1_step4096_recovery_v2_nonfinite_failure.json"
    decision = _load_json(LOOP166_MANIFEST_ROOT / "phase_b1_nonfinite_decision.json")

    assert decision["failure_incident"]["sha256"] == _sha256(incident_path)
    assert decision["recipe_disposition"]["retired"] is True
    assert decision["recipe_disposition"]["same_lineage_retry_resume_or_recovery_allowed"] is False
    assert decision["successor_program"]["candidate_id"] == "Loop167"
    assert decision["successor_program"]["route"] == "ember_v3_novel_delta_structural_control"
    assert decision["successor_program"]["execution_authorized_by_this_decision"] is False
    assert decision["research_champion"] == "Loop151"
    assert decision["target_achieved"] is False


def test_loop167_preregistration_freezes_causal_controls_and_scope() -> None:
    proposal_path = LOOP167_MANIFEST_ROOT / "proposal.json"
    proposal = _load_json(proposal_path)
    authorization = _load_json(LOOP167_MANIFEST_ROOT / "authorization.json")

    dimension_proof = proposal["official_v3_dimension_proof"]
    assert sum(value for key, value in dimension_proof.items() if key != "total") == 2568
    assert dimension_proof["total"] == 2568
    mapping = proposal["phase_a_semantic_delta_freeze"]
    assert mapping["raw_opens_allowed"] == 0
    assert mapping["mapping_requirements"]["official_columns_classified_exactly_once"] == 2568
    assert mapping["mapping_requirements"]["minimum_genuinely_novel_semantic_groups"] == 3
    assert set(mapping["forced_overlap_controls"]) == {"authenticode", "data_directories"}

    inventory = proposal["axon_structural_inventory"]
    for name, binding in inventory.items():
        if isinstance(binding, dict) and "path" not in binding and "source" in binding:
            assert _sha256(PROJECT_ROOT / binding["source"]) == binding["sha256"], name
    assert inventory["baseline_total_dimension_before_deduplicated_allowlist_freeze"] == 572

    phase_b = proposal["phase_b_train_only_oof"]
    _assert_binding(phase_b["input_folds"])
    assert (
        _sha256(PROJECT_ROOT / phase_b["input_folds"]["summary_path"])
        == phase_b["input_folds"]["summary_sha256"]
    )
    assert phase_b["execution_ready"] is False
    assert phase_b["seeds"] == [41, 42, 43]
    assert phase_b["hard_threshold"] == 0.5
    assert phase_b["threshold_sweep_allowed"] is False
    assert set(phase_b["arms"]) == {"B0", "B1", "M", "A", "CF"}
    assert phase_b["fit_count"]["maximum_total_fits"] == 75
    assert phase_b["raw_access"]["exact_train_raw_passes"] == 1
    assert phase_b["raw_access"]["maximum_raw_open_attempts"] == 20000

    gate = proposal["quality_gate"]
    assert gate["per_seed_all_required"]["minimum_repairs"] == 50
    assert gate["per_seed_all_required"]["minimum_override_precision"] == 0.8
    assert gate["per_seed_all_required"]["minimum_net_positive_folds"] == 4
    assert gate["per_seed_all_required"]["minimum_M_net_reduction_advantage_over_CF"] == 30
    assert gate["component_bootstrap"]["replicates"] == 200000
    assert gate["failure_decision"] == "close_ember_v3_novel_delta_control"

    assert authorization["proposal_binding"]["sha256"] == _sha256(proposal_path)
    assert authorization["authorization_granted"] is True
    assert authorization["phase_a_limits"]["raw_sample_opens_allowed"] == 0
    assert (
        authorization["phase_b_conditional_authority"]["execution_authorization_granted"] is False
    )
    assert authorization["phase_b_conditional_authority"]["execution_ready"] is False
    assert authorization["authority"]["public_key_required"] is False
    assert authorization["ready_for"]["phase_a_static_mapping"] is True
    assert all(
        authorization["ready_for"][key] is False
        for key in (
            "phase_b_raw_access",
            "phase_b_train_oof",
            "val",
            "test10k",
            "legacy_full_test",
            "promotion",
            "certification",
        )
    )
