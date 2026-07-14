import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_ml_recommendation_status import build_status  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_build_ml_recommendation_status_summarizes_goal_states():
    with _case_dir("ml_recommendation_status") as tmp_path:
        evidence = tmp_path / "scripts" / "build_model_review_report.py"
        evidence.parent.mkdir(parents=True)
        evidence.write_text("", encoding="utf-8")

        payload = build_status(tmp_path)

    by_id = {row["id"]: row for row in payload["recommendations"]}
    assert by_id["current_strict_best_loop151"]["status"] == "current_strict_best"
    assert by_id["current_strict_best_loop151"]["funnel_review"]["full_test"].startswith("F1 0.9908541911")
    assert "Loop136" in by_id["current_strict_best_loop151"]["funnel_review"]["fallback"]
    assert (
        "reports/phase3_loop151/loop151_trusted_signer_guard_full_eval.json"
        in by_id["current_strict_best_loop151"]["evidence_status"]
    )
    assert (
        by_id["loop164_whole_file_residual_expert"]["status"]
        == "parked_surrogate_negative_formal_gate_not_run"
    )
    assert (
        by_id["loop164_whole_file_residual_expert"]["priority"]
        == "parked_surrogate_negative"
    )
    assert by_id["loop164_whole_file_residual_expert"]["remove_from_pending"] is True
    assert "Val errors <=152" in by_id["loop164_whole_file_residual_expert"]["preflight_review"]["program_gate"]
    assert "48 pass only with FN <=24" in by_id["loop164_whole_file_residual_expert"][
        "preflight_review"
    ]["point_target"]
    assert "one-sided 97.5% component-bootstrap" in by_id[
        "loop164_whole_file_residual_expert"
    ]["preflight_review"]["certification_protocol"]
    assert (
        by_id["loop164_whole_file_residual_expert"]["preflight_review"]["decision"]
        == "park_current_loop164_recipe_surrogate_negative_exact_loop151_gate_not_run"
    )
    assert "F1 0.9620420177" in by_id["loop164_whole_file_residual_expert"][
        "preflight_review"
    ]["local_oof_diagnostic"]
    assert (
        "manifests/roadmap_9997/loop164_whole_file_residual_expert/preflight.json"
        in by_id["loop164_whole_file_residual_expert"]["evidence_status"]
    )
    assert (
        "scripts/validate_loop164_nested_oof_execution_receipt.py"
        in by_id["loop164_whole_file_residual_expert"]["evidence_status"]
    )
    assert (
        "scripts/validate_loop164_fold_scope_plan.py"
        in by_id["loop164_whole_file_residual_expert"]["evidence_status"]
    )
    assert (
        "scripts/validate_loop164_training_authority.py"
        in by_id["loop164_whole_file_residual_expert"]["evidence_status"]
    )
    assert (
        "scripts/validate_loop164_whole_file_implementation.py"
        in by_id["loop164_whole_file_residual_expert"]["evidence_status"]
    )
    assert (
        "tests/test_loop164_whole_file_gcg.py"
        in by_id["loop164_whole_file_residual_expert"]["evidence_status"]
    )
    assert (
        "manifests/roadmap_9997/loop164_whole_file_residual_expert/"
        "implementation_contract_authorization.json"
        in by_id["loop164_whole_file_residual_expert"]["evidence_status"]
    )
    assert (
        "manifests/roadmap_9997/champion_registry.json"
        in by_id["loop164_whole_file_residual_expert"]["evidence_status"]
    )
    assert "formal Loop151 gate was not run" in by_id["loop164_whole_file_residual_expert"]["preflight_review"][
        "promotion_blockers"
    ]
    assert "repairs/breaks 75/595" in by_id["loop164_whole_file_residual_expert"][
        "preflight_review"
    ]["surrogate_complementarity"]
    assert "park this Loop164 recipe" in by_id[
        "loop164_whole_file_residual_expert"
    ]["next_action"]
    assert (
        "reports/roadmap_9997/loop165/loop69_loop164_surrogate_complementarity.json"
        in by_id["loop164_whole_file_residual_expert"]["evidence_status"]
    )
    assert (
        by_id["loop166_code_section_foundation"]["status"]
        == "closed_b1_nonfinite_current_recipe_retired"
    )
    assert by_id["loop166_code_section_foundation"]["remove_from_pending"] is True
    assert "251/256" in by_id["loop166_code_section_foundation"]["phase_review"]["phase_a"]
    assert "step 8192/28768" in by_id["loop166_code_section_foundation"]["phase_review"]["phase_b1"]
    assert (
        by_id["loop166_code_section_foundation"]["phase_review"]["decision"]
        == "close_loop166_bpe1024_mlm_recipe_and_preregister_loop167_ember_v3_novel_delta_control"
    )
    assert "Do not retry" in by_id["loop166_code_section_foundation"]["next_action"]
    assert (
        "manifests/roadmap_9997/loop166_code_section_foundation/phase_b1_step4096_recovery_v2_nonfinite_failure.json"
        in by_id["loop166_code_section_foundation"]["evidence_status"]
    )
    assert (
        by_id["loop167_ember_v3_novel_delta"]["status"]
        == "preregistered_phase_a_static_only_phase_b_source_closure_pending"
    )
    assert by_id["loop167_ember_v3_novel_delta"]["remove_from_pending"] is False
    assert "arms B0/B1/M/A/CF" in by_id["loop167_ember_v3_novel_delta"]["phase_review"]["phase_b"]
    assert "without a public key" in by_id["loop167_ember_v3_novel_delta"]["phase_review"]["authority"]
    assert (
        by_id["loop167_ember_v3_novel_delta"]["phase_review"]["decision"]
        == "grant_phase_a_static_only_and_withhold_phase_b_until_new_source_closed_authorization"
    )
    assert (
        "manifests/roadmap_9997/loop167_ember_v3_novel_delta/proposal.json"
        in by_id["loop167_ember_v3_novel_delta"]["evidence_status"]
    )
    assert by_id["loop152_val_noise_redraw_readiness"]["status"] == "awaiting_independent_val_verdicts"
    assert "rows 86" in by_id["loop152_val_noise_redraw_readiness"]["redraw_review"]["current_counts"]
    assert (
        "reports/phase3_loop152/loop152_loop150_val86_redraw_readiness_summary.json"
        in by_id["loop152_val_noise_redraw_readiness"]["evidence_status"]
    )
    assert by_id["loop153_current_best_val_noise_focus"]["status"] == "awaiting_independent_val_verdicts"
    assert "focus rows 73" in by_id["loop153_current_best_val_noise_focus"]["redraw_review"]["current_counts"]
    assert "162" in by_id["loop153_current_best_val_noise_focus"]["redraw_review"]["source_alignment"]
    assert (
        "reports/phase3_loop153/loop153_loop151_val_noise_focus_summary.json"
        in by_id["loop153_current_best_val_noise_focus"]["evidence_status"]
    )
    assert (
        "docs/phase3_loop153_current_best_val_noise_focus_report.md"
        in by_id["loop153_current_best_val_noise_focus"]["evidence_status"]
    )
    assert by_id["loop156_current_best_val_all_error_review"]["status"] == "awaiting_independent_val_verdicts"
    assert "review_rows 162" in by_id["loop156_current_best_val_all_error_review"]["redraw_review"]["current_counts"]
    assert "73/162" in by_id["loop156_current_best_val_all_error_review"]["redraw_review"]["coverage"]
    assert (
        "reports/phase3_loop156/loop156_loop151_val_all_errors_summary.json"
        in by_id["loop156_current_best_val_all_error_review"]["evidence_status"]
    )
    assert (
        "tests/test_build_loop156_current_best_val_full_error_review.py"
        in by_id["loop156_current_best_val_all_error_review"]["evidence_status"]
    )
    assert by_id["loop157_external_annotation_package"]["status"] == "ready_for_external_content_annotation"
    assert "rows 162" in by_id["loop157_external_annotation_package"]["external_review"]["package_state"]
    assert "automatic verdict" in by_id["loop157_external_annotation_package"]["external_review"]["authorization"]
    assert (
        "reports/phase3_loop157/loop157_loop151_val_all_errors_external_package_summary.json"
        in by_id["loop157_external_annotation_package"]["evidence_status"]
    )
    assert (
        "tests/test_export_loop157_current_best_val_external_annotation_package.py"
        in by_id["loop157_external_annotation_package"]["evidence_status"]
    )
    assert by_id["loop158_loop157_external_annotation_ingress"]["status"] == "ready_noop_no_external_annotations"
    assert "context rows 162" in by_id["loop158_loop157_external_annotation_ingress"]["external_review"]["current_counts"]
    assert (
        "exact four columns"
        in by_id["loop158_loop157_external_annotation_ingress"]["external_review"]["accepted_schema"]
    )
    assert (
        "identity/model note terms"
        in by_id["loop158_loop157_external_annotation_ingress"]["next_action"]
    )
    assert (
        "reports/phase3_loop158/loop158_loop157_external_annotation_import_summary.json"
        in by_id["loop158_loop157_external_annotation_ingress"]["evidence_status"]
    )
    assert (
        "tests/test_import_loop158_current_best_val_external_annotations.py"
        in by_id["loop158_loop157_external_annotation_ingress"]["evidence_status"]
    )
    assert by_id["loop154_trusted_signer_threshold_t0995"]["status"] == "experimentally_not_useful_retain_record"
    assert by_id["loop154_trusted_signer_threshold_t0995"]["failure_review"]["evidence_strength"] == "强证据"
    assert "identical Val" in by_id["loop154_trusted_signer_threshold_t0995"]["failure_review"]["failure_observation"]
    assert (
        "reports/phase3_loop154/loop154_trusted_signer_guard_t0995_full_eval.json"
        in by_id["loop154_trusted_signer_threshold_t0995"]["evidence_status"]
    )
    assert by_id["loop155_candidate_governance_audit"]["status"] == "completed_governance_record"
    assert "full-test 1460 errors" in by_id["loop155_candidate_governance_audit"]["governance_review"]["full_test_mirage"]
    assert (
        "reports/phase3_loop155/loop155_candidate_governance_audit.json"
        in by_id["loop155_candidate_governance_audit"]["evidence_status"]
    )
    assert (
        "tests/test_build_loop155_candidate_governance_audit.py"
        in by_id["loop155_candidate_governance_audit"]["evidence_status"]
    )
    assert by_id["loop159_r11_only_recall_candidate"]["status"] == "high_recall_tradeoff_not_strict_best"
    assert "errors 155" in by_id["loop159_r11_only_recall_candidate"]["governance_review"]["val"]
    assert "errors 78" in by_id["loop159_r11_only_recall_candidate"]["governance_review"]["test10k"]
    assert "errors 1491" in by_id["loop159_r11_only_recall_candidate"]["governance_review"]["full_test"]
    assert "FN improves by 58" in by_id["loop159_r11_only_recall_candidate"]["governance_review"]["tradeoff"]
    assert (
        "reports/phase3_loop159/loop159_r11_only_candidate_audit.json"
        in by_id["loop159_r11_only_recall_candidate"]["evidence_status"]
    )
    assert (
        "tests/test_build_loop159_r11_only_candidate_audit.py"
        in by_id["loop159_r11_only_recall_candidate"]["evidence_status"]
    )
    assert by_id["loop160_lowprob_r11_gate"]["status"] == "failed_full_test_after_val_test10k_pass"
    assert "<= 0.2487261742" in by_id["loop160_lowprob_r11_gate"]["governance_review"]["selection"]
    assert "accepted 3" in by_id["loop160_lowprob_r11_gate"]["governance_review"]["val"]
    assert "accepted 1" in by_id["loop160_lowprob_r11_gate"]["governance_review"]["test10k"]
    assert "accepted 41" in by_id["loop160_lowprob_r11_gate"]["governance_review"]["full_test"]
    assert "did not generalize" in by_id["loop160_lowprob_r11_gate"]["governance_review"]["failure_mode"]
    assert (
        "reports/phase3_loop160/loop160_lowprob_r11_gate_audit.json"
        in by_id["loop160_lowprob_r11_gate"]["evidence_status"]
    )
    assert (
        "tests/test_build_loop160_lowprob_r11_gate.py"
        in by_id["loop160_lowprob_r11_gate"]["evidence_status"]
    )
    assert by_id["loop161_test10k_promotion_margin_guard"]["status"] == "active_promotion_guard"
    assert "Test-10k improvement >= 3" in by_id["loop161_test10k_promotion_margin_guard"]["governance_review"]["rule"]
    assert "Val -17" in by_id["loop161_test10k_promotion_margin_guard"]["governance_review"]["accepted"]
    assert "Loop160: Test-10k -1" in by_id["loop161_test10k_promotion_margin_guard"]["governance_review"]["rejected"]
    assert (
        "reports/phase3_loop161/loop161_test10k_promotion_margin_guard.json"
        in by_id["loop161_test10k_promotion_margin_guard"]["evidence_status"]
    )
    assert (
        "tests/test_build_loop161_test10k_promotion_margin_guard.py"
        in by_id["loop161_test10k_promotion_margin_guard"]["evidence_status"]
    )
    assert by_id["loop162_loop160_failure_posthoc"]["status"] == "posthoc_failure_record_only"
    assert "accepted 3" in by_id["loop162_loop160_failure_posthoc"]["failure_review"]["val"]
    assert "accepted 1" in by_id["loop162_loop160_failure_posthoc"]["failure_review"]["test10k"]
    assert "accepted 41" in by_id["loop162_loop160_failure_posthoc"]["failure_review"]["full_test"]
    assert "Posthoc only" in by_id["loop162_loop160_failure_posthoc"]["failure_review"]["selection_policy"]
    assert (
        "reports/phase3_loop162/loop162_loop160_failure_posthoc.json"
        in by_id["loop162_loop160_failure_posthoc"]["evidence_status"]
    )
    assert (
        "tests/test_build_loop162_loop160_failure_posthoc.py"
        in by_id["loop162_loop160_failure_posthoc"]["evidence_status"]
    )
    assert by_id["loop163_r11_rescue_support_audit"]["status"] == "reject_low_support_no_selector_training"
    assert "Val disagreements >= 30" in by_id["loop163_r11_rescue_support_audit"]["governance_review"]["thresholds"]
    assert "disagreements 9" in by_id["loop163_r11_rescue_support_audit"]["governance_review"]["val"]
    assert "fixes/breaks 3/3" in by_id["loop163_r11_rescue_support_audit"]["governance_review"]["test10k"]
    assert "fixes/breaks 58/83" in by_id["loop163_r11_rescue_support_audit"]["governance_review"]["full_test_posthoc"]
    assert (
        "reports/phase3_loop163/loop163_r11_rescue_support_audit.json"
        in by_id["loop163_r11_rescue_support_audit"]["evidence_status"]
    )
    assert (
        "tests/test_build_loop163_r11_rescue_support_audit.py"
        in by_id["loop163_r11_rescue_support_audit"]["evidence_status"]
    )
    assert by_id["unified_model_review_gate"]["remove_from_pending"] is True
    assert by_id["fixed_v2_20w_uncompressed_cache"]["remove_from_pending"] is True
    assert by_id["fixed_v2_20w_uncompressed_cache"]["status"] == "completed_removed_from_pending"
    assert (
        "reports/random_20w_split/random_20w_8192_uncompressed_cache_coverage_audit.json"
        in by_id["fixed_v2_20w_uncompressed_cache"]["evidence_status"]
    )
    assert (
        "reports/random_20w_split/random_20w_8192_uncompressed_test10k_after_lookup_opt_eval.json"
        in by_id["fixed_v2_20w_uncompressed_cache"]["evidence_status"]
    )
    assert (
        "scripts/audit_split_cache_coverage.py"
        in by_id["fixed_v2_20w_uncompressed_cache"]["evidence_status"]
    )
    assert "99.9%" in by_id["fixed_v2_20w_uncompressed_cache"]["validation_review"]["target_result"]
    assert by_id["rl_mainline"]["status"] == "experimentally_not_useful_retain_record"
    assert by_id["rl_mainline"]["failure_review"]["evidence_strength"] == "强证据"
    assert "reward" in by_id["rl_mainline"]["failure_review"]["inferred_cause"]
    assert by_id["probability_calibration"]["remove_from_pending"] is True
    assert (
        "reports/model_review/final_model_selection/ab_strict_reverification_report.json"
        in by_id["probability_calibration"]["evidence_status"]
    )
    assert (
        "reports/model_review/final_model_selection/probability_calibrator_test_strict_full.json"
        in by_id["probability_calibration"]["evidence_status"]
    )
    assert (
        "reports/model_review/final_model_selection/probability_calibrator_hard_fn_holdout_strict_full.json"
        in by_id["probability_calibration"]["evidence_status"]
    )
    assert (
        "reports/model_review/final_model_selection/probability_calibrator_high_value_benign_strict_full.json"
        in by_id["probability_calibration"]["evidence_status"]
    )
    assert (
        "reports/model_review/final_model_selection/ab_strict_reverification_report.md"
        in by_id["ga_feature_mask"]["evidence_status"]
    )
    assert (
        "reports/model_review/final_model_selection/ga_feature_mask_full_holdout_summary.json"
        in by_id["ga_feature_mask"]["evidence_status"]
    )
    assert (
        "reports/model_review/final_model_selection/high_value_benign_baseline_analysis/prediction_error_summary.json"
        in by_id["probability_calibration"]["evidence_status"]
    )
    assert (
        "reports/model_review/final_model_selection/high_value_benign_ga_mask_analysis/prediction_error_summary.json"
        in by_id["ga_feature_mask"]["evidence_status"]
    )
    assert by_id["ga_feature_mask"]["tradeoff_review"]["default_decision"].startswith("High-security")
    assert by_id["hard_example_replay"]["status"] == "experimentally_not_useful_retain_record"
    assert by_id["hard_example_replay"]["failure_review"]["evidence_strength"] == "强证据"
    assert (
        "models/generalization_group_isolated_seed_confirm/seed_plan.json"
        in by_id["byte_noise_near_threshold"]["evidence_status"]
    )
    assert by_id["byte_noise_near_threshold"]["status"] == "experimentally_not_useful_retain_record"
    assert by_id["byte_noise_near_threshold"]["failure_review"]["evidence_strength"] == "强证据"
    assert (
        "models/generalization_group_isolated_seed_confirm/summary.md"
        in by_id["byte_noise_near_threshold"]["evidence_status"]
    )
    assert (
        "reports/hard_family_finetune/balanced_replay_strict_source_group_threshold063/hard_error_finetune_plan.json"
        in by_id["hard_example_replay"]["evidence_status"]
    )
    assert by_id["unified_model_review_gate"]["evidence_status"]["scripts/build_model_review_report.py"] == "present"
    assert payload["summary"]["completed_removed"] >= 2
    assert payload["summary"]["negative_retained"] >= 4
    assert payload["summary"]["open"] >= 1
