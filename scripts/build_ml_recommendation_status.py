#!/usr/bin/env python3
"""Build a machine-readable status ledger for ML improvement recommendations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

RECOMMENDATIONS = [
    {
        "id": "current_strict_best_loop151",
        "title": "当前 strict best：Loop151 trusted signer guard",
        "priority": "current_best",
        "status": "current_strict_best",
        "remove_from_pending": False,
        "evidence": [
            "docs/phase3_loop151_trusted_signer_guard_report.md",
            "scripts/evaluate_authenticode_trusted_signer_guard.py",
            "tests/test_evaluate_authenticode_trusted_signer_guard.py",
            "reports/phase3_loop151/loop151_trusted_signer_guard_val_eval.json",
            "reports/phase3_loop151/loop151_trusted_signer_guard_test10k_eval.json",
            "reports/phase3_loop151/loop151_trusted_signer_guard_full_eval.json",
            "reports/phase3_loop151/loop151_trusted_signer_guard_full_predictions.csv",
        ],
        "funnel_review": {
            "val": "F1 0.9919193935, errors 162, FP/FN 105/57",
            "test10k": "F1 0.9921921922, errors 78, FP/FN 49/29",
            "full_test": "F1 0.9908541911, errors 1466/160000, FP/FN 879/587",
            "delta_vs_loop136_full": "errors -78, FP -79, FN +1",
            "fallback": "Loop136 remains recall fallback if the business cost of FN dominates total-error/F1.",
        },
        "next_action": (
            "Use Loop151 as the strict-best F1/total-error reference. Freeze signer terms; any publisher expansion "
            "must restart at Val and must not be selected from full-test observations."
        ),
    },
    {
        "id": "loop164_whole_file_residual_expert",
        "title": "Loop164 whole-file GCG：当前 recipe 代理负结果后停算",
        "priority": "parked_surrogate_negative",
        "status": "parked_surrogate_negative_formal_gate_not_run",
        "remove_from_pending": True,
        "evidence": [
            "docs/phase3_loop164_whole_file_residual_expert_proposal.md",
            "scripts/build_loop164_mainline_preflight.py",
            "tests/test_build_loop164_mainline_preflight.py",
            "scripts/validate_loop164_isolation_contract.py",
            "scripts/validate_loop164_nested_oof_execution_receipt.py",
            "tests/test_validate_loop164_nested_oof_execution_receipt.py",
            "scripts/validate_loop164_fold_scope_plan.py",
            "tests/test_validate_loop164_fold_scope_plan.py",
            "scripts/validate_loop164_training_authority.py",
            "tests/test_validate_loop164_training_authority.py",
            "scripts/validate_loop164_whole_file_implementation.py",
            "tests/test_validate_loop164_whole_file_implementation.py",
            "tests/test_loop164_whole_file_gcg.py",
            "scripts/run_loop164_local_whole_file_oof.py",
            "scripts/analyze_loop164_local_oof_result.py",
            "tests/test_loop164_local_oof.py",
            "tests/test_analyze_loop164_local_oof_result.py",
            "docs/phase3_loop164_local_whole_file_oof_diagnostic_report.md",
            "reports/roadmap_9997/loop164/local_whole_file_oof_report.json",
            "reports/roadmap_9997/loop164/local_whole_file_oof_predictions.jsonl",
            "reports/roadmap_9997/loop164/local_whole_file_oof_analysis.json",
            "scripts/audit_loop165_loop69_loop164_surrogate_complementarity.py",
            "tests/test_audit_loop165_loop69_loop164_surrogate_complementarity.py",
            "docs/phase3_loop165_loop69_loop164_surrogate_complementarity.md",
            "reports/roadmap_9997/loop165/loop69_loop164_surrogate_complementarity.json",
            "manifests/roadmap_9997/loop165_surrogate_complementarity/decision.json",
            "manifests/roadmap_9997/loop164_whole_file_residual_expert/frontier_sources.json",
            "manifests/roadmap_9997/loop164_whole_file_residual_expert/proposal.json",
            "manifests/roadmap_9997/loop164_whole_file_residual_expert/authorization.json",
            "manifests/roadmap_9997/loop164_whole_file_residual_expert/implementation_contract_authorization.json",
            "manifests/roadmap_9997/loop164_whole_file_residual_expert/preflight.json",
            "manifests/roadmap_9997/champion_registry.json",
        ],
        "preflight_review": {
            "scope": "A1 aggregate-only static preflight; no raw, checkpoint, prediction-row, cache-row, training, fitting, or heldout evaluation.",
            "selected_evidence": "MalConv2-style low-memory whole-file byte GCG plus strict Train-OOF residual fusion.",
            "point_target": (
                "Legacy development reference uses the exact 9997/10000 inequality: 47 total errors always pass; "
                "48 pass only with FN <=24; 49 always fail. It is not certification evidence."
            ),
            "certification_protocol": (
                "A1 preregisters two ordered A3 sealed windows, per-window one-sided 97.5% component-bootstrap "
                "LCB >=0.9997, 200000 bootstrap replicates, and a hash-bound 50000-simulation joint-power gate; "
                "all runtime evidence remains missing."
            ),
            "program_gate": "Val errors <=152 with FP/FN <=105/57; frozen Test-10k errors <=73 with FP/FN <=49/29.",
            "execution_blockers": (
                "A2 metadata isolation v2 authority and v3 receipt, full-pool group/time manifest, custodian fold scope plan plus its "
                "aggregate-only validation receipt, post-isolation static implementation manifest, separately pinned A2 training authority, "
                "Loop151 Train OOF input bundle, resource guard, final lease, and validated nested OOF receipt are missing."
            ),
            "promotion_blockers": (
                "The formal Loop151 gate was not run: Loop69 is a Loop61-style surrogate, its random folds do not "
                "match Loop164 content-component folds, and four current rows lack a Loop69 baseline."
            ),
            "local_oof_diagnostic": (
                "One-seed five-fold Train-only content-group diagnostic: supported 19540/20000, fixed-0.5 "
                "F1 0.9620420177, errors 748 (FP/FN 488/260), conservative all-missing-wrong F1 "
                "0.9400971933, fold F1 0.9496402878..0.9709694142. No OOM/nonfinite or heldout access."
            ),
            "surrogate_complementarity": (
                "SHA-aligned common rows 19996; supported 19540; decision changes 670; repairs/breaks "
                "75/595; blind-switch precision 0.1119402985; net error reduction -520. All five "
                "Loop164 diagnostic folds are net negative."
            ),
            "formal_gate": "not_run_blocked_wrong_base_lineage_and_fold_scope",
            "decision": "park_current_loop164_recipe_surrogate_negative_exact_loop151_gate_not_run",
        },
        "next_action": (
            "Keep Loop151 as champion and park this Loop164 recipe: no more standalone seeds, epochs, threshold "
            "search, heldout runs, or Loop151 OOF reconstruction solely for it. Build materially independent "
            "foundation, structural, behavior, and label-quality evidence first. Rebuild decision-aligned OOF only "
            "when multiple stronger experts justify a shared router; the current surrogate is not a formal Loop151 gate."
        ),
    },
    {
        "id": "loop166_code_section_foundation",
        "title": "Loop166 code-section BPE/MLM：B1 nonfinite 后关闭当前 recipe",
        "priority": "closed_negative_foundation_recipe",
        "status": "closed_b1_nonfinite_current_recipe_retired",
        "remove_from_pending": True,
        "evidence": [
            "docs/phase3_loop166_code_section_foundation_proposal.md",
            "docs/phase3_loop166_code_section_extractor_probe.md",
            "docs/phase3_loop166_phase_b1_nonfinite_closure.md",
            "manifests/roadmap_9997/loop166_code_section_foundation/proposal.json",
            "manifests/roadmap_9997/loop166_code_section_foundation/phase_a_decision.json",
            "manifests/roadmap_9997/loop166_code_section_foundation/phase_b1_step4096_recovery_v2.json",
            "manifests/roadmap_9997/loop166_code_section_foundation/phase_b1_step4096_recovery_v2_authorization.json",
            "manifests/roadmap_9997/loop166_code_section_foundation/phase_b1_step4096_recovery_v2_nonfinite_failure.json",
            "manifests/roadmap_9997/loop166_code_section_foundation/phase_b1_nonfinite_decision.json",
            "src/loop166/__init__.py",
            "src/loop166/code_sections.py",
            "scripts/run_loop166_code_section_extractor_probe.py",
            "tests/test_loop166_code_sections.py",
            "tests/test_loop166_v2_failure_and_loop167_preregistration.py",
            "reports/roadmap_9997/loop166/code_section_extractor_probe.json",
            "reports/roadmap_9997/loop166/phase_b1_step4096_recovery_v2_consumed.json",
            "reports/roadmap_9997/loop166/phase_b1_step4096_recovery_v2_raw_progress.jsonl",
            "reports/roadmap_9997/loop166/phase_b1_step4096_recovery_v2_exit_receipt.json",
            "reports/roadmap_9997/loop166/phase_b1_step4096_recovery_v2_stderr.log",
            "models/roadmap_9997/loop166/phase_b1_step4096_recovery_v2_tiny_mlm.pt",
        ],
        "phase_review": {
            "claim_scope": "Local Train-only failure closure; no quality metric, threshold, heldout, promotion, or certification.",
            "external_basis": (
                "MalwarePT arXiv:2605.16455 v1 motivates PE code-section BPE/MLM but has no public code/weights; "
                "Axon freezes a 10-15M scaled diagnostic instead of claiming an 86M reproduction."
            ),
            "phase_a": (
                "251/256 successful, 5 explicit no-executable-section missing, coverage 0.98046875, "
                "silent drop 0, elapsed 1.2943352s, peak RSS 48603136 bytes, raw-code artifact 0 bytes."
            ),
            "phase_b1": (
                "Recovery v2 completed the authorized Train scan: 16000 ledger records, 15988/15988 raw "
                "opens/successes, 19239582561 bytes, outer-holdout 0/0. A durable partial checkpoint reached "
                "step 8192/28768 before the worker raised a non-finite gradient norm; the success report and "
                "final-verify receipt remain absent."
            ),
            "raw_pass_accounting": (
                "Physical completed full-fit passes are 2..3 because v1 is unknown; charged full-pass "
                "equivalents are 3. The v2 lease, ledger, partial checkpoint, receipts, and logs are immutable."
            ),
            "decision": "close_loop166_bpe1024_mlm_recipe_and_preregister_loop167_ember_v3_novel_delta_control",
            "blocked": "The current BPE-1024/MLM recipe is retired. Same-lineage retry, resume, numerical recipe repair, Phase C, five-fold, Val/Test/full, and promotion are blocked.",
        },
        "next_action": (
            "Do not retry or repair Loop166. Preserve all v2 evidence and proceed only through the independent "
            "Loop167 EMBER-v3 novel-delta preregistration; no public key is required for local Train-only work."
        ),
    },
    {
        "id": "loop167_ember_v3_novel_delta",
        "title": "Loop167 EMBER-v3 novel-delta structural control",
        "priority": "P0_frontier_fallback_control",
        "status": "preregistered_phase_a_static_only_phase_b_source_closure_pending",
        "remove_from_pending": False,
        "evidence": [
            "docs/phase3_loop167_ember_v3_novel_delta_proposal.md",
            "manifests/roadmap_9997/loop167_ember_v3_novel_delta/proposal.json",
            "manifests/roadmap_9997/loop167_ember_v3_novel_delta/authorization.json",
            "tests/test_loop166_v2_failure_and_loop167_preregistration.py",
        ],
        "phase_review": {
            "claim_scope": "Static semantic-delta preregistration only; Phase B raw access and Train-OOF are not yet executable.",
            "external_basis": (
                "EMBER2024 commit 0ef753e81d98bf209f71b03cd331dfc190b5b54d defines a 2568-dimensional "
                "v3 PE feature schema. Axon must classify every column as exact overlap, partial overlap, "
                "genuinely novel, or forbidden before opening raw samples."
            ),
            "phase_a": (
                "Raw opens/training/dependency installs are zero. Dimension conservation, at least three novel "
                "semantic groups, feature-order commitment, reference vectors, finite outputs, and missing "
                "semantics must pass; Authenticode and data directories are forced overlap controls."
            ),
            "phase_b": (
                "Preregistered but source-closure blocked: Train-only 20000-row five-fold OOF, seeds 41/42/43, "
                "fixed threshold 0.5 and HGB, arms B0/B1/M/A/CF, one raw pass, at most 75 fits and 8h total wall."
            ),
            "quality_gate": (
                "Every seed requires net error reduction >=max(30, 10% of the stronger control errors), at "
                "least 50 repairs, override precision >=0.80, 4/5 positive folds, one-sided component LCB >0, "
                "bounded FP/FN regression, low delta error overlap, and a >=30 causal advantage over shuffled delta."
            ),
            "authority": (
                "Phase A static work is authorized without a public key. Phase B requires a new source-closed "
                "authorization and one-shot lease; Val/Test/full/promotion remain denied."
            ),
            "decision": "grant_phase_a_static_only_and_withhold_phase_b_until_new_source_closed_authorization",
        },
        "next_action": (
            "Implement the project-native semantic mapping, extractor, frozen deduplicated baseline allowlist, "
            "controller, synthetic tests, and resource guard. Issue the separate one-shot Phase B authorization "
            "only after all hashes and canonical paths are closed; do not open raw data before then."
        ),
    },
    {
        "id": "loop152_val_noise_redraw_readiness",
        "title": "Loop150 Val 86 噪声复核到 fresh redraw readiness",
        "priority": "P0_data_governance",
        "status": "awaiting_independent_val_verdicts",
        "remove_from_pending": False,
        "evidence": [
            "docs/phase3_loop152_loop150_val_redraw_readiness_report.md",
            "scripts/run_loop152_loop150_val_focus_redraw_readiness.py",
            "tests/test_run_loop152_loop150_val_focus_redraw_readiness.py",
            "reports/phase3_loop152/loop152_loop150_val86_preflight.json",
            "reports/phase3_loop152/loop152_loop150_val86_redraw_readiness_summary.json",
            "reports/phase3_loop152/loop152_strict_adjustment_plan.json",
            "reports/phase3_loop152/loop152_loop76_readiness.json",
        ],
        "redraw_review": {
            "scope": "Loop150 Val high-conflict focus rows only; full-test focus rows are not model-selection evidence.",
            "current_counts": "rows 86, annotated_rows 0, replacement_required 0, decision await_external_verdicts",
            "split_invariant": "200000 rows, train/val/test 20000/20000/160000, per-split label balance preserved",
            "replacement_rule": "confirmed bad rows become exclude_and_replace with the original locked label; direct relabel and self-fill are forbidden.",
        },
        "next_action": (
            "Wait for independent content/external verdicts on the Val 86 package. If actionable bad rows appear, "
            "run Loop152 -> Loop76 -> candidate pool -> corrected split -> replacement/cache/metadata audits before any Val-first training."
        ),
    },
    {
        "id": "loop153_current_best_val_noise_focus",
        "title": "Loop151 current-best Val 73 噪声复核包",
        "priority": "P0_data_governance",
        "status": "awaiting_independent_val_verdicts",
        "remove_from_pending": False,
        "evidence": [
            "docs/phase3_loop153_current_best_val_noise_focus_report.md",
            "scripts/build_loop153_current_best_val_noise_focus.py",
            "tests/test_build_loop153_current_best_val_noise_focus.py",
            "reports/phase3_loop153/loop153_loop151_val_noise_focus_summary.json",
            "reports/phase3_loop153/loop153_loop151_val_noise_focus_blinded.csv",
            "reports/phase3_loop153/loop153_loop151_val_noise_focus_private_map.csv",
            "reports/phase3_loop153/loop153_loop151_val_noise_focus_preflight.json",
            "reports/phase3_loop153/loop153_loop151_val_noise_focus_redraw_readiness_summary.json",
        ],
        "redraw_review": {
            "scope": "Loop151 current strict-best Val errors only; full-test rows are not model-selection evidence.",
            "current_counts": "Loop151 Val errors 162, focus rows 73, FP/FN 52/21, annotated_rows 0, replacement_required 0",
            "source_alignment": "Loop136 Val neighbor/content evidence filtered 179 -> 162 rows with no missing current-error SHA rows.",
            "decision": "await_external_verdicts; train_val/test10k/full_test all false until independent content/external verdicts exist.",
            "replacement_rule": "confirmed bad rows become exclude_and_replace with the original locked label; direct relabel and bad-row self-fill are forbidden.",
        },
        "next_action": (
            "Use Loop153, not the older Loop150/Loop136 Val package, as the current-best reviewer queue. "
            "After independent verdicts, rerun Loop152-style readiness in the Loop153 output directory before any redraw or Val-first training."
        ),
    },
    {
        "id": "loop156_current_best_val_all_error_review",
        "title": "Loop151 current-best Val 162 全错误盲化复核包",
        "priority": "P0_data_governance",
        "status": "awaiting_independent_val_verdicts",
        "remove_from_pending": False,
        "evidence": [
            "docs/phase3_loop156_current_best_val_full_error_review_report.md",
            "scripts/build_loop156_current_best_val_full_error_review.py",
            "tests/test_build_loop156_current_best_val_full_error_review.py",
            "reports/phase3_loop156/loop156_loop151_val_all_errors_summary.json",
            "reports/phase3_loop156/loop156_loop151_val_all_errors_blinded.csv",
            "reports/phase3_loop156/loop156_loop151_val_all_errors_private_map.csv",
            "reports/phase3_loop156/loop156_loop151_val_all_errors_preflight.json",
            "reports/phase3_loop156/loop156_loop151_val_all_errors_redraw_readiness_summary.json",
        ],
        "redraw_review": {
            "scope": "All 162 Loop151 current strict-best Val errors; full-test rows are not included.",
            "current_counts": "review_rows 162, FP/FN 105/57, annotated_rows 0, replacement_required 0",
            "coverage": "Loop153 high-conflict focus covers 73/162; Loop156 expands reviewer coverage to all current-best Val errors.",
            "decision": "await_external_verdicts; fresh_redraw/train_val/test10k/full_test all false until independent content/external verdicts exist.",
            "replacement_rule": "confirmed bad rows become exclude_and_replace with the original locked label; direct relabel and bad-row self-fill are forbidden.",
        },
        "next_action": (
            "Use Loop156 when full Val error-surface coverage is needed. Once independent verdicts exist, rerun "
            "preflight and readiness before any same-original-label redraw or Val-first retraining."
        ),
    },
    {
        "id": "loop157_external_annotation_package",
        "title": "Loop156 Val 162 外部标注安全包",
        "priority": "P0_data_governance",
        "status": "ready_for_external_content_annotation",
        "remove_from_pending": False,
        "evidence": [
            "docs/phase3_loop157_external_annotation_package_report.md",
            "scripts/export_loop157_current_best_val_external_annotation_package.py",
            "tests/test_export_loop157_current_best_val_external_annotation_package.py",
            "reports/phase3_loop157/loop157_loop151_val_all_errors_external_package_summary.json",
            "reports/phase3_loop157/loop157_loop151_val_all_errors_external_context.csv",
            "reports/phase3_loop157/loop157_loop151_val_all_errors_annotation_template.csv",
            "reports/phase3_loop157/loop157_loop151_val_all_errors_reviewer_guide.json",
        ],
        "external_review": {
            "scope": "All 162 Loop151 current-best Val errors from Loop156.",
            "package_state": "rows 162, context_field_count 33, blockers [], context_value_violation_count 0",
            "label_error_counts": "label 0/1 = 105/57; error FP/FN = 105/57",
            "authorization": "ready for independent content/external annotation only; automatic verdict/relabel/replacement/train/test all false.",
        },
        "next_action": (
            "Fill the Loop157 annotation template with independent content/external verdicts only, then run Loop158 "
            "strict import/preflight. Do not add identity/model fields to the returned file."
        ),
    },
    {
        "id": "loop158_loop157_external_annotation_ingress",
        "title": "Loop157 外部标注返回导入闸门",
        "priority": "P0_data_governance",
        "status": "ready_noop_no_external_annotations",
        "remove_from_pending": False,
        "evidence": [
            "docs/phase3_loop158_loop157_external_annotation_ingress_report.md",
            "scripts/import_loop158_current_best_val_external_annotations.py",
            "tests/test_import_loop158_current_best_val_external_annotations.py",
            "reports/phase3_loop158/loop158_loop157_external_annotation_import_summary.json",
            "reports/phase3_loop158/loop158_loop157_external_annotation_import_summary.md",
        ],
        "external_review": {
            "scope": "Loop157 returned annotations for all 162 Loop151 current-best Val errors.",
            "current_counts": "context rows 162, returned annotation rows 0, annotated rows 0, blockers []",
            "accepted_schema": "exact four columns only: review_focus_id/manual_label_verdict/manual_verdict_note/recommended_action",
            "authorization": "private map join, redraw readiness, replacement, training, Test-10k, and full-test all remain false in the header-only no-op state.",
        },
        "next_action": (
            "When real independent verdict rows are returned, run Loop158 first. It will block identity/model columns "
            "and identity/model note terms before Loop126 preflight or any private-map join."
        ),
    },
    {
        "id": "loop154_trusted_signer_threshold_t0995",
        "title": "Loop151 trusted signer score threshold 0.995 收紧复验",
        "priority": "negative_record",
        "status": "experimentally_not_useful_retain_record",
        "remove_from_pending": False,
        "evidence": [
            "docs/phase3_loop154_trusted_signer_threshold_tightening_report.md",
            "scripts/evaluate_authenticode_trusted_signer_guard.py",
            "reports/phase3_loop154/loop154_trusted_signer_guard_t0995_val_eval.json",
            "reports/phase3_loop154/loop154_trusted_signer_guard_t0995_test10k_eval.json",
            "reports/phase3_loop154/loop154_trusted_signer_guard_t0995_full_eval.json",
        ],
        "failure_review": {
            "failure_observation": "Threshold 0.995 produced identical Val, Test-10k, and full-test predictions to Loop151 threshold 1.0.",
            "inferred_cause": "All frozen trusted-signer downgrades already have stage2 scores <= 0.995 in the evaluated splits.",
            "evidence_strength": "强证据",
            "do_not_repeat_until": "A new score source or a new externally approved signer term list changes the action set.",
        },
        "next_action": "Keep Loop151 as the canonical current best; do not spend more cycles sweeping signer score thresholds around 1.0.",
    },
    {
        "id": "loop155_candidate_governance_audit",
        "title": "Loop151 邻近候选漏斗治理审计",
        "priority": "P0_eval_governance",
        "status": "completed_governance_record",
        "remove_from_pending": False,
        "evidence": [
            "docs/phase3_loop155_candidate_governance_audit.md",
            "scripts/build_loop155_candidate_governance_audit.py",
            "tests/test_build_loop155_candidate_governance_audit.py",
            "reports/phase3_loop155/loop155_candidate_governance_audit.json",
            "reports/phase3_loop155/loop155_candidate_governance_audit.md",
        ],
        "governance_review": {
            "current_best": "Loop151 remains deployable strict best: Val 162, Test-10k 78, full-test 1466 errors.",
            "test10k_reject": "Loop144 union + signer improves Val to 150 errors but regresses Test-10k to 81 errors.",
            "full_test_mirage": "OOF-noise/R5 + signer has full-test 1460 errors but fails Val with 173 errors, so it is not selectable.",
            "equivalent_negative": "Loop154 threshold 0.995 is metric-equivalent to Loop151 on all evaluated splits.",
        },
        "next_action": (
            "Do not replace Loop151 using full-test-only evidence. Use Loop155 as the guardrail when reviewing "
            "nearby candidates and continue with Val-first orthogonal evidence or Loop153 noise verdicts."
        ),
    },
    {
        "id": "loop159_r11_only_recall_candidate",
        "title": "Loop159 R11-only + trusted signer 高召回候选",
        "priority": "negative_record",
        "status": "high_recall_tradeoff_not_strict_best",
        "remove_from_pending": False,
        "evidence": [
            "docs/phase3_loop159_r11_only_candidate_audit.md",
            "scripts/build_loop159_r11_only_candidate_audit.py",
            "tests/test_build_loop159_r11_only_candidate_audit.py",
            "reports/phase3_loop159/loop159_r11_only_candidate_audit.json",
            "reports/phase3_loop159/loop159_r11_only_trusted_signer_val_eval.json",
            "reports/phase3_loop159/loop159_r11_only_trusted_signer_test10k_eval.json",
            "reports/phase3_loop151/loop151_trusted_signer_guard_on_r11_filtered_full_eval.json",
        ],
        "governance_review": {
            "val": "F1 0.9922720247, errors 155, FP/FN 106/49; errors -7 vs Loop151",
            "test10k": "F1 0.9921968788, errors 78, FP/FN 52/26; total errors equal to Loop151",
            "full_test": "F1 0.9907064008, errors 1491, FP/FN 962/529; errors +25 vs Loop151",
            "tradeoff": "Full-test FN improves by 58 but FP increases by 83, so strict F1/total-error regresses.",
        },
        "next_action": (
            "Do not replace Loop151 under the current F1/total-error objective. Keep Loop159 only as a high-recall "
            "trade-off candidate if the product later accepts substantially higher FP for fewer FN."
        ),
    },
    {
        "id": "loop160_lowprob_r11_gate",
        "title": "Loop160 low-probability R11 rescue gate",
        "priority": "negative_record",
        "status": "failed_full_test_after_val_test10k_pass",
        "remove_from_pending": False,
        "evidence": [
            "docs/phase3_loop160_lowprob_r11_gate.md",
            "scripts/build_loop160_lowprob_r11_gate.py",
            "tests/test_build_loop160_lowprob_r11_gate.py",
            "reports/phase3_loop160/loop160_lowprob_r11_gate_audit.json",
            "reports/phase3_loop160/loop160_lowprob_r11_val_predictions.csv",
            "reports/phase3_loop160/loop160_lowprob_r11_test10k_predictions.csv",
            "reports/phase3_loop160/loop160_lowprob_r11_full_predictions.csv",
        ],
        "governance_review": {
            "selection": "Val-only smallest baseline_prob_malicious threshold <= 0.2487261742 with min error improvement 3 and no FP increase.",
            "val": "accepted 3, correct/wrong 3/0, errors 159, FP/FN 105/54, delta errors -3",
            "test10k": "accepted 1, correct/wrong 1/0, errors 77, FP/FN 49/28, delta errors -1",
            "full_test": "accepted 41, correct/wrong 18/23, errors 1471, FP/FN 902/569, delta errors +5",
            "failure_mode": "Test-10k one-row gain did not generalize; full-test FP spillover exceeded FN rescue.",
        },
        "next_action": (
            "Do not use probability-threshold-only R11 rescue as strict best. Future recall rescue needs stronger "
            "content/external evidence before entering Test-10k."
        ),
    },
    {
        "id": "loop161_test10k_promotion_margin_guard",
        "title": "Loop161 Test-10k 晋级边际闸门",
        "priority": "P0_eval_governance",
        "status": "active_promotion_guard",
        "remove_from_pending": False,
        "evidence": [
            "docs/phase3_loop161_test10k_promotion_margin_guard.md",
            "scripts/build_loop161_test10k_promotion_margin_guard.py",
            "tests/test_build_loop161_test10k_promotion_margin_guard.py",
            "reports/phase3_loop161/loop161_test10k_promotion_margin_guard.json",
            "reports/phase3_loop161/loop161_test10k_promotion_margin_guard.md",
        ],
        "governance_review": {
            "rule": "Val improvement >= 3 errors and Test-10k improvement >= 3 errors before full-test promotion.",
            "accepted": "Loop151 trusted signer guard: Val -17, Test-10k -5.",
            "rejected": "Loop144 + signer: Test-10k +3; Loop159: Test-10k 0; Loop160: Test-10k -1.",
            "purpose": "Prevent one-row Test-10k gains from consuming full-test or becoming full-test-tuned candidates.",
        },
        "next_action": (
            "Apply Loop161 before future full-test runs. Candidates with marginal Test-10k changes should remain "
            "Val/Test-10k records unless a new independent evidence source widens the margin."
        ),
    },
    {
        "id": "loop162_loop160_failure_posthoc",
        "title": "Loop162 Loop160 full-test failure posthoc",
        "priority": "P0_error_analysis",
        "status": "posthoc_failure_record_only",
        "remove_from_pending": False,
        "evidence": [
            "docs/phase3_loop162_loop160_failure_posthoc.md",
            "scripts/build_loop162_loop160_failure_posthoc.py",
            "tests/test_build_loop162_loop160_failure_posthoc.py",
            "reports/phase3_loop162/loop162_loop160_failure_posthoc.json",
            "reports/phase3_loop162/loop162_loop160_failure_posthoc_public.csv",
            "reports/phase3_loop162/loop162_loop160_failure_posthoc_private_map.csv",
        ],
        "failure_review": {
            "val": "accepted 3, correct/wrong 3/0",
            "test10k": "accepted 1, correct/wrong 1/0",
            "full_test": "accepted 41, correct/wrong 18/23",
            "interpretation": "Val/Test-10k accepted rows were too sparse; full-test wrong accepted rows outnumbered correct accepted rows.",
            "selection_policy": "Posthoc only; not model, threshold, signer-term, GA mask, replacement, or production-rule evidence.",
        },
        "next_action": (
            "Do not repeat probability-only R11 rescue. Continue only with independent Val-side content/external "
            "evidence that can separate true FN rescue from FP spillover."
        ),
    },
    {
        "id": "loop163_r11_rescue_support_audit",
        "title": "Loop163 R11 rescue 支撑度熔断",
        "priority": "P0_eval_governance",
        "status": "reject_low_support_no_selector_training",
        "remove_from_pending": False,
        "evidence": [
            "docs/phase3_loop163_r11_rescue_support_audit.md",
            "scripts/build_loop163_r11_rescue_support_audit.py",
            "tests/test_build_loop163_r11_rescue_support_audit.py",
            "reports/phase3_loop163/loop163_r11_rescue_support_audit.json",
            "reports/phase3_loop163/loop163_r11_rescue_support_public.csv",
            "reports/phase3_loop163/loop163_r11_rescue_support_private_map.csv",
        ],
        "governance_review": {
            "thresholds": "Val disagreements >= 30, Val fixes >= 10, Val breaks <= 0 required before selector training.",
            "val": "rows 20000, disagreements 9, fixes/breaks 8/1",
            "test10k": "rows 10000, disagreements 6, fixes/breaks 3/3",
            "full_test_posthoc": "rows 160000, disagreements 141, fixes/breaks 58/83",
            "decision_reason": "Val support is too small and already has a break row; probability-only/R11-only selector search is likely overfit.",
        },
        "next_action": (
            "Stop R11-only/probability-only selector searches. Resume recall rescue only with new independent Val-side "
            "content or external evidence."
        ),
    },
    {
        "id": "unified_model_review_gate",
        "title": "统一模型评审闸门",
        "priority": "P0",
        "status": "completed_removed_from_pending",
        "remove_from_pending": True,
        "evidence": [
            "scripts/build_model_review_report.py",
            "tests/test_build_model_review_report.py",
            "reports/model_review/final_model_selection/model_review_report.json",
            "reports/model_review/final_model_selection/model_review_summary.md",
        ],
        "next_action": "Keep as reusable gate; do not re-add to pending advice.",
    },
    {
        "id": "fixed_v2_20w_uncompressed_cache",
        "title": "fixed-v2 20w 未压缩 cache 覆盖审计",
        "priority": "P0",
        "status": "completed_removed_from_pending",
        "remove_from_pending": True,
        "evidence": [
            "scripts/recover_missing_feature_cache.py",
            "scripts/audit_split_cache_coverage.py",
            "scripts/evaluate_split_from_cache.py",
            "tests/test_recover_missing_feature_cache.py",
            "tests/test_audit_split_cache_coverage.py",
            "tests/test_evaluate_split_from_cache.py",
            "reports/random_20w_split/random_20w_8192_uncompressed_cache_rebuild_full_current_split.json",
            "reports/random_20w_split/random_20w_8192_uncompressed_cache_coverage_audit.json",
            "reports/random_20w_split/random_20w_8192_uncompressed_missing_cache.csv",
            "reports/random_20w_split/random_20w_8192_uncompressed_test10k_after_lookup_opt_eval.json",
            "reports/random_20w_split/random_20w_8192_uncompressed_test10k_after_lookup_opt_missing_cache.csv",
            "data/.cache/manifest_38672ba0.json",
        ],
        "validation_review": {
            "split": "reports/random_20w_split/random_20w_split.csv",
            "checkpoint": "models/random_20w_8192/best_model.pt",
            "manifest": "data/.cache/manifest_38672ba0.json",
            "cache_coverage": "199870/200000 = 99.935%",
            "test_probe": "10k raw test rows, 9994 predicted, 6 missing cache",
            "threshold_050": "F1 0.9302, AUC 0.9782, FP 270, FN 415",
            "target_result": "F1 >= 99.9% not achieved on current evidence.",
        },
        "next_action": (
            "Treat cache coverage as resolved for the random-20w fixed-v2 8192 run. "
            "The 10k test probe is far below F1 99.9%; improve model quality before another expensive full test."
        ),
    },
    {
        "id": "probability_calibration",
        "title": "阈值和概率校准正式产品化",
        "priority": "P1",
        "status": "completed_removed_from_pending",
        "remove_from_pending": True,
        "evidence": [
            "reports/model_review/final_model_selection/ab_strict_reverification_report.json",
            "reports/model_review/final_model_selection/ab_strict_reverification_report.md",
            "reports/hard_family_finetune/clean_hyperparam_search/train_calibrator_no_metadata_test_confirmation_scripted.json",
            "reports/model_review/final_model_selection/probability_calibrator_test_strict_full.json",
            "reports/model_review/final_model_selection/probability_calibrator_hard_fn_holdout_strict_full.json",
            "reports/model_review/final_model_selection/probability_calibrator_hard_error_holdout_strict_full.json",
            "reports/model_review/final_model_selection/probability_calibrator_high_value_benign_strict_full.json",
            "reports/model_review/final_model_selection/high_value_benign_baseline_analysis/prediction_error_summary.json",
            "reports/model_review/final_model_selection/high_value_benign_ga_mask_analysis/prediction_error_summary.json",
            "reports/model_review/final_model_selection/cache_coverage_audit.json",
        ],
        "next_action": "Keep as reusable completed record; do not re-add to pending advice.",
    },
    {
        "id": "rl_mainline",
        "title": "RL 主线扩大",
        "priority": "negative_record",
        "status": "experimentally_not_useful_retain_record",
        "remove_from_pending": False,
        "evidence": ["reports/pro_runs/fixed_pe256_2k_summary.json"],
        "failure_review": {
            "failure_observation": "Three-seed RL average F1 was 0.8427 versus CE baseline 0.8910.",
            "inferred_cause": "The current bandit reward does not provide a more stable learning signal than cross-entropy and mainly shifts FP/FN trade-offs.",
            "evidence_strength": "强证据",
            "do_not_repeat_until": "A new reward design beats CE in a same-protocol multi-seed small experiment.",
        },
        "next_action": "Do not scale without a new reward design and same-protocol evidence.",
    },
    {
        "id": "swa_ema_all_combined",
        "title": "SWA / EMA / all combined",
        "priority": "negative_record",
        "status": "experimentally_not_useful_retain_record",
        "remove_from_pending": False,
        "evidence": [
            "models/comparison_experiments_from_cache/results.json",
            "reports/model_review/final_model_selection/training_trick_summary.json",
        ],
        "failure_review": {
            "failure_observation": "20k cache-sample F1 dropped from baseline 0.9287 to SWA 0.8882, EMA 0.9132, and all-combined 0.8549.",
            "inferred_cause": "The current training tricks disturb the already useful convergence path; combining many tricks makes the regression uninterpretable.",
            "evidence_strength": "中到强证据",
            "do_not_repeat_until": "A single-variable multi-seed diagnostic improves F1 without increasing FP/FN costs.",
        },
        "next_action": "Do not prioritize near-term; keep the negative record visible.",
    },
    {
        "id": "ga_feature_mask",
        "title": "GA 特征掩码",
        "priority": "P1",
        "status": "useful_high_security_candidate_not_default",
        "remove_from_pending": False,
        "evidence": [
            "reports/model_review/final_model_selection/ab_strict_reverification_report.json",
            "reports/model_review/final_model_selection/ab_strict_reverification_report.md",
            "reports/feature_mask_eval_all20000_thresholds.json",
            "reports/feature_mask_eval_by_source_dir.json",
            "reports/model_review/final_model_selection/ga_feature_mask_full_holdout_summary.json",
            "reports/model_review/final_model_selection/high_value_benign_manifest.csv",
            "reports/model_review/final_model_selection/high_value_benign_manifest_summary.json",
            "reports/model_review/final_model_selection/high_value_benign_baseline_analysis/prediction_error_summary.json",
            "reports/model_review/final_model_selection/high_value_benign_ga_mask_analysis/prediction_error_summary.json",
            "reports/model_review/final_model_selection/cache_coverage_audit.json",
        ],
        "tradeoff_review": {
            "observation": "20k errors improved 1340→1210 and FN 958→670, but FP increased 382→540 and high-value benign FP increased 604→638.",
            "inferred_cause": "The selected feature subset favors recall and removes or downweights signals that help protect benign files.",
            "evidence_strength": "强证据",
            "default_decision": "High-security optional mode only, not default.",
        },
        "next_action": "Keep as a high-security optional mode only; do not enable by default.",
    },
    {
        "id": "byte_noise_near_threshold",
        "title": "byte noise / near-threshold weighting",
        "priority": "negative_record",
        "status": "experimentally_not_useful_retain_record",
        "remove_from_pending": False,
        "evidence": [
            "models/generalization_group_isolated/results.json",
            "reports/model_review/final_model_selection/training_trick_summary.json",
            "models/generalization_group_isolated_seed_confirm/seed_plan.json",
            "models/generalization_group_isolated_seed_confirm/results.json",
            "models/generalization_group_isolated_seed_confirm/summary.md",
        ],
        "failure_review": {
            "failure_observation": "Multi-seed test F1 mean dropped from baseline 0.9444 to byte-noise 0.9260 and near-threshold 0.9200; FN mean increased.",
            "inferred_cause": "The apparent single-seed gain was seed-sensitive; perturbing bytes or overweighting boundary samples made the decision boundary less stable.",
            "evidence_strength": "强证据",
            "do_not_repeat_until": "A constrained variant shows multi-seed FN mean no higher than baseline.",
        },
        "next_action": (
            "Do not enable by default. Multi-seed cache-covered group-isolated confirmation "
            "showed lower mean test F1 and higher mean FN than baseline."
        ),
    },
    {
        "id": "gated_residual_fusion",
        "title": "gated / residual fusion",
        "priority": "negative_record",
        "status": "experimentally_not_useful_retain_record",
        "remove_from_pending": False,
        "evidence": [
            "reports/hard_family_finetune/gated_full_threshold_sweep.json",
            "reports/hard_family_finetune/residual_stat_gate_full_threshold_sweep.json",
            "reports/hard_family_finetune/clean_hyperparam_search/f1_probe_residual_channel_gate_val_sweep.json",
        ],
        "failure_review": {
            "failure_observation": "Gated and residual-stat full sweeps underperformed the concat candidate; residual-channel probe did not beat known stronger baselines.",
            "inferred_cause": "The current gate/residual designs did not learn a reliable per-sample branch trust policy and added optimization complexity.",
            "evidence_strength": "强证据",
            "do_not_repeat_until": "A new constrained gate design first shows interpretable gate behavior on validation errors.",
        },
        "next_action": "Do not repeat the same gate designs unless a new constrained design is proposed.",
    },
    {
        "id": "hard_example_replay",
        "title": "hard-example replay",
        "priority": "P1",
        "status": "experimentally_not_useful_retain_record",
        "remove_from_pending": False,
        "evidence": [
            "reports/hard_family_finetune/hard_error_finetune_threshold055/hard_error_finetuned_full_threshold_sweep.json",
            "reports/hard_family_finetune/model_selection_final/final_model_selection_report.json",
            "reports/hard_family_finetune/balanced_replay_from_current_candidate_threshold063/hard_error_finetune_plan.json",
            "reports/hard_family_finetune/balanced_replay_from_current_candidate_threshold063/README.md",
            "reports/hard_family_finetune/balanced_replay_from_current_candidate_threshold063/hard_error_finetune_split.csv",
            "reports/hard_family_finetune/balanced_replay_strict_source_group_threshold063/hard_error_finetune_plan.json",
            "reports/hard_family_finetune/balanced_replay_strict_source_group_threshold063/hard_error_holdout_predictions.csv",
            "reports/hard_family_finetune/balanced_replay_strict_source_group_threshold063/baseline_holdout_predictions.csv",
            "models/balanced_replay_strict_source_group_threshold063/best_model.pt",
        ],
        "blocker": "Strict-source-group replayed model did not achieve balanced improvement: FN holdout regressed from 39→15 correct at threshold 0.63. Net gain only +4/423 on hard-error holdout. Trade-off direction depends on threshold but no single threshold simultaneously improves both FP and FN holdout.",
        "failure_review": {
            "failure_observation": "At threshold 0.63, FN holdout correct fell 39→15 while FP holdout correct rose 41→79; total gain was only +4/423.",
            "inferred_cause": "The recipe shifts model aggressiveness and overfits hard examples instead of learning a stable FP/FN boundary.",
            "evidence_strength": "强证据",
            "do_not_repeat_until": "A fundamentally different replay protocol uses strict source-group isolation, clean replay, and dual FP/FN holdouts.",
        },
        "next_action": "Keep as negative record. Do not repeat the same 4-epoch 1e-5 LR recipe unless a fundamentally different approach is proposed.",
    },
    {
        "id": "speakeasyx_dynamic_features",
        "title": "SpeakeasyX 动态行为特征",
        "priority": "P2",
        "status": "second_stage_only_not_mainline",
        "remove_from_pending": False,
        "evidence": ["reports/hard_family_finetune/clean_hyperparam_search/speakeasy_feasibility_report.md"],
        "tradeoff_review": {
            "observation": "Fixed timeout filter reduced selected test FP 122→0 but increased FN 120→168.",
            "inferred_cause": "Timeout/unsupported execution states are useful false-positive signals but are not benign-only; some malicious samples also timeout.",
            "evidence_strength": "中到强证据",
            "default_decision": "Second-stage triage signal only, not an automatic mainline downgrade.",
        },
        "next_action": "Only test conservative FP triage rules; do not merge into main classifier directly.",
    },
]


def _exists(path: str, root: Path) -> bool:
    return (root / path).exists()


def build_status(root: Path) -> dict:
    rows = []
    for row in RECOMMENDATIONS:
        evidence = row.get("evidence", [])
        rows.append(
            {
                **row,
                "evidence_status": {
                    path: "present" if _exists(path, root) else "missing"
                    for path in evidence
                },
            }
        )
    return {
        "schema": "axon_ml_recommendation_status_v1",
        "recommendations": rows,
        "summary": {
            "completed_removed": sum(1 for row in rows if row["status"] == "completed_removed_from_pending"),
            "negative_retained": sum("not_useful" in row["status"] for row in rows),
            "open": sum(
                1
                for row in rows
                if not row.get("remove_from_pending", False)
                and row["status"] not in {
                    "completed_removed_from_pending",
                    "experimentally_not_useful_retain_record",
                }
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ML recommendation status ledger JSON.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_status(args.root.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
