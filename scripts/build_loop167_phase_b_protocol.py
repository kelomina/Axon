#!/usr/bin/env python3
"""Build or verify the immutable static protocol for Loop167 Phase B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "manifests" / "roadmap_9997" / "loop167_ember_v3_novel_delta"
PROTOCOL_PATH = ARTIFACT_ROOT / "phase_b_protocol.json"
PHASE_A_PATHS = {
    "proposal": ARTIFACT_ROOT / "proposal.json",
    "authorization": ARTIFACT_ROOT / "authorization.json",
    "semantic_delta_mapping": ARTIFACT_ROOT / "semantic_delta_mapping.json",
    "frozen_deduplicated_baseline_allowlist": ARTIFACT_ROOT / "frozen_deduplicated_baseline_allowlist.json",
    "source_semantics_addendum": ARTIFACT_ROOT / "phase_a_source_semantics_addendum.json",
    "source_closure": ARTIFACT_ROOT / "phase_a_source_closure.json",
    "static_decision": ARTIFACT_ROOT / "phase_a_static_decision.json",
}


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _phase_a_bindings() -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for name, path in PHASE_A_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing Phase-A binding: {path}")
        bindings[name] = {"path": _relative(path), "sha256": sha256_file(path)}
    return bindings


def build_phase_b_protocol_payload() -> dict[str, Any]:
    return {
        "schema": "axon_loop167_phase_b_protocol_v1",
        "loop_id": "loop167_ember_v3_novel_delta",
        "status": "static_contract_sealed_raw_execution_requires_source_closure_runtime_guard_and_lease",
        "claim_scope": "local_train_only_structural_delta_diagnostic_not_model_quality_promotion_or_full_test",
        "phase_a_bindings": _phase_a_bindings(),
        "input_contract": {
            "folds": {
                "path": "reports/roadmap_9997/loop164/local_train_diagnostic_folds.jsonl",
                "sha256": "00a31a1bd86d7b887447f3e86e5e753ebcaaee45be74311199332e073a3880a5",
                "record_schema": "axon_loop164_local_train_diagnostic_fold_record_v1",
                "split_role": "train",
                "rows": 20000,
                "folds": 5,
                "rows_per_fold": 4000,
                "val_test_or_full_access": False,
            },
            "scope_drift_is_fatal": True,
            "source_sha256_verified_in_same_stream": True,
        },
        "feature_contract": {
            "raw_context": {
                "single_open_per_ordinal": True,
                "single_pe_parse_per_available_ordinal": True,
                "maximum_source_file_bytes": 67108864,
                "reader_chunk_bytes": 1048576,
                "raw_path_or_label_enters_feature_vector": False,
                "persist_raw_bytes_strings_import_tokens_or_disassembly": False,
            },
            "b0": {
                "value_dimension": 571,
                "missing_indicator_dimension": 6,
                "missing_indicator_names": [
                    "missing_fixed_v2",
                    "missing_stat",
                    "missing_content_pe_v1",
                    "missing_content_pe_v2",
                    "missing_content_string",
                    "missing_content_cert",
                ],
                "directory_parse_failure_policy": "retain_compatible_parsed_pe_values_and_audit_reason",
            },
            "b1": {
                "value_dimension": 536,
                "missing_indicator_dimension": 4,
                "missing_indicator_names": [
                    "missing_b1_byte_context",
                    "missing_b1_pe_context",
                    "missing_b1_directory_context",
                    "missing_b1_authenticode",
                ],
                "sampling_indicator_dimension": 3,
                "sampling_indicator_names": [
                    "b1_string_sampled_to_native_cap",
                    "b1_string_candidate_cap_reached",
                    "b1_section_or_overlay_entropy_sampled",
                ],
                "string_head_bytes": 262144,
                "string_tail_bytes": 65536,
                "string_max_candidates": 4096,
            },
            "novel": {
                "value_dimension": 292,
                "complete_mask_required": True,
                "missing_policy": "zero_fill_then_M_and_CF_score_and_decision_bitwise_fallback_to_B0",
            },
            "authenticode": {
                "dimension": 8,
                "unsigned_policy": "complete_zero_vector",
                "signed_cms_without_pinned_parser_policy": "zero_vector_plus_missing_b1_authenticode",
                "public_key_required": False,
                "new_dependency_or_external_tool_allowed": False,
            },
        },
        "fit_contract": {
            "arms": ["B0", "B1", "M", "A", "CF"],
            "outer_folds": [0, 1, 2, 3, 4],
            "replay_seeds": [41, 42, 43],
            "seed_policy": "deterministic_replay_not_independent_robustness_trials",
            "required_replay_matrix_and_prediction_hash_equality": True,
            "maximum_total_fits": 75,
            "estimator": {
                "type": "sklearn.ensemble.HistGradientBoostingClassifier",
                "loss": "log_loss",
                "learning_rate": 0.06,
                "max_iter": 260,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 0.0,
                "max_bins": 255,
                "early_stopping": False,
                "threshold": 0.5,
                "threshold_search_allowed": False,
                "hyperparameter_search_allowed": False,
            },
        },
        "evaluation_contract": {
            "primary_comparator": "global_20000_row_lower_error_of_B0_and_B1_with_B0_winning_ties",
            "per_fold_or_per_row_oracle_selection_allowed": False,
            "m_cf_novel_missing_policy": "copy_same_row_B0_score_and_hard_decision_bitwise",
            "required_rows_per_arm_seed": 20000,
            "silent_drop_allowed": False,
        },
        "resource_contract": {
            "maximum_raw_open_attempts": 20000,
            "maximum_raw_bytes": 26843545600,
            "maximum_feature_cache_bytes": 1073741824,
            "maximum_extraction_peak_rss_bytes": 4294967296,
            "maximum_training_peak_rss_bytes": 8589934592,
            "maximum_extraction_wall_seconds": 6000,
            "maximum_training_wall_seconds": 18000,
            "reserved_seal_evaluation_wall_seconds": 4800,
            "maximum_total_wall_seconds": 28800,
            "worker_count": 1,
            "thread_count": 1,
            "maximum_gpu_allocated_bytes": 0,
            "kill_conditions": [
                "oom",
                "nonfinite",
                "timeout",
                "unrecoverable_feature_cache",
                "second_raw_pass_requested",
                "source_or_scope_drift",
                "replay_hash_mismatch",
            ],
        },
        "runtime_contract": {
            "isolated_python_required": True,
            "required_environment": {
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            "required_packages": ["numpy", "scipy", "scikit-learn", "pefile", "threadpoolctl"],
            "raw_worker_may_not_receive_labels_or_model_scores": True,
            "fit_worker_may_not_receive_raw_root": True,
        },
        "one_shot_lease": {
            "lease_id": "loop167-phase-b-train-oof-v1",
            "marker_path": "reports/roadmap_9997/loop167/phase_b_execution_consumed.json",
            "consume_before_first_raw_open": True,
            "failed_attempt_consumes_lease": True,
            "retry_resume_or_rescan_allowed": False,
        },
        "forbidden": [
            "val_test10k_legacy_full_sentinel_or_sealed_window_access",
            "threshold_search",
            "hyperparameter_search",
            "path_extension_directory_sha_row_fold_or_label_feature",
            "loop151_loop69_loop164_prediction_or_score_input",
            "unlocked_authenticode_parser_or_public_key_requirement",
            "same_lease_retry",
        ],
        "ready_for": {
            "static_source_closure": True,
            "raw_access": False,
            "fit": False,
            "val": False,
            "test10k": False,
            "legacy_full_test": False,
            "promotion": False,
        },
    }


def write_new(path: Path, payload: dict[str, Any]) -> str:
    content = canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(content).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.write) == bool(args.check):
        raise SystemExit("Specify exactly one of --write or --check")
    payload = build_phase_b_protocol_payload()
    expected = canonical_json_bytes(payload)
    if args.write:
        digest = write_new(PROTOCOL_PATH, payload)
    else:
        if not PROTOCOL_PATH.is_file() or PROTOCOL_PATH.read_bytes() != expected:
            raise SystemExit("Phase-B protocol is missing or drifted")
        digest = sha256_file(PROTOCOL_PATH)
    print(json.dumps({"path": _relative(PROTOCOL_PATH), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
