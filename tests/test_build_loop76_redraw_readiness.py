from __future__ import annotations

import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop76_redraw_readiness import build_readiness  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _split_summary() -> dict:
    return {
        "rows": 200000,
        "split_counts": {"test": 160000, "train": 20000, "val": 20000},
        "label_split_counts": {
            "test": {"0": 80000, "1": 80000},
            "train": {"0": 10000, "1": 10000},
            "val": {"0": 10000, "1": 10000},
        },
    }


def _import_payload(*, ready: bool = True, training_policy_rows: int = 0) -> dict:
    return {
        "schema": "axon_loop74_external_verdict_import_v1",
        "decision": "ready_noop_no_actionable_verdicts" if ready else "blocked_invalid_verdicts",
        "import_ready": ready,
        "review_rows": 1868,
        "expected_rows": 1868,
        "invalid_rows": 0 if ready else 1,
        "training_policy_rows": training_policy_rows,
        "blocking_issues": [] if ready else ["invalid_manual_label_verdict"],
        "input_alignment": {
            "sample_index_match_count": 1868 if ready else 1867,
            "missing_split_rows": 0,
            "duplicate_review_rows": 0,
        },
        "split_summary": _split_summary(),
        "target_feasibility": {"confirmed_bad_rows": 0},
        "confirmed_bad_rows": {"total": 0},
        "manual_quality": {},
    }


def _loop87_import_payload(*, rows: int = 1868) -> dict:
    return {
        "schema": "axon_loop87_review_evidence_verdict_import_v1",
        "decision": "ready_noop_no_actionable_verdicts",
        "import_ready": True,
        "rows": rows,
        "expected_rows": 1868,
        "invalid_rows": 0,
        "training_policy_rows": 0,
        "blocking_issues": [],
        "duplicate_sample_index_rows": 0,
        "manual_quality": {
            "blank_verdict_rows": rows,
            "actionable_verdict_missing_note_rows": 0,
            "evidence_note_missing_content_or_external_rows": 0,
            "evidence_note_identity_or_score_only_rows": 0,
        },
    }


def _adjustment_payload(*, replacement_required: int = 0, training_policy_rows: int = 0) -> dict:
    return {
        "schema": "axon_manual_review_adjustment_plan_v1",
        "review_rows": 1868,
        "planned_rows": replacement_required,
        "ignored_rows": 1868 - replacement_required,
        "unknown_verdict_rows": 0,
        "missing_split_rows": 0,
        "duplicate_review_rows": 0,
        "replacement_required": replacement_required,
        "replacement_counts_by_original_label": {"0": replacement_required, "1": 0} if replacement_required else {},
        "training_policy_rows": training_policy_rows,
        "review_rows_in_test_split": 1868,
        "action_counts": {"exclude_and_replace": replacement_required} if replacement_required else {},
        "split_action_counts": {"test:exclude_and_replace": replacement_required} if replacement_required else {},
        "split_summary": _split_summary(),
    }


def _candidate_payload(*, enough: bool = True) -> dict:
    return {
        "schema": "axon_replacement_candidate_pool_v1",
        "rows": 3 if enough else 0,
        "label_counts": {"0": 3} if enough else {},
        "required_replacements": {"0": 2, "1": 0},
        "replacement_shortfall": {} if enough else {"0": 2},
        "content_hash_required_for_strict_redraw": True,
        "source_sha256_origin_counts": {"content_hash": 3} if enough else {},
        "enough_for_required_replacements": enough,
    }


def _corrected_payload() -> dict:
    return {
        "schema": "axon_corrected_split_from_manual_plan_v1",
        "allow_test_replacements": True,
        "excluded_rows": 2,
        "relabeled_rows": 0,
        "corrected_summary": _split_summary(),
        "replacement_summary": {"shortfall": {}, "selected_replacements": 2},
    }


def _replacement_audit_payload(*, ok: bool = True) -> dict:
    return {
        "schema": "axon_corrected_split_replacement_integrity_v1",
        "replacement_integrity_ok": ok,
        "integrity_failures": [] if ok else ["fresh replacement counts do not match replacement requests"],
        "row_count_ok": ok,
        "label_balance_enforced": True,
        "fresh_replacement_rows": 2 if ok else 1,
        "replacement_requests": 2,
        "test_replacement_requests": 2,
        "corrected_summary": _split_summary(),
    }


def _cache_ready_payload(*, ready: bool = True) -> dict:
    return {
        "schema": "axon_corrected_split_cache_ready_v1",
        "cache_ready": ready,
        "total_rows": 200000,
        "covered_rows": 200000 if ready else 199999,
        "missing_rows": 0 if ready else 1,
        "coverage_ratio": 1.0 if ready else 0.999995,
        "shape_failures": [],
        "missing_label_counts": {} if ready else {"0": 1},
        "missing_split_counts": {} if ready else {"test": 1},
        "missing_reason_counts": {} if ready else {"manifest_missing": 1},
        "missing_cache_output": "missing.csv",
        "cache_metadata_validation_enabled": True,
        "metadata_checked_rows": 200000 if ready else 199999,
        "metadata_failure_rows": 0,
        "metadata_issue_counts": {},
        "metadata_issue_output": "metadata_issues.csv",
        "label_balance_enforced": True,
        "label_balance_drift": [],
    }


def _split_metadata_payload(*, ready: bool = True, validate_npz: bool = True, expect_20w: bool = True) -> dict:
    return {
        "schema": "axon_strict_split_metadata_audit_v1",
        "audit_ready": ready,
        "rows": 200000,
        "manifest_samples": 200000,
        "validate_npz": validate_npz,
        "expect_20w": expect_20w,
        "row_issue_count": 0 if ready else 1,
        "metadata_issue_counts": {} if ready else {"source_sha256_mismatch_split_npz": 1},
        "shape_failures": [],
        "match_counts": {"source_sha256": 200000 if ready else 199999},
    }


def _build(
    tmp_path: Path,
    *,
    import_payload: dict,
    adjustment_payload: dict,
    candidate=None,
    corrected=None,
    replacement=None,
    cache=None,
    split_metadata=None,
):
    import_json = _write_json(tmp_path / "import.json", import_payload)
    adjustment_json = _write_json(tmp_path / "adjustment.json", adjustment_payload)
    candidate_json = _write_json(tmp_path / "candidate.json", candidate) if candidate is not None else None
    corrected_json = _write_json(tmp_path / "corrected.json", corrected) if corrected is not None else None
    replacement_json = _write_json(tmp_path / "replacement.json", replacement) if replacement is not None else None
    cache_json = _write_json(tmp_path / "cache.json", cache) if cache is not None else None
    split_metadata_json = _write_json(tmp_path / "split_metadata.json", split_metadata) if split_metadata is not None else None
    return build_readiness(
        strict_import_json=import_json,
        adjustment_plan_json=adjustment_json,
        candidate_pool_json=candidate_json,
        corrected_split_json=corrected_json,
        replacement_audit_json=replacement_json,
        cache_ready_json=cache_json,
        split_metadata_json=split_metadata_json,
        split_csv=tmp_path / "split.csv",
        plan_csv=tmp_path / "plan.csv",
        candidate_csv=tmp_path / "candidate.csv",
        corrected_split_csv=tmp_path / "corrected.csv",
        manifest_json=tmp_path / "manifest.json",
        data_dir=tmp_path / "data",
        output_prefix=tmp_path / "loop76",
    )


def test_noop_import_waits_for_external_verdicts():
    with _case_dir("loop76_noop") as tmp_path:
        payload = _build(tmp_path, import_payload=_import_payload(), adjustment_payload=_adjustment_payload())

    assert payload["decision"] == "await_external_verdicts"
    assert payload["strict_failures"] == []
    assert payload["next_step"] == "no_redraw_required_until_actionable_verdicts"
    assert payload["memory_leak_profile"]["loads_model"] is False
    assert payload["ready_for"]["test10k"] is False


def test_loop87_noop_import_schema_is_accepted_when_adjustment_plan_proves_split_shape():
    with _case_dir("loop76_loop87_noop") as tmp_path:
        payload = _build(
            tmp_path,
            import_payload=_loop87_import_payload(),
            adjustment_payload=_adjustment_payload(),
        )

    assert payload["decision"] == "await_external_verdicts"
    assert payload["strict_import"]["review_rows"] == 1868
    assert payload["strict_import"]["sample_index_match_count"] == 1868
    assert payload["strict_failures"] == []


def test_invalid_import_blocks_before_redraw():
    with _case_dir("loop76_invalid_import") as tmp_path:
        payload = _build(tmp_path, import_payload=_import_payload(ready=False), adjustment_payload=_adjustment_payload())

    assert payload["decision"] == "blocked_before_redraw"
    assert "strict_import_not_ready" in payload["strict_failures"]
    assert payload["next_step"] == "fix_strict_import_or_adjustment_plan"


def test_relabel_action_in_adjustment_plan_blocks_full_error_redraw_path():
    with _case_dir("loop76_blocks_relabel_action") as tmp_path:
        adjustment = _adjustment_payload(replacement_required=0, training_policy_rows=0)
        adjustment["planned_rows"] = 1
        adjustment["action_counts"] = {"relabel": 1}
        adjustment["split_action_counts"] = {"train:relabel": 1}
        payload = _build(tmp_path, import_payload=_import_payload(), adjustment_payload=adjustment)

    assert payload["decision"] == "blocked_before_redraw"
    assert "adjustment_plan_contains_non_replacement_actions" in payload["strict_failures"]
    assert "adjustment_plan_contains_non_replacement_rows" in payload["strict_failures"]


def test_held_out_test_verdict_only_action_blocks_until_import_is_regenerated_as_redraw():
    with _case_dir("loop76_blocks_held_out_test_action") as tmp_path:
        adjustment = _adjustment_payload(replacement_required=0, training_policy_rows=0)
        adjustment["planned_rows"] = 1
        adjustment["action_counts"] = {"held_out_test_verdict_only": 1}
        adjustment["split_action_counts"] = {"test:held_out_test_verdict_only": 1}
        payload = _build(tmp_path, import_payload=_import_payload(), adjustment_payload=adjustment)

    assert payload["decision"] == "blocked_before_redraw"
    assert "adjustment_plan_contains_non_replacement_actions" in payload["strict_failures"]


def test_replacement_plan_requires_candidate_pool_first():
    with _case_dir("loop76_needs_candidates") as tmp_path:
        payload = _build(
            tmp_path,
            import_payload=_import_payload(),
            adjustment_payload=_adjustment_payload(replacement_required=2),
        )

    assert payload["decision"] == "needs_replacement_candidate_pool"
    assert payload["next_step"] == "build_replacement_candidate_pool"
    assert "--required-label0 2" in payload["commands"]["build_replacement_candidate_pool"]
    assert "--enforce-label-balance" in payload["commands"]["audit_replacements"]
    assert "--enforce-label-balance" in payload["commands"]["audit_cache_ready"]
    assert "--metadata-issue-output" in payload["commands"]["audit_cache_ready"]


def test_candidate_shortfall_blocks_redraw():
    with _case_dir("loop76_candidate_shortfall") as tmp_path:
        payload = _build(
            tmp_path,
            import_payload=_import_payload(),
            adjustment_payload=_adjustment_payload(replacement_required=2),
            candidate=_candidate_payload(enough=False),
        )

    assert payload["decision"] == "blocked_candidate_shortfall"
    assert payload["strict_failures"] == ["replacement_candidate_shortfall"]
    assert payload["next_step"] == "collect_more_valid_same_label_candidates"


def test_candidate_pool_without_content_hash_blocks_redraw():
    with _case_dir("loop76_candidate_pool_no_hash") as tmp_path:
        candidate = _candidate_payload(enough=True)
        candidate["content_hash_required_for_strict_redraw"] = False
        payload = _build(
            tmp_path,
            import_payload=_import_payload(),
            adjustment_payload=_adjustment_payload(replacement_required=2),
            candidate=candidate,
        )

    assert payload["decision"] == "blocked_candidate_pool"
    assert payload["strict_failures"] == ["candidate_pool_content_hash_not_required"]


def test_candidate_pool_with_unhashed_rows_blocks_redraw():
    with _case_dir("loop76_candidate_pool_unhashed") as tmp_path:
        candidate = _candidate_payload(enough=True)
        candidate["source_sha256_origin_counts"] = {"content_hash": 2, "missing": 1}
        payload = _build(
            tmp_path,
            import_payload=_import_payload(),
            adjustment_payload=_adjustment_payload(replacement_required=2),
            candidate=candidate,
        )

    assert payload["decision"] == "blocked_candidate_pool"
    assert payload["strict_failures"] == ["candidate_pool_contains_unhashed_rows"]


def test_full_clean_chain_is_ready_for_val_first_reverification():
    with _case_dir("loop76_ready") as tmp_path:
        payload = _build(
            tmp_path,
            import_payload=_import_payload(),
            adjustment_payload=_adjustment_payload(replacement_required=2),
            candidate=_candidate_payload(enough=True),
            corrected=_corrected_payload(),
            replacement=_replacement_audit_payload(ok=True),
            cache=_cache_ready_payload(ready=True),
            split_metadata=_split_metadata_payload(ready=True),
        )

    assert payload["decision"] == "ready_for_val_first_reverification"
    assert payload["strict_failures"] == []
    assert payload["next_step"] == "restart_val_first_funnel"
    assert payload["cache_ready"]["cache_ready"] is True
    assert payload["ready_for"]["train_val_only"] is True
    assert payload["ready_for"]["full_test"] is False


def test_cache_ready_chain_requires_strict_split_metadata_audit_before_training():
    with _case_dir("loop76_needs_split_metadata") as tmp_path:
        payload = _build(
            tmp_path,
            import_payload=_import_payload(),
            adjustment_payload=_adjustment_payload(replacement_required=2),
            candidate=_candidate_payload(enough=True),
            corrected=_corrected_payload(),
            replacement=_replacement_audit_payload(ok=True),
            cache=_cache_ready_payload(ready=True),
        )

    assert payload["decision"] == "needs_strict_split_metadata_audit"
    assert payload["next_step"] == "audit_strict_split_metadata"
    assert "--expect-20w" in payload["commands"]["audit_strict_split_metadata"]
    assert payload["ready_for"]["train_val_only"] is False
    assert payload["ready_for"]["split_metadata_audit"] is True


def test_split_metadata_audit_without_npz_validation_blocks_training():
    with _case_dir("loop76_split_metadata_no_npz") as tmp_path:
        payload = _build(
            tmp_path,
            import_payload=_import_payload(),
            adjustment_payload=_adjustment_payload(replacement_required=2),
            candidate=_candidate_payload(enough=True),
            corrected=_corrected_payload(),
            replacement=_replacement_audit_payload(ok=True),
            cache=_cache_ready_payload(ready=True),
            split_metadata=_split_metadata_payload(ready=True, validate_npz=False),
        )

    assert payload["decision"] == "blocked_split_metadata"
    assert payload["strict_failures"] == ["split_metadata_npz_validation_not_enabled"]


def test_split_metadata_audit_without_20w_shape_check_blocks_training():
    with _case_dir("loop76_split_metadata_no_shape") as tmp_path:
        payload = _build(
            tmp_path,
            import_payload=_import_payload(),
            adjustment_payload=_adjustment_payload(replacement_required=2),
            candidate=_candidate_payload(enough=True),
            corrected=_corrected_payload(),
            replacement=_replacement_audit_payload(ok=True),
            cache=_cache_ready_payload(ready=True),
            split_metadata=_split_metadata_payload(ready=True, expect_20w=False),
        )

    assert payload["decision"] == "blocked_split_metadata"
    assert payload["strict_failures"] == ["split_metadata_20w_shape_not_checked"]


def test_failed_split_metadata_audit_blocks_training():
    with _case_dir("loop76_split_metadata_failed") as tmp_path:
        payload = _build(
            tmp_path,
            import_payload=_import_payload(),
            adjustment_payload=_adjustment_payload(replacement_required=2),
            candidate=_candidate_payload(enough=True),
            corrected=_corrected_payload(),
            replacement=_replacement_audit_payload(ok=True),
            cache=_cache_ready_payload(ready=True),
            split_metadata=_split_metadata_payload(ready=False),
        )

    assert payload["decision"] == "blocked_split_metadata"
    assert payload["strict_failures"] == ["split_metadata_audit_failed"]


def test_cache_miss_requires_recovery_before_eval():
    with _case_dir("loop76_cache_missing") as tmp_path:
        payload = _build(
            tmp_path,
            import_payload=_import_payload(),
            adjustment_payload=_adjustment_payload(replacement_required=2),
            candidate=_candidate_payload(enough=True),
            corrected=_corrected_payload(),
            replacement=_replacement_audit_payload(ok=True),
            cache=_cache_ready_payload(ready=False),
        )

    assert payload["decision"] == "needs_cache_recovery"
    assert payload["next_step"] == "recover_missing_cache_then_rerun_cache_ready"
    assert payload["cache_ready"]["missing_rows"] == 1


def test_replacement_audit_without_label_balance_blocks():
    with _case_dir("loop76_replacement_balance_missing") as tmp_path:
        replacement = _replacement_audit_payload(ok=True)
        replacement["label_balance_enforced"] = False
        payload = _build(
            tmp_path,
            import_payload=_import_payload(),
            adjustment_payload=_adjustment_payload(replacement_required=2),
            candidate=_candidate_payload(enough=True),
            corrected=_corrected_payload(),
            replacement=replacement,
        )

    assert payload["decision"] == "blocked_replacement_integrity"
    assert payload["strict_failures"] == ["replacement_integrity_label_balance_not_enforced"]


def test_cache_ready_without_label_balance_blocks():
    with _case_dir("loop76_cache_balance_missing") as tmp_path:
        cache = _cache_ready_payload(ready=True)
        cache["label_balance_enforced"] = False
        payload = _build(
            tmp_path,
            import_payload=_import_payload(),
            adjustment_payload=_adjustment_payload(replacement_required=2),
            candidate=_candidate_payload(enough=True),
            corrected=_corrected_payload(),
            replacement=_replacement_audit_payload(ok=True),
            cache=cache,
        )

    assert payload["decision"] == "blocked_cache_readiness"
    assert payload["strict_failures"] == ["cache_ready_label_balance_not_enforced"]


def test_cache_ready_without_metadata_validation_blocks():
    with _case_dir("loop76_cache_metadata_disabled") as tmp_path:
        cache = _cache_ready_payload(ready=True)
        cache["cache_metadata_validation_enabled"] = False
        payload = _build(
            tmp_path,
            import_payload=_import_payload(),
            adjustment_payload=_adjustment_payload(replacement_required=2),
            candidate=_candidate_payload(enough=True),
            corrected=_corrected_payload(),
            replacement=_replacement_audit_payload(ok=True),
            cache=cache,
        )

    assert payload["decision"] == "blocked_cache_readiness"
    assert payload["strict_failures"] == ["cache_metadata_validation_not_enabled"]


def test_cache_metadata_failures_block_redraw_not_cache_recovery():
    with _case_dir("loop76_cache_metadata_failures") as tmp_path:
        cache = _cache_ready_payload(ready=False)
        cache["covered_rows"] = 200000
        cache["missing_rows"] = 0
        cache["coverage_ratio"] = 1.0
        cache["missing_label_counts"] = {}
        cache["missing_split_counts"] = {}
        cache["missing_reason_counts"] = {}
        cache["metadata_failure_rows"] = 2
        cache["metadata_issue_counts"] = {"label_mismatch": 1, "shape_mismatch": 1}
        payload = _build(
            tmp_path,
            import_payload=_import_payload(),
            adjustment_payload=_adjustment_payload(replacement_required=2),
            candidate=_candidate_payload(enough=True),
            corrected=_corrected_payload(),
            replacement=_replacement_audit_payload(ok=True),
            cache=cache,
        )

    assert payload["decision"] == "blocked_cache_readiness"
    assert payload["strict_failures"] == ["cache_metadata_failures_present"]
    assert payload["next_step"] == "quarantine_bad_cache_rows_then_redraw_same_label"
