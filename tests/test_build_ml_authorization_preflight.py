import csv
import json
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_ml_authorization_preflight import (  # noqa: E402
    ALLOWED_IDENTITY_USES,
    EXPECTED_LABEL_SPLIT_COUNTS,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TOTAL_ROWS,
    IDENTITY_FIELDS,
    build_preflight,
)


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_csv(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_path", "label"])
        writer.writeheader()
        for index in range(rows):
            writer.writerow({"source_path": f"sample-{index}", "label": "0"})


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _completed_plan() -> dict:
    return {
        "authorization_packages": [
            {
                "id": "A_probability_calibration_strict_productization",
                "recommendation_id": "probability_calibration",
                "heavy_authorization_required": False,
                "completed": True,
                "acceptance_criteria": ["completed"],
            },
            {
                "id": "B_ga_feature_mask_full_hard_whitelist_validation",
                "recommendation_id": "ga_feature_mask",
                "heavy_authorization_required": False,
                "completed": True,
                "acceptance_criteria": ["completed"],
            },
            {
                "id": "C_byte_noise_near_threshold_multiseed",
                "recommendation_id": "byte_noise_near_threshold",
                "heavy_authorization_required": False,
                "completed": True,
                "acceptance_criteria": ["completed negative record"],
            },
        ]
    }


def _plan_with_open_heavy_packages() -> dict:
    payload = _completed_plan()
    payload["authorization_packages"].extend(
        [
            {
                "id": "D_hard_example_balanced_replay",
                "recommendation_id": "hard_example_replay",
                "heavy_authorization_required": True,
                "completed": False,
                "bounded_recovery_inputs": {},
                "acceptance_criteria": ["strict validation"],
            },
            {
                "id": "E_speakeasyx_conservative_second_stage_probe",
                "recommendation_id": "speakeasyx_dynamic_features",
                "heavy_authorization_required": True,
                "completed": False,
                "bounded_recovery_inputs": {},
                "acceptance_criteria": ["val first"],
            },
        ]
    )
    return payload


def _cache_ready_payload(**overrides) -> dict:
    payload = {
        "schema": "axon_corrected_split_cache_ready_v1",
        "split_summary": {
            "rows": EXPECTED_TOTAL_ROWS,
            "split_counts": EXPECTED_SPLIT_COUNTS,
            "label_split_counts": EXPECTED_LABEL_SPLIT_COUNTS,
        },
        "total_rows": EXPECTED_TOTAL_ROWS,
        "covered_rows": EXPECTED_TOTAL_ROWS,
        "missing_rows": 0,
        "cache_metadata_validation_enabled": True,
        "metadata_checked_rows": EXPECTED_TOTAL_ROWS,
        "metadata_failure_rows": 0,
        "label_balance_enforced": True,
        "cache_ready": True,
    }
    payload.update(overrides)
    return payload


def _current_state_payload(**cache_overrides) -> dict:
    cache_section = {
        "total_rows": EXPECTED_TOTAL_ROWS,
        "covered_rows": EXPECTED_TOTAL_ROWS,
        "missing_rows": 0,
        "label_balance_enforced": True,
        "cache_metadata_validation_enabled": True,
        "metadata_checked_rows": EXPECTED_TOTAL_ROWS,
        "metadata_failure_rows": 0,
        "sampled_rows": 2000,
        "sample_failed_rows": 0,
    }
    cache_section.update(cache_overrides)
    return {
        "schema": "axon_loop79_current_state_gate_v1",
        "decision": "pass",
        "sections": {
            "fixed_v2_replacement_130": {
                "replacement_rows": 130,
                "self_replacements": 0,
                "selection_status_counts": {"strict_extracted": 130},
            },
            "current_split_cache": cache_section,
        },
    }


def _route_audit_payload(
    *,
    decision: str = "await_independent_blinded_verdicts",
    actionable_rows: int = 0,
    replacement_required_rows: int = 0,
    training_policy_rows: int = 0,
    training_allowed_now: bool = False,
    test10k_allowed_now: bool = False,
    full_test_allowed_now: bool = False,
    ready_for_redraw_preflight: bool = False,
) -> dict:
    return {
        "schema": "axon_loop98_identity_safe_route_audit_v1",
        "decision": decision,
        "blockers": [],
        "route_sections": {
            "full_queue_review": {
                "evidence": {
                    "actionable_rows": actionable_rows,
                    "replacement_required_rows": replacement_required_rows,
                    "training_policy_rows": training_policy_rows,
                }
            }
        },
        "identity_feature_policy": {
            "forbidden_as_model_or_verdict_evidence": IDENTITY_FIELDS,
            "allowed_uses": ALLOWED_IDENTITY_USES,
        },
        "decisions": {
            "training_allowed_now": training_allowed_now,
            "test10k_allowed_now": test10k_allowed_now,
            "full_test_allowed_now": full_test_allowed_now,
            "ready_for_redraw_preflight": ready_for_redraw_preflight,
            "next_allowed_step": "read-only review only",
        },
    }


def _write_gate_inputs(
    tmp_path: Path,
    *,
    route: dict | None = None,
    current_state: dict | None = None,
    cache_ready: dict | None = None,
) -> tuple[Path, Path, Path]:
    route_path = tmp_path / "route.json"
    current_path = tmp_path / "current.json"
    cache_path = tmp_path / "cache.json"
    _write_json(route_path, route if route is not None else _route_audit_payload())
    _write_json(current_path, current_state if current_state is not None else _current_state_payload())
    _write_json(cache_path, cache_ready if cache_ready is not None else _cache_ready_payload())
    return route_path, current_path, cache_path


def test_build_ml_authorization_preflight_checks_bounded_inputs_and_counts_rows():
    with _case_dir("ml_authorization_preflight") as tmp_path:
        csv_path = tmp_path / "missing.csv"
        _write_csv(csv_path, rows=2)
        plan_path = tmp_path / "plan.json"
        status_path = tmp_path / "status.json"
        plan_path.write_text(
            json.dumps(
                {
                    "authorization_packages": [
                        {
                            "id": "A_probability_calibration_strict_productization",
                            "recommendation_id": "probability_calibration",
                            "heavy_authorization_required": False,
                            "completed": True,
                            "acceptance_criteria": ["completed"],
                        },
                        {
                            "id": "B_ga_feature_mask_full_hard_whitelist_validation",
                            "recommendation_id": "ga_feature_mask",
                            "heavy_authorization_required": True,
                            "bounded_recovery_inputs": {
                                "present_csv": str(csv_path),
                                "missing_csv": str(tmp_path / "missing-input.csv"),
                            },
                            "acceptance_criteria": ["strict eval"],
                        },
                        {"id": "C_not_checked"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        status_path.write_text(json.dumps({"summary": {"open": 5}}), encoding="utf-8")

        preflight = build_preflight(tmp_path, plan_path, status_path)

    completed_check = preflight["package_checks"][0]
    check = preflight["package_checks"][1]
    assert completed_check["completed"] is True
    assert completed_check["ready_for_authorization"] is False
    assert preflight["all_bounded_inputs_present"] is False
    assert check["bounded_recovery_inputs"]["present_csv"]["rows"] == 2
    assert check["missing_inputs"] == ["missing_csv"]
    assert check["heavy_authorization_required"] is True


def test_build_ml_authorization_preflight_treats_only_completed_packages_as_not_blocked():
    with _case_dir("ml_authorization_preflight_completed_only") as tmp_path:
        plan_path = tmp_path / "plan.json"
        status_path = tmp_path / "status.json"
        plan_path.write_text(
            json.dumps(
                {
                    "authorization_packages": [
                        {
                            "id": "A_probability_calibration_strict_productization",
                            "recommendation_id": "probability_calibration",
                            "heavy_authorization_required": False,
                            "completed": True,
                            "acceptance_criteria": ["completed"],
                        },
                        {
                            "id": "B_ga_feature_mask_full_hard_whitelist_validation",
                            "recommendation_id": "ga_feature_mask",
                            "heavy_authorization_required": False,
                            "completed": True,
                            "acceptance_criteria": ["completed"],
                        },
                        {
                            "id": "C_byte_noise_near_threshold_multiseed",
                            "recommendation_id": "byte_noise_near_threshold",
                            "heavy_authorization_required": False,
                            "completed": True,
                            "acceptance_criteria": ["completed negative record"],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        status_path.write_text(json.dumps({"summary": {"open": 4}}), encoding="utf-8")

        preflight = build_preflight(tmp_path, plan_path, status_path)

    assert {check["id"] for check in preflight["package_checks"]} == {
        "A_probability_calibration_strict_productization",
        "B_ga_feature_mask_full_hard_whitelist_validation",
        "C_byte_noise_near_threshold_multiseed",
    }
    assert all(check["completed"] for check in preflight["package_checks"])
    assert preflight["all_bounded_inputs_present"] is True


def test_build_ml_authorization_preflight_blocks_operations_when_route_awaits_verdicts():
    with _case_dir("ml_authorization_preflight_route_blocks") as tmp_path:
        plan_path = tmp_path / "plan.json"
        status_path = tmp_path / "status.json"
        _write_json(plan_path, _completed_plan())
        _write_json(status_path, {"summary": {"open": 0}})
        route_path, current_path, cache_path = _write_gate_inputs(tmp_path)

        preflight = build_preflight(
            tmp_path,
            plan_path,
            status_path,
            route_audit_path=route_path,
            current_state_gate_path=current_path,
            cache_ready_path=cache_path,
        )

    auth = preflight["operation_authorization"]
    assert "authorization audit summaries" in preflight["evidence_semantics"]
    assert "not model features" in preflight["evidence_semantics"]
    assert auth["decisions"]["read_only_review_allowed"] is True
    assert auth["decisions"]["package_completion_grants_operations"] is False
    assert auth["decisions"]["train_val_allowed"] is False
    assert auth["decisions"]["threshold_sweep_allowed"] is False
    assert auth["decisions"]["test10k_allowed"] is False
    assert auth["decisions"]["full_test_allowed"] is False
    assert "route_audit_awaits_independent_blinded_verdicts" in preflight["route_gate"]["blockers"]
    assert "no_actionable_independent_verdicts" in preflight["route_gate"]["blockers"]
    assert "route_audit_training_allowed_now_false" in auth["operation_blockers"]["train_val"]
    assert "route_audit_test10k_allowed_now_false" in auth["operation_blockers"]["test10k"]


def test_build_ml_authorization_preflight_requires_metadata_ready_cache_even_if_route_allows():
    with _case_dir("ml_authorization_preflight_metadata_blocks") as tmp_path:
        plan_path = tmp_path / "plan.json"
        status_path = tmp_path / "status.json"
        _write_json(plan_path, _completed_plan())
        _write_json(status_path, {"summary": {"open": 0}})
        route_path, current_path, cache_path = _write_gate_inputs(
            tmp_path,
            route=_route_audit_payload(
                decision="ready_for_non_destructive_redraw_preflight",
                actionable_rows=2,
                replacement_required_rows=2,
                training_allowed_now=True,
                test10k_allowed_now=True,
                full_test_allowed_now=True,
                ready_for_redraw_preflight=True,
            ),
            current_state=_current_state_payload(metadata_failure_rows=1),
            cache_ready=_cache_ready_payload(metadata_failure_rows=1),
        )

        preflight = build_preflight(
            tmp_path,
            plan_path,
            status_path,
            route_audit_path=route_path,
            current_state_gate_path=current_path,
            cache_ready_path=cache_path,
        )

    auth = preflight["operation_authorization"]
    assert preflight["cache_ready_gate"]["status"] == "block"
    assert preflight["current_state_gate"]["status"] == "block"
    assert "cache_ready_metadata_failures_present" in auth["shared_blockers"]
    assert "current_state_metadata_failures_present" in auth["shared_blockers"]
    assert auth["decisions"]["train_val_allowed"] is False
    assert auth["decisions"]["test10k_allowed"] is False
    assert auth["decisions"]["full_test_allowed"] is False


def test_build_ml_authorization_preflight_does_not_turn_redraw_preflight_into_training():
    with _case_dir("ml_authorization_preflight_redraw_is_not_training") as tmp_path:
        plan_path = tmp_path / "plan.json"
        status_path = tmp_path / "status.json"
        _write_json(plan_path, _completed_plan())
        _write_json(status_path, {"summary": {"open": 0}})
        route_path, current_path, cache_path = _write_gate_inputs(
            tmp_path,
            route=_route_audit_payload(
                decision="ready_for_non_destructive_redraw_preflight",
                actionable_rows=2,
                replacement_required_rows=2,
                training_allowed_now=True,
                test10k_allowed_now=True,
                full_test_allowed_now=True,
                ready_for_redraw_preflight=True,
            ),
        )

        preflight = build_preflight(
            tmp_path,
            plan_path,
            status_path,
            route_audit_path=route_path,
            current_state_gate_path=current_path,
            cache_ready_path=cache_path,
        )

    auth = preflight["operation_authorization"]
    assert auth["decisions"]["redraw_preflight_allowed"] is True
    assert auth["decisions"]["train_val_allowed"] is False
    assert auth["decisions"]["test10k_allowed"] is False
    assert auth["decisions"]["full_test_allowed"] is False
    assert "route_audit_only_allows_redraw_preflight" in auth["operation_blockers"]["train_val"]


def test_build_ml_authorization_preflight_enforces_identity_policy_fields():
    with _case_dir("ml_authorization_preflight_identity_policy") as tmp_path:
        plan_path = tmp_path / "plan.json"
        status_path = tmp_path / "status.json"
        _write_json(plan_path, _completed_plan())
        _write_json(status_path, {"summary": {"open": 0}})
        route = _route_audit_payload(
            decision="ready_for_non_destructive_redraw_preflight",
            actionable_rows=1,
            replacement_required_rows=1,
            training_allowed_now=True,
        )
        route["identity_feature_policy"]["forbidden_as_model_or_verdict_evidence"] = ["filename"]
        route_path, current_path, cache_path = _write_gate_inputs(tmp_path, route=route)

        preflight = build_preflight(
            tmp_path,
            plan_path,
            status_path,
            route_audit_path=route_path,
            current_state_gate_path=current_path,
            cache_ready_path=cache_path,
        )

    route_gate = preflight["route_gate"]
    assert "identity_policy_missing_required_forbidden_fields" in route_gate["blockers"]
    missing = route_gate["evidence"]["missing_forbidden_identity_fields"]
    assert "path" in missing
    assert "source_sha256" in missing
    assert preflight["operation_authorization"]["decisions"]["train_val_allowed"] is False


def test_build_ml_authorization_preflight_includes_open_heavy_packages_without_authorizing():
    with _case_dir("ml_authorization_preflight_package_scope") as tmp_path:
        plan_path = tmp_path / "plan.json"
        status_path = tmp_path / "status.json"
        _write_json(plan_path, _plan_with_open_heavy_packages())
        _write_json(status_path, {"summary": {"open": 2}})
        route_path, current_path, cache_path = _write_gate_inputs(tmp_path)

        preflight = build_preflight(
            tmp_path,
            plan_path,
            status_path,
            route_audit_path=route_path,
            current_state_gate_path=current_path,
            cache_ready_path=cache_path,
        )

    scope = preflight["package_scope_audit"]
    assert set(scope["included_package_ids"]) == {
        "A_probability_calibration_strict_productization",
        "B_ga_feature_mask_full_hard_whitelist_validation",
        "C_byte_noise_near_threshold_multiseed",
        "D_hard_example_balanced_replay",
        "E_speakeasyx_conservative_second_stage_probe",
    }
    assert scope["ignored_package_ids"] == []
    assert scope["open_heavy_packages"] == [
        "D_hard_example_balanced_replay",
        "E_speakeasyx_conservative_second_stage_probe",
    ]
    assert preflight["ml_gate_result"]["passed"] is False
    assert preflight["operation_authorization"]["decisions"]["package_completion_grants_operations"] is False
