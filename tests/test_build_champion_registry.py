from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_champion_registry import SOURCE_PATHS, build_registry  # noqa: E402


def write_json(root: Path, relative_path: Path, payload: dict) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def metric(f1: float, errors: int, fp: int, fn: int, rows: int) -> dict:
    return {
        "candidate": {
            "f1": f1,
            "errors": errors,
            "false_positive": fp,
            "false_negative": fn,
            "sample_count": rows,
        }
    }


def create_fixture(root: Path) -> None:
    write_json(
        root,
        SOURCE_PATHS["truth_manifest"],
        {
            "decision": "artifact_freeze_complete_raw_replay_pending",
            "champion_registry": {
                "champion_id": "current_strict_best_loop151",
                "status": "current_strict_best",
            },
            "metrics": {
                "val": metric(0.91, 162, 105, 57, 20000),
                "test10k": metric(0.92, 78, 49, 29, 10000),
                "legacy_full_test": metric(0.93, 1466, 879, 587, 160000),
            },
            "capability_boundary": {
                "classification": "prediction_level_policy_freeze",
                "raw_file_to_report_replay_ready": False,
                "native_loop151_ready": False,
                "connected_system_ready": False,
                "certification_ready": False,
                "blockers": ["Loop151 raw runtime is incomplete"],
            },
        },
    )
    write_json(
        root,
        SOURCE_PATHS["native_smoke_receipt"],
        {
            "decision": "native_loop28_train_smoke_passed_full_loop151_blocked",
            "claim_scope": {"quality_claim_allowed": False},
            "dag": {
                "stages": [
                    {
                        "stage_id": "loop28_native",
                        "raw_executable": True,
                        "artifacts": [
                            {
                                "path": "models/loop28.onnx",
                                "exists": True,
                                "sha256": "a" * 64,
                                "size_bytes": 123,
                            }
                        ],
                    }
                ]
            },
        },
    )
    write_json(
        root,
        SOURCE_PATHS["native_parity_receipt"],
        {
            "decision": "native_loop28_parity_blocked",
            "parity": {"requested_count": 1, "passed_count": 0},
        },
    )
    write_json(
        root,
        SOURCE_PATHS["loop28_pause"],
        {"decision": "pause_native_decode_compat_without_execution_resume_loop151_f1_mainline"},
    )
    write_json(
        root,
        SOURCE_PATHS["recommendation_status"],
        {
            "recommendations": [
                {"id": "current_strict_best_loop151", "status": "current_strict_best"},
                {
                    "id": "loop164_whole_file_residual_expert",
                    "status": "parked_surrogate_negative_formal_gate_not_run",
                    "preflight_review": {
                        "decision": (
                            "park_current_loop164_recipe_surrogate_negative_"
                            "exact_loop151_gate_not_run"
                        )
                    },
                },
                {
                    "id": "loop166_code_section_foundation",
                    "status": "closed_b1_nonfinite_current_recipe_retired",
                    "phase_review": {
                        "decision": (
                            "close_loop166_bpe1024_mlm_recipe_and_preregister_"
                            "loop167_ember_v3_novel_delta_control"
                        )
                    },
                },
                {
                    "id": "loop167_ember_v3_novel_delta",
                    "status": "closed_cache_only_train_oof_negative_do_not_retry",
                    "phase_review": {
                        "decision": (
                            "close_loop167_cache_only_novel_delta_negative_do_not_retry_or_promote"
                        )
                    },
                },
            ]
        },
    )


def test_registry_keeps_research_native_and_connected_scopes_separate(tmp_path: Path):
    create_fixture(tmp_path)

    payload = build_registry(tmp_path, generated_at_utc="2026-07-12T00:00:00Z")

    assert payload["champions"]["research"]["candidate_id"] == "Loop151"
    assert payload["champions"]["research"]["metrics"]["legacy_full_test"]["errors"] == 1466
    assert payload["champions"]["research"]["next_candidate"]["candidate_id"] is None
    assert (
        payload["champions"]["research"]["next_candidate"]["decision"]
        == "close_loop167_cache_only_novel_delta_negative_do_not_retry_or_promote"
    )
    assert (
        payload["champions"]["research"]["recently_closed_candidate"]["candidate_id"]
        == "Loop166"
    )
    assert (
        payload["champions"]["research"]["recently_parked_candidate"]["candidate_id"]
        == "Loop164"
    )
    assert payload["champions"]["native_offline"]["candidate_id"] == "Loop28"
    assert payload["champions"]["native_offline"]["parity_passed"] == 0
    assert payload["champions"]["native_offline"]["quality_claim_allowed"] is False
    assert payload["champions"]["connected_system"]["candidate_id"] is None
    assert payload["invariants"]["native_reference_is_not_research_champion"] is True


def test_registry_rejects_a_native_parity_success_claim_not_in_receipt(tmp_path: Path):
    create_fixture(tmp_path)
    parity_path = tmp_path / SOURCE_PATHS["native_parity_receipt"]
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity["decision"] = "native_loop28_parity_passed"
    parity_path.write_text(json.dumps(parity), encoding="utf-8")

    with pytest.raises(ValueError, match="parity decision mismatch"):
        build_registry(tmp_path, generated_at_utc="2026-07-12T00:00:00Z")


def test_registry_rejects_loop164_as_current_champion(tmp_path: Path):
    create_fixture(tmp_path)
    status_path = tmp_path / SOURCE_PATHS["recommendation_status"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["recommendations"][1]["status"] = "current_strict_best"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(ValueError, match="parked candidate status mismatch"):
        build_registry(tmp_path, generated_at_utc="2026-07-12T00:00:00Z")


def test_registry_rejects_closed_loop166_or_unready_loop167_as_champion(tmp_path: Path):
    create_fixture(tmp_path)
    status_path = tmp_path / SOURCE_PATHS["recommendation_status"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in status["recommendations"]}
    by_id["loop166_code_section_foundation"]["status"] = "current_strict_best"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(ValueError, match="closed candidate status mismatch"):
        build_registry(tmp_path, generated_at_utc="2026-07-12T00:00:00Z")

    create_fixture(tmp_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in status["recommendations"]}
    by_id["loop167_ember_v3_novel_delta"]["status"] = "current_strict_best"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(ValueError, match="next candidate status mismatch"):
        build_registry(tmp_path, generated_at_utc="2026-07-12T00:00:00Z")
