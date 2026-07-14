from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from audit_loop165_loop69_loop164_surrogate_complementarity import (  # noqa: E402
    CANONICAL_SHA256,
    DEFAULT_LOOP69_PREDICTIONS,
    SurrogateAuditError,
    _bind_raw,
    audit,
)


def test_canonical_surrogate_gate_closes_loop164_complementarity_lineage():
    payload = audit()

    assert payload["alignment_receipt"]["common_sha256_rows"] == 19996
    assert payload["alignment_receipt"]["loop69_only_rows"] == 4
    assert payload["alignment_receipt"]["loop164_only_rows"] == 4
    assert [
        row["sample_index"] for row in payload["alignment_receipt"]["same_index_replacements"]
    ] == [1, 2, 3, 4]
    assert payload["coverage"]["supported_common_rows"] == 19540
    assert payload["coverage"]["missing_common_rows"] == 456
    assert payload["coverage"]["loop164_total_missing_rows"] == 460
    assert payload["overlap"]["base_error_repairs"] == 75
    assert payload["overlap"]["base_correct_breaks"] == 595
    assert payload["overlap"]["both_wrong"] == 153
    assert payload["overlap"]["supported_disagreements"] == 670
    assert payload["overlap"]["blind_switch_precision"] == 75 / 670
    assert payload["overlap"]["net_error_reduction"] == -520
    assert payload["partition_compatibility"] == {
        "loop69_partition": "random_stratified_five_fold_seed_69_one_based",
        "loop164_partition": "content_component_five_fold_seed_164_zero_based",
        "common_rows_with_same_numeric_fold_after_normalization": 4076,
        "common_rows_with_different_fold_after_normalization": 15920,
        "non_singleton_loop164_components_on_common_rows": 393,
        "non_singleton_components_crossing_loop69_folds": 356,
        "rows_in_crossing_components": 3368,
        "shared_outer_partition": False,
    }
    assert [
        (
            payload["overlap"]["by_loop164_diagnostic_fold"][str(fold)]["repairs"],
            payload["overlap"]["by_loop164_diagnostic_fold"][str(fold)]["breaks"],
        )
        for fold in range(5)
    ] == [(13, 130), (11, 101), (15, 116), (13, 84), (23, 164)]
    assert payload["metrics"]["loop69_common_denominator"]["errors"] == 252
    assert (
        payload["metrics"][
            "unattainable_oracle_choose_correct_expert_retain_base_on_missing"
        ]["errors"]
        == 177
    )
    assert payload["surrogate_cost_fuse"]["checks"] == {
        "supported_disagreements": True,
        "base_error_repairs": True,
        "blind_switch_precision": False,
        "net_error_reduction": False,
    }
    assert payload["surrogate_cost_fuse"]["passed"] is False
    assert payload["loop151_exact_oof"]["authorized_for_loop164"] is False
    assert payload["formal_loop151_complementarity_gate"] == {
        "status": "not_run",
        "decision": "blocked_wrong_base_lineage_and_fold_scope",
        "blockers": [
            "loop69_is_loop61_style_not_decision_aligned_loop151",
            "loop69_random_folds_do_not_match_loop164_content_component_folds",
            "four_current_snapshot_rows_lack_loop69_baseline",
            "loop69_report_lacks_complete_recipe_input_and_output_sha_provenance",
        ],
    }
    assert payload["current_loop164_recipe"]["formal_lineage_closed"] is False
    assert payload["target_status"]["promotion_evidence"] is False
    assert payload["decision"].startswith("park_current_loop164_recipe_surrogate_negative")


def test_sha_binding_fails_closed_on_tampering():
    raw = DEFAULT_LOOP69_PREDICTIONS.read_bytes()

    try:
        _bind_raw(raw + b"\n", CANONICAL_SHA256["loop69_predictions"], "tampered fixture")
    except SurrogateAuditError as exc:
        assert "SHA-256 drifted" in str(exc)
    else:
        raise AssertionError("Tampered input must fail its canonical SHA-256 binding")


def test_decision_manifest_binds_final_loop165_artifacts():
    manifest_path = (
        PROJECT_ROOT
        / "manifests"
        / "roadmap_9997"
        / "loop165_surrogate_complementarity"
        / "decision.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    for binding in payload["artifacts"].values():
        raw = (PROJECT_ROOT / binding["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == binding["sha256"]
    assert payload["formal_loop151_gate"]["status"] == "not_run"
    assert payload["surrogate_result"]["cost_fuse_passed"] is False
    assert payload["target_achieved"] is False
