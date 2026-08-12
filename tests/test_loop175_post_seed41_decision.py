from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.decide_loop175_post_seed41_branch import build_decision


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def inputs(
    tmp_path: Path,
    *,
    model_passed: bool,
    resource_passed: bool,
    causal_passed: bool = True,
    a_errors: int = 200,
    c_errors: int = 150,
) -> tuple[Path, Path, Path, Path]:
    decision = "seed41_pass_allow_seed42_43" if model_passed else "closed_seed41_gate"
    primary = [
        {"name": "C_net_advantage_over_D", "passed": causal_passed},
        {"name": "C_D_component_bootstrap_one_sided_95_lcb", "passed": causal_passed},
    ]
    final = tmp_path / "final.json"
    write_json(final, {
        "schema": "axon_loop175_seed41_oof_receipt_v1",
        "loop_id": "Loop175",
        "seed": 41,
        "decision": decision,
        "evaluation": {
            "arm_metrics": {"A": {"errors": a_errors}, "C": {"errors": c_errors}},
            "gates": {
                "primary_C": primary,
                "runtime": [{"name": "seed_wall_seconds", "passed": True}],
            },
        },
        "val_rows_opened": 0,
        "test10k_rows_opened": 0,
        "full_test_rows_opened": 0,
        "val_test_or_full_rows_opened": 0,
    })
    audit = tmp_path / "audit.json"
    write_json(audit, {
        "schema": "axon_loop175_seed41_interruption_adjusted_audit_v1",
        "evidence": {"final_receipt": {"sha256": hashlib.sha256(final.read_bytes()).hexdigest()}},
        "model_gate": {"decision": decision, "passed": model_passed},
        "resource_gate": {"passed": resource_passed},
    })
    predecision = tmp_path / "predecision.json"
    write_json(predecision, {
        "schema": "axon_loop175_post_seed41_architecture_predecision_v1",
        "receipt_branches": {
            "engineering_or_resource_failure": "resource",
            "seed41_and_three_seed_feasibility_pass": "seeds",
            "seed41_pass_but_three_seed_20pct_impossible": "close",
            "scientific_failure_region_causal": "hgconv",
            "scientific_failure_region_rejected": "new-evidence",
        },
    })
    proposal = tmp_path / "proposal.json"
    write_json(proposal, {
        "three_seed_gate": {"minimum_relative_error_reduction_vs_A_each_seed": 0.2},
    })
    return final, audit, predecision, proposal


def decide(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    return build_decision(*inputs(tmp_path, **kwargs))


def test_pass_and_feasible_allows_new_seed_authorization(tmp_path: Path) -> None:
    result = decide(tmp_path, model_passed=True, resource_passed=True)
    assert result["decision"] == "allow_new_authorization_and_source_closure_for_seed42_43_only"
    assert result["seed42_43_execution_authorized"] is False
    assert result["new_authorization_and_source_closure_permitted"] is True


def test_adjusted_resource_failure_blocks_seed_authorization(tmp_path: Path) -> None:
    result = decide(tmp_path, model_passed=True, resource_passed=False)
    assert result["decision"] == "close_loop175_allow_loop179_resource_proposal_only"
    assert result["new_authorization_and_source_closure_permitted"] is False


def test_causal_scientific_failure_routes_to_hgconv(tmp_path: Path) -> None:
    result = decide(tmp_path, model_passed=False, resource_passed=True, causal_passed=True)
    assert result["decision"] == "close_loop175_allow_loop179_hgconv_region_proposal_only"


def test_rejected_region_causality_closes_lineage(tmp_path: Path) -> None:
    result = decide(tmp_path, model_passed=False, resource_passed=True, causal_passed=False)
    assert result["decision"] == "close_region_section_lineage_require_new_isolated_evidence_proposal"


def test_sub_twenty_percent_pass_closes_without_more_seeds(tmp_path: Path) -> None:
    result = decide(
        tmp_path,
        model_passed=True,
        resource_passed=True,
        a_errors=200,
        c_errors=170,
    )
    assert result["decision"] == "close_loop175_without_seed42_43"
