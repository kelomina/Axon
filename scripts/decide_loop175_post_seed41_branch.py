#!/usr/bin/env python3
"""Select the preregistered Loop175 post-seed41 research branch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

OUTPUT_SCHEMA = "axon_loop175_post_seed41_decision_v1"


class DecisionError(ValueError):
    pass


def load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecisionError(f"cannot read JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise DecisionError(f"JSON root must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate_map(receipt: dict[str, Any], group: str) -> dict[str, dict[str, Any]]:
    try:
        gates = receipt["evaluation"]["gates"][group]
    except (KeyError, TypeError) as error:
        raise DecisionError(f"missing {group} gates") from error
    if not isinstance(gates, list):
        raise DecisionError(f"{group} gates must be a list")
    result: dict[str, dict[str, Any]] = {}
    for gate in gates:
        if not isinstance(gate, dict) or not isinstance(gate.get("name"), str):
            raise DecisionError(f"invalid {group} gate")
        if gate["name"] in result or not isinstance(gate.get("passed"), bool):
            raise DecisionError(f"duplicate or malformed {group} gate: {gate.get('name')}")
        result[gate["name"]] = gate
    return result


def build_decision(
    final_path: Path,
    audit_path: Path,
    predecision_path: Path,
    proposal_path: Path,
) -> dict[str, Any]:
    final = load_object(final_path)
    audit = load_object(audit_path)
    predecision = load_object(predecision_path)
    proposal = load_object(proposal_path)
    if final.get("schema") != "axon_loop175_seed41_oof_receipt_v1":
        raise DecisionError("final receipt schema drifted")
    if audit.get("schema") != "axon_loop175_seed41_interruption_adjusted_audit_v1":
        raise DecisionError("adjusted audit schema drifted")
    if predecision.get("schema") != "axon_loop175_post_seed41_architecture_predecision_v1":
        raise DecisionError("predecision schema drifted")
    if final.get("loop_id") != "Loop175" or final.get("seed") != 41:
        raise DecisionError("final receipt identity drifted")
    try:
        bound_final_sha = audit["evidence"]["final_receipt"]["sha256"]
    except (KeyError, TypeError) as error:
        raise DecisionError("adjusted audit final receipt binding is missing") from error
    if bound_final_sha != sha256_file(final_path):
        raise DecisionError("adjusted audit does not bind the current final receipt")
    if audit.get("model_gate", {}).get("decision") != final.get("decision"):
        raise DecisionError("adjusted audit model decision drifted")
    if any(final.get(field) != 0 for field in (
        "val_rows_opened",
        "test10k_rows_opened",
        "full_test_rows_opened",
        "val_test_or_full_rows_opened",
    )):
        raise DecisionError("held-out access was reported")

    primary = gate_map(final, "primary_C")
    runtime = gate_map(final, "runtime")
    runtime_passed = all(gate["passed"] for gate in runtime.values())
    adjusted_resource_passed = audit.get("resource_gate", {}).get("passed") is True
    model_passed = audit.get("model_gate", {}).get("passed") is True
    if model_passed != (final.get("decision") == "seed41_pass_allow_seed42_43"):
        raise DecisionError("adjusted audit model pass flag drifted")

    try:
        arm_metrics = final["evaluation"]["arm_metrics"]
        baseline_errors = int(arm_metrics["A"]["errors"])
        candidate_errors = int(arm_metrics["C"]["errors"])
    except (KeyError, TypeError, ValueError) as error:
        raise DecisionError("A/C error metrics are missing") from error
    if baseline_errors < 0 or candidate_errors < 0:
        raise DecisionError("A/C error metrics are invalid")
    relative_reduction = (
        (baseline_errors - candidate_errors) / baseline_errors if baseline_errors else 0.0
    )
    try:
        required_relative_reduction = float(
            proposal["three_seed_gate"]["minimum_relative_error_reduction_vs_A_each_seed"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DecisionError("three-seed relative reduction gate is missing") from error

    reasons: list[str] = []
    if not runtime_passed or not adjusted_resource_passed:
        branch = "engineering_or_resource_failure"
        decision = "close_loop175_allow_loop179_resource_proposal_only"
        reasons.append("raw_or_interruption_adjusted_resource_gate_failed")
    elif model_passed:
        if relative_reduction >= required_relative_reduction:
            branch = "seed41_and_three_seed_feasibility_pass"
            decision = "allow_new_authorization_and_source_closure_for_seed42_43_only"
            reasons.append("seed41_and_single_seed_three_seed_feasibility_gates_passed")
        else:
            branch = "seed41_pass_but_three_seed_20pct_impossible"
            decision = "close_loop175_without_seed42_43"
            reasons.append("seed41_relative_error_reduction_below_three_seed_requirement")
    else:
        causal_gate_names = (
            "C_net_advantage_over_D",
            "C_D_component_bootstrap_one_sided_95_lcb",
        )
        missing = [name for name in causal_gate_names if name not in primary]
        if missing:
            raise DecisionError(f"causal gates missing: {','.join(missing)}")
        region_causal = all(primary[name]["passed"] for name in causal_gate_names)
        if region_causal:
            branch = "scientific_failure_region_causal"
            decision = "close_loop175_allow_loop179_hgconv_region_proposal_only"
            reasons.append("region_ownership_causal_but_primary_C_gate_failed")
        else:
            branch = "scientific_failure_region_rejected"
            decision = "close_region_section_lineage_require_new_isolated_evidence_proposal"
            reasons.append("region_ownership_causal_gate_failed")

    expected_branch = predecision.get("receipt_branches", {}).get(branch)
    if not isinstance(expected_branch, str):
        raise DecisionError(f"predecision branch is missing: {branch}")
    return {
        "schema": OUTPUT_SCHEMA,
        "loop_id": "Loop175",
        "seed": 41,
        "claim_scope": "train_only_post_seed41_branch_selection_no_new_execution_or_heldout_access",
        "evidence": {
            "final_receipt": {"path": final_path.as_posix(), "sha256": sha256_file(final_path)},
            "interruption_adjusted_audit": {"path": audit_path.as_posix(), "sha256": sha256_file(audit_path)},
            "predecision": {"path": predecision_path.as_posix(), "sha256": sha256_file(predecision_path)},
            "proposal": {"path": proposal_path.as_posix(), "sha256": sha256_file(proposal_path)},
        },
        "observations": {
            "model_gate_passed": model_passed,
            "raw_runtime_gates_passed": runtime_passed,
            "interruption_adjusted_resource_gate_passed": adjusted_resource_passed,
            "A_errors": baseline_errors,
            "C_errors": candidate_errors,
            "C_relative_error_reduction_vs_A": relative_reduction,
            "three_seed_minimum_relative_error_reduction": required_relative_reduction,
        },
        "selected_branch": branch,
        "predecision_branch_text": expected_branch,
        "reasons": reasons,
        "seed42_43_execution_authorized": False,
        "new_authorization_and_source_closure_permitted": decision == "allow_new_authorization_and_source_closure_for_seed42_43_only",
        "heldout_access": {
            "val_rows_opened": 0,
            "test10k_rows_opened": 0,
            "full_test_rows_opened": 0,
        },
        "decision": decision,
    }


def write_idempotent(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    if path.exists():
        if path.read_bytes() != encoded:
            raise DecisionError(f"existing postdecision differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-receipt", type=Path, required=True)
    parser.add_argument("--adjusted-audit", type=Path, required=True)
    parser.add_argument("--predecision", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--if-ready", action="store_true")
    arguments = parser.parse_args()
    if arguments.if_ready and not (
        arguments.final_receipt.is_file() and arguments.adjusted_audit.is_file()
    ):
        print(json.dumps({"schema": OUTPUT_SCHEMA, "status": "waiting_for_final_evidence"}, sort_keys=True))
        return 0
    decision = build_decision(
        arguments.final_receipt.resolve(),
        arguments.adjusted_audit.resolve(),
        arguments.predecision.resolve(),
        arguments.proposal.resolve(),
    )
    write_idempotent(arguments.output.resolve(), decision)
    print(json.dumps(decision, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
