#!/usr/bin/env python3
"""Validate the aggregate-only Loop164 fold scope plan before A2 training.

This A1 utility checks only proposal, contract, isolation receipt, and a
custodian-produced aggregate scope plan. It never opens JSONL inventory rows,
raw files, caches, prediction rows, checkpoints, or model payloads.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from validate_loop164_nested_oof_execution_receipt import (  # noqa: E402
    ISOLATION_AUTHORIZATION_PROVENANCE_SCHEMA,
    ISOLATION_CONTRACT_SCHEMA,
    ISOLATION_METADATA_AUTHORITY_SCOPE,
    ISOLATION_RECEIPT_SCHEMA,
    LOOP_ID,
    REQUIRED_EMBARGO_SECONDS,
    REQUIRED_FOLDS,
    REQUIRED_FUSION_FIELDS,
    SCOPE_PLAN_SCHEMA,
    _validate_contract,
    _validate_isolation_receipt,
    _validate_scope_plan,
    read_json_object,
    resolve_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SCHEMA = "axon_loop164_fold_scope_plan_validation_v1"
DEFAULT_PROPOSAL = (
    PROJECT_ROOT / "manifests/roadmap_9997/loop164_whole_file_residual_expert/proposal.json"
)
DEFAULT_CONTRACT = PROJECT_ROOT / "reports/roadmap_9997/loop164/full_pool_group_manifest.json"
DEFAULT_ISOLATION_RECEIPT = (
    PROJECT_ROOT / "reports/roadmap_9997/loop164/full_pool_isolation_validation.json"
)
DEFAULT_SCOPE_PLAN = PROJECT_ROOT / "reports/roadmap_9997/loop164/fold_scope_plan.json"

__all__ = (
    "ISOLATION_AUTHORIZATION_PROVENANCE_SCHEMA",
    "ISOLATION_CONTRACT_SCHEMA",
    "ISOLATION_METADATA_AUTHORITY_SCOPE",
    "ISOLATION_RECEIPT_SCHEMA",
    "LOOP_ID",
    "REQUIRED_EMBARGO_SECONDS",
    "REQUIRED_FOLDS",
    "REQUIRED_FUSION_FIELDS",
    "SCOPE_PLAN_SCHEMA",
    "validate_loop164_fold_scope_plan",
)


def _empty_result() -> dict[str, Any]:
    return {
        "schema": VALIDATION_SCHEMA,
        "loop_id": LOOP_ID,
        "aggregate_only_verified": False,
        "proposal_binding_verified": False,
        "contract_binding_verified": False,
        "isolation_receipt_binding_verified": False,
        "scope_plan_binding_verified": False,
        "binding_fingerprints": {},
        "plan_summary": {
            "outer_fold_count": 0,
            "inner_fold_count_per_outer_fold": 0,
            "eligible_rows": None,
            "warmup_rows": None,
            "fold_assignment_fingerprint": None,
        },
        "blockers": [],
        "ready_for": {
            "fold_scope_frozen": False,
            "a2_training_authorization": False,
            "train_oof": False,
            "val_a": False,
            "test10k": False,
            "full_test": False,
        },
        "decision": "block",
        "notes": [
            "This validates aggregate scope commitments only and does not authorize training.",
            "A separate A2 scope-plan authority and later A2 training authority are required.",
        ],
    }


def _read(path: Path, *, label: str, failures: Counter[str]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    try:
        return read_json_object(path)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        failures[f"{label}_unreadable"] += 1
        return None, None


def validate_loop164_fold_scope_plan(
    *,
    proposal_json: Path = DEFAULT_PROPOSAL,
    contract_json: Path = DEFAULT_CONTRACT,
    isolation_receipt_json: Path = DEFAULT_ISOLATION_RECEIPT,
    scope_plan_json: Path = DEFAULT_SCOPE_PLAN,
) -> dict[str, Any]:
    """Fail closed when the future scope plan cannot be proven frozen and isolated."""

    result = _empty_result()
    failures: Counter[str] = Counter()
    try:
        proposal_path = resolve_path(proposal_json)
        contract_path = resolve_path(contract_json)
        isolation_receipt_path = resolve_path(isolation_receipt_json)
        scope_plan_path = resolve_path(scope_plan_json)
    except ValueError:
        result["blockers"] = ["scope_plan_path_binding_invalid"]
        return result

    proposal, proposal_sha256 = _read(proposal_path, label="proposal", failures=failures)
    contract, contract_sha256 = _read(contract_path, label="contract", failures=failures)
    isolation_receipt, isolation_sha256 = _read(
        isolation_receipt_path, label="isolation_receipt", failures=failures
    )
    scope_plan, scope_plan_sha256 = _read(scope_plan_path, label="scope_plan", failures=failures)
    result["binding_fingerprints"] = {
        name: value
        for name, value in {
            "proposal_sha256": proposal_sha256,
            "contract_sha256": contract_sha256,
            "isolation_receipt_sha256": isolation_sha256,
            "scope_plan_sha256": scope_plan_sha256,
        }.items()
        if value is not None
    }

    if proposal is not None:
        if proposal.get("loop_id") != LOOP_ID:
            failures["proposal_loop_id_invalid"] += 1
        if proposal.get("decision") != "propose_loop164_whole_file_residual_expert_no_execution":
            failures["proposal_decision_invalid"] += 1
        result["proposal_binding_verified"] = not any(
            code.startswith("proposal_") for code in failures
        )
    if contract is not None:
        _validate_contract(contract, failures)
        result["contract_binding_verified"] = not any(
            code.startswith("contract_") for code in failures
        )
    fold_fingerprint, eligible_rows, warmup_rows = _validate_isolation_receipt(
        isolation_receipt,
        contract_sha256=contract_sha256,
        failures=failures,
    )
    result["isolation_receipt_binding_verified"] = not any(
        code.startswith("isolation_receipt_") for code in failures
    )
    scopes = _validate_scope_plan(
        scope_plan,
        contract_sha256=contract_sha256,
        isolation_receipt_sha256=isolation_sha256,
        expected_fingerprint=fold_fingerprint,
        expected_eligible_rows=eligible_rows,
        expected_warmup_rows=warmup_rows,
        failures=failures,
    )
    result["scope_plan_binding_verified"] = not any(
        code.startswith("scope_plan_") for code in failures
    )
    result["plan_summary"] = {
        "outer_fold_count": len(scopes),
        "inner_fold_count_per_outer_fold": len(
            next(iter(scopes.values())).get("inner_scopes", [])
        )
        if scopes
        else 0,
        "eligible_rows": eligible_rows,
        "warmup_rows": warmup_rows,
        "fold_assignment_fingerprint": fold_fingerprint,
    }
    if len(scopes) != len(REQUIRED_FOLDS):
        failures["scope_plan_outer_fold_coverage_invalid"] += 1
    result["aggregate_only_verified"] = not any(
        code.endswith("unexpected_fields") or code.endswith("not_object") for code in failures
    )
    result["blockers"] = sorted(failures)
    result["decision"] = "pass" if not failures else "block"
    result["ready_for"]["fold_scope_frozen"] = not failures
    return result


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("Refusing to overwrite an existing scope-plan validation receipt") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an aggregate-only Loop164 fold scope plan before A2 training."
    )
    parser.add_argument("--proposal-json", type=Path, default=DEFAULT_PROPOSAL)
    parser.add_argument("--contract-json", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--isolation-receipt-json", type=Path, default=DEFAULT_ISOLATION_RECEIPT)
    parser.add_argument("--scope-plan-json", type=Path, default=DEFAULT_SCOPE_PLAN)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = validate_loop164_fold_scope_plan(
        proposal_json=args.proposal_json,
        contract_json=args.contract_json,
        isolation_receipt_json=args.isolation_receipt_json,
        scope_plan_json=args.scope_plan_json,
    )
    try:
        _write_json_exclusive(resolve_path(args.output_json), payload)
    except (OSError, ValueError) as exc:
        print(json.dumps({"decision": "block", "blockers": [str(exc)]}, indent=2))
        return 2
    print(
        json.dumps(
            {"decision": payload["decision"], "blockers": payload["blockers"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
