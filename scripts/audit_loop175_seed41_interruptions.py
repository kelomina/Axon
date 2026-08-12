#!/usr/bin/env python3
"""Audit Loop175 seed-41 wall time with external interruption charges."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

FINAL_SCHEMA = "axon_loop175_seed41_oof_receipt_v1"
INTERRUPTION_SCHEMA = "axon_loop175_external_interruption_attestation_v1"
OUTPUT_SCHEMA = "axon_loop175_seed41_interruption_adjusted_audit_v1"


class AuditError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise AuditError(f"JSON root must be an object: {path}")
    return payload


def finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise AuditError(f"{field} must be finite and non-negative")
    return normalized


def runtime_gate(receipt: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        gates = receipt["evaluation"]["gates"]["runtime"]
    except (KeyError, TypeError) as error:
        raise AuditError("final receipt runtime gates are missing") from error
    matches = [gate for gate in gates if isinstance(gate, dict) and gate.get("name") == name]
    if len(matches) != 1:
        raise AuditError(f"expected exactly one runtime gate named {name}")
    return matches[0]


def build_audit(final_path: Path, protocol_path: Path, interruptions_directory: Path) -> dict[str, Any]:
    final_receipt = load_object(final_path)
    protocol = load_object(protocol_path)
    if final_receipt.get("schema") != FINAL_SCHEMA or final_receipt.get("loop_id") != "Loop175":
        raise AuditError("final receipt identity drifted")
    if final_receipt.get("seed") != 41:
        raise AuditError("final receipt seed drifted")
    if final_receipt.get("claim_scope") != "train_only_outer_oof_not_val_test10k_or_full_test":
        raise AuditError("final receipt claim scope drifted")
    if any(final_receipt.get(field) != 0 for field in (
        "val_rows_opened",
        "test10k_rows_opened",
        "full_test_rows_opened",
        "val_test_or_full_rows_opened",
    )):
        raise AuditError("final receipt reports held-out access")
    if protocol.get("loop_id") != "Loop175":
        raise AuditError("protocol identity drifted")
    expected_protocol = sha256_file(protocol_path)
    if final_receipt.get("protocol_commitment") != expected_protocol:
        raise AuditError("final receipt protocol commitment drifted")

    maximum_seed_wall = finite_number(
        protocol.get("resource_contract", {}).get("maximum_seed41_wall_seconds"),
        field="maximum_seed41_wall_seconds",
    )
    controller_gate = runtime_gate(final_receipt, "seed_wall_seconds")
    controller_wall = finite_number(controller_gate.get("observed"), field="controller_seed_wall_seconds")

    interruption_paths = sorted(interruptions_directory.glob("*.json"))
    if not interruption_paths:
        raise AuditError("no interruption attestations found")
    charges: list[dict[str, Any]] = []
    for path in interruption_paths:
        attestation = load_object(path)
        if attestation.get("schema") != INTERRUPTION_SCHEMA:
            raise AuditError(f"interruption schema drifted: {path}")
        if attestation.get("loop_id") != "Loop175" or attestation.get("seed") != 41:
            raise AuditError(f"interruption identity drifted: {path}")
        accounting = attestation.get("wall_accounting")
        if not isinstance(accounting, dict) or accounting.get("require_addition_to_final_seed_wall_gate") is not True:
            raise AuditError(f"interruption accounting contract drifted: {path}")
        charged_seconds = finite_number(
            accounting.get("conservative_charged_seconds"),
            field=f"{path.name}.conservative_charged_seconds",
        )
        charges.append({
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "charged_seconds": charged_seconds,
            "cause": attestation.get("cause"),
            "blocks_seed42_43_authorization": attestation.get("blocks_seed42_43_authorization") is True,
        })

    total_charge = sum(float(item["charged_seconds"]) for item in charges)
    adjusted_wall = controller_wall + total_charge
    numeric_resource_passed = adjusted_wall <= maximum_seed_wall
    provenance_blockers = [
        str(item["path"])
        for item in charges
        if item["blocks_seed42_43_authorization"] is True
    ]
    resource_passed = numeric_resource_passed and not provenance_blockers
    evaluation = final_receipt.get("evaluation")
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("gates"), dict):
        raise AuditError("final receipt evaluation gate structure is missing")
    model_decision = final_receipt.get("decision")
    if evaluation.get("decision") != model_decision:
        raise AuditError("top-level and evaluation decisions disagree")
    evaluation_all_passed = evaluation["gates"].get("all_passed")
    if not isinstance(evaluation_all_passed, bool):
        raise AuditError("evaluation all_passed flag is missing")
    model_passed = model_decision == "seed41_pass_allow_seed42_43"
    if model_passed != evaluation_all_passed:
        raise AuditError("model decision and gate aggregate disagree")
    if model_passed and resource_passed:
        decision = "seed41_pass_allow_seed42_43_with_interruption_adjusted_resource_contract"
    elif not model_passed:
        decision = "closed_seed41_model_gate"
    else:
        decision = "closed_seed41_interruption_adjusted_resource_or_provenance_gate"
    return {
        "schema": OUTPUT_SCHEMA,
        "loop_id": "Loop175",
        "seed": 41,
        "claim_scope": "train_only_oof_and_resource_accounting_not_val_test10k_or_full_test",
        "evidence": {
            "final_receipt": {"path": final_path.as_posix(), "sha256": sha256_file(final_path)},
            "protocol": {"path": protocol_path.as_posix(), "sha256": expected_protocol},
            "interruptions": charges,
        },
        "model_gate": {"decision": model_decision, "passed": model_passed},
        "resource_gate": {
            "controller_seed_wall_seconds": controller_wall,
            "external_interruption_charged_seconds": total_charge,
            "interruption_adjusted_seed_wall_seconds": adjusted_wall,
            "maximum_seed41_wall_seconds": maximum_seed_wall,
            "numeric_passed": numeric_resource_passed,
            "provenance_blockers": provenance_blockers,
            "passed": resource_passed,
        },
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
            raise AuditError(f"existing audit differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def audit_exit_code(audit: dict[str, Any]) -> int:
    return 0 if audit["model_gate"]["passed"] and audit["resource_gate"]["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--interruptions-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--if-ready", action="store_true")
    arguments = parser.parse_args()
    if arguments.if_ready and not arguments.final_receipt.is_file():
        print(json.dumps({
            "schema": OUTPUT_SCHEMA,
            "status": "waiting_for_final_receipt",
            "final_receipt": arguments.final_receipt.as_posix(),
        }, ensure_ascii=True, sort_keys=True))
        return 0
    audit = build_audit(
        arguments.final_receipt.resolve(),
        arguments.protocol.resolve(),
        arguments.interruptions_directory.resolve(),
    )
    write_idempotent(arguments.output.resolve(), audit)
    print(json.dumps(audit, ensure_ascii=True, sort_keys=True))
    return audit_exit_code(audit)


if __name__ == "__main__":
    raise SystemExit(main())
