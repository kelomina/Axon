from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_loop175_seed41_interruptions import AuditError, audit_exit_code, build_audit


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def fixture_files(tmp_path: Path, *, decision: str, all_passed: bool) -> tuple[Path, Path, Path]:
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {
        "loop_id": "Loop175",
        "resource_contract": {"maximum_seed41_wall_seconds": 108000},
    })
    protocol_sha256 = hashlib.sha256(protocol.read_bytes()).hexdigest()
    final = tmp_path / "final.json"
    write_json(final, {
        "schema": "axon_loop175_seed41_oof_receipt_v1",
        "loop_id": "Loop175",
        "seed": 41,
        "claim_scope": "train_only_outer_oof_not_val_test10k_or_full_test",
        "protocol_commitment": protocol_sha256,
        "decision": decision,
        "evaluation": {
            "decision": decision,
            "gates": {
                "all_passed": all_passed,
                "runtime": [{"name": "seed_wall_seconds", "observed": 80000, "passed": True}],
            },
        },
        "val_rows_opened": 0,
        "test10k_rows_opened": 0,
        "full_test_rows_opened": 0,
        "val_test_or_full_rows_opened": 0,
    })
    interruptions = tmp_path / "interruptions"
    for index, charge in enumerate((3443, 21600)):
        write_json(interruptions / f"{index}.json", {
            "schema": "axon_loop175_external_interruption_attestation_v1",
            "loop_id": "Loop175",
            "seed": 41,
            "cause": f"cause-{index}",
            "wall_accounting": {
                "require_addition_to_final_seed_wall_gate": True,
                "conservative_charged_seconds": charge,
            },
        })
    return final, protocol, interruptions


def test_combined_pass_accumulates_all_interruptions(tmp_path: Path) -> None:
    final, protocol, interruptions = fixture_files(
        tmp_path,
        decision="seed41_pass_allow_seed42_43",
        all_passed=True,
    )
    audit = build_audit(final, protocol, interruptions)
    assert audit["resource_gate"]["external_interruption_charged_seconds"] == 25043
    assert audit["resource_gate"]["interruption_adjusted_seed_wall_seconds"] == 105043
    assert audit["decision"] == "seed41_pass_allow_seed42_43_with_interruption_adjusted_resource_contract"
    assert audit_exit_code(audit) == 0


def test_model_failure_returns_nonzero_even_when_resources_pass(tmp_path: Path) -> None:
    final, protocol, interruptions = fixture_files(
        tmp_path,
        decision="closed_seed41_gate",
        all_passed=False,
    )
    audit = build_audit(final, protocol, interruptions)
    assert audit["resource_gate"]["passed"] is True
    assert audit["model_gate"]["passed"] is False
    assert audit_exit_code(audit) == 2


def test_unresolved_interruption_provenance_blocks_authorization(tmp_path: Path) -> None:
    final, protocol, interruptions = fixture_files(
        tmp_path,
        decision="seed41_pass_allow_seed42_43",
        all_passed=True,
    )
    path = interruptions / "1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["blocks_seed42_43_authorization"] = True
    write_json(path, payload)
    audit = build_audit(final, protocol, interruptions)
    assert audit["resource_gate"]["numeric_passed"] is True
    assert audit["resource_gate"]["passed"] is False
    assert audit["resource_gate"]["provenance_blockers"] == [path.as_posix()]
    assert audit_exit_code(audit) == 2


def test_decision_gate_disagreement_fails_closed(tmp_path: Path) -> None:
    final, protocol, interruptions = fixture_files(
        tmp_path,
        decision="seed41_pass_allow_seed42_43",
        all_passed=False,
    )
    with pytest.raises(AuditError, match="model decision and gate aggregate disagree"):
        build_audit(final, protocol, interruptions)
